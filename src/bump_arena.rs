use core::alloc::Layout;
use core::cell::Cell;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr::NonNull;
use core::slice;

/// A fixed‑capacity, single‑threaded bump allocator.
///
/// The arena hands out mutable slices of [`MaybeUninit<u8>`] that
/// are logically uninitialised.  The caller must initialise the
/// memory before reading from it.  The backing store is a boxed
/// slice whose capacity is set once at construction and **never
/// changes**, so addresses remain stable.  For zero capacity, the
/// boxed slice may be a dangling, non‑allocated value.
///
/// # Thread safety
///
/// `BumpArena` is **`!Send` and `!Sync`** -- it contains a raw
/// pointer marker, which is `!Send` and `!Sync`.
///
/// # Examples
///
/// ```
/// use core::alloc::Layout;
///
/// use linalloc::BumpArena;
///
/// let bump = BumpArena::new(1024);
///
/// // Allocate space for a `u64`.
/// let layout = Layout::new::<u64>();
/// let slice = bump.alloc_raw_slice(layout).unwrap();
/// let ptr = slice.as_mut_ptr().cast::<u64>();
/// unsafe { ptr.write(42) };
/// let val = unsafe { &*ptr };
/// assert_eq!(*val, 42);
///
/// // Memory is freed when `bump` goes out of scope.
/// ```
pub struct BumpArena {
    base: NonNull<[MaybeUninit<u8>]>,
    offset: Cell<usize>,
    _invariant: PhantomData<*const ()>,
}

impl BumpArena {
    /// Creates a bump allocator with exactly `capacity` bytes of memory.
    ///
    /// The memory is allocated from the global allocator and is
    /// **uninitialised**.  No zeroing or default‑initialisation is
    /// performed.
    ///
    /// # Panics
    ///
    /// If allocation fails, the global allocator error handler is
    /// invoked (typically aborting the process).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            // SAFETY: `Box` is guaranteed to be non-null.
            base: unsafe { NonNull::new_unchecked(Box::into_raw(Box::new_uninit_slice(capacity))) },
            offset: Cell::new(0),
            _invariant: PhantomData,
        }
    }

    /// Allocates a mutable slice of [`MaybeUninit<u8>`] that satisfies
    /// `layout`.
    ///
    /// The returned memory is **logically uninitialised** -- it must be
    /// initialised (e.g. with [`ptr::write`]) before any reads are
    /// performed.
    ///
    /// The slice borrows the arena immutably (`&self`), so the arena
    /// cannot be dropped or moved while the slice is alive.  The
    /// backing store is never resized, so non‑zero allocations remain
    /// valid until the arena is dropped or [`reset`] is called.  A
    /// zero‑size allocation returns a well‑aligned dangling slice and
    /// does not advance the bump pointer.
    ///
    /// # Returns
    ///
    /// `None` if the arena does not have enough free space after
    /// accounting for the requested size and alignment.
    ///
    /// [`ptr::write`]: core::ptr::write
    /// [`reset`]: BumpArena::reset
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_raw_slice(&self, layout: Layout) -> Option<&mut [MaybeUninit<u8>]> {
        let size = layout.size();
        if size == 0 {
            let ptr = layout.dangling_ptr().as_ptr().cast::<MaybeUninit<u8>>();
            return Some(unsafe { slice::from_raw_parts_mut(ptr, 0) });
        }

        let align = layout.align();
        let offset = self.offset.get();
        let base = self.base.as_ptr().cast::<MaybeUninit<u8>>();

        let pad = unsafe { base.add(offset) }.align_offset(align);
        if pad == usize::MAX {
            return None;
        }

        let aligned = offset.checked_add(pad)?;
        let offset = aligned.checked_add(size)?;
        if offset > self.capacity() {
            return None;
        }

        self.offset.set(offset);

        // Safety:
        // - `base` is a non‑null, heap‑allocated box -- the region
        //   [aligned, aligned+size) is within the allocation.
        // - The bump pointer is monotonically advanced -- no two
        //   allocations overlap.
        // - The returned reference borrows `self`, tying its lifetime
        //   to the arena.
        unsafe { Some(slice::from_raw_parts_mut(base.add(aligned), size)) }
    }

    /// Resets the bump pointer to the beginning, making the entire
    /// capacity available for new allocations.
    ///
    /// # Safety
    ///
    /// All previously returned slices must no longer be in use.
    /// This method **does not** run any destructors -- the caller is
    /// responsible for dropping all values placed in the arena before
    /// calling `reset`.
    pub unsafe fn reset(&self) {
        self.offset.set(0);
    }

    /// Returns the total capacity of the backing memory, in bytes.
    pub fn capacity(&self) -> usize {
        self.base.len()
    }

    /// Returns the number of bytes that have been allocated so far.
    pub fn used(&self) -> usize {
        self.offset.get()
    }
}

impl Drop for BumpArena {
    fn drop(&mut self) {
        unsafe {
            drop(Box::from_raw(self.base.as_ptr()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_alignment_and_length() {
        let bump = BumpArena::new(128);
        let base = bump.base.as_ptr().cast::<MaybeUninit<u8>>() as usize;
        let mut prev_end = 0usize;

        for align in [1, 2, 4, 8, 16] {
            let layout = Layout::from_size_align(3, align).unwrap();
            let slice = bump.alloc_raw_slice(layout).unwrap();
            let ptr = slice.as_ptr() as usize;
            assert_eq!(ptr % align, 0);
            assert_eq!(slice.len(), 3);

            let start = ptr - base;
            let end = start + slice.len();
            assert!(end > prev_end);
            assert_eq!(bump.used(), end);
            assert!(bump.used() <= bump.capacity());
            prev_end = end;
        }
    }

    #[test]
    fn alloc_no_overlap() {
        let bump = BumpArena::new(64);
        let a = bump.alloc_raw_slice(Layout::from_size_align(16, 8).unwrap()).unwrap();
        let b = bump.alloc_raw_slice(Layout::from_size_align(8, 8).unwrap()).unwrap();

        let a_start = a.as_ptr() as usize;
        let a_end = a_start + a.len();
        let b_start = b.as_ptr() as usize;
        let b_end = b_start + b.len();

        assert!(a_end <= b_start || b_end <= a_start);
    }

    #[test]
    fn alloc_oom_does_not_advance() {
        let bump = BumpArena::new(16);
        let layout = Layout::from_size_align(8, 1).unwrap();
        bump.alloc_raw_slice(layout).unwrap();
        let used_before = bump.used();

        let too_large = Layout::from_size_align(9, 1).unwrap();
        assert!(bump.alloc_raw_slice(too_large).is_none());
        assert_eq!(bump.used(), used_before);
        assert!(bump.used() <= bump.capacity());
    }

    #[test]
    fn reset_reuses_base() {
        let bump = BumpArena::new(32);
        let layout = Layout::from_size_align(8, 4).unwrap();
        let first = bump.alloc_raw_slice(layout).unwrap();
        let first_ptr = first.as_ptr() as usize;

        unsafe { bump.reset() };
        assert_eq!(bump.used(), 0);

        let second = bump.alloc_raw_slice(layout).unwrap();
        let second_ptr = second.as_ptr() as usize;
        assert_eq!(first_ptr, second_ptr);
    }

    #[test]
    fn zero_capacity_rejects_nonzero_alloc_raw_slice() {
        let bump = BumpArena::new(0);
        let layout = Layout::from_size_align(1, 1).unwrap();
        assert!(bump.alloc_raw_slice(layout).is_none());
        assert_eq!(bump.used(), 0);
    }

    #[test]
    fn zero_size_alloc_does_not_advance() {
        let bump = BumpArena::new(8);
        let layout = Layout::from_size_align(0, 8).unwrap();
        let slice = bump.alloc_raw_slice(layout).unwrap();
        assert_eq!(slice.len(), 0);
        assert_eq!(bump.used(), 0);
    }
}
