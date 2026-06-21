use core::alloc::Layout;
use core::cell::Cell;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr::NonNull;
use core::slice;

use crate::UninitAllocator;

/// A fixed‑capacity, single‑threaded bump allocator.
///
/// The arena hands out mutable slices of [`MaybeUninit<u8>`] that
/// are logically uninitialised. The caller must initialise the
/// memory before reading from it. The backing store is a boxed
/// slice whose capacity is set once at construction and **never
/// changes**, so addresses remain stable. For zero capacity, the
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
/// let slice = bump.alloc_uninit_slice(layout).unwrap();
/// let ptr = slice.as_mut_ptr().cast::<u64>();
/// unsafe { ptr.write(42) };
/// let val = unsafe { &*ptr };
/// assert_eq!(*val, 42);
///
/// // Memory is freed when `bump` goes out of scope.
/// ```
#[derive(Debug)]
pub struct BumpArena {
    base: NonNull<[MaybeUninit<u8>]>,
    offset: Cell<usize>,
    _invariant: PhantomData<*const ()>,
}

impl BumpArena {
    /// Creates a bump allocator with exactly `capacity` bytes of memory.
    ///
    /// The memory is allocated from the global allocator and is
    /// **uninitialised**. No zeroing or default‑initialisation is
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
    /// cannot be dropped or moved while the slice is alive. The
    /// backing store is never resized, so non‑zero allocations remain
    /// valid until the arena is dropped or [`reset`] is called. A
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
    pub fn alloc_uninit_slice(&self, layout: Layout) -> Option<&mut [MaybeUninit<u8>]> {
        let size = layout.size();
        if size == 0 {
            let ptr = layout.dangling_ptr().as_ptr().cast::<MaybeUninit<u8>>();
            return Some(unsafe { slice::from_raw_parts_mut(ptr, 0) });
        }

        let align = layout.align();
        let offset = self.offset.get();
        let base = self.base.as_ptr().cast::<MaybeUninit<u8>>();

        let base_addr = base as usize;
        let addr = base_addr + offset;
        let align_mask = align - 1;
        let aligned_addr = addr.checked_add(align_mask)? & !align_mask;
        let aligned = aligned_addr - base_addr;
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

// Safety: all safety invariants required by `UninitAllocator` are upheld by `BumpArena`.
unsafe impl UninitAllocator for BumpArena {
    fn alloc_uninit_slice(&self, layout: Layout) -> Option<&mut [MaybeUninit<u8>]> {
        self.alloc_uninit_slice(layout)
    }
}

// Safety:
//
// `BumpArena` provides correctly aligned, non‑overlapping memory that remains
// stable until the arena is dropped or reset. The `Allocator` contract is
// upheld: `allocate` hands out memory from the bump pointer, `deallocate` is a
// deliberate no‑op (the arena cannot reclaim individual blocks), and `grow` /
// `shrink` only resize the most recent allocation in place when it is the last
// block.
#[cfg(feature = "nightly")]
unsafe impl core::alloc::Allocator for &BumpArena {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, core::alloc::AllocError> {
        let slice = self.alloc_uninit_slice(layout).ok_or(core::alloc::AllocError)?;
        // SAFETY: `slice` is guaranteed to be non-null and valid for `layout.size()` bytes.
        let ptr = unsafe { NonNull::new_unchecked(slice.as_mut_ptr().cast()) };
        Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        // Bump allocator memory is reclaimed only via `reset` or `Drop`.
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, core::alloc::AllocError> {
        let offset = self.offset.get();
        let base = self.base.as_ptr().cast::<MaybeUninit<u8>>() as usize;
        let old_ptr = ptr.as_ptr() as usize;
        let old_offset = old_ptr.checked_sub(base).ok_or(core::alloc::AllocError)?;
        let old_size = old_layout.size();
        let new_size = new_layout.size();
        let old_end = old_offset.checked_add(old_size).ok_or(core::alloc::AllocError)?;
        let is_last = old_end == offset;

        if !is_last || new_size <= old_size || !old_ptr.is_multiple_of(new_layout.align()) {
            return Err(core::alloc::AllocError);
        }

        let required_offset = old_offset.checked_add(new_size).ok_or(core::alloc::AllocError)?;
        if required_offset > self.capacity() {
            return Err(core::alloc::AllocError);
        }

        self.offset.set(required_offset);
        let new_ptr = unsafe {
            NonNull::new_unchecked(
                self.base.as_ptr().cast::<MaybeUninit<u8>>().add(old_offset).cast::<u8>(),
            )
        };
        Ok(NonNull::slice_from_raw_parts(new_ptr, new_size))
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, core::alloc::AllocError> {
        let new_ptr = unsafe { self.grow(ptr, old_layout, new_layout)? };
        let old_size = old_layout.size();
        // Zero the newly added tail.
        let new_bytes = unsafe { new_ptr.as_ptr().cast::<u8>().add(old_size) };
        unsafe { core::ptr::write_bytes(new_bytes, 0, new_layout.size() - old_size) };
        Ok(new_ptr)
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, core::alloc::AllocError> {
        let offset = self.offset.get();
        let base = self.base.as_ptr().cast::<MaybeUninit<u8>>() as usize;
        let old_ptr = ptr.as_ptr() as usize;
        let old_offset = old_ptr.checked_sub(base).ok_or(core::alloc::AllocError)?;
        let old_size = old_layout.size();
        let new_size = new_layout.size();
        let old_end = old_offset.checked_add(old_size).ok_or(core::alloc::AllocError)?;
        let is_last = old_end == offset;

        if !is_last || new_size > old_size || !old_ptr.is_multiple_of(new_layout.align()) {
            return Err(core::alloc::AllocError);
        }

        let new_offset = old_offset.checked_add(new_size).ok_or(core::alloc::AllocError)?;
        self.offset.set(new_offset);
        let new_ptr = unsafe {
            NonNull::new_unchecked(
                self.base.as_ptr().cast::<MaybeUninit<u8>>().add(old_offset).cast::<u8>(),
            )
        };
        Ok(NonNull::slice_from_raw_parts(new_ptr, new_size))
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "nightly")]
    use core::alloc::Allocator;

    use super::*;

    #[cfg(feature = "nightly")]
    fn block_ptr(block: NonNull<[u8]>) -> NonNull<u8> {
        // SAFETY: `Allocator::allocate` never returns a null block pointer.
        unsafe { NonNull::new_unchecked(block.as_ptr().cast::<u8>()) }
    }

    #[cfg(feature = "nightly")]
    fn allocate_last_block_misaligned_to(bump: &BumpArena, align: usize) -> (NonNull<u8>, Layout) {
        let layout = Layout::from_size_align(8, 1).unwrap();
        let pad = Layout::from_size_align(1, 1).unwrap();

        for _ in 0..=align {
            let block = (&bump).allocate(layout).unwrap();
            let ptr = block_ptr(block);
            if !(ptr.as_ptr() as usize).is_multiple_of(align) {
                return (ptr, layout);
            }
            (&bump).allocate(pad).unwrap();
        }

        panic!("could not create a misaligned last allocation");
    }

    #[test]
    fn alloc_alignment_and_length() {
        let bump = BumpArena::new(128);
        let base = bump.base.as_ptr().cast::<MaybeUninit<u8>>() as usize;
        let mut prev_end = 0usize;

        for align in [1, 2, 4, 8, 16] {
            let layout = Layout::from_size_align(3, align).unwrap();
            let slice = bump.alloc_uninit_slice(layout).unwrap();
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
        let a = bump.alloc_uninit_slice(Layout::from_size_align(16, 8).unwrap()).unwrap();
        let b = bump.alloc_uninit_slice(Layout::from_size_align(8, 8).unwrap()).unwrap();

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
        bump.alloc_uninit_slice(layout).unwrap();
        let used_before = bump.used();

        let too_large = Layout::from_size_align(9, 1).unwrap();
        assert!(bump.alloc_uninit_slice(too_large).is_none());
        assert_eq!(bump.used(), used_before);
        assert!(bump.used() <= bump.capacity());
    }

    #[test]
    fn reset_reuses_base() {
        let bump = BumpArena::new(32);
        let layout = Layout::from_size_align(8, 4).unwrap();
        let first = bump.alloc_uninit_slice(layout).unwrap();
        let first_ptr = first.as_ptr() as usize;

        unsafe { bump.reset() };
        assert_eq!(bump.used(), 0);

        let second = bump.alloc_uninit_slice(layout).unwrap();
        let second_ptr = second.as_ptr() as usize;
        assert_eq!(first_ptr, second_ptr);
    }

    #[test]
    fn zero_capacity_rejects_nonzero_alloc_uninit_slice() {
        let bump = BumpArena::new(0);
        let layout = Layout::from_size_align(1, 1).unwrap();
        assert!(bump.alloc_uninit_slice(layout).is_none());
        assert_eq!(bump.used(), 0);
    }

    #[test]
    fn zero_size_alloc_does_not_advance() {
        let bump = BumpArena::new(8);
        let layout = Layout::from_size_align(0, 8).unwrap();
        let slice = bump.alloc_uninit_slice(layout).unwrap();
        assert_eq!(slice.len(), 0);
        assert_eq!(bump.used(), 0);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn allocator_grow_and_shrink_resize_last_allocation() {
        let bump = BumpArena::new(64);
        let alloc = &bump;
        let old = Layout::from_size_align(8, 4).unwrap();
        let grown = Layout::from_size_align(16, 4).unwrap();
        let shrunk = Layout::from_size_align(4, 4).unwrap();

        let block = alloc.allocate(old).unwrap();
        let ptr = block_ptr(block);
        let start =
            (ptr.as_ptr() as usize) - (bump.base.as_ptr().cast::<MaybeUninit<u8>>() as usize);
        unsafe { ptr.as_ptr().write_bytes(0xAB, old.size()) };

        let grown_block = unsafe { alloc.grow(ptr, old, grown).unwrap() };
        let grown_ptr = block_ptr(grown_block);
        assert_eq!(grown_ptr, ptr);
        assert_eq!(bump.used(), start + grown.size());
        assert_eq!(unsafe { grown_ptr.as_ptr().read() }, 0xAB);
        assert_eq!(unsafe { grown_ptr.as_ptr().add(old.size() - 1).read() }, 0xAB);

        let shrunk_block = unsafe { alloc.shrink(grown_ptr, grown, shrunk).unwrap() };
        assert_eq!(block_ptr(shrunk_block), ptr);
        assert_eq!(bump.used(), start + shrunk.size());
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn allocator_grow_zeroed_zeroes_new_tail() {
        let bump = BumpArena::new(64);
        let alloc = &bump;
        let old = Layout::from_size_align(4, 1).unwrap();
        let grown = Layout::from_size_align(12, 1).unwrap();

        let block = alloc.allocate(old).unwrap();
        let ptr = block_ptr(block);
        unsafe { ptr.as_ptr().write_bytes(0xAB, old.size()) };

        let grown_block = unsafe { alloc.grow_zeroed(ptr, old, grown).unwrap() };
        let grown_ptr = block_ptr(grown_block);
        assert_eq!(grown_ptr, ptr);
        for index in 0..old.size() {
            assert_eq!(unsafe { grown_ptr.as_ptr().add(index).read() }, 0xAB);
        }
        for index in old.size()..grown.size() {
            assert_eq!(unsafe { grown_ptr.as_ptr().add(index).read() }, 0);
        }
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn vec_try_reserve_can_grow_inside_allocator() {
        let bump = BumpArena::new(64);
        let mut values = Vec::with_capacity_in(1, &bump);
        values.push(1);

        assert!(values.try_reserve(1).is_ok());
        values.push(2);

        assert_eq!(&values, &[1, 2]);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn allocator_rejects_resize_when_pointer_does_not_fit_new_alignment() {
        let bump = BumpArena::new(256);
        let (ptr, old) = allocate_last_block_misaligned_to(&bump, 8);
        let used = bump.used();
        let grown = Layout::from_size_align(16, 8).unwrap();

        assert!(unsafe { (&bump).grow(ptr, old, grown) }.is_err());
        assert_eq!(bump.used(), used);

        let bump = BumpArena::new(256);
        let (ptr, old) = allocate_last_block_misaligned_to(&bump, 8);
        let used = bump.used();
        let shrunk = Layout::from_size_align(4, 8).unwrap();

        assert!(unsafe { (&bump).shrink(ptr, old, shrunk) }.is_err());
        assert_eq!(bump.used(), used);
    }
}
