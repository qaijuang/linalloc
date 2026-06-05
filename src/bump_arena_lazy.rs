use core::alloc::Layout;
use core::cell::Cell;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr::NonNull;
use core::slice;

use crate::sys;

/// A fixed‑capacity, single‑threaded bump allocator backed by lazy‑committed
/// virtual memory.
///
/// `BumpArenaLazy` provides mutable slices of [`MaybeUninit<u8>`] that are
/// logically uninitialised. The caller must initialise the memory before
/// reading from it. The backing store is a reserved virtual‑memory region
/// whose total capacity is set once at construction and **never changes**.
/// Physical memory is committed on demand as the bump pointer advances, so
/// the arena can be created with a very large capacity without immediately
/// consuming physical memory.
///
/// # Memory commitment strategy
///
/// The arena uses incremental commitment: the initial physical footprint is
/// tiny, and pages are committed in chunks as allocations request more memory.
/// Committed memory is never decommitted until the entire arena is dropped.
/// This gives stable addresses, predictable performance, and minimal upfront
/// resource usage.
///
/// # Thread safety
///
/// `BumpArenaLazy` is **`!Send` and `!Sync`** -- it contains a raw‑pointer marker
/// that prevents the value from leaving the thread where it was created. The
/// arena is therefore safe to use in single‑threaded contexts only.
///
/// # Examples
///
/// ```
/// use core::alloc::Layout;
///
/// use linalloc::BumpArenaLazy;
///
/// let bump = BumpArenaLazy::new(1024);
///
/// // Allocate space for a `u64`.
/// let layout = Layout::new::<u64>();
/// let slice = bump.alloc_uninit_slice(layout).expect("out of memory");
/// let ptr = slice.as_mut_ptr().cast::<u64>();
/// unsafe { ptr.write(42) };
/// let val = unsafe { &*ptr };
/// assert_eq!(*val, 42);
///
/// // Memory is freed when `bump` goes out of scope.
/// ```
pub struct BumpArenaLazy {
    base: NonNull<u8>,
    capacity: usize,
    offset: Cell<usize>,
    commit: Cell<usize>,
    last_os_error: Cell<i32>,
    _invariant: PhantomData<*const ()>,
}

impl BumpArenaLazy {
    /// Creates a bump allocator that can grow up to `capacity` bytes.
    ///
    /// The memory is **reserved** but not committed -- physical pages are
    /// allocated only when needed, as the bump pointer moves forward.
    /// If `capacity` is zero, the arena is empty and will reject all non‑zero
    /// allocations.
    ///
    /// # Panics
    ///
    /// Panics if the operating system cannot reserve the requested address
    /// range. A zero‑capacity arena never panics.
    ///
    /// # Examples
    ///
    /// ```
    /// use linalloc::BumpArenaLazy;
    ///
    /// let arena = BumpArenaLazy::new(1024);
    /// assert_eq!(arena.capacity(), 1024);
    /// assert_eq!(arena.used(), 0);
    /// ```
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self::try_new(capacity).expect("BumpArenaLazy::new failed to reserve memory")
    }

    /// Like [`new`], but with no panic behaviour.
    ///
    /// # Errors
    ///
    /// Returns OS error code if reservation fails.
    ///
    /// [`new`]: BumpArenaLazy::new
    pub fn try_new(capacity: usize) -> Result<Self, i32> {
        // saves us one unnecessary syscall.
        if capacity == 0 {
            return Ok(Self {
                base: NonNull::dangling(),
                capacity: 0,
                offset: Cell::new(0),
                commit: Cell::new(0),
                last_os_error: Cell::new(0),
                _invariant: PhantomData,
            });
        }

        let base = sys::reserve(capacity).ok_or_else(sys::last_os_error)?;

        Ok(Self {
            base,
            capacity,
            offset: Cell::new(0),
            commit: Cell::new(0),
            last_os_error: Cell::new(0),
            _invariant: PhantomData,
        })
    }

    /// Allocates a mutable slice of [`MaybeUninit<u8>`] that satisfies
    /// `layout`.
    ///
    /// The returned memory is **logically uninitialised** -- it must be
    /// initialised before any reads are performed (for example, using
    /// [`ptr::write`]).
    ///
    /// The slice borrows the arena immutably (`&self`), so the arena cannot
    /// be dropped or moved while the slice is alive. This guarantees that
    /// multiple allocations can coexist without aliasing.
    ///
    /// A zero‑size allocation returns a well‑aligned dangling slice and does
    /// **not** advance the bump pointer.
    ///
    /// # Returns
    ///
    /// `None` if the arena does not have enough free space after accounting
    /// for the requested size and alignment, or if a required memory commit
    /// fails.
    ///
    /// [`ptr::write`]: core::ptr::write
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_uninit_slice(&self, layout: Layout) -> Option<&mut [MaybeUninit<u8>]> {
        let size = layout.size();
        if size == 0 {
            let ptr = layout.dangling_ptr().as_ptr().cast::<MaybeUninit<u8>>();
            return Some(unsafe { slice::from_raw_parts_mut(ptr, 0) });
        }

        let align = layout.align();
        let offset = self.offset.get();
        let base = self.base.as_ptr();

        let base_addr = base as usize;
        let addr = base_addr + offset;
        let align_mask = align - 1;
        let aligned_addr = addr.checked_add(align_mask)? & !align_mask;
        let aligned = aligned_addr - base_addr;
        let offset = aligned.checked_add(size)?;
        if offset > self.capacity {
            return None;
        }

        if offset > self.commit.get() {
            self.try_commit(offset)?;
        }
        self.offset.set(offset);

        // Safety: [aligned, offset) lies within the reservation and is
        // backed by committed memory. The bump pointer is monotonically
        // advanced, so no two allocations overlap. The returned slice borrows
        // `self`, tying its lifetime to the arena.
        unsafe {
            let ptr = base.add(aligned);
            Some(slice::from_raw_parts_mut(ptr.cast(), size))
        }
    }

    // With the code in `try_commit()` out of the way, `alloc_uninit_slice()` compiles down to some super tight assembly.
    #[cold]
    fn try_commit(&self, required_offset: usize) -> Option<()> {
        let page = sys::page_size();
        let current = self.commit.get();

        // Round required_offset up to the next page boundary, capped by capacity.
        let needed = required_offset.checked_next_multiple_of(page)?.min(self.capacity);

        // Safety:
        // > `current` is page‑aligned and within the reservation.
        // > `needed - current` is a multiple of the page size.
        // > The range has not been committed before, so no overlapping commit.
        unsafe {
            let addr = NonNull::new_unchecked(self.base.as_ptr().add(current));
            if let Err(()) = sys::commit(addr, needed - current) {
                // capture the OS error immediately
                self.last_os_error.set(sys::last_os_error());
                return None;
            }
        }

        self.commit.set(needed);
        Some(())
    }

    /// Returns the OS error code from the last failed allocation or commit
    /// operation, if any.
    ///
    /// The returned value is the raw platform‑specific error code:
    /// - On Unix: the `errno` value (positive integer).
    /// - On Windows: the `GetLastError` code.
    ///
    /// Returns `None` if no failure has occurred since the arena was created
    /// or since the last successful operation.
    ///
    /// # Semantics
    ///
    /// This method behaves analogously to [`std::io::Error::last_os_error`] at
    /// the point of the failed internal system call. The error code is stable
    /// until the next failure overwrites it.
    ///
    /// [`std::io::Error::last_os_error`]: std::io::Error::last_os_error
    pub fn last_os_error_code(&self) -> Option<i32> {
        let code = self.last_os_error.get();
        if code == 0 { None } else { Some(code) }
    }

    /// Resets the bump pointer to the beginning, reusing already‑committed
    /// memory.
    ///
    /// # Safety
    ///
    /// All previously returned slices must no longer be in use.
    /// This method does **not** run any destructors -- the caller is
    /// responsible for dropping all values placed in the arena before calling
    /// `reset`.
    pub unsafe fn reset(&self) {
        self.offset.set(0);
    }

    /// Returns the total capacity of the backing memory, in bytes.
    ///
    /// This is the value passed to [`new`] and never changes.
    ///
    /// [`new`]: BumpArenaLazy::new
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of bytes that have been allocated so far.
    pub fn used(&self) -> usize {
        self.offset.get()
    }
}

impl Drop for BumpArenaLazy {
    fn drop(&mut self) {
        if self.capacity > 0 {
            unsafe {
                sys::release(self.base, self.capacity);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_alignment_and_length() {
        let bump = BumpArenaLazy::new(128);
        let base = bump.base.as_ptr() as usize;
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
        let bump = BumpArenaLazy::new(64);
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
        let bump = BumpArenaLazy::new(16);
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
        let bump = BumpArenaLazy::new(32);
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
        let bump = BumpArenaLazy::new(0);
        let layout = Layout::from_size_align(1, 1).unwrap();
        assert!(bump.alloc_uninit_slice(layout).is_none());
        assert_eq!(bump.used(), 0);
    }

    #[test]
    fn zero_size_alloc_does_not_advance() {
        let bump = BumpArenaLazy::new(8);
        let layout = Layout::from_size_align(0, 8).unwrap();
        let slice = bump.alloc_uninit_slice(layout).unwrap();
        assert_eq!(slice.len(), 0);
        assert_eq!(bump.used(), 0);
    }
}
