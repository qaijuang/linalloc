use core::alloc::Layout;
use core::cell::Cell;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr::NonNull;
use core::slice;

use crate::{UninitAllocator, sys};

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
/// let slice = bump.try_alloc_uninit(layout).expect("out of memory");
/// let ptr = slice.as_mut_ptr().cast::<u64>();
/// unsafe { ptr.write(42) };
/// let val = unsafe { &*ptr };
/// assert_eq!(*val, 42);
///
/// // Memory is freed when `bump` goes out of scope.
/// ```
#[derive(Debug)]
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

        let base = sys::reserve(capacity)?;

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
    /// `layout`, panicking if the allocation fails.
    ///
    /// See [`BumpArenaLazy::try_alloc_uninit`] for fallible allocation semantics.
    ///
    /// # Panics
    ///
    /// Panics if the arena does not have enough free space after accounting for
    /// the requested size and alignment, or if a required memory commit fails.
    pub fn alloc_uninit(&self, layout: Layout) -> &mut [MaybeUninit<u8>] {
        self.alloc_uninit_impl(layout).expect("BumpArenaLazy allocation failed")
    }

    /// Allocates a mutable slice of [`MaybeUninit<u8>`] that satisfies
    /// `layout`.
    ///
    /// The returned memory is **logically uninitialised** -- it must be
    /// initialised before any reads are performed (for example, using
    /// [`core::ptr::write`]).
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
    pub fn try_alloc_uninit(&self, layout: Layout) -> Option<&mut [MaybeUninit<u8>]> {
        self.alloc_uninit_impl(layout)
    }

    /// Allocates a mutable slice of [`MaybeUninit<u8>`] that satisfies
    /// `layout`.
    #[deprecated(since = "1.2.0", note = "Use `BumpArenaLazy::try_alloc_uninit` instead.")]
    pub fn alloc_uninit_slice(&self, layout: Layout) -> Option<&mut [MaybeUninit<u8>]> {
        self.alloc_uninit_impl(layout)
    }

    #[allow(clippy::mut_from_ref)]
    fn alloc_uninit_impl(&self, layout: Layout) -> Option<&mut [MaybeUninit<u8>]> {
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
            return self.alloc_uninit_bump(aligned, offset, size);
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

    // With the code in `alloc_uninit_bump()` out of the way, `alloc_uninit_impl()` compiles down to some super tight assembly.
    #[cold]
    #[inline(never)]
    #[allow(clippy::mut_from_ref)]
    fn alloc_uninit_bump(
        &self,
        aligned: usize,
        offset: usize,
        size: usize,
    ) -> Option<&mut [MaybeUninit<u8>]> {
        let page = sys::page_size();
        let current = self.commit.get();

        // Round offset up to the next page boundary, capped by capacity.
        let needed = offset.checked_next_multiple_of(page)?.min(self.capacity);

        // Safety:
        // > `current` is page‑aligned and within the reservation.
        // > `needed - current` is a multiple of the page size.
        // > The range has not been committed before, so no overlapping commit.
        unsafe {
            let addr = NonNull::new_unchecked(self.base.as_ptr().add(current));
            if let Err(code) = sys::commit(addr, needed - current) {
                // capture the OS error code immediately
                self.last_os_error.set(code);
                return None;
            }
        }

        self.commit.set(needed);
        self.offset.set(offset);

        unsafe {
            let ptr = self.base.as_ptr().add(aligned);
            Some(slice::from_raw_parts_mut(ptr.cast(), size))
        }
    }

    /// Returns the OS error code from the last failed allocation or commit
    /// operation, if any.
    ///
    /// The returned value is the raw platform‑specific error code:
    /// - On Unix: the `errno` value (positive integer).
    /// - On Windows: the `GetLastError` code.
    ///
    /// Returns `None` if no OS-backed reserve or commit failure has been
    /// recorded for this arena.
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

// Safety: all safety invariants required by `UninitAllocator` are upheld by `BumpArenaLazy`.
unsafe impl UninitAllocator for BumpArenaLazy {
    fn try_alloc_uninit(&self, layout: Layout) -> Option<&mut [MaybeUninit<u8>]> {
        self.alloc_uninit_impl(layout)
    }
}

// Safety:
//
// Same contract as for `&BumpArena`, with the addition that `grow` may
// trigger a virtual‑memory commit if the new size requires it.
#[cfg(feature = "nightly")]
unsafe impl core::alloc::Allocator for BumpArenaLazy {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, core::alloc::AllocError> {
        let slice = self.alloc_uninit_impl(layout).ok_or(core::alloc::AllocError)?;
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
        let base = self.base.as_ptr() as usize;
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
        if required_offset > self.capacity {
            return Err(core::alloc::AllocError);
        }

        if required_offset > self.commit.get() {
            let slice = self
                .alloc_uninit_bump(old_offset, required_offset, new_size)
                .ok_or(core::alloc::AllocError)?;
            let ptr = unsafe { NonNull::new_unchecked(slice.as_mut_ptr().cast()) };
            return Ok(NonNull::slice_from_raw_parts(ptr, new_size));
        }

        self.offset.set(required_offset);
        let new_ptr = unsafe { NonNull::new_unchecked(self.base.as_ptr().add(old_offset)) };
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
        let base = self.base.as_ptr() as usize;
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
        let new_ptr = unsafe { NonNull::new_unchecked(self.base.as_ptr().add(old_offset)) };
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
    fn allocate_last_block_misaligned_to(
        bump: &BumpArenaLazy,
        align: usize,
    ) -> (NonNull<u8>, Layout) {
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
        let bump = BumpArenaLazy::new(128);
        let base = bump.base.as_ptr() as usize;
        let mut prev_end = 0usize;

        for align in [1, 2, 4, 8, 16] {
            let layout = Layout::from_size_align(3, align).unwrap();
            let slice = bump.try_alloc_uninit(layout).unwrap();
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
        let a = bump.try_alloc_uninit(Layout::from_size_align(16, 8).unwrap()).unwrap();
        let b = bump.try_alloc_uninit(Layout::from_size_align(8, 8).unwrap()).unwrap();

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
        bump.try_alloc_uninit(layout).unwrap();
        let used_before = bump.used();

        let too_large = Layout::from_size_align(9, 1).unwrap();
        assert!(bump.try_alloc_uninit(too_large).is_none());
        assert_eq!(bump.used(), used_before);
        assert!(bump.used() <= bump.capacity());
    }

    #[test]
    fn reset_reuses_base() {
        let bump = BumpArenaLazy::new(32);
        let layout = Layout::from_size_align(8, 4).unwrap();
        let first = bump.try_alloc_uninit(layout).unwrap();
        let first_ptr = first.as_ptr() as usize;

        unsafe { bump.reset() };
        assert_eq!(bump.used(), 0);

        let second = bump.try_alloc_uninit(layout).unwrap();
        let second_ptr = second.as_ptr() as usize;
        assert_eq!(first_ptr, second_ptr);
    }

    #[test]
    fn zero_capacity_rejects_nonzero_alloc_uninit() {
        let bump = BumpArenaLazy::new(0);
        let layout = Layout::from_size_align(1, 1).unwrap();
        assert!(bump.try_alloc_uninit(layout).is_none());
        assert_eq!(bump.used(), 0);
    }

    #[test]
    fn zero_size_alloc_does_not_advance() {
        let bump = BumpArenaLazy::new(8);
        let layout = Layout::from_size_align(0, 8).unwrap();
        let slice = bump.try_alloc_uninit(layout).unwrap();
        assert_eq!(slice.len(), 0);
        assert_eq!(bump.used(), 0);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn allocator_grow_and_shrink_resize_last_allocation() {
        let bump = BumpArenaLazy::new(64);
        let alloc = &bump;
        let old = Layout::from_size_align(8, 4).unwrap();
        let grown = Layout::from_size_align(16, 4).unwrap();
        let shrunk = Layout::from_size_align(4, 4).unwrap();

        let block = alloc.allocate(old).unwrap();
        let ptr = block_ptr(block);
        let start = (ptr.as_ptr() as usize) - (bump.base.as_ptr() as usize);
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
        let bump = BumpArenaLazy::new(64);
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
    fn allocator_grow_across_commit_preserves_the_block() {
        let page = sys::page_size();
        let bump = BumpArenaLazy::new(page + 64);
        let alloc = &bump;
        let old = Layout::from_size_align(page - 8, 1).unwrap();
        let grown = Layout::from_size_align(page + 16, 1).unwrap();

        let block = alloc.allocate(old).unwrap();
        let ptr = block_ptr(block);
        unsafe { ptr.as_ptr().write_bytes(0xAB, old.size()) };
        assert_eq!(bump.used(), old.size());
        assert_eq!(bump.commit.get(), page);

        let grown_block = unsafe { alloc.grow(ptr, old, grown).unwrap() };
        let grown_ptr = block_ptr(grown_block);

        assert_eq!(grown_ptr, ptr);
        assert_eq!(bump.used(), grown.size());
        assert!(bump.commit.get() >= grown.size());
        assert_eq!(unsafe { grown_ptr.as_ptr().read() }, 0xAB);
        assert_eq!(unsafe { grown_ptr.as_ptr().add(old.size() - 1).read() }, 0xAB);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn vec_try_reserve_can_grow_inside_allocator() {
        let bump = BumpArenaLazy::new(64);
        let mut values = Vec::with_capacity_in(1, &bump);
        values.push(1);

        assert!(values.try_reserve(1).is_ok());
        values.push(2);

        assert_eq!(&values, &[1, 2]);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn allocator_rejects_resize_when_pointer_does_not_fit_new_alignment() {
        let bump = BumpArenaLazy::new(256);
        let (ptr, old) = allocate_last_block_misaligned_to(&bump, 8);
        let used = bump.used();
        let grown = Layout::from_size_align(16, 8).unwrap();

        assert!(unsafe { bump.grow(ptr, old, grown) }.is_err());
        assert_eq!(bump.used(), used);

        let bump = BumpArenaLazy::new(256);
        let (ptr, old) = allocate_last_block_misaligned_to(&bump, 8);
        let used = bump.used();
        let shrunk = Layout::from_size_align(4, 8).unwrap();

        assert!(unsafe { bump.shrink(ptr, old, shrunk) }.is_err());
        assert_eq!(bump.used(), used);
    }
}
