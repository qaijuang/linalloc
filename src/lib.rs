#![warn(clippy::pedantic)]
#![doc = include_str!("../README.md")]

#[cfg(not(any(unix, windows)))]
compile_error!("`linalloc` only supports Unix and Windows targets");

use core::alloc::Layout;
use core::cell::Cell;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr::NonNull;
use core::slice;

/// A fixed-capacity bump allocator backed by virtual memory.
///
/// Use [`Self::alloc`] to store values or [`Self::alloc_uninit`] for raw storage.
/// Stored values aren't dropped automatically.
///
/// It reserves address space up front and makes pages writable as needed.
/// Only dropping the arena releases its reservation. Allocations stay at the
/// same address until reset or drop, but allocator API resizing may move a block.
///
/// # Thread safety
///
/// The arena is `!Send` and `!Sync`. Returned references follow their type's
/// normal `Send` and `Sync` bounds.
///
/// # Examples
///
/// ```
/// use linalloc::BumpArena;
///
/// let arena = BumpArena::new(1024);
/// let value = arena.alloc(42);
/// assert_eq!(*value, 42);
/// ```
#[derive(Debug)]
pub struct BumpArena {
    base: NonNull<u8>,
    capacity: usize,
    // Query the page size once
    page_size: usize,
    offset: Cell<usize>,
    commit: Cell<usize>,
    last_os_error: Cell<i32>,
    _invariant: PhantomData<*const ()>,
}

impl BumpArena {
    /// Creates an arena with `capacity` bytes.
    ///
    /// Capacity is fixed. Zero capacity allows only zero-sized allocations.
    ///
    /// # Panics
    ///
    /// Panics if address space cannot be reserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use linalloc::BumpArena;
    ///
    /// let arena = BumpArena::new(1024);
    /// assert_eq!(arena.capacity(), 1024);
    /// assert_eq!(arena.used(), 0);
    /// ```
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self::try_new(capacity).expect("BumpArena::new failed to reserve memory")
    }

    /// Fallible version of [`Self::new`].
    ///
    /// # Errors
    ///
    /// Returns the OS error code if reservation fails.
    pub fn try_new(capacity: usize) -> Result<Self, i32> {
        // No mapping needed for an empty arena
        if capacity == 0 {
            return Ok(Self {
                base: NonNull::dangling(),
                capacity: 0,
                page_size: usize::MAX,
                offset: Cell::new(0),
                commit: Cell::new(0),
                last_os_error: Cell::new(0),
                _invariant: PhantomData,
            });
        }

        let base = linalloc_sys::reserve(capacity)?;
        let page_size = linalloc_sys::page_size();

        Ok(Self {
            base,
            capacity,
            page_size,
            offset: Cell::new(0),
            commit: Cell::new(0),
            last_os_error: Cell::new(0),
            _invariant: PhantomData,
        })
    }

    /// Allocates `value` in the arena.
    ///
    /// See [`Self::try_alloc`] for failure behavior.
    ///
    /// # Panics
    ///
    /// Panics if allocation fails. The rejected value is dropped first.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::mem::MaybeUninit;
    ///
    /// use linalloc::BumpArena;
    ///
    /// let arena = BumpArena::new(128);
    /// let source = String::from("hello");
    /// let text = arena.alloc(source.as_str());
    /// let number = arena.alloc(MaybeUninit::<u64>::uninit()).write(42);
    /// assert_eq!(*text, "hello");
    /// assert_eq!(*number, 42);
    /// ```
    ///
    /// The returned reference cannot outlive the arena:
    ///
    /// ```compile_fail
    /// use linalloc::BumpArena;
    ///
    /// let value;
    /// {
    ///     let arena = BumpArena::new(8);
    ///     value = arena.alloc(42_u64);
    /// }
    /// assert_eq!(*value, 42);
    /// ```
    ///
    /// `&mut T` is invariant in `T`:
    ///
    /// ```compile_fail
    /// use linalloc::BumpArena;
    ///
    /// fn shorten_inner<'a>(arena: &'a BumpArena) -> &'a mut &'a str {
    ///     arena.alloc::<&'static str>("hello")
    /// }
    /// ```
    #[inline]
    pub fn alloc<T>(&self, value: T) -> &mut T {
        self.try_alloc(value).expect("BumpArena allocation failed")
    }

    /// Allocates `value` in the arena.
    ///
    /// Returns `None` if it doesn't fit or a page commit fails.
    /// Failed allocations drop `value`. Stored values aren't dropped automatically.
    /// Zero-sized values use no capacity.
    ///
    /// The returned reference borrows the arena. Values may borrow other data.
    ///
    /// # Panics
    ///
    /// Panics if dropping a rejected value panics.
    ///
    /// # Examples
    ///
    /// ```
    /// use linalloc::BumpArena;
    ///
    /// let arena = BumpArena::new(8);
    /// assert_eq!(*arena.try_alloc(42_u64).unwrap(), 42);
    /// assert!(arena.try_alloc(7_u64).is_none());
    /// ```
    ///
    /// Allocation does not extend the lifetime of borrowed data:
    ///
    /// ```compile_fail
    /// use linalloc::BumpArena;
    ///
    /// let arena = BumpArena::new(128);
    /// let value;
    /// {
    ///     let source = String::from("hello");
    ///     value = arena.try_alloc(source.as_str()).unwrap();
    /// }
    /// assert_eq!(*value, "hello");
    /// ```
    #[allow(clippy::mut_from_ref)]
    #[inline]
    pub fn try_alloc<T>(&self, value: T) -> Option<&mut T> {
        let storage = self.alloc_uninit_impl(Layout::new::<T>())?;
        // SAFETY: The raw allocator provides fresh, committed storage with the
        // size and alignment of T, or an aligned non-null dangling pointer for
        // a zero-sized T. All bit patterns are valid for MaybeUninit<T>, and
        // the slot is exclusive for the returned borrow of self.
        let slot = unsafe { &mut *storage.as_mut_ptr().cast::<MaybeUninit<T>>() };
        Some(slot.write(value))
    }

    /// Allocates uninitialized bytes for `layout`.
    ///
    /// See [`Self::try_alloc_uninit`] for failure behavior.
    ///
    /// # Panics
    ///
    /// Panics if the layout doesn't fit or a page commit fails.
    #[inline]
    pub fn alloc_uninit(&self, layout: Layout) -> &mut [MaybeUninit<u8>] {
        self.alloc_uninit_impl(layout).expect("BumpArena allocation failed")
    }

    /// Allocates uninitialized bytes for `layout`.
    ///
    /// The slice has `layout.size()` bytes and satisfies `layout.align()`.
    /// Initialize the memory before reading it.
    ///
    /// Allocations are disjoint. Each returned slice borrows the arena.
    /// A zero-sized layout returns an empty aligned slice without consuming capacity.
    ///
    /// Returns `None` if the layout doesn't fit or a page commit fails.
    #[inline]
    pub fn try_alloc_uninit(&self, layout: Layout) -> Option<&mut [MaybeUninit<u8>]> {
        self.alloc_uninit_impl(layout)
    }

    #[allow(clippy::mut_from_ref)]
    #[inline]
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

        // SAFETY: [aligned, offset) is writable, in bounds, and disjoint from
        // earlier allocations. The slice borrows self.
        unsafe {
            let ptr = base.add(aligned);
            Some(slice::from_raw_parts_mut(ptr.cast(), size))
        }
    }

    // Keep commit work off the allocation fast path
    #[allow(clippy::mut_from_ref)]
    #[cold]
    #[inline(never)]
    fn alloc_uninit_bump(
        &self,
        aligned: usize,
        offset: usize,
        size: usize,
    ) -> Option<&mut [MaybeUninit<u8>]> {
        let current = self.commit.get();

        let needed = offset.checked_next_multiple_of(self.page_size)?.min(self.capacity);

        // SAFETY: [current, needed) is nonempty, reserved, and uncommitted.
        // current is page-aligned. The OS rounds a partial final page up,
        // just as it rounded the reservation.
        unsafe {
            let addr = NonNull::new_unchecked(self.base.as_ptr().add(current));
            if let Err(code) = linalloc_sys::commit(addr, needed - current) {
                self.last_os_error.set(code);
                return None;
            }
        }

        self.commit.set(needed);
        self.offset.set(offset);

        // SAFETY: The caller supplied an in-bounds range for a fresh allocation
        // or an exclusive resize. The range is now writable.
        unsafe {
            let ptr = self.base.as_ptr().add(aligned);
            Some(slice::from_raw_parts_mut(ptr.cast(), size))
        }
    }

    /// Returns the OS error code from the last failed page commit.
    ///
    /// Returns `None` until a commit fails. Success, reset, and capacity failures
    /// don't clear the code.
    ///
    /// The code is `errno` on Unix or `GetLastError` on Windows.
    pub fn last_os_error_code(&self) -> Option<i32> {
        let code = self.last_os_error.get();
        if code == 0 { None } else { Some(code) }
    }

    /// Rewinds the bump pointer without releasing pages or dropping values.
    ///
    /// # Safety
    ///
    /// End all borrows and drop stored values and allocator-backed collections
    /// before resetting. No pointer to an old allocation may be used afterward.
    pub unsafe fn reset(&self) {
        self.offset.set(0);
    }

    /// Returns the fixed capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns bytes consumed, including alignment padding.
    pub fn used(&self) -> usize {
        self.offset.get()
    }
}

impl Drop for BumpArena {
    fn drop(&mut self) {
        if self.capacity > 0 {
            // SAFETY: self owns this live reservation. Safe borrows cannot
            // outlive the arena.
            unsafe {
                linalloc_sys::release(self.base, self.capacity);
            }
        }
    }
}

// SAFETY: Allocations are aligned, disjoint, and valid across arena moves.
// Resizing preserves the required bytes and leaves the old block valid on
// failure. Safe allocation methods never invalidate other blocks.
// Reset is unsafe, and deallocation is a no-op.
unsafe impl core::alloc::Allocator for BumpArena {
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

        if new_size < old_size {
            return Err(core::alloc::AllocError);
        }

        if old_ptr.is_multiple_of(new_layout.align()) {
            if new_size == old_size {
                return Ok(NonNull::slice_from_raw_parts(ptr, new_size));
            }

            if is_last {
                let required_offset =
                    old_offset.checked_add(new_size).ok_or(core::alloc::AllocError)?;
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
                return Ok(NonNull::slice_from_raw_parts(new_ptr, new_size));
            }
        }

        let new_block = self.allocate(new_layout)?;
        unsafe {
            core::ptr::copy_nonoverlapping(
                ptr.as_ptr(),
                new_block.as_ptr().cast::<u8>(),
                old_layout.size(),
            );
        }
        Ok(new_block)
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

        if new_size > old_size {
            return Err(core::alloc::AllocError);
        }

        if old_ptr.is_multiple_of(new_layout.align()) {
            if is_last {
                let new_offset = old_offset.checked_add(new_size).ok_or(core::alloc::AllocError)?;
                self.offset.set(new_offset);
            }
            return Ok(NonNull::slice_from_raw_parts(ptr, new_size));
        }

        let new_block = self.allocate(new_layout)?;
        unsafe {
            core::ptr::copy_nonoverlapping(
                ptr.as_ptr(),
                new_block.as_ptr().cast::<u8>(),
                new_layout.size(),
            );
        }
        Ok(new_block)
    }
}

#[cfg(test)]
mod tests {

    use core::alloc::Allocator;

    use super::*;

    struct DropCounter<'a>(&'a Cell<usize>);

    impl Drop for DropCounter<'_> {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    fn block_ptr(block: NonNull<[u8]>) -> NonNull<u8> {
        // SAFETY: `Allocator::allocate` never returns a null block pointer.
        unsafe { NonNull::new_unchecked(block.as_ptr().cast::<u8>()) }
    }

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
    fn typed_alloc_preserves_borrowed_values() {
        let bump = BumpArena::new(128);
        let source = String::from("hello");
        let first = bump.alloc(source.as_str());
        let second = bump.try_alloc(source.as_bytes()).unwrap();

        assert_eq!(*first, "hello");
        assert_eq!(*second, b"hello");
    }

    #[test]
    fn typed_allocations_coexist_with_raw_allocations_and_vec_growth() {
        let bump = BumpArena::new(256);
        let first = bump.alloc(41_u64);
        let raw = bump.alloc_uninit(Layout::new::<[u8; 8]>()).write_copy_of_slice(&[0xAB; 8]);
        let mut values = Vec::with_capacity_in(1, &bump);
        values.push(7_u64);
        let second = bump.try_alloc(1_u64).unwrap();

        values.try_reserve(1).unwrap();
        values.push(8);
        *first += 1;
        *second += 1;
        raw[0] = 0xCD;

        assert_eq!(*first, 42);
        assert_eq!(*second, 2);
        assert_eq!(raw, &[0xCD, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB]);
        assert_eq!(&values, &[7, 8]);
    }

    #[test]
    fn typed_alloc_respects_alignment_after_raw_allocation() {
        #[repr(align(64))]
        struct Aligned([u8; 64]);

        let bump = BumpArena::new(256);
        bump.alloc_uninit(Layout::new::<u8>());
        let value = bump.try_alloc(Aligned([0xAB; 64])).unwrap();

        assert!((core::ptr::from_mut(value) as usize).is_multiple_of(64));
        assert_eq!(value.0, [0xAB; 64]);
    }

    #[test]
    fn typed_alloc_zero_sized_values_need_no_capacity_or_destruction() {
        #[repr(align(64))]
        struct Aligned;

        impl Drop for Aligned {
            fn drop(&mut self) {
                panic!("the arena must not drop allocated values");
            }
        }

        let bump = BumpArena::new(0);
        let first = bump.alloc(Aligned);
        let second = bump.try_alloc(Aligned).unwrap();

        assert!((core::ptr::from_mut(first) as usize).is_multiple_of(64));
        assert!((core::ptr::from_mut(second) as usize).is_multiple_of(64));
        assert_eq!(bump.used(), 0);
    }

    #[test]
    fn typed_alloc_does_not_drop_successful_values() {
        let drops = Cell::new(0);
        let bump = BumpArena::new(128);
        bump.alloc(DropCounter(&drops));
        bump.try_alloc(DropCounter(&drops)).unwrap();

        assert_eq!(drops.get(), 0);
        drop(bump);
        assert_eq!(drops.get(), 0);
    }

    #[test]
    fn typed_try_alloc_drops_rejected_value_without_consuming_capacity() {
        let drops = Cell::new(0);
        let bump = BumpArena::new(8);
        let first = bump.alloc(42_u64);
        let used = bump.used();

        assert!(bump.try_alloc(DropCounter(&drops)).is_none());
        assert_eq!(drops.get(), 1);
        assert_eq!(bump.used(), used);
        assert_eq!(*first, 42);
    }

    #[test]
    fn typed_alloc_drops_rejected_value_before_panicking() {
        let drops = Cell::new(0);
        let bump = BumpArena::new(0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            bump.alloc(DropCounter(&drops));
        }));

        assert!(result.is_err());
        assert_eq!(drops.get(), 1);
        assert_eq!(bump.used(), 0);
    }

    #[test]
    fn typed_alloc_supports_uninitialized_values() {
        let bump = BumpArena::new(128);
        let first = bump.alloc(MaybeUninit::<u64>::uninit()).write(42);
        let second = bump.try_alloc(MaybeUninit::<u64>::uninit()).unwrap().write(7);

        assert_eq!(*first, 42);
        assert_eq!(*second, 7);
    }

    #[test]
    fn alloc_alignment_and_length() {
        let bump = BumpArena::new(128);
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
        let bump = BumpArena::new(64);
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
        let bump = BumpArena::new(16);
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
        let bump = BumpArena::new(32);
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
        let bump = BumpArena::new(0);
        let layout = Layout::from_size_align(1, 1).unwrap();
        assert!(bump.try_alloc_uninit(layout).is_none());
        assert_eq!(bump.used(), 0);
    }

    #[test]
    fn zero_size_alloc_does_not_advance() {
        let bump = BumpArena::new(8);
        let layout = Layout::from_size_align(0, 8).unwrap();
        let slice = bump.try_alloc_uninit(layout).unwrap();
        assert_eq!(slice.len(), 0);
        assert_eq!(bump.used(), 0);
    }

    #[test]
    fn allocator_grow_and_shrink_resize_last_allocation() {
        let bump = BumpArena::new(64);
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

    #[test]
    fn allocator_grow_across_commit_preserves_the_block() {
        let page = linalloc_sys::page_size();
        let bump = BumpArena::new(page + 64);
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

    #[test]
    fn vec_try_reserve_can_grow_inside_allocator() {
        let bump = BumpArena::new(64);
        let mut values = Vec::with_capacity_in(1, &bump);
        values.push(1);

        assert!(values.try_reserve(1).is_ok());
        values.push(2);

        assert_eq!(&values, &[1, 2]);
    }

    #[test]
    fn allocator_grow_relocates_non_last_allocation() {
        let alloc = &BumpArena::new(128);
        let old = Layout::from_size_align(8, 1).unwrap();
        let blocker = Layout::from_size_align(8, 1).unwrap();
        let grown = Layout::from_size_align(16, 1).unwrap();

        let block = alloc.allocate(old).unwrap();
        let ptr = block_ptr(block);
        unsafe { ptr.as_ptr().write_bytes(0xAB, old.size()) };
        let blocker = alloc.allocate(blocker).unwrap();
        let blocker_ptr = block_ptr(blocker);
        unsafe { blocker_ptr.as_ptr().write_bytes(0xCD, 8) };

        let grown_block = unsafe { alloc.grow(ptr, old, grown).unwrap() };
        let grown_ptr = block_ptr(grown_block);

        assert_ne!(grown_ptr, ptr);
        for index in 0..old.size() {
            assert_eq!(unsafe { grown_ptr.as_ptr().add(index).read() }, 0xAB);
        }
        for index in 0..8 {
            assert_eq!(unsafe { blocker_ptr.as_ptr().add(index).read() }, 0xCD);
        }
    }

    #[test]
    fn allocator_grow_zeroed_relocation_zeroes_new_tail() {
        let alloc = &BumpArena::new(128);
        let old = Layout::from_size_align(4, 1).unwrap();
        let blocker = Layout::from_size_align(8, 1).unwrap();
        let grown = Layout::from_size_align(12, 1).unwrap();

        let block = alloc.allocate(old).unwrap();
        let ptr = block_ptr(block);
        unsafe { ptr.as_ptr().write_bytes(0xAB, old.size()) };
        alloc.allocate(blocker).unwrap();

        let grown_block = unsafe { alloc.grow_zeroed(ptr, old, grown).unwrap() };
        let grown_ptr = block_ptr(grown_block);

        assert_ne!(grown_ptr, ptr);
        for index in 0..old.size() {
            assert_eq!(unsafe { grown_ptr.as_ptr().add(index).read() }, 0xAB);
        }
        for index in old.size()..grown.size() {
            assert_eq!(unsafe { grown_ptr.as_ptr().add(index).read() }, 0);
        }
    }

    #[test]
    fn allocator_grow_error_keeps_old_allocation_untouched() {
        let bump = BumpArena::new(24);
        let alloc = &bump;
        let old = Layout::from_size_align(8, 1).unwrap();
        let blocker = Layout::from_size_align(16, 1).unwrap();
        let grown = Layout::from_size_align(16, 1).unwrap();

        let block = alloc.allocate(old).unwrap();
        let ptr = block_ptr(block);
        unsafe { ptr.as_ptr().write_bytes(0xAB, old.size()) };
        alloc.allocate(blocker).unwrap();
        let used = bump.used();

        assert!(unsafe { alloc.grow(ptr, old, grown) }.is_err());
        assert_eq!(bump.used(), used);
        for index in 0..old.size() {
            assert_eq!(unsafe { ptr.as_ptr().add(index).read() }, 0xAB);
        }
    }

    #[test]
    fn vec_try_reserve_relocates_when_buffer_is_not_last_allocation() {
        let bump = BumpArena::new(128);
        let mut values = Vec::with_capacity_in(1, &bump);
        values.push(1u64);
        let original = values.as_ptr();
        let blocker = bump.allocate(Layout::new::<u64>()).unwrap();
        let blocker_ptr = block_ptr(blocker);
        unsafe { blocker_ptr.as_ptr().write_bytes(0xCD, core::mem::size_of::<u64>()) };

        assert!(values.try_reserve(1).is_ok());
        values.push(2);

        assert_ne!(values.as_ptr(), original);
        assert_eq!(&values, &[1, 2]);
        for index in 0..core::mem::size_of::<u64>() {
            assert_eq!(unsafe { blocker_ptr.as_ptr().add(index).read() }, 0xCD);
        }
    }

    #[test]
    fn try_new_rejects_unreservable_capacity() {
        let err = BumpArena::try_new(usize::MAX).unwrap_err();
        #[cfg(not(miri))]
        assert_eq!(Some(err), std::io::Error::last_os_error().raw_os_error());
        assert_ne!(err, 0);
    }

    #[test]
    fn allocator_relocates_when_pointer_does_not_fit_new_alignment() {
        let bump = BumpArena::new(256);
        let (ptr, old) = allocate_last_block_misaligned_to(&bump, 8);
        unsafe { ptr.as_ptr().write_bytes(0xAB, old.size()) };
        let grown = Layout::from_size_align(16, 8).unwrap();

        let grown_block = unsafe { bump.grow(ptr, old, grown).unwrap() };
        let grown_ptr = block_ptr(grown_block);
        assert_ne!(grown_ptr, ptr);
        assert_eq!((grown_ptr.as_ptr() as usize) % grown.align(), 0);
        for index in 0..old.size() {
            assert_eq!(unsafe { grown_ptr.as_ptr().add(index).read() }, 0xAB);
        }

        let bump = BumpArena::new(256);
        let (ptr, old) = allocate_last_block_misaligned_to(&bump, 8);
        unsafe { ptr.as_ptr().write_bytes(0xAB, old.size()) };
        let shrunk = Layout::from_size_align(4, 8).unwrap();

        let shrunk_block = unsafe { bump.shrink(ptr, old, shrunk).unwrap() };
        let shrunk_ptr = block_ptr(shrunk_block);
        assert_ne!(shrunk_ptr, ptr);
        assert_eq!((shrunk_ptr.as_ptr() as usize) % shrunk.align(), 0);
        for index in 0..shrunk.size() {
            assert_eq!(unsafe { shrunk_ptr.as_ptr().add(index).read() }, 0xAB);
        }
    }
}
