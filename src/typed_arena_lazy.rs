use core::cell::Cell;
use core::marker::PhantomData;
use core::mem::{MaybeUninit, size_of};
use core::ptr::{NonNull, drop_in_place};

use crate::sys;

/// A fixed‑capacity, single‑threaded arena that allocates values of type `T`
/// and automatically drops them in reverse allocation order.
///
/// The backing store is a reserved virtual‑memory region whose total capacity
/// is set at construction in *elements*. Physical memory is committed on
/// demand as elements are allocated, so the arena can be created with a very
/// large capacity without immediately consuming physical memory.
///
/// # Invariance and thread safety
///
/// `TypedArenaLazy<T>` is **invariant** in `T` and **`!Send + !Sync`**. The
/// marker field prevents unsound subtyping and cross‑thread usage. This
/// guarantees:
///
/// - No unsound subtyping (e.g. treating a `String` arena as a `dyn Display`
///   arena, which would break `Drop`).
/// - The arena is confined to a single thread.
///
/// # Examples
///
/// ```
/// use linalloc::TypedArenaLazy;
///
/// let mut arena = TypedArenaLazy::<String>::new(5);
///
/// let s = arena.try_alloc("hello".to_string()).unwrap();
/// assert_eq!(s, "hello");
///
/// // All values are dropped when `arena` goes out of scope.
/// ```
#[derive(Debug)]
pub struct TypedArenaLazy<T> {
    base: NonNull<MaybeUninit<T>>,
    capacity: usize,
    // pays the cost of syscall upfront.
    page_size: usize,
    offset: Cell<usize>,
    commit: Cell<usize>,
    last_os_error: Cell<i32>,
    #[allow(clippy::type_complexity)]
    _invariant: PhantomData<(*const (), fn(T) -> T)>,
}

impl<T> TypedArenaLazy<T> {
    /// Creates a new typed arena that can hold up to `capacity` elements of
    /// type `T`.
    ///
    /// The backing memory is **reserved** but not committed -- physical pages
    /// are allocated only as elements are added. If `capacity` is zero, or
    /// `T` is a zero‑sized type (e.g. `()`), the arena is empty and will
    /// reject all non‑zero‑sized allocations.
    ///
    /// # Panics
    ///
    /// Panics if the operating system cannot reserve the requested address
    /// range, or if the required byte size overflows.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self::try_new(capacity).expect("TypedArenaLazy::new failed to reserve memory")
    }

    /// Like [`new`], but returns a `Result`.
    ///
    /// # Panics
    ///
    /// Panics if the required byte size overflows.
    ///
    /// # Errors
    ///
    /// Returns OS error code if reservation fails.
    ///
    /// [`new`]: TypedArenaLazy::new
    pub fn try_new(capacity: usize) -> Result<Self, i32> {
        let page_size = sys::page_size();

        if capacity == 0 || size_of::<T>() == 0 {
            return Ok(Self {
                base: NonNull::dangling(),
                capacity,
                page_size,
                offset: Cell::new(0),
                commit: Cell::new(0),
                last_os_error: Cell::new(0),
                _invariant: PhantomData,
            });
        }

        let size_bytes =
            capacity.checked_mul(size_of::<T>()).expect("TypedArenaLazy: capacity overflow");
        let base = sys::reserve(size_bytes)?;

        Ok(Self {
            base: base.cast(),
            capacity,
            page_size,
            offset: Cell::new(0),
            commit: Cell::new(0),
            last_os_error: Cell::new(0),
            _invariant: PhantomData,
        })
    }

    /// Just like [`TypedArenaLazy::try_alloc`], but panics if the allocation
    /// fails.
    ///
    /// # Panics
    ///
    /// Panics if the arena is full or a required memory commit fails.
    pub fn alloc(&self, value: T) -> &mut T {
        self.alloc_impl(value).expect("TypedArenaLazy allocation failed")
    }

    /// Allocates a new `T` by moving `value` into the arena.
    ///
    /// The returned mutable reference borrows the arena immutably (`&self`),
    /// so the arena is frozen (cannot be dropped or reset) until the reference
    /// goes out of scope. Multiple allocations can coexist without aliasing.
    ///
    /// Zero‑sized types (e.g. `()`) are handled specially: they consume no
    /// capacity and always succeed.
    ///
    /// # Returns
    ///
    /// `None` if the arena is full (i.e. [`len`](Self::len) == [`capacity`](Self::capacity)).
    ///
    /// # Examples
    ///
    /// ```
    /// use linalloc::TypedArenaLazy;
    ///
    /// let arena = TypedArenaLazy::<i32>::new(10);
    /// let x = arena.try_alloc(42).unwrap();
    /// assert_eq!(*x, 42);
    /// ```
    pub fn try_alloc(&self, value: T) -> Option<&mut T> {
        self.alloc_impl(value)
    }

    /// Allocates a new `T` by moving `value` into the arena.
    #[deprecated(since = "1.2.0", note = "Use `TypedArenaLazy::try_alloc` instead.")]
    pub fn alloc_raw(&self, value: T) -> Option<&mut T> {
        self.alloc_impl(value)
    }

    #[allow(clippy::mut_from_ref)]
    fn alloc_impl(&self, value: T) -> Option<&mut T> {
        // ZST's never consume capacity.
        if size_of::<T>() == 0 {
            unsafe {
                let dangling = NonNull::<T>::dangling();
                dangling.as_ptr().write(value);
                return Some(&mut *dangling.as_ptr());
            }
        }

        let idx = self.offset.get();
        if idx >= self.capacity {
            return None;
        }

        // `new` guarantees `capacity * size_of::<T>()` fits, and `idx < capacity`.
        let required_bytes = (idx + 1) * size_of::<T>();

        // Ensure enough memory is committed.
        if required_bytes > self.commit.get() {
            return self.alloc_bump(idx, required_bytes, value);
        }

        // Initialise the slot.
        unsafe {
            let slot = self.base.as_ptr().add(idx);
            let r = (&mut *slot).write(value);
            self.offset.set(idx + 1);
            Some(r)
        }
    }

    // With the code in `alloc_bump()` out of the way, `alloc_impl()` compiles down to some super tight assembly.
    #[cold]
    #[inline(never)]
    #[allow(clippy::mut_from_ref)]
    fn alloc_bump(&self, idx: usize, required_bytes: usize, value: T) -> Option<&mut T> {
        let current = self.commit.get();

        // `new` guarantees `capacity * size_of::<T>()` fits.
        let total_bytes = self.capacity * size_of::<T>();

        // Next page rounding
        let needed = required_bytes.checked_next_multiple_of(self.page_size)?.min(total_bytes);

        // Safety:
        // > `current` is page‑aligned and within the reservation.
        // > `needed - current` is a multiple of the page size.
        unsafe {
            let addr = NonNull::new_unchecked(self.base.as_ptr().cast::<u8>().add(current));
            if let Err(code) = sys::commit(addr, needed - current) {
                // capture the OS error code immediately
                self.last_os_error.set(code);
                return None;
            }
        }

        self.commit.set(needed);

        // Initialise the slot.
        unsafe {
            let slot = self.base.as_ptr().add(idx);
            let r = (&mut *slot).write(value);
            self.offset.set(idx + 1);
            Some(r)
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

    /// Returns the number of elements currently allocated in the arena.
    ///
    /// # Examples
    ///
    /// ```
    /// use linalloc::TypedArenaLazy;
    ///
    /// let mut arena = TypedArenaLazy::<i32>::new(10);
    /// assert_eq!(arena.len(), 0);
    /// arena.try_alloc(1);
    /// assert_eq!(arena.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.offset.get()
    }

    /// Returns `true` if the arena contains no allocated elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use linalloc::TypedArenaLazy;
    ///
    /// let arena = TypedArenaLazy::<i32>::new(10);
    /// assert!(arena.is_empty());
    /// arena.try_alloc(1);
    /// assert!(!arena.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the maximum number of elements the arena can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Drops all live `T` values in reverse allocation order and resets the
    /// arena for reuse.
    ///
    /// Because this method takes `&mut self`, the borrow checker guarantees
    /// that no references to the arena's contents are currently alive. After
    /// the call, [`len`](Self::len) returns `0`.
    ///
    /// Committed memory is not decommitted -- subsequent allocations will reuse
    /// the already‑committed pages.
    ///
    /// # Examples
    ///
    /// ```
    /// use linalloc::TypedArenaLazy;
    ///
    /// let mut arena = TypedArenaLazy::<Vec<i32>>::new(5);
    /// {
    ///     let v = arena.try_alloc(vec![1, 2, 3]).unwrap();
    /// } // v goes out of scope -- can call `reset` now.
    /// arena.reset();
    /// assert_eq!(arena.len(), 0);
    /// ```
    pub fn reset(&mut self) {
        let offset = self.offset.replace(0);
        unsafe {
            let start = self.base.as_ptr().cast::<T>();
            // Drop in reverse order per Rust's usual drop semantics.
            for i in (0..offset).rev() {
                drop_in_place(start.add(i));
            }
        }
    }
}

impl<T> Drop for TypedArenaLazy<T> {
    fn drop(&mut self) {
        let offset = self.offset.get();
        unsafe {
            let start = self.base.as_ptr().cast::<T>();
            for i in (0..offset).rev() {
                drop_in_place(start.add(i));
            }
            if self.capacity > 0 && size_of::<T>() > 0 {
                let total_bytes = self.capacity * size_of::<T>();
                sys::release(self.base.cast(), total_bytes);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use core::{mem, ptr};

    use super::*;

    // A helper that records drop order and count.
    struct DropTracker<'a> {
        id: u32,
        order: &'a Cell<Vec<u32>>,
    }

    impl Drop for DropTracker<'_> {
        fn drop(&mut self) {
            let mut v = self.order.take();
            v.push(self.id);
            self.order.set(v);
        }
    }

    #[test]
    fn drop_order_reverse_allocation() {
        let order = Cell::new(Vec::new());
        let arena = TypedArenaLazy::<DropTracker>::new(10);

        arena.try_alloc(DropTracker { id: 1, order: &order }).unwrap();
        arena.try_alloc(DropTracker { id: 2, order: &order }).unwrap();
        arena.try_alloc(DropTracker { id: 3, order: &order }).unwrap();

        drop(arena);

        assert_eq!(order.take(), vec![3, 2, 1]);
    }

    #[test]
    fn reset_drops_values_and_reuses_memory() {
        let order = Cell::new(Vec::new());

        let mut arena = TypedArenaLazy::<DropTracker>::new(10);
        let ptr1 = ptr::from_mut(arena.try_alloc(DropTracker { id: 1, order: &order }).unwrap());
        let _ptr2 = ptr::from_mut(arena.try_alloc(DropTracker { id: 2, order: &order }).unwrap());

        arena.reset();

        // After reset, all previous values must have been dropped.
        assert_eq!(order.take(), vec![2, 1]);

        // New allocation reuses the first slot.
        let ptr3 = ptr::from_mut(arena.try_alloc(DropTracker { id: 3, order: &order }).unwrap());
        assert_eq!(ptr1, ptr3);

        drop(arena);
        assert_eq!(order.take(), vec![3]);
    }

    #[test]
    fn no_double_drop_after_reset() {
        struct Counter<'a>(&'a Cell<u32>);
        impl Drop for Counter<'_> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let count = Cell::new(0u32);

        let mut arena = TypedArenaLazy::<Counter>::new(10);
        arena.try_alloc(Counter(&count)).unwrap();
        arena.try_alloc(Counter(&count)).unwrap();

        arena.reset();
        assert_eq!(count.get(), 2); // both dropped exactly once

        arena.try_alloc(Counter(&count)).unwrap();
        drop(arena);
        assert_eq!(count.get(), 3); // only the new one dropped
    }

    #[test]
    fn reset_clears_len_before_dropping_values() {
        struct PanicOnDrop<'a>(&'a Cell<u32>);
        impl Drop for PanicOnDrop<'_> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
                panic!("drop panic");
            }
        }

        let drops = Cell::new(0u32);
        let mut arena = mem::ManuallyDrop::new(TypedArenaLazy::<PanicOnDrop>::new(2));
        arena.try_alloc(PanicOnDrop(&drops)).unwrap();
        arena.try_alloc(PanicOnDrop(&drops)).unwrap();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| arena.reset()));

        assert!(result.is_err());
        assert_eq!(drops.get(), 1);
        assert_eq!(arena.len(), 0);
        unsafe {
            mem::ManuallyDrop::drop(&mut arena);
        }
    }

    #[test]
    fn oom_does_not_advance_offset() {
        let arena = TypedArenaLazy::<u64>::new(1); // holds exactly 1 u64
        assert!(arena.try_alloc(1u64).is_some());
        assert_eq!(arena.len(), 1);
        assert!(arena.try_alloc(2u64).is_none());
        assert_eq!(arena.len(), 1);
    }

    #[test]
    fn zst_does_not_advance_offset() {
        let arena = TypedArenaLazy::<()>::new(0);
        assert!(arena.try_alloc(()).is_some());
        assert_eq!(arena.len(), 0);
        assert!(arena.try_alloc(()).is_some());
        assert_eq!(arena.len(), 0);
    }

    #[test]
    fn allocated_value_is_valid() {
        let arena = TypedArenaLazy::<String>::new(1);
        let s = arena.try_alloc("hello".to_string()).unwrap();
        assert_eq!(s, "hello");
        s.push_str(" world");
        assert_eq!(s, "hello world");
    }

    #[test]
    #[should_panic(expected = "TypedArenaLazy: capacity overflow")]
    fn new_panics_on_capacity_overflow() {
        let _arena = TypedArenaLazy::<u16>::new(usize::MAX);
    }
}
