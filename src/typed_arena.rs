use core::cell::Cell;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr::{NonNull, drop_in_place};

/// A fixed‑capacity, single‑threaded arena that allocates values of
/// type `T` and automatically drops them in reverse allocation order.
///
/// The backing store is a `NonNull<[MaybeUninit<T>]>` whose capacity is
/// set at construction.  Each call to [`alloc_raw`] writes a value into
/// the next free slot and returns a mutable reference.  When the
/// arena is dropped (or when [`reset`] is called), all live values
/// are dropped and the memory is made available for reuse.
///
/// # Invariance and thread safety
///
/// `TypedArena<T>` is **invariant** in `T` (because `MaybeUninit<T>`
/// is invariant) and **`!Send + !Sync`** (the raw‑pointer marker
/// prevents cross‑thread usage).  This guarantees:
///
/// - No unsound subtyping (e.g., treating a `String` arena as a
///   `dyn Display` arena, which would break `Drop`).
/// - The arena is confined to a single thread.
///
/// # Examples
///
/// ```
/// use linalloc::TypedArena;
///
/// let mut arena = TypedArena::<String>::new(5);
///
/// let s = arena.alloc_raw("hello".to_string()).unwrap();
/// assert_eq!(s, "hello");
///
/// // All values are dropped when `arena` goes out of scope.
/// ```
///
/// [`alloc_raw`]: TypedArena::alloc_raw
/// [`reset`]: TypedArena::reset
pub struct TypedArena<T> {
    base: NonNull<[MaybeUninit<T>]>,
    offset: Cell<usize>,
    _invariant: PhantomData<*const ()>,
}

impl<T> TypedArena<T> {
    /// Creates a new typed arena that can hold up to `capacity`
    /// elements of type `T`.
    ///
    /// The backing memory is allocated but **uninitialised**.
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

    /// Allocates a new `T` by moving `value` into the arena.
    ///
    /// The returned mutable reference borrows the arena immutably
    /// (`&self`), so the arena is frozen (cannot be dropped or reset)
    /// until the reference goes out of scope.
    ///
    /// # Returns
    ///
    /// `None` if the arena is full (i.e., `len() == capacity()`).
    ///
    /// # Examples
    ///
    /// ```
    /// use linalloc::TypedArena;
    ///
    /// let arena = TypedArena::<i32>::new(10);
    /// let x = arena.alloc_raw(42).unwrap();
    /// assert_eq!(*x, 42);
    /// ```
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_raw(&self, value: T) -> Option<&mut T> {
        if size_of::<T>() == 0 {
            unsafe {
                let dangling = NonNull::<T>::dangling();
                dangling.as_ptr().write(value);
                return Some(&mut *dangling.as_ptr());
            }
        }

        let idx = self.offset.get();
        if idx >= self.capacity() {
            return None;
        }

        // Safety:
        // - `idx` is within the capacity -- the slot is valid.
        // - The slot is uninitialised -- `write` initialises it.
        // - The returned reference borrows `self` -- lifetime tied to the arena.
        unsafe {
            let beg = self.base.as_ptr().cast::<MaybeUninit<T>>();
            let slot = &mut *beg.add(idx);
            let r = slot.write(value);
            self.offset.set(idx + 1);
            Some(r)
        }
    }

    /// Returns the number of elements currently allocated in the arena.
    ///
    /// # Examples
    ///
    /// ```
    /// use linalloc::TypedArena;
    ///
    /// let mut arena = TypedArena::<i32>::new(10);
    /// assert_eq!(arena.len(), 0);
    /// arena.alloc_raw(1);
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
    /// use linalloc::TypedArena;
    ///
    /// let arena = TypedArena::<i32>::new(10);
    /// assert!(arena.is_empty());
    /// arena.alloc_raw(1);
    /// assert!(!arena.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the maximum number of elements the arena can hold.
    pub fn capacity(&self) -> usize {
        self.base.len()
    }

    /// Drops all live `T` values in reverse allocation order and
    /// resets the arena for reuse.
    ///
    /// Because this method takes `&mut self`, the borrow checker
    /// guarantees that no references to the arena's contents are
    /// currently alive.  After the call, `len()` returns `0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use linalloc::TypedArena;
    ///
    /// let mut arena = TypedArena::<Vec<i32>>::new(5);
    /// {
    ///     let v = arena.alloc_raw(vec![1, 2, 3]).unwrap();
    /// } // v goes out of scope -- can call `reset` now.
    /// arena.reset();
    /// assert_eq!(arena.len(), 0);
    /// ```
    pub fn reset(&mut self) {
        let offset = self.offset.get();
        unsafe {
            let start = self.base.as_ptr().cast::<MaybeUninit<T>>().cast::<T>();
            // Drop in reverse order per Rust's usual drop semantics.
            for i in (0..offset).rev() {
                drop_in_place(start.add(i));
            }
        }
        self.offset.set(0);
    }
}

impl<T> Drop for TypedArena<T> {
    fn drop(&mut self) {
        let offset = self.offset.get();
        unsafe {
            let start = self.base.as_ptr().cast::<MaybeUninit<T>>().cast::<T>();
            for i in (0..offset).rev() {
                drop_in_place(start.add(i));
            }
            drop(Box::from_raw(self.base.as_ptr()));
        }
    }
}

#[cfg(test)]
mod tests {
    use core::ptr;

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
        let arena = TypedArena::<DropTracker>::new(10);

        arena.alloc_raw(DropTracker { id: 1, order: &order }).unwrap();
        arena.alloc_raw(DropTracker { id: 2, order: &order }).unwrap();
        arena.alloc_raw(DropTracker { id: 3, order: &order }).unwrap();

        drop(arena);

        assert_eq!(order.take(), vec![3, 2, 1]);
    }

    #[test]
    fn reset_drops_values_and_reuses_memory() {
        let order = Cell::new(Vec::new());

        let mut arena = TypedArena::<DropTracker>::new(10);
        let ptr1 = ptr::from_mut(arena.alloc_raw(DropTracker { id: 1, order: &order }).unwrap());
        let _ptr2 = ptr::from_mut(arena.alloc_raw(DropTracker { id: 2, order: &order }).unwrap());

        arena.reset();

        // After reset, all previous values must have been dropped.
        assert_eq!(order.take(), vec![2, 1]);

        // New allocation reuses the first slot.
        let ptr3 = ptr::from_mut(arena.alloc_raw(DropTracker { id: 3, order: &order }).unwrap());
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

        let mut arena = TypedArena::<Counter>::new(10);
        arena.alloc_raw(Counter(&count)).unwrap();
        arena.alloc_raw(Counter(&count)).unwrap();

        arena.reset();
        assert_eq!(count.get(), 2); // both dropped exactly once

        arena.alloc_raw(Counter(&count)).unwrap();
        drop(arena);
        assert_eq!(count.get(), 3); // only the new one dropped
    }

    #[test]
    fn oom_does_not_advance_offset() {
        let arena = TypedArena::<u64>::new(1); // holds exactly 1 u64
        assert!(arena.alloc_raw(1u64).is_some());
        assert_eq!(arena.len(), 1);
        assert!(arena.alloc_raw(2u64).is_none());
        assert_eq!(arena.len(), 1);
    }

    #[test]
    fn zst_does_not_advance_offset() {
        let arena = TypedArena::<()>::new(0);
        assert!(arena.alloc_raw(()).is_some());
        assert_eq!(arena.len(), 0);
        assert!(arena.alloc_raw(()).is_some());
        assert_eq!(arena.len(), 0);
        drop(arena);
    }

    #[test]
    fn allocated_value_is_valid() {
        let arena = TypedArena::<String>::new(1);
        let s = arena.alloc_raw("hello".to_string()).unwrap();
        assert_eq!(s, "hello");
        s.push_str(" world");
        assert_eq!(s, "hello world");
    }
}
