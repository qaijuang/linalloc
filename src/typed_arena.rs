use core::alloc::Layout;
use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::mem::{self, needs_drop, size_of};
use core::ptr::{self, drop_in_place};

use crate::UninitAllocator;

/// A typed arena that allocates values of type `T` from a borrowed backing allocator.
///
/// Multiple `TypedArena`s can share the same underlying allocator,
/// allowing different types to be allocated in the same memory region
/// while being dropped independently. The backing allocator is specified by the
/// type parameter `A`, which must implement [`crate::UninitAllocator`].
///
/// Values allocated in this arena are automatically dropped in reverse
/// allocation order when the `TypedArena` is dropped or [`TypedArena::reset`] is called.
/// The memory in the backing allocator is **not** freed or rewound -- only the
/// objects’ destructors are executed. Reuse of the underlying memory is
/// governed by the allocator’s own life cycle (e.g., manually reset after all
/// `TypedArena`s have been dropped).
///
/// # Thread safety
///
/// `TypedArena` is **`!Send` and `!Sync`** because it contains a
/// raw pointer marker that prevents the value from leaving the thread where it was created.
/// This holds regardless of whether the backing allocator `A` is `Send` or `Sync`.
///
/// # Invariance
///
/// `TypedArena<T>` is **invariant** in `T`. The internal tracking list
/// contains `*mut T` pointers, which are invariant. This forbids
/// unsound subtyping (e.g., treating a `String` arena as a `dyn Display` arena),
/// which would otherwise break `Drop`.
///
/// # Examples
///
/// ```
/// use linalloc::{BumpArena, TypedArena};
///
/// let bump = BumpArena::new(4 * 1024);
///
/// {
///     let mut strings = TypedArena::<String, _>::new_in(&bump);
///     let mut ints = TypedArena::<i32, _>::new_in(&bump);
///
///     let s = strings.try_alloc("hello".to_string()).unwrap();
///     let i = ints.try_alloc(42).unwrap();
///     assert_eq!(*s, "hello");
///     assert_eq!(*i, 42);
///     // strings and ints are dropped here, values are destroyed.
/// }
///
/// // The bump memory is still allocated, but no live objects remain.
/// unsafe { bump.reset() }; // safe because all references have ended
/// ```
#[derive(Debug)]
pub struct TypedArena<'a, T, A: UninitAllocator> {
    allocator: &'a A,
    // Tracks the addresses of every allocated `T`. Interior mutability via
    // `UnsafeCell` allows pushing from `&self` during `alloc`. The list is
    // only read or cleared when we have `&mut self` (in `Drop` and `reset`).
    allocations: UnsafeCell<Vec<*mut T>>,
    // Makes the struct unconditionally `!Send + !Sync`.
    _marker: PhantomData<*const ()>,
}

impl<'a, T, A: UninitAllocator> TypedArena<'a, T, A> {
    /// Creates a new `TypedArena` that allocates objects inside the given
    /// backing allocator.
    ///
    /// The allocator must outlive the `TypedArena` and all references
    /// returned by [`TypedArena::try_alloc`].
    pub fn new_in(allocator: &'a A) -> Self {
        Self { allocator, allocations: UnsafeCell::new(Vec::new()), _marker: PhantomData }
    }

    /// Just like [`TypedArena::try_alloc`], but panics
    /// when allocation fails.
    ///
    /// # Panics
    ///
    /// if the backing allocator cannot satisfy the allocation request.
    pub fn alloc(&self, value: T) -> &mut T {
        self.alloc_impl(value).expect("TypedArena allocation failed")
    }

    /// Allocates a new `T` by moving `value` into the arena.
    ///
    /// The returned mutable reference borrows the `TypedArena` immutably
    /// (`&self`), so the arena is frozen (cannot be dropped or reset) until
    /// the reference goes out of scope. Multiple allocations can coexist
    /// without aliasing.
    ///
    /// Zero‑sized types (e.g., `()`) are handled specially: they consume no
    /// space in the backing allocator and always succeed.
    ///
    /// # Returns
    ///
    /// `None` if the backing allocator cannot satisfy the allocation
    /// request.
    ///
    /// # Examples
    ///
    /// ```
    /// use linalloc::{BumpArena, TypedArena};
    ///
    /// let bump = BumpArena::new(1024);
    /// let arena = TypedArena::<i32, _>::new_in(&bump);
    /// let x = arena.try_alloc(42).unwrap();
    /// assert_eq!(*x, 42);
    /// ```
    pub fn try_alloc(&self, value: T) -> Option<&mut T> {
        self.alloc_impl(value)
    }

    #[allow(clippy::mut_from_ref)]
    fn alloc_impl(&self, value: T) -> Option<&mut T> {
        if size_of::<T>() == 0 {
            unsafe {
                let dangling = ptr::NonNull::<T>::dangling();
                if needs_drop::<T>() {
                    (*self.allocations.get()).push(dangling.as_ptr());
                }
                dangling.as_ptr().write(value);
                return Some(&mut *dangling.as_ptr());
            }
        }

        let layout = Layout::new::<T>();
        let slice = self.allocator.try_alloc_uninit(layout)?;
        let ptr = slice.as_mut_ptr().cast::<T>();

        unsafe {
            // Push the pointer into the tracking list. Because this method
            // takes `&self`, we need interior mutability -- `UnsafeCell` gives
            // us a unique access path that does not alias with any &mut borrow
            // of the arena (which would require `&mut self`).
            (*self.allocations.get()).push(ptr);
            // Initialise the freshly allocated memory after tracking succeeds.
            ptr.write(value);
            // Return a mutable reference that borrows `self`, freezing the
            // arena while the reference is alive.
            Some(&mut *ptr)
        }
    }

    /// Returns a reference to the backing allocator.
    pub fn allocator(&self) -> &A {
        self.allocator
    }

    /// Returns the number of elements currently allocated in this arena.
    pub fn len(&self) -> usize {
        // Safety: we only read the length, which is a plain integer access
        // that does not alias with any other operation. The `UnsafeCell`
        // guarantees that this is a valid read.
        unsafe { (*self.allocations.get()).len() }
    }

    /// Returns `true` if the arena contains no allocated elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Consumes the arena and returns an iterator that drains all allocated
    /// values in **allocation order** (FIFO).
    ///
    /// This method does **not** allocate extra memory. The backing allocator
    /// remains borrowed for the iterator’s lifetime, preventing it from being
    /// dropped or reset until iteration is complete.
    ///
    /// # Examples
    ///
    /// ```
    /// use linalloc::{BumpArena, TypedArena};
    ///
    /// let bump = BumpArena::new(128);
    /// let mut arena = TypedArena::<String, _>::new_in(&bump);
    /// arena.try_alloc("first".to_string()).unwrap();
    /// arena.try_alloc("second".to_string()).unwrap();
    ///
    /// let mut d = arena.drain();
    /// assert_eq!(d.next(), Some("first".to_string()));
    /// assert_eq!(d.next(), Some("second".to_string()));
    /// assert_eq!(d.next(), None);
    /// ```
    pub fn drain(self) -> DrainIter<'a, T, A> {
        // Disable the arena's own Drop.
        let this = mem::ManuallyDrop::new(self);

        let allocations: Vec<*mut T> = unsafe { ptr::read(this.allocations.get()) };

        DrainIter { pointers: allocations.into_iter(), _allocator: this.allocator }
    }

    /// Drops all live `T` values in reverse allocation order and clears the
    /// tracking list.
    ///
    /// Because this method takes `&mut self`, the borrow checker guarantees
    /// that no references to the arena’s contents are currently alive.
    /// After the call, [`TypedArena::len`] returns `0`.
    ///
    /// The memory in the backing allocator is **not** freed or rewound -- only
    /// the destructors of the allocated values are executed. Future
    /// allocations request fresh memory from the backing allocator. Reuse is
    /// governed by that allocator’s own reset/drop lifecycle.
    ///
    /// # Examples
    ///
    /// ```
    /// use linalloc::{BumpArena, TypedArena};
    ///
    /// let bump = BumpArena::new(1024);
    /// let mut arena = TypedArena::<Vec<i32>, _>::new_in(&bump);
    /// arena.try_alloc(vec![1, 2, 3]).unwrap();
    /// // No references are alive, so reset is safe.
    /// arena.reset();
    /// assert!(arena.is_empty());
    /// ```
    pub fn reset(&mut self) {
        let allocs = self.allocations.get_mut();
        // Drop in reverse order, mirroring Rust’s own drop semantics.
        while let Some(ptr) = allocs.pop() {
            unsafe {
                drop_in_place(ptr);
            }
        }
    }
}

impl<T, A: UninitAllocator> Drop for TypedArena<'_, T, A> {
    fn drop(&mut self) {
        self.reset();
    }
}

/// Yields `T` in **allocation order** (FIFO).
/// Created by [`TypedArena::drain`].
///
/// This iterator does **not** allocate extra memory -- it reuses the arena’s
/// internal tracking list. The backing allocator remains borrowed for the
/// iterator’s lifetime, preventing premature deallocation. If dropped before
/// fully consumed, remaining elements are destroyed in **reverse allocation
/// order** (LIFO), mirroring Rust own's drop semantics.
pub struct DrainIter<'a, T, A: UninitAllocator> {
    // The remaining raw pointers, taken from the arena's tracking list.
    pointers: std::vec::IntoIter<*mut T>,
    // Keeps the backing allocator alive.
    _allocator: &'a A,
}

impl<T, A: UninitAllocator> Iterator for DrainIter<'_, T, A> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let ptr = self.pointers.next()?;
        // SAFETY:
        // > `ptr` is non‑null and properly aligned for `T` (guaranteed by
        //   the allocator and the arena's tracking).
        // > The memory pointed to is still live because `_allocator` keeps
        //   the allocator borrowed.
        // > No other reference to this memory exists -- the arena is consumed.
        Some(unsafe { ptr::read(ptr) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.pointers.size_hint()
    }
}

impl<T, A: UninitAllocator> ExactSizeIterator for DrainIter<'_, T, A> {
    fn len(&self) -> usize {
        self.pointers.len()
    }
}

impl<T, A: UninitAllocator> core::iter::FusedIterator for DrainIter<'_, T, A> {}

impl<T, A: UninitAllocator> Drop for DrainIter<'_, T, A> {
    fn drop(&mut self) {
        let remaining = self.pointers.as_slice();
        // Same order as `TypedArena::reset`.
        for &ptr in remaining.iter().rev() {
            // SAFETY: ptr is valid, unique, and the allocator is still alive.
            unsafe {
                drop_in_place(ptr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use core::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::BumpArena;

    #[test]
    fn zst_with_drop_is_dropped_when_arena_drops() {
        static DROPS: AtomicUsize = AtomicUsize::new(0);

        struct Zst;

        impl Drop for Zst {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::Relaxed);
            }
        }

        DROPS.store(0, Ordering::Relaxed);
        let bump = BumpArena::new(0);
        {
            let arena = TypedArena::<Zst, _>::new_in(&bump);
            assert!(arena.try_alloc(Zst).is_some());
            assert!(arena.try_alloc(Zst).is_some());
            assert_eq!(arena.len(), 2);
        }

        assert_eq!(DROPS.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn reset_removes_pointer_before_dropping_value() {
        static DROPS: AtomicUsize = AtomicUsize::new(0);

        struct PanicOnFirstDrop(u8);

        impl Drop for PanicOnFirstDrop {
            fn drop(&mut self) {
                let _ = self.0;
                assert!(DROPS.fetch_add(1, Ordering::Relaxed) != 0, "drop panic");
            }
        }

        DROPS.store(0, Ordering::Relaxed);
        let bump = BumpArena::new(128);
        let mut arena = TypedArena::<PanicOnFirstDrop, _>::new_in(&bump);
        arena.try_alloc(PanicOnFirstDrop(1)).unwrap();
        arena.try_alloc(PanicOnFirstDrop(2)).unwrap();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| arena.reset()));

        assert!(result.is_err());
        assert_eq!(DROPS.load(Ordering::Relaxed), 1);
        assert_eq!(arena.len(), 1);

        drop(arena);
        assert_eq!(DROPS.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn reset_does_not_rewind_the_backing_allocator() {
        let bump = BumpArena::new(size_of::<u64>());
        let mut arena = TypedArena::<u64, _>::new_in(&bump);

        assert!(arena.try_alloc(1).is_some());
        arena.reset();

        assert!(arena.try_alloc(2).is_none());
        assert_eq!(arena.len(), 0);
    }
}
