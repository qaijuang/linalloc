use core::alloc::Layout;
use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::mem::{self, size_of};
use core::ptr::{self, drop_in_place};
use std::vec;

use crate::UninitAllocator;

/// A typed arena that allocates values of type `T` from a borrowed backing allocator.
///
/// Multiple `TypedArenaRef`s can share the same underlying allocator,
/// allowing different types to be allocated in the same memory region
/// while being dropped independently. The backing allocator is specified by the
/// type parameter `A`, which must implement [`crate::UninitAllocator`].
///
/// Values allocated in this arena are automatically dropped in reverse
/// allocation order when the `TypedArenaRef` is dropped or [`TypedArenaRef::reset`] is called.
/// The memory in the backing allocator is **not** freed -- only the objects’
/// destructors are executed. Reuse of the underlying memory is governed by the
/// allocator’s own life cycle (e.g., manually reset after all
/// `TypedArenaRef`s have been dropped).
///
/// # Thread safety
///
/// `TypedArenaRef` is **`!Send` and `!Sync`** because it contains a
/// raw pointer marker that prevents the value from leaving the thread where it was created.
/// This holds regardless of whether the backing allocator `A` is `Send` or `Sync`.
///
/// # Invariance
///
/// `TypedArenaRef<T>` is **invariant** in `T`. The internal tracking list
/// contains `*mut T` pointers, which are invariant. This forbids
/// unsound subtyping (e.g., treating a `String` arena as a `dyn Display` arena),
/// which would otherwise break `Drop`.
///
/// # Examples
///
/// ```
/// use linalloc::{BumpArena, TypedArenaRef};
///
/// let bump = BumpArena::new(4 * 1024);
///
/// {
///     let mut strings = TypedArenaRef::<String, _>::new_in(&bump);
///     let mut ints = TypedArenaRef::<i32, _>::new_in(&bump);
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
pub struct TypedArenaRef<'a, T, A: UninitAllocator + 'a> {
    allocator: &'a A,
    // Tracks the addresses of every allocated `T`. Interior mutability via
    // `UnsafeCell` allows pushing from `&self` during `alloc`. The list is
    // only read or cleared when we have `&mut self` (in `Drop` and `reset`).
    allocations: UnsafeCell<Vec<*mut T>>,
    // Makes the struct unconditionally `!Send + !Sync`.
    _marker: PhantomData<*const ()>,
}

impl<'a, T, A: UninitAllocator + 'a> TypedArenaRef<'a, T, A> {
    /// Creates a new `TypedArenaRef` that allocates objects inside the given
    /// backing allocator.
    ///
    /// The allocator must outlive the `TypedArenaRef` and all references
    /// returned by [`TypedArenaRef::try_alloc`].
    pub fn new_in(allocator: &'a A) -> Self {
        Self { allocator, allocations: UnsafeCell::new(Vec::new()), _marker: PhantomData }
    }

    /// Just like [`TypedArenaRef::try_alloc`], but panics
    /// when allocation fails.
    ///
    /// # Panics
    ///
    /// if the backing allocator cannot satisfy the allocation request.
    pub fn alloc(&self, value: T) -> &mut T {
        self.alloc_impl(value).expect("TypedArenaRef allocation failed")
    }

    /// Allocates a new `T` by moving `value` into the arena.
    ///
    /// The returned mutable reference borrows the `TypedArenaRef` immutably
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
    /// use linalloc::{BumpArena, TypedArenaRef};
    ///
    /// let bump = BumpArena::new(1024);
    /// let arena = TypedArenaRef::<i32, _>::new_in(&bump);
    /// let x = arena.try_alloc(42).unwrap();
    /// assert_eq!(*x, 42);
    /// ```
    pub fn try_alloc(&self, value: T) -> Option<&mut T> {
        self.alloc_impl(value)
    }

    #[allow(clippy::mut_from_ref)]
    fn alloc_impl(&self, value: T) -> Option<&mut T> {
        // Zero‑sized types never consume memory -- they just write to a
        // dangling pointer. No tracking is necessary because ZSTs have no
        // destructors and no drop glue.
        if size_of::<T>() == 0 {
            unsafe {
                let dangling = ptr::NonNull::<T>::dangling();
                dangling.as_ptr().write(value);
                return Some(&mut *dangling.as_ptr());
            }
        }

        let layout = Layout::new::<T>();
        let slice = self.allocator.try_alloc_uninit(layout)?;
        let ptr = slice.as_mut_ptr().cast::<T>();

        unsafe {
            // Initialise the freshly allocated memory.
            ptr.write(value);
            // Push the pointer into the tracking list. Because this method
            // takes `&self`, we need interior mutability -- `UnsafeCell` gives
            // us a unique access path that does not alias with any &mut borrow
            // of the arena (which would require `&mut self`).
            (*self.allocations.get()).push(ptr);
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

    /// Drops all live `T` values in reverse allocation order and clears the
    /// tracking list.
    ///
    /// Because this method takes `&mut self`, the borrow checker guarantees
    /// that no references to the arena’s contents are currently alive.
    /// After the call, [`TypedArenaRef::len`] returns `0`.
    ///
    /// The memory in the backing allocator is **not** freed -- only the
    /// destructors of the allocated values are executed. The memory can be
    /// reused by further allocations through this `TypedArenaRef` (or other
    /// borrows of the same allocator), but it will not be physically released until
    /// the allocator is dropped or reset.
    ///
    /// # Examples
    ///
    /// ```
    /// use linalloc::{BumpArena, TypedArenaRef};
    ///
    /// let bump = BumpArena::new(1024);
    /// let mut arena = TypedArenaRef::<Vec<i32>, _>::new_in(&bump);
    /// arena.try_alloc(vec![1, 2, 3]).unwrap();
    /// // No references are alive, so reset is safe.
    /// arena.reset();
    /// assert!(arena.is_empty());
    /// ```
    pub fn reset(&mut self) {
        let allocs = self.allocations.get_mut();
        // Drop in reverse order, mirroring Rust’s own drop semantics.
        for &ptr in allocs.iter().rev() {
            unsafe {
                drop_in_place(ptr);
            }
        }
        allocs.clear();
    }
}

impl<'a, T, A: UninitAllocator + 'a> Drop for TypedArenaRef<'a, T, A> {
    fn drop(&mut self) {
        self.reset();
    }
}

/// An iterator that yields the elements of a consumed [`TypedArenaRef`].
///
/// This struct is created by the [`TypedArenaRef::into_iter`]
/// (provided by the [`IntoIterator`] trait). It yields the allocated values
/// in **reverse allocation order** -- the most recently allocated element is
/// returned first.
///
/// # Thread safety
///
/// `IntoIter` is **`!Send` and `!Sync`** -- the arena is single‑threaded and
/// this iterator must remain on the thread where it was created.
pub struct IntoIter<'a, T, A: UninitAllocator + 'a> {
    // keeps the backing memory alive
    allocator: &'a A,
    // the remaining raw pointers (in reverse order)
    pointers: vec::IntoIter<*mut T>,
}

impl<'a, T, A: UninitAllocator + 'a> IntoIter<'a, T, A> {
    /// Returns a reference to the backing allocator.
    #[must_use]
    pub fn allocator(&self) -> &A {
        self.allocator
    }
}

impl<'a, T, A: UninitAllocator + 'a> Iterator for IntoIter<'a, T, A> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let ptr = self.pointers.next()?;
        // SAFETY: The pointer comes from the tracking list and points to a
        // valid, initialised `T`. The backing memory is still alive (guaranteed
        // by `allocator`). No other reference to this memory exists.
        Some(unsafe { ptr::read(ptr) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.pointers.size_hint()
    }
}

impl<'a, T, A: UninitAllocator + 'a> Drop for IntoIter<'a, T, A> {
    fn drop(&mut self) {
        // Drop any remaining elements to prevent leaks.
        for ptr in self.pointers.by_ref() {
            // SAFETY: The pointer is valid and points to an initialised `T`.
            unsafe {
                ptr::drop_in_place(ptr);
            }
        }
    }
}

impl<'a, T, A: UninitAllocator + 'a> IntoIterator for TypedArenaRef<'a, T, A> {
    type Item = T;
    type IntoIter = IntoIter<'a, T, A>;

    /// Consumes the arena and returns an iterator over its allocated values in
    /// reverse allocation order.
    ///
    /// The iterator yields **owned** values. The memory in the backing
    /// allocator is **not** freed -- it stays reserved and can be reused after
    /// a manual reset of the underlying bump allocator (once all references to
    /// the memory have ended).
    ///
    /// If the iterator is dropped before it is fully consumed, the remaining
    /// elements are automatically dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// use linalloc::{BumpArena, TypedArenaRef};
    ///
    /// let bump = BumpArena::new(128);
    /// let mut arena = TypedArenaRef::<String, _>::new_in(&bump);
    ///
    /// arena.try_alloc("foo".to_string()).unwrap();
    /// arena.try_alloc("bar".to_string()).unwrap();
    ///
    /// // Consume the arena.
    /// let s = arena.into_iter().collect::<Vec<_>>();
    /// assert_eq!(s, ["bar", "foo"]);
    /// ```
    fn into_iter(self) -> IntoIter<'a, T, A> {
        // Prevent the TypedArenaRef's Drop from running -- we are taking over
        // ownership of the tracking list and the responsibility for dropping
        // the values.
        let this = mem::ManuallyDrop::new(self);

        // SAFETY: `allocations` is a `UnsafeCell<Vec<*mut T>>` and we have
        // exclusive access to the arena (it is being consumed). Moving the
        // `Vec` out of the `UnsafeCell` is sound.
        let allocations = unsafe { ptr::read(this.allocations.get()) };

        // Build a vector of pointers in reverse allocation order so that the
        // iterator yields the most recently allocated element first.
        let pointers: Vec<*mut T> = allocations.into_iter().rev().collect();

        IntoIter { allocator: this.allocator, pointers: pointers.into_iter() }
    }
}

impl<'a, T, A: UninitAllocator + 'a> IntoIterator for &'a TypedArenaRef<'a, T, A> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    /// Returns an iterator over the allocated values in reverse allocation
    /// order (LIFO).
    ///
    /// The iterator yields `&T` and borrows the arena immutably. New
    /// allocations are possible while the iterator is alive, but they will
    /// not appear in this iterator because it captures a snapshot of the
    /// current state.
    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T, A: UninitAllocator + 'a> IntoIterator for &'a mut TypedArenaRef<'a, T, A> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    /// Returns a mutable iterator over the allocated values in reverse
    /// allocation order (LIFO).
    ///
    /// The iterator yields `&mut T` and borrows the arena mutably. No other
    /// allocations or resets are possible while the iterator is alive.
    fn into_iter(self) -> IterMut<'a, T> {
        self.iter_mut()
    }
}

impl<'a, T, A: UninitAllocator + 'a> DoubleEndedIterator for IntoIter<'a, T, A> {
    /// Removes and returns an element from the back of the iterator.
    ///
    /// This yields the elements in **forward allocation order** -- the first
    /// allocated element is returned first, which is the reverse of the
    /// default iteration order (LIFO).
    ///
    /// # Examples
    ///
    /// ```
    /// use linalloc::{BumpArena, TypedArenaRef};
    ///
    /// let bump = BumpArena::new(128);
    /// let mut arena = TypedArenaRef::<String, _>::new_in(&bump);
    /// arena.try_alloc("first".to_string()).unwrap();
    /// arena.try_alloc("second".to_string()).unwrap();
    ///
    /// let mut iter = arena.into_iter();
    /// assert_eq!(iter.next_back(), Some("first".to_string())); // forward
    /// assert_eq!(iter.next(), Some("second".to_string())); // LIFO
    /// ```
    fn next_back(&mut self) -> Option<T> {
        let ptr = self.pointers.next_back()?;
        // SAFETY: The pointer is valid and points to an initialised `T`.
        // See `next` for the full safety argument.
        Some(unsafe { ptr::read(ptr) })
    }
}

/// An immutable iterator over the elements of a [`TypedArenaRef`].
///
/// This iterator yields `&T` in **reverse allocation order** (LIFO).
/// It borrows the arena immutably, so the arena cannot be dropped while
/// the iterator exists. New allocations **are** allowed; they will not
/// appear in this iterator because it operates on a snapshot of the
/// tracking list taken at creation time.
///
/// # Thread safety
///
/// `Iter` is `!Send` and `!Sync` -- the contained raw pointers are neither
/// `Send` nor `Sync`, keeping the iterator confined to a single thread.
pub struct Iter<'s, T> {
    // Snapshotted pointers in reverse order.
    pointers: vec::IntoIter<*const T>,
    // Ensures the arena cannot be dropped while the iterator exists.
    _marker: PhantomData<&'s ()>,
}

impl<'s, T: 's> Iterator for Iter<'s, T> {
    type Item = &'s T;

    fn next(&mut self) -> Option<&'s T> {
        let ptr = self.pointers.next()?;
        // SAFETY: The pointer comes from a snapshot of the tracking list,
        // which contains only valid, initialised `T`s. The arena cannot be
        // dropped (we hold `&self`) and `reset` requires `unsafe` or
        // `&mut self`, so the memory remains stable.
        Some(unsafe { &*ptr })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.pointers.size_hint()
    }
}

impl<'s, T: 's> DoubleEndedIterator for Iter<'s, T> {
    fn next_back(&mut self) -> Option<&'s T> {
        let ptr = self.pointers.next_back()?;
        Some(unsafe { &*ptr })
    }
}

/// A mutable iterator over the elements of a [`TypedArenaRef`].
///
/// This iterator yields `&mut T` in **reverse allocation order** (LIFO).
/// Because it takes `&mut self` on the arena, the borrow checker prevents
/// any additional allocations or resets while the iterator is alive.
///
/// # Thread safety
///
/// `IterMut` is `!Send` and `!Sync` for the same reasons as [`Iter`].
pub struct IterMut<'s, T> {
    // Borrows the tracking list directly.
    slice: &'s [*mut T],
    // Position from the end of the slice.
    pos: usize,
}

impl<'s, T> Iterator for IterMut<'s, T> {
    type Item = &'s mut T;

    fn next(&mut self) -> Option<&'s mut T> {
        if self.pos == 0 {
            return None;
        }
        self.pos -= 1;
        let ptr = self.slice[self.pos];
        // SAFETY: We have exclusive access to the arena (&mut self).
        // The pointer is valid and unique.
        Some(unsafe { &mut *ptr })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.pos, Some(self.pos))
    }
}

impl<'a, T, A: UninitAllocator + 'a> TypedArenaRef<'a, T, A> {
    /// Returns an immutable iterator over all allocated elements, in
    /// **reverse allocation order** (LIFO).
    ///
    /// The iterator yields `&T` and borrows the arena immutably. New
    /// allocations are possible while the iterator is alive, but they will
    /// not appear in this iterator because it captures a snapshot of the
    /// current state.
    ///
    /// # Examples
    ///
    /// ```
    /// use linalloc::{BumpArena, TypedArenaRef};
    ///
    /// let bump = BumpArena::new(128);
    /// let mut arena = TypedArenaRef::<String, _>::new_in(&bump);
    /// arena.try_alloc("foo".to_string()).unwrap();
    /// arena.try_alloc("bar".to_string()).unwrap();
    ///
    /// let mut iter = arena.iter();
    /// assert_eq!(iter.next(), Some(&"bar".to_string())); // LIFO
    /// assert_eq!(iter.next(), Some(&"foo".to_string()));
    /// assert_eq!(iter.next(), None);
    ///
    /// // A new allocation does not affect `iter`.
    /// let _third = arena.try_alloc("baz".to_string()).unwrap();
    /// ```
    pub fn iter(&self) -> Iter<'_, T> {
        let allocations = unsafe { &*self.allocations.get() };
        // Clone the pointers and reverse to get LIFO order.
        let mut pointers: Vec<*const T> =
            allocations.iter().copied().map(<*mut T>::cast_const).collect();
        pointers.reverse();
        Iter { pointers: pointers.into_iter(), _marker: PhantomData }
    }

    /// Returns a mutable iterator over all allocated elements, in
    /// **reverse allocation order** (LIFO).
    ///
    /// This method requires `&mut self`, so no other allocations or resets
    /// are possible while the iterator is alive. The iterator yields
    /// `&mut T`, giving exclusive access to each element.
    ///
    /// # Examples
    ///
    /// ```
    /// use linalloc::{BumpArena, TypedArenaRef};
    ///
    /// let bump = BumpArena::new(128);
    /// let mut arena = TypedArenaRef::<String, _>::new_in(&bump);
    /// arena.try_alloc("foo".to_string()).unwrap();
    /// arena.try_alloc("bar".to_string()).unwrap();
    ///
    /// for s in arena.iter_mut() {
    ///     *s = format!("new-{s}");
    /// }
    /// ```
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        let allocs = self.allocations.get_mut();
        let len = allocs.len();
        IterMut { slice: allocs.as_slice(), pos: len }
    }
}
