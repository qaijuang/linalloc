#![warn(clippy::pedantic)]
#![doc = include_str!("../README.md")]

#[cfg(all(feature = "lazy", not(any(unix, windows))))]
compile_error!("the `lazy` feature is currently supported only on Unix and Windows targets");

mod bump_arena;
#[cfg(all(feature = "lazy", any(unix, windows)))]
mod bump_arena_lazy;
#[cfg(all(feature = "lazy", any(unix, windows)))]
pub(crate) mod sys;
mod typed_arena;
#[cfg(all(feature = "lazy", any(unix, windows)))]
mod typed_arena_lazy;
mod typed_arena_ref;

pub use bump_arena::*;
#[cfg(all(feature = "lazy", any(unix, windows)))]
pub use bump_arena_lazy::*;
pub use typed_arena::*;
#[cfg(all(feature = "lazy", any(unix, windows)))]
pub use typed_arena_lazy::*;
pub use typed_arena_ref::*;

/// An untyped allocator that provides mutable slices of uninitialised memory.
///
/// Types implementing this trait can serve as the backing store for
/// [`TypedArenaRef`], which adds automatic destructor execution and
/// type‑safe allocation on top of the raw memory.
///
/// # Safety
///
/// Implementors must uphold the following invariants. Violating any of them
/// will cause **undefined behaviour** in safe code that uses [`TypedArenaRef`].
///
/// - **Alignment** -- every slice returned by [`UninitAllocator::alloc_uninit_slice`] is aligned
///   to at least the requested `layout.align()`.
/// - **No overlap** -- the memory regions handed out never overlap. For
///   example, a bump allocator achieves this by monotonically advancing a
///   pointer; other strategies are possible as long as they guarantee
///   disjointness.
/// - **Stable addresses** -- the memory backing a returned slice remains valid
///   and at the same address until the allocator is dropped or explicitly
///   reset. The allocator must never invalidate previously‑returned pointers
///   while they may still be in use.
/// - **Single‑threaded confinement** -- the allocator is designed for
///   single‑threaded use. Even if the type implements `Sync`, the user
///   **must not** share a reference to the allocator across threads. Doing
///   so can cause data races on the allocator’s internal state and lead to
///   undefined behaviour.
pub unsafe trait UninitAllocator {
    /// Allocates a mutable slice of [`MaybeUninit<u8>`] that satisfies
    /// `layout`.
    ///
    /// Returns `None` if the allocator cannot satisfy the request.
    #[allow(clippy::mut_from_ref)]
    fn alloc_uninit_slice(
        &self,
        layout: core::alloc::Layout,
    ) -> Option<&mut [core::mem::MaybeUninit<u8>]>;
}

#[cfg(all(test, feature = "lazy", any(unix, windows)))]
mod tests {
    use std::io::Error;

    use super::{BumpArenaLazy, TypedArenaLazy};

    #[test]
    fn try_new_reports_os_error_for_unreservable_capacity() {
        let err = BumpArenaLazy::try_new(usize::MAX).unwrap_err();
        assert_eq!(Some(err), Error::last_os_error().raw_os_error());
        assert_ne!(err, 0);

        let err = TypedArenaLazy::<u8>::try_new(usize::MAX).unwrap_err();
        assert_eq!(Some(err), Error::last_os_error().raw_os_error());
        assert_ne!(err, 0);
    }
}
