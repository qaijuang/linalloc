#![warn(clippy::pedantic)]
#![cfg_attr(feature = "nightly", feature(allocator_api))]
#![doc = include_str!("../README.md")]

#[cfg(not(any(unix, windows)))]
compile_error!("`linalloc` only supports Unix and Windows targets");

mod bump_arena;
mod typed_arena;

pub(crate) mod sys;

pub use bump_arena::*;
pub use typed_arena::*;

/// An untyped allocator that provides mutable slices of uninitialised memory.
///
/// Types implementing this trait can serve as the backing store for
/// [`TypedArena`], which adds automatic destructor execution and
/// type‑safe allocation on top of the raw memory.
///
/// # Safety
///
/// Implementors must uphold the following invariants. Violating any of them
/// will cause **undefined behaviour** in safe code that uses [`TypedArena`].
///
/// - **Alignment** -- every slice returned by [`UninitAllocator::try_alloc_uninit`] is aligned
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
    /// Allocates a mutable slice of [`core::mem::MaybeUninit`] that satisfies
    /// `layout`.
    ///
    /// Returns `None` if the allocator cannot satisfy the request.
    #[allow(clippy::mut_from_ref)]
    fn try_alloc_uninit(
        &self,
        layout: core::alloc::Layout,
    ) -> Option<&mut [core::mem::MaybeUninit<u8>]>;
}
