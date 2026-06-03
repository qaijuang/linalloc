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

pub use bump_arena::*;
#[cfg(all(feature = "lazy", any(unix, windows)))]
pub use bump_arena_lazy::*;
pub use typed_arena::*;
#[cfg(all(feature = "lazy", any(unix, windows)))]
pub use typed_arena_lazy::*;
