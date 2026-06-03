#![warn(clippy::pedantic)]
#![doc = include_str!("../README.md")]

mod bump_arena;
#[cfg(feature = "lazy")]
mod bump_arena_lazy;
#[cfg(feature = "lazy")]
pub(crate) mod sys;
mod typed_arena;
#[cfg(feature = "lazy")]
mod typed_arena_lazy;

pub use bump_arena::*;
#[cfg(feature = "lazy")]
pub use bump_arena_lazy::*;
pub use typed_arena::*;
#[cfg(feature = "lazy")]
pub use typed_arena_lazy::*;
