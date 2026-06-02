#![warn(clippy::pedantic)]
#![doc = include_str!("../README.md")]

mod bump_arena;
#[cfg(feature = "lazy")]
mod bump_arena_lazy;
#[allow(dead_code, reason = "work in progress")]
pub(crate) mod sys;
mod typed_arena;

pub use bump_arena::*;
#[cfg(feature = "lazy")]
pub use bump_arena_lazy::*;
pub use typed_arena::*;
