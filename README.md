# linalloc (Linear Allocator)

[![miri](https://github.com/qaijuang/linalloc/actions/workflows/ci.yml/badge.svg)](https://github.com/qaijuang/linalloc/actions/workflows/ci.yml)

[![license](https://img.shields.io/github/license/qaijuang/linalloc)]

Allocator primitives for single-threaded, fixed-capacity arenas.

This crate provides two arena types:

- [`BumpArena`] allocates untyped memory and **DOES NOT** automatically drop values.
- [`TypedArena`] allocates values of a specific type `T` and drops them in reverse allocation order.

## Contributing

Security reports, bug fixes, test and documentation improvements are very welcome. Please open an issue or a pull request.
