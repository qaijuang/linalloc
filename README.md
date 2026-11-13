# linalloc (Linear Allocator)

[![CI](https://github.com/qaijuang/linalloc/actions/workflows/ci.yml/badge.svg)](https://github.com/qaijuang/linalloc/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/crates/msrv/linalloc)](https://crates.io/crates/linalloc)
[![crates.io](https://img.shields.io/crates/v/linalloc)](https://crates.io/crates/linalloc)
[![docs.rs](https://img.shields.io/docsrs/linalloc)](https://docs.rs/linalloc)
[![license](https://img.shields.io/github/license/qaijuang/linalloc)](https://github.com/qaijuang/linalloc/blob/main/LICENSE)

A fixed-capacity bump allocator backed by virtual memory.

It reserves address space up front and makes pages writable as needed.
For single-threaded Rust on Unix and Windows.

## Typed allocation

`alloc` stores a value and returns `&mut T`. `try_alloc` returns `None` on
failure and drops the rejected value. Stored values aren't dropped automatically.

```rust
use linalloc::BumpArena;

let arena = BumpArena::new(128);
let source = String::from("hello");
let number = arena.alloc(42_u64);
let text = arena.try_alloc(source.as_str()).unwrap();

assert_eq!(*number, 42);
assert_eq!(*text, "hello");
```

For delayed initialization, allocate a `MaybeUninit<T>` and call `write`.

## Raw allocation

`alloc_uninit` and `try_alloc_uninit` return uninitialized bytes for a `Layout`.
Initialize the memory before reading it.

```rust
use core::alloc::Layout;

use linalloc::BumpArena;

let arena = BumpArena::new(128);
let slot = arena.alloc_uninit(Layout::new::<u64>());
let ptr = slot.as_mut_ptr().cast::<u64>();

unsafe {
    ptr.write(42);
    assert_eq!(*ptr, 42);
}
```

## Vec support

On Rust 1.100 and later, `Vec` can allocate through `&arena`.
Growing a `Vec` may move its buffer.

```rust
use linalloc::BumpArena;

let arena = BumpArena::new(128);
let mut values = Vec::with_capacity_in(1, &arena);
values.push(1);
values.try_reserve(1).unwrap();
values.push(2);

assert_eq!(&values, &[1, 2]);
```

## OS errors

`try_new` returns the OS error code if reservation fails.
`last_os_error_code` reports the last commit failure for an existing arena.
It returns `None` until a commit fails. Success, reset, and capacity exhaustion
don't clear it. Use `std::io::Error::from_raw_os_error(code)` to format the code.

## Safety

The arena doesn't run destructors for stored values. Drop values that own
resources before resetting or dropping the arena.

`reset` is unsafe. End all borrows and drop stored values and allocator-backed
collections first. Don't use old references or pointers afterward.

## Testing

```sh
cargo test
cargo miri test
```

Under Miri, `linalloc-sys` uses aligned heap storage instead of OS mappings.
This checks the arena's memory operations, not page protection, lazy commitment,
or OS error codes. Native tests use the OS backend.
