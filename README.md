# linalloc (Linear Allocator)

[![CI](https://github.com/qaijuang/linalloc/actions/workflows/ci.yml/badge.svg)](https://github.com/qaijuang/linalloc/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/crates/msrv/linalloc)](https://crates.io/crates/linalloc)
[![crates.io](https://img.shields.io/crates/v/linalloc)](https://crates.io/crates/linalloc)
[![docs.rs](https://img.shields.io/docsrs/linalloc)](https://docs.rs/linalloc)
[![license](https://img.shields.io/github/license/qaijuang/linalloc)](https://github.com/qaijuang/linalloc/blob/main/LICENSE)

Small, fixed-capacity arena allocator for single-threaded Rust programs.

You pick the capacity up front. The arena capacity never grows.
Addresses stay stable. When it is full, fallible allocation returns `None`.

## Choose an arena

| Type                   | What it gives you                                | Drop behavior                                 |
| ---------------------- | ------------------------------------------------ | --------------------------------------------- |
| `BumpArena`            | Raw byte allocation from reserved virtual memory | Values must be dropped by the caller          |
| `TypedArena<'a, T, A>` | Values of one type from a backing allocator      | Drops live values in reverse allocation order |

## Feature flags

- `nightly` requires a nightly Rust toolchain and enables the unstable
  standard-library `allocator_api` implementation for `BumpArena`.

## Allocation APIs

Both arenas expose `try_*` for fallible allocation and `alloc` / `alloc_*` for the
panicking variant. The older inherent methods, `TypedArena::alloc_raw`, `BumpArena::alloc_uninit_slice`, and
`BumpArena::alloc_uninit_slice`, remain available for compatibility but are
deprecated.

## Using bump arena

`BumpArena` gives you uninitialized bytes. You choose the layout, initialize
the memory, and drop any values you place there.

```rust
use core::alloc::Layout;

use linalloc::BumpArena;

let arena = BumpArena::new(128);
let slot = arena.try_alloc_uninit(Layout::new::<u64>()).unwrap();
let ptr = slot.as_mut_ptr().cast::<u64>();

unsafe { ptr.write(42) };
assert_eq!(unsafe { *ptr }, 42);
```

### With standard-library allocator

Enable `nightly` when you want bump arena to back standard-library
collections that use the unstable allocator API:

```toml
[dependencies]
linalloc = { version = "1", features = ["nightly"] }
```

```rust
#![feature(allocator_api)]

# #[cfg(feature = "nightly")]
# {
    use linalloc::BumpArena;

    let arena = BumpArena::new(128);
    let mut values = Vec::with_capacity_in(1, &arena);

    values.push(1);
    values.try_reserve(1).unwrap();
    values.push(2);

    assert_eq!(&values, &[1, 2]);
# }
```

### In typed arena as backing allocator

`TypedArena<'a, T, A>` stores initialized `T` values in a backing allocator
`A` that implements the `UninitAllocator` trait
and drops the live values when the arena is reset or dropped.

```rust
use linalloc::{BumpArena, TypedArena};

let bump = BumpArena::new(128); // Implements `UninitAllocator`
let mut foo_arena = TypedArena::<String, _>::new_in(&bump);
let mut bar_arena = TypedArena::<String, _>::new_in(&bump);

let foo = foo_arena.try_alloc("foo".to_owned()).unwrap();
let bar = bar_arena.try_alloc("bar".to_owned()).unwrap();

assert_eq!(foo, "foo");
assert_eq!(bar, "bar");
```

## Reading OS errors

Bump arena keeps the raw OS code from the last failed reserve or commit call.
Use it when `try_new` fails, or when allocation returns `None` and you need to
know whether the OS refused more committed memory.

```rust
use core::alloc::Layout;

use linalloc::BumpArena;

if let Err(code) = BumpArena::try_new(usize::MAX) {
    assert_eq!(Some(code), std::io::Error::last_os_error().raw_os_error());
}


let arena = BumpArena::new(128);
let _slot = arena.try_alloc_uninit(Layout::new::<u64>()).unwrap();
assert_eq!(arena.last_os_error_code(), None);
```

## Safety

Bump arenas hand you uninitialized bytes. Do not read them until you have
written them. Values stored in bump arenas are not dropped automatically.

Typed arenas own initialized values. `reset` takes `&mut self`, drops live
values in reverse allocation order, and then reuses the storage.

For bump arenas, `reset` is unsafe. All returned slices must be dead, and
any values stored in the arena must already have been dropped.

With `nightly`, `BumpArena` implement
`core::alloc::Allocator`. Per-block `deallocate` is a no-op -- memory is reclaimed
only by `reset` or by dropping the arena. `grow`, `grow_zeroed`, and `shrink`
resize only the most recent allocation in place. Drop all collections and values
that use an arena allocator before calling `reset`.
