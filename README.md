# linalloc (Linear Allocator)

[![miri](https://github.com/qaijuang/linalloc/actions/workflows/miri.yml/badge.svg)](https://github.com/qaijuang/linalloc/actions/workflows/miri.yml)
[![MSRV](https://img.shields.io/crates/msrv/linalloc)](https://crates.io/crates/linalloc)
[![crates.io](https://img.shields.io/crates/v/linalloc)](https://crates.io/crates/linalloc)
[![docs.rs](https://img.shields.io/docsrs/linalloc)](https://docs.rs/linalloc)
[![license](https://img.shields.io/github/license/qaijuang/linalloc)](https://github.com/qaijuang/linalloc/blob/main/LICENSE)

Small, fixed-capacity arena allocators for single-threaded Rust programs.

You pick the capacity up front. The arena never grows.
Addresses stay stable. When it is full, allocation returns `None`.

## Choose an arena

| Type                | Feature | What it gives you                                | Drop behavior                                 |
| ------------------- | ------- | ------------------------------------------------ | --------------------------------------------- |
| `BumpArena`         | default | Raw byte allocation from a fixed heap buffer     | Values must be dropped by the caller          |
| `TypedArena<T>`     | default | Values of one type from a fixed heap buffer      | Drops live values in reverse allocation order |
| `BumpArenaLazy`     | `lazy`  | Raw byte allocation from reserved virtual memory | Values must be dropped by the caller          |
| `TypedArenaLazy<T>` | `lazy`  | Values of one type from reserved virtual memory  | Drops live values in reverse allocation order |

All arenas are `!Send` and `!Sync`. They are deliberately single-threaded.

## Use a bump arena

`BumpArena` gives you uninitialized bytes. You choose the layout, initialize
the memory, and drop any values you place there.

```rust
use core::alloc::Layout;

use linalloc::BumpArena;

let arena = BumpArena::new(128);
let slot = arena.alloc_uninit_slice(Layout::new::<u64>()).unwrap();
let ptr = slot.as_mut_ptr().cast::<u64>();

unsafe { ptr.write(42) };
assert_eq!(unsafe { *ptr }, 42);
```

## Use a typed arena

`TypedArena<T>` stores initialized `T` values and drops the live values when
the arena is reset or dropped.

```rust
use linalloc::TypedArena;

let arena = TypedArena::<String>::new(4);
let value = arena.alloc_raw("hello".to_owned()).unwrap();

value.push_str(" world");
assert_eq!(value, "hello world");
```

The borrow checker prevents resetting a typed arena while references into it
are still live:

```rust,compile_fail
use linalloc::TypedArena;

let mut arena = TypedArena::<String>::new(1);
let value = arena.alloc_raw("held".to_owned()).unwrap();
//          ----- immutable borrow occurs here
arena.reset();
//^^^^^^^^^^^^^ mutable borrow occurs here
drop(value);
//   ----- immutable borrow later used here
```

## Use lazy arenas

Enable `lazy` when you want to reserve a large virtual address range and
commit physical memory only as allocation advances:

```toml
[dependencies]
linalloc = { version = "1", features = ["lazy"] }
```

The lazy feature is supported on Unix and Windows targets.

## Read lazy OS errors

Lazy arenas keep the raw OS code from the last failed reserve or commit call.
Use it when `try_new` fails, or when allocation returns `None` and you need to
know whether the OS refused more committed memory.

```rust
#[cfg(all(feature = "lazy", any(unix, windows), not(miri)))]
{
    use core::alloc::Layout;

    use linalloc::{BumpArenaLazy, TypedArenaLazy};

    if let Err(code) = BumpArenaLazy::try_new(usize::MAX) {
        assert_eq!(Some(code), std::io::Error::last_os_error().raw_os_error());
    }

    let typed = TypedArenaLazy::<u8>::new(1);
    assert!(typed.alloc_raw(1).is_some());
    assert!(typed.alloc_raw(2).is_none());
    assert_eq!(typed.last_os_error_code(), None);
}
```

## Safety

Untyped arenas hand you uninitialized bytes. Do not read them until you have
written them. Values stored in untyped arenas are not dropped automatically.

Typed arenas own initialized values. `reset` takes `&mut self`, drops live
values in reverse allocation order, and then reuses the storage.

For untyped arenas, `reset` is unsafe. All returned slices must be dead, and
any values stored in the arena must already have been dropped.

## Miri

Miri covers the default eager arenas.

The lazy arenas use platform virtual-memory calls. Miri does not currently
support every protection mode used by that path.
