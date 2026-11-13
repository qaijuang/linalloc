use core::alloc::{Allocator, Layout};
use core::hint::black_box;
use core::ptr::NonNull;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use linalloc::BumpArena;

const CAPACITY: usize = 1024 * 1024 * 64;

fn linalloc() -> BumpArena {
    BumpArena::new(CAPACITY)
}

fn warm_arena(bytes: usize) -> BumpArena {
    let arena = linalloc();
    black_box(arena.alloc_uninit(Layout::from_size_align(bytes, 1).unwrap()));
    // SAFETY: The allocation's borrow has ended and it holds no values to drop
    unsafe { arena.reset() };
    arena
}

fn bench_alloc_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("alloc_small");
    let layout = Layout::new::<u64>();

    group.bench_function("warm_raw", |b| {
        let arena = warm_arena(layout.size());
        b.iter(|| {
            unsafe { arena.reset() };
            black_box(arena.try_alloc_uninit(layout).unwrap())
        });
    });
    group.bench_function("warm_typed", |b| {
        let arena = warm_arena(layout.size());
        b.iter(|| {
            unsafe { arena.reset() };
            black_box(arena.try_alloc(black_box(42_u64)).unwrap())
        });
    });
    group.bench_function("warm_allocator", |b| {
        let arena = warm_arena(layout.size());
        b.iter(|| {
            unsafe { arena.reset() };
            black_box(arena.allocate(layout).unwrap())
        });
    });
    group.bench_function("capacity_failure", |b| {
        let arena = BumpArena::new(layout.size());
        arena.alloc(0_u64);
        b.iter(|| black_box(arena.try_alloc(black_box(42_u64))));
    });
    group.bench_function("zst", |b| {
        let arena = BumpArena::new(0);
        b.iter(|| black_box(arena.try_alloc(()).unwrap()));
    });

    group.finish();
}

fn bench_alloc_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("alloc_throughput");
    let layout = Layout::new::<u64>();

    for &n in &[100, 10_000, 100_000] {
        group.bench_with_input(BenchmarkId::new("warm", n), &n, |b, &n| {
            let arena = warm_arena(n * layout.size());
            b.iter(|| {
                unsafe { arena.reset() };
                for _ in 0..n {
                    black_box(arena.try_alloc_uninit(layout).unwrap());
                }
            });
        });
        group.bench_with_input(BenchmarkId::new("fresh", n), &n, |b, &n| {
            b.iter_batched(
                linalloc,
                |arena| {
                    for _ in 0..n {
                        black_box(arena.try_alloc_uninit(layout).unwrap());
                    }
                    // Return ownership so Criterion releases pages after timing
                    arena
                },
                BatchSize::PerIteration,
            );
        });
    }

    group.finish();
}

fn bench_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("alignment_warm");
    let prefix = Layout::from_size_align(1, 64).unwrap();

    for align in [1, 4, 8, 16, 64] {
        let layout = Layout::from_size_align(8, align).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(align), &layout, |b, &layout| {
            let arena = warm_arena(128);
            b.iter(|| {
                unsafe { arena.reset() };
                // Include a one-byte predecessor so alignments above one need padding
                black_box(arena.alloc_uninit(prefix));
                black_box(arena.try_alloc_uninit(layout).unwrap())
            });
        });
    }

    group.finish();
}

fn bench_grow(c: &mut Criterion) {
    let mut group = c.benchmark_group("grow_warm");
    let old_layout = Layout::new::<u64>();
    let new_layout = Layout::new::<[u64; 8]>();

    for (name, zeroed, relocating) in [
        ("in_place", false, false),
        ("relocating", false, true),
        ("zeroed_in_place", true, false),
        ("zeroed_relocating", true, true),
    ] {
        group.bench_function(name, |b| {
            let arena = warm_arena(128);
            b.iter(|| {
                unsafe { arena.reset() };
                let block = arena.allocate(old_layout).unwrap();
                let ptr = NonNull::new(block.as_ptr().cast::<u8>()).unwrap();
                if relocating {
                    black_box(arena.alloc_uninit(Layout::new::<u8>()));
                }
                // SAFETY: ptr is a live block matching old_layout, with no live references
                let block = unsafe {
                    if zeroed {
                        arena.grow_zeroed(ptr, old_layout, new_layout).unwrap()
                    } else {
                        arena.grow(ptr, old_layout, new_layout).unwrap()
                    }
                };
                black_box(block)
            });
        });
    }

    group.finish();
}

fn bench_shrink(c: &mut Criterion) {
    let mut group = c.benchmark_group("shrink_warm");
    let old_layout = Layout::from_size_align(16, 1).unwrap();
    let prefix = Layout::from_size_align(1, 64).unwrap();

    for (name, align) in [("in_place", 1), ("realigning", 64)] {
        let new_layout = Layout::from_size_align(8, align).unwrap();
        group.bench_function(name, |b| {
            let arena = warm_arena(128);
            b.iter(|| {
                unsafe { arena.reset() };
                if align > 1 {
                    black_box(arena.alloc_uninit(prefix));
                }
                let block = arena.allocate(old_layout).unwrap();
                let ptr = NonNull::new(block.as_ptr().cast::<u8>()).unwrap();
                // SAFETY: ptr matches old_layout, has no live references, and the size decreases
                black_box(unsafe { arena.shrink(ptr, old_layout, new_layout).unwrap() })
            });
        });
    }

    group.finish();
}

fn bench_vec_push(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec_push_warm");

    for &n in &[100, 10_000] {
        for (name, interleaved) in [("isolated", false), ("interleaved", true)] {
            group.bench_with_input(BenchmarkId::new(name, n), &n, |b, &n| {
                let arena = warm_arena(usize::try_from(n).unwrap() * 32);
                b.iter(|| {
                    // SAFETY: The previous iteration's Vec was dropped before this reset
                    unsafe { arena.reset() };
                    let mut v: Vec<u64, &BumpArena> = Vec::with_capacity_in(1, &arena);
                    for i in 0..n {
                        if interleaved {
                            black_box(arena.alloc_uninit(Layout::new::<u8>()));
                        }
                        v.push(black_box(i));
                    }
                    black_box(v);
                });
            });
        }
    }

    group.finish();
}

fn bench_reset(c: &mut Criterion) {
    c.bench_function("reset_empty", |b| {
        let arena = linalloc();
        b.iter(|| unsafe { black_box(&arena).reset() });
    });
}

fn bench_commit_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("commit_boundary");
    let page = linalloc_sys::page_size();
    let small_layout = Layout::from_size_align(page - 8, 1).unwrap();
    let across_layout = Layout::from_size_align(16, 1).unwrap();
    let old_layout = Layout::from_size_align(page, 1).unwrap();
    let new_layout = Layout::from_size_align(page * 2, 1).unwrap();

    group.bench_function("allocate", |b| {
        b.iter_batched(
            || {
                let arena = BumpArena::new(page * 2);
                arena.alloc_uninit(small_layout);
                arena
            },
            |arena| {
                black_box(arena.try_alloc_uninit(across_layout).unwrap());
                arena
            },
            BatchSize::PerIteration,
        );
    });
    group.bench_function("grow", |b| {
        b.iter_batched(
            || {
                let arena = BumpArena::new(page * 2);
                let block = arena.allocate(old_layout).unwrap();
                let ptr = NonNull::new(block.as_ptr().cast::<u8>()).unwrap();
                (arena, ptr)
            },
            |(arena, ptr)| {
                // SAFETY: ptr is the live last block with old_layout and no live references
                black_box(unsafe { arena.grow(ptr, old_layout, new_layout).unwrap() });
                arena
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_alloc_small,
    bench_alloc_throughput,
    bench_alignment,
    bench_reset,
    bench_commit_boundary,
    bench_grow,
    bench_shrink,
    bench_vec_push
);

criterion_main!(benches);
