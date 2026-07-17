#![cfg_attr(feature = "nightly", feature(allocator_api))]

use core::alloc::Layout;
use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use linalloc::{BumpArena, sys};

const CAPACITY: usize = 1024 * 1024 * 64;

fn linalloc() -> BumpArena {
    BumpArena::new(CAPACITY)
}

fn bench_alloc_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("alloc_small");
    let layout = Layout::new::<u64>();

    group.bench_function("linalloc", |b| {
        let arena = linalloc();
        b.iter(|| {
            unsafe { arena.reset() }
            let slice = arena.try_alloc_uninit(layout).unwrap();
            black_box(slice)
        });
    });

    group.finish();
}

fn bench_alloc_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("alloc_throughput");
    let layout = Layout::new::<u64>();

    for &n in &[100, 10_000, 100_000] {
        group.bench_with_input(BenchmarkId::new("linalloc", n), &n, |b, &n| {
            let arena = linalloc();
            b.iter(|| {
                unsafe { arena.reset() };
                for _ in 0..n {
                    let slice = arena.try_alloc_uninit(layout).unwrap();
                    black_box(slice);
                }
            });
        });
    }

    group.finish();
}

fn bench_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("alignment");
    let layouts = [
        Layout::from_size_align(8, 1).unwrap(),
        Layout::from_size_align(8, 4).unwrap(),
        Layout::from_size_align(8, 8).unwrap(),
        Layout::from_size_align(8, 16).unwrap(),
        Layout::from_size_align(8, 64).unwrap(),
    ];

    for layout in layouts {
        let id = format!("{}-align-{}", layout.size(), layout.align());
        group.bench_function(format!("linalloc/{id}"), |b| {
            let arena = linalloc();
            b.iter(|| {
                unsafe { arena.reset() };
                arena.try_alloc_uninit(layout).unwrap();
            });
        });
    }

    group.finish();
}

#[cfg(feature = "nightly")]
fn bench_grow_in_place(c: &mut Criterion) {
    use core::alloc::Allocator;

    let mut group = c.benchmark_group("grow_in_place");
    let old_layout = Layout::new::<u64>();
    let new_layout = Layout::new::<[u64; 8]>();

    group.bench_function("linalloc", |b| {
        let arena = linalloc();
        b.iter(|| {
            use core::ptr::NonNull;

            unsafe { arena.reset() };
            // Allocate a single u64
            let ptr = arena.allocate(old_layout).unwrap();
            let ptr = NonNull::new(ptr.as_ptr().cast::<u8>()).unwrap();
            let ptr = unsafe { arena.grow(ptr, old_layout, new_layout).unwrap() };
            black_box(ptr)
        });
    });
    group.finish();
}

#[cfg(feature = "nightly")]
fn bench_vec_push(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec_push");

    for &n in &[100, 10_000] {
        group.bench_with_input(BenchmarkId::new("linalloc", n), &n, |b, &n| {
            let arena = linalloc();
            b.iter(|| {
                unsafe { arena.reset() };
                let mut v: Vec<u64, &BumpArena> = Vec::with_capacity_in(1, &arena);
                for i in 0u64..n {
                    v.push(i);
                }
                black_box(v);
            });
        });
    }

    group.finish();
}

fn bench_reset(c: &mut Criterion) {
    let mut group = c.benchmark_group("reset");

    group.bench_function("linalloc", |b| {
        let arena = linalloc();
        // Fill it a bit to simulate state
        for _ in 0..100 {
            arena.try_alloc_uninit(Layout::new::<u64>()).unwrap();
        }
        b.iter(|| unsafe { arena.reset() });
    });

    group.finish();
}

fn bench_commit_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("commit_boundary");
    let page = sys::page_size();
    dbg!(page);
    let small_layout = Layout::from_size_align(page - 8, 1).unwrap();
    let across_layout = Layout::from_size_align(16, 1).unwrap();

    group.bench_function("linalloc", |b| {
        let arena = linalloc();
        arena.try_alloc_uninit(small_layout).unwrap();
        b.iter(|| {
            let slice = arena.try_alloc_uninit(across_layout).unwrap();
            black_box(slice);
            unsafe { arena.reset() };
        });
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
);

#[cfg(feature = "nightly")]
criterion_group!(nightly_benches, bench_grow_in_place, bench_vec_push,);

#[cfg(feature = "nightly")]
criterion_main!(benches, nightly_benches);

#[cfg(not(feature = "nightly"))]
criterion_main!(benches);
