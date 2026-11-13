//! Heap storage for Miri. Page protection and OS errors aren't modeled.

use core::alloc::Layout;
use core::ptr::NonNull;
use std::alloc::{alloc, dealloc};

const PAGE_SIZE: usize = 4096;
const RESERVATION_FAILED: i32 = 1;

fn reservation_layout(size: usize) -> Result<Layout, i32> {
    if size == 0 {
        return Err(RESERVATION_FAILED);
    }
    Layout::from_size_align(size, PAGE_SIZE)
        .map(|layout| layout.pad_to_align())
        .map_err(|_| RESERVATION_FAILED)
}

pub fn reserve(size: usize) -> Result<NonNull<u8>, i32> {
    let layout = reservation_layout(size)?;
    // SAFETY: layout is valid and nonzero. Padding keeps the whole reservation
    // within a single allocation with page alignment.
    NonNull::new(unsafe { alloc(layout) }).ok_or(RESERVATION_FAILED)
}

#[allow(clippy::unnecessary_wraps, reason = "matches the native backend")]
pub fn commit(_addr: NonNull<u8>, _size: usize) -> Result<(), i32> {
    // The whole heap allocation is already writable
    Ok(())
}

pub unsafe fn release(addr: NonNull<u8>, size: usize) {
    let layout = reservation_layout(size).expect("size must match a live reservation");
    // SAFETY: The caller supplies the pointer and size from reserve, exactly
    // once. The same calculation recovers the original allocation layout.
    unsafe { dealloc(addr.as_ptr(), layout) };
}

pub fn page_size() -> usize {
    PAGE_SIZE
}
