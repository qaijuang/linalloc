//! Platform‑specific memory management for incremental commitment.
//!
//! This module provides a thin, safe‑to‑use abstraction over the operating
//! system’s virtual memory primitives. It is **not** part of the public API
//! of `linalloc` -- it exists solely to support the arena allocator’s lazy
//! commit strategy.

use core::ptr::NonNull;

/// Reserve `size` bytes of address space without committing physical memory.
///
/// The returned pointer is guaranteed to be page‑aligned and non‑null.
///
/// # Returns
///
/// `None` if the operating system fails to reserve the requested address
/// range (e.g. out of address space).
pub fn reserve(size: usize) -> Option<NonNull<u8>> {
    platform::reserve(size)
}

/// Commit `size` bytes starting at `addr`, making the memory readable and
/// writable.
///
/// # Requirements
///
/// - `addr` must be page‑aligned and must point to memory previously reserved
///   but not yet committed.
/// - `size` must be non-zero and fit within the reservation. The platform
///   commits every page containing at least one byte in `[addr, addr + size)`.
/// - The range `[addr, addr + size)` must not contain any already‑committed
///   memory (the operating system may allow overlapping commits, but the
///   allocator never does so).
///
/// # Errors
///
/// Returns `Err(())` if the system call fails (e.g. out of physical memory).
pub fn commit(addr: NonNull<u8>, size: usize) -> Result<(), ()> {
    platform::commit(addr, size)
}

/// Release the entire reservation, including all committed pages.
///
/// # Safety
///
/// - `addr` must be the exact pointer returned by a previous call to
///   [`reserve`], and `size` must be the exact reservation size.
/// - This function must be called at most once for a given reservation.
/// - No pointers into the released region may be used afterwards.
pub unsafe fn release(addr: NonNull<u8>, size: usize) {
    unsafe { platform::release(addr, size) }
}

/// Returns the system’s page size in bytes.
///
/// The value is cached after the first call -- subsequent calls are
/// extremely cheap (a relaxed atomic load).
///
/// On Unix targets, if the page size query fails, this falls back to 4KiB.
pub fn page_size() -> usize {
    platform::page_size()
}

#[cfg(unix)]
mod platform {
    use core::ffi::c_void;
    use core::ptr::NonNull;
    use core::sync::atomic::{AtomicUsize, Ordering};

    unsafe extern "C" {
        fn mmap(
            addr: *mut c_void,
            length: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut c_void;

        fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32;

        fn munmap(addr: *mut c_void, length: usize) -> i32;

        fn sysconf(name: i32) -> i64;
    }

    const PROT_NONE: i32 = 0;
    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;

    const MAP_PRIVATE: i32 = 0x02;
    #[cfg(target_os = "linux")]
    const MAP_ANONYMOUS: i32 = 0x20;
    #[cfg(not(target_os = "linux"))]
    const MAP_ANONYMOUS: i32 = 0x1000; // MAP_ANON on macOS/BSD

    const MAP_FAILED: *mut c_void = -1isize as *mut c_void;

    const _SC_PAGESIZE: i32 = 29;

    pub fn reserve(size: usize) -> Option<NonNull<u8>> {
        #[cfg(target_os = "netbsd")]
        // NetBSD allows an mmap(2) caller to specify what protection flags they
        // will use later via mprotect. It does not allow a caller to move from
        // PROT_NONE to PROT_READ | PROT_WRITE.
        //
        // see PROT_MPROTECT in man 2 mmap
        const DESIRED_PROT: i32 = (PROT_READ | PROT_WRITE) << 3;

        #[cfg(not(target_os = "netbsd"))]
        const DESIRED_PROT: i32 = PROT_NONE;

        let ptr = unsafe {
            mmap(core::ptr::null_mut(), size, DESIRED_PROT, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
        };
        if ptr.is_null() || ptr == MAP_FAILED {
            None
        } else {
            // SAFETY: The pointer is already checked and is not null
            Some(unsafe { NonNull::new_unchecked(ptr.cast()) })
        }
    }

    pub fn commit(addr: NonNull<u8>, size: usize) -> Result<(), ()> {
        let ret = unsafe { mprotect(addr.as_ptr().cast(), size, PROT_READ | PROT_WRITE) };
        if ret == 0 { Ok(()) } else { Err(()) }
    }

    pub unsafe fn release(addr: NonNull<u8>, size: usize) {
        unsafe { munmap(addr.as_ptr().cast(), size) };
    }

    static PAGE_SIZE: AtomicUsize = AtomicUsize::new(0);

    #[inline]
    pub fn page_size() -> usize {
        let sz = PAGE_SIZE.load(Ordering::Relaxed);
        if sz != 0 {
            return sz;
        }
        page_size_store()
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "sysconf returns a long, but page sizes are never that large"
    )]
    #[cold]
    fn page_size_store() -> usize {
        let raw = unsafe { sysconf(_SC_PAGESIZE) };
        let sz = if raw < 0 { 4 * 1024 } else { raw as usize };
        PAGE_SIZE.store(sz, Ordering::Relaxed);
        sz
    }
}

#[cfg(target_os = "windows")]
mod platform {
    use core::ffi::c_void;
    use core::ptr::NonNull;
    use core::sync::atomic::{AtomicUsize, Ordering};

    unsafe extern "system" {
        fn VirtualAlloc(
            lpAddress: *mut c_void,
            dwSize: usize,
            flAllocationType: u32,
            flProtect: u32,
        ) -> *mut c_void;

        fn VirtualFree(lpAddress: *mut c_void, dwSize: usize, dwFreeType: u32) -> i32;

        fn GetSystemInfo(lpSystemInfo: *mut SYSTEM_INFO);
    }

    const MEM_RESERVE: u32 = 0x0000_2000;
    const MEM_COMMIT: u32 = 0x0000_1000;
    const MEM_RELEASE: u32 = 0x0000_8000;

    const PAGE_NOACCESS: u32 = 0x01;
    const PAGE_READWRITE: u32 = 0x04;

    #[allow(non_snake_case, reason = "matches Windows API naming")]
    #[repr(C)]
    struct SYSTEM_INFO {
        _u: u32,
        dwPageSize: u32,
        lpMinimumApplicationAddress: *mut c_void,
        lpMaximumApplicationAddress: *mut c_void,
        dwActiveProcessorMask: usize,
        dwNumberOfProcessors: u32,
        dwProcessorType: u32,
        dwAllocationGranularity: u32,
        wProcessorLevel: u16,
        wProcessorRevision: u16,
    }

    pub fn reserve(size: usize) -> Option<NonNull<u8>> {
        let ptr = unsafe { VirtualAlloc(core::ptr::null_mut(), size, MEM_RESERVE, PAGE_NOACCESS) };
        NonNull::new(ptr.cast())
    }

    pub fn commit(addr: NonNull<u8>, size: usize) -> Result<(), ()> {
        let ptr = unsafe { VirtualAlloc(addr.as_ptr().cast(), size, MEM_COMMIT, PAGE_READWRITE) };
        if ptr.is_null() { Err(()) } else { Ok(()) }
    }

    pub unsafe fn release(addr: NonNull<u8>, _size: usize) {
        unsafe { VirtualFree(addr.as_ptr().cast(), 0, MEM_RELEASE) };
    }

    static PAGE_SIZE: AtomicUsize = AtomicUsize::new(0);

    #[inline]
    pub fn page_size() -> usize {
        let sz = PAGE_SIZE.load(Ordering::Relaxed);
        if sz != 0 {
            return sz;
        }
        page_size_store()
    }

    #[cold]
    fn page_size_store() -> usize {
        let mut info: SYSTEM_INFO = unsafe { core::mem::zeroed() };
        unsafe { GetSystemInfo(&raw mut info) };
        let sz = info.dwPageSize as usize;
        PAGE_SIZE.store(sz, Ordering::Relaxed);
        sz
    }
}
