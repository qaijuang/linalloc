//! Virtual memory primitives used by linalloc.

use core::ptr::NonNull;

/// Reserves `size` bytes of address space.
///
/// Returns a non-null, page-aligned pointer. The OS rounds `size` up to whole pages.
///
/// # Errors
///
/// Returns the OS error code on failure.
pub fn reserve(size: usize) -> Result<NonNull<u8>, i32> {
    platform::reserve(size)
}

/// Makes `size` bytes at `addr` readable and writable.
///
/// Uses `mprotect` on Unix and `VirtualAlloc(MEM_COMMIT)` on Windows.
/// The OS rounds the range up to whole pages.
///
/// # Safety
///
/// - `addr` is page-aligned and belongs to a live reservation from [`reserve`]
/// - `size` is nonzero and the range fits within the reservation
/// - The range has not been committed
///
/// # Errors
///
/// Returns the OS error code on failure.
pub unsafe fn commit(addr: NonNull<u8>, size: usize) -> Result<(), i32> {
    platform::commit(addr, size)
}

/// Releases a reservation and its committed pages.
///
/// # Safety
///
/// - `addr` and `size` match a live reservation from [`reserve`]
/// - The reservation is released only once
/// - No pointers into it are used afterward
pub unsafe fn release(addr: NonNull<u8>, size: usize) {
    unsafe { platform::release(addr, size) }
}

/// Returns the page size in bytes.
///
/// Uses 4096 if the Unix page size query fails.
#[must_use]
pub fn page_size() -> usize {
    platform::page_size()
}

#[cfg(miri)]
#[path = "miri.rs"]
mod platform;

#[cfg(all(unix, not(miri)))]
mod platform {
    use core::ffi::c_void;
    use core::ptr::NonNull;

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

        fn getpagesize() -> i32;

        // yanked from https://github.com/rust-lang/rust/blob/main/library/std/src/sys/io/error/unix.rs
        #[cfg(not(any(target_os = "dragonfly", target_os = "vxworks", target_os = "rtems")))]
        #[cfg_attr(
            any(
                target_os = "linux",
                target_os = "emscripten",
                target_os = "fuchsia",
                target_os = "l4re",
                target_os = "hurd",
            ),
            link_name = "__errno_location"
        )]
        #[cfg_attr(
            any(
                target_os = "netbsd",
                target_os = "openbsd",
                target_os = "cygwin",
                target_os = "android",
                target_os = "redox",
                target_os = "nuttx",
                target_env = "newlib"
            ),
            link_name = "__errno"
        )]
        #[cfg_attr(any(target_os = "solaris", target_os = "illumos"), link_name = "___errno")]
        #[cfg_attr(target_os = "nto", link_name = "__get_errno_ptr")]
        #[cfg_attr(any(target_os = "freebsd", target_vendor = "apple"), link_name = "__error")]
        #[cfg_attr(target_os = "haiku", link_name = "_errnop")]
        #[cfg_attr(target_os = "aix", link_name = "_Errno")]
        fn errno_location() -> *mut i32;
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

    pub fn reserve(size: usize) -> Result<NonNull<u8>, i32> {
        #[cfg(target_os = "netbsd")]
        // PROT_MPROTECT(PROT_READ | PROT_WRITE) allows later read/write access
        // See mmap(2)
        const DESIRED_PROT: i32 = (PROT_READ | PROT_WRITE) << 3;

        #[cfg(not(target_os = "netbsd"))]
        const DESIRED_PROT: i32 = PROT_NONE;

        let ptr = unsafe {
            mmap(core::ptr::null_mut(), size, DESIRED_PROT, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
        };
        if ptr.is_null() || ptr == MAP_FAILED {
            Err(last_os_error())
        } else {
            // SAFETY: ptr is non-null and not MAP_FAILED.
            Ok(unsafe { NonNull::new_unchecked(ptr.cast()) })
        }
    }

    pub fn commit(addr: NonNull<u8>, size: usize) -> Result<(), i32> {
        let ret = unsafe { mprotect(addr.as_ptr().cast(), size, PROT_READ | PROT_WRITE) };
        if ret == 0 { Ok(()) } else { Err(last_os_error()) }
    }

    pub unsafe fn release(addr: NonNull<u8>, size: usize) {
        unsafe { munmap(addr.as_ptr().cast(), size) };
    }

    pub fn last_os_error() -> i32 {
        unsafe { *errno_location() }
    }

    #[allow(clippy::cast_sign_loss)]
    #[inline]
    pub fn page_size() -> usize {
        let sz = unsafe { getpagesize() };
        if sz <= 0 { 4 * 1024 } else { sz as usize }
    }
}

#[cfg(all(target_os = "windows", not(miri)))]
mod platform {
    use core::ffi::c_void;
    use core::ptr::NonNull;

    unsafe extern "system" {
        fn VirtualAlloc(
            lpAddress: *mut c_void,
            dwSize: usize,
            flAllocationType: u32,
            flProtect: u32,
        ) -> *mut c_void;

        fn VirtualFree(lpAddress: *mut c_void, dwSize: usize, dwFreeType: u32) -> i32;

        fn GetSystemInfo(lpSystemInfo: *mut SYSTEM_INFO);

        fn GetLastError() -> u32;
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

    pub fn reserve(size: usize) -> Result<NonNull<u8>, i32> {
        let ptr = unsafe { VirtualAlloc(core::ptr::null_mut(), size, MEM_RESERVE, PAGE_NOACCESS) };
        NonNull::new(ptr.cast()).ok_or_else(last_os_error)
    }

    pub fn commit(addr: NonNull<u8>, size: usize) -> Result<(), i32> {
        let ptr = unsafe { VirtualAlloc(addr.as_ptr().cast(), size, MEM_COMMIT, PAGE_READWRITE) };
        if ptr.is_null() { Err(last_os_error()) } else { Ok(()) }
    }

    pub unsafe fn release(addr: NonNull<u8>, _size: usize) {
        unsafe { VirtualFree(addr.as_ptr().cast(), 0, MEM_RELEASE) };
    }

    pub fn last_os_error() -> i32 {
        unsafe { GetLastError() }.cast_signed()
    }

    #[inline]
    pub fn page_size() -> usize {
        let mut info: SYSTEM_INFO = unsafe { core::mem::zeroed() };
        unsafe { GetSystemInfo(&raw mut info) };
        info.dwPageSize as usize
    }
}
