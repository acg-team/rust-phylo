use libc::{
    self, c_void, MAP_ANONYMOUS, MAP_FAILED, MAP_HUGETLB, MAP_PRIVATE, PROT_READ, PROT_WRITE,
};
use std::{
    alloc::{GlobalAlloc, Layout},
    ffi::CString,
    fs::File,
    io::Read,
    ptr::null_mut,
};

// https://www.kernel.org/doc/Documentation/vm/hugetlbpage.txt
//
// The output of "cat /proc/meminfo" will include lines like:
// ...
// HugePages_Total: uuu
// HugePages_Free:  vvv
// HugePages_Rsvd:  www
// HugePages_Surp:  xxx
// Hugepagesize:    yyy kB
// Hugetlb:         zzz kB

// constant.
const MEMINFO_PATH: &str = "/proc/meminfo";
const HUGEPAGESIZE_PREFIX: &str = "Hugepagesize:";

lazy_static::lazy_static! {
    static ref HUGEPAGE_SIZE: usize = {
        // TODO: lazy read lines
        let buf = File::open(MEMINFO_PATH).map_or("".to_owned(), |mut f| {
            let mut s = String::new();
            let _ = f.read_to_string(&mut s);
            s
        });
        let hugepage_size = parse_hugepage_size(&buf);
        assert_eq!(hugepage_size, 2048 * 1024);
        hugepage_size
    };
}

fn parse_hugepage_size(s: &str) -> usize {
    for line in s.lines() {
        if let Some(without_prefix) = line.strip_prefix(HUGEPAGESIZE_PREFIX) {
            let mut parts = without_prefix.split_whitespace();

            let p = parts.next().unwrap();
            let mut hugepage_size = p.parse::<usize>().unwrap();

            hugepage_size *= parts.next().map_or(1, |x| match x {
                "kB" => 1024,
                "" => 1,
                _ => panic!("unrecognized hugepage size unit"),
            });

            return hugepage_size;
        }
    }

    panic!("failed to read huge page configuration")
}

fn align_to(size: usize, align: usize) -> usize {
    (size + align - 1) & !(align - 1)
}

// hugepage allocator.
pub(crate) struct HugePageAllocator;

impl HugePageAllocator {
    #[allow(dead_code)]
    pub unsafe fn mmap_hugetlbfs(layout: Layout) -> *mut u8 {
        let len = layout
            .align_to(*HUGEPAGE_SIZE)
            .unwrap()
            .pad_to_align()
            .size();
        let p = unsafe {
            libc::mmap(
                null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, // if this fails you probably have
                // nr_hugepages = 0 in your kernel, see justfile
                -1,
                0,
            )
        };
        if p == MAP_FAILED {
            let layout_size = layout.size();
            panic!(
                "hugetlbfs alloc failed for len {layout_size}/{len}: '{:?}', did you allocate hugetables in your kernel?",
                CString::from_raw(libc::strerror(*libc::__errno_location()))
            );
            // return null_mut();
        }

        p as *mut u8
    }

    pub unsafe fn mmap_transparent_hugepages(layout: Layout) -> *mut u8 {
        let len = align_to(layout.size(), *HUGEPAGE_SIZE);
        let p = unsafe {
            libc::mmap(
                null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            )
        };

        libc::madvise(p, len, libc::MADV_HUGEPAGE);
        if p == MAP_FAILED {
            return null_mut();
        }

        p as *mut u8
    }
}

unsafe impl GlobalAlloc for HugePageAllocator {
    #[cfg(feature = "hugetlbfs")]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        HugePageAllocator::mmap_hugetlbfs(layout)
    }
    #[cfg(not(feature = "hugetlbfs"))]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        HugePageAllocator::mmap_transparent_hugepages(layout)
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        libc::munmap(
            p as *mut c_void,
            layout
                .align_to(*HUGEPAGE_SIZE)
                .unwrap()
                .pad_to_align()
                .size(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{mem, ptr};

    #[test]
    fn test_parse_hugepage_size() {
        // correct.
        assert_eq!(parse_hugepage_size("Hugepagesize:1024"), 1024);
        assert_eq!(parse_hugepage_size("Hugepagesize: 2 kB"), 2048);
    }

    #[test]
    fn test_align_to() {
        assert_eq!(align_to(8, 4), 8);
        assert_eq!(align_to(8, 16), 16);
    }

    #[test]
    fn test_allocator() {
        let hugepage_alloc = HugePageAllocator;

        // u16.
        unsafe {
            let layout = Layout::new::<u16>();
            let p = hugepage_alloc.alloc(layout);
            assert_ne!(p, null_mut());
            *p = 20;
            assert_eq!(*p, 20);
            hugepage_alloc.dealloc(p, layout);
        }

        // array.
        unsafe {
            let layout = Layout::array::<char>(2048).unwrap();
            let dst = hugepage_alloc.alloc(layout);
            assert_ne!(dst, null_mut());

            let src = String::from("hello rust");
            let len = src.len();
            ptr::copy_nonoverlapping(src.as_ptr(), dst, len);
            let s = String::from_raw_parts(dst, len, len);
            assert_eq!(s, src);
            mem::forget(s);

            hugepage_alloc.dealloc(dst, layout);
        }
    }
}
