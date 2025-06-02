// taken from hugepage_rs because they forgot to relax the sized bound
// on their boxed type :)

use crate::util::mem::allocate;
use std::alloc::{GlobalAlloc, Layout};
use std::iter::zip;
use std::mem::MaybeUninit;
use std::num::NonZero;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::slice;

/// A pointer type for hugepage allocation.
///
/// Allocates memory on the hugepage and then places x into it.
#[derive(Debug, PartialEq)]
pub struct BoxSlice<T> {
    data: NonNull<[T]>,
}

// safety:
// only we own the underlying memory so no other threads
// could access the data
unsafe impl<T> Send for BoxSlice<T> {}

impl<T> BoxSlice<T> {
    pub fn alloc_slice(value: T, len: NonZero<usize>) -> Self
    where
        T: Copy,
    {
        let slice = alloc_huge_slice_value(value, len.into());
        // safety: slice is villed with elements of T
        unsafe { Self::from_raw(slice) }
    }
    pub fn alloc_slice_uninit(len: NonZero<usize>) -> BoxSlice<MaybeUninit<T>> {
        let slice = alloc_huge_slice_uninit(len.into());
        // safety: slice is villed with elements of T
        unsafe { BoxSlice::from_raw(slice) }
    }
    pub fn alloc_slice_transparent_hugepages(value: T, len: NonZero<usize>) -> Self
    where
        T: Copy,
    {
        let slice = alloc_huge_slice_value_transparent_hugepages(value, len.into());
        // safety: slice is villed with elements of T
        unsafe { Self::from_raw(slice) }
    }
    /// # Safety
    /// - remember to drop this using BoxSlice::from_raw
    pub unsafe fn leak<'a>(self) -> &'a mut [T] {
        let mut this = std::mem::ManuallyDrop::new(self);
        // safety: data is valid and initialized
        unsafe { this.data.as_mut() }
    }

    /// # Safety
    /// raw must be a correctly aligned and fully initialized slice of T
    pub const unsafe fn from_raw(raw: *mut [T]) -> Self {
        Self {
            data: NonNull::new(raw).unwrap(),
        }
    }

    pub fn clone_manual(&self) -> Self
    where
        T: Copy,
    {
        let len = self.data.len();

        // safety: data is not modified and always initializated valid
        let this = unsafe { self.data.as_ref() };
        let other = alloc_huge_slice_uninit::<T>(len);

        for (from, to) in zip(this, &mut *other) {
            *to = MaybeUninit::new(*from);
        }

        // safety: other will be fully filled with valid elements one line below
        let other = unsafe { std::mem::transmute::<&mut [MaybeUninit<T>], &mut [T]>(other) };

        Self {
            data: NonNull::from(other),
        }
    }
}

impl<T> Drop for BoxSlice<T> {
    fn drop(&mut self) {
        // safety: pointer is never modified after initialization
        let layout = Layout::array::<T>(self.data.len()).unwrap();
        unsafe {
            // std::alloc::dealloc(self.data.as_ptr() as *mut u8, layout);
            allocate::HugePageAllocator.dealloc(self.data.as_ptr() as *mut u8, layout);
        }
    }
}

impl<T: Copy> Clone for BoxSlice<T> {
    fn clone(&self) -> Self {
        let len = self.data.len();

        // safety: data is not modified and always initializated valid
        let this = unsafe { self.data.as_ref() };
        let other = alloc_huge_slice_uninit::<T>(len);
        //
        // safety: other will be fully filled with valid elements one line below
        let other = unsafe { std::mem::transmute::<&mut [MaybeUninit<T>], &mut [T]>(other) };
        // safety:
        // - other was newly allocated thus is distinc from this
        // - other and this have the same length
        unsafe {
            std::ptr::copy_nonoverlapping(this.as_ptr(), other.as_mut_ptr(), len);
        }

        Self {
            data: NonNull::from(other),
        }
    }
}

impl<T> Deref for BoxSlice<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { self.data.as_ref() }
    }
}

impl<T> DerefMut for BoxSlice<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { self.data.as_mut() }
    }
}

fn alloc_huge_slice_uninit_transparent_hugepages<'a, T>(len: usize) -> &'a mut [MaybeUninit<T>] {
    let layout = Layout::array::<T>(len).unwrap();
    let mem = unsafe { allocate::HugePageAllocator::mmap_transparent_hugepages(layout) }
        as *mut MaybeUninit<T>;

    if mem.is_null() {
        panic!("alloc failed");
    }

    // safety: we allocated the memory for exactly `len` T's above with the correct alignment
    unsafe { slice::from_raw_parts_mut(mem, len) }
}
fn alloc_huge_slice_value_transparent_hugepages<'a, T: Copy>(value: T, len: usize) -> &'a mut [T] {
    let slice = alloc_huge_slice_uninit_transparent_hugepages(len);
    for element in &mut *slice {
        *element = MaybeUninit::new(value);
    }
    // safety: all elements of the slice have been initialized with a valid value
    // of T
    unsafe { std::mem::transmute(slice) }
}

fn alloc_huge_slice_uninit<'a, T>(len: usize) -> &'a mut [MaybeUninit<T>] {
    let layout = Layout::array::<T>(len).unwrap();
    let mem = unsafe { allocate::HugePageAllocator.alloc(layout) } as *mut MaybeUninit<T>;
    // let mem = unsafe { std::alloc::alloc(layout) } as *mut MaybeUninit<T>;

    if mem.is_null() {
        panic!("alloc failed");
    }

    // safety: we allocated the memory for exactly `len` T's above with the correct alignment
    unsafe { slice::from_raw_parts_mut(mem, len) }
}

fn alloc_huge_slice_value<'a, T: Copy>(value: T, len: usize) -> &'a mut [T] {
    let slice = alloc_huge_slice_uninit(len);
    for element in &mut *slice {
        *element = MaybeUninit::new(value);
    }
    // safety: all elements of the slice have been initialized with a valid value
    // of T
    unsafe { std::mem::transmute(slice) }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_boxed() {
        // variable.
        {
            let len = 5;

            let mut box_slice = BoxSlice::alloc_slice(0.0, NonZero::try_from(len).unwrap());

            for i in box_slice.deref_mut() {
                *i = 0.0;
            }
            box_slice[0] += 42.;

            assert_eq!(box_slice.len(), len);

            assert_eq!(box_slice[0], 42.0);
            assert_eq!(box_slice[1], 0.0);
            assert_eq!(box_slice[2], 0.0);
            assert_eq!(box_slice[3], 0.0);
            assert_eq!(box_slice[4], 0.0);
        }
    }
}
