use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountedSystem;

thread_local! {
    static ALLOCATED: Cell<Option<usize>> = const { Cell::new(None) };
}

fn count(bytes: usize) {
    let _ = ALLOCATED.try_with(|counter| {
        if let Some(total) = counter.get() {
            counter.set(Some(total + bytes));
        }
    });
}

// SAFETY: every operation forwards its original pointers and layouts to System.
unsafe impl GlobalAlloc for CountedSystem {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        // SAFETY: the allocator contract supplies a valid layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        // SAFETY: the allocator contract supplies a valid layout.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: allocation and deallocation both use System with the caller's layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size);
        // SAFETY: the caller supplies an allocated pointer and valid replacement size.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountedSystem = CountedSystem;

/// Measure allocations on this test thread; restore recording even if the closure panics.
pub(crate) fn allocated<T>(run: impl FnOnce() -> T) -> (T, usize) {
    struct Recording;
    impl Drop for Recording {
        fn drop(&mut self) {
            ALLOCATED.set(None);
        }
    }
    ALLOCATED.with(|counter| assert!(counter.replace(Some(0)).is_none()));
    let recording = Recording;
    let value = run();
    let bytes = ALLOCATED.with(|counter| counter.get().unwrap());
    drop(recording);
    (value, bytes)
}
