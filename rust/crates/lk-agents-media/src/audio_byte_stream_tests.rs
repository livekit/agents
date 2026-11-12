use super::audio_byte_stream::AudioByteStream;
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

/// Framing allocates only linear storage while preserving every PCM byte. https://github.com/livekit/agents/pull/4247
#[test]
#[ignore = "https://github.com/livekit/agents/pull/4247 reproduces: framing repeatedly allocates and copies the unconsumed tail"]
fn extracting_many_frames_does_not_reallocate_the_remaining_tail_each_time() {
    for frames in [128, 256, 512] {
        let input: Vec<_> = (0..frames * 320).map(|i| (i % 251) as u8).collect();
        let mut stream = AudioByteStream::new(16_000, 1, Some(160), false);
        ALLOCATED.with(|counter| counter.set(Some(0)));
        let out = stream.push(&input);
        let allocated = ALLOCATED.with(|counter| counter.replace(None).unwrap());
        assert_eq!(out.len(), frames);
        assert_eq!(out.into_iter().flat_map(|f| f.data).collect::<Vec<_>>(), input);
        assert!(
            allocated < input.len() * 8,
            "framing {frames} frames allocated {allocated} bytes for {} input bytes",
            input.len()
        );
    }
}
