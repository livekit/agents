//! Bounds temporary allocation while diffing realtime conversation histories.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Instant, UNIX_EPOCH};

use lk_agents_abi::{ChatContext, ChatMessage, Role, compute_chat_ctx_diff};

struct CountAllocations;
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

// This binary runs one test. Count requested bytes, including allocations later freed, so a
// bounded peak cannot hide repeatedly allocating a quadratic amount of scratch space.
// SAFETY: Every pointer and layout is forwarded to System unchanged; the counter
// observes allocation sizes without altering allocation ownership or alignment.
unsafe impl GlobalAlloc for CountAllocations {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATED.fetch_add(new_size, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountAllocations = CountAllocations;

fn context(ids: impl IntoIterator<Item = usize>) -> ChatContext {
    ChatContext::new(
        ids.into_iter()
            .map(|id| ChatMessage::text(id.to_string(), Role::User, "text", UNIX_EPOCH).into())
            .collect(),
    )
}

fn diff_allocation_scaling() {
    for count in [1_000, 10_000] {
        let old = context(0..count);
        let appended = context(0..count + 1);
        let reversed = context((0..count).rev());
        for (name, new) in [("identical", &old), ("append", &appended), ("reorder", &reversed)] {
            let baseline = ALLOCATED.load(Ordering::Relaxed);
            let started = Instant::now();
            let diff = compute_chat_ctx_diff(&old, new);
            let elapsed = started.elapsed();
            let allocated = ALLOCATED.load(Ordering::Relaxed) - baseline;
            println!("{name} {count} items: {elapsed:?}, {allocated} allocated bytes");
            // The budget covers indexes, the retained subsequence and the resulting edit script.
            // Each input item gets 512 bytes; no input-dependent product of the two lengths fits.
            assert!(
                allocated
                    <= if name == "reorder" {
                        512 * (old.len() + new.len())
                    } else {
                        128 * new.len()
                    },
                "{name}: {allocated} temporary/output bytes for {count} items"
            );
            match name {
                "identical" => assert_eq!(diff, Default::default()),
                "append" => {
                    assert!(diff.to_remove.is_empty());
                    assert!(diff.to_update.is_empty());
                    assert_eq!(
                        diff.to_create,
                        vec![(Some((count - 1).to_string()), count.to_string())]
                    );
                }
                "reorder" => {
                    assert_eq!(diff.to_remove.len(), count - 1);
                    assert_eq!(diff.to_create.len(), count - 1);
                    assert_eq!(diff.to_remove[0], "0");
                    assert_eq!(
                        diff.to_create[0],
                        (Some((count - 1).to_string()), (count - 2).to_string())
                    );
                }
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn realtime_ctx_diff_scaling() {
    diff_allocation_scaling();
}
