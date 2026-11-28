//! Allocation regression for every steady-state DSP path; counts only this test's thread.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::time::Duration;

use lk_agents_media::playout::{Config, Playout};

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

struct Counting;

// SAFETY: all allocations and deallocations are forwarded unchanged to the system allocator.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ENABLED.try_with(Cell::get).unwrap_or(false) {
            let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
            let _ = BYTES.try_with(|count| count.set(count.get() + layout.size()));
        }
        // SAFETY: forwarding the caller's valid allocation layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: pointer and layout are those returned by the system allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[test]
fn normal_stretch_conceal_recover_and_reset_do_not_allocate() {
    BYTES.set(0);
    ENABLED.set(true);
    let mut playout = Playout::new(Config::default()).unwrap();
    ENABLED.set(false);
    assert!(BYTES.get() <= 100_000, "default controller heap budget exceeded");
    let voice: Vec<_> = (0..2400)
        .map(|i| ((i as f64 * std::f64::consts::TAU / 120.0).sin() * 12_000.0) as i16)
        .collect();
    let mut expansion = Playout::new(Config {
        minimum_delay: Duration::from_millis(100),
        maximum_stretch: 0.05,
        ..Config::default()
    })
    .unwrap();
    let mut out = [0; 240];
    ALLOCATIONS.set(0);
    ENABLED.set(true);
    for at in (0..2000).step_by(10) {
        if at < 1000 && at % 100 == 0 || at == 1200 {
            playout.push(Duration::from_millis(at), &voice).unwrap();
        }
        playout.render(Duration::from_millis(at), &mut out).unwrap();
    }
    expansion.push(Duration::ZERO, &voice).unwrap();
    expansion.push(Duration::from_millis(100), &voice).unwrap();
    for at in (100..310).step_by(10) {
        expansion.render(Duration::from_millis(at), &mut out).unwrap();
    }
    playout.finish();
    playout.reset();
    // Backlog triggers acceleration as well as ordinary chunk ingestion.
    for _ in 0..4 {
        playout.push(Duration::ZERO, &voice).unwrap();
    }
    for at in (20..220).step_by(10) {
        playout.render(Duration::from_millis(at), &mut out).unwrap();
    }
    ENABLED.set(false);
    assert_eq!(ALLOCATIONS.get(), 0);
    assert!(expansion.statistics().stretched > 0);
    assert!(playout.statistics().accelerated > 0);
}
