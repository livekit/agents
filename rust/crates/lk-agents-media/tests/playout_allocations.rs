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
        minimum_delay: Duration::ZERO,
        maximum_stretch: 0.05,
        ..Config::default()
    })
    .unwrap();
    let mut fast = Playout::new(Config::default()).unwrap();
    let mut out = [0; 240];
    ALLOCATIONS.set(0);
    ENABLED.set(true);
    for at in (0..2000).step_by(10) {
        if at < 1000 && at % 100 == 0 || at == 1200 {
            playout.push(Duration::from_millis(at), &voice).unwrap();
        }
        playout.render(Duration::from_millis(at), &mut out).unwrap();
    }
    for at in (0..500).step_by(10) {
        if at % 40 == 0 {
            expansion.push(Duration::from_millis(at), &voice[..960]).unwrap();
        }
        expansion.render(Duration::from_millis(at), &mut out).unwrap();
    }
    // Enter an actual voiced alignment prefix, not only an unvoiced gain ramp.
    playout.reset();
    playout.push(Duration::ZERO, &voice).unwrap();
    for at in (0..140).step_by(10) {
        playout.render(Duration::from_millis(at), &mut out).unwrap();
    }
    playout.push(Duration::from_millis(140), &voice[30..]).unwrap();
    playout.render(Duration::from_millis(140), &mut out).unwrap();
    let merged = playout.statistics().merge_inserted;
    playout.finish();
    playout.reset();
    // Backlog triggers acceleration as well as ordinary chunk ingestion.
    for _ in 0..4 {
        playout.push(Duration::ZERO, &voice).unwrap();
    }
    for at in (100..310).step_by(10) {
        playout.render(Duration::from_millis(at), &mut out).unwrap();
    }
    for at in (0..1000_u64).step_by(10) {
        if at.is_multiple_of(100) {
            fast.push(Duration::from_millis(at), &voice).unwrap();
        }
        fast.render(Duration::from_millis(at), &mut out).unwrap();
    }
    for at in (1000..2500).step_by(100) {
        fast.push(Duration::from_millis(at), &voice).unwrap();
    }
    for at in (2500..2580).step_by(10) {
        fast.render(Duration::from_millis(at), &mut out).unwrap();
    }
    ENABLED.set(false);
    assert_eq!(ALLOCATIONS.get(), 0);
    assert!(merged > 0);
    assert!(fast.statistics().fast_accelerated > 0);
    assert!(expansion.statistics().stretched > 0);
    assert!(playout.statistics().accelerated > 0);
}
