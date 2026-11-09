use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

pub use crate::record::{Fault, Site};

// Panic and never-answer faults at the fakes' boundary, seeded per run and recorded like every
// other fault. Stress frequencies, not measured rates: a stressed run draws once per host call
// and once per provider stream (at its open), and one draw in `ONE_IN` strikes, half of those a
// panic and half a fake that never answers. A stream's fault lands at one of its first eight
// steps, or at its close, a quarter of the time. A panic unwinds out of the fake's future inside
// the product's task, where the runtime's spawn wrapper (`lk_agents_net::task::launch`) reports
// it and resumes it, as in production.

/// One draw in this many strikes.
pub const ONE_IN: u32 = 24;

/// The prefix of every injected panic, so the rules tell it from a production panic.
pub const INJECTED: &str = "injected fault";

/// Faults struck in this process's runs: the survey's evidence the faults are reached.
static STRUCK: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// `(panics, hangs)` struck in this process.
pub fn struck() -> (u64, u64) {
    (STRUCK[0].load(Ordering::Relaxed), STRUCK[1].load(Ordering::Relaxed))
}

pub struct Hazards(Mutex<fastrand::Rng>);

impl Hazards {
    pub fn new(seed: u64) -> Self {
        quiet_injected_panics();
        Hazards(Mutex::new(fastrand::Rng::with_seed(seed ^ 0x6861_7a61_7264)))
    }

    pub fn draw(&self) -> Option<Fault> {
        let mut rng = self.0.lock();
        if rng.u32(0..ONE_IN) != 0 {
            return None;
        }
        Some(if rng.bool() { Fault::Panic } else { Fault::Hang })
    }

    /// A stream's fault: at its `next` after this many steps, or at its `close`.
    pub fn draw_stream(&self, next: Site, close: Site) -> StreamHazard {
        let Some(fault) = self.draw() else { return StreamHazard(None) };
        let mut rng = self.0.lock();
        let site = if rng.u32(0..4) == 0 { close } else { next };
        StreamHazard(Some((site, fault, rng.u32(0..8))))
    }
}

/// The one fault a provider stream plays out, if any, and when.
#[derive(Default)]
pub struct StreamHazard(Option<(Site, Fault, u32)>);

impl StreamHazard {
    /// The fault due at this step of `site`, given once; the caller records it.
    pub fn due(&mut self, site: Site) -> Option<Fault> {
        let (at, fault, steps) = self.0.as_mut()?;
        if *at != site {
            return None;
        }
        if *steps > 0 {
            *steps -= 1;
            return None;
        }
        let fault = *fault;
        self.0 = None;
        Some(fault)
    }
}

pub(super) fn count(fault: Fault) {
    STRUCK[usize::from(fault == Fault::Hang)].fetch_add(1, Ordering::Relaxed);
}

/// The fault played out: a panic at the poll, or a wait that never ends (but is dropped).
pub async fn strike<T>(fault: Fault, site: Site) -> T {
    match fault {
        Fault::Panic => panic!("{INJECTED}: {site:?}"),
        Fault::Hang => std::future::pending().await,
    }
}

/// Injected panics are expected: the default hook would print each one.
fn quiet_injected_panics() {
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let message = info.payload().downcast_ref::<String>();
            if !message.is_some_and(|m| m.starts_with(INJECTED)) {
                previous(info);
            }
        }));
    });
}
