//! A cost contract: a call wakes its thread a bounded number of times per second of virtual time.
//!
//! Every time the run's one thread goes idle and is woken again (a timer, a task polling) is
//! counted. A session's own work comes to tens of wake-ups a second (at most 29 over 2,048 survey
//! seeds); a task that polls instead of waiting, as the idle background audio mixer did every
//! millisecond before a61e6faa0, is a thousand more for as long as the call lasts. On a real call
//! that is CPU spent on nothing, multiplied by every call on the worker.
use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::Recording,
};

/// Wake-ups per second of virtual time a call may average.
pub const BOUND_PER_S: u64 = 200;
/// A run shorter than this is not judged: its start-up dominates.
const MIN_MS: u64 = 10_000;

pub struct IdleIsCheap;

impl Rule for IdleIsCheap {
    const CONTRACT: Contract = Contract {
        id: "idle_is_cheap",
        guarantee: "A call wakes its thread no more than a bound per second of virtual time, averaged over the call.",
        examples: &[a_quiet_call],
        counterexamples: &[a_polling_call],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let ms = r.now().at_ms;
        if ms < MIN_MS {
            return;
        }
        let parks = r.parks();
        check.expect_with(Subject::Session, parks * 1000 <= BOUND_PER_S * ms, || {
            format!(
                "{parks} wake-ups in {} s of virtual time: {} a second (bound {BOUND_PER_S})",
                ms / 1000,
                parks * 1000 / ms
            )
        });
    }
}

/// A minute of call, a hundred wake-ups a second.
fn a_quiet_call() -> Recording {
    let mut r = Authored::new().later(60_000).done();
    r.set_parks(6_000);
    r
}

/// A minute of call, a task polling every millisecond.
fn a_polling_call() -> Recording {
    let mut r = Authored::new().later(60_000).done();
    r.set_parks(66_000);
    r
}
