//! The inference simulations: the request machines of `lk-agents-inference`, driven on supplied
//! time against fake gateways that answer as the real ones do -- calibrated latencies, refused
//! connections, rejections, streams that drop or stall midway, replies that end without their
//! terminator, transports slow (or failing) to release an attempt -- while their consumer waits,
//! applies backpressure and gives up.
//!
//! What a fake will report enters the shared scheduler ([`crate::sched`]) only once due, as it
//! reaches a channel in production; the scheduler then orders what is pending across sources
//! (each attempt's transport, the consumer, the caller), a due wake first. Virtual time moves only
//! when nothing is pending, to the next report or deadline.
//!
//! Each simulation records every event it delivered with the effects it returned, and its rules
//! are functions of that trace, so a rule's counterexample is a trace written by hand.

pub mod avatar;
pub mod llm;

use std::time::Duration;

use lk_agents_abi::Ts;

use crate::sched::{Sched, Source};

/// Milliseconds on the simulation's clock, as the machines' [`Ts`].
pub fn ts(ms: u64) -> Ts {
    Ts::from_nanos(ms * 1_000_000)
}

/// A [`Ts`] back in milliseconds, rounded up: a deadline is due at the first whole millisecond
/// at or after it.
pub fn ms(at: Ts) -> u64 {
    at.as_nanos().div_ceil(1_000_000)
}

/// A duration in milliseconds.
pub fn dur(ms: u64) -> Duration {
    Duration::from_millis(ms)
}

/// What the fakes will report, and when: an item reaches the scheduler once it is due.
pub struct Timeline<S, E> {
    items: Vec<(u64, u64, S, E)>,
    seq: u64,
}

impl<S: Source, E> Default for Timeline<S, E> {
    fn default() -> Self {
        Timeline { items: Vec::new(), seq: 0 }
    }
}

impl<S: Source, E> Timeline<S, E> {
    /// `event` from `source` is reported at `due` ms.
    pub fn at(&mut self, due: u64, source: S, event: E) {
        self.items.push((due, self.seq, source, event));
        self.seq += 1;
    }

    /// Everything due by `now` reaches the scheduler, in the order it fell due.
    pub fn release(&mut self, now: u64, sched: &mut Sched<S, E>) {
        let mut due: Vec<(u64, u64, S, E)> = Vec::new();
        let mut i = 0;
        while i < self.items.len() {
            if self.items[i].0 <= now {
                due.push(self.items.swap_remove(i));
            } else {
                i += 1;
            }
        }
        due.sort_by_key(|(at, seq, ..)| (*at, *seq));
        for (_, _, source, event) in due {
            sched.push(source, event);
        }
    }

    /// When the next report falls due.
    pub fn next_due(&self) -> Option<u64> {
        self.items.iter().map(|(at, ..)| *at).min()
    }

    /// Drops what `source` has not reported yet (its transport was stopped); whether any of it
    /// was `what`.
    pub fn forget_where(&mut self, source: &S, what: impl Fn(&E) -> bool) -> bool {
        let mut found = false;
        self.items.retain(|(_, _, s, e)| {
            let forgotten = s == source;
            found |= forgotten && what(e);
            !forgotten
        });
        found
    }
}
