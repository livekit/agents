//! What the realtime server remembers the agent saying is what the person heard of it.
//!
//! A reply the person cut short is truncated on the server where the playout stopped, and one
//! they heard none of is deleted: otherwise the model later refers to words nobody heard. Judged
//! on what the server held as the session closed its connection, for each message of a generation
//! whose playout had ended (finished or cut) before the session was asked to close.
use std::collections::BTreeMap;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::Input,
    record::{Kind, Recording},
};

/// Rounding between the playout's position and the frames it covers.
const SLACK_MS: u64 = 20;

pub struct RealtimeServerHoldsWhatWasHeard;

impl Rule for RealtimeServerHoldsWhatWasHeard {
    const CONTRACT: Contract = Contract {
        id: "realtime_server_holds_what_was_heard",
        guarantee: "As a realtime connection closes, no assistant message its server generated holds more audio than the person heard of it, once its playout ended before the close.",
        examples: &[truncated_to_what_was_heard],
        counterexamples: &[kept_whole_after_a_cut],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let close = r.inputs_view().close_asked_at();
        // An output that threw its playback count away leaves a speech's tail in the next one's
        // segment: what played of which generation is then the speaker's guess, not a fact.
        if r.inputs_view().accepted().any(|a| matches!(a.input, Input::SinkReset { .. })) {
            return;
        }
        // By generation: what was heard, when its playout last settled, and whether all of it
        // had by then.
        let mut heard: BTreeMap<u32, (u64, crate::record::Observed, bool)> = BTreeMap::new();
        for (at, generation, ms, whole) in r.heard_generations() {
            let entry = heard.entry(*generation).or_insert((0, *at, false));
            entry.0 += ms;
            entry.1 = entry.1.max(*at);
            entry.2 = *whole;
        }
        // Each socket's connection as the session closed it: what its server ended up
        // remembering. One closed before (a swap, a recycle) took its conversation with it.
        let Some(close) = close else { return };
        let mut last: BTreeMap<u32, &Vec<(u32, u64)>> = BTreeMap::new();
        for (_, socket, items) in r.held().iter().filter(|(at, ..)| *at >= close) {
            last.insert(*socket, items);
        }
        for (socket, items) in last {
            for (generation, kept) in items {
                // Never played out here (still playing at the close, or on another output).
                let Some((ms, ended, whole)) = heard.get(generation) else { continue };
                // Still playing (some of it never settled), or settled only as the close came.
                if !whole || close <= *ended {
                    continue;
                }
                check.expect_with(Subject::Resource(Kind::Realtime, socket), *kept <= ms + SLACK_MS, || {
                    format!("generation {generation}'s message keeps {kept} ms of audio; the person heard {ms} ms")
                });
            }
        }
    }
}

/// The person heard 120 ms of a 400 ms message, and the server kept 120.
fn truncated_to_what_was_heard() -> Recording {
    Authored::new().heard(3, 120).sent(Input::Close { drain: true }).held(0, vec![(3, 120)]).done()
}

/// The person heard 120 ms, and the server kept all 400.
fn kept_whole_after_a_cut() -> Recording {
    Authored::new().heard(3, 120).sent(Input::Close { drain: true }).held(0, vec![(3, 400)]).done()
}
