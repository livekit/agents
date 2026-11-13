use std::time::Duration;

use lk_agents_abi::Ts;

use crate::{
    check::{Check, Contract, Rule, Subject},
    fakes::room::{HeardFrame, Speaker},
    record::Recording,
};

// One capture chunk is what the output may have handed the source as the clear lands. The SDK's
// `clear_queue` cuts only what it buffered: a capture longer than its queue refills the speaker
// after it, so this holds only while the output never hands it one.
const CHUNK: Duration = Duration::from_millis(50);

pub struct RoomInterruptClearsWithinAChunk;

impl Rule for RoomInterruptClearsWithinAChunk {
    const CONTRACT: Contract = Contract {
        id: "room_interrupt_clears_within_a_chunk",
        guarantee: "After the agent clears a room audio source, nothing it had captured before plays more than one 50 ms capture chunk later.",
        examples: &[stops_in_a_chunk],
        counterexamples: &[keeps_talking],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        for speaker in r.room_speakers() {
            for (nth, clear) in speaker.clears.iter().enumerate() {
                let late = speaker
                    .heard
                    .iter()
                    .filter(|h| h.clears_before <= nth && h.until > clear.saturating_add(CHUNK))
                    .map(|h| h.until)
                    .max();
                check.expect_with(Subject::Session, late.is_none(), || {
                    format!(
                        "room audio cleared at {} ms, and audio captured before it played until {} ms",
                        ms(*clear),
                        late.map_or(0, ms)
                    )
                });
            }
        }
    }
}

fn ms(at: Ts) -> u64 {
    at.saturating_duration_since(Ts::ZERO).as_millis() as u64
}

fn played_until(until_ms: u64) -> Recording {
    let at = |ms: u64| Ts::ZERO.saturating_add(Duration::from_millis(ms));
    let mut r = Recording::default();
    r.set_room_speakers(vec![Speaker {
        clears: vec![at(100)],
        heard: vec![
            HeardFrame { clears_before: 0, until: at(100) },
            HeardFrame { clears_before: 0, until: at(until_ms) },
            // Captured after the clear: the next reply.
            HeardFrame { clears_before: 1, until: at(900) },
        ],
    }]);
    r
}

/// The chunk in flight at the clear plays out 50 ms later.
fn stops_in_a_chunk() -> Recording {
    played_until(150)
}

/// The rest of a 500 ms capture refills the speaker after the clear.
fn keeps_talking() -> Recording {
    played_until(500)
}
