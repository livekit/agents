//! An interruption stops the avatar within one frame: once the agent's `lk.clear_buffer` reaches
//! the runner, no audio received before it, and no video lip-synced to that audio, is pushed out
//! more than one frame period later.
//!
//! One frame is what a generator has in flight when the clear lands: the tick it already took
//! from its buffer. Everything after it is the generator's to drop, once the runner has cleared
//! it. What the synchronizer had already queued still plays, as in the reference, which never
//! clears it (`rtc.AVSynchronizer.clear_queue` is not called).
use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{AvatarFrameLabel, AvatarStep, Observed, Recording},
    rules::avatar_av_in_sync::started,
};

pub struct AvatarInterruptClearsWithinAFrame;

impl Rule for AvatarInterruptClearsWithinAFrame {
    const CONTRACT: Contract = Contract {
        id: "avatar_interrupt_clears_within_a_frame",
        guarantee: "After a clear reaches the avatar runner, nothing it had received before is pushed out more than one frame period later.",
        examples: &[stops_in_a_frame],
        counterexamples: &[keeps_talking],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let Some((period, _, _)) = started(r) else { return };
        for (clear, _) in r.avatar().iter().filter(|(_, s)| matches!(s, AvatarStep::Clear)) {
            let late: Vec<_> = r
                .avatar()
                .iter()
                .filter(|(at, step)| {
                    at.seq > clear.seq
                        && at.at_ms > clear.at_ms + period
                        && matches!(step, AvatarStep::Pushed { frame: AvatarFrameLabel { from: Some(from), .. }, .. }
                            if from.seq < clear.seq)
                })
                .collect();
            check.expect_with(Subject::Session, late.is_empty(), || {
                format!(
                    "clear at {} ms, then {} frames of the cut audio pushed, the first at {} ms",
                    clear.at_ms,
                    late.len(),
                    late[0].0.at_ms
                )
            });
        }
    }
}

fn label(video: bool, tick: u64) -> AvatarFrameLabel {
    AvatarFrameLabel { video, tick, segment: Some(0), from: Some(Observed::at(2, 10)) }
}

fn pushed_after_clear(at_ms: u64) -> Recording {
    Authored::new()
        .avatar_at(0, AvatarStep::Started { period_ms: 40, latency_ms: 10, queue_ms: 100 })
        .avatar_at(10, AvatarStep::Received { segment: 0 })
        .avatar_at(100, AvatarStep::Clear)
        .avatar_at(at_ms, AvatarStep::Pushed { frame: label(true, 3), plays_ms: at_ms })
        .avatar_at(at_ms, AvatarStep::Pushed { frame: label(false, 3), plays_ms: at_ms })
        .done()
}

/// The tick in flight goes out 30 ms after the clear.
fn stops_in_a_frame() -> Recording {
    pushed_after_clear(130)
}

/// The cut segment is still being pushed 200 ms after the clear.
fn keeps_talking() -> Recording {
    pushed_after_clear(300)
}
