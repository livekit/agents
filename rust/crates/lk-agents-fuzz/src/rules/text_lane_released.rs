use crate::check::{Check, Subject};
use crate::text::{LaneRecord, TextContract, TextRecording, TextRule};

pub struct TextLaneReleased;

impl TextRule for TextLaneReleased {
    const CONTRACT: TextContract = TextContract {
        id: "text_lane_released",
        guarantee: "Once every turn ended and the lane ran its next turn, no lease holds the lane past its TTL and every run a turn started has ended, unless the database was down, lost a write or lost a worker.",
        examples: &[released, faulted_run_left],
        counterexamples: &[held, run_never_ended, held_after_a_fault],
    };

    /// A fault may leave a run unended (its end was the lost write); a lease always lapses.
    fn on_complete(r: &TextRecording, check: &mut Check) {
        let Some(lane) = r.lane else { return };
        check.expect(Subject::Session, !lane.held, "a lease still holds the lane past its TTL");
        if !lane.faulted {
            check.expect_with(Subject::Session, lane.unended == 0, || {
                format!("{} runs never ended", lane.unended)
            });
        }
    }
}

fn lane(unended: u32, held: bool, faulted: bool) -> TextRecording {
    TextRecording { lane: Some(LaneRecord { unended, held, faulted }), ..TextRecording::default() }
}

fn released() -> TextRecording {
    lane(0, false, false)
}

fn faulted_run_left() -> TextRecording {
    lane(1, false, true)
}

fn held() -> TextRecording {
    lane(0, true, false)
}

fn run_never_ended() -> TextRecording {
    lane(1, false, false)
}

fn held_after_a_fault() -> TextRecording {
    lane(0, true, true)
}
