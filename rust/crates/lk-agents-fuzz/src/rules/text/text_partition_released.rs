use crate::check::{Check, Subject};
use crate::text::{PartitionRecord, TextContract, TextRecording, TextRule};

pub struct TextPartitionReleased;

impl TextRule for TextPartitionReleased {
    const CONTRACT: TextContract = TextContract {
        id: "text_partition_released",
        guarantee: "Once every turn ended and the partition_key ran its next turn, no lease holds the partition_key past its TTL and every run a turn started has ended, unless the database was down, lost a write or lost a worker.",
        examples: &[released, faulted_run_left],
        counterexamples: &[held, run_never_ended, held_after_a_fault],
    };

    /// A fault may leave a run unended (its end was the lost write); a lease always lapses.
    fn on_complete(r: &TextRecording, check: &mut Check) {
        let Some(partition_key) = r.partition_key else { return };
        check.expect(Subject::Session, !partition_key.held, "a lease still holds the partition_key past its TTL");
        if !partition_key.faulted {
            check.expect_with(Subject::Session, partition_key.unended == 0, || {
                format!("{} runs never ended", partition_key.unended)
            });
        }
    }
}

fn partition(unended: u32, held: bool, faulted: bool) -> TextRecording {
    TextRecording { partition_key: Some(PartitionRecord { unended, held, faulted }), ..TextRecording::default() }
}

fn released() -> TextRecording {
    partition(0, false, false)
}

fn faulted_run_left() -> TextRecording {
    partition(1, false, true)
}

fn held() -> TextRecording {
    partition(0, true, false)
}

fn run_never_ended() -> TextRecording {
    partition(1, false, false)
}

fn held_after_a_fault() -> TextRecording {
    partition(0, true, true)
}
