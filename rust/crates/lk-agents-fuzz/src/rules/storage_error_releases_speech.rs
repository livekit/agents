//! A failed storage receipt still releases the speech's completion barrier.
use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Recording, StorageOperation, StorageStep},
};
use lk_agents_abi::storage::StorageError;
use lk_agents_voice::events::SpeechOutcome;

pub struct StorageErrorReleasesSpeech;

impl Rule for StorageErrorReleasesSpeech {
    const CONTRACT: Contract = Contract {
        id: "storage_error_releases_speech",
        guarantee: "Unavailable and VersionConflict settle and abandon receipts release the speech's completion barrier.",
        examples: &[unavailable_settled, conflict_abandoned],
        counterexamples: &[unavailable_stuck, conflict_stuck],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let host = r.host_view();
        for (at, step) in host.storage() {
            let StorageStep::Answered { speech, operation, result: Err(error), delivered: true } =
                step
            else {
                continue;
            };
            if !matches!(operation, StorageOperation::Settle | StorageOperation::Abandon)
                || !matches!(
                    error,
                    StorageError::Unavailable(_) | StorageError::VersionConflict { .. }
                )
            {
                continue;
            }
            let done = host.speech(*speech).done_announcements().iter().any(|(done, _)| at < done);
            check.expect(
                Subject::Speech(*speech),
                done,
                "storage error left speech waiting for completion",
            );
        }
    }
}

fn failed(operation: StorageOperation) -> StorageStep {
    StorageStep::Answered {
        speech: 1,
        operation,
        delivered: true,
        result: Err(match operation {
            StorageOperation::Settle => StorageError::Unavailable("outage".into()),
            _ => StorageError::VersionConflict { lane: "voice".into(), expected: 1 },
        }),
    }
}

fn unavailable_settled() -> Recording {
    Authored::new()
        .storage(failed(StorageOperation::Settle))
        .speech_done(1, SpeechOutcome::Completed)
        .done()
}

fn conflict_abandoned() -> Recording {
    Authored::new()
        .storage(failed(StorageOperation::Abandon))
        .speech_done(1, SpeechOutcome::Completed)
        .done()
}

fn unavailable_stuck() -> Recording {
    Authored::new().storage(failed(StorageOperation::Settle)).later(30_000).done()
}

fn conflict_stuck() -> Recording {
    Authored::new().storage(failed(StorageOperation::Abandon)).later(30_000).done()
}
