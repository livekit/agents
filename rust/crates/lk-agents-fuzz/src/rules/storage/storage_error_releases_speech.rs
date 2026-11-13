use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Recording, StorageOperation, StorageStep},
};
use lk_agents_abi::storage::StorageError;
use lk_agents_voice::{
    agent_session::machine::activity::scheduling::SpeechId,
    events::SpeechOutcome,
    trace::{InterruptStep, Step},
};

pub struct StorageErrorReleasesSpeech;

impl Rule for StorageErrorReleasesSpeech {
    const CONTRACT: Contract = Contract {
        id: "storage_error_releases_speech",
        guarantee: "Unavailable and VersionConflict settle and abandon receipts release the speech's completion barrier or close the bound call with an error.",
        examples: &[unavailable_settled, conflict_abandoned, released_after_watchdog, error_closed],
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
            // The watchdog already released a speech it ended; this receipt is its partition's release.
            let timed_out = r.internal_view().speech(*speech).timed_out_at();
            if timed_out.is_some_and(|ended| ended < *at) {
                continue;
            }
            let done = host.speech(*speech).done_announcements().iter().any(|(done, _)| at < done);
            check.expect(
                Subject::Speech(*speech),
                done || r.internal_view().error_close_requested_after(*at),
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
            _ => StorageError::VersionConflict { partition_key: "voice".into(), expected: 1 },
        }),
    }
}

fn unavailable_settled() -> Recording {
    Authored::new()
        .storage(failed(StorageOperation::Settle))
        .speech_done(1, SpeechOutcome::Completed)
        .done()
}

fn error_closed() -> Recording {
    Authored::new().storage(failed(StorageOperation::Settle))
        .journal(Step::Close(lk_agents_voice::trace::CloseStep::Requested {
            reason: lk_agents_abi::voice::events::CloseReason::Error, drain: false,
        })).done()
}

fn conflict_abandoned() -> Recording {
    Authored::new()
        .storage(failed(StorageOperation::Abandon))
        .speech_done(1, SpeechOutcome::Completed)
        .done()
}

/// Speech 2's watchdog ended it unreported: its partition's failed release comes after it is done.
fn released_after_watchdog() -> Recording {
    Authored::new()
        .storage(failed(StorageOperation::Settle))
        .speech_done(1, SpeechOutcome::Completed)
        .journal(Step::Interrupt(InterruptStep::InterruptionTimedOut { speech: SpeechId(2) }))
        .speech_done(2, SpeechOutcome::Interrupted)
        .later(30_000)
        .storage(StorageStep::Answered {
            speech: 2,
            operation: StorageOperation::Abandon,
            delivered: true,
            result: Err(StorageError::Unavailable("outage".into())),
        })
        .done()
}

fn unavailable_stuck() -> Recording {
    Authored::new().storage(failed(StorageOperation::Settle)).later(30_000).done()
}

fn conflict_stuck() -> Recording {
    Authored::new().storage(failed(StorageOperation::Abandon)).later(30_000).done()
}
