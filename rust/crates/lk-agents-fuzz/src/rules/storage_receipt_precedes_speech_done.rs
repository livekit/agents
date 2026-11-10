use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Recording, StorageOperation, StorageStep},
};
use lk_agents_voice::{
    agent_session::machine::activity::scheduling::SpeechId,
    events::SpeechOutcome,
    trace::{CloseAct, CloseStep, InterruptStep, Step},
};

pub struct StorageReceiptPrecedesSpeechDone;

impl Rule for StorageReceiptPrecedesSpeechDone {
    const CONTRACT: Contract = Contract {
        id: "storage_receipt_precedes_speech_done",
        guarantee: "A speech is announced done only after its settle or abandon receipt, including slow and failed commits. A speech the interruption watchdog ended never reports its turn: its lane is released afterwards. Nor does one the close released at its bound, its commit still pending (a lane another live worker holds) or reported afterwards, within the lane's release bound.",
        examples: &[
            waited,
            released_after_watchdog,
            released_by_the_close,
            reported_after_the_close,
        ],
        counterexamples: &[
            done_before_receipt,
            done_without_receipt,
            watchdog_cut_a_pending_commit,
            released_after_done,
        ],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let host = r.host_view();
        for (requested, step) in host.storage() {
            let StorageStep::Requested { speech, operation } = step else { continue };
            if !matches!(operation, StorageOperation::Settle | StorageOperation::Abandon) {
                continue;
            }
            let timed_out = r.internal_view().speech(*speech).timed_out_at();
            if timed_out.is_some_and(|ended| ended < *requested) {
                continue;
            }
            // The close's release announces the speech done itself, whenever its turn reports.
            let released = r.journal().iter().find_map(|(at, step)| {
                matches!(step, Step::Close(CloseStep::Performed(CloseAct::Released { speech: s }))
                    if s.0 == *speech)
                .then_some(*at)
            });
            for (done, _) in host.speech(*speech).done_announcements() {
                if released.is_some_and(|released| released < done) {
                    continue;
                }
                let received = host.storage().iter().any(|(at, step)| {
                    *requested < *at
                        && *at < done
                        && matches!(step,
                        StorageStep::Answered { speech: id, operation: op, delivered: true, .. }
                        if id == speech && op == operation)
                });
                check.expect(
                    Subject::Speech(*speech),
                    received,
                    "speech completed before its storage receipt",
                );
            }
        }
    }
}

fn request() -> StorageStep {
    StorageStep::Requested { speech: 1, operation: StorageOperation::Settle }
}

fn receipt() -> StorageStep {
    StorageStep::Answered {
        speech: 1,
        operation: StorageOperation::Settle,
        delivered: true,
        result: Err(lk_agents_abi::storage::StorageError::Unavailable("outage".into())),
    }
}

fn waited() -> Recording {
    Authored::new()
        .storage(request())
        .later(1500)
        .storage(receipt())
        .speech_done(1, SpeechOutcome::Completed)
        .done()
}

/// The close released speech 2 at its bound with its commit pending; speech 1 waited.
fn released_by_the_close() -> Recording {
    let settle = StorageOperation::Settle;
    Authored::new()
        .storage(request())
        .storage(receipt())
        .speech_done(1, SpeechOutcome::Completed)
        .storage(StorageStep::Requested { speech: 2, operation: settle })
        .journal(Step::Close(CloseStep::Performed(CloseAct::Released { speech: SpeechId(2) })))
        .speech_done(2, SpeechOutcome::Interrupted)
        .done()
}

/// Speech 1 waited; the close released speech 2 at its bound, its cut turn reported afterwards.
fn reported_after_the_close() -> Recording {
    let settle = StorageOperation::Settle;
    Authored::new()
        .storage(request())
        .storage(receipt())
        .speech_done(1, SpeechOutcome::Completed)
        .journal(Step::Close(CloseStep::Performed(CloseAct::Released { speech: SpeechId(2) })))
        .speech_done(2, SpeechOutcome::Interrupted)
        .storage(StorageStep::Requested { speech: 2, operation: settle })
        .storage(StorageStep::Answered {
            speech: 2,
            operation: settle,
            delivered: true,
            result: Ok(()),
        })
        .done()
}

/// Announced done before its receipt, then released by the close: the release did not announce it.
fn released_after_done() -> Recording {
    Authored::new()
        .storage(request())
        .speech_done(1, SpeechOutcome::Interrupted)
        .journal(Step::Close(CloseStep::Performed(CloseAct::Released { speech: SpeechId(1) })))
        .storage(receipt())
        .done()
}

/// Speech 2's watchdog ended it unreported: its lane is released after it is done.
fn released_after_watchdog() -> Recording {
    let abandon = StorageOperation::Abandon;
    Authored::new()
        .storage(request())
        .storage(receipt())
        .speech_done(1, SpeechOutcome::Completed)
        .journal(Step::Interrupt(InterruptStep::InterruptionTimedOut { speech: SpeechId(2) }))
        .speech_done(2, SpeechOutcome::Interrupted)
        .later(30_000)
        .storage(StorageStep::Requested { speech: 2, operation: abandon })
        .storage(StorageStep::Answered {
            speech: 2,
            operation: abandon,
            delivered: true,
            result: Ok(()),
        })
        .done()
}

/// The watchdog ended a speech whose commit was still in flight.
fn watchdog_cut_a_pending_commit() -> Recording {
    Authored::new()
        .storage(request())
        .journal(Step::Interrupt(InterruptStep::InterruptionTimedOut { speech: SpeechId(1) }))
        .speech_done(1, SpeechOutcome::Interrupted)
        .later(1500)
        .storage(receipt())
        .done()
}

fn done_before_receipt() -> Recording {
    Authored::new()
        .storage(request())
        .speech_done(1, SpeechOutcome::Completed)
        .later(1500)
        .storage(receipt())
        .done()
}

fn done_without_receipt() -> Recording {
    Authored::new().storage(request()).speech_done(1, SpeechOutcome::Completed).done()
}
