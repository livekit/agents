//! A speech waits for the storage receipt even if the commit is slow or fails.
use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Recording, StorageOperation, StorageStep},
};
use lk_agents_voice::events::SpeechOutcome;

pub struct StorageReceiptPrecedesSpeechDone;

impl Rule for StorageReceiptPrecedesSpeechDone {
    const CONTRACT: Contract = Contract {
        id: "storage_receipt_precedes_speech_done",
        guarantee: "A speech is announced done only after its settle or abandon receipt, including slow and failed commits.",
        examples: &[waited],
        counterexamples: &[done_before_receipt, done_without_receipt],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let host = r.host_view();
        for (requested, step) in host.storage() {
            let StorageStep::Requested { speech, operation } = step else { continue };
            if !matches!(operation, StorageOperation::Settle | StorageOperation::Abandon) {
                continue;
            }
            for (done, _) in host.speech(*speech).done_announcements() {
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
