use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Recording, StorageOperation, StorageStep},
};
use lk_agents_abi::voice::events::CloseReason;
use lk_agents_voice::trace::{CloseStep, Step};

pub struct StorageSaveIsAnswered;

impl Rule for StorageSaveIsAnswered {
    const CONTRACT: Contract = Contract {
        id: "storage_save_is_answered",
        guarantee: "Every turn's save asked for, a commit or an abort, is answered, unless the call closes first.",
        examples: &[answered, closed_before_the_answer],
        counterexamples: &[never_answered],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let saves = r.host_view().storage();
        let closing = r.close_begun().is_some();
        for (requested, step) in saves {
            let StorageStep::Requested { speech, operation } = step else { continue };
            if !matches!(operation, StorageOperation::Settle | StorageOperation::Abandon) {
                continue;
            }
            let answered = saves.iter().any(|(at, step)| {
                requested < at
                    && matches!(step, StorageStep::Answered { speech: s, operation: op, .. }
                        if s == speech && op == operation)
            });
            check.expect_with(Subject::Speech(*speech), answered || closing, || {
                format!("its {operation:?} was never answered")
            });
        }
    }
}

fn requested(speech: u64) -> StorageStep {
    StorageStep::Requested { speech, operation: StorageOperation::Settle }
}

fn answered() -> Recording {
    Authored::new()
        .storage(requested(1))
        .storage(StorageStep::Answered {
            speech: 1,
            operation: StorageOperation::Settle,
            delivered: true,
            result: Ok(()),
        })
        .done()
}

/// The call closed with the save in flight: its answer may never come.
fn closed_before_the_answer() -> Recording {
    let close = CloseStep::Requested { reason: CloseReason::UserInitiated, drain: false };
    Authored::new().storage(requested(1)).journal(Step::Close(close)).done()
}

fn never_answered() -> Recording {
    Authored::new().storage(requested(1)).done()
}
