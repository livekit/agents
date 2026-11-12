use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Recording, StorageOperation, StorageStep},
};
use lk_agents_abi::voice::events::CloseReason;
use lk_agents_voice::{
    agent_session::machine::activity::scheduling::SpeechId,
    trace::{CloseStep, SpeechStep, Step},
};

pub struct StorageSavesFollowTurnOrder;

impl Rule for StorageSavesFollowTurnOrder {
    const CONTRACT: Contract = Contract {
        id: "storage_saves_follow_turn_order",
        guarantee: "Turns are saved in the order they were taken: a turn the person cut while its tools ran is committed or aborted before any later turn's save is asked for. Every save asked for is answered, unless the call closes first.",
        examples: &[in_turn_order, closed_before_the_answer],
        counterexamples: &[next_saved_first, never_answered],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let host = r.host_view();
        let saves = host.storage();
        let ends = |op: &StorageOperation| {
            matches!(op, StorageOperation::Settle | StorageOperation::Abandon)
        };
        let closing = r.close_begun().is_some();
        for (requested, step) in saves {
            let StorageStep::Requested { speech, operation } = step else { continue };
            if !ends(operation) {
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
        let order = r.speech_order();
        let turn = |speech: u64| order.get(&speech).copied().unwrap_or(usize::MAX);
        let cut = r.journal().iter().filter_map(|(_, step)| match step {
            Step::Speech(SpeechStep::ReleasedToTools { speech }) => Some(speech.0),
            _ => None,
        });
        for earlier in cut {
            let ended = saves.iter().find_map(|(at, step)| match step {
                StorageStep::Answered { speech, operation, .. }
                    if *speech == earlier && ends(operation) =>
                {
                    Some(*at)
                }
                _ => None,
            });
            for (asked, step) in saves {
                let StorageStep::Requested { speech, operation: StorageOperation::Settle } = step
                else {
                    continue;
                };
                if turn(*speech) <= turn(earlier) {
                    continue;
                }
                check.expect_with(
                    Subject::Speech(*speech),
                    ended.is_some_and(|ended| ended < *asked),
                    || format!("saved before turn {earlier}, cut while its tools ran, had ended"),
                );
            }
        }
    }
}

fn requested(speech: u64) -> StorageStep {
    StorageStep::Requested { speech, operation: StorageOperation::Settle }
}

fn answered(speech: u64) -> StorageStep {
    StorageStep::Answered {
        speech,
        operation: StorageOperation::Settle,
        delivered: true,
        result: Ok(()),
    }
}

fn cut(speech: u64) -> Step {
    Step::Speech(SpeechStep::ReleasedToTools { speech: SpeechId(speech) })
}

fn in_turn_order() -> Recording {
    Authored::new()
        .journal(cut(1))
        .later(1500)
        .storage(requested(1))
        .storage(answered(1))
        .storage(requested(2))
        .storage(answered(2))
        .done()
}

/// The call closed with the save in flight: its answer may never come.
fn closed_before_the_answer() -> Recording {
    let close = CloseStep::Requested { reason: CloseReason::UserInitiated, drain: false };
    Authored::new().storage(requested(1)).journal(Step::Close(close)).done()
}

/// The next turn was saved while the cut turn's tools still ran.
fn next_saved_first() -> Recording {
    Authored::new()
        .journal(cut(1))
        .storage(requested(2))
        .storage(answered(2))
        .later(1500)
        .storage(requested(1))
        .storage(answered(1))
        .done()
}

fn never_answered() -> Recording {
    Authored::new().storage(requested(1)).done()
}
