//! `on_user_turn_completed` is awaited until it returns: the reference never cancels user code
//! nor moves on without it, so a slow hook, or a second user turn arriving while it runs, never
//! commits and answers the turn without the hook's edits (or despite its `StopResponse`).
use lk_agents_voice::agent_session::{HookId, HookKind};
use lk_agents_voice::trace::{HandoffStep, HookEnd, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::Recording,
};

pub struct TurnHookIsAwaited;

impl Rule for TurnHookIsAwaited {
    const CONTRACT: Contract = Contract {
        id: "turn_hook_is_awaited",
        guarantee: "on_user_turn_completed is never given up on while the session runs: it is answered, or abandoned by the close.",
        examples: &[answered],
        counterexamples: &[timed_out],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        for (_, step) in r.journal() {
            let Step::Handoff(HandoffStep::HookFinished {
                hook,
                kind: HookKind::OnUserTurnCompleted,
                how,
            }) = step
            else {
                continue;
            };
            check.expect(
                Subject::Session,
                !matches!(how, HookEnd::TimedOut),
                format!("on_user_turn_completed {hook:?} given up on before it returned"),
            );
        }
    }
}

fn finished(how: HookEnd) -> Recording {
    let hook = HookId(1);
    Authored::new()
        .journal(Step::Handoff(HandoffStep::HookIssued {
            hook,
            kind: HookKind::OnUserTurnCompleted,
            agent: None,
            speech: None,
            turn: None,
        }))
        .journal(Step::Handoff(HandoffStep::HookFinished {
            hook,
            kind: HookKind::OnUserTurnCompleted,
            how,
        }))
        .done()
}

/// The hook answered, with its edits.
fn answered() -> Recording {
    finished(HookEnd::Returned { edits: 1 })
}

/// A second turn (or a bound) ended the wait: the turn went on without the hook.
fn timed_out() -> Recording {
    finished(HookEnd::TimedOut)
}
