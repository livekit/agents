//! A frame's `on_exit` runs once, whichever of a host `drain()`, a handoff or the close asks for
//! it first.
//!
//! The three share one exit: a handoff or a close that follows a drain adopts the drain's hook
//! while it runs and skips the frame's `on_exit` once it has (C-10). A frame is the stretch of the
//! journal between two `AgentUpdated` steps -- a swap, a task entered, a parent resumed -- and a
//! second `HookIssued{OnExit}` for the same agent inside one is user code run twice: a goodbye
//! said twice, a transfer dialled twice.
use lk_agents_voice::{
    agent_session::{HookId, HookKind},
    trace::{HandoffStep, Step},
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, Setup},
    input::{AgentText, Input, Millis},
    record::Recording,
};

pub struct OnExitOncePerFrame;

impl Rule for OnExitOncePerFrame {
    const CONTRACT: Contract = Contract {
        id: "on_exit_once_per_frame",
        guarantee: "Each frame's on_exit is issued at most once, whether a drain, a handoff or the close asked for it.",
        examples: &[exited_once_per_frame],
        counterexamples: &[exited_twice],
        probe: Some(drain_then_close),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let internal = r.internal_view();
        // The agents whose `on_exit` was issued in the frame being read; a new frame starts empty.
        let mut exited: Vec<String> = Vec::new();
        for (at, step) in internal.steps() {
            match step {
                Step::Handoff(HandoffStep::AgentUpdated { .. }) => exited.clear(),
                Step::Handoff(HandoffStep::HookIssued {
                    kind: HookKind::OnExit,
                    agent: Some(agent),
                    ..
                }) => {
                    let again = exited.contains(agent);
                    check.expect(
                        Subject::Session,
                        !again,
                        format!("on_exit issued a second time for {agent} at seq {}", at.seq),
                    );
                    exited.push(agent.clone());
                }
                _ => {}
            }
        }
    }
}

fn on_exit(hook: u64, agent: &str) -> Step {
    Step::Handoff(HandoffStep::HookIssued {
        hook: HookId(hook),
        kind: HookKind::OnExit,
        agent: Some(agent.into()),
        speech: None,
        turn: None,
        at_tail: None,
    })
}

fn swapped() -> Step {
    Step::Handoff(HandoffStep::AgentUpdated {
        tools: 0,
        chat_ctx: false,
        reopened: "",
        legs: Default::default(),
    })
}

/// The drained frame's `on_exit`, a swap, the next frame's `on_exit` at the close.
fn exited_once_per_frame() -> Recording {
    Authored::new()
        .journal(on_exit(1, "agent"))
        .journal(swapped())
        .journal(on_exit(3, "agent"))
        .done()
}

/// A drain's `on_exit`, then the close issuing its own for the same frame.
fn exited_twice() -> Recording {
    Authored::new().journal(on_exit(1, "agent")).journal(on_exit(2, "agent")).done()
}

/// Prerequisite: a drain's `on_exit` still running when a handoff and then the close ask for the
/// same frame's exit, on a session whose host runs hooks -- the handoff must adopt the drain's hook
/// and the close the handoff's, not issue a second. The runner answers the hook in its release
/// phase, after both were asked.
fn drain_then_close() -> Case {
    Case {
        setup: Setup { hook_bound_ms: Some(2_000), ..Setup::default() },
        inputs: vec![
            Input::Drain { wait: true },
            Input::Handoff {
                agent: "next".into(),
                instructions: AgentText("You are next.".into()),
                tools: Vec::new(),
                pause_previous: false,
                resume: false,
            },
            Input::Advance { ms: Millis(50) },
            Input::Close { drain: false },
        ],
        ..Case::default()
    }
}
