//! A host `drain()` is answered `Ok` only once the frame has exited and nothing it was saying is
//! left.
//!
//! The reference's `AgentActivity.drain()` awaits `on_exit`, then `_pause_scheduling_task()`,
//! which waits for every speech task to finish, tools included. The driver answers at its
//! `Drained{refused: None}` step, so at that point of its journal the `on_exit` of the agent it
//! drains has finished (`HookFinished`, whoever issued it), and every speech the host was told
//! about (`Emitted(SpeechCreated)`) has been told done (`Emitted(SpeechDone)`). An `Ok` with no
//! such step behind it is an call the driver never decided.
use lk_agents_voice::{
    agent_activity::scheduling::{SpeechId, SpeechKind},
    agent_session::{HookId, HookKind},
    events::SessionEvent,
    speech_handle::{InputDetails, SpeechHandle},
    trace::{HandoffStep, HookEnd, Step},
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    drive::{Case, Setup},
    input::{AgentText, Input, Millis, SayInput},
    record::{Answer, Kind, Recording},
};

pub struct DrainAnswersOnceSettled;

impl Rule for DrainAnswersOnceSettled {
    const CONTRACT: Contract = Contract {
        id: "drain_answers_once_settled",
        guarantee: "drain() answers Ok only after its on_exit ended and every announced speech was announced done.",
        examples: &[settled_then_answered, answered_while_next_agent_exits],
        counterexamples: &[answered_without_step, answered_while_speaking, answered_before_exit],
        probe: Some(drain_while_speaking),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let host = r.host_view();
        let internal = r.internal_view();
        for call in host.api_calls() {
            if !matches!(call.input(), Input::Drain { .. }) {
                continue;
            }
            let Some((at, Answer::DrainOk)) = call.returned() else { continue };

            // The step the call was decided at: the last `Drained{None}` before the call arrived.
            let steps = internal.steps();
            let decided = steps.iter().rposition(|(when, step)| {
                *when <= at && matches!(step, Step::Handoff(HandoffStep::Drained { refused: None }))
            });
            check.expect(
                &call,
                decided.is_some(),
                format!("drain answered Ok at seq {} with no Drained step before it", at.seq),
            );
            let Some(decided) = decided else { continue };
            let before = &steps[..decided];

            // Its agent's `on_exit` ended before it, whichever of the drain, a handoff or the
            // close issued it. The drain is bound to the agent driving when it is asked: the frame
            // (the steps between two `AgentUpdated`) it arrived in. An agent swapped in since may
            // have its own drain waiting on its own exit.
            let swap =
                |step: &Step| matches!(step, Step::Handoff(HandoffStep::AgentUpdated { .. }));
            let asked = steps.partition_point(|(when, _)| *when < call.called_at());
            let frame = steps[..asked].iter().filter(|(_, s)| swap(s)).count();
            let mut at_frame = 0;
            for (_, step) in before {
                if swap(step) {
                    at_frame += 1;
                }
                let Step::Handoff(HandoffStep::HookIssued { hook, kind: HookKind::OnExit, .. }) =
                    step
                else {
                    continue;
                };
                if at_frame != frame {
                    continue;
                }
                let ended = before.iter().any(|(_, step)| {
                    matches!(step, Step::Handoff(HandoffStep::HookFinished { hook: h, kind: HookKind::OnExit, .. }) if h == hook)
                });
                check.expect(
                    &call,
                    ended,
                    format!("drain answered while its on_exit {hook:?} was still running"),
                );
            }

            // Nothing announced is still open.
            for (_, step) in before {
                let Step::Emitted(SessionEvent::SpeechCreated { speech, .. }) = step else {
                    continue;
                };
                let id = speech.id();
                let done = before.iter().any(|(_, s)| {
                    matches!(s, Step::Emitted(SessionEvent::SpeechDone { speech: d, .. }) if *d == id)
                });
                check.expect(
                    &call,
                    done,
                    format!("drain answered Ok while speech {} was still open", id.0),
                );
            }
        }
    }
}

fn drain() -> Authored {
    Authored::new()
        .sent_opening(Input::Drain { wait: true }, &[(Kind::Answer, crate::record::Detail::None)])
}

fn exit_issued(hook: u64) -> Step {
    Step::Handoff(HandoffStep::HookIssued {
        hook: HookId(hook),
        kind: HookKind::OnExit,
        agent: Some("agent".into()),
        speech: None,
        turn: None,
        at_tail: None,
    })
}

fn exit_finished() -> Step {
    Step::Handoff(HandoffStep::HookFinished {
        hook: HookId(2),
        kind: HookKind::OnExit,
        how: HookEnd::Returned { edits: 0 },
    })
}

fn drained() -> Step {
    Step::Handoff(HandoffStep::Drained { refused: None })
}

/// The exit ended, the frame drained, then the call.
fn settled_then_answered() -> Recording {
    drain()
        .journal(exit_issued(2))
        .journal(exit_finished())
        .journal(drained())
        .result(0, Answer::DrainOk)
        .done()
}

/// A drain asked of an agent an inline task then paused: answered once that agent's exit ended,
/// while the task's agent runs the exit its own drain asked for.
fn answered_while_next_agent_exits() -> Recording {
    let swapped = Step::Handoff(HandoffStep::AgentUpdated {
        tools: 0,
        chat_ctx: false,
        reopened: "",
        legs: Default::default(),
    });
    drain()
        .journal(exit_issued(2))
        .journal(swapped)
        .journal(exit_issued(4))
        .journal(exit_finished())
        .journal(drained())
        .result(0, Answer::DrainOk)
        .done()
}

/// An `Ok` the driver never decided.
fn answered_without_step() -> Recording {
    drain().journal(exit_issued(2)).result(0, Answer::DrainOk).done()
}

/// A speech announced and never announced done, yet the drain says the frame is quiet.
fn answered_while_speaking() -> Recording {
    let handle =
        SpeechHandle::new(SpeechId(7), String::from("speech_7"), true, InputDetails::default());
    let announced = Step::Emitted(SessionEvent::SpeechCreated {
        speech: std::sync::Arc::new(handle),
        kind: SpeechKind::Say,
        user_initiated: true,
    });
    drain()
        .journal(exit_issued(2))
        .journal(announced)
        .journal(exit_finished())
        .journal(drained())
        .result(0, Answer::DrainOk)
        .done()
}

/// Drained while its own `on_exit` never ended.
fn answered_before_exit() -> Recording {
    drain().journal(exit_issued(2)).journal(drained()).result(0, Answer::DrainOk).done()
}

/// Prerequisite: a drain asked while a `say()` is still being spoken, on a session whose host runs
/// hooks. The call must wait for the hook the runner answers and for the speech the runner lets
/// finish in its release phase.
fn drain_while_speaking() -> Case {
    Case {
        setup: Setup { hook_bound_ms: Some(2_000), ..Setup::default() },
        inputs: vec![
            Input::Say {
                text: AgentText("Goodbye for now.".into()),
                input: SayInput::Words,
                allow: None,
                to_ctx: true,
            },
            Input::Advance { ms: Millis(50) },
            Input::Drain { wait: true },
            Input::Advance { ms: Millis(50) },
        ],
        ..Case::default()
    }
}
