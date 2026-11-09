use std::collections::BTreeSet;

use lk_agents_voice::{
    agent_session::{HookId, HookKind, machine::activity::scheduling::SpeechId},
    events::{SpeechOutcome, ToolEndStatus},
    trace::{HandoffStep, HookEnd, InterruptCause, InterruptStep, Step, ToolStep},
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Observed, Recording, StorageOperation, StorageStep},
};

pub struct InterruptEndsTheReply;

impl Rule for InterruptEndsTheReply {
    const CONTRACT: Contract = Contract {
        id: "interrupt_ends_the_reply",
        guarantee: "An interrupted speech's reply ends on its own: the interruption watchdog ends it only while the reply awaits what an interrupt does not cancel, application code (one of its tool calls, or a host hook issued for it) or the voice lane it commits the person's turn to (its take, or its settle or abandon receipt).",
        examples: &[
            ended_before_the_watchdog,
            watchdog_ends_a_running_tool,
            watchdog_ends_an_awaited_hook,
            watchdog_ends_a_lane_wait,
            watchdog_ends_a_receipt_wait,
            wait_ends_in_the_watchdogs_instant,
        ],
        counterexamples: &[
            watchdog_ends_a_stuck_reply,
            tool_ended_before_the_watchdog,
            hook_answered_before_the_watchdog,
            lane_taken_before_the_watchdog,
            lane_loss_watched,
        ],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let internal = r.internal_view();
        let steps = internal.steps();
        let storage = r.host_view().storage();
        let interrupted: BTreeSet<u64> = steps
            .iter()
            .filter_map(|(_, step)| match step {
                Step::Interrupt(InterruptStep::Interrupted { speech, .. }) => Some(speech.0),
                _ => None,
            })
            .collect();
        for speech in interrupted {
            let Some(fired) = internal.speech(speech).timed_out_at() else {
                check.expect(Subject::Speech(speech), true, "");
                continue;
            };
            // Whether a wait begun at `from` was still open when the watchdog fired. One that
            // ended in the watchdog's own instant left the reply no time to end.
            let open_at_watchdog = |from: Observed, until: Option<Observed>| {
                from < fired && until.is_none_or(|until| until.at_ms >= fired.at_ms)
            };
            // An interrupt cuts every provider open and read, node call and playout wait of the
            // reply. It does not cut application code (agents.md, "Cancellation and cleanup"): a
            // tool call handed to the runner, or a host hook the reply awaits
            // (`function_tools_executed`), runs on, and the watchdog is what cancels the speech's
            // tasks (`SpeechTasks`). Nor the lane: the voice turn holds it to the commit of its
            // reply (voice-session.md), an interrupted reply still commits the person's turn, and
            // its storage receipt precedes its completion.
            let application_code = steps.iter().any(|(from, step)| {
                let ended = |is_end: &dyn Fn(&Step) -> bool| {
                    steps.iter().find(|(at, step)| from < at && is_end(step)).map(|(at, _)| *at)
                };
                let until = match step {
                    Step::Tool(ToolStep::CallStarted { speech: on, call_id, hosted: true, .. })
                        if on.0 == speech =>
                    {
                        ended(&|step| {
                            matches!(step, Step::Tool(ToolStep::CallEnded { call_id: c, .. }) if c == call_id)
                        })
                    }
                    Step::Handoff(HandoffStep::HookIssued { hook, speech: Some(on), .. })
                        if on.0 == speech =>
                    {
                        ended(&|step| {
                            matches!(step, Step::Handoff(HandoffStep::HookFinished { hook: h, .. }) if h == hook)
                        })
                    }
                    _ => return false,
                };
                open_at_watchdog(*from, until)
            });
            // The lane's take and its settle or abandon receipt; observing its loss is no wait.
            let lane = storage.iter().any(|(asked, step)| {
                let StorageStep::Requested { speech: s, operation } = step else { return false };
                let answered = storage.iter().find_map(|(at, step)| match step {
                    StorageStep::Answered { speech: a, operation: o, .. }
                        if a == s && o == operation && asked < at =>
                    {
                        Some(*at)
                    }
                    _ => None,
                });
                *s == speech
                    && *operation != StorageOperation::ObserveLoss
                    && open_at_watchdog(*asked, answered)
            });
            check.expect_with(Subject::Speech(speech), application_code || lane, || {
                format!(
                    "the interruption watchdog ended the reply at {} ms while it awaited no tool call, hook or lane: a wait the interrupt should cut kept it alive",
                    fired.at_ms
                )
            });
        }
    }
}

fn interrupted(speech: u64) -> Step {
    Step::Interrupt(InterruptStep::Interrupted {
        speech: SpeechId(speech),
        cause: InterruptCause::Host,
        force: true,
    })
}

fn watchdog(speech: u64) -> Step {
    Step::Interrupt(InterruptStep::InterruptionTimedOut { speech: SpeechId(speech) })
}

fn tool(speech: u64, call_id: &str, ended: bool) -> Step {
    let (speech, call_id) = (SpeechId(speech), call_id.to_owned());
    match ended {
        false => Step::Tool(ToolStep::CallStarted {
            speech,
            call_id,
            name: "lookup".into(),
            hosted: true,
        }),
        true => Step::Tool(ToolStep::CallEnded { speech, call_id, end: Some(ToolEndStatus::Done) }),
    }
}

fn hook(speech: u64, finished: bool) -> Step {
    let (hook, kind) = (HookId(1), HookKind::FunctionToolsExecuted);
    match finished {
        false => Step::Handoff(HandoffStep::HookIssued {
            hook,
            kind,
            agent: None,
            speech: Some(SpeechId(speech)),
            turn: None,
        }),
        true => Step::Handoff(HandoffStep::HookFinished {
            hook,
            kind,
            how: HookEnd::Returned { edits: 0 },
        }),
    }
}

fn lane(operation: StorageOperation, answered: bool) -> StorageStep {
    match answered {
        false => StorageStep::Requested { speech: 1, operation },
        true => StorageStep::Answered { speech: 1, operation, result: Ok(()), delivered: true },
    }
}

/// The interrupted reply wound down by itself.
fn ended_before_the_watchdog() -> Recording {
    Authored::new()
        .chat(1)
        .journal(interrupted(1))
        .later(40)
        .speech_done(1, SpeechOutcome::Interrupted)
        .done()
}

/// The reply awaited its tool call, still running when the watchdog cancelled it.
fn watchdog_ends_a_running_tool() -> Recording {
    Authored::new()
        .journal(tool(1, "call_1", false))
        .journal(interrupted(1))
        .later(5_000)
        .journal(watchdog(1))
        .journal(tool(1, "call_1", true))
        .speech_done(1, SpeechOutcome::Interrupted)
        .done()
}

/// The reply's tools settled and it awaits the host's `function_tools_executed` handlers.
fn watchdog_ends_an_awaited_hook() -> Recording {
    Authored::new()
        .journal(interrupted(1))
        .journal(hook(1, false))
        .later(5_000)
        .journal(watchdog(1))
        .speech_done(1, SpeechOutcome::Interrupted)
        .done()
}

/// The reply still waits for the lane a slow database holds.
fn watchdog_ends_a_lane_wait() -> Recording {
    Authored::new()
        .storage(lane(StorageOperation::Acquire, false))
        .journal(interrupted(1))
        .later(5_000)
        .journal(watchdog(1))
        .speech_done(1, SpeechOutcome::Interrupted)
        .done()
}

/// The hook's own deadline and the watchdog fell in the same instant: the hook ended first, but
/// the reply had no time to go on.
fn wait_ends_in_the_watchdogs_instant() -> Recording {
    Authored::new()
        .journal(interrupted(1))
        .journal(hook(1, false))
        .later(5_000)
        .journal(hook(1, true))
        .journal_at_once(watchdog(1))
        .speech_done(1, SpeechOutcome::Interrupted)
        .done()
}

/// Nothing the interrupt spares was running: the reply sat on a wait the interrupt should cut.
fn watchdog_ends_a_stuck_reply() -> Recording {
    Authored::new()
        .chat(1)
        .journal(interrupted(1))
        .later(5_000)
        .journal(watchdog(1))
        .speech_done(1, SpeechOutcome::Interrupted)
        .done()
}

/// The reply's tool call ended, then the reply still waited out the watchdog; another speech's
/// running call does not excuse it.
fn tool_ended_before_the_watchdog() -> Recording {
    Authored::new()
        .journal(tool(1, "call_1", false))
        .journal(tool(2, "call_2", false))
        .journal(interrupted(1))
        .later(100)
        .journal(tool(1, "call_1", true))
        .later(4_900)
        .journal(watchdog(1))
        .speech_done(1, SpeechOutcome::Interrupted)
        .done()
}

/// The hook answered, then the reply still waited out the watchdog.
fn hook_answered_before_the_watchdog() -> Recording {
    Authored::new()
        .journal(interrupted(1))
        .journal(hook(1, false))
        .later(100)
        .journal(hook(1, true))
        .later(4_900)
        .journal(watchdog(1))
        .speech_done(1, SpeechOutcome::Interrupted)
        .done()
}

/// The reply took the lane, then waited out the watchdog.
fn lane_taken_before_the_watchdog() -> Recording {
    Authored::new()
        .storage(lane(StorageOperation::Acquire, false))
        .journal(interrupted(1))
        .later(100)
        .storage(lane(StorageOperation::Acquire, true))
        .later(4_900)
        .journal(watchdog(1))
        .speech_done(1, SpeechOutcome::Interrupted)
        .done()
}

/// The reply's lane release waits on a slow database: its receipt precedes its completion.
fn watchdog_ends_a_receipt_wait() -> Recording {
    Authored::new()
        .storage(lane(StorageOperation::Acquire, false))
        .storage(lane(StorageOperation::Acquire, true))
        .journal(interrupted(1))
        .storage(lane(StorageOperation::Abandon, false))
        .later(5_000)
        .journal(watchdog(1))
        .speech_done(1, SpeechOutcome::Interrupted)
        .done()
}

/// Watching for the lane's loss is no wait: the reply had nothing left to await.
fn lane_loss_watched() -> Recording {
    Authored::new()
        .storage(lane(StorageOperation::Acquire, false))
        .storage(lane(StorageOperation::Acquire, true))
        .storage(lane(StorageOperation::ObserveLoss, false))
        .journal(interrupted(1))
        .later(5_000)
        .journal(watchdog(1))
        .speech_done(1, SpeechOutcome::Interrupted)
        .done()
}
