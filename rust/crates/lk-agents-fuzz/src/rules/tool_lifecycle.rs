//! A tool body the runner executed is announced to the host once as started and, once its speech
//! has finished, once as ended, in that order.
//!
//! The end is owed at the join, not when the body answers. The driver awaits a step's tool calls
//! in the model's order *after* the step is spoken, so a call that has answered is still legally
//! silent while an earlier sibling is outstanding or the step is still playing. What the product
//! does guarantee is that no started call outlives its speech: `tool_executor::end_live_tools`
//! settles whatever is still live when the speech is cut or the session closes. So the trigger
//! here is the speech's own completion.
use lk_agents_voice::events::ToolEndStatus;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    input::{AgentText, Input, Ref, ToolAnswer},
    record::Recording,
};

pub struct ToolLifecycle;

impl Rule for ToolLifecycle {
    const CONTRACT: Contract = Contract {
        id: "tool_lifecycle",
        guarantee: "Each executed tool call is announced started once and ended once, in that order, and is settled by the time its speech is done.",
        examples: &[full_lifecycle, answered_while_speech_runs],
        counterexamples: &[started_twice, ended_before_start, speech_done_without_end],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let inputs = r.inputs_view();
        let host = r.host_view();
        for tool in host.tools_run() {
            let (call_id, speech) = (tool.call_id(), tool.speech());
            let (starts, ends) = (tool.started_announcements(), tool.ended_announcements());

            // The host hears "started" at most once per call: a second one would make the host
            // believe two bodies ran.
            check.expect(&tool, starts.len() <= 1, format!("{call_id} started {} times", starts.len()));

            // The host hears "ended" at most once per call: a second one is a duplicate result.
            check.expect(&tool, ends.len() <= 1, format!("{call_id} ended {} times", ends.len()));

            // When both were heard, "started" came first.
            if let (Some(start), Some(end)) = (starts.first(), ends.first()) {
                check.expect(&tool, start < end, format!("{call_id} ended before it started"));
            }

            // No started call outlives its speech: once the host has been told the speech is done,
            // the call is settled too, one way or another. Until then a call that has answered may
            // legitimately be silent, waiting on its step or on an earlier sibling.
            let Some((done, _)) = host.speech(speech).done_announcements().first().copied() else {
                continue;
            };
            if inputs.driver_ran_after(done) {
                check.expect(
                    &tool,
                    !starts.is_empty(),
                    format!("{call_id} ran, its speech finished, the host was never told it started"),
                );
                check.expect(
                    &tool,
                    !ends.is_empty(),
                    format!(
                        "{call_id} ran, its speech {speech} finished, the host was never told it ended"
                    ),
                );
            }
        }
    }
}

/// Started, ended, and the speech finished: the whole announced lifecycle.
fn full_lifecycle() -> Recording {
    use lk_agents_voice::{agent_activity::scheduling::SpeechKind, events::SpeechOutcome};
    Authored::new()
        .tool_call("c1", 1)
        .tool_started("c1")
        .tool_ended("c1", ToolEndStatus::Done)
        .speech_created(1, SpeechKind::GenerateReply)
        .speech_done(1, SpeechOutcome::Completed)
        .then_act()
        .done()
}

/// The body answered and the host heard "started"; the speech is still running, so the end is
/// not owed yet. This is the case the rule must accept, and the one it used to reject.
fn answered_while_speech_runs() -> Recording {
    Authored::new()
        .tool_call("c1", 1)
        .tool_started("c1")
        .sent(Input::ToolAnswered {
            tool: Ref::new(0),
            answer: ToolAnswer::Return,
            text: AgentText("done".into()),
        })
        .then_act()
        .done()
}

/// One body, two "started" announcements.
fn started_twice() -> Recording {
    Authored::new().tool_call("c1", 1).tool_started("c1").tool_started("c1").done()
}

/// "Ended" reaches the host before "started".
fn ended_before_start() -> Recording {
    Authored::new()
        .tool_call("c1", 1)
        .tool_ended("c1", ToolEndStatus::Done)
        .tool_started("c1")
        .done()
}

/// The speech finished and the runner moved on, with a started call never settled.
fn speech_done_without_end() -> Recording {
    use lk_agents_voice::{agent_activity::scheduling::SpeechKind, events::SpeechOutcome};
    Authored::new()
        .tool_call("c1", 1)
        .tool_started("c1")
        .speech_created(1, SpeechKind::GenerateReply)
        .speech_done(1, SpeechOutcome::Completed)
        .then_act()
        .done()
}
