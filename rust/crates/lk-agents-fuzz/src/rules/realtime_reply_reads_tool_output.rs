//! A realtime reply asked once a tool finished reads the tool's output.
use lk_agents_voice::trace::{RealtimeStep, Step, ToolStep};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Breach, Kind, Observed, Recording},
};

pub struct RealtimeReplyReadsToolOutput;

impl Rule for RealtimeReplyReadsToolOutput {
    const CONTRACT: Contract = Contract {
        id: "realtime_reply_reads_tool_output",
        guarantee: "A realtime reply the driver asks for once a tool's results are committed is asked after the tool's output reached the server: the server answers from what it holds when asked, and an output sent after the `response.create` (in the same breath, no reply asked after it) is read by no reply. A reply asked before the results were committed cannot read them, and is not judged.",
        examples: &[asked_while_the_tool_ran],
        counterexamples: &[asked_before_the_output],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let steps = r.internal_view().steps();
        let asked_at = |request: &str| {
            steps.iter().find_map(|(at, s)| match s {
                Step::Realtime(RealtimeStep::Sent { req_id: Some(id), .. }) if id == request => {
                    Some(*at)
                }
                _ => None,
            })
        };
        // When the call's results were committed: its speech's tools finished after it ended.
        let committed_at = |call: &str| {
            let (ended, speech) = steps.iter().find_map(|(at, s)| match s {
                Step::Tool(ToolStep::CallEnded { call_id, speech, .. }) if call_id == call => {
                    Some((*at, *speech))
                }
                _ => None,
            })?;
            steps.iter().find_map(|(at, s)| match s {
                Step::Tool(ToolStep::Finished { speech: s, .. }) if *s == speech && *at > ended => {
                    Some(*at)
                }
                _ => None,
            })
        };
        for (at, socket, breach) in r.inference_view().realtime_breaches() {
            let Breach::AskedBeforeOutput { response, request, call_id } = breach else { continue };
            let asked = request.as_deref().and_then(asked_at);
            let ran = |asked: Observed| committed_at(call_id).is_none_or(|at| asked < at);
            check.expect(
                Subject::Resource(Kind::Realtime, *socket),
                asked.is_some_and(ran),
                format!(
                    "{response} was asked after {call_id}'s results were committed and before its output, which arrived at {} ms: the reply read the call without it",
                    at.at_ms
                ),
            );
        }
    }
}

fn early() -> Breach {
    Breach::AskedBeforeOutput {
        response: "resp_1".into(),
        request: Some("req_1".into()),
        call_id: "call_1".into(),
    }
}

fn ended() -> Step {
    Step::Tool(ToolStep::CallEnded {
        speech: lk_agents_voice::agent_activity::scheduling::SpeechId(1),
        call_id: "call_1".into(),
        end: lk_agents_voice::tool_executor::ToolEnd::Done,
    })
}

fn committed() -> Step {
    Step::Tool(ToolStep::Finished {
        speech: lk_agents_voice::agent_activity::scheduling::SpeechId(1),
        committed: 2,
    })
}

fn sent() -> Step {
    Step::Realtime(RealtimeStep::Sent {
        kind: lk_agents_abi::RealtimeCommandKind::GenerateReply,
        req_id: Some("req_1".into()),
    })
}

/// A reply asked before the tool's results were committed, its output pushed in that breath.
fn asked_while_the_tool_ran() -> Recording {
    Authored::new().journal(ended()).journal(sent()).journal(committed()).breach(0, early()).done()
}

/// The follow-up's `response.create` went out after the commit and before the output.
fn asked_before_the_output() -> Recording {
    Authored::new().journal(ended()).journal(committed()).journal(sent()).breach(0, early()).done()
}
