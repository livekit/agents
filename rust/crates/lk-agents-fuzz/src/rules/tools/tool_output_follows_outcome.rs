use lk_agents_abi::{ChatItem, FunctionCall, Role, llm::tool_executor::WAITING_PREFIX};
use lk_agents_voice::{
    agent::Source,
    events::SessionEvent,
    trace::{HandoffStep, Step, ToolStep},
};
use std::collections::BTreeSet;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{AgentText, Input, Ref, ToolAnswer},
    record::{Detail, Kind, Recording},
};

// Mirrors the reference's executor and `_apply_transition`. A cut step's outputs and calls the
// close settles as errors are not held to the reply flag. A `VoiceTask`'s `Completed` ends the
// task, so its agent asks for no reply. A detached body's final uses its unique item ID.

pub struct ToolOutputFollowsOutcome;

impl Rule for ToolOutputFollowsOutcome {
    const CONTRACT: Contract = Contract {
        id: "tool_output_follows_outcome",
        guarantee: "Each tool outcome retains its output, error flag and reply request. Human responses retain their own identity and reach an admitted input occurrence once, without overwriting its original question.",
        examples: &[
            a_value_asking_for_no_reply,
            a_question_answered,
            an_unselected_question_waits,
            a_reply_only_answers_the_selected_question,
        ],
        counterexamples: &[
            a_completion_recorded_silent,
            a_failure_not_an_error,
            a_question_overwritten_by_its_response,
            a_response_delivered_twice,
        ],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let inputs = r.inputs_view();
        let host = r.host_view();
        let closing = inputs.closing_asked_at();
        // Each output as first announced, and every announcement in order.
        let announced: Vec<_> = r
            .journal()
            .iter()
            .filter_map(|(at, s)| match s {
                Step::Emitted(SessionEvent::ConversationItemAdded { item, source, .. }) => {
                    Some((*at, &**item, source.speech()))
                }
                _ => None,
            })
            .collect();
        // When each speech was told done.
        let mut done = std::collections::BTreeMap::new();
        for (at, step) in r.journal() {
            if let Step::Emitted(SessionEvent::SpeechDone { speech, .. }) = step {
                done.entry(speech.0).or_insert(*at);
            }
        }
        // Each task's stretch, from its entry to its end: a speech created in one is the task's.
        let mut stretches = Vec::new();
        let mut open = Vec::new();
        for (at, step) in r.journal() {
            match step {
                Step::Handoff(HandoffStep::TaskEntered { task, .. }) => open.push((*task, *at)),
                Step::Handoff(HandoffStep::TaskEnded { task, .. }) => {
                    if let Some(i) = open.iter().position(|(t, _)| t == task) {
                        stretches.push((open.remove(i).1, *at));
                    }
                }
                _ => {}
            }
        }
        stretches.extend(open.into_iter().map(|(_, from)| (from, crate::record::Observed::END)));
        let in_task = |speech: u64| {
            let created = host.speech(speech).created_at();
            created.is_some_and(|c| stretches.iter().any(|(from, to)| *from < c && c <= *to))
        };
        // A tool's first report may be the host's: progress detaches the body, which answers
        // with its final; an end answers the call, or a detached body's final, and the body's
        // own answer comes too late.
        let mut detached = BTreeSet::new();
        let mut ended = BTreeSet::new();
        for applied in inputs.accepted() {
            let (tool, answer, outcome) = match &applied.input {
                Input::ToolUpdate { tool, progress, end } => {
                    if ended.contains(&tool.nth) {
                        continue;
                    }
                    if progress.is_some() {
                        detached.insert(tool.nth);
                        continue;
                    }
                    ended.insert(tool.nth);
                    (tool, end, crate::fakes::tools::update_outcome(None, *end))
                }
                Input::ToolAnswered { tool, answer, text: AgentText(text) } => {
                    if *answer == ToolAnswer::Detached {
                        detached.insert(tool.nth);
                    }
                    if ended.contains(&tool.nth)
                        || matches!(answer, ToolAnswer::Cancelled | ToolAnswer::Detached)
                    {
                        continue;
                    }
                    (tool, answer, crate::fakes::tools::outcome_of(*answer, text))
                }
                _ => continue,
            };
            let Some(Detail::Tool { item_id, call_id, speech, .. }) =
                r.resource(Kind::Tool, tool.nth).map(|t| t.detail().clone())
            else {
                continue;
            };
            let detached = detached.contains(&tool.nth);
            let recorded =
                if detached { format!("{item_id}_final_out") } else { format!("{item_id}_out") };
            let Some((at, ChatItem::FunctionCallOutput(output), _)) =
                announced.iter().find(|(_, item, _)| match item {
                    ChatItem::FunctionCallOutput(output) => output.id == recorded,
                    _ => false,
                })
            else {
                continue;
            };
            // The close settles a call it cut as an error of its own, a cut speech as cancelled.
            if closing.is_some_and(|c| c <= *at) || host.speech(speech).is_interrupted() {
                continue;
            }
            let mut shape = lk_agents_core::tool_outcome::shape(
                &outcome,
                lk_agents_core::tool_outcome::Mode::Voice,
            );
            let over = done.get(&speech).is_some_and(|done| done < at);
            if *answer == ToolAnswer::Completed && in_task(speech) || over && !detached {
                shape.reply_required = false;
            }
            let want =
                lk_agents_core::tool_outcome::exact_output_note(&outcome).unwrap_or(shape.text);
            let subject = Subject::Resource(Kind::Tool, tool.nth);
            check.expect_with(subject.clone(), output.output == want, || {
                format!(
                    "{call_id} answered {answer:?} is recorded {:?}, not {want:?}",
                    output.output
                )
            });
            check.expect_with(subject.clone(), output.is_error == shape.is_error, || {
                format!(
                    "{call_id} answered {answer:?} is recorded with is_error={}",
                    output.is_error
                )
            });
            check.expect_with(subject, output.reply_required == shape.reply_required, || {
                format!(
                    "{call_id} answered {answer:?} asks for a reply: {}, not {}",
                    output.reply_required, shape.reply_required
                )
            });
            if *answer == ToolAnswer::Ask && !detached {
                let replaced = r.journal().iter().any(|(later, step)| {
                    *at < *later
                        && matches!(step,
                            Step::Emitted(SessionEvent::ConversationItemAdded {
                                item,
                                source: Source::ToolOutput { .. } | Source::ToolUpdate { .. },
                                ..
                            }) if matches!(&**item, ChatItem::FunctionCallOutput(next)
                                if next.id == recorded && next.output != want))
                });
                check.expect_with(Subject::Resource(Kind::Tool, tool.nth), !replaced, || {
                    format!("the original question for {item_id} was replaced by a human response")
                });
            }
        }
        let mut delivered = BTreeSet::new();
        for (_, step) in r.journal() {
            let Step::Tool(ToolStep::InputResponseDelivered { request, message_id, .. }) = step
            else {
                continue;
            };
            check.expect_with(Subject::Session, delivered.insert(request), || {
                format!("the same input occurrence {request:?} received a response twice")
            });
            let human = announced.iter().any(|(_, item, _)| {
                matches!(item, ChatItem::Message(message)
                    if message.id == *message_id && message.role == Role::User)
            });
            check.expect_with(Subject::Session, human, || {
                format!("the response for {request:?} lost its human message {message_id}")
            });
        }
    }
}

fn output(call_id: &str, text: &str, is_error: bool, reply_required: bool) -> Step {
    use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId;
    let mut out = lk_agents_abi::FunctionCallOutput::new(
        format!("item_{call_id}_out"),
        call_id,
        text,
        is_error,
        std::time::SystemTime::UNIX_EPOCH,
    );
    out.reply_required = reply_required;
    Step::Emitted(SessionEvent::ConversationItemAdded {
        item: Box::new(ChatItem::FunctionCallOutput(out)),
        agent: "agent".into(),
        source: lk_agents_voice::agent::Source::ToolOutput { speech: SpeechId(1), step: 1 },
        visibility: lk_agents_voice::agent::Visibility::Both,
    })
}

fn user(text: &str) -> Step {
    let m =
        lk_agents_abi::ChatMessage::text("u1", Role::User, text, std::time::SystemTime::UNIX_EPOCH);
    Step::Emitted(SessionEvent::ConversationItemAdded {
        item: Box::new(ChatItem::Message(m)),
        agent: "agent".into(),
        source: lk_agents_voice::agent::Source::UserTyped,
        visibility: lk_agents_voice::agent::Visibility::Both,
    })
}

fn answered(answer: ToolAnswer, text: &str) -> Authored {
    Authored::new().tool_call("c1", 1).sent(Input::ToolAnswered {
        tool: Ref::new(0),
        answer,
        text: AgentText(text.into()),
    })
}

fn question_call(call_id: &str) -> Step {
    let call = FunctionCall {
        id: format!("item_{call_id}"),
        call_id: call_id.into(),
        name: "lookup".into(),
        arguments: "{}".into(),
        created_at: std::time::SystemTime::UNIX_EPOCH,
        extra: Default::default(),
        group_id: None,
    };
    Step::Emitted(SessionEvent::ConversationItemAdded {
        item: Box::new(call.into()),
        agent: "agent".into(),
        source: lk_agents_voice::agent::Source::ToolCall {
            speech: lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId(1),
            step: 1,
        },
        visibility: lk_agents_voice::agent::Visibility::Both,
    })
}

fn delivered_response(call_id: &str) -> Step {
    Step::Tool(ToolStep::InputResponseDelivered {
        request: lk_agents_abi::session::InputRequestRef::Agent {
            agent_id: "agent".into(),
            request_id: format!("request/{call_id}"),
        },
        call_id: Some(format!("item_{call_id}")),
        message_id: "u1".into(),
    })
}

/// `ToolResult("sent DTMF 1", reply_required=False)`: kept, asking for nothing.
fn a_value_asking_for_no_reply() -> Recording {
    answered(ToolAnswer::Recorded, "sent DTMF 1")
        .journal(output("c1", "sent DTMF 1", false, false))
        .done()
}

/// The response reaches its admitted occurrence; the original tool outcome remains a question.
fn a_question_answered() -> Recording {
    let question = format!("{WAITING_PREFIX}Which day?");
    answered(ToolAnswer::Ask, "Which day?")
        .journal(question_call("c1"))
        .journal(output("c1", &question, false, false))
        .journal(user("Monday"))
        .journal(delivered_response("c1"))
        .done()
}

fn an_unselected_question_waits() -> Recording {
    answered(ToolAnswer::Ask, "Which day?")
        .journal(output("c1", &format!("{WAITING_PREFIX}Which day?"), false, false))
        .journal(user("Monday"))
        .done()
}

fn a_reply_only_answers_the_selected_question() -> Recording {
    answered(ToolAnswer::Ask, "Which day?")
        .journal(question_call("c1"))
        .journal(output("c2", &format!("{WAITING_PREFIX}What time?"), false, false))
        .journal(output("c1", &format!("{WAITING_PREFIX}Which day?"), false, false))
        .journal(user("Monday"))
        .journal(delivered_response("c1"))
        .done()
}

/// `Completed("booked")` recorded empty and silent: the agent says nothing.
fn a_completion_recorded_silent() -> Recording {
    answered(ToolAnswer::Completed, "booked").journal(output("c1", "", false, false)).done()
}

/// `Failed("no slot")` recorded as an ordinary value.
fn a_failure_not_an_error() -> Recording {
    answered(ToolAnswer::Failed, "no slot").journal(output("c1", "no slot", false, true)).done()
}

/// A human answer overwrites the tool's original question instead of retaining both records.
fn a_question_overwritten_by_its_response() -> Recording {
    let question = format!("{WAITING_PREFIX}Which day?");
    answered(ToolAnswer::Ask, "Which day?")
        .journal(question_call("c1"))
        .journal(output("c1", &question, false, false))
        .journal(user("Monday"))
        .journal(output("c1", "Monday", false, false))
        .journal(delivered_response("c1"))
        .done()
}

/// A repeated delivery spends the same occurrence twice even though its human input is present.
fn a_response_delivered_twice() -> Recording {
    answered(ToolAnswer::Ask, "Which day?")
        .journal(question_call("c1"))
        .journal(output("c1", &format!("{WAITING_PREFIX}Which day?"), false, false))
        .journal(user("Monday"))
        .journal(delivered_response("c1"))
        .journal(delivered_response("c1"))
        .done()
}
