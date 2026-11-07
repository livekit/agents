//! A tool call is recorded with the output its outcome means, as the reference's executor and
//! `_apply_transition` record it: a value's text (also when it asks for no reply), a transition's
//! text (`Completed`'s value, `Failed`'s error, an `InputRequired`'s question), an error flagged
//! as one, and a reply asked for exactly when the outcome asks for one. An `InputRequired` left
//! open is answered by the person's next message only when selected: that call's output becomes
//! what they said, while sibling questions remain pending.
//!
//! A step cut short records its outputs asking for no reply, and the close settles calls still
//! running as an error, so neither is held to the reply flag; a body cancelled has no outcome. A
//! step whose speech ended before its outputs were recorded asks for no reply to them. A
//! `VoiceTask`'s speech runs its tools on the task's runner, whose `Completed` ends the task: its
//! agent asks for no reply to it. A body its first `ctx.update()` detached answers with its final,
//! recorded under `<call_id>_final`; a host update ending a body first answers its call.
use lk_agents_abi::{ChatItem, FunctionCall, Role, llm::tool_executor::WAITING_PREFIX};
use lk_agents_core::tts::markup::ChatMessageExt;
use lk_agents_voice::{
    events::SessionEvent,
    trace::{HandoffStep, Step},
};
use std::collections::BTreeSet;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{AgentText, Input, Ref, ToolAnswer},
    record::{Detail, Kind, Recording},
};

pub struct ToolOutputFollowsOutcome;

impl Rule for ToolOutputFollowsOutcome {
    const CONTRACT: Contract = Contract {
        id: "tool_output_follows_outcome",
        guarantee: "Each answered tool call is recorded with its outcome's output, error flag and reply request, and the selected call waiting on the person is answered by their next message.",
        examples: &[
            a_value_asking_for_no_reply,
            a_question_answered,
            an_unselected_question_waits,
            a_reply_only_answers_the_selected_question,
        ],
        counterexamples: &[
            a_completion_recorded_silent,
            a_failure_not_an_error,
            a_question_never_answered,
        ],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let inputs = r.inputs_view();
        let host = r.host_view();
        let internal = r.internal_view();
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
            let Some(Detail::Tool { call_id, speech, .. }) =
                r.resource(Kind::Tool, tool.nth).map(|t| t.detail().clone())
            else {
                continue;
            };
            let recorded = if detached.contains(&tool.nth) {
                format!("{call_id}_final")
            } else {
                call_id.clone()
            };
            let Some((at, ChatItem::FunctionCallOutput(output), _)) = announced.iter().find(
                |(_, i, _)| matches!(i, ChatItem::FunctionCallOutput(o) if o.call_id == recorded),
            ) else {
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
            if *answer == ToolAnswer::Completed && in_task(speech)
                || over && !detached.contains(&tool.nth)
            {
                shape.reply_required = false;
            }
            let want = lk_agents_core::tool_outcome::delivered(&outcome).unwrap_or(shape.text);
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
        }
        let mut selected = BTreeSet::new();
        let mut waiting: Vec<(String, String)> = Vec::new();
        for (i, (at, item, speech)) in announced.iter().enumerate() {
            match item {
                ChatItem::FunctionCall(call) => {
                    let flag =
                        call.extra.get(lk_agents_core::agent_run::planning::INPUT_REQUESTED_KEY);
                    if flag.is_some_and(|flag| {
                        flag.as_bool() == Some(true) || flag.as_str() == Some("true")
                    }) {
                        selected.insert(call.call_id.clone());
                    } else {
                        selected.remove(&call.call_id);
                    }
                }
                // A step cut before its tools were awaited asks nothing; one cut after it put
                // its question leaves the question asked.
                ChatItem::FunctionCallOutput(o) if o.output.starts_with(WAITING_PREFIX) => {
                    let cut = speech.is_some_and(|s| {
                        internal.speech(s.0).interrupted_at().is_some_and(|cut| cut <= *at)
                    });
                    if !cut {
                        match waiting.iter_mut().find(|(call, _)| *call == o.call_id) {
                            Some((_, id)) => *id = o.id.clone(),
                            None => waiting.push((o.call_id.clone(), o.id.clone())),
                        }
                    }
                }
                ChatItem::Message(m) if m.role == Role::User => {
                    if closing.is_some_and(|c| c <= *at) {
                        continue;
                    }
                    let Some(index) = waiting.iter().position(|(call, _)| selected.contains(call))
                    else {
                        continue;
                    };
                    let (_, id) = waiting.remove(index);
                    let said = m.text_content().unwrap_or_default();
                    let answered = announced[..i]
                        .iter()
                        .rev()
                        .find_map(|(_, prior, _)| match prior {
                            ChatItem::FunctionCallOutput(o) if o.id == id => Some(
                                !o.output.starts_with(WAITING_PREFIX)
                                    && o.output.contains(said.as_str()),
                            ),
                            _ => None,
                        })
                        .unwrap_or(false);
                    check.expect_with(Subject::Session, answered, || {
                        format!("the question {id} waits on was never answered by {said:?}")
                    });
                }
                _ => {}
            }
        }
    }
}

fn output(call_id: &str, text: &str, is_error: bool, reply_required: bool) -> Step {
    use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId;
    let mut out = lk_agents_abi::FunctionCallOutput::new(
        format!("{call_id}_out"),
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

fn selected_question(call_id: &str) -> Step {
    let mut call = FunctionCall {
        id: call_id.into(),
        call_id: call_id.into(),
        name: "lookup".into(),
        arguments: "{}".into(),
        created_at: std::time::SystemTime::UNIX_EPOCH,
        extra: Default::default(),
        group_id: None,
    };
    call.extra.insert(lk_agents_core::agent_run::planning::INPUT_REQUESTED_KEY.into(), true.into());
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

/// `ToolResult("sent DTMF 1", reply_required=False)`: kept, asking for nothing.
fn a_value_asking_for_no_reply() -> Recording {
    answered(ToolAnswer::Recorded, "sent DTMF 1")
        .journal(output("c1", "sent DTMF 1", false, false))
        .done()
}

/// The question is asked, and the person's answer becomes the call's output.
fn a_question_answered() -> Recording {
    let question = format!("{WAITING_PREFIX}Which day?");
    answered(ToolAnswer::Ask, "Which day?")
        .journal(selected_question("c1"))
        .journal(output("c1", &question, false, false))
        .journal(output("c1", "Monday", false, false))
        .journal(user("Monday"))
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
        .journal(selected_question("c1"))
        .journal(output("c2", &format!("{WAITING_PREFIX}What time?"), false, false))
        .journal(output("c1", &format!("{WAITING_PREFIX}Which day?"), false, false))
        .journal(output("c1", "Monday", false, false))
        .journal(user("Monday"))
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

/// The person answered, and the call still waits.
fn a_question_never_answered() -> Recording {
    let question = format!("{WAITING_PREFIX}Which day?");
    answered(ToolAnswer::Ask, "Which day?")
        .journal(selected_question("c1"))
        .journal(output("c1", &question, false, false))
        .journal(user("Monday"))
        .done()
}
