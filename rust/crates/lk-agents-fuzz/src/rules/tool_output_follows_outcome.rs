//! A tool call is recorded with the output its outcome means, as the reference's executor and
//! `_apply_transition` record it: a value's text (also when it asks for no reply), a transition's
//! text (`Completed`'s value, `Failed`'s error, an `InputRequired`'s question), an error flagged
//! as one, and a reply asked for exactly when the outcome asks for one. An `InputRequired` left
//! open is answered by the person's next message: that call's output becomes what they said.
//!
//! A step cut short records its outputs asking for no reply, and the close settles calls still
//! running as an error, so neither is held to the reply flag; a body cancelled has no outcome. A
//! `VoiceTask`'s speech runs its tools on the task's runner, whose `Completed` ends the task: its
//! agent asks for no reply to it.
use lk_agents_abi::{ChatItem, Role, llm::tool_executor::WAITING_PREFIX};
use lk_agents_voice::{
    events::SessionEvent,
    trace::{HandoffStep, Step},
};

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
        guarantee: "Each answered tool call is recorded with its outcome's output, error flag and reply request, and a call left waiting on the person is answered by their next message.",
        examples: &[a_value_asking_for_no_reply, a_question_answered],
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
        for applied in inputs.accepted() {
            let Input::ToolAnswered { tool, answer, text: AgentText(text) } = &applied.input else {
                continue;
            };
            if matches!(answer, ToolAnswer::Cancelled | ToolAnswer::Detached) {
                continue;
            }
            let Some(Detail::Tool { call_id, speech, .. }) =
                r.resource(Kind::Tool, tool.nth).map(|t| t.detail().clone())
            else {
                continue;
            };
            let Some((at, ChatItem::FunctionCallOutput(output), _)) = announced.iter().find(
                |(_, i, _)| matches!(i, ChatItem::FunctionCallOutput(o) if o.call_id == call_id),
            ) else {
                continue;
            };
            // The close settles a call it cut as an error of its own, a cut speech as cancelled.
            if closing.is_some_and(|c| c <= *at) || host.speech(speech).is_interrupted() {
                continue;
            }
            let outcome = crate::fakes::tools::outcome_of(*answer, text);
            let mut shape = lk_agents_core::tool_outcome::shape(&outcome, lk_agents_core::tool_outcome::Mode::Voice);
            if *answer == ToolAnswer::Completed && in_task(speech) {
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
        // An open question is the latest output still waiting; the person's next message is it.
        let mut waiting: Option<String> = None;
        for (i, (at, item, speech)) in announced.iter().enumerate() {
            match item {
                // A step cut before its tools were awaited asks nothing.
                ChatItem::FunctionCallOutput(o) if o.output.starts_with(WAITING_PREFIX) => {
                    let cut = speech.is_some_and(|s| host.speech(s.0).is_interrupted());
                    waiting = (!cut).then(|| o.id.clone());
                }
                ChatItem::Message(m) if m.role == Role::User => {
                    let Some(id) = waiting.take() else { continue };
                    if closing.is_some_and(|c| c <= *at) {
                        continue;
                    }
                    let said = m.text_content().unwrap_or_default();
                    let answered = announced[..i].iter().rev().any(|(_, prior, _)| {
                        matches!(prior, ChatItem::FunctionCallOutput(o)
                            if o.id == id && !o.output.starts_with(WAITING_PREFIX)
                                && (o.output.contains(said.as_str())
                                    || o.output.starts_with("not run yet: the person answered")))
                    });
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
    use lk_agents_voice::agent_activity::scheduling::SpeechId;
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
        run: None,
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
        run: None,
    })
}

fn answered(answer: ToolAnswer, text: &str) -> Authored {
    Authored::new().tool_call("c1", 1).sent(Input::ToolAnswered {
        tool: Ref::new(0),
        answer,
        text: AgentText(text.into()),
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
        .journal(output("c1", &question, false, false))
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
        .journal(output("c1", &question, false, false))
        .journal(user("Monday"))
        .done()
}
