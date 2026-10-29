//! A tool declared with `needs_approval=True` runs exactly when the model repeats, with the same
//! arguments, the call whose read-back the person has answered, in the reply to that answer, and
//! only once per answer. Any other call of it asks its read-back: an answer the person moved on
//! from (spoke again) grants nothing, and neither does an answer that came after the repeat's step
//! began calling tools (the step's model request had not read it). The calls carry the marks (`lk.approval`: `asked`,
//! `granted`) the next call reads.
//!
//! Read off what the host was told: each call as announced, each output as last announced before
//! the next call. The runs' approval tool is `fakes::tools::APPROVAL_TOOL`.
use std::collections::BTreeMap;

use lk_agents_abi::{
    ChatItem, FunctionCall, FunctionCallOutput, llm::tool_executor::WAITING_PREFIX,
};
use lk_agents_voice::{
    agent::Source,
    events::{SessionEvent, ToolLifecycle},
    tool_executor::APPROVAL_KEY,
    trace::{Step, ToolStep},
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    fakes::tools::APPROVAL_TOOL,
    record::Recording,
};

pub struct ApprovalRunsOnTheAnswer;

impl Rule for ApprovalRunsOnTheAnswer {
    const CONTRACT: Contract = Contract {
        id: "approval_runs_on_the_answer",
        guarantee: "A tool that requires approval runs on the repeat of a call whose read-back the person answered, before they speak again, once per answer, and asks its read-back otherwise.",
        examples: &[asked_then_granted, answered_then_moved_on_asks_again],
        counterexamples: &[
            asked_again_after_the_answer,
            granted_unasked,
            granted_to_an_earlier_step,
        ],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let journal = r.journal();
        let key = |args: &str| {
            serde_json::from_str::<serde_json::Value>(if args.is_empty() { "{}" } else { args })
                .map(|v| v.to_string())
                .unwrap_or_else(|_| args.to_owned())
        };
        let mark = |c: &FunctionCall| {
            c.extra.get(APPROVAL_KEY).and_then(|v| v.as_str()).map(str::to_owned)
        };
        // Each approval call as committed, where its body started, and every output announcement.
        let mut calls: Vec<(usize, FunctionCall)> = Vec::new();
        let mut started: BTreeMap<String, usize> = BTreeMap::new();
        // Each call's step, and where each step started its first call.
        let mut step_of: BTreeMap<String, (u64, u32)> = BTreeMap::new();
        let mut step_open: BTreeMap<(u64, u32), usize> = BTreeMap::new();
        let mut outputs: Vec<(usize, &FunctionCallOutput)> = Vec::new();
        // The person's messages: the one after an answer is the answering one, and the next ends
        // what that answer can grant.
        let mut said: Vec<usize> = Vec::new();
        for (n, (_, step)) in journal.iter().enumerate() {
            match step {
                Step::Emitted(SessionEvent::ConversationItemAdded { item, source, .. }) => {
                    if let ChatItem::FunctionCall(c) = &**item
                        && let Source::ToolCall { speech, step }
                        | Source::ToolOutput { speech, step } = source
                    {
                        step_of.insert(c.call_id.clone(), (speech.0, *step));
                    }
                    match &**item {
                        ChatItem::FunctionCall(c) if c.name == APPROVAL_TOOL => {
                            calls.push((n, c.clone()))
                        }
                        ChatItem::FunctionCallOutput(o) => outputs.push((n, o)),
                        ChatItem::Message(m) if m.role == lk_agents_abi::Role::User => said.push(n),
                        _ => {}
                    }
                }
                Step::Emitted(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Started {
                    call,
                })) => {
                    started.entry(call.call_id.clone()).or_insert(n);
                }
                Step::Tool(ToolStep::CallStarted { call_id, .. }) => {
                    started.entry(call_id.clone()).or_insert(n);
                }
                _ => {}
            }
        }
        for (call_id, step) in &step_of {
            if let Some(&n) = started.get(call_id) {
                step_open.entry(*step).and_modify(|o| *o = (*o).min(n)).or_insert(n);
            }
        }
        // When a read-back was put to the person, and when they answered it.
        let asked_at = |call_id: &str| {
            outputs
                .iter()
                .find(|(_, o)| o.call_id == call_id && o.output.starts_with(WAITING_PREFIX))
                .map(|(n, _)| *n)
        };
        let answered_at = |call_id: &str| {
            outputs
                .iter()
                .find(|(_, o)| o.call_id == call_id && !o.output.starts_with(WAITING_PREFIX))
                .map(|(n, _)| *n)
        };
        for (_, call) in &calls {
            let (Some(m), Some(&start)) = (mark(call), started.get(&call.call_id)) else {
                continue;
            };
            let same = |c: &FunctionCall| {
                c.call_id != call.call_id && key(&c.arguments) == key(&call.arguments)
            };
            // The tool's latest read-back before this call, whatever its arguments: an answer is
            // to the question asked last. A read-back of the same step counts from its call's
            // start, as the tool machine decides a step's calls in the model's order; one of
            // another step counts once its output is committed, when the question is put.
            let step = step_of.get(&call.call_id);
            let latest = calls
                .iter()
                .filter(|(_, c)| c.call_id != call.call_id && mark(c).as_deref() == Some("asked"))
                .filter_map(|(_, c)| {
                    let asked = asked_at(&c.call_id)?;
                    let at = if step.is_some() && step_of.get(&c.call_id) == step {
                        *started.get(&c.call_id)?
                    } else {
                        asked
                    };
                    (at < start).then_some((at, c))
                })
                .max_by_key(|(n, _)| *n);
            let Some((_, readback)) = latest else {
                check.expect_with(Subject::Session, m != "granted", || {
                    format!("{} was granted with no read-back put before it", call.call_id)
                });
                continue;
            };
            let moved_on =
                |a: usize| said.iter().filter(|n| **n > a).nth(1).is_some_and(|n| *n < start);
            let opened = step_of.get(&call.call_id).and_then(|s| step_open.get(s)).copied();
            let answer = answered_at(&readback.call_id).filter(|n| {
                *n < opened.unwrap_or(start).min(start)
                    && !moved_on(*n)
                    && key(&readback.arguments) == key(&call.arguments)
            });
            // Another repeat whose body started after the answer, before this one, may have spent
            // it; a cut repeat's grant is not on the record, so it counts too.
            let spent = answer.is_some_and(|a| {
                calls.iter().any(|(_, c)| {
                    same(c)
                        && c.call_id != readback.call_id
                        && started.get(&c.call_id).is_some_and(|s| a < *s && *s < start)
                        && mark(c).as_deref() != Some("asked")
                })
            });
            let owed = answer.is_some() && !spent;
            let holds = match m.as_str() {
                "granted" => answer.is_some(),
                _ => !owed,
            };
            check.expect_with(Subject::Session, holds, || {
                format!(
                    "{} {} was marked {m:?}; the read-back {} {}",
                    call.name,
                    call.call_id,
                    readback.call_id,
                    if owed { "was answered and unspent" } else { "was not answered" }
                )
            });
        }
    }
}

fn started(call_id: &str) -> Step {
    let c = function_call(call_id, None);
    Step::Emitted(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Started { call: c }))
}

fn function_call(call_id: &str, mark: Option<&str>) -> lk_agents_abi::FunctionCall {
    let mut c = lk_agents_abi::FunctionCall {
        id: format!("item_{call_id}"),
        call_id: call_id.into(),
        arguments: r#"{"label":"human"}"#.into(),
        name: APPROVAL_TOOL.into(),
        created_at: std::time::SystemTime::UNIX_EPOCH,
        extra: Default::default(),
        group_id: None,
    };
    if let Some(mark) = mark {
        c.extra.insert(APPROVAL_KEY.into(), serde_json::Value::String(mark.into()));
    }
    c
}

/// A call committed by step `step` of speech 1.
fn call(call_id: &str, mark: &str, step: u32) -> Step {
    emitted(ChatItem::FunctionCall(function_call(call_id, Some(mark))), step)
}

fn output(call_id: &str, text: &str) -> Step {
    let o = lk_agents_abi::FunctionCallOutput::new(
        format!("{call_id}_out"),
        call_id,
        text,
        false,
        std::time::SystemTime::UNIX_EPOCH,
    );
    emitted(ChatItem::FunctionCallOutput(o), 1)
}

fn emitted(item: ChatItem, step: u32) -> Step {
    use lk_agents_voice::agent_activity::scheduling::SpeechId;
    Step::Emitted(SessionEvent::ConversationItemAdded {
        item: Box::new(item),
        agent: "agent".into(),
        source: lk_agents_voice::agent::Source::ToolOutput { speech: SpeechId(1), step },
        visibility: lk_agents_voice::agent::Visibility::Both,
        run: None,
    })
}

fn asked() -> Authored {
    Authored::new()
        .journal(started("c1"))
        .journal(call("c1", "asked", 1))
        .journal(output("c1", &format!("{WAITING_PREFIX}Please confirm. Yes or no?")))
}

/// Read back, answered, repeated: it runs.
fn asked_then_granted() -> Recording {
    asked()
        .journal(output("c1", "yes"))
        .journal(started("c2"))
        .journal(call("c2", "granted", 2))
        .done()
}

/// The person said yes and the repeat asks again: the booking never happens.
fn asked_again_after_the_answer() -> Recording {
    asked()
        .journal(output("c1", "yes"))
        .journal(started("c2"))
        .journal(call("c2", "asked", 2))
        .done()
}

/// Run before the person answered.
/// The person answered, then said something else: the repeat asks again.
fn answered_then_moved_on_asks_again() -> Recording {
    asked()
        .journal(output("c1", "yes"))
        .journal(said("yes"))
        .journal(said("actually, wait"))
        .journal(started("c2"))
        .journal(call("c2", "asked", 2))
        .done()
}

fn said(text: &str) -> Step {
    Step::Emitted(SessionEvent::ConversationItemAdded {
        item: Box::new(ChatItem::Message(lk_agents_abi::ChatMessage::new(
            format!("msg_{text}"),
            lk_agents_abi::Role::User,
            Vec::new(),
            std::time::SystemTime::UNIX_EPOCH,
        ))),
        agent: "agent".into(),
        source: lk_agents_voice::agent::Source::UserTyped,
        visibility: lk_agents_voice::agent::Visibility::Both,
        run: None,
    })
}

/// The repeat's step began calling tools before the person answered: its request never read it.
fn granted_to_an_earlier_step() -> Recording {
    asked()
        .journal(started("c2"))
        .journal(output("c1", "yes"))
        .journal(started("c3"))
        .journal(call("c2", "asked", 2))
        .journal(call("c3", "granted", 2))
        .done()
}

fn granted_unasked() -> Recording {
    asked().journal(started("c2")).journal(call("c2", "granted", 2)).done()
}
