use std::collections::{BTreeMap, BTreeSet};

use lk_agents_abi::{
    ChatItem, FunctionCall, FunctionCallOutput, llm::tool_executor::WAITING_PREFIX,
};
use lk_agents_voice::{
    agent::Source,
    events::{SessionEvent, ToolLifecycle},
    tool_executor::APPROVAL_KEY,
    trace::{Step, ToolStep},
};

use lk_agents_core::approval::{APPROVAL_IDENTITY_KEY, APPROVAL_SOURCE_KEY};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    fakes::tools::APPROVAL_TOOL,
    record::Recording,
};

// Each model decision names the exact canonical read-backs its owning invocation admitted.
// A validated repeat spends one of those independently delivered occurrences and records its
// source. Native admission and committed approval marks are claims checked against that evidence.

pub struct ApprovalRunsOnTheAnswer;

impl Rule for ApprovalRunsOnTheAnswer {
    const CONTRACT: Contract = Contract {
        id: "approval_runs_on_the_answer",
        guarantee: "A tool requiring approval runs only when its decision scope explicitly admits an answered canonical read-back with the same validated identity, once per occurrence, and records exactly which read-back it spent.",
        examples: &[
            asked_then_granted,
            a_new_invocation_without_admitted_origins_asks_again,
            two_matched_confirmations_grant_different_validated_arguments,
            an_identical_sibling_can_be_answered_after_the_first_was_granted,
            a_restored_call_spends_its_grant,
            a_reused_provider_id_keeps_the_asking_item_distinct,
        ],
        counterexamples: &[
            asked_again_after_the_answer,
            granted_unasked,
            granted_to_an_earlier_step,
            granted_twice_for_one_answer,
            a_grant_names_an_unmatched_canonical_readback,
        ],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let journal = r.journal();
        let mut calls: BTreeMap<String, (usize, &FunctionCall)> = BTreeMap::new();
        let mut started: BTreeMap<String, &FunctionCall> = BTreeMap::new();
        let mut proposals = BTreeMap::new();
        let mut admitted = BTreeMap::new();
        let mut asked = BTreeMap::new();
        let mut outputs = BTreeMap::new();
        let mut responses: BTreeMap<String, Vec<(usize, &str)>> = BTreeMap::new();
        let mut humans = BTreeMap::new();
        let mut scopes = BTreeMap::new();
        for (n, (_, step)) in journal.iter().enumerate() {
            match step {
                Step::Tool(ToolStep::Calls { item_id, speech, step, input_answers, .. }) => {
                    proposals.entry(item_id.clone()).or_insert(Proposal {
                        at: n,
                        scope: (speech.0, *step),
                        answers: input_answers,
                    });
                }
                Step::Tool(ToolStep::CallAdmitted {
                    item_id, speech, identity, source, ..
                }) => {
                    admitted.entry(item_id.clone()).or_insert(Admission {
                        at: n,
                        speech: speech.0,
                        identity: identity.as_deref(),
                        source: source.as_deref(),
                    });
                }
                Step::Tool(ToolStep::InputResponseDelivered {
                    request: lk_agents_abi::session::InputRequestRef::Agent { .. },
                    call_id: Some(call_id),
                    message_id,
                }) => {
                    responses.entry(call_id.clone()).or_default().push((n, message_id));
                }
                Step::Emitted(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Started {
                    call,
                })) => {
                    started.insert(call.id.clone(), call);
                }
                Step::Emitted(SessionEvent::ConversationItemAdded { item, source, .. }) => {
                    match &**item {
                        ChatItem::FunctionCall(call) => {
                            if let Source::ToolCall { speech, step }
                            | Source::ToolOutput { speech, step } = source
                            {
                                scopes.insert(call.id.clone(), (speech.0, *step));
                            }
                            if mark(call) == Some("asked") {
                                asked.entry(call.id.clone()).or_insert(n);
                            }
                            if mark(call).is_some() {
                                calls
                                    .entry(call.id.clone())
                                    .and_modify(|(_, stored)| *stored = call)
                                    .or_insert((n, call));
                            }
                        }
                        ChatItem::FunctionCallOutput(output)
                            if output.output.starts_with(WAITING_PREFIX) =>
                        {
                            outputs.entry(output.id.clone()).or_insert(n);
                        }
                        ChatItem::Message(message)
                            if message.role == lk_agents_abi::Role::User
                                && message
                                    .raw_text_content()
                                    .is_some_and(|text| !text.trim().is_empty()) =>
                        {
                            humans.entry(message.id.clone()).or_insert(n);
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        // Actual body admission spends its grant even when interruption prevented a final
        // ConversationItemAdded. A committed Granted mark has the same once-only obligation.
        let ids: BTreeSet<_> = calls.keys().chain(admitted.keys()).cloned().collect();
        let mut decisions: Vec<_> = ids
            .into_iter()
            .filter_map(|id| {
                let committed = calls.get(&id);
                let admission = admitted.get(&id);
                let call =
                    committed.map(|(_, call)| *call).or_else(|| started.get(&id).copied())?;
                if call.name != APPROVAL_TOOL {
                    return None;
                }
                let at = committed
                    .map(|(n, _)| *n)
                    .into_iter()
                    .chain(admission.map(|admission| admission.at))
                    .min()?;
                Some((at, id, call))
            })
            .collect();
        decisions.sort_by_key(|(at, id, _)| (*at, id.clone()));
        let mut spent = BTreeSet::new();
        for (_, id, call) in decisions {
            let admission = admitted.get(&id);
            let proposal = proposals.get(&id);
            let recorded_source =
                call.extra.get(APPROVAL_SOURCE_KEY).and_then(serde_json::Value::as_str);
            let identity = match admission {
                Some(admission) => admission.identity.map(str::to_owned),
                None => validated_identity(call),
            };
            let source = match admission {
                Some(admission) => admission.source,
                None => recorded_source,
            };
            let consistent = admission.is_none_or(|admission| {
                calls.get(&id).is_none_or(|(_, recorded)| {
                    validated_identity(recorded).as_deref() == admission.identity
                        && recorded_source == admission.source
                })
            });
            let claim = mark(call).unwrap_or("granted");
            let scope_matches = proposal.is_some_and(|proposal| {
                scopes.get(&id).is_none_or(|scope| *scope == proposal.scope)
                    && admission.is_none_or(|admission| admission.speech == proposal.scope.0)
            });
            let eligible = |source_id: &str| {
                let Some(proposal) = proposal else { return false };
                let Some((_, readback)) = calls.get(source_id) else { return false };
                let Some(&asked_at) = asked.get(source_id) else { return false };
                let Some(&output_at) = outputs.get(&readback.output_id()) else { return false };
                scope_matches
                    && proposal.answers.contains(source_id)
                    && !spent.contains(source_id)
                    && readback.name == call.name
                    && mark(readback) == Some("asked")
                    && identity.is_some()
                    && validated_identity(readback) == identity
                    && asked_at <= output_at
                    && responses.get(source_id).is_some_and(|responses| {
                        responses.iter().any(|(delivery, message)| {
                            output_at < *delivery
                                && *delivery < proposal.at
                                && humans.get(*message).is_some_and(|human| *human < *delivery)
                        })
                    })
            };
            let owed =
                proposal.is_some_and(|proposal| proposal.answers.iter().any(|id| eligible(id)));
            let holds = consistent
                && identity.is_some()
                && match claim {
                    "granted" => source.is_some_and(eligible),
                    "asked" => !owed && admission.is_none() && source.is_none(),
                    _ => false,
                };
            check.expect_with(Subject::Session, holds, || format!(
                "{} canonical call {id} claimed {claim:?} with source {source:?}; scope {:?}, matched and unspent={owed}",
                call.name, proposal.map(|proposal| proposal.scope),
            ));
            if claim == "granted" && holds {
                spent.insert(source.expect("a valid grant names its source").to_owned());
            }
        }
    }
}

struct Proposal<'a> {
    at: usize,
    scope: (u64, u32),
    answers: &'a BTreeSet<String>,
}

struct Admission<'a> {
    at: usize,
    speech: u64,
    identity: Option<&'a str>,
    source: Option<&'a str>,
}

fn mark(call: &FunctionCall) -> Option<&str> {
    call.extra.get(APPROVAL_KEY).and_then(serde_json::Value::as_str)
}

fn validated_identity(call: &FunctionCall) -> Option<String> {
    call.extra.get(APPROVAL_IDENTITY_KEY).map(|identity| match identity {
        serde_json::Value::String(identity) => identity.clone(),
        value => canonical(value),
    })
}

fn canonical(value: &serde_json::Value) -> String {
    fn sorted(value: &serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(values) => {
                let ordered: BTreeMap<_, _> =
                    values.iter().map(|(key, value)| (key, sorted(value))).collect();
                serde_json::to_value(ordered).expect("JSON values serialize")
            }
            serde_json::Value::Array(values) => {
                serde_json::Value::Array(values.iter().map(sorted).collect())
            }
            value => value.clone(),
        }
    }
    sorted(value).to_string()
}

fn proposed(call_id: &str, step: u32, answers: &[&str]) -> Step {
    proposed_in(1, call_id, step, answers)
}

fn proposed_in(speech: u64, call_id: &str, step: u32, answers: &[&str]) -> Step {
    Step::Tool(ToolStep::Calls {
        speech: lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId(speech),
        step,
        item_id: format!("item_{call_id}"),
        names: vec![APPROVAL_TOOL.into()],
        input_answers: answers.iter().map(|id| format!("item_{id}")).collect(),
    })
}

fn started(call_id: &str) -> Step {
    Step::Emitted(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Started {
        call: function_call(call_id, None),
    }))
}

fn function_call(call_id: &str, mark: Option<&str>) -> FunctionCall {
    let mut call = FunctionCall {
        id: format!("item_{call_id}"),
        call_id: call_id.into(),
        arguments: r#"{"label":"human"}"#.into(),
        name: APPROVAL_TOOL.into(),
        created_at: std::time::UNIX_EPOCH,
        extra: Default::default(),
        group_id: None,
    };
    if let Some(mark) = mark {
        call.extra.insert(APPROVAL_KEY.into(), mark.into());
        call.extra.insert(APPROVAL_IDENTITY_KEY.into(), r#"{"label":"human"}"#.into());
        if mark == "granted" {
            call.extra.insert(APPROVAL_SOURCE_KEY.into(), "item_c1".into());
        }
    }
    call
}

fn call(call_id: &str, mark: &str, step: u32) -> Step {
    emitted(function_call(call_id, Some(mark)).into(), step)
}

fn with_identity(call_id: &str, mark: &str, step: u32, label: &str, source: Option<&str>) -> Step {
    let mut call = function_call(call_id, Some(mark));
    // The application validated this identity; its raw model spelling is not authorization.
    call.extra.insert(
        APPROVAL_IDENTITY_KEY.into(),
        canonical(&serde_json::json!({"label":label})).into(),
    );
    if let Some(source) = source {
        call.extra.insert(APPROVAL_SOURCE_KEY.into(), format!("item_{source}").into());
    }
    emitted(call.into(), step)
}

fn output(call_id: &str, text: &str) -> Step {
    emitted(
        FunctionCallOutput::new(
            format!("item_{call_id}_out"),
            call_id,
            text,
            false,
            std::time::UNIX_EPOCH,
        )
        .into(),
        1,
    )
}

fn emitted(item: ChatItem, step: u32) -> Step {
    emitted_in(1, item, step)
}

fn emitted_in(speech: u64, item: ChatItem, step: u32) -> Step {
    use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId;
    Step::Emitted(SessionEvent::ConversationItemAdded {
        item: Box::new(item),
        agent: "agent".into(),
        source: Source::ToolOutput { speech: SpeechId(speech), step },
        visibility: lk_agents_voice::agent::Visibility::Both,
    })
}

fn asked() -> Authored {
    Authored::new()
        .journal(proposed("c1", 1, &[]))
        .journal(started("c1"))
        .journal(call("c1", "asked", 1))
        .journal(output("c1", &format!("{WAITING_PREFIX}Please confirm. Yes or no?")))
}

fn asked_then_granted() -> Recording {
    asked()
        .journal(said("yes"))
        .journal(delivered("c1", "yes"))
        .journal(proposed("c2", 2, &["c1"]))
        .journal(started("c2"))
        .journal(call("c2", "granted", 2))
        .done()
}

fn asked_again_after_the_answer() -> Recording {
    asked()
        .journal(said("yes"))
        .journal(delivered("c1", "yes"))
        .journal(proposed("c2", 2, &["c1"]))
        .journal(started("c2"))
        .journal(call("c2", "asked", 2))
        .done()
}

/// The admission spends its exact source even if interruption prevented a committed grant.
fn a_restored_call_spends_its_grant() -> Recording {
    asked()
        .journal(said("yes"))
        .journal(delivered("c1", "yes"))
        .journal(proposed("c2", 2, &["c1"]))
        .journal(started("c2"))
        .journal(Step::Tool(ToolStep::CallAdmitted {
            speech: lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId(1),
            call_id: "c2".into(),
            item_id: "item_c2".into(),
            identity: Some(r#"{"label":"human"}"#.into()),
            source: Some("item_c1".into()),
        }))
        .journal(proposed("c3", 3, &["c1"]))
        .journal(started("c3"))
        .journal(call("c3", "asked", 3))
        .done()
}

fn a_reused_provider_id_keeps_the_asking_item_distinct() -> Recording {
    let reused = |id: &str, mark| {
        let mut call = function_call(id, mark);
        call.call_id = "reused".into();
        call
    };
    let start = |id| {
        Step::Emitted(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Started {
            call: reused(id, None),
        }))
    };
    let question = FunctionCallOutput::new(
        "item_c1_out",
        "reused",
        format!("{WAITING_PREFIX}Please confirm. Yes or no?"),
        false,
        std::time::UNIX_EPOCH,
    );
    Authored::new()
        .journal(proposed("c1", 1, &[]))
        .journal(start("c1"))
        .journal(emitted(reused("c1", Some("asked")).into(), 1))
        .journal(emitted(question.into(), 1))
        .journal(said("yes"))
        .journal(delivered("c1", "yes"))
        .journal(proposed("c2", 2, &["c1"]))
        .journal(start("c2"))
        .journal(emitted(reused("c2", Some("granted")).into(), 2))
        .done()
}

fn two_matched_confirmations_grant_different_validated_arguments() -> Recording {
    Authored::new()
        .journal(proposed("c1", 1, &[]))
        .journal(started("c1"))
        .journal(with_identity("c1", "asked", 1, "Monday", None))
        .journal(output("c1", &format!("{WAITING_PREFIX}Cancel Monday?")))
        .journal(proposed("c2", 2, &[]))
        .journal(started("c2"))
        .journal(with_identity("c2", "asked", 2, "Tuesday", None))
        .journal(output("c2", &format!("{WAITING_PREFIX}Cancel Tuesday?")))
        .journal(said("yes to both"))
        .journal(delivered("c1", "yes to both"))
        .journal(delivered("c2", "yes to both"))
        .journal(proposed("c3", 3, &["c1", "c2"]))
        .journal(started("c3"))
        .journal(with_identity("c3", "granted", 3, "Monday", Some("c1")))
        .journal(proposed("c4", 3, &["c1", "c2"]))
        .journal(started("c4"))
        .journal(with_identity("c4", "granted", 3, "Tuesday", Some("c2")))
        .done()
}

fn an_identical_sibling_can_be_answered_after_the_first_was_granted() -> Recording {
    asked()
        .journal(proposed("c2", 2, &[]))
        .journal(started("c2"))
        .journal(call("c2", "asked", 2))
        .journal(output("c2", &format!("{WAITING_PREFIX}Please also confirm this one.")))
        .journal(said("first yes"))
        .journal(delivered("c1", "first yes"))
        .journal(proposed("c3", 3, &["c1"]))
        .journal(started("c3"))
        .journal(call("c3", "granted", 3))
        .journal(said("second yes"))
        .journal(delivered("c2", "second yes"))
        .journal(proposed("c4", 4, &["c2"]))
        .journal(started("c4"))
        .journal(with_identity("c4", "granted", 4, "human", Some("c2")))
        .done()
}

fn granted_twice_for_one_answer() -> Recording {
    asked()
        .journal(said("yes"))
        .journal(delivered("c1", "yes"))
        .journal(proposed("c2", 2, &["c1"]))
        .journal(started("c2"))
        .journal(call("c2", "granted", 2))
        .journal(proposed("c3", 3, &["c1"]))
        .journal(started("c3"))
        .journal(call("c3", "granted", 3))
        .done()
}

fn a_grant_names_an_unmatched_canonical_readback() -> Recording {
    asked()
        .journal(proposed("c2", 2, &[]))
        .journal(started("c2"))
        .journal(call("c2", "asked", 2))
        .journal(output("c2", &format!("{WAITING_PREFIX}Confirm the second request?")))
        .journal(said("yes"))
        .journal(delivered("c1", "yes"))
        .journal(proposed("c3", 3, &["c1"]))
        .journal(started("c3"))
        .journal(with_identity("c3", "granted", 3, "human", Some("c2")))
        .done()
}

/// A new invocation receives no authority merely because another invocation delivered an answer.
fn a_new_invocation_without_admitted_origins_asks_again() -> Recording {
    asked()
        .journal(said("yes"))
        .journal(delivered("c1", "yes"))
        .journal(said("actually, wait"))
        .journal(proposed_in(2, "c2", 1, &[]))
        .journal(started("c2"))
        .journal(emitted_in(2, function_call("c2", Some("asked")).into(), 1))
        .done()
}

fn said(text: &str) -> Step {
    Step::Emitted(SessionEvent::ConversationItemAdded {
        item: Box::new(
            lk_agents_abi::ChatMessage::text(
                format!("msg_{text}"),
                lk_agents_abi::Role::User,
                text,
                std::time::UNIX_EPOCH,
            )
            .into(),
        ),
        agent: "agent".into(),
        source: Source::UserTyped,
        visibility: lk_agents_voice::agent::Visibility::Both,
    })
}

fn delivered(call_id: &str, response: &str) -> Step {
    Step::Tool(ToolStep::InputResponseDelivered {
        request: lk_agents_abi::session::InputRequestRef::Agent {
            agent_id: "agent".into(),
            request_id: format!("request/{call_id}"),
        },
        call_id: Some(format!("item_{call_id}")),
        message_id: format!("msg_{response}"),
    })
}

/// A response arriving after a step began calling does not enter that step's explicit authority.
fn granted_to_an_earlier_step() -> Recording {
    asked()
        .journal(proposed("c2", 2, &[]))
        .journal(started("c2"))
        .journal(said("yes"))
        .journal(delivered("c1", "yes"))
        .journal(proposed("c3", 2, &[]))
        .journal(started("c3"))
        .journal(call("c2", "asked", 2))
        .journal(call("c3", "granted", 2))
        .done()
}

fn granted_unasked() -> Recording {
    asked()
        .journal(proposed("c2", 2, &[]))
        .journal(started("c2"))
        .journal(call("c2", "granted", 2))
        .done()
}
