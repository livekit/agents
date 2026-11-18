use super::*;
use crate::agent_run::machine::tests::{input, string_tool, text_agent, user};
use lk_agents_abi::FunctionCallOutput;
use lk_agents_abi::llm::tool_executor::{REMOTE_REPORT_KEY, RemoteReport, ToolOutcome};
use lk_agents_abi::resources::ResourceId;
use serde_json::json;

fn original() -> FunctionCall {
    FunctionCall {
        id: "canonical-original".into(),
        call_id: "repeated-provider-id".into(),
        name: "booking".into(),
        arguments: r#"{"request":"Book a seat"}"#.into(),
        created_at: std::time::UNIX_EPOCH,
        group_id: None,
        extra: [(REMOTE_REPORT_KEY.into(), serde_json::to_value(RemoteReport::Pending).unwrap())]
            .into_iter()
            .collect(),
    }
}

#[test]
fn remote_reconciliation_uses_the_original_call_and_upserts_its_result_without_a_model() {
    let original = original();
    let mut remote = string_tool("booking", "request");
    remote.remote = Some("booking-service".into());
    // These policies must not reinterpret the already-admitted original operation as a new
    // duplicate or a fresh approval request.
    remote.needs_approval = Some(lk_agents_abi::agent::ReadBack::Default);
    remote.on_duplicate = lk_agents_abi::llm::provider_format::tools::DuplicateMode::Reject;
    let history = vec![
        ChatItem::FunctionCall(original.clone()),
        FunctionCallOutput::new(
            original.output_id(),
            &original.call_id,
            "Checking",
            false,
            std::time::UNIX_EPOCH,
        )
        .into(),
    ];
    let mut run = AgentRun::new(
        text_agent(vec![remote]),
        input(Opening::Message(Box::new(user("unused", "unused"))), history),
    )
    .unwrap();
    let mut effects = run.resume_remote(vec![original.clone()], Ts::ZERO);
    let call = effects
        .iter()
        .find_map(|effect| match effect {
            AgentRunEffect::ToolPrepare { call, function_call, .. } => {
                assert_eq!(function_call, &original);
                Some(*call)
            }
            _ => None,
        })
        .unwrap();
    effects.extend(run.handle(
        AgentRunEvent::ToolPrepared {
            call,
            prepared: ResourceId(10),
            identity: lk_agents_abi::agent::ArgumentIdentity::Json(
                json!({"request":"Book a seat"}),
            ),
        },
        Ts::ZERO,
    ));
    assert!(effects.iter().any(|effect| matches!(effect, AgentRunEffect::ToolExecute { function_call, .. } if function_call == &original)));
    effects.extend(
        run.handle(
            AgentRunEvent::ToolSettled {
                call,
                report: ToolOutcome::Update {
                    message: "Still checking".into(),
                    raw: "Still checking".into(),
                    reply_required: true,
                }
                .into(),
            },
            Ts::ZERO,
        ),
    );
    assert!(!effects.iter().any(|effect| matches!(effect, AgentRunEffect::Finished { .. })));
    effects.extend(
        run.handle(
            AgentRunEvent::ToolSettled {
                call,
                report: ToolOutcome::Value {
                    text: "Booked".into(),
                    reply_required: true,
                    verbatim: false,
                    more: None,
                }
                .into(),
            },
            Ts::ZERO,
        ),
    );
    assert!(!effects.iter().any(|effect| matches!(
        effect,
        AgentRunEffect::LlmRequestStart { .. } | AgentRunEffect::ReadBackRender { .. }
    )));
    let result = effects
        .iter()
        .find_map(|effect| match effect {
            AgentRunEffect::Finished { result, .. } => Some(result.as_ref().as_ref().unwrap()),
            _ => None,
        })
        .unwrap();
    assert_eq!(result.all_messages.iter().filter(|item| item.id() == original.id).count(), 1);
    let outputs: Vec<_> = result
        .all_messages
        .iter()
        .filter_map(|item| match item {
            ChatItem::FunctionCallOutput(output) if output.id == original.output_id() => {
                Some(output)
            }
            _ => None,
        })
        .collect();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].output, "Booked");
    assert_eq!(outputs[0].call_id, original.call_id);
    assert_eq!(
        result.new_messages.iter().filter(|item| item.id() == original.output_id()).count(),
        1
    );
    assert!(!result.all_messages.iter().any(|item| item.id() == "unused"));
}

#[test]
fn remote_reconciliation_refuses_a_local_coroutine_without_preparing_it() {
    let mut run = AgentRun::new(
        text_agent(vec![string_tool("booking", "request")]),
        input(Opening::Message(Box::new(user("unused", "unused"))), vec![]),
    )
    .unwrap();
    let effects = run.resume_remote(vec![original()], Ts::ZERO);
    assert!(effects.iter().any(
        |effect| matches!(effect, AgentRunEffect::Finished { result, .. } if result.is_err())
    ));
    assert!(!effects.iter().any(|effect| matches!(
        effect,
        AgentRunEffect::ToolPrepare { .. } | AgentRunEffect::LlmRequestStart { .. }
    )));
}
