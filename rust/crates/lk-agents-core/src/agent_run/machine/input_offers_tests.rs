use super::tests::*;
use super::*;
use crate::delegation::remote::routing::{INPUT_OFFERS_KEY, captured_offers};
use lk_agents_abi::FunctionToolCall;
use lk_agents_abi::session::{InputCandidate, InputRequestRef};
use serde_json::json;

fn remote_tool() -> lk_agents_abi::agent::FunctionToolInfo {
    let mut tool = string_tool("flights", "request");
    tool.remote = Some("https://flights.test".into());
    tool
}

fn candidate() -> InputCandidate {
    InputCandidate {
        reference: InputRequestRef::RemoteTask {
            remote_id: "flights".into(),
            task: lk_agents_abi::delegation::remote::RemoteTaskRef {
                context_id: "trip".into(),
                task_id: "flight".into(),
            },
            request_id: "flight/input_1".into(),
        },
        question: "Which day?".into(),
        presentation: ChatMessage::text(
            "asked",
            Role::Assistant,
            "Which day?",
            std::time::UNIX_EPOCH,
        ),
        response: Some(user("answer", "Friday")),
    }
}

#[test]
fn admitted_remote_calls_retain_only_the_owners_frozen_occurrences() {
    let offered = vec![candidate()];
    let run = AgentRun::new(
        text_agent(vec![remote_tool()]),
        input(Opening::Message(Box::new(user("answer", "Friday"))), vec![]),
    )
    .unwrap()
    .with_input_offers(offered.clone());
    let mut driver = Driver::start(run);
    driver.reply("", &[("flights", json!({"request":"Friday"}))]);
    let prepared = driver
        .effects
        .iter()
        .find_map(|effect| match effect {
            AgentRunEffect::ToolPrepare { function_call, .. } => Some(function_call),
            _ => None,
        })
        .unwrap();
    assert_eq!(captured_offers(prepared).unwrap(), offered);
}

#[test]
fn a_direct_run_overwrites_model_supplied_input_authority_with_empty_offers() {
    let mut driver = Driver::new(text_agent(vec![remote_tool()]), "Hi", vec![]);
    let (request, _, _) = driver.request();
    driver.send(AgentRunEvent::LlmChunk {
        request,
        chunk: chunk(
            None,
            vec![FunctionToolCall {
                name: "flights".into(),
                arguments: json!({"request":"Friday"}).to_string(),
                call_id: "call".into(),
                extra: Some(
                    [
                        (INPUT_OFFERS_KEY.into(), serde_json::to_value(vec![candidate()]).unwrap()),
                        ("google".into(), json!({"signature":"kept"})),
                    ]
                    .into_iter()
                    .collect(),
                ),
            }],
            None,
        ),
    });
    let prepared = driver
        .effects
        .iter()
        .find_map(|effect| match effect {
            AgentRunEffect::ToolPrepare { function_call, .. } => Some(function_call),
            _ => None,
        })
        .unwrap();
    assert!(captured_offers(prepared).unwrap().is_empty());
    assert_eq!(prepared.extra[INPUT_OFFERS_KEY], json!([]));
    assert_eq!(prepared.extra["google"], json!({"signature":"kept"}));
}

#[test]
fn ordinary_local_calls_keep_provider_extras_without_accepting_forged_remote_offers() {
    for spoofed in [false, true] {
        let mut driver =
            Driver::new(text_agent(vec![string_tool("lookup", "request")]), "Hi", vec![]);
        let (request, _, _) = driver.request();
        let mut extra =
            serde_json::Map::from_iter([("google".into(), json!({"signature":"kept"}))]);
        if spoofed {
            extra.insert(INPUT_OFFERS_KEY.into(), serde_json::to_value(vec![candidate()]).unwrap());
        }
        driver.send(AgentRunEvent::LlmChunk {
            request,
            chunk: chunk(
                None,
                vec![FunctionToolCall {
                    name: "lookup".into(),
                    arguments: json!({"request":"Friday"}).to_string(),
                    call_id: "call".into(),
                    extra: Some(extra),
                }],
                None,
            ),
        });
        let prepared = driver
            .effects
            .iter()
            .find_map(|effect| match effect {
                AgentRunEffect::ToolPrepare { function_call, .. } => Some(function_call),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            prepared.extra,
            serde_json::Map::from_iter([("google".into(), json!({"signature":"kept"}))])
        );
    }
}
