//! Provider declarations and client-function namespaces share one validated tool update.

use super::*;
use crate::realtime::dialect::XAI;
use lk_agents_abi::llm::realtime::UpdateTools;
use lk_agents_abi::{ToolFlags, ToolSpec};
use serde_json::{Value, json};

fn function(name: &str) -> ToolSpec {
    ToolSpec {
        name: name.into(),
        description: "lookup".into(),
        json_schema: json!({"type": "object", "properties": {}}),
        flags: ToolFlags::default(),
        provider_tool: None,
    }
}

fn provider(payload: Value) -> ToolSpec {
    ToolSpec {
        name: format!("xai_{}", payload["type"].as_str().unwrap_or("unknown")),
        description: String::new(),
        json_schema: Value::Null,
        flags: ToolFlags::default(),
        provider_tool: Some(lk_agents_abi::llm::types::ProviderToolDeclaration {
            provider: Some("xai".into()),
            payload,
        }),
    }
}

fn xai() -> RealtimeMachine {
    let opts = SessionOptions::from_dialect(&XAI, false);
    let caps = XAI.capabilities(opts.turn_detection.as_ref(), false, true, true);
    RealtimeMachine::new(XAI, opts, caps, "test_".into())
}

fn rejected(step: &Step, message: &str) {
    assert!(step.frames.is_empty(), "a rejected tool update must not reach the provider");
    assert!(step.events.iter().any(|event| matches!(event,
        Event::Error(error) if !error.recoverable && error.message.contains(message))));
}

#[test]
fn xai_provider_function_collisions_reject_the_update_and_keep_the_previous_tools() {
    let reserved = [
        (json!({"type": "web_search"}), vec!["web_search", "browse_page"]),
        (
            json!({"type": "x_search"}),
            vec!["x_keyword_search", "x_semantic_search", "x_user_search", "x_thread_fetch"],
        ),
        (
            json!({"type": "file_search", "vector_store_ids": []}),
            vec!["collections_search", "file_search"],
        ),
    ];
    for (payload, names) in reserved {
        for name in names {
            let mut machine = xai();
            machine.open(at(0));
            machine.command(
                at(0),
                Command::UpdateTools(UpdateTools { tools: vec![function("lookup")] }),
            );
            let step = machine.command(
                at(1),
                Command::UpdateTools(UpdateTools {
                    tools: vec![provider(payload.clone()), function(name)],
                }),
            );
            rejected(&step, name);
            let replay = machine.reconnect(at(2));
            let names = replay
                .frames
                .iter()
                .find_map(|frame| match frame {
                    Outgoing::ToolsUpdate { tools, .. } => {
                        Some(tools.iter().map(|tool| tool.name.as_str()).collect::<Vec<_>>())
                    }
                    _ => None,
                })
                .unwrap();
            assert_eq!(names, ["lookup"], "reconnect must retain the last accepted tool set");
        }
    }
}

#[test]
fn xai_provider_schema_errors_never_replace_the_current_tools() {
    for payload in [
        json!({}),
        json!({"type": "other"}),
        json!({"type": "x_search", "allowed_x_handles": [4]}),
        json!({"type": "file_search"}),
        json!({"type": "file_search", "vector_store_ids": [3]}),
        json!({"type": "file_search", "vector_store_ids": [], "max_num_results": 1.5}),
    ] {
        let mut machine = xai();
        machine.open(at(0));
        let step = machine
            .command(at(1), Command::UpdateTools(UpdateTools { tools: vec![provider(payload)] }));
        rejected(&step, "provider tool");
    }
}

#[test]
fn xai_only_reserves_names_for_the_provider_tools_that_are_enabled() {
    let harmless = [
        (Some(json!({"type": "web_search"})), "view_image"),
        (Some(json!({"type": "web_search"})), "harmless_control"),
        (Some(json!({"type": "web_search"})), "collections_search"),
        (Some(json!({"type": "x_search"})), "x_search"),
        (Some(json!({"type": "x_search"})), "harmless_control"),
        (Some(json!({"type": "file_search", "vector_store_ids": []})), "view_document"),
        (Some(json!({"type": "file_search", "vector_store_ids": []})), "harmless_control"),
        (Some(json!({"type": "file_search", "vector_store_ids": []})), "web_search"),
        (None, "collections_search"),
        (None, "file_search"),
        (None, "web_search"),
        (None, "browse_page"),
    ];
    for (payload, name) in harmless {
        let mut machine = xai();
        machine.open(at(0));
        let mut tools = payload.into_iter().map(provider).collect::<Vec<_>>();
        tools.push(function(name));
        let step = machine.command(at(1), Command::UpdateTools(UpdateTools { tools }));
        assert_eq!(step.frames.len(), 1, "a harmless name must remain available: {name}");
        assert!(step.events.is_empty());
    }
}

#[test]
fn xai_file_search_alone_is_valid() {
    let mut machine = xai();
    machine.open(at(0));
    let step = machine.command(
        at(1),
        Command::UpdateTools(UpdateTools {
            tools: vec![provider(json!({"type": "file_search", "vector_store_ids": []}))],
        }),
    );
    assert_eq!(step.frames.len(), 1);
    assert!(step.events.is_empty());
}

#[test]
fn realtime_only_offers_the_current_dialects_supported_provider_tools() {
    use crate::realtime::dialect::{INFERENCE, OPENAI};
    // Azure mode uses this same OPENAI core dialect (transport/auth differ in runtime).
    for dialect in [XAI, OPENAI, INFERENCE] {
        for family in [None, Some("openai"), Some("gemini"), Some("anthropic"), Some("mistral")] {
            let opts = SessionOptions::from_dialect(&dialect, false);
            let caps = dialect.capabilities(opts.turn_detection.as_ref(), false, true, true);
            let mut machine = RealtimeMachine::new(dialect.clone(), opts, caps, "test_".into());
            machine.open(at(0));
            let mut foreign = provider(json!({"googleSearch": {}}));
            foreign.provider_tool.as_mut().unwrap().provider = family.map(str::to_owned);
            let mut same_wire_type = provider(json!({"type": "web_search"}));
            same_wire_type.provider_tool.as_mut().unwrap().provider = Some("openai".into());
            let step = machine.command(
                at(1),
                Command::UpdateTools(UpdateTools {
                    tools: vec![foreign, same_wire_type, function("lookup")],
                }),
            );
            assert!(step.events.is_empty(), "foreign declarations do not fail a tool update");
            let names = step
                .frames
                .iter()
                .find_map(|frame| match frame {
                    Outgoing::ToolsUpdate { tools, .. } => {
                        Some(tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>())
                    }
                    _ => None,
                })
                .unwrap();
            assert_eq!(names, ["lookup"]);
        }
    }
    let opts = SessionOptions::from_dialect(&OPENAI, false);
    let caps = OPENAI.capabilities(opts.turn_detection.as_ref(), false, true, true);
    let mut machine = RealtimeMachine::new(OPENAI, opts, caps, "test_".into());
    machine.open(at(0));
    let step = machine.command(
        at(1),
        Command::UpdateTools(UpdateTools {
            tools: vec![provider(json!({"type": "web_search"})), function("lookup")],
        }),
    );
    assert!(matches!(&step.frames[0], Outgoing::ToolsUpdate { tools, .. }
        if tools.len() == 1 && tools[0].name == "lookup"));
}
