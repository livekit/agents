//! Realtime contracts observed at the synchronous typed boundary.

use super::*;

mod provider_tools {
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
            let step = machine.command(
                at(1),
                Command::UpdateTools(UpdateTools { tools: vec![provider(payload)] }),
            );
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
            for family in [None, Some("openai"), Some("gemini"), Some("anthropic"), Some("mistral")]
            {
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
}
#[path = "contract_review_tests.rs"]
mod review_fixes;
mod server_items {
    //! Unsolicited items keep their position across delayed deletion receipts and reconnects.

    use super::*;

    const DELETED_SUCCESSORS: usize = 300;
    const UNSUBMITTED_SUFFIX: usize = 300;

    fn pending_server_item(
        suffix: usize,
        server_item: ChatItem,
    ) -> (RealtimeMachine, Vec<Outgoing>, u64) {
        let mut m = machine();
        m.open(at(0));
        let initial = ChatContext::new(
            std::iter::once(message("a"))
                .chain((0..DELETED_SUCCESSORS).map(|id| message(&format!("deleted-{id}"))))
                .chain(std::iter::once(message("c")))
                .collect(),
        );
        let initial = m.command(at(0), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: initial }));
        let initial = finish_context(&mut m, at(0), initial);
        acknowledge_page(&mut m, initial);
        finish_context(&mut m, at(0), Step::default());

        let target = ChatContext::new(
            [message("a"), message("b"), message("c")]
                .into_iter()
                .chain((0..suffix).map(|id| message(&format!("tail-{id}"))))
                .collect(),
        );
        let mut page = m.command(at(1), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: target }));
        let mut submitted = Vec::new();
        loop {
            assert!(page.fatal.is_none());
            assert!(page.frames.len() <= 64);
            let started_creating = creates(&page).contains(&"b");
            submitted.extend(page.frames);
            if started_creating {
                break;
            }
            assert!(m.context_work_pending(), "the replacement never began its create phase");
            page = m.continue_context(at(1));
            retire_context(&mut m);
        }
        assert_eq!(
            submitted.iter().filter(|frame| matches!(frame, Outgoing::ItemDelete { .. })).count(),
            DELETED_SUCCESSORS
        );
        if suffix == 0 {
            submitted.extend(finish_context(&mut m, at(1), Step::default()).frames);
            assert!(
                !m.context_pages_pending(),
                "no prior preparation may mask the gap walk's yield"
            );
        } else {
            assert!(
                submitted
                    .iter()
                    .filter(|frame| matches!(frame, Outgoing::ItemCreate { .. }))
                    .count()
                    < suffix + 1,
                "the schedule must retain an unsubmitted target suffix"
            );
        }

        let generation = m.connection_generation();
        let inserted = m.incoming_on(
            generation,
            at(2),
            Incoming::ItemAdded { previous_item_id: Some("a".into()), item: server_item },
        );
        assert!(
            m.chat_ctx().get_by_id("u").is_some(),
            "the acknowledged mirror must update immediately"
        );
        assert!(inserted.frames.is_empty(), "the deleted-successor walk must yield after one page");
        assert!(m.context_pages_pending(), "the successor gap must retain a continuation");
        assert!(
            m.tick(at(2)).frames.is_empty(),
            "control must remain serviceable between gap pages"
        );
        (m, submitted, generation)
    }

    fn expected_ids(suffix: usize) -> Vec<String> {
        ["a", "b", "u", "c"]
            .into_iter()
            .map(str::to_owned)
            .chain((0..suffix).map(|id| format!("tail-{id}")))
            .collect()
    }

    fn acknowledge_frames(m: &mut RealtimeMachine, frames: Vec<Outgoing>) {
        for frame in frames {
            let incoming = match frame {
                Outgoing::ItemCreate { item, previous_item_id, .. } => Incoming::ItemAdded {
                    previous_item_id: previous_item_id.filter(|id| id != "root"),
                    item,
                },
                Outgoing::ItemDelete { item_id, .. } => Incoming::ItemDeleted { item_id },
                _ => continue,
            };
            m.incoming_on(m.connection_generation(), at(3), incoming);
            retire_context(m);
        }
    }

    #[test]
    fn server_item_placement_yields_across_deleted_successors_and_keeps_local_neighbors() {
        let (mut m, mut submitted, _) = pending_server_item(0, message("u"));
        let rest = finish_context(&mut m, at(3), Step::default());
        submitted.extend(rest.frames);
        assert!(
            !submitted.iter().any(|frame| matches!(frame,
                Outgoing::ItemCreate { item, .. } if item.id() == "u"
            )),
            "an acknowledged unsolicited item must not be created again"
        );
        acknowledge_frames(&mut m, submitted);
        assert_eq!(
            m.chat_ctx().items.iter().map(|item| item.id().to_owned()).collect::<Vec<_>>(),
            expected_ids(0)
        );
    }

    #[test]
    fn reconnect_during_server_item_placement_preserves_pending_item_and_unsubmitted_target() {
        let (mut m, _, previous_generation) = pending_server_item(UNSUBMITTED_SUFFIX, message("u"));
        let first = m.reconnect(at(3));
        assert!(creates(&first).is_empty(), "the old successor walk must finish before replay");
        assert!(m.context_work_pending());
        let second = m.reconnect(at(4));
        assert!(creates(&second).is_empty());
        m.incoming_on(
            previous_generation,
            at(4),
            Incoming::ItemAdded { previous_item_id: None, item: message("obsolete") },
        );
        assert!(
            m.chat_ctx().get_by_id("obsolete").is_none(),
            "an older socket changed the current mirror"
        );
        assert!(m.tick(at(4)).frames.is_empty());
        let replay = finish_context(&mut m, at(4), second);
        assert_eq!(
            creates(&replay),
            expected_ids(UNSUBMITTED_SUFFIX),
            "replay must preserve ordered, exactly-once identities"
        );
        acknowledge_page(&mut m, replay);
        assert_eq!(
            m.chat_ctx().items.iter().map(|item| item.id().to_owned()).collect::<Vec<_>>(),
            expected_ids(UNSUBMITTED_SUFFIX)
        );
    }

    #[test]
    fn current_connection_items_survive_completion_of_an_older_server_item_gap() {
        let (mut m, _, _) = pending_server_item(UNSUBMITTED_SUFFIX, message("u"));
        let first = m.reconnect(at(3));
        assert!(creates(&first).is_empty());
        m.incoming_on(
            m.connection_generation(),
            at(3),
            Incoming::ItemAdded { previous_item_id: None, item: message("current") },
        );
        retire_context(&mut m);
        assert!(m.chat_ctx().get_by_id("current").is_some());
        let replay = finish_context(&mut m, at(3), Step::default());
        assert!(
            m.chat_ctx().get_by_id("current").is_some(),
            "resuming an old gap walk cleared the current acknowledged mirror"
        );
        assert_eq!(
            creates(&replay),
            expected_ids(UNSUBMITTED_SUFFIX),
            "an already acknowledged current item was recreated"
        );
        acknowledge_frames(&mut m, replay.frames);
        let mut expected = expected_ids(UNSUBMITTED_SUFFIX);
        expected.push("current".into());
        assert_eq!(
            m.chat_ctx().items.iter().map(|item| item.id().to_owned()).collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn a_tool_output_waits_for_its_acknowledged_call_during_server_item_placement() {
        let call = ChatItem::FunctionCall(lk_agents_abi::FunctionCall {
            id: "u".into(),
            call_id: "expert".into(),
            arguments: "{}".into(),
            name: "lookup".into(),
            created_at: SystemTime::UNIX_EPOCH,
            extra: Default::default(),
            group_id: None,
        });
        let (mut m, mut submitted, _) = pending_server_item(0, call);
        let output = |id: &str, call_id: &str| {
            ChatItem::FunctionCallOutput(lk_agents_abi::FunctionCallOutput::new(
                id,
                call_id,
                "resolved",
                false,
                SystemTime::UNIX_EPOCH,
            ))
        };
        let rejected = m.command(
            at(3),
            Command::PushItem(PushItem {
                item: Box::new(output("unknown-output", "unknown-call")),
                reply_required: false,
            }),
        );
        assert!(rejected.frames.is_empty());
        let queued = m.command(
            at(3),
            Command::PushItem(PushItem {
                item: Box::new(output("tool-result", "expert")),
                reply_required: false,
            }),
        );
        assert!(
            queued.frames.is_empty(),
            "the call placement must settle before its output is sent"
        );
        let rest = finish_context(&mut m, at(3), queued);
        submitted.extend(rest.frames);
        assert_eq!(submitted.iter().filter(|frame| matches!(frame, Outgoing::ItemCreate { item: ChatItem::FunctionCallOutput(output), .. } if output.id == "tool-result" && output.call_id == "expert" && output.output == "resolved")).count(), 1);
        assert!(!submitted.iter().any(
            |frame| matches!(frame, Outgoing::ItemCreate { item, .. } if item.id() == "unknown-output")
        ));
    }
}
use crate::realtime::dialect::OPENAI;
use crate::realtime::protocol::Incoming;
use lk_agents_abi::llm::realtime::{GenerateReply, PushItem, UpdateChatCtx};
use lk_agents_abi::{ChatMessage, Role};

fn at(seconds: u64) -> Ts {
    Ts::from_nanos(seconds * 1_000_000_000)
}

fn message(id: &str) -> ChatItem {
    ChatMessage::text(id, Role::User, id, SystemTime::UNIX_EPOCH).into()
}

fn machine() -> RealtimeMachine {
    let opts = SessionOptions::from_dialect(&OPENAI, false);
    let caps = OPENAI.capabilities(opts.turn_detection.as_ref(), false, true, true);
    RealtimeMachine::new(OPENAI, opts, caps, "test_".into())
}

fn acknowledge(m: &mut RealtimeMachine, now: Ts, id: &str, previous: Option<&str>) {
    m.incoming_on(
        m.connection_generation(),
        now,
        Incoming::ItemAdded { previous_item_id: previous.map(str::to_owned), item: message(id) },
    );
}

fn creates(step: &Step) -> Vec<&str> {
    step.frames
        .iter()
        .filter_map(|frame| match frame {
            Outgoing::ItemCreate { item, .. } => Some(item.id()),
            _ => None,
        })
        .collect()
}

#[test]
fn realtime_pending_response_deadline_is_owned_by_the_core() {
    let mut m = machine();
    m.open(at(0));
    m.command(
        at(0),
        Command::GenerateReply(GenerateReply {
            req_id: "pending".into(),
            instructions: None,
            tool_choice: None,
        }),
    );
    m.response_sent(m.connection_generation(), at(0), "pending", false);
    assert_eq!(m.next_deadline(), Some(at(5)));
    let step = m.tick(at(5));
    assert!(step.events.iter().any(|event| matches!(event,
        Event::GenerateReplyFailed(failed) if failed.req_id == "pending")));
    assert!(!m.has_active_generation());
    let late = m.incoming_on(
        m.connection_generation(),
        at(6),
        Incoming::ResponseCreated {
            response_id: "late".into(),
            client_event_id: Some("pending".into()),
        },
    );
    assert!(!late.events.iter().any(|event| matches!(event, Event::GenerationCreated(_))));
    assert!(late.frames.iter().any(|frame| matches!(frame,
        Outgoing::ResponseCancel { response_id: Some(id) } if id == "late")));
}

#[test]
fn realtime_connection_generation_rejects_a_previous_socket_ack() {
    let mut m = machine();
    m.open(at(0));
    let previous = m.connection_generation();
    m.reconnect(at(1));
    m.incoming_on(
        previous,
        at(2),
        Incoming::ItemAdded { previous_item_id: None, item: message("stale") },
    );
    assert!(m.chat_ctx().is_empty());
    acknowledge(&mut m, at(2), "current", None);
    assert_eq!(m.chat_ctx().items[0].id(), "current");
}

#[test]
fn realtime_reconnect_mirror_preserves_pushed_unconfirmed_items_without_duplicates() {
    let mut m = machine();
    m.open(at(0));
    acknowledge(&mut m, at(0), "a", None);
    let push = m.command(
        at(1),
        Command::PushItem(PushItem { item: Box::new(message("b")), reply_required: false }),
    );
    assert_eq!(creates(&push), ["b"]);
    // A pushed item is desired state even before the server acknowledges it.
    assert_eq!(m.chat_ctx().len(), 1);
    let target = ChatContext::new(vec![message("a"), message("b"), message("c")]);
    let update =
        m.command(at(1), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: target.clone() }));
    assert_eq!(creates(&update), ["c"]);
    let replay = m.reconnect(at(2));
    assert_eq!(creates(&replay), ["a", "b", "c"]);
    assert!(m.chat_ctx().is_empty());
    let repeated = m.command(at(2), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: target }));
    assert!(repeated.frames.is_empty());
    acknowledge(&mut m, at(3), "a", None);
    acknowledge(&mut m, at(3), "b", Some("a"));
    acknowledge(&mut m, at(3), "c", Some("b"));
    assert_eq!(m.chat_ctx().items.iter().map(ChatItem::id).collect::<Vec<_>>(), ["a", "b", "c"]);
}

#[test]
fn realtime_reconnect_mirror_keeps_unsolicited_items_beside_pending_insertions() {
    let mut m = machine();
    m.open(at(0));
    acknowledge(&mut m, at(0), "a", None);
    acknowledge(&mut m, at(0), "c", Some("a"));
    m.command(
        at(1),
        Command::UpdateChatCtx(UpdateChatCtx {
            chat_ctx: ChatContext::new(vec![message("a"), message("b"), message("c")]),
        }),
    );
    acknowledge(&mut m, at(2), "unsolicited", Some("a"));
    let replay = m.reconnect(at(3));
    assert_eq!(creates(&replay), ["a", "b", "unsolicited", "c"]);
}

#[test]
fn realtime_session_recycle_deadline_suspends_while_a_response_is_active() {
    let mut machine = machine();
    machine.open(at(0));
    let deadline = machine.next_deadline().unwrap();
    let generation = machine.connection_generation();
    machine.incoming_on(
        generation,
        at(1),
        Incoming::ResponseCreated { response_id: "speaking".into(), client_event_id: None },
    );
    machine.tick(deadline);
    assert!(
        machine.next_deadline().is_none(),
        "an overdue recycle timer cannot spin while the agent speaks"
    );
}

#[test]
fn identical_pending_updates_still_reconcile_unsolicited_server_edits() {
    let mut m = machine();
    m.open(at(0));
    acknowledge(&mut m, at(0), "a", None);
    let target = ChatContext::new(vec![message("a"), message("b")]);
    m.command(at(1), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: target.clone() }));
    m.incoming_on(
        m.connection_generation(),
        at(2),
        Incoming::TranscriptionCompleted(crate::realtime::protocol::TranscriptionCompleted {
            item_id: "a".into(),
            content_index: 0,
            transcript: "changed on the server".into(),
            confidence: None,
            in_progress: false,
            event_id: "transcript".into(),
            usage: None,
        }),
    );
    let update = m.command(at(2), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: target }));
    assert_eq!(creates(&update), ["a"], "the pending item b must not be created twice");
    assert!(update.frames.iter().any(|frame| matches!(frame,
        Outgoing::ItemDelete { item_id, .. } if item_id == "a")));
}

#[test]
fn pushed_items_follow_unsolicited_server_items_while_an_update_is_pending() {
    let mut m = machine();
    m.open(at(0));
    acknowledge(&mut m, at(0), "a", None);
    m.command(
        at(1),
        Command::UpdateChatCtx(UpdateChatCtx {
            chat_ctx: ChatContext::new(vec![message("a"), message("b")]),
        }),
    );
    acknowledge(&mut m, at(2), "unsolicited", Some("a"));
    m.command(
        at(2),
        Command::PushItem(PushItem { item: Box::new(message("c")), reply_required: false }),
    );
    let replay = m.reconnect(at(3));
    assert_eq!(creates(&replay), ["a", "b", "unsolicited", "c"]);
}

#[test]
fn only_a_duplicate_rejection_confirms_an_unacknowledged_create() {
    for (code, acknowledged) in [("item_create_duplicate_item_id", true), ("invalid_item", false)] {
        let mut m = machine();
        m.open(at(0));
        acknowledge(&mut m, at(0), "a", None);
        let target = ChatContext::new(vec![message("a"), message("b")]);
        let update =
            m.command(at(1), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: target.clone() }));
        let event_id = update
            .frames
            .iter()
            .find_map(|frame| match frame {
                Outgoing::ItemCreate { event_id, item, .. } if item.id() == "b" => {
                    Some(event_id.clone())
                }
                _ => None,
            })
            .unwrap();
        m.incoming_on(
            m.connection_generation(),
            at(2),
            Incoming::Error(crate::realtime::protocol::ServerError {
                event_id: Some(event_id),
                code: Some(code.into()),
                type_: None,
                message: code.into(),
            }),
        );
        assert_eq!(m.chat_ctx().get_by_id("b").is_some(), acknowledged);
        let repeated = m.command(at(3), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: target }));
        assert_eq!(creates(&repeated).is_empty(), acknowledged);
    }
}

fn long_context(count: usize) -> ChatContext {
    ChatContext::new((0..count).map(|id| message(&id.to_string())).collect())
}

fn retire_context(m: &mut RealtimeMachine) {
    while let Some(retirement) = m.take_context_retirement() {
        drop(retirement);
    }
}

fn finish_context(m: &mut RealtimeMachine, now: Ts, mut first: Step) -> Step {
    retire_context(m);
    while m.context_work_pending() {
        let page = m.continue_context(now);
        retire_context(m);
        assert!(page.frames.len() <= 64, "one submission must fit a context page");
        first.merge(page);
    }
    first
}

fn first_context_page(m: &mut RealtimeMachine, mut page: Step) -> Step {
    retire_context(m);
    while page.frames.is_empty() && m.context_work_pending() {
        page = m.continue_context(at(0));
        retire_context(m);
    }
    assert!(!page.frames.is_empty());
    assert!(page.frames.len() <= 64);
    page
}

fn acknowledge_page(m: &mut RealtimeMachine, page: Step) {
    for frame in page.frames {
        if let Outgoing::ItemCreate { item, previous_item_id, .. } = frame {
            m.incoming_on(
                m.connection_generation(),
                at(0),
                Incoming::ItemAdded {
                    previous_item_id: previous_item_id.filter(|id| id != "root"),
                    item,
                },
            );
        }
    }
}

#[test]
fn acknowledged_pages_keep_the_entire_unsubmitted_reconnect_intent() {
    let mut m = machine();
    m.open(at(0));
    let first =
        m.command(at(0), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(260) }));
    let first = first_context_page(&mut m, first);
    acknowledge_page(&mut m, first);
    let replay = m.reconnect(at(1));
    let replay = finish_context(&mut m, at(1), replay);
    assert_eq!(creates(&replay), (0..260).map(|id| id.to_string()).collect::<Vec<_>>());
}

#[test]
fn superseding_context_pages_never_create_the_old_unsent_suffix() {
    let mut m = machine();
    m.open(at(0));
    let first =
        m.command(at(0), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(260) }));
    let first = first_context_page(&mut m, first);
    let submitted = creates(&first).len();
    acknowledge_page(&mut m, first);
    let mut target = long_context(5);
    target.items.push(message("replacement"));
    let update = m.command(at(1), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: target }));
    let update = finish_context(&mut m, at(1), update);
    assert_eq!(creates(&update), ["replacement"]);
    assert!(update.frames.iter().all(|frame| !matches!(frame,
        Outgoing::ItemCreate { item, .. } if item.id().parse::<usize>().is_ok_and(|id| id >= submitted))));
    let replay = m.reconnect(at(2));
    let replay = finish_context(&mut m, at(2), replay);
    assert_eq!(creates(&replay), ["0", "1", "2", "3", "4", "replacement"]);
}

#[test]
fn queued_responses_start_the_read_deadline_only_after_the_write_receipt() {
    let mut m = machine();
    m.open(at(0));
    m.command(at(0), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(260) }));
    let queued = m.command(
        at(0),
        Command::GenerateReply(GenerateReply {
            req_id: "queued".into(),
            instructions: None,
            tool_choice: None,
        }),
    );
    assert!(queued.frames.is_empty());
    assert!(
        !m.tick(at(10)).events.iter().any(|event| matches!(event, Event::GenerateReplyFailed(_)))
    );
    let sent = finish_context(&mut m, at(10), Step::default());
    assert_eq!(creates(&sent).len(), 260);
    assert!(
        matches!(sent.frames.last(), Some(Outgoing::ResponseCreate { req_id, .. }) if req_id == "queued")
    );
    m.response_sent(m.connection_generation(), at(12), "queued", false);
    assert_eq!(m.next_deadline(), Some(at(17)));
    assert!(m.tick(at(17)).events.iter().any(|event| matches!(event,
        Event::GenerateReplyFailed(failure) if failure.req_id == "queued")));
}

#[test]
fn canceled_unsent_say_stays_identifiable_after_a_delayed_write_receipt() {
    use crate::realtime::dialect::XAI;
    use lk_agents_abi::llm::realtime::{CancelReply, Say};
    let opts = SessionOptions::from_dialect(&XAI, false);
    let caps = XAI.capabilities(opts.turn_detection.as_ref(), false, true, true);
    let mut m = RealtimeMachine::new(XAI, opts, caps, "say_".into());
    m.open(at(0));
    m.command(
        at(0),
        Command::Say(Say {
            req_id: "canceled".into(),
            text: "hello".into(),
            add_to_chat_ctx: true,
        }),
    );
    m.command(at(0), Command::CancelReply(CancelReply { req_id: "canceled".into() }));
    assert!(!m.tick(at(20)).recycle, "an unsent say has no stale-wire timeout");
    m.response_sent(m.connection_generation(), at(20), "canceled", true);
    let late = m.incoming_on(
        m.connection_generation(),
        at(21),
        Incoming::ResponseCreated { response_id: "late".into(), client_event_id: None },
    );
    assert!(!late.events.iter().any(|event| matches!(event, Event::GenerationCreated(_))));
    assert!(late.frames.iter().any(|frame| matches!(frame,
        Outgoing::ResponseCancel { response_id: Some(id) } if id == "late")));
}

#[test]
fn expiring_a_sent_say_tombstone_recycles_before_processing_old_socket_traffic() {
    use crate::realtime::dialect::XAI;
    use lk_agents_abi::llm::realtime::{CancelReply, Say};
    let opts = SessionOptions::from_dialect(&XAI, false);
    let caps = XAI.capabilities(opts.turn_detection.as_ref(), false, true, true);
    let mut m = RealtimeMachine::new(XAI, opts, caps, "say_".into());
    m.open(at(0));
    m.command(
        at(0),
        Command::Say(Say {
            req_id: "canceled".into(),
            text: "hello".into(),
            add_to_chat_ctx: true,
        }),
    );
    m.response_sent(m.connection_generation(), at(0), "canceled", true);
    m.command(at(0), Command::CancelReply(CancelReply { req_id: "canceled".into() }));
    let old_generation = m.connection_generation();
    let late = m.incoming_on(
        old_generation,
        at(10),
        Incoming::ResponseCreated { response_id: "late".into(), client_event_id: None },
    );
    assert!(late.recycle);
    assert!(!late.events.iter().any(|event| matches!(event, Event::GenerationCreated(_))));
    m.reconnect(at(10));
    let old = m.incoming_on(
        old_generation,
        at(11),
        Incoming::ResponseCreated { response_id: "older".into(), client_event_id: None },
    );
    assert!(old.events.is_empty());
}

fn audio(bytes: usize) -> Command {
    Command::PushAudio(lk_agents_abi::llm::realtime::PushAudio {
        frame: lk_agents_abi::AudioFrame {
            data: vec![0; bytes].into(),
            sample_rate: 24_000,
            num_channels: 1,
            samples_per_channel: (bytes / 2) as u32,
        },
    })
}

#[test]
fn replay_defers_audio_commit_and_clear_in_their_original_order() {
    use lk_agents_abi::llm::realtime::{ClearAudio, CommitAudio};
    let mut m = machine();
    m.open(at(0));
    m.command(at(0), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(260) }));
    assert!(m.command(at(0), audio(9_600)).frames.is_empty());
    assert!(m.command(at(0), Command::CommitAudio(CommitAudio {})).frames.is_empty());
    assert!(m.command(at(0), Command::ClearAudio(ClearAudio {})).frames.is_empty());
    assert!(
        m.command(
            at(0),
            Command::GenerateReply(GenerateReply {
                req_id: "after_audio".into(),
                instructions: None,
                tool_choice: None,
            })
        )
        .frames
        .is_empty()
    );
    let sent = finish_context(&mut m, at(0), Step::default());
    let operations: Vec<&str> = sent
        .frames
        .iter()
        .filter_map(|frame| match frame {
            Outgoing::AudioAppend { .. } => Some("append"),
            Outgoing::AudioCommit => Some("commit"),
            Outgoing::AudioClear => Some("clear"),
            Outgoing::ResponseCreate { .. } => Some("response"),
            _ => None,
        })
        .collect();
    assert_eq!(operations, ["append", "commit", "clear", "response"]);
    assert_eq!(creates(&sent).len(), 260);
}

#[test]
fn replay_media_capacity_is_refused_explicitly_by_command_count_and_bytes() {
    use lk_agents_abi::llm::realtime::ClearAudio;
    for byte_bound in [false, true] {
        let mut m = machine();
        m.open(at(0));
        m.command(at(0), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(260) }));
        if byte_bound {
            assert!(m.command(at(0), audio(1024 * 1024)).events.is_empty());
        } else {
            for _ in 0..128 {
                assert!(m.command(at(0), Command::ClearAudio(ClearAudio {})).events.is_empty());
            }
        }
        let rejected = m.command(at(0), audio(2));
        assert!(rejected.recycle);
        assert!(rejected.events.iter().any(|event| matches!(event,
            Event::Error(error) if error.message.contains("media capacity exceeded"))));
    }
}

#[test]
fn superseding_partial_edits_preserves_content_that_has_not_been_submitted() {
    let text = |value: &str| -> ChatItem {
        ChatMessage::text("a", Role::User, value, SystemTime::UNIX_EPOCH).into()
    };
    let mut m = machine();
    m.open(at(0));
    let old = m.command(
        at(0),
        Command::PushItem(PushItem { item: Box::new(text("old")), reply_required: false }),
    );
    assert_eq!(creates(&old), ["a"]);
    let mut target = ChatContext::new(vec![text("new")]);
    target.items.extend((0..300).map(|id| message(&format!("new-{id}"))));
    let first =
        m.command(at(1), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: target.clone() }));
    let first = first_context_page(&mut m, first);
    assert!(
        first.frames.iter().all(|frame| !matches!(frame,
            Outgoing::ItemCreate { item, .. } if item.id() == "a"
        )),
        "the schedule must supersede before a's new content is submitted"
    );
    let superseded = m.command(at(1), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: target }));
    let superseded = finish_context(&mut m, at(1), superseded);
    assert!(superseded.frames.iter().any(|frame| matches!(frame,
        Outgoing::ItemCreate { item, .. } if item.id() == "a" && item.as_message().unwrap().raw_text_content().as_deref() == Some("new")
    )), "a prepared but unsent edit was mistaken for the submitted conversation");
}

#[test]
fn an_older_create_ack_cannot_discard_the_newer_unconfirmed_replacement() {
    let text = |value: &str| -> ChatItem {
        ChatMessage::text("a", Role::User, value, SystemTime::UNIX_EPOCH).into()
    };
    let mut m = machine();
    m.open(at(0));
    m.command(
        at(0),
        Command::PushItem(PushItem { item: Box::new(text("old")), reply_required: false }),
    );
    let update = m.command(
        at(1),
        Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: ChatContext::new(vec![text("new")]) }),
    );
    let _ = finish_context(&mut m, at(1), update);
    m.incoming_on(
        m.connection_generation(),
        at(1),
        Incoming::ItemAdded { previous_item_id: None, item: text("old") },
    );
    m.incoming_on(m.connection_generation(), at(1), Incoming::ItemDeleted { item_id: "a".into() });
    let _ = finish_context(&mut m, at(1), Step::default());
    // The writer may lose the socket after the acknowledged delete but before the replacement.
    let replay = m.reconnect(at(2));
    let replay = finish_context(&mut m, at(2), replay);
    assert!(replay.frames.iter().any(|frame| matches!(frame,
        Outgoing::ItemCreate { item, .. } if item.id() == "a" && item.as_message().unwrap().raw_text_content().as_deref() == Some("new")
    )), "the older create receipt settled a newer replacement on the same item");
}

#[test]
fn server_items_arriving_between_preparation_pages_are_reconciled_once() {
    let mut m = machine();
    m.open(at(0));
    let initial =
        m.command(at(0), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(260) }));
    let initial = finish_context(&mut m, at(0), initial);
    acknowledge_page(&mut m, initial);
    let update =
        m.command(at(1), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(261) }));
    assert!(update.frames.is_empty(), "the schedule must interrupt context preparation");
    acknowledge(&mut m, at(1), "unsolicited", Some("50"));
    let update = finish_context(&mut m, at(1), update);
    assert_eq!(creates(&update), ["260"]);
    assert_eq!(update.frames.iter().filter(|frame| matches!(frame, Outgoing::ItemDelete { item_id, .. } if item_id == "unsolicited")).count(), 1);
    for frame in update.frames {
        match frame {
            Outgoing::ItemCreate { item, previous_item_id, .. } => {
                m.incoming_on(
                    m.connection_generation(),
                    at(2),
                    Incoming::ItemAdded {
                        previous_item_id: previous_item_id.filter(|id| id != "root"),
                        item,
                    },
                );
            }
            Outgoing::ItemDelete { item_id, .. } => {
                m.incoming_on(m.connection_generation(), at(2), Incoming::ItemDeleted { item_id });
            }
            _ => {}
        }
    }
    assert_eq!(
        m.chat_ctx().items.iter().map(ChatItem::id).collect::<Vec<_>>(),
        (0..261).map(|id| id.to_string()).collect::<Vec<_>>()
    );
}

#[test]
fn reconnecting_during_edit_preparation_replays_the_complete_new_intent() {
    let mut m = machine();
    m.open(at(0));
    let initial =
        m.command(at(0), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(260) }));
    let initial = finish_context(&mut m, at(0), initial);
    acknowledge_page(&mut m, initial);
    let mut target = long_context(260);
    target.items.reverse();
    target.items.push(message("new"));
    let update =
        m.command(at(1), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: target.clone() }));
    assert!(update.frames.is_empty());
    let old_generation = m.connection_generation();
    m.reconnect(at(2));
    let replay = m.reconnect(at(3));
    m.incoming_on(
        old_generation,
        at(3),
        Incoming::ItemAdded { previous_item_id: None, item: message("stale") },
    );
    let replay = finish_context(&mut m, at(3), replay);
    assert_eq!(creates(&replay), target.items.iter().map(ChatItem::id).collect::<Vec<_>>());
    assert!(m.chat_ctx().is_empty());
}

#[test]
fn superseding_between_an_edit_delete_and_create_preserves_the_replacement() {
    let mut m = machine();
    m.open(at(0));
    let initial =
        m.command(at(0), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(4) }));
    let initial = finish_context(&mut m, at(0), initial);
    acknowledge_page(&mut m, initial);
    retire_context(&mut m);
    let mut target = long_context(4);
    for item in &mut target.items {
        item.as_message_mut().unwrap().content =
            vec![lk_agents_abi::ChatContent::Text("edited".into())];
    }
    target.items.push(message("tail"));
    let page = m.command(at(1), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: target.clone() }));
    let removed = match page.frames.last() {
        Some(Outgoing::ItemDelete { item_id, .. }) => item_id.clone(),
        other => panic!("the schedule must cut after an edit delete: {other:?}"),
    };
    let replacement = m.command(at(1), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: target }));
    let replacement = finish_context(&mut m, at(1), replacement);
    assert!(replacement.frames.iter().any(|frame| matches!(frame,
        Outgoing::ItemCreate { item, .. } if item.id() == removed && item.as_message().unwrap().raw_text_content().as_deref() == Some("edited")
    )));
    let replay = m.reconnect(at(2));
    let replay = finish_context(&mut m, at(2), replay);
    assert_eq!(creates(&replay), ["0", "1", "2", "3", "tail"]);
    assert!(replay.frames.iter().all(|frame| !matches!(frame,
        Outgoing::ItemCreate { item, .. } if item.id() != "tail" && item.as_message().unwrap().raw_text_content().as_deref() != Some("edited")
    )));
}

#[test]
fn delete_receipts_cannot_grow_retained_payloads_while_cleanup_admission_is_blocked() {
    let mut m = machine();
    m.open(at(0));
    let initial =
        m.command(at(0), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(1_000) }));
    let initial = finish_context(&mut m, at(0), initial);
    acknowledge_page(&mut m, initial);
    retire_context(&mut m);
    let mut overloaded = false;
    for id in 0..300 {
        let step = m.incoming_on(
            m.connection_generation(),
            at(1),
            Incoming::ItemDeleted { item_id: id.to_string() },
        );
        if let Some(error) = step.fatal {
            assert_eq!(error.message, "realtime context retirement capacity exceeded");
            overloaded = true;
            break;
        }
    }
    assert!(overloaded, "blocked cleanup must stop accepting a stream of deleted histories");
    assert!(
        m.has_context_retirement(),
        "overload must retain the rejected work for the cleanup owner"
    );
    retire_context(&mut m);
    assert!(!m.has_context_retirement());
}
