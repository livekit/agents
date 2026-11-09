use super::*;

#[path = "machine_server_tests.rs"]
mod server;
#[path = "machine_simulation_tests.rs"]
mod sim;
#[path = "machine_sync_tests.rs"]
mod sync;

mod provider_tools {
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
mod review_fixes {
    use super::*;

    #[test]
    fn retirement_only_never_defers_audio_replies_or_items() {
        let mut m = machine();
        m.open(at(0));
        let first =
            m.command(at(0), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(32) }));
        let mut frames = first.frames;
        for _ in 0..32 {
            if !m.context_pages_pending() {
                break;
            }
            let page = m.continue_context(at(0));
            assert!(page.fatal.is_none());
            frames.extend(page.frames);
        }
        assert!(!m.context_pages_pending());
        assert!(m.has_context_retirement());
        for _ in 0..160 {
            let step = m.command(at(0), audio(2));
            assert!(!step.recycle && step.fatal.is_none());
            assert_eq!(step.frames.len(), 1, "cleanup must not defer audio");
        }
        let reply = m.command(
            at(0),
            Command::GenerateReply(GenerateReply {
                req_id: "during-cleanup".into(),
                instructions: None,
                tool_choice: None,
            }),
        );
        assert_eq!(reply.frames.len(), 1);
        let pushed = m.command(
            at(0),
            Command::PushItem(PushItem {
                item: Box::new(message("after-cleanup")),
                reply_required: false,
            }),
        );
        assert_eq!(creates(&pushed), ["after-cleanup"]);
        // Replay must preserve the pushed tail after an exact page-sized emission.
        retire_context(&mut m);
        let first = m.reconnect(at(1));
        let replay = finish_context(&mut m, at(1), first);
        assert!(creates(&replay).contains(&"after-cleanup"));
    }

    #[test]
    fn ordinary_reconnect_waits_for_retirement_capacity() {
        let mut m = machine();
        m.open(at(0));
        let first = m.command(
            at(0),
            Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(10_000) }),
        );
        let initial = finish_context(&mut m, at(0), first);
        acknowledge_page(&mut m, initial);
        retire_context(&mut m);
        let first = m.reconnect(at(1));
        assert!(first.fatal.is_none());
        let mut submitted: Vec<String> = creates(&first).into_iter().map(str::to_owned).collect();
        for _ in 0..10_000 {
            if !m.context_continuation_ready() {
                break;
            }
            let page = m.continue_context(at(1));
            submitted.extend(creates(&page).into_iter().map(str::to_owned));
            assert!(page.fatal.is_none(), "ordinary continuation must backpressure");
            assert!(!m.closing);
            assert!(m.retired.len() <= 4);
            if !m.context_pages_pending() {
                break;
            }
        }
        assert!(m.has_context_retirement());
        assert!(m.context_pages_pending(), "replay must pause until cleanup can be admitted");
        assert!(!m.context_continuation_ready());
        retire_context(&mut m);
        let rest = finish_context(&mut m, at(1), Step::default());
        assert!(rest.fatal.is_none() && !m.closing);
        submitted.extend(creates(&rest).into_iter().map(str::to_owned));
        assert_eq!(submitted, (0..10_000).map(|id| id.to_string()).collect::<Vec<_>>());
        assert!(!m.context_pages_pending());
    }

    #[test]
    fn content_only_changes_do_not_restart_snapshot_progress() {
        let mut m = machine();
        m.open(at(0));
        let first =
            m.command(at(0), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(300) }));
        let initial = finish_context(&mut m, at(0), first);
        acknowledge_page(&mut m, initial);
        retire_context(&mut m);
        m.command(at(0), Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(301) }));
        let mut completed = false;
        for _ in 0..100 {
            m.incoming_on(
                m.connection_generation(),
                at(0),
                Incoming::TranscriptionCompleted(
                    crate::realtime::protocol::TranscriptionCompleted {
                        item_id: "0".into(),
                        content_index: 0,
                        transcript: "continued".into(),
                        confidence: None,
                        in_progress: false,
                        event_id: "transcript".into(),
                        usage: None,
                    },
                ),
            );
            let page = m.continue_context(at(0));
            assert!(page.fatal.is_none());
            retire_context(&mut m);
            if !m.context_pages_pending() {
                completed = true;
                break;
            }
        }
        assert!(completed, "steady content updates must not starve structural traversal");
    }

    #[test]
    fn rejected_delete_preserves_order_after_another_deleted_predecessor() {
        let mut m = machine();
        m.open(at(0));
        acknowledge(&mut m, at(0), "keep", None);
        acknowledge(&mut m, at(0), "a", Some("keep"));
        acknowledge(&mut m, at(0), "b", Some("a"));
        acknowledge(&mut m, at(0), "last", Some("b"));
        let first = m.command(
            at(0),
            Command::UpdateChatCtx(UpdateChatCtx {
                chat_ctx: ChatContext::new(vec![message("keep"), message("last")]),
            }),
        );
        let edits = finish_context(&mut m, at(0), first);
        let event_id = edits
            .frames
            .iter()
            .find_map(|frame| match frame {
                Outgoing::ItemDelete { item_id, event_id } if item_id == "b" => {
                    Some(event_id.clone())
                }
                _ => None,
            })
            .unwrap();
        let rejection = m.incoming_on(
            m.connection_generation(),
            at(0),
            Incoming::Error(crate::realtime::protocol::ServerError {
                event_id: Some(event_id),
                code: Some("delete_rejected".into()),
                type_: None,
                message: "cannot delete b".into(),
            }),
        );
        finish_context(&mut m, at(0), rejection);
        let first = m.reconnect(at(1));
        let replay = finish_context(&mut m, at(1), first);
        assert_eq!(creates(&replay), ["keep", "b", "last"]);
    }

    #[test]
    fn deferred_audio_and_truncation_share_the_remaining_page_allowance() {
        use lk_agents_abi::llm::realtime::Truncate;
        let mut m = machine();
        m.open(at(0));
        // Duplicate IDs are filtered during replay. The remaining62 validation units
        // emit nothing and leave one desired item before the deferred63audio + truncate.
        let first = m.command(
            at(0),
            Command::UpdateChatCtx(UpdateChatCtx {
                chat_ctx: ChatContext::new(vec![message("only"); 63]),
            }),
        );
        assert_eq!(creates(&first), ["only"]);
        for _ in 0..63 {
            assert!(m.command(at(0), audio(2)).frames.is_empty());
        }
        assert!(
            m.command(
                at(0),
                Command::Truncate(Truncate {
                    message_id: "only".into(),
                    has_audio: false,
                    audio_end_ms: 0,
                    audio_transcript: Some("cut".into()),
                })
            )
            .frames
            .is_empty()
        );
        let mut audio_frames = 0;
        let mut edited = false;
        for _ in 0..100 {
            let page = m.continue_context(at(0));
            assert!(page.fatal.is_none());
            assert!(
                page.frames.len() <= 64,
                "nested truncate exceeded its page: {}",
                page.frames.len()
            );
            for frame in page.frames {
                match frame {
                    Outgoing::AudioAppend { .. } => audio_frames += 1,
                    Outgoing::ItemCreate { item: ChatItem::Message(message), .. } => {
                        edited |=
                            message.content == vec![lk_agents_abi::ChatContent::Text("cut".into())];
                    }
                    _ => {}
                }
            }
            retire_context(&mut m);
            if !m.context_pages_pending() {
                break;
            }
        }
        assert_eq!(audio_frames, 63);
        assert!(edited);
        assert!(!m.context_pages_pending());
    }

    #[test]
    fn repeated_reconnects_preserve_replay_with_initial_cleanup_withheld() {
        let mut m = machine();
        m.open(at(0));
        let mut initial = m.command(
            at(0),
            Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(10_000) }),
        );
        for _ in 0..1_000 {
            if !m.context_pages_pending() {
                break;
            }
            assert!(m.context_continuation_ready(), "initial submission unexpectedly parked");
            let page = m.continue_context(at(0));
            assert!(page.fatal.is_none());
            initial.merge(page);
        }
        assert!(!m.context_pages_pending());
        assert_eq!(creates(&initial).len(), 10_000);
        acknowledge_page(&mut m, initial);
        assert!(m.has_context_retirement());
        assert!(m.retired.len() <= 3);

        let first = m.reconnect(at(1));
        assert!(first.fatal.is_none(), "the first reconnect exhausted cleanup slots");
        assert!(first.frames.iter().all(|frame| !matches!(frame, Outgoing::ItemCreate { .. })));
        let queued = m.retired.len();
        let second = m.reconnect(at(2));
        assert!(second.fatal.is_none(), "an empty old connection consumed a fifth cleanup slot");
        assert!(!m.closing);
        assert_eq!(m.retired.len(), queued, "empty connection state must not add a cleanup batch");
        assert!(!m.context_continuation_ready(), "replay must await withheld disposal admission");

        retire_context(&mut m);
        let replay = finish_context(&mut m, at(2), second);
        assert!(replay.fatal.is_none() && !m.closing);
        assert_eq!(creates(&replay), (0..10_000).map(|id| id.to_string()).collect::<Vec<_>>());
    }

    fn completed_update_with_cleanup_withheld() -> RealtimeMachine {
        let mut m = machine();
        m.open(at(0));
        let mut submitted = m.command(
            at(0),
            Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(10_000) }),
        );
        while m.context_pages_pending() {
            assert!(m.context_continuation_ready());
            let page = m.continue_context(at(0));
            assert!(page.fatal.is_none());
            submitted.merge(page);
        }
        acknowledge_page(&mut m, submitted);
        assert_eq!(m.retired.len(), 2);
        m
    }

    fn expect_terminal_context_refusal(m: &mut RealtimeMachine, step: Step, req_id: &str) {
        let error = step.fatal.expect("external context bursts must refuse at the ownership cap");
        assert!(!error.recoverable);
        assert_eq!(error.message, "realtime context retirement capacity exceeded");
        assert_eq!(
            step.events
                .iter()
                .filter(|event| matches!(event,
            Event::GenerateReplyFailed(failed) if failed.req_id == req_id))
                .count(),
            1,
            "terminal refusal must settle the outstanding reply exactly once"
        );
        assert!(m.closing);
        assert!(m.has_context_retirement(), "terminal refusal must retain cleanup ownership");
        // Merely observing the closed machine or trying to continue it cannot dispose its owners.
        assert!(m.continue_context(at(1)).frames.is_empty());
        assert!(m.has_context_retirement());
        let mut admitted = Vec::new();
        while let Some(owner) = m.take_context_retirement() {
            admitted.push(owner);
        }
        assert!(!admitted.is_empty());
        assert!(!m.has_context_retirement());
        // In production these opaque owners are dropped only by the admitted cleanup worker.
        drop(admitted);
    }

    #[test]
    fn external_update_burst_refuses_terminally_when_prepare_owners_exhaust_cleanup_capacity() {
        let mut m = completed_update_with_cleanup_withheld();
        m.command(
            at(0),
            Command::GenerateReply(GenerateReply {
                req_id: "prepare-burst".into(),
                instructions: None,
                tool_choice: None,
            }),
        );
        // U2 parks. U3 and U4 replace its entire 10k Prepare owner, filling slots 3 and 4.
        for expected in [2, 3, 4] {
            let step = m.command(
                at(0),
                Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(10_000) }),
            );
            assert!(step.fatal.is_none());
            assert_eq!(m.retired.len(), expected);
            assert!(!m.context_continuation_ready());
        }
        let refused = m.command(
            at(0),
            Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(10_000) }),
        );
        expect_terminal_context_refusal(&mut m, refused, "prepare-burst");
    }

    #[test]
    fn external_update_burst_refuses_terminally_when_overlay_pending_inputs_exhaust_cleanup_capacity()
     {
        let mut m = completed_update_with_cleanup_withheld();
        let reconnect = m.reconnect(at(1));
        assert!(reconnect.fatal.is_none());
        assert_eq!(m.retired.len(), 3);
        m.command(
            at(1),
            Command::GenerateReply(GenerateReply {
                req_id: "overlay-burst".into(),
                instructions: None,
                tool_choice: None,
            }),
        );
        // U2 becomes the pending input; U3 retires that input as the fourth large owner.
        for expected in [3, 4] {
            let step = m.command(
                at(1),
                Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(10_000) }),
            );
            assert!(step.fatal.is_none());
            assert_eq!(m.retired.len(), expected);
            assert!(!m.context_continuation_ready());
        }
        let refused = m.command(
            at(1),
            Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: long_context(10_000) }),
        );
        expect_terminal_context_refusal(&mut m, refused, "overlay-burst");
    }
}
use crate::realtime::dialect::OPENAI;
use crate::realtime::protocol::Incoming;
use lk_agents_abi::llm::realtime::{GenerateReply, PushItem, UpdateChatCtx};
use lk_agents_abi::{ChatMessage, Role};

pub(super) fn at(seconds: u64) -> Ts {
    Ts::from_nanos(seconds * 1_000_000_000)
}

pub(super) fn message(id: &str) -> ChatItem {
    ChatMessage::text(id, Role::User, id, SystemTime::UNIX_EPOCH).into()
}

pub(super) fn machine() -> RealtimeMachine {
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

pub(super) fn creates(step: &Step) -> Vec<&str> {
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
    m.response_sent(m.connection_generation(), at(0), "pending", ReplyKind::Generated);
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

pub(super) fn retire_context(m: &mut RealtimeMachine) {
    while let Some(retirement) = m.take_context_retirement() {
        drop(retirement);
    }
}

pub(super) fn finish_context(m: &mut RealtimeMachine, now: Ts, mut first: Step) -> Step {
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

pub(super) fn acknowledge_page(m: &mut RealtimeMachine, page: Step) {
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
    m.response_sent(m.connection_generation(), at(12), "queued", ReplyKind::Generated);
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
    m.response_sent(m.connection_generation(), at(20), "canceled", ReplyKind::Say);
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
    m.response_sent(m.connection_generation(), at(0), "canceled", ReplyKind::Say);
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
