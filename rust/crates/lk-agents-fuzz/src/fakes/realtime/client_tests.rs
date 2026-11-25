use super::*;
use crate::{drive::simulate, record::Stamps};
use lk_agents_abi::{
    ChatItem, ChatMessage, Role,
    llm::realtime::{CancelReply, GenerateReply, UpdateChatCtx},
};
use lk_agents_runtime::{
    net::{clock::SimClock, task::TokioSpawner},
    voice::session::Timebase,
};

/// No async provider implementation: commands, deadlines and server facts cross the actual
/// production machine, while this fixture controls when the provider answers.
fn socket(
    delays: RealtimeDelays,
) -> (Arc<Socket>, tokio::sync::mpsc::UnboundedReceiver<super::super::Record>) {
    let clock: Arc<dyn Clock> = Arc::new(SimClock::new());
    let spawner: Arc<dyn Spawner> = Arc::new(TokioSpawner::current());
    let (registry, records) =
        Registry::new(Stamps::new(Timebase::of_clock(clock.clone())), spawner.clone());
    let setup = ServerSetup {
        provider: lk_agents_abi::llm::realtime::RealtimeProvider::Inference,
        vad: None,
        seed: 7,
        connect: Duration::ZERO,
        tool_calls: Some(0),
        text_only: false,
        fallback: false,
        timing: Default::default(),
        delays,
        user_transcription: true,
    };
    let fake = FakeRealtime::new(registry.clone(), setup.clone(), clock.clone(), spawner);
    let caps = capabilities(None, false, false, true);
    let socket = Arc::new(Socket {
        setup,
        caps,
        clock,
        registry,
        slot: OnceLock::new(),
        state: Mutex::new(State::new(Ts::ZERO, None, fake.machine(None, caps, 7))),
        wake: tokio::sync::Notify::new(),
        ready: tokio::sync::Notify::new(),
    });
    let slot =
        socket.registry.register(Kind::Realtime, Detail::None, Control::Realtime(socket.clone()));
    assert!(socket.slot.set(slot).is_ok(), "one registration");
    {
        let mut st = socket.state.lock().unwrap();
        st.server = socket.new_server(1);
        socket.advance(&mut st);
    }
    (socket, records)
}

fn command(socket: &Socket, command: RealtimeCommand) {
    let mut st = socket.state.lock().unwrap();
    st.commands.push_back(command);
    socket.advance(&mut st);
}

async fn advance(socket: &Socket, millis: u64) {
    tokio::time::advance(Duration::from_millis(millis)).await;
    socket.advance(&mut socket.state.lock().unwrap());
}

fn reply(id: &str) -> RealtimeCommand {
    RealtimeCommand::GenerateReply(GenerateReply {
        req_id: id.into(),
        instructions: None,
        tool_choice: None,
    })
}

#[test]
fn cancelled_before_creation_uses_production_correlation_and_suppresses_late_audio() {
    simulate(async {
        let (socket, _records) =
            socket(RealtimeDelays { created: Some(1_000), ..Default::default() });
        command(&socket, reply("cancelled"));
        command(&socket, RealtimeCommand::CancelReply(CancelReply { req_id: "cancelled".into() }));
        advance(&socket, 1_001).await;
        advance(&socket, 20_000).await;
        let st = socket.state.lock().unwrap();
        assert!(
            st.events
                .iter()
                .any(|e| matches!(e, Event::GenerateReplyFailed(f) if f.req_id == "cancelled"))
        );
        assert!(
            !st.events
                .iter()
                .any(|e| matches!(e, Event::GenerationCreated(_) | Event::AudioDelta(_)))
        );
        assert!(st.generations.is_empty(), "the simulated server settled its cancelled response");
        assert!(!st.machine.has_active_generation());
    });
}

#[test]
fn production_reply_deadline_cancels_a_response_the_server_creates_late() {
    simulate(async {
        let (socket, _records) =
            socket(RealtimeDelays { created: Some(11_000), ..Default::default() });
        command(&socket, reply("late"));
        advance(&socket, 10_001).await;
        assert!(
            socket
                .state
                .lock()
                .unwrap()
                .events
                .iter()
                .any(|e| matches!(e, Event::GenerateReplyFailed(f) if f.req_id == "late"))
        );
        advance(&socket, 1_000).await;
        advance(&socket, 20_000).await;
        let st = socket.state.lock().unwrap();
        assert!(
            !st.events
                .iter()
                .any(|e| matches!(e, Event::GenerationCreated(_) | Event::AudioDelta(_)))
        );
        assert!(st.generations.is_empty());
        assert!(!st.machine.has_active_generation());
    });
}

#[test]
fn reconnect_replays_unacknowledged_context_through_the_production_machine() {
    simulate(async {
        let (socket, _records) =
            socket(RealtimeDelays { context_ack: Some(100), ..Default::default() });
        let item =
            ChatItem::Message(ChatMessage::text("kept", Role::User, "hello", socket.clock.wall()));
        command(
            &socket,
            RealtimeCommand::UpdateChatCtx(UpdateChatCtx {
                chat_ctx: ChatContext::new(vec![item]),
            }),
        );
        {
            let st = socket.state.lock().unwrap();
            assert!(st.server.holds("kept"), "write applied before acknowledgement");
            assert!(st.machine.chat_ctx().items.is_empty(), "client has not been told yet");
        }
        assert!(deliver(
            &socket,
            &Input::RealtimeEvent {
                socket: crate::input::Ref::new(0),
                event: RtEvent::Disconnected
            }
        ));
        assert!(deliver(
            &socket,
            &Input::RealtimeEvent {
                socket: crate::input::Ref::new(0),
                event: RtEvent::Reconnected
            }
        ));
        advance(&socket, 101).await;
        assert_eq!(socket.state.lock().unwrap().machine.chat_ctx().items[0].id(), "kept");
        command(
            &socket,
            RealtimeCommand::UpdateChatCtx(UpdateChatCtx { chat_ctx: ChatContext::new(vec![]) }),
        );
        {
            let st = socket.state.lock().unwrap();
            assert!(!st.server.holds("kept"));
            assert_eq!(
                st.machine.chat_ctx().items.len(),
                1,
                "deletion acknowledgement is delayed too"
            );
        }
        advance(&socket, 101).await;
        assert!(socket.state.lock().unwrap().machine.chat_ctx().items.is_empty());
    });
}

#[test]
fn server_refuses_invalid_history_writes_with_the_original_event_identity() {
    let mut server = Server::new(7, "server".into());
    let item = ChatItem::Message(ChatMessage::text(
        "one",
        Role::User,
        "hello",
        std::time::SystemTime::UNIX_EPOCH,
    ));
    server.create_request(Ts::ZERO, "missing-parent".into(), &item, Some("missing".into()));
    assert!(!server.holds("one"));
    server.create_request(Ts::ZERO, "accepted".into(), &item, Some("root".into()));
    server.create_request(Ts::ZERO, "duplicate".into(), &item, None);
    assert_eq!(server.ids(), ["one"]);
    let now = Ts::ZERO.saturating_add(Duration::from_secs(60));
    let mut rejected = Vec::new();
    while let Some(happening) = server.pop(now) {
        if let server::Happening::Error { request, code, .. } = happening {
            rejected.push((request.expect("correlated rejection"), code.expect("error code")));
        }
    }
    assert_eq!(
        rejected,
        [
            ("missing-parent".into(), "item_create_invalid_previous_item_id"),
            ("duplicate".into(), "item_create_duplicate_item_id")
        ]
    );
}

/// A failed ASR job and a lost connection preserve the same accepted partial both in caller
/// events and in the model's replayed conversation, rather than saving only one of those views.
#[test]
fn a_failed_transcript_is_replayed_as_the_received_partial() {
    simulate(async {
        let (socket, _records) = socket(RealtimeDelays {
            transcription: Some(20_000),
            transcription_partial: Some(100),
            ..Default::default()
        });
        let frame = lk_agents_abi::AudioFrame::new(vec![0; 14_400], 24_000, 1, 7_200);
        command(&socket, RealtimeCommand::PushAudio(lk_agents_abi::PushAudio { frame }));
        command(
            &socket,
            RealtimeCommand::CommitAudio(lk_agents_abi::llm::realtime::CommitAudio {}),
        );
        advance(&socket, 500).await;
        assert!(deliver(
            &socket,
            &Input::RealtimeEvent {
                socket: crate::input::Ref::new(0),
                event: RtEvent::TranscriptionFailed
            }
        ));
        socket.advance(&mut socket.state.lock().unwrap());
        let (id, text) = {
            let st = socket.state.lock().unwrap();
            let final_ = st
                .events
                .iter()
                .find_map(|e| match e {
                    Event::InputTranscription(t) if t.is_final => Some(t),
                    _ => None,
                })
                .expect("the failed stream settles its partial");
            assert_eq!(
                st.machine.chat_ctx().items[0].as_message().unwrap().raw_text_content().as_deref(),
                Some(final_.transcript.as_str())
            );
            (final_.item_id.clone(), final_.transcript.clone())
        };
        assert!(deliver(
            &socket,
            &Input::RealtimeEvent {
                socket: crate::input::Ref::new(0),
                event: RtEvent::Disconnected
            }
        ));
        assert!(deliver(
            &socket,
            &Input::RealtimeEvent {
                socket: crate::input::Ref::new(0),
                event: RtEvent::Reconnected
            }
        ));
        advance(&socket, 1_000).await;
        let st = socket.state.lock().unwrap();
        let context = st.machine.chat_ctx();
        assert_eq!(context.items.len(), 1);
        assert_eq!(context.items[0].id(), id);
        assert_eq!(
            context.items[0].as_message().unwrap().raw_text_content().as_deref(),
            Some(text.as_str())
        );
    });
}

#[test]
fn an_invalid_named_tool_is_accepted_then_fails_without_poisoning_the_next_reply() {
    simulate(async {
        let (socket, _records) = socket(Default::default());
        command(
            &socket,
            RealtimeCommand::GenerateReply(GenerateReply {
                req_id: "invalid-tool".into(),
                instructions: None,
                tool_choice: Some(lk_agents_abi::ToolChoice::Named("not_registered".into())),
            }),
        );
        advance(&socket, 1_000).await;
        {
            let st = socket.state.lock().unwrap();
            assert!(st.events.iter().any(|e| matches!(e, Event::GenerationCreated(_))));
            assert!(st.events.iter().any(|e| matches!(e, Event::GenerationDone(d) if d.status == lk_agents_abi::llm::realtime::GenerationStatus::Failed)));
            assert!(st.events.iter().any(|e| matches!(e, Event::Error(e) if e.code.as_deref() == Some("invalid_tool_choice") && e.recoverable)));
            assert!(!st.machine.has_active_generation());
        }
        command(&socket, reply("valid"));
        advance(&socket, 1_000).await;
        advance(&socket, 20_000).await;
        let st = socket.state.lock().unwrap();
        assert!(st.events.iter().any(|e| matches!(e, Event::GenerationDone(d) if d.status == lk_agents_abi::llm::realtime::GenerationStatus::Completed)));
        assert!(!st.machine.has_active_generation());
    });
}

#[test]
fn imported_long_item_ids_are_aliased_on_the_server_and_unchanged_in_client_history() {
    simulate(async {
        let (socket, _records) = socket(Default::default());
        let id = "12345678-1234-1234-1234-123456789abc";
        command(
            &socket,
            RealtimeCommand::UpdateChatCtx(UpdateChatCtx {
                chat_ctx: ChatContext::new(vec![ChatItem::Message(ChatMessage::text(
                    id,
                    Role::User,
                    "book Tuesday",
                    socket.clock.wall(),
                ))]),
            }),
        );
        advance(&socket, 1_000).await;
        {
            let st = socket.state.lock().unwrap();
            assert_eq!(st.machine.chat_ctx().items[0].id(), id);
            assert_eq!(st.server.ids().len(), 1);
            assert_eq!(st.server.ids()[0].len(), 32);
            assert_ne!(st.server.ids()[0], id);
            assert!(!st.events.iter().any(|e| matches!(e, Event::Error(_))));
        }
        command(
            &socket,
            RealtimeCommand::UpdateChatCtx(UpdateChatCtx { chat_ctx: ChatContext::default() }),
        );
        advance(&socket, 1_000).await;
        let st = socket.state.lock().unwrap();
        assert!(st.server.ids().is_empty(), "deletion uses the same alias as creation");
        assert!(st.machine.chat_ctx().items.is_empty());
    });
}

/// These errors are sent by the server even when the client treats one as a harmless race.
#[test]
fn cancel_and_empty_commit_refusals_reach_the_real_client() {
    let mut server = Server::new(1, String::new());
    server.cancel(Ts::ZERO, None);
    server.cancel(Ts::ZERO, Some("unknown"));
    server.commit_audio(Ts::ZERO);
    let mut codes = Vec::new();
    while let Some(h) = server.pop(Ts::ZERO.saturating_add(Duration::from_secs(60))) {
        if let server::Happening::Error { code, .. } = h {
            codes.push(code.unwrap());
        }
    }
    assert_eq!(
        codes,
        [
            "response_cancel_not_active",
            "response_cancel_not_found",
            "input_audio_buffer_commit_empty"
        ]
    );
}

#[test]
fn a_busy_response_rejection_settles_only_the_second_request() {
    simulate(async {
        let (socket, _records) = socket(Default::default());
        command(&socket, reply("first"));
        command(&socket, reply("second"));
        advance(&socket, 1_000).await;
        {
            let st = socket.state.lock().unwrap();
            assert!(
                st.events
                    .iter()
                    .any(|e| matches!(e, Event::GenerateReplyFailed(f) if f.req_id == "second"))
            );
            assert!(
                !st.events
                    .iter()
                    .any(|e| matches!(e, Event::GenerateReplyFailed(f) if f.req_id == "first"))
            );
            assert!(st.machine.has_active_generation());
        }
        advance(&socket, 20_000).await;
        let st = socket.state.lock().unwrap();
        assert_eq!(st.events.iter().filter(|e| matches!(e, Event::GenerationDone(d) if d.status == lk_agents_abi::llm::realtime::GenerationStatus::Completed)).count(), 1);
        assert!(!st.machine.has_active_generation());
    });
}
