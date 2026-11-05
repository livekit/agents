//! Conversation regression tests.

mod agent_state_written_on_connect {
    //! The agent state reaches the room: written by the session as it changes, and a state taken
    //! before the room connected written once it does.
    use std::sync::Arc;
    use std::time::Duration;

    use lk_agents_abi::ATTRIBUTE_AGENT_STATE;
    use lk_agents_abi::voice::room_owner::RoomIo as _;
    use lk_agents_abi::voice::rtc::{ConnectionState, RtcRoom};
    use lk_agents_fuzz::fakes::room::FakeRtc;
    use lk_agents_runtime::net::task::TokioSpawner;
    use lk_agents_runtime::room_io::RoomIo;
    use lk_agents_voice::room_io::RoomIoConfig;

    // Catches: `lk.agent.state` written only while the room is connected, and the session started
    // alongside the job's `connect()`: the `listening` it takes at start was dropped, so until the
    // agent first spoke a frontend (or a caller waiting for the agent to be ready) saw no state.
    #[tokio::test]
    async fn a_state_taken_before_the_connect_is_written_on_it() {
        let (rtc, clock) =
            FakeRtc::with_virtual_clock(lk_agents_runtime::rtc::events::RoomEventBus::port);
        let room = rtc.room("agent-state-on-connect");
        room.set_connection_state(ConnectionState::Disconnected);
        let spawner = Arc::new(TokioSpawner::new(tokio::runtime::Handle::current()));
        let ids =
            Arc::new(lk_agents_runtime::room_io::output::transcription::SegmentIds::default());
        let config = RoomIoConfig::default();
        let rtc = room.clone() as Arc<dyn RtcRoom>;
        let io = RoomIo::new(rtc, clock, spawner, ids, config, None, None);
        io.clone().start(lk_agents_abi::telemetry::Cx::NONE);
        io.clone().on_agent_state_changed("listening");
        tokio::time::sleep(Duration::from_millis(50)).await;
        room.set_connection_state(ConnectionState::Connected);
        tokio::time::sleep(Duration::from_millis(50)).await;
        let state = room.local_participant().attributes().get(ATTRIBUTE_AGENT_STATE).cloned();
        assert_eq!(state.as_deref(), Some("listening"));
        assert_eq!(io.aclose().await, lk_agents_abi::Closed::Clean);
    }

    // Catches: `RoomIo::on_agent_state_changed` with no caller, so the agent never writes
    // `lk.agent.state` and a frontend keyed on it shows the agent stuck.
    #[tokio::test(start_paused = true)]
    async fn the_session_writes_the_agent_state_to_the_room_as_it_changes() {
        use lk_agents_abi::voice::session::{VoiceSession as _, VoiceSessionArgs};
        use lk_agents_runtime::room_io::FixedRoom;
        use lk_agents_runtime::voice::session::{SessionRunner, Timebase};
        use lk_agents_runtime::voice::{Executor, VoiceSession};
        use lk_agents_voice::agent_session::{Command, HookOutcome, VoiceHostEvent};

        let (rtc, clock) =
            FakeRtc::with_virtual_clock(lk_agents_runtime::rtc::events::RoomEventBus::port);
        let room = rtc.room("agent-state");
        let fixed =
            FixedRoom { rtc: Some(room.clone()), time: Timebase::from_virtual(clock), pool: None };
        let args = VoiceSessionArgs {
            agent: "agent".into(),
            room_io_config: Default::default(),
            ..Default::default()
        };
        let env = lk_agents_runtime::net::env::ProcessEnv::current_runtime();
        let session = VoiceSession::new(&env, Arc::new(fixed), args, None).expect("a valid room");
        let opening = session
            .open(
                Box::new(Executor::new(session.time().clone(), session.spawner())),
                None,
                move |mut runner: SessionRunner| {
                    Box::pin(async move {
                        while let Some(event) = runner.next_host_event().await {
                            if let VoiceHostEvent::Hook(call) = event {
                                let outcome = HookOutcome::Returned;
                                let _ =
                                    runner.api().send(Command::HookDone { id: call.id(), outcome });
                            }
                        }
                    })
                },
            )
            .expect("a ready session opens");
        opening.await.expect("the loop starts");
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }
        let local = room.local_participant();
        let written: Vec<String> = local
            .attribute_calls()
            .iter()
            .filter_map(|attributes| attributes.get(ATTRIBUTE_AGENT_STATE).cloned())
            .collect();
        assert!(!written.is_empty(), "no `lk.agent.state` write reached the room");
        assert_eq!(
            local.attributes().get(ATTRIBUTE_AGENT_STATE).map(String::as_str),
            Some("listening"),
            "the room's attribute is not the agent's current state (writes: {written:?})"
        );
    }
}

mod stored_turn_keeps_the_history {
    //! A turn that takes the voice lane keeps the session's history whole, whatever its agent reads.
    use lk_agents_abi::ChatItem;
    use lk_agents_core::tts::markup::ChatMessageExt;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, Role, UserText};
    use lk_agents_voice::events::SessionEvent;

    fn reply(chat: u32, said: &str) -> [Input; 9] {
        [
            Input::GenerateReply {
                user_input: Some(UserText("hello?".into())),
                instructions: None,
                allow: None,
            },
            Input::Advance { ms: Millis(20) },
            Input::LlmText { chat: Ref::new(chat), text: AgentText(said.into()) },
            Input::LlmEnd { chat: Ref::new(chat) },
            Input::Advance { ms: Millis(20) },
            Input::PlayoutStarted { segment: Ref::new(chat) },
            Input::PlayoutProgressed { segment: Ref::new(chat), ms: Millis(5_000) },
            Input::PlayoutFinished { segment: Ref::new(chat) },
            Input::Advance { ms: Millis(1_000) },
        ]
    }

    // Catches: the session's history was replaced by the lane storage gave back for a turn's
    // request, the active agent's own view. Items that view does not hold (a question an earlier
    // agent left waiting, what `update_chat_ctx` dropped from the view) left the history, so the
    // person's answer never reached the waiting call and the next commit wrote the lane without them
    // (`tool_output_follows_outcome`, survey seeds 13386, 15374).
    #[test]
    fn what_the_agent_does_not_read_stays_in_the_history() {
        // A store that keeps its lease for the whole call.
        let setup = Setup { storage_lost_at: Some(1_000), ..Setup::default() };
        // The agent's view becomes one note of the developer's; its first exchange stays in history.
        let dropped = Input::UpdateChatCtx {
            items: vec![Role::Assistant],
            text: AgentText("The office is closed on Monday.".into()),
        };
        let inputs: Vec<Input> = reply(0, "Hi, how can I help?")
            .into_iter()
            .chain([dropped])
            .chain(reply(1, "Sure."))
            .collect();
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        // The history as the host holds it: told item by item, replaced whole by a reload.
        let mut history: Vec<ChatItem> = Vec::new();
        for (_, event) in run.record.host_view().events() {
            match event {
                SessionEvent::HistoryReloaded { items } => history = items.clone(),
                SessionEvent::ConversationItemAdded { item, .. } => {
                    match history.iter_mut().find(|i| i.id() == item.id()) {
                        Some(told) => *told = (**item).clone(),
                        None => history.push((**item).clone()),
                    }
                }
                _ => {}
            }
        }
        let said = |text: &str| {
            history
                .iter()
                .any(|i| i.as_message().is_some_and(|m| m.text_content().as_deref() == Some(text)))
        };
        assert!(said("Hi, how can I help?"), "the first reply left the history: {history:#?}");
        assert!(said("Sure."), "the second reply is in the history: {history:#?}");
    }
}

mod turn_hook_sees_agent_context {
    //! `on_user_turn_completed` edits a copy of the driving agent's own context, not the session's
    //! history of every agent, and a second turn waits for the first turn's hook.
    use lk_agents_abi::{ChatItem, Role};
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, HookAnswer, Input, Millis, Ref, TurnMode, UserText};
    use lk_agents_fuzz::record::{Host, Kind};
    use lk_agents_voice::agent_session::HookKind;
    use lk_agents_voice::events::SessionEvent;

    fn turn(utterance: u32, words: &str, skip_reply: bool) -> Vec<Input> {
        vec![
            Input::SttStart { stt: Ref::new(0) },
            Input::SttFinal {
                utterance: Ref::new(utterance),
                text: UserText(words.into()),
                speaker: None,
            },
            Input::SttEnd { utterance: Ref::new(utterance) },
            Input::Advance { ms: Millis(20) },
            Input::CommitUserTurn {
                transcript_timeout: Millis(100),
                stt_flush: Millis(0),
                skip_reply,
            },
            Input::Advance { ms: Millis(1_000) },
        ]
    }

    /// Every hook still waited on, answered as returning nothing.
    async fn answer_hooks(run: &mut Run) {
        for nth in run.record.live().of(Kind::Hook) {
            let answer = Input::HookDone {
                hook: Ref::new(nth),
                answer: HookAnswer::Returned,
                text: AgentText(String::new()),
            };
            run.apply(&answer).await;
        }
        run.apply(&Input::Advance { ms: Millis(100) }).await;
    }

    fn turn_hooks(run: &Run) -> Vec<Vec<String>> {
        run.record
            .host()
            .iter()
            .filter_map(|(_, h)| match h {
                Host::Hook { kind: HookKind::OnUserTurnCompleted, ctx_ids, .. } => {
                    Some(ctx_ids.clone())
                }
                _ => None,
            })
            .collect()
    }

    // Catches: the host built the hook's `turn_ctx` from `session.history` (every agent's items,
    // the switch's records included) and sent all of it back as the reply's context, so a RAG edit
    // after a handoff put the session's records into the new agent's prompt.
    #[test]
    fn the_turn_hook_gets_the_agents_own_context() {
        let setup = Setup {
            stt: true,
            turn: TurnMode::Manual,
            hook_bound_ms: Some(1_000),
            ..Setup::default()
        };
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            answer_hooks(&mut run).await;
            for input in turn(0, "first agent hears this", true) {
                run.apply(&input).await;
            }
            answer_hooks(&mut run).await;
            let handoff = Input::Handoff {
                agent: "next".into(),
                instructions: AgentText("You are next.".into()),
                tools: vec![],
                turn: None,
            };
            run.apply(&handoff).await;
            run.apply(&Input::Advance { ms: Millis(100) }).await;
            answer_hooks(&mut run).await;
            for input in turn(1, "second agent hears this", false) {
                run.apply(&input).await;
            }
            run
        });
        let mut user = Vec::new();
        let mut records = Vec::new();
        for (_, ev) in run.record.host_view().events() {
            if let SessionEvent::ConversationItemAdded { item, .. } = ev {
                match &**item {
                    ChatItem::Message(m) if m.role == Role::User => user.push(m.id.clone()),
                    ChatItem::AgentHandoff(_) => records.push(item.id().to_owned()),
                    _ => {}
                }
            }
        }
        assert!(!user.is_empty(), "the first turn was committed");
        let hooks = turn_hooks(&run);
        assert_eq!(hooks.len(), 1, "the first turn asks for no reply, so no hook");
        // The switch carries the conversation forward, so the first turn is in the agent's view; the
        // session's record of the switch is history, not the agent's context.
        assert!(hooks[0].contains(&user[0]), "the carried conversation: {:?}", hooks[0]);
        assert!(!records.is_empty(), "the switch was recorded");
        assert!(
            records.iter().all(|r| !hooks[0].contains(r)),
            "the hook was handed the session's handoff record: {:?}",
            hooks[0]
        );
    }

    // Catches: a second turn decided while the first turn's hook ran ended that wait with no edits
    // and answered the first turn anyway; the reference awaits the old task ("we never cancel user
    // code").
    #[test]
    fn a_second_turn_waits_for_the_first_turns_hook() {
        let setup = Setup {
            stt: true,
            turn: TurnMode::Manual,
            hook_bound_ms: Some(1_000),
            ..Setup::default()
        };
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            answer_hooks(&mut run).await;
            for input in turn(0, "first words", false) {
                run.apply(&input).await;
            }
            for input in turn(1, "second words", false) {
                run.apply(&input).await;
            }
            run.apply(&Input::Advance { ms: Millis(20_000) }).await;
            run
        });
        assert_eq!(turn_hooks(&run).len(), 1, "the second turn's hook waits for the first's");
        let replies = run.record.host_view().speeches().len();
        assert_eq!(replies, 0, "no turn is answered while the first hook runs");
    }
}
