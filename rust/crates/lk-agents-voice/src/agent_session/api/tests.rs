//! Session decisions and host hook outcomes, without a voice executor.

mod recorder;

use super::*;
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::commands::{HookId, HookOutcome};
use crate::agent_session::host_events::HookCall;
use crate::agent_session::{OpeningLegs, QueueOwner, SessionCores, SessionExecutor};
use crate::events::SessionEvent;
use crate::test_support::FixedRoom;
use crate::tts::markup::ChatMessageExt;
use lk_agents_abi::errors::ProviderError;

/// Builds the cores of a session with no STT or VAD; nothing here runs them.
struct NoLegs;

impl SessionExecutor for NoLegs {
    fn tts_adapter(&self) -> crate::agent_session::TtsAdapter {
        Arc::new(|tts, _, _| tts)
    }

    fn stt_adapter(&self) -> crate::agent_session::SttAdapter {
        Arc::new(|stt, _, _| stt)
    }

    fn open_legs<'a>(
        &'a mut self,
        _: &'a lk_agents_abi::resources::Resources,
        stt: Option<Arc<dyn lk_agents_abi::Stt>>,
        vad: Option<Arc<dyn lk_agents_abi::Vad>>,
    ) -> lk_agents_abi::BoxFuture<'a, Result<OpeningLegs, DriverError>> {
        assert!(stt.is_none() && vad.is_none(), "these sessions open no streams");
        Box::pin(async { Ok(OpeningLegs::default()) })
    }

    fn open_realtime<'a>(
        &'a mut self,
        _: &'a lk_agents_abi::resources::Resources,
        _: Arc<dyn lk_agents_abi::Realtime>,
        _: bool,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            (Arc<dyn lk_agents_abi::RealtimeSession>, lk_agents_abi::resources::ResourceId),
            DriverError,
        >,
    > {
        unreachable!("these sessions open no realtime session")
    }

    fn own(self: Box<Self>, _: SessionCores) -> Box<dyn QueueOwner> {
        unreachable!("the tests step the cores themselves")
    }
}

#[tokio::test(start_paused = true)]
async fn realtime_transport_timeout_keeps_its_phase_and_request_identity_on_session_error() {
    use crate::agent_activity::realtime::RealtimeWake;
    use crate::agent_session::effects::Effect;
    use lk_agents_abi::Ts;
    use lk_agents_abi::errors::{ApiError, ApiErrorKind, TimeoutPhase};
    use lk_agents_abi::llm::realtime as rt;

    let api = unobserved(fake_room("realtime-timeout-identity").1, None);
    let Stage::Ready(ready) = std::mem::replace(&mut *api.lock(), Stage::Closed(None)) else {
        panic!("session must be ready");
    };
    let (mut cores, _) = ready.builder.start(&mut NoLegs).await.unwrap();
    let driver = &mut cores.driver;
    let mut out = driver.out();
    let mut timeout = ApiError::timed_out(TimeoutPhase::Connect);
    timeout.request_id = Some("handshake-7".into());
    driver.on_realtime(
        Ts::ZERO,
        RealtimeWake::Event(rt::Event::Error(rt::RealtimeModelError {
            timestamp: Ts::ZERO,
            label: "realtime".into(),
            message: timeout.message.clone(),
            recoverable: true,
            req_id: None,
            code: None,
            provider_error: Some(timeout),
        })),
        &mut out,
    );
    let effects = driver.settle_step(out);
    let error = effects
        .into_iter()
        .find_map(|effect| match effect {
            Effect::Emit(SessionEvent::Error(error)) => Some(error),
            _ => None,
        })
        .expect("the session did not report the transport error");
    let ProviderError::Local(error) = error.error else {
        panic!("a native transport error changed to a host error");
    };
    assert_eq!(error.kind, ApiErrorKind::Timeout);
    assert_eq!(error.phase, Some(TimeoutPhase::Connect));
    assert_eq!(error.request_id.as_deref(), Some("handshake-7"));
}

#[tokio::test(start_paused = true)]
async fn late_realtime_transcript_keeps_its_source_without_starting_another_speech() {
    use crate::agent_activity::realtime::RealtimeWake;
    use crate::agent_session::effects::{Effect, Planned};
    use lk_agents_abi::llm::realtime as rt;
    use lk_agents_abi::{ChatMessage, Role, Ts};

    let api = unobserved(fake_room("late-realtime-transcript").1, None);
    let Stage::Ready(ready) = std::mem::replace(&mut *api.lock(), Stage::Closed(None)) else {
        panic!("session must be ready");
    };
    let (mut cores, _) = ready.builder.start(&mut NoLegs).await.unwrap();
    let driver = &mut cores.driver;
    let mut out = driver.out();
    let source =
        ChatMessage::text("old-response", Role::Assistant, "Already heard", api.time().read().wall);
    let source_time = source.created_at;
    driver.commit_item(
        source.into(),
        crate::agent::Source::Remote { item_id: "old-response".into() },
        crate::agent::Visibility::Both,
        &mut out,
    );
    driver.on_realtime(
        Ts::ZERO,
        RealtimeWake::Event(rt::Event::GenerationCreated(rt::GenerationCreated {
            response_id: "new-response".into(),
            req_id: None,
        })),
        &mut out,
    );
    let players = driver.activity.realtime_players.clone();
    out.take();
    for event in [
        rt::Event::TranscriptAttribution(rt::TranscriptAttribution {
            response_id: "late-transcript".into(),
            source_response_id: "old-response".into(),
        }),
        rt::Event::GenerationCreated(rt::GenerationCreated {
            response_id: "late-transcript".into(),
            req_id: None,
        }),
        rt::Event::MessageStarted(rt::MessageStarted {
            message_id: "late-transcript".into(),
            response_id: Some("late-transcript".into()),
        }),
        rt::Event::TextDelta(rt::TextDelta {
            message_id: "late-transcript".into(),
            response_id: Some("late-transcript".into()),
            delta: "A delayed transcription".into(),
            start_time: None,
        }),
        rt::Event::MessageDone(rt::MessageDone {
            message_id: "late-transcript".into(),
            response_id: Some("late-transcript".into()),
        }),
        rt::Event::GenerationDone(rt::GenerationDone {
            response_id: "late-transcript".into(),
            status: rt::GenerationStatus::Completed,
            usage: Default::default(),
        }),
    ] {
        driver.on_realtime(Ts::ZERO, RealtimeWake::Event(event), &mut out);
    }
    assert_eq!(driver.activity.realtime_players, players, "a correction took the voice floor");
    assert!(
        !out.take().iter().any(|effect| matches!(
            effect,
            Planned::Effect(Effect::RealtimePlayer { .. } | Effect::SpeechStart { .. })
        )),
        "a correction was sent through speech playback"
    );
    let correction =
        &driver.session.chat_ctx.entry("late-transcript").expect("late words were lost").item;
    let correction = correction.as_message().unwrap();
    assert_eq!(correction.text_content().as_deref(), Some("A delayed transcription"));
    assert_eq!(
        correction.metrics.provider_request_ids.as_deref(),
        Some(["old-response".to_owned()].as_slice())
    );
    assert_eq!(correction.created_at, source_time);
}

// Catches (voice-session.md "releases the lane within 30 seconds"): a speech whose tasks were
// aborted (the interruption watchdog, `cancel_tasks`) reports neither done nor failed, so its
// lane was never released and the store renewed its lease until the call closed.
#[tokio::test(start_paused = true)]
async fn a_speech_ended_without_its_turn_releases_the_lane_within_30_s() {
    use crate::agent_session::effects::Effect;
    use crate::agent_session::storage::LANE_RELEASE_BOUND;
    use crate::agent_session::time::Now;
    use lk_agents_abi::Ts;

    let api = unobserved(fake_room("aborted-speech-lane").1, None);
    let Stage::Ready(ready) = std::mem::replace(&mut *api.lock(), Stage::Closed(None)) else {
        panic!("session must be ready");
    };
    let (mut cores, _) = ready.builder.start(&mut NoLegs).await.unwrap();
    let driver = &mut cores.driver;
    driver.session.storage.enabled = true;
    let speech = SpeechId(7);
    driver.storage_started(speech);
    let mut out = driver.out();
    driver.release_speech(Ts::ZERO, speech, &mut out);
    let released = driver.settle_step(out);
    assert!(!released.iter().any(|e| matches!(e, Effect::StorageAbort(_))));
    let wall = std::time::UNIX_EPOCH;
    let due = Ts::ZERO.saturating_add(LANE_RELEASE_BOUND);
    assert_eq!(driver.wake_at(), Some(due), "the release is bounded");
    let effects = driver.handle(None, Now { at: due, wall });
    assert!(
        effects.iter().any(|e| matches!(e, Effect::StorageAbort(s) if *s == speech)),
        "the lane was never released: {effects:?}"
    );
}

/// A fake room on a virtual clock, and the same room as a session's.
fn fake_room(name: &str) -> (Arc<lk_agents_test_support::rtc::FakeRoom>, FixedRoom) {
    let (rtc, clock) = lk_agents_test_support::rtc::FakeRtc::with_virtual_clock(
        crate::test_support::unread_room_events,
    );
    let room = rtc.room(name);
    // The `FakeRtc` is the room's owner; the room keeps it alive through its own `Arc`.
    std::mem::forget(rtc);
    (room.clone(), FixedRoom { rtc: room, time: Timebase::from_virtual(clock) })
}

fn unobserved(room: FixedRoom, hook_bound: Option<f64>) -> SessionApi {
    assembled(room, hook_bound, None)
}

/// [`unobserved`], with the host's `SessionOptions` object when given.
fn assembled(
    room: FixedRoom,
    hook_bound: Option<f64>,
    options: Option<&crate::agent_activity::options::HostOptions>,
) -> SessionApi {
    let spawner = Arc::new(lk_agents_abi::TokioSpawner::new(tokio::runtime::Handle::current()));
    #[rustfmt::skip]
    let api = SessionApi::host_new(
        Arc::new(room), None, None, "agent".into(), None, None, None, None, None, None, None,
        None, None, None, None, true, 24_000, 24_000, options, None, None, None, None, None, true,
        Some(crate::room_io::RoomIoConfig::default()), None, None, None, None, None, None,
        None, None,
        None, None, None, None, false, None, false, hook_bound, 3, None, None, None, Maybe::NotGiven, None,
        None, None, None, spawner,
    );
    api.expect("valid native room ports")
}

// Catches: a second session in one job recording, or hosting the remote session, or a requested
// recording on it silently dropped instead of refused.
#[test]
fn only_the_primary_session_records_or_hosts() {
    let plan = |requested, primary_taken, text_only| {
        plan_session_start(requested, true, true, primary_taken, None, text_only)
    };
    let first = plan(None, false, false).unwrap();
    assert!(first.is_primary && first.hosting && first.records && first.recording.audio);
    assert!(!plan(None, false, true).unwrap().recording.audio, "a text-only session records audio");
    let second = plan(None, true, false).unwrap();
    assert!(!second.is_primary && !second.hosting && !second.records);
    assert!(plan(Some(RecordingRequest::All(true)), true, false).is_err());
    assert!(plan(Some(RecordingRequest::All(false)), true, false).is_ok());
    let declined = plan_session_start(None, true, true, false, Some(false), false).unwrap();
    assert!(declined.is_primary && !declined.hosting);
}

struct NoRegistry;

impl lk_agents_abi::RefRegistry for NoRegistry {
    fn release(&self, _id: u64) {}
}

fn host_ref(id: u64) -> lk_agents_abi::HostRef {
    let registry: Arc<dyn lk_agents_abi::RefRegistry> = Arc::new(NoRegistry);
    lk_agents_abi::HostRef::new(id, &registry)
}

/// Hooks that answer as configured, and record what failed.
#[derive(Default)]
struct ScriptedHooks {
    raise: bool,
    user_text: Option<String>,
    stop: bool,
    #[allow(clippy::disallowed_types, reason = "S0: test harness runtime resource")]
    failed: Mutex<Vec<String>>,
    #[allow(clippy::disallowed_types, reason = "S0: test harness runtime resource")]
    toolsets_closed: Mutex<usize>,
}

impl ScriptedHooks {
    fn result<T>(&self, value: T) -> Result<T, HostError> {
        if self.raise {
            Err(HostError {
                kind: "ValueError".into(),
                message: "the hook raised".into(),
                host_ref: None,
            })
        } else {
            Ok(value)
        }
    }
}

impl SessionHooks for ScriptedHooks {
    fn on_enter<'a>(
        &'a self,
        _: &'a lk_agents_abi::HostRef,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async move { self.result(()) })
    }
    fn on_exit<'a>(
        &'a self,
        _: &'a lk_agents_abi::HostRef,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async move { self.result(()) })
    }
    fn close_toolsets<'a>(
        &'a self,
        _: Option<&'a lk_agents_abi::HostRef>,
    ) -> lk_agents_abi::BoxFuture<'a, ()> {
        Box::pin(async move {
            *self.toolsets_closed.lock().unwrap() += 1;
        })
    }
    fn setup_tools<'a>(
        &'a self,
        _: &'a lk_agents_abi::HostRef,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Vec<lk_agents_abi::ToolSpec>, HostError>> {
        Box::pin(async move { self.result(Vec::new()) })
    }
    fn closing(&self) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async move { self.result(()) })
    }
    fn on_user_turn_completed<'a>(
        &'a self,
        _: &'a lk_agents_abi::HostRef,
        _: Vec<lk_agents_abi::ChatItem>,
        _: &'a str,
        _: f64,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<(Option<String>, Option<Vec<lk_agents_abi::ChatItem>>, bool), HostError>,
    > {
        Box::pin(async move { self.result((self.user_text.clone(), Some(Vec::new()), self.stop)) })
    }
    fn on_user_turn_exceeded<'a>(
        &'a self,
        _: &'a lk_agents_abi::HostRef,
        _: &'a lk_agents_abi::voice::events::UserTurnExceededEvent,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async move { self.result(()) })
    }
    fn function_tools_executed(
        &self,
        _: Vec<lk_agents_abi::ChatItem>,
        outputs: Vec<lk_agents_abi::ChatItem>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Vec<bool>, HostError>> {
        Box::pin(async move { self.result(vec![false; outputs.len()]) })
    }
    fn filler_source<'a>(
        &'a self,
        _: &'a lk_agents_abi::HostRef,
        step: u32,
    ) -> lk_agents_abi::BoxFuture<'a, Result<FillerReply, HostError>> {
        Box::pin(async move {
            self.result(match step {
                0 => FillerReply::Text("one moment".into()),
                1 => FillerReply::Spoke,
                _ => FillerReply::Declined,
            })
        })
    }
    fn hook_failed(&self, hook: &str, _: Option<lk_agents_abi::errors::CapturedError>) {
        self.failed.lock().unwrap().push(hook.to_owned());
    }
}

#[allow(clippy::disallowed_methods, reason = "S0: test harness runtime resource")]
/// What `answer_hook` sends back for `call`.
async fn answered(hooks: &Arc<ScriptedHooks>, call: HookCall) -> HookOutcome {
    use crate::agent_session::commands::Sent;
    use crate::agent_session::driver::{Event, Input, Tx};
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let agents = HostAgents::default();
    agents.install("agent", Some(host_ref(1)));
    let spawner = Arc::new(lk_agents_abi::TokioSpawner::new(tokio::runtime::Handle::current()));
    let hooks: Arc<dyn SessionHooks> = hooks.clone();
    super::answer_hook(hooks, call, agents, Tx::new(tx), spawner).await;
    match rx.recv().await {
        Some(Event::Input(Input::Host(Sent {
            command: Command::HookDone { outcome, .. },
            ..
        }))) => outcome,
        _ => panic!("the hook was not answered"),
    }
}

// Catches: a hook's answer translated wrongly for the driver: a `StopResponse` lost, an unchanged
// transcript sent as an edit, an `on_exit` that leaves the agent's toolsets open, a filler's
// reply misread, or a raising hook not reported as failed.
#[tokio::test]
async fn hook_answers_reach_the_driver_as_the_host_gave_them() {
    use crate::agent::TurnEdit;
    let hooks = Arc::new(ScriptedHooks {
        stop: true,
        user_text: Some("edited".into()),
        ..Default::default()
    });
    let (id, agent) = (HookId(1), "agent".to_owned());
    assert!(matches!(
        answered(&hooks, HookCall::OnEnter { id, agent: agent.clone() }).await,
        HookOutcome::Returned
    ));
    assert!(matches!(
        answered(&hooks, HookCall::OnEnter { id, agent: "unknown".into() }).await,
        HookOutcome::Failed
    ));
    assert!(matches!(
        answered(&hooks, HookCall::OnExit { id, agent: agent.clone() }).await,
        HookOutcome::Returned
    ));
    assert!(matches!(
        answered(&hooks, HookCall::OnExit { id, agent: "unknown".into() }).await,
        HookOutcome::Failed
    ));
    settle().await;
    assert_eq!(*hooks.toolsets_closed.lock().unwrap(), 1, "on_exit left the agent's toolsets open");

    let turn = |text: &str| HookCall::OnUserTurnCompleted {
        id,
        agent: "agent".into(),
        chat_ctx: Vec::new(),
        turn: crate::audio_recognition::TurnId(1),
        new_transcript: text.into(),
        transcript_confidence: 0.9,
    };
    let HookOutcome::Turn(edits) = answered(&hooks, turn("original")).await else {
        panic!("not a turn")
    };
    assert!(matches!(
        edits[..],
        [TurnEdit::StopResponse, TurnEdit::UpdateUserMessage(_), TurnEdit::ReplaceItems(_)]
    ));
    let HookOutcome::Turn(edits) = answered(&hooks, turn("edited")).await else {
        panic!("not a turn")
    };
    assert!(
        !edits.iter().any(|e| matches!(e, TurnEdit::UpdateUserMessage(_))),
        "an unchanged text sent as an edit"
    );

    let event = lk_agents_abi::voice::events::ToolsExecutedEvent {
        function_calls: Vec::new(),
        function_call_outputs: vec![lk_agents_abi::FunctionCallOutput::new(
            "item_1",
            "call_1",
            "done",
            false,
            std::time::UNIX_EPOCH,
        )],
        handoff_required: false,
    };
    let executed =
        HookCall::FunctionToolsExecuted { id, speech: SpeechId(1), event: Box::new(event) };
    assert!(matches!(
        answered(&hooks, executed).await,
        HookOutcome::FunctionToolsExecuted { reply_required } if reply_required == [false]
    ));

    let filler = |step, source| HookCall::FillerSource {
        id,
        filler: crate::filler_scheduler::FillerId(1),
        step,
        source,
    };
    assert!(matches!(
        answered(&hooks, filler(0, Some(host_ref(2)))).await,
        HookOutcome::Filler { text: Some(_), spoke: false }
    ));
    assert!(matches!(
        answered(&hooks, filler(1, Some(host_ref(2)))).await,
        HookOutcome::Filler { text: None, spoke: true }
    ));
    assert!(matches!(
        answered(&hooks, filler(2, Some(host_ref(2)))).await,
        HookOutcome::Filler { text: None, spoke: false }
    ));
    assert!(matches!(
        answered(&hooks, filler(0, None)).await,
        HookOutcome::Filler { text: None, spoke: false }
    ));
    assert!(hooks.failed.lock().unwrap().is_empty());

    let raising = Arc::new(ScriptedHooks { raise: true, ..Default::default() });
    assert!(matches!(answered(&raising, turn("x")).await, HookOutcome::Failed));
    assert_eq!(*raising.failed.lock().unwrap(), ["on_user_turn_completed"]);
}

async fn settle() {
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }
}
