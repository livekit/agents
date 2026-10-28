//! `SessionApi` over a fake room: the host's `start()` / `aclose()` pair without a host.

use super::*;
use crate::agent_session::commands::{ActivityCommand, HookId, HookOutcome, TurnCommand};
use crate::agent_session::host_events::{HookCall, HookKind};
use crate::agent_session::{HandoffRefused, SpeechId};
use crate::speech_handle::SpeechHandle;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::events::CloseReason;
#[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
use std::sync::atomic::{AtomicBool, Ordering};

#[tokio::test(start_paused = true)]
async fn realtime_transport_timeout_keeps_its_phase_and_request_identity_on_session_error() {
    use crate::agent_activity::realtime::RealtimeWake;
    use crate::agent_session::effects::Effect;
    use lk_agents_abi::Ts;
    use lk_agents_abi::errors::{ApiError, ApiErrorKind, TimeoutPhase};
    use lk_agents_abi::llm::realtime as rt;

    let api = unobserved(fake_room("realtime-timeout-identity").1, None);
    let Stage::Ready(ready) = std::mem::replace(&mut *api.lock(), Stage::Closed) else {
        panic!("session must be ready");
    };
    let (mut cores, _) = ready.builder.start().await.unwrap();
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
    let Stage::Ready(ready) = std::mem::replace(&mut *api.lock(), Stage::Closed) else {
        panic!("session must be ready");
    };
    let (mut cores, _) = ready.builder.start().await.unwrap();
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

fn agent_update(agent: &str, instructions: Instructions) -> AgentUpdate {
    AgentUpdate {
        agent: agent.to_owned(),
        instructions,
        dynamic_instructions: None,
        toolset: crate::tool_executor::Toolset::default(),
        chat_ctx: None,
        llm: Maybe::NotGiven,
        tts: Maybe::NotGiven,
        stt: Maybe::NotGiven,
        vad: Maybe::NotGiven,
        options: Default::default(),
        expressive: None,
        output_nodes: Default::default(),
        call: None,
    }
}

/// A fake room on a virtual clock, and the same room as a session's.
fn fake_room(name: &str) -> (Arc<lk_agents_rtc::FakeRoom>, FixedRoom) {
    let (rtc, clock) = lk_agents_rtc::FakeRtc::with_virtual_clock();
    let room = rtc.room(name);
    // The `FakeRtc` is the room's owner; the room keeps it alive through its own `Arc`.
    std::mem::forget(rtc);
    (room.clone(), FixedRoom { rtc: room, time: Timebase::from_virtual(clock) })
}

/// A session and the steps it journals.
struct Recorded {
    api: SessionApi,
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    steps: Arc<Mutex<Vec<Step>>>,
}

impl std::ops::Deref for Recorded {
    type Target = SessionApi;
    fn deref(&self) -> &SessionApi {
        &self.api
    }
}

impl Recorded {
    fn steps(&self) -> Vec<Step> {
        self.steps.lock().unwrap().clone()
    }
}

#[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
/// A room-only session whose journal the test reads.
fn room_only(room: FixedRoom, hook_bound: Option<f64>) -> Recorded {
    let api = unobserved(room, hook_bound);
    let steps = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&steps);
    api.observe(move |step: &Step| sink.lock().unwrap().push(step.clone()));
    Recorded { api, steps }
}

/// A room-only session, every provider left out, through the host's constructor.
fn unobserved(room: FixedRoom, hook_bound: Option<f64>) -> SessionApi {
    assembled(room, hook_bound, None)
}

#[tokio::test(start_paused = true)]
async fn driver_open_defers_host_delivery_until_runtime_executes_effects() {
    let api = room_only(fake_room("effect-boundary").1, None);
    let Stage::Ready(ready) = std::mem::replace(&mut *api.lock(), Stage::Closed) else {
        panic!("session must be ready");
    };
    let (cores, mut handle) = ready.builder.start().await.unwrap();
    let mut runtime = crate::runtime::Runtime::new(cores);
    while handle.events.try_recv().is_ok() {}
    let before = api.steps();
    let effects = runtime.driver.open(api.time().read());
    assert!(handle.events.try_recv().is_err(), "the decision step delivered a host event");
    assert_eq!(api.steps(), before, "the decision step invoked the journal observer");
    runtime.perform_voice(effects);
    assert!(handle.events.try_recv().is_ok(), "runtime never delivered the opening event");
    assert!(api.steps().len() > before.len(), "runtime never delivered journal observations");
}

#[tokio::test(start_paused = true)]
async fn missing_provider_open_reports_correlated_failure() {
    use crate::agent_session::effects::{Effect, StreamKind};
    let api = unobserved(fake_room("missing-provider").1, None);
    let Stage::Ready(ready) = std::mem::replace(&mut *api.lock(), Stage::Closed) else {
        panic!("session must be ready");
    };
    let (cores, _) = ready.builder.start().await.unwrap();
    let mut runtime = crate::runtime::Runtime::new(cores);
    while runtime.queue.try_recv().is_ok() {}
    for leg in [StreamKind::Stt, StreamKind::Vad] {
        let gone = lk_agents_abi::resources::ResourceId(u64::MAX);
        runtime.perform_voice(vec![match leg {
            StreamKind::Stt => Effect::SttStreamOpen { open: 42, stt: gone },
            StreamKind::Vad => Effect::VadStreamOpen { open: 42, vad: gone },
        }]);
        assert!(
            matches!(
                runtime.queue.try_recv(),
                Ok(crate::agent_session::Event::LegOpened(
                    crate::room_io::StreamOpened { open: 42, leg: reported, outcome: Err(_) }
                )) if reported == leg
            ),
            "a failed launch must settle its request"
        );
    }
}

#[tokio::test(start_paused = true)]
async fn close_is_serviced_before_ready_media() {
    let api = unobserved(fake_room("saturated-media").1, None);
    let Stage::Ready(ready) = std::mem::replace(&mut *api.lock(), Stage::Closed) else {
        panic!("session must be ready");
    };
    let (cores, handle) = ready.builder.start().await.unwrap();
    let mut runtime = crate::runtime::Runtime::new(cores);
    let (media, frames) = tokio::sync::mpsc::channel(64);
    for _ in 0..64 {
        media.try_send(lk_agents_abi::AudioFrame::new(vec![0; 320], 16_000, 1, 160)).unwrap();
    }
    runtime.frames = frames;
    handle.api.send(Command::Close(Close::default())).unwrap();
    runtime.run().await;
    assert_eq!(runtime.frames.len(), 64, "ready media delayed close");
}

#[tokio::test(start_paused = true)]
async fn accepted_views_are_published_before_receipts() {
    use crate::agent_session::effects::Effect;
    let api = unobserved(fake_room("view-before-receipt").1, None);
    let Stage::Ready(ready) = std::mem::replace(&mut *api.lock(), Stage::Closed) else {
        panic!("session must be ready");
    };
    let (cores, handle) = ready.builder.start().await.unwrap();
    let mut runtime = crate::runtime::Runtime::new(cores);
    let mut accepted = (**api.view.borrow()).clone();
    accepted.config.configure(&crate::agent_activity::options::aec_warmup_layer(Some(9.0)));
    // The receipt's waker reads the view at the instant the answer is sent.
    struct ReadsView(tokio::sync::watch::Receiver<Arc<crate::agent_session::SessionView>>);
    impl std::task::Wake for ReadsView {
        fn wake(self: Arc<Self>) {
            assert_eq!(
                self.0.borrow().config.effective.timers.aec_warmup(),
                Some(Duration::from_secs(9)),
                "receipt observed options from before acceptance"
            );
        }
    }
    let (call, mut receipt) = open(RequestId(0), Answer::interrupt);
    let call_id = call.id;
    let waker = std::task::Waker::from(Arc::new(ReadsView(api.view.clone())));
    let polled = std::pin::Pin::new(&mut receipt).poll(&mut std::task::Context::from_waker(&waker));
    assert!(polled.is_pending());
    runtime.handle(
        crate::agent_session::Event::Awaited(
            crate::agent_session::Input::Host(crate::agent_session::Sent {
                origin: Origin::Host,
                command: Command::SetStaticKeyterms(Vec::new()),
            }),
            call,
        ),
        handle.peer.time.read(),
    );
    runtime.perform_voice(vec![
        Effect::Answer(call_id, Answer::Interrupt(Ok(()))),
        Effect::SessionView(Arc::new(accepted)),
    ]);
    assert!(receipt.await.is_ok(), "the receipt was not delivered");
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
        Arc::new(room), None, None, "agent".into(), None, None, None, None, None, None, None, None,
        None, None, None, None, true, 24_000, 24_000, options, None, None, None, None, None, true,
        Some(crate::room_io::RoomIoConfig::default()), None, None, None, None, None, None,
        None, None,
        None, None, None, None, false, None, false, hook_bound, 3, None, None, None, Maybe::NotGiven, None,
        None, None, None, spawner,
    );
    api
}

// Catches: a `close` that waits for the host's drain while still holding a `PeerHandle` sender on
// the host stream, so the drain can never end.
#[allow(clippy::disallowed_methods, reason = "S0: test harness runtime resource")]
#[allow(clippy::disallowed_types, reason = "S0: test harness runtime resource")]
#[tokio::test(start_paused = true)]
async fn close_on_a_virtual_clock_ends_the_drain_and_returns() {
    let api = room_only(fake_room("session-api-test").1, None);
    let saw_close = Arc::new(AtomicBool::new(false));
    let drain_ended = Arc::new(AtomicBool::new(false));
    let (saw, ended) = (Arc::clone(&saw_close), Arc::clone(&drain_ended));
    let opening = api
        .open(false, move |mut runner: SessionRunner| {
            Box::pin(async move {
                while let Some(ev) = runner.next_host_event().await {
                    match ev {
                        HostEvent::Hook(call) => {
                            let _ = runner.api().send(Command::HookDone {
                                id: call.id(),
                                outcome: HookOutcome::Returned,
                            });
                        }
                        HostEvent::Session(SessionEvent::Close { .. }) => {
                            saw.store(true, Ordering::SeqCst);
                        }
                        HostEvent::Session(_) | HostEvent::Room(_) => {}
                    }
                }
                ended.store(true, Ordering::SeqCst);
            })
        })
        .expect("a ready session opens");
    opening.await.expect("the loop starts");

    // Bounded on the paused runtime clock: a close parked on the virtual clock fails here.
    tokio::time::timeout(
        Duration::from_secs(60),
        api.close(CloseReason::UserInitiated, false, None),
    )
    .await
    .expect("close hung: the drain was still waiting on a sender the stage held");

    assert!(saw_close.load(Ordering::SeqCst), "the drain never saw the `close` event");
    assert!(
        drain_ended.load(Ordering::SeqCst),
        "close returned before the drain ended: the host's `close` handlers were not queued"
    );
    assert!(matches!(api.send(TurnCommand::Clear), Err(SessionRefused::Closed)));
}

// Catches: a production session (no observer) that keeps every step it decides, so a long call's
// memory grows with its length.
#[tokio::test(start_paused = true)]
async fn a_session_nobody_observes_keeps_no_journal() {
    let api = unobserved(fake_room("no-journal").1, None);
    let opening = api
        .open(false, move |mut runner: SessionRunner| {
            Box::pin(async move {
                while let Some(ev) = runner.next_host_event().await {
                    if let HostEvent::Hook(call) = ev {
                        let _ = runner.api().send(Command::HookDone {
                            id: call.id(),
                            outcome: HookOutcome::Returned,
                        });
                    }
                }
            })
        })
        .expect("a ready session opens");
    opening.await.expect("the loop starts");
    for _ in 0..1000 {
        api.send(TurnCommand::Clear).expect("the loop runs");
    }
    settle().await;
    let Stage::Running(running) = &*api.lock() else { panic!("the session is not running") };
    assert!(!running.peer.journal.is_on(), "a session nobody observes journals its steps");
}

// Catches: `RoomIo::on_agent_state_changed` with no caller, so the agent never writes
// `lk.agent.state` and a frontend keyed on it shows the agent stuck.
#[tokio::test(start_paused = true)]
async fn the_agent_state_is_written_to_the_room_as_it_changes() {
    let (room, fixed) = fake_room("agent-state");
    let api = room_only(fixed, None);
    let opening = api
        .open(false, move |mut runner: SessionRunner| {
            Box::pin(async move {
                while let Some(ev) = runner.next_host_event().await {
                    if let HostEvent::Hook(call) = ev {
                        let _ = runner.api().send(Command::HookDone {
                            id: call.id(),
                            outcome: HookOutcome::Returned,
                        });
                    }
                }
            })
        })
        .expect("a ready session opens");
    opening.await.expect("the loop starts");
    settle().await;

    let written: Vec<String> = room
        .local_participant()
        .attribute_calls()
        .iter()
        .filter_map(|a| a.get(lk_agents_abi::ATTRIBUTE_AGENT_STATE).cloned())
        .collect();
    assert!(!written.is_empty(), "no `lk.agent.state` write reached the room");
    assert_eq!(
        room.local_participant()
            .attributes()
            .get(lk_agents_abi::ATTRIBUTE_AGENT_STATE)
            .map(String::as_str),
        Some("listening"),
        "the room's attribute is not the agent's current state (writes: {written:?})"
    );
}

#[allow(clippy::disallowed_methods, reason = "S0: test harness runtime resource")]
#[allow(clippy::disallowed_types, reason = "S0: test harness runtime resource")]
/// A room-only session whose host answers every hook but `on_exit`, which the test answers.
async fn session_holding_exits() -> (Recorded, tokio::sync::mpsc::UnboundedReceiver<HookId>) {
    // A host that runs hooks: without a bound the driver treats `on_exit` as over at once.
    let api = room_only(fake_room("session-api-test").1, Some(10.0));
    let (exits_tx, exits) = tokio::sync::mpsc::unbounded_channel();
    let opening = api
        .open(false, move |mut runner: SessionRunner| {
            Box::pin(async move {
                while let Some(ev) = runner.next_host_event().await {
                    if let HostEvent::Hook(call) = ev {
                        if matches!(call, HookCall::OnExit { .. }) {
                            let _ = exits_tx.send(call.id());
                        } else {
                            let _ = runner.api().send(Command::HookDone {
                                id: call.id(),
                                outcome: HookOutcome::Returned,
                            });
                        }
                    }
                }
            })
        })
        .expect("a ready session opens");
    opening.await.expect("the loop starts");
    (api, exits)
}

fn answer_exit(api: &Recorded, id: HookId) {
    api.send(Command::HookDone { id, outcome: HookOutcome::Returned }).expect("the loop runs");
}

/// The `on_exit`s issued, of `agent` or of any.
fn on_exits(api: &Recorded, agent: Option<&str>) -> usize {
    api.steps()
        .iter()
        .filter(|s| {
            matches!(
                s,
                Step::Handoff(crate::trace::HandoffStep::HookIssued {
                    kind: HookKind::OnExit, agent: a, ..
                }) if agent.is_none() || a.as_deref() == agent
            )
        })
        .count()
}

async fn settle() {
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }
}

/// Whether `call` has no answer yet.
fn unanswered<T>(call: &mut Awaited<T>) -> bool {
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    std::pin::Pin::new(call).poll(&mut cx).is_pending()
}

// Catches: a `drain()` refused while a handoff is in flight; it must wait for the swap,
// without running the arriving agent's `on_exit`.
#[tokio::test(start_paused = true)]
async fn a_drain_during_a_handoff_joins_it_and_is_answered_at_the_swap() {
    let (api, mut exits) = session_holding_exits().await;
    let (call, swap) = api.call(Answer::handoff).expect("the loop runs");
    let mut update = agent_update("next", Instructions::plain("You are next."));
    update.call = Some(call.id);
    api.send_awaited(Origin::Host, ActivityCommand::Update(Box::new(update)), call)
        .expect("the loop runs");
    let exit = exits.recv().await.expect("the handoff issues the frame's on_exit");

    let (call, mut drain) = api.call(Answer::drain).expect("the loop runs");
    api.send_awaited(Origin::Host, ActivityCommand::Drain { call: Some(call.id) }, call)
        .expect("the loop runs");
    settle().await;
    assert!(unanswered(&mut drain), "the drain was answered while the handoff's on_exit still ran");

    answer_exit(&api, exit);
    assert_eq!(swap.await.expect("answered"), Ok(()), "the handoff lands");
    assert_eq!(drain.await.expect("answered"), Ok(()), "joined, not refused");
    assert_eq!(on_exits(&api, None), 1, "the arriving frame's on_exit ran: the drain moved frames");
    api.close(CloseReason::UserInitiated, false, None).await;
}

// Catches: a `drain()` refused during the close; it must wait for the close's drain.
#[tokio::test(start_paused = true)]
async fn a_drain_during_the_close_is_answered_by_the_closes_drain() {
    let (api, mut exits) = session_holding_exits().await;
    api.send(Command::Close(Close::default())).expect("the loop runs");
    let exit = exits.recv().await.expect("the close issues the frame's on_exit");

    let (call, mut drain) = api.call(Answer::drain).expect("the loop runs");
    api.send_awaited(Origin::Host, ActivityCommand::Drain { call: Some(call.id) }, call)
        .expect("the loop runs");
    settle().await;
    assert!(unanswered(&mut drain), "the drain was answered while the close's on_exit still ran");

    answer_exit(&api, exit);
    assert_eq!(drain.await.expect("answered"), Ok(()), "joined the close, not refused");
    assert_eq!(on_exits(&api, None), 1, "one on_exit for the frame, the close's");
    api.close(CloseReason::UserInitiated, false, None).await;
}

/// A `say(audio=)` from `origin`, its frames held by the returned sender, and its handle.
fn say_from(
    api: &Recorded,
    origin: Origin,
) -> (tokio::sync::mpsc::Sender<lk_agents_abi::AudioFrame>, Arc<SpeechHandle>) {
    let (words, rx) = tokio::sync::mpsc::channel(4);
    let (handle, call, reply) = api.speech_call(None).expect("the loop runs");
    let text = SayText::Audio { text: "Hello there.".into(), rx };
    let say = SpeechCommand::Say {
        text,
        allow_interruptions: None,
        add_to_chat_ctx: true,
        call: Some(call),
    };
    api.send_awaited(origin, say, reply).expect("the loop runs");
    (words, handle)
}

#[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
/// A host handoff whose drain waits on a speech still being said: the window a switch holds the
/// host's speech in. Dropping the returned sender ends that speech, and the swap follows.
async fn handoff_draining(
    api: &Recorded,
    #[allow(clippy::disallowed_types, reason = "S0: test harness runtime resource")]
    exits: &mut tokio::sync::mpsc::UnboundedReceiver<HookId>,
) -> (tokio::sync::mpsc::Sender<lk_agents_abi::AudioFrame>, Awaited<Result<(), HandoffRefused>>) {
    let (words, _) = say_from(api, Origin::Host);
    settle().await;
    let (call, swap) = api.call(Answer::handoff).expect("the loop runs");
    let mut update = agent_update("next", Instructions::plain("You are next."));
    update.call = Some(call.id);
    api.send_awaited(Origin::Host, ActivityCommand::Update(Box::new(update)), call)
        .expect("the loop runs");
    let exit = exits.recv().await.expect("the handoff issues the frame's on_exit");
    answer_exit(api, exit);
    settle().await;
    (words, swap)
}

fn position(api: &Recorded, step: impl Fn(&Step) -> bool) -> Option<usize> {
    api.steps().iter().position(step)
}

fn rejected(
    api: &Recorded,
    speech: SpeechId,
) -> Option<crate::agent_activity::scheduling::Rejected> {
    api.steps().iter().find_map(|s| match s {
        Step::Speech(crate::trace::SpeechStep::ScheduleRejected { speech: s, reason })
            if *s == speech =>
        {
            Some(*reason)
        }
        _ => None,
    })
}

// Catches: a host `say()` during a switch refused as draining; the reference schedules it on
// `_next_activity`, so the arriving agent speaks it, before its `on_enter`.
#[tokio::test(start_paused = true)]
async fn a_host_say_during_a_handoff_is_spoken_by_the_arriving_agent_before_its_on_enter() {
    let (api, mut exits) = session_holding_exits().await;
    let (words, swap) = handoff_draining(&api, &mut exits).await;
    let (_held, handle) = say_from(&api, Origin::Host);
    settle().await;
    assert!(handle.created().is_none(), "created on the leaving agent: {:#?}", api.steps());

    drop(words);
    assert_eq!(swap.await.expect("answered"), Ok(()), "the handoff lands");
    handle.wait_until_created().await.expect("created at the swap");
    let speech = handle.id();
    settle().await;
    assert_eq!(rejected(&api, speech), None, "{:#?}", api.steps());
    let updated = position(&api, |s| {
        matches!(s, Step::Handoff(crate::trace::HandoffStep::AgentUpdated { .. }))
    });
    let created = position(
        &api,
        |s| matches!(s, Step::Speech(crate::trace::SpeechStep::ReplyCreated { speech: s }) if *s == speech),
    );
    let on_enter = position(&api, |s| {
        matches!(
            s,
            Step::Handoff(crate::trace::HandoffStep::HookIssued {
                kind: HookKind::OnEnter, agent: Some(a), ..
            }) if a == "next"
        )
    });
    assert!(updated.is_some() && updated < created, "created before the swap: {:#?}", api.steps());
    assert!(created < on_enter, "the arriving agent's on_enter went first: {:#?}", api.steps());
    api.close(CloseReason::UserInitiated, false, None).await;
}

// Catches: the leaving agent's own speech (a tool's, a hook's) carried to the arriving agent; it
// is still refused by the paused scheduler, as in the reference.
#[tokio::test(start_paused = true)]
async fn the_leaving_agents_own_say_during_a_handoff_is_still_refused() {
    let (api, mut exits) = session_holding_exits().await;
    let (_words, _swap) = handoff_draining(&api, &mut exits).await;
    for origin in [Origin::Tool { call_id: "call_1".into() }, Origin::Hook] {
        let (_held, handle) = say_from(&api, origin.clone());
        handle.wait_until_created().await.expect("created at once");
        assert_eq!(
            rejected(&api, handle.id()),
            Some(crate::agent_activity::scheduling::Rejected::Draining),
            "{origin:?}: {:#?}",
            api.steps()
        );
        assert!(handle.interrupted(), "{origin:?}");
        assert!(handle.done(), "refused speech waited for cleanup of work that never started");
    }
    api.close(CloseReason::UserInitiated, false, None).await;
}

// Catches: a held say the host interrupted before the swap played on the arriving agent anyway;
// the interrupt is kept and lands as the speech is created.
#[tokio::test(start_paused = true)]
async fn a_held_say_interrupted_before_the_swap_does_not_play() {
    let (api, mut exits) = session_holding_exits().await;
    let (words, swap) = handoff_draining(&api, &mut exits).await;
    let (_held, handle) = say_from(&api, Origin::Host);
    let spawner = Arc::new(lk_agents_abi::TokioSpawner::new(tokio::runtime::Handle::current()));
    handle.interrupt_for_host(false, spawner).expect("an interruptible say");
    assert!(handle.interrupted_for_host(), "the interrupt is the handle's at once");

    drop(words);
    assert_eq!(swap.await.expect("answered"), Ok(()), "the handoff lands");
    settle().await;
    assert!(handle.created().is_some(), "created at the swap");
    assert!(handle.interrupted() && handle.done(), "{:#?}", api.steps());
    let played = position(
        &api,
        |s| matches!(s, Step::Playout(crate::trace::PlayoutStep::PlaybackStarted { speech, .. }) if *speech == handle.id()),
    );
    assert_eq!(played, None, "{:#?}", api.steps());
    api.close(CloseReason::UserInitiated, false, None).await;
}

// Catches: a held say left unanswered when the close refuses the switch it waited on: the host
// awaiting it hung. It is refused as any speech into a closing session is.
#[allow(clippy::disallowed_macros, reason = "S0: test harness runtime resource")]
#[tokio::test(start_paused = true)]
async fn a_held_say_is_refused_when_the_close_refuses_the_handoff() {
    for drain in [true, false] {
        let (api, mut exits) = session_holding_exits().await;
        let (words, swap) = handoff_draining(&api, &mut exits).await;
        let (_held, handle) = say_from(&api, Origin::Host);
        settle().await;
        let close = Close { drain, ..Close::default() };
        api.send(Command::Close(close)).expect("the loop runs");
        settle().await;
        drop(words);
        handle.wait_until_created().await.expect("created, not hung");

        assert_eq!(swap.await.expect("answered"), Err(HandoffRefused::Closing), "drain={drain}");
        settle().await;
        assert!(rejected(&api, handle.id()).is_some(), "drain={drain}: {:#?}", api.steps());
        assert!(handle.interrupted(), "drain={drain}");
        let closing = api.close(CloseReason::UserInitiated, false, None);
        tokio::pin!(closing);
        loop {
            tokio::select! {
                _ = &mut closing => break,
                Some(id) = exits.recv() => answer_exit(&api, id),
            }
        }
    }
}

/// A session opened over a host that answers every hook at once.
async fn running(api: SessionApi) -> SessionApi {
    let opening = api
        .open(false, move |mut runner: SessionRunner| {
            Box::pin(async move {
                while let Some(ev) = runner.next_host_event().await {
                    if let HostEvent::Hook(call) = ev {
                        let _ = runner.api().send(Command::HookDone {
                            id: call.id(),
                            outcome: HookOutcome::Returned,
                        });
                    }
                }
            })
        })
        .expect("a ready session opens");
    opening.await.expect("the loop starts");
    api
}

/// A host stream of these items, then its end.
struct Items<T>(std::collections::VecDeque<T>);

impl<T: Send + 'static> lk_agents_abi::StreamSource<T> for Items<T> {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<T>, ProviderError>> {
        Box::pin(async move { Ok(self.0.pop_front()) })
    }

    fn close(
        self: Box<Self>,
        _deadline: lk_agents_abi::Ts,
    ) -> lk_agents_abi::BoxFuture<'static, lk_agents_abi::Closed> {
        Box::pin(async move { lk_agents_abi::Closed::Clean })
    }
}

fn invalid<T>(r: Result<T, CommandRefused>) -> String {
    match r {
        Err(CommandRefused::Invalid(why)) => why,
        Err(other) => panic!("expected an invalid-argument refusal, got {other:?}"),
        Ok(_) => panic!("expected an invalid-argument refusal, got an answer"),
    }
}

// Catches: a malformed host argument reaching the driver (a negative duration panics
// `Duration::from_secs_f64` on the loop), or refused with a message that does not name it.
#[tokio::test(start_paused = true)]
async fn host_arguments_are_checked_before_anything_is_sent() {
    let api = unobserved(fake_room("bad-arguments").1, None);
    assert!(invalid(api.ivr_start(-1.0, None, None)).contains("max_silence"));
    assert!(invalid(api.ivr_start(f64::NAN, None, None)).contains("max_silence"));
    let filler = |delay, interval, speech: &str| {
        api.filler_start("call".into(), speech, delay, interval, None, Some("hm".into()), None)
    };
    invalid(filler(-1.0, None, "speech_1"));
    invalid(filler(f64::NAN, None, "speech_1"));
    invalid(filler(0.5, Some(-1.0), "speech_1"));
    assert!(invalid(filler(0.5, None, "nope")).contains("nope"));
    assert!(invalid(api.set_io_enabled("microphone", true)).contains("microphone"));
    assert!(invalid(api.dtmf_start(4, false, 4.0, "x".into())).contains("\"x\""));
    assert!(invalid(api.dtmf_start(4, false, -1.0, "#".into())).contains("dtmf_input_timeout"));
    use lk_agents_abi::metrics::base::{AgentMetrics, EouMetrics};
    let eou = AgentMetrics::Eou(EouMetrics {
        timestamp: 0.0,
        end_of_utterance_delay: 0.0,
        transcription_delay: 0.0,
        on_user_turn_completed_delay: 0.0,
        speech_id: None,
        metadata: None,
    });
    assert!(invalid(api.host_metrics(eou)).contains("count it twice"));
    assert!(api.aclose("bogus", false).is_err());
    // Well-formed, but nothing runs yet.
    assert!(matches!(api.set_io_enabled("audio_input", true), Err(CommandRefused::Session(_))));
}

// Catches: the host's rarer calls (a streamed or pre-synthesised `say`, the outbound-SIP warmup,
// the reads `session.*` makes) refused or wrong on a running session.
#[tokio::test(start_paused = true)]
async fn rare_host_calls_reach_a_running_session() {
    use crate::agent_activity::options::{HostOptions, tests::kwargs};
    let options = HostOptions::new(kwargs(None));
    let api = running(assembled(fake_room("rare-calls").1, None, Some(&options))).await;

    assert_eq!(api.resolved_options().unwrap().aec_warmup_duration, Some(3.0));
    api.set_aec_warmup(None).unwrap();
    api.interrupt(false).unwrap().await.unwrap();
    assert_eq!(api.resolved_options().unwrap().aec_warmup_duration, None);
    api.update_turn_options(Default::default()).unwrap();

    let words = Items(["Hello ".to_owned(), "there.".to_owned()].into());
    let streamed = api.say_stream(Box::new(words), None, true, None).unwrap();
    streamed.wait_until_created().await.unwrap();
    let frames = Items(Default::default());
    let recorded = api.say_audio("Hi.".into(), Box::new(frames), None, true, None).unwrap();
    recorded.wait_until_created().await.unwrap();
    // Outside any speech: nothing to end; no output to set.
    api.end_call(&SpeechHandle::detached(String::new(), true)).unwrap();
    api.set_run_output(&streamed, None, None).unwrap();

    assert_eq!(api.keyterms(), None);
    assert!(!api.amd_listening());
    assert!(!api.dtmf_reply_running());
    assert!(api.usage().is_empty());
    assert_eq!(api.linked_participant(), None);
    api.unset_participant().await;

    api.shutdown(false);
    for _ in 0..20 {
        settle().await;
        tokio::time::advance(Duration::from_secs(1)).await;
    }
    assert!(matches!(api.send(TurnCommand::Clear), Err(SessionRefused::Closed)));
}

#[tokio::test(start_paused = true)]
async fn options_view_changes_only_after_core_acceptance() {
    use crate::agent_activity::options::{HostOptions, tests::kwargs};
    let options = HostOptions::new(kwargs(None));
    let api = running(assembled(fake_room("accepted-options").1, None, Some(&options))).await;
    api.set_aec_warmup(None).unwrap();
    assert_eq!(api.resolved_options().unwrap().aec_warmup_duration, Some(3.0));
    let (call, answer) = api.call(Answer::idle).unwrap();
    let wait = SpeechCommand::WaitForIdle { hold: None, call: Some(call.id) };
    api.send_awaited(Origin::Host, wait, call).unwrap();
    assert!(answer.await.unwrap().is_ok());
    assert_eq!(api.resolved_options().unwrap().aec_warmup_duration, None);
    api.close(CloseReason::UserInitiated, false, None).await;
    assert!(api.set_aec_warmup(Some(9.0)).is_err());
    assert_eq!(api.resolved_options().unwrap().aec_warmup_duration, None);
}

// Catches: a session built with no `SessionOptions` answering a configuration it does not have.
#[tokio::test(start_paused = true)]
async fn a_session_without_options_resolves_none() {
    let api = running(unobserved(fake_room("no-options").1, None)).await;
    assert!(api.resolved_options().is_none());
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
