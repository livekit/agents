//! `SessionApi` over a fake room: the host's `start()` / `aclose()` pair without a host.

use super::*;
use crate::agent_session::commands::{ActivityCommand, HookId, HookOutcome, TurnCommand};
use crate::agent_session::host_events::{HookCall, HookKind};
use crate::agent_session::{HandoffRefused, SpeechId};
use crate::speech_handle::SpeechHandle;
use lk_agents_abi::voice::events::CloseReason;
use std::sync::atomic::{AtomicBool, Ordering};

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
        None, None, None, true, 24_000, 24_000, options, None, None, false, None, true,
        Some(crate::room_io::RoomIoConfig::default()), None, None, None, None, None, None,
        None, None,
        None, None, None, None, false, None, false, hook_bound, 3, None, None, None, Maybe::NotGiven, None,
        None, None, None, spawner,
    );
    api.expect("a room-only session assembles")
}

// Catches: a `close` that waits for the host's drain while still holding a `PeerHandle` sender on
// the host stream, so the drain can never end.
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

// Catches: a `drain()` refused while a handoff is in flight; it must wait for the swap,
// without running the arriving agent's `on_exit`.
#[tokio::test(start_paused = true)]
async fn a_drain_during_a_handoff_joins_it_and_is_answered_at_the_swap() {
    let (api, mut exits) = session_holding_exits().await;
    let (call, swap) = api.call(Answer::handoff).expect("the loop runs");
    let mut update = agent_update("next", Instructions::plain("You are next."));
    update.call = Some(call);
    api.send(ActivityCommand::Update(Box::new(update))).expect("the loop runs");
    let exit = exits.recv().await.expect("the handoff issues the frame's on_exit");

    let (call, mut drain) = api.call(Answer::drain).expect("the loop runs");
    api.send(ActivityCommand::Drain { call: Some(call) }).expect("the loop runs");
    settle().await;
    assert!(
        matches!(drain.try_recv(), Err(tokio::sync::oneshot::error::TryRecvError::Empty)),
        "the drain was answered while the handoff's on_exit still ran"
    );

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
    api.send(ActivityCommand::Drain { call: Some(call) }).expect("the loop runs");
    settle().await;
    assert!(
        matches!(drain.try_recv(), Err(tokio::sync::oneshot::error::TryRecvError::Empty)),
        "the drain was answered while the close's on_exit still ran"
    );

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
    let (handle, call) = api.speech_call(None).expect("the loop runs");
    let text = SayText::Audio { text: "Hello there.".into(), rx };
    let say = SpeechCommand::Say {
        text,
        allow_interruptions: None,
        add_to_chat_ctx: true,
        call: Some(call),
    };
    api.send_from(origin, say).expect("the loop runs");
    (words, handle)
}

/// A host handoff whose drain waits on a speech still being said: the window a switch holds the
/// host's speech in. Dropping the returned sender ends that speech, and the swap follows.
async fn handoff_draining(
    api: &Recorded,
    exits: &mut tokio::sync::mpsc::UnboundedReceiver<HookId>,
) -> (
    tokio::sync::mpsc::Sender<lk_agents_abi::AudioFrame>,
    tokio::sync::oneshot::Receiver<Result<(), HandoffRefused>>,
) {
    let (words, _) = say_from(api, Origin::Host);
    settle().await;
    let (call, swap) = api.call(Answer::handoff).expect("the loop runs");
    let mut update = agent_update("next", Instructions::plain("You are next."));
    update.call = Some(call);
    api.send(ActivityCommand::Update(Box::new(update))).expect("the loop runs");
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

#[async_trait::async_trait]
impl<T: Send> lk_agents_abi::StreamSource<T> for Items<T> {
    async fn next(&mut self) -> Result<Option<T>, ProviderError> {
        Ok(self.0.pop_front())
    }

    async fn close(self: Box<Self>, _deadline: lk_agents_abi::Ts) -> lk_agents_abi::Closed {
        lk_agents_abi::Closed::Clean
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
    assert!(invalid(api.host_metrics(HostMetricsRecord(eou))).contains("count it twice"));
    assert!(api.aclose("bogus", false, false).is_err());
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

// Catches: a session built with no `SessionOptions` answering a configuration it does not have.
#[tokio::test(start_paused = true)]
async fn a_session_without_options_resolves_none() {
    let api = running(unobserved(fake_room("no-options").1, None)).await;
    assert!(api.resolved_options().is_none());
}
