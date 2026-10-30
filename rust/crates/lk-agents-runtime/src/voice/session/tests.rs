//! `VoiceSession` over a fake room: the host's `start()` / `aclose()` pair without a host.

mod hooks;
mod playback;
mod recorder;

use lk_agents_abi::voice::speech_handle::SpeechHandle as HostSpeech;

use super::*;
use crate::room_io::FixedRoom;
use crate::voice::speech_handle::SpeechHandle;
use crate::voice::{Executor, remote_session};
use lk_agents_abi::Maybe;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::events::CloseReason;
use lk_agents_abi::voice::session::VoiceSession as _;
use lk_agents_voice::agent_activity::scheduling::SpeechId;
use lk_agents_voice::agent_session::HandoffRefused;
use lk_agents_voice::agent_session::commands::{ActivityCommand, HookId, HookOutcome, TurnCommand};
use lk_agents_voice::agent_session::host_events::{HookCall, HookKind};
use lk_agents_voice::agent_session::*;
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::generation::Instructions;
use lk_agents_voice::trace::Step;
use std::future::Future as _;
use parking_lot::Mutex;
use std::time::Duration;

use std::sync::atomic::{AtomicBool, Ordering};

fn agent_update(agent: &str, instructions: Instructions) -> AgentUpdate {
    AgentUpdate {
        agent: agent.to_owned(),
        instructions,
        dynamic_instructions: None,
        toolset: lk_agents_voice::tool_executor::Toolset::default(),
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

/// A session's room without RTC, on a virtual clock.
fn no_room() -> FixedRoom {
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    FixedRoom { rtc: None, time: Timebase::from_virtual(clock), pool: None }
}

/// An output that never reports playback; it latches the segment its first capture was keyed to.
#[derive(Default)]
struct MissingPlaybackReports(
    Arc<lk_agents_abi::voice::io::PlaybackTracker>,
    std::sync::OnceLock<lk_agents_abi::voice::io::PlaybackSegment>,
);

impl lk_agents_abi::voice::io::AudioOutput for MissingPlaybackReports {
    fn label(&self) -> &str {
        "missing playback reports"
    }
    fn tracker(&self) -> &Arc<lk_agents_abi::voice::io::PlaybackTracker> {
        &self.0
    }
    fn capabilities(&self) -> lk_agents_abi::voice::io::AudioOutputCapabilities {
        Default::default()
    }
    fn capture<'a>(
        &'a self,
        _: &'a lk_agents_abi::AudioFrame,
    ) -> lk_agents_abi::BoxFuture<'a, lk_agents_abi::voice::io::Result<()>> {
        Box::pin(async {
            self.1.set(self.0.current_segment().expect("admitted capture")).unwrap();
            Ok(())
        })
    }
    fn flush_segment(&self) {}
    fn clear_buffer(&self) {}
}

/// A session and the steps it journals.
struct Recorded {
    api: VoiceSession,

    steps: Arc<Mutex<Vec<Step>>>,
}

impl std::ops::Deref for Recorded {
    type Target = VoiceSession;
    fn deref(&self) -> &VoiceSession {
        &self.api
    }
}

impl Recorded {
    fn steps(&self) -> Vec<Step> {
        self.steps.lock().clone()
    }
}

/// A room-only session whose journal the test reads.
fn room_only(room: FixedRoom, hook_bound: Option<f64>) -> Recorded {
    let api = unobserved(room, hook_bound);
    let steps = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&steps);
    api.observe(move |step: &Step| sink.lock().push(step.clone()));
    Recorded { api, steps }
}

/// A room-only session, every provider left out, through the host's constructor.
fn unobserved(room: FixedRoom, hook_bound: Option<f64>) -> VoiceSession {
    assembled(room, hook_bound, None)
}

/// [`unobserved`], with the host's `SessionOptions` object when given.
fn assembled(
    room: FixedRoom,
    hook_bound: Option<f64>,
    options: Option<SessionOptions>,
) -> VoiceSession {
    let room = Arc::new(room) as Arc<dyn lk_agents_abi::voice::room_owner::SessionRoom>;
    let args = lk_agents_abi::voice::session::VoiceSessionArgs {
        agent: "agent".into(),
        room_io_config: Some(Default::default()),
        hook_bound,
        ..Default::default()
    };
    let env = lk_agents_net::env::ProcessEnv::current_runtime();
    let api = <VoiceSession as lk_agents_abi::voice::session::VoiceSession>::new(
        &env, room, args, options,
    );
    api.expect("valid native room ports")
}

// Catches: a `close` that waits for the host's drain while still holding a `PeerHandle` sender on
// the host stream, so the drain can never end.

#[tokio::test(start_paused = true)]
async fn close_on_a_virtual_clock_ends_the_drain_and_returns() {
    let api = room_only(no_room(), None);
    let saw_close = Arc::new(AtomicBool::new(false));
    let drain_ended = Arc::new(AtomicBool::new(false));
    let (saw, ended) = (Arc::clone(&saw_close), Arc::clone(&drain_ended));
    let opening = api
        .open(
            Box::new(Executor::new(api.time().clone(), api.spawner())),
            None,
            move |mut runner: SessionRunner| {
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
                            HostEvent::Session(_)
                            | HostEvent::Room(_)
                            | HostEvent::SpeechRetired(_) => {}
                        }
                    }
                    ended.store(true, Ordering::SeqCst);
                })
            },
        )
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

/// The production executor, keeping the journal of the cores it takes over.
struct Observing(Executor, Arc<Mutex<Option<lk_agents_voice::trace::Journal>>>);

impl SessionExecutor for Observing {
    fn tts_adapter(&self) -> crate::voice::session::TtsAdapter {
        self.0.tts_adapter()
    }

    fn stt_adapter(&self) -> crate::voice::session::SttAdapter {
        self.0.stt_adapter()
    }

    fn open_legs<'a>(
        &'a mut self,
        resources: &'a crate::voice::resources::Resources,
        stt: Option<Arc<dyn lk_agents_abi::Stt>>,
        vad: Option<Arc<dyn lk_agents_abi::Vad>>,
    ) -> lk_agents_abi::BoxFuture<'a, Result<OpeningLegs, DriverError>> {
        self.0.open_legs(resources, stt, vad)
    }

    fn open_realtime<'a>(
        &'a mut self,
        resources: &'a crate::voice::resources::Resources,
        model: Arc<dyn lk_agents_abi::Realtime>,
        disabled: bool,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            (Arc<dyn lk_agents_abi::RealtimeSession>, lk_agents_abi::resources::ResourceId),
            DriverError,
        >,
    > {
        self.0.open_realtime(resources, model, disabled)
    }

    fn own(self: Box<Self>, cores: SessionCores) -> Box<dyn QueueOwner> {
        *self.1.lock() = Some(cores.peer.journal.clone());
        Box::new(self.0).own(cores)
    }
}

// Catches: a production session (no observer) that keeps every step it decides, so a long call's
// memory grows with its length.
#[tokio::test(start_paused = true)]
async fn a_session_nobody_observes_keeps_no_journal() {
    let api = unobserved(no_room(), None);
    let journal = Arc::new(Mutex::new(None));
    let observed = journal.clone();
    let opening = api
        .open(
            Box::new(Observing(Executor::new(api.time().clone(), api.spawner()), observed)),
            None,
            move |mut runner: SessionRunner| {
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
            },
        )
        .expect("a ready session opens");
    opening.await.expect("the loop starts");
    for _ in 0..1000 {
        api.send(TurnCommand::Clear).expect("the loop runs");
    }
    settle().await;
    assert!(
        !journal.lock().as_ref().unwrap().is_on(),
        "a session nobody observes journals its steps"
    );
}

/// A room-only session whose host answers every hook but `on_exit`, which the test answers.
async fn session_holding_exits() -> (Recorded, tokio::sync::mpsc::UnboundedReceiver<HookId>) {
    // A host that runs hooks: without a bound the driver treats `on_exit` as over at once.
    let api = room_only(no_room(), Some(10.0));
    let (exits_tx, exits) = tokio::sync::mpsc::unbounded_channel();
    let opening = api
        .open(
            Box::new(Executor::new(api.time().clone(), api.spawner())),
            None,
            move |mut runner: SessionRunner| {
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
            },
        )
        .expect("a ready session opens");
    opening.await.expect("the loop starts");
    // An output the held `say(audio=)` streams are captured to.
    api.set_audio_output(Some(Arc::new(MissingPlaybackReports::default()))).unwrap();
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
                Step::Handoff(lk_agents_voice::trace::HandoffStep::HookIssued {
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
    handle.hold_say(crate::voice::speech_handle::SayStream::Audio(rx));
    let text = SayText::Audio { text: "Hello there.".into() };
    let say = SpeechCommand::Say {
        text,
        allow_interruptions: None,
        add_to_chat_ctx: true,
        call: Some(call),
    };
    api.send_awaited(origin, say, reply).expect("the loop runs");
    (words, handle)
}

/// A host handoff whose drain waits on a speech still being said: the window a switch holds the
/// host's speech in. Dropping the returned sender ends that speech, and the swap follows.
async fn handoff_draining(
    api: &Recorded,

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
) -> Option<lk_agents_voice::agent_activity::scheduling::Rejected> {
    api.steps().iter().find_map(|s| match s {
        Step::Speech(lk_agents_voice::trace::SpeechStep::ScheduleRejected {
            speech: s,
            reason,
        }) if *s == speech => Some(*reason),
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
    HostSpeech::wait_until_created(&*handle).await.expect("created at the swap");
    let speech = handle.id();
    settle().await;
    assert_eq!(rejected(&api, speech), None, "{:#?}", api.steps());
    let updated = position(&api, |s| {
        matches!(s, Step::Handoff(lk_agents_voice::trace::HandoffStep::AgentUpdated { .. }))
    });
    let created = position(
        &api,
        |s| matches!(s, Step::Speech(lk_agents_voice::trace::SpeechStep::ReplyCreated { speech: s }) if *s == speech),
    );
    let on_enter = position(&api, |s| {
        matches!(
            s,
            Step::Handoff(lk_agents_voice::trace::HandoffStep::HookIssued {
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
        HostSpeech::wait_until_created(&*handle).await.expect("created at once");
        assert_eq!(
            rejected(&api, handle.id()),
            Some(lk_agents_voice::agent_activity::scheduling::Rejected::Draining),
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
    HostSpeech::interrupt(&*handle, false).expect("an interruptible say");
    assert!(HostSpeech::interrupted(&*handle), "the interrupt is the handle's at once");

    drop(words);
    assert_eq!(swap.await.expect("answered"), Ok(()), "the handoff lands");
    settle().await;
    assert!(handle.created().is_some(), "created at the swap");
    assert!(handle.interrupted() && handle.done(), "{:#?}", api.steps());
    let played = position(
        &api,
        |s| matches!(s, Step::Playout(lk_agents_voice::trace::PlayoutStep::PlaybackStarted { speech, .. }) if *speech == handle.id()),
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
        HostSpeech::wait_until_created(&*handle).await.expect("created, not hung");

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
async fn running(api: VoiceSession) -> VoiceSession {
    let opening = api
        .open(
            Box::new(Executor::new(api.time().clone(), api.spawner())),
            None,
            move |mut runner: SessionRunner| {
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
            },
        )
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
    let api = unobserved(no_room(), None);
    assert!(invalid(api.ivr_start(-1.0, None, None)).contains("max_silence"));
    assert!(invalid(api.ivr_start(f64::NAN, None, None)).contains("max_silence"));
    let filler = |delay, interval, speech: &str| {
        api.filler_start(
            "call".into(),
            speech.into(),
            delay,
            interval,
            None,
            Some("hm".into()),
            None,
        )
    };
    invalid(filler(-1.0, None, "speech_1"));
    invalid(filler(f64::NAN, None, "speech_1"));
    invalid(filler(0.5, Some(-1.0), "speech_1"));
    assert!(invalid(filler(0.5, None, "nope")).contains("nope"));
    assert!(invalid(api.set_io_enabled("microphone".into(), true)).contains("microphone"));
    assert!(invalid(api.collect_dtmf(4, false, 4.0, "x".into())).contains("\"x\""));
    assert!(invalid(api.collect_dtmf(4, false, -1.0, "#".into())).contains("timeout"));
    assert!(api.aclose("bogus".into(), false).is_err());
    // Well-formed, but nothing runs yet.
    assert!(matches!(
        api.set_io_enabled("audio_input".into(), true),
        Err(CommandRefused::Session(_))
    ));
}

// Catches: the host's rarer calls (a streamed or pre-synthesised `say`, the reads `session.*`
// makes) refused or wrong on a running session.
#[tokio::test(start_paused = true)]
async fn rare_host_calls_reach_a_running_session() {
    let options =
        <SessionOptions as lk_agents_abi::voice::session::SessionOptions>::new(option_kwargs());
    let api = running(assembled(no_room(), None, Some(options))).await;

    assert_eq!(api.resolved_options().unwrap().aec_warmup_duration, Some(3.0));
    api.interrupt(false).unwrap().await.unwrap();
    api.update_turn_options(Default::default()).unwrap();

    let words = Items(["Hello ".to_owned(), "there.".to_owned()].into());
    let streamed = api.say_stream(Box::new(words), None, true, None).unwrap();
    HostSpeech::wait_until_created(&*streamed).await.unwrap();
    let frames = Items(Default::default());
    let recorded = api.say_audio("Hi.".into(), Box::new(frames), None, true, None).unwrap();
    HostSpeech::wait_until_created(&*recorded).await.unwrap();
    // Outside any speech: nothing to end.
    api.end_call(String::new()).unwrap();

    assert_eq!(api.keyterms(), None);
    assert!(!api.amd_listening());
    assert!(api.usage().models.is_empty());

    api.shutdown(false);
    for _ in 0..20 {
        settle().await;
        tokio::time::advance(Duration::from_secs(1)).await;
    }
    assert!(matches!(api.send(TurnCommand::Clear), Err(SessionRefused::Closed)));
}

#[tokio::test(start_paused = true)]
async fn options_view_changes_only_after_core_acceptance() {
    use lk_agents_voice::agent_activity::options::aec_warmup_layer;
    let options =
        <SessionOptions as lk_agents_abi::voice::session::SessionOptions>::new(option_kwargs());
    let api = running(assembled(no_room(), None, Some(options))).await;
    api.send(TurnCommand::Configure(aec_warmup_layer(None))).unwrap();
    assert_eq!(api.resolved_options().unwrap().aec_warmup_duration, Some(3.0));
    let (call, answer) = api.call(Answer::idle).unwrap();
    let wait = SpeechCommand::WaitForIdle { hold: None, call: Some(call.id) };
    api.send_awaited(Origin::Host, wait, call).unwrap();
    assert!(answer.await.unwrap().is_ok());
    assert_eq!(api.resolved_options().unwrap().aec_warmup_duration, None);
    api.close(CloseReason::UserInitiated, false, None).await;
    assert!(api.send(TurnCommand::Configure(aec_warmup_layer(Some(9.0)))).is_err());
    assert_eq!(api.resolved_options().unwrap().aec_warmup_duration, None);
}

// Catches: an outbound SIP call kept the default AEC warmup, muting the callee's first words
// (`_on_room_io_participant_linked`); an application's own `aec_warmup_duration` is kept.
#[tokio::test(start_paused = true)]
async fn an_outbound_sip_call_has_no_aec_warmup_unless_the_application_set_one() {
    use lk_agents_abi::voice::room_io::ParticipantLinked;
    use lk_agents_abi::voice::room_owner::RoomIoEvent;
    async fn linked(explicit: Maybe<Option<f64>>) -> Option<f64> {
        let kwargs = SessionKwargs { aec_warmup_duration: explicit, ..option_kwargs() };
        let options = <SessionOptions as lk_agents_abi::voice::session::SessionOptions>::new(kwargs);
        let api = running(assembled(no_room(), None, Some(options))).await;
        let participant = ParticipantLinked { identity: "callee".into() };
        api.api().unwrap().send(RoomIoEvent::ParticipantLinked { participant, outbound_sip: true }).unwrap();
        let (call, answer) = api.call(Answer::idle).unwrap();
        let wait = SpeechCommand::WaitForIdle { hold: None, call: Some(call.id) };
        api.send_awaited(Origin::Host, wait, call).unwrap();
        assert!(answer.await.unwrap().is_ok());
        api.resolved_options().unwrap().aec_warmup_duration
    }
    use lk_agents_abi::voice::session_options::SessionKwargs;
    assert_eq!(linked(Maybe::NotGiven).await, None);
    assert_eq!(linked(Maybe::Given(Some(2.0))).await, Some(2.0));
}

pub(super) fn option_kwargs() -> lk_agents_abi::voice::session_options::SessionKwargs {
    use lk_agents_abi::voice::session_options::{HostValue, SessionKwargs, TurnHandlingKwargs};
    SessionKwargs {
        turn_handling: Some(TurnHandlingKwargs::default()),
        stt_context_options: HostValue::NotGiven,
        max_tool_steps: 3,
        use_tts_aligned_transcript: Maybe::NotGiven,
        tts_text_transforms: Maybe::NotGiven,
        min_consecutive_speech_delay: 0.0,
        expressive: Default::default(),
        aec_warmup_duration: Maybe::NotGiven,
        ivr_detection: false,
        user_away_timeout: Some(15.0),
        transcription_timeout: None,
        session_close_transcript_timeout: 2.0,
        conn_options: Default::default(),
    }
}

/// A remote host whose transport records whether opening was attempted after session close.
struct RemoteTransport {
    started: AtomicBool,
    closed: AtomicBool,
}

impl lk_agents_abi::voice::console::SessionTransport for RemoteTransport {
    fn start(&self) -> lk_agents_abi::BoxFuture<'_, Result<(), lk_agents_abi::errors::HostError>> {
        Box::pin(async move {
            self.started.store(true, Ordering::SeqCst);
            Ok(())
        })
    }

    fn send_message<'a>(
        &'a self,
        _message: &'a livekit_protocol::agent::AgentSessionMessage,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), lk_agents_abi::errors::HostError>> {
        Box::pin(async { Ok(()) })
    }

    fn recv(
        &self,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<
            Option<livekit_protocol::agent::AgentSessionMessage>,
            lk_agents_abi::errors::HostError,
        >,
    > {
        Box::pin(std::future::pending())
    }

    fn close(&self) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(async move { self.closed.store(true, Ordering::SeqCst) })
    }
}

struct UnusedRemotePort;

impl lk_agents_abi::worker::remote_session::SessionPort for UnusedRemotePort {
    fn chat_history(
        &self,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Vec<lk_agents_abi::ChatItem>, lk_agents_abi::errors::HostError>,
    > {
        Box::pin(async move { unreachable!("the lifecycle test sends no remote requests") })
    }
    fn agent_info(
        &self,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<lk_agents_abi::worker::remote_session::AgentInfo, lk_agents_abi::errors::HostError>,
    > {
        Box::pin(async move { unreachable!("the lifecycle test sends no remote requests") })
    }
    fn run_input<'a>(
        &'a self,
        _text: &'a str,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<lk_agents_abi::worker::remote_session::RunOutcome, lk_agents_abi::errors::HostError>,
    > {
        Box::pin(async move { unreachable!("the lifecycle test sends no remote requests") })
    }
    fn session_state(
        &self,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<
            lk_agents_abi::worker::remote_session::SessionState,
            lk_agents_abi::errors::HostError,
        >,
    > {
        Box::pin(async move { unreachable!("the lifecycle test sends no remote requests") })
    }
    fn rtc_stats(
        &self,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<
            Option<lk_agents_abi::worker::remote_session::RtcStats>,
            lk_agents_abi::errors::HostError,
        >,
    > {
        Box::pin(async move { unreachable!("the lifecycle test sends no remote requests") })
    }
    fn session_usage(
        &self,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<lk_agents_abi::usage::Usage, lk_agents_abi::errors::HostError>,
    > {
        Box::pin(async move { unreachable!("the lifecycle test sends no remote requests") })
    }
    fn framework_info(&self) -> (String, String) {
        ("rust".to_owned(), "test".to_owned())
    }
    fn update_io(
        &self,
        _update: lk_agents_abi::worker::remote_session::IoUpdate,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(), lk_agents_abi::errors::HostError>> {
        Box::pin(async move { unreachable!("the lifecycle test sends no remote requests") })
    }
    fn finalize_simulation(
        &self,
        _verdict: lk_agents_abi::worker::remote_session::SimulationVerdict,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<
            lk_agents_abi::worker::remote_session::VerdictOutcome,
            lk_agents_abi::errors::HostError,
        >,
    > {
        Box::pin(async move { unreachable!("the lifecycle test sends no remote requests") })
    }
}

#[tokio::test(start_paused = true)]
async fn close_during_opening_retires_only_the_accepted_remote_host() {
    let api = unobserved(no_room(), None);
    let make_host = || {
        let transport = Arc::new(RemoteTransport {
            started: AtomicBool::new(false),
            closed: AtomicBool::new(false),
        });
        let host = Arc::new(remote_session::driver::RemoteSessionLink::new(
            Arc::new(UnusedRemotePort),
            transport.clone(),
            None,
            api.time().clone(),
            Arc::new(lk_agents_net::task::TokioSpawner::current()),
        ));
        (host, transport)
    };
    let (accepted, accepted_transport) = make_host();
    let (refused, refused_transport) = make_host();
    let opening = api
        .open(
            Box::new(Executor::new(api.time().clone(), api.spawner())),
            Some(accepted.clone()),
            |_| Box::pin(async {}),
        )
        .expect("the first opening is accepted");
    assert!(matches!(
        api.open(
            Box::new(Executor::new(api.time().clone(), api.spawner())),
            Some(refused.clone()),
            |_| Box::pin(async {})
        ),
        Err(OpenRefused::AlreadyStarted),
    ));
    api.aclose("user_initiated".into(), false).unwrap().await;
    // The still-pending startup must not reopen this host after close has returned.
    accepted.start().await.unwrap();
    refused.start().await.unwrap();
    let accepted_started = accepted_transport.started.load(Ordering::SeqCst);
    let refused_started = refused_transport.started.load(Ordering::SeqCst);
    accepted.aclose().await;
    refused.aclose().await;
    drop(opening);
    assert!(!accepted_started, "closing during opening lost the accepted remote host");
    assert!(refused_started, "a refused opening closed a host the session never owned");
    assert!(refused_transport.closed.load(Ordering::SeqCst));
}

// Catches: the session holding every speech's live handle (its items, generations and the host's
// object) until it ends, which a long call grows without bound.
#[tokio::test(start_paused = true)]
async fn a_finished_speech_leaves_the_sessions_registry() {
    let api = running(unobserved(no_room(), None)).await;
    api.set_audio_output(None).unwrap();
    for _ in 0..8 {
        let speech = api.say("Done.".into(), None, true, None).unwrap();
        settle().await;
        api.time().advance(Duration::from_secs(1));
        settle().await;
        assert!(speech.done(), "the say never finished");
    }
    assert_eq!(api.speeches.len(), 0, "finished speeches stayed in the session's registry");
    api.close(CloseReason::UserInitiated, false, None).await;
}
