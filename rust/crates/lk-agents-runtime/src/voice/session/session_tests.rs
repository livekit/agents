use lk_agents_abi::voice::speech_handle::SpeechHandle as HostSpeech;

use super::*;
use crate::room_io::FixedRoom;
use crate::voice::Executor;
use crate::voice::speech_handle::SpeechHandle;
use lk_agents_abi::errors::{HostError, ProviderError};
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::voice::events::CloseReason;
use lk_agents_abi::voice::session::{FillerSource, VoiceSession as _};
use lk_agents_abi::{Maybe, OneOrStream};
use lk_agents_voice::agent_session::HandoffRefused;
use lk_agents_voice::agent_session::commands::{ActivityCommand, HookId, HookOutcome, TurnCommand};
use lk_agents_voice::agent_session::host_events::{HookCall, HookKind};
use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId;
use lk_agents_voice::agent_session::*;
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::generation::Instructions;
use lk_agents_voice::trace::Step;
use parking_lot::Mutex;
use std::future::Future as _;
use std::time::Duration;

use std::sync::atomic::{AtomicBool, Ordering};

pub(super) fn agent_update(agent: &str, instructions: Instructions) -> AgentUpdate {
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
        activation: None,
    }
}

/// A session's room without RTC, on a virtual clock.
pub(super) fn no_room() -> FixedRoom {
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    FixedRoom { rtc: None, time: Timebase::from_virtual(clock), pool: None }
}

/// An output that never reports playback; it latches the segment its first capture was keyed to.
struct MissingPlaybackReports(
    Arc<dyn lk_agents_abi::voice::io::PlaybackTracker>,
    std::sync::OnceLock<lk_agents_abi::voice::io::PlaybackSegment>,
);

impl Default for MissingPlaybackReports {
    fn default() -> Self {
        Self(Arc::new(crate::voice::playback::Tracker::new()), Default::default())
    }
}

impl lk_agents_abi::voice::io::AudioOutput for MissingPlaybackReports {
    fn label(&self) -> &str {
        "missing playback reports"
    }
    fn tracker(&self) -> &Arc<dyn lk_agents_abi::voice::io::PlaybackTracker> {
        &self.0
    }
    fn capabilities(&self) -> lk_agents_abi::voice::io::AudioOutputCapabilities {
        Default::default()
    }
    fn capture<'a>(
        &'a self,
        _cx: &Cx,
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
pub(super) fn unobserved(room: FixedRoom, hook_bound: Option<f64>) -> VoiceSession {
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
        room_io_config: Default::default(),
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
                            VoiceHostEvent::Hook(call) => {
                                let _ = runner.api().send(Command::HookDone {
                                    id: call.id(),
                                    outcome: HookOutcome::Returned,
                                });
                            }
                            VoiceHostEvent::Session(SessionEvent::Close { .. }) => {
                                saw.store(true, Ordering::SeqCst);
                            }
                            VoiceHostEvent::Session(_)
                            | VoiceHostEvent::Room(_)
                            | VoiceHostEvent::SpeechRetired(_) => {}
                        }
                    }
                    ended.store(true, Ordering::SeqCst);
                })
            },
        )
        .expect("a ready session opens");
    opening.await.expect("the loop starts");

    // Bounded on the paused runtime clock: a close parked on the virtual clock fails here.
    tokio::time::timeout(Duration::from_secs(60), close_session(&api))
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
struct Observing(Executor, Arc<Mutex<Option<super::shared::Journal>>>);

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
        cx: lk_agents_abi::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<'a, Result<OpeningLegs, DriverError>> {
        self.0.open_legs(resources, stt, vad, cx)
    }

    fn open_realtime<'a>(
        &'a mut self,
        resources: &'a crate::voice::resources::Resources,
        model: Arc<dyn lk_agents_abi::Realtime>,
        disabled: bool,
        tools: Option<lk_agents_abi::HostRef>,
        cx: lk_agents_abi::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            (Arc<dyn lk_agents_abi::RealtimeSession>, lk_agents_abi::resources::ResourceId),
            DriverError,
        >,
    > {
        self.0.open_realtime(resources, model, disabled, tools, cx)
    }

    fn tasks(&self) -> Arc<lk_agents_net::task::TaskSet> {
        self.0.tasks()
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
                        if let VoiceHostEvent::Hook(call) = ev {
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
                        if let VoiceHostEvent::Hook(call) = ev {
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

pub(super) async fn settle() {
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }
}

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
    api.send_awaited(VoiceCommandOrigin::Host, ActivityCommand::Update(Box::new(update)), call)
        .expect("the loop runs");
    let exit = exits.recv().await.expect("the handoff issues the frame's on_exit");

    let (call, mut drain) = api.call(Answer::drain).expect("the loop runs");
    api.send_awaited(
        VoiceCommandOrigin::Host,
        ActivityCommand::Drain { call: Some(call.id) },
        call,
    )
    .expect("the loop runs");
    settle().await;
    assert!(unanswered(&mut drain), "the drain was answered while the handoff's on_exit still ran");

    answer_exit(&api, exit);
    assert_eq!(swap.await.expect("answered"), Ok(()), "the handoff lands");
    assert_eq!(drain.await.expect("answered"), Ok(()), "joined, not refused");
    assert_eq!(on_exits(&api, None), 1, "the arriving frame's on_exit ran: the drain moved frames");
    close_session(&api).await;
}

// Catches: a `drain()` refused during the close; it must wait for the close's drain.
#[tokio::test(start_paused = true)]
async fn a_drain_during_the_close_is_answered_by_the_closes_drain() {
    let (api, mut exits) = session_holding_exits().await;
    api.send(Command::Close(Close::default())).expect("the loop runs");
    let exit = exits.recv().await.expect("the close issues the frame's on_exit");

    let (call, mut drain) = api.call(Answer::drain).expect("the loop runs");
    api.send_awaited(
        VoiceCommandOrigin::Host,
        ActivityCommand::Drain { call: Some(call.id) },
        call,
    )
    .expect("the loop runs");
    settle().await;
    assert!(unanswered(&mut drain), "the drain was answered while the close's on_exit still ran");

    answer_exit(&api, exit);
    assert_eq!(drain.await.expect("answered"), Ok(()), "joined the close, not refused");
    assert_eq!(on_exits(&api, None), 1, "one on_exit for the frame, the close's");
    close_session(&api).await;
}

/// A `say(audio=)` from `origin`, its frames held by the returned sender, and its handle.
fn say_from(
    api: &Recorded,
    origin: VoiceCommandOrigin,
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
    let (words, _) = say_from(api, VoiceCommandOrigin::Host);
    settle().await;
    let (call, swap) = api.call(Answer::handoff).expect("the loop runs");
    let mut update = agent_update("next", Instructions::plain("You are next."));
    update.call = Some(call.id);
    api.send_awaited(VoiceCommandOrigin::Host, ActivityCommand::Update(Box::new(update)), call)
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
) -> Option<lk_agents_voice::agent_session::machine::activity::scheduling::Rejected> {
    api.steps().iter().find_map(|s| match s {
        Step::Speech(lk_agents_voice::trace::SpeechStep::ScheduleRejected {
            speech: s,
            reason,
        }) if *s == speech => Some(*reason),
        _ => None,
    })
}

// Catches: a host `say()` during a switch refused as draining; the arriving agent must speak it,
// before its `on_enter`.
#[tokio::test(start_paused = true)]
async fn a_host_say_during_a_handoff_is_spoken_by_the_arriving_agent_before_its_on_enter() {
    let (api, mut exits) = session_holding_exits().await;
    let (words, swap) = handoff_draining(&api, &mut exits).await;
    let (_held, handle) = say_from(&api, VoiceCommandOrigin::Host);
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
    close_session(&api).await;
}

// Catches: the leaving agent's own speech (a tool's, a hook's) carried to the arriving agent; it
// is still refused by the paused scheduler.
#[tokio::test(start_paused = true)]
async fn the_leaving_agents_own_say_during_a_handoff_is_still_refused() {
    let (api, mut exits) = session_holding_exits().await;
    let (_words, _swap) = handoff_draining(&api, &mut exits).await;
    for origin in [
        VoiceCommandOrigin::Tool {
            invocation_id: "test-run".into(),
            operation: 1,
            call_id: "call_1".into(),
        },
        VoiceCommandOrigin::Hook { hook: "on_enter".into() },
    ] {
        let (_held, handle) = say_from(&api, origin.clone());
        HostSpeech::wait_until_created(&*handle).await.expect("created at once");
        assert_eq!(
            rejected(&api, handle.id()),
            Some(lk_agents_voice::agent_session::machine::activity::scheduling::Rejected::Draining),
            "{origin:?}: {:#?}",
            api.steps()
        );
        assert!(handle.interrupted(), "{origin:?}");
        assert!(handle.done(), "refused speech waited for cleanup of work that never started");
    }
    close_session(&api).await;
}

// Catches: a held say the host interrupted before the swap played on the arriving agent anyway;
// the interrupt is kept and lands as the speech is created.
#[tokio::test(start_paused = true)]
async fn a_held_say_interrupted_before_the_swap_does_not_play() {
    let (api, mut exits) = session_holding_exits().await;
    let (words, swap) = handoff_draining(&api, &mut exits).await;
    let (_held, handle) = say_from(&api, VoiceCommandOrigin::Host);
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
    close_session(&api).await;
}

// Catches: a held say left unanswered when the close refuses the switch it waited on: the host
// awaiting it hung. It is refused as any speech into a closing session is.

#[tokio::test(start_paused = true)]
async fn a_held_say_is_refused_when_the_close_refuses_the_handoff() {
    for drain in [true, false] {
        let (api, mut exits) = session_holding_exits().await;
        let (words, swap) = handoff_draining(&api, &mut exits).await;
        let (_held, handle) = say_from(&api, VoiceCommandOrigin::Host);
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
        let closing = close_session(&api);
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
pub(super) async fn running(api: VoiceSession) -> VoiceSession {
    let opening = api
        .open(
            Box::new(Executor::new(api.time().clone(), api.spawner())),
            None,
            move |mut runner: SessionRunner| {
                Box::pin(async move {
                    while let Some(ev) = runner.next_host_event().await {
                        if let VoiceHostEvent::Hook(call) = ev {
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

impl<T: Send + 'static> lk_agents_abi::StreamOutput<T> for Items<T> {
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

impl<T: Send + 'static> lk_agents_abi::utils::stream::ProviderStream<T> for Items<T> {}

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
            FillerSource::Text("hm".into()),
            delay,
            interval,
            None,
            None,
        )
    };
    invalid(filler(-1.0, None, "speech_1"));
    invalid(filler(f64::NAN, None, "speech_1"));
    invalid(filler(0.5, Some(-1.0), "speech_1"));
    assert!(invalid(filler(0.5, None, "nope")).contains("nope"));
    let pound = lk_agents_abi::voice::workflows::dtmf::DtmfEvent::Pound;
    assert!(invalid(api.collect_dtmf(4, false, -1.0, pound)).contains("timeout"));
    assert!(api.aclose("bogus".into(), false).is_err());
    // Well-formed, but nothing runs yet.
    assert!(matches!(
        api.set_io_enabled(lk_agents_abi::voice::session::IoKind::AudioInput, true),
        Err(CommandRefused::Session(_))
    ));
}

// Catches: the host's rarer calls (a streamed or pre-synthesised `say`, the reads `session.*`
// makes) refused or wrong on a running session.
#[tokio::test(start_paused = true)]
async fn rare_host_calls_reach_a_running_session() {
    let options = <SessionOptions as lk_agents_abi::voice::session::SessionOptions>::new(
        option_kwargs(),
        Default::default(),
    );
    let api = running(assembled(no_room(), None, Some(options))).await;

    assert_eq!(api.resolved_options().unwrap().aec_warmup_duration, Some(3.0));
    api.interrupt(false).unwrap().await.unwrap();
    api.update_turn_options(Default::default()).unwrap();

    let words = Items(["Hello ".to_owned(), "there.".to_owned()].into());
    let streamed = api.say(OneOrStream::Stream(Box::new(words)), None, None, true, None).unwrap();
    HostSpeech::wait_until_created(&*streamed).await.unwrap();
    let frames = Items(Default::default());
    let recorded =
        api.say(OneOrStream::One("Hi.".into()), Some(Box::new(frames)), None, true, None).unwrap();
    HostSpeech::wait_until_created(&*recorded).await.unwrap();

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
    use lk_agents_voice::agent_session::options::aec_warmup_layer;
    let options = <SessionOptions as lk_agents_abi::voice::session::SessionOptions>::new(
        option_kwargs(),
        Default::default(),
    );
    let api = running(assembled(no_room(), None, Some(options))).await;
    api.send(TurnCommand::Configure(Box::new(aec_warmup_layer(None)))).unwrap();
    assert_eq!(api.resolved_options().unwrap().aec_warmup_duration, Some(3.0));
    let (call, answer) = api.call(Answer::idle).unwrap();
    let wait = SpeechCommand::WaitForIdle { hold: None, call: Some(call.id) };
    api.send_awaited(VoiceCommandOrigin::Host, wait, call).unwrap();
    assert!(answer.await.unwrap().is_ok());
    assert_eq!(api.resolved_options().unwrap().aec_warmup_duration, None);
    close_session(&api).await;
    assert!(api.send(TurnCommand::Configure(Box::new(aec_warmup_layer(Some(9.0))))).is_err());
    assert_eq!(api.resolved_options().unwrap().aec_warmup_duration, None);
}

// Catches: an outbound SIP call kept the default AEC warmup, muting the callee's first words;
// an application's own `aec_warmup_duration` is kept.
#[tokio::test(start_paused = true)]
async fn an_outbound_sip_call_has_no_aec_warmup_unless_the_application_set_one() {
    use lk_agents_abi::voice::room_io::ParticipantLinked;
    use lk_agents_abi::voice::room_owner::RoomIoEvent;
    async fn linked(explicit: Maybe<Option<f64>>) -> Option<f64> {
        let kwargs = SessionKwargs { aec_warmup_duration: explicit, ..option_kwargs() };
        let options = <SessionOptions as lk_agents_abi::voice::session::SessionOptions>::new(
            kwargs,
            Default::default(),
        );
        let api = running(assembled(no_room(), None, Some(options))).await;
        let participant = ParticipantLinked { identity: "callee".into() };
        api.api()
            .unwrap()
            .send(RoomIoEvent::ParticipantLinked { participant, outbound_sip: true })
            .unwrap();
        let (call, answer) = api.call(Answer::idle).unwrap();
        let wait = SpeechCommand::WaitForIdle { hold: None, call: Some(call.id) };
        api.send_awaited(VoiceCommandOrigin::Host, wait, call).unwrap();
        assert!(answer.await.unwrap().is_ok());
        api.resolved_options().unwrap().aec_warmup_duration
    }
    use lk_agents_abi::voice::session_options::SessionKwargs;
    assert_eq!(linked(Maybe::NotGiven).await, None);
    assert_eq!(linked(Maybe::Given(Some(2.0))).await, Some(2.0));
}

pub(super) fn option_kwargs() -> lk_agents_abi::voice::session_options::SessionKwargs {
    use lk_agents_abi::voice::session_options::{SessionKwargs, TurnHandlingKwargs};
    SessionKwargs {
        turn_handling: Some(TurnHandlingKwargs::default()),
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
    }
}

// Catches: the session holding every speech's live handle (its items, generations and the host's
// object) until it ends, which a long call grows without bound.
#[tokio::test(start_paused = true)]
async fn a_finished_speech_leaves_the_sessions_registry() {
    let api = running(unobserved(no_room(), None)).await;
    api.set_audio_output(None).unwrap();
    for _ in 0..8 {
        let speech = api.say(OneOrStream::One("Done.".into()), None, None, true, None).unwrap();
        settle().await;
        api.time().advance(Duration::from_secs(1));
        settle().await;
        assert!(speech.done(), "the say never finished");
    }
    assert_eq!(api.speeches.len(), 0, "finished speeches stayed in the session's registry");
    close_session(&api).await;
}

// Catches: an override layer held before `start()` lost to a session build that failed, so the
// retried start ran without it, or left with the options once the session took it.
#[tokio::test(start_paused = true)]
async fn a_failed_build_leaves_the_held_override_for_the_retried_start() {
    use lk_agents_abi::voice::session::SessionOptions as _;
    use lk_agents_abi::voice::session_options::{
        EndpointingKwargs, SessionLegs, TurnHandlingKwargs, UpdateKwargs,
    };
    let options = SessionOptions::new(option_kwargs(), Default::default());
    options.override_options(UpdateKwargs {
        turn_handling: Some(TurnHandlingKwargs {
            endpointing: Some(EndpointingKwargs { min_delay: Some(0.9), ..Default::default() }),
            ..Default::default()
        }),
        ..Default::default()
    });
    // A remote session needs a room or a console: this build fails.
    let args = lk_agents_abi::voice::session::VoiceSessionArgs {
        agent: "agent".into(),
        room_io_config: Default::default(),
        remote_session: Some(Arc::new(UnusedRemotePort)),
        ..Default::default()
    };
    let room = Arc::new(no_room()) as Arc<dyn lk_agents_abi::voice::room_owner::SessionRoom>;
    let env = lk_agents_net::env::ProcessEnv::current_runtime();
    let failed = <VoiceSession as lk_agents_abi::voice::session::VoiceSession>::new(
        &env,
        room,
        args,
        Some(options.clone()),
    );
    assert!(failed.is_err());

    let api = assembled(no_room(), None, Some(options.clone()));
    assert_eq!(api.resolved_options().unwrap().endpointing.min_delay, 0.9);
    let legs = SessionLegs { stt: true, vad: true, llm: true, tts: true, ..SessionLegs::none() };
    assert_eq!(options.resolved(legs).endpointing.min_delay, 0.5);
}

/// A camera that says whether the session reads it.
struct Camera(AtomicBool);

impl lk_agents_abi::voice::io::VideoInput for Camera {
    fn label(&self) -> &str {
        "camera"
    }
    fn next_frame(
        &self,
        _cx: &Cx,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        lk_agents_abi::voice::io::Result<Option<lk_agents_abi::voice::rtc::VideoFrameRef>>,
    > {
        Box::pin(std::future::pending())
    }
    fn on_attached(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
    fn on_detached(&self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

// Catches: the camera unset and set again before the core took the unset. The same object kept
// its id, which the unset's step released, so the set named an id the table no longer had and
// the camera was silently never read again.
#[tokio::test(start_paused = true)]
async fn a_camera_unset_and_set_again_at_once_is_read() {
    let api = running(unobserved(no_room(), None)).await;
    let camera = Arc::new(Camera(AtomicBool::new(false)));
    api.set_video_input(Some(camera.clone())).unwrap();
    settle().await;
    assert!(camera.0.load(Ordering::SeqCst), "the camera is read");
    api.set_video_input(None).unwrap();
    api.set_video_input(Some(camera.clone())).unwrap();
    settle().await;
    assert!(camera.0.load(Ordering::SeqCst), "the camera set again is not read");
    api.shutdown(false);
}

#[derive(Default)]
struct Microphone {
    attached: AtomicBool,
    reading: AtomicBool,
}

impl lk_agents_abi::voice::io::AudioInput for Microphone {
    fn label(&self) -> &str {
        "microphone"
    }
    fn next_frame(
        &self,
        _cx: &Cx,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        lk_agents_abi::voice::io::Result<Option<lk_agents_abi::AudioFrame>>,
    > {
        self.reading.store(true, Ordering::SeqCst);
        Box::pin(std::future::pending())
    }
    fn on_attached(&self) {
        self.attached.store(true, Ordering::SeqCst);
    }
    fn on_detached(&self) {
        self.attached.store(false, Ordering::SeqCst);
    }
}

#[tokio::test(start_paused = true)]
async fn microphone_replacement_transfers_ownership_to_the_queued_reader() {
    let api = running(unobserved(no_room(), None)).await;
    let first = Arc::new(Microphone::default());
    let second = Arc::new(Microphone::default());
    api.set_audio_input(Some(first.clone())).unwrap();
    settle().await;
    assert!(first.attached.load(Ordering::SeqCst), "the admitted microphone was released");
    assert!(first.reading.load(Ordering::SeqCst), "the microphone's reader never started");
    api.set_audio_input(Some(second.clone())).unwrap();
    settle().await;
    assert!(!first.attached.load(Ordering::SeqCst), "replacement did not detach the old source");
    assert!(second.attached.load(Ordering::SeqCst));
    assert!(second.reading.load(Ordering::SeqCst));
    api.set_audio_input(None).unwrap();
    settle().await;
    assert!(!second.attached.load(Ordering::SeqCst));
    assert!(api.resources.live().iter().all(|(_, kind)| *kind != "audio input"));
    api.aclose("user_initiated".into(), false).unwrap().await;
}

pub(super) struct UnusedRemotePort;

impl lk_agents_abi::worker::remote_session::SessionPort for UnusedRemotePort {
    fn chat_history(
        &self,
        _cx: &Cx,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Vec<lk_agents_abi::ChatItem>, HostError>> {
        Box::pin(async move { unreachable!("the lifecycle test sends no remote requests") })
    }
    fn agent_info(
        &self,
        _cx: &Cx,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<lk_agents_abi::worker::remote_session::AgentInfo, HostError>,
    > {
        Box::pin(async move { unreachable!("the lifecycle test sends no remote requests") })
    }
    fn run_input<'a>(
        &'a self,
        _cx: &Cx,
        _text: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Vec<lk_agents_abi::ChatItem>, HostError>> {
        Box::pin(async move { unreachable!("the lifecycle test sends no remote requests") })
    }
    fn session_state(
        &self,
        _cx: &Cx,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<lk_agents_abi::worker::remote_session::RemoteSessionState, HostError>,
    > {
        Box::pin(async move { unreachable!("the lifecycle test sends no remote requests") })
    }
    fn rtc_stats(
        &self,
        _cx: &Cx,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Option<lk_agents_abi::worker::remote_session::RtcStats>, HostError>,
    > {
        Box::pin(async move { unreachable!("the lifecycle test sends no remote requests") })
    }
    fn session_usage(
        &self,
        _cx: &Cx,
    ) -> lk_agents_abi::BoxFuture<'_, Result<lk_agents_abi::usage::Usage, HostError>> {
        Box::pin(async move { unreachable!("the lifecycle test sends no remote requests") })
    }
    fn framework_info(&self) -> lk_agents_abi::worker::remote_session::FrameworkInfo {
        lk_agents_abi::worker::remote_session::FrameworkInfo {
            sdk: "rust".to_owned(),
            sdk_version: "test".to_owned(),
        }
    }
    fn update_io(
        &self,
        _cx: &Cx,
        _update: lk_agents_abi::worker::remote_session::IoUpdate,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async move { unreachable!("the lifecycle test sends no remote requests") })
    }
    fn finalize_simulation(
        &self,
        _cx: &Cx,
        _verdict: lk_agents_abi::worker::SimulationVerdict,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Option<lk_agents_abi::worker::SimulationVerdict>, HostError>,
    > {
        Box::pin(async move { unreachable!("the lifecycle test sends no remote requests") })
    }
}

pub(super) async fn close_session(api: &VoiceSession) {
    let closing = api.close(CloseReason::UserInitiated, Default::default(), None);
    tokio::pin!(closing);
    for _ in 0..30_000 {
        tokio::select! {
            biased;
            () = &mut closing => return,
            () = tokio::time::sleep(std::time::Duration::from_millis(1)) => {
                api.time().advance(std::time::Duration::from_millis(20));
            }
        }
    }
    panic!("session close did not finish within the test clock budget");
}

#[tokio::test(start_paused = true)]
async fn missing_playback_progress_and_finish_ends_the_speech_without_inventing_heard_audio() {
    let api = running(unobserved(no_room(), None)).await;
    let output = Arc::new(MissingPlaybackReports::default());
    api.set_audio_output(Some(output.clone())).unwrap();
    let frames = Items([lk_agents_abi::AudioFrame::new(vec![0; 320], 16_000, 1, 160)].into());
    let speech = api
        .say(
            OneOrStream::One("Not confirmed heard.".into()),
            Some(Box::new(frames)),
            None,
            true,
            None,
        )
        .unwrap();
    settle().await;
    assert!(output.1.get().is_some(), "the output never accepted the segment");
    // Ten milliseconds delivered, the transport's two-second allowance, then one clock tick.
    api.time().advance(Duration::from_millis(2011));
    settle().await;
    assert!(speech.done(), "missing progress left the playback finish deadline unarmed");
    assert!(speech.exception().unwrap().is_some(), "missing reports fabricated clean playback");
    assert_eq!(output.0.position(), Duration::ZERO, "unreported audio was counted as heard");
    let key = *output.1.get().expect("the sink latched its external operation key");
    assert!(output.0.has_pending_segment(key), "timeout fabricated a leaf completion");
    // A replacement speech cannot turn an unavailable output into successful completion while
    // its input remains open. The refusal is separate from interruption or normal EOF.
    let (blocked, still_open) =
        api.say_audio_channel("Still open.".into(), None, true, None).unwrap();
    settle().await;
    assert!(blocked.done(), "unavailable output did not refuse the replacement speech");
    let failure = blocked.exception().unwrap().expect("output refusal was reported as success");
    assert!(failure.to_string().contains("output admission refused"), "{failure:?}");
    drop(still_open);
    // Explicit resource cleanup settles the retained uncertain binding.
    output.0.reset();
    close_session(&api).await;
}

struct PendingTextCleanup {
    captured: AtomicBool,
    cancelled: lk_agents_net::signal::Producer<bool>,
    allow_cleanup: lk_agents_net::signal::Producer<bool>,
    completed: lk_agents_net::signal::Producer<bool>,
    retained: Mutex<Option<Box<dyn Send>>>,
}

impl Default for PendingTextCleanup {
    fn default() -> Self {
        Self {
            captured: AtomicBool::new(false),
            cancelled: lk_agents_net::signal::signal(false).0,
            allow_cleanup: lk_agents_net::signal::signal(false).0,
            completed: lk_agents_net::signal::signal(false).0,
            retained: Mutex::new(None),
        }
    }
}

struct ClosingTextNode(Arc<PendingTextCleanup>);

impl lk_agents_abi::voice::nodes::TextNode for ClosingTextNode {
    fn capture_text(&self, _: &str) {
        self.0.captured.store(true, Ordering::SeqCst);
    }
    fn capture_timed(&self, text: &lk_agents_abi::TranscriptChunk) {
        self.capture_text(&text.text);
    }
    fn flush(&self) {}
    fn retain_until_drained(&self, owner: Box<dyn Send>) {
        let mut retained = self.0.retained.lock();
        if !self.0.completed.get() {
            assert!(retained.is_none());
            *retained = Some(owner);
        } else {
            drop(retained);
            drop(owner);
        }
    }
    fn cancel(&self) {
        self.0.cancelled.set(true);
    }
    fn drain(&self) -> lk_agents_abi::BoxFuture<'_, lk_agents_abi::voice::io::Result<()>> {
        Box::pin(async move {
            let _ = self.0.completed.cell().wait_for(|done| *done).await;
            Ok(())
        })
    }
}

impl Drop for ClosingTextNode {
    fn drop(&mut self) {
        self.0.cancelled.set(true);
    }
}

struct ClosingTextFactory(Arc<PendingTextCleanup>);

impl lk_agents_abi::voice::nodes::NodeFactory for ClosingTextFactory {
    fn text_node(
        &self,
        below: Arc<dyn lk_agents_abi::voice::io::TextOutput>,
    ) -> Result<Arc<dyn lk_agents_abi::voice::nodes::TextNode>, lk_agents_abi::errors::HostError>
    {
        let state = self.0.clone();
        // The host drain owns its output until its explicitly gated source close finishes.
        tokio::spawn(async move {
            let _ = state.cancelled.cell().wait_for(|cancelled| *cancelled).await;
            let _ = state.allow_cleanup.cell().wait_for(|allowed| *allowed).await;
            below.flush();
            let owner = {
                let mut retained = state.retained.lock();
                state.completed.set(true);
                retained.take()
            };
            drop(owner);
        });
        Ok(Arc::new(ClosingTextNode(self.0.clone())))
    }
    fn llm_node<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a lk_agents_abi::ChatRequest,
    ) -> lk_agents_abi::BoxFuture<'a, Result<lk_agents_abi::LlmStream, ProviderError>> {
        Box::pin(async { Err(lk_agents_abi::errors::ApiError::connection(None).into()) })
    }
    fn tts_node(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: bool,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<
            lk_agents_abi::DuplexStream<lk_agents_abi::TtsSink, lk_agents_abi::TtsSource>,
            ProviderError,
        >,
    > {
        Box::pin(async { Err(lk_agents_abi::errors::ApiError::connection(None).into()) })
    }
    fn stt_node(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<
            lk_agents_abi::DuplexStream<lk_agents_abi::SttSink, lk_agents_abi::SttSource>,
            ProviderError,
        >,
    > {
        Box::pin(async { Err(lk_agents_abi::errors::ApiError::connection(None).into()) })
    }
}

struct TextDestination(AtomicBool);

impl lk_agents_abi::voice::io::TextOutput for TextDestination {
    fn label(&self) -> &str {
        "text destination"
    }
    fn capture_text<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, lk_agents_abi::voice::io::Result<()>> {
        Box::pin(async { Ok(()) })
    }
    fn flush(&self) {}
    fn on_detached(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[tokio::test(start_paused = true)]
async fn a_hard_aborted_generation_keeps_its_text_binding_until_source_cleanup_finishes() {
    let api = running(unobserved(no_room(), None)).await;
    api.set_audio_output(None).unwrap();
    let old = Arc::new(TextDestination(AtomicBool::new(false)));
    api.set_transcription_output(Some(old.clone())).unwrap();
    let state = Arc::new(PendingTextCleanup::default());
    let mut update = agent_update("buffering", Instructions::plain("Speak clearly."));
    update.output_nodes.transcription = Some(
        api.resources.register_transcription_node(Arc::new(ClosingTextFactory(state.clone()))),
    );
    let (call, reply) = api.call(Answer::handoff).unwrap();
    update.call = Some(call.id);
    api.send_awaited(VoiceCommandOrigin::Host, ActivityCommand::Update(Box::new(update)), call)
        .unwrap();
    assert_eq!(reply.await.unwrap(), Ok(()));
    let speech =
        api.say(OneOrStream::One("Still processing.".into()), None, None, true, None).unwrap();
    settle().await;
    assert!(
        state.captured.load(Ordering::SeqCst),
        "the text segment never started: resources={:?}, done={}, exception={:?}",
        api.resources.live(),
        speech.done(),
        speech.exception()
    );
    api.set_transcription_output(Some(Arc::new(TextDestination(AtomicBool::new(false))))).unwrap();
    settle().await;
    assert!(!old.0.load(Ordering::SeqCst), "selection retired an admitted text segment");
    // The interruption watchdog cancels the speech's tasks while the generation's drain waits.
    lk_agents_abi::voice::speech_handle::SpeechHandle::interrupt(&*speech, true).unwrap();
    api.time().advance(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
    settle().await;
    assert!(state.cancelled.get(), "hard cancellation did not reach the source owner");
    assert!(
        !old.0.load(Ordering::SeqCst),
        "the old text binding retired before its source cleanup completed"
    );
    state.allow_cleanup.set(true);
    settle().await;
    assert!(state.completed.get(), "the cancelled source did not finish cleanup");
    assert!(old.0.load(Ordering::SeqCst), "the completed old binding was never retired");
    close_session(&api).await;
}
