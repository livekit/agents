//! The session's actor: the loop, its input queue and timers, and the handle a host and its peer
//! systems hold.

use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::AgentSession;
use crate::agent_session::SessionShared;
use crate::agent_session::commands::{
    Answer, CallId, Close, Command, HostRecorder, IvrCommand, Origin, Sent,
};
use crate::agent_session::cycle::{Cycle, TurnTimer};
use crate::agent_session::host_events::HostEvent;
use crate::agent_session::time::Now;
use crate::audio::LegEnd;
use crate::events::SessionEvent;
use crate::generation::GenEvent;
use crate::speech_handle::{InputDetails, SpeechHandle};
use crate::trace::{
    CloseAct, CloseStep, CoreStep, HandoffStep, PlayoutStep, RoomStep, SpeechStep, Step, Trace,
};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::events::AgentState;
use lk_agents_abi::{FrameId, SpeechEvent, TimerSet, Ts, VadEvent};
use lk_agents_rtc::RoomEvent;
use std::sync::{Arc, Mutex};

/// Only what a host could read, in loop order; never machine state.
#[derive(Debug, Clone)]
pub enum Watched {
    Session(SessionEvent),
    Vad(lk_agents_abi::VadEvent),
    /// After the speaker filter.
    Stt(lk_agents_abi::SpeechEvent),
    Ivr(IvrCommand),
    CloseRequested,
}

/// One unbounded channel per subscriber; a departed one is forgotten on publish.
#[derive(Clone, Default)]
pub struct WatchHub {
    taps: Arc<Mutex<Vec<tokio::sync::mpsc::UnboundedSender<Watched>>>>,
}

impl WatchHub {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn subscribe(&self) -> tokio::sync::mpsc::UnboundedReceiver<Watched> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.taps.lock().unwrap_or_else(|p| p.into_inner()).push(tx);
        rx
    }

    /// Cheap with none: the common session runs no system.
    pub fn publish(&self, item: Watched) {
        let mut taps = self.taps.lock().unwrap_or_else(|p| p.into_inner());
        taps.retain(|tap| tap.send(item.clone()).is_ok());
    }

    pub fn is_empty(&self) -> bool {
        self.taps.lock().unwrap_or_else(|p| p.into_inner()).is_empty()
    }
}

impl std::fmt::Debug for WatchHub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WatchHub")
            .field("taps", &self.taps.lock().map(|t| t.len()).unwrap_or(0))
            .finish()
    }
}

/// Everything a peer of the host may touch, and nothing else.
#[derive(Clone)]
pub struct Peer {
    pub api: Tx<Command>,
    pub(super) events: tokio::sync::mpsc::UnboundedSender<HostEvent>,
    pub trace: Arc<Mutex<Trace>>,
    pub time: crate::agent_session::Timebase,
    pub spawner: Arc<dyn lk_agents_abi::Spawner>,
    pub(super) watch: WatchHub,
    pub(super) legs: Arc<crate::audio::Legs>,
    /// Flips at `begin_closing`: a system's own drain ends on it like the driver's do.
    pub closing: tokio::sync::watch::Receiver<bool>,
}

impl Peer {
    pub fn subscribe(&self) -> tokio::sync::mpsc::UnboundedReceiver<Watched> {
        self.watch.subscribe()
    }

    pub fn send(&self, origin: Origin, cmd: impl Into<Command>) -> Result<(), QueueClosed> {
        self.api.send_from(origin, cmd.into())
    }

    pub fn journal(&self, step: Step) {
        self.trace.lock().unwrap_or_else(|p| p.into_inner()).push(step);
    }

    pub fn emit(&self, ev: SessionEvent) {
        self.journal(Step::Emitted(ev.journal_form()));
        let _ = self.events.send(HostEvent::Session(ev));
    }

    /// The microphone's own frames, never the gate's silence. A second tap replaces the first.
    pub fn install_audio_tap(
        &self,
        sink: Arc<dyn lk_agents_abi::StreamSink<lk_agents_abi::AudioFrame>>,
    ) -> crate::audio::Ticket {
        self.legs.install(crate::audio::LegId::Amd, sink)
    }

    pub fn clear_audio_tap(&self) {
        self.legs.clear(crate::audio::LegId::Amd);
    }

    /// Forgets the row only if this install is still the one in place.
    pub fn audio_tap_ended(&self, ticket: crate::audio::Ticket) -> bool {
        self.legs.ended(ticket)
    }
}

impl std::fmt::Debug for Peer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Peer").field("time", &self.time).field("watch", &self.watch).finish()
    }
}

pub struct SessionHandle {
    pub api: Tx<Command>,
    /// Every event, in the order the driver emitted it.
    pub events: Rx<HostEvent>,
    /// Each awaited command's answer, once, by its call id.
    pub answers: Rx<(CallId, Answer)>,
    pub trace: Arc<Mutex<Trace>>,
    /// What a system runs on; the host starts the systems it configures on this after `start()`.
    pub peer: Peer,
    pub ivr_detection: bool,
}

pub(super) type Rx<T> = tokio::sync::mpsc::UnboundedReceiver<T>;

/// The loop receives one `Input` at a time in arrival order and merges it with its timers by a
/// stated rule, so the trace is a function of the input sequence.
pub struct Tx<T> {
    inner: tokio::sync::mpsc::UnboundedSender<Input>,
    wrap: fn(T) -> Input,
}

impl<T> Clone for Tx<T> {
    fn clone(&self) -> Self {
        Tx { inner: self.inner.clone(), wrap: self.wrap }
    }
}

impl<T> Tx<T> {
    pub(super) fn new(
        inner: tokio::sync::mpsc::UnboundedSender<Input>,
        wrap: fn(T) -> Input,
    ) -> Self {
        Tx { inner, wrap }
    }

    pub fn send(&self, value: T) -> Result<(), QueueClosed> {
        self.inner.send((self.wrap)(value)).map_err(|_| QueueClosed)
    }

    /// Resolves when the loop has dropped its receiver.
    pub async fn closed(&self) {
        self.inner.closed().await
    }
}

pub(super) fn host_input(command: Command) -> Input {
    Input::Host(Sent { origin: Origin::Host, command })
}

impl Tx<Command> {
    pub fn disconnected() -> Self {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        Tx::new(tx, host_input)
    }

    pub fn send_from(&self, origin: Origin, command: Command) -> Result<(), QueueClosed> {
        self.inner.send(Input::Host(Sent { origin, command })).map_err(|_| QueueClosed)
    }
}

impl<T> std::fmt::Debug for Tx<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Tx")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueClosed;

/// Timers are the loop's own.
#[derive(lk_agents_abi::Vocabulary)]
pub enum Input {
    Host(Sent),
    Stt(crate::audio::Tagged<SpeechEvent>),
    Vad(VadEvent),
    Room(RoomEvent),
    /// Forwarded from the loop so they share one arrival order.
    RoomIo(crate::room_io::RoomIoEvent),
    Gen(GenEvent),
    Realtime(crate::agent_activity::realtime::RealtimeWake),
    Eot(
        lk_agents_abi::turn::detector::ReqId,
        Result<lk_agents_abi::EotPrediction, lk_agents_abi::errors::ProviderError>,
    ),
    Overlap(lk_agents_abi::turn::detector::LegEvent<crate::audio_recognition::OverlapVerdict>),
    LegEnded(LegEnd),
    LegOpened(crate::room_io::LegOpened),
    Video(lk_agents_rtc::VideoFrameRef),
}

/// What the queue owner hands [`Driver::handle`]: an input in arrival order, or time reaching
/// [`Driver::wake_at`].
pub enum Event {
    Input(Input),
    Wake,
}

/// The session's actor: the one loop that dequeues every [`Input`] in arrival order, merges it with
/// its timers, and orchestrates the components in sequence: the [`AgentSession`], its running
/// [`AgentActivity`](crate::agent_activity::AgentActivity), and the [`AgentInput`] legs. Every
/// decision is journalled into the shared trace, dated on the session clock.
pub struct Driver {
    pub(crate) session: AgentSession,
    pub(crate) activity: crate::agent_activity::AgentActivity,
    pub(crate) input: crate::io::AgentInput,
    /// Machines never see it; [`Cycle`] carries it for them.
    pub(crate) current: FrameId,
    pub(crate) timers: TimerSet<TurnTimer>,
    /// What the spawned tasks share with the loop, the session clock among it.
    pub(crate) shared: Arc<SessionShared>,
    pub(crate) events: tokio::sync::mpsc::UnboundedSender<HostEvent>,
    pub(super) answers: tokio::sync::mpsc::UnboundedSender<(CallId, Answer)>,
    pub(super) inputs: Rx<Input>,
    pub(super) trace: Arc<Mutex<Trace>>,
    /// The core knows no subscriber.
    pub(crate) watch: WatchHub,
    /// So a player's `ToolsDone` is ordered with the socket's events.
    pub(crate) realtime_tx: Tx<crate::agent_activity::realtime::RealtimeWake>,
    pub(crate) stt_tx: Tx<crate::audio::Tagged<SpeechEvent>>,
    pub(crate) vad_tx: Tx<VadEvent>,
    pub(crate) leg_opens: Tx<crate::room_io::LegOpened>,
    /// Decides ownership (a tool's hold is its run's).
    pub(crate) command_origin: Origin,
    /// The time of the step being handled, or of the last one.
    pub(crate) step: crate::agent_session::time::StepClock,
    pub(crate) stopped: bool,
    pub(super) opened: bool,
}

impl Driver {
    /// Only while performing an effect, when no cycle exists.
    pub(crate) fn journal(&self, step: Step) {
        self.trace.lock().unwrap_or_else(|p| p.into_inner()).push(step);
    }

    /// An event sent to the host where no cycle exists, journalled first.
    pub(crate) fn emit_now(&self, ev: SessionEvent) {
        self.journal(Step::Emitted(ev.journal_form()));
        let _ = self.events.send(HostEvent::Session(ev));
    }

    pub(crate) fn cycle(&self) -> Cycle {
        Cycle::new(self.current, Arc::clone(&self.trace))
    }

    /// Performed with its events on the host stream, so `start()` returns after them. Idempotent.
    pub fn open(&mut self, now: Now) {
        if std::mem::replace(&mut self.opened, true) {
            return;
        }
        self.step.set(now);
        let now = now.at;
        // The AEC warmup is *not* armed here but at the first `speaking` edge; armed now, it would
        // refuse every barge-in for the session.
        self.publish_gate();
        // The away deadline runs from start: both are silent until somebody speaks.
        {
            let mut cycle = self.cycle();
            // `start()`'s `_update_agent_state("listening")`: the host's first state edge.
            self.set_agent_state(now, AgentState::Listening, &mut cycle);
            let closing = self.session.closing_past_edge();
            cycle.machine().append(self.session.presence.reconcile_away_timer(now, closing));
            if let Some(recorder) = self.session.opening.recorder.take() {
                self.attach_recorder(now, recorder, &mut cycle);
            }
            self.issue_on_enter(now, &mut cycle);
            self.execute(cycle);
        }
    }

    /// The queue owner: every input in arrival order, and a wake when [`Self::wake_at`] comes, a due
    /// wake before any same-instant input.
    pub async fn run(&mut self) {
        let time = self.shared.time.clone();
        self.open(time.read());
        while !self.stopped {
            let wake_at = self.wake_at();
            let event = if wake_at.is_some_and(|at| at <= time.now()) {
                Event::Wake
            } else {
                let settled = self.settled();
                tokio::select! {
                    biased;
                    () = time.wait_until_opt(wake_at) => Event::Wake,
                    () = settled => Event::Wake,
                    next = self.inputs.recv() => match next {
                        Some(input) => Event::Input(input),
                        None => break,
                    },
                }
            };
            self.handle(event, time.read());
        }
        let undelivered = std::iter::from_fn(|| self.try_next_input()).collect();
        self.finish(undelivered);
    }

    /// The next input in arrival order, if one is waiting.
    pub fn try_next_input(&mut self) -> Option<Input> {
        self.inputs.try_recv().ok()
    }

    /// The next input; `None` once every sender is gone.
    pub async fn next_input(&mut self) -> Option<Input> {
        self.inputs.recv().await
    }

    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// Answer an awaited command; `None` is a command nobody awaits.
    pub(crate) fn answer(&self, call: Option<CallId>, answer: Answer) {
        if let Some(call) = call {
            let _ = self.answers.send((call, answer));
        }
    }

    /// Resolves when the host settles the watched task on its side, a settle the core reads as a
    /// wake; pending while no task is watched.
    pub fn settled(&self) -> impl std::future::Future<Output = ()> + 'static {
        let handle = self.activity.tasks.watched().map(|t| t.handle().clone());
        async move {
            match handle {
                Some(handle) => handle.wait_settled().await,
                None => std::future::pending().await,
            }
        }
    }

    /// The loop stopped. What was queued and never delivered is refused, then the session's last
    /// event goes out, then every waiter and task the loop leaves is released: nothing settles
    /// after `close`.
    pub fn finish(&mut self, undelivered: Vec<Input>) {
        let mut cycle = self.cycle();
        for input in undelivered {
            // Only a host command carries a promise somebody awaits.
            if let Input::Host(Sent { origin: _, command }) = input {
                self.refuse_command(command, &mut cycle);
            }
        }
        drop(cycle);
        self.run_close_at_end();
        let mut cycle = self.cycle();
        if let Some((reason, error)) =
            self.session.closing.as_ref().map(|c| (c.reason, c.error.clone()))
        {
            cycle.machine().emit(SessionEvent::Close { reason, error });
        }
        self.execute(cycle);
        self.settle_waiters_at_end();
        self.release_tasks_at_end();
    }

    /// When the core next wants [`Event::Wake`]: its earliest deadline, or the last step's instant
    /// when a watched task is already due.
    pub fn wake_at(&self) -> Option<Ts> {
        let task = self.task_due().map(|_| self.step.now);
        [self.next_deadline(), task].into_iter().flatten().min()
    }

    fn next_deadline(&self) -> Option<Ts> {
        [
            self.timers.next_deadline(),
            self.activity.scheduling_task.next_deadline(),
            self.activity.pending_turn.as_ref().map(|p| p.deadline),
            self.session.close_phase().and_then(super::ClosePhase::deadline),
            self.activity.audio_recognition.commit_deadline(),
            self.activity.callbacks.next_deadline(),
        ]
        .into_iter()
        .flatten()
        .min()
    }

    /// The innermost task whose settle the loop has not seen: its handle settled, or the speech
    /// awaiting it ended.
    fn task_due(&self) -> Option<crate::workflows::TaskId> {
        let live = self.activity.tasks.watched()?;
        let speech_ended = live
            .speech()
            .is_some_and(|s| self.activity.scheduling_task.handle(s).is_none_or(|h| h.done()));
        (live.handle().done() || speech_ended).then(|| live.handle().id())
    }

    /// One step: `event` at `now`, then everything it made due in the same instant.
    pub fn handle(&mut self, event: Event, now: Now) {
        self.step.set(now);
        let now = now.at;
        let mut cycle = self.cycle();
        match event {
            Event::Wake => {
                if self.next_deadline().is_some_and(|at| at <= now) {
                    self.on_timers(now, &mut cycle);
                } else if let Some(task) = self.task_due() {
                    self.on_task_wake(now, task, &mut cycle);
                }
            }
            Event::Input(Input::Stt(ev)) => self.on_stt(now, ev, &mut cycle),
            Event::Input(Input::Vad(ev)) => self.on_vad(now, ev, &mut cycle),
            Event::Input(Input::Room(ev)) => self.on_rtc(now, ev, &mut cycle),
            Event::Input(Input::Host(Sent { origin, command })) => {
                cycle.trace(CoreStep::CommandReceived {
                    origin: origin.clone(),
                    kind: command.kind(),
                });
                self.command_origin = origin;
                // Past the close's hard edge no command takes effect: its promise is answered.
                // A `Close` joins; a hook's answer is still read.
                let late = self.session.closing_past_edge()
                    && !matches!(command, Command::Close(_) | Command::HookDone { .. });
                if late {
                    self.refuse_command(command, &mut cycle);
                    self.pump_schedule(now, &mut cycle);
                    self.execute(cycle);
                    self.publish_gate();
                    return;
                }
                self.on_api(now, command, &mut cycle);
            }
            Event::Input(Input::RoomIo(ev)) => self.on_room_io(now, ev, &mut cycle),
            Event::Input(Input::Gen(ev)) => self.on_generation(now, ev, &mut cycle),
            Event::Input(Input::Realtime(ev)) => self.on_realtime(now, ev, &mut cycle),
            Event::Input(Input::Eot(req, p)) => self.on_eot(now, req, p, &mut cycle),
            Event::Input(Input::Overlap(ev)) => self.on_overlap_verdict(now, ev, &mut cycle),
            Event::Input(Input::LegEnded(end)) => self.on_leg_ended(now, end, &mut cycle),
            Event::Input(Input::LegOpened(opened)) => self.on_leg_opened(now, opened, &mut cycle),
            Event::Input(Input::Video(frame)) => self.on_video_frame(now, frame, &mut cycle),
        }
        // Scheduled in the same cycle as the wake that changed it.
        self.pump_schedule(now, &mut cycle);
        self.execute(cycle);
        // The handoff's drain settled: the swap lands now, at the same instant. A loop: the swap
        // may begin another frame change whose drain settles at once.
        let mut cycle = self.cycle();
        while self.handoff_ready(&mut cycle) {
            self.perform_handoff(now, &mut cycle);
            self.pump_schedule(now, &mut cycle);
        }
        // The run's settle check, after the wake's own effects; a re-prompt is scheduled in
        // this cycle.
        if self.run_check(now, &mut cycle) {
            self.pump_schedule(now, &mut cycle);
        }
        self.drain_ready(&mut cycle);
        // The close's drain settled in this wake: the hard edge lands now, in its own cycle.
        self.close_ready(now, &mut cycle);
        self.settle_waiters(now, &mut cycle);
        self.execute(cycle);
        // The close's next phases, each in its own cycle (the socket's `aclose` awaited here).
        self.close_advance(now);
        self.publish_gate();
    }

    /// **Already done**, not dropped: an unattached handle would answer `done() == false` forever.
    pub(crate) fn settle_undelivered(&mut self, what: &str, call: Option<CallId>) {
        let id = self.session.ids.next_speech_id();
        let handle = Arc::new(SpeechHandle::new(
            id,
            crate::speech_handle::SpeechWireId(id).to_string(),
            self.session.config.effective.turn.interruption.enabled,
            InputDetails::default(),
        ));
        handle.mark_done(Some(lk_agents_abi::errors::ProviderError::Local(
            lk_agents_abi::errors::ApiError::new(format!(
                "the session loop ended before {what}() ran"
            )),
        )));
        // The host stream still exists, so the speech is announced and ended on it like every other.
        let kind = if what == "say" {
            crate::agent_activity::scheduling::SpeechKind::Say
        } else {
            crate::agent_activity::scheduling::SpeechKind::GenerateReply
        };
        let announce =
            SessionEvent::SpeechCreated { speech: Arc::clone(&handle), kind, user_initiated: true };
        self.emit_now(announce);
        // Born past the close's edge: the scheduling task's refusal and the close's release.
        self.journal(Step::Speech(SpeechStep::ReplyCreated { speech: id }));
        self.journal(Step::Speech(SpeechStep::ScheduleRejected {
            speech: id,
            reason: crate::agent_activity::scheduling::Rejected::Stopped,
        }));
        self.journal(Step::Close(CloseStep::Performed(CloseAct::Released { speech: id })));
        let done =
            SessionEvent::SpeechDone { speech: id, outcome: crate::events::SpeechOutcome::Failed };
        self.emit_now(done);
        self.answer(call, Answer::Speech(handle));
    }

    pub(super) fn on_rtc(&mut self, now: Ts, ev: RoomEvent, cycle: &mut Cycle) {
        cycle.trace(RoomStep::Event(crate::trace::RoomEventKind::of(&ev)));
        match ev {
            // The room going away is the job ending (`CloseReason.JOB_SHUTDOWN`, no drain).
            RoomEvent::Close { .. } => {
                let reason = lk_agents_abi::voice::events::CloseReason::JobShutdown;
                self.begin_closing(now, Close { reason, drain: false, error: None }, cycle)
            }
            // The departure arrives as the link's own edge (`ParticipantUnlinked`); the rest is
            // `RoomIo`'s.
            RoomEvent::TrackSubscribed { .. }
            | RoomEvent::TrackPublished { .. }
            | RoomEvent::TrackUnpublished { .. }
            | RoomEvent::LocalTrackPublished { .. }
            | RoomEvent::ParticipantConnected { .. }
            | RoomEvent::ParticipantDisconnected { .. }
            | RoomEvent::ConnectionStateChanged { .. } => {}
        }
    }

    pub(super) fn on_api(&mut self, now: Ts, cmd: Command, cycle: &mut Cycle) {
        match cmd {
            Command::Speech(cmd) => self.on_speech(now, cmd, cycle),
            Command::Frame(cmd) => self.on_frame(now, cmd, cycle),
            Command::Turn(cmd) => self.on_turn(now, cmd, cycle),
            Command::Video(image) => {
                self.push_video(image, cycle);
            }
            Command::Recorder(crate::agent_session::RecorderCommand::Attach(HostRecorder(
                recorder,
            ))) => self.attach_recorder(now, recorder, cycle),
            Command::Recorder(crate::agent_session::RecorderCommand::Detach) => {
                self.detach_recorder(now, cycle)
            }
            Command::Close(close) => self.begin_closing(now, close, cycle),
            Command::HookDone { id, outcome } => self.on_hook_done(now, id, outcome, cycle),
            Command::Task(cmd) => self.on_task(now, cmd, cycle),
            Command::Run(spec) => self.on_run(now, *spec, cycle),
            Command::Dtmf(cmd) => self.on_dtmf(now, cmd, cycle),
            Command::HostMetrics(record) => {
                cycle.machine().emit(crate::events::SessionEvent::MetricsCollected {
                    metrics: record,
                    speech: None,
                });
            }
            Command::SetStaticKeyterms(terms) => {
                self.keyterms_step(
                    now,
                    crate::keyterm_detection::KeytermIn::StaticTerms(terms),
                    cycle,
                );
            }
            // Addressed to the IVR navigator: journalled and relayed on the feed, acted on by
            // nobody: except `AddTools`, since the core owns what a request is offered.
            Command::Ivr(IvrCommand::AddTools(tools)) => {
                let added = self.session.extend_tools(tools);
                cycle.trace(crate::trace::ToolStep::SessionToolsExtended { added });
            }
            Command::Ivr(command) => {
                self.watch.publish(Watched::Ivr(command));
            }
        }
    }

    pub(super) fn on_timers(&mut self, now: Ts, cycle: &mut Cycle) {
        // The close's drain bound. The edge cancels every timer, so none fires in its instant.
        if self.close_tick(now, cycle) {
            return;
        }
        // `commit_user_turn`'s `transcript_timeout`: the commit promotes a held interim.
        if self.activity.audio_recognition.commit_deadline().is_some_and(|at| at <= now) {
            self.perform_pending_commit(now, cycle);
        }
        // The end-of-turn hook's bound: the turn proceeds with no edits, once.
        if self.activity.pending_turn.as_ref().is_some_and(|p| p.deadline <= now)
            && let Some(pending) = self.activity.pending_turn.take()
        {
            if let Some(hook) = pending.hook {
                self.activity.callbacks.finish(hook);
                cycle.trace(HandoffStep::HookFinished {
                    hook,
                    kind: crate::agent_session::HookKind::OnUserTurnCompleted,
                    how: crate::trace::HookEnd::TimedOut,
                });
            }
            self.finish_turn(now, pending.finished(now), Vec::new(), cycle);
            self.pump_schedule(now, cycle);
        }
        self.expire_hooks(now, cycle);
        loop {
            // Disarm deadlines an earlier fire of this pass cancelled or moved: the cancel effect
            // runs after the pass. asyncio's `_run_once` likewise skips a handle cancelled earlier
            // in the same tick.
            for touched in cycle.timers_touched() {
                self.timers.cancel(touched);
            }
            let Some(timer) = self.timers.pop_due(now) else { break };
            cycle.trace(CoreStep::TimerFired(timer, now));
            match timer {
                // `PausedSpeech` owns this one; every other `TurnTimer` is the turn machine's.
                TurnTimer::FalseInterruption => self.on_false_interruption_timeout(now, cycle),
                // `_on_aec_warmup_expired`: the one place `aec_warmup_armed` goes back to `false`
                // before close.
                TurnTimer::AecWarmup => {
                    let closing = self.session.closing_past_edge();
                    self.session.aec.expire(closing);
                }
                TurnTimer::UserAway => {
                    cycle.machine().append(self.session.presence.on_user_away(now));
                }
                TurnTimer::RecorderFlush => self.flush_recorder(now, true, cycle),
                // A peer's deadlines, fired on the peer's own set. Never here.
                TurnTimer::Amd(_) | TurnTimer::IvrSilence => {}
                TurnTimer::KeytermTimeout => self.on_keyterm_timeout(now, cycle),
                TurnTimer::DtmfDebounce => self.on_dtmf_debounce(now, cycle),
                // A realtime request the server never answered: the machine settles it.
                TurnTimer::RealtimeRequest(key) => self.realtime_request_timed_out(key, cycle),
                TurnTimer::Filler(id) => self.on_filler_timer(now, id, cycle),
                TurnTimer::EotPredictionTimeout
                | TurnTimer::EndpointingDelay
                | TurnTimer::TranscriptionTimeout
                | TurnTimer::BackchannelBoundary
                | TurnTimer::UserTurnDuration => {
                    let eff = self.activity.audio_recognition.on_timer(now, timer);
                    cycle.machine().append(eff);
                }
            }
        }
    }

    /// Journalled only when the verdict changed. Through [`Driver::journal`]: no cycle exists here.
    pub(super) fn publish_gate(&self) {
        self.shared.launch.publish(self.launch_hold());
        let session = &self.session;
        let inputs = crate::audio::GateInputs {
            agent_state: session.presence.agent_state(),
            aec_warmup_remaining: session.aec.remaining(),
            aec_warmup_armed: session.aec.armed(),
            uninterruptible_speech_active: self
                .current_handle()
                .is_some_and(|h| !h.done() && !h.interrupted() && !h.allow_interruptions()),
            discard_audio_if_uninterruptible: session
                .config
                .effective
                .turn
                .interruption
                .discard_audio_if_uninterruptible,
        };
        if let Some(discard) = self.input.publish_gate(&inputs) {
            // The silence the closing verdict produced, before the verdict that ends it.
            for (leg, frames) in self.input.legs().take_silenced() {
                if frames > 0 {
                    self.journal(Step::Playout(PlayoutStep::GateSilenced { leg, frames }));
                }
            }
            self.journal(Step::Playout(PlayoutStep::GatePublished {
                stt_real: !discard,
                realtime_real: !discard,
                vad_real: true,
                amd_real: true,
            }));
        }
    }

    /// Interruptible speeches only (an uninterruptible greeting plays over the user by design); a
    /// speech whose first frame is out is never held again.
    pub(super) fn launch_hold(&self) -> std::collections::BTreeSet<SpeechId> {
        // Read off the gate, not the scheduling task's current speech, which a block or
        // supersession can move away from a speech still parked at launch.
        if !self.activity.audio_recognition.user_speaking() {
            return std::collections::BTreeSet::new();
        }
        self.shared
            .launch
            .launching()
            .into_iter()
            .filter(|speech| {
                self.activity.scheduling_task.handle(*speech).is_some_and(|h| h.interruptible())
            })
            .collect()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DriverError {
    /// **Moved, never stringified.**
    #[error("provider: {0}")]
    Provider(#[from] ProviderError),

    #[error("resample: {0}")]
    Resample(String),

    #[error("the session has no {0}")]
    Missing(&'static str),
}

/// The IVR navigator, when the session detects IVRs; the AMD is the host's to start.
pub fn start_ivr(peer: &Peer, ivr_detection: bool) {
    crate::ivr::ivr_activity::IvrSystem::spawn(peer.clone(), ivr_detection);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SystemGone;
