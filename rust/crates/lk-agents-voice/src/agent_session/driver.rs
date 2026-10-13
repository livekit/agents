//! The session's actor: the loop, its input queue and timers, the builder that assembles it, and
//! the handle a host and its peer systems hold.

use crate::agent_session::cycle::Cycle;
use crate::agent_activity::pause::{PausedSpeech, sink_facts};
use crate::agent_activity::scheduling::{SchedulingTask, SpeechId};
use crate::generation::GenEvent;
use crate::agent_session::SessionShared;
use crate::agent_session::cycle::TurnTimer;
use crate::agent_activity::{ActivityParts, DriverConfig, GateSnapshot, PIPELINE_SAMPLE_RATE};
use crate::agent_session::AgentSession;
use crate::audio::{AudioPumpParts, LegEnd, LegId, Legs, spawn_drain};
use crate::audio_recognition::AudioRecognition;
use crate::agent_session::commands::{
    Command, FrameCommand, HostAudio, HostRecorder, IvrCommand, Origin, Sent,
};
use crate::events::SessionEvent;
use crate::generation::Instructions;
use crate::agent_session::host_events::HostEvent;
use crate::speech_handle::{InputDetails, SpeechHandle};
use crate::agent_session::time::Timebase;
use crate::trace::{
    CloseAct, CloseStep, CoreStep, HandoffStep, PlayoutStep, RealtimeStep, RoomStep, SpeechStep,
    Step, Trace,
};
use lk_agents_abi::turn::detector::DetectorLeg;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::utils::frame::FrameTable;
use lk_agents_abi::voice::events::AgentState;
use lk_agents_abi::{
    FrameId, Llm, Spawner, SpawnerExt, SpeechEvent, Stt, TimerSet, Ts, Tts, Vad, VadEvent,
};
use lk_agents_rtc::RoomEvent;
use lk_agents_voice::io::{AudioInput, AudioOutput, TextOutput};
use std::sync::{Arc, Mutex};
use std::time::Duration;

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
    events: tokio::sync::mpsc::UnboundedSender<HostEvent>,
    pub trace: Arc<Mutex<Trace>>,
    pub time: crate::agent_session::Timebase,
    pub spawner: Arc<dyn lk_agents_abi::Spawner>,
    watch: WatchHub,
    legs: Arc<crate::audio::Legs>,
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

pub(super) enum Wake {
    Input(Input),
    Timer,
    Task(crate::workflows::TaskId),
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
    pub(crate) stopped: bool,
    pub(super) opened: bool,
}

impl Driver {
    /// Only while performing an effect, when no cycle exists.
    pub(crate) fn journal(&self, step: Step) {
        self.trace.lock().unwrap_or_else(|p| p.into_inner()).push(step);
    }

    pub(crate) fn ports(&self) -> Cycle {
        Cycle::new(self.current, Arc::clone(&self.trace))
    }

    /// Performed with its events on the host stream, so `start()` returns after them. Idempotent.
    pub async fn open(&mut self) {
        if std::mem::replace(&mut self.opened, true) {
            return;
        }
        let time = self.shared.time.clone();
        // The AEC warmup is *not* armed here but at the first `speaking` edge; armed now, it would
        // refuse every barge-in for the session.
        self.publish_gate();
        // The away deadline runs from start: both are silent until somebody speaks.
        {
            let now = time.now();
            let mut cycle = self.ports();
            // `start()`'s `_update_agent_state("listening")`: the host's first state edge.
            self.set_agent_state(now, AgentState::Listening, &mut cycle);
            cycle.machine().append(self.session.reconcile_away_timer(now));
            if let Some(recorder) = self.session.take_opening_recorder() {
                self.attach_recorder(now, recorder, &mut cycle);
            }
            self.issue_on_enter(now, &mut cycle);
            self.execute(cycle).await;
        }
    }

    pub async fn run(&mut self) {
        self.open().await;
        // Cloned once: the `select!` holds `&mut self.inputs`.
        let time = self.shared.time.clone();

        while !self.stopped {
            let deadline = [
                self.timers.next_deadline(),
                self.activity.scheduling_task().next_deadline(),
                self.activity.pending_turn().map(|p| p.deadline),
                self.session.close_deadline(),
                self.activity.audio_recognition().commit_deadline(),
                self.activity.hooks().next_deadline(),
            ]
            .into_iter()
            .flatten()
            .min();

            // The merge rule: a due timer fires before any same-instant input; otherwise inputs in
            // arrival order.
            let wake = if deadline.is_some_and(|at| at <= time.now()) {
                Wake::Timer
            } else {
                let task_wake = self.task_wake();
                let inputs = &mut self.inputs;
                tokio::select! {
                    biased;
                    () = time.wait_until_opt(deadline) => Wake::Timer,
                    task = async {
                        match task_wake {
                            Some(w) => w.wait().await,
                            None => std::future::pending().await,
                        }
                    } => Wake::Task(task),
                    next = inputs.recv() => match next {
                        Some(input) => Wake::Input(input),
                        None => {
                            self.stopped = true;
                            continue;
                        }
                    },
                }
            };

            let now = time.now();
            let mut cycle = self.ports();
            match wake {
                Wake::Input(Input::Stt(ev)) => self.on_stt(now, ev, &mut cycle),
                Wake::Input(Input::Vad(ev)) => self.on_vad(now, ev, &mut cycle),
                Wake::Input(Input::Room(ev)) => self.on_rtc(now, ev, &mut cycle),
                // The handoff is the one command that must *await* (reopening a leg opens a
                // stream); `on_api` is sync.
                Wake::Input(Input::Host(Sent { origin, command })) => {
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
                        self.execute(cycle).await;
                        self.publish_gate();
                        continue;
                    }
                    match command {
                        // A sequence: `on_exit`, the drain, then the swap.
                        Command::Frame(FrameCommand::Update(update)) => {
                            self.begin_handoff(now, update, &mut cycle);
                        }
                        Command::Frame(FrameCommand::UpdateModels { stt, vad, llm, tts }) => {
                            self.update_models(stt, vad, llm, tts, &mut cycle);
                        }
                        // A re-attached microphone opens a fresh STT stream, which awaits.
                        Command::Frame(FrameCommand::SetAudioInput(input)) => {
                            self.set_audio_input(
                                now,
                                input.map(|HostAudio(input)| input),
                                &mut cycle,
                            );
                        }
                        cmd => self.on_api(now, cmd, &mut cycle),
                    }
                }
                Wake::Input(Input::RoomIo(ev)) => self.on_room_io(now, ev, &mut cycle),
                Wake::Input(Input::Gen(ev)) => self.on_generation(now, ev, &mut cycle),
                Wake::Input(Input::Realtime(ev)) => self.on_realtime(now, ev, &mut cycle),
                Wake::Input(Input::Eot(req, p)) => self.on_eot(now, req, p, &mut cycle),
                Wake::Input(Input::Overlap(ev)) => self.on_overlap_verdict(now, ev, &mut cycle),
                Wake::Input(Input::LegEnded(end)) => self.on_leg_ended(now, end, &mut cycle),
                Wake::Input(Input::LegOpened(opened)) => {
                    self.on_leg_opened(now, opened, &mut cycle)
                }
                Wake::Input(Input::Video(frame)) => self.on_video_frame(now, frame, &mut cycle),
                Wake::Timer => self.on_timers(now, &mut cycle),
                Wake::Task(task) => self.on_task_wake(now, task, &mut cycle),
            }
            // Scheduled in the same cycle as the wake that changed it.
            self.pump_schedule(now, &mut cycle);
            self.execute(cycle).await;
            // The handoff's drain settled: the swap lands now, at the same instant. A loop: the swap
            // may begin another frame change whose drain settles at once.
            let mut cycle = self.ports();
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
            self.execute(cycle).await;
            // The close's next phases, each in its own cycle (the socket's `aclose` awaited here).
            self.close_advance().await;
            self.publish_gate();
        }

        self.report_undelivered_commands();
        self.settle_waiters_at_end();
        self.release_tasks_at_end();
    }

    /// A dropped queued `oneshot` would leave the caller a promise nobody fulfils. Nothing is
    /// traced when the queue is empty, so a trace mentioning this is a finding.
    pub(crate) fn report_undelivered_commands(&mut self) {
        let mut cycle = self.ports();
        while let Ok(input) = self.inputs.try_recv() {
            // Only a host command carries a promise somebody awaits.
            let Input::Host(Sent { origin: _, command }) = input else { continue };
            self.refuse_command(command, &mut cycle);
        }
        drop(cycle);
        self.run_close_at_end();
    }

    /// **Already done**, not dropped: an unattached handle would answer `done() == false` forever.
    pub(crate) fn settle_undelivered(
        &mut self,
        what: &str,
        waiting: tokio::sync::oneshot::Sender<Arc<SpeechHandle>>,
    ) {
        let id = self.session.next_speech_id();
        let handle = Arc::new(SpeechHandle::new(
            id,
            crate::speech_handle::SpeechWireId(id).to_string(),
            self.session.options().turn.interruption.enabled,
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
        self.journal(Step::Emitted(announce.clone()));
        let _ = self.events.send(HostEvent::Session(announce));
        // Born past the close's edge: the scheduling task's refusal and the close's release.
        self.journal(Step::Speech(SpeechStep::ReplyCreated { speech: id }));
        self.journal(Step::Speech(SpeechStep::ScheduleRejected {
            speech: id,
            reason: crate::agent_activity::scheduling::Rejected::Stopped,
        }));
        self.journal(Step::Close(CloseStep::Performed(CloseAct::Released { speech: id })));
        let done = SessionEvent::SpeechDone {
            speech: id,
            outcome: crate::events::SpeechOutcome::Failed,
        };
        self.journal(Step::Emitted(done.clone()));
        let _ = self.events.send(HostEvent::Session(done));
        let _ = waiting.send(handle);
    }

    pub(super) fn on_rtc(&mut self, now: Ts, ev: RoomEvent, cycle: &mut Cycle) {
        cycle.trace(RoomStep::Event(crate::trace::RoomEventKind::of(&ev)));
        match ev {
            // The room going away is the job ending (`CloseReason.JOB_SHUTDOWN`, no drain).
            RoomEvent::Close { .. } => {
                self.close_for_room(now, lk_agents_abi::voice::events::CloseReason::JobShutdown, cycle)
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
            Command::Recorder(crate::agent_session::RecorderCommand::Attach(
                HostRecorder(recorder),
            )) => self.attach_recorder(now, recorder, cycle),
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
            // nobody -- except `AddTools`, since the core owns what a request is offered.
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
        if self.activity.audio_recognition().commit_deadline().is_some_and(|at| at <= now) {
            self.perform_pending_commit(now, cycle);
        }
        // The end-of-turn hook's bound: the turn proceeds with no edits, once.
        if self.activity.pending_turn().is_some_and(|p| p.deadline <= now)
            && let Some(pending) = self.activity.take_pending_turn()
        {
            if let Some(hook) = pending.hook {
                self.activity.hooks_mut().forget(hook);
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
                    self.session.expire_aec_warmup();
                }
                TurnTimer::UserAway => {
                    cycle.machine().append(self.session.on_user_away(now));
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
                    let eff = self.activity.audio_recognition_mut().on_timer(now, timer);
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
            agent_state: session.agent_state(),
            aec_warmup_remaining: session.aec_warmup_remaining(),
            aec_warmup_armed: session.aec_warmup_armed(),
            uninterruptible_speech_active: self
                .current_handle()
                .is_some_and(|h| !h.done() && !h.interrupted() && !h.allow_interruptions()),
            discard_audio_if_uninterruptible: self
                .session
                .options()
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
        if !self.activity.audio_recognition().user_speaking() {
            return std::collections::BTreeSet::new();
        }
        self.shared
            .launch
            .launching()
            .into_iter()
            .filter(|speech| {
                self.activity
                    .scheduling_task()
                    .handle(*speech)
                    .is_some_and(|h| !h.interrupted() && h.allow_interruptions())
            })
            .collect()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DriverError {
    /// **Moved, never stringified.**
    #[error("provider: {0}")]
    Provider(#[from] ProviderError),

    #[error("rtc: {0}")]
    Rtc(#[from] lk_agents_rtc::RtcError),

    #[error("cycle: {0}")]
    Io(#[from] lk_agents_voice::io::IoError),

    #[error("resample: {0}")]
    Resample(String),

    #[error("the session has no {0}")]
    Missing(&'static str),
}

pub struct DriverBuilder {
    pub time: Timebase,
    pub spawner: Arc<dyn Spawner>,
    pub config: DriverConfig,
    pub thresholds: lk_agents_abi::tokenize::language::ThresholdOptions,
    pub agent: String,
    pub instructions: Instructions,
    /// The session starts from the agent's constructor context. Seeded before the instructions, so
    /// the system message lands in it.
    pub chat_ctx: Option<Vec<lk_agents_abi::ChatItem>>,
    /// `Some(bound)`: the hooks the sequence needs wait up to `bound`; an overrun is journalled and
    /// skipped. `None`: each hook is finished in place, and a tool's handoff cannot be described.
    pub hook_bound: Option<Duration>,
    /// Opened before the opening frame's handoff record, so that record and the opening `on_enter`
    /// are the run's.
    pub capture_run: Option<crate::run_result::RunSpec>,
    /// Consecutive unrecoverable errors of one leg tolerated before closing (default `3`).
    pub max_unrecoverable_errors: u32,
    pub llm: Option<Arc<dyn Llm>>,
    /// A TTS with no duplex path arrives wrapped in `StreamAdapter`.
    pub tts: Option<Arc<dyn Tts>>,
    pub stt: Option<Arc<dyn Stt>>,
    pub speaker_filter: Option<crate::audio::SpeakerFilterConfig>,
    /// `None`: nothing is pushed.
    pub keyterms: Option<crate::keyterm_detection::KeytermSetup>,
    pub vad: Option<Arc<dyn Vad>>,
    pub audio_in: Option<Arc<dyn AudioInput>>,
    pub video_in: Option<Arc<dyn lk_agents_voice::io::VideoInput>>,
    pub recorder: Option<Arc<dyn lk_agents_media::recorder::AudioRecorder>>,
    pub audio_out: Option<Arc<dyn AudioOutput>>,
    pub tts_text_transforms:
        Vec<lk_agents_voice::transcription::text_transforms::TransformSpec>,
    pub realtime_audio_node: Option<Arc<dyn crate::io::RealtimeAudioNode>>,
    pub text_out: Option<Arc<dyn TextOutput>>,
    pub room_events: Option<Rx<RoomEvent>>,
    pub room_io_events: Option<Rx<crate::room_io::RoomIoEvent>>,
    pub room_link: Option<crate::agent_activity::lifecycle::RoomLinkConfig>,
    /// `None` commits turns on silence alone.
    pub eot: Option<Arc<dyn lk_agents_abi::EotDetector>>,
    /// `None` is VAD-only barge-in: every burst past `min_interruption_duration`, a "mm-hm"
    /// included, stops the agent.
    pub interruption: Option<Arc<dyn crate::agent_activity::pause::InterruptionPort>>,
    /// `None` forwards **every** frame the host pushes.
    pub video_sampler: Option<Arc<lk_agents_voice::room_io::_input::VideoSampler>>,
    /// Instead of an LLM+TTS pair: `llm`/`tts` are ignored while it is set.
    pub realtime: Option<Arc<dyn lk_agents_abi::Realtime>>,
    pub toolset: crate::tool_executor::Toolset,
}

impl DriverBuilder {
    /// `async` because the provider factories must run on the host loop.
    pub async fn start(mut self) -> Result<(Driver, SessionHandle), DriverError> {
        let keyterms = self.keyterms.take();
        // An advertised tool with nothing to run it is refused here, not at the first call.
        if !self.toolset.specs.is_empty() && self.toolset.runner.is_none() {
            return Err(DriverError::Missing("tool_runner"));
        }
        if let Some(llm) = &self.llm {
            llm.prewarm();
        }
        if let Some(stt) = &self.stt {
            stt.prewarm();
        }
        if let Some(tts) = &self.tts {
            tts.prewarm();
        }
        let (queue_tx, queue_rx) = tokio::sync::mpsc::unbounded_channel::<Input>();
        let stt_tx: Tx<crate::audio::Tagged<SpeechEvent>> =
            Tx::new(queue_tx.clone(), Input::Stt);
        let vad_tx: Tx<VadEvent> = Tx::new(queue_tx.clone(), Input::Vad);
        let api_tx: Tx<Command> = Tx::new(queue_tx.clone(), host_input);
        let gen_tx: Tx<GenEvent> = Tx::new(queue_tx.clone(), Input::Gen);
        let legs_tx: Tx<LegEnd> = Tx::new(queue_tx.clone(), Input::LegEnded);
        let leg_opens: Tx<crate::room_io::LegOpened> =
            Tx::new(queue_tx.clone(), Input::LegOpened);
        let video_tx: Tx<lk_agents_rtc::VideoFrameRef> = Tx::new(queue_tx.clone(), Input::Video);
        let eot_events: Tx<lk_agents_abi::turn::detector::LegEvent<lk_agents_abi::EotPrediction>> =
            Tx::new(queue_tx.clone(), eot_input);
        let verdict_tx: Tx<
            lk_agents_abi::turn::detector::LegEvent<crate::audio_recognition::OverlapVerdict>,
        > = Tx::new(queue_tx.clone(), Input::Overlap);
        let rt_tx: Tx<crate::agent_activity::realtime::RealtimeWake> =
            Tx::new(queue_tx.clone(), Input::Realtime);
        let (ev_tx, ev_rx) = tokio::sync::mpsc::unbounded_channel::<HostEvent>();
        let watch = WatchHub::new();
        let ev_tx_for_peers = ev_tx.clone();
        // Created first so forwarders and drains can subscribe.
        let closing = tokio::sync::watch::channel(false).0;
        // Forwarders end at `Closing`, so no task outlives teardown.
        if let Some(rx) = self.room_io_events.take() {
            let tx = Tx::new(queue_tx.clone(), Input::RoomIo);
            spawn_forward(&self.spawner, rx, tx, closing.subscribe());
        }
        if let Some(rx) = self.room_events.take() {
            let tx = Tx::new(queue_tx.clone(), Input::Room);
            spawn_forward(&self.spawner, rx, tx, closing.subscribe());
        }

        // --- the one leg table: presence is read off it and nowhere else ---
        let legs = Legs::new();
        if let Some(stt) = &self.stt {
            let (sink, source) =
                stt.stream(None, None).await.ok_or(DriverError::Missing("a streaming STT"))??;
            // The speaker filter rides the STT row, a fresh detector per stream.
            let ticket =
                legs.install_stt(
                    Arc::from(sink),
                    self.speaker_filter.as_ref().map(crate::audio::speaker_filter),
                );
            spawn_drain(
                &self.spawner,
                self.time.clone(),
                closing.subscribe(),
                crate::audio::tagged(ticket, source),
                stt_tx.clone(),
                Arc::clone(&legs),
                ticket,
                legs_tx.clone(),
            );
        }

        if let Some(vad) = &self.vad {
            let (sink, source) = vad.stream().await?;
            let ticket = legs.install(LegId::Vad, Arc::from(sink));
            spawn_drain(
                &self.spawner,
                self.time.clone(),
                closing.subscribe(),
                source,
                vad_tx.clone(),
                Arc::clone(&legs),
                ticket,
                legs_tx.clone(),
            );
        }

        // --- the realtime session, opened once for the driver's life (a handoff updates it) ---
        // Detector presence has one source, the leg; with a leg and no protocol named, batch.
        {
            use crate::agent_activity::options::DetectorKind;
            let eot = self.eot.is_some();
            self.config.edit(|_, legs| {
                legs.detector = match (eot, legs.detector) {
                    (false, None | Some(DetectorKind::Batch | DetectorKind::Streaming)) => None,
                    (true, None) => Some(DetectorKind::Batch),
                    (true, Some(kind @ (DetectorKind::Batch | DetectorKind::Streaming))) => {
                        Some(kind)
                    }
                };
            });
        }
        let eot = self
            .eot
            .clone()
            .map(|d| Arc::new(lk_agents_abi::EotLeg::new(d, Arc::clone(&self.spawner))));
        if let Some(leg) = &eot {
            if let Some(audio) = leg.audio() {
                legs.install(LegId::Eot, audio);
            }
            spawn_leg_pump(&self.spawner, Arc::clone(leg), eot_events, closing.subscribe());
        }
        // The interruption detector's verdicts and audio leg come off one object, so neither
        // exists without the other.
        if let Some(detector) = self.interruption.clone() {
            legs.install(LegId::Interruption, detector.audio());
            spawn_leg_pump(&self.spawner, detector, verdict_tx, closing.subscribe());
        }

        let mut realtime_session = None;
        let mut realtime_caps = None;
        let mut realtime_disabled = None;
        let mut realtime_resolved = None;
        if let Some(model) = &self.realtime {
            // Who ends the turn, decided once from the model's capabilities and the session's
            // inputs (`_resolve_rt_turn_detection_enabled`).
            let caps = model.capabilities();
            let side = crate::agent_activity::realtime::SessionSide {
                stt: self.stt.is_some(),
                vad: self.vad.is_some(),
                detection: self.config.effective.turn.detection,
                streaming_detector: self.config.legs.detector
                    == Some(crate::agent_activity::options::DetectorKind::Streaming),
                interruption_given: self.config.effective.turn.interruption.mode.is_some(),
                turn_detection_mode: self.config.effective.turn.owner,
            };
            let disabled = crate::agent_activity::realtime::turn_detection_disabled(&caps, side);
            realtime_disabled = Some(disabled);
            // The refined mode goes into the config; the pre-refinement answer is kept so a later
            // re-resolution is compared against it.
            realtime_resolved = Some(side.turn_detection_mode);
            self.config.effective.turn.owner =
                crate::agent_activity::realtime::realtime_turn_detection_mode(&caps, side, disabled);
            let session = model.session(disabled).await?;
            realtime_caps = Some(caps);
            legs.install(
                LegId::Realtime,
                Arc::new(crate::agent_activity::realtime::RealtimeAudioSink::new(Arc::clone(
                    &session,
                ))),
            );
            // The opening frame is sent ungated by `mutable_*` (they govern mid-session changes)
            // through `push_frame` once the driver exists.
            crate::agent_activity::realtime::spawn_event_pump(
                &self.spawner,
                Arc::clone(&session),
                rt_tx.clone(),
            );
            realtime_session = Some(session);
        }

        // --- the one audio conversion, fanned out ----------------------------------------------
        let gate = GateSnapshot::new();
        // Held, so `SetAudioInput` can point a fresh pump at the same table.
        let recorder_cell = Arc::new(std::sync::RwLock::new(None));
        let pump_parts = AudioPumpParts {
            last_input: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            first_input_at: Arc::new(std::sync::atomic::AtomicU64::new(
                crate::audio::NO_INPUT_YET,
            )),
            input_enabled: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            recorder: Arc::clone(&recorder_cell),
            time: self.time.clone(),
            gate: Arc::clone(&gate),
            legs,
            rate: PIPELINE_SAMPLE_RATE,
            closing: closing.subscribe(),
            ended: legs_tx.clone(),
        };
        let audio_pump = self
            .audio_in
            .clone()
            .map(|input| self.spawner.spawn_unboxed(pump_parts.clone().pump(input)));
        // The camera, the same way: one task per source, replaced whole by `SetVideoInput`.
        let video_parts = crate::audio::VideoPumpParts {
            closing: closing.subscribe(),
            ended: legs_tx.clone(),
            frames: video_tx,
        };
        let video_pump = self
            .video_in
            .clone()
            .map(|input| self.spawner.spawn_unboxed(video_parts.clone().pump(input)));

        let shared = Arc::new(SessionShared {
            time: self.time.clone(),
            opts: self.config.effective,
            audio_out: std::sync::RwLock::new(self.audio_out.clone()),
            audio_out_enabled: std::sync::atomic::AtomicBool::new(true),
            text_out_enabled: std::sync::atomic::AtomicBool::new(true),
            recorder: recorder_cell,
            recorder_observer: std::sync::Mutex::new(None),
            launch: crate::audio::LaunchGate::new(),
            text_out: std::sync::RwLock::new(self.text_out.clone()),
            tts_transforms: self.tts_text_transforms.clone(),
            realtime_audio: self.realtime_audio_node.clone(),
            events: gen_tx,
            closing,
            draining: tokio::sync::watch::channel(false).0,
            hook_host: self.hook_bound.is_some(),
            spawner: Arc::clone(&self.spawner),
            frames: Arc::new(FrameTable::new()),
            gen_ids: std::sync::atomic::AtomicU64::new(0),
            frame: tokio::sync::watch::channel(crate::agent_session::FrameView::default())
                .0,
        });
        let tts = self.tts.clone();

        let config = self.config.clone();
        let mut audio_recognition =
            AudioRecognition::new(&config.effective, &config.legs, self.thresholds);
        let paused_speech = {
            // The sink's facts at start; re-read at every pause decision.
            let (sink, audio_enabled) = sink_facts(self.audio_out.as_deref());
            PausedSpeech::new(&config.effective, sink, audio_enabled)
        };
        let scheduling_task =
            SchedulingTask::new(config.effective.speech.min_consecutive_speech_delay);

        // `_adaptive_interruption_active` is "a detector was supplied"; without it the sentinels
        // are unreachable.
        audio_recognition.set_interruption_active(self.interruption.is_some());

        // The constructor context as the agent's view, not announced (the host has it). The prompt
        // is rendered per request.
        let session = AgentSession::new(
            config,
            crate::agent_session::SessionParts {
                llm: self.llm,
                tts,
                stt: self.stt,
                vad: self.vad,
                max_unrecoverable_errors: self.max_unrecoverable_errors,
                opening_recorder: self.recorder,
                opening_run: self.capture_run,
                room_link: self.room_link,
                agent: self.agent.clone(),
                chat_ctx: self.chat_ctx.clone(),
            },
        );

        let trace = Arc::new(Mutex::new(Trace::new(self.time.clock())));
        {
            let mut t = trace.lock().unwrap_or_else(|p| p.into_inner());
            t.push(Step::Core(CoreStep::ConfigApplied {
                layer: crate::agent_activity::options::ConfigLayer::Session,
                effective: Box::new(*session.options()),
            }));
            // The wall clock at `Ts::ZERO`: the session may already be past it.
            let since_zero = self.time.now().saturating_duration_since(Ts::ZERO);
            t.push(Step::Core(CoreStep::WallEpoch {
                at: self.time.wall().checked_sub(since_zero).unwrap_or(std::time::UNIX_EPOCH),
            }));
            if let Some(turn_detection_disabled) = realtime_disabled {
                t.push(Step::Realtime(RealtimeStep::Opened { turn_detection_disabled }));
            }
        }

        let resolved_mode = realtime_resolved.unwrap_or(session.options().turn.owner);
        audio_recognition.set_detectors(eot, self.interruption);
        let activity = crate::agent_activity::AgentActivity::new(ActivityParts {
            agent: self.agent,
            instructions: self.instructions,
            toolset: self.toolset,
            audio_recognition,
            scheduling_task,
            paused_speech,
            hook_bound: self.hook_bound,
            rt_session: realtime_session,
            realtime_caps: realtime_caps.unwrap_or_default(),
            realtime_turn_detection_disabled: realtime_disabled.unwrap_or(false),
            realtime_resolved_mode: resolved_mode,
            realtime_requested_detection: session.options().turn.detection,
        });
        let input = crate::io::AgentInput::new(
            (pump_parts, audio_pump, self.audio_in),
            (video_parts, video_pump, self.video_in),
            self.video_sampler,
            gate,
            self.speaker_filter,
        );
        let mut driver = Driver {
            session,
            activity,
            input,
            current: FrameId::ROOT,
            timers: TimerSet::new(),
            shared,
            events: ev_tx,
            inputs: queue_rx,
            trace: Arc::clone(&trace),
            watch: watch.clone(),
            realtime_tx: rt_tx,
            stt_tx,
            vad_tx,
            leg_opens,
            command_origin: Origin::Host,
            stopped: false,
            opened: false,
        };
        driver.publish_frame();

        // The first facts, in order: the gate, then the opening frame's records (the handoff from
        // nobody, the starting config), and the opening frame on the realtime socket.
        driver.publish_gate();
        {
            let mut cycle = driver.ports();
            // `start(capture_run=True)`'s run exists before the opening handoff is recorded, so the
            // handoff and the opening `on_enter` are the run's (agent_session.py:1058-1064, :1764).
            if let Some(spec) = driver.session.take_opening_run() {
                cycle.trace(CoreStep::CommandReceived {
                    origin: Origin::Host,
                    kind: "Command::Run",
                });
                let now = driver.shared.time.now();
                driver.on_run(now, spec, &mut cycle);
            }
            driver.record_frame(None, &mut cycle);
            if driver.activity.rt_session().is_some() {
                let opening = driver.session.chat_ctx().for_realtime();
                driver.push_frame(Some(opening), &mut cycle);
            }
            driver.execute(cycle).await;
        }
        // `KeytermDetector.start`: the static terms reach the recogniser now.
        if let Some(setup) = keyterms {
            let mut cycle = driver.ports();
            driver.keyterms_start(setup, &mut cycle);
            driver.execute(cycle).await;
        }
        // `ivr_detection=True` is the IVR system's to start; the core starts no system.
        let ivr_detection = driver.session.options().turn.ivr_detection;
        let peer = Peer {
            api: api_tx.clone(),
            events: ev_tx_for_peers,
            trace: Arc::clone(&trace),
            time: self.time.clone(),
            spawner: Arc::clone(&self.spawner),
            watch,
            legs: Arc::clone(driver.input.legs()),
            closing: driver.shared.closing.subscribe(),
        };

        Ok((driver, SessionHandle { api: api_tx, events: ev_rx, trace, peer, ivr_detection }))
    }
}

/// Ends on `Closed` (forwarded first), when the loop is gone, or at `Closing`. None is an error:
/// the session goes on with VAD-only barge-in.
pub(super) fn spawn_leg_pump<L>(
    spawner: &Arc<dyn Spawner>,
    leg: Arc<L>,
    tx: Tx<lk_agents_abi::turn::detector::LegEvent<L::Verdict>>,
    mut closing: tokio::sync::watch::Receiver<bool>,
) where
    L: DetectorLeg + ?Sized + 'static,
    L::Verdict: 'static,
{
    spawner.spawn_unboxed(async move {
        loop {
            let ev = tokio::select! {
                biased;
                _ = closing.wait_for(|c| *c) => break,
                ev = leg.next() => ev,
            };
            let closed = matches!(ev, lk_agents_abi::turn::detector::LegEvent::Closed);
            if tx.send(ev).is_err() || closed {
                break;
            }
        }
    });
}

fn spawn_forward<T: Send + 'static>(
    spawner: &Arc<dyn Spawner>,
    mut rx: Rx<T>,
    tx: Tx<T>,
    mut closing: tokio::sync::watch::Receiver<bool>,
) {
    spawner.spawn_unboxed(async move {
        loop {
            let ev = tokio::select! {
                biased;
                ev = rx.recv() => ev,
                _ = closing.wait_for(|c| *c) => None,
            };
            if ev.is_none_or(|ev| tx.send(ev).is_err()) {
                break;
            }
        }
    });
}

/// `Closed` is the pump's own end.
fn eot_input(ev: lk_agents_abi::turn::detector::LegEvent<lk_agents_abi::EotPrediction>) -> Input {
    use lk_agents_abi::turn::detector::{LegEvent, ReqId};
    match ev {
        LegEvent::Verdict { req, verdict } => Input::Eot(req.unwrap_or(ReqId(0)), Ok(verdict)),
        LegEvent::Error { req, recoverable: _, err } => {
            Input::Eot(req.unwrap_or(ReqId(0)), Err(err))
        }
        LegEvent::Closed => Input::LegEnded(LegEnd {
            leg: "eot",
            open: 0,
            cause: crate::trace::LegEndCause::SourceEnded,
            closed: None,
            error: None,
        }),
    }
}

/// The AMD is the host's to start.
pub fn start(peer: &Peer, ivr_detection: bool) {
    crate::ivr::ivr_activity::IvrSystem::spawn(peer.clone(), ivr_detection);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SystemGone;
