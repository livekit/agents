//! Typed transition outputs and private signals settled by the driver.
//! Runtime executes external effects after the core returns.

use crate::agent_activity::drain::{Drain, DrainEvent};
use crate::agent_activity::false_interruption::{AudioInterruptTrigger, TurnSettled};
use crate::agent_activity::handoff::{Handoff, HandoffEvent, HandoffId};
use crate::agent_activity::hooks::Leave;
use crate::agent_activity::scheduling::{PauseCause, SpeechId};
use crate::agent_session::{
    AgentUpdate, Answer, ClosePhase, Driver, HandoffRefused, HookCall, HookId, HostEvent, RequestId,
};
use crate::audio_recognition::{EndOfTurnInfo, InterruptionSignal, TurnDetectionMode};
use crate::events::SessionEvent;
use crate::run_result::Owned;
use crate::trace::{CoreStep, InputStep, InterruptStep, RealtimeStep, Step, ToolStep, TurnStep};
use crate::tts::markup::ChatMessageExt;
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::turn::detector::ReqId;
use lk_agents_abi::voice::events::UserState;
use lk_agents_abi::{FrameId, Ts};
use std::collections::VecDeque;
use std::sync::Arc;

pub struct Out {
    frame: FrameId,
    /// What the machines asked for, not yet performed or settled.
    todo: Vec<Planned>,
    /// What a machine writes into. Drained into `todo` by [`Out::absorb`].
    machine: Effects,
    pub(crate) signals: VecDeque<Signal>,
    journal: bool,
}

impl std::fmt::Debug for Out {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Out").field("frame", &self.frame).field("todo", &self.todo).finish()
    }
}

impl Out {
    /// `frame` is the driver's `current` at the moment the event was dequeued.
    pub fn new(frame: FrameId, journal: bool) -> Self {
        Out { frame, todo: Vec::new(), machine: Effects::new(), signals: VecDeque::new(), journal }
    }

    pub(crate) fn signal(&mut self, signal: Signal) {
        self.absorb();
        self.signals.push_back(signal);
    }

    /// A host call answered, sent when the cycle is performed.
    pub(crate) fn answer(&mut self, call: Option<RequestId>, answer: Answer) {
        if let Some(call) = call {
            self.absorb();
            self.todo.push(Planned::Effect(Effect::Answer(call, answer)));
        }
    }

    pub fn frame(&self) -> FrameId {
        self.frame
    }

    /// Where this cycle's effects are written; [`Out::take`] reads them.
    pub fn machine(&mut self) -> &mut Effects {
        &mut self.machine
    }

    /// Move the machine's effects into the cycle's to-do list, recording each as a step at the
    /// instant it was decided. Called after every transition, so the journal reads in causal
    /// order; `perform` records nothing.
    pub fn absorb(&mut self) {
        for transition in self.machine.take() {
            let planned = match transition {
                Transition::Effect(effect) => Planned::Effect(effect),
                Transition::Op(op) => Planned::Op(op),
                Transition::Signal(signal) => {
                    if self.journal {
                        let step = match &signal {
                            Signal::SessionEvent(event) => {
                                Some(Step::Emitted(event.journal_form()))
                            }
                            Signal::Recognition(decision) => Some(Step::Decided(decision.clone())),
                            _ => None,
                        };
                        if let Some(step) = step {
                            self.todo.push(Planned::Effect(Effect::Journal(Box::new(step))));
                        }
                    }
                    self.signals.push_back(signal);
                    continue;
                }
            };
            if self.journal
                && let Some(step) = journal_form(&planned)
            {
                self.todo.push(Planned::Effect(Effect::Journal(Box::new(step))));
            }
            self.todo.push(planned);
        }
    }

    /// Record a driver decision, after the effects decided before it.
    pub fn trace(&mut self, step: impl Into<Step>) {
        self.absorb();
        if self.journal {
            self.todo.push(Planned::Effect(Effect::Journal(Box::new(step.into()))));
        }
    }

    pub(crate) fn take(&mut self) -> Vec<Planned> {
        self.absorb();
        std::mem::take(&mut self.todo)
    }

    /// The timers a pending `Arm` or `Cancel` of this cycle touches, in order. Timer ops settle
    /// after the loop's timer pass, so without this a deadline cancelled by an earlier fire in the
    /// same instant would still be popped (`Driver::on_timers`).
    pub fn timers_touched(&mut self) -> Vec<Timer> {
        self.absorb();
        self.todo
            .iter()
            .filter_map(|planned| match planned {
                Planned::Op(StepOp::Arm(t, _) | StepOp::Cancel(t)) => Some(*t),
                Planned::Op(StepOp::FlushInput(_) | StepOp::Interruption(_))
                | Planned::Effect(_) => None,
            })
            .collect()
    }

    /// Whether this cycle's last `Arm` or `Cancel` of `timer` arms it; `None` when it touches none.
    pub fn arms(&mut self, timer: Timer) -> Option<bool> {
        self.absorb();
        self.todo.iter().rev().find_map(|planned| match planned {
            Planned::Op(StepOp::Arm(t, _)) if *t == timer => Some(true),
            Planned::Op(StepOp::Cancel(t)) if *t == timer => Some(false),
            _ => None,
        })
    }

    /// Whether this cycle cancels `timer` anywhere.
    pub(crate) fn cancels(&mut self, timer: Timer) -> bool {
        self.absorb();
        self.todo
            .iter()
            .any(|planned| matches!(planned, Planned::Op(StepOp::Cancel(t)) if *t == timer))
    }
}

/// The step the journal records for what a machine decided, at the instant it decided it; `None`
/// for what is journalled elsewhere.
fn journal_form(planned: &Planned) -> Option<Step> {
    let effect = match planned {
        Planned::Op(StepOp::Arm(t, at)) => {
            return Some(Step::Core(CoreStep::TimerArmed { timer: (*t).into(), at: *at }));
        }
        Planned::Op(StepOp::Cancel(t)) => {
            return Some(Step::Core(CoreStep::TimerCancelled((*t).into())));
        }
        Planned::Op(StepOp::FlushInput(StreamKind::Stt)) => {
            return Some(Step::Input(InputStep::ProviderCalled(ProviderCall::FlushStt)));
        }
        Planned::Op(StepOp::FlushInput(StreamKind::Vad)) => {
            return Some(Step::Input(InputStep::ProviderCalled(ProviderCall::FlushVad)));
        }
        Planned::Op(StepOp::Interruption(signal)) => {
            return Some(Step::Input(InputStep::DetectorCalled(DetectorCall::Signal(*signal))));
        }
        Planned::Effect(effect) => effect,
    };
    let step = match effect {
        // The journal's copy never pins a host exception (`SessionEvent::journal_form`).
        Effect::Emit(_) => return None,
        Effect::SttStreamOpen { open, stt } => {
            ProviderCall::OpenLeg { leg: StreamKind::Stt, open: *open, resource: *stt }.into()
        }
        Effect::VadStreamOpen { open, vad } => {
            ProviderCall::OpenLeg { leg: StreamKind::Vad, open: *open, resource: *vad }.into()
        }
        Effect::ProviderPrewarm(id) => ProviderCall::Prewarm(*id).into(),
        Effect::ProviderRelease(id) => ProviderCall::Release(*id).into(),
        Effect::RealtimeClose(id) => ProviderCall::CloseRealtime(*id).into(),
        Effect::EotPredict { req, .. } => DetectorCall::RequestEot { req: *req }.into(),
        Effect::EotCancel { req, timed_out } => {
            DetectorCall::CancelEot { req: *req, timed_out: *timed_out }.into()
        }
        Effect::EotFlush(reason) => DetectorCall::Flush(*reason).into(),
        Effect::EotClose => DetectorCall::Close.into(),
        Effect::Audio(AudioOp::Gate { speech, op, cause, output: _ }) => {
            let (speech, op, cause) = (*speech, *op, *cause);
            Step::Interrupt(InterruptStep::PlayoutGate { speech, op, cause })
        }
        // The source swap's own step (`AudioInputSet`, `VideoInputSet`) is the handler's.
        Effect::Audio(
            AudioOp::SetVideoInput(_)
            | AudioOp::SetAudioInput(_)
            | AudioOp::SetAudioOutput { .. }
            | AudioOp::SetTranscriptionOutput { .. }
            | AudioOp::SetOutputEnabled { .. },
        ) => return None,
        Effect::Journal(_)
        | Effect::Answer(..)
        | Effect::Host(_)
        | Effect::ChatContext(_)
        | Effect::StorageCommit { .. }
        | Effect::StorageAbort(_)
        | Effect::Outputs(_)
        | Effect::Tap(_)
        | Effect::SessionView(_)
        | Effect::Closing
        | Effect::SttConversationItem { .. }
        | Effect::SpeechStart { .. }
        | Effect::PushKeyterms { .. }
        | Effect::RequestKeyterms { .. }
        | Effect::CancelKeyterms(_)
        | Effect::AttachRecorder { .. }
        | Effect::DetachRecorder(_)
        | Effect::FlushRecorder(_)
        | Effect::RealtimePlayer { .. }
        | Effect::CloseRealtimePlayer(_)
        | Effect::CloseRealtimePlayers
        | Effect::CancelSpeech(_)
        | Effect::Speech(..)
        | Effect::FlushStream(_)
        | Effect::InterruptionSignal { .. } => return None,
        // Journalled through the data plane's reports.
        Effect::Forwarder(_) | Effect::ActivateLeg { .. } | Effect::DiscardLeg(_) => return None,
        Effect::Realtime { command: cmd, .. } => Step::Realtime(RealtimeStep::Sent {
            kind: cmd.kind(),
            req_id: cmd.req_id().map(str::to_owned),
        }),
        Effect::CancelTool { call_id, .. } => {
            Step::Tool(ToolStep::Cancelled { call_id: call_id.clone() })
        }
        Effect::ValidateOutput { .. }
        | Effect::RenderTemplate { .. }
        | Effect::AnswerExpert { .. } => return None,
    };
    Some(step)
}

impl From<ProviderCall> for Step {
    fn from(call: ProviderCall) -> Self {
        Step::Input(InputStep::ProviderCalled(call))
    }
}

impl From<DetectorCall> for Step {
    fn from(call: DetectorCall) -> Self {
        Step::Input(InputStep::DetectorCalled(call))
    }
}

/// What a step asks the runtime to do, in one ordered list, not per-kind fields: order within and
/// across kinds is inherited from the reference, where these ran inline in exactly this order.
/// What the step settles itself ([`StepOp`], [`Signal`]) never appears here.
#[derive(Debug)]
pub enum Effect {
    /// Comes back as `Input::StreamOpened{open}`: the loop never awaits `stream()`.
    SttStreamOpen {
        open: u64,
        stt: ResourceId,
    },
    VadStreamOpen {
        open: u64,
        vad: ResourceId,
    },
    /// `prewarm()`: open the provider's connection pool early.
    ProviderPrewarm(ResourceId),
    /// The core holds the provider no more.
    ProviderRelease(ResourceId),
    /// The realtime session's `aclose()`, off the loop; comes back as `RealtimeWake::Closed`.
    RealtimeClose(ResourceId),
    Emit(SessionEvent),
    Journal(Box<Step>),
    Answer(RequestId, Answer),
    Host(HostEvent),
    ChatContext(Arc<lk_agents_abi::ChatContext>),
    StorageAbort(SpeechId),
    StorageCommit {
        speech: SpeechId,
        history: Vec<lk_agents_abi::ChatItem>,
    },
    Outputs(crate::agent_session::Outputs),
    Tap(crate::agent_session::SystemTap),
    SessionView(Arc<crate::agent_session::SessionView>),
    Closing,
    SttConversationItem {
        stt: ResourceId,
        role: String,
        text: String,
    },
    SpeechStart {
        speech: SpeechId,
        context: Box<crate::generation::ReplyContext>,
        llm: Option<ResourceId>,
        tts: Option<ResourceId>,
        /// The speech takes the voice lane: the store reloads the history before the model reads
        /// it and watches the lease.
        stored: bool,
    },
    /// Per-frame `PushAudio` never comes this way.
    Realtime {
        session: ResourceId,
        command: lk_agents_abi::RealtimeCommand,
    },
    FlushStream(ResourceId),
    InterruptionSignal {
        overlap: Option<ReqId>,
        signal: InterruptionSignal,
    },
    Audio(AudioOp),
    /// The answer carries `req` back.
    EotPredict {
        req: ReqId,
        timeout: Option<std::time::Duration>,
    },
    /// `cancel_inference(timed_out=..)`: `true` only from the prediction timeout.
    EotCancel {
        req: ReqId,
        timed_out: bool,
    },
    /// `AudioRecognition._clear_user_turn`, `._bounce_eou_task`.
    EotFlush(FlushReason),
    /// A prediction still in flight answers a turn nobody will decide.
    EotClose,
    RealtimePlayer {
        speech: SpeechId,
        event: Box<lk_agents_abi::llm::realtime::Event>,
    },
    CloseRealtimePlayer(SpeechId),
    CloseRealtimePlayers,
    CancelSpeech(SpeechId),
    /// What the core published about a speech, for its live handle.
    Speech(SpeechId, crate::speech_handle::SpeechUpdate),
    AttachRecorder {
        recorder: crate::agent_session::commands::HostRecorder,
        output: Option<ResourceId>,
    },
    DetachRecorder(Option<ResourceId>),
    FlushRecorder(f64),
    PushKeyterms {
        stt: Option<ResourceId>,
        terms: Vec<String>,
        warn_unsupported: bool,
    },
    RequestKeyterms {
        llm: ResourceId,
        req: u64,
        request: Box<lk_agents_abi::ChatRequest>,
    },
    CancelKeyterms(u64),
    ActivateLeg {
        channel: crate::audio::AudioChannel,
        sink: ResourceId,
    },
    DiscardLeg(ResourceId),
    /// Sent to the data plane through the queue.
    Forwarder(crate::data_plane::ForwarderCommand),
    /// The body running under `call_id` is cancelled; the driver has told its end.
    CancelTool {
        call_id: String,
        hosted: crate::tool_executor::HostedTool,
    },
    /// The person's reply, forwarded to the delegated call's body for its expert.
    AnswerExpert {
        hosted: crate::tool_executor::HostedTool,
        answer: String,
    },
    /// The reply's agent validates an output candidate of `speech`'s run, reported as
    /// [`GenEvent::OutputValidated`](crate::generation::GenEvent::OutputValidated).
    ValidateOutput {
        speech: crate::agent_activity::scheduling::SpeechId,
        validation: lk_agents_abi::agent::run::ValidationId,
        candidate: lk_agents_abi::agent::OutputCandidate,
        hosted: crate::tool_executor::HostedTool,
    },
    /// `hosted`'s agent renders a `tool_handling` template, reported as
    /// [`GenEvent::TemplateRendered`](crate::generation::GenEvent::TemplateRendered).
    RenderTemplate {
        rendering: crate::tool_executor::Rendering,
        template: lk_agents_abi::agent::ToolTemplate,
        hosted: crate::tool_executor::HostedTool,
    },
}

/// What the step settles itself before it returns: the runtime never sees these.
#[derive(Debug)]
pub(crate) enum StepOp {
    /// Arming an armed timer *replaces* it ([`TimerSet`](lk_agents_abi::TimerSet)).
    Arm(Timer, Ts),
    /// Idempotent: cancelling an unarmed timer is a no-op, so cleanup paths cancel unconditionally.
    Cancel(Timer),
    /// Flush the open stream of that leg, when one is open.
    FlushInput(StreamKind),
    /// To the interruption detector (`_interruption_ch`), with the overlap open once the
    /// transition that raised it returned.
    Interruption(InterruptionSignal),
}

/// One entry of a step's to-do list, in the order decided.
#[derive(Debug)]
pub(crate) enum Planned {
    Effect(Effect),
    Op(StepOp),
}

#[derive(Debug, Default)]
#[must_use = "effects a transition asked for must reach the driver's out"]
pub struct Effects {
    pub(crate) transitions: Vec<Transition>,
}

#[derive(Debug)]
pub(crate) enum Transition {
    Effect(Effect),
    Op(StepOp),
    Signal(Signal),
}

impl Effects {
    pub fn new() -> Self {
        Effects::default()
    }

    pub(crate) fn take(&mut self) -> Vec<Transition> {
        std::mem::take(&mut self.transitions)
    }

    pub fn push(&mut self, e: Effect) {
        self.transitions.push(Transition::Effect(e));
    }

    /// What a component returned, after what is already here, in its order.
    pub fn append(&mut self, returned: Effects) {
        self.transitions.extend(returned.transitions);
    }

    pub fn arm(&mut self, timer: Timer, at: Ts) {
        self.transitions.push(Transition::Op(StepOp::Arm(timer, at)));
    }

    pub fn cancel(&mut self, timer: Timer) {
        self.transitions.push(Transition::Op(StepOp::Cancel(timer)));
    }

    /// The open STT or VAD stream flushed (`flush()`), when one is open.
    pub fn flush_input(&mut self, leg: StreamKind) {
        self.transitions.push(Transition::Op(StepOp::FlushInput(leg)));
    }

    pub fn emit(&mut self, ev: SessionEvent) {
        self.transitions.push(Transition::Signal(Signal::SessionEvent(ev)));
    }

    pub fn request_eot(&mut self, req: ReqId, timeout: Option<std::time::Duration>) {
        self.push(Effect::EotPredict { req, timeout });
    }

    pub fn cancel_eot(&mut self, req: ReqId, timed_out: bool) {
        self.push(Effect::EotCancel { req, timed_out });
    }

    pub fn signal(&mut self, signal: InterruptionSignal) {
        self.transitions.push(Transition::Op(StepOp::Interruption(signal)));
    }

    pub fn decide(&mut self, decision: Decision) {
        self.transitions.push(Transition::Signal(Signal::Recognition(decision)));
    }

    pub fn flush_turn_detector(&mut self, reason: FlushReason) {
        self.push(Effect::EotFlush(reason));
    }

    /// The close's edge and its `CloseLegs` phase.
    pub fn close_turn_detector(&mut self) {
        self.push(Effect::EotClose);
    }
}

/// Every deadline of the session, fired by the driver's one [`TimerSet`](lk_agents_abi::TimerSet).
/// `arm` on an armed timer replaces it (the reference's `handle.cancel(); handle =
/// call_later(..)`). Deadlines due at one instant fire in declaration order: the close's bound,
/// a waiting commit, the parked turn, the callbacks, then the rest. The scheduler's two, last,
/// fire in the scheduling pass, wherever it runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Timer {
    /// The current close phase's bound.
    Close,
    /// `commit_user_turn`'s `transcript_timeout`: the commit promotes a held interim.
    Commit,
    /// A callback the driver waits on: past it, the callback counts as failed.
    Hook(crate::agent_session::HookId),
    /// `audio_recognition._bounce_eou_task`.
    EotPredictionTimeout,
    /// `audio_recognition._bounce_eou_task`.
    EndpointingDelay,
    /// `_arm_transcription_timeout`.
    TranscriptionTimeout,
    /// `_backchannel_boundary_timer`: after agent speech onset, a backchannel verdict is ignored.
    BackchannelBoundary,
    /// `_false_interruption_timer`.
    FalseInterruption,
    /// InterruptByAudioActivity-in refused, STT hears silence.
    AecWarmup,
    /// Armed while both sides listen and no tool runs.
    UserAway,
    /// From the first speech onset since the agent last spoke; fires `UserTurnExceeded` once per
    /// crossing.
    UserTurnDuration,
    /// `RecorderIO._write_task`'s 2.5 s.
    RecorderFlush,
    /// `keyterm_detection.timeout`.
    KeytermTimeout,
    /// No `GenerationCreated` (or failure) back yet; on firing it fails with a timeout.
    RealtimeRequest(crate::agent_activity::realtime::RequestKey),
    /// `RunContext.with_filler`.
    Filler(crate::filler_scheduler::FillerId),
    /// Re-armed by every digit, cancelled while either side speaks.
    DtmfDebounce,
    /// An ended call's realtime reply to the tool result has not started yet.
    EndCallReply,
    /// An interrupted speech that never reported its end is released (`SpeechHandle._cancel`).
    InterruptionTimeout(SpeechId),
    /// `min_consecutive_speech_delay` ran out.
    ConsecutiveSpeechDelay,
    /// A speech that ended without reporting its turn: past it, its lane is released.
    LaneRelease(SpeechId),
}

/// The journal's record of a call on a provider (`InputStep::ProviderCalled`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderCall {
    FlushVad,
    FlushStt,
    /// Comes back as `Input::StreamOpened{open}`: the loop never awaits `stream()`.
    OpenLeg {
        leg: StreamKind,
        open: u64,
        resource: ResourceId,
    },
    /// `prewarm()`: open the provider's connection pool early.
    Prewarm(ResourceId),
    /// The core holds the provider no more.
    Release(ResourceId),
    /// The realtime session's `aclose()`, off the loop; comes back as `RealtimeWake::Closed`.
    CloseRealtime(ResourceId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StreamKind {
    Stt,
    Vad,
}

impl StreamKind {
    pub fn label(self) -> &'static str {
        match self {
            StreamKind::Stt => "stt",
            StreamKind::Vad => "vad",
        }
    }
}

/// The journal's record of a call on a detector (`InputStep::DetectorCalled`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorCall {
    /// `req` is the id the answer carries back.
    RequestEot { req: ReqId },
    /// `cancel_inference(timed_out=..)`: `true` only from the prediction timeout.
    CancelEot { req: ReqId, timed_out: bool },
    /// `AudioRecognition._clear_user_turn`, `._bounce_eou_task`.
    Flush(FlushReason),
    /// `_interruption_ch`.
    Signal(InterruptionSignal),
    /// A prediction still in flight answers a turn nobody will decide.
    Close,
}

/// The reference's free-form `reason`, closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlushReason {
    /// `audio_recognition._bounce_eou_task`: the user turn was accepted.
    TurnCommitted,
    /// `AudioRecognition._clear_user_turn`: the turn was dropped and its state cleared.
    ClearUserTurn,
}

impl FlushReason {
    /// So the host logs the same words the reference does.
    pub fn as_str(self) -> &'static str {
        match self {
            FlushReason::TurnCommitted => "turn committed",
            FlushReason::ClearUserTurn => "clear_user_turn",
        }
    }
}

/// The outputs are named by id, as the core decided them when it raised the op: the data plane
/// keeps no record of its own.
#[derive(Debug)]
pub enum AudioOp {
    /// Pushed only when a sink exists, so the journal's `PlayoutGate` is exactly the sink's
    /// history.
    Gate {
        speech: SpeechId,
        op: GateOp,
        cause: GateCause,
        output: ResourceId,
    },
    SetVideoInput(Option<crate::agent_session::commands::HostVideo>),
    SetAudioInput(Option<crate::agent_session::commands::HostAudio>),
    /// `AgentOutput.audio`'s setter (`io.py`): `previous` is detached, `next` attached or detached
    /// as the switch says. A reply in flight keeps the sink it started with.
    SetAudioOutput {
        previous: Option<ResourceId>,
        next: Option<ResourceId>,
        enabled: bool,
    },
    SetTranscriptionOutput {
        previous: Option<ResourceId>,
        next: Option<ResourceId>,
        enabled: bool,
    },
    /// The output stays installed; its chain is told.
    SetOutputEnabled {
        output: crate::trace::OutputKind,
        sink: Option<ResourceId>,
        enabled: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateOp {
    Pause,
    Resume,
    /// `clear_buffer()`, after a flush that closes the segment; answered with where playback
    /// stopped.
    ClearBuffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateCause {
    /// The launch hold: `Pause` before the first frame, `Resume` when the user stopped.
    Launch,
    ByAudioActivity,
    FalseInterruptionResolved,
    Interrupted,
    /// A hold that no longer applies: the parked speech became uninterruptible.
    Cancelled,
    /// Released a sink pause the facts no longer permit (sink swap, audio off).
    Released,
}

#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum Decision {
    /// `decide_commit`, tell the host, answer the machine, and maybe reply.
    EndOfTurn(EndOfTurnInfo),
    TurnSettled(TurnSettled),
    AudioInterruptGuard(AudioInterruptGuard),
    /// Still subject to `decide_interrupt`'s guards.
    InterruptByAudioActivity(AudioInterruptTrigger),
    /// Journal-only.
    EndpointingDelayUpdated {
        min_delay: std::time::Duration,
    },
    /// Even when a VAD end already closed the user's speech. Journal-only.
    TurnDeclared {
        owner: TurnDetectionMode,
    },
}

/// Guard 1 of `decide_interrupt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioInterruptGuard {
    /// `false` while the detector classifies an overlap.
    Set(bool),
    /// The session's configured default: the option, not `true`.
    Restore,
}

impl Driver {
    /// Settle private signals in FIFO order before returning external work to the runtime.
    pub(crate) fn settle_step(&mut self, mut out: Out) -> Vec<Effect> {
        let mut effects = Vec::new();
        loop {
            effects.append(&mut self.outbox);
            for planned in out.take() {
                match planned {
                    Planned::Effect(effect) => effects.push(effect),
                    Planned::Op(StepOp::Arm(timer, at)) => {
                        self.timers.arm(timer, at);
                    }
                    Planned::Op(StepOp::Cancel(timer)) => {
                        self.timers.cancel(timer);
                    }
                    Planned::Op(StepOp::FlushInput(leg)) => {
                        let channel = match leg {
                            StreamKind::Stt => crate::audio::AudioChannel::Stt,
                            StreamKind::Vad => crate::audio::AudioChannel::Vad,
                        };
                        if let Some(&sink) = self.input.sinks.get(&channel) {
                            effects.push(Effect::FlushStream(sink));
                        }
                    }
                    Planned::Op(StepOp::Interruption(signal)) => {
                        if let InterruptionSignal::OverlapEnded { .. } = signal {
                            self.session.trace.overlap_ended();
                        }
                        let overlap = self.activity.audio_recognition.open_overlap();
                        effects.push(Effect::InterruptionSignal { overlap, signal });
                    }
                }
            }
            if let Some(signal) = out.signals.pop_front() {
                self.dispatch(signal, &mut out);
                continue;
            }
            if self.settle_reply_plans(&mut out) || self.settle_deferred_calls(&mut out) {
                continue;
            }
            if !self.publish_current_speech(&mut out) {
                break;
            }
        }
        self.flush_activity();
        let speeches = self.activity.speech_scheduler.take_updates();
        effects.extend(speeches.into_iter().map(|(id, update)| Effect::Speech(id, update)));
        let launch_holds = self.launch_hold();
        let view = &self.view;
        if view.launch_holds != launch_holds
            || view.config != self.session.config
            || view.agent != self.activity.agent
            || view.vad != self.session.vad
        {
            self.view = Arc::new(crate::agent_session::SessionView {
                launch_holds,
                config: self.session.config.clone(),
                agent: self.activity.agent.clone(),
                vad: self.session.vad.clone(),
            });
            effects.push(Effect::SessionView(Arc::clone(&self.view)));
        }
        effects.append(&mut self.outbox);
        // Last, behind every event of the step that names them.
        let (runs, tools) = (&self.session.runs, &self.activity.tools);
        let named = |speech| runs.names(speech) || tools.live(speech).next().is_some();
        let retired = self.activity.speech_scheduler.take_retired(named);
        effects.extend(retired.into_iter().map(|id| Effect::Host(HostEvent::SpeechRetired(id))));
        effects
    }

    fn on_recognition_decision(&mut self, now: Ts, decision: Decision, out: &mut Out) {
        match decision {
            Decision::EndOfTurn(info) => {
                self.on_end_of_turn(now, info, out);
                // Scheduled in the same cycle: a reply created here is authorised now.
                self.pump_schedule(now, out);
            }
            // `_on_turn_settled`: a pause left in `AwaitingTurn` resolves the way the
            // false-interruption timer would have.
            Decision::TurnSettled(settled) => self.on_turn_settled(now, settled, out),
            Decision::AudioInterruptGuard(AudioInterruptGuard::Set(value)) => {
                self.session.by_audio_activity = value;
            }
            Decision::AudioInterruptGuard(AudioInterruptGuard::Restore) => {
                self.restore_audio_activity_guard();
            }
            // Through `interrupt_by_audio_activity`, so all its guards still apply. The flush is
            // skipped when the pause already ran `on_end_of_agent_speech`: running it twice would
            // shrink the `min`-folded ignore window again.
            Decision::InterruptByAudioActivity(trigger) => {
                self.interrupt_by_audio_activity(now, out, trigger);
                if let AudioInterruptTrigger::DetectorVerdict { ignore_user_transcript_until } =
                    trigger
                    && self.activity.paused_speech.paused_handle().is_none()
                {
                    let eff = self
                        .activity
                        .audio_recognition
                        .on_end_of_agent_speech(now, ignore_user_transcript_until);
                    out.machine().append(eff);
                }
            }
            Decision::EndpointingDelayUpdated { .. } | Decision::TurnDeclared { .. } => {}
        }
    }

    fn on_session_event(&mut self, now: Ts, ev: SessionEvent, out: &mut Out) {
        // `on_backchannel_confirmed`: clear the realtime model's buffered audio so a
        // "mm-hm" cannot prefix the next turn. The reference's `turn_detection not in
        // ("manual", "realtime_llm")` guard is unported: `TurnConfig::mode` is private.
        if matches!(ev, SessionEvent::BackchannelConfirmed)
            && self.session.config.effective.turn.interruption.detection_enabled()
            && self.activity.rt_session.is_some()
        {
            self.rt_send(
                lk_agents_abi::RealtimeCommand::ClearAudio(
                    lk_agents_abi::llm::realtime::ClearAudio {},
                ),
                out,
            );
        }
        // The journal names the utterance's segments by one id: the realtime item's, else
        // one minted here (the pure tier cannot). The event carries only the provider's, as
        // the reference's STT path carries none.
        if let SessionEvent::UserInputTranscribed(t) = &ev {
            let minted = self.session.ids.user_transcript_item(t.is_final);
            let id = t.item_id.clone().unwrap_or(minted);
            let text = t.ev.best_alternative().map(|a| a.text.clone()).unwrap_or_default();
            out.trace(Step::Turn(TurnStep::TranscriptSegment {
                id: id.clone(),
                text,
                is_final: t.is_final,
            }));
        }
        // The user state and away deadline are decided on the way to the host.
        self.on_session_edge(now, &ev, out);
        // The span edges this event makes go out right before it.
        let mut spans = Vec::new();
        let step = &self.step;
        let wall = |at: Ts| step.wall_secs_at(at);
        let agent = self.activity.agent.as_str();
        self.session.trace.observe(wall(now), &wall, agent, &ev, &mut spans);
        for span in spans {
            if self.journalled {
                out.trace(Step::Emitted(span.journal_form()));
            }
            out.machine().push(Effect::Emit(span));
        }
        let closed = matches!(ev, SessionEvent::Close { .. });
        if let SessionEvent::PreemptiveGeneration(pg) = &ev {
            self.on_preemptive_generation(
                now,
                pg.new_transcript.clone(),
                pg.started_speaking_at,
                out,
            );
        }
        out.machine().push(Effect::Emit(ev));
        if closed {
            self.on_session_closed();
        }
    }

    /// Speech boundaries move the user state, dated to the boundary's instant; a final transcript
    /// restores `away` or pushes the deadline out; every state edge reconciles the away timer.
    pub(crate) fn on_session_edge(&mut self, now: Ts, ev: &SessionEvent, out: &mut Out) {
        // A DTMF collection reads the same edges (and its read-back's end) off the same events.
        self.dtmf_on_event(ev, out);
        self.end_call_on_event(now, ev, out);
        match ev {
            SessionEvent::StartOfSpeech(e) => {
                out.machine()
                    .append(self.session.states.update_user_state(e.at, UserState::Speaking));
            }
            SessionEvent::EndOfSpeech(e) => {
                out.machine()
                    .append(self.session.states.update_user_state(e.at, UserState::Listening));
            }
            SessionEvent::UserInputTranscribed(t) => {
                // Any non-empty transcript is the STT recovering
                // (`AgentSession._user_input_transcribed`).
                if let Some(a) = t.ev.best_alternative() {
                    self.session.errors.on_transcript(&a.text);
                }
                if t.is_final {
                    let closing = self.session.closing_past_edge();
                    out.machine().append(self.session.states.on_final_transcript(now, closing));
                }
            }
            // The IVR navigator reads the two state edges off the feed, as `IVRActivity` does.
            SessionEvent::AgentStateChanged { .. } | SessionEvent::UserStateChanged { .. } => {
                let speaking = lk_agents_abi::voice::events::AgentState::Speaking;
                if matches!(ev, SessionEvent::AgentStateChanged { new, .. } if *new == speaking) {
                    self.turn_limit_agent_speaking(out);
                }
                let closing = self.session.closing_past_edge();
                out.machine().append(self.session.states.reconcile_away_timer(now, closing));
            }
            SessionEvent::UserTurnExceeded(e) => self.on_user_turn_exceeded(e, out),
            SessionEvent::MetricsCollected { .. }
            | SessionEvent::SpanStarted(_)
            | SessionEvent::SpanEnded(_)
            | SessionEvent::Error(_)
            | SessionEvent::OverlappingSpeech(_)
            | SessionEvent::Diagnostic(_)
            | SessionEvent::EotPrediction(_)
            | SessionEvent::AgentBackchannelOpportunity(_)
            | SessionEvent::PreemptiveGeneration(_)
            | SessionEvent::UserTranscriptionTimeout(_)
            | SessionEvent::BackchannelConfirmed
            | SessionEvent::OverlapInterruption(_)
            | SessionEvent::AgentFalseInterruption { .. }
            | SessionEvent::UserTurnCompleted(_)
            | SessionEvent::SpeechCreated { .. }
            | SessionEvent::FunctionToolsExecuted(_)
            | SessionEvent::ToolExecutionUpdated(_)
            | SessionEvent::HistoryReloaded { .. }
            | SessionEvent::SpeechDone { .. }
            | SessionEvent::FloorChanged { .. }
            | SessionEvent::AmdResult { .. }
            | SessionEvent::RunStarted { .. }
            | SessionEvent::RunItemAdded { .. }
            | SessionEvent::RunRefused { .. }
            | SessionEvent::RunSettled { .. }
            | SessionEvent::ContextCarried { .. }
            | SessionEvent::Close { .. } => {}
            // `_on_conversation_item_added`: the keyterm pass's turn gate.
            SessionEvent::ConversationItemAdded { item, .. } => {
                self.forward_chat_context(item, out);
                if self.session.keyterms.is_some() {
                    let item = (**item).clone();
                    self.keyterms_on_item(now, &item, out);
                }
            }
        }
    }

    /// `stt._push_conversation_item` on `conversation_item_added`: a message's role and text reach
    /// an STT that consumes conversation context, unless `forward_chat_context` is off.
    fn forward_chat_context(&self, item: &lk_agents_abi::ChatItem, out: &mut Out) {
        if !self.session.forward_chat_context {
            return;
        }
        let Some(message) = item.as_message() else { return };
        let Some(text) = message.text_content().filter(|t| !t.is_empty()) else { return };
        let Some(stt) = self.session.stt.as_ref() else { return };
        out.machine().push(Effect::SttConversationItem {
            stt: stt.id,
            role: message.role.as_str().to_owned(),
            text,
        });
    }

    /// `SessionEvent::FloorChanged` when `_current_speech` differs from the last announced one.
    /// Returns whether an emit was queued.
    pub(crate) fn publish_current_speech(&mut self, out: &mut Out) -> bool {
        let current = self.activity.speech_scheduler.current_speech();
        let Some(holder) = self.activity.announce_current_speech(current) else { return false };
        out.machine().emit(SessionEvent::FloorChanged { holder });
        true
    }
}

/// What one part of the session tells another within a step: raised into [`Out`], dispatched in
/// the order raised until none is left, never seen outside the step.
#[derive(Debug)]
pub(crate) enum Signal {
    SessionEvent(SessionEvent),
    Recognition(Decision),
    /// `_pause_scheduling_task` for `cause`, answered with `Paused`.
    PauseScheduling(PauseCause),
    Paused(PauseCause, PauseResult),
    /// The last speech held was released.
    Quiet,
    /// An `on_exit` ended: answered, timed out, or with nobody to ask.
    OnExitEnded(HookId),
    RunOwn(Owned),
    /// The switch's drain settled: the swap is due.
    Swap,
    RefuseHandoff {
        id: Option<HandoffId>,
        update: Box<AgentUpdate>,
        cause: HandoffRefused,
    },
    /// The switch in progress ended without a swap.
    AgentStays,
    /// Outside a switch, a held user turn reaches the agent.
    DeliverHeldTurn,
    /// The switch ended without a swap: the host's held speech goes to the agent left driving.
    ReplayHeldSpeech,
    NextHandoff,
    /// What the close's phase waits for may be over: a phase began, the trailing commit ended,
    /// the legs closed or their bound passed.
    CloseProgress,
    InterruptForDtmf,
    /// The digits read back for the caller to confirm.
    ReadBackDtmf(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PauseResult {
    Settled,
    Refused,
}

/// What a machine reads of the rest of the session, and the agent it may leave.
pub(crate) struct Cx<'a> {
    /// No speech is held.
    pub(crate) quiet: bool,
    pub(crate) closing: bool,
    pub(crate) switching: bool,
    pub(crate) leave: Leave<'a>,
}

impl Driver {
    pub(crate) fn dispatch(&mut self, signal: Signal, out: &mut Out) {
        let now = self.step.now;
        match signal {
            Signal::SessionEvent(event) => self.on_session_event(now, event, out),
            Signal::Recognition(decision) => self.on_recognition_decision(now, decision, out),
            Signal::PauseScheduling(cause) => self.pause_scheduling(now, cause, out),
            Signal::Paused(PauseCause::Handoff, result) => {
                self.handoff_step(HandoffEvent::Paused(result), out)
            }
            Signal::Paused(PauseCause::Drain, result) => {
                self.drain_step(DrainEvent::Paused(result), out)
            }
            // Settled or refused, the close's drain is over.
            Signal::Paused(PauseCause::Close, _) => {
                if matches!(self.session.close_phase(), Some(ClosePhase::Draining)) {
                    self.close_edge(now, out);
                }
            }
            Signal::Quiet => {
                self.drain_step(DrainEvent::Quiet, out);
                self.close_progress(now, out);
            }
            Signal::CloseProgress => self.close_progress(now, out),
            Signal::OnExitEnded(hook) => {
                self.activity.on_exit.ended(hook);
                self.handoff_step(HandoffEvent::OnExitEnded(hook), out);
                self.drain_step(DrainEvent::OnExitEnded(hook), out);
                self.close_exit_done(now, hook, out);
            }
            Signal::RunOwn(what) => self.run_own(what, out),
            Signal::Swap => self.perform_handoff(now, out),
            Signal::RefuseHandoff { id, update, cause } => {
                self.refuse_handoff(id, update, cause, out)
            }
            Signal::AgentStays => self.drain_step(DrainEvent::AgentStays, out),
            Signal::DeliverHeldTurn => self.deliver_held_turn(now, out),
            Signal::ReplayHeldSpeech => self.replay_held_speech(now, out),
            Signal::NextHandoff => {
                self.handoff_step(HandoffEvent::Next, out);
                self.pump_schedule(now, out);
            }
            Signal::InterruptForDtmf => self.interrupt_for_dtmf(now, out),
            Signal::ReadBackDtmf(digits) => self.read_back_dtmf(now, digits, out),
        }
    }

    /// `drain()`: under a close, its callers wait on the close's own drain.
    pub(crate) fn drain(&mut self, waiters: Vec<RequestId>, out: &mut Out) {
        if self.session.closing.is_some() {
            return self.join_close_drain(waiters, out);
        }
        // A VoiceTask holds the floor: its parent's tool waits on it, so nothing would settle.
        if self.activity.voice_tasks.running() {
            let refused = crate::agent_session::DrainRefused::TaskRunning;
            out.trace(crate::trace::HandoffStep::Drained { refused: Some(refused) });
            for call in waiters {
                out.answer(Some(call), Answer::Drain(Err(refused)));
            }
            return;
        }
        self.drain_step(DrainEvent::Asked(waiters), out);
    }

    pub(crate) fn handoff_step(&mut self, ev: HandoffEvent, out: &mut Out) {
        let (cx, handoff, _) = self.machines();
        handoff.handle(ev, cx, out);
        self.issue_claimed_exits(out);
    }

    pub(crate) fn drain_step(&mut self, ev: DrainEvent, out: &mut Out) {
        let (cx, _, drain) = self.machines();
        drain.handle(ev, cx, out);
        self.issue_claimed_exits(out);
    }

    /// The `on_exit`s claimed in the last step, asked of the host; with none, over at once.
    pub(crate) fn issue_claimed_exits(&mut self, out: &mut Out) {
        let now = self.step.now;
        for (hook, agent) in self.activity.on_exit.take_claimed() {
            let wait = self.hook_wait();
            if !self.issue_hook(now, HookCall::OnExit { id: hook, agent }, Some(wait), out) {
                out.signal(Signal::OnExitEnded(hook));
            }
        }
    }

    fn machines(&mut self) -> (Cx<'_>, &mut Handoff, &mut Drain) {
        let closing = self.session.closing.is_some();
        let a = &mut self.activity;
        let cx = Cx {
            quiet: a.speech_scheduler.is_empty(),
            closing,
            switching: a.handoff.switching(),
            leave: Leave { on_exit: &mut a.on_exit, hooks: &mut a.hooks, agent: &a.agent },
        };
        (cx, &mut a.handoff, &mut a.drain)
    }

    /// The driving agent, for the first to leave it to claim its `on_exit`.
    pub(crate) fn leave(&mut self) -> Leave<'_> {
        let a = &mut self.activity;
        Leave { on_exit: &mut a.on_exit, hooks: &mut a.hooks, agent: &a.agent }
    }
}
