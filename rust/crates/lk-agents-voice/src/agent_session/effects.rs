//! Voice session effect declarations and their typed payloads.
//! Producer: `machine::VoiceDriver`; consumer: runtime `voice::VoiceRuntime`.
//! Ordered dispatch, collectors, and timer execution are not part of this declaration file.

use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::{Answer, RequestId, VoiceHostEvent};
use crate::audio_recognition::InterruptionSignal;
use crate::events::SessionEvent;
use crate::trace::Step;
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::turn::detector::ReqId;
use std::sync::Arc;

/// What a step asks the runtime to do, in one ordered list, not per-kind fields: order within and
/// across kinds is inherited from the reference, where these ran inline in exactly this order.
/// Driver-owned deadlines and deferred domain callbacks never cross this runtime boundary.
#[derive(Debug)]
pub enum VoiceEffect {
    /// Comes back as `VoiceEvent::StreamOpened{open}`: the loop never awaits `stream()`.
    OpenSttStream {
        open: u64,
        stt: ResourceId,
        /// The activity the stream runs under (`start_agent_activity`), as the reference's STT
        /// pump does; an `stt_node`'s own spans open there.
        span: Option<lk_agents_abi::telemetry::record::SpanContext>,
    },
    OpenVadStream {
        open: u64,
        vad: ResourceId,
    },
    /// `prewarm()`: open the provider's connection pool early.
    ProviderPrewarm(ResourceId),
    /// The core holds the provider no more.
    ProviderRelease(ResourceId),
    /// The realtime session's `aclose()`, off the loop; comes back as `RealtimeWake::Closed`.
    CloseRealtimeSession(ResourceId),
    Emit(SessionEvent),
    Journal(Box<Step>),
    Answer(RequestId, Answer),
    Host(VoiceHostEvent),
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
    StartSpeech {
        speech: SpeechId,
        context: Box<crate::generation::ReplyContext>,
        llm: Option<ResourceId>,
        tts: Option<ResourceId>,
        /// The speech takes the voice lane: the store reloads the history before the model reads
        /// it and watches the lease.
        stored: bool,
        /// `agent_turn`: the parent of the speech's nodes, realtime inference and tool calls.
        span: Option<lk_agents_abi::telemetry::record::SpanContext>,
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
        recorder: crate::agent_session::commands::RecorderHandle,
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
        /// `keyterm_detection`, the request's parent.
        span: Option<lk_agents_abi::telemetry::record::SpanContext>,
    },
    CancelKeyterms(u64),
    AdoptOpenedInput {
        channel: crate::audio::AudioChannel,
        sink: ResourceId,
    },
    DiscardOpenedInput(ResourceId),
    /// Sent to the data plane through the queue.
    UpdateAudioInputRouting(crate::data_plane::AudioInputRouterCommand),
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
        /// The speech's turn, which the callback runs under.
        turn: Option<crate::telemetry::SpanContext>,
    },
    /// Cancel one output validator, leaving unrelated callbacks and background tools running.
    CancelOutputValidation {
        speech: crate::agent_activity::scheduling::SpeechId,
        validation: lk_agents_abi::agent::run::ValidationId,
    },
    /// Ask the host to format tool-handling prompt text (duplicate handling or a deferred
    /// tool reply), not to generate speech or render a UI. Reported as
    /// [`GenEvent::ToolHandlingTemplateRendered`](crate::generation::GenEvent::ToolHandlingTemplateRendered).
    RenderToolHandlingTemplate {
        rendering: crate::tool_executor::Rendering,
        template: lk_agents_abi::agent::ToolTemplate,
        hosted: crate::tool_executor::HostedTool,
        /// The speech's turn, which the rendering runs under; `None` once it ended.
        turn: Option<crate::telemetry::SpanContext>,
    },
    /// Cancel only this call's refusal renderer. Deferred tool-reply rendering and other
    /// calls keep their independent lifetimes. Idempotent when no renderer is pending.
    CancelToolRefusal {
        speech: crate::agent_activity::scheduling::SpeechId,
        call: lk_agents_core::agent_run::CallId,
    },
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
    /// Comes back as `VoiceEvent::StreamOpened{open}`: the loop never awaits `stream()`.
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
    SetVideoInput(Option<lk_agents_abi::resources::ResourceId>),
    SetAudioInput(Option<crate::agent_session::commands::AudioInputHandle>),
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

/// Guard 1 of `decide_interrupt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioInterruptGuard {
    /// `false` while the detector classifies an overlap.
    Set(bool),
    /// The session's configured default: the option, not `true`.
    Restore,
}

impl StreamKind {
    pub fn label(self) -> &'static str {
        match self {
            StreamKind::Stt => "stt",
            StreamKind::Vad => "vad",
        }
    }
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
