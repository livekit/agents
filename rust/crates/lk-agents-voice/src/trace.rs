use crate::agent_activity::false_interruption::InterruptRefusal;
use crate::agent_activity::scheduling::{PreemptiveVerdict, SpeechId};
use crate::agent_session::effects::TurnTimer;
pub use crate::agent_session::effects::{GateCause, GateOp};
use crate::audio_recognition::CommitVerdict;
use crate::events::SessionEvent;
use lk_agents_abi::Ts;
use std::collections::BTreeSet;

/// The exhaustive match in `Step::speech` makes a new variant say whether it names a speech.
#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum Step {
    /// Selected for publication; not an acknowledgement of host delivery.
    Emitted(SessionEvent),
    /// Journalled where decided (before `decide_commit` rules on an `EndOfTurn`).
    Decided(crate::agent_session::effects::Decision),

    Core(CoreStep),
    Turn(TurnStep),
    Speech(SpeechStep),
    Interrupt(InterruptStep),
    Generation(GenerationStep),
    Tool(ToolStep),
    Playout(PlayoutStep),
    Input(InputStep),
    Realtime(RealtimeStep),
    Handoff(HandoffStep),
    Close(CloseStep),
    Detector(DetectorStep),
    Room(RoomStep),
}

macro_rules! step_families {
    ($($family:ident($ty:ident)),* $(,)?) => {$(
        impl From<$ty> for Step {
            fn from(step: $ty) -> Step {
                Step::$family(step)
            }
        }
    )*};
}
step_families!(
    Core(CoreStep),
    Turn(TurnStep),
    Speech(SpeechStep),
    Interrupt(InterruptStep),
    Generation(GenerationStep),
    Tool(ToolStep),
    Playout(PlayoutStep),
    Input(InputStep),
    Realtime(RealtimeStep),
    Handoff(HandoffStep),
    Close(CloseStep),
    Detector(DetectorStep),
    Room(RoomStep),
);

#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum CoreStep {
    /// A host command entered the loop: who asked and which leaf. The first step of its cycle.
    CommandReceived {
        origin: crate::agent_session::Origin,
        kind: &'static str,
    },
    TimerArmed {
        timer: crate::agent_session::effects::TurnTimer,
        at: lk_agents_abi::Ts,
    },
    TimerCancelled(crate::agent_session::effects::TurnTimer),
    /// Recorded as the cycle that performs it begins.
    TimerFired(TurnTimer, Ts),
    /// The whole configuration in effect after a layer of the option stack was applied. What a
    /// re-resolution changed is the diff against the previous `ConfigApplied`. Boxed: largest, rarest.
    ConfigApplied {
        layer: crate::agent_activity::options::ConfigLayer,
        effective: Box<crate::agent_activity::options::SessionConfig>,
    },
    WallEpoch {
        at: std::time::SystemTime,
    },
    /// Never silent.
    PortFailure(&'static str),
}

#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum TurnStep {
    /// The countdown reached `Committing` and `decide_commit` has ruled.
    EndOfTurn {
        transcript: String,
        verdict: CommitVerdict,
        committed: bool,
    },
    /// The user-turn claim depth moved; while > 0 the user state is pinned to `speaking`.
    UserTurnClaims {
        depth: u32,
    },
    /// `TurnCommand::Commit` at its arm, after its flush and final wait.
    CommitRequested {
        skip_reply: bool,
    },
    /// `CommitUserTurn` answered; `""` when there is no STT or the session is closing.
    CommitAnswered {
        transcript: String,
    },
    /// A newer `CommitUserTurn` cancelled this waiting one: no commit, no answer.
    CommitSuperseded,
    TranscriptSegment {
        id: String,
        text: String,
        is_final: bool,
    },
    /// STT was flushed with `frames` frames of silence (the end-of-track / manual-commit flush).
    SttFlushed {
        frames: u32,
    },
    PreemptiveReconciled {
        speech: SpeechId,
        verdict: PreemptiveVerdict,
    },
    PreemptiveRefused(crate::agent_activity::scheduling::PreemptiveRefusal),
}

#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum SpeechStep {
    /// A reply was created and queued; its task exists before scheduling.
    ReplyCreated {
        speech: SpeechId,
    },
    ReplyFinished {
        speech: SpeechId,
        text: String,
        interrupted: bool,
    },
    /// A provider failed; the error itself goes to the handle.
    ReplyFailed {
        speech: SpeechId,
        class: FailureClass,
    },
    /// The queue refused a speech (draining or stopped): force-interrupted, nothing plays.
    ScheduleRejected {
        speech: SpeechId,
        reason: crate::agent_activity::scheduling::Rejected,
    },
    /// A `say()` refused before any speech existed, as the reference raises at
    /// `AgentActivity.say`; no `SpeechCreated` is emitted.
    SayRefused {
        why: &'static str,
    },
    Authorized {
        speech: SpeechId,
    },
    /// A reply step was spoken and marked done. The floor is released here, before its tools run,
    /// so a tool's `say()` isn't queued behind its own speech (#4534).
    StepSpoken {
        speech: SpeechId,
        step: u32,
        interrupted: bool,
    },
    /// `FirstFrameEffects` executed. **The only place `_agent_speaking` is ever set.**
    AgentStartedSpeaking {
        speech: SpeechId,
    },
    /// A speech's tool is blocked on an inline `AgentTask`, or no longer. A missing unblock shows
    /// only as a session gone quiet, hence the step.
    Blocked {
        speech: SpeechId,
        blocked: bool,
    },
    ItemsRecorded {
        speech: SpeechId,
        ids: Vec<String>,
    },
    FloorArmed {
        until: Ts,
    },
    FloorReleased,
    IdleHoldReleased {
        hold: crate::agent_session::HoldId,
    },
    /// Scheduling stopped accepting new turns. `except` is every speech the pause doesn't wait for
    /// (the asking tool's and every other blocked on an inline task); a set, as `_drain_blocked_tasks`.
    SchedulingPaused {
        except: BTreeSet<SpeechId>,
    },
    SchedulingPauseSettled,
    SchedulingPauseRefused {
        cause: &'static str,
    },
    /// AMD's hold went on: the queue keeps its head and stops authorizing it.
    AuthorizationPaused,
    /// The hold came off; the held speech takes the floor on this same iteration.
    AuthorizationResumed,
    /// A `WaitForIdle`/`WaitForIdleAndHold` answered; `closing` when it was the refusal.
    IdleAnswered {
        hold: Option<crate::agent_session::HoldId>,
        closing: bool,
    },
    /// A tool's idle wait withdrawn with its speech: the reference cancels an interrupted speech's
    /// tools, releasing the hold in `_wait_for_idle_and_hold`'s `finally`.
    IdleAskWithdrawn {
        hold: Option<crate::agent_session::HoldId>,
        speech: SpeechId,
    },
    WatchdogArmed {
        speech: SpeechId,
        at: Ts,
    },
    WatchdogCleared {
        speech: SpeechId,
    },
}

#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum InterruptStep {
    ByAudioActivity(Box<crate::agent_activity::false_interruption::InterruptReport>),
    /// A interruption check refused before any effect ran.
    ByAudioActivityRefused(InterruptRefusal),
    /// Only produced by [`folded`](crate::trace::folded): the barge step above repeated `count` times.
    ByAudioActivityRepeated {
        count: u32,
    },
    Interrupted {
        speech: SpeechId,
        cause: InterruptCause,
        force: bool,
    },
    /// The interruption watchdog fired: tasks cancelled and the speech marked done
    /// (`SpeechHandle._cancel`).
    InterruptionTimedOut {
        speech: SpeechId,
    },
    InterruptionsToggled {
        speech: SpeechId,
        allow: bool,
    },
    /// `SpeechCommand::Interrupt` at its arm (`None`: session-level); `Interrupted` steps follow.
    HostInterrupt {
        speech: Option<SpeechId>,
        force: bool,
    },
    /// `SpeechCommand::Interrupt` answered, with the speeches the walk reached, all done now.
    InterruptSettled {
        speeches: Vec<SpeechId>,
    },
    /// The non-forced walk met a speech that forbids interruptions: it and those behind it play.
    InterruptWalkStopped {
        speech: SpeechId,
    },
    OverlapVerdict {
        is_interruption: bool,
        agent_ended: bool,
    },
    OverlapRefused(crate::audio_recognition::overlap::OverlapRefusal),
    PlayoutGate {
        speech: SpeechId,
        op: GateOp,
        cause: GateCause,
    },
}

#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum GenerationStep {
    ChatRequested {
        speech: SpeechId,
        step: u32,
        ctx_hash: u64,
        tools: Vec<String>,
        instructions_hash: u64,
        ctx_ids: Vec<String>,
        /// `None` is the frame default, `NOT_GIVEN`.
        tool_choice: Option<lk_agents_abi::ToolChoice>,
    },
    ChatFinished {
        speech: SpeechId,
        step: u32,
        error: Option<FailureClass>,
    },
    ChatCtxUpdated {
        items: usize,
    },
    TtsRequested {
        speech: SpeechId,
        segment: u32,
        text: String,
    },
    /// A provider attempt failed before output; `attempt` (from one) is the failed one.
    RetryArmed {
        speech: SpeechId,
        leg: ProviderKind,
        attempt: u32,
        at: Ts,
    },
    /// The dwell waits for idle from here.
    FillerStarted {
        filler: crate::filler_scheduler::FillerId,
        speech: SpeechId,
    },
    FillerFired {
        filler: crate::filler_scheduler::FillerId,
        step: u32,
        spoke: bool,
        speech: Option<SpeechId>,
    },
    FillerStopped {
        filler: crate::filler_scheduler::FillerId,
        cause: FillerEnd,
    },
}

#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum ToolStep {
    /// The model asked for these tools (names only; arguments can be pages of JSON).
    Calls {
        speech: SpeechId,
        names: Vec<String>,
    },
    /// A tool call handed to the runner (`hosted`) or answered by the driver (an unadvertised
    /// name). Only a hosted call holds the away window.
    CallStarted {
        speech: SpeechId,
        call_id: String,
        name: String,
        hosted: bool,
    },
    CallEnded {
        speech: SpeechId,
        call_id: String,
        end: crate::tool_executor::ToolEnd,
    },
    Updated {
        call_id: String,
        id: String,
        is_final: bool,
    },
    /// A `ToolUpdate` for a call the driver isn't running: dropped.
    UpdateRefused {
        call_id: String,
    },
    StepStarted {
        speech: SpeechId,
        step: u32,
    },
    /// The tools ran and `next_step` ruled; `committed` items went into the agent's history.
    Finished {
        speech: SpeechId,
        committed: usize,
        handoff: bool,
    },
    ToolsetUpdated {
        tools: usize,
    },
    /// `AgentSession._tools.extend(..)`: the session's tools grew by `added`.
    SessionToolsExtended {
        added: Vec<String>,
    },
    ReplyScheduled {
        speech: SpeechId,
        update_ids: Vec<String>,
        at_tail: bool,
    },
    /// The deferred reply's speech ended, and how the reference reports it.
    ReplyDone {
        speech: SpeechId,
        status: crate::events::ToolReplyStatus,
    },
}

#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum PlayoutStep {
    PlaybackStarted {
        speech: SpeechId,
    },
    /// One played run, reported at a discontinuity only (flush, dry, clear).
    PlaybackProgressed {
        speech: SpeechId,
        offset: std::time::Duration,
        duration: std::time::Duration,
    },
    PlaybackFinished {
        speech: SpeechId,
        position: std::time::Duration,
        interrupted: bool,
    },
    /// Frames the output played for a speech, per pushed batch. The anchor of "what was heard".
    AudioPlayed {
        speech: SpeechId,
        frames: u64,
    },
    SinkCalled {
        speech: Option<SpeechId>,
        call: SinkCall,
    },
    /// The sink's playback count was reset (`_reset_playback_count`) under `speech`'s segment.
    SinkReset {
        speech: SpeechId,
    },
    OutputReplaced {
        audio: bool,
        transcription: bool,
    },
    /// The audio gate's verdict changed (published on change only).
    GatePublished {
        stt_real: bool,
        realtime_real: bool,
        vad_real: bool,
        amd_real: bool,
    },
    /// Silence frames the forwarder substituted on `leg`, reported at the gate change that ends the
    /// verdict.
    GateSilenced {
        leg: crate::audio::AudioChannel,
        frames: u64,
    },
    TranscriptForwarded {
        speech: SpeechId,
        text: String,
    },
    /// The recorder was detached after one last flush.
    RecorderDetached,
    /// The flush timer fired and the recorder wrote what had settled; `ok: false` on failure.
    RecorderFlushed {
        ok: bool,
    },
}

#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum InputStep {
    /// A detector leg was asked (`req` correlates the answer).
    LegRequested {
        leg: DetectorName,
        req: u64,
    },
    LegVerdict {
        leg: DetectorName,
        req: u64,
    },
    LegFailed {
        leg: DetectorName,
        recoverable: bool,
    },
    /// An audio leg was reopened on a host-given provider (the open at `start()` is not journalled).
    StreamOpened {
        leg: &'static str,
    },
    /// A leg's stream came back for an open no longer waited for: closed unused.
    LegOpenDiscarded {
        leg: &'static str,
    },
    /// An item from a retired stream of `leg`, dropped before anything sees it; the reference
    /// cancels the old consumer at `_update_stt`.
    LegItemDiscarded {
        leg: &'static str,
        stream: lk_agents_abi::resources::ResourceId,
    },
    ChannelEnded {
        leg: &'static str,
        cause: ChannelEndCause,
    },
    /// A leg's frame counters at `Closing`: frames pushed, and how many were silence.
    LegFrames {
        leg: crate::audio::AudioChannel,
        frames: u64,
        silent: u64,
    },
    LegShape {
        leg: crate::audio::AudioChannel,
        sample_rate: u32,
        num_channels: u16,
        samples_per_channel: u32,
    },
    /// The microphone's frame shape and the fan-out's converted shape, at `Closing`.
    InputShape {
        sample_rate: u32,
        num_channels: u16,
        samples_per_channel: u32,
        converted_rate: u32,
        converted_samples_per_channel: u32,
    },
    /// `SetAudioInput` attached or removed a source. A detach under the user's speech ends it; a
    /// re-attach opens a fresh STT stream.
    AudioInputSet {
        attached: bool,
    },
    VideoInputSet {
        attached: bool,
    },
    /// The video pump handed a sampled frame to the model, or had none to hand it to (`sent: false`).
    VideoFramePushed {
        sent: bool,
    },
    /// The microphone's enable flag flipped; the leg stays installed.
    InputEnabled {
        leg: InputKind,
        enabled: bool,
    },
    OutputEnabled {
        leg: OutputKind,
        enabled: bool,
    },
    ModelsUpdated {
        stt_given: bool,
        vad_given: bool,
        stt_reopened: bool,
        vad_reopened: bool,
    },
    ProviderCalled(crate::agent_session::effects::ProviderCall),
    /// A detector call the machine asked for, recorded where it was decided.
    DetectorCalled(crate::agent_session::effects::DetectorCall),
    /// A host stream was closed by the driver, and how the close went (bounded on the session clock).
    HostStreamClosed {
        label: &'static str,
        how: lk_agents_abi::Closed,
    },
}

#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum RealtimeStep {
    Opened {
        turn_detection_disabled: bool,
    },
    /// A command sent to the realtime socket (every one but `PushAudio`).
    Sent {
        kind: lk_agents_abi::RealtimeCommandKind,
        req_id: Option<String>,
    },
    GenerationStatus {
        speech: SpeechId,
        status: String,
    },
}

#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum HandoffStep {
    /// A host `drain()` began; `exit_hook` is the frame's `on_exit` it waits for (`None` if it ran).
    DrainStarted { exit_hook: Option<crate::agent_session::HookId> },
    /// Every waiter of the frame's drain was answered; `refused` is `None` on success.
    Drained { refused: Option<crate::agent_session::DrainRefused> },

    /// The agent was replaced; `reopened` names the audio legs whose provider differed.
    AgentUpdated { tools: usize, chat_ctx: bool, reopened: &'static str, legs: ProviderPresence },
    /// One call into user code issued, by kind. `OnExit` precedes the handoff's pause, `OnEnter`
    /// follows `AgentUpdated`. `HookFinished` closes it.
    HookIssued {
        hook: crate::agent_session::HookId,
        kind: crate::agent_session::HookKind,
        agent: Option<String>,
        speech: Option<SpeechId>,
        turn: Option<crate::audio_recognition::TurnId>,
        at_tail: Option<bool>,
    },
    /// A hook's lifetime ended, and how; the turn hook's `Returned{edits}` counts its edits.
    HookFinished {
        hook: crate::agent_session::HookId,
        kind: crate::agent_session::HookKind,
        how: HookEnd,
    },
    /// The open run took ownership of something (`RunResult._watch_handle`).
    RunOwned { run: crate::events::RunId, what: crate::run_result::Owned },
    /// The run let go of something it owned (`_unwatch_handle`).
    RunReleased { run: crate::events::RunId, what: crate::run_result::Owned },
    /// Nothing owned left and no output: the run re-prompted the model (`_maybe_retry_output`).
    RunRetried { run: crate::events::RunId, attempt: u32 },
    /// `AgentTask.complete()` recorded the run's output (`_maybe_run_final_output`).
    RunOutput { speech: SpeechId, ok: bool },
}

#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum CloseStep {
    Requested { reason: lk_agents_abi::voice::events::CloseReason, drain: bool },
    Performed(CloseAct),
}

#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum DetectorStep {
    AmdRefused(crate::amd::classifier::Refusal),
    IvrTriggered { origin: crate::ivr::ivr_activity::IvrReplyOrigin },
    IvrLifecycle(IvrLifecycle),
    KeytermsPushed { terms: Vec<String> },
    KeytermsRebound { same_stt: bool },
}

#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum RoomStep {
    Event(RoomEventKind),
    Link(RoomLinkStep),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillerEnd {
    Stopped,
    /// The tool's speech ended or was interrupted (`wait_if_not_interrupted`).
    SpeechEnded,
    /// `interval=None` fired once, or `max_steps` was reached.
    Done,
    /// A `FillerStart` for an id already open or for a speech the loop does not hold, or a
    /// fire whose say the scheduler refused.
    Refused,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IvrLifecycle {
    Started,
    /// A second start while one runs: refused, the running one stands.
    AlreadyStarted,
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ProviderKind {
    Llm,
    Tts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SinkCall {
    Push { frames: u64 },
    Flush,
    Clear,
    End,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookEnd {
    /// The host answered; `edits` counts a turn hook's `TurnEdit`s.
    Returned {
        edits: usize,
    },
    Failed,
    /// The bound passed with no answer; the sequence went ahead without it.
    TimedOut,
    /// Nothing waits on the running hook any more (the session closed, or the filler it sources
    /// ended); user code is not cancelled.
    Abandoned,
    /// No host to run hooks: issued and finished in the same step.
    NoHost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureClass {
    Retryable,
    NonRetryable,
    Timeout,
    Cancelled,
}

impl FailureClass {
    pub fn of(err: &lk_agents_abi::errors::ProviderError) -> Self {
        use lk_agents_abi::errors::ApiErrorKind as K;
        match err.api_kind() {
            Some(K::Timeout) => FailureClass::Timeout,
            Some(K::Cancelled) => FailureClass::Cancelled,
            Some(K::Api) | Some(K::Status) | Some(K::Connection) | None => {
                if err.retryable() {
                    FailureClass::Retryable
                } else {
                    FailureClass::NonRetryable
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CloseAct {
    TranscriptWaitEnded { transcript: String },
    Released { speech: SpeechId },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptCause {
    ByAudioActivity,
    Host,
    /// A detector's verdict (AMD: a machine answered), forced past `allow_interruptions`.
    Detector,
    Superseded,
    Closing,
    ScheduleRejected,
    Watchdog,
    Handoff,
    UserTurn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorName {
    Eot,
    Interruption,
    Amd,
    /// One request per `turn_interval` user items.
    Keyterms,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProviderPresence {
    pub stt: bool,
    pub vad: bool,
    pub llm: bool,
    pub tts: bool,
    pub realtime: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoomLinkStep {
    /// `RoomIO` linked a participant (`room_io.RoomIO._on_participant_connected`).
    Linked,
    /// The linked participant left with `reason`; `closes` is the machine's verdict.
    Unlinked { reason: lk_agents_rtc::DisconnectReason, closes: bool },
    /// `delete_room_on_close`: the host was asked to delete the room, after `Close`.
    DeleteRoomRequested,
}

/// One [`RoomEvent`](lk_agents_rtc::RoomEvent) variant, payload dropped: what the journal keeps of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoomEventKind {
    TrackSubscribed,
    TrackPublished,
    TrackUnpublished,
    LocalTrackPublished,
    ParticipantConnected,
    ParticipantDisconnected,
    ConnectionStateChanged,
    Close,
}

impl RoomEventKind {
    pub fn of(ev: &lk_agents_rtc::RoomEvent) -> Self {
        use lk_agents_rtc::RoomEvent as E;
        match ev {
            E::TrackSubscribed { .. } => RoomEventKind::TrackSubscribed,
            E::TrackPublished { .. } => RoomEventKind::TrackPublished,
            E::TrackUnpublished { .. } => RoomEventKind::TrackUnpublished,
            E::LocalTrackPublished { .. } => RoomEventKind::LocalTrackPublished,
            E::ParticipantConnected { .. } => RoomEventKind::ParticipantConnected,
            E::ParticipantDisconnected { .. } => RoomEventKind::ParticipantDisconnected,
            E::ConnectionStateChanged { .. } => RoomEventKind::ConnectionStateChanged,
            E::Close { .. } => RoomEventKind::Close,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelEndCause {
    SourceEnded,
    Error,
    Closing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub enum InputKind {
    Audio,
    Video,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub enum OutputKind {
    Audio,
    Transcription,
}

impl Step {
    pub fn speech(&self) -> Option<SpeechId> {
        use crate::events::{SessionEvent as E, ToolLifecycle};
        match self {
            Step::Emitted(event) => match event {
                E::SpeechCreated { speech, .. } => Some(speech.id()),
                E::SpeechDone { speech, .. } => Some(*speech),
                E::FloorChanged { holder } => *holder,
                E::ToolExecutionUpdated(ToolLifecycle::Reply { speech, .. }) => Some(*speech),
                _ => None,
            },
            Step::Decided(_) => None,
            Step::Core(_) => None,
            Step::Turn(step) => match step {
                TurnStep::PreemptiveReconciled { speech, .. } => Some(*speech),
                TurnStep::EndOfTurn { .. }
                | TurnStep::UserTurnClaims { .. }
                | TurnStep::CommitRequested { .. }
                | TurnStep::CommitAnswered { .. }
                | TurnStep::CommitSuperseded
                | TurnStep::TranscriptSegment { .. }
                | TurnStep::SttFlushed { .. }
                | TurnStep::PreemptiveRefused(_) => None,
            },
            Step::Speech(step) => match step {
                SpeechStep::ReplyCreated { speech }
                | SpeechStep::ReplyFinished { speech, .. }
                | SpeechStep::ReplyFailed { speech, .. }
                | SpeechStep::ScheduleRejected { speech, .. }
                | SpeechStep::Authorized { speech }
                | SpeechStep::StepSpoken { speech, .. }
                | SpeechStep::AgentStartedSpeaking { speech }
                | SpeechStep::Blocked { speech, .. }
                | SpeechStep::ItemsRecorded { speech, .. }
                | SpeechStep::IdleAskWithdrawn { speech, .. }
                | SpeechStep::WatchdogArmed { speech, .. }
                | SpeechStep::WatchdogCleared { speech } => Some(*speech),
                SpeechStep::SayRefused { .. }
                | SpeechStep::FloorArmed { .. }
                | SpeechStep::FloorReleased
                | SpeechStep::IdleHoldReleased { .. }
                | SpeechStep::SchedulingPaused { .. }
                | SpeechStep::SchedulingPauseSettled
                | SpeechStep::SchedulingPauseRefused { .. }
                | SpeechStep::AuthorizationPaused
                | SpeechStep::AuthorizationResumed
                | SpeechStep::IdleAnswered { .. } => None,
            },
            Step::Interrupt(step) => match step {
                InterruptStep::Interrupted { speech, .. }
                | InterruptStep::InterruptionTimedOut { speech }
                | InterruptStep::InterruptionsToggled { speech, .. }
                | InterruptStep::InterruptWalkStopped { speech }
                | InterruptStep::PlayoutGate { speech, .. } => Some(*speech),
                InterruptStep::HostInterrupt { speech, .. } => *speech,
                InterruptStep::ByAudioActivity(_)
                | InterruptStep::ByAudioActivityRefused(_)
                | InterruptStep::ByAudioActivityRepeated { .. }
                | InterruptStep::InterruptSettled { .. }
                | InterruptStep::OverlapVerdict { .. }
                | InterruptStep::OverlapRefused(_) => None,
            },
            Step::Generation(step) => match step {
                GenerationStep::ChatRequested { speech, .. }
                | GenerationStep::ChatFinished { speech, .. }
                | GenerationStep::TtsRequested { speech, .. }
                | GenerationStep::RetryArmed { speech, .. }
                | GenerationStep::FillerStarted { speech, .. } => Some(*speech),
                GenerationStep::FillerFired { speech, .. } => *speech,
                GenerationStep::ChatCtxUpdated { .. } | GenerationStep::FillerStopped { .. } => {
                    None
                }
            },
            Step::Tool(step) => match step {
                ToolStep::Calls { speech, .. }
                | ToolStep::CallStarted { speech, .. }
                | ToolStep::CallEnded { speech, .. }
                | ToolStep::StepStarted { speech, .. }
                | ToolStep::Finished { speech, .. }
                | ToolStep::ReplyScheduled { speech, .. }
                | ToolStep::ReplyDone { speech, .. } => Some(*speech),
                ToolStep::Updated { .. }
                | ToolStep::UpdateRefused { .. }
                | ToolStep::ToolsetUpdated { .. }
                | ToolStep::SessionToolsExtended { .. } => None,
            },
            Step::Playout(step) => match step {
                PlayoutStep::PlaybackStarted { speech }
                | PlayoutStep::PlaybackProgressed { speech, .. }
                | PlayoutStep::PlaybackFinished { speech, .. }
                | PlayoutStep::AudioPlayed { speech, .. }
                | PlayoutStep::SinkReset { speech }
                | PlayoutStep::TranscriptForwarded { speech, .. } => Some(*speech),
                PlayoutStep::SinkCalled { speech, .. } => *speech,
                PlayoutStep::OutputReplaced { .. }
                | PlayoutStep::GatePublished { .. }
                | PlayoutStep::GateSilenced { .. }
                | PlayoutStep::RecorderDetached
                | PlayoutStep::RecorderFlushed { .. } => None,
            },
            Step::Input(_) => None,
            Step::Realtime(step) => match step {
                RealtimeStep::GenerationStatus { speech, .. } => Some(*speech),
                RealtimeStep::Opened { .. } | RealtimeStep::Sent { .. } => None,
            },
            Step::Handoff(step) => match step {
                HandoffStep::RunOutput { speech, .. } => Some(*speech),
                HandoffStep::HookIssued { speech, .. } => *speech,
                HandoffStep::DrainStarted { .. }
                | HandoffStep::Drained { .. }
                | HandoffStep::AgentUpdated { .. }
                | HandoffStep::HookFinished { .. }
                | HandoffStep::RunOwned { .. }
                | HandoffStep::RunReleased { .. }
                | HandoffStep::RunRetried { .. } => None,
            },
            Step::Close(_) | Step::Detector(_) | Step::Room(_) => None,
        }
    }
}

/// `assert_step_is_covered` makes a family missing from this list a compile error.
pub fn step_names() -> Vec<String> {
    use lk_agents_abi::utils::vocabulary::{Coverage, Covers};

    struct Names;
    macro_rules! names {
        ($($variant:ident => $names:expr;)*) => {$(
            impl Covers<step_vocabulary::$variant> for Names {
                type With = &'static [&'static str];
                fn coverage() -> Coverage<Self::With> {
                    Coverage::Covered($names)
                }
            }
        )*};
    }
    names! {
        Emitted => crate::events::session_event_vocabulary::NAMES;
        Core => core_step_vocabulary::NAMES;
        Turn => turn_step_vocabulary::NAMES;
        Speech => speech_step_vocabulary::NAMES;
        Interrupt => interrupt_step_vocabulary::NAMES;
        Generation => generation_step_vocabulary::NAMES;
        Tool => tool_step_vocabulary::NAMES;
        Playout => playout_step_vocabulary::NAMES;
        Input => input_step_vocabulary::NAMES;
        Realtime => realtime_step_vocabulary::NAMES;
        Handoff => handoff_step_vocabulary::NAMES;
        Close => close_step_vocabulary::NAMES;
        Detector => detector_step_vocabulary::NAMES;
        Room => room_step_vocabulary::NAMES;
        Decided => crate::agent_session::effects::decision_vocabulary::NAMES;
    }
    assert_step_is_covered::<Names, &'static [&'static str]>();

    step_coverage::<Names, &'static [&'static str]>()
        .into_iter()
        .flat_map(|(family, arm)| match arm {
            Coverage::Covered([]) => vec![family.to_string()],
            Coverage::Covered(names) => {
                names.iter().map(|name| format!("{family}::{name}")).collect::<Vec<_>>()
            }
            Coverage::Skipped(_) => vec![],
        })
        .collect()
}

fn is_barge(step: &Step) -> bool {
    matches!(
        step,
        Step::Interrupt(InterruptStep::ByAudioActivity(_))
            | Step::Interrupt(InterruptStep::ByAudioActivityRefused(_))
    )
}

/// Consecutive identical barge steps become the first plus [`InterruptStep::ByAudioActivityRepeated`].
pub fn folded(steps: &[Step]) -> Vec<Step> {
    let mut out: Vec<Step> = Vec::with_capacity(steps.len());
    let mut run: Option<(&Step, u32)> = None;
    for step in steps {
        if is_barge(step)
            && let Some((first, count)) = &mut run
            && *first == step
        {
            *count += 1;
            match out.last_mut() {
                Some(Step::Interrupt(InterruptStep::ByAudioActivityRepeated { count: c })) => {
                    *c = *count
                }
                Some(_) | None => {
                    out.push(Step::Interrupt(InterruptStep::ByAudioActivityRepeated {
                        count: *count,
                    }))
                }
            }
            continue;
        }
        run = is_barge(step).then_some((step, 0));
        out.push(step.clone());
    }
    out
}

pub struct Trace {
    steps: Vec<Step>,
    /// `None` in production.
    observer: Option<Box<dyn FnMut(&Step) + Send>>,
}

impl std::fmt::Debug for Trace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.steps.iter()).finish()
    }
}

impl Trace {
    pub fn new() -> Self {
        Trace { steps: Vec::new(), observer: None }
    }

    /// Replaces any earlier observer.
    pub fn observe(&mut self, observer: Box<dyn FnMut(&Step) + Send>) {
        self.observer = Some(observer);
    }

    pub fn push(&mut self, step: Step) {
        if let Some(observer) = &mut self.observer {
            observer(&step);
        }
        self.steps.push(step);
    }

    pub fn steps(&self) -> Vec<Step> {
        self.steps.clone()
    }
}
