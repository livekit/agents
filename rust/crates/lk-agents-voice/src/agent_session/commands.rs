//! The session's command vocabulary: everything a host or a system may ask of the driver, each
//! answered (when it is) through the channel it carries.

use crate::agent_activity::false_interruption::InterruptRefusal;
use crate::agent_activity::scheduling::SpeechId;
use crate::filler_scheduler::FillerId;
use crate::generation::Instructions;
use crate::tool_executor::Toolset;
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::{ChatContext, Maybe};
use std::sync::Arc;

/// Enqueueing a command does not mean the operation was accepted or completed.
#[derive(Debug, lk_agents_abi::Vocabulary)]
pub enum Command {
    Speech(SpeechCommand),
    Activity(ActivityCommand),
    Turn(TurnCommand),
    /// The pixels stay in the binding's side table (an `ImageSource::Frame` key); the host mints
    /// the `img_` id.
    Video(lk_agents_abi::ImageContent),
    Close(Close),
    /// An id nobody waits on is dropped.
    HookDone {
        id: HookId,
        outcome: HookOutcome,
    },
    /// Plugin-measured only (VAD, realtime usage, EOT/interruption); LLM/TTS/STT/EOU never come
    /// from here.
    HostMetrics(Box<lk_agents_abi::metrics::base::AgentMetrics>),
    Ivr(IvrCommand),
    /// `update_options(keyterms=..)`: replaces the user list; the effective list is pushed to the
    /// STT.
    SetStaticKeyterms(Vec<String>),
    Recorder(RecorderCommand),
    Dtmf(DtmfCommand),
    /// Answered by `RunStarted` or `RunRefused`, settled by `RunSettled`.
    Run(Box<crate::run_result::RunSpec>),
    VoiceTask(crate::agent_activity::voice_task::VoiceTaskCommand),
    /// From `VoiceCommandOrigin::Amd`: the detector attached or released its verdict; its `amd`
    /// span is the session trace's.
    AmdSpan(AmdSpan),
}

/// The answering-machine detector's span, as the detector reports it.
#[derive(Debug, Clone)]
pub enum AmdSpan {
    /// Attached: `amd` opens with the context the detector's classification runs under.
    Attached(
        lk_agents_core::telemetry::span::Reserved<lk_agents_abi::telemetry::spans::session::Amd>,
    ),
    /// Released its verdict: `amd` ends.
    Verdict(lk_agents_abi::voice::amd::AmdPrediction),
}

/// Monotonic within a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HookId(pub u64);

/// `AMD.execute()` sends `AddTools`, `Start` then `SeedTranscript` on a `machine-ivr` verdict.
#[derive(Debug, Clone, lk_agents_abi::Vocabulary)]
pub enum IvrCommand {
    /// A running navigator stands.
    Start {
        max_silence: std::time::Duration,
    },
    /// A final counted as if heard (AMD's greeting), so the first menu repeat is the second chunk.
    SeedTranscript {
        text: String,
    },
    Close,
    /// Added to the **session's** tools before `Start` and AMD's resume; acted on by the core, not
    /// relayed.
    AddTools(crate::tool_executor::Toolset),
}

#[derive(Debug, lk_agents_abi::Vocabulary)]
pub enum HookOutcome {
    /// A hook that raised is `Failed`, and the sequence continues.
    Returned,
    /// `StopResponse` is one of them.
    Turn(Vec<crate::agent::TurnEdit>),
    Filler {
        text: Option<String>,
        spoke: bool,
    },
    /// `function_tools_executed` handled: each output's `reply_required`, in order, as the
    /// handlers left them.
    FunctionToolsExecuted {
        reply_required: Vec<bool>,
    },
    /// `on_user_turn_completed` skips the reply.
    Failed,
    /// The hook raised: `Failed`, with the exception its span records.
    Raised(Box<lk_agents_abi::errors::HostError>),
}

pub use lk_agents_abi::voice::session::VoiceCommandOrigin;

pub use lk_agents_abi::voice::session::{CommitSuperseded, IdleRefused};

/// The id a caller gives a command it awaits; the answer carries it back. `None` where a command
/// takes one means nobody awaits the answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestId(pub u64);

impl RequestId {
    /// The calls a session's own tasks await take ids with this bit set, apart from the host's.
    const TASK: u64 = 1 << 63;

    /// The id of the `n`th call the session's tasks await.
    pub fn task(n: u64) -> Self {
        RequestId(n | Self::TASK)
    }
}

/// A `say()` / `generate_reply()` the host awaits: its call, and the id its handle already has,
/// which the loop creates the speech under.
#[derive(Debug)]
pub struct SpeechCall {
    pub id: RequestId,
    pub speech: SpeechId,
}

/// What an awaited command is answered with, once, on the session's answer stream.
#[derive(Debug)]
pub enum Answer {
    /// The speech of a [`SpeechCall`] is created.
    Speech,
    ReplyPlan(Option<Box<crate::generation::PlannedReply>>),
    OutputBinding(Result<Option<super::output::SegmentBinding>, super::output::OutputBindingError>),
    OutputActivation(
        Result<Option<super::output::SegmentBinding>, super::output::OutputBindingError>,
    ),
    /// A tool call's next move.
    ToolGo(crate::tool_executor::ToolGo),
    /// A tool step's next step, decided by the driver.
    ToolStep(crate::tool_executor::StepNext),
    Interrupt(Result<(), InterruptRefusal>),

    Commit(Result<String, CommitSuperseded>),
    Idle(Result<(), IdleRefused>),
    Hold(Result<HoldId, IdleRefused>),
    Drain(Result<(), DrainRefused>),
    Handoff(Result<(), HandoffRefused>),
    Dtmf(crate::workflows::dtmf::DtmfOutcome),
    /// The value of the task's validated `Completed`.
    VoiceTask(
        Result<
            lk_agents_abi::voice::session::VoiceTaskValue,
            crate::agent_activity::voice_task::VoiceTaskEnded,
        >,
    ),
}

macro_rules! answer_picks {
    ($($pick:ident: $variant:ident($payload:ty)),* $(,)?) => {
        impl Answer {
            $(pub fn $pick(self) -> Option<$payload> {
                match self {
                    Answer::$variant(payload) => Some(payload),
                    _ => None,
                }
            })*
        }
    };
}

answer_picks! {
    reply_plan: ReplyPlan(Option<Box<crate::generation::PlannedReply>>),
    output_binding: OutputBinding(Result<Option<super::output::SegmentBinding>, super::output::OutputBindingError>),
    output_activated: OutputActivation(Result<Option<super::output::SegmentBinding>, super::output::OutputBindingError>),
    tool_go: ToolGo(crate::tool_executor::ToolGo),
    tool_step: ToolStep(crate::tool_executor::StepNext),
    interrupt: Interrupt(Result<(), InterruptRefusal>),
    commit: Commit(Result<String, CommitSuperseded>),
    idle: Idle(Result<(), IdleRefused>),
    hold: Hold(Result<HoldId, IdleRefused>),
    drain: Drain(Result<(), DrainRefused>),
    handoff: Handoff(Result<(), HandoffRefused>),
    dtmf: Dtmf(crate::workflows::dtmf::DtmfOutcome),
    voice_task: VoiceTask(Result<lk_agents_abi::voice::session::VoiceTaskValue, crate::agent_activity::voice_task::VoiceTaskEnded>),
}

#[derive(Debug)]
pub struct Sent {
    pub origin: VoiceCommandOrigin,
    pub command: Command,
}

impl Command {
    pub fn kind(&self) -> &'static str {
        macro_rules! leaf {
            ($c:expr; $E:ident: $($V:ident),* $(,)?) => {
                match $c { $($E::$V { .. } => concat!(stringify!($E), "::", stringify!($V)),)* }
            };
        }
        match self {
            Command::Speech(c) => leaf!(c; SpeechCommand: Say, GenerateReply, Interrupt,
                InterruptSpeech, CommitUserTurn, WaitForIdle, WaitForIdleAndHold, ReleaseIdleHold, ClaimUserTurn,
                ReleaseUserTurn, SetAllowInterruptions, HoldInterruptions, EndCall, ToolUpdate, FillerStart,
                FillerStop, PauseAuthorization, ResumeAuthorization),
            Command::Activity(c) => leaf!(c; ActivityCommand: Update, UpdateInstructions,
                UpdateTools, UpdateExpressive, SetAudioInput, SetVideoInput, SetIoEnabled, UpdateChatCtx, SetAudioOutput, SetTranscriptionOutput, UpdateModels, Drain),
            Command::Turn(c) => leaf!(c; TurnCommand: Commit, Clear, Configure, Override,
                ReleaseOverride, SkipReplies),
            Command::Ivr(c) => leaf!(c; IvrCommand: Start, SeedTranscript, Close, AddTools),
            Command::Dtmf(c) => leaf!(c; DtmfCommand: Start, Cancel),
            Command::VoiceTask(c) => {
                use crate::agent_activity::voice_task::VoiceTaskCommand;
                leaf!(c; VoiceTaskCommand: Run, Cancel)
            }
            Command::Video(_) => "Command::Video",
            Command::Close(_) => "Command::Close",
            Command::HookDone { .. } => "Command::HookDone",
            Command::HostMetrics(_) => "Command::HostMetrics",
            Command::SetStaticKeyterms(_) => "Command::SetStaticKeyterms",
            Command::Run(_) => "Command::Run",
            Command::AmdSpan(_) => "Command::AmdSpan",
            Command::Recorder(RecorderCommand::Attach(_)) => "Command::Recorder::Attach",
            Command::Recorder(RecorderCommand::Detach) => "Command::Recorder::Detach",
        }
    }
}

#[derive(Debug)]
pub struct Close {
    pub reason: lk_agents_abi::voice::events::CloseReason,
    pub drain: bool,
    /// The unrecoverable error that closes the session: the `error` event's, which `close`
    /// carries too.
    pub error: Option<crate::events::ErrorEvent>,
}

impl Default for Close {
    /// `aclose()` with no arguments: `USER_INITIATED`, no drain, no error.
    fn default() -> Self {
        Close {
            reason: lk_agents_abi::voice::events::CloseReason::UserInitiated,
            drain: false,
            error: None,
        }
    }
}

impl From<Close> for Command {
    fn from(c: Close) -> Command {
        Command::Close(c)
    }
}

impl From<SpeechCommand> for Command {
    fn from(c: SpeechCommand) -> Command {
        Command::Speech(c)
    }
}

impl From<IvrCommand> for Command {
    fn from(c: IvrCommand) -> Command {
        Command::Ivr(c)
    }
}

impl From<ActivityCommand> for Command {
    fn from(c: ActivityCommand) -> Command {
        Command::Activity(c)
    }
}

impl From<crate::agent_activity::voice_task::VoiceTaskCommand> for Command {
    fn from(c: crate::agent_activity::voice_task::VoiceTaskCommand) -> Command {
        Command::VoiceTask(c)
    }
}

impl From<TurnCommand> for Command {
    fn from(c: TurnCommand) -> Command {
        Command::Turn(c)
    }
}

/// Outlives the agent across handoffs.
#[derive(Debug, lk_agents_abi::Vocabulary)]
pub enum TurnCommand {
    /// Commit the turn now (`commit_user_turn`'s tail); a typed turn is
    /// `generate_reply(user_input=)`.
    Commit {
        /// `commit_user_turn(skip_reply=True)`: close the turn, keep the transcript, generate nothing.
        skip_reply: bool,
    },
    /// A realtime server buffer is cleared too.
    Clear,
    /// Unnamed leaves keep their answer; a changed endpointing replaces the model (a running
    /// countdown keeps its deadline); `turn_detection=None` arrives as `Some(Auto)`.
    Configure(Box<crate::agent_activity::options::AgentSessionOptions>),
    /// `override_options(..)`: a layer over the base until its `ReleaseOverride`.
    Override(Box<crate::agent_activity::options::OverrideLayer>),
    /// The `override_options` block ended, however it ended.
    ReleaseOverride(u64),
    /// `AMD._on_end_of_turn`, from `VoiceCommandOrigin::Amd`: while the detector holds a machine verdict under
    /// `interrupt_on_machine`, every turn is committed with no reply (the developer is leaving a
    /// voicemail over the greeting's tail). Cleared when the detector closes.
    SkipReplies(bool),
}

/// One collection at a time.
#[derive(Debug, lk_agents_abi::Vocabulary)]
pub enum DtmfCommand {
    /// Answered once with the first outcome, or `Stopped`; the confirm arm is spoken by the
    /// driver.
    Start { collector: crate::workflows::dtmf::DtmfCollector, call: Option<RequestId> },
    /// The awaiter of `call` went away: its collection stops; a newer one is left running.
    Cancel { call: RequestId },
}

impl From<DtmfCommand> for Command {
    fn from(c: DtmfCommand) -> Command {
        Command::Dtmf(c)
    }
}

#[derive(Debug, lk_agents_abi::Vocabulary)]
pub enum SpeechCommand {
    Say {
        /// A stream is one segment per chunk.
        text: SayText,
        /// `None` takes the session default.
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        call: Option<SpeechCall>,
    },
    GenerateReply {
        /// `None` replies to the context as it stands.
        user_input: Option<String>,
        /// An *extra* system message for this turn only.
        instructions: Option<String>,
        allow_interruptions: Option<bool>,
        /// **`None` is `NOT_GIVEN`, not `"none"`**: the deferred tool-result reply passes `"none"`
        /// so it cannot start another round.
        tool_choice: Option<lk_agents_abi::ToolChoice>,
        /// Planned against instead of the record (untouched).
        chat_ctx: Option<ChatContext>,
        /// Whole, because specs, host objects and runner must agree.
        toolset: Option<Arc<Toolset>>,
        /// `generate_reply(tools=)`: the agent's toolset narrowed to these names, its provider
        /// tools kept, for this reply only.
        tool_names: Option<Vec<String>>,
        on_enter: bool,
        call: Option<SpeechCall>,
    },
    /// A pending preemptive generation is dropped; the current speech is interrupted (a refusal is
    /// the whole answer); the socket is told once; the queue is walked in pop order up to the first
    /// refusal, each speech whole.
    Interrupt {
        force: bool,
        call: Option<RequestId>,
    },
    /// The whole speech: its playout, its tools and any follow-up; `force` past a protection.
    InterruptSpeech {
        speech: SpeechId,
        force: bool,
    },
    /// Flush, wait for the final (skipped if one landed in the last 0.5 s), commit. No STT or
    /// closing: `""`, no commit. A newer commit cancels a waiting one; realtime commits and asks
    /// for a reply first.
    CommitUserTurn {
        transcript_timeout: std::time::Duration,
        stt_flush_duration: std::time::Duration,
        skip_reply: bool,
        call: Option<RequestId>,
    },
    /// Idle: nothing current, queued or being decided, the user silent, no claim or foreign hold.
    WaitForIdle {
        hold: Option<HoldId>,
        call: Option<RequestId>,
    },
    /// `RunContext.foreground()`: wait for idle, then hold it until `ReleaseIdleHold`; a waiter naming
    /// the hold passes through.
    WaitForIdleAndHold {
        call: Option<RequestId>,
    },
    /// Unknown ids are a no-op.
    ReleaseIdleHold {
        hold: HoldId,
    },
    /// Pins the user state to `speaking` and holds idle open. Reentrant.
    ClaimUserTurn,
    /// The last release re-derives the user state.
    ReleaseUserTurn,
    /// Refused quietly on a done handle.
    SetAllowInterruptions {
        speech: SpeechId,
        allow: bool,
    },
    HoldInterruptions {
        speech: SpeechId,
        hold: bool,
    },
    /// `EndCallTool`: close the session once `speech` (the tool's) is done; on a realtime model
    /// answering tool results by itself, once that answer is done.
    EndCall {
        speech: SpeechId,
    },
    /// A background tool speaks again after its first `ctx.update()`: the driver mints and commits
    /// the `{call_id}_update_N` / `{call_id}_final` pair and coalesces one deferred reply at idle.
    ToolUpdate {
        invocation_id: String,
        operation: u64,
        call_id: String,
        update: ToolUpdate,
    },
    /// While the tool runs, say the source after `delay` idle, then every `interval`. A repeated
    /// `id` is refused.
    FillerStart {
        id: FillerId,
        call_id: String,
        speech: SpeechId,
        delay: std::time::Duration,
        interval: Option<std::time::Duration>,
        max_steps: Option<u32>,
        source: FillerSource,
    },
    /// The dwell or cooldown in flight is dropped; a fire already made stays.
    FillerStop {
        id: FillerId,
    },
    /// `VoiceCommandOrigin::Amd` only: nothing is authorized until `ResumeAuthorization`. A flag, not a
    /// counter.
    PauseAuthorization,
    /// The head is authorized in this cycle.
    ResumeAuthorization,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FillerSource {
    Text(String),
    /// Asked at every fire: the host's callable, handed back on its `filler_source` hook.
    Host(Option<lk_agents_abi::HostRef>),
}

#[derive(Debug, Clone)]
pub enum ToolUpdate {
    /// `raw` as written; `message` rendered by the host.
    Progress { message: String, raw: String },
    /// Decided like a blocking tool's, except a silent return voices nothing.
    Final { report: crate::tool_executor::ToolReport },
}

impl From<crate::tool_executor::ToolReport> for ToolUpdate {
    /// A later `ctx.update()` is progress; any other outcome is the tool's end.
    fn from(report: crate::tool_executor::ToolReport) -> Self {
        match report.outcome {
            crate::tool_executor::ToolOutcome::Update { message, raw } => {
                ToolUpdate::Progress { message, raw }
            }
            _ => ToolUpdate::Final { report },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HoldId(pub u64);

#[derive(Debug)]
pub enum SayText {
    Text(String),
    /// Pre-synthesised frames, which the host's handle holds for the speech.
    Audio {
        text: String,
    },
    /// Text chunks the host's handle holds for the speech; closing the sender ends it.
    Stream,
}

pub struct AudioInputHandle(pub ResourceId);

/// Host handles are newtypes so the commands carrying them keep their `Debug`.
macro_rules! host_handle_debug {
    ($T:ident) => {
        impl std::fmt::Debug for $T {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(stringify!($T))
            }
        }
    };
}
host_handle_debug!(AudioInputHandle);
host_handle_debug!(RecorderHandle);

#[derive(Debug)]
pub enum RecorderCommand {
    /// Flushes every 2.5 s of session time.
    Attach(RecorderHandle),
    /// The file's `finish` stays the host's.
    Detach,
}

pub struct RecorderHandle(pub ResourceId);

/// Video input is handled at the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub enum IoKind {
    AudioInput,
    AudioOutput,
    Transcription,
}

/// Providers replaced together by one session update.
#[derive(Debug)]
pub struct ModelUpdate {
    pub stt: Maybe<Option<lk_agents_abi::resources::SttRef>>,
    pub vad: Maybe<Option<lk_agents_abi::resources::VadRef>>,
    pub llm: Maybe<Option<lk_agents_abi::resources::LlmRef>>,
    pub tts: Maybe<Option<lk_agents_abi::resources::TtsRef>>,
}

/// Scheduling pauses are the driver's own, never a command.
#[derive(Debug, lk_agents_abi::Vocabulary)]
pub enum ActivityCommand {
    /// No stream is torn down, so no word is dropped. Order (`_update_activity`): refuse when
    /// closing; `on_exit` before the pause; drain; closing re-check; swap; `on_enter` not awaited.
    /// A second queues.
    Update(Box<AgentUpdate>),
    // --- the running agent, changed in place: not a handoff, which replaces the whole agent ---
    /// Replies in flight keep theirs. Carries the full triple: workflow tasks put their persona in
    /// the modality variants.
    UpdateInstructions(Instructions),
    UpdateTools(crate::tool_executor::Toolset),
    /// `update_options(expressive=)`: the session's setting, read by the next reply.
    UpdateExpressive(crate::agent_activity::options::Expressive),
    /// The old pump is aborted first, so two sources never feed the legs at once.
    SetAudioInput(Option<AudioInputHandle>),
    SetVideoInput(Option<lk_agents_abi::resources::VideoInputRef>),
    /// The leg stays: a muted mic stops feeding STT; a disabled output plans speech without a
    /// voice.
    SetIoEnabled {
        leg: IoKind,
        enabled: bool,
    },
    /// **Replaces** the running agent's context (already filtered host-side).
    UpdateChatCtx(Vec<lk_agents_abi::ChatItem>),
    /// A playing speech is **interrupted** (nobody can hear it); the pause facts are re-read at
    /// once.
    SetAudioOutput(Option<lk_agents_abi::resources::AudioOutputRef>),
    /// A reply in flight keeps its leg.
    SetTranscriptionOutput(Option<lk_agents_abi::resources::TextOutputRef>),
    /// **Not a handoff**: only the providers change; the streaming legs reopen as a handoff reopens
    /// them.
    UpdateModels(Box<ModelUpdate>),
    /// Runs the driving agent's `on_exit` (once per agent), then pauses scheduling until the speech
    /// in flight finishes, tools included. A second joins. Refused only if the pause is lifted or
    /// the loop ends.
    Drain {
        call: Option<RequestId>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, lk_agents_abi::Vocabulary)]
pub enum DrainRefused {
    /// The loop ended with no close to answer the drain.
    Closing,
    /// Scheduling was resumed under the drain's pause: the agent runs again.
    ActivityChanging,
    /// A `VoiceTask` holds the floor; its parent waits on it.
    TaskRunning,
}

impl DrainRefused {
    pub fn as_str(self) -> &'static str {
        match self {
            DrainRefused::Closing => "closing",
            DrainRefused::ActivityChanging => "activity_changing",
            DrainRefused::TaskRunning => "voice_task_running",
        }
    }
}

pub struct AgentUpdate {
    /// Stamped on every `SessionEvent::ConversationItemAdded`; the host maps it back to its `Agent`.
    pub agent: String,
    /// Workflow tasks put their persona in the modality variants.
    pub instructions: Instructions,
    pub dynamic_instructions: Option<lk_agents_abi::resources::ResourceId>,
    /// The executor is per-agent; its agent's definition carries the retry budgets.
    pub toolset: crate::tool_executor::Toolset,
    /// **Replaces** the old one: an agent starts from what it was constructed with, so a merge
    /// would take that choice away.
    pub chat_ctx: Option<Vec<lk_agents_abi::ChatItem>>,
    /// `NotGiven` keeps, `Given(x)` installs, `Given(None)` is `Agent(llm=None)`. The host picks
    /// `NotGiven` on identity, so warm state is preserved.
    pub llm: Maybe<Option<lk_agents_abi::resources::LlmRef>>,
    pub tts: Maybe<Option<lk_agents_abi::resources::TtsRef>>,
    /// `Given` closes the leg and reopens it.
    pub stt: Maybe<Option<lk_agents_abi::resources::SttRef>>,
    pub vad: Maybe<Option<lk_agents_abi::resources::VadRef>>,
    /// **Replaced whole** and re-resolved over the resulting legs.
    pub options: crate::agent_activity::options::AgentSessionOptions,
    /// The agent's `expressive`, replacing the previous agent's; `None` defers to the session's.
    pub expressive: Option<crate::agent_activity::options::Expressive>,
    /// The agent's `transcription_node`, replacing the previous
    /// agent's.
    pub output_nodes: crate::agent_session::OutputNodes,
    pub call: Option<RequestId>,
    /// The spans its toolset setup ran under, which the handoff opens.
    pub activation: Option<crate::telemetry::Activation>,
}

/// Checked before and after the drain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandoffRefused {
    Closing,
    SetupFailed,
    DrainRefused,
    /// A `VoiceTask` holds the floor; its parent waits on it.
    TaskRunning,
}

/// The provider a three-state argument hands over.
fn handed<R>(arg: &Maybe<Option<R>>, id: impl Fn(&R) -> ResourceId) -> Option<ResourceId> {
    match arg {
        Maybe::Given(Some(r)) => Some(id(r)),
        Maybe::Given(None) | Maybe::NotGiven => None,
    }
}

impl Command {
    /// The providers it hands the core.
    pub fn resources(&self) -> Vec<ResourceId> {
        match self {
            Command::Activity(ActivityCommand::Update(update)) => update.resources().collect(),
            Command::VoiceTask(crate::agent_activity::voice_task::VoiceTaskCommand::Run(run)) => {
                run.update.resources().collect()
            }
            Command::Activity(ActivityCommand::SetAudioInput(Some(input))) => vec![input.0],
            Command::Recorder(RecorderCommand::Attach(recorder)) => vec![recorder.0],
            Command::Activity(ActivityCommand::SetAudioOutput(Some(out))) => vec![out.id],
            Command::Activity(ActivityCommand::SetVideoInput(Some(input))) => vec![input.id],
            Command::Activity(ActivityCommand::SetTranscriptionOutput(Some(out))) => vec![out.id],
            Command::Activity(ActivityCommand::UpdateModels(models)) => {
                let ModelUpdate { stt, vad, llm, tts } = models.as_ref();
                [
                    handed(stt, |r| r.id),
                    handed(vad, |r| r.id),
                    handed(llm, |r| r.id),
                    handed(tts, |r| r.id),
                ]
                .into_iter()
                .flatten()
                .collect()
            }
            // Classify new command/activity variants explicitly: no resource-bearing update
            // may silently bypass queue refusal and retirement accounting.
            Command::Activity(
                ActivityCommand::UpdateInstructions(_)
                | ActivityCommand::UpdateTools(_)
                | ActivityCommand::UpdateExpressive(_)
                | ActivityCommand::SetAudioInput(None)
                | ActivityCommand::SetVideoInput(None)
                | ActivityCommand::SetIoEnabled { .. }
                | ActivityCommand::UpdateChatCtx(_)
                | ActivityCommand::SetAudioOutput(None)
                | ActivityCommand::SetTranscriptionOutput(None)
                | ActivityCommand::Drain { .. },
            )
            | Command::VoiceTask(crate::agent_activity::voice_task::VoiceTaskCommand::Cancel {
                ..
            })
            | Command::Recorder(RecorderCommand::Detach)
            | Command::Speech(_)
            | Command::Turn(_)
            | Command::Video(_)
            | Command::Close(_)
            | Command::HookDone { .. }
            | Command::HostMetrics(_)
            | Command::Ivr(_)
            | Command::SetStaticKeyterms(_)
            | Command::Dtmf(_)
            | Command::Run(_)
            | Command::AmdSpan(_) => Vec::new(),
        }
    }
}

impl AgentUpdate {
    /// Registrations retained by a prepared frame until the runtime submits it to the core.
    pub fn resources(&self) -> impl Iterator<Item = ResourceId> {
        // Classify every field so adding another registration requires an ownership review.
        let Self {
            agent: _,
            instructions: _,
            dynamic_instructions,
            toolset: _,
            chat_ctx: _,
            llm,
            tts,
            stt,
            vad,
            options: _,
            expressive: _,
            output_nodes,
            call: _,
            activation: _,
        } = self;
        let crate::agent_session::OutputNodes { transcription } = output_nodes;
        [
            handed(llm, |r| r.id),
            handed(tts, |r| r.id),
            handed(stt, |r| r.id),
            handed(vad, |r| r.id),
            *dynamic_instructions,
            *transcription,
        ]
        .into_iter()
        .flatten()
    }

    pub(crate) fn reopens(&self) -> &'static str {
        match (self.stt.is_given(), self.vad.is_given()) {
            (false, false) => "",
            (true, false) => "stt",
            (false, true) => "vad",
            (true, true) => "stt+vad",
        }
    }
}

impl std::fmt::Debug for AgentUpdate {
    /// Hand-written: providers are not `Debug`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentUpdate")
            .field("instructions", &self.instructions)
            .field("dynamic_instructions", &self.dynamic_instructions.is_some())
            .field("tools", &self.toolset.specs.len())
            .field("chat_ctx_replaced", &self.chat_ctx.is_some())
            .field("llm", &self.llm)
            .field("tts", &self.tts)
            .field("stt", &self.stt)
            .field("vad", &self.vad)
            .field("options", &self.options)
            .field("expressive", &self.expressive)
            .field("call", &self.call)
            .finish()
    }
}
