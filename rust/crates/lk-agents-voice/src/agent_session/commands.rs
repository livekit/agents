//! The session's command vocabulary: everything a host or a system may ask of the driver, each
//! answered (when it is) through the channel it carries.

use crate::agent_activity::false_interruption::InterruptRefusal;
use crate::agent_activity::scheduling::SpeechId;
use crate::filler_scheduler::FillerId;
use crate::generation::Instructions;
use crate::speech_handle::SpeechHandle;
use crate::tool_executor::Toolset;
use crate::workflows::{TaskHandle, TaskId};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::voice::io::AudioInput;
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
    Task(TaskCommand),
    Recorder(RecorderCommand),
    Dtmf(DtmfCommand),
    /// Answered by `RunStarted` or `RunRefused`, settled by `RunSettled`.
    Run(Box<crate::run_result::RunSpec>),
}

/// `AgentTask.__await_impl`: the awaiter parks on its [`TaskHandle`], which resolves on `Exited`
/// once the parent frame is back.
#[derive(Debug, lk_agents_abi::Vocabulary)]
pub enum TaskCommand {
    Enter(Box<TaskEntry>),
}

pub struct TaskEntry {
    pub handle: TaskHandle<lk_agents_abi::HostRef>,
    /// `previous`/`next` are the driver's to set.
    pub frame: AgentUpdate,
    /// `None` from `on_enter`/`on_exit`.
    pub speech: Option<SpeechId>,
    pub preserve_function_call_history: bool,
    /// `None`: the awaiting speech's slot, then the driving frame's.
    pub parent: Option<String>,
}

impl std::fmt::Debug for TaskEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskEntry")
            .field("task", &self.handle.id())
            .field("agent", &self.frame.agent)
            .field("speech", &self.speech)
            .field("preserve_function_call_history", &self.preserve_function_call_history)
            .finish()
    }
}

/// Monotonic within a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HookId(pub u64);

/// `AMD.execute()` sends `AddTools`, `Start` then `SeedTranscript` on a `machine-ivr` verdict.
#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
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
    AddTools(crate::tool_executor::SessionTools),
}

#[derive(Debug, lk_agents_abi::Vocabulary)]
pub enum HookOutcome {
    /// A hook that raised is `Failed`, and the sequence continues.
    Returned,
    /// `StopResponse` is one of them.
    Turn(Vec<crate::agent::TurnEdit>),
    Described(Box<AgentUpdate>),
    Rendered(String),
    Filler {
        text: Option<String>,
        spoke: bool,
    },
    /// `function_tools_executed` handled: each output's `reply_required`, in order, and the event's
    /// `handoff_required`, as the handlers left them.
    FunctionToolsExecuted {
        reply_required: Vec<bool>,
        handoff_required: bool,
    },
    /// `on_user_turn_completed` skips the reply; a describe drops the handoff.
    Failed,
}

/// The host and the systems are peers that write only commands under their own origin; the core
/// names no system.
#[derive(Debug, Clone, PartialEq, Eq, lk_agents_abi::Vocabulary)]
pub enum Origin {
    Host,
    Tool { call_id: String },
    Hook,
    Task,
    Amd,
    Ivr,
}

/// `"host"`, `"hook"`, `"task"` or `"tool:<call_id>"`; anything else is refused.
impl std::str::FromStr for Origin {
    type Err = String;

    fn from_str(s: &str) -> Result<Origin, String> {
        match s {
            "host" => Ok(Origin::Host),
            "hook" => Ok(Origin::Hook),
            "task" => Ok(Origin::Task),
            other => match other.strip_prefix("tool:") {
                Some(call_id) if !call_id.is_empty() => {
                    Ok(Origin::Tool { call_id: call_id.to_owned() })
                }
                Some(_) | None => Err(format!(
                    "origin must be None, \"host\", \"hook\", \"task\" or \"tool:<call_id>\", got {other:?}"
                )),
            },
        }
    }
}

/// A `commit_user_turn()` superseded by a newer one; the host's future is cancelled.
#[lk_agents_abi::host(error(kind = "asyncio.CancelledError"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("superseded by a newer commit_user_turn()")]
pub struct CommitSuperseded;

/// The id a caller gives a command it awaits; the answer carries it back. `None` where a command
/// takes one means nobody awaits the answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestId(pub u64);

/// What an awaited command is answered with, once, on the session's answer stream.
#[derive(Debug)]
pub enum Answer {
    Speech(Arc<SpeechHandle>),
    Interrupt(Result<(), InterruptRefusal>),
    Commit(Result<String, CommitSuperseded>),
    Idle(Result<(), IdleRefused>),
    Hold(Result<HoldId, IdleRefused>),
    Drain(Result<(), DrainRefused>),
    Handoff(Result<(), HandoffRefused>),
    Dtmf(crate::workflows::builtins::dtmf::DtmfOutcome),
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
    speech: Speech(Arc<SpeechHandle>),
    interrupt: Interrupt(Result<(), InterruptRefusal>),
    commit: Commit(Result<String, CommitSuperseded>),
    idle: Idle(Result<(), IdleRefused>),
    hold: Hold(Result<HoldId, IdleRefused>),
    drain: Drain(Result<(), DrainRefused>),
    handoff: Handoff(Result<(), HandoffRefused>),
    dtmf: Dtmf(crate::workflows::builtins::dtmf::DtmfOutcome),
}

#[derive(Debug)]
pub struct Sent {
    pub origin: Origin,
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
                ReleaseUserTurn, SetAllowInterruptions, SetRunOutput, ToolUpdate, FillerStart,
                FillerStop, PauseAuthorization, ResumeAuthorization),
            Command::Activity(c) => leaf!(c; ActivityCommand: Update, UpdateInstructions,
                UpdateTools, SetAudioInput, SetVideoInput, SetIoEnabled, UpdateChatCtx, SetAudioOutput, SetTranscriptionOutput, UpdateModels, Drain),
            Command::Turn(c) => leaf!(c; TurnCommand: Commit, Clear, Configure),
            Command::Ivr(c) => leaf!(c; IvrCommand: Start, SeedTranscript, Close, AddTools),
            Command::Task(c) => leaf!(c; TaskCommand: Enter),
            Command::Dtmf(c) => leaf!(c; DtmfCommand: Start, Digit, Stop),
            Command::Video(_) => "Command::Video",
            Command::Close(_) => "Command::Close",
            Command::HookDone { .. } => "Command::HookDone",
            Command::HostMetrics(_) => "Command::HostMetrics",
            Command::SetStaticKeyterms(_) => "Command::SetStaticKeyterms",
            Command::Run(_) => "Command::Run",
            Command::Recorder(RecorderCommand::Attach(_)) => "Command::Recorder::Attach",
            Command::Recorder(RecorderCommand::Detach) => "Command::Recorder::Detach",
        }
    }
}

#[derive(Debug)]
pub struct Close {
    pub reason: lk_agents_abi::voice::events::CloseReason,
    pub drain: bool,
    pub error: Option<lk_agents_abi::errors::ProviderError>,
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

impl From<TurnCommand> for Command {
    fn from(c: TurnCommand) -> Command {
        Command::Turn(c)
    }
}

impl From<TaskCommand> for Command {
    fn from(c: TaskCommand) -> Command {
        Command::Task(c)
    }
}

/// Outlives the frame across handoffs.
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
    Configure(crate::agent_activity::options::SessionPatch),
}

/// One collection at a time.
#[derive(Debug, lk_agents_abi::Vocabulary)]
pub enum DtmfCommand {
    /// Answered once with the first outcome, or `Stopped`; the confirm arm is spoken by the
    /// driver.
    Start {
        collector: crate::workflows::builtins::dtmf::DtmfCollector,
        call: Option<RequestId>,
    },
    /// Ignored with no collection running.
    Digit {
        digit: String,
    },
    Stop,
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
        call: Option<RequestId>,
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
        on_enter: bool,
        call: Option<RequestId>,
    },
    /// A pending preemptive generation is dropped; the current speech is interrupted (a refusal is
    /// the whole answer); the socket is told once; the queue is walked in pop order up to the first
    /// refusal, each speech whole.
    Interrupt { force: bool, call: Option<RequestId> },
    /// Unforced it is scoped to the playing step; `force` ends the speech, tools and all.
    InterruptSpeech { speech: SpeechId, force: bool },
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
    WaitForIdle { hold: Option<HoldId>, call: Option<RequestId> },
    /// `RunContext.foreground()`: wait for idle, then hold it until `ReleaseIdleHold`; a waiter naming
    /// the hold passes through.
    WaitForIdleAndHold { call: Option<RequestId> },
    /// Unknown ids are a no-op.
    ReleaseIdleHold { hold: HoldId },
    /// Pins the user state to `speaking` and holds idle open. Reentrant.
    ClaimUserTurn,
    /// The last release re-derives the user state.
    ReleaseUserTurn,
    /// Refused quietly on a done handle.
    SetAllowInterruptions { speech: SpeechId, allow: bool },
    /// `AgentTask.complete(result)` from `speech`'s tool: the run ending on that speech reads it as
    /// output.
    SetRunOutput { speech: SpeechId, output: crate::run_result::RunOutput },
    /// A background tool speaks again after its first `ctx.update()`: the driver mints and commits
    /// the `{call_id}_update_N` / `{call_id}_final` pair and coalesces one deferred reply at idle.
    ToolUpdate { call_id: String, update: ToolUpdate },
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
    FillerStop { id: FillerId },
    /// `Origin::Amd` only: nothing is authorized until `ResumeAuthorization`. A flag, not a
    /// counter.
    PauseAuthorization,
    /// The head is authorized in this cycle.
    ResumeAuthorization,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FillerSource {
    Text(String),
    /// Asked at every fire.
    Host,
}

#[derive(Debug)]
pub enum ToolUpdate {
    /// `raw` as written; `message` rendered by the host.
    Progress { message: String, raw: String },
    /// Decided like a blocking tool's, except an `Agent` is refused and a silent return voices
    /// nothing.
    Final { answer: crate::tool_executor::ToolAnswer },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HoldId(pub u64);

/// Why a `WaitForIdle`/`WaitForIdleAndHold` was answered without reaching idle: `_core.SessionClosing`,
/// which the shim raises as `ActivityClosedError`.
#[lk_agents_abi::host(error(kind = "livekit.agents._core.SessionClosing"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, lk_agents_abi::Vocabulary)]
pub enum IdleRefused {
    /// The session closed (or its loop ended) first: the reference's `ActivityClosedError`.
    Closing,
}

impl std::fmt::Display for IdleRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("session is closing")
    }
}

#[derive(Debug)]
pub enum SayText {
    Text(String),
    Audio {
        text: String,
        rx: tokio::sync::mpsc::Receiver<lk_agents_abi::AudioFrame>,
    },
    /// Closing the sender ends the speech.
    Stream(tokio::sync::mpsc::Receiver<String>),
}

pub struct HostAudio(pub Arc<dyn AudioInput>);

/// Host handles are newtypes so the commands carrying them keep their `Debug`.
macro_rules! host_handle_debug {
    ($T:ident) => {
        impl std::fmt::Debug for $T {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(stringify!($T))
            }
        }
    };
    ($T:ident labelled) => {
        impl std::fmt::Debug for $T {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_tuple(stringify!($T)).field(&self.0.label()).finish()
            }
        }
    };
}
host_handle_debug!(HostAudio);
host_handle_debug!(HostRecorder);
host_handle_debug!(HostVideo labelled);

#[derive(Debug)]
pub enum RecorderCommand {
    /// Flushes every 2.5 s of session time.
    Attach(HostRecorder),
    /// The file's `finish` stays the host's.
    Detach,
}

pub struct HostRecorder(pub Arc<dyn lk_agents_media::recorder::AudioRecorder>);

pub struct HostVideo(pub Arc<dyn lk_agents_abi::voice::io::VideoInput>);

/// Video input is handled at the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub enum IoKind {
    AudioInput,
    AudioOutput,
    Transcription,
}

/// Scheduling pauses are the driver's own, never a command.
#[derive(Debug, lk_agents_abi::Vocabulary)]
pub enum ActivityCommand {
    /// No stream is torn down, so no word is dropped. Order (`_update_activity`): refuse when
    /// closing; `on_exit` before the pause; drain; closing re-check; swap; `on_enter` not awaited.
    /// A second queues.
    Update(Box<AgentUpdate>),
    // --- the running frame, changed in place: not a handoff, which replaces the whole frame ---
    /// Replies in flight keep theirs. Carries the full triple: workflow tasks put their persona in
    /// the modality variants.
    UpdateInstructions(Instructions),
    UpdateTools(crate::tool_executor::Toolset),
    /// The old pump is aborted first, so two sources never feed the legs at once.
    SetAudioInput(Option<HostAudio>),
    SetVideoInput(Option<HostVideo>),
    /// The leg stays: a muted mic stops feeding STT; a disabled output plans speech without a
    /// voice.
    SetIoEnabled {
        leg: IoKind,
        enabled: bool,
    },
    /// **Replaces** the running frame's context (already filtered host-side).
    UpdateChatCtx(Vec<lk_agents_abi::ChatItem>),
    /// A playing speech is **interrupted** (nobody can hear it); the pause facts are re-read at
    /// once.
    SetAudioOutput(Option<lk_agents_abi::resources::AudioOutputRef>),
    /// A reply in flight keeps its leg.
    SetTranscriptionOutput(Option<lk_agents_abi::resources::TextOutputRef>),
    /// **Not a handoff**: only the providers change; the streaming legs reopen as a handoff reopens
    /// them.
    UpdateModels {
        stt: Maybe<Option<lk_agents_abi::resources::SttRef>>,
        vad: Maybe<Option<lk_agents_abi::resources::VadRef>>,
        llm: Maybe<Option<lk_agents_abi::resources::LlmRef>>,
        tts: Maybe<Option<lk_agents_abi::resources::TtsRef>>,
    },
    /// Runs the driving frame's `on_exit` (once per frame), then pauses scheduling until the speech
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
    /// Scheduling was resumed under the drain's pause: the frame runs again.
    ActivityChanging,
}

impl DrainRefused {
    pub fn as_str(self) -> &'static str {
        match self {
            DrainRefused::Closing => "closing",
            DrainRefused::ActivityChanging => "activity_changing",
        }
    }
}

pub struct AgentUpdate {
    /// Stamped on every `SessionEvent::ConversationItemAdded`; the host maps it back to its `Agent`.
    pub agent: String,
    /// Workflow tasks put their persona in the modality variants.
    pub instructions: Instructions,
    /// The executor is per-agent.
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
    pub options: crate::agent_activity::options::SessionPatch,
    /// `Close` runs the leaving frame's `on_exit`; `Pause` keeps it for a later resume, no hook.
    pub previous: PreviousActivity,
    /// `Start` runs the arriving frame's `on_enter`; `Resume` does not.
    pub next: NewActivity,
    pub call: Option<RequestId>,
    /// `None` is a host handoff.
    pub task: Option<TaskId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviousActivity {
    Close,
    Pause,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NewActivity {
    Start,
    Resume,
}

/// Checked before and after the drain. A resume while closing is not refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandoffRefused {
    Closing,
    DrainRefused,
    /// Or never answered within the bound.
    NotDescribed,
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
    pub(crate) fn resources(&self) -> Vec<ResourceId> {
        match self {
            Command::Activity(ActivityCommand::Update(update))
            | Command::HookDone { outcome: HookOutcome::Described(update), .. } => {
                update.resources().collect()
            }
            Command::Task(TaskCommand::Enter(entry)) => entry.frame.resources().collect(),
            Command::Activity(ActivityCommand::SetAudioOutput(Some(out))) => vec![out.id],
            Command::Activity(ActivityCommand::SetTranscriptionOutput(Some(out))) => vec![out.id],
            Command::Activity(ActivityCommand::UpdateModels { stt, vad, llm, tts }) => [
                handed(stt, |r| r.id),
                handed(vad, |r| r.id),
                handed(llm, |r| r.id),
                handed(tts, |r| r.id),
            ]
            .into_iter()
            .flatten()
            .collect(),
            _ => Vec::new(),
        }
    }
}

impl AgentUpdate {
    /// The providers it hands over.
    pub(crate) fn resources(&self) -> impl Iterator<Item = ResourceId> {
        [
            handed(&self.llm, |r| r.id),
            handed(&self.tts, |r| r.id),
            handed(&self.stt, |r| r.id),
            handed(&self.vad, |r| r.id),
        ]
        .into_iter()
        .flatten()
    }

    pub fn frame(agent: impl Into<String>, instructions: Instructions) -> Self {
        AgentUpdate {
            agent: agent.into(),
            instructions,
            toolset: crate::tool_executor::Toolset::default(),
            chat_ctx: None,
            llm: Maybe::NotGiven,
            tts: Maybe::NotGiven,
            stt: Maybe::NotGiven,
            vad: Maybe::NotGiven,
            options: Default::default(),
            previous: PreviousActivity::Close,
            next: NewActivity::Start,
            call: None,
            task: None,
        }
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
            .field("tools", &self.toolset.specs.len())
            .field("chat_ctx_replaced", &self.chat_ctx.is_some())
            .field("llm", &self.llm)
            .field("tts", &self.tts)
            .field("stt", &self.stt)
            .field("vad", &self.vad)
            .field("options", &self.options)
            .field("previous", &self.previous)
            .field("next", &self.next)
            .field("call", &self.call)
            .finish()
    }
}

// `IdleRefused`'s `#[host(error(kind = "livekit.agents._core.SessionClosing"))]` names this class.
#[cfg(feature = "python")]
pyo3::create_exception!(
    _core,
    SessionClosing,
    pyo3::exceptions::PyRuntimeError,
    "`wait_for_idle()`/`wait_for_idle_and_hold()` refused: the session is closing. The shim raises `ActivityClosedError`."
);

#[cfg(feature = "python")]
lk_agents_abi::python::inventory::submit! {
    lk_agents_abi::python::Registration(|m| {
        use pyo3::types::PyModuleMethods as _;
        m.add("SessionClosing", m.py().get_type::<SessionClosing>())
    })
}
