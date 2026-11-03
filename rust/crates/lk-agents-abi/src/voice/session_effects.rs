//! What `start()` decides. The decision is in `lk-agents-voice/src/agent_session/`.

/// A user-turn hook result: updated text, history and whether it stopped the response.
pub type CompletedUserTurn = (Option<String>, Option<Vec<crate::ChatItem>>, bool);

use crate::voice::session_options::RecordingOptions;
use std::sync::Arc;

#[crate::host(file(module = "session_wire"))]
const _: () = ();

#[crate::host(constant)]
pub const DEFAULT_HOOK_BOUND: f64 = 15.0;
#[crate::host(constant)]
pub const DEFAULT_MAX_UNRECOVERABLE_ERRORS: i64 = 3;
#[crate::host(constant)]
pub const DEFAULT_SPEAKING_FPS: f64 = 1.0;
#[crate::host(constant)]
pub const DEFAULT_SILENT_FPS: f64 = 0.3;

#[crate::host(record(module = "session_wire", emit))]
#[derive(Clone)]
pub struct DefaultModelPlan {
    pub vad_model: Option<DefaultVadModel>,
    pub turn_detector: bool,
}

#[crate::host(record(module = "session_wire", emit))]
#[derive(Clone)]
pub enum DefaultVadModel {
    #[host(rename = "silero")]
    Silero,
}

#[crate::host(record(module = "session_wire", typed_dict, emit))]
#[derive(Clone)]
pub struct ExceededTurnReply {
    pub instructions: String,
    pub allow_interruptions: bool,
    pub tool_choice: crate::ToolChoice,
}

/// The decision `start()` gets back.
#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StartPlan {
    pub recording: RecordingOptions,
    pub is_primary: bool,
    /// `is_primary and (session_host if given else True)`: a second session must never host.
    pub hosting: bool,
    /// `records_anything` on the resolved options.
    pub records: bool,
}

/// How a job's session starts, decided before its loop opens.
#[crate::host(functions)]
pub trait SessionStart {
    /// Retain requested local tools and every provider-owned tool, in declaration order.
    #[host(function)]
    fn reply_tool_indices(tools: Vec<crate::ToolSpec>, names: Vec<String>) -> Vec<usize>;
    #[host(function)]
    fn default_session_models(
        vad_supplied: bool,
        detector_supplied: bool,
        realtime: bool,
    ) -> DefaultModelPlan;

    #[host(function)]
    fn user_turn_exceeded_reply() -> ExceededTurnReply;
    /// The job's recording and whether this session is its primary (records) and hosts the
    /// remote session protocol; `another_session_is_primary` is the job's claim already taken.
    #[host(function)]
    fn plan_session_start(
        #[host(kw)] job_default: bool,
        another_session_is_primary: bool,
    ) -> StartPlan;
}

/// What a filler's callable answered: nothing, its text to say, or anything else (a speech it
/// scheduled itself).
#[crate::host(record(module = "session_effects", extract, transparent))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FillerReply {
    Declined,
    Text(String),
    #[host(host_type = "builtins.object")]
    Spoke,
}

/// The agent's hooks, run by the host and awaited by the core: each hook call is answered with
/// what its method returns, or `Failed` when it raises (after `hook_failed`). A hook still running
/// when the loop has returned is cancelled.
#[crate::host(proxy(host = "livekit.agents.voice.agent_session._AgentHooks"))]
pub trait SessionHooks: Send + Sync {
    /// `Agent.on_enter()` for `agent`, the `Agent` the host installed, with its `on_enter` span
    /// current (`trace`).
    #[host(await, name = "_core_on_enter")]
    fn on_enter<'a>(
        &'a self,
        agent: &'a crate::HostRef,
        #[host(context)] trace: crate::telemetry::record::TraceContext,
    ) -> crate::BoxFuture<'a, Result<(), crate::errors::HostError>>;
    /// `Agent.on_exit()`, with its `on_exit` span current.
    #[host(await, name = "_core_on_exit")]
    fn on_exit<'a>(
        &'a self,
        agent: &'a crate::HostRef,
        #[host(context)] trace: crate::telemetry::record::TraceContext,
    ) -> crate::BoxFuture<'a, Result<(), crate::errors::HostError>>;
    /// Framework ownership is native; the host only identifies its Python toolset objects.
    #[host(read_once, name = "_core_toolset_scopes")]
    fn toolset_scopes(&self) -> &Arc<dyn crate::llm::toolsets::ScopesPort>;
    #[host(call, name = "_core_toolset_resources")]
    fn toolset_resources(
        &self,
        tools: &crate::HostRef,
    ) -> Result<Vec<Arc<dyn crate::llm::toolsets::ToolsetHooks>>, crate::errors::HostError>;
    /// Python reflection runs after native acquisition so dynamically registered tools are visible.
    #[host(call, name = "_core_tool_infos")]
    fn tool_infos(
        &self,
        tools: &crate::HostRef,
    ) -> Result<Vec<crate::agent::FunctionToolInfo>, crate::errors::HostError>;
    /// The session closes: what runs beside it (AMD) closes first.
    #[host(await, name = "_core_closing")]
    fn closing(&self) -> crate::BoxFuture<'_, Result<(), crate::errors::HostError>>;
    /// `agent.on_user_turn_completed(turn_ctx, new_message)`, `turn_ctx` built from `chat_ctx`
    /// (a copy of the agent's own context): the message's text as the hook left it, the turn's
    /// items when it changed them, and whether it raised `StopResponse`. Its span is current.
    #[host(await, name = "_core_on_user_turn_completed")]
    fn on_user_turn_completed<'a>(
        &'a self,
        agent: &'a crate::HostRef,
        chat_ctx: Vec<crate::ChatItem>,
        new_transcript: &'a str,
        transcript_confidence: f64,
        #[host(context)] trace: crate::telemetry::record::TraceContext,
    ) -> crate::BoxFuture<'a, Result<CompletedUserTurn, crate::errors::HostError>>;
    /// `agent.on_user_turn_exceeded(ev)`.
    #[host(await, name = "_core_on_user_turn_exceeded")]
    fn on_user_turn_exceeded<'a>(
        &'a self,
        agent: &'a crate::HostRef,
        ev: &'a crate::voice::events::UserTurnExceededEvent,
    ) -> crate::BoxFuture<'a, Result<(), crate::errors::HostError>>;
    /// The `function_tools_executed` event emitted: each output's `reply_required` as its
    /// handlers left it.
    #[host(await, name = "_core_function_tools_executed")]
    fn function_tools_executed(
        &self,
        function_calls: Vec<crate::ChatItem>,
        function_call_outputs: Vec<crate::ChatItem>,
    ) -> crate::BoxFuture<'_, Result<Vec<bool>, crate::errors::HostError>>;
    /// A filler's callable `source` called for `step`, awaited when it returns an awaitable.
    #[host(await, name = "_core_filler_source")]
    fn filler_source<'a>(
        &'a self,
        source: &'a crate::HostRef,
        step: u32,
    ) -> crate::BoxFuture<'a, Result<FillerReply, crate::errors::HostError>>;
    /// Hook `hook` raised `error` (the exception itself, when there is one): logged.
    #[host(notify, name = "_core_hook_failed")]
    fn hook_failed(&self, hook: &str, error: Option<crate::errors::CapturedError>);
}

/// Where a host session's events go: one notification per event, delivered in driver order on
/// the host loop.
#[crate::host(proxy(host = "livekit.agents.voice.agent_session.VoiceSession"))]
pub trait SessionEvents: Send + Sync {
    #[host(notify, name = "_core_event")]
    fn event(&self, wire: crate::voice::events::HostWire);

    /// `str()` of the error record `error` is emitted with: the message the remote session sends.
    #[host(call, name = "_core_error_message")]
    fn error_message(
        &self,
        error: &crate::voice::events::SessionErrorWire,
    ) -> Result<String, crate::errors::HostError> {
        Ok(error.error.to_string())
    }
}
