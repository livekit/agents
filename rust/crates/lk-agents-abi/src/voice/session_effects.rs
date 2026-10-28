//! What `start()` decides. The decision is in `lk-agents-voice/src/agent_session/`.

use crate::voice::session_options::RecordingOptions;
use lk_agents_host_derive::Host;

/// The decision `start()` gets back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Host)]
#[binding(emit)]
pub struct StartPlan {
    pub recording: RecordingOptions,
    pub is_primary: bool,
    /// `is_primary and (session_host if given else True)`: a second session must never host.
    pub hosting: bool,
    /// `records_anything` on the resolved options.
    pub records: bool,
}

/// What a filler's callable answered: nothing, its text to say, or anything else (a speech it
/// scheduled itself).
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(extract, untagged)]
#[crate::host(handwritten(extract, host = "str | None | object"))]
pub enum FillerReply {
    #[binding(none)]
    Declined,
    Text(String),
    #[binding(rest)]
    Spoke,
}

/// The agent's hooks, run by the host and awaited by the core: each hook call is answered with
/// what its method returns, or `Failed` when it raises (after `hook_failed`). A hook still running
/// when the loop has returned is cancelled.
#[crate::host(proxy(host = "livekit.agents.voice.agent_session._AgentHooks"))]
pub trait SessionHooks: Send + Sync {
    /// `Agent.on_enter()` for `agent`, the `Agent` the host installed.
    #[host(await, name = "_core_on_enter")]
    fn on_enter<'a>(
        &'a self,
        agent: &'a crate::HostRef,
    ) -> crate::BoxFuture<'a, Result<(), crate::errors::HostError>>;
    /// `Agent.on_exit()`.
    #[host(await, name = "_core_on_exit")]
    fn on_exit<'a>(
        &'a self,
        agent: &'a crate::HostRef,
    ) -> crate::BoxFuture<'a, Result<(), crate::errors::HostError>>;
    /// The toolsets of `agent` (after its `on_exit` was answered), or of the session (`None`, as
    /// it closes) closed.
    #[host(await, name = "_core_close_toolsets")]
    fn close_toolsets<'a>(&'a self, agent: Option<&'a crate::HostRef>) -> crate::BoxFuture<'a, ()>;
    /// The toolsets among `tools` (an agent's and the session's, a list) set up; the specs of every
    /// tool they hold then.
    #[host(await, name = "_core_setup_tools")]
    fn setup_tools<'a>(
        &'a self,
        tools: &'a crate::HostRef,
    ) -> crate::BoxFuture<'a, Result<Vec<crate::ToolSpec>, crate::errors::HostError>>;
    /// The session closes: what runs beside it (AMD) closes first.
    #[host(await, name = "_core_closing")]
    fn closing(&self) -> crate::BoxFuture<'_, Result<(), crate::errors::HostError>>;
    /// `agent.on_user_turn_completed(turn_ctx, new_message)`, `turn_ctx` built from `chat_ctx`
    /// (a copy of the agent's own context): the message's text as the hook left it, the turn's
    /// items when it changed them, and whether it raised `StopResponse`.
    #[host(await, name = "_core_on_user_turn_completed")]
    fn on_user_turn_completed<'a>(
        &'a self,
        agent: &'a crate::HostRef,
        chat_ctx: Vec<crate::ChatItem>,
        new_transcript: &'a str,
        transcript_confidence: f64,
    ) -> crate::BoxFuture<
        'a,
        Result<(Option<String>, Option<Vec<crate::ChatItem>>, bool), crate::errors::HostError>,
    >;
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
