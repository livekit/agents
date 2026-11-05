//! The remote tool adapter's typed decisions. Live resources remain on the endpoint client.

use serde_json::{Map, Value};

use crate::ChatItem;
use crate::a2a::client::{AgentCard, AgentClientError, Message};
use crate::llm::tool_executor::ToolOutcome;

#[derive(Debug, Clone, PartialEq)]
#[crate::host(record(module = "a2a_remote", emit))]
pub enum RemoteEvent {
    Pending,
    Progress {
        text: String,
    },
    InputRequired {
        question: String,
        request_id: String,
        verbatim: Option<String>,
        verbatim_full: bool,
        outcome: ToolOutcome,
    },
    Outcome {
        outcome: ToolOutcome,
    },
    Follow {
        task_id: String,
        thread_id: String,
    },
}

/// What a delegation runs on the host: its tool call's progress and retained state and, on a
/// call, the person's answers to the expert. Each runs in the tool call's context.
#[crate::host(proxy(host = "livekit.agents.a2a._Delegation"))]
pub trait DelegationHost: Send + Sync {
    /// The interaction changed: `extra` is the function call's retained state, `text` a
    /// progress message for the person (`ctx.update()`).
    #[host(await, name = "_core_progress")]
    fn progress<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        #[host(context)] context: &'a crate::invoke::CallContext,
        text: Option<String>,
        extra: Map<String, Value>,
    ) -> crate::BoxFuture<'a, Result<(), crate::errors::HostError>>;
    /// On a call, the request reached the expert: the conversation goes on while it works.
    #[host(await, name = "_core_delegated")]
    fn delegated<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        #[host(context)] context: &'a crate::invoke::CallContext,
    ) -> crate::BoxFuture<'a, Result<(), crate::errors::HostError>>;
    /// On a call, the expert's question for the person; the person's answer.
    #[host(await, name = "_core_ask")]
    fn ask<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        #[host(context)] context: &'a crate::invoke::CallContext,
        question: String,
        verbatim: bool,
    ) -> crate::BoxFuture<'a, Result<String, crate::errors::HostError>>;
}

/// One delegation of a tool call to a remote agent (docs/agents.md "Remote agents").
#[crate::host(class(name = "_core_RemoteInteraction"))]
pub trait RemoteInteraction: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    /// `project_url`, `api_key` and `api_secret` are the server's LiveKit project, which grants
    /// and verifies a shared thread; empty when there is none.
    #[host(new)]
    fn new(
        endpoint: String,
        tool_id: String,
        function_call: ChatItem,
        history: Vec<ChatItem>,
        project_url: String,
        api_key: String,
        api_secret: String,
    ) -> Result<Self, AgentClientError>;

    /// The call's message; `thread_id` shares the caller's thread under `token`'s grant for it.
    #[host(call)]
    fn message(
        &self,
        arguments: Value,
        structured: bool,
        offers_extension: bool,
        history: Vec<ChatItem>,
        #[host(default = None)] thread_id: Option<String>,
        #[host(default = None)] token: Option<String>,
    ) -> Result<Message, AgentClientError>;

    /// Run the delegation on the endpoint: connect with `token`, send the call's message
    /// (`thread_id` shares the caller's thread under `token`'s grant), then follow its updates:
    /// each reaches `host.progress`, a moved task is followed, and on a call (`live`) the
    /// request is announced through `host.delegated` and each question answered through
    /// `host.ask`. Returns the outcome, or, off a call, the expert's question. The client and
    /// its reply are closed before it returns.
    #[host(await)]
    #[allow(clippy::too_many_arguments, reason = "Mirrors the host call's arguments")]
    async fn run(
        &self,
        env: &Self::Env,
        host: std::sync::Arc<dyn DelegationHost>,
        arguments: Value,
        structured: bool,
        history: Vec<ChatItem>,
        token: Option<String>,
        thread_id: Option<String>,
        conn_options: Option<crate::ApiConnectOptions>,
        live: bool,
        #[host(context)] context: crate::invoke::CallContext,
        cx: crate::telemetry::Cx,
    ) -> Result<RemoteEvent, AgentClientError>;

    /// The tool call was cancelled: the interaction records it and a running delegation stops,
    /// closing its reply (which cancels the remote task) and client; `settled()` waits for that.
    #[host(call)]
    fn cancelled(&self) -> Result<(), AgentClientError>;

    /// After `cancelled()`: the running delegation's cleanup, bounded by the teardown limit.
    #[host(await)]
    async fn settled(&self);

    #[host(getter)]
    fn extra(&self) -> Result<Map<String, Value>, AgentClientError>;
}

#[crate::host(functions)]
pub trait RemoteConfiguration: Send + Sync + Sized + 'static {
    /// The delegation tool declared in code: `name` and `description` are required, and the tool
    /// takes the request, stated in full. Nothing is fetched from the endpoint.
    #[host(function)]
    fn _core_delegation_tool(
        name: String,
        description: String,
    ) -> Result<AgentCard, AgentClientError>;

    /// Resolve an endpoint credential: `token` as given; else, for an endpoint on the server's
    /// project's origin (`project_url`, signed with `api_key`/`api_secret`; empty when there is
    /// none), a grant for the caller's `thread_id`, or a project credential without one.
    #[host(function)]
    fn _core_remote_token(
        endpoint: String,
        project_url: String,
        api_key: String,
        api_secret: String,
        #[host(default = None)] token: Option<String>,
        #[host(default = None)] thread_id: Option<String>,
    ) -> Result<Option<String>, AgentClientError>;
}
