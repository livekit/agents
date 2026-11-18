use serde_json::{Map, Value};

use crate::ChatItem;
use crate::a2a::client::{AgentCard, AgentClientError, Message};
use crate::llm::tool_executor::ToolOutcome;

/// The storage partition whose caller remains active during a delegation.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
#[crate::host(record(module = "a2a_remote", emit, extract))]
pub struct DelegationPartition {
    pub storage_location: String,
    pub storage_project: String,
    pub thread_id: String,
    pub partition_key: String,
}

/// Internal handoff to a live delegate. Local interactions receive text in `request`;
/// remote task messages can carry business input without claiming a response reservation.
#[derive(Debug, Clone, PartialEq)]
#[crate::host(record(module = "a2a_remote", emit, extract))]
pub struct DelegationInput {
    pub arguments: Value,
    pub structured: bool,
    pub input_request: Option<crate::session::InputRequestRef>,
}

/// Transient ownership of one exact canonical response before its journal Prepare. Only
/// native admission receipts and concrete queued/running submissions retain this token.
pub struct RemoteResponseClaim;

/// Native resources accompany pure routing data without entering snapshots or host APIs.
#[derive(Clone)]
pub struct RemoteCallAdmission {
    pub admission: crate::delegation::routing::CallAdmission,
    pub driver: Option<std::sync::Arc<dyn RemoteTaskDriver>>,
    pub response_claim: Option<std::sync::Arc<RemoteResponseClaim>>,
}

/// The one live native owner of a task's messages and observations. A conversation retains
/// weak handles only; a restored task reference is never authority to revive an old driver.
pub trait RemoteTaskDriver: Send + Sync {
    /// A sender retained by another caller cannot keep a dead command consumer eligible.
    fn is_live(&self) -> bool;

    /// Resolves after the peer acknowledges the message. Dropping this waiter does not
    /// retract a message already admitted by the driver.
    fn input<'a>(
        &'a self,
        input: DelegationInput,
        response_claim: Option<std::sync::Arc<RemoteResponseClaim>>,
        origin: Option<crate::session::ToolCallRef>,
    ) -> crate::BoxFuture<'a, Result<(), AgentClientError>>;

    /// Dispatch the immutable operation an owner proved was prepared but never attempted.
    /// It uses the same observation owner and retains the saved message/request identities.
    fn resume<'a>(
        &'a self,
        request_id: String,
        request: Box<crate::a2a::protocol::SendMessageRequest>,
        streaming: bool,
        input_request: Option<crate::session::InputRequestRef>,
        origin: Option<crate::session::ToolCallRef>,
    ) -> crate::BoxFuture<'a, Result<(), AgentClientError>>;

    /// An explicit task control, serialized with its messages and observations. Success
    /// requires checkpointed evidence of cancellation, rather than dispatch alone.
    fn cancel<'a>(
        &'a self,
        task: crate::delegation::remote::RemoteTaskRef,
    ) -> crate::BoxFuture<'a, Result<(), AgentClientError>>;

    /// Reconcile already-admitted work through this task's original observer. Observation
    /// failure is delivered to its original call; the accepted duplicate must not fail or
    /// suggest retrying business work. The wait also ends when the observer dies.
    fn reconcile(&self) -> crate::BoxFuture<'_, ()>;
}

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
#[crate::host(proxy(host = "livekit.agents.a2a._Delegation", owns))]
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
    /// Present a remote question; its response returns through the task's native driver.
    #[host(await, name = "_core_question")]
    fn question<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        #[host(context)] context: &'a crate::invoke::CallContext,
        question: String,
        verbatim: bool,
        input_request: Option<crate::session::InputRequestRef>,
    ) -> crate::BoxFuture<'a, Result<(), crate::errors::HostError>>;
    /// A transient local interaction's question and its in-memory answer.
    #[host(await, name = "_core_ask")]
    fn ask<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        #[host(context)] context: &'a crate::invoke::CallContext,
        question: String,
        verbatim: bool,
        input_request: Option<crate::session::InputRequestRef>,
    ) -> crate::BoxFuture<'a, Result<DelegationInput, crate::errors::HostError>>;
}

/// One delegation of a tool call to a remote agent.
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
        project_url: String,
        api_key: String,
        api_secret: String,
    ) -> Result<Self, AgentClientError>;

    /// Caller identities carried to the receiving endpoint for partition re-entry checks.
    #[host(call)]
    fn set_partition_chain(&self, chain: Vec<DelegationPartition>) -> Result<(), AgentClientError>;

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
    /// request releases the frontend through `host.delegated` and each question is presented
    /// through `host.question`. Task messages reach the same native driver. Returns the outcome,
    /// or, off a call, the expert's question. The client and its reply close before it returns.
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
        #[host(default = None)] owner: Option<std::sync::Arc<dyn crate::session::SessionCommands>>,
        #[host(default = None)] origin: Option<crate::session::ToolCallRef>,
        #[host(context)] context: crate::invoke::CallContext,
        cx: crate::telemetry::Cx,
    ) -> Result<RemoteEvent, AgentClientError>;

    /// Stop the local tool wait and close its observation resources. Remote business work is
    /// unchanged; `settled()` waits for local cleanup, not remote task cancellation.
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
    #[host(function)]
    fn _core_delegation_chain(
        metadata: Map<String, Value>,
    ) -> Result<Vec<DelegationPartition>, AgentClientError>;

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
