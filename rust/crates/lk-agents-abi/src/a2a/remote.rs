//! The remote tool adapter's typed decisions. Live resources remain on the endpoint client.

use serde_json::{Map, Value};

use crate::ChatItem;
use crate::a2a::client::{AgentCard, AgentClientError, Message, TaskUpdate};
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

#[crate::host(class(name = "_core_RemoteInteraction"))]
pub trait RemoteInteraction: Send + Sync + Sized + 'static {
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

    /// The person's answer to the retained input request `request_id`, on its task; refused
    /// once that request was settled or superseded.
    #[host(call)]
    fn answer(&self, request_id: String, text: String) -> Result<Message, AgentClientError>;

    #[host(call)]
    fn update(&self, update: TaskUpdate) -> Result<RemoteEvent, AgentClientError>;

    #[host(call)]
    fn cancelled(&self) -> Result<(), AgentClientError>;

    #[host(getter)]
    fn extra(&self) -> Result<Map<String, Value>, AgentClientError>;
}

#[crate::host(functions)]
pub trait RemoteConfiguration: Send + Sync + Sized + 'static {
    /// The delegation tool declared in code: `name` and `description` are required, and the tool
    /// takes the request, stated in full. Nothing is fetched from the endpoint.
    #[host(function)]
    fn _core_delegation_tool(name: String, description: String)
    -> Result<AgentCard, AgentClientError>;

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
