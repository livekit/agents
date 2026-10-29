//! The remote tool adapter's typed decisions. Live resources remain on the endpoint client.

use serde_json::{Map, Value};

use crate::ChatItem;
use crate::a2a::client::{AgentClientError, Message, TaskUpdate};
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
    #[host(new)]
    fn new(
        endpoint: String,
        tool_id: String,
        function_call: ChatItem,
        history: Vec<ChatItem>,
    ) -> Result<Self, AgentClientError>;

    #[host(call)]
    fn message(
        &self,
        arguments: Value,
        structured: bool,
        offers_extension: bool,
        history: Vec<ChatItem>,
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
    /// Resolve an endpoint credential from explicit input or the configured project's origin.
    #[host(function)]
    fn _core_remote_token(
        endpoint: String,
        #[host(default = None)] token: Option<String>,
    ) -> Result<Option<String>, AgentClientError>;
}
