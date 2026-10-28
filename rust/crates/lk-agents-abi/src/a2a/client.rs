//! The endpoint client's generated records and acknowledged reply resources.

use serde_json::{Map, Value};

use crate::errors::ProviderError;
use crate::{ApiConnectOptions, ChatItem};

#[derive(Debug, Clone, Default, PartialEq)]
#[crate::host(record(module = "a2a_client", emit, extract))]
pub struct MessageMetadata {
    #[host(default = None)]
    pub kind: Option<String>,
    #[host(default = None)]
    pub channel: Option<String>,
    #[host(default = None)]
    pub verbatim: Option<String>,
    #[host(default = None)]
    pub answer_tool: Option<String>,
    #[host(default = None)]
    pub merged_into: Option<String>,
    #[host(default = None)]
    pub reason: Option<String>,
    #[host(default = None)]
    pub interrupted: Option<bool>,
    #[host(default = None)]
    pub interruption_reason: Option<String>,
    #[host(default = None)]
    pub application: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, PartialEq)]
#[crate::host(record(module = "a2a_client", emit, extract))]
pub enum MessagePart {
    #[host(name = "TextPart")]
    TextPart {
        text: String,
        #[host(default = None)]
        metadata: Option<Map<String, Value>>,
    },
    #[host(name = "DataPart")]
    DataPart {
        data: Value,
        #[host(default = None)]
        metadata: Option<Map<String, Value>>,
    },
    #[host(name = "FilePart")]
    FilePart {
        #[host(default = None)]
        url: Option<String>,
        #[host(default = None)]
        data: Option<Vec<u8>>,
        #[host(default = None)]
        filename: Option<String>,
        #[host(default = None)]
        media_type: Option<String>,
        #[host(default = None)]
        metadata: Option<Map<String, Value>>,
    },
}

#[derive(Debug, Clone, PartialEq)]
#[crate::host(record(module = "a2a_client", emit, extract))]
#[python(
    property(
        name = "text",
        ret = "str",
        hook = "_core_a2a_message_text",
        doc = "The text parts, joined by newlines."
    ),
    property(
        name = "data",
        ret = "dict[str, object] | None",
        hook = "_core_a2a_message_data",
        doc = "The first object data part, if any."
    )
)]
pub struct Message {
    pub parts: Vec<MessagePart>,
    #[host(default = None)]
    pub metadata: Option<MessageMetadata>,
    #[host(default = None)]
    pub message_id: Option<String>,
    #[host(default = None)]
    pub task_id: Option<String>,
    #[host(default = None)]
    pub context_id: Option<String>,
    #[host(default = "user")]
    pub role: String,
    #[host(default = None)]
    pub channel: Option<String>,
    #[host(default = None)]
    pub received_at: Option<f64>,
    #[host(default = None)]
    pub raw: Option<Value>,
}

/// The text parts of a generated A2A message, joined by newlines.
#[crate::host(export)]
pub fn _core_a2a_message_text(message: Message) -> String {
    message
        .parts
        .iter()
        .filter_map(|part| match part {
            MessagePart::TextPart { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The first object data part of a generated A2A message.
#[crate::host(export)]
pub fn _core_a2a_message_data(message: Message) -> Option<Map<String, Value>> {
    message.parts.into_iter().find_map(|part| match part {
        MessagePart::DataPart { data: Value::Object(data), .. } => Some(data),
        _ => None,
    })
}

/// A string is shorthand for a generated message with one text part.
#[derive(Debug, Clone)]
#[crate::host(handwritten(extract, host = "str | Message"))]
pub enum MessageInput {
    Text(String),
    Message(Message),
}

#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for MessageInput {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::types::PyAnyMethods;
        if obj.is_instance_of::<pyo3::types::PyString>() {
            Ok(Self::Text(obj.extract()?))
        } else {
            Ok(Self::Message(obj.extract()?))
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
#[crate::host(record(module = "a2a_client", emit, extract))]
pub struct Artifact {
    pub artifact_id: String,
    #[host(default = None)]
    pub name: Option<String>,
    #[host(default = None)]
    pub description: Option<String>,
    pub parts: Vec<MessagePart>,
    pub metadata: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq)]
#[crate::host(record(module = "a2a_client", emit, extract))]
pub struct TaskUpdate {
    pub state: String,
    pub text: String,
    pub task_id: String,
    pub context_id: String,
    pub metadata: Map<String, Value>,
    pub artifacts: Vec<Artifact>,
}

#[derive(Debug, Clone, PartialEq)]
#[crate::host(record(module = "a2a_client", emit))]
pub struct AgentCard {
    pub name: String,
    pub description: String,
    pub extensions: Vec<String>,
    pub input_schema: Map<String, Value>,
    pub structured_input: bool,
}

#[derive(Debug, thiserror::Error)]
#[crate::host(error(kind = "RuntimeError"))]
pub enum AgentClientError {
    #[error("{0}")]
    #[host(kind = "ValueError")]
    InvalidInput(String),
    #[error("{0}")]
    Protocol(String),
    #[error("another pull of this reply is still in flight")]
    #[host(kind = "livekit.agents._exceptions.StreamStateError")]
    Overlap,
    #[error(transparent)]
    #[host(delegate)]
    Provider(#[from] ProviderError),
}

#[crate::host(class(name = "Reply"))]
pub trait Reply: Send + Sync + Sized + 'static {
    #[host(getter)]
    fn task_id(&self) -> String;
    #[host(getter)]
    fn thread_id(&self) -> String;
    #[host(next)]
    async fn next(&self) -> Result<Option<TaskUpdate>, AgentClientError>;
    #[host(await)]
    async fn aclose(&self);
}

#[crate::host(class(name = "AgentClient"))]
pub trait AgentClient: Send + Sync + Sized + 'static {
    #[host(class(name = "Reply"))]
    type Reply;

    #[host(new)]
    fn new(
        endpoint_url: String,
        #[host(kw, default = None)] token: Option<String>,
        #[host(default = None)] conn_options: Option<ApiConnectOptions>,
    ) -> Result<Self, AgentClientError>;
    #[host(await)]
    async fn connect(&self) -> Result<AgentCard, AgentClientError>;
    #[host(getter)]
    fn offers_extension(&self) -> bool;
    #[host(getter)]
    fn extensions(&self) -> Vec<String>;
    #[host(await)]
    async fn send(
        &self,
        message: MessageInput,
        #[host(kw, default = None)] thread_id: Option<String>,
        #[host(default = None)] task_id: Option<String>,
        #[host(default = None)] metadata: Option<MessageMetadata>,
        #[host(default = None)] chat_ctx: Option<Vec<ChatItem>>,
    ) -> Result<Self::Reply, AgentClientError>;
    #[host(await)]
    async fn _core_subscribe(
        &self,
        task_id: String,
        thread_id: String,
    ) -> Result<Self::Reply, AgentClientError>;
    #[host(await)]
    async fn cancel(
        &self,
        task_id: String,
        #[host(kw, default = None)] thread_id: Option<String>,
        #[host(default = None)] reason: Option<String>,
    ) -> Result<TaskUpdate, AgentClientError>;
    #[host(await)]
    async fn aclose(&self);
}
