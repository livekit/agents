//! Public stream events and the host-neutral request stream owner.

use super::resources::ResourceError;
use crate::{BoxFuture, errors::ProviderError};

#[crate::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StreamRefused {
    #[error("a RunStream is entered once and cannot be reused")]
    Reentered,
    #[error("enter the RunStream before iterating it or reading its result")]
    NotEntered,
    #[error("the run stream has not finished starting")]
    NotStarted,
    #[error("the stream already has an iterator")]
    IteratorClaimed,
    #[error("result() while an iterator consumes the stream")]
    IteratorActive,
    #[error("another consumer is reading the run stream")]
    ConsumerActive,
    #[error("the stream iterator is closed")]
    CursorClosed,
    #[error("the run stream already has an update source")]
    AlreadyBound,
}

#[crate::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, thiserror::Error)]
pub enum StreamError {
    #[error(transparent)]
    #[host(delegate)]
    Refused(#[from] StreamRefused),
    #[error(transparent)]
    #[host(delegate)]
    Start(#[from] super::run::InvalidRun),
    #[error("the stream outcome has already been hydrated by its host")]
    OutcomeTaken,
    #[error("the run was cancelled")]
    #[host(kind = "asyncio.CancelledError")]
    Cancelled,
    #[error(transparent)]
    #[host(delegate)]
    Provider(#[from] ProviderError),
    #[error(transparent)]
    #[host(delegate)]
    Resources(#[from] ResourceError),
}

#[crate::host(record(
    module = "run_stream_events",
    path = "livekit.agents.voice.run_result",
    emit,
    constant(type = "text_delta")
))]
#[derive(Debug, Clone, PartialEq)]
pub struct TextDeltaEvent {
    pub sequence: u64,
    pub agent_name: String,
    pub invocation_id: String,
    #[host(default = None)]
    pub frame_id: Option<String>,
    pub message_id: String,
    pub delta: String,
}

#[crate::host(record(
    module = "run_stream_events",
    path = "livekit.agents.voice.run_result",
    emit,
    constant(type = "text_end")
))]
#[derive(Debug, Clone, PartialEq)]
pub struct TextEndEvent {
    pub sequence: u64,
    pub agent_name: String,
    pub invocation_id: String,
    #[host(default = None)]
    pub frame_id: Option<String>,
    pub message_id: String,
    pub interrupted: bool,
}

#[crate::host(record(
    module = "run_stream_events",
    path = "livekit.agents.voice.run_result",
    emit,
    constant(type = "tool_call")
))]
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallEvent {
    pub sequence: u64,
    pub agent_name: String,
    pub invocation_id: String,
    #[host(default = None)]
    pub frame_id: Option<String>,
    pub call_id: String,
    pub tool_name: String,
    #[host(annotation = "dict[str, pydantic.JsonValue]")]
    pub arguments: serde_json::Value,
}

#[crate::host(record(
    module = "run_stream_events",
    path = "livekit.agents.voice.run_result",
    emit,
    constant(type = "tool_result")
))]
#[derive(Debug, Clone, PartialEq)]
pub struct ToolResultEvent {
    pub sequence: u64,
    pub agent_name: String,
    pub invocation_id: String,
    #[host(default = None)]
    pub frame_id: Option<String>,
    pub call_id: String,
    pub tool_name: String,
    #[host(annotation = "typing.Literal['completed', 'failed', 'cancelled']")]
    pub outcome: super::run::ToolResultStatus,
    #[host(annotation = "pydantic.JsonValue")]
    pub output: serde_json::Value,
    pub error: Option<String>,
}

#[crate::host(record(
    module = "run_stream_events",
    path = "livekit.agents.voice.run_result",
    emit,
    constant(type = "progress")
))]
#[derive(Debug, Clone, PartialEq)]
pub struct ProgressEvent {
    pub sequence: u64,
    pub agent_name: String,
    pub invocation_id: String,
    #[host(default = None)]
    pub frame_id: Option<String>,
    pub message: String,
    pub call_id: Option<String>,
    pub tool_name: Option<String>,
}

#[crate::host(record(module = "run_stream_events", emit, transparent))]
#[derive(Debug, Clone, PartialEq)]
pub enum PresentedEvent {
    TextDelta(TextDeltaEvent),
    TextEnd(TextEndEvent),
    ToolCall(ToolCallEvent),
    ToolResult(ToolResultEvent),
    Progress(ProgressEvent),
}

/// Only terminal updates cross to the host's result hydration callbacks.
#[crate::host(record(module = "run_stream_events", emit, transparent))]
#[derive(Debug, Clone)]
pub enum StreamOutcome {
    Run(super::run::RunUpdate),
    Session(crate::text_session::TextSessionUpdate),
}

#[crate::host(class(name = "_core_RunStream"))]
pub trait RunStream: Send + Sync + Sized + 'static {
    #[host(env)]
    type Env;
    #[host(class(name = "_core_StreamEvents"))]
    type Events: Send + Sync + 'static;
    #[host(class(name = "_core_StreamText"))]
    type Text: Send + Sync + 'static;

    #[host(new)]
    fn new(env: &Self::Env) -> Self;
    #[host(call)]
    fn enter(&self) -> Result<(), StreamRefused>;
    #[host(call)]
    fn events(&self) -> Result<Self::Events, StreamRefused>;
    #[host(call)]
    fn text(&self, delta: bool) -> Result<Self::Text, StreamRefused>;
    #[host(call)]
    fn begin_result(&self) -> Result<u64, StreamRefused>;
    #[host(call)]
    fn end_result(&self, consumer: u64, hydrated: bool);
    #[host(await)]
    fn result(&self, consumer: u64) -> BoxFuture<'static, Result<StreamOutcome, StreamError>>;
    #[host(call)]
    fn cancel(&self);
    #[host(await)]
    fn aclose(&self) -> BoxFuture<'static, Result<(), StreamError>>;
    #[host(await)]
    fn rollback(&self) -> BoxFuture<'static, ()>;
    #[host(getter)]
    fn done(&self) -> bool;
    #[host(getter)]
    fn cancelled(&self) -> bool;
    #[host(getter)]
    fn usage(&self) -> crate::usage::Usage;
}

#[crate::host(class(name = "_core_StreamEvents"))]
pub trait StreamEvents: Send + Sync + Sized + 'static {
    #[host(call)]
    fn begin_read(&self) -> Result<bool, StreamError>;
    #[host(call)]
    fn end_read(&self, accepted: bool);
    #[host(await)]
    async fn recv(&self) -> Result<Option<PresentedEvent>, StreamError>;
    #[host(await)]
    async fn aclose(&self) -> Result<(), StreamError>;
}

#[crate::host(class(name = "_core_StreamText"))]
pub trait StreamText: Send + Sync + Sized + 'static {
    #[host(call)]
    fn begin_read(&self) -> Result<bool, StreamError>;
    #[host(call)]
    fn end_read(&self, accepted: bool);
    #[host(await)]
    async fn recv(&self) -> Result<Option<String>, StreamError>;
    #[host(await)]
    async fn aclose(&self) -> Result<(), StreamError>;
}

// Public data contracts. Python behavior remains in its adapters.

#[crate::host(record(
    module = "run_events",
    path = "livekit.agents.voice.run_result",
    dataclass,
    constant(type = "message")
))]
pub struct ChatMessageEvent {
    pub item: crate::ChatMessage,
}

#[crate::host(record(
    module = "run_events",
    path = "livekit.agents.voice.run_result",
    dataclass,
    constant(type = "function_call")
))]
pub struct FunctionCallEvent {
    pub item: crate::FunctionCall,
}

#[crate::host(record(
    module = "run_events",
    path = "livekit.agents.voice.run_result",
    dataclass,
    constant(type = "function_call_output")
))]
pub struct FunctionCallOutputEvent {
    pub item: crate::FunctionCallOutput,
}
