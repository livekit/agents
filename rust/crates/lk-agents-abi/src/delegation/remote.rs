//! Durable observations and request intent. No connection, credential or running future is saved.

use std::collections::BTreeMap;

use crate::a2a_protocol as a2a;
pub use crate::session::InputRequestRef;
use crate::session::{InputRequestState, ToolCallRef};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "operation", content = "request", rename_all = "snake_case")]
#[crate::host(record(module = "delegation_remote", transparent, emit, extract))]
pub enum A2ARequest {
    Send(Box<crate::a2a::protocol::SendMessageRequest>),
    Get(crate::a2a::protocol::GetTaskRequest),
    List(crate::a2a::protocol::ListTasksRequest),
    Cancel(crate::a2a::protocol::CancelTaskRequest),
    Subscribe(crate::a2a::protocol::SubscribeToTaskRequest),
}

impl A2ARequest {
    /// The SDK owns method spellings. Streaming send uses the same SDK request body.
    pub fn method(&self, streaming: bool) -> &'static str {
        use a2a::jsonrpc::methods;
        match self {
            Self::Send(_) if streaming => methods::SEND_STREAMING_MESSAGE,
            Self::Send(_) => methods::SEND_MESSAGE,
            Self::Get(_) => methods::GET_TASK,
            Self::List(_) => methods::LIST_TASKS,
            Self::Cancel(_) => methods::CANCEL_TASK,
            Self::Subscribe(_) => methods::SUBSCRIBE_TO_TASK,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "response_type", content = "response", rename_all = "snake_case")]
#[crate::host(record(module = "delegation_remote", transparent, emit, extract))]
pub enum A2AResponse {
    Send(crate::a2a::protocol::SendMessageResponse),
    Stream(crate::a2a::protocol::StreamResponse),
    Task(crate::a2a::protocol::Task),
    List(crate::a2a::protocol::ListTasksResponse),
}

/// Persisted failure evidence, without credentials, Python exceptions or retry decisions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "failure", rename_all = "snake_case")]
#[crate::host(record(module = "delegation_remote", emit, extract))]
pub enum RemoteFailure {
    Rpc { error: crate::a2a::protocol::JsonRpcError },
    Transport { kind: crate::errors::ErrorKind, message: String, status_code: Option<i64> },
    Protocol { message: String },
}

#[crate::host(record(module = "delegation_remote", emit, extract))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum A2AJournalKind {
    /// Outgoing SDK request content; dispatch progress is stored on its request state.
    RequestRecorded,
    /// Typed peer content was received. Protocol validation may still refuse its projection.
    ResponseReceived,
    /// Communication failed; remote business work may nevertheless have happened.
    RequestError,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "payload_type", content = "payload", rename_all = "snake_case")]
#[crate::host(record(module = "delegation_remote", transparent, emit, extract))]
pub enum A2AJournalPayload {
    Request(A2ARequest),
    Response(Box<A2AResponse>),
    Error(RemoteFailure),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[crate::host(record(module = "delegation_remote", emit, extract))]
pub struct A2AJournalEntry {
    /// Local observation order; neither remote causal order nor a transport RPC ID.
    pub sequence: u64,
    /// Stable logical operation identity shared by its request, responses and errors.
    pub request_id: String,
    pub kind: A2AJournalKind,
    /// Original typed SDK content. Dispatch markers never append duplicate payloads.
    pub payload: A2AJournalPayload,
}

#[crate::host(record(
    module = "delegation_remote",
    path = "livekit.agents.storage.session_state",
    emit,
    extract,
    dataclass,
    frozen,
    kw_only,
    slots
))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputRequest {
    /// Response handling for this occurrence, separate from the remote task status.
    pub state: InputRequestState,
    /// Optional question in the canonical observed-message index.
    pub source_message_id: Option<String>,
    /// Reserved outgoing A2A message, retained unchanged while its delivery is uncertain.
    pub response_message_id: Option<String>,
    /// Canonical human-facing delivery; received remote content is not presentation evidence.
    #[host(skip)]
    pub presented_message_id: Option<String>,
    /// Canonical human response retained while the current agent cannot dispatch to this task.
    #[host(skip)]
    pub human_response_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[crate::host(record(module = "delegation_remote", emit, extract))]
pub struct RemoteTaskRef {
    pub context_id: String,
    pub task_id: String,
}

/// A caller's immutable observation before waiting for an owner or fence. This is raw
/// journal evidence, not permission to retry or a second cancellation state machine.
#[derive(Debug, Clone, PartialEq)]
pub struct RemoteCancellationEvidence {
    pub request_id: String,
    pub dispatch_started: bool,
    pub ended: bool,
    pub failure: Option<RemoteFailure>,
}

/// Credential-free evidence that the original send used a verified project delegation grant.
/// It cannot mint authority by itself: control rechecks the current bound thread and project.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteControlScope {
    pub project_origin: String,
    /// Public JWT issuer identifier; never a bearer token or signing secret.
    pub project_issuer: String,
    pub caller_identity: String,
    pub thread_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[crate::host(record(module = "delegation_remote", emit, extract))]
pub struct RemoteRequestState {
    /// SDK method constant, fixed when the request is prepared.
    pub method: String,
    /// Journal sequence of the immutable SDK request; not a second payload copy.
    pub request_event: u64,
    /// Persisted before I/O: an attempt may have happened, not proof of admission.
    pub dispatch_started: bool,
    /// The local observation stream settled; this does not complete remote business work.
    /// Preparing the unchanged request may reopen it only before dispatch started.
    pub ended: bool,
    /// A known remote task for reconciliation after a lost connection.
    pub task: Option<RemoteTaskRef>,
    /// Local tool call that initiated this operation, not relationship-wide ownership.
    pub origin: Option<ToolCallRef>,
    /// A selected input occurrence, if this operation sends its response.
    pub input_request: Option<InputRequestRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[host(skip)]
    pub control_scope: Option<RemoteControlScope>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[crate::host(record(module = "delegation_remote", emit, extract))]
pub struct RemoteTaskState {
    /// Latest protocol observation, including protocol states not understood by presentation.
    pub task: crate::a2a::protocol::Task,
    pub input_requests: BTreeMap<String, InputRequest>,
    /// Explicit intent for this task only; closing a frontend never sets it.
    pub cancel_requested: bool,
    /// Monotonic local occurrence identity, independent of question wording or stream order.
    pub next_input: u64,
    /// First successfully projected terminal status; raw stale/rejected responses are not
    /// outcome authority. The journal supplies its exact status metadata and stream prefix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[host(skip)]
    pub terminal_event: Option<u64>,
    /// Artifact ID -> request whose observation last set its projected content. A chunk from
    /// another request cannot know whether that snapshot already contains it, so it is not
    /// appended a second time.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    #[host(skip)]
    pub artifact_sources: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[crate::host(record(module = "delegation_remote", emit, extract))]
pub struct RemoteAgentState {
    /// Pinned destination. Restoration cannot silently redirect existing work.
    pub endpoint: String,
    pub events: Vec<A2AJournalEntry>,
    /// Mutable operation facts; the journal contains communication only.
    pub requests: BTreeMap<String, RemoteRequestState>,
    pub tasks: BTreeMap<String, RemoteTaskState>,
    /// Canonical observed messages in either direction, indexed by messageId.
    pub messages: BTreeMap<String, crate::a2a::protocol::Message>,
    /// Reused for new work; observing a different task context does not switch it.
    pub default_context_id: Option<String>,
}

impl RemoteAgentState {
    /// The journal entry at a local sequence. Compaction removes settled entries, so a
    /// sequence is an ordered key rather than a position.
    pub fn event(&self, sequence: u64) -> Option<&A2AJournalEntry> {
        let position = self.events.binary_search_by_key(&sequence, |event| event.sequence).ok()?;
        self.events.get(position)
    }

    /// Read only the latest cancellation's recorded facts. The core decides whether its
    /// typed failure proves refusal; callers never infer that from error wording.
    pub fn cancellation_evidence(
        &self,
        target: &RemoteTaskRef,
    ) -> Option<RemoteCancellationEvidence> {
        let (id, request) = self
            .requests
            .iter()
            .filter(|(_, request)| {
                request.task.as_ref() == Some(target)
                && self.event(request.request_event).is_some_and(|event| {
                    matches!(&event.payload, A2AJournalPayload::Request(A2ARequest::Cancel(cancel))
                        if cancel.id == target.task_id)
                })
            })
            .max_by_key(|(_, request)| request.request_event)?;
        let failure =
            self.events.iter().rev().filter(|event| event.request_id == *id).find_map(|event| {
                match &event.payload {
                    A2AJournalPayload::Error(error) => Some(error.clone()),
                    _ => None,
                }
            });
        Some(RemoteCancellationEvidence {
            request_id: id.clone(),
            dispatch_started: request.dispatch_started,
            ended: request.ended,
            failure,
        })
    }

    pub fn new(endpoint: String) -> Self {
        Self {
            endpoint,
            events: Vec::new(),
            requests: BTreeMap::new(),
            tasks: BTreeMap::new(),
            messages: BTreeMap::new(),
            default_context_id: None,
        }
    }
}

/// Inputs to the native conversation owner. Runtime waits for its checkpoint before sending.
#[derive(Debug, Clone)]
pub enum RemoteInput {
    Prepare {
        request_id: String,
        request: A2ARequest,
        streaming: bool,
        origin: Option<ToolCallRef>,
        input_request: Option<InputRequestRef>,
        control_scope: Option<Box<RemoteControlScope>>,
    },
    DispatchStarted {
        request_id: String,
    },
    Response {
        request_id: String,
        response: Box<A2AResponse>,
    },
    Error {
        request_id: String,
        error: RemoteFailure,
    },
    Ended {
        request_id: String,
    },
}

#[derive(Debug, Clone)]
pub struct RemoteCommand {
    pub remote_id: String,
    pub endpoint: String,
    pub input: RemoteInput,
}

/// A native recovery decision, read from the saved request rather than inferred by I/O.
#[derive(Debug, Clone, PartialEq)]
pub enum RemoteRecovery {
    /// No dispatch marker was committed. Only this original operation may be sent.
    Dispatch {
        request_id: String,
        request: Box<A2ARequest>,
        streaming: bool,
        input_request: Option<InputRequestRef>,
    },
    /// A send may have reached the peer. Read this task; never repeat its business message.
    GetTask(RemoteTaskRef),
    /// An accepted final task observation or direct message; never a cached nonterminal poll.
    Observed(Box<crate::a2a::protocol::SendMessageResponse>),
    /// Dispatch may have happened, but no accepted task or reply identifies its outcome.
    Uncertain,
}
