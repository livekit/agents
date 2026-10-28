//! Stored tasks projected into the upstream A2A records, a response stream's resume cursor, and
//! the JSON-RPC envelope encoded once for HTTP and server-sent events.

use std::time::{Duration, UNIX_EPOCH};

use a2a::{
    Artifact, JsonRpcId, JsonRpcResponse, Part, StreamResponse, Task, TaskState, TaskStatus,
};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use lk_agents_abi::a2a::door::DoorFrame;
use lk_agents_abi::a2a_protocol as a2a;
use lk_agents_abi::storage::StorageError;
use lk_agents_abi::storage::TaskState as StoredState;
use lk_agents_abi::storage::thread::TaskRow;
use serde::{Deserialize, Serialize};

/// Where a resubscribe resumes: after `offset` in one of the task's response streams. Opaque to
/// clients.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cursor {
    pub task_id: String,
    pub stream_id: String,
    pub offset: String,
}

impl Cursor {
    pub fn encode(&self) -> String {
        URL_SAFE_NO_PAD.encode(serde_json::to_vec(self).expect("stream cursor"))
    }

    pub fn decode(token: &str) -> Result<Self, StorageError> {
        if token.len() > 4096 {
            return Err(StorageError::Invalid("stream cursor exceeds its budget".into()));
        }
        URL_SAFE_NO_PAD
            .decode(token)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .ok_or_else(|| StorageError::Invalid("invalid response stream cursor".into()))
    }
}

pub fn status(state: StoredState) -> TaskState {
    match state {
        StoredState::Submitted => TaskState::Submitted,
        StoredState::Working => TaskState::Working,
        StoredState::InputRequired => TaskState::InputRequired,
        StoredState::Completed => TaskState::Completed,
        StoredState::Failed => TaskState::Failed,
        StoredState::Canceled => TaskState::Canceled,
        StoredState::Rejected => TaskState::Rejected,
    }
}

fn timestamp<T: From<std::time::SystemTime>>(ms: i64) -> Option<T> {
    u64::try_from(ms).ok().map(|ms| (UNIX_EPOCH + Duration::from_millis(ms)).into())
}

pub fn task(row: &TaskRow, context: &str, history: Option<usize>, artifacts: bool) -> Task {
    let length = history.unwrap_or(row.data.history.len());
    let mut messages = row.data.history[row.data.history.len().saturating_sub(length)..].to_vec();
    for message in &mut messages {
        message.task_id = Some(row.task_id.clone());
        message.context_id = Some(context.to_owned());
    }
    Task {
        id: row.task_id.clone(),
        context_id: context.to_owned(),
        status: TaskStatus {
            state: status(row.status),
            message: row.data.status_message.clone(),
            timestamp: timestamp(row.updated_at),
        },
        artifacts: artifacts.then(|| row.data.artifacts.clone()).flatten(),
        history: (!messages.is_empty()).then_some(messages),
        metadata: (!row.data.metadata.is_empty())
            .then(|| row.data.metadata.clone().into_iter().collect()),
    }
}

pub fn submitted(id: &str, context: &str, now_ms: i64) -> StreamResponse {
    StreamResponse::Task(Task {
        id: id.to_owned(),
        context_id: context.into(),
        status: TaskStatus {
            state: TaskState::Submitted,
            message: None,
            timestamp: timestamp(now_ms),
        },
        history: None,
        artifacts: None,
        metadata: None,
    })
}

/// A `working` update `message_id` with the progress text, at `now_ms`.
pub fn progress(
    id: &str,
    context: &str,
    text: String,
    message_id: String,
    now_ms: i64,
) -> StreamResponse {
    let mut message = a2a::Message::new(a2a::Role::Agent, vec![Part::text(text)]);
    message.message_id = message_id;
    message.task_id = Some(id.to_owned());
    message.context_id = Some(context.to_owned());
    StreamResponse::StatusUpdate(a2a::TaskStatusUpdateEvent {
        task_id: id.to_owned(),
        context_id: context.into(),
        status: TaskStatus {
            state: TaskState::Working,
            message: Some(message),
            timestamp: timestamp(now_ms),
        },
        metadata: None,
    })
}

pub fn artifact(
    id: &str,
    context: &str,
    artifact_id: &str,
    text: Option<String>,
    append: bool,
    last: bool,
) -> StreamResponse {
    StreamResponse::ArtifactUpdate(a2a::TaskArtifactUpdateEvent {
        task_id: id.to_owned(),
        context_id: context.into(),
        artifact: Artifact {
            artifact_id: artifact_id.to_owned(),
            name: Some(lk_agents_abi::a2a::metadata::ANSWER_ARTIFACT_NAME.into()),
            description: None,
            parts: text.into_iter().map(Part::text).collect(),
            metadata: None,
            extensions: None,
        },
        append: Some(append),
        last_chunk: Some(last),
        metadata: None,
    })
}

pub fn final_status(row: &TaskRow, context: &str) -> StreamResponse {
    let task = task(row, context, Some(0), false);
    StreamResponse::StatusUpdate(a2a::TaskStatusUpdateEvent {
        task_id: task.id,
        context_id: task.context_id,
        status: task.status,
        metadata: task.metadata,
    })
}

pub fn success<T: Serialize>(id: JsonRpcId, value: T, cursor: Option<&str>) -> DoorFrame {
    let result = serde_json::to_value(value).expect("A2A wire record");
    frame(crate::a2a::rpc::success(id, result), cursor)
}

pub fn frame(response: JsonRpcResponse, cursor: Option<&str>) -> DoorFrame {
    let is_error = response.error.is_some();
    let json = serde_json::to_string(&response).expect("A2A JSON-RPC envelope");
    let mut sse = String::new();
    if let Some(cursor) = cursor {
        sse.push_str("id: ");
        sse.push_str(cursor);
        sse.push('\n');
    }
    if is_error {
        sse.push_str("event: error\n");
    }
    sse.push_str("data: ");
    sse.push_str(&json);
    sse.push_str("\n\n");
    DoorFrame { json, sse, is_error }
}
