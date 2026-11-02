//! Stored tasks projected into the upstream A2A records, and
//! the JSON-RPC envelope encoded once for HTTP and server-sent events.

use std::time::{Duration, UNIX_EPOCH};

use a2a::{
    Artifact, JsonRpcId, JsonRpcResponse, Part, StreamResponse, Task, TaskState, TaskStatus,
};
use lk_agents_abi::a2a::door::DoorFrame;
use lk_agents_abi::a2a::turn::TurnEvent;
use lk_agents_abi::a2a_protocol as a2a;
use lk_agents_abi::storage::TaskState as StoredState;
use lk_agents_abi::storage::thread::TaskRow;
use serde::Serialize;

use crate::a2a::TurnResult;

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

/// What a live observer is sent of one stream event.
#[derive(Debug, Clone)]
pub enum Delivery {
    Send(DoorFrame),
    /// It fell behind: an error frame, and it is detached.
    Detach(DoorFrame),
}

/// The delivery of `event` to an observer whose queue has `room` free slots; the last slot is
/// kept for the detaching error.
pub fn deliver(request_id: JsonRpcId, event: DoorFrame, room: usize) -> Delivery {
    if room > 1 {
        return Delivery::Send(event);
    }
    let error = crate::a2a::rpc::rpc_error(
        a2a::error_code::INTERNAL_ERROR,
        "slow subscriber detached; retrieve the task outcome with GetTask",
    );
    Delivery::Detach(frame(crate::a2a::rpc::failure(request_id, error), None))
}

/// What one request's response stream says, decided from the turn: the task announced once, its
/// progress and answer tokens while they belong to that task, then the answer artifact (closing
/// the streamed one when the tokens already were the answer) and the final status.
#[derive(Debug)]
pub struct Publication {
    /// The announced task; `None` until announced.
    task_id: Option<String>,
    artifact_id: String,
    /// The answer tokens streamed so far.
    text: String,
}

impl Publication {
    pub fn new(artifact_id: String) -> Self {
        Self { task_id: None, artifact_id, text: String::new() }
    }

    /// The `submitted` announcement of `task_id`, once.
    pub fn announce(
        &mut self,
        task_id: &str,
        context: &str,
        now_ms: i64,
    ) -> Option<StreamResponse> {
        if self.task_id.is_some() {
            return None;
        }
        self.task_id = Some(task_id.to_owned());
        Some(submitted(task_id, context, now_ms))
    }

    /// A handler event of the announced task as a stream event; `message_id` names a progress
    /// message.
    pub fn event(
        &mut self,
        event: &TurnEvent,
        context: &str,
        message_id: String,
        now_ms: i64,
    ) -> Option<StreamResponse> {
        match event {
            TurnEvent::Progress { task_id, text } if self.announced(task_id) => {
                Some(progress(task_id, context, text.clone(), message_id, now_ms))
            }
            TurnEvent::Token { task_id, delta } if self.announced(task_id) && !delta.is_empty() => {
                let append = !self.text.is_empty();
                self.text.push_str(delta);
                Some(artifact(
                    task_id,
                    context,
                    &self.artifact_id,
                    Some(delta.clone()),
                    append,
                    false,
                ))
            }
            _ => None,
        }
    }

    fn announced(&self, task_id: &str) -> bool {
        self.task_id.as_deref() == Some(task_id)
    }

    /// The turn's end: its announcement if none was made, the answer artifact, the final status.
    pub fn finish(
        &mut self,
        result: &TurnResult,
        context: &str,
        now_ms: i64,
    ) -> Vec<StreamResponse> {
        let row = &result.task;
        let mut events: Vec<StreamResponse> =
            self.announce(&row.task_id, context, now_ms).into_iter().collect();
        if let Some(answer) = row.data.artifacts.as_ref().and_then(|a| a.first()) {
            let streamed = !self.text.is_empty()
                && result.output.as_ref().and_then(serde_json::Value::as_str)
                    == Some(self.text.as_str());
            events.push(if streamed {
                artifact(&row.task_id, context, &self.artifact_id, None, true, true)
            } else {
                StreamResponse::ArtifactUpdate(a2a::TaskArtifactUpdateEvent {
                    task_id: row.task_id.clone(),
                    context_id: context.into(),
                    artifact: answer.clone(),
                    append: Some(false),
                    last_chunk: Some(true),
                    metadata: None,
                })
            });
        }
        events.push(final_status(row, context));
        events
    }
}
