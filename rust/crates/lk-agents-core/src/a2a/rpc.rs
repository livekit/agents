//! An endpoint's JSON-RPC 2.0 requests (docs/protocol/a2a.md), decoded once into the upstream A2A
//! records plus LiveKit's additions (`contextId` on the task methods, `offset` on a subscription),
//! and the envelopes the answers go out in. A malformed request is a JSON-RPC error here; what a
//! well-formed request meets later is its method's outcome.

use a2a::{
    CancelTaskRequest, GetTaskRequest, JsonRpcError, JsonRpcId, JsonRpcResponse, ListTasksRequest,
    SendMessageRequest, SubscribeToTaskRequest, error_code,
};
use lk_agents_abi::a2a_protocol as a2a;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use lk_agents_abi::a2a::door::DoorFrame;
use lk_agents_abi::a2a::metadata::REASON_KEY;
use lk_agents_abi::storage::thread::{REASON_THREAD_NOT_FOUND, TaskRow};
use lk_agents_abi::storage::{StorageError, StreamInfo, TaskState};

use crate::a2a::wire;

/// Wire versions are separate from the accepted historical method aliases.
const VERSIONS: [&str; 1] = ["1.0"];

/// The largest `ListTasks` page A2A allows.
pub const MAX_PAGE_SIZE: i64 = 100;
/// A `ListTasks` page without `pageSize`.
pub const DEFAULT_PAGE_SIZE: i64 = 50;

/// One well-formed request.
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub id: JsonRpcId,
    pub call: Call,
}

impl Request {
    /// Whether the answer is a stream of server-sent events.
    pub fn is_streaming(&self) -> bool {
        matches!(self.call, Call::Send { streaming: true, .. } | Call::Subscribe { .. })
    }
}

/// A task named by its id and the thread it is in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskRef {
    pub id: String,
    pub context_id: String,
}

/// What a request asks for.
#[derive(Debug, Clone, PartialEq)]
pub enum Call {
    /// `SendMessage`, or `SendStreamingMessage` when `streaming`.
    Send { message: a2a::Message, history_length: Option<usize>, streaming: bool },
    /// `GetTask`.
    Get { task: TaskRef, history_length: Option<usize> },
    /// `ListTasks`.
    List(TaskList),
    /// `CancelTask`, with the caller's `livekit/reason`.
    Cancel { task: TaskRef, reason: Option<String> },
    /// `SubscribeToTask`, resuming after `offset` when given.
    Subscribe { task: TaskRef, offset: Option<String> },
}

/// `ListTasks`: one thread's tasks by last update.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskList {
    pub context_id: String,
    pub status: Option<TaskState>,
    pub page_size: i64,
    pub page_token: Option<String>,
    pub history_length: Option<usize>,
    /// Unix milliseconds: only tasks updated after it; zero for every task.
    pub updated_after: i64,
    pub include_artifacts: bool,
}

/// A request answered with a JSON-RPC error, under the request's id when it had a valid one.
#[derive(Debug, Clone, PartialEq)]
pub struct Refusal {
    pub id: JsonRpcId,
    pub error: JsonRpcError,
}

/// A JSON-RPC error object.
pub fn rpc_error(code: i32, message: impl Into<String>) -> JsonRpcError {
    JsonRpcError { code, message: message.into(), data: None }
}

fn invalid_params(message: impl Into<String>) -> JsonRpcError {
    rpc_error(error_code::INVALID_PARAMS, message)
}

/// A method's success envelope.
pub fn success(id: JsonRpcId, result: Value) -> JsonRpcResponse {
    JsonRpcResponse::success(id, result)
}

/// A method's error envelope.
pub fn failure(id: JsonRpcId, error: JsonRpcError) -> JsonRpcResponse {
    JsonRpcResponse::error(id, error)
}

/// The `A2A-Version` header selects the served wire format.
pub fn check_version(header: Option<&str>) -> Result<(), JsonRpcError> {
    match header.map(str::trim).filter(|v| !v.is_empty()) {
        None => Ok(()),
        Some(v) if VERSIONS.contains(&v) => Ok(()),
        Some(v) => {
            Err(rpc_error(error_code::VERSION_NOT_SUPPORTED, format!("version not supported: {v}")))
        }
    }
}

/// One request body. A batch, a body that is not JSON-RPC 2.0 and an unknown method are refused
/// with the matching JSON-RPC error; so are params of the wrong shape.
pub fn decode(body: &[u8]) -> Result<Request, Box<Refusal>> {
    let refuse = |id: JsonRpcId, error| Box::new(Refusal { id, error });
    let value: Value = serde_json::from_slice(body)
        .map_err(|e| refuse(JsonRpcId::Null, rpc_error(error_code::PARSE_ERROR, e.to_string())))?;
    let envelope = match value {
        Value::Object(o) => o,
        Value::Array(_) => {
            let e = rpc_error(error_code::INVALID_REQUEST, "batch requests are not supported");
            return Err(refuse(JsonRpcId::Null, e));
        }
        _ => {
            let e = rpc_error(error_code::INVALID_REQUEST, "a request is a JSON object");
            return Err(refuse(JsonRpcId::Null, e));
        }
    };
    let id = match envelope.get("id") {
        Some(Value::String(s)) => JsonRpcId::String(s.clone()),
        Some(Value::Number(n)) if n.as_i64().is_some() => {
            JsonRpcId::Number(n.as_i64().unwrap_or(0))
        }
        _ => {
            let e = rpc_error(error_code::INVALID_REQUEST, "'id' is a string or an integer");
            return Err(refuse(JsonRpcId::Null, e));
        }
    };
    if envelope.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        let e = rpc_error(error_code::INVALID_REQUEST, "'jsonrpc' must be exactly '2.0'");
        return Err(refuse(id, e));
    }
    let Some(method) = envelope.get("method").and_then(Value::as_str) else {
        return Err(refuse(id, rpc_error(error_code::INVALID_REQUEST, "'method' is a string")));
    };
    let params = match envelope.get("params") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(p)) => p.clone(),
        Some(_) => return Err(refuse(id, invalid_params("params must be an object"))),
    };
    match call(method, params) {
        Ok(call) => Ok(Request { id, call }),
        Err(error) => Err(refuse(id, error)),
    }
}

fn call(method: &str, params: Map<String, Value>) -> Result<Call, JsonRpcError> {
    match method {
        "SendMessage" | "message/send" => send(params, false),
        "SendStreamingMessage" | "message/stream" => send(params, true),
        "GetTask" | "tasks/get" => {
            let task = task_ref(&params)?;
            let r: GetTaskRequest = typed(params)?;
            Ok(Call::Get { task, history_length: history_length(r.history_length)? })
        }
        "ListTasks" | "tasks/list" => list(params),
        "CancelTask" | "tasks/cancel" => {
            let task = task_ref(&params)?;
            let r: CancelTaskRequest = typed(params)?;
            let reason = match r.metadata.as_ref().and_then(|m| m.get(REASON_KEY)) {
                None | Some(Value::Null) => None,
                Some(Value::String(s)) => Some(s.clone()),
                Some(_) => return Err(invalid_params("the cancellation reason is a string")),
            };
            Ok(Call::Cancel { task, reason })
        }
        "SubscribeToTask" | "tasks/resubscribe" => {
            let task = task_ref(&params)?;
            let offset = match params.get("offset") {
                None | Some(Value::Null) => None,
                Some(Value::String(s)) => Some(s.clone()),
                Some(_) => return Err(invalid_params("offset is an opaque string cursor")),
            };
            let _: SubscribeToTaskRequest = typed(params)?;
            Ok(Call::Subscribe { task, offset })
        }
        a2a::methods::CREATE_PUSH_CONFIG
        | a2a::methods::GET_PUSH_CONFIG
        | a2a::methods::LIST_PUSH_CONFIGS
        | a2a::methods::DELETE_PUSH_CONFIG => Err(rpc_error(
            error_code::PUSH_NOTIFICATION_NOT_SUPPORTED,
            "push notifications are not supported",
        )),
        a2a::methods::GET_EXTENDED_AGENT_CARD => {
            Err(rpc_error(error_code::UNSUPPORTED_OPERATION, "there is no extended agent card"))
        }
        other => Err(rpc_error(error_code::METHOD_NOT_FOUND, format!("method not found: {other}"))),
    }
}

fn typed<T: DeserializeOwned>(params: Map<String, Value>) -> Result<T, JsonRpcError> {
    serde_json::from_value(Value::Object(params)).map_err(|e| invalid_params(e.to_string()))
}

fn send(params: Map<String, Value>, streaming: bool) -> Result<Call, JsonRpcError> {
    if !params.get("message").is_some_and(Value::is_object) {
        return Err(invalid_params("params.message is required"));
    }
    if !params["message"]
        .get("messageId")
        .is_some_and(|v| v.as_str().is_some_and(|s| !s.is_empty()))
    {
        return Err(invalid_params("a message needs a messageId"));
    }
    crate::a2a::attachments::validate_wire_parts(&params["message"]).map_err(invalid_params)?;
    let r: SendMessageRequest = typed(params)?;
    let mut message = r.message;
    message.task_id = message.task_id.filter(|s| !s.is_empty());
    message.context_id = message.context_id.filter(|s| !s.is_empty());
    let length = r.configuration.and_then(|c| c.history_length);
    Ok(Call::Send { message, history_length: history_length(length)?, streaming })
}

fn list(params: Map<String, Value>) -> Result<Call, JsonRpcError> {
    let context_id = context_id(&params)?;
    let r: ListTasksRequest = typed(params)?;
    let status = match r.status {
        None | Some(a2a::TaskState::Unspecified) => None,
        Some(s) => Some(task_state(&s).ok_or_else(|| invalid_params("no task is in that state"))?),
    };
    let page_size = match r.page_size {
        None => DEFAULT_PAGE_SIZE,
        Some(n) if (1..=MAX_PAGE_SIZE).contains(&i64::from(n)) => i64::from(n),
        Some(_) => return Err(invalid_params(format!("pageSize is 1 to {MAX_PAGE_SIZE}"))),
    };
    Ok(Call::List(TaskList {
        context_id,
        status,
        page_size,
        page_token: r.page_token.filter(|t| !t.is_empty()),
        history_length: history_length(r.history_length)?,
        updated_after: r.status_timestamp_after.map_or(0, |t| t.timestamp_millis()),
        include_artifacts: r.include_artifacts.unwrap_or(false),
    }))
}

/// The framework's state for an A2A one; `None` for one no task is in.
pub fn task_state(state: &a2a::TaskState) -> Option<TaskState> {
    Some(match state {
        a2a::TaskState::Submitted => TaskState::Submitted,
        a2a::TaskState::Working => TaskState::Working,
        a2a::TaskState::InputRequired => TaskState::InputRequired,
        a2a::TaskState::Completed => TaskState::Completed,
        a2a::TaskState::Failed => TaskState::Failed,
        a2a::TaskState::Canceled => TaskState::Canceled,
        a2a::TaskState::Rejected => TaskState::Rejected,
        a2a::TaskState::Unspecified | a2a::TaskState::AuthRequired => return None,
    })
}

/// The task methods name the thread with `contextId`: missing, the task is not found
/// (a2a.md#tasks); not a string, the params are invalid.
fn context_id(params: &Map<String, Value>) -> Result<String, JsonRpcError> {
    match params.get("contextId") {
        None | Some(Value::Null) => Err(rpc_error(
            error_code::TASK_NOT_FOUND,
            "task methods need the thread's contextId beside the task id",
        )),
        Some(Value::String(s)) if s.is_empty() => Err(rpc_error(
            error_code::TASK_NOT_FOUND,
            "task methods need the thread's contextId beside the task id",
        )),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(_) => Err(invalid_params("contextId is a string")),
    }
}

fn task_ref(params: &Map<String, Value>) -> Result<TaskRef, JsonRpcError> {
    let id = match params.get("id") {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        _ => return Err(invalid_params("id is the task's id")),
    };
    Ok(TaskRef { id, context_id: context_id(params)? })
}

fn history_length(n: Option<i32>) -> Result<Option<usize>, JsonRpcError> {
    n.map(|n| usize::try_from(n).map_err(|_| invalid_params("historyLength is not negative")))
        .transpose()
}

/// Optional extension negotiation does not confer thread authority.
pub fn extensions_header(requested: Option<&str>) -> Option<String> {
    let supported = lk_agents_abi::a2a::metadata::EXTENSION_URI;
    requested
        .filter(|value| value.split(',').any(|part| part.trim() == supported))
        .map(|_| supported.to_owned())
}

/// A request body admitted by the door: refused when over the transport's payload budget, when
/// the endpoint's request budget is spent, when malformed, or when its `A2A-Version` is not
/// served.
pub fn admit(
    body: &[u8],
    size: usize,
    max: usize,
    admitted: bool,
    version: Option<&str>,
) -> Result<Request, Box<Refusal>> {
    let refuse = |code, message: &str| {
        Box::new(Refusal { id: JsonRpcId::Null, error: rpc_error(code, message) })
    };
    if size > max {
        return Err(refuse(error_code::INVALID_PARAMS, "A2A payload exceeds the transport budget"));
    }
    if !admitted {
        return Err(refuse(error_code::INTERNAL_ERROR, "endpoint request budget exhausted"));
    }
    let request = decode(body)?;
    match check_version(version) {
        Ok(()) => Ok(request),
        Err(error) => Err(Box::new(Refusal { id: request.id, error })),
    }
}

/// A method's failure on a storage error: a `reject` lane's refusal names the running task,
/// a caller error is invalid params, anything else internal.
pub fn storage_failure(id: JsonRpcId, error: &StorageError) -> DoorFrame {
    let mut rpc = rpc_error(error_code::INTERNAL_ERROR, error.to_string());
    match error {
        StorageError::Rejected { running_task_id } => {
            rpc.code = error_code::UNSUPPORTED_OPERATION;
            rpc.data =
                running_task_id.as_ref().map(|id| serde_json::json!({ "runningTaskId": id }));
        }
        StorageError::Invalid(_) => rpc.code = error_code::INVALID_PARAMS,
        _ => {}
    }
    wire::frame(failure(id, rpc), None)
}

/// `TaskNotFoundError` for a task not in the named thread.
pub fn task_not_found(id: JsonRpcId, task: &TaskRef) -> DoorFrame {
    let message = format!("task {} is not in thread {}", task.id, task.context_id);
    wire::frame(failure(id, rpc_error(error_code::TASK_NOT_FOUND, message)), None)
}

/// `GetTask`'s answer.
pub fn got(
    id: JsonRpcId,
    task: &TaskRef,
    row: Option<&TaskRow>,
    history: Option<usize>,
) -> DoorFrame {
    match row {
        Some(row) => wire::success(id, wire::task(row, &task.context_id, history, true), None),
        None => task_not_found(id, task),
    }
}

/// What a client's cancel writes on `row`: its metadata with the caller's `reason` merged in;
/// `None` for a terminal row, which stays as it is.
pub fn cancellation(row: &TaskRow, reason: Option<String>) -> Option<Map<String, Value>> {
    if row.status.is_terminal() {
        return None;
    }
    let mut metadata = row.data.metadata.clone();
    if let Some(reason) = reason {
        metadata.insert(REASON_KEY.into(), Value::String(reason));
    }
    Some(metadata)
}

/// `CancelTask`'s answer from the row after the cancel: canceled, else not cancelable.
pub fn canceled(id: JsonRpcId, task: &TaskRef, row: Option<&TaskRow>) -> DoorFrame {
    match row {
        None => task_not_found(id, task),
        Some(row) if row.status == TaskState::Canceled => {
            wire::success(id, wire::task(row, &task.context_id, None, true), None)
        }
        Some(_) => {
            let error = rpc_error(error_code::TASK_NOT_CANCELABLE, "the task is already terminal");
            wire::frame(failure(id, error), None)
        }
    }
}

/// `ListTasks`' answer: one page of the thread's tasks.
pub fn listed(
    id: JsonRpcId,
    query: &TaskList,
    tasks: &[TaskRow],
    next_page_token: String,
    page_size: i64,
    total_size: i64,
) -> DoorFrame {
    let history = Some(query.history_length.unwrap_or(0));
    let result = a2a::ListTasksResponse {
        tasks: tasks
            .iter()
            .map(|row| wire::task(row, &query.context_id, history, query.include_artifacts))
            .collect(),
        next_page_token,
        page_size: page_size as i32,
        total_size: i32::try_from(total_size).unwrap_or(i32::MAX),
    };
    wire::success(id, result, None)
}

/// What a message naming `taskId` meets before its turn: the task's row on the thread. A task
/// not in the thread is not found, a terminal one takes no follow-up; a question of this lane
/// waiting is the task the answer is announced under.
pub fn follow_up(
    id: JsonRpcId,
    task: &TaskRef,
    row: Option<&TaskRow>,
    lane: &str,
) -> Result<Option<String>, DoorFrame> {
    match row {
        None => Err(task_not_found(id, task)),
        Some(row) if row.terminal() => {
            let message = "a follow-up to a terminal task must start a new task";
            Err(wire::frame(
                failure(id, rpc_error(error_code::UNSUPPORTED_OPERATION, message)),
                None,
            ))
        }
        Some(row) => Ok((row.lane == lane
            && row.status == TaskState::InputRequired
            && row.internal.as_ref().is_some_and(|internal| internal.ask.is_some()))
        .then(|| task.id.clone())),
    }
}

/// The task of a message for a thread that does not exist: failed `thread-not-found`, never
/// stored.
pub fn unknown_thread_task(task_id: &str, lane: &str, now_ms: i64, message_id: String) -> TaskRow {
    crate::a2a::unpersisted_failed(task_id, lane, None, REASON_THREAD_NOT_FOUND, now_ms, message_id)
}

/// The answer to a message whose task failed before any storage held it (an unknown thread):
/// the `submitted` and final events on a stream, else the task.
pub fn unpersisted(
    id: JsonRpcId,
    row: &TaskRow,
    context: &str,
    streaming: bool,
    history: Option<usize>,
) -> Vec<DoorFrame> {
    if streaming {
        vec![
            wire::success(id.clone(), wire::submitted(&row.task_id, context, row.updated_at), None),
            wire::success(id, wire::final_status(row, context), None),
        ]
    } else {
        let task = a2a::SendMessageResponse::Task(wire::task(row, context, history, true));
        vec![wire::success(id, task, None)]
    }
}

/// `SendMessage`'s answer once the turn ended; a stream already carried it.
pub fn sent(id: JsonRpcId, row: &TaskRow, context: &str, history: Option<usize>) -> DoorFrame {
    wire::success(id, a2a::SendMessageResponse::Task(wire::task(row, context, history, true)), None)
}

/// One response stream a resubscribe reads.
#[derive(Debug, Clone, PartialEq)]
pub struct Replay<'a> {
    pub stream: &'a StreamInfo,
    /// The offset to read after; `None` from the start.
    pub after: Option<String>,
    /// Keep reading as it grows: only the last stream of a task still running.
    pub follow: bool,
}

/// What a resubscribe to `row` reads: its response streams from the cursor's, the first from
/// the cursor's offset. A cursor of another task or stream is invalid.
pub fn resume<'a>(
    task: &TaskRef,
    row: &'a TaskRow,
    offset: Option<&str>,
) -> Result<Vec<Replay<'a>>, StorageError> {
    let refs =
        row.internal.as_ref().map(|internal| internal.streams.as_slice()).unwrap_or_default();
    let cursor = offset.map(wire::Cursor::decode).transpose()?;
    let start = match &cursor {
        None => 0,
        Some(c) => refs
            .iter()
            .position(|s| s.stream_id == c.stream_id)
            .filter(|_| c.task_id == task.id)
            .ok_or_else(|| {
                StorageError::Invalid("response cursor belongs to a different task".into())
            })?,
    };
    let last = refs.len().saturating_sub(start + 1);
    Ok(refs[start..]
        .iter()
        .enumerate()
        .map(|(i, stream)| Replay {
            stream,
            after: cursor.as_ref().filter(|_| i == 0).map(|c| c.offset.clone()),
            follow: !row.status.is_terminal() && i == last,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(method: &str, params: Value) -> Vec<u8> {
        serde_json::to_vec(&json!({"jsonrpc": "2.0", "id": 7, "method": method, "params": params}))
            .unwrap()
    }

    fn refused(body: &[u8]) -> (JsonRpcId, i32) {
        let r = decode(body).unwrap_err();
        (r.id, r.error.code)
    }

    fn message(id: &str) -> Value {
        json!({"messageId": id, "role": "ROLE_USER", "parts": [{"text": "hi"}]})
    }

    // Catches: the envelope errors a2a.md lists becoming internal errors or a coerced request
    // (a batch answered, `[]` params read as `{}`, a fractional id accepted).
    #[test]
    fn malformed_envelopes_are_their_json_rpc_errors() {
        let num = JsonRpcId::Number(7);
        assert_eq!(refused(b"{not json"), (JsonRpcId::Null, error_code::PARSE_ERROR));
        assert_eq!(refused(b"[]"), (JsonRpcId::Null, error_code::INVALID_REQUEST));
        let batch = format!("[{}]", String::from_utf8(request("GetTask", json!({}))).unwrap());
        assert_eq!(refused(batch.as_bytes()), (JsonRpcId::Null, error_code::INVALID_REQUEST));
        let fractional = br#"{"jsonrpc": "2.0", "id": 1.5, "method": "GetTask"}"#;
        assert_eq!(refused(fractional), (JsonRpcId::Null, error_code::INVALID_REQUEST));
        let old = br#"{"jsonrpc": "1.0", "id": 7, "method": "GetTask"}"#;
        assert_eq!(refused(old), (num.clone(), error_code::INVALID_REQUEST));
        assert_eq!(refused(&request("tasks/subscribe", json!({}))).1, error_code::METHOD_NOT_FOUND);
        assert_eq!(
            refused(&request("Nope", json!({}))),
            (num.clone(), error_code::METHOD_NOT_FOUND)
        );
        assert_eq!(refused(&request("SendMessage", json!([]))), (num, error_code::INVALID_PARAMS));
        assert_eq!(
            refused(&request("CreateTaskPushNotificationConfig", json!({}))).1,
            error_code::PUSH_NOTIFICATION_NOT_SUPPORTED
        );
    }

    // Catches: a message without a messageId or with malformed parts reaching the turn (it used
    // to surface as an InternalError), and both method families not decoding to one call.
    #[test]
    fn messages_decode_once_under_both_method_names() {
        for (method, streaming) in [
            ("SendMessage", false),
            ("message/send", false),
            ("SendStreamingMessage", true),
            ("message/stream", true),
        ] {
            let r = decode(&request(
                method,
                json!({"message": message("m1"),
                "configuration": {"historyLength": 2}}),
            ))
            .unwrap();
            assert_eq!(r.is_streaming(), streaming);
            let Call::Send { message, history_length, .. } = r.call else { panic!() };
            assert_eq!((message.message_id.as_str(), history_length), ("m1", Some(2)));
        }
        let bad = [
            json!({}),
            json!({"message": "hi"}),
            json!({"message": {"role": "ROLE_USER", "parts": []}}),
            json!({"message": {"messageId": "", "role": "ROLE_USER", "parts": []}}),
            json!({"message": {"messageId": 3, "role": "ROLE_USER", "parts": []}}),
            json!({"message": {"messageId": "m", "role": "ROLE_USER", "parts": [{"bogus": 1}]}}),
            json!({"message": {"messageId": "m", "role": "ROLE_USER", "parts": [], "taskId": 4}}),
            json!({"message": message("m"), "configuration": {"historyLength": -1}}),
        ];
        for params in bad {
            assert_eq!(
                refused(&request("SendMessage", params.clone())).1,
                error_code::INVALID_PARAMS,
                "{params}"
            );
        }
        let r = decode(&request(
            "SendMessage",
            json!({"message": {"messageId": "m",
            "role": "ROLE_USER", "parts": [], "taskId": "", "contextId": "",
            "referenceTaskIds": ["AST_0"]}}),
        ))
        .unwrap();
        let Call::Send { message, .. } = r.call else { panic!() };
        assert_eq!((message.task_id, message.context_id), (None, None));
        assert_eq!(message.reference_task_ids, Some(vec!["AST_0".to_owned()]));
    }

    // Catches: a task method without contextId answered from some other thread, a snake_case
    // context_id accepted, or a non-string id/offset/reason coerced to text.
    #[test]
    fn task_methods_need_their_thread_and_typed_params() {
        let task = TaskRef { id: "AST_1".into(), context_id: "AT_1".into() };
        let get = decode(&request(
            "GetTask",
            json!({"id": "AST_1", "contextId": "AT_1",
            "historyLength": 0}),
        ))
        .unwrap();
        assert_eq!(get.call, Call::Get { task: task.clone(), history_length: Some(0) });
        let sub = decode(&request(
            "tasks/resubscribe",
            json!({"id": "AST_1", "contextId": "AT_1",
            "offset": "c-9"}),
        ))
        .unwrap();
        assert!(sub.is_streaming());
        assert_eq!(sub.call, Call::Subscribe { task: task.clone(), offset: Some("c-9".into()) });
        let reason = json!({REASON_KEY: "caller-canceled", "livekit/reason": 5});
        let cancel = decode(&request(
            "CancelTask",
            json!({"id": "AST_1", "contextId": "AT_1",
            "metadata": reason}),
        ))
        .unwrap();
        assert_eq!(cancel.call, Call::Cancel { task, reason: Some("caller-canceled".into()) });

        for method in ["GetTask", "CancelTask", "SubscribeToTask", "ListTasks"] {
            let missing = json!({"id": "AST_1", "context_id": "AT_1"});
            assert_eq!(
                refused(&request(method, missing)).1,
                error_code::TASK_NOT_FOUND,
                "{method}"
            );
            let empty = json!({"id": "AST_1", "contextId": ""});
            assert_eq!(refused(&request(method, empty)).1, error_code::TASK_NOT_FOUND, "{method}");
            let typed = json!({"id": "AST_1", "contextId": 1});
            assert_eq!(refused(&request(method, typed)).1, error_code::INVALID_PARAMS, "{method}");
        }
        let bad = [
            ("GetTask", json!({"id": 1, "contextId": "AT_1"})),
            ("GetTask", json!({"contextId": "AT_1"})),
            ("GetTask", json!({"id": "AST_1", "contextId": "AT_1", "historyLength": "2"})),
            ("SubscribeToTask", json!({"id": "AST_1", "contextId": "AT_1", "offset": 3})),
            (
                "CancelTask",
                json!({"id": "AST_1", "contextId": "AT_1",
                "metadata": {REASON_KEY: true}}),
            ),
        ];
        for (method, params) in bad {
            assert_eq!(
                refused(&request(method, params.clone())).1,
                error_code::INVALID_PARAMS,
                "{params}"
            );
        }
    }

    #[test]
    fn list_tasks_reads_its_filters_and_bounds_its_page() {
        let r = decode(&request(
            "ListTasks",
            json!({"contextId": "AT_1",
            "status": "TASK_STATE_WORKING", "pageSize": 10, "pageToken": "p",
            "statusTimestampAfter": "1970-01-01T00:00:01Z", "includeArtifacts": true}),
        ))
        .unwrap();
        assert_eq!(
            r.call,
            Call::List(TaskList {
                context_id: "AT_1".into(),
                status: Some(TaskState::Working),
                page_size: 10,
                page_token: Some("p".into()),
                history_length: None,
                updated_after: 1000,
                include_artifacts: true,
            })
        );
        for params in [
            json!({"contextId": "AT_1", "pageSize": 0}),
            json!({"contextId": "AT_1", "pageSize": 101}),
            json!({"contextId": "AT_1", "status": "TASK_STATE_AUTH_REQUIRED"}),
            json!({"contextId": "AT_1", "status": "done"}),
        ] {
            assert_eq!(
                refused(&request("tasks/list", params.clone())).1,
                error_code::INVALID_PARAMS,
                "{params}"
            );
        }
    }

    // Catches: the error table of a2a.md#errors drifting while the answers moved out of the door
    // (a follow-up to a terminal task, cancel of a terminal task, a cursor of another task).
    #[test]
    fn method_answers_keep_the_documented_errors() {
        let task = TaskRef { id: "AST_1".into(), context_id: "AT_1".into() };
        let row = |status| {
            lk_agents_abi::storage::thread::TaskWrite {
                task_id: "AST_1".into(),
                status,
                message_id: Some("m".into()),
                name: None,
                data: Default::default(),
                internal: None,
            }
            .row("desk")
        };
        let code =
            |f: DoorFrame| serde_json::from_str::<Value>(&f.json).unwrap()["error"]["code"].clone();
        let id = || JsonRpcId::Number(1);
        assert_eq!(
            code(follow_up(id(), &task, None, "desk").unwrap_err()),
            json!(error_code::TASK_NOT_FOUND)
        );
        let done = row(TaskState::Completed);
        let refused = follow_up(id(), &task, Some(&done), "desk").unwrap_err();
        assert_eq!(code(refused), json!(error_code::UNSUPPORTED_OPERATION));
        assert_eq!(
            code(canceled(id(), &task, Some(&done))),
            json!(error_code::TASK_NOT_CANCELABLE)
        );
        assert_eq!(code(canceled(id(), &task, Some(&row(TaskState::Canceled)))), Value::Null);
        let mut running = row(TaskState::Working);
        let stream = |id: &str| StreamInfo { stream_id: id.into(), expires_at: 0 };
        running.internal = Some(lk_agents_abi::storage::TaskInternal {
            streams: vec![stream("s1"), stream("s2")],
            ..Default::default()
        });
        let other =
            wire::Cursor { task_id: "AST_2".into(), stream_id: "s1".into(), offset: "3".into() };
        assert!(resume(&task, &running, Some(&other.encode())).is_err());
        let ours = wire::Cursor { task_id: "AST_1".into(), ..other };
        let from = resume(&task, &running, Some(&ours.encode())).unwrap();
        assert_eq!(from[0].after.as_deref(), Some("3"));
        // only the last stream of a running task is followed
        assert_eq!(from.iter().map(|r| r.follow).collect::<Vec<_>>(), [false, true]);
        let ended = TaskRow { status: TaskState::Completed, ..running.clone() };
        assert!(resume(&task, &ended, None).unwrap().iter().all(|r| !r.follow));
    }

    // Catches: a `reject` lane answering INTERNAL_ERROR, which hides the task the client must
    // cancel first (server.md, interruption policies).
    #[test]
    fn a_rejected_message_names_the_running_task() {
        let rejected = StorageError::Rejected { running_task_id: Some("AST_r".into()) };
        let frame = storage_failure(JsonRpcId::Number(1), &rejected);
        let error = &serde_json::from_str::<Value>(&frame.json).unwrap()["error"];
        assert_eq!(error["code"], json!(error_code::UNSUPPORTED_OPERATION));
        assert_eq!(error["data"], json!({"runningTaskId": "AST_r"}));
    }

    #[test]
    fn the_version_header_names_a_served_version() {
        assert!(check_version(None).is_ok());
        assert!(check_version(Some("1.0")).is_ok());
        assert_eq!(
            check_version(Some(" 0.3")).unwrap_err().code,
            error_code::VERSION_NOT_SUPPORTED
        );
        assert_eq!(check_version(Some("2.0")).unwrap_err().code, error_code::VERSION_NOT_SUPPORTED);
    }
}
