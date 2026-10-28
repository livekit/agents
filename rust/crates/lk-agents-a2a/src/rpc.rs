//! An endpoint's JSON-RPC 2.0 requests (docs/protocol/a2a.md), decoded once into the upstream A2A
//! records plus LiveKit's additions (`contextId` on the task methods, `offset` on a subscription),
//! and the envelopes the answers go out in. A malformed request is a JSON-RPC error here; what a
//! well-formed request meets later is its method's outcome.

use a2a::{
    CancelTaskRequest, GetTaskRequest, JsonRpcError, JsonRpcId, JsonRpcResponse, ListTasksRequest,
    SendMessageRequest, SubscribeToTaskRequest, error_code,
};
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use lk_agents_abi::a2a::metadata::REASON_KEY;
use lk_agents_abi::storage::TaskState;

/// Wire versions are separate from the accepted historical method aliases.
const VERSIONS: [&str; 1] = ["1.0"];

/// The largest `ListTasks` page A2A allows.
pub const MAX_PAGE_SIZE: i64 = 100;

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
    pub page_size: Option<i64>,
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
    crate::attachments::validate_wire_parts(&params["message"]).map_err(invalid_params)?;
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
        None => None,
        Some(n) if (1..=MAX_PAGE_SIZE).contains(&i64::from(n)) => Some(i64::from(n)),
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
                page_size: Some(10),
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
