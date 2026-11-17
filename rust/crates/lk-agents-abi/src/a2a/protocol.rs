//! SDK values cross as SDK protobuf messages. No second Python wire model is declared here.

#[crate::host(codec(
    extract,
    emit,
    host = "a2a.types.a2a_pb2.Message",
    py = "py_protocol::a2a_message"
))]
pub type Message = crate::a2a_protocol::Message;

#[crate::host(codec(extract, emit, host = "a2a.types.a2a_pb2.Task", py = "py_protocol::a2a_task"))]
pub type Task = crate::a2a_protocol::Task;

#[crate::host(codec(
    extract,
    emit,
    host = "a2a.types.a2a_pb2.StreamResponse",
    py = "py_protocol::a2a_stream_response"
))]
pub type StreamResponse = crate::a2a_protocol::StreamResponse;

#[crate::host(codec(
    extract,
    emit,
    host = "a2a.types.a2a_pb2.SendMessageRequest",
    py = "py_protocol::a2a_send_request"
))]
pub type SendMessageRequest = crate::a2a_protocol::SendMessageRequest;

#[crate::host(codec(
    extract,
    emit,
    host = "a2a.types.a2a_pb2.SendMessageResponse",
    py = "py_protocol::a2a_send_response"
))]
pub type SendMessageResponse = crate::a2a_protocol::SendMessageResponse;

#[crate::host(codec(
    extract,
    emit,
    host = "a2a.types.a2a_pb2.GetTaskRequest",
    py = "py_protocol::a2a_get_request"
))]
pub type GetTaskRequest = crate::a2a_protocol::GetTaskRequest;

#[crate::host(codec(
    extract,
    emit,
    host = "a2a.types.a2a_pb2.CancelTaskRequest",
    py = "py_protocol::a2a_cancel_request"
))]
pub type CancelTaskRequest = crate::a2a_protocol::CancelTaskRequest;

#[crate::host(codec(
    extract,
    emit,
    host = "a2a.types.a2a_pb2.SubscribeToTaskRequest",
    py = "py_protocol::a2a_subscribe_request"
))]
pub type SubscribeToTaskRequest = crate::a2a_protocol::SubscribeToTaskRequest;

#[crate::host(codec(
    extract,
    emit,
    host = "a2a.types.a2a_pb2.ListTasksRequest",
    py = "py_protocol::a2a_list_request"
))]
pub type ListTasksRequest = crate::a2a_protocol::ListTasksRequest;

#[crate::host(codec(
    extract,
    emit,
    host = "a2a.types.a2a_pb2.ListTasksResponse",
    py = "py_protocol::a2a_list_response"
))]
pub type ListTasksResponse = crate::a2a_protocol::ListTasksResponse;

// The pinned Python SDK ships protobuf operation models but no JSON-RPC error class.
// Preserve its native code/message/data as JSON rather than declaring a second wire model.
#[crate::host(codec(
    extract,
    emit,
    host = "dict[str, pydantic.JsonValue]",
    py = "py_protocol::a2a_rpc_error"
))]
pub type JsonRpcError = crate::a2a_protocol::JsonRpcError;
