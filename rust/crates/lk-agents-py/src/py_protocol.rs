//! Host-specific protobuf JSON codecs for the upstream A2A SDK records.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use serde::{Serialize, de::DeserializeOwned};

use crate::generated::accessors as access;

pub mod interfaces {
    #[lk_agents_host_derive::host(interface(
        module = "google.protobuf.json_format",
        attrs(to_json(name = "MessageToJson"), parse(name = "Parse"))
    ))]
    pub struct ProtobufJson;

    #[lk_agents_host_derive::host(interface(
        module = "a2a.types.a2a_pb2",
        attrs(
            message(name = "Message"),
            task(name = "Task"),
            task_state(name = "TaskState"),
            stream_response(name = "StreamResponse"),
            send_request(name = "SendMessageRequest"),
            send_response(name = "SendMessageResponse"),
            get_request(name = "GetTaskRequest"),
            cancel_request(name = "CancelTaskRequest"),
            subscribe_request(name = "SubscribeToTaskRequest"),
            list_request(name = "ListTasksRequest"),
            list_response(name = "ListTasksResponse")
        )
    ))]
    pub struct A2AProtocol;

    #[lk_agents_host_derive::host(interface(
        host = "google.protobuf.internal.enum_type_wrapper.EnumTypeWrapper",
        attrs(name(name = "Name"), value(name = "Value"))
    ))]
    pub struct A2AStateEnum;
}

fn from_py<T: DeserializeOwned>(obj: &Bound<'_, PyAny>, class: &Bound<'_, PyAny>) -> PyResult<T> {
    if !obj.is_instance(class)? {
        return Err(PyTypeError::new_err("expected the upstream A2A SDK protobuf record"));
    }
    let module = access::ProtobufJson::import(obj.py())?;
    let json = crate::py_calls::call(&access::ProtobufJson::to_json(&module)?, (obj,), None)?;
    serde_json::from_str(&json.extract::<String>()?)
        .map_err(|error| PyValueError::new_err(format!("A2A protocol value: {error}")))
}

fn to_py<'py, T: Serialize>(
    py: Python<'py>,
    value: &T,
    class: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let json = serde_json::to_string(value)
        .map_err(|error| PyValueError::new_err(format!("A2A protocol value: {error}")))?;
    let message = crate::py_calls::call(class, (), None)?;
    let module = access::ProtobufJson::import(py)?;
    crate::py_calls::call(&access::ProtobufJson::parse(&module)?, (json, message), None)
}

macro_rules! codec {
    ($from:ident, $to:ident, $native:ident, $class:ident) => {
        pub fn $from(obj: &Bound<'_, PyAny>) -> PyResult<lk_agents_abi::a2a::protocol::$native> {
            let module = access::A2AProtocol::import(obj.py())?;
            from_py(obj, &access::A2AProtocol::$class(&module)?)
        }

        pub fn $to(
            py: Python<'_>,
            value: lk_agents_abi::a2a::protocol::$native,
        ) -> PyResult<Bound<'_, PyAny>> {
            let module = access::A2AProtocol::import(py)?;
            to_py(py, &value, &access::A2AProtocol::$class(&module)?)
        }
    };
}

codec!(a2a_message_from_py, a2a_message_to_py, Message, message);
codec!(a2a_task_from_py, a2a_task_to_py, Task, task);
codec!(a2a_stream_response_from_py, a2a_stream_response_to_py, StreamResponse, stream_response);
codec!(a2a_send_request_from_py, a2a_send_request_to_py, SendMessageRequest, send_request);
codec!(a2a_send_response_from_py, a2a_send_response_to_py, SendMessageResponse, send_response);
codec!(a2a_get_request_from_py, a2a_get_request_to_py, GetTaskRequest, get_request);
codec!(a2a_cancel_request_from_py, a2a_cancel_request_to_py, CancelTaskRequest, cancel_request);
codec!(
    a2a_subscribe_request_from_py,
    a2a_subscribe_request_to_py,
    SubscribeToTaskRequest,
    subscribe_request
);
codec!(a2a_list_request_from_py, a2a_list_request_to_py, ListTasksRequest, list_request);
codec!(a2a_list_response_from_py, a2a_list_response_to_py, ListTasksResponse, list_response);

pub fn a2a_state_from_py(
    obj: &Bound<'_, PyAny>,
) -> PyResult<lk_agents_abi::a2a::protocol::TaskState> {
    let value = crate::py_values::number_from_py::<i32>(obj, "A2A TaskState")?;
    let module = access::A2AProtocol::import(obj.py())?;
    let states = access::A2AProtocol::task_state(&module)?;
    let name = crate::py_calls::call(&access::A2AStateEnum::name(&states)?, (value,), None)?;
    serde_json::from_value(serde_json::Value::String(name.extract()?))
        .map_err(|error| PyValueError::new_err(format!("A2A TaskState: {error}")))
}

pub fn a2a_state_to_py(
    py: Python<'_>,
    state: lk_agents_abi::a2a::protocol::TaskState,
) -> PyResult<Bound<'_, PyAny>> {
    // SDK names bridge the Rust enum and Python protobuf numbers; neither mapping is copied.
    let name = serde_json::to_value(state)
        .map_err(|error| PyValueError::new_err(format!("A2A TaskState: {error}")))?;
    let serde_json::Value::String(name) = name else {
        return Err(PyValueError::new_err("the SDK TaskState is not an enum name"));
    };
    let module = access::A2AProtocol::import(py)?;
    let states = access::A2AProtocol::task_state(&module)?;
    crate::py_calls::call(&access::A2AStateEnum::value(&states)?, (name,), None)
}

pub fn a2a_rpc_error_from_py(
    obj: &Bound<'_, PyAny>,
) -> PyResult<lk_agents_abi::a2a::protocol::JsonRpcError> {
    serde_json::from_value(crate::py_values::value_from_py(obj, "A2A JSON-RPC error")?)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub fn a2a_rpc_error_to_py(
    py: Python<'_>,
    value: lk_agents_abi::a2a::protocol::JsonRpcError,
) -> PyResult<Bound<'_, PyAny>> {
    let value =
        serde_json::to_value(value).map_err(|error| PyValueError::new_err(error.to_string()))?;
    crate::py_values::value_to_py(py, &value)
}
