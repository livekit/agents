//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(record)]` declarations of `lk-agents-abi`.
//!
//! The `text_session` records.

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/text_session.rs.
static FRAME_ID: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::text_session::FrameId {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(FRAME_ID.import(py, "livekit.agents._generated.text_session", "FrameId")?)? {
            return Err(super::not_a(obj, "FrameId"));
        }
        Ok(Self {
            value: ::lk_agents_abi::python::attr_with(obj, "FrameId", "value", |v| v.extract::<::std::string::String>())?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::text_session::FrameId {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { value } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("value", ::pyo3::IntoPyObjectExt::into_bound_py_any(value, py)?)?;
        FRAME_ID.import(py, "livekit.agents._generated.text_session", "FrameId")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/text_session.rs.
static PARENT_CALL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::text_session::ParentCall {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(PARENT_CALL.import(py, "livekit.agents._generated.text_session", "ParentCall")?)? {
            return Err(super::not_a(obj, "ParentCall"));
        }
        Ok(Self {
            frame_id: ::lk_agents_abi::python::attr_with(obj, "ParentCall", "frame_id", |v| v.extract::<::lk_agents_abi::text_session::FrameId>().map_err(::std::convert::Into::<PyErr>::into))?,
            call_id: ::lk_agents_abi::python::attr_with(obj, "ParentCall", "call_id", |v| v.extract::<::std::string::String>())?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::text_session::ParentCall {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { frame_id, call_id } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("frame_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(frame_id, py)?)?;
        kwargs.set_item("call_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(call_id, py)?)?;
        PARENT_CALL.import(py, "livekit.agents._generated.text_session", "ParentCall")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/text_session.rs.
static CHILD_CALL_STATUS_APPROVAL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static CHILD_CALL_STATUS_READY: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static CHILD_CALL_STATUS_RUNNING: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::text_session::ChildCallStatus {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if obj.is_instance(CHILD_CALL_STATUS_APPROVAL.import(py, "livekit.agents._generated.text_session", "ChildCallStatusApproval")?)? {
            return Ok(Self::Approval);
        }
        if obj.is_instance(CHILD_CALL_STATUS_READY.import(py, "livekit.agents._generated.text_session", "ChildCallStatusReady")?)? {
            return Ok(Self::Ready);
        }
        if obj.is_instance(CHILD_CALL_STATUS_RUNNING.import(py, "livekit.agents._generated.text_session", "ChildCallStatusRunning")?)? {
            return Ok(Self::Running { frame_id: ::lk_agents_abi::python::attr_with(obj, "ChildCallStatusRunning", "frame_id", |v| v.extract::<::lk_agents_abi::text_session::FrameId>().map_err(::std::convert::Into::<PyErr>::into))? });
        }
        Err(super::not_a(obj, "ChildCallStatusApproval | ChildCallStatusReady | ChildCallStatusRunning"))
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::text_session::ChildCallStatus {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Approval => CHILD_CALL_STATUS_APPROVAL.import(py, "livekit.agents._generated.text_session", "ChildCallStatusApproval")?.call0(),
            Self::Ready => CHILD_CALL_STATUS_READY.import(py, "livekit.agents._generated.text_session", "ChildCallStatusReady")?.call0(),
            Self::Running { frame_id } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("frame_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(frame_id, py)?)?;
                CHILD_CALL_STATUS_RUNNING.import(py, "livekit.agents._generated.text_session", "ChildCallStatusRunning")?.call((), Some(&kwargs))
            },
        }
    }
}

// Declared in rust/crates/lk-agents-abi/src/text_session.rs.
static PENDING_CHILD_CALL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::text_session::PendingChildCall {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(PENDING_CHILD_CALL.import(py, "livekit.agents._generated.text_session", "PendingChildCall")?)? {
            return Err(super::not_a(obj, "PendingChildCall"));
        }
        Ok(Self {
            parent: ::lk_agents_abi::python::attr_with(obj, "PendingChildCall", "parent", |v| v.extract::<::lk_agents_abi::text_session::ParentCall>().map_err(::std::convert::Into::<PyErr>::into))?,
            invocation_id: ::lk_agents_abi::python::attr_with(obj, "PendingChildCall", "invocation_id", |v| v.extract::<::std::string::String>())?,
            execution_id: ::lk_agents_abi::python::attr_with(obj, "PendingChildCall", "execution_id", |v| v.extract::<::std::string::String>())?,
            tool_name: ::lk_agents_abi::python::attr_with(obj, "PendingChildCall", "tool_name", |v| v.extract::<::std::string::String>())?,
            agent_name: ::lk_agents_abi::python::attr_with(obj, "PendingChildCall", "agent_name", |v| v.extract::<::std::string::String>())?,
            validated_arguments: ::lk_agents_abi::python::attr_with(obj, "PendingChildCall", "validated_arguments", |v| ::lk_agents_abi::python::value_from_py(v, "PendingChildCall.validated_arguments"))?,
            identity: ::lk_agents_abi::python::attr_with(obj, "PendingChildCall", "identity", |v| v.extract::<::std::string::String>())?,
            status: ::lk_agents_abi::python::attr_with(obj, "PendingChildCall", "status", |v| v.extract::<::lk_agents_abi::text_session::ChildCallStatus>().map_err(::std::convert::Into::<PyErr>::into))?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::text_session::PendingChildCall {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { parent, invocation_id, execution_id, tool_name, agent_name, validated_arguments, identity, status } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("parent", ::pyo3::IntoPyObjectExt::into_bound_py_any(parent, py)?)?;
        kwargs.set_item("invocation_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(invocation_id, py)?)?;
        kwargs.set_item("execution_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(execution_id, py)?)?;
        kwargs.set_item("tool_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(tool_name, py)?)?;
        kwargs.set_item("agent_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(agent_name, py)?)?;
        kwargs.set_item("validated_arguments", ::lk_agents_abi::python::value_to_py(py, &validated_arguments)?)?;
        kwargs.set_item("identity", ::pyo3::IntoPyObjectExt::into_bound_py_any(identity, py)?)?;
        kwargs.set_item("status", ::pyo3::IntoPyObjectExt::into_bound_py_any(status, py)?)?;
        PENDING_CHILD_CALL.import(py, "livekit.agents._generated.text_session", "PendingChildCall")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/text_session.rs.
static SESSION_FRAME: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::text_session::SessionFrame {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(SESSION_FRAME.import(py, "livekit.agents._generated.text_session", "SessionFrame")?)? {
            return Err(super::not_a(obj, "SessionFrame"));
        }
        Ok(Self {
            id: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "id", |v| v.extract::<::lk_agents_abi::text_session::FrameId>().map_err(::std::convert::Into::<PyErr>::into))?,
            agent_name: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "agent_name", |v| v.extract::<::std::string::String>())?,
            history: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "history", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::lk_agents_abi::ChatItem>().map_err(::std::convert::Into::<PyErr>::into) }).collect::<PyResult<Vec<_>>>())?,
            parent: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "parent", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::text_session::ParentCall>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
            context_scope: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "context_scope", |v| v.extract::<::std::string::String>())?,
            completed: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "completed", |v| v.extract::<bool>())?,
            pending_calls: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "pending_calls", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::lk_agents_abi::text_session::PendingChildCall>().map_err(::std::convert::Into::<PyErr>::into) }).collect::<PyResult<Vec<_>>>())?,
            answer_tool: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "answer_tool", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            visible_dialogue: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "visible_dialogue", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::lk_agents_abi::ChatMessage>().map_err(::std::convert::Into::<PyErr>::into) }).collect::<PyResult<Vec<_>>>())?,
            starting_history_ids: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "starting_history_ids", |v| v.try_iter()?.map(|item| { let item = &item?; item.extract::<::std::string::String>() }).collect::<PyResult<::std::collections::BTreeSet<_>>>())?,
            transferred_message_ids: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "transferred_message_ids", |v| v.try_iter()?.map(|item| { let item = &item?; item.extract::<::std::string::String>() }).collect::<PyResult<::std::collections::BTreeSet<_>>>())?,
            merge_chat_history: ::lk_agents_abi::python::attr_with(obj, "SessionFrame", "merge_chat_history", |v| v.extract::<bool>())?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::text_session::SessionFrame {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { id, agent_name, history, parent, context_scope, completed, pending_calls, answer_tool, visible_dialogue, starting_history_ids, transferred_message_ids, merge_chat_history } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("id", ::pyo3::IntoPyObjectExt::into_bound_py_any(id, py)?)?;
        kwargs.set_item("agent_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(agent_name, py)?)?;
        kwargs.set_item("history", ::pyo3::types::PyList::new(py, history.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("parent", match parent { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("context_scope", ::pyo3::IntoPyObjectExt::into_bound_py_any(context_scope, py)?)?;
        kwargs.set_item("completed", ::pyo3::IntoPyObjectExt::into_bound_py_any(completed, py)?)?;
        kwargs.set_item("pending_calls", ::pyo3::types::PyList::new(py, pending_calls.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("answer_tool", match answer_tool { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("visible_dialogue", ::pyo3::types::PyList::new(py, visible_dialogue.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("starting_history_ids", ::pyo3::types::PyFrozenSet::new(py, starting_history_ids.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("transferred_message_ids", ::pyo3::types::PyFrozenSet::new(py, transferred_message_ids.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("merge_chat_history", ::pyo3::IntoPyObjectExt::into_bound_py_any(merge_chat_history, py)?)?;
        SESSION_FRAME.import(py, "livekit.agents._generated.text_session", "SessionFrame")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/text_session.rs.
static TEXT_SESSION_STATE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::text_session::TextSessionState {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(TEXT_SESSION_STATE.import(py, "livekit.agents._generated.text_session", "TextSessionState")?)? {
            return Err(super::not_a(obj, "TextSessionState"));
        }
        Ok(Self {
            session_id: ::lk_agents_abi::python::attr_with(obj, "TextSessionState", "session_id", |v| v.extract::<::std::string::String>())?,
            execution_id: ::lk_agents_abi::python::attr_with(obj, "TextSessionState", "execution_id", |v| v.extract::<::std::string::String>())?,
            root_frame_id: ::lk_agents_abi::python::attr_with(obj, "TextSessionState", "root_frame_id", |v| v.extract::<::lk_agents_abi::text_session::FrameId>().map_err(::std::convert::Into::<PyErr>::into))?,
            active_frame_id: ::lk_agents_abi::python::attr_with(obj, "TextSessionState", "active_frame_id", |v| v.extract::<::lk_agents_abi::text_session::FrameId>().map_err(::std::convert::Into::<PyErr>::into))?,
            ancestors: ::lk_agents_abi::python::attr_with(obj, "TextSessionState", "ancestors", |v| v.cast::<::pyo3::types::PyList>()?.iter().map(|item| { let item = &item; item.extract::<::lk_agents_abi::text_session::FrameId>().map_err(::std::convert::Into::<PyErr>::into) }).collect::<PyResult<Vec<_>>>())?,
            frames: ::lk_agents_abi::python::attr_with(obj, "TextSessionState", "frames", |v| v.cast::<::pyo3::types::PyDict>()?.iter().map(|(key, item)| { let (key, item) = (&key, &item); Ok((key.extract::<::lk_agents_abi::text_session::FrameId>().map_err(::std::convert::Into::<PyErr>::into)?, item.extract::<::lk_agents_abi::text_session::SessionFrame>().map_err(::std::convert::Into::<PyErr>::into)?)) }).collect::<PyResult<::std::collections::BTreeMap<_, _>>>())?,
            next_id: ::lk_agents_abi::python::attr_with(obj, "TextSessionState", "next_id", |v| ::lk_agents_abi::python::number_from_py::<u64>(v, "TextSessionState.next_id"))?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::text_session::TextSessionState {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { session_id, execution_id, root_frame_id, active_frame_id, ancestors, frames, next_id } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("session_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(session_id, py)?)?;
        kwargs.set_item("execution_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(execution_id, py)?)?;
        kwargs.set_item("root_frame_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(root_frame_id, py)?)?;
        kwargs.set_item("active_frame_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(active_frame_id, py)?)?;
        kwargs.set_item("ancestors", ::pyo3::types::PyList::new(py, ancestors.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("frames", { let d = ::pyo3::types::PyDict::new(py); for (k, x) in frames { d.set_item(::pyo3::IntoPyObjectExt::into_bound_py_any(k, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?; } d.into_any() })?;
        kwargs.set_item("next_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(next_id, py)?)?;
        TEXT_SESSION_STATE.import(py, "livekit.agents._generated.text_session", "TextSessionState")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/text_session.rs.

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::text_session::ChatHistory {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
        match &*value {
            "fresh" => Ok(Self::Fresh),
            "fork" => Ok(Self::Fork),
            other => Err(::pyo3::exceptions::PyValueError::new_err(format!("ChatHistory: {other:?} is not one of 'fresh', 'fork'"))),
        }
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::text_session::ChatHistory {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let value = match self {
            Self::Fresh => "fresh",
            Self::Fork => "fork",
        };
        Ok(::pyo3::types::PyString::new(py, value).into_any())
    }
}

// Declared in rust/crates/lk-agents-abi/src/text_session.rs.
static LOCAL_AGENT_TOOL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::text_session::LocalAgentTool {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(LOCAL_AGENT_TOOL.import(py, "livekit.agents._generated.text_session", "LocalAgentTool")?)? {
            return Err(super::not_a(obj, "LocalAgentTool"));
        }
        Ok(Self {
            agent_name: ::lk_agents_abi::python::attr_with(obj, "LocalAgentTool", "agent_name", |v| v.extract::<::std::string::String>())?,
            chat_history: ::lk_agents_abi::python::attr_with(obj, "LocalAgentTool", "chat_history", |v| v.extract::<::lk_agents_abi::text_session::ChatHistory>().map_err(::std::convert::Into::<PyErr>::into))?,
            merge_chat_history: ::lk_agents_abi::python::attr_with(obj, "LocalAgentTool", "merge_chat_history", |v| v.extract::<bool>())?,
            context_scope: ::lk_agents_abi::python::attr_with(obj, "LocalAgentTool", "context_scope", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::text_session::LocalAgentTool {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { agent_name, chat_history, merge_chat_history, context_scope } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("agent_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(agent_name, py)?)?;
        kwargs.set_item("chat_history", ::pyo3::IntoPyObjectExt::into_bound_py_any(chat_history, py)?)?;
        kwargs.set_item("merge_chat_history", ::pyo3::IntoPyObjectExt::into_bound_py_any(merge_chat_history, py)?)?;
        kwargs.set_item("context_scope", match context_scope { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        LOCAL_AGENT_TOOL.import(py, "livekit.agents._generated.text_session", "LocalAgentTool")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/text_session.rs.
static ROUTING_ERROR_MAX_DEPTH: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static ROUTING_ERROR_UNKNOWN_AGENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static ROUTING_ERROR_UNKNOWN_SCOPE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static ROUTING_ERROR_UNKNOWN_TOOL: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static ROUTING_ERROR_MISSING_OUTPUT_TYPE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static ROUTING_ERROR_INVALID_STATE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static ROUTING_ERROR_MISSING_ACCEPTED_OUTPUT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::text_session::RoutingError {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::MaxDepth { max_depth } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("max_depth", ::pyo3::IntoPyObjectExt::into_bound_py_any(max_depth, py)?)?;
                ROUTING_ERROR_MAX_DEPTH.import(py, "livekit.agents._generated.text_session", "RoutingErrorMaxDepth")?.call((), Some(&kwargs))
            },
            Self::UnknownAgent(message) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                ROUTING_ERROR_UNKNOWN_AGENT.import(py, "livekit.agents._generated.text_session", "RoutingErrorUnknownAgent")?.call((), Some(&kwargs))
            },
            Self::UnknownScope(message) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                ROUTING_ERROR_UNKNOWN_SCOPE.import(py, "livekit.agents._generated.text_session", "RoutingErrorUnknownScope")?.call((), Some(&kwargs))
            },
            Self::UnknownTool(message) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                ROUTING_ERROR_UNKNOWN_TOOL.import(py, "livekit.agents._generated.text_session", "RoutingErrorUnknownTool")?.call((), Some(&kwargs))
            },
            Self::MissingOutputType(message) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                ROUTING_ERROR_MISSING_OUTPUT_TYPE.import(py, "livekit.agents._generated.text_session", "RoutingErrorMissingOutputType")?.call((), Some(&kwargs))
            },
            Self::InvalidState(message) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
                ROUTING_ERROR_INVALID_STATE.import(py, "livekit.agents._generated.text_session", "RoutingErrorInvalidState")?.call((), Some(&kwargs))
            },
            Self::MissingAcceptedOutput => ROUTING_ERROR_MISSING_ACCEPTED_OUTPUT.import(py, "livekit.agents._generated.text_session", "RoutingErrorMissingAcceptedOutput")?.call0(),
        }
    }
}

// Declared in rust/crates/lk-agents-abi/src/text_session.rs.
static TEXT_SESSION_TURN: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::text_session::TextSessionTurn {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { result, state } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("result", ::pyo3::IntoPyObjectExt::into_bound_py_any(result, py)?)?;
        kwargs.set_item("state", ::pyo3::IntoPyObjectExt::into_bound_py_any(state, py)?)?;
        TEXT_SESSION_TURN.import(py, "livekit.agents._generated.text_session", "TextSessionTurn")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/text_session.rs.
static TEXT_SESSION_ERROR_ROUTING: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static TEXT_SESSION_ERROR_RUN: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::text_session::TextSessionError {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Routing(error) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("error", ::pyo3::IntoPyObjectExt::into_bound_py_any(error, py)?)?;
                TEXT_SESSION_ERROR_ROUTING.import(py, "livekit.agents._generated.text_session", "TextSessionErrorRouting")?.call((), Some(&kwargs))
            },
            Self::Run(error) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("error", ::pyo3::IntoPyObjectExt::into_bound_py_any(error, py)?)?;
                TEXT_SESSION_ERROR_RUN.import(py, "livekit.agents._generated.text_session", "TextSessionErrorRun")?.call((), Some(&kwargs))
            },
        }
    }
}

// Declared in rust/crates/lk-agents-abi/src/text_session.rs.
static TEXT_SESSION_AGENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::text_session::TextSessionAgent {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(TEXT_SESSION_AGENT.import(py, "livekit.agents._generated.text_session", "TextSessionAgent")?)? {
            return Err(super::not_a(obj, "TextSessionAgent"));
        }
        Ok(Self {
            definition: ::lk_agents_abi::python::attr_with(obj, "TextSessionAgent", "definition", |v| v.extract::<::lk_agents_abi::agent::AgentDefinition>().map_err(::std::convert::Into::<PyErr>::into))?,
            llm: ::lk_agents_abi::python::attr_with(obj, "TextSessionAgent", "llm", |v| <dyn ::lk_agents_abi::Llm as crate::python::HostProxy>::from_host(v, None))?,
            host: ::lk_agents_abi::python::attr_with(obj, "TextSessionAgent", "host", |v| <dyn ::lk_agents_abi::agent::AgentHost as crate::python::HostProxy>::from_host(v, None))?,
            output_type_explicit: ::lk_agents_abi::python::attr_with(obj, "TextSessionAgent", "output_type_explicit", |v| v.extract::<bool>())?,
            agent_tools: ::lk_agents_abi::python::attr_with(obj, "TextSessionAgent", "agent_tools", |v| v.cast::<::pyo3::types::PyDict>()?.iter().map(|(key, item)| { let (key, item) = (&key, &item); Ok((key.extract::<::std::string::String>()?, item.extract::<::lk_agents_abi::text_session::LocalAgentTool>().map_err(::std::convert::Into::<PyErr>::into)?)) }).collect::<PyResult<::std::collections::BTreeMap<_, _>>>())?,
            host_context: ::lk_agents_abi::python::attr_with(obj, "TextSessionAgent", "host_context", |v| v.extract::<::lk_agents_abi::HostRef>().map_err(::std::convert::Into::<PyErr>::into))?,
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/text_session.rs.
static TEXT_SESSION_REQUEST: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::text_session::TextSessionRequest {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(TEXT_SESSION_REQUEST.import(py, "livekit.agents._generated.text_session", "TextSessionRequest")?)? {
            return Err(super::not_a(obj, "TextSessionRequest"));
        }
        Ok(Self {
            session_id: ::lk_agents_abi::python::attr_with(obj, "TextSessionRequest", "session_id", |v| v.extract::<::std::string::String>())?,
            execution_id: ::lk_agents_abi::python::attr_with(obj, "TextSessionRequest", "execution_id", |v| v.extract::<::std::string::String>())?,
            root_scope: ::lk_agents_abi::python::attr_with(obj, "TextSessionRequest", "root_scope", |v| v.extract::<::std::string::String>())?,
            contexts: ::lk_agents_abi::python::attr_with(obj, "TextSessionRequest", "contexts", |v| v.cast::<::pyo3::types::PyDict>()?.iter().map(|(key, item)| { let (key, item) = (&key, &item); Ok((key.extract::<::std::string::String>()?, item.extract::<::lk_agents_abi::HostRef>().map_err(::std::convert::Into::<PyErr>::into)?)) }).collect::<PyResult<::std::collections::BTreeMap<_, _>>>())?,
            separate_contexts: ::lk_agents_abi::python::attr_with(obj, "TextSessionRequest", "separate_contexts", |v| v.extract::<bool>())?,
            message: ::lk_agents_abi::python::attr_with(obj, "TextSessionRequest", "message", |v| v.extract::<::lk_agents_abi::ChatItem>().map_err(::std::convert::Into::<PyErr>::into))?,
            state: ::lk_agents_abi::python::attr_with(obj, "TextSessionRequest", "state", |v| if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::text_session::TextSessionState>().map_err(::std::convert::Into::<PyErr>::into)).map(Some) })?,
            delegated: ::lk_agents_abi::python::attr_with(obj, "TextSessionRequest", "delegated", |v| v.extract::<bool>())?,
            conn_options: ::lk_agents_abi::python::attr_with(obj, "TextSessionRequest", "conn_options", |v| v.extract::<::lk_agents_abi::ApiConnectOptions>().map_err(::std::convert::Into::<PyErr>::into))?,
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/text_session.rs.
static TEXT_SESSION_SETTLED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::text_session::TextSessionSettled {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { turn, output } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("turn", ::pyo3::IntoPyObjectExt::into_bound_py_any(turn, py)?)?;
        kwargs.set_item("output", match output { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        TEXT_SESSION_SETTLED.import(py, "livekit.agents._generated.text_session", "TextSessionSettled")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/text_session.rs.
static TEXT_SESSION_UPDATE_STREAM: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static TEXT_SESSION_UPDATE_INVOCATION_SETTLED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static TEXT_SESSION_UPDATE_PARKED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static TEXT_SESSION_UPDATE_INCOMPLETE_CLEANUP: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static TEXT_SESSION_UPDATE_FINISHED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static TEXT_SESSION_UPDATE_FAILED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::text_session::TextSessionUpdate {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Stream { sequence, frame_id, agent_name, invocation_id, event } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("sequence", ::pyo3::IntoPyObjectExt::into_bound_py_any(sequence, py)?)?;
                kwargs.set_item("frame_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(frame_id, py)?)?;
                kwargs.set_item("agent_name", ::pyo3::IntoPyObjectExt::into_bound_py_any(agent_name, py)?)?;
                kwargs.set_item("invocation_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(invocation_id, py)?)?;
                kwargs.set_item("event", ::pyo3::IntoPyObjectExt::into_bound_py_any(event, py)?)?;
                TEXT_SESSION_UPDATE_STREAM.import(py, "livekit.agents._generated.text_session", "TextSessionUpdateStream")?.call((), Some(&kwargs))
            },
            Self::InvocationSettled { invocation_id, usage } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("invocation_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(invocation_id, py)?)?;
                kwargs.set_item("usage", ::pyo3::IntoPyObjectExt::into_bound_py_any(usage, py)?)?;
                TEXT_SESSION_UPDATE_INVOCATION_SETTLED.import(py, "livekit.agents._generated.text_session", "TextSessionUpdateInvocationSettled")?.call((), Some(&kwargs))
            },
            Self::Parked { invocation_id, usage } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("invocation_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(invocation_id, py)?)?;
                kwargs.set_item("usage", ::pyo3::IntoPyObjectExt::into_bound_py_any(usage, py)?)?;
                TEXT_SESSION_UPDATE_PARKED.import(py, "livekit.agents._generated.text_session", "TextSessionUpdateParked")?.call((), Some(&kwargs))
            },
            Self::IncompleteCleanup(pending) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("pending", ::pyo3::types::PyList::new(py, pending.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
                TEXT_SESSION_UPDATE_INCOMPLETE_CLEANUP.import(py, "livekit.agents._generated.text_session", "TextSessionUpdateIncompleteCleanup")?.call((), Some(&kwargs))
            },
            Self::Finished(settled) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("settled", ::pyo3::IntoPyObjectExt::into_bound_py_any(settled, py)?)?;
                TEXT_SESSION_UPDATE_FINISHED.import(py, "livekit.agents._generated.text_session", "TextSessionUpdateFinished")?.call((), Some(&kwargs))
            },
            Self::Failed(error) => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("error", ::pyo3::IntoPyObjectExt::into_bound_py_any(error, py)?)?;
                TEXT_SESSION_UPDATE_FAILED.import(py, "livekit.agents._generated.text_session", "TextSessionUpdateFailed")?.call((), Some(&kwargs))
            },
        }
    }
}

// Declared in rust/crates/lk-agents-abi/src/text_session.rs.
static TEXT_SESSION_DELIVERY: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::text_session::TextSessionDelivery {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(TEXT_SESSION_DELIVERY.import(py, "livekit.agents._generated.text_session", "TextSessionDelivery")?)? {
            return Err(super::not_a(obj, "TextSessionDelivery"));
        }
        Ok(Self {
            invocation_id: ::lk_agents_abi::python::attr_with(obj, "TextSessionDelivery", "invocation_id", |v| v.extract::<::std::string::String>())?,
            messages: ::lk_agents_abi::python::attr_with(obj, "TextSessionDelivery", "messages", |v| v.cast::<::pyo3::types::PyDict>()?.iter().map(|(key, item)| { let (key, item) = (&key, &item); Ok((key.extract::<::std::string::String>()?, item.extract::<::std::string::String>()?)) }).collect::<PyResult<::std::collections::BTreeMap<_, _>>>())?,
        })
    }
}
