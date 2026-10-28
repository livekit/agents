//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(record)]` declarations of `lk-agents-abi`.
//!
//! The `storage` records.

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/storage/mod.rs.
static THREAD_INFO: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::storage::ThreadInfo {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { thread_id, scope, attributes, created_at, last_active, expires_at, idle_ttl_seconds, database_id } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("thread_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(thread_id, py)?)?;
        kwargs.set_item("scope", ::pyo3::IntoPyObjectExt::into_bound_py_any(scope, py)?)?;
        kwargs.set_item("attributes", ::lk_agents_abi::python::extra_to_py(py, &attributes)?.into_any())?;
        kwargs.set_item("created_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(created_at, py)?)?;
        kwargs.set_item("last_active", ::pyo3::IntoPyObjectExt::into_bound_py_any(last_active, py)?)?;
        kwargs.set_item("expires_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(expires_at, py)?)?;
        kwargs.set_item("idle_ttl_seconds", ::pyo3::IntoPyObjectExt::into_bound_py_any(idle_ttl_seconds, py)?)?;
        kwargs.set_item("database_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(database_id, py)?)?;
        THREAD_INFO.import(py, "livekit.agents._generated.storage_class", "ThreadInfo")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/storage/mod.rs.
static THREAD_PAGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::storage::ThreadPage {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { threads, before } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("threads", ::pyo3::types::PyList::new(py, threads.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("before", ::pyo3::IntoPyObjectExt::into_bound_py_any(before, py)?)?;
        THREAD_PAGE.import(py, "livekit.agents._generated.storage_class", "ThreadPage")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/storage/mod.rs.
static THREAD_ITEM: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::storage::ThreadItem {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { agent, seq, message_id, item } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("agent", ::pyo3::IntoPyObjectExt::into_bound_py_any(agent, py)?)?;
        kwargs.set_item("seq", ::pyo3::IntoPyObjectExt::into_bound_py_any(seq, py)?)?;
        kwargs.set_item("message_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(message_id, py)?)?;
        kwargs.set_item("item", ::lk_agents_abi::python::value_to_py(py, &item)?)?;
        THREAD_ITEM.import(py, "livekit.agents._generated.storage_class", "ThreadItem")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/storage/mod.rs.
static THREAD_ITEM_PAGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::storage::ThreadItemPage {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { items, after } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("items", ::pyo3::types::PyList::new(py, items.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("after", ::pyo3::IntoPyObjectExt::into_bound_py_any(after, py)?)?;
        THREAD_ITEM_PAGE.import(py, "livekit.agents._generated.storage_class", "ThreadItemPage")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/storage/mod.rs.
static THREAD_TASK: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::storage::ThreadTask {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { task_id, thread_id, name, status, updated_at } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("task_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(task_id, py)?)?;
        kwargs.set_item("thread_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(thread_id, py)?)?;
        kwargs.set_item("name", match name { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("status", ::pyo3::IntoPyObjectExt::into_bound_py_any(status, py)?)?;
        kwargs.set_item("updated_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(updated_at, py)?)?;
        THREAD_TASK.import(py, "livekit.agents._generated.storage_class", "ThreadTask")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/storage/rows.rs.

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::storage::TaskState {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let value = obj.extract::<::pyo3::pybacked::PyBackedStr>()?;
        match &*value {
            "submitted" => Ok(Self::Submitted),
            "working" => Ok(Self::Working),
            "input-required" => Ok(Self::InputRequired),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            "canceled" => Ok(Self::Canceled),
            "rejected" => Ok(Self::Rejected),
            other => Err(::pyo3::exceptions::PyValueError::new_err(format!("TaskState: {other:?} is not one of 'submitted', 'working', 'input-required', 'completed', 'failed', 'canceled', 'rejected'"))),
        }
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::storage::TaskState {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let value = match self {
            Self::Submitted => "submitted",
            Self::Working => "working",
            Self::InputRequired => "input-required",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Canceled => "canceled",
            Self::Rejected => "rejected",
        };
        Ok(::pyo3::types::PyString::new(py, value).into_any())
    }
}

// Declared in rust/crates/lk-agents-abi/src/storage/rows.rs.
static ASK: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::storage::Ask {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { question, tool, call_id, answer_tool } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("question", ::pyo3::IntoPyObjectExt::into_bound_py_any(question, py)?)?;
        kwargs.set_item("tool", match tool { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("call_id", match call_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("answer_tool", match answer_tool { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        ASK.import(py, "livekit.agents._generated.storage_class", "Ask")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/storage/rows.rs.
static STREAM_INFO: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::storage::StreamInfo {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { stream_id, expires_at } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("stream_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(stream_id, py)?)?;
        kwargs.set_item("expires_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(expires_at, py)?)?;
        STREAM_INFO.import(py, "livekit.agents._generated.storage_class", "StreamInfo")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/storage/rows.rs.
static ITEM_ROW: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::storage::ItemRow {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(ITEM_ROW.import(py, "livekit.agents._generated.storage_class", "ItemRow")?)? {
            return Err(super::not_a(obj, "ItemRow"));
        }
        Ok(Self {
            id: ::lk_agents_abi::python::attr_with(obj, "ItemRow", "id", |v| v.extract::<::std::string::String>())?,
            message_id: ::lk_agents_abi::python::attr_with(obj, "ItemRow", "message_id", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
            payload: ::lk_agents_abi::python::attr_with(obj, "ItemRow", "payload", |v| ::lk_agents_abi::python::value_from_py(v, "ItemRow.payload"))?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::storage::ItemRow {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { id, message_id, payload } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("id", ::pyo3::IntoPyObjectExt::into_bound_py_any(id, py)?)?;
        kwargs.set_item("message_id", match message_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("payload", ::lk_agents_abi::python::value_to_py(py, &payload)?)?;
        ITEM_ROW.import(py, "livekit.agents._generated.storage_class", "ItemRow")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/storage/rows.rs.
static TASK_DATA: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::storage::TaskData {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { history, artifacts, metadata, status_message } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("history", ::pyo3::types::PyList::new(py, history.into_iter().map(|x| PyResult::Ok(::lk_agents_abi::python::value_to_py(py, &x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("artifacts", match artifacts { None => py.None().into_bound(py), Some(x) => ::pyo3::types::PyList::new(py, x.into_iter().map(|x| PyResult::Ok(::lk_agents_abi::python::value_to_py(py, &x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any() })?;
        kwargs.set_item("metadata", ::lk_agents_abi::python::extra_to_py(py, &metadata)?.into_any())?;
        kwargs.set_item("status_message", match status_message { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::value_to_py(py, &x)? })?;
        TASK_DATA.import(py, "livekit.agents._generated.storage_class", "TaskData")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/storage/rows.rs.
static TASK_RECORD: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::storage::TaskRecord {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { task_id, lane, seq, name, message_id, status, updated_at, data, internal } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("task_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(task_id, py)?)?;
        kwargs.set_item("lane", ::pyo3::IntoPyObjectExt::into_bound_py_any(lane, py)?)?;
        kwargs.set_item("seq", ::pyo3::IntoPyObjectExt::into_bound_py_any(seq, py)?)?;
        kwargs.set_item("name", match name { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("message_id", match message_id { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("status", ::pyo3::IntoPyObjectExt::into_bound_py_any(status, py)?)?;
        kwargs.set_item("updated_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(updated_at, py)?)?;
        kwargs.set_item("data", ::pyo3::IntoPyObjectExt::into_bound_py_any(data, py)?)?;
        kwargs.set_item("internal", match internal { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        TASK_RECORD.import(py, "livekit.agents._generated.storage_class", "TaskRecord")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/storage/rows.rs.
static TASK_PAGE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::storage::TaskPage {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { tasks, next_page_token, page_size, total_size } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("tasks", ::pyo3::types::PyList::new(py, tasks.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("next_page_token", ::pyo3::IntoPyObjectExt::into_bound_py_any(next_page_token, py)?)?;
        kwargs.set_item("page_size", ::pyo3::IntoPyObjectExt::into_bound_py_any(page_size, py)?)?;
        kwargs.set_item("total_size", ::pyo3::IntoPyObjectExt::into_bound_py_any(total_size, py)?)?;
        TASK_PAGE.import(py, "livekit.agents._generated.storage_class", "TaskPage")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/storage/rows.rs.
static LANE_ROW: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::storage::LaneRow {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { lane, version, state, lease_holder, lease_until, waiter, drained_id } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("lane", ::pyo3::IntoPyObjectExt::into_bound_py_any(lane, py)?)?;
        kwargs.set_item("version", ::pyo3::IntoPyObjectExt::into_bound_py_any(version, py)?)?;
        kwargs.set_item("state", match state { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::value_to_py(py, &x)? })?;
        kwargs.set_item("lease_holder", match lease_holder { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("lease_until", ::pyo3::IntoPyObjectExt::into_bound_py_any(lease_until, py)?)?;
        kwargs.set_item("waiter", match waiter { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("drained_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(drained_id, py)?)?;
        LANE_ROW.import(py, "livekit.agents._generated.storage_class", "LaneRow")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/storage/rows.rs.
static EVENT_ROW: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::storage::EventRow {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { id, ts_ms, lane, run_id, r#type, payload } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("id", ::pyo3::IntoPyObjectExt::into_bound_py_any(id, py)?)?;
        kwargs.set_item("ts_ms", ::pyo3::IntoPyObjectExt::into_bound_py_any(ts_ms, py)?)?;
        kwargs.set_item("lane", ::pyo3::IntoPyObjectExt::into_bound_py_any(lane, py)?)?;
        kwargs.set_item("run_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(run_id, py)?)?;
        kwargs.set_item("type", ::pyo3::IntoPyObjectExt::into_bound_py_any(r#type, py)?)?;
        kwargs.set_item("payload", ::lk_agents_abi::python::value_to_py(py, &payload)?)?;
        EVENT_ROW.import(py, "livekit.agents._generated.storage_class", "EventRow")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/storage/rows.rs.
static INBOX_ROW: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::storage::InboxRow {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { id, lane, task_id, message_id, message, received_at } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("id", ::pyo3::IntoPyObjectExt::into_bound_py_any(id, py)?)?;
        kwargs.set_item("lane", ::pyo3::IntoPyObjectExt::into_bound_py_any(lane, py)?)?;
        kwargs.set_item("task_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(task_id, py)?)?;
        kwargs.set_item("message_id", ::pyo3::IntoPyObjectExt::into_bound_py_any(message_id, py)?)?;
        kwargs.set_item("message", ::lk_agents_abi::python::value_to_py(py, &message)?)?;
        kwargs.set_item("received_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(received_at, py)?)?;
        INBOX_ROW.import(py, "livekit.agents._generated.storage_class", "InboxRow")?.call((), Some(&kwargs))
    }
}
