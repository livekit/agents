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
        kwargs.set_item("name", ::pyo3::IntoPyObjectExt::into_bound_py_any(name, py)?)?;
        kwargs.set_item("status", ::pyo3::IntoPyObjectExt::into_bound_py_any(status, py)?)?;
        kwargs.set_item("updated_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(updated_at, py)?)?;
        THREAD_TASK.import(py, "livekit.agents._generated.storage_class", "ThreadTask")?.call((), Some(&kwargs))
    }
}
