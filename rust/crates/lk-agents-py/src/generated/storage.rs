//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `storage` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/storage/mod.rs.
type StorageImpl = <crate::Selected as super::Implementations>::Storage;

/// Threads in one storage namespace (docs/internals/storage.md#thread-lifecycle): a local
/// directory, or the LiveKit Cloud project of a key pair. The same contract on both backends.
#[pyclass(frozen, name = "Storage", module = "livekit.agents._core")]
pub struct Storage(::std::sync::Arc<StorageImpl>);

#[pymethods]
impl Storage {
    /// `directory`: threads in that local directory, resolved to an absolute path once.
    /// Otherwise agentstore through the public API derived from `url` (the project's
    /// `LIVEKIT_URL`), with tokens minted from the key pair. Opens nothing until first use.
    #[new]
    #[pyo3(signature = (*, directory = None, url = None, api_key = None, api_secret = None, lease_ttl = None, renew_interval = None))]
    fn new(directory: ::std::option::Option<::std::string::String>, url: ::std::option::Option<::std::string::String>, api_key: ::std::option::Option<::std::string::String>, api_secret: ::std::option::Option<::std::string::String>, lease_ttl: ::std::option::Option<f64>, renew_interval: ::std::option::Option<f64>) -> PyResult<Storage> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::new(directory, url, api_key, api_secret, lease_ttl, renew_interval).map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(Storage(::std::sync::Arc::new(out)))
    }

    /// The absolute local directory, or the public API agentstore is reached through.
    #[getter]
    fn location(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::location(&self.0);
        Ok(out)
    }

    /// A new `AT_` thread, returned once its metadata and database exist. `attributes` maps
    /// strings to strings.
    #[pyo3(signature = (*, scope = "".to_owned(), attributes = None, idle_ttl_seconds = 0))]
    fn create_thread<'py>(&self, py: Python<'py>, scope: ::std::string::String, attributes: Option<Bound<'_, PyAny>>, idle_ttl_seconds: i64) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let attributes = attributes.map(|v| match ::lk_agents_abi::python::value_from_py(&v, "attributes")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("attributes", ": expected a dict"))) }).transpose()?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::create_thread(&this, scope, attributes, idle_ttl_seconds)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// Unknown, deleted or expired threads raise `ThreadNotFound`.
    #[pyo3(signature = (thread_id))]
    fn get_thread<'py>(&self, py: Python<'py>, thread_id: ::std::string::String) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::get_thread(&this, thread_id)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// The threads of one scope, newest first; `before` is the previous page's cursor.
    #[pyo3(signature = (scope, *, before = "".to_owned(), limit = 0))]
    fn list_threads<'py>(&self, py: Python<'py>, scope: ::std::string::String, before: ::std::string::String, limit: i64) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::list_threads(&this, scope, before, limit)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// Merges `attributes` (an empty value deletes the key) and sets the idle TTL when given:
    /// the activity boundary that moves expiry.
    #[pyo3(signature = (thread_id, *, attributes = None, idle_ttl_seconds = None))]
    fn update_thread<'py>(&self, py: Python<'py>, thread_id: ::std::string::String, attributes: Option<Bound<'_, PyAny>>, idle_ttl_seconds: ::std::option::Option<i64>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let attributes = attributes.map(|v| match ::lk_agents_abi::python::value_from_py(&v, "attributes")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("attributes", ": expected a dict"))) }).transpose()?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::update_thread(&this, thread_id, attributes, idle_ttl_seconds)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// Makes the thread unavailable and reclaims its content; repeat-safe.
    #[pyo3(signature = (thread_id))]
    fn delete_thread<'py>(&self, py: Python<'py>, thread_id: ::std::string::String) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::delete_thread(&this, thread_id)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// The thread's chat items, oldest first, of one agent's lane when `agent` is given.
    #[pyo3(signature = (thread_id, *, agent = "".to_owned(), after = "".to_owned(), limit = 0))]
    fn get_thread_items<'py>(&self, py: Python<'py>, thread_id: ::std::string::String, agent: ::std::string::String, after: ::std::string::String, limit: i64) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::get_thread_items(&this, thread_id, agent, after, limit)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// The thread's A2A tasks, of one `status` when given.
    #[pyo3(signature = (thread_id, *, status = None))]
    fn get_thread_tasks<'py>(&self, py: Python<'py>, thread_id: ::std::string::String, status: ::std::option::Option<::std::string::String>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::get_thread_tasks(&this, thread_id, status)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// The existing thread's handle, its writes under `holder`. Takes no lease.
    #[pyo3(signature = (thread_id, holder))]
    fn open<'py>(&self, py: Python<'py>, thread_id: ::std::string::String, holder: ::std::string::String) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::open(&this, thread_id, holder)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(crate::handles::Thread::from(out), py))
        })
    }

    /// The storage's response streams, shared by its threads.
    fn streams<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::streams(&this, )
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(crate::handles::Streams::from(out), py))
        })
    }

    /// Releases backend resources once open handles settled; repeat-safe.
    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}
