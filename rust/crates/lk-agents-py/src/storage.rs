//! The storage surface of `_core` (plan 13.4): `lk-agents-storage` as Python sees it. `Storage`
//! holds the URLs and the key pair; `open(database_id, holder)` dials one connection and returns
//! a `Thread` (`turn.rs`); `Streams` and `Threads` are the answer streams and the threads service.
//!
//! Records cross as the pydantic models `lk-agents-codegen` writes from the crate's
//! `#[derive(Host)]` records (`livekit.agents._generated.storage*`): an argument is a model (or a
//! dict in its shape, which is what the model dumps to), and every result comes back as an instance
//! of its model, validated by the class from the JSON the crate serialised. Python never sees SQL.
//! Every async method is an awaitable on the host loop.
//!
//! Errors are the exceptions of plan 5.2 and 13.5, all under `StorageError`: `ThreadNotFound`,
//! `VersionConflict`, `NotSerializable`, `TooLarge`, `StorageUnavailable` (retryable) and
//! `RunRejected` (the `reject` policy, with `running_task_id`); a turn the host cancelled raises
//! `asyncio.CancelledError`; any other failure is a plain `StorageError`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use livekit_protocol::agent::agent_task::{GetTasksRequest, Status as TaskStatus, Task};
use livekit_protocol::agent::agent_thread::{
    CreateRequest, GetItemsRequest, GetItemsResponse, Item, ListRequest, ListResponse, Thread,
    UpdateRequest,
};
use livekit_protocol::agent::chat_context::ChatItem;
use livekit_protocol::agent_db;
use lk_agents_abi::python::{check_pid, pyo3_async_runtimes, value_from_py, value_to_py};

use crate::turn::PyThread;
use lk_agents_host_derive::Host;
use lk_agents_storage::{
    StorageError as CoreError, StorageOptions, TokenMinter, service_error, thread_error,
};
use pyo3::exceptions::asyncio::CancelledError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pyo3::create_exception!(
    _core,
    StorageError,
    PyRuntimeError,
    "A session storage failure (plan 5.2)."
);
pyo3::create_exception!(
    _core,
    ThreadNotFound,
    StorageError,
    "An id the threads service does not know, or a database agent-db does not have."
);
pyo3::create_exception!(
    _core,
    VersionConflict,
    StorageError,
    "A fenced commit matched no row: the lane moved past the version the run loaded."
);
pyo3::create_exception!(
    _core,
    NotSerializable,
    StorageError,
    "A field that could not be serialized; the message names it."
);
pyo3::create_exception!(
    _core,
    TooLarge,
    StorageError,
    "A value over its size limit; the message names what and the limit."
);
pyo3::create_exception!(
    _core,
    StorageUnavailable,
    StorageError,
    "Retryable: the connection was lost, the lease lapsed, or a transaction was discarded by a reconnect."
);
pyo3::create_exception!(
    _core,
    RunRejected,
    StorageError,
    "The reject policy refused the message while a turn runs; `running_task_id` names the turn when the journal does."
);

pub(crate) fn to_py_err(e: CoreError) -> PyErr {
    let msg = e.to_string();
    match e {
        CoreError::ThreadNotFound(_) => ThreadNotFound::new_err(msg),
        CoreError::VersionConflict { .. } => VersionConflict::new_err(msg),
        CoreError::NotSerializable(..) => NotSerializable::new_err(msg),
        CoreError::TooLarge { .. } => TooLarge::new_err(msg),
        CoreError::Unavailable(_) => StorageUnavailable::new_err(msg),
        CoreError::Invalid(_) => PyValueError::new_err(msg),
        CoreError::RunRejected { running_task_id } => {
            let err = RunRejected::new_err(msg);
            // the attribute the Python `RunRejected` carries; set on the instance, so the
            // exception's text stays the plain sentence
            Python::attach(|py| {
                let _ = err.value(py).setattr("running_task_id", running_task_id);
            });
            err
        }
        // the handler was cancelled from Python: the door sees the exception it raised
        CoreError::Cancelled(_) => CancelledError::new_err(msg),
        _ => StorageError::new_err(msg),
    }
}

/// A request into the crate's serde type: a pydantic model (its JSON dump) or a dict in the
/// model's shape; a malformed one is a `ValueError` naming the field.
pub(crate) fn parse_request<T: DeserializeOwned>(
    obj: &Bound<'_, PyAny>,
    what: &str,
) -> PyResult<T> {
    let dumped;
    let obj = if obj.hasattr("model_dump")? {
        let kwargs = PyDict::new(obj.py());
        kwargs.set_item("mode", "json")?;
        dumped = obj.call_method("model_dump", (), Some(&kwargs))?;
        &dumped
    } else {
        obj
    };
    let v = value_from_py(obj, what)?;
    serde_json::from_value(v).map_err(|e| PyValueError::new_err(format!("{what}: {e}")))
}

/// A plain result (a bool, a number, a string, or nothing) as its Python value.
pub(crate) fn result<T: Serialize>(py: Python<'_>, v: &T) -> PyResult<Py<PyAny>> {
    let v = serde_json::to_value(v).map_err(|e| StorageError::new_err(e.to_string()))?;
    Ok(value_to_py(py, &v)?.unbind())
}

/// The generated pydantic class `livekit.agents._generated.<module>.<name>`.
fn model_class<'py>(py: Python<'py>, module: &str, name: &str) -> PyResult<Bound<'py, PyAny>> {
    py.import(format!("livekit.agents._generated.{module}"))?.getattr(name)
}

/// `class.model_validate(value)`: the JSON the crate serialised, as an instance of its model.
pub(crate) fn validate(py: Python<'_>, class: &Bound<'_, PyAny>, v: &Value) -> PyResult<Py<PyAny>> {
    Ok(class.call_method1("model_validate", (value_to_py(py, v)?,))?.unbind())
}

/// A record as an instance of its generated model.
pub(crate) fn model<T: Serialize>(
    py: Python<'_>,
    module: &str,
    name: &str,
    v: &T,
) -> PyResult<Py<PyAny>> {
    let v = serde_json::to_value(v).map_err(|e| StorageError::new_err(e.to_string()))?;
    validate(py, &model_class(py, module, name)?, &v)
}

/// An optional record: its model, or `None`.
pub(crate) fn opt_model<T: Serialize>(
    py: Python<'_>,
    module: &str,
    name: &str,
    v: &Option<T>,
) -> PyResult<Py<PyAny>> {
    match v {
        Some(v) => model(py, module, name, v),
        None => Ok(py.None()),
    }
}

/// Records as a list of their model's instances, one class lookup for the list.
pub(crate) fn models<T: Serialize>(
    py: Python<'_>,
    module: &str,
    name: &str,
    items: impl IntoIterator<Item = T>,
) -> PyResult<Py<PyAny>> {
    let class = model_class(py, module, name)?;
    let out = PyList::empty(py);
    for item in items {
        let v = serde_json::to_value(&item).map_err(|e| StorageError::new_err(e.to_string()))?;
        out.append(validate(py, &class, &v)?)?;
    }
    Ok(out.into_any().unbind())
}

/// The tokio runtime the awaitables run on, entered for the synchronous entry points that spawn.
fn enter_runtime() -> tokio::runtime::EnterGuard<'static> {
    pyo3_async_runtimes::tokio::get_runtime().enter()
}

/// An awaitable of `$body`'s `Ok`, as `$into(py, out)` builds it: a plain value by default, a
/// model (or a list of them) when named.
macro_rules! awaitable {
    ($py:expr, $body:expr) => {
        awaitable!($py, $body, |py, out| result(py, &out))
    };
    ($py:expr, $body:expr, $into:expr) => {
        pyo3_async_runtimes::tokio::future_into_py($py, async move {
            let out = $body.await.map_err(to_py_err)?;
            Python::attach(|py| ($into)(py, out))
        })
    };
}

/// The root: credentials and the agentstore URL (plan 13.4), the public API every request goes
/// through, derived from `LIVEKIT_URL`.
#[pyclass(name = "Storage", module = "livekit.agents._core", frozen)]
pub struct PyStorage {
    inner: lk_agents_storage::Storage,
}

#[pymethods]
impl PyStorage {
    /// `url` is the project's `LIVEKIT_URL`; the public API agentstore is reached through is
    /// derived from it.
    #[new]
    #[pyo3(signature = (*, url, api_key, api_secret, lease_ttl=None, renew_interval=None))]
    fn new(
        url: &str,
        api_key: String,
        api_secret: String,
        lease_ttl: Option<f64>,
        renew_interval: Option<f64>,
    ) -> PyResult<Self> {
        check_pid()?;
        let mut opts = StorageOptions::new(url);
        if let Some(seconds) = lease_ttl {
            opts.lease_ttl = Duration::try_from_secs_f64(seconds)
                .map_err(|e| PyValueError::new_err(format!("lease_ttl: {e}")))?;
        }
        if let Some(seconds) = renew_interval {
            opts.renew_interval = Duration::try_from_secs_f64(seconds)
                .map_err(|e| PyValueError::new_err(format!("renew_interval: {e}")))?;
        }
        let minter = TokenMinter::new(api_key, api_secret).map_err(to_py_err)?;
        let inner = lk_agents_storage::Storage::new(opts, minter).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// The public API agentstore is reached through.
    #[getter]
    fn url(&self) -> String {
        self.inner.options().url.clone()
    }

    /// One connection to a thread's database, as a `Thread` whose leases and waiter marks carry
    /// `holder` and a handle id of its own. Awaitable.
    fn open<'py>(
        &self,
        py: Python<'py>,
        database_id: String,
        holder: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let storage = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let thread = storage.open(&database_id, &holder).await.map_err(to_py_err)?;
            Ok(PyThread { inner: thread })
        })
    }

    /// The answer streams client, shared by every thread.
    #[getter]
    fn streams(&self) -> PyStreams {
        PyStreams { inner: self.inner.streams().clone() }
    }

    /// The threads client.
    #[getter]
    fn threads(&self) -> PyThreads {
        PyThreads { inner: self.inner.clone() }
    }

    /// `DatabaseService.CreateDatabase`, for development and tests: a `DatabaseInfo`.
    #[pyo3(signature = (ttl_seconds=0))]
    fn create_database<'py>(
        &self,
        py: Python<'py>,
        ttl_seconds: i64,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let storage = self.inner.clone();
        let req = agent_db::CreateRequest { ttl: duration_of(ttl_seconds) };
        awaitable!(
            py,
            async move {
                let created = storage.databases().create(req).await.map_err(service_error)?;
                Ok::<_, CoreError>(DatabaseInfo::from(created))
            },
            |py, out| model(py, "storage", "DatabaseInfo", &out)
        )
    }

    /// `DatabaseService.DeleteDatabase`.
    fn delete_database<'py>(
        &self,
        py: Python<'py>,
        database_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let storage = self.inner.clone();
        awaitable!(py, async move {
            storage.databases().delete(&database_id).await.map_err(service_error)
        })
    }
}

/// The answer streams (plan 6) over `/v1/streams`.
#[pyclass(name = "Streams", module = "livekit.agents._core", frozen)]
pub struct PyStreams {
    inner: lk_agents_storage::Streams,
}

#[pymethods]
impl PyStreams {
    /// A new stream: a `StreamInfo`. `ttl_seconds=None` is the default, a fixed day.
    #[pyo3(signature = (ttl_seconds=None, region=None))]
    fn create<'py>(
        &self,
        py: Python<'py>,
        ttl_seconds: Option<u64>,
        region: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let streams = self.inner.clone();
        let ttl = ttl_seconds.map(std::time::Duration::from_secs);
        awaitable!(py, streams.create(ttl, region.as_deref()), |py, out| model(
            py,
            "storage",
            "StreamInfo",
            &out
        ))
    }

    /// Queues one record; sent in order, a failure is logged and does not fail the turn.
    fn append(&self, stream_id: &str, record: &[u8]) -> PyResult<()> {
        check_pid()?;
        let _rt = enter_runtime();
        self.inner.append(stream_id, record.to_vec());
        Ok(())
    }

    /// One append, awaited: the record's offset token.
    fn append_now<'py>(
        &self,
        py: Python<'py>,
        stream_id: String,
        record: Vec<u8>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let streams = self.inner.clone();
        awaitable!(py, streams.append_now(&stream_id, &record))
    }

    /// Waits until every queued append of the stream was answered.
    fn flush<'py>(&self, py: Python<'py>, stream_id: String) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let streams = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            streams.flush(&stream_id).await;
            Ok(())
        })
    }

    /// Flushes, then closes: the terminal frame.
    fn close<'py>(&self, py: Python<'py>, stream_id: String) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let streams = self.inner.clone();
        awaitable!(py, streams.close(&stream_id))
    }

    fn delete<'py>(&self, py: Python<'py>, stream_id: String) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let streams = self.inner.clone();
        awaitable!(py, streams.delete(&stream_id))
    }

    /// A reader from `after` (an offset token; `None` is the beginning). `live` follows the tail
    /// until the stream closes.
    #[pyo3(signature = (stream_id, after=None, live=false))]
    fn read(&self, stream_id: &str, after: Option<String>, live: bool) -> PyResult<PyStreamReader> {
        check_pid()?;
        let reader = self.inner.read(stream_id, after.as_deref(), live);
        Ok(PyStreamReader { inner: Arc::new(tokio::sync::Mutex::new(reader)) })
    }
}

/// A read position over one stream; `next()` yields `bytes`, `None` at the end.
#[pyclass(name = "StreamReader", module = "livekit.agents._core", frozen)]
pub struct PyStreamReader {
    inner: Arc<tokio::sync::Mutex<lk_agents_storage::StreamReader>>,
}

#[pymethods]
impl PyStreamReader {
    /// The next record, or `None`: the tail of a catch-up read, the close of a live one.
    fn next<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let reader = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let record = reader.lock().await.next().await.map_err(to_py_err)?;
            Python::attach(|py| Ok(record.map(|r| PyBytes::new(py, &r).into_any().unbind())))
        })
    }

    /// The offset to resume from after the records returned so far. Awaitable.
    fn next_offset<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let reader = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            Ok(reader.lock().await.next_offset().to_owned())
        })
    }

    /// Whether the stream's close was seen. Awaitable.
    fn is_closed<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let reader = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            Ok(reader.lock().await.is_closed())
        })
    }
}

/// The threads service's client (plan 4.3). Threads come back as `ThreadInfo`, the proto's
/// fields.
#[pyclass(name = "Threads", module = "livekit.agents._core", frozen)]
pub struct PyThreads {
    inner: lk_agents_storage::Storage,
}

#[pymethods]
impl PyThreads {
    /// A new thread with its database.
    #[pyo3(signature = (*, scope="", attributes=None, idle_ttl_seconds=0))]
    fn create_thread<'py>(
        &self,
        py: Python<'py>,
        scope: &str,
        attributes: Option<&Bound<'py, PyAny>>,
        idle_ttl_seconds: i64,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let attributes = attributes.map(|a| value_from_py(a, "attributes")).transpose()?;
        let attributes = attributes_from_json(attributes.as_ref()).map_err(to_py_err)?;
        let (threads, scope) = (self.inner.clone(), scope.to_owned());
        awaitable!(
            py,
            async move {
                let req =
                    CreateRequest { scope, attributes, idle_ttl: duration_of(idle_ttl_seconds) };
                let thread = threads.threads().create(req).await.map_err(thread_error);
                thread.map(ThreadInfo::from)
            },
            thread_model
        )
    }

    fn get_thread<'py>(&self, py: Python<'py>, thread_id: String) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let threads = self.inner.clone();
        awaitable!(
            py,
            async move {
                threads.threads().get(&thread_id).await.map_err(thread_error).map(ThreadInfo::from)
            },
            thread_model
        )
    }

    /// The threads of one subject under one agent, newest first: a `ThreadPage`, whose
    /// `before` is the cursor for the next page, empty at the end.
    #[pyo3(signature = (scope, *, before="", limit=0))]
    fn list_threads<'py>(
        &self,
        py: Python<'py>,
        scope: &str,
        before: &str,
        limit: i32,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let (threads, scope, before) = (self.inner.clone(), scope.to_owned(), before.to_owned());
        awaitable!(
            py,
            async move {
                let req = ListRequest { scope, page_token: before, limit };
                threads.threads().list(req).await.map_err(thread_error).map(ThreadPage::from)
            },
            |py, out| model(py, "storage_threads", "ThreadPage", &out)
        )
    }

    /// Merges `attributes` (an empty value deletes the entry) and sets the idle TTL when given.
    #[pyo3(signature = (thread_id, *, attributes=None, idle_ttl_seconds=None))]
    fn update_thread<'py>(
        &self,
        py: Python<'py>,
        thread_id: &str,
        attributes: Option<&Bound<'py, PyAny>>,
        idle_ttl_seconds: Option<i64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let attributes = attributes.map(|a| value_from_py(a, "attributes")).transpose()?;
        let attributes = attributes_from_json(attributes.as_ref()).map_err(to_py_err)?;
        let (threads, thread_id) = (self.inner.clone(), thread_id.to_owned());
        awaitable!(
            py,
            async move {
                let idle_ttl = idle_ttl_seconds.and_then(duration_of);
                let req = UpdateRequest { thread_id, attributes, idle_ttl };
                let thread = threads.threads().update(req).await.map_err(thread_error);
                thread.map(ThreadInfo::from)
            },
            thread_model
        )
    }

    fn delete_thread<'py>(
        &self,
        py: Python<'py>,
        thread_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let threads = self.inner.clone();
        awaitable!(
            py,
            async move { threads.threads().delete(&thread_id).await.map_err(thread_error) }
        )
    }

    /// The thread's chat items, oldest first: a `ThreadItemPage`, whose `after` is the cursor
    /// for the next page, empty at the end. `agent` keeps one agent's items.
    #[pyo3(signature = (thread_id, *, agent="", after="", limit=0))]
    fn get_thread_items<'py>(
        &self,
        py: Python<'py>,
        thread_id: &str,
        agent: &str,
        after: &str,
        limit: i32,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let threads = self.inner.clone();
        let (thread_id, agent, after) = (thread_id.to_owned(), agent.to_owned(), after.to_owned());
        awaitable!(
            py,
            async move {
                let req = GetItemsRequest {
                    thread_id,
                    agent,
                    page_token: after,
                    limit,
                    ..Default::default()
                };
                threads.threads().items(req).await.map_err(thread_error).map(ThreadItemPage::from)
            },
            |py, out| model(py, "storage_threads", "ThreadItemPage", &out)
        )
    }

    /// The thread's background tasks, oldest first: `ThreadTask`s, of one `status` when given.
    #[pyo3(signature = (thread_id, *, status=None))]
    fn get_thread_tasks<'py>(
        &self,
        py: Python<'py>,
        thread_id: &str,
        status: Option<&str>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let status = task_status(status).map_err(to_py_err)?;
        let (threads, thread_id) = (self.inner.clone(), thread_id.to_owned());
        awaitable!(
            py,
            async move {
                let mut tasks = Vec::new();
                let mut page_token = String::new();
                loop {
                    let req = GetTasksRequest {
                        thread_id: thread_id.clone(),
                        status: status.map(i32::from),
                        page_token,
                        ..Default::default()
                    };
                    let r = threads.threads().tasks(req).await.map_err(thread_error)?;
                    tasks.extend(r.tasks.into_iter().map(ThreadTask::from));
                    if r.next_page_token.is_empty() {
                        return Ok::<_, CoreError>(tasks);
                    }
                    page_token = r.next_page_token;
                }
            },
            |py, out| models(py, "storage_threads", "ThreadTask", out)
        )
    }
}

// The thread records: what Python reads threads as (plan 4.3): the proto's fields, times as unix seconds and TTLs as
// seconds, `0` for never. Python's livekit-protocol does not carry the thread messages.

/// A database made through `DatabaseService.CreateDatabase`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage", config = "frozen=True")]
pub struct DatabaseInfo {
    /// `DB_..`.
    pub database_id: String,
    /// Unix seconds; `0` for never.
    pub expires_at: i64,
}

impl From<agent_db::CreateResponse> for DatabaseInfo {
    fn from(r: agent_db::CreateResponse) -> Self {
        Self { database_id: r.database_id, expires_at: seconds_of(r.expires_at) }
    }
}

/// A `Timestamp` as unix seconds; `None` is `0`, never.
pub fn seconds_of(t: Option<pbjson_types::Timestamp>) -> i64 {
    t.map_or(0, |t| t.seconds)
}

/// A `Duration` as whole seconds; `None` is `0`.
pub fn duration_seconds(d: Option<pbjson_types::Duration>) -> i64 {
    d.map_or(0, |d| d.seconds)
}

/// Seconds as a `Duration` for the wire; `0` sends none, the server's default.
pub fn duration_of(seconds: i64) -> Option<pbjson_types::Duration> {
    (seconds > 0).then(|| pbjson_types::Duration::from(Duration::from_secs(seconds as u64)))
}

/// A thread as the host reads it (plan 4.3): the proto's fields, times as unix seconds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_threads", config = "frozen=True")]
pub struct ThreadInfo {
    pub thread_id: String,
    /// The subject the thread belongs to, the one a grant is scoped by.
    pub scope: String,
    pub attributes: HashMap<String, String>,
    pub created_at: i64,
    pub last_active: i64,
    /// Unix seconds; `0` for never.
    pub expires_at: i64,
    pub idle_ttl_seconds: i64,
    /// The thread's database on agent-db, `DB_..`, created with the thread.
    pub database_id: String,
}

impl From<Thread> for ThreadInfo {
    fn from(t: Thread) -> Self {
        Self {
            thread_id: t.thread_id,
            scope: t.scope,
            attributes: t.attributes,
            created_at: seconds_of(t.created_at),
            last_active: seconds_of(t.last_active),
            expires_at: seconds_of(t.expires_at),
            idle_ttl_seconds: duration_seconds(t.idle_ttl),
            database_id: t.database_id,
        }
    }
}

/// One page of `ListThreads`, newest first.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_threads", config = "frozen=True")]
pub struct ThreadPage {
    pub threads: Vec<ThreadInfo>,
    /// The cursor for the next page; empty at the end.
    pub before: String,
}

impl From<ListResponse> for ThreadPage {
    fn from(r: ListResponse) -> Self {
        Self {
            threads: r.threads.into_iter().map(ThreadInfo::from).collect(),
            before: r.next_page_token,
        }
    }
}

/// One chat item of a thread as the host reads it; `item` is the `ChatContext.ChatItem` as
/// protobuf JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_threads", config = "frozen=True")]
pub struct ThreadItem {
    pub agent: String,
    /// Position in the agent's history.
    pub seq: i64,
    pub message_id: String,
    #[python(ty = "dict[str, Any]", import = "typing:Any")]
    pub item: ChatItem,
}

impl From<Item> for ThreadItem {
    fn from(i: Item) -> Self {
        Self {
            agent: i.agent,
            seq: i.seq,
            message_id: i.message_id,
            item: i.item.unwrap_or_default(),
        }
    }
}

/// One page of `GetThreadItems`, oldest first.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_threads", config = "frozen=True")]
pub struct ThreadItemPage {
    pub items: Vec<ThreadItem>,
    /// The cursor for the next page; empty at the end.
    pub after: String,
}

impl From<GetItemsResponse> for ThreadItemPage {
    fn from(r: GetItemsResponse) -> Self {
        Self {
            items: r.items.into_iter().map(ThreadItem::from).collect(),
            after: r.next_page_token,
        }
    }
}

/// One background task of a thread, as `GetThreadTasks` answers it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_threads", config = "frozen=True")]
pub struct ThreadTask {
    pub task_id: String,
    pub thread_id: String,
    pub name: String,
    /// `running`, `completed`, `failed` or `canceled`.
    pub status: String,
    pub updated_at: i64,
}

impl From<Task> for ThreadTask {
    fn from(t: Task) -> Self {
        let status = TaskStatus::try_from(t.status).unwrap_or_default();
        Self {
            task_id: t.task_id,
            thread_id: t.thread_id,
            name: t.name,
            status: status.as_str_name().to_lowercase(),
            updated_at: seconds_of(t.updated_at),
        }
    }
}

/// A task status from the host, `running` and so on; `None` for every status.
pub fn task_status(status: Option<&str>) -> Result<Option<TaskStatus>, CoreError> {
    status
        .map(|s| {
            TaskStatus::from_str_name(&s.to_uppercase())
                .ok_or_else(|| CoreError::Invalid(format!("unknown task status {s}")))
        })
        .transpose()
}

/// `attributes` from the host: a string map.
pub fn attributes_from_json(v: Option<&Value>) -> Result<HashMap<String, String>, CoreError> {
    let Some(v) = v else { return Ok(HashMap::new()) };
    let obj = v
        .as_object()
        .ok_or_else(|| CoreError::Invalid("attributes must be an object of strings".into()))?;
    obj.iter()
        .map(|(k, v)| {
            v.as_str()
                .map(|s| (k.clone(), s.to_owned()))
                .ok_or_else(|| CoreError::Invalid(format!("attribute {k} is not a string")))
        })
        .collect()
}

/// A `ThreadInfo` as its model.
fn thread_model(py: Python<'_>, s: ThreadInfo) -> PyResult<Py<PyAny>> {
    model(py, "storage_threads", "ThreadInfo", &s)
}

lk_agents_abi::register_host!(class PyStorage);
lk_agents_abi::register_host!(class PyStreams);
lk_agents_abi::register_host!(class PyStreamReader);
lk_agents_abi::register_host!(class PyThreads);
lk_agents_abi::register_host!(exception StorageError);
lk_agents_abi::register_host!(exception ThreadNotFound);
lk_agents_abi::register_host!(exception VersionConflict);
lk_agents_abi::register_host!(exception NotSerializable);
lk_agents_abi::register_host!(exception TooLarge);
lk_agents_abi::register_host!(exception StorageUnavailable);
lk_agents_abi::register_host!(exception RunRejected);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_errors_map_to_their_exceptions() {
        type Class = fn(Python<'_>) -> Bound<'_, pyo3::types::PyType>;
        Python::initialize();
        Python::attach(|py| {
            let cases: [(CoreError, Class); 6] = [
                (CoreError::ThreadNotFound("x".into()), |py| py.get_type::<ThreadNotFound>()),
                (CoreError::VersionConflict { lane: "l".into(), expected: 1 }, |py| {
                    py.get_type::<VersionConflict>()
                }),
                (CoreError::NotSerializable("state".into(), "deps".into()), |py| {
                    py.get_type::<NotSerializable>()
                }),
                (CoreError::TooLarge { what: "state".into(), limit: 1 }, |py| {
                    py.get_type::<TooLarge>()
                }),
                (CoreError::Unavailable("lost".into()), |py| py.get_type::<StorageUnavailable>()),
                (CoreError::Server { code: "exec".into(), message: "m".into() }, |py| {
                    py.get_type::<StorageError>()
                }),
            ];
            for (err, class) in cases {
                let e = to_py_err(err);
                assert!(e.get_type(py).is(class(py)), "{e}");
                assert!(e.is_instance_of::<StorageError>(py));
            }
            assert!(to_py_err(CoreError::Invalid("x".into())).is_instance_of::<PyValueError>(py));
            let rejected =
                to_py_err(CoreError::RunRejected { running_task_id: Some("AST_1".into()) });
            assert!(rejected.is_instance_of::<RunRejected>(py));
            assert!(rejected.is_instance_of::<StorageError>(py));
            let running: Option<String> =
                rejected.value(py).getattr("running_task_id").unwrap().extract().unwrap();
            assert_eq!(running.as_deref(), Some("AST_1"));
            assert!(
                to_py_err(CoreError::Cancelled("x".into())).is_instance_of::<CancelledError>(py)
            );
        });
    }

    // Catches: a request that does not deserialize raising the storage error instead of a
    // `ValueError` that names the field, a model crossing as its object rather than its dump, or
    // a result handed to the class under the wrong name.
    #[test]
    fn requests_and_results_cross_as_models() {
        Python::initialize();
        Python::attach(|py| {
            let d = py.eval(c"{'id': 'i'}", None, None).unwrap();
            let e = parse_request::<lk_agents_storage::run::ItemRow>(&d, "item").unwrap_err();
            assert!(e.is_instance_of::<PyValueError>(py), "{e}");
            assert!(e.to_string().contains("payload"), "{e}");

            // a pydantic model is whatever `model_dump(mode="json")` says it is
            let ns = PyDict::new(py);
            py.run(
                c"class M:\n    def model_dump(self, mode):\n        assert mode == 'json'\n        return {'id': 'i', 'payload': {'message': {'id': 'i'}}}\n",
                None,
                Some(&ns),
            )
            .unwrap();
            let m = ns.get_item("M").unwrap().unwrap().call0().unwrap();
            let item: lk_agents_storage::run::ItemRow = parse_request(&m, "item").unwrap();
            assert_eq!((item.id.as_str(), item.message_id.as_deref()), ("i", None));

            // a result reaches `model_validate` as the crate's JSON
            py.run(
                c"class Row:\n    @classmethod\n    def model_validate(cls, d):\n        return ('validated', d)\n",
                None,
                Some(&ns),
            )
            .unwrap();
            let class = ns.get_item("Row").unwrap().unwrap();
            let out = validate(py, &class, &serde_json::to_value(&item).unwrap()).unwrap();
            let (tag, d): (String, Bound<'_, PyDict>) = out.extract(py).unwrap();
            assert_eq!(tag, "validated");
            assert_eq!(d.get_item("id").unwrap().unwrap().extract::<String>().unwrap(), "i");
            assert!(d.get_item("message_id").unwrap().unwrap().is_none());
        });
    }

    #[test]
    fn a_thread_renders_with_the_proto_field_names() {
        let t =
            Thread { thread_id: "AT_1".into(), database_id: "DB_1".into(), ..Default::default() };
        let v = serde_json::to_value(ThreadInfo::from(t)).unwrap();
        assert_eq!(v["thread_id"], "AT_1");
        assert_eq!(v["expires_at"], 0, "no timestamp is never");
        assert_eq!(seconds_of(Some(pbjson_types::Timestamp { seconds: 7, nanos: 0 })), 7);
        assert_eq!(duration_of(0), None);
        assert_eq!(duration_seconds(duration_of(90)), 90);
        assert_eq!(v["database_id"], "DB_1");
        assert!(attributes_from_json(Some(&serde_json::json!({"a": 1}))).is_err());
        let task = Task { status: TaskStatus::Running.into(), ..Default::default() };
        assert_eq!(ThreadTask::from(task).status, "running");
        assert_eq!(task_status(Some("completed")).unwrap(), Some(TaskStatus::Completed));
        assert!(task_status(Some("nope")).is_err());
    }

    #[test]
    fn a_thread_item_keeps_its_chat_item() {
        use livekit_protocol::agent::chat_context::{ChatItem, chat_item};
        let message = chat_item::Item::Message(Default::default());
        let item =
            Item { seq: 3, item: Some(ChatItem { item: Some(message) }), ..Default::default() };
        let v = serde_json::to_value(ThreadItem::from(item)).unwrap();
        assert_eq!(v["seq"], 3);
        assert!(v["item"].get("message").is_some(), "the item crosses, not null: {v}");
    }
}
