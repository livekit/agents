//! The storage surface of `_core` (plan 13.4): `lk-agents-storage` as Python sees it. `Storage`
//! holds the URLs and the key pair; `open(database_id)` dials one connection and returns a
//! `ThreadDb` whose methods are the typed statements of plan 13.5; `Streams` and `Threads` are
//! the answer streams and the threads service; `Lease` is a run's renewal timer.
//!
//! Requests and results cross as the pydantic models `lk-agents-codegen` writes from the crate's
//! `#[derive(Host)]` records (`livekit.agents._generated.storage*`): a request is a model (or a
//! dict in its shape, which is what the model dumps to), and every result, row included, comes back
//! as an instance of its model, validated by the class from the JSON the crate serialised. The
//! transport underneath is JSON either way; Python never sees SQL. Every async method is an
//! awaitable on the host loop.
//!
//! Errors are the exceptions of plan 5.2 and 13.5, all under `StorageError`: `ThreadNotFound`,
//! `VersionConflict`, `NotSerializable`, `TooLarge`, `StorageUnavailable` (retryable) and
//! `RunRejected` (the `reject` policy, with `running_task_id`); a turn the host cancelled raises
//! `asyncio.CancelledError`; any other failure is a plain `StorageError`.

use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::python::{check_pid, pyo3_async_runtimes, value_from_py, value_to_py};
use lk_agents_storage::statements::{
    AcquireRequest, CommitRequest, Fence, ItemWrite, Receive, ReceiveRequest, RunEndedRequest,
    SubmittedTask,
};
use lk_agents_storage::threads::{ThreadInfo, ThreadTask, attributes_from_json, task_status};
use lk_agents_storage::{StorageError as CoreError, StorageOptions, TokenMinter};
use pyo3::exceptions::asyncio::CancelledError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use serde::Serialize;
use serde::de::DeserializeOwned;
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
    /// `api_key` and `api_secret` default to `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`.
    #[new]
    #[pyo3(signature = (*, api_key=None, api_secret=None, lease_ttl=None, renew_interval=None))]
    fn new(
        api_key: Option<String>,
        api_secret: Option<String>,
        lease_ttl: Option<f64>,
        renew_interval: Option<f64>,
    ) -> PyResult<Self> {
        check_pid()?;
        let mut opts = StorageOptions::from_env().map_err(to_py_err)?;
        if let Some(seconds) = lease_ttl {
            opts.lease_ttl = Duration::try_from_secs_f64(seconds)
                .map_err(|e| PyValueError::new_err(format!("lease_ttl: {e}")))?;
        }
        if let Some(seconds) = renew_interval {
            opts.renew_interval = Duration::try_from_secs_f64(seconds)
                .map_err(|e| PyValueError::new_err(format!("renew_interval: {e}")))?;
        }
        let minter = match (api_key, api_secret) {
            (Some(k), Some(s)) => TokenMinter::new(k, s),
            (None, None) => TokenMinter::from_env(),
            _ => Err(CoreError::Invalid("api_key and api_secret go together".into())),
        }
        .map_err(to_py_err)?;
        let inner = lk_agents_storage::Storage::new(opts, minter).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// The public API agentstore is reached through.
    #[getter]
    fn url(&self) -> String {
        self.inner.options().url.clone()
    }

    /// One connection to the session's database, with its schema version read. Awaitable.
    fn open<'py>(&self, py: Python<'py>, database_id: String) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let storage = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let db = storage.open(&database_id).await.map_err(to_py_err)?;
            Ok(PyThreadDb { inner: db })
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
        PyThreads { inner: self.inner.threads().clone() }
    }

    /// `DatabaseService.CreateDatabase`, for development and tests: a `DatabaseInfo`.
    #[pyo3(signature = (ttl_seconds=0))]
    fn create_database<'py>(
        &self,
        py: Python<'py>,
        ttl_seconds: i64,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let databases = self.inner.databases().clone();
        awaitable!(py, databases.create_database(ttl_seconds), |py, out| model(
            py,
            "storage",
            "DatabaseInfo",
            &out
        ))
    }

    /// `DatabaseService.DeleteDatabase`.
    fn delete_database<'py>(
        &self,
        py: Python<'py>,
        database_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let databases = self.inner.databases().clone();
        awaitable!(py, databases.delete_database(&database_id))
    }
}

/// One session's database over one connection (plan 13.4), closed at run end. Requests are
/// the models of `_generated/storage.py`; rows come back as their row models.
#[pyclass(name = "ThreadDb", module = "livekit.agents._core", frozen)]
pub struct PyThreadDb {
    pub(crate) inner: lk_agents_storage::ThreadDb,
}

#[pymethods]
impl PyThreadDb {
    #[getter]
    fn database_id(&self) -> String {
        self.inner.conn().database_id().to_owned()
    }

    /// The durable commit sequence last observed on this connection.
    #[getter]
    fn tip(&self) -> i64 {
        self.inner.conn().tip()
    }

    /// Ends the connection; every later call raises `StorageUnavailable`.
    fn close(&self) {
        self.inner.close();
    }

    /// The acquire and the load, one batch (plan 3.3.2): an `AcquireRequest`. The `Loaded` lane,
    /// or `None` when another holder has the lease.
    fn acquire<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let req: AcquireRequest = parse_request(request, "acquire request")?;
        let db = self.inner.clone();
        awaitable!(py, db.acquire(&req), |py, out| opt_model(py, "storage", "Loaded", &out))
    }

    /// Releases the lease and writes nothing else: the path of an `already_committed` run.
    fn release<'py>(
        &self,
        py: Python<'py>,
        lane: String,
        lease_token: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let db = self.inner.clone();
        awaitable!(py, db.release(&lane, &lease_token))
    }

    /// One renewal; `False` when the lease is no longer this holder's.
    fn renew<'py>(
        &self,
        py: Python<'py>,
        lane: String,
        holder: String,
        lease_token: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let db = self.inner.clone();
        awaitable!(py, db.renew(&lane, &holder, &lease_token))
    }

    /// The renewal timer: renews every three seconds until `Lease.stop()`; `Lease.lost()` resolves
    /// with `StorageUnavailable` when a renew fails, which cancels the run.
    fn start_renewal(&self, lane: &str, holder: &str, lease_token: &str) -> PyResult<PyLease> {
        check_pid()?;
        let _rt = enter_runtime();
        Ok(PyLease { inner: self.inner.start_renewal(lane, holder, lease_token) })
    }

    /// The fenced commit (plan 5.2): a `CommitRequest`, answered with a `CommitResult`; raises
    /// `VersionConflict` when the fence matched no row. A `None` fence is the `concurrent`
    /// policy's append-only commit.
    fn commit<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let req: CommitRequest = parse_request(request, "commit request")?;
        let db = self.inner.clone();
        awaitable!(py, db.commit(&req), |py, out| model(py, "storage", "CommitResult", &out))
    }

    /// Items written ahead of a commit, under its `Fence`, without moving the lane's version:
    /// `ItemWrite`s another lane must read while the holder still works. Raises
    /// `VersionConflict` when the fence matched no row.
    fn write_items<'py>(
        &self,
        py: Python<'py>,
        lane: String,
        fence: &Bound<'py, PyAny>,
        items: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let fence: Fence = parse_request(fence, "fence")?;
        let items: Vec<ItemWrite> = parse_request(items, "items")?;
        let db = self.inner.clone();
        awaitable!(py, db.write_items(&lane, &fence, &items))
    }

    /// Leaving by exception: `run_ended` with the outcome, and the release; nothing else. A
    /// `RunEndedRequest`.
    fn run_ended<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let req: RunEndedRequest = parse_request(request, "run_ended request")?;
        let db = self.inner.clone();
        awaitable!(py, db.run_ended(&req))
    }

    /// The inbox receive (plan 3.3.4): a `ReceiveRequest`, answered with a `Received` (the row's
    /// id, and the version when this worker took the lease) or a `Duplicate` (the task the
    /// `messageId` already belongs to).
    fn receive<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let req: ReceiveRequest = parse_request(request, "receive request")?;
        let db = self.inner.clone();
        awaitable!(py, db.receive(&req), |py, out: Receive| {
            let name = match out {
                Receive::Received(_) => "Received",
                Receive::Duplicate(_) => "Duplicate",
            };
            model(py, "storage", name, &out)
        })
    }

    /// The early `submitted` row under interrupt, queue, concurrent and reject, before the lease
    /// is tried: a `SubmittedTask`. `True` when inserted, `False` when the task or the message
    /// already had a row.
    fn insert_submitted_task<'py>(
        &self,
        py: Python<'py>,
        task: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let task: SubmittedTask = parse_request(task, "submitted task")?;
        let db = self.inner.clone();
        awaitable!(py, db.insert_submitted_task(&task))
    }

    /// The drain check: a `DrainCheck`.
    fn drain_check<'py>(
        &self,
        py: Python<'py>,
        lane: String,
        lease_token: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let db = self.inner.clone();
        awaitable!(py, db.drain_check(&lane, &lease_token), |py, out| model(
            py,
            "storage",
            "DrainCheck",
            &out
        ))
    }

    /// Records a waiter under `interrupt`; `False` when the lease is free (acquire instead).
    fn set_waiter<'py>(
        &self,
        py: Python<'py>,
        lane: String,
        waiter: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let db = self.inner.clone();
        awaitable!(py, db.set_waiter(&lane, &waiter))
    }

    /// The holder's boundary check: the waiting worker, or `None`.
    fn waiter<'py>(&self, py: Python<'py>, lane: String) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let db = self.inner.clone();
        awaitable!(py, db.waiter(&lane))
    }

    /// Takes a `submitted` or orphaned `working` task to `working`: the status it had (`working`
    /// means a dead holder had it), or `None` when it is terminal.
    fn pickup_task<'py>(
        &self,
        py: Python<'py>,
        lane: String,
        task_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let db = self.inner.clone();
        awaitable!(py, db.pickup_task(&lane, &task_id))
    }

    /// A `working` row whose holder died ends `failed` with `status_message` (reason `worker-lost`).
    fn mark_orphan_failed<'py>(
        &self,
        py: Python<'py>,
        lane: String,
        task_id: String,
        status_message: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let message = value_from_py(status_message, "status_message")?;
        let db = self.inner.clone();
        awaitable!(py, db.mark_orphan_failed(&lane, &task_id, &message))
    }

    /// `tasks/cancel` on a running row: the request and its reason go into its metadata, for
    /// the holder to act on. `False` when the row is no longer `working`.
    #[pyo3(signature = (lane, task_id, reason=None))]
    fn request_cancel<'py>(
        &self,
        py: Python<'py>,
        lane: String,
        task_id: String,
        reason: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let db = self.inner.clone();
        awaitable!(py, db.request_cancel(&lane, &task_id, reason.as_deref()))
    }

    /// The journal, every lane, after `after_id`: `EventRow`s.
    #[pyo3(signature = (after_id=0, limit=100))]
    fn events<'py>(
        &self,
        py: Python<'py>,
        after_id: i64,
        limit: i64,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let db = self.inner.clone();
        awaitable!(py, db.events(after_id, limit), |py, out| models(py, "storage", "EventRow", out))
    }

    /// One lane's journal after `after_id`: `EventRow`s.
    #[pyo3(signature = (lane, after_id=0))]
    fn lane_events<'py>(
        &self,
        py: Python<'py>,
        lane: String,
        after_id: i64,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let db = self.inner.clone();
        awaitable!(py, db.lane_events(&lane, after_id), |py, out| models(
            py, "storage", "EventRow", out
        ))
    }

    /// Every lane row: `LaneRow`s.
    fn lanes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let db = self.inner.clone();
        awaitable!(py, db.lanes(), |py, out| models(py, "storage", "LaneRow", out))
    }

    /// One `LaneRow`, or `None`.
    fn lane<'py>(&self, py: Python<'py>, lane: String) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let db = self.inner.clone();
        awaitable!(py, db.lane(&lane), |py, out| opt_model(py, "storage", "LaneRow", &out))
    }

    /// The whole history of a lane: `ChatItemRow`s in `seq` order.
    fn history<'py>(&self, py: Python<'py>, lane: String) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let db = self.inner.clone();
        awaitable!(py, db.history(&lane), |py, out| models(py, "storage", "ChatItemRow", out))
    }

    /// `TaskRow`s, newest first, optionally one status.
    #[pyo3(signature = (lane, status=None, limit=50))]
    fn tasks<'py>(
        &self,
        py: Python<'py>,
        lane: String,
        status: Option<String>,
        limit: i64,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let db = self.inner.clone();
        awaitable!(py, db.tasks(&lane, status.as_deref(), limit), |py, out| models(
            py, "storage", "TaskRow", out
        ))
    }

    /// `working` rows a crashed holder left, in `seq` order: `TaskRow`s.
    fn orphans<'py>(&self, py: Python<'py>, lane: String) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let db = self.inner.clone();
        awaitable!(py, db.orphans(&lane), |py, out| models(py, "storage", "TaskRow", out))
    }

    /// Every inbox row of the lane, drained ones included: `InboxRow`s.
    fn inbox<'py>(&self, py: Python<'py>, lane: String) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let db = self.inner.clone();
        awaitable!(py, db.inbox(&lane), |py, out| models(py, "storage", "InboxRow", out))
    }

    /// The inbox rows above `drained_id`: `InboxRow`s.
    fn inbox_pending<'py>(&self, py: Python<'py>, lane: String) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let db = self.inner.clone();
        awaitable!(py, db.inbox_pending(&lane), |py, out| models(py, "storage", "InboxRow", out))
    }
}

/// A held lease with its renewal timer (plan 3.3.2).
#[pyclass(name = "Lease", module = "livekit.agents._core", frozen)]
pub struct PyLease {
    inner: lk_agents_storage::Lease,
}

#[pymethods]
impl PyLease {
    #[getter]
    fn lane(&self) -> String {
        self.inner.lane().to_owned()
    }

    #[getter]
    fn holder(&self) -> String {
        self.inner.holder().to_owned()
    }

    #[getter]
    fn token(&self) -> String {
        self.inner.token().to_owned()
    }

    #[getter]
    fn is_lost(&self) -> bool {
        self.inner.is_lost()
    }

    /// Resolves by raising `StorageUnavailable` once the lease is lost; pending while it is held.
    /// Race it against the run.
    fn lost<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let lease = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let e = lease.lost().await;
            Err::<(), PyErr>(to_py_err(e))
        })
    }

    /// Ends the timer; the lease itself is released by the commit or `run_ended`.
    fn stop(&self) {
        self.inner.stop();
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
    inner: lk_agents_storage::Threads,
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
                let thread = threads.create_thread(&scope, attributes, idle_ttl_seconds).await;
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
            async move { threads.get_thread(&thread_id).await.map(ThreadInfo::from) },
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
            async move { threads.list_threads(&scope, &before, limit).await },
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
                let thread = threads.update_thread(&thread_id, attributes, idle_ttl_seconds).await;
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
        awaitable!(py, threads.delete_thread(&thread_id))
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
            async move { threads.get_thread_items(&thread_id, &agent, &after, limit).await },
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
                let tasks = threads.get_thread_tasks(&thread_id, status).await?;
                Ok::<_, CoreError>(tasks.into_iter().map(ThreadTask::from).collect::<Vec<_>>())
            },
            |py, out| models(py, "storage_threads", "ThreadTask", out)
        )
    }
}

/// A `ThreadInfo` as its model.
fn thread_model(py: Python<'_>, s: ThreadInfo) -> PyResult<Py<PyAny>> {
    model(py, "storage_threads", "ThreadInfo", &s)
}

lk_agents_abi::register_host!(class PyStorage);
lk_agents_abi::register_host!(class PyThreadDb);
lk_agents_abi::register_host!(class PyLease);
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

    // Catches: an error of plan 13.5 landing on the wrong Python class, or `Invalid` (a caller's
    // mistake) raising the retryable one.
    #[test]
    fn core_errors_map_to_their_exceptions() {
        Python::initialize();
        Python::attach(|py| {
            let cases: [(CoreError, fn(Python<'_>) -> Bound<'_, pyo3::types::PyType>); 6] = [
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
                assert!(e.get_type(py).is(&class(py)), "{e}");
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
            let d = py.eval(c"{'lane': 'l', 'lease_token': 't'}", None, None).unwrap();
            let e = parse_request::<RunEndedRequest>(&d, "run_ended request").unwrap_err();
            assert!(e.is_instance_of::<PyValueError>(py), "{e}");
            assert!(e.to_string().contains("run_id"), "{e}");

            // a pydantic model is whatever `model_dump(mode="json")` says it is
            let ns = PyDict::new(py);
            py.run(
                c"class M:\n    def model_dump(self, mode):\n        assert mode == 'json'\n        return {'lane': 'l', 'run_id': 'r', 'lease_token': 't', 'ts_ms': 1, 'payload': {}}\n",
                None,
                Some(&ns),
            )
            .unwrap();
            let m = ns.get_item("M").unwrap().unwrap().call0().unwrap();
            let req: RunEndedRequest = parse_request(&m, "run_ended request").unwrap();
            assert_eq!(req.run_id, "r");

            // a result reaches `model_validate` as the crate's JSON
            py.run(
                c"class Row:\n    @classmethod\n    def model_validate(cls, d):\n        return ('validated', d)\n",
                None,
                Some(&ns),
            )
            .unwrap();
            let class = ns.get_item("Row").unwrap().unwrap();
            let row = lk_agents_storage::schema::ChatItemRow {
                id: "i".into(),
                lane: "l".into(),
                seq: 1,
                message_id: None,
                payload: serde_json::json!({"type": "message", "role": "user"}),
            };
            let out = validate(py, &class, &serde_json::to_value(&row).unwrap()).unwrap();
            let (tag, d): (String, Bound<'_, PyDict>) = out.extract(py).unwrap();
            assert_eq!(tag, "validated");
            assert_eq!(d.get_item("lane").unwrap().unwrap().extract::<String>().unwrap(), "l");
            assert!(d.get_item("message_id").unwrap().unwrap().is_none());
        });
    }
}
