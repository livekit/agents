//! The turn surface of `_core`: `lk_agents_storage` and `lk_agents_runtime::a2a` as Python
//! sees them.
//! `Thread` is one open thread (`Storage.open`); `Thread.open_run` takes a lane for a `Run` the host
//! drives itself (the voice lane); `Turn(thread, lane, host, options)` is one text door's turns,
//! and `turn.run(message)` takes one A2A message through its policy. The host is the door's
//! `TurnHost`, driven through its generated proxy (`_core_handle`).

use lk_agents_abi::SpawnerExt as _;
use std::sync::Arc;

use lk_agents_abi::a2a::options::TurnOptions;
use crate::host::check_pid;
use lk_agents_abi::python::{value_from_py, value_to_py};
use lk_agents_abi::storage as host;
use lk_agents_runtime::a2a::Turn;
use lk_agents_runtime::storage::db::TaskOrder;
use lk_agents_runtime::storage::namespace::thread_item_page;
use lk_agents_runtime::storage::{
    CommitOptions, ItemRow, Run, RunOptions, RunOutcome, TaskQuery, TaskState, Thread,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::storage::to_py_err;

/// Core rows as a list of their host records (`livekit.agents._generated.storage_class`), each by
/// its generated conversion.
fn records<T, H: From<T>>(
    py: Python<'_>,
    rows: Vec<T>,
    to_py: for<'py> fn(Python<'py>, H) -> PyResult<Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let rows = rows.into_iter().map(|row| to_py(py, H::from(row))).collect::<PyResult<Vec<_>>>()?;
    Ok(pyo3::types::PyList::new(py, rows)?.into_any().unbind())
}

/// A core row, or `None`, as its host record.
fn one<T, H: From<T>>(
    py: Python<'_>,
    row: Option<T>,
    to_py: for<'py> fn(Python<'py>, H) -> PyResult<Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    match row {
        Some(row) => Ok(to_py(py, H::from(row))?.unbind()),
        None => Ok(py.None()),
    }
}

/// One open thread (plan 5.2 `SessionHandle`), from `Storage.open`. Its holder is the worker's
/// name with the handle's own id appended, so two handles of one worker wait on each other like
/// two workers do.
#[pyclass(name = "Thread", module = "livekit.agents._core", frozen)]
pub struct PyThread {
    pub(crate) inner: Thread,
}

impl From<Thread> for PyThread {
    fn from(inner: Thread) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyThread {
    /// The database id.
    #[getter]
    fn id(&self) -> String {
        self.inner.id().to_owned()
    }

    /// `<worker holder>/<handle>`: what this handle's leases and waiter marks carry.
    #[getter]
    fn holder(&self) -> String {
        self.inner.holder().to_owned()
    }

    /// Every lane as a `LaneRow`. Awaitable.
    fn lanes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let thread = self.inner.clone();
        crate::host::future_into_py(py, async move {
            let lanes = thread.lanes().await.map_err(to_py_err)?;
            Python::attach(|py| {
                records(py, lanes, crate::generated::records::storage_class::lane_row_to_py)
            })
        })
    }

    /// The journal, every lane, after `after_id`: `EventRow`s. Awaitable.
    #[pyo3(signature = (after_id=0, limit=100))]
    fn journal<'py>(
        &self,
        py: Python<'py>,
        after_id: i64,
        limit: i64,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let thread = self.inner.clone();
        crate::host::future_into_py(py, async move {
            let events = thread.journal(after_id, limit).await.map_err(to_py_err)?;
            Python::attach(|py| {
                records(py, events, crate::generated::records::storage_class::event_row_to_py)
            })
        })
    }

    /// A lane's history as stored `ItemRow`s, in order. Awaitable.
    fn history<'py>(&self, py: Python<'py>, lane: &str) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let lane = self.inner.lane(lane);
        crate::host::future_into_py(py, async move {
            let items = lane.history().await.map_err(to_py_err)?;
            Python::attach(|py| {
                records(py, items, crate::generated::records::storage_class::item_row_to_py)
            })
        })
    }

    /// A lane's task rows as `TaskRow`s, newest first, up to `limit`. Awaitable.
    #[pyo3(signature = (lane, limit=50))]
    fn tasks<'py>(&self, py: Python<'py>, lane: &str, limit: i64) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let lane = self.inner.lane(lane);
        crate::host::future_into_py(py, async move {
            let tasks = lane.tasks(None, limit).await.map_err(to_py_err)?;
            Python::attach(|py| {
                records(py, tasks, crate::generated::records::storage_class::task_record_to_py)
            })
        })
    }

    /// One task as a `TaskRow`, whichever lane it is on; `None` when the thread has none of
    /// that id. Awaitable.
    fn task<'py>(&self, py: Python<'py>, task_id: String) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let thread = self.inner.clone();
        crate::host::future_into_py(py, async move {
            let record = thread.task(&task_id).await.map_err(to_py_err)?;
            Python::attach(|py| {
                one(py, record, crate::generated::records::storage_class::task_record_to_py)
            })
        })
    }

    /// One page of the thread's tasks as a `TaskPage`: `order` `"updated"`, last updated first
    /// (A2A `tasks/list`), or `"inserted"`, oldest first; of `lane` and `status` when given.
    /// Awaitable.
    #[pyo3(signature = (lane=None, status=None, order="updated", page_size=50, page_token=None))]
    fn list_tasks<'py>(
        &self,
        py: Python<'py>,
        lane: Option<String>,
        status: Option<Bound<'py, PyAny>>,
        order: &str,
        page_size: i64,
        page_token: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let status = status
            .map(|s| crate::generated::records::storage_class::task_state_from_py(&s))
            .transpose()?;
        let status = status.and_then(|s| TaskState::parse(s.as_str()));
        let order = match order {
            "updated" => TaskOrder::Updated,
            "inserted" => TaskOrder::Inserted,
            other => return Err(PyValueError::new_err(format!("unknown task order {other:?}"))),
        };
        let query = TaskQuery { lane, status, order, page_size, page_token, updated_after: 0 };
        let thread = self.inner.clone();
        crate::host::future_into_py(py, async move {
            let page = thread.list_tasks(&query).await.map_err(to_py_err)?;
            Python::attach(|py| {
                Ok(crate::generated::records::storage_class::task_page_to_py(
                    py,
                    host::TaskPage::from(page),
                )?
                .unbind())
            })
        })
    }

    /// One page of the thread's committed items in insertion order, of `lane` when given, after
    /// the previous page's `after` cursor: a `ThreadItemPage`. Awaitable.
    #[pyo3(signature = (lane=None, after=None, limit=100))]
    fn list_items<'py>(
        &self,
        py: Python<'py>,
        lane: Option<String>,
        after: Option<String>,
        limit: i64,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let thread = self.inner.clone();
        crate::host::future_into_py(py, async move {
            let page = thread
                .list_items(lane.as_deref(), after.as_deref().filter(|a| !a.is_empty()), limit)
                .await
                .and_then(thread_item_page)
                .map_err(to_py_err)?;
            Python::attach(|py| {
                Ok(crate::generated::records::storage_class::thread_item_page_to_py(py, page)?
                    .unbind())
            })
        })
    }

    /// A lane's row as a `LaneRow`, `None` before its first run. Awaitable.
    fn lane_info<'py>(&self, py: Python<'py>, lane: &str) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let lane = self.inner.lane(lane);
        crate::host::future_into_py(py, async move {
            let info = lane.info().await.map_err(to_py_err)?;
            Python::attach(|py| {
                one(py, info, crate::generated::records::storage_class::lane_row_to_py)
            })
        })
    }

    /// A lane's inbox rows as `InboxRow`s in arrival order: every one, or the pending ones.
    /// Awaitable.
    #[pyo3(signature = (lane, pending=false))]
    fn inbox<'py>(
        &self,
        py: Python<'py>,
        lane: &str,
        pending: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let lane = self.inner.lane(lane);
        crate::host::future_into_py(py, async move {
            let rows = lane.inbox(pending).await.map_err(to_py_err)?;
            Python::attach(|py| {
                records(py, rows, crate::generated::records::storage_class::inbox_row_to_py)
            })
        })
    }

    /// `tasks/cancel` (plan 5.5): the row `canceled` with `metadata` merged in; the `TaskRow`
    /// after, `None` when the task is not the lane's. Awaitable.
    fn cancel_task<'py>(
        &self,
        py: Python<'py>,
        lane: &str,
        task_id: String,
        metadata: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let metadata = match value_from_py(metadata, "metadata")? {
            serde_json::Value::Object(m) => m,
            _ => return Err(PyValueError::new_err("metadata must be a dict")),
        };
        let lane = self.inner.lane(lane);
        crate::host::future_into_py(py, async move {
            let record = lane.cancel_task(&task_id, metadata).await.map_err(to_py_err)?;
            Python::attach(|py| {
                one(py, record, crate::generated::records::storage_class::task_record_to_py)
            })
        })
    }

    /// Takes the lane for one run (plan 5.2), waiting up to `wait_timeout` seconds for another
    /// holder's lease. Awaitable; a `Run`.
    #[pyo3(signature = (lane, run_id, *, channel=None, wait_timeout=None))]
    fn open_run<'py>(
        &self,
        py: Python<'py>,
        lane: &str,
        run_id: String,
        channel: Option<String>,
        wait_timeout: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let mut opts = RunOptions { channel, ..RunOptions::new(run_id.clone(), run_id) };
        if let Some(secs) = wait_timeout {
            opts.wait_timeout = std::time::Duration::try_from_secs_f64(secs)
                .map_err(|e| PyValueError::new_err(format!("wait_timeout: {e}")))?;
        }
        let lane = self.inner.lane(lane);
        crate::host::future_into_py(py, async move {
            let run = lane.open(opts).await.map_err(to_py_err)?;
            Ok(PyRun {
                version: run.version(),
                closed: run.closed,
                history: run.history.clone(),
                state: run.state.clone(),
                context_scopes: run.context_scopes.clone(),
                run: Arc::new(tokio::sync::Mutex::new(run)),
            })
        })
    }

    /// Ends the database connection; every later call raises `StorageUnavailable`.
    fn close(&self) {
        let thread = self.inner.clone();
        match crate::host::process_handle() {
            // Detached: `close()` is synchronous for the host; the close ends on its own.
            Ok(runtime) => lk_agents_runtime::net::task::TokioSpawner::new(runtime)
                .spawn_detached(async move { thread.close().await }),
            Err(error) => log::warn!("the database connection could not close: {error}"),
        }
    }
}

/// One run the host drives itself (the voice lane): what it loaded, then one `commit` of the
/// history and state the host ends with, or `abandon`.
#[pyclass(name = "Run", module = "livekit.agents._core", frozen)]
pub struct PyRun {
    run: Arc<tokio::sync::Mutex<Run>>,
    version: i64,
    closed: bool,
    history: Vec<ItemRow>,
    state: Option<serde_json::Value>,
    context_scopes: Vec<host::context::ContextScope>,
}

#[pymethods]
impl PyRun {
    /// The version loaded: the commit's fence.
    #[getter]
    fn version(&self) -> i64 {
        self.version
    }

    /// The run's id already committed: `commit` only releases.
    #[getter]
    fn closed(&self) -> bool {
        self.closed
    }

    /// The history as loaded, `ItemRow`s in order.
    #[getter]
    fn history(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        records(py, self.history.clone(), crate::generated::records::storage_class::item_row_to_py)
    }

    /// The state as loaded, `None` for a fresh lane.
    #[getter]
    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        value_to_py(py, self.state.as_ref().unwrap_or(&serde_json::Value::Null))
    }

    #[getter]
    fn context_scopes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        records(
            py,
            self.context_scopes.clone(),
            crate::generated::records::storage_context::context_scope_to_py,
        )
    }

    /// Writes what changed since the load: `history` (`ItemRow`s, the whole of it) and `state`
    /// (`None` keeps the stored one), then releases. The version after. Awaitable.
    #[pyo3(signature = (history, state=None, *, scope_writes=None))]
    fn commit<'py>(
        &self,
        py: Python<'py>,
        history: &Bound<'py, PyAny>,
        state: Option<&Bound<'py, PyAny>>,
        scope_writes: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let scope_writes = scope_writes
            .map(|w| {
                w.try_iter()?
                    .map(|w| {
                        crate::generated::records::storage_context::context_scope_write_from_py(&w?)
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        let history: Vec<ItemRow> = history
            .try_iter()?
            .map(|item| {
                ItemRow::try_from(crate::generated::records::storage_class::item_row_from_py(
                    &item?,
                )?)
                .map_err(to_py_err)
            })
            .collect::<PyResult<_>>()?;
        let state = state.map(|s| value_from_py(s, "state")).transpose()?;
        let run = self.run.clone();
        crate::host::future_into_py(py, async move {
            let mut run = run.lock().await;
            run.history = history;
            run.scope_writes = scope_writes.unwrap_or_default();
            if state.is_some() {
                run.state = state;
            }
            let committed = run.commit(CommitOptions::default()).await.map_err(to_py_err)?;
            Ok(committed.version)
        })
    }

    /// Ends the run by error: `run_ended` with `outcome` and `reason`, and the release. Awaitable.
    fn abandon<'py>(
        &self,
        py: Python<'py>,
        outcome: &str,
        reason: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let outcome = RunOutcome::parse(outcome)
            .ok_or_else(|| PyValueError::new_err(format!("unknown outcome {outcome}")))?;
        let run = self.run.clone();
        crate::host::future_into_py(py, async move {
            run.lock().await.abandon(outcome, reason).await.map_err(to_py_err)
        })
    }
}

/// One text door's turns (`TurnRunner`): its lane, its policy map, its host.
#[pyclass(name = "Turn", module = "livekit.agents._core", frozen)]
pub struct PyTurn {
    inner: Arc<Turn>,
}

#[pymethods]
impl PyTurn {
    /// Construct on the host event loop: the host's coroutines are scheduled on it. `options` is
    /// a generated `TurnOptions` record; a field left out is the crate default.
    #[new]
    #[pyo3(signature = (thread, lane, host, options=None))]
    fn new(
        thread: &PyThread,
        lane: &str,
        host: &Bound<'_, PyAny>,
        options: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        check_pid()?;
        let opts: TurnOptions = options
            .map(crate::generated::records::storage_options::turn_options_from_py)
            .transpose()?
            .unwrap_or_default();
        let clock = lk_agents_runtime::process::host_clock();
        let host = crate::generated::a2a::TurnHostProxy::from_host(host)?;
        Ok(Self { inner: Arc::new(Turn::new(thread.inner.clone(), lane, host, clock, opts)) })
    }

    #[getter]
    fn lane(&self) -> String {
        self.inner.lane().to_owned()
    }

    /// The policy of a channel, as the options resolve it.
    fn policy_for(&self, channel: &str) -> String {
        self.inner.policy_for(channel).as_str().to_owned()
    }

    /// One incoming A2A message through its policy (`TurnRunner.run_turn`). `task_id` is the
    /// answer stream's id when the caller created one. Awaitable: the `TurnResult`. Raises `RunRejected` under
    /// `reject`, `asyncio.CancelledError` when the handler was cancelled, the storage errors
    /// otherwise.
    #[pyo3(signature = (message, task_id=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        message: &Bound<'py, PyAny>,
        task_id: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let message = value_from_py(message, "message")?;
        let context = crate::host::capture_context(py)?;
        let turn = self.inner.as_ref().clone().with_context(context);
        crate::host::future_into_py(py, async move {
            let out = turn.run(message, task_id).await.map_err(to_py_err)?;
            Python::attach(|py| {
                Ok(crate::generated::records::storage_turn::turn_result_to_py(
                    py,
                    lk_agents_abi::a2a::turn::TurnResult::from(out),
                )?
                .unbind())
            })
        })
    }
}

lk_agents_abi::register_host!(class PyThread);
lk_agents_abi::register_host!(class PyRun);
lk_agents_abi::register_host!(class PyTurn);
