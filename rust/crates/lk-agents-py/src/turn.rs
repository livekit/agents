//! The turn surface of `_core`: `lk_agents_storage::run` and `lk_agents_a2a` as Python sees them.
//! `Thread` is one open thread (`Storage.open`); `Thread.open_run` takes a lane for a `Run` the host
//! drives itself (the voice lane); `Turn(thread, lane, host, options)` is one text door's turns,
//! and `turn.run(message)` takes one A2A message through its policy. The host is the door's
//! `TurnHost`, driven through its generated proxy (`_core_handle`).

use std::sync::Arc;

use lk_agents_a2a::{Turn, TurnHost, TurnOptions};
use lk_agents_abi::python::{
    HostProxy, check_pid, pyo3_async_runtimes, value_from_py, value_to_py,
};
use lk_agents_abi::{Clock, RealClock};
use lk_agents_storage::db::TaskQuery;
use lk_agents_storage::run::{
    CommitOptions, ItemRow, Run, RunOptions, RunOutcome, TaskState, Thread,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::storage::{model, models, opt_model, parse_request, to_py_err};

/// One open thread (plan 5.2 `SessionHandle`), from `Storage.open`. Its holder is the worker's
/// name with the handle's own id appended, so two handles of one worker wait on each other like
/// two workers do.
#[pyclass(name = "Thread", module = "livekit.agents._core", frozen)]
pub struct PyThread {
    pub(crate) inner: Thread,
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

    /// The durable commit sequence last observed on the connection.
    #[getter]
    fn tip(&self) -> i64 {
        self.inner.tip()
    }

    /// Every lane as a `LaneRow`. Awaitable.
    fn lanes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let thread = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let lanes = thread.lanes().await.map_err(to_py_err)?;
            Python::attach(|py| models(py, "storage_run", "LaneRow", lanes))
        })
    }

    /// The journal, every lane, after `after_id`: `EventRow`s. Awaitable.
    #[pyo3(signature = (after_id=0, limit=100))]
    fn events<'py>(
        &self,
        py: Python<'py>,
        after_id: i64,
        limit: i64,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let thread = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let events = thread.events(after_id, limit).await.map_err(to_py_err)?;
            Python::attach(|py| models(py, "storage", "EventRow", events))
        })
    }

    /// A lane's history as stored `ItemRow`s, in order. Awaitable.
    fn history<'py>(&self, py: Python<'py>, lane: &str) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let lane = self.inner.lane(lane);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let items = lane.history().await.map_err(to_py_err)?;
            Python::attach(|py| models(py, "storage_run", "ItemRow", items))
        })
    }

    /// A lane's task rows as `TaskRow`s, newest first, up to `limit`. Awaitable.
    #[pyo3(signature = (lane, limit=50))]
    fn tasks<'py>(&self, py: Python<'py>, lane: &str, limit: i64) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let lane = self.inner.lane(lane);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let tasks = lane.tasks(None, limit).await.map_err(to_py_err)?;
            Python::attach(|py| models(py, "storage_run", "TaskRow", tasks))
        })
    }

    /// One task as a `TaskRow`, whichever lane it is on; `None` when the thread has none of
    /// that id. Awaitable.
    fn task<'py>(&self, py: Python<'py>, task_id: String) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let thread = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let record = thread.task(&task_id).await.map_err(to_py_err)?;
            Python::attach(|py| opt_model(py, "storage_run", "TaskRow", &record))
        })
    }

    /// One page of every lane's tasks, last updated first, as a `TaskPage` (A2A `tasks/list`).
    /// Awaitable.
    #[pyo3(signature = (status=None, updated_after=0, page_size=50, page_token=None))]
    fn list_tasks<'py>(
        &self,
        py: Python<'py>,
        status: Option<&str>,
        updated_after: i64,
        page_size: i64,
        page_token: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let status = status
            .map(|s| {
                TaskState::from_wire(s)
                    .ok_or_else(|| PyValueError::new_err(format!("task state {s:?}")))
            })
            .transpose()?;
        let query = TaskQuery { status, updated_after, page_size, page_token };
        let thread = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let page = thread.list_tasks(&query).await.map_err(to_py_err)?;
            Python::attach(|py| model(py, "storage_run", "TaskPage", &page))
        })
    }

    /// A lane's row as a `LaneRow`, `None` before its first run. Awaitable.
    fn lane_info<'py>(&self, py: Python<'py>, lane: &str) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let lane = self.inner.lane(lane);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let info = lane.info().await.map_err(to_py_err)?;
            Python::attach(|py| opt_model(py, "storage_run", "LaneRow", &info))
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let rows = lane.db().inbox(pending).await.map_err(to_py_err)?;
            Python::attach(|py| models(py, "storage", "InboxRow", rows))
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let record = lane.cancel_task(&task_id, metadata).await.map_err(to_py_err)?;
            Python::attach(|py| opt_model(py, "storage_run", "TaskRow", &record))
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let run = lane.open(opts).await.map_err(to_py_err)?;
            Ok(PyRun {
                version: run.version(),
                closed: run.closed,
                history: run.history.clone(),
                state: run.state.clone(),
                run: Arc::new(tokio::sync::Mutex::new(run)),
            })
        })
    }

    /// Ends the database connection; every later call raises `StorageUnavailable`.
    fn close(&self) {
        self.inner.close();
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
        models(py, "storage_run", "ItemRow", self.history.clone())
    }

    /// The state as loaded, `None` for a fresh lane.
    #[getter]
    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        value_to_py(py, self.state.as_ref().unwrap_or(&serde_json::Value::Null))
    }

    /// Writes what changed since the load: `history` (`ItemRow`s, the whole of it) and `state`
    /// (`None` keeps the stored one), then releases. The version after. Awaitable.
    #[pyo3(signature = (history, state=None))]
    fn commit<'py>(
        &self,
        py: Python<'py>,
        history: &Bound<'py, PyAny>,
        state: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let history: Vec<ItemRow> = history
            .try_iter()?
            .map(|item| parse_request(&item?, "history item"))
            .collect::<PyResult<_>>()?;
        let state = state.map(|s| value_from_py(s, "state")).transpose()?;
        let run = self.run.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut run = run.lock().await;
            run.history = history;
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
        let outcome = RunOutcome::from_wire(outcome)
            .ok_or_else(|| PyValueError::new_err(format!("unknown outcome {outcome}")))?;
        let run = self.run.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
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
    /// a `TurnOptions` model (or a dict in its shape); a field left out is the crate default.
    #[new]
    #[pyo3(signature = (thread, lane, host, options=None))]
    fn new(
        thread: &PyThread,
        lane: &str,
        host: &Bound<'_, PyAny>,
        options: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        check_pid()?;
        let opts: TurnOptions =
            options.map(|o| parse_request(o, "options")).transpose()?.unwrap_or_default();
        let clock: Arc<dyn Clock> = Arc::new(RealClock::new());
        let host = <dyn TurnHost as HostProxy>::from_host(host, Some(clock.clone()))?;
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
        let turn = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let out = turn.run(message, task_id).await.map_err(to_py_err)?;
            Python::attach(|py| model(py, "storage_turn", "TurnResult", &out))
        })
    }
}

lk_agents_abi::register_host!(class PyThread);
lk_agents_abi::register_host!(class PyRun);
lk_agents_abi::register_host!(class PyTurn);

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: an options dict read as the wrong spec (a str policy applied per channel, a
    // debounce map read as one value), or an unknown policy accepted.
    #[test]
    fn options_cross_as_a_dict() {
        use std::time::Duration;

        use lk_agents_a2a::{DebounceSpec, Policy, PolicySpec};

        Python::initialize();
        Python::attach(|py| {
            let d = py.eval(c"{}", None, None).unwrap();
            assert_eq!(
                parse_request::<TurnOptions>(&d, "options").unwrap(),
                TurnOptions::default()
            );

            let d = py
                .eval(c"{'policy': 'interrupt', 'debounce': 0, 'wait_timeout': 1.5}", None, None)
                .unwrap();
            let o: TurnOptions = parse_request(&d, "options").unwrap();
            assert_eq!(o.policy, PolicySpec::All(Policy::Interrupt));
            assert_eq!(o.debounce, DebounceSpec::All(Duration::ZERO));
            assert_eq!(o.wait_timeout, Some(Duration::from_millis(1500)));

            let d = py
                .eval(c"{'policy': {'sms': 'batch'}, 'debounce': {'sms': 2.0}}", None, None)
                .unwrap();
            let o: TurnOptions = parse_request(&d, "options").unwrap();
            assert_eq!(
                o.policy,
                PolicySpec::ByChannel([("sms".to_owned(), Policy::Batch)].into_iter().collect())
            );
            assert_eq!(
                o.debounce,
                DebounceSpec::ByChannel(
                    [("sms".to_owned(), Duration::from_secs(2))].into_iter().collect()
                )
            );

            let d = py.eval(c"{'policy': 'later'}", None, None).unwrap();
            let e = parse_request::<TurnOptions>(&d, "options").unwrap_err();
            assert!(e.is_instance_of::<PyValueError>(py) && e.to_string().contains("later"), "{e}");
        });
    }
}
