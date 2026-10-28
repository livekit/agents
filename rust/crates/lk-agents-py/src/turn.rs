//! The turn surface of `_core` (plan 13.8): `lk_agents_storage::{run, turn}` as Python sees it.
//! `Thread(db, holder)` is the run module's handle over an open `ThreadDb`; `Turn(thread,
//! lane, host, options)` is one text door's turns, and `turn.run(message)` takes one A2A message
//! through its policy with the handler on the Python side; `Boundary` is what the handler polls
//! from inside `handle` to ask whether to yield.
//!
//! The host is any Python object with three coroutine methods, `handle(ctx)`, `on_update(task_id,
//! text)` and `on_token(task_id, delta)`, and under `live` four more, `accepted(task_id, local)`,
//! `on_end(task, output)`, `cancel(task_id)` and `background(wait)`. Rust calls them from the
//! tokio runtime the way the worker calls `host.open`: the coroutine is scheduled on the event
//! loop that built the `Turn` with `run_coroutine_threadsafe`, its result arrives on a oneshot
//! channel the done callback fills, and dropping the Rust future (the turn cancelled from Python)
//! cancels the Python task. The GIL is held only to build the call and to read its result, never
//! across an await, and nothing here blocks the event loop thread on the runtime.
//!
//! `handle(ctx, boundary)` receives the `TurnContext` model of `_generated/storage_turn.py`
//! (every field of the crate's record as JSON: `history` the items, `tasks` the task records,
//! `batch` the parsed messages) and the `Boundary` object beside it, and returns a
//! `HandlerOutput` model (or a dict in its shape). An exception from `handle` is the handler's
//! error, `"<TypeName>: <text>"`, which the turn writes on the failed row;
//! `asyncio.CancelledError` ends the run `canceled`; a `StorageUnavailable` or `ThreadNotFound`
//! the handler met fails the run as that error.

use std::sync::Arc;

use lk_agents_abi::python::{
    CancelPythonTask, check_pid, log, pyo3_async_runtimes, schedule_coroutine, value_from_py,
    value_to_py,
};
use lk_agents_storage::StorageError as CoreError;
use lk_agents_storage::run::{TaskRecord, Thread};
use lk_agents_storage::turn::{
    Background, Boundary, HandlerOutput, HostError, Turn, TurnContext, TurnHost, TurnOptions,
};
use pyo3::exceptions::asyncio::CancelledError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3_async_runtimes::TaskLocals;

use crate::storage::{
    PyThreadDb, StorageUnavailable, ThreadNotFound, model, models, parse_request, to_py_err,
};

/// The run module's handle over an open database (plan 5.2 `SessionHandle`): `Thread::new`.
/// `holder` is the worker's name; the handle's own id is appended, so two handles of one worker
/// wait on each other like two workers do.
#[pyclass(name = "Thread", module = "livekit.agents._core", frozen)]
pub struct PyThread {
    inner: Thread,
}

#[pymethods]
impl PyThread {
    #[new]
    fn new(db: &PyThreadDb, holder: &str) -> PyResult<Self> {
        check_pid()?;
        Ok(Self { inner: Thread::new(db.inner.clone(), holder) })
    }

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

    #[getter]
    fn now_ms(&self) -> i64 {
        self.inner.now_ms()
    }

    /// Every lane as a `LaneInfo`. Awaitable.
    fn lanes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let thread = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let lanes = thread.lanes().await.map_err(to_py_err)?;
            Python::attach(|py| models(py, "storage_run", "LaneInfo", lanes))
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

    /// A lane's history as stored `Item`s, in order. Awaitable.
    fn history<'py>(&self, py: Python<'py>, lane: &str) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let lane = self.inner.lane(lane);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let items = lane.history().await.map_err(to_py_err)?;
            Python::attach(|py| models(py, "storage_run", "Item", items))
        })
    }

    /// A lane's task rows as `TaskRecord`s, newest first, up to `limit`. Awaitable.
    #[pyo3(signature = (lane, limit=50))]
    fn tasks<'py>(&self, py: Python<'py>, lane: &str, limit: i64) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let lane = self.inner.lane(lane);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let tasks = lane.tasks(None, limit).await.map_err(to_py_err)?;
            Python::attach(|py| models(py, "storage_run", "TaskRecord", tasks))
        })
    }

    /// Ends the database connection; every later call raises `StorageUnavailable`.
    fn close(&self) {
        self.inner.close();
    }
}

/// The `boundary` of the handler's context: `check()` before each model call and once more
/// before the answer streams; `True` once means `True` from then on.
#[pyclass(name = "Boundary", module = "livekit.agents._core", frozen)]
pub struct PyBoundary {
    inner: Boundary,
}

#[pymethods]
impl PyBoundary {
    /// The task the boundary guards.
    #[getter]
    fn task_id(&self) -> String {
        self.inner.task_id().to_owned()
    }

    /// A check answered `True` at some point.
    #[getter]
    fn tripped(&self) -> bool {
        self.inner.tripped()
    }

    /// The yield was a client's `tasks/cancel`.
    #[getter]
    fn client_canceled(&self) -> bool {
        self.inner.client_canceled()
    }

    /// Whether to yield now. Awaitable; raises the storage error when the check could not read.
    fn check<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let boundary = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            boundary.check().await.map_err(to_py_err)
        })
    }
}

// --------------------------------------------------------------------- the host

/// The Python host object and the loop it runs on.
struct PyTurnHost {
    obj: Py<PyAny>,
    locals: TaskLocals,
}

impl PyTurnHost {
    /// `host.<name>(*args)` as a scheduled coroutine, awaited from the runtime; dropping the
    /// operation cancels the Python task.
    async fn call<F>(&self, name: &'static str, args: F) -> PyResult<Py<PyAny>>
    where
        F: for<'py> FnOnce(Python<'py>) -> PyResult<Bound<'py, PyTuple>>,
    {
        let (task, receiver): (CancelPythonTask, _) = Python::attach(|py| {
            let coro = self.obj.bind(py).call_method1(name, args(py)?)?;
            schedule_coroutine(py, &self.locals, coro)
        })?;
        let result = receiver
            .await
            .map_err(|_| PyRuntimeError::new_err(format!("{name}: the Python task disappeared")))?;
        drop(task);
        result
    }
}

/// `TurnContext` as the two arguments `handle` receives: the `TurnContext` model (the record
/// minus the boundary) and the `Boundary` object.
fn context_to_py<'py>(py: Python<'py>, ctx: &TurnContext) -> PyResult<(Py<PyAny>, Py<PyBoundary>)> {
    let record = serde_json::json!({
        "thread_id": ctx.thread_id,
        "lane": ctx.lane,
        "message": ctx.message,
        "batch": ctx.batch,
        "history": ctx.history,
        "state": ctx.state,
        "task_id": ctx.task_id,
        "tasks": ctx.tasks,
        "agent_stack": ctx.agent_stack,
        "caller_history": ctx.caller_history,
        "answer_tool": ctx.answer_tool,
        "answer_ref": ctx.answer_ref,
        "live": ctx.live,
    });
    let model = model(py, "storage_turn", "TurnContext", &record)?;
    let boundary = Py::new(py, PyBoundary { inner: ctx.boundary.clone() })?;
    Ok((model, boundary))
}

/// `"<TypeName>: <text>"`, the text alone when the exception has none, as `python_reason` does.
fn error_text(py: Python<'_>, e: &PyErr) -> String {
    let name = e.get_type(py).name().map(|n| n.to_string()).unwrap_or_else(|_| "Exception".into());
    let text = e.value(py).str().map(|s| s.to_string()).unwrap_or_default();
    if text.is_empty() { name } else { format!("{name}: {text}") }
}

/// What a raising `handle` means to the turn: a cancellation ends the run `canceled`, a storage
/// error the handler met fails the run, anything else is the handler's error on the failed row.
fn host_error(py: Python<'_>, e: &PyErr) -> Result<HandlerOutput, HostError> {
    let text = e.value(py).str().map(|s| s.to_string()).unwrap_or_default();
    if e.is_instance_of::<CancelledError>(py) {
        return Err(HostError::Cancelled(text));
    }
    if e.is_instance_of::<StorageUnavailable>(py) {
        return Err(HostError::Storage(CoreError::Unavailable(text)));
    }
    if e.is_instance_of::<ThreadNotFound>(py) {
        return Err(HostError::Storage(CoreError::ThreadNotFound(text)));
    }
    Ok(HandlerOutput { error: Some(error_text(py, e)), ..HandlerOutput::default() })
}

#[async_trait::async_trait]
impl TurnHost for PyTurnHost {
    async fn handle(&self, ctx: TurnContext) -> Result<HandlerOutput, HostError> {
        let res = self
            .call("handle", |py| {
                let (record, boundary) = context_to_py(py, &ctx)?;
                Ok((record, boundary).into_pyobject(py)?.into_any().cast_into()?)
            })
            .await;
        Python::attach(|py| match res {
            Ok(v) => match parse_request::<HandlerOutput>(v.bind(py), "handle output") {
                Ok(out) => Ok(out),
                // the host's bug, named on the row rather than crashing the turn
                Err(e) => Ok(HandlerOutput {
                    error: Some(format!("ValueError: handle returned an invalid output: {e}")),
                    ..HandlerOutput::default()
                }),
            },
            Err(e) => host_error(py, &e),
        })
    }

    async fn on_update(&self, task_id: &str, text: &str) {
        let (task_id, text) = (task_id.to_owned(), text.to_owned());
        if let Err(e) = self
            .call("on_update", move |py| {
                Ok((task_id, text).into_pyobject(py)?.into_any().cast_into()?)
            })
            .await
        {
            log::warn!(target: "livekit.agents", "the host's on_update raised: {e}");
        }
    }

    async fn on_token(&self, task_id: &str, delta: &str) {
        let (task_id, delta) = (task_id.to_owned(), delta.to_owned());
        if let Err(e) = self
            .call("on_token", move |py| {
                Ok((task_id, delta).into_pyobject(py)?.into_any().cast_into()?)
            })
            .await
        {
            log::warn!(target: "livekit.agents", "the host's on_token raised: {e}");
        }
    }

    async fn accepted(&self, task_id: &str, local: bool) {
        let task_id = task_id.to_owned();
        if let Err(e) = self
            .call("accepted", move |py| {
                Ok((task_id, local).into_pyobject(py)?.into_any().cast_into()?)
            })
            .await
        {
            log::warn!(target: "livekit.agents", "the host's accepted raised: {e}");
        }
    }

    async fn on_end(&self, task: &TaskRecord, output: Option<&serde_json::Value>) {
        let (task, output) = (task.clone(), output.cloned().unwrap_or_default());
        if let Err(e) = self
            .call("on_end", move |py| {
                let task = model(py, "storage_run", "TaskRecord", &task)?;
                Ok((task, value_to_py(py, &output)?).into_pyobject(py)?.into_any().cast_into()?)
            })
            .await
        {
            log::warn!(target: "livekit.agents", "the host's on_end raised: {e}");
        }
    }

    async fn cancel(&self, task_id: &str) {
        let task_id = task_id.to_owned();
        if let Err(e) = self
            .call("cancel", move |py| Ok((task_id,).into_pyobject(py)?.into_any().cast_into()?))
            .await
        {
            log::warn!(target: "livekit.agents", "the host's cancel raised: {e}");
        }
    }

    async fn background(&self, wait: std::time::Duration) -> Background {
        let wait = wait.as_secs_f64();
        let res = self
            .call("background", move |py| Ok((wait,).into_pyobject(py)?.into_any().cast_into()?))
            .await;
        Python::attach(|py| {
            res.and_then(|v| parse_request::<Background>(v.bind(py), "background")).unwrap_or_else(
                |e| {
                    // nothing known to run: the holder lets the lane go
                    log::warn!(target: "livekit.agents", "the host's background raised: {e}");
                    Background::default()
                },
            )
        })
    }
}

// --------------------------------------------------------------------- the turn

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
        py: Python<'_>,
        thread: &PyThread,
        lane: &str,
        host: Py<PyAny>,
        options: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        check_pid()?;
        for name in ["handle", "on_update", "on_token"] {
            if !host.bind(py).hasattr(name)? {
                return Err(PyValueError::new_err(format!("host object has no {name}")));
            }
        }
        let locals = TaskLocals::with_running_loop(py)?.copy_context(py)?;
        let opts: TurnOptions =
            options.map(|o| parse_request(o, "options")).transpose()?.unwrap_or_default();
        let host: Arc<dyn TurnHost> = Arc::new(PyTurnHost { obj: host, locals });
        Ok(Self { inner: Arc::new(Turn::new(thread.inner.clone(), lane, host, opts)) })
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
lk_agents_abi::register_host!(class PyBoundary);
lk_agents_abi::register_host!(class PyTurn);

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: an options dict read as the wrong spec (a str policy applied per channel, a
    // debounce map read as one value), or an unknown policy accepted.
    #[test]
    fn options_cross_as_a_dict() {
        use std::time::Duration;

        use lk_agents_storage::turn::{DebounceSpec, Policy, PolicySpec};

        Python::initialize();
        Python::attach(|py| {
            let d = py.eval(c"{}", None, None).unwrap();
            assert_eq!(
                parse_request::<TurnOptions>(&d, "options").unwrap(),
                TurnOptions::default()
            );

            let d = py
                .eval(
                    c"{'policy': 'interrupt', 'debounce': 0, 'wait_timeout': 1.5, 'state_default': {'n': 1}}",
                    None,
                    None,
                )
                .unwrap();
            let o: TurnOptions = parse_request(&d, "options").unwrap();
            assert_eq!(o.policy, PolicySpec::All(Policy::Interrupt));
            assert_eq!(o.debounce, DebounceSpec::All(Duration::ZERO));
            assert_eq!(o.wait_timeout, Some(Duration::from_millis(1500)));
            assert_eq!(o.state_default, Some(serde_json::json!({"n": 1})));

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

    // Catches: a handler exception losing its type name, or `CancelledError` written on the row
    // as a handler error instead of ending the run `canceled`.
    #[test]
    fn a_raising_handler_is_the_rows_error_and_a_cancel_is_a_cancel() {
        Python::initialize();
        Python::attach(|py| {
            let boom = PyRuntimeError::new_err("boom");
            let out = host_error(py, &boom).unwrap();
            assert_eq!(out.error.as_deref(), Some("RuntimeError: boom"));
            assert_eq!(error_text(py, &PyRuntimeError::new_err(())), "RuntimeError");
            assert!(matches!(
                host_error(py, &CancelledError::new_err("gone")),
                Err(HostError::Cancelled(m)) if m == "gone"
            ));
            assert!(matches!(
                host_error(py, &StorageUnavailable::new_err("lost")),
                Err(HostError::Storage(CoreError::Unavailable(m))) if m == "lost"
            ));
        });
    }
}
