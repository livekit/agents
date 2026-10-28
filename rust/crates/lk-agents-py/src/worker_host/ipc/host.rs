//! Process bootstrap callbacks and diagnostics.
//! Python host boundary conversions and callback diagnostics.

use super::super::callback::{self, Callback as CancellingHandle, Callback as HostHandle};
use super::ProcessLaunch;
use super::{
    Child, Closed, Extra, ExtraValue, Level, MemoryError, PoolEvent, PoolHost, ProcHost, PyJobRun,
    PyProc, PyProcHost, StartJob, ToChild, ToParent,
};
use lk_agents_abi::lock;
use lk_agents_abi::python::pyo3_async_runtimes;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::sync::Arc;
use std::time::Duration;

/// The shared ABI job record, without a second handwritten wire projection.
pub(super) fn fields_of_job<'py>(py: Python<'py>, job: &StartJob) -> PyResult<Bound<'py, PyAny>> {
    Ok(job.clone().into_pyobject(py)?.into_any())
}

/// `msg` on the `livekit.agents` logger, with `exc`'s traceback.
pub(super) fn py_log(level: Level, msg: &str, exc: Option<PyErr>) {
    py_log_extra(level, msg, exc, Vec::new());
}

/// A log record's `extra`.
pub(super) fn extra_dict(py: Python<'_>, extra: Extra) -> PyResult<Bound<'_, PyDict>> {
    let fields = PyDict::new(py);
    for (key, value) in extra {
        match value {
            ExtraValue::Int(v) => fields.set_item(key, v)?,
            ExtraValue::Float(v) => fields.set_item(key, v)?,
            ExtraValue::Str(v) => fields.set_item(key, v)?,
            ExtraValue::Bool(v) => fields.set_item(key, v)?,
        }
    }
    Ok(fields)
}

/// `msg` on the `livekit.agents` logger, with `exc`'s traceback and the `extra` fields.
pub(super) fn py_log_extra(level: Level, msg: &str, exc: Option<PyErr>, extra: Extra) {
    let msg = msg.to_owned();
    let logged = callback::call_now(move |py| {
        let logger = crate::host::construct(
            &py.import("logging")?.getattr("getLogger")?,
            ("livekit.agents",),
            None,
        )?;
        let kwargs = PyDict::new(py);
        if let Some(exc) = exc {
            kwargs.set_item("exc_info", exc.into_value(py))?;
        }
        if !extra.is_empty() {
            kwargs.set_item("extra", extra_dict(py, extra)?)?;
        }
        crate::host::construct(&logger.getattr("log")?, (level as i32, msg), Some(&kwargs))
            .map(Bound::unbind)
    });
    if let Err(error) = logged {
        Python::attach(|py| error.print(py));
    }
}

/// `CancelledError`, what a cancelled call raises.
pub(super) fn cancelled() -> PyErr {
    pyo3::exceptions::asyncio::CancelledError::new_err(())
}

pub(super) fn is_cancelled(e: &PyErr) -> bool {
    Python::attach(|py| e.is_instance_of::<pyo3::exceptions::asyncio::CancelledError>(py))
}

/// A callable's name, as the shutdown spans label it.
pub(super) fn callable_name(callable: &Bound<'_, PyAny>) -> String {
    ["__qualname__", "__name__"]
        .iter()
        .find_map(|attr| callable.getattr(*attr).ok().and_then(|n| n.extract::<String>().ok()))
        .unwrap_or_else(|| callable.repr().map(|r| r.to_string()).unwrap_or_default())
}

impl PyProcHost {
    fn emit(&self, level: Level, message: &str, extra: Extra, error: Option<PyErr>) {
        let Some(host) = lock(&self.object).as_ref().cloned() else {
            return;
        };
        queue_process_diagnostic(host, level, message, extra, error);
    }
}

fn queue_process_diagnostic(
    host: Arc<Py<PyAny>>,
    level: Level,
    message: &str,
    mut extra: Extra,
    error: Option<PyErr>,
) {
    let message = message.to_owned();
    crate::worker::diagnostics::submit_host(move |dropped| {
        if dropped > 0 {
            extra.push(("dropped_supervisor_logs", ExtraValue::Int(dropped as i64)));
        }
        let logged = callback::call_now(move |py| {
            let fields = extra_dict(py, extra)?;
            let error = error.map(|error| error.into_value(py));
            crate::host::construct(
                &host.bind(py).getattr("_core_log")?,
                (level as i32, message, fields, error),
                None,
            )
            .map(Bound::unbind)
        });
        if let Err(error) = logged {
            Python::attach(|py| error.print(py));
        }
    });
}

impl ProcHost for PyProcHost {
    type Error = PyErr;

    fn log(&self, level: Level, msg: &str, extra: Extra) {
        self.emit(level, msg, extra, None);
    }

    fn log_exception(&self, msg: &str, err: PyErr) {
        self.emit(Level::Error, msg, Vec::new(), Some(err));
    }

    fn is_alive(&self) -> bool {
        self.process.borrow().is_some()
    }

    fn request_stack_dump(&self) -> Result<bool, String> {
        let process = self.process.borrow();
        process.as_ref().map_or(Ok(false), |process| {
            process.request_stack_dump().map_err(|error| error.to_string())
        })
    }

    fn kill(&self) -> Result<(), String> {
        let process = self.process.borrow();
        process.as_ref().map_or(Ok(()), |process| process.kill().map_err(|error| error.to_string()))
    }

    fn sample_memory(&self) -> Result<(f64, String), MemoryError<PyErr>> {
        let pid = self
            .process
            .borrow()
            .as_ref()
            .map(|process| process.pid)
            .ok_or_else(|| MemoryError::Unavailable("process exited".into()))?;
        crate::worker::ipc::process::sample_memory(pid)
            .map_err(|error| MemoryError::Unavailable(error.to_string()))
    }

    fn incomplete_cleanup(&self, observation: crate::worker::ipc::supervisor::CleanupObservation) {
        self.emit(
            Level::Error,
            "process cleanup incomplete",
            vec![
                ("pid", ExtraValue::Int(i64::from(observation.pid))),
                ("exited", ExtraValue::Bool(observation.exited)),
                ("reaped", ExtraValue::Bool(observation.reaped)),
            ],
            None,
        );
    }

    fn signal_failed(&self, observation: crate::worker::ipc::supervisor::SignalFailure) {
        self.emit(
            Level::Error,
            "native process signal failed",
            vec![
                ("pid", ExtraValue::Int(i64::from(observation.pid))),
                ("cause", ExtraValue::Str(observation.cause)),
            ],
            None,
        );
    }

    fn release(&self) {
        // Py handles defer decref when the GIL is unavailable; native cleanup never waits for it.
        drop(lock(&self.object).take());
    }
}

impl Child {
    pub(super) async fn send(&self, msg: ToParent) -> Result<(), Closed> {
        self.client.send(msg).await
    }

    /// One message from the parent; `false` ends the serving.
    pub(super) async fn on_message(
        self: &Arc<Self>,
        host: &CancellingHandle,
        msg: ToChild,
    ) -> PyResult<bool> {
        match msg {
            // answered once the loop ran a callback: the pong measures the loop's responsiveness
            ToChild::Ping { sent_ms } => {
                let loop_ran = HostHandle::from_parts(
                    Python::attach(|py| host.object().clone_ref(py)),
                    host.task_locals().clone(),
                );
                if loop_ran.run_on_loop(|_| Ok(())).await.is_err() {
                    return Ok(false);
                }
                Ok(self.send(ToParent::Pong { sent_ms }).await.is_ok())
            }
            // the job starts once: its run ends with the parent's shutdown
            ToChild::StartJob { request_id, job } => {
                let accepted: PyResult<()> = async {
                    if lock(&self.run).is_some() {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            "a job is already running",
                        ));
                    }
                    let run = host
                        .await_call(move |h| {
                            crate::host::construct(
                                &h.getattr("_core_start_job")?,
                                (fields_of_job(h.py(), &job)?,),
                                None,
                            )
                        })
                        .await?;
                    let run: Arc<PyJobRun> = Python::attach(|py| -> PyResult<_> {
                        Ok(run.bind(py).cast::<crate::generated::worker::JobRun>()?.get().0.clone())
                    })?;
                    let _ = run.0.parent.set(Arc::clone(self));
                    *lock(&self.run) = Some(run);
                    Ok(())
                }
                .await;
                let error = accepted.as_ref().err().map(ToString::to_string);
                let sent = self.send(ToParent::JobAccepted { request_id, error }).await.is_ok();
                accepted?;
                Ok(sent)
            }
            // acked, then the process ends with its job (at once without one)
            ToChild::Shutdown { reason } => {
                if self.send(ToParent::ShutdownAck).await.is_err() {
                    return Ok(false);
                }
                let run = lock(&self.run).clone();
                let Some(run) = run else {
                    let _ = self.send(ToParent::ShuttingDown).await;
                    return Ok(false);
                };
                let reason =
                    if reason.is_empty() { "parent process shutdown".into() } else { reason };
                run.0.end(reason, false);
                Ok(true)
            }
            ToChild::DumpStacks => {
                host.await_call(|h| crate::host::read(h, "_core_dump_stacks")).await?;
                Ok(true)
            }
            ToChild::Hello(_) => Ok(true),
        }
    }
}

/// The Python `ProcPool`: it makes the executors (`_core_create`), reports a failed
/// initialization (`_core_init_failed`) and emits the events; the executors run their own steps.
pub(super) struct PyPoolHost(pub(super) CancellingHandle);

impl PyPoolHost {
    async fn step(&self, proc: &PyProc, method: &'static str) -> PyResult<()> {
        let proc = Python::attach(|py| self.0.child(proc.0.clone_ref(py)));
        proc.await_call(move |p| crate::host::read(p, method)).await.map(drop)
    }

    fn soon(
        &self,
        method: &'static str,
        args: impl for<'py> FnOnce(Python<'py>) -> PyResult<Bound<'py, PyTuple>> + Send + 'static,
    ) {
        if let Err(e) = self.0.call_soon(method, args) {
            Python::attach(|py| e.print(py));
        }
    }
}

#[async_trait::async_trait]
impl PoolHost for PyPoolHost {
    type Proc = PyProc;
    type Job = Arc<Py<PyAny>>;
    type Error = PyErr;

    async fn create(&self) -> PyResult<PyProc> {
        let pool = Python::attach(|py| self.0.object().clone_ref(py));
        let handle = Python::attach(|py| {
            HostHandle::from_parts(pool.clone_ref(py), self.0.task_locals().clone())
        });
        handle
            .run_on_loop(move |py| Ok(crate::host::read(pool.bind(py), "_core_create")?.unbind()))
            .await
            .map(|p| PyProc(Arc::new(p), Arc::new(std::sync::OnceLock::new())))
            .map_err(|e| Python::attach(|py| lk_agents_abi::python::raise_provider(py, e)))
    }

    async fn start(&self, proc: &PyProc) -> PyResult<()> {
        let host = Python::attach(|py| self.0.child(proc.0.clone_ref(py)));
        let result = host.await_call(|host| crate::host::read(host, "_core_start")).await?;
        let supervisor = Python::attach(|py| -> PyResult<_> {
            Ok(result
                .bind(py)
                .cast::<crate::generated::worker::ProcSupervisor>()?
                .get()
                .0
                .sup
                .clone())
        })?;
        proc.1.set(supervisor).map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("process start already reported")
        })
    }

    async fn initialize(&self, proc: &PyProc) -> PyResult<()> {
        self.step(proc, "initialize").await
    }

    /// The dispatch timeline: a warm process is now handling the job (`launched_at`).
    async fn launch(&self, proc: &PyProc, job: &Self::Job) -> PyResult<()> {
        let (proc, job) = Python::attach(|py| -> PyResult<_> {
            let now = lk_agents_abi::unix_now();
            job.bind(py).setattr("launched_at", now)?;
            Ok((self.0.child(proc.0.clone_ref(py)), job.clone_ref(py)))
        })?;
        proc.await_call(move |p| crate::host::construct(&p.getattr("launch_job")?, (job,), None))
            .await
            .map(drop)
    }

    async fn close(&self, proc: &PyProc) -> PyResult<()> {
        self.step(proc, "aclose").await
    }

    fn close_failed(&self, proc: &PyProc, error: PyErr) {
        queue_process_diagnostic(
            proc.0.clone(),
            Level::Error,
            "error closing process",
            Vec::new(),
            Some(error),
        );
    }

    fn cleanup_incomplete(&self, observation: crate::worker::ipc::pool::PoolCleanupObservation) {
        crate::worker::diagnostics::emit(
            log::Level::Error,
            format!("process pool cleanup incomplete: {observation:?}"),
        );
    }

    async fn wait_exit(&self, proc: &PyProc) -> PyResult<()> {
        let supervisor = proc.1.get().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("ready process has no native supervisor")
        })?;
        if supervisor.wait_exit().await {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "ready process finished without an OS exit observation",
            ))
        }
    }

    fn exit_failed(&self, proc: &PyProc, error: PyErr) {
        queue_process_diagnostic(
            proc.0.clone(),
            Level::Error,
            "error observing process exit",
            Vec::new(),
            Some(error),
        );
    }

    async fn join(&self, proc: &PyProc) -> PyResult<crate::worker::ipc::pool::PoolJoinOutcome> {
        if let Some(supervisor) = proc.1.get() {
            return Ok(if supervisor.wait_cleaned().await {
                crate::worker::ipc::pool::PoolJoinOutcome::Reaped
            } else {
                crate::worker::ipc::pool::PoolJoinOutcome::NotLaunched
            });
        }
        let proc = Python::attach(|py| self.0.child(proc.0.clone_ref(py)));
        let result = proc.await_call(|p| crate::host::read(p, "_core_join")).await?;
        let reaped = Python::attach(|py| result.bind(py).extract::<bool>())?;
        Ok(if reaped {
            crate::worker::ipc::pool::PoolJoinOutcome::Reaped
        } else {
            crate::worker::ipc::pool::PoolJoinOutcome::NotLaunched
        })
    }

    fn join_failed(&self, proc: &PyProc, error: PyErr) {
        queue_process_diagnostic(
            proc.0.clone(),
            Level::Error,
            "error joining process",
            Vec::new(),
            Some(error),
        );
    }

    fn emit(&self, event: PoolEvent, proc: &PyProc) {
        let proc = proc.0.clone();
        self.soon("emit", move |py| {
            PyTuple::new(py, [event.as_str().into_pyobject(py)?.into_any(), proc.bind(py).clone()])
        });
    }

    fn init_failed(&self, proc: &PyProc, error: PyErr) {
        queue_process_diagnostic(
            proc.0.clone(),
            Level::Error,
            "error initializing process",
            Vec::new(),
            Some(error),
        );
    }
}

/// Takes ownership of a connected socket (an fd of `ipc_pair`'s, or `socket.detach()`). Needs a
/// runtime context.
pub(super) fn socket(fd: i64) -> PyResult<Socket> {
    #[cfg(unix)]
    let std = {
        use std::os::fd::FromRawFd;
        let fd = i32::try_from(fd).map_err(|_| PyValueError::new_err("bad socket fd"))?;
        // SAFETY: the caller hands the descriptor's ownership over.
        unsafe { StdSocket::from_raw_fd(fd) }
    };
    #[cfg(windows)]
    let std = {
        use std::os::windows::io::FromRawSocket;
        // SAFETY: the caller hands the socket's ownership over.
        unsafe { StdSocket::from_raw_socket(fd as u64) }
    };
    std.set_nonblocking(true)?;
    Ok(Socket::from_std(std)?)
}

#[cfg(unix)]
pub(super) type Socket = tokio::net::UnixStream;
#[cfg(windows)]
pub(super) type Socket = tokio::net::TcpStream;
#[cfg(unix)]
pub(super) type StdSocket = std::os::unix::net::UnixStream;
#[cfg(windows)]
pub(super) type StdSocket = std::net::TcpStream;

/// The socket to the other process closed.
pub(super) fn closed(_: Closed) -> PyErr {
    pyo3::exceptions::PyConnectionError::new_err("the IPC socket is closed")
}

pub(super) fn run_process_owner<'py>(
    py: Python<'py>,
    this: &super::PyProcSupervisor,
    joined: Py<PyAny>,
    host: Py<PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let host = Arc::new(CancellingHandle::new(py, host)?);
    let sup = this.sup.clone();
    let process = this.process.clone();
    let ran = this.ran.clone();
    let finished = ran.subscribe();
    let initialized = this.initialized.subscribe();
    let launch = this.launch.clone();
    let mut launched = launch.subscribe();
    let launch_host = host.clone();
    let launch_sup = sup.clone();
    pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
        match launch_host.await_call(move |h| Ok(joined.bind(h.py()).clone())).await {
            Ok(_) => {
                let _ = launch_host.await_call(|h| crate::host::read(h, "_core_launched")).await;
            }
            Err(error) => {
                let cause = error.to_string();
                let mut before_receipt = false;
                launch.send_if_modified(|state| {
                    if matches!(state, ProcessLaunch::Pending) {
                        *state = ProcessLaunch::Failed(cause.clone());
                        before_receipt = true;
                        true
                    } else {
                        false
                    }
                });
                if !before_receipt {
                    launch_sup.launch_failed(cause.clone()).await;
                }
                let _ = launch_host
                    .await_call(move |h| {
                        crate::host::construct(&h.getattr("_core_launch_failed")?, (cause,), None)
                    })
                    .await;
            }
        }
    });
    let launch = this.launch.clone();
    pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
        let _ = launched.wait_for(|state| !matches!(state, ProcessLaunch::Pending)).await;
        let mut owned = ProcessLaunch::Claimed;
        launch.send_modify(|state| std::mem::swap(state, &mut owned));
        let mut resource = match owned {
            ProcessLaunch::Started(resource) => resource,
            ProcessLaunch::Failed(cause) => {
                sup.not_launched(cause).await;
                ran.send_replace(Some(false));
                return;
            }
            _ => return,
        };
        if resource.control().is_none() {
            sup.launch_failed("forkserver bootstrap incomplete before PID receipt".into()).await;
        }
        let mut identity_announced = false;
        let mut supervising = false;
        let mut observed_exit = false;
        let mut reported = false;
        loop {
            if !identity_announced {
                match resource.poll_identity() {
                    Ok(true) => {
                        let control = resource.control().expect("identity observed").clone();
                        process.send_replace(Some(control.clone()));
                        sup.launched(control.pid).await;
                        identity_announced = true;
                    }
                    Ok(false) => {
                        tokio::time::sleep(Duration::from_millis(10)).await;
                        continue;
                    }
                    Err(error) => {
                        if !reported {
                            reported = true;
                            sup.launch_failed(format!(
                                "forkserver process identity receipt failed: {error}"
                            ))
                            .await;
                        }
                        tokio::time::sleep(Duration::from_millis(10)).await;
                        continue;
                    }
                }
            }
            if !supervising && initialized.borrow().is_some() {
                supervising = true;
                sup.supervise().await;
            }
            let polled = if observed_exit {
                resource.poll_reaped().map(|code| (true, code))
            } else {
                resource.poll_exit().map(|exited| (exited, None))
            };
            match polled {
                Ok((true, _)) if !observed_exit => {
                    observed_exit = true;
                    process.send_replace(None);
                    sup.exited(resource.exitcode()).await;
                }
                Ok((_, Some(code))) => {
                    let pid = resource.control().expect("identity observed").pid;
                    drop(resource);
                    sup.reaped(code).await;
                    let completion = sup.completion(i32::try_from(code).ok());
                    let _ = host
                        .await_call(move |h| {
                            crate::host::construct(
                                &h.getattr("_core_completed")?,
                                (completion, pid),
                                None,
                            )
                        })
                        .await;
                    ran.send_replace(Some(true));
                    return;
                }
                Err(error) if !reported => {
                    reported = true;
                    crate::worker::diagnostics::emit(
                        log::Level::Error,
                        format!(
                            "native process {} observation failed: {error}",
                            resource.control().expect("identity observed").pid
                        ),
                    );
                }
                _ => {}
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });
    wait_process_completion(py, finished)
}

pub(super) fn wait_process_completion<'py>(
    py: Python<'py>,
    mut finished: tokio::sync::watch::Receiver<Option<bool>>,
) -> PyResult<Bound<'py, PyAny>> {
    crate::host::future_into_py(py, async move {
        let outcome = finished
            .wait_for(Option::is_some)
            .await
            .map(|done| done.unwrap_or(false))
            .unwrap_or(false);
        Ok(outcome)
    })
}
