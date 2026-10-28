//! Runtime-owned host adapter, compiled by lk-agents-runtime through #[path].
//! `_core.ipc_pair`, `_core.ProcSupervisor`, `_core.IpcClient`, `_core.ProcPool` and
//! `_core.JobRun`: the job processes as `ipc/__init__.py` (the worker's side) and `ipc/child.py`
//! (a process's) drive them.
//!
//! `ProcSupervisor(fd, host, ...)` takes over the parent's socket. `host` is the `JobExecutor`:
//! Rust owns launch completion, native signals, exit observation and the start-method reaping
//! receipt. The host supplies bootstrap and receives queued diagnostics and completion callbacks.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use tokio::sync::watch;

use lk_agents_abi::python::{CancellingHandle, HostBridge, check_pid, pyo3_async_runtimes};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[path = "python/host.rs"]
mod host;
use host::*;

use super::pool::{LaunchError, PoolEvent, PoolHost, PoolOptions, ProcPool};
use super::process::{ProcessControl, ProcessLaunch, ProcessResource, StartMethod};
use super::{
    Closed, Extra, ExtraValue, HEALTH_CHECK_GRACE, Hello, InitError, IpcClient, Level, MemoryError,
    PROTOCOL_VERSION, ProcHost, ProcSupervisor, StartJob, SupervisorOptions, ToChild, ToParent,
    lock, secs,
};

/// The `SupervisedProc` as the supervisor's host, until the process exited.

struct PyProcHost {
    object: Mutex<Option<Arc<Py<PyAny>>>>,
    process: watch::Receiver<Option<Arc<ProcessControl>>>,
}

/// The control socket of one process: the parent's end and the child's, as descriptors the caller
/// owns (the parent's goes to `ProcSupervisor`, the child's to the process, then `IpcClient`).
/// Both are closed on exec.
#[pyfunction]
fn ipc_pair() -> PyResult<(i64, i64)> {
    #[cfg(unix)]
    {
        use std::os::fd::IntoRawFd;
        let (parent, child) = StdSocket::pair()?;
        Ok((parent.into_raw_fd().into(), child.into_raw_fd().into()))
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::IntoRawSocket;
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0))?;
        let child = StdSocket::connect(listener.local_addr()?)?;
        let (parent, _) = listener.accept()?;
        Ok((parent.into_raw_socket() as i64, child.into_raw_socket() as i64))
    }
}

lk_agents_abi::register_host!(fn ipc_pair);

/// The parent's side of one job process (`ipc/__init__.py`): its initialization, supervision
/// and close, from the spawn the host made to the exit.
#[pyclass(name = "ProcSupervisor", module = "livekit.agents._core", frozen)]
pub struct PyProcSupervisor {
    launch: watch::Sender<ProcessLaunch>,
    sup: ProcSupervisor,
    /// The child's health timeout, from the supervision's ping settings.
    health_timeout_ms: Option<u64>,
    /// Whether the process initialized, once that settled (answered, failed, timed out, killed).
    initialized: watch::Sender<Option<bool>>,
    /// Completed launch: false without an OS child, true only with its reaping receipt.
    ran: watch::Sender<Option<bool>>,
    killable: bool,
    process: watch::Sender<Option<Arc<ProcessControl>>>,
}

impl PyProcSupervisor {
    fn settle(&self, initialized: bool) {
        self.initialized.send_if_modified(|v| v.is_none() && v.replace(initialized).is_none());
    }
}

#[pymethods]
impl PyProcSupervisor {
    #[new]
    #[pyo3(signature = (
        fd, host, *, pid, initialize_timeout, close_timeout, memory_warn_mb,
        memory_limit_mb, ping_interval, ping_timeout, high_ping_threshold, killable = true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        fd: i64,
        host: Py<PyAny>,
        pid: u32,
        initialize_timeout: f64,
        close_timeout: f64,
        memory_warn_mb: f64,
        memory_limit_mb: f64,
        ping_interval: f64,
        ping_timeout: f64,
        high_ping_threshold: f64,
        killable: bool,
    ) -> PyResult<Self> {
        check_pid()?;
        // Thread runners and direct supervisors need diagnostics before any OS launch.
        crate::worker::diagnostics::initialize().map_err(PyRuntimeError::new_err)?;
        let _rt = pyo3_async_runtimes::tokio::get_runtime().enter();
        let opts = SupervisorOptions {
            pid,
            initialize_timeout,
            close_timeout,
            memory_warn_mb,
            memory_limit_mb,
            ping_interval,
            ping_timeout,
            high_ping_threshold,
            killable,
        };
        // the child calls itself orphaned when the parent goes quiet past its pong deadline (+10 s);
        // a job thread's parent has none
        let health_timeout_ms = (ping_interval > 0.0 && ping_timeout > 0.0)
            .then(|| secs(ping_timeout + HEALTH_CHECK_GRACE).as_millis() as u64);
        let (process, native) = watch::channel(None);
        Ok(Self {
            launch: watch::channel(ProcessLaunch::Pending).0,
            sup: ProcSupervisor::new(
                socket(fd)?,
                PyProcHost { object: Mutex::new(Some(Arc::new(host))), process: native },
                opts,
                crate::process::workers()
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?,
            ),
            health_timeout_ms,
            initialized: watch::channel(None).0,
            ran: watch::channel(None).0,
            killable,
            process,
        })
    }

    /// Sends the process its `Hello` (the loop's debug mode, the HTTP proxy, the health timeout)
    /// and returns the seconds it took to be ready; raises `TimeoutError` when it was not in time
    /// (a process is then killed).
    #[pyo3(signature = (*, asyncio_debug, http_proxy))]
    fn initialize<'py>(
        slf: Bound<'py, Self>,
        asyncio_debug: bool,
        http_proxy: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let this = slf.clone().unbind();
        let sup = slf.get().sup.clone();
        let hello = Hello {
            version: PROTOCOL_VERSION,
            asyncio_debug,
            http_proxy: http_proxy.filter(|p| !p.is_empty()),
            health_timeout_ms: this.get().health_timeout_ms,
        };
        let runner = if this.get().killable { "process" } else { "runner" };
        pyo3_async_runtimes::tokio::future_into_py(slf.py(), async move {
            let outcome = sup.initialize(hello).await;
            this.get().settle(matches!(outcome, Ok(Some(_))));
            match outcome {
                Ok(Some(elapsed)) => Ok(elapsed.as_secs_f64()),
                Ok(None) => Err(Python::attach(|py| -> PyResult<PyErr> {
                    let timeout = py.import("asyncio")?.getattr("TimeoutError")?;
                    Ok(PyErr::from_value(
                        timeout.call1((format!("{runner} initialization timed out"),))?,
                    ))
                })
                .unwrap_or_else(|e| e)),
                Err(InitError::Closed) => Err(closed(Closed)),
                Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
            }
        })
    }

    /// Whether the process answered its initialization.
    #[getter]
    fn initialized(&self) -> bool {
        *self.initialized.borrow() == Some(true)
    }

    /// Called on the launch worker immediately after multiprocessing creates the child.
    fn _core_launched_process(
        &self,
        pid: u32,
        sentinel: i64,
        method: &str,
        owned: Vec<i64>,
    ) -> PyResult<()> {
        if !matches!(*self.launch.borrow(), ProcessLaunch::Pending) {
            return Err(PyRuntimeError::new_err("launch already reported"));
        }
        let resource = ProcessResource::new(pid, sentinel, StartMethod::parse(method)?, owned)?;
        self.launch.send_replace(ProcessLaunch::Started(resource));
        Ok(())
    }

    /// Own launch/bootstrap before invoking multiprocessing; native exit and reaping continue
    /// even if the Python caller cancels this future or its event loop stops responding.
    fn run<'py>(
        slf: Bound<'py, Self>,
        joined: Py<PyAny>,
        host: Py<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        run_process_owner(slf, joined, host)
    }

    fn _core_wait_exit<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let sup = self.sup.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move { Ok(sup.wait_exit().await) })
    }

    fn _core_join<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        wait_process_completion(py, self.ran.subscribe())
    }

    /// A graceful close: an initialized process is asked to shut down (killed when late), any
    /// other is killed; returns once it exited.
    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let initialized = self.initialized();
        self.settle(false);
        let sup = self.sup.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            if initialized {
                sup.shutdown().await;
            } else {
                sup.kill().await;
                sup.wait_cleanup().await;
            }
            Ok(())
        })
    }

    /// The kill sequence; returns once the process exited.
    fn kill<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.settle(false);
        let sup = self.sup.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            sup.kill().await;
            sup.wait_cleanup().await;
            Ok(())
        })
    }

    /// Hands the process its validated ABI job record.
    fn start_job<'py>(
        &self,
        py: Python<'py>,
        fields: lk_agents_abi::worker::job::StartJobFields,
    ) -> PyResult<Bound<'py, PyAny>> {
        let job = fields;
        let sup = self.sup.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            sup.start_job(job)
                .await
                .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
        })
    }

    #[getter]
    fn killed(&self) -> bool {
        self.sup.killed()
    }
}

lk_agents_abi::register_host!(class PyProcSupervisor);

// ----------------------------------------------------------------------------------- client

/// The child's side of the parent's socket (`ipc/child.py`): the initialize handshake, then it
/// serves the parent's messages (`serve`), answers the pings itself once the child's loop has run
/// a callback, starts the one job and acks the parent's shutdown.
#[pyclass(name = "IpcClient", module = "livekit.agents._core", frozen)]
pub struct PyIpcClient(Arc<Child>);

struct Child {
    client: IpcClient,
    /// `stop()`: `serve` returns.
    stop: watch::Sender<bool>,

    /// The job the parent started, once it did.
    run: Mutex<Option<Py<PyJobRun>>>,
    asyncio_debug: bool,
}

#[pymethods]
impl PyIpcClient {
    /// The handshake on the parent's socket `fd`, blocking: the parent's `Hello`, then
    /// `initialize(http_proxy)` and `Ready` (with the error it raised, re-raised here). Once
    /// initialized, the child calls itself orphaned when the parent goes quiet past the health
    /// timeout the `Hello` gave.
    #[new]
    #[pyo3(signature = (fd, initialize, *, process = false))]
    fn new(py: Python<'_>, fd: i64, initialize: Bound<'_, PyAny>, process: bool) -> PyResult<Self> {
        check_pid()?;
        match py.import("asyncio")?.call_method0("get_running_loop") {
            Ok(_) => {
                return Err(PyRuntimeError::new_err(
                    "IpcClient handshake must precede the event loop",
                ));
            }
            Err(error) if error.is_instance_of::<PyRuntimeError>(py) => {}
            Err(error) => return Err(error),
        }
        let rt = pyo3_async_runtimes::tokio::get_runtime();
        let sock = {
            let _rt = rt.enter();
            socket(fd)?
        };
        let failed =
            |e: super::WireError| PyRuntimeError::new_err(format!("failed to initialize: {e}"));
        let (client, hello) = py
            .detach(|| {
                rt.block_on(IpcClient::connect_with_health_action(sock, move || {
                    if process {
                        super::process::terminate_current_process();
                    }
                }))
            })
            .map_err(failed)?;
        let initialized = initialize.call1((hello.http_proxy.clone(),)).and_then(|value| {
            // The host callback warms Python SDKs; only runtime can construct this native client.
            // Both complete under this bootstrap barrier before Ready or a job loop.
            py.detach(crate::http::GatewayHttp::shared).map_err(PyRuntimeError::new_err)?;
            Ok(value)
        });
        let error = initialized.as_ref().err().map(|e| e.value(py).to_string());
        let ready = ToParent::Ready { version: PROTOCOL_VERSION, error };
        let sent = py.detach(|| rt.block_on(client.send(ready)));
        if let Err(e) = initialized {
            py.detach(|| rt.block_on(client.close()));
            return Err(e);
        }
        sent.map_err(closed)?;
        Ok(Self(Arc::new(Child {
            client,
            stop: watch::channel(false).0,
            run: Mutex::new(None),
            asyncio_debug: hello.asyncio_debug,
        })))
    }

    /// The parent's loop runs in debug mode: so does the child's.
    #[getter]
    fn asyncio_debug(&self) -> bool {
        self.0.asyncio_debug
    }

    /// Serve the parent's messages until the channel closes, the shutdown ends the process, or
    /// `stop()`. `host` (on the running loop) runs `_core_start_job(fields)` (returning the job's
    /// `JobRun`) and `_core_dump_stacks()`, each a coroutine.
    fn serve<'py>(&self, py: Python<'py>, host: Py<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let host = CancellingHandle::new(py, host)?;
        let child = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut stop = child.stop.subscribe();
            let served = async {
                loop {
                    let Some(msg) = child.client.recv().await else {
                        return child
                            .client
                            .wait_reader()
                            .await
                            .map_err(|error| PyRuntimeError::new_err(error.to_string()));
                    };
                    if !child.on_message(&host, msg).await? {
                        return Ok(());
                    }
                }
            };
            let outcome = tokio::select! {
                r = served => r,
                _ = stop.wait_for(|stopped| *stopped) => Ok(()),
            };
            outcome
        })
    }

    /// `serve` returns.
    fn stop(&self) {
        self.0.stop.send_replace(true);
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let child = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            child.client.close().await;
            Ok(())
        })
    }
}

lk_agents_abi::register_host!(class PyIpcClient);

// ------------------------------------------------------------------------------------- pool

/// An executor, compared by identity.
#[derive(Clone)]
pub struct PyProc(Arc<Py<PyAny>>, Arc<std::sync::OnceLock<ProcSupervisor>>);

impl PartialEq for PyProc {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ptr() == other.0.as_ptr()
    }
}

/// `ipc/proc_pool.py`'s `ProcPool`: the warm processes and the job launches.
#[pyclass(name = "ProcPool", module = "livekit.agents._core", frozen)]
pub struct PyProcPool(ProcPool<PyPoolHost>);

#[pymethods]
impl PyProcPool {
    /// `host` is the Python `ProcPool`; must be called inside its running loop.
    #[new]
    #[pyo3(signature = (host, *, num_idle_processes, initialize_timeout, close_timeout))]
    fn new(
        py: Python<'_>,
        host: Py<PyAny>,
        num_idle_processes: usize,
        initialize_timeout: f64,
        close_timeout: f64,
    ) -> PyResult<Self> {
        check_pid()?;
        let opts = PoolOptions {
            num_idle_processes,
            initialize_timeout: super::secs(initialize_timeout),
            close_timeout: super::secs(close_timeout),
        };
        Ok(Self(ProcPool::new(PyPoolHost(CancellingHandle::new(py, host)?), opts)))
    }

    #[getter]
    fn processes(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.0.processes().iter().map(|p| p.0.clone_ref(py)).collect()
    }

    #[getter]
    fn jobs_waiting(&self) -> usize {
        self.0.jobs_waiting()
    }

    #[getter]
    fn target_idle_processes(&self) -> usize {
        self.0.target_idle()
    }

    #[setter]
    fn set_target_idle_processes(&self, n: usize) {
        self.0.set_target_idle(n);
    }

    fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let pool = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            pool.start().await.map_err(|error| PyRuntimeError::new_err(error.to_string()))
        })
    }

    fn launch_job<'py>(&self, py: Python<'py>, info: Py<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let job_id: String = info.bind(py).getattr("job")?.getattr("id")?.extract()?;
        let pool = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            pool.launch_job(&job_id, Arc::new(info)).await.map_err(|e| match e {
                LaunchError::Admission(reason) => PyRuntimeError::new_err(reason.to_string()),
                LaunchError::AcquireExhausted { attempts } => PyRuntimeError::new_err(format!(
                    "no process became available after {attempts} attempts"
                )),
                LaunchError::NoProcess => {
                    PyRuntimeError::new_err("no process became available after 3 attempts")
                }
                LaunchError::Launch(e) => e,
            })
        })
    }

    /// Waits until every process a job was launched on exited (the drain).
    fn join_jobs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let pool = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            pool.join_jobs().await;
            Ok(())
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let pool = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            pool.aclose().await;
            Ok(())
        })
    }
}

lk_agents_abi::register_host!(class PyProcPool);

/// How long the entrypoint may take to return once the job shuts down, before it is cancelled.
const ENTRYPOINT_EXIT_TIMEOUT: Duration = Duration::from_secs(15);

/// A job's dispatch timeline, for its root span: one event per known stage instant (unix
/// nanoseconds) and the seconds between adjacent stages as attributes (they sum to the dispatch
/// latency). A zero instant is unknown (a simulation, the console, a resumed job) and skipped.
/// `server_started_at` is the server's own start (`JobState.started_at`, unix nanoseconds, or
/// milliseconds or seconds should that change).
#[pyfunction]
#[allow(clippy::type_complexity)]
fn job_timeline(
    received_at: f64,
    accepted_at: f64,
    assigned_at: f64,
    launched_at: f64,
    entrypoint_started_at: f64,
    server_started_at: i64,
) -> (Vec<(&'static str, i64)>, Vec<(&'static str, f64)>) {
    let ns = |ts: f64| (ts * 1e9) as i64;
    let stages = [
        ("job_received", received_at),
        ("job_accepted", accepted_at),
        ("job_assigned", assigned_at),
        ("process_assigned", launched_at),
        ("entrypoint_started", entrypoint_started_at),
    ];
    let mut events: Vec<_> =
        stages.iter().filter(|(_, ts)| *ts != 0.0).map(|(name, ts)| (*name, ns(*ts))).collect();
    let gaps = [
        ("lk.job.accept_latency", received_at, accepted_at),
        ("lk.job.assignment_latency", accepted_at, assigned_at),
        ("lk.job.launch_latency", assigned_at, launched_at),
        ("lk.job.entrypoint_latency", launched_at, entrypoint_started_at),
        ("lk.job.dispatch_latency", received_at, entrypoint_started_at),
    ];
    let attributes = gaps
        .iter()
        .filter(|(_, start, end)| *start != 0.0 && *end != 0.0)
        .map(|(attr, start, end)| (*attr, (end - start).max(0.0)))
        .collect();
    if server_started_at > 0 {
        let v = server_started_at as f64;
        let started = if v > 1e17 {
            v / 1e9
        } else if v > 1e11 {
            v / 1e3
        } else {
            v
        };
        events.push(("job_started_on_server", ns(started)));
    }
    (events, attributes)
}

lk_agents_abi::register_host!(fn job_timeline);

/// The default bound on `VoiceSession.aclose()` during the shutdown, a guardrail against close
/// paths that hang: past it the rest of the shutdown (on_session_end, ShuttingDown,
/// room.disconnect, the shutdown callbacks) runs anyway.
const SESSION_ACLOSE_TIMEOUT: f64 = 60.0;

/// The entrypoint has this long to connect the room before a warning.
const CONNECT_WARNING_DELAY: Duration = Duration::from_secs(10);

use lk_agents_abi::ATTRIBUTE_SIMULATOR;

#[derive(Default)]
struct JobRunState {
    /// The first reason the job ends with, and whether the user asked for it.
    shutdown: Option<(String, bool)>,
    connected: bool,
    user_shutdown: bool,
    /// `add_participant_entrypoint`'s, in order, each with the participant kinds it runs for.
    entrypoints: Vec<(Py<PyAny>, Vec<i32>)>,

    /// The participant entrypoints running, by participant and entrypoint, each its run's number.
    participant_runs: HashMap<(String, usize), u64>,
    runs: u64,
    /// The API calls (`delete_room`, ...) still running.
    tasks: Vec<tokio::task::JoinHandle<()>>,
    /// Whether `init_recording` ran, and the redaction it resolved (the job's until then).
    recording_initialized: bool,
    redaction: bool,
    /// The job's primary session: the first started for it.
    primary_session: Option<Py<PyAny>>,
    /// `ctx.add_shutdown_callback`'s, in order.
    shutdown_callbacks: Vec<ShutdownCallback>,
}

/// A shutdown callback: whether it takes the reason, whether livekit-agents registered it (no
/// span of its own) and its name (its span's `lk.callback.name`).
struct ShutdownCallback {
    callback: Py<PyAny>,
    takes_reason: bool,
    framework: bool,
    name: String,
}

/// A job's run, for `JobContext` and `job_proc_lazy_main.py`'s `_JobProc`: what ends the job (the
/// room, the user, the parent, a crash, the simulator leaving; the first wins), the entrypoint's
/// supervision, the order and bounds of the shutdown (whose steps the host, `_JobProc`, runs and
/// opens the spans of), the participant entrypoints, the room connected once, the API calls in
/// flight and the recording's setup.
#[pyclass(name = "JobRun", module = "livekit.agents._core", frozen)]
pub struct PyJobRun(Arc<JobRunInner>);

struct JobRunInner {
    host: std::sync::OnceLock<CancellingHandle>,
    /// The parent the job process reports its end to (none for a job run outside one).
    parent: std::sync::OnceLock<Arc<Child>>,
    fake: bool,
    session_end_timeout: f64,
    /// The name of the door's `on_session_end`, when it has one.
    session_end: Option<String>,
    /// The bound on `VoiceSession.aclose()` during the shutdown.
    session_aclose_timeout: f64,
    /// The job's LiveKit Cloud observability endpoint, when it has one.
    observability_url: Option<String>,
    /// The server asked for the job's recording.
    recording: bool,
    state: Mutex<JobRunState>,
    ended: watch::Sender<bool>,

    /// Held while the room connects, so a second `connect()` waits for the first.
    connecting: tokio::sync::Mutex<()>,
    /// Set once the job ends: the API calls still running are cancelled.
    tasks_closed: watch::Sender<bool>,
}

impl JobRunInner {
    fn host(&self) -> &CancellingHandle {
        self.host.get().expect("JobRun.run() sets the host first")
    }

    fn end(&self, reason: String, user_initiated: bool) {
        let mut st = lock(&self.state);
        st.user_shutdown |= user_initiated;
        if st.shutdown.is_none() {
            st.shutdown = Some((reason, user_initiated));
            self.ended.send_replace(true);
        }
    }

    async fn call(
        &self,
        f: impl for<'py> FnOnce(&Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        self.host().await_call(f).await
    }

    fn log(&self, level: Level, msg: &str, exc: Option<PyErr>) {
        py_log(level, msg, exc);
    }

    /// The entrypoints of `kind` start for the participant, each in the caller's context; one
    /// still running for an earlier participant of the same identity is warned about.
    fn participant_joined(
        self: &Arc<Self>,
        py: Python<'_>,
        ctx: &Py<PyAny>,
        participant: &Py<PyAny>,
        identity: &str,
        kind: i32,
    ) -> PyResult<()> {
        let mut starts = Vec::new();
        let mut overlapping = false;
        {
            let mut st = lock(&self.state);
            let matching: Vec<(usize, Py<PyAny>)> = st
                .entrypoints
                .iter()
                .enumerate()
                .filter(|(_, (_, kinds))| kinds.contains(&kind))
                .map(|(i, (entry, _))| (i, entry.clone_ref(py)))
                .collect();
            for (index, entry) in matching {
                let key = (identity.to_owned(), index);
                st.runs += 1;
                let run = st.runs;
                overlapping |= st.participant_runs.insert(key.clone(), run).is_some();
                starts.push((key, run, entry));
            }
        }
        if overlapping {
            self.log(Level::Warning, &format!("a participant has joined before a prior participant task matching the same identity has finished: '{identity}'"), None);
        }
        for (key, run, entry) in starts {
            let name: String =
                entry.bind(py).getattr("__name__").and_then(|n| n.extract()).unwrap_or_default();
            let handle = CancellingHandle::new(py, entry)?;
            let (ctx, participant) = (ctx.clone_ref(py), participant.clone_ref(py));
            let inner = Arc::clone(self);
            pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
                let r = handle.await_call(|entry| entry.call1((ctx, participant))).await;
                {
                    let mut st = lock(&inner.state);
                    if st.participant_runs.get(&key) == Some(&run) {
                        st.participant_runs.remove(&key);
                    }
                }
                if let Err(e) = r.map(drop) {
                    if !is_cancelled(&e) {
                        let msg = format!("error in participant entrypoint {name} for '{}'", key.0);
                        inner.log(Level::Error, &msg, Some(e));
                    }
                }
            });
        }
        Ok(())
    }

    /// A span of the shutdown's, named `name`, labelled with the callback it runs.
    fn span(&self, name: &str, callback: Option<&str>) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            Ok(self.host().object().bind(py).call_method1("_core_span", (name, callback))?.unbind())
        })
    }

    fn span_end(&self, span: Py<PyAny>, error: Option<PyErr>, timed_out: bool) {
        let r = Python::attach(|py| -> PyResult<()> {
            let error = error.map(|e| e.into_value(py).into_any());
            self.host()
                .object()
                .bind(py)
                .call_method1("_core_span_end", (span, error, timed_out))?;
            Ok(())
        });
        if let Err(e) = r {
            Python::attach(|py| e.print(py));
        }
    }

    /// One step run inside a span of its own: `error_msg` is logged with a failure.
    async fn spanned(
        &self,
        name: &str,
        callback: Option<&str>,
        method: &'static str,
        timeout: Option<f64>,
        error_msg: &str,
    ) {
        let span = match self.span(name, callback) {
            Ok(span) => span,
            Err(e) => return Python::attach(|py| e.print(py)),
        };
        let run = self.call(|h| h.call_method1(method, (span.clone_ref(h.py()),)));
        let outcome = match timeout {
            Some(t) => tokio::time::timeout(secs(t), run).await.map_err(|_| ()),
            None => Ok(run.await),
        };
        match outcome {
            Ok(Ok(_)) => self.span_end(span, None, false),
            Ok(Err(e)) => {
                let shown = Python::attach(|py| e.clone_ref(py));
                self.span_end(span, Some(e), false);
                self.log(Level::Error, error_msg, Some(shown));
            }
            Err(()) => {
                self.span_end(span, None, true);
                let msg = format!(
                    "on_session_end timed out after {}s",
                    timeout.unwrap_or_default() as i64
                );
                self.log(Level::Error, &msg, None);
            }
        }
    }

    #[allow(clippy::disallowed_macros, clippy::disallowed_methods, reason = "S0: move to runtime")]
    async fn run(self: Arc<Self>) -> PyResult<()> {
        let entry = self.call(|h| h.call_method0("_core_start_entrypoint")).await?;
        let wait_entry = {
            let entry = Python::attach(|py| entry.clone_ref(py));
            self.call(move |h| h.call_method1("_core_wait", (entry,)))
        };
        tokio::pin!(wait_entry);
        let warn_at = tokio::time::sleep(CONNECT_WARNING_DELAY);
        tokio::pin!(warn_at);
        let mut ended = self.ended.subscribe();
        let (mut entry_done, mut warned) = (false, self.fake);
        loop {
            tokio::select! {
                r = &mut wait_entry, if !entry_done => {
                    entry_done = true;
                    match r {
                        Err(e) => {
                            self.log(Level::Error, "unhandled exception while running the job task", Some(e));
                            // a crash before ctx.connect() resolves nothing else: the shutdown
                            // still runs, so the cleanup can flush the crash logs
                            self.end("job crashed".into(), false);
                        }
                        Ok(_) => {
                            let st = lock(&self.state);
                            if !st.connected && !st.user_shutdown && !self.fake {
                                drop(st);
                                self.log(Level::Warning, "The job task completed without establishing a connection or performing a proper shutdown. Ensure that job_ctx.connect()/job_ctx.shutdown() is called and the job is correctly finalized.", None);
                            }
                        }
                    }
                }
                () = &mut warn_at, if !entry_done && !warned => {
                    warned = true;
                    let st = lock(&self.state);
                    if !st.connected && !st.user_shutdown {
                        drop(st);
                        self.log(Level::Warning, "The room connection was not established within 10 seconds after calling job_entry. This might mean that job_ctx.connect() was never invoked, or that no VoiceSession with an active RoomIO has been started.", None);
                    }
                }
                _ = ended.wait_for(|e| *e) => break,
            }
        }
        let (reason, user_initiated) = lock(&self.state).shutdown.clone().unwrap_or_default();
        self.call(|h| h.call_method1("_core_shutdown_began", (reason.as_str(), user_initiated)))
            .await?;

        // the entrypoint returns, or is cancelled
        if !entry_done
            && tokio::time::timeout(ENTRYPOINT_EXIT_TIMEOUT, &mut wait_entry).await.is_err()
        {
            self.log(Level::Warning, "entrypoint did not exit in time, cancelling", None);
            let entry = Python::attach(|py| entry.clone_ref(py));
            let _ = self.call(move |h| h.call_method1("_core_cancel", (entry,))).await;
        }

        // the bound on `VoiceSession.aclose()`: past it, the rest of the shutdown still runs
        let aclose_timeout = self.session_aclose_timeout;
        let close = self.call(|h| h.call_method0("_core_close_session"));
        match tokio::time::timeout(secs(aclose_timeout), close).await {
            Ok(Err(e)) => Python::attach(|py| e.print(py)),
            Ok(Ok(_)) => {}
            Err(_) => self.log(
                Level::Error,
                &format!(
                    "VoiceSession.aclose() timed out after {aclose_timeout:.1}s; proceeding with shutdown so registered callbacks still run."
                ),
                None,
            ),
        }

        if let Some(session_end) = &self.session_end {
            self.spanned(
                "on_session_end",
                Some(session_end),
                "_core_session_end",
                Some(self.session_end_timeout),
                "error while executing the on_session_end callback",
            )
            .await;
        }
        self.spanned(
            "session_end_upload",
            None,
            "_core_upload",
            None,
            "error in job_ctx._on_session_end",
        )
        .await;
        // the parent learns the job is ending, then why
        if let Some(parent) = self.parent.get() {
            let _ = parent.send(ToParent::ShuttingDown).await;
        }
        let extra = vec![
            ("reason", ExtraValue::Str(reason.clone())),
            ("user_initiated", ExtraValue::Bool(user_initiated)),
        ];
        py_log_extra(Level::Debug, "shutting down job task", None, extra);
        if let Some(parent) = self.parent.get() {
            let _ = parent.send(ToParent::Exiting { reason: reason.clone() }).await;
        }
        let disconnect = self.span("room_disconnect", None)?;
        let r = self
            .call(|h| h.call_method1("_core_disconnect", (disconnect.clone_ref(h.py()),)))
            .await;
        self.span_end(disconnect, None, false);
        r?;

        // the shutdown callbacks run together; a framework one gets no span of its own
        let callbacks = std::mem::take(&mut lock(&self.state).shutdown_callbacks);
        let mut runs = Vec::new();
        for ShutdownCallback { callback, takes_reason, framework, name } in callbacks {
            let inner = self.clone();
            let reason = reason.clone();
            runs.push(tokio::spawn(async move {
                let span = if framework {
                    None
                } else {
                    Some(inner.span("shutdown_callback", Some(&name))?)
                };
                let r = inner
                    .call(|h| {
                        let py = h.py();
                        let span = span.as_ref().map(|s| s.clone_ref(py));
                        let args = match takes_reason {
                            true => PyTuple::new(py, [reason.as_str()])?,
                            false => PyTuple::empty(py),
                        };
                        h.call_method1("_core_shutdown_callback", (callback, args, span))
                    })
                    .await;
                if let Some(span) = span {
                    let error = r.as_ref().err().map(|e| Python::attach(|py| e.clone_ref(py)));
                    inner.span_end(span, error, false);
                }
                r.map(drop)
            }));
        }
        for run in runs {
            if let Ok(Err(e)) = run.await {
                self.log(Level::Error, "error while shutting down the job", Some(e));
            }
        }
        Ok(())
    }
}

#[pymethods]
impl PyJobRun {
    #[allow(clippy::disallowed_methods, clippy::disallowed_types, reason = "S0: move to runtime")]
    /// `redaction` and `recording` are the job's own (`Job.enable_redaction`,
    /// `Job.enable_recording`), `url` the server's; `session_end` is the door's `on_session_end`.
    #[new]
    #[pyo3(signature = (
        *, fake_job, session_end_timeout, redaction, url = String::new(), recording = false,
        session_end = None, session_aclose_timeout = SESSION_ACLOSE_TIMEOUT,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        fake_job: bool,
        session_end_timeout: f64,
        redaction: bool,
        url: String,
        recording: bool,
        session_end: Option<Bound<'_, PyAny>>,
        session_aclose_timeout: f64,
    ) -> Self {
        let env = std::env::var("LIVEKIT_OBSERVABILITY_URL").ok();
        Self(Arc::new(JobRunInner {
            host: std::sync::OnceLock::new(),
            parent: std::sync::OnceLock::new(),
            fake: fake_job,
            session_end_timeout,
            session_end: session_end.as_ref().map(callable_name),
            session_aclose_timeout,
            observability_url: lk_agents_abi::worker::job::observability_url(&url, env.as_deref()),
            recording,
            state: Mutex::new(JobRunState { redaction, ..JobRunState::default() }),
            ended: watch::channel(false).0,
            connecting: tokio::sync::Mutex::new(()),
            tasks_closed: watch::channel(false).0,
        }))
    }

    /// `ctx.add_shutdown_callback`: `callback(reason)`, or `callback()` when it takes no argument,
    /// runs as the job shuts down.
    fn add_shutdown_callback(&self, callback: Bound<'_, PyAny>) -> PyResult<()> {
        let bound_method =
            callback.py().import("inspect")?.call_method1("ismethod", (&callback,))?.is_truthy()?;
        let argcount: usize = callback.getattr("__code__")?.getattr("co_argcount")?.extract()?;
        let module: String =
            callback.getattr("__module__").and_then(|m| m.extract()).unwrap_or_default();
        lock(&self.0.state).shutdown_callbacks.push(ShutdownCallback {
            takes_reason: argcount > usize::from(bound_method),
            framework: module == "livekit.agents" || module.starts_with("livekit.agents."),
            name: callable_name(&callback),
            callback: callback.unbind(),
        });
        Ok(())
    }

    /// The job's LiveKit Cloud observability endpoint: `LIVEKIT_OBSERVABILITY_URL`, else the
    /// server's when it is LiveKit Cloud's.
    #[getter]
    fn observability_url(&self) -> Option<String> {
        self.0.observability_url.clone()
    }

    /// Whether the job buffers its logs from the start, for the recording to upload them: the
    /// server asked for the recording of a real job that has an observability endpoint.
    #[getter]
    fn buffer_logs(&self) -> bool {
        !self.0.fake && self.0.recording && self.0.observability_url.is_some()
    }

    /// The job ends with `reason`; the first reason given is the one kept.
    fn shutdown(&self, reason: String, user_initiated: bool) {
        self.0.end(reason, user_initiated);
    }

    /// The room is connected (by `connect`, or by a caller that owns the connection).
    fn connected(&self) {
        lock(&self.0.state).connected = true;
    }

    /// `ctx.connect()`: awaits `connect()` once; a call while it runs waits for it, a call after
    /// it connected returns at once, a failed one leaves the next call to try again.
    fn connect<'py>(&self, py: Python<'py>, connect: Py<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let handle = CancellingHandle::new(py, connect)?;
        let inner = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let _connecting = inner.connecting.lock().await;
            if lock(&inner.state).connected {
                return Ok(());
            }
            handle.await_call(|connect| connect.call0()).await?;
            lock(&inner.state).connected = true;
            Ok(())
        })
    }

    /// `ctx.add_participant_entrypoint`: `entrypoint(ctx, participant)` runs for each participant
    /// of one of `kinds` that joins.
    fn add_participant_entrypoint(
        &self,
        py: Python<'_>,
        entrypoint: Py<PyAny>,
        kinds: Vec<i32>,
    ) -> PyResult<()> {
        let mut st = lock(&self.0.state);
        for (known, _) in &st.entrypoints {
            if known.bind(py).eq(entrypoint.bind(py))? {
                return Err(PyValueError::new_err("entrypoints cannot be added more than once"));
            }
        }
        st.entrypoints.push((entrypoint, kinds));
        Ok(())
    }

    /// A participant joined, or was in the room when it connected; `ctx` is the `JobContext`.
    fn participant_joined(
        &self,
        py: Python<'_>,
        ctx: Py<PyAny>,
        participant: Py<PyAny>,
        identity: String,
        kind: i32,
    ) -> PyResult<()> {
        self.0.participant_joined(py, &ctx, &participant, &identity, kind)
    }

    /// A participant left: a simulator leaving ends the job.
    fn participant_left(&self, attributes: HashMap<String, String>) {
        if attributes.contains_key(ATTRIBUTE_SIMULATOR) {
            self.0.log(Level::Debug, "simulator disconnected, shutting down the job", None);
            self.0.end("simulation completed".into(), true);
        }
    }

    #[allow(clippy::disallowed_methods, clippy::disallowed_macros, reason = "S0: move to runtime")]
    /// `ctx.<name>()`, an API call of the job's (`delete_room`, ...): `call()`'s awaitable, awaited
    /// on the caller's loop, as a future. A failure is logged whether or not the caller awaits
    /// it, and it is cancelled once the job ends (`close_tasks`). A console job makes no call: its
    /// future is `console_result`.
    fn api_call<'py>(
        &self,
        py: Python<'py>,
        call: Py<PyAny>,
        name: String,
        console_result: Py<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if self.0.fake {
            self.0.log(
                Level::Warning,
                &format!("job_ctx.{name}() is not executed while in console mode"),
                None,
            );
            return pyo3_async_runtimes::tokio::future_into_py(
                py,
                async move { Ok(console_result) },
            );
        }
        let handle = CancellingHandle::new(py, call.bind(py).call0()?.unbind())?;
        let (tx, rx) = tokio::sync::oneshot::channel::<PyResult<Py<PyAny>>>();
        let mut closed = self.0.tasks_closed.subscribe();
        let task = pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
            let r = tokio::select! {
                r = handle.await_call(|awaitable| Ok(awaitable.clone())) => r,
                _ = closed.wait_for(|closed| *closed) => Err(cancelled()),
            };
            if let Err(e) = &r {
                if !is_cancelled(e) {
                    let e = Python::attach(|py| e.clone_ref(py));
                    py_log(Level::Error, &format!("error in {name}"), Some(e));
                }
            }
            let _ = tx.send(r);
        });
        let mut st = lock(&self.0.state);
        st.tasks.retain(|t| !t.is_finished());
        st.tasks.push(task);
        drop(st);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            rx.await.unwrap_or_else(|_| Err(cancelled()))
        })
    }

    /// Cancels the API calls still running and waits for them.
    fn close_tasks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.0.tasks_closed.send_replace(true);
        let tasks = std::mem::take(&mut lock(&self.0.state).tasks);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            for task in tasks {
                let _ = task.await;
            }
            Ok(())
        })
    }

    /// `ctx.init_recording(options)`: the redaction to record with when the cloud upload is to be
    /// set up now, `None` when not (set up already, nothing to upload, or no observability
    /// endpoint). Audio without its transcript under redaction is refused.
    fn init_recording(&self, options: HashMap<String, bool>) -> PyResult<Option<bool>> {
        let observability = self.0.observability_url.is_some();
        let get = |key: &str, default: bool| options.get(key).copied().unwrap_or(default);
        let mut st = lock(&self.0.state);
        let redaction = st.redaction || get("redaction", false);
        if redaction && get("audio", true) && !get("transcript", true) {
            return Err(PyValueError::new_err(
                "audio upload requires transcript upload when redaction is enabled",
            ));
        }
        if st.recording_initialized {
            return Ok(None);
        }
        st.recording_initialized = true;
        st.redaction = redaction;
        let uploads = ["traces", "logs", "audio", "transcript"].iter().any(|key| get(key, true));
        Ok((uploads && observability).then_some(redaction))
    }

    /// A session started for the job (on its room, or the console's): whether it is the job's
    /// primary one, which the job closes and reports for. The first claims it.
    fn claim_session(&self, py: Python<'_>, session: Py<PyAny>) -> bool {
        let mut st = lock(&self.0.state);
        match &st.primary_session {
            Some(primary) => primary.is(&session),
            None => {
                st.primary_session = Some(session.clone_ref(py));
                true
            }
        }
    }

    /// The job's primary session, once one started.
    #[getter]
    fn primary_session(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        lock(&self.0.state).primary_session.as_ref().map(|s| s.clone_ref(py))
    }

    /// Whether `init_recording` ran.
    #[getter]
    fn recording_initialized(&self) -> bool {
        lock(&self.0.state).recording_initialized
    }

    /// The job's redaction, as `init_recording` resolved it.
    #[getter]
    fn redaction(&self) -> bool {
        lock(&self.0.state).redaction
    }

    /// The entrypoint, then the shutdown sequence once the job ends; `host` is the `_JobProc`,
    /// which runs each step.
    fn run<'py>(&self, py: Python<'py>, host: Py<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        if self.0.host.set(CancellingHandle::new(py, host)?).is_err() {
            return Err(PyRuntimeError::new_err("the job already runs"));
        }
        let inner = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move { inner.run().await })
    }
}

lk_agents_abi::register_host!(class PyJobRun);
