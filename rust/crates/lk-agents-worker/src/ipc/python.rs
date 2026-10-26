//! `_core.ipc_encode`, `_core.ipc_decode`, `_core.ProcSupervisor` and `_core.IpcClient`: the job
//! process IPC as `ipc/channel.py`, `ipc/supervised_proc.py` and `ipc/proc_client.py` drive it.
//!
//! `ProcSupervisor(fd, host, ...)` takes over the parent's socket. `host` is the `SupervisedProc`:
//! Rust calls its `_core_log(level, msg, extra, exc)`, `_core_is_alive()`, `_core_kill()`,
//! `_sample_memory_mb()` and `_core_has_running_job()`, and drops it once the process exited.

use std::borrow::Cow;
use std::sync::Mutex;

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use lk_agents_abi::python::{CancellingHandle, HostBridge, HostHandle, check_pid, pyo3_async_runtimes};
use pyo3::exceptions::{PyAssertionError, PyException, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};

use super::pool::{LaunchError, PoolEvent, PoolHost, PoolOptions, ProcPool};

use super::{
    Closed, ExtraValue, FieldKind, InitError, IpcClient, IpcMessage, Level, MESSAGES, MemoryError,
    ProcHost, ProcSupervisor, SupervisorOptions, Value, lock,
};

/// `duplex_unix.DuplexClosed`, what the duplex the supervisor and client replace raised.
fn duplex_closed(_: Closed) -> PyErr {
    Python::attach(|py| {
        match py
            .import("livekit.agents.utils.aio.duplex_unix")
            .and_then(|m| m.getattr("DuplexClosed"))
        {
            Ok(cls) => cls.call0().map(PyErr::from_value).unwrap_or_else(|e| e),
            Err(e) => e,
        }
    })
}

// ------------------------------------------------------------------------------------ codec

/// A message's payload from its `MSG_ID` and its fields by name (a message dataclass's `vars`).
#[pyfunction]
fn ipc_encode<'py>(
    py: Python<'py>,
    msg_id: i32,
    fields: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyBytes>> {
    let (_, schema) = usize::try_from(msg_id)
        .ok()
        .and_then(|i| MESSAGES.get(i))
        .ok_or_else(|| PyValueError::new_err(format!("unknown IPC message id {msg_id}")))?;
    let mut values = Vec::with_capacity(schema.len());
    for (name, kind) in *schema {
        let v = fields
            .get_item(name)?
            .ok_or_else(|| PyValueError::new_err(format!("IPC message field {name} is missing")))?;
        values.push(match kind {
            FieldKind::Bool => Value::Bool(v.extract()?),
            FieldKind::Float => Value::Float(v.extract::<f64>()? as f32),
            FieldKind::Double => Value::Double(v.extract()?),
            FieldKind::Long => Value::Long(v.extract()?),
            FieldKind::Str => Value::Str(v.extract()?),
            FieldKind::Bytes => Value::Bytes(v.extract::<Cow<[u8]>>()?.into_owned()),
            FieldKind::OptBytes if v.is_none() => Value::Absent,
            FieldKind::OptBytes => Value::Bytes(v.extract::<Cow<[u8]>>()?.into_owned()),
        });
    }
    let msg = IpcMessage::new(msg_id, values).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, &msg.encode()))
}

/// A payload's `MSG_ID` and its fields by name.
#[pyfunction]
fn ipc_decode<'py>(py: Python<'py>, data: &[u8]) -> PyResult<(i32, Bound<'py, PyDict>)> {
    let msg = IpcMessage::decode(data).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let fields = PyDict::new(py);
    for (name, value) in msg.fields() {
        match value {
            Value::Bool(v) => fields.set_item(name, v)?,
            Value::Float(v) => fields.set_item(name, f64::from(*v))?,
            Value::Double(v) => fields.set_item(name, v)?,
            Value::Long(v) => fields.set_item(name, v)?,
            Value::Str(v) => fields.set_item(name, v)?,
            Value::Bytes(v) => fields.set_item(name, PyBytes::new(py, v))?,
            Value::Absent => fields.set_item(name, py.None())?,
        }
    }
    Ok((msg.id(), fields))
}

lk_agents_abi::register_host!(fn ipc_encode);
lk_agents_abi::register_host!(fn ipc_decode);

// ------------------------------------------------------------------------------- supervisor

/// The `SupervisedProc` as the supervisor's host, until the process exited.
struct PyProcHost(Mutex<Option<Py<PyAny>>>);

impl PyProcHost {
    /// Calls the host; an error is printed (the callbacks are the host's own and never raise).
    fn call<R>(&self, f: impl FnOnce(&Bound<'_, PyAny>) -> PyResult<R>) -> Option<R> {
        Python::attach(|py| {
            let host = lock(&self.0).as_ref()?.clone_ref(py);
            f(host.bind(py)).inspect_err(|e| e.print(py)).ok()
        })
    }

    fn emit(&self, level: Level, msg: &str, extra: super::Extra, exc: Option<PyErr>) {
        self.call(|host| {
            let py = host.py();
            let fields = PyDict::new(py);
            for (key, value) in extra {
                match value {
                    ExtraValue::Int(v) => fields.set_item(key, v)?,
                    ExtraValue::Float(v) => fields.set_item(key, v)?,
                    ExtraValue::Str(v) => fields.set_item(key, v)?,
                    ExtraValue::Bool(v) => fields.set_item(key, v)?,
                }
            }
            let exc = exc.map(|e| e.into_value(py));
            host.call_method1("_core_log", (level as i32, msg, fields, exc)).map(drop)
        });
    }
}

impl ProcHost for PyProcHost {
    type Error = PyErr;

    fn log(&self, level: Level, msg: &str, extra: super::Extra) {
        self.emit(level, msg, extra, None);
    }

    fn log_exception(&self, msg: &str, err: PyErr) {
        self.emit(Level::Error, msg, Vec::new(), Some(err));
    }

    fn is_alive(&self) -> bool {
        self.call(|host| host.call_method0("_core_is_alive")?.extract()).unwrap_or(false)
    }

    fn kill(&self) {
        self.call(|host| host.call_method0("_core_kill").map(drop));
    }

    fn sample_memory(&self) -> Result<(f64, String), MemoryError<PyErr>> {
        Python::attach(|py| {
            let Some(host) = lock(&self.0).as_ref().map(|h| h.clone_ref(py)) else {
                return Err(MemoryError::Unavailable("process exited".into()));
            };
            let err =
                match host.bind(py).call_method0("_sample_memory_mb").and_then(|r| r.extract()) {
                    Ok(sample) => return Ok(sample),
                    Err(err) => err,
                };
            if !err.is_instance_of::<PyException>(py) {
                return Err(MemoryError::Interrupted);
            }
            let unavailable = py.import("psutil").is_ok_and(|psutil| {
                ["NoSuchProcess", "AccessDenied"].iter().any(|name| {
                    psutil
                        .getattr(*name)
                        .is_ok_and(|cls| err.value(py).is_instance(&cls).unwrap_or(false))
                })
            });
            match unavailable {
                true => Err(MemoryError::Unavailable(err.value(py).to_string())),
                false => Err(MemoryError::Failed(err)),
            }
        })
    }

    fn has_running_job(&self) -> bool {
        self.call(|host| host.call_method0("_core_has_running_job")?.extract()).unwrap_or(false)
    }

    fn release(&self) {
        let host = lock(&self.0).take();
        Python::attach(|_| drop(host));
    }
}

#[cfg(unix)]
type Socket = tokio::net::UnixStream;
#[cfg(windows)]
type Socket = tokio::net::TcpStream;

/// Takes ownership of a connected socket (`socket.detach()`). Needs a runtime context.
fn socket(fd: i64) -> PyResult<Socket> {
    #[cfg(unix)]
    let std = {
        use std::os::fd::FromRawFd;
        let fd = i32::try_from(fd).map_err(|_| PyValueError::new_err("bad socket fd"))?;
        // SAFETY: the caller detached the descriptor and hands its ownership over.
        unsafe { std::os::unix::net::UnixStream::from_raw_fd(fd) }
    };
    #[cfg(windows)]
    let std = {
        use std::os::windows::io::FromRawSocket;
        // SAFETY: the caller detached the socket and hands its ownership over.
        unsafe { std::net::TcpStream::from_raw_socket(fd as u64) }
    };
    std.set_nonblocking(true)?;
    Ok(Socket::from_std(std)?)
}

/// The parent's side of one job or inference process (`ipc/supervised_proc.py`).
#[pyclass(name = "ProcSupervisor", module = "livekit.agents._core", frozen)]
pub struct PyProcSupervisor(ProcSupervisor<PyProcHost>);

#[pymethods]
impl PyProcSupervisor {
    #[new]
    #[pyo3(signature = (
        fd, host, *, pid, process_kind, initialize_timeout, close_timeout, memory_warn_mb,
        memory_limit_mb, ping_interval, ping_timeout, high_ping_threshold, killable = true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        fd: i64,
        host: Py<PyAny>,
        pid: u32,
        process_kind: String,
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
        let _rt = pyo3_async_runtimes::tokio::get_runtime().enter();
        let opts = SupervisorOptions {
            pid,
            process_kind,
            initialize_timeout,
            close_timeout,
            memory_warn_mb,
            memory_limit_mb,
            ping_interval,
            ping_timeout,
            high_ping_threshold,
            killable,
        };
        Ok(Self(ProcSupervisor::new(socket(fd)?, PyProcHost(Mutex::new(Some(host))), opts)))
    }

    /// Seconds the process took to answer the InitializeRequest the host sent, or `None` when it
    /// timed out (the process was killed).
    fn initialize<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let sup = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            match sup.initialize().await {
                Ok(elapsed) => Ok(elapsed.map(|d| d.as_secs_f64())),
                Err(InitError::Closed) => Err(duplex_closed(Closed)),
                Err(InitError::Unexpected) => {
                    Err(PyAssertionError::new_err(InitError::Unexpected.to_string()))
                }
                Err(e @ InitError::Failed(_)) => Err(PyRuntimeError::new_err(e.to_string())),
                Err(InitError::Codec(e)) => Err(PyValueError::new_err(e.to_string())),
            }
        })
    }

    /// Starts the read loop, the pings and the memory monitor.
    fn supervise(&self) {
        let _rt = pyo3_async_runtimes::tokio::get_runtime().enter();
        self.0.supervise();
    }

    fn recv_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let sup = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            sup.recv().await.map_err(duplex_closed)
        })
    }

    fn send_bytes<'py>(&self, py: Python<'py>, data: Cow<[u8]>) -> PyResult<Bound<'py, PyAny>> {
        let sup = self.0.clone();
        let data = data.into_owned();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            sup.send(&data).await.map_err(duplex_closed)
        })
    }

    fn shutdown<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let sup = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            sup.shutdown().await;
            Ok(())
        })
    }

    fn kill<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let sup = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            sup.kill().await;
            Ok(())
        })
    }

    fn exited<'py>(&self, py: Python<'py>, exitcode: Option<i64>) -> PyResult<Bound<'py, PyAny>> {
        let sup = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            sup.exited(exitcode).await;
            Ok(())
        })
    }

    #[getter]
    fn killed(&self) -> bool {
        self.0.killed()
    }
}

lk_agents_abi::register_host!(class PyProcSupervisor);

// ----------------------------------------------------------------------------------- client

/// The child's side of the socket after the initialize handshake (`ipc/proc_client.py`).
#[pyclass(name = "IpcClient", module = "livekit.agents._core", frozen)]
pub struct PyIpcClient(std::sync::Arc<IpcClient>);

#[pymethods]
impl PyIpcClient {
    /// `health_timeout`: seconds of parent silence after which the child calls itself orphaned.
    #[new]
    #[pyo3(signature = (fd, health_timeout = None))]
    fn new(fd: i64, health_timeout: Option<f64>) -> PyResult<Self> {
        check_pid()?;
        let _rt = pyo3_async_runtimes::tokio::get_runtime().enter();
        let health_timeout = health_timeout.map(super::secs);
        Ok(Self(std::sync::Arc::new(IpcClient::new(socket(fd)?, health_timeout))))
    }

    fn recv_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            client.recv().await.map_err(duplex_closed)
        })
    }

    fn send_bytes<'py>(&self, py: Python<'py>, data: Cow<[u8]>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.0.clone();
        let data = data.into_owned();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            client.send(&data).await.map_err(duplex_closed)
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            client.close().await;
            Ok(())
        })
    }
}

lk_agents_abi::register_host!(class PyIpcClient);

// ------------------------------------------------------------------------------------- pool

/// An executor, compared by identity.
#[derive(Clone)]
pub struct PyProc(Arc<Py<PyAny>>);

impl PartialEq for PyProc {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ptr() == other.0.as_ptr()
    }
}

/// The Python `ProcPool`: it makes the executors (`_core_create`), reports a failed
/// initialization (`_core_init_failed`) and emits the events; the executors run their own steps.
struct PyPoolHost(CancellingHandle);

impl PyPoolHost {
    async fn step(&self, proc: &PyProc, method: &'static str) -> PyResult<()> {
        let proc = Python::attach(|py| self.0.child(proc.0.clone_ref(py)));
        proc.await_call(|p| p.call_method0(method)).await.map(drop)
    }

    fn soon(&self, method: &'static str, args: impl for<'py> FnOnce(Python<'py>) -> PyResult<Bound<'py, PyTuple>>) {
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
        let handle = Python::attach(|py| HostHandle::from_parts(pool.clone_ref(py), self.0.task_locals().clone()));
        handle
            .run_on_loop(move |py| Ok(pool.bind(py).call_method0("_core_create")?.unbind()))
            .await
            .map(|p| PyProc(Arc::new(p)))
            .map_err(|e| Python::attach(|py| lk_agents_abi::python::raise_provider(py, e)))
    }

    async fn start(&self, proc: &PyProc) -> PyResult<()> {
        self.step(proc, "start").await
    }

    async fn initialize(&self, proc: &PyProc) -> PyResult<()> {
        self.step(proc, "initialize").await
    }

    /// The dispatch timeline: a warm process is now handling the job (`launched_at`).
    async fn launch(&self, proc: &PyProc, job: &Self::Job) -> PyResult<()> {
        let (proc, job) = Python::attach(|py| -> PyResult<_> {
            let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or_default();
            job.bind(py).setattr("launched_at", now)?;
            Ok((self.0.child(proc.0.clone_ref(py)), job.clone_ref(py)))
        })?;
        proc.await_call(|p| p.call_method1("launch_job", (job,))).await.map(drop)
    }

    async fn close(&self, proc: &PyProc) {
        if let Err(e) = self.step(proc, "aclose").await {
            Python::attach(|py| e.print(py));
        }
    }

    async fn join(&self, proc: &PyProc) {
        if let Err(e) = self.step(proc, "join").await {
            Python::attach(|py| e.print(py));
        }
    }

    fn emit(&self, event: PoolEvent, proc: &PyProc) {
        let proc = proc.0.clone();
        self.soon("emit", move |py| PyTuple::new(py, [event.as_str().into_pyobject(py)?.into_any(), proc.bind(py).clone()]));
    }

    fn init_failed(&self, proc: &PyProc, error: PyErr) {
        let proc = proc.0.clone();
        self.soon("_core_init_failed", move |py| {
            PyTuple::new(py, [proc.bind(py).clone(), error.into_value(py).into_bound(py).into_any()])
        });
    }
}

/// `ipc/proc_pool.py`'s `ProcPool`: the warm processes and the job launches.
#[pyclass(name = "ProcPool", module = "livekit.agents._core", frozen)]
pub struct PyProcPool(ProcPool<PyPoolHost>);

#[pymethods]
impl PyProcPool {
    /// `host` is the Python `ProcPool`; must be called inside its running loop.
    #[new]
    #[pyo3(signature = (host, *, num_idle_processes, initialize_timeout))]
    fn new(py: Python<'_>, host: Py<PyAny>, num_idle_processes: usize, initialize_timeout: f64) -> PyResult<Self> {
        check_pid()?;
        let opts = PoolOptions { num_idle_processes, initialize_timeout: super::secs(initialize_timeout) };
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
            pool.start().await;
            Ok(())
        })
    }

    fn launch_job<'py>(&self, py: Python<'py>, info: Py<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let job_id: String = info.bind(py).getattr("job")?.getattr("id")?.extract()?;
        let pool = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            pool.launch_job(&job_id, Arc::new(info)).await.map_err(|e| match e {
                LaunchError::NoProcess => PyRuntimeError::new_err(
                    "no process became available after 3 attempts",
                ),
                LaunchError::Launch(e) => e,
            })
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
