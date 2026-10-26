//! `_core.ipc_encode`, `_core.ipc_decode`, `_core.ProcSupervisor` and `_core.IpcClient`: the job
//! process IPC as `ipc/channel.py`, `ipc/supervised_proc.py` and `ipc/proc_client.py` drive it.
//!
//! `ProcSupervisor(fd, host, ...)` takes over the parent's socket. `host` is the `SupervisedProc`:
//! Rust calls its `_core_log(level, msg, extra, exc)`, `_core_is_alive()`, `_core_kill()`,
//! `_sample_memory_mb()` and `_core_has_running_job()`, and drops it once the process exited.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::sync::watch;

use lk_agents_abi::python::{CancellingHandle, HostBridge, HostHandle, check_pid, pyo3_async_runtimes};
use pyo3::exceptions::{PyAssertionError, PyException, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};

use super::pool::{LaunchError, PoolEvent, PoolHost, PoolOptions, ProcPool};

use super::{
    Closed, ExtraValue, FieldKind, InitError, IpcClient, IpcMessage, Level, MESSAGES, MemoryError,
    ProcHost, ProcSupervisor, SupervisorOptions, Value, lock, secs,
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
    Ok((msg.id(), fields_of(py, &msg)?))
}

/// A message's fields by name.
fn fields_of<'py>(py: Python<'py>, msg: &IpcMessage) -> PyResult<Bound<'py, PyDict>> {
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
    Ok(fields)
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

/// The parent's side of one job or inference process (`ipc/supervised_proc.py`): its
/// initialization, supervision, messages and close, from the spawn the host made to the exit.
#[pyclass(name = "ProcSupervisor", module = "livekit.agents._core", frozen)]
pub struct PyProcSupervisor {
    sup: ProcSupervisor<PyProcHost>,
    /// Whether the process initialized, once that settled (answered, failed, timed out, killed).
    initialized: watch::Sender<Option<bool>>,
    /// Set once `run` ended: the process exited and was reaped.
    ran: watch::Sender<bool>,
    inferences: Arc<Inferences>,
}

impl PyProcSupervisor {
    fn settle(&self, initialized: bool) {
        self.initialized.send_if_modified(|v| v.is_none() && v.replace(initialized).is_none());
    }
}

/// One message the process sent while supervised: an inference it asks for runs on the host's
/// `_core_inference(method, data)` and is answered; an answer to one of ours resolves it.
async fn on_proc_message(
    sup: ProcSupervisor<PyProcHost>,
    inferences: Arc<Inferences>,
    host: Arc<CancellingHandle>,
    data: Vec<u8>,
    inflight: &mut tokio::task::JoinSet<()>,
) {
    let msg = match IpcMessage::decode(&data) {
        Ok(msg) => msg,
        Err(e) => return log::error!(target: "livekit.agents", "Error in _read_ipc_task: {e}"),
    };
    match msg.id() {
        super::INFERENCE_RESPONSE => inferences.answered(&msg),
        super::INFERENCE_REQUEST => {
            let str_of = |name: &str| match msg.get(name) {
                Some(Value::Str(s)) => s.clone(),
                _ => String::new(),
            };
            let (method, request_id) = (str_of("method"), str_of("request_id"));
            let data = match msg.get("data") {
                Some(Value::Bytes(data)) => data.clone(),
                _ => Vec::new(),
            };
            inflight.spawn(async move {
                let r = host
                    .await_call(|h| h.call_method1("_core_inference", (method, PyBytes::new(h.py(), &data))))
                    .await;
                let (data, error) = Python::attach(|py| match r {
                    Ok(out) if out.is_none(py) => (Value::Absent, String::new()),
                    Ok(out) => match out.extract::<Vec<u8>>(py) {
                        Ok(bytes) => (Value::Bytes(bytes), String::new()),
                        Err(e) => (Value::Absent, e.value(py).to_string()),
                    },
                    Err(e) => (Value::Absent, e.value(py).to_string()),
                });
                let reply = IpcMessage::new(super::INFERENCE_RESPONSE, vec![Value::Str(request_id), data, Value::Str(error)]);
                if let Ok(reply) = reply {
                    let _ = sup.send(&reply.encode()).await;
                }
            });
        }
        _ => {}
    }
}

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
        Ok(Self {
            sup: ProcSupervisor::new(socket(fd)?, PyProcHost(Mutex::new(Some(host))), opts),
            initialized: watch::channel(None).0,
            ran: watch::channel(false).0,
            inferences: Arc::default(),
        })
    }

    /// Seconds the process took to answer the InitializeRequest the host sent, or `None` when it
    /// timed out (the process was killed).
    fn initialize<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let this = slf.clone().unbind();
        let sup = slf.get().sup.clone();
        pyo3_async_runtimes::tokio::future_into_py(slf.py(), async move {
            let outcome = sup.initialize().await;
            this.get().settle(matches!(outcome, Ok(Some(_))));
            match outcome {
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

    /// Whether the process answered its initialization.
    #[getter]
    fn initialized(&self) -> bool {
        *self.initialized.borrow() == Some(true)
    }

    /// The process from its initialization to its exit: once the initialization settled, the
    /// pings, the memory monitor and the messages (`_core_inference(method, data)` answers the
    /// inferences it asks for); then `joined` (the process was reaped), `host._core_reap()` (its
    /// exit code) and the end of the supervision.
    fn run<'py>(slf: Bound<'py, Self>, joined: Py<PyAny>, host: Py<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let host = Arc::new(CancellingHandle::new(py, host)?);
        let this = slf.get();
        let (sup, inferences) = (this.sup.clone(), this.inferences.clone());
        let mut initialized = this.initialized.subscribe();
        let ran = this.ran.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let _ = initialized.wait_for(Option::is_some).await;
            sup.supervise();
            let serving = {
                let (sup, inferences, host) = (sup.clone(), inferences.clone(), host.clone());
                tokio::spawn(async move {
                    let mut inflight = tokio::task::JoinSet::new();
                    while let Ok(data) = sup.recv().await {
                        on_proc_message(sup.clone(), inferences.clone(), host.clone(), data, &mut inflight).await;
                    }
                    // the socket closed: what is still running is cancelled once the process is reaped
                    std::future::pending::<()>().await;
                })
            };
            let joined = host.await_call(|h| Ok(joined.bind(h.py()).clone())).await;
            let exitcode = joined.and_then(|_| {
                Python::attach(|py| host.object().bind(py).call_method0("_core_reap")?.extract::<Option<i64>>())
            });
            serving.abort();
            let _ = serving.await;
            inferences.clear();
            let exitcode = exitcode.unwrap_or_else(|e| {
                Python::attach(|py| e.print(py));
                None
            });
            sup.exited(exitcode).await;
            ran.send_replace(true);
            Ok(())
        })
    }

    /// A graceful close: an initialized process is asked to shut down (killed when late), any
    /// other is killed; returns once it exited.
    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let initialized = self.initialized();
        self.settle(false);
        let sup = self.sup.clone();
        let mut ran = self.ran.subscribe();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            if initialized {
                sup.shutdown().await;
            } else {
                sup.kill().await;
            }
            let _ = ran.wait_for(|ran| *ran).await;
            Ok(())
        })
    }

    /// The kill sequence; returns once the process exited.
    fn kill<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.settle(false);
        let sup = self.sup.clone();
        let mut ran = self.ran.subscribe();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            sup.kill().await;
            let _ = ran.wait_for(|ran| *ran).await;
            Ok(())
        })
    }

    /// `InferenceExecutor.do_inference`: the process runs `method` on `data`.
    fn do_inference<'py>(&self, py: Python<'py>, method: String, data: Cow<[u8]>) -> PyResult<Bound<'py, PyAny>> {
        let (sup, inferences) = (self.sup.clone(), self.inferences.clone());
        let data = data.into_owned();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let send = |msg: IpcMessage| async move { sup.send(&msg.encode()).await };
            inferences.request("inference_req", method, data, send).await
        })
    }

    fn send_bytes<'py>(&self, py: Python<'py>, data: Cow<[u8]>) -> PyResult<Bound<'py, PyAny>> {
        let sup = self.sup.clone();
        let data = data.into_owned();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            sup.send(&data).await.map_err(duplex_closed)
        })
    }

    #[getter]
    fn killed(&self) -> bool {
        self.sup.killed()
    }
}

lk_agents_abi::register_host!(class PyProcSupervisor);

// ----------------------------------------------------------------------------------- client

/// The child's side of the socket after the initialize handshake (`ipc/proc_client.py`): it
/// serves the parent's messages (`serve`), answers the pings itself once the child's loop has
/// run a callback, and matches the inference requests the job makes (`do_inference`) to their
/// responses.
#[pyclass(name = "IpcClient", module = "livekit.agents._core", frozen)]
pub struct PyIpcClient(Arc<Child>);

struct Child {
    client: IpcClient,
    inferences: Inferences,
    /// `stop()`: `serve` returns.
    stop: watch::Sender<bool>,
}

/// An `InferenceResponse`'s data and error.
type InferenceReply = tokio::sync::oneshot::Sender<(Option<Vec<u8>>, String)>;

/// The inference requests one side of the socket sent, awaiting their `InferenceResponse`.
#[derive(Default)]
struct Inferences {
    pending: Mutex<HashMap<String, InferenceReply>>,
    last_request: AtomicU64,
}

impl Inferences {
    /// An `InferenceResponse`: its request's answer.
    fn answered(&self, msg: &IpcMessage) {
        let request_id = match msg.get("request_id") {
            Some(Value::Str(s)) => s.clone(),
            _ => String::new(),
        };
        let data = match msg.get("data") {
            Some(Value::Bytes(data)) => Some(data.clone()),
            _ => None,
        };
        let error = match msg.get("error") {
            Some(Value::Str(s)) => s.clone(),
            _ => String::new(),
        };
        match lock(&self.pending).remove(&request_id) {
            Some(reply) => {
                let _ = reply.send((data, error));
            }
            None => log::warn!(
                target: "livekit.agents", request_id = request_id.as_str();
                "received unexpected inference response"
            ),
        }
    }

    /// Sends an `InferenceRequest` for `method` on `data` (ids `<prefix>_<n>`) and awaits its
    /// answer: raises the `RuntimeError` of a failed inference, `CancelledError` once nothing
    /// will answer.
    async fn request<F: std::future::Future<Output = Result<(), Closed>>>(
        &self,
        prefix: &str,
        method: String,
        data: Vec<u8>,
        send: impl FnOnce(IpcMessage) -> F,
    ) -> PyResult<Option<Py<PyBytes>>> {
        let n = self.last_request.fetch_add(1, Ordering::Relaxed) + 1;
        let request_id = format!("{prefix}_{n}");
        let (tx, rx) = tokio::sync::oneshot::channel();
        lock(&self.pending).insert(request_id.clone(), tx);
        let request = IpcMessage::new(
            super::INFERENCE_REQUEST,
            vec![Value::Str(method.clone()), Value::Str(request_id.clone()), Value::Bytes(data)],
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if let Err(e) = send(request).await {
            lock(&self.pending).remove(&request_id);
            return Err(duplex_closed(e));
        }
        let Ok((data, error)) = rx.await else {
            return Err(pyo3::exceptions::asyncio::CancelledError::new_err(()));
        };
        if !error.is_empty() {
            return Err(PyRuntimeError::new_err(format!("inference of {method} failed: {error}")));
        }
        Ok(data.map(|d| Python::attach(|py| PyBytes::new(py, &d).unbind())))
    }

    fn clear(&self) {
        lock(&self.pending).clear();
    }
}

impl Child {
    async fn send(&self, msg: IpcMessage) -> Result<(), Closed> {
        self.client.send(&msg.encode()).await
    }

    /// One message from the parent; `false` ends the serving.
    async fn on_message(&self, host: &CancellingHandle, msg: IpcMessage) -> PyResult<bool> {
        let str_of = |name: &str| match msg.get(name) {
            Some(Value::Str(s)) => s.clone(),
            _ => String::new(),
        };
        match msg.id() {
            // answered once the loop ran a callback: the pong measures the loop's responsiveness
            super::PING_REQUEST => {
                let Some(Value::Long(timestamp)) = msg.get("timestamp").cloned() else {
                    return Ok(true);
                };
                let loop_ran = HostHandle::from_parts(
                    Python::attach(|py| host.object().clone_ref(py)),
                    host.task_locals().clone(),
                );
                if loop_ran.run_on_loop(|_| Ok(())).await.is_err() {
                    return Ok(false);
                }
                let now = super::time_ms().max(0) as u64;
                let pong = IpcMessage::new(
                    super::PONG_RESPONSE,
                    vec![Value::Long(timestamp), Value::Long(now)],
                )
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(self.send(pong).await.is_ok())
            }
            super::INFERENCE_RESPONSE => {
                self.inferences.answered(&msg);
                Ok(true)
            }
            super::START_JOB_REQUEST => {
                host.await_call(|h| h.call_method1("_core_start_job", (fields_of(h.py(), &msg)?,)))
                    .await?;
                Ok(true)
            }
            super::SHUTDOWN_REQUEST => {
                let reason = str_of("reason");
                let done = host.await_call(|h| h.call_method1("_core_shutdown", (reason,))).await?;
                Ok(!Python::attach(|py| done.bind(py).is_truthy())?)
            }
            super::INFERENCE_REQUEST => {
                let data = match msg.get("data") {
                    Some(Value::Bytes(data)) => data.clone(),
                    _ => Vec::new(),
                };
                let (method, request_id) = (str_of("method"), str_of("request_id"));
                host.await_call(|h| {
                    let data = PyBytes::new(h.py(), &data);
                    h.call_method1("_core_inference", (method, request_id, data))
                })
                .await?;
                Ok(true)
            }
            super::DUMP_STACK_TRACE_REQUEST => {
                host.await_call(|h| h.call_method0("_core_dump_stacks")).await?;
                Ok(true)
            }
            _ => Ok(true),
        }
    }
}

#[pymethods]
impl PyIpcClient {
    /// `health_timeout`: seconds of parent silence after which the child calls itself orphaned.
    #[new]
    #[pyo3(signature = (fd, health_timeout = None))]
    fn new(fd: i64, health_timeout: Option<f64>) -> PyResult<Self> {
        check_pid()?;
        let _rt = pyo3_async_runtimes::tokio::get_runtime().enter();
        let health_timeout = health_timeout.map(super::secs);
        Ok(Self(Arc::new(Child {
            client: IpcClient::new(socket(fd)?, health_timeout),
            inferences: Inferences::default(),
            stop: watch::channel(false).0,
        })))
    }

    /// Serve the parent's messages until the channel closes, `host._core_shutdown(reason)`
    /// answers true, or `stop()`. `host` (on the running loop) runs `_core_start_job(fields)`,
    /// `_core_shutdown(reason)`, `_core_inference(method, request_id, data)` and
    /// `_core_dump_stacks()`, each a coroutine. The inference requests still in flight at the end
    /// are cancelled.
    fn serve<'py>(&self, py: Python<'py>, host: Py<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let host = CancellingHandle::new(py, host)?;
        let child = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut stop = child.stop.subscribe();
            let served = async {
                while let Ok(data) = child.client.recv().await {
                    let msg = match IpcMessage::decode(&data) {
                        Ok(msg) => msg,
                        Err(e) => {
                            log::error!(target: "livekit.agents", "Error in _read_ipc_task: {e}");
                            continue;
                        }
                    };
                    if !child.on_message(&host, msg).await? {
                        break;
                    }
                }
                PyResult::Ok(())
            };
            let outcome = tokio::select! {
                r = served => r,
                _ = stop.wait_for(|stopped| *stopped) => Ok(()),
            };
            child.inferences.clear();
            outcome
        })
    }

    /// `serve` returns.
    fn stop(&self) {
        self.0.stop.send_replace(true);
    }

    /// `InferenceExecutor.do_inference`: the parent runs `method` on `data`; raises the
    /// `RuntimeError` of a failed inference, `CancelledError` once serving ended.
    fn do_inference<'py>(
        &self,
        py: Python<'py>,
        method: String,
        data: Cow<[u8]>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let child = self.0.clone();
        let data = data.into_owned();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // unique across the job processes of one worker, whose inference process answers all
            let prefix = format!("inference_job_{:x}", std::process::id());
            let sender = child.clone();
            let send = |msg: IpcMessage| async move { sender.send(msg).await };
            child.inferences.request(&prefix, method, data, send).await
        })
    }

    fn send_bytes<'py>(&self, py: Python<'py>, data: Cow<[u8]>) -> PyResult<Bound<'py, PyAny>> {
        let child = self.0.clone();
        let data = data.into_owned();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            child.client.send(&data).await.map_err(duplex_closed)
        })
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

// ------------------------------------------------------------------------------------ job run

/// How long the entrypoint may take to return once the job shuts down, before it is cancelled.
const ENTRYPOINT_EXIT_TIMEOUT: Duration = Duration::from_secs(15);

/// The entrypoint has this long to connect the room before a warning.
const CONNECT_WARNING_DELAY: Duration = Duration::from_secs(10);

/// The participant attribute a simulator joins with (`types.ATTRIBUTE_SIMULATOR`).
const ATTRIBUTE_SIMULATOR: &str = "lk.simulator";

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
    fake: bool,
    session_end_timeout: f64,
    state: Mutex<JobRunState>,
    ended: watch::Sender<bool>,
    /// Held while the room connects, so a second `connect()` waits for the first.
    connecting: tokio::sync::Mutex<()>,
    /// Set once the job ends: the API calls still running are cancelled.
    tasks_closed: watch::Sender<bool>,
}

/// `msg` on the `livekit.agents` logger, with `exc`'s traceback.
fn py_log(level: Level, msg: &str, exc: Option<PyErr>) {
    Python::attach(|py| {
        let logged = (|| -> PyResult<()> {
            let logger = py.import("logging")?.call_method1("getLogger", ("livekit.agents",))?;
            let kwargs = PyDict::new(py);
            if let Some(exc) = exc {
                kwargs.set_item("exc_info", exc.into_value(py))?;
            }
            logger.call_method("log", (level as i32, msg), Some(&kwargs))?;
            Ok(())
        })();
        if let Err(e) = logged {
            e.print(py);
        }
    })
}

/// `CancelledError`, what a cancelled call raises.
fn cancelled() -> PyErr {
    pyo3::exceptions::asyncio::CancelledError::new_err(())
}

fn is_cancelled(e: &PyErr) -> bool {
    Python::attach(|py| e.is_instance_of::<pyo3::exceptions::asyncio::CancelledError>(py))
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

    async fn call(&self, f: impl for<'py> FnOnce(&Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>) -> PyResult<Py<PyAny>> {
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
            let name: String = entry.bind(py).getattr("__name__").and_then(|n| n.extract()).unwrap_or_default();
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

    fn span(&self, name: &str, callback: Option<&Py<PyAny>>) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let callback = callback.map(|c| c.clone_ref(py));
            Ok(self.host().object().bind(py).call_method1("_core_span", (name, callback))?.unbind())
        })
    }

    fn span_end(&self, span: Py<PyAny>, error: Option<PyErr>, timed_out: bool) {
        let r = Python::attach(|py| -> PyResult<()> {
            let error = error.map(|e| e.into_value(py).into_any());
            self.host().object().bind(py).call_method1("_core_span_end", (span, error, timed_out))?;
            Ok(())
        });
        if let Err(e) = r {
            Python::attach(|py| e.print(py));
        }
    }

    /// One step run inside a span of its own: `error_msg` is logged with a failure.
    async fn spanned(&self, name: &str, method: &'static str, timeout: Option<f64>, error_msg: &str) {
        let span = match self.span(name, None) {
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
                let msg = format!("on_session_end timed out after {}s", timeout.unwrap_or_default() as i64);
                self.log(Level::Error, &msg, None);
            }
        }
    }

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
                        self.log(Level::Warning, "The room connection was not established within 10 seconds after calling job_entry. This might mean that job_ctx.connect() was never invoked, or that no LiveSession with an active RoomIO has been started.", None);
                    }
                }
                _ = ended.wait_for(|e| *e) => break,
            }
        }
        let (reason, user_initiated) = lock(&self.state).shutdown.clone().unwrap_or_default();
        self.call(|h| h.call_method1("_core_shutdown_began", (reason.as_str(), user_initiated))).await?;

        // the entrypoint returns, or is cancelled
        if !entry_done && tokio::time::timeout(ENTRYPOINT_EXIT_TIMEOUT, &mut wait_entry).await.is_err() {
            self.log(Level::Warning, "entrypoint did not exit in time, cancelling", None);
            let entry = Python::attach(|py| entry.clone_ref(py));
            let _ = self.call(move |h| h.call_method1("_core_cancel", (entry,))).await;
        }

        // the bound on `LiveSession.aclose()`: past it, the rest of the shutdown still runs
        let aclose_timeout = Python::attach(|py| -> PyResult<f64> {
            self.host().object().bind(py).getattr("_session_aclose_timeout")?.extract()
        })?;
        let close = self.call(|h| h.call_method0("_core_close_session"));
        match tokio::time::timeout(secs(aclose_timeout), close).await {
            Ok(Err(e)) => Python::attach(|py| e.print(py)),
            Ok(Ok(_)) => {}
            Err(_) => self.log(
                Level::Error,
                &format!(
                    "LiveSession.aclose() timed out after {aclose_timeout:.1}s; proceeding with shutdown so registered callbacks still run."
                ),
                None,
            ),
        }

        let has_session_end = Python::attach(|py| -> PyResult<bool> {
            self.host().object().bind(py).call_method0("_core_has_session_end")?.extract()
        })?;
        if has_session_end {
            self.spanned(
                "on_session_end",
                "_core_session_end",
                Some(self.session_end_timeout),
                "error while executing the on_session_end callback",
            )
            .await;
        }
        self.spanned("session_end_upload", "_core_upload", None, "error in job_ctx._on_session_end").await;
        self.call(|h| h.call_method1("_core_finish", (reason.as_str(), user_initiated))).await?;
        let disconnect = self.span("room_disconnect", None)?;
        let r = self
            .call(|h| h.call_method1("_core_disconnect", (disconnect.clone_ref(h.py()),)))
            .await;
        self.span_end(disconnect, None, false);
        r?;

        // the shutdown callbacks run together; a framework one gets no span of its own
        let callbacks: Vec<(Py<PyAny>, bool)> = Python::attach(|py| {
            self.host().object().bind(py).call_method0("_core_shutdown_callbacks")?.extract()
        })?;
        let mut runs = Vec::new();
        for (callback, framework) in callbacks {
            let inner = self.clone();
            let reason = reason.clone();
            runs.push(tokio::spawn(async move {
                let span = if framework { None } else { Some(inner.span("shutdown_callback", Some(&callback))?) };
                let r = inner
                    .call(|h| {
                        let span = span.as_ref().map(|s| s.clone_ref(h.py()));
                        h.call_method1("_core_shutdown_callback", (callback, reason.as_str(), span))
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
    /// `redaction` is the job's own (`Job.enable_redaction`).
    #[new]
    #[pyo3(signature = (*, fake_job, session_end_timeout, redaction))]
    fn new(fake_job: bool, session_end_timeout: f64, redaction: bool) -> Self {
        Self(Arc::new(JobRunInner {
            host: std::sync::OnceLock::new(),
            fake: fake_job,
            session_end_timeout,
            state: Mutex::new(JobRunState { redaction, ..JobRunState::default() }),
            ended: watch::channel(false).0,
            connecting: tokio::sync::Mutex::new(()),
            tasks_closed: watch::channel(false).0,
        }))
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
    fn add_participant_entrypoint(&self, py: Python<'_>, entrypoint: Py<PyAny>, kinds: Vec<i32>) -> PyResult<()> {
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
    fn participant_joined(&self, py: Python<'_>, ctx: Py<PyAny>, participant: Py<PyAny>, identity: String, kind: i32) -> PyResult<()> {
        self.0.participant_joined(py, &ctx, &participant, &identity, kind)
    }

    /// A participant left: a simulator leaving ends the job.
    fn participant_left(&self, attributes: HashMap<String, String>) {
        if attributes.contains_key(ATTRIBUTE_SIMULATOR) {
            self.0.log(Level::Debug, "simulator disconnected, shutting down the job", None);
            self.0.end("simulation completed".into(), true);
        }
    }

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
            self.0.log(Level::Warning, &format!("job_ctx.{name}() is not executed while in console mode"), None);
            return pyo3_async_runtimes::tokio::future_into_py(py, async move { Ok(console_result) });
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move { rx.await.unwrap_or_else(|_| Err(cancelled())) })
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
    fn init_recording(&self, options: HashMap<String, bool>, observability: bool) -> PyResult<Option<bool>> {
        let get = |key: &str, default: bool| options.get(key).copied().unwrap_or(default);
        let mut st = lock(&self.0.state);
        let redaction = st.redaction || get("redaction", false);
        if redaction && get("audio", true) && !get("transcript", true) {
            return Err(PyValueError::new_err("audio upload requires transcript upload when redaction is enabled"));
        }
        if st.recording_initialized {
            return Ok(None);
        }
        st.recording_initialized = true;
        st.redaction = redaction;
        let uploads = ["traces", "logs", "audio", "transcript"].iter().any(|key| get(key, true));
        Ok((uploads && observability).then_some(redaction))
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
