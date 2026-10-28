//! Runtime-owned host adapter, compiled by lk-agents-runtime through #[path].
//! `_core.Worker` and `_core.EndpointStream`: the worker as Python drives it (plan 13.4).
//!
//! `Worker(options, host)` owns the `/agent` connection; `host.open(kind, args)` is the one
//! callback Rust makes, in the generator's shape: it returns an awaitable, scheduled on the event
//! loop that constructed the worker, and cancelled when Rust drops the operation. The kinds:
//!
//! | kind              | args                                   | result                      |
//! |-------------------|----------------------------------------|-----------------------------|
//! | `"job_request"`   | a `JobRequest` dict, for `on_request`  | a decision dict             |
//! | `"launch_job"`    | a `RunningJobInfo` dict                | the LiveKit job id          |
//! | `"terminate_job"` | `job_id`                           | ignored                     |
//! | `"load"`          | `{}`, for a `load_fnc`                 | a float                     |
//! | `"endpoint"`      | the preamble fields and an `EndpointStream` | ignored; serve to completion |
//!
//! Python owns process supervision, signals and the ASGI bridge; Rust owns every socket.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use lk_agents_abi::python::{
    CancelPythonTask, capture_host, check_pid, pyo3_async_runtimes, schedule_coroutine,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_async_runtimes::TaskLocals;

use std::sync::atomic::{AtomicBool, Ordering};

use crate::worker::transport::{
    AgentEndpointKind, EndpointControl, EndpointReader, EndpointWriter, HttpStreamResetCode,
};
use crate::worker::{
    EndpointRequest, EnvOption, HEARTBEAT_INTERVAL, HostError, JobDecision, JobId, TransportConfig,
    WorkerCommand, WorkerHost, protocol,
};

// --------------------------------------------------------------------- EndpointStream

type Half<T> = Arc<tokio::sync::Mutex<Option<T>>>;

/// One exchange stream after its preamble, as Python reads and writes it. The two directions are
/// independent: a `read` and a `write` may be in flight at once. Dropping it without `finish`
/// resets with code 0 (`HSR_ABORT`).
#[pyclass(name = "EndpointStream", module = "livekit.agents._core")]
pub struct PyEndpointStream {
    reader: Half<EndpointReader>,
    writer: Half<EndpointWriter>,
    control: EndpointControl,
}

/// The largest single `read`; a body streams through many.
const MAX_READ: usize = 1 << 20;

fn reset_code(code: i32) -> PyResult<HttpStreamResetCode> {
    HttpStreamResetCode::try_from(code)
        .map_err(|_| PyValueError::new_err(format!("unknown HttpStreamResetCode {code}")))
}

#[pymethods]
impl PyEndpointStream {
    /// Up to `max_bytes` of the peer's bytes; `b""` once the peer half-closed (FIN). Raises when
    /// the peer reset the stream.
    #[pyo3(signature = (max_bytes = 65536))]
    fn read<'py>(&self, py: Python<'py>, max_bytes: usize) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let reader = self.reader.clone();
        let max_bytes = max_bytes.clamp(1, MAX_READ);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = reader.lock().await;
            let Some(reader) = guard.as_mut() else {
                return Err(PyRuntimeError::new_err("endpoint stream is closed"));
            };
            let mut buf = vec![0u8; max_bytes];
            match reader.read(&mut buf).await {
                Ok(Some(n)) => {
                    buf.truncate(n);
                    Ok(buf)
                }
                Ok(None) => Ok(Vec::new()),
                Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
            }
        })
        .map(|awaitable| awaitable.into_any())
    }

    /// Writes all of `data`, waiting on QUIC flow control.
    fn write<'py>(&self, py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let writer = self.writer.clone();
        let data = data.to_vec();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = writer.lock().await;
            let Some(writer) = guard.as_mut() else {
                return Err(PyRuntimeError::new_err("endpoint stream is closed"));
            };
            writer.write_all(&data).await.map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// FIN: the response is complete. Idempotent; a stream already reset raises.
    fn finish(&self) -> PyResult<()> {
        check_pid()?;
        self.control.finish();
        if let Ok(mut writer) = self.writer.try_lock() {
            if let Some(writer) = writer.as_mut() {
                writer
                    .apply_control()
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            }
        }
        Ok(())
    }

    /// RESET_STREAM and STOP_SENDING, admitted independently of pending data I/O.
    fn reset(&self, code: i32) -> PyResult<()> {
        check_pid()?;
        self.control.reset(reset_code(code)?);
        if let Ok(mut writer) = self.writer.try_lock() {
            if let Some(writer) = writer.as_mut() {
                let _ = writer.apply_control();
            }
        }
        self.apply_stop();
        Ok(())
    }

    /// STOP_SENDING only: the response went out whole and nothing more is wanted.
    fn stop(&self, code: i32) -> PyResult<()> {
        check_pid()?;
        self.control.stop(reset_code(code)?);
        self.apply_stop();
        Ok(())
    }
}

impl PyEndpointStream {
    fn apply_stop(&self) {
        if let Ok(mut reader) = self.reader.try_lock() {
            if let Some(reader) = reader.as_mut() {
                reader.apply_control();
            }
        }
    }
}

// --------------------------------------------------------------------- EndpointServer

use crate::worker::endpoint::{
    self, AppError, AsgiApp, Conn as AsgiConn, Received, RequestInfo, Scope, Sent, StreamIo,
};
use lk_agents_abi::python::{CancellingHandle, HostBridge};
use pyo3::types::PyBytes;

/// `_core.EndpointStream` as the exchange reads and writes it.
struct NativeStream {
    reader: Half<EndpointReader>,
    writer: Half<EndpointWriter>,
    control: EndpointControl,
}

#[async_trait]
impl StreamIo for NativeStream {
    async fn read(&self, max: usize) -> Result<Vec<u8>, String> {
        let mut guard = self.reader.lock().await;
        let reader = guard.as_mut().ok_or("endpoint stream is closed")?;
        let mut buf = vec![0u8; max.clamp(1, MAX_READ)];
        match reader.read(&mut buf).await.map_err(|e| e.to_string())? {
            Some(n) => {
                buf.truncate(n);
                Ok(buf)
            }
            None => Ok(Vec::new()),
        }
    }

    async fn write(&self, data: &[u8]) -> Result<(), String> {
        let mut guard = self.writer.lock().await;
        let writer = guard.as_mut().ok_or("endpoint stream is closed")?;
        writer.write_all(data).await.map_err(|e| e.to_string())
    }

    async fn finish(&self) {
        self.control.finish();
        if let Ok(mut writer) = self.writer.try_lock() {
            if let Some(writer) = writer.as_mut() {
                let _ = writer.apply_control();
            }
        }
    }

    async fn reset(&self, code: HttpStreamResetCode) {
        self.control.reset(code);
        if let Ok(mut writer) = self.writer.try_lock() {
            if let Some(writer) = writer.as_mut() {
                let _ = writer.apply_control();
            }
        }
        self.stop(code).await;
    }

    async fn stop(&self, code: HttpStreamResetCode) {
        self.control.stop(code);
        if let Ok(mut reader) = self.reader.try_lock() {
            if let Some(reader) = reader.as_mut() {
                reader.apply_control();
            }
        }
    }
}

/// Any object with `EndpointStream`'s five methods (a test's fake): the awaitables are awaited
/// and the plain calls made on its loop, in order.
struct HostStream(CancellingHandle);

impl HostStream {
    fn soon(&self, method: &'static str, code: Option<i32>) {
        let scheduled = Python::attach(|py| -> PyResult<()> {
            let f = self.0.object().bind(py).getattr(method)?;
            let event_loop = self.0.task_locals().event_loop(py);
            match code {
                Some(c) => event_loop.call_method1("call_soon_threadsafe", (f, c))?,
                None => event_loop.call_method1("call_soon_threadsafe", (f,))?,
            };
            Ok(())
        });
        if let Err(e) = scheduled {
            Python::attach(|py| e.print(py));
        }
    }
}

#[async_trait]
impl StreamIo for HostStream {
    async fn read(&self, max: usize) -> Result<Vec<u8>, String> {
        let v = self
            .0
            .await_call(|s| s.call_method1("read", (max,)))
            .await
            .map_err(|e| e.to_string())?;
        Python::attach(|py| v.extract::<Vec<u8>>(py)).map_err(|e| e.to_string())
    }

    async fn write(&self, data: &[u8]) -> Result<(), String> {
        let data = data.to_vec();
        self.0
            .await_call(move |s| s.call_method1("write", (PyBytes::new(s.py(), &data),)))
            .await
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    async fn finish(&self) {
        self.soon("finish", None);
    }

    async fn reset(&self, code: HttpStreamResetCode) {
        self.soon("reset", Some(code as i32));
    }

    async fn stop(&self, code: HttpStreamResetCode) {
        self.soon("stop", Some(code as i32));
    }
}

/// The ASGI application, called on its loop; `request` is `request.state.lk_request`.
struct PyApp {
    app: CancellingHandle,
    request: Py<PyAny>,
}

fn headers_py<'py>(py: Python<'py>, headers: &endpoint::Headers) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, headers.iter().map(|(k, v)| (PyBytes::new(py, k), PyBytes::new(py, v))))
}

fn scope_py<'py>(
    py: Python<'py>,
    scope: &Scope,
    request: &Py<PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let asgi = PyDict::new(py);
    asgi.set_item("version", "3.0")?;
    asgi.set_item("spec_version", "2.3")?;
    d.set_item("type", scope.kind)?;
    d.set_item("asgi", asgi)?;
    d.set_item("http_version", "1.1")?;
    if scope.kind == "http" {
        d.set_item("method", &scope.method)?;
    }
    d.set_item("scheme", &scope.scheme)?;
    d.set_item("path", &scope.path)?;
    d.set_item("raw_path", PyBytes::new(py, &scope.raw_path))?;
    d.set_item("query_string", PyBytes::new(py, &scope.query_string))?;
    d.set_item("root_path", "")?;
    d.set_item("headers", headers_py(py, &scope.headers)?)?;
    d.set_item("client", scope.client.clone())?;
    d.set_item("server", py.None())?;
    if scope.kind == "websocket" {
        d.set_item("subprotocols", scope.subprotocols.clone())?;
    }
    let state = PyDict::new(py);
    state.set_item("lk_request", request.bind(py))?;
    d.set_item("state", state)?;
    Ok(d)
}

fn received_py(py: Python<'_>, msg: Received) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    match msg {
        Received::Request { body, more_body } => {
            d.set_item("type", "http.request")?;
            d.set_item("body", PyBytes::new(py, &body))?;
            d.set_item("more_body", more_body)?;
        }
        Received::Disconnect => d.set_item("type", "http.disconnect")?,
        Received::WsConnect => d.set_item("type", "websocket.connect")?,
        Received::WsText(t) => {
            d.set_item("type", "websocket.receive")?;
            d.set_item("text", t)?;
        }
        Received::WsBytes(b) => {
            d.set_item("type", "websocket.receive")?;
            d.set_item("bytes", PyBytes::new(py, &b))?;
        }
        Received::WsDisconnect { code } => {
            d.set_item("type", "websocket.disconnect")?;
            d.set_item("code", code)?;
        }
    }
    Ok(d)
}

/// `bytes(x)`: ASGI headers and bodies may be any bytes-like object.
fn bytes_of(v: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    v.py().import("builtins")?.getattr("bytes")?.call1((v,))?.extract()
}

fn sent_from_py(m: &Bound<'_, PyAny>) -> PyResult<Sent> {
    let get = |k: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
        let v = m.call_method1("get", (k,))?;
        Ok((!v.is_none()).then_some(v))
    };
    let headers = || -> PyResult<endpoint::Headers> {
        let Some(h) = get("headers")? else { return Ok(Vec::new()) };
        h.try_iter()?
            .map(|kv| {
                let (k, v): (Bound<'_, PyAny>, Bound<'_, PyAny>) = kv?.extract()?;
                Ok((bytes_of(&k)?, bytes_of(&v)?))
            })
            .collect()
    };
    let body = || -> PyResult<Vec<u8>> { get("body")?.map_or(Ok(Vec::new()), |b| bytes_of(&b)) };
    let more_body = || -> PyResult<bool> { get("more_body")?.map_or(Ok(false), |b| b.is_truthy()) };
    let kind: String = m.get_item("type")?.extract()?;
    Ok(match kind.as_str() {
        "http.response.start" => {
            Sent::ResponseStart { status: m.get_item("status")?.extract()?, headers: headers()? }
        }
        "http.response.body" => Sent::ResponseBody { body: body()?, more_body: more_body()? },
        "http.response.trailers" => Sent::ResponseTrailers,
        "websocket.accept" => Sent::WsAccept {
            subprotocol: get("subprotocol")?.map(|s| s.extract()).transpose()?,
            headers: headers()?,
        },
        "websocket.send" => match get("bytes")? {
            Some(b) => Sent::WsBytes(bytes_of(&b)?),
            None => Sent::WsText(get("text")?.map_or(Ok(String::new()), |t| t.extract())?),
        },
        "websocket.close" => Sent::WsClose {
            code: get("code")?.map_or(Ok(1000), |c| c.extract())?,
            reason: get("reason")?.map_or(Ok(String::new()), |r| r.extract())?,
        },
        "websocket.http.response.start" => Sent::WsHttpResponseStart {
            status: m.get_item("status")?.extract()?,
            headers: headers()?,
        },
        "websocket.http.response.body" => {
            Sent::WsHttpResponseBody { body: body()?, more_body: more_body()? }
        }
        _ => Sent::Unknown(kind),
    })
}

/// The ASGI `receive` callable.
#[pyclass(frozen)]
struct AsgiReceive(AsgiConn);

#[pymethods]
impl AsgiReceive {
    fn __call__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let conn = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let msg = conn.receive().await;
            Python::attach(|py| received_py(py, msg).map(Bound::unbind))
        })
    }
}

/// The ASGI `send` callable.
#[pyclass(frozen)]
struct AsgiSend(AsgiConn);

#[pymethods]
impl AsgiSend {
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        message: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let msg = sent_from_py(message)?;
        let conn = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            conn.send(msg).await.map_err(PyRuntimeError::new_err)
        })
    }
}

#[async_trait]
impl AsgiApp for PyApp {
    async fn call(&self, scope: Scope, conn: AsgiConn) -> Result<(), AppError> {
        let request = Python::attach(|py| self.request.clone_ref(py));
        self.app
            .await_call(move |app| {
                let scope = scope_py(app.py(), &scope, &request)?;
                app.call1((scope, AsgiReceive(conn.clone()), AsgiSend(conn)))
            })
            .await
            .map(|_| ())
            .map_err(|e| {
                Python::attach(|py| AppError {
                    summary: format!(
                        "{}: {}",
                        e.get_type(py).qualname().map(|n| n.to_string()).unwrap_or_default(),
                        e.value(py)
                    ),
                    traceback: e.traceback(py).and_then(|tb| tb.format().ok()).unwrap_or_default(),
                })
            })
    }
}

/// `AgentHttp`'s server half: each `open("endpoint", ..)` stream served with the ASGI app
/// (`endpoint.rs`), and the open WebSockets it drains.
#[pyclass(name = "EndpointServer", module = "livekit.agents._core")]
#[derive(Default)]
pub struct PyEndpointServer {
    websockets: Arc<endpoint::Registry>,
}

#[pymethods]
impl PyEndpointServer {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Serve `stream` (an `EndpointStream` after its preamble) with `app` to completion;
    /// `request` is the `EndpointRequest` the handler sees as `request.state.lk_request`.
    fn serve<'py>(
        &self,
        py: Python<'py>,
        stream: &Bound<'py, PyAny>,
        app: Py<PyAny>,
        request: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let info: RequestInfo = request.extract()?;
        let io: Arc<dyn StreamIo> = match stream.cast::<PyEndpointStream>() {
            Ok(s) => {
                let s = s.borrow();
                Arc::new(NativeStream {
                    reader: s.reader.clone(),
                    writer: s.writer.clone(),
                    control: s.control.clone(),
                })
            }
            Err(_) => Arc::new(HostStream(CancellingHandle::new(py, stream.clone().unbind())?)),
        };
        let app = PyApp { app: CancellingHandle::new(py, app)?, request: request.clone().unbind() };
        let registry = self.websockets.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            endpoint::serve(io, info, &app, &registry).await;
            Ok(())
        })
    }

    /// A close frame to every open WebSocket, so clients reconnect to another worker.
    #[pyo3(signature = (code = 1001, reason = "worker draining".to_owned()))]
    fn close_websockets<'py>(
        &self,
        py: Python<'py>,
        code: u16,
        reason: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let registry = self.websockets.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            registry.close_all(code, &reason).await;
            Ok(())
        })
    }
}

lk_agents_abi::register_host!(class PyEndpointServer);

// --------------------------------------------------------------------- the host

#[derive(Clone)]
struct PyWorkerHost {
    obj: Arc<Py<PyAny>>,
    locals: TaskLocals,
    /// The user gave a `load_fnc` the worker may call (not on Cloud).
    user_load: bool,
    /// The server has a voice door, and that door an `on_request`.
    has_rtc: bool,
    has_on_request: bool,
}

impl PyWorkerHost {
    /// `host.open(kind, args)` as a scheduled coroutine; dropping the operation cancels the task.
    async fn open<F>(&self, kind: &'static str, args: F) -> Result<Py<PyAny>, HostError>
    where
        F: for<'py> FnOnce(Python<'py>) -> PyResult<Bound<'py, PyAny>>,
    {
        let (task, receiver): (CancelPythonTask, _) = Python::attach(|py| -> PyResult<_> {
            let coro = self.obj.bind(py).call_method1("open", (kind, args(py)?))?;
            schedule_coroutine(py, &self.locals, coro)
        })
        .map_err(|e| Python::attach(|py| capture_host(py, &e)))?;
        let result = receiver
            .await
            .map_err(|_| HostError::new(format!("open({kind:?}): Python task disappeared")))?;
        drop(task);
        result.map_err(|e| Python::attach(|py| capture_host(py, &e)))
    }
}

#[async_trait]
impl WorkerHost for PyWorkerHost {
    async fn on_job_request(&self, req: crate::worker::JobRequest) -> JobDecision {
        if let Some(decision) = server::own_decision(&req, self.has_rtc, self.has_on_request) {
            return decision;
        }
        let job_id = req.job.id.clone();
        // `resuming` is set when a call migrates off a draining worker.
        let res = self.open("job_request", |py| Ok(req.into_pyobject(py)?.into_any())).await;
        match res {
            // A failed `request_fnc` rejects without terminating, so the server can redispatch.
            Err(e) => JobDecision::Failed(e),
            Ok(v) => Python::attach(|py| parse_decision(py, v.bind(py)))
                .unwrap_or_else(|e| JobDecision::Failed(HostError::new(format!("{job_id}: {e}")))),
        }
    }

    async fn launch_job(&self, info: crate::worker::RunningJobInfo) -> Result<JobId, HostError> {
        let v = self.open("launch_job", |py| Ok(info.into_pyobject(py)?.into_any())).await?;
        Python::attach(|py| v.extract::<String>(py))
            .map_err(|e| HostError::new(format!("launch_job returned a non-str job id: {e}")))
    }

    async fn terminate_job(&self, id: JobId) -> Result<(), HostError> {
        self.open("terminate_job", |py| Ok(id.into_pyobject(py)?.into_any())).await.map(|_| ())
    }

    async fn user_load(&self) -> Option<f32> {
        if !self.user_load {
            return None;
        }
        // A raising `load_fnc` must not read as idle (that over-schedules); `None` falls back to
        // the CPU monitor.
        let v = self.open("load", |py| Ok(PyDict::new(py).into_any())).await.ok()?;
        Python::attach(|py| v.extract::<Option<f32>>(py).ok()).flatten()
    }

    async fn serve_endpoint(&self, request: EndpointRequest) {
        let EndpointRequest { preamble, stream } = request;
        let request_id = preamble.request_id.clone();
        let (writer, reader) = stream.split();
        let control = writer.control();
        let reader: Half<EndpointReader> = Arc::new(tokio::sync::Mutex::new(Some(reader)));
        let writer: Half<EndpointWriter> = Arc::new(tokio::sync::Mutex::new(Some(writer)));
        let halves = (reader.clone(), writer.clone());
        let kind = match AgentEndpointKind::try_from(preamble.kind) {
            Ok(AgentEndpointKind::AekText) => "text",
            _ => "http",
        };
        let entered = std::sync::atomic::AtomicBool::new(false);
        let outcome = self
            .open("endpoint", |py| {
                let d = PyDict::new(py);
                d.set_item("stream", PyEndpointStream { reader, writer, control })?;
                d.set_item("kind", kind)?;
                d.set_item("request_id", &preamble.request_id)?;
                d.set_item("authorized", preamble.authorized)?;
                d.set_item("route", &preamble.route)?;
                d.set_item("timeout_ms", preamble.timeout_ms)?;
                d.set_item("client_addr", &preamble.client_addr)?;
                d.set_item("scheme", &preamble.scheme)?;
                entered.store(true, std::sync::atomic::Ordering::Relaxed);
                Ok(d.into_any())
            })
            .await;
        if let Err(error) = outcome {
            // `open` itself raising means Python never held the stream: refused is sound. A
            // coroutine that raised may have run the application: internal.
            let code = if entered.load(std::sync::atomic::Ordering::Relaxed) {
                HttpStreamResetCode::HsrInternal
            } else {
                HttpStreamResetCode::HsrRefused
            };
            crate::worker::diagnostics::emit(
                log::Level::Warn,
                format!(
                    "endpoint request failed in the host request_id={request_id} error={error}"
                ),
            );
            let (reader, writer) = halves;
            if let Some(w) = writer.lock().await.as_mut() {
                w.reset(code);
            }
            if let Some(r) = reader.lock().await.as_mut() {
                r.stop(code);
            }
        }
    }
}

/// The host's answer to `open("job_request", ..)` as a [`JobDecision`]. A malformed dict becomes an
/// error, which the caller turns into `JobDecision::Failed` (reject without terminating).
fn parse_decision(py: Python<'_>, v: &Bound<'_, PyAny>) -> Result<JobDecision, String> {
    let d = v.cast::<PyDict>().map_err(|_| "expected a dict".to_string())?;
    let host_err = |e: PyErr| capture_host(py, &e).to_string();
    let raw: crate::worker::RawJobDecision = d.as_any().extract().map_err(host_err)?;
    JobDecision::from_wire(
        &raw.decision,
        || d.as_any().extract::<crate::worker::AcceptArguments>().map_err(host_err),
        || raw.error.clone().unwrap_or_default(),
    )
}

// --------------------------------------------------------------------- the routes

/// One manifest entry: `{"path", "methods", "kind", "public"}`, with `kind` `"http"` (default) or
/// `"text"` for a session endpoint (plan 9.1).
fn endpoint_from_py(item: &Bound<'_, PyAny>) -> PyResult<protocol::AgentEndpoint> {
    let d =
        item.cast::<PyDict>().map_err(|_| PyValueError::new_err("endpoints: expected dicts"))?;
    let path: String = d
        .get_item("path")?
        .ok_or_else(|| PyValueError::new_err("endpoints: an entry has no path"))?
        .extract()?;
    let methods: Vec<String> = match d.get_item("methods")? {
        Some(v) => v.extract()?,
        None => Vec::new(),
    };
    let kind = match d.get_item("kind")?.map(|v| v.extract::<String>()).transpose()?.as_deref() {
        None | Some("http") => AgentEndpointKind::AekHttp,
        Some("text") => AgentEndpointKind::AekText,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "endpoints: {path}: unknown kind {other:?}, expected \"http\" or \"text\""
            )));
        }
    };
    let public = match d.get_item("public")? {
        Some(v) => v.extract()?,
        None => false,
    };
    Ok(protocol::AgentEndpoint { path, methods, kind: kind as i32, public })
}

fn env_flag(name: &str) -> bool {
    matches!(std::env::var(name).unwrap_or_default().to_lowercase().as_str(), "1" | "true" | "yes")
}

/// The transport the routes need: the endpoint data plane rides WebTransport; without routes the
/// WebSocket control connection suffices (`LIVEKIT_WORKER_WEBTRANSPORT` asks for it anyway).
fn transport_for(routes: bool) -> TransportConfig {
    if routes || env_flag("LIVEKIT_WORKER_WEBTRANSPORT") {
        TransportConfig::WebTransport { insecure: env_flag("LIVEKIT_WORKER_INSECURE") }
    } else {
        TransportConfig::WebSocket { heartbeat: HEARTBEAT_INTERVAL }
    }
}

// --------------------------------------------------------------------- ServerConfig

use crate::worker::server::{self, ServerSettings};

/// `AgentServer`'s options (`_core.ServerConfig`): what it was given, with the environment read
/// where an option was left empty. Writable until the server starts.
#[pyclass(name = "ServerConfig", module = "livekit.agents._core")]
pub struct PyServerConfig {
    settings: Arc<std::sync::Mutex<ServerSettings>>,
    running: Arc<AtomicBool>,
}

impl PyServerConfig {
    fn get(&self) -> std::sync::MutexGuard<'_, ServerSettings> {
        self.settings.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[pymethods]
impl PyServerConfig {
    /// `options`: `AgentServer`'s arguments by name; `http_proxy` left out reads `HTTPS_PROXY`,
    /// else `HTTP_PROXY`.
    #[new]
    fn new(options: lk_agents_abi::worker::worker::ServerOptions) -> PyResult<Self> {
        let mut s = ServerSettings::default();
        let env = |name: &str| std::env::var(name).ok();
        s.http_proxy = ServerSettings::env_proxy(env);
        s.apply(options).map_err(PyValueError::new_err)?;
        Ok(PyServerConfig {
            settings: Arc::new(std::sync::Mutex::new(s.with_env(env))),
            running: Arc::default(),
        })
    }

    /// `update_options`: the given options, before the server starts.
    fn update(&self, options: lk_agents_abi::worker::worker::ServerOptions) -> PyResult<()> {
        if self.running.load(Ordering::Acquire) {
            return Err(PyRuntimeError::new_err("cannot update options after starting the server"));
        }
        self.get().apply(options).map_err(PyValueError::new_err)
    }

    #[getter]
    fn ws_url(&self) -> String {
        self.get().ws_url.clone()
    }

    #[getter]
    fn api_key(&self) -> String {
        self.get().api_key.clone()
    }

    #[getter]
    fn api_secret(&self) -> String {
        self.get().api_secret.clone()
    }

    #[getter]
    fn agent_name(&self) -> String {
        self.get().agent_name.clone()
    }

    #[getter]
    fn deployment(&self) -> String {
        self.get().deployment.clone()
    }

    /// Hosted on Cloud (`LIVEKIT_WORKER_TOKEN` set).
    #[getter]
    fn cloud(&self) -> bool {
        self.get().cloud()
    }

    #[getter]
    fn drain_timeout(&self) -> f64 {
        self.get().drain_timeout
    }

    #[getter]
    fn simulation(&self) -> bool {
        self.get().simulation
    }

    #[setter]
    fn set_simulation(&self, value: bool) {
        self.get().simulation = value;
    }
}

lk_agents_abi::register_host!(class PyServerConfig);

// --------------------------------------------------------------------- ServerRuntime

fn empty(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    Ok(PyDict::new(py).into_any())
}

#[async_trait]
impl crate::worker::server_driver::ServerHost for PyWorkerHost {
    fn configured(&self, has_rtc: bool, has_on_request: bool, user_load: bool) -> Self {
        Self { has_rtc, has_on_request, user_load, ..self.clone() }
    }

    async fn start(
        &self,
        start: ServerSettings,
        devmode: bool,
        has_rtc: bool,
    ) -> Result<(), String> {
        self.open("start", move |py| {
            use lk_agents_abi::worker::worker::{ProcessPoolOptions, ProcessStartOptions};
            Ok(ProcessStartOptions {
                devmode,
                job_processes: has_rtc,
                env: start
                    .child_env()
                    .into_iter()
                    .map(|(key, value)| (key.into(), value))
                    .collect(),
                job_executor_type: start.job_executor_type.clone(),
                multiprocessing_context: start.multiprocessing_context.clone(),
                pool: ProcessPoolOptions {
                    num_idle_processes: start
                        .num_idle_processes
                        .unwrap_or(EnvOption::Fixed(0))
                        .get(devmode),
                    initialize_timeout: start.initialize_process_timeout,
                    close_timeout: start.shutdown_process_timeout,
                    session_end_timeout: start.session_end_timeout,
                    memory_warn_mb: start.job_memory_warn_mb,
                    memory_limit_mb: start.job_memory_limit_mb,
                    http_proxy: start.http_proxy.filter(|proxy| !proxy.is_empty()),
                },
            }
            .into_pyobject(py)?
            .into_any())
        })
        .await
        .map(|_| ())
        .map_err(|error| error.to_string())
    }

    async fn started(&self) -> Result<(), String> {
        self.open("started", empty).await.map(|_| ()).map_err(|error| error.to_string())
    }

    async fn registered(&self, info: crate::worker::WorkerRegistered) -> Result<(), String> {
        self.open("registered", move |py| Ok(info.into_pyobject(py)?.into_any()))
            .await
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    async fn idle_processes(&self, target: u32) -> Result<(), String> {
        self.open("idle_processes", move |py| Ok(target.into_pyobject(py)?.into_any()))
            .await
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    async fn worker_load(&self, load: f32) -> Result<(), String> {
        self.open("worker_load", move |py| Ok(load.into_pyobject(py)?.into_any()))
            .await
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    async fn close_websockets(&self) -> Result<(), String> {
        self.open("close_websockets", empty).await.map(|_| ()).map_err(|error| error.to_string())
    }

    async fn join_jobs(&self) -> Result<(), String> {
        self.open("join_jobs", empty).await.map(|_| ()).map_err(|error| error.to_string())
    }

    async fn close(&self) -> Result<(), String> {
        self.open("close", empty).await.map(|_| ()).map_err(|error| error.to_string())
    }

    async fn active_jobs(&self) -> usize {
        match self.open("health", empty).await {
            Ok(value) => Python::attach(|py| value.extract::<usize>(py)).unwrap_or(0),
            Err(_) => 0,
        }
    }

    async fn metrics(&self) -> Option<(String, String)> {
        let value = self.open("metrics", empty).await.ok()?;
        Python::attach(|py| value.extract::<(String, String)>(py).ok())
    }
}

#[pyclass(name = "ServerRuntime", module = "livekit.agents._core")]
pub struct PyServer {
    settings: Arc<std::sync::Mutex<ServerSettings>>,

    /// Seals configuration when the server starts; lifecycle lives in `ServerLifecycle`.
    running: Arc<AtomicBool>,
    runtime: crate::worker::server_driver::ServerRuntime<PyWorkerHost>,
}

impl PyServer {
    fn settings(&self) -> ServerSettings {
        self.settings.lock().unwrap_or_else(|error| error.into_inner()).clone()
    }

    fn id(&self) -> String {
        self.runtime.snapshot().worker_id
    }
}

#[pymethods]
impl PyServer {
    #[new]
    fn new(py: Python<'_>, config: &PyServerConfig, host: Py<PyAny>) -> PyResult<Self> {
        check_pid()?;
        if !host.bind(py).hasattr("open")? {
            return Err(PyValueError::new_err("host object has no open(kind, args)"));
        }
        let locals = TaskLocals::with_running_loop(py)?.copy_context(py)?;
        let host = PyWorkerHost {
            obj: Arc::new(host),
            locals,
            user_load: true,
            has_rtc: false,
            has_on_request: false,
        };
        Ok(Self {
            settings: config.settings.clone(),
            running: config.running.clone(),
            runtime: crate::worker::server_driver::ServerRuntime::new(host, None),
        })
    }

    #[getter]
    fn worker_id(&self) -> String {
        self.id()
    }

    #[getter]
    fn draining(&self) -> bool {
        self.runtime.snapshot().phase == crate::worker::server_lifecycle::ServerPhase::Draining
    }

    #[getter]
    fn http_port(&self) -> u16 {
        self.runtime.snapshot().http_port
    }

    #[pyo3(signature = (*, devmode, unregistered, has_rtc, has_text_doors, publisher, routes, has_load_fnc, has_on_request))]
    #[allow(clippy::too_many_arguments)]
    fn run<'py>(
        &self,
        py: Python<'py>,
        devmode: bool,
        unregistered: bool,
        has_rtc: bool,
        has_text_doors: bool,
        publisher: bool,
        routes: &Bound<'py, PyList>,
        has_load_fnc: bool,
        has_on_request: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let endpoints =
            routes.iter().map(|item| endpoint_from_py(&item)).collect::<PyResult<Vec<_>>>()?;
        let start = crate::worker::server_driver::ServerStart {
            settings: self.settings(),
            devmode,
            unregistered,
            has_rtc,
            has_text_doors,
            publisher,
            transport: transport_for(!endpoints.is_empty()),
            endpoints,
            has_load_fnc,
            has_on_request,
            worker_url: std::env::var("LIVEKIT_WORKER_URL").ok().filter(|url| !url.is_empty()),
        };
        self.running.store(true, Ordering::Release);
        let runtime = self.runtime.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            runtime.run(start).await.map_err(PyRuntimeError::new_err)
        })
    }

    #[pyo3(signature = (timeout=None))]
    fn drain<'py>(&self, py: Python<'py>, timeout: Option<f64>) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let timeout = timeout
            .filter(|timeout| *timeout != 0.0)
            .map(Duration::try_from_secs_f64)
            .transpose()
            .map_err(|_| PyValueError::new_err("drain timeout must be finite and nonnegative"))?;
        let runtime = self.runtime.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            runtime.drain(timeout).await.map_err(|error| {
                if error == "drain timed out" {
                    pyo3::exceptions::PyTimeoutError::new_err(error)
                } else {
                    PyRuntimeError::new_err(error)
                }
            })
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let runtime = self.runtime.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            runtime.close().await.map_err(PyRuntimeError::new_err)
        })
    }

    fn job_status(&self, job_id: String, status: &str) {
        let status = match status {
            "failed" => protocol::JobStatus::JsFailed,
            "success" => protocol::JobStatus::JsSuccess,
            _ => protocol::JobStatus::JsRunning,
        };
        if let Some(worker) = self.runtime.snapshot().worker {
            if worker.send(WorkerCommand::JobStatus { job_id: job_id.clone(), status }).is_err() {
                crate::worker::diagnostics::emit(
                    log::Level::Error,
                    format!("worker job status admission failed job_id={job_id}"),
                );
            }
        }
    }

    #[pyo3(signature = (room, *, fake_job, agent_identity=None, room_info=None, token=None))]
    fn simulate_job<'py>(
        &self,
        py: Python<'py>,
        room: String,
        fake_job: bool,
        agent_identity: Option<String>,
        room_info: Option<Vec<u8>>,
        token: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        if !self.runtime.snapshot().has_rtc {
            return Err(PyRuntimeError::new_err("a server with no rtc_session runs no job"));
        }
        if self.runtime.snapshot().phase != crate::worker::server_lifecycle::ServerPhase::Running {
            return Err(PyRuntimeError::new_err("the server is not accepting jobs"));
        }
        let room_info = room_info
            .map(|bytes| server::decode_room(&bytes))
            .transpose()
            .map_err(PyValueError::new_err)?;
        let info = server::simulated_job(
            &self.settings(),
            &self.id(),
            &room,
            fake_job,
            agent_identity,
            room_info,
            token,
        )
        .map_err(PyValueError::new_err)?;
        let runtime = self.runtime.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            runtime.simulate_job(info).await.map_err(PyRuntimeError::new_err)
        })
    }
}

lk_agents_abi::register_host!(class PyServer);
lk_agents_abi::register_host!(class PyEndpointStream);

#[cfg(test)]
mod tests {
    use super::*;

    fn attach<T>(f: impl FnOnce(Python<'_>) -> T) -> T {
        Python::initialize();
        Python::attach(f)
    }

    #[test]
    fn typed_permissions_record_reaches_configuration() {
        attach(|py| {
            let module = py.import("livekit.agents._generated.worker_options").unwrap();
            let fields = PyDict::new(py);
            fields.set_item("hidden", true).unwrap();
            fields.set_item("can_publish", false).unwrap();
            fields.set_item("can_publish_sources", vec![1, 2]).unwrap();
            let permissions =
                module.getattr("WorkerPermissions").unwrap().call((), Some(&fields)).unwrap();
            let options = PyDict::new(py);
            options.set_item("permissions", permissions).unwrap();
            let options =
                module.getattr("ServerOptions").unwrap().call((), Some(&options)).unwrap();
            let mut config = ServerSettings::default();
            config.apply(options.extract().unwrap()).unwrap();
            assert!(config.permissions.hidden);
            assert!(!config.permissions.can_publish);
            assert_eq!(config.permissions.can_publish_sources, vec![1, 2]);
            assert!(config.permissions.can_subscribe && config.permissions.can_publish_data);
        });
    }

    // Catches: an acceptance field lost on the way in, a missing one refused rather than
    // defaulted, or a wrong-typed one read as empty.
    #[test]
    fn an_accept_decision_carries_its_fields_defaults_missing_ones_and_refuses_wrong_types() {
        attach(|py| {
            let d = PyDict::new(py);
            d.set_item("decision", "accept").unwrap();
            d.set_item("name", "agent-7").unwrap();
            d.set_item("identity", "id-7").unwrap();
            d.set_item("metadata", "{}").unwrap();
            d.set_item("attributes", vec![("k".to_string(), "v".to_string())]).unwrap();
            match parse_decision(py, d.as_any()).unwrap() {
                JobDecision::Accept(a) => {
                    assert_eq!(a.name, "agent-7");
                    assert_eq!(a.identity, "id-7");
                    assert_eq!(a.metadata, "{}");
                    assert_eq!(a.attributes, vec![("k".to_string(), "v".to_string())]);
                }
                other => panic!("{other:?}"),
            }

            let bare = PyDict::new(py);
            bare.set_item("decision", "accept").unwrap();
            match parse_decision(py, bare.as_any()).unwrap() {
                JobDecision::Accept(a) => {
                    assert_eq!(a.name, "");
                    assert!(a.attributes.is_empty());
                }
                other => panic!("{other:?}"),
            }

            bare.set_item("name", 7).unwrap();
            let err =
                parse_decision(py, bare.as_any()).expect_err("a non-str name must be reported");
            assert!(err.contains("name") || err.contains("str"), "{err}");
        });
    }

    // Catches: a manifest entry's kind or methods lost between Python and the register message,
    // or routes not selecting the WebTransport transport the data plane needs.
    #[test]
    fn endpoints_reach_the_manifest_with_their_kind_and_select_webtransport() {
        attach(|py| {
            let text = PyDict::new(py);
            text.set_item("path", "/a2a").unwrap();
            text.set_item("methods", vec!["POST"]).unwrap();
            text.set_item("kind", "text").unwrap();
            let http = PyDict::new(py);
            http.set_item("path", "/items/{id}").unwrap();
            http.set_item("methods", vec!["GET", "HEAD"]).unwrap();
            http.set_item("public", true).unwrap();
            let text = endpoint_from_py(text.as_any()).unwrap();
            let http = endpoint_from_py(http.as_any()).unwrap();
            assert_eq!(text.kind, AgentEndpointKind::AekText as i32);
            assert_eq!(http.kind, AgentEndpointKind::AekHttp as i32);
            assert_eq!(http.methods, vec!["GET", "HEAD"]);
            assert!(http.public && !text.public);
            assert!(matches!(transport_for(true), TransportConfig::WebTransport { .. }));

            let bad = PyDict::new(py);
            bad.set_item("path", "/x").unwrap();
            bad.set_item("kind", "grpc").unwrap();
            assert!(endpoint_from_py(bad.as_any()).is_err(), "an unknown kind must be refused");
        });
    }
}
