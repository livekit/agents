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
//! | `"terminate_job"` | `{"job_id"}`                           | ignored                     |
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
use pyo3::types::{IntoPyDict, PyDict, PyList};
use std::sync::atomic::{AtomicBool, Ordering};
use pyo3_async_runtimes::TaskLocals;

use crate::transport::{AgentEndpointKind, EndpointReader, EndpointWriter, HttpStreamResetCode};
use crate::{
    EndpointRequest, EnvOption, HEARTBEAT_INTERVAL, HostError, JobDecision, JobId, LOG,
    ServerType, TransportConfig, Worker, WorkerCommand, WorkerHandle, WorkerHost,
    WorkerNotification, protocol,
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
    fn finish(&self, py: Python<'_>) -> PyResult<()> {
        check_pid()?;
        let writer = self.writer.clone();
        py.detach(|| {
            let mut guard = writer.blocking_lock();
            match guard.as_mut() {
                Some(writer) => writer.finish().map_err(|e| PyRuntimeError::new_err(e.to_string())),
                None => Ok(()),
            }
        })
    }

    /// RESET_STREAM with an `HttpStreamResetCode` number, and STOP_SENDING with the same.
    fn reset(&self, py: Python<'_>, code: i32) -> PyResult<()> {
        check_pid()?;
        let code = reset_code(code)?;
        let (writer, reader) = (self.writer.clone(), self.reader.clone());
        py.detach(|| {
            if let Some(w) = writer.blocking_lock().as_mut() {
                w.reset(code);
            }
            if let Some(r) = reader.blocking_lock().as_mut() {
                r.stop(code);
            }
        });
        Ok(())
    }

    /// STOP_SENDING only: the response went out whole and nothing more of the request is wanted.
    fn stop(&self, py: Python<'_>, code: i32) -> PyResult<()> {
        check_pid()?;
        let code = reset_code(code)?;
        let reader = self.reader.clone();
        py.detach(|| {
            if let Some(r) = reader.blocking_lock().as_mut() {
                r.stop(code);
            }
        });
        Ok(())
    }
}

// --------------------------------------------------------------------- EndpointServer

use crate::endpoint::{self, AppError, AsgiApp, Conn as AsgiConn, Received, RequestInfo, Scope, Sent, StreamIo};
use lk_agents_abi::python::{CancellingHandle, HostBridge};
use pyo3::types::PyBytes;

/// `_core.EndpointStream` as the exchange reads and writes it.
struct NativeStream {
    reader: Half<EndpointReader>,
    writer: Half<EndpointWriter>,
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
        if let Some(w) = self.writer.lock().await.as_mut() {
            let _ = w.finish();
        }
    }

    async fn reset(&self, code: HttpStreamResetCode) {
        if let Some(w) = self.writer.lock().await.as_mut() {
            w.reset(code);
        }
        self.stop(code).await;
    }

    async fn stop(&self, code: HttpStreamResetCode) {
        if let Some(r) = self.reader.lock().await.as_mut() {
            r.stop(code);
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
        let v = self.0.await_call(|s| s.call_method1("read", (max,))).await.map_err(|e| e.to_string())?;
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

fn scope_py<'py>(py: Python<'py>, scope: &Scope, request: &Py<PyAny>) -> PyResult<Bound<'py, PyDict>> {
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
    fn __call__<'py>(&self, py: Python<'py>, message: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
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
        let info = RequestInfo {
            request_id: request.getattr("request_id")?.extract()?,
            route: request.getattr("route")?.extract()?,
            scheme: request.getattr("scheme")?.extract()?,
            client_addr: request.getattr("client_addr")?.extract()?,
            timeout_ms: request.getattr("timeout_ms")?.extract::<i64>()?.max(0) as u64,
        };
        let io: Arc<dyn StreamIo> = match stream.cast::<PyEndpointStream>() {
            Ok(s) => {
                let s = s.borrow();
                Arc::new(NativeStream { reader: s.reader.clone(), writer: s.writer.clone() })
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
    fn close_websockets<'py>(&self, py: Python<'py>, code: u16, reason: String) -> PyResult<Bound<'py, PyAny>> {
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
    async fn on_job_request(&self, req: crate::JobRequest) -> JobDecision {
        if let Some(decision) = server::own_decision(&req, self.has_rtc, self.has_on_request) {
            return decision;
        }
        let job_id = req.job.id.clone();
        // `resuming` is set when a call migrates off a draining worker.
        let res = self.open("job_request", |py| Ok((&req).into_pyobject(py)?.into_any())).await;
        match res {
            // A failed `request_fnc` rejects without terminating, so the server can redispatch.
            Err(e) => JobDecision::Failed(e),
            Ok(v) => Python::attach(|py| parse_decision(py, v.bind(py)))
                .unwrap_or_else(|e| JobDecision::Failed(HostError::new(format!("{job_id}: {e}")))),
        }
    }

    async fn launch_job(&self, info: crate::RunningJobInfo) -> Result<JobId, HostError> {
        let v = self.open("launch_job", |py| Ok((&info).into_pyobject(py)?.into_any())).await?;
        Python::attach(|py| v.extract::<String>(py))
            .map_err(|e| HostError::new(format!("launch_job returned a non-str job id: {e}")))
    }

    async fn terminate_job(&self, id: JobId) -> Result<(), HostError> {
        self.open("terminate_job", |py| {
            let d = PyDict::new(py);
            d.set_item("job_id", &id)?;
            Ok(d.into_any())
        })
        .await
        .map(|_| ())
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
                d.set_item("stream", PyEndpointStream { reader, writer })?;
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
            log::warn!(target: LOG, request_id = request_id.as_str(), error = error.to_string().as_str();
                "endpoint request failed in the host");
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
    let raw: crate::RawJobDecision = d.as_any().extract().map_err(host_err)?;
    JobDecision::from_wire(
        &raw.decision,
        || d.as_any().extract::<crate::AcceptArguments>().map_err(host_err),
        || raw.error.clone().unwrap_or_default(),
    )
}

// --------------------------------------------------------------------- the routes

/// One manifest entry: `{"path", "methods", "kind", "public"}`, with `kind` `"http"` (default) or
/// `"text"` for a session endpoint (plan 9.1).
fn endpoint_from_py(item: &Bound<'_, PyAny>) -> PyResult<protocol::AgentEndpoint> {
    let d = item.cast::<PyDict>().map_err(|_| PyValueError::new_err("endpoints: expected dicts"))?;
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

use crate::server::{self, ServerSettings, ServerStatus};

/// `AgentServer._endpoint_url_base`.
#[pyfunction]
fn endpoint_url_base(ws_url: &str, agent_name: &str, deployment: &str) -> Option<String> {
    server::endpoint_url_base(ws_url, agent_name, deployment)
}

lk_agents_abi::register_host!(fn endpoint_url_base);

/// The given options written over `s`; an unknown one is refused.
fn apply(s: &mut ServerSettings, options: &Bound<'_, PyDict>) -> PyResult<()> {
    fn text(v: &Bound<'_, PyAny>) -> PyResult<String> {
        Ok(v.extract::<Option<String>>()?.unwrap_or_default())
    }
    for (key, v) in options.iter() {
        let key: String = key.extract()?;
        let wrap = |e: PyErr| PyValueError::new_err(format!("{key}: {e}"));
        match key.as_str() {
            "ws_url" => s.ws_url = text(&v)?,
            "api_key" => s.api_key = text(&v)?,
            "api_secret" => s.api_secret = text(&v)?,
            "agent_name" => s.agent_name = text(&v)?,
            "host" => s.host = text(&v)?,
            "port" => s.port = EnvOption::Fixed(v.extract().map_err(wrap)?),
            "load_threshold" => s.load_threshold = EnvOption::Fixed(v.extract().map_err(wrap)?),
            "num_idle_processes" => s.num_idle_processes = EnvOption::Fixed(v.extract().map_err(wrap)?),
            "drain_timeout" => s.drain_timeout = v.extract::<Option<f64>>()?.unwrap_or(0.0),
            "max_retry" => s.max_retry = v.extract()?,
            "permissions" => s.permissions = v.extract()?,
            "http_proxy" => s.http_proxy = v.extract()?,
            "prometheus_port" => s.prometheus_port = v.extract()?,
            "prometheus_multiproc_dir" => s.prometheus_multiproc_dir = v.extract()?,
            "version" => s.version = v.extract()?,
            "job_executor_type" => s.job_executor_type = v.extract()?,
            "multiprocessing_context" => s.multiprocessing_context = v.extract()?,
            "job_memory_warn_mb" => s.job_memory_warn_mb = v.extract()?,
            "job_memory_limit_mb" => s.job_memory_limit_mb = v.extract()?,
            "shutdown_process_timeout" => s.shutdown_process_timeout = v.extract()?,
            "session_end_timeout" => s.session_end_timeout = v.extract()?,
            "initialize_process_timeout" => s.initialize_process_timeout = v.extract()?,
            other => return Err(PyValueError::new_err(format!("unknown server option {other:?}"))),
        }
    }
    Ok(())
}

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
    fn new(options: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut s = ServerSettings::default();
        let env = |name: &str| std::env::var(name).ok();
        s.http_proxy = ServerSettings::env_proxy(env);
        apply(&mut s, options)?;
        Ok(PyServerConfig {
            settings: Arc::new(std::sync::Mutex::new(s.with_env(env))),
            running: Arc::default(),
        })
    }

    /// `update_options`: the given options, before the server starts.
    fn update(&self, options: &Bound<'_, PyDict>) -> PyResult<()> {
        if self.running.load(Ordering::Acquire) {
            return Err(PyRuntimeError::new_err("cannot update options after starting the server"));
        }
        apply(&mut self.get(), options)
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

#[derive(PartialEq)]
enum Phase {
    Idle,
    Running,
    Closed,
}

struct Life {
    phase: Phase,
    worker: Option<WorkerHandle>,
    conn: Option<tokio::task::JoinHandle<()>>,
    load: Option<tokio::task::AbortHandle>,
    health: Option<tokio::task::AbortHandle>,
    metrics: Option<tokio::task::AbortHandle>,
}

/// How a run ended: `Ok` once closed, the connection's error when it failed.
type Outcome = tokio::sync::watch::Sender<Option<Result<(), String>>>;

/// The health listener's view of the host.
struct HostHealth(PyWorkerHost);

#[async_trait]
impl server::HealthProbe for HostHealth {
    async fn health(&self) -> (bool, usize) {
        match self.0.open("health", |py| Ok(PyDict::new(py).into_any())).await {
            Ok(v) => Python::attach(|py| v.extract::<(bool, usize)>(py)).unwrap_or((true, 0)),
            Err(_) => (true, 0),
        }
    }

    async fn metrics(&self) -> Option<(String, String)> {
        let v = self.0.open("metrics", |py| Ok(PyDict::new(py).into_any())).await.ok()?;
        Python::attach(|py| v.extract::<(String, String)>(py).ok())
    }
}

/// Whatever `run()` started, stopped: the connection, the load loop, the host's job processes
/// (`close`) and the listeners; `run()` returns.
async fn shut_down(life: &mut Life, host: &PyWorkerHost, running: &AtomicBool, outcome: &Outcome) -> PyResult<()> {
    life.phase = Phase::Closed;
    if let Some(worker) = life.worker.take() {
        let _ = worker.send(WorkerCommand::Close);
    }
    if let Some(conn) = life.conn.take() {
        let _ = conn.await;
    }
    if let Some(load) = life.load.take() {
        load.abort();
    }
    let closed = host.open("close", empty).await;
    for listener in [life.health.take(), life.metrics.take()].into_iter().flatten() {
        listener.abort();
    }
    running.store(false, Ordering::Release);
    outcome.send_if_modified(|o| o.is_none().then(|| *o = Some(Ok(()))).is_some());
    closed.map(|_| ()).map_err(lk_agents_abi::python::raise_host_error)
}

fn empty(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    Ok(PyDict::new(py).into_any())
}

/// `AgentServer`'s lifecycle (`_core.ServerRuntime`), over the host's `open(kind, args)`: the
/// worker's callbacks, and `start` (the job processes, from the resolved options), `started`,
/// `registered`, `idle_processes`, `worker_load`, `health`, `close_websockets`, `join_jobs` and
/// `close`.
#[pyclass(name = "ServerRuntime", module = "livekit.agents._core")]
pub struct PyServer {
    settings: Arc<std::sync::Mutex<ServerSettings>>,
    running: Arc<AtomicBool>,
    host: PyWorkerHost,
    life: Arc<tokio::sync::Mutex<Life>>,
    status: Arc<ServerStatus>,
    worker_id: Arc<std::sync::Mutex<String>>,
    draining: Arc<AtomicBool>,
    http_port: Arc<std::sync::atomic::AtomicU16>,
    outcome: Arc<Outcome>,
    /// The registered worker, for the job statuses the host reports.
    worker: Arc<std::sync::Mutex<Option<WorkerHandle>>>,
    /// Set by `run()`: only a voice door runs jobs.
    has_rtc: Arc<AtomicBool>,
}

impl PyServer {
    fn settings(&self) -> ServerSettings {
        self.settings.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn id(&self) -> String {
        self.worker_id.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

#[pymethods]
impl PyServer {
    /// Constructed on the host's loop, where every `open` runs.
    #[new]
    fn new(py: Python<'_>, config: &PyServerConfig, host: Py<PyAny>) -> PyResult<Self> {
        check_pid()?;
        if !host.bind(py).hasattr("open")? {
            return Err(PyValueError::new_err("host object has no open(kind, args)"));
        }
        let locals = TaskLocals::with_running_loop(py)?.copy_context(py)?;
        Ok(PyServer {
            settings: config.settings.clone(),
            running: config.running.clone(),
            host: PyWorkerHost {
                obj: Arc::new(host),
                locals,
                user_load: true,
                has_rtc: false,
                has_on_request: false,
            },
            has_rtc: Arc::default(),
            life: Arc::new(tokio::sync::Mutex::new(Life {
                phase: Phase::Idle,
                worker: None,
                conn: None,
                load: None,
                health: None,
                metrics: None,
            })),
            status: Arc::default(),
            worker_id: Arc::new(std::sync::Mutex::new("unregistered".into())),
            draining: Arc::default(),
            http_port: Arc::default(),
            outcome: Arc::new(tokio::sync::watch::channel(None).0),
            worker: Arc::default(),
        })
    }

    #[getter]
    fn worker_id(&self) -> String {
        self.id()
    }

    #[getter]
    fn draining(&self) -> bool {
        self.draining.load(Ordering::Acquire)
    }

    /// The health listener's port; 0 without one.
    #[getter]
    fn http_port(&self) -> u16 {
        self.http_port.load(Ordering::Acquire)
    }

    /// Start the job processes (the host's `start`), the health listener and the load loop,
    /// register unless `unregistered`, and serve until `aclose()`; raises when the connection fails
    /// for good. `routes` is the endpoint manifest; `publisher` the voice door's job type. Job
    /// processes start only for a voice door; the host is asked about a job request only when
    /// that door has an `on_request` (`server::own_decision`).
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
        let mut s = self.settings();
        let (life, running, outcome) =
            (self.life.clone(), self.running.clone(), self.outcome.clone());
        let (status, worker_id, http_port) =
            (self.status.clone(), self.worker_id.clone(), self.http_port.clone());
        let mut host = self.host.clone();
        host.has_rtc = has_rtc;
        host.has_on_request = has_on_request;
        self.has_rtc.store(has_rtc, Ordering::Release);
        if !has_rtc {
            s.num_idle_processes = EnvOption::Fixed(0);
        }
        let reporting = self.worker.clone();
        let server_type = if publisher { ServerType::Publisher } else { ServerType::Room };
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = life.lock().await;
            if guard.phase != Phase::Idle {
                return Err(PyRuntimeError::new_err("worker is already running"));
            }
            if !has_rtc && !has_text_doors {
                return Err(PyRuntimeError::new_err(
                    "No RTC session entrypoint has been registered.\n\
                     Define one using the @server.rtc_session() decorator, for example:\n    \
                     @server.rtc_session()\n    async def my_agent(ctx: JobContext):\n        ...\n",
                ));
            }
            if s.cloud() {
                if has_load_fnc {
                    log::warn!(target: LOG,
                        "custom load_fnc is not supported when hosting on Cloud, reverting to default");
                }
                if s.load_threshold != server::DEFAULT_LOAD_THRESHOLD {
                    log::warn!(target: LOG,
                        "custom load_threshold is not supported when hosting on Cloud, reverting to default");
                    s.load_threshold = server::DEFAULT_LOAD_THRESHOLD;
                }
            }
            host.user_load = has_load_fnc && !s.cloud();
            if s.simulation {
                log::info!(target: LOG, "simulation mode enabled: worker load limit disabled");
            }
            if !unregistered {
                s.check_registration(!endpoints.is_empty()).map_err(PyValueError::new_err)?;
            }
            running.store(true, Ordering::Release);
            // from here `aclose()` shuts down whatever started, and so does a failure
            guard.phase = Phase::Running;
            let started = async {
                s.prometheus_multiproc_dir =
                    server::prepare_prometheus_dir(s.prometheus_multiproc_dir.as_deref());
                // the listeners before the job processes, as the reference: a port in use fails
                // the start before any process is spawned
                let settings = Arc::new(s.clone());
                if !s.simulation {
                    let (listener, port) = server::bind_listener(&s.host, s.port.get(devmode))
                        .await
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    http_port.store(port, Ordering::Release);
                    log::info!(target: LOG, "HTTP server listening on {}:{port}", s.host);
                    let probe = Arc::new(HostHealth(host.clone()));
                    let serve =
                        server::serve_health(listener, settings.clone(), server_type, status.clone(), probe);
                    guard.health = Some(tokio::spawn(serve).abort_handle());
                }
                if let Some(port) = s.prometheus_port {
                    let (listener, port) = server::bind_listener(&s.host, port)
                        .await
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    log::info!(target: LOG, "Prometheus metrics exposed at http://{}:{port}/metrics", s.host);
                    let serve = server::serve_metrics(listener, Arc::new(HostHealth(host.clone())));
                    guard.metrics = Some(tokio::spawn(serve).abort_handle());
                }
                let start = s.clone();
                host.open("start", move |py| {
                    let d = PyDict::new(py);
                    d.set_item("devmode", devmode)?;
                    d.set_item("job_processes", has_rtc)?;
                    d.set_item("env", start.child_env().into_py_dict(py)?)?;
                    d.set_item("job_executor_type", &start.job_executor_type)?;
                    d.set_item("multiprocessing_context", &start.multiprocessing_context)?;
                    // `ProcPool`'s keyword arguments
                    let pool = PyDict::new(py);
                    pool.set_item("num_idle_processes", start.num_idle_processes.get(devmode))?;
                    pool.set_item("initialize_timeout", start.initialize_process_timeout)?;
                    pool.set_item("close_timeout", start.shutdown_process_timeout)?;
                    pool.set_item("session_end_timeout", start.session_end_timeout)?;
                    pool.set_item("memory_warn_mb", start.job_memory_warn_mb)?;
                    pool.set_item("memory_limit_mb", start.job_memory_limit_mb)?;
                    pool.set_item("http_proxy", start.http_proxy.as_deref().filter(|p| !p.is_empty()))?;
                    d.set_item("pool", pool)?;
                    Ok(d.into_any())
                })
                .await
                .map_err(lk_agents_abi::python::raise_host_error)?;
                PyResult::Ok(())
            }
            .await;
            if let Err(e) = started {
                let _ = shut_down(&mut guard, &host, &running, &outcome).await;
                return Err(e);
            }
            if !unregistered {
                let worker_url = std::env::var("LIVEKIT_WORKER_URL").ok().filter(|u| !u.is_empty());
                let transport = transport_for(!endpoints.is_empty());
                let options = s.worker_options(devmode, server_type, endpoints.clone(), transport, worker_url);
                if !endpoints.is_empty() {
                    let routes: Vec<String> =
                        endpoints.iter().map(|e| format!("{} {}", e.methods.join(","), e.path)).collect();
                    let routes = routes.join(", ");
                    log::info!(target: LOG, routes = routes.as_str(); "serving agent endpoints");
                }
                let (mut conn, handle) = Worker::new(options, host.clone());
                let mut notes = handle.subscribe();
                let (registered, failed, outcome) = (host.clone(), status.clone(), outcome.clone());
                let worker_id = worker_id.clone();
                *reporting.lock().unwrap_or_else(|e| e.into_inner()) = Some(handle.clone());
                guard.worker = Some(handle);
                guard.conn = Some(tokio::spawn(async move {
                    let forward = async {
                        while let Ok(note) = notes.recv().await {
                            match note {
                                WorkerNotification::Registered(r) => {
                                    *worker_id.lock().unwrap_or_else(|e| e.into_inner()) =
                                        r.worker_id.clone();
                                    let _ = registered
                                        .open("registered", move |py| Ok(r.into_pyobject(py)?.into_any()))
                                        .await;
                                }
                                WorkerNotification::Stopped(_) => break,
                                _ => {}
                            }
                        }
                    };
                    let (result, ()) = tokio::join!(conn.run(), forward);
                    if let Err(e) = result {
                        failed.connection_failed.store(true, Ordering::Release);
                        log::error!(target: LOG, "worker connection failed: {e}");
                        outcome.send_if_modified(|o| {
                            o.is_none().then(|| *o = Some(Err(e.to_string()))).is_some()
                        });
                    }
                }));
            }
            let (load_host, load_status, worker) = (host.clone(), status.clone(), guard.worker.clone());
            guard.load = Some(
                tokio::spawn(async move {
                    let mut tick = tokio::time::interval(crate::UPDATE_LOAD_INTERVAL);
                    loop {
                        tick.tick().await;
                        if let Some(worker) = &worker {
                            let snapshot = worker.snapshot();
                            load_status.set_load(snapshot.worker_load);
                            if let Some(target) = snapshot.target_idle_processes {
                                let _ = load_host
                                    .open("idle_processes", move |py| Ok(target.into_pyobject(py)?.into_any()))
                                    .await;
                            }
                        }
                        let load = load_status.load();
                        let _ = load_host
                            .open("worker_load", move |py| Ok(load.into_pyobject(py)?.into_any()))
                            .await;
                    }
                })
                .abort_handle(),
            );
            let _ = host.open("started", empty).await;
            drop(guard);
            let mut ended = outcome.subscribe();
            let result = ended
                .wait_for(Option::is_some)
                .await
                .map(|o| o.clone().unwrap_or(Ok(())))
                .unwrap_or(Ok(()));
            result.map_err(PyRuntimeError::new_err)
        })
    }

    /// Stop taking jobs and wait for the running ones, within `timeout` seconds (0 or `None`: no
    /// deadline; `TimeoutError` past it). Once; a server never run has nothing to drain.
    #[pyo3(signature = (timeout=None))]
    fn drain<'py>(&self, py: Python<'py>, timeout: Option<f64>) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let (life, draining, host, id) =
            (self.life.clone(), self.draining.clone(), self.host.clone(), self.id());
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = life.lock().await;
            if guard.phase != Phase::Running || draining.swap(true, Ordering::AcqRel) {
                return Ok(());
            }
            let shown = timeout.map(|t| t.to_string()).unwrap_or_else(|| "None".into());
            log::info!(target: LOG, id = id.as_str(), timeout = shown.as_str(); "draining worker");
            // clients of an open WebSocket get a close frame and reconnect to another worker
            let _ = host.open("close_websockets", empty).await;
            let deadline = timeout.filter(|t| *t > 0.0).map(Duration::from_secs_f64);
            let timed_out = || pyo3::exceptions::PyTimeoutError::new_err("drain timed out");
            match &guard.worker {
                Some(worker) => worker.drain(deadline).await.map_err(|e| {
                    if e == "drain timed out" { timed_out() } else { PyRuntimeError::new_err(e) }
                }),
                None => {
                    let joined = host.open("join_jobs", empty);
                    let joined = match deadline {
                        Some(d) => tokio::time::timeout(d, joined).await.map_err(|_| timed_out())?,
                        None => joined.await,
                    };
                    joined.map(|_| ()).map_err(lk_agents_abi::python::raise_host_error)
                }
            }
        })
    }

    /// Stop the connection and the load loop, close the host's job processes (`close`), then the
    /// health listener; `run()` returns. A server never run, or already closed, does nothing.
    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let (life, running, outcome, host, id) = (
            self.life.clone(),
            self.running.clone(),
            self.outcome.clone(),
            self.host.clone(),
            self.id(),
        );
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = life.lock().await;
            if guard.phase != Phase::Running {
                return Ok(());
            }
            log::info!(target: LOG, id = id.as_str(); "shutting down worker");
            shut_down(&mut guard, &host, &running, &outcome).await
        })
    }

    /// A job process's status (`running`, `failed`, `success`), by the LiveKit job id; nothing
    /// before registration.
    fn job_status(&self, job_id: String, status: &str) {
        let status = match status {
            "failed" => protocol::JobStatus::JsFailed,
            "success" => protocol::JobStatus::JsSuccess,
            _ => protocol::JobStatus::JsRunning,
        };
        if let Some(worker) = self.worker.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            let _ = worker.send(WorkerCommand::JobStatus { job_id, status });
        }
    }

    /// Launch a job under `room` without a dispatch (`simulate_job`); `room_info` is a
    /// `models.Room`'s bytes.
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
        if !self.has_rtc.load(Ordering::Acquire) {
            return Err(PyRuntimeError::new_err("a server with no rtc_session runs no job"));
        }
        let room_info = room_info.map(|b| server::decode_room(&b)).transpose().map_err(PyValueError::new_err)?;
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
        let (life, host) = (self.life.clone(), self.host.clone());
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // not held across the launch: a close while the process warms up is not delayed
            if life.lock().await.phase != Phase::Running {
                return Err(PyRuntimeError::new_err("the server is not running"));
            }
            host.launch_job(info).await.map(|_| ()).map_err(lk_agents_abi::python::raise_host_error)
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

    // Catches: `ServerOptions.permissions` not reaching the config (`hidden=True` registering a
    // visible agent), or a partial dict raising instead of defaulting.
    #[test]
    fn the_permissions_dict_reaches_the_config_and_missing_keys_default() {
        attach(|py| {
            let d = PyDict::new(py);
            d.set_item("hidden", true).unwrap();
            d.set_item("can_publish", false).unwrap();
            d.set_item("can_publish_sources", vec![1, 2]).unwrap();
            let p: crate::WorkerPermissions = d.as_any().extract().unwrap();
            assert!(p.hidden && !p.can_publish, "{p:?}");
            assert_eq!(p.can_publish_sources, vec![1, 2]);
            assert!(p.can_subscribe && p.can_publish_data && p.can_update_metadata, "{p:?}");
            let empty: crate::WorkerPermissions = PyDict::new(py).as_any().extract().unwrap();
            assert_eq!(empty, crate::WorkerPermissions::default());
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
            let err = parse_decision(py, bare.as_any()).expect_err("a non-str name must be reported");
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
