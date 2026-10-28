//! Host object conversion for worker endpoint streams and the ASGI application.

use std::sync::Arc;

use async_trait::async_trait;
use lk_agents_abi::HostRef;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::python::{capture_host, check_pid};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::worker::transport::{
    EndpointControl, EndpointReader, EndpointWriter, HttpStreamResetCode,
};

// --------------------------------------------------------------------- EndpointStream

pub(super) type Half<T> = Arc<tokio::sync::Mutex<Option<T>>>;

/// One exchange stream after its preamble, as Python reads and writes it. The two directions are
/// independent: a `read` and a `write` may be in flight at once. Dropping it without `finish`
/// resets with code 0 (`HSR_ABORT`).
pub struct PyEndpointStream {
    pub(super) reader: Half<EndpointReader>,
    pub(super) writer: Half<EndpointWriter>,
    pub(super) control: EndpointControl,
}

/// The largest single `read`; a body streams through many.
const MAX_READ: usize = 1 << 20;

fn reset_code(code: i32) -> PyResult<HttpStreamResetCode> {
    HttpStreamResetCode::try_from(code)
        .map_err(|_| PyValueError::new_err(format!("unknown HttpStreamResetCode {code}")))
}

impl PyEndpointStream {
    /// Up to `max_bytes` of the peer's bytes; `b""` once the peer half-closed (FIN). Raises when
    /// the peer reset the stream.
    fn read_py<'py>(&self, py: Python<'py>, max_bytes: usize) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let reader = self.reader.clone();
        let max_bytes = max_bytes.clamp(1, MAX_READ);
        crate::host::future_into_py(py, async move {
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
    fn write_py<'py>(&self, py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let writer = self.writer.clone();
        let data = data.to_vec();
        crate::host::future_into_py(py, async move {
            let mut guard = writer.lock().await;
            let Some(writer) = guard.as_mut() else {
                return Err(PyRuntimeError::new_err("endpoint stream is closed"));
            };
            writer.write_all(&data).await.map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// FIN: the response is complete. Idempotent; a stream already reset raises.
    fn finish_py(&self) -> PyResult<()> {
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
    fn reset_py(&self, code: i32) -> PyResult<()> {
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
    fn stop_py(&self, code: i32) -> PyResult<()> {
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

use super::callback::Callback as CancellingHandle;
use crate::worker::endpoint::{
    self, AppError, AsgiApp, Conn as AsgiConn, Received, RequestInfo, Scope, Sent, StreamIo,
};
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
        let scheduled = self.0.call_soon(method, move |py| match code {
            Some(code) => PyTuple::new(py, [code]),
            None => Ok(PyTuple::empty(py)),
        });
        if let Err(error) = scheduled {
            Python::attach(|py| error.print(py));
        }
    }
}

#[async_trait]
impl StreamIo for HostStream {
    async fn read(&self, max: usize) -> Result<Vec<u8>, String> {
        let v = self
            .0
            .await_call(move |s| crate::host::construct(&s.getattr("read")?, (max,), None))
            .await
            .map_err(|e| e.to_string())?;
        Python::attach(|py| v.extract::<Vec<u8>>(py)).map_err(|e| e.to_string())
    }

    async fn write(&self, data: &[u8]) -> Result<(), String> {
        let data = data.to_vec();
        self.0
            .await_call(move |s| {
                crate::host::construct(&s.getattr("write")?, (PyBytes::new(s.py(), &data),), None)
            })
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
    crate::host::construct(&v.py().import("builtins")?.getattr("bytes")?, (v,), None)?.extract()
}

fn sent_from_py(m: &Bound<'_, PyAny>) -> PyResult<Sent> {
    let get = |k: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
        let v = crate::host::construct(&m.getattr("get")?, (k,), None)?;
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

#[async_trait]
impl AsgiApp for PyApp {
    async fn call(&self, scope: Scope, conn: AsgiConn) -> Result<(), AppError> {
        let request = Python::attach(|py| self.request.clone_ref(py));
        self.app
            .await_call(move |app| {
                let scope = scope_py(app.py(), &scope, &request)?;
                let receive = crate::generated::worker::_core_AsgiReceive(Arc::new(AsgiReceive {
                    conn: conn.clone(),
                }));
                let send = crate::generated::worker::_core_AsgiSend(Arc::new(AsgiSend { conn }));
                crate::host::construct(app, (scope, receive, send), None)
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

/// A receive callable whose argument binding is generated from its ABI declaration.
pub struct AsgiReceive {
    conn: AsgiConn,
}

impl lk_agents_abi::worker::endpoint_host::AsgiReceive for AsgiReceive {
    fn __call__(&self) -> Result<HostRef, HostError> {
        let conn = self.conn.clone();
        Python::attach(|py| {
            crate::host::future_into_py(py, async move {
                let message = conn.receive().await;
                Python::attach(|py| received_py(py, message).map(Bound::unbind))
            })
            .map(awaitable_ref)
        })
        .map_err(host_error)
    }
}

/// A send callable whose argument binding is generated from its ABI declaration.
pub struct AsgiSend {
    conn: AsgiConn,
}

impl lk_agents_abi::worker::endpoint_host::AsgiSend for AsgiSend {
    fn __call__(&self, message: HostRef) -> Result<HostRef, HostError> {
        let conn = self.conn.clone();
        Python::attach(|py| {
            let message = lk_agents_abi::python::resolve_key(py, message.id())?;
            let message = sent_from_py(message.bind(py))?;
            crate::host::future_into_py(py, async move {
                conn.send(message).await.map_err(PyRuntimeError::new_err)
            })
            .map(awaitable_ref)
        })
        .map_err(host_error)
    }
}

/// `AgentHttp`'s server half: each `open("endpoint", ..)` stream served with the ASGI app
/// (`endpoint.rs`), and the open WebSockets it drains.
#[derive(Default)]
pub struct PyEndpointServer {
    websockets: Arc<endpoint::Registry>,
}

impl PyEndpointServer {
    /// Serve `stream` (an `EndpointStream` after its preamble) with `app` to completion;
    /// `request` is the `EndpointRequest` the handler sees as `request.state.lk_request`.
    fn serve_py<'py>(
        &self,
        py: Python<'py>,
        stream: &Bound<'py, PyAny>,
        app: Py<PyAny>,
        request: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let info: RequestInfo = request.extract()?;
        let io: Arc<dyn StreamIo> = match stream.cast::<crate::generated::worker::EndpointStream>()
        {
            Ok(s) => {
                let s = &s.get().0;
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
        crate::host::future_into_py(py, async move {
            endpoint::serve(io, info, &app, &registry).await;
            Ok(())
        })
    }

    /// A close frame to every open WebSocket, so clients reconnect to another worker.
    fn close_websockets_py<'py>(
        &self,
        py: Python<'py>,
        code: u16,
        reason: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let registry = self.websockets.clone();
        crate::host::future_into_py(py, async move {
            registry.close_all(code, &reason).await;
            Ok(())
        })
    }
}

fn host_error(error: PyErr) -> HostError {
    Python::attach(|py| capture_host(py, &error))
}

fn awaitable_ref(future: Bound<'_, PyAny>) -> HostRef {
    lk_agents_abi::python::intern_host(future.unbind())
}

impl lk_agents_abi::worker::endpoint_host::EndpointStream for PyEndpointStream {
    fn read(&self, max_bytes: usize) -> Result<HostRef, HostError> {
        Python::attach(|py| self.read_py(py, max_bytes).map(awaitable_ref)).map_err(host_error)
    }

    fn write(&self, data: Vec<u8>) -> Result<HostRef, HostError> {
        Python::attach(|py| self.write_py(py, &data).map(awaitable_ref)).map_err(host_error)
    }

    fn finish(&self) -> Result<(), HostError> {
        self.finish_py().map_err(host_error)
    }

    fn reset(&self, code: i32) -> Result<(), HostError> {
        self.reset_py(code).map_err(host_error)
    }

    fn stop(&self, code: i32) -> Result<(), HostError> {
        self.stop_py(code).map_err(host_error)
    }
}

impl lk_agents_abi::worker::endpoint_host::EndpointServer for PyEndpointServer {
    fn new() -> Self {
        Self::default()
    }

    fn serve(&self, stream: HostRef, app: HostRef, request: HostRef) -> Result<HostRef, HostError> {
        Python::attach(|py| {
            let stream = lk_agents_abi::python::resolve_key(py, stream.id())?;
            let app = lk_agents_abi::python::resolve_key(py, app.id())?;
            let request = lk_agents_abi::python::resolve_key(py, request.id())?;
            self.serve_py(py, stream.bind(py), app, request.bind(py)).map(awaitable_ref)
        })
        .map_err(host_error)
    }

    fn close_websockets(&self, code: u16, reason: String) -> Result<HostRef, HostError> {
        Python::attach(|py| self.close_websockets_py(py, code, reason).map(awaitable_ref))
            .map_err(host_error)
    }
}
