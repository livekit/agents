//! `_core.InterruptionDetector` and `_core.InterruptionDetectorStream`; the detector is the
//! session's [`InterruptionSource`] as a native class, never a proxy.

use std::sync::Arc;

use lk_agents_abi::voice::events::{DetectorErrorWire, DetectorOverlapWire};
use lk_agents_abi::metrics::base::InterruptionMetricsModel;
use lk_agents_abi::turn::interruption::InterruptionSource;
use lk_agents_abi::python::{Ambient, check_pid, extract_generated, intern_host, provider_error_to_py};
use lk_agents_abi::SpawnerExt as _;
use lk_agents_abi::telemetry::report::{SpanUpdateAttrsWire, SpanUpdateWire};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{
    AdaptiveInterruptionDetector, ConnectionConfig, DetectorEvent, InputEvent, InterruptionDetectionError,
    InterruptionOptions, InterruptionStream, OverlapEnd, OverlapStart, OverlappingSpeechEvent, StreamEvent,
};
use crate::python::{conn_options, header_provider, register_class, register_fn};
use crate::transport::{TransportError, WsMessage, WsReceiver, WsSender, WsTransport};
use lk_agents_abi::python::{CancellingHandle, HostBridge as _};

/// `inference/interruption.py`'s WebSocket, over `lk_agents_inference::interruption`.
///
/// The crate owns the socket and the scoring; the pydantic models, emitter and OTel span stay
/// Python. `stream()` is awaitable (sync in the reference) because the crate's `stream()` calls
/// `tokio::spawn`, which panics outside the runtime.
#[pyclass(frozen, name = "InterruptionDetector", module = "livekit.agents._core")]
pub struct InterruptionDetector {
    inner: Arc<AdaptiveInterruptionDetector>,
    /// `http_session=`: an `aiohttp.ClientSession` every stream dials through (`ws_connect`), as
    /// the reference's `_connect_ws` does; `None` dials natively.
    http_session: Option<Py<PyAny>>,
    /// One receiver: one detector object per model.
    events: Arc<tokio::sync::Mutex<tokio::sync::mpsc::UnboundedReceiver<DetectorEvent>>>,
}

register_class!(InterruptionDetector);

lk_agents_abi::python::inventory::submit! {
    lk_agents_abi::python::HostNative {
        trait_id: || std::any::TypeId::of::<dyn InterruptionSource>(),
        read: |obj| {
            let d = obj.extract::<PyRef<'_, InterruptionDetector>>().ok()?;
            let v: Arc<dyn InterruptionSource> = Arc::clone(&d.inner) as Arc<dyn InterruptionSource>;
            Some(Box::new(v) as Box<dyn std::any::Any>)
        },
    }
}

#[pymethods]
impl InterruptionDetector {
    #[new]
    #[pyo3(signature = (
        *,
        base_url,
        api_key,
        api_secret,
        threshold = None,
        min_interruption_duration,
        max_audio_duration,
        audio_prefix_duration,
        detection_interval,
        inference_timeout,
        headers = None,
        timeout = 10.0,
        token_ttl_secs = 600,
        http_session = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        base_url: String,
        api_key: String,
        api_secret: String,
        threshold: Option<f64>,
        min_interruption_duration: f64,
        max_audio_duration: f64,
        audio_prefix_duration: f64,
        detection_interval: f64,
        inference_timeout: f64,
        headers: Option<Py<PyAny>>,
        timeout: f64,
        token_ttl_secs: u64,
        http_session: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        check_pid()?;
        let opts = InterruptionOptions::new(
            threshold,
            min_interruption_duration,
            max_audio_duration,
            audio_prefix_duration,
            detection_interval,
            inference_timeout,
            base_url,
            api_key,
            api_secret,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // `headers` is a callable, called per connection, returning the host's job headers without
        // `Authorization`: the crate mints that.
        let headers = headers.map_or_else(|| ConnectionConfig::default().headers, header_provider);
        let config = ConnectionConfig {
            conn_options: conn_options(timeout),
            headers,
            token_ttl_secs,
            ..ConnectionConfig::default()
        };
        let clock = <Arc<dyn lk_agents_abi::Clock> as Ambient>::ambient();
        let (detector, rx) = AdaptiveInterruptionDetector::new(opts, config, clock);
        Ok(InterruptionDetector {
            inner: Arc::new(detector),
            http_session,
            events: Arc::new(tokio::sync::Mutex::new(rx)),
        })
    }

    /// `update_options`. The options are shared with live streams, which reconnect to pick them up.
    #[pyo3(signature = (*, threshold = None, min_interruption_duration = None))]
    fn update_options(&self, threshold: Option<f64>, min_interruption_duration: Option<f64>) -> PyResult<()> {
        check_pid()?;
        self.inner.update_options(threshold, min_interruption_duration);
        Ok(())
    }

    /// The next event of the session's stream (`overlapping_speech`, `error` or `metrics`) as a
    /// dict, or `None` when the detector is gone.
    fn recv_event<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let events = Arc::clone(&self.events);
        lk_agents_abi::python::future_into_py(py, async move {
            let ev = events.lock().await.recv().await;
            match ev {
                None => Ok(None),
                Some(ev) => Python::attach(|py| Ok(Some(detector_event_to_py(py, &ev)?.unbind()))),
            }
        })
    }

    /// Open a detection stream, whose events reach only its own `recv`, for one attempt of the
    /// host's `_main_task` (which owns the retries). Awaitable -- see the type docs for why.
    fn stream<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let inner = Arc::clone(&self.inner);
        // On this loop, which the session's awaitables run on.
        let transport = match &self.http_session {
            Some(session) => {
                let handle = CancellingHandle::new(py, session.clone_ref(py))?;
                Some(Arc::new(HostWs { session: handle }) as Arc<dyn WsTransport>)
            }
            None => None,
        };
        lk_agents_abi::python::future_into_py(py, async move {
            let stream = inner.private_stream_once(transport);
            let sender = stream.sender();
            Ok(InterruptionDetectorStream {
                inner: Arc::new(tokio::sync::Mutex::new(Some(stream))),
                sender: Arc::new(std::sync::Mutex::new(sender)),
            })
        })
    }
}

fn detector_event_to_py<'py>(py: Python<'py>, ev: &DetectorEvent) -> PyResult<Bound<'py, PyDict>> {
    match ev {
        DetectorEvent::OverlappingSpeech(event) => overlap_to_py(py, event),
        DetectorEvent::Error(err) => error_to_py(py, err),
        // No wire struct: `InterruptionMetrics` converts itself; `metadata` is the host's.
        DetectorEvent::MetricsCollected(metrics) => (&**metrics).into_pyobject(py),
    }
}

fn stream_event_to_py<'py>(py: Python<'py>, ev: &StreamEvent) -> PyResult<Bound<'py, PyDict>> {
    match ev {
        StreamEvent::Overlap(event) => overlap_to_py(py, event),
        StreamEvent::Error(err) => stream_error_to_py(py, err),
        StreamEvent::SpanUpdate { span, attrs } => SpanUpdateWire {
            span: span.id(),
            attrs: SpanUpdateAttrsWire {
                is_interruption: attrs.is_interruption.to_owned(),
                probability: attrs.probability,
                total_duration: attrs.total_duration,
                prediction_duration: attrs.prediction_duration,
                detection_delay: attrs.detection_delay,
            },
        }
        .into_pyobject(py),
    }
}

fn error_to_py<'py>(py: Python<'py>, err: &InterruptionDetectionError) -> PyResult<Bound<'py, PyDict>> {
    DetectorErrorWire { message: err.error.to_string(), recoverable: err.recoverable }.into_pyobject(py)
}

/// A stream's failure also carries the error itself (`APIStatusError(status_code=500,
/// retryable=False)`, ...), which the host's `_run` raises into its retry ladder.
fn stream_error_to_py<'py>(py: Python<'py>, err: &InterruptionDetectionError) -> PyResult<Bound<'py, PyDict>> {
    let d = error_to_py(py, err)?;
    let exc = provider_error_to_py(lk_agents_abi::errors::ProviderError::Local(err.error.clone()));
    d.set_item("error", exc.value(py))?;
    Ok(d)
}

/// `{"type": "overlapping_speech", "event": ..}`. The arrays travel as lists (the host wraps them
/// in numpy). `None` stays `None`: "no prediction" is not an empty window.
fn overlap_to_py<'py>(py: Python<'py>, ev: &OverlappingSpeechEvent) -> PyResult<Bound<'py, PyDict>> {
    DetectorOverlapWire { event: () }.host_dict(py, ev.into_pyobject(py)?.into_any())
}

/// One detection stream.
#[pyclass(frozen, name = "InterruptionDetectorStream", module = "livekit.agents._core")]
pub struct InterruptionDetectorStream {
    /// The verdict half, held across an await by `recv`.
    inner: Arc<tokio::sync::Mutex<Option<InterruptionStream>>>,
    /// The input half. Synchronous so 50 fps pushes never queue behind a verdict. Dropped by
    /// `end_input`, which lets the receive half observe the end.
    sender: Arc<std::sync::Mutex<Option<tokio::sync::mpsc::UnboundedSender<InputEvent>>>>,
}

register_class!(InterruptionDetectorStream);

#[pymethods]
impl InterruptionDetectorStream {
    /// Push one audio frame. Synchronous: 50 fps.
    fn push_audio(&self, frame: &Bound<'_, PyAny>) -> PyResult<()> {
        check_pid()?;
        let frame: lk_agents_abi::AudioFrame = extract_generated(frame)?;
        self.send(InputEvent::Audio(frame))
    }

    /// One of the four sentinels: `agent_speech_started`, `agent_speech_ended`,
    /// `overlap_speech_started`, `overlap_speech_ended`.
    #[pyo3(signature = (
        kind,
        *,
        started_at = 0.0,
        ended_at = 0.0,
        speech_duration = 0.0,
        agent_ended = false,
        user_speaking_span = None,
    ))]
    fn push_event(
        &self,
        kind: &str,
        started_at: f64,
        ended_at: f64,
        speech_duration: f64,
        agent_ended: bool,
        user_speaking_span: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        check_pid()?;
        let ev = match kind {
            "agent_speech_started" => InputEvent::AgentSpeechStarted,
            "agent_speech_ended" => InputEvent::AgentSpeechEnded,
            "overlap_speech_started" => InputEvent::OverlapSpeechStarted(OverlapStart {
                // How long VAD says the user has already spoken; trims leading silence once.
                // Zero is "no VAD said", not "no silence".
                speech_duration,
                started_at,
                // Interned so the crate can hand it back on `span_update` untouched.
                user_speaking_span: user_speaking_span.map(intern_host),
            }),
            "overlap_speech_ended" => InputEvent::OverlapSpeechEnded(OverlapEnd { ended_at, agent_ended }),
            other => {
                return Err(PyValueError::new_err(format!("unknown interruption input event {other:?}")));
            }
        };
        self.send(ev)
    }

    fn flush(&self) -> PyResult<()> {
        check_pid()?;
        self.send(InputEvent::Flush)
    }

    /// `end_input()`: flush, then close the input channel.
    ///
    /// Drops both senders; leaving the clone alive parks the stream's task and leaks a socket per turn.
    fn end_input(&self) -> PyResult<()> {
        check_pid()?;
        let sender = self.sender.lock().unwrap_or_else(|p| p.into_inner()).take();
        match sender {
            None => Ok(()), // already ended: idempotent, like the reference's
            Some(sender) => {
                let _ = sender.send(InputEvent::Flush);
                drop(sender);
                let inner = Arc::clone(&self.inner);
                // The stream's own sender is behind the async lock; queued, since `end_input` is sync.
                <Arc<dyn lk_agents_abi::Spawner> as Ambient>::ambient().spawn_unboxed(async move {
                    if let Some(stream) = inner.lock().await.as_mut() {
                        let _ = stream.end_input();
                    }
                });
                Ok(())
            }
        }
    }

    /// The next event (`overlapping_speech`, `error` or `span_update`) as a dict, or `None` when
    /// the stream is done.
    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let inner = Arc::clone(&self.inner);
        lk_agents_abi::python::future_into_py(py, async move {
            let mut guard = inner.lock().await;
            let Some(stream) = guard.as_mut() else { return Ok(None) };
            match stream.recv_event().await {
                None => Ok(None),
                Some(ev) => Python::attach(|py| Ok(Some(stream_event_to_py(py, &ev)?.unbind()))),
            }
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let inner = Arc::clone(&self.inner);
        let self_sender = Arc::clone(&self.sender);
        lk_agents_abi::python::future_into_py(py, async move {
            let _ = self_sender.lock().unwrap_or_else(|p| p.into_inner()).take();
            if let Some(stream) = inner.lock().await.take() {
                stream
                    .aclose()
                    .await
                    .map_err(|e| provider_error_to_py(lk_agents_abi::errors::ProviderError::Local(e)))?;
            }
            Ok(())
        })
    }
}

impl InterruptionDetectorStream {
    /// A closed stream is an error: pushing into it would silently drop audio.
    fn send(&self, ev: InputEvent) -> PyResult<()> {
        let guard = self.sender.lock().unwrap_or_else(|p| p.into_inner());
        match guard.as_ref() {
            None => Err(PyRuntimeError::new_err("this interruption stream has ended its input")),
            Some(sender) => sender.send(ev).map_err(|_| PyRuntimeError::new_err("this interruption stream is closed")),
        }
    }
}

// `_metrics_monitor_task`'s record for one overlap, off the host's `OverlappingSpeechEvent`.
#[pyfunction]
fn interruption_metrics<'py>(
    py: Python<'py>,
    event: &Bound<'_, PyAny>,
    wall: f64,
    model: &str,
    provider: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let ev: OverlappingSpeechEvent = event.extract()?;
    let metrics = super::interruption_metrics(&ev, wall, model, provider);
    InterruptionMetricsModel::host_dict(py, &metrics)
}

register_fn!(interruption_metrics);

// --------------------------------------------------------------------- the host's socket

/// `http_session.ws_connect(url, headers=..)` (`_connect_ws`): the caller's `aiohttp` session as
/// the stream's socket. Every awaitable it starts is cancelled when its Rust future is dropped.
struct HostWs {
    session: CancellingHandle,
}

/// `aiohttp.WSMsgType`'s values the stream tells apart.
const WS_TEXT: i64 = 1;
const WS_CLOSE: i64 = 8;
const WS_CLOSING: i64 = 256;
const WS_CLOSED: i64 = 257;
const WS_ERROR: i64 = 258;

/// `_connect_ws`'s except block, as the transport's failures: a timeout, a rejected upgrade's
/// status (`aiohttp.ClientResponseError`), anything else a connection failure.
fn connect_failure(py: Python<'_>, err: &PyErr) -> TransportError {
    if err.is_instance_of::<pyo3::exceptions::PyTimeoutError>(py) {
        return TransportError::Timeout;
    }
    let value = err.value(py);
    let is_response_error = py
        .import("aiohttp")
        .and_then(|m| m.getattr("ClientResponseError"))
        .and_then(|cls| value.is_instance(&cls))
        .unwrap_or(false);
    if is_response_error && let Ok(status) = value.getattr("status").and_then(|s| s.extract::<i64>()) {
        let message = value.getattr("message").and_then(|m| m.extract::<String>()).unwrap_or_default();
        return TransportError::Status { status, message, headers: Vec::new(), body: None };
    }
    TransportError::Connect(err.to_string())
}

fn io(err: PyErr) -> TransportError {
    TransportError::Io(err.to_string())
}

#[async_trait::async_trait]
impl WsTransport for HostWs {
    async fn connect(
        &self,
        url: &str,
        headers: Vec<(String, String)>,
    ) -> Result<(Arc<dyn WsSender>, Box<dyn WsReceiver>), TransportError> {
        let url = url.to_owned();
        let connect = self.session.await_call(move |session| {
            let py = session.py();
            let kwargs = PyDict::new(py);
            kwargs.set_item("headers", headers.into_iter().collect::<std::collections::HashMap<_, _>>())?;
            session.call_method("ws_connect", (url,), Some(&kwargs))
        });
        let ws = match connect.await {
            Ok(ws) => ws,
            Err(err) => return Err(Python::attach(|py| connect_failure(py, &err))),
        };
        let ws = Arc::new(self.session.child(ws));
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let pump = tokio::spawn(pump_receive(Arc::clone(&ws), tx));
        Ok((
            Arc::new(HostWsSender { ws }) as Arc<dyn WsSender>,
            Box::new(HostWsReceiver { rx, pump, close_code: None }) as Box<dyn WsReceiver>,
        ))
    }
}

/// One `await ws.receive()` at a time, as aiohttp requires: `recv` reads the channel, so racing
/// it never starts a second receive.
async fn pump_receive(
    ws: Arc<CancellingHandle>,
    tx: tokio::sync::mpsc::UnboundedSender<Result<(WsMessage, Option<u16>), TransportError>>,
) {
    loop {
        let next = ws.await_call(|ws| ws.call_method0("receive")).await;
        let item = Python::attach(|py| -> Result<(WsMessage, Option<u16>), TransportError> {
            let msg = next.map_err(io)?;
            let msg = msg.bind(py);
            let kind: i64 = msg.getattr("type").and_then(|t| t.extract()).map_err(io)?;
            Ok(match kind {
                WS_TEXT => (WsMessage::Text(msg.getattr("data").and_then(|d| d.extract()).map_err(io)?), None),
                WS_CLOSE | WS_CLOSING | WS_CLOSED => {
                    let code = ws.read("close_code", |c| c.extract::<Option<u16>>()).ok().flatten();
                    (WsMessage::Closed, code)
                }
                WS_ERROR => return Err(TransportError::Io(format!("websocket error: {msg}"))),
                _ => (WsMessage::Other, None),
            })
        });
        let last = !matches!(item, Ok((WsMessage::Text(_) | WsMessage::Other, _)));
        if tx.send(item).is_err() || last {
            return;
        }
    }
}

struct HostWsSender {
    ws: Arc<CancellingHandle>,
}

#[async_trait::async_trait]
impl WsSender for HostWsSender {
    async fn send_text(&self, text: String) -> Result<(), TransportError> {
        self.ws.await_call(move |ws| ws.call_method1("send_str", (text,))).await.map(drop).map_err(io)
    }

    async fn send_binary(&self, data: Vec<u8>) -> Result<(), TransportError> {
        let send = self.ws.await_call(move |ws| {
            ws.call_method1("send_bytes", (pyo3::types::PyBytes::new(ws.py(), &data),))
        });
        send.await.map(drop).map_err(io)
    }

    fn is_closed(&self) -> bool {
        self.ws.read("closed", |c| c.extract::<bool>()).unwrap_or(true)
    }

    async fn close(&self) {
        let _ = self.ws.await_call(|ws| ws.call_method0("close")).await;
    }
}

struct HostWsReceiver {
    rx: tokio::sync::mpsc::UnboundedReceiver<Result<(WsMessage, Option<u16>), TransportError>>,
    pump: tokio::task::JoinHandle<()>,
    close_code: Option<u16>,
}

#[async_trait::async_trait]
impl WsReceiver for HostWsReceiver {
    async fn recv(&mut self) -> Result<WsMessage, TransportError> {
        match self.rx.recv().await {
            Some(Ok((msg, code))) => {
                if matches!(msg, WsMessage::Closed) {
                    self.close_code = code;
                }
                Ok(msg)
            }
            Some(Err(e)) => Err(e),
            None => Ok(WsMessage::Closed),
        }
    }

    fn close_code(&self) -> Option<u16> {
        self.close_code
    }
}

/// The pending `receive()` is cancelled with the pump: a hung socket leaves no host task behind.
impl Drop for HostWsReceiver {
    fn drop(&mut self) {
        self.pump.abort();
    }
}
