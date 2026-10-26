//! `_core.DuplexRealtimeModel`, `_core.FixedGate`, `_core.AdaptiveNoiseGate`. Hand-written: the
//! model's host is a `_core_view.Duplex` view, and a gate is either one of these classes or any
//! object with `update(frame) -> bool` and `deactivate()`.

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use lk_agents_abi::llm::realtime::{
    Event, GenerateReply, GenerateReplyFailed, HostRealtimeError, PushAudio, RealtimeError,
    RealtimeModelError,
};
use lk_agents_abi::python::{CancellingHandle, FromHostErr, HostBridge, check_pid, extract_generated};
use lk_agents_abi::{AbortHandle, AudioFrame, Realtime, Spawner, SpawnerExt};
use tokio::sync::mpsc;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use super::{
    AdaptiveNoiseGate, AudioGate, DuplexCapabilities, DuplexCommand, DuplexEvent, DuplexHost,
    DuplexHostSession, DuplexModel, FixedGate, Fragment,
};
use crate::python::{clear, register_class};
use lk_agents_abi::errors::{ApiError, ProviderError};

// ------------------------------------------------------------------------------------ gates

/// A gate the host holds and the session drives: one state, whoever calls.
#[derive(Clone)]
struct SharedGate(Arc<Mutex<dyn AudioGate>>);

impl AudioGate for SharedGate {
    fn update(&mut self, frame: &AudioFrame) -> Result<bool, String> {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).update(frame)
    }

    fn deactivate(&mut self) {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).deactivate();
    }
}

/// `FixedGate`: opens on output that stands out from a silence the plugin already knows.
#[pyclass(frozen, name = "FixedGate", module = "livekit.agents._core")]
pub struct PyFixedGate(SharedGate);

/// `AdaptiveNoiseGate`: opens on output that stands out from the model's own silence.
#[pyclass(frozen, name = "AdaptiveNoiseGate", module = "livekit.agents._core")]
pub struct PyAdaptiveNoiseGate(SharedGate);

register_class!(PyFixedGate, PyAdaptiveNoiseGate);

#[pymethods]
impl PyFixedGate {
    #[new]
    #[pyo3(signature = (silence, *, activation_ratio = 3.0, deactivation_ratio = 1.8, min_silence_duration = 0.5))]
    fn new(silence: f64, activation_ratio: f64, deactivation_ratio: f64, min_silence_duration: f64) -> Self {
        let gate = FixedGate::new(silence, activation_ratio, deactivation_ratio, min_silence_duration);
        PyFixedGate(SharedGate(Arc::new(Mutex::new(gate))))
    }

    /// True while the frame belongs to an open burst of output.
    fn update(&self, frame: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.0.clone().update(&extract_generated(frame)?).map_err(PyRuntimeError::new_err)
    }

    /// End the open burst, keeping whatever has been learned about the model.
    fn deactivate(&self) {
        self.0.clone().deactivate();
    }
}

#[pymethods]
impl PyAdaptiveNoiseGate {
    #[new]
    #[pyo3(signature = (*, activation_ratio = 3.0, deactivation_ratio = 1.8, min_silence_duration = 0.5, window = 10.0))]
    fn new(activation_ratio: f64, deactivation_ratio: f64, min_silence_duration: f64, window: f64) -> Self {
        let gate = AdaptiveNoiseGate::new(activation_ratio, deactivation_ratio, min_silence_duration, window);
        PyAdaptiveNoiseGate(SharedGate(Arc::new(Mutex::new(gate))))
    }

    /// True while the frame belongs to an open burst of output.
    fn update(&self, frame: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.0.clone().update(&extract_generated(frame)?).map_err(PyRuntimeError::new_err)
    }

    /// End the open burst, keeping whatever has been learned about the model.
    fn deactivate(&self) {
        self.0.clone().deactivate();
    }
}

/// Any other object with the `AudioGate` methods, called under the GIL frame by frame.
struct HostGate(Py<PyAny>);

impl AudioGate for HostGate {
    fn update(&mut self, frame: &AudioFrame) -> Result<bool, String> {
        Python::attach(|py| {
            let frame = frame.clone().into_pyobject(py).map_err(|e| e.to_string())?;
            self.0.bind(py).call_method1("update", (frame,)).and_then(|v| v.is_truthy()).map_err(|e| {
                let message = format!("{}: {}", e.get_type(py).qualname().map(|n| n.to_string()).unwrap_or_default(), e.value(py));
                clear(py, e);
                message
            })
        })
    }

    fn deactivate(&mut self) {
        Python::attach(|py| {
            if let Err(e) = self.0.bind(py).call_method0("deactivate") {
                e.print(py);
            }
        });
    }
}

fn gate_from_py(obj: &Bound<'_, PyAny>) -> Option<Box<dyn AudioGate>> {
    if obj.is_none() {
        return None;
    }
    if let Ok(g) = obj.cast::<PyFixedGate>() {
        return Some(Box::new(g.get().0.clone()));
    }
    if let Ok(g) = obj.cast::<PyAdaptiveNoiseGate>() {
        return Some(Box::new(g.get().0.clone()));
    }
    Some(Box::new(HostGate(obj.clone().unbind())))
}

// ------------------------------------------------------------------------------------ the host

/// `_core_view.Duplex`: the adapter's model.
struct PyDuplexHost {
    view: CancellingHandle,
    label: String,
    caps: DuplexCapabilities,
    spawner: Arc<dyn Spawner>,
}

fn host_error(e: PyErr) -> ProviderError {
    ProviderError::Local(ApiError::new(e.to_string()))
}

#[async_trait::async_trait]
impl DuplexHost for PyDuplexHost {
    fn label(&self) -> String {
        self.label.clone()
    }

    fn capabilities(&self) -> DuplexCapabilities {
        self.caps
    }

    async fn open(&self) -> Result<(Arc<dyn DuplexHostSession>, Option<Box<dyn AudioGate>>), ProviderError> {
        let pair = self.view.await_call(|v| v.call_method0("_core_session")).await.map_err(host_error)?;
        let (session, audio, gate) = Python::attach(|py| -> PyResult<_> {
            let (session, gate): (Bound<'_, PyAny>, Bound<'_, PyAny>) = pair.bind(py).extract()?;
            let audio = session.call_method0("_core_audio_stream")?;
            Ok((self.view.child(session.unbind()), self.view.child(audio.unbind()), gate_from_py(&gate)))
        })
        .map_err(host_error)?;
        Ok((PyDuplexSession::open(session, audio, self.label.clone(), &self.spawner), gate))
    }
}

/// `_core_view._DuplexSession`: the plugin's session. Updates apply one after the other, the
/// output audio is pulled here, and the session's callbacks report the rest one event at a time.
struct PyDuplexSession {
    view: Arc<CancellingHandle>,
    lane: mpsc::UnboundedSender<DuplexCommand>,
    /// Every event, the audio and the refused replies among them; `None` once the session closed.
    events: tokio::sync::Mutex<mpsc::UnboundedReceiver<Option<DuplexEvent>>>,
    tx: mpsc::UnboundedSender<Option<DuplexEvent>>,
    spawner: Arc<dyn Spawner>,
    tasks: Mutex<Vec<AbortHandle>>,
}

fn command_to_py<'py>(py: Python<'py>, cmd: DuplexCommand) -> PyResult<Bound<'py, PyAny>> {
    let d = PyDict::new(py);
    match cmd {
        DuplexCommand::PushAudio(frame) => return Ok((&PushAudio { frame }).into_pyobject(py)?.into_any()),
        DuplexCommand::GenerateReply { req_id, instructions, tool_choice } => {
            return Ok((&GenerateReply { req_id, instructions, tool_choice }).into_pyobject(py)?.into_any());
        }
        DuplexCommand::UpdateInstructions(instructions) => {
            d.set_item("kind", "update_instructions")?;
            d.set_item("instructions", instructions)?;
        }
        DuplexCommand::AppendItems(items) => {
            d.set_item("kind", "append_items")?;
            d.set_item("items", &items)?;
        }
        DuplexCommand::UpdateTools(names) => {
            d.set_item("kind", "update_tools")?;
            d.set_item("names", names)?;
        }
        DuplexCommand::Configure { instructions, items, tools } => {
            d.set_item("kind", "configure")?;
            d.set_item("instructions", instructions)?;
            d.set_item("items", items.as_ref())?;
            d.set_item("names", tools)?;
        }
    }
    Ok(d.into_any())
}

fn event_from_py(d: &Bound<'_, PyAny>) -> PyResult<DuplexEvent> {
    let kind: String = d.get_item("kind")?.extract()?;
    Ok(match kind.as_str() {
        "transcript_delta" => DuplexEvent::Transcript(Fragment {
            text: d.get_item("text")?.extract()?,
            start_ms: d.get_item("start_ms")?.extract()?,
            end_ms: d.get_item("end_ms")?.extract()?,
        }),
        _ => DuplexEvent::Session(d.extract::<Event>()?),
    })
}

/// `{type(e).__name__}: {e}`, as the output stream's failure is reported.
fn describe(py: Python<'_>, e: &PyErr) -> String {
    let name = e.get_type(py).name().map(|n| n.to_string()).unwrap_or_default();
    format!("{name}: {}", e.value(py))
}

fn wall_now() -> f64 {
    SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or_default()
}

impl PyDuplexSession {
    fn open(view: CancellingHandle, audio: CancellingHandle, label: String, spawner: &Arc<dyn Spawner>) -> Arc<Self> {
        let (lane, lane_rx) = mpsc::unbounded_channel();
        let (tx, events) = mpsc::unbounded_channel();
        let session = Arc::new(PyDuplexSession {
            view: Arc::new(view),
            lane,
            events: tokio::sync::Mutex::new(events),
            tx: tx.clone(),
            spawner: Arc::clone(spawner),
            tasks: Mutex::new(Vec::new()),
        });
        session.spawn(Self::forward(Arc::clone(&session.view), tx.clone()));
        session.spawn(Self::pump_audio(audio, tx.clone()));
        session.spawn(Self::run_lane(Arc::clone(&session.view), lane_rx, label, tx));
        session
    }

    fn spawn(&self, f: impl std::future::Future<Output = ()> + Send + 'static) {
        let mut tasks = self.tasks.lock().unwrap_or_else(|p| p.into_inner());
        tasks.retain(|t| !t.is_finished());
        tasks.push(self.spawner.spawn_unboxed(f));
    }

    /// The session callbacks' events, until the view reports the end.
    async fn forward(view: Arc<CancellingHandle>, tx: mpsc::UnboundedSender<Option<DuplexEvent>>) {
        loop {
            let Ok(ev) = view.await_call(|s| s.call_method0("_core_next_event")).await else { return };
            let parsed = Python::attach(|py| -> PyResult<Option<DuplexEvent>> {
                let ev = ev.bind(py);
                if ev.is_none() { Ok(None) } else { event_from_py(ev).map(Some) }
            });
            match parsed {
                Ok(None) => {
                    let _ = tx.send(None);
                    return;
                }
                Ok(Some(ev)) => {
                    if tx.send(Some(ev)).is_err() {
                        return;
                    }
                }
                // a malformed event is the view's bug; the session goes on
                Err(e) => log::warn!(target: "livekit.agents", error = e.to_string().as_str(); "duplex event refused"),
            }
        }
    }

    /// The model's output audio, then how it ended.
    async fn pump_audio(audio: CancellingHandle, tx: mpsc::UnboundedSender<Option<DuplexEvent>>) {
        let error = loop {
            match audio.anext().await {
                Ok(Some(item)) => {
                    let ev = Python::attach(|py| -> PyResult<DuplexEvent> {
                        let item = item.bind(py);
                        Ok(DuplexEvent::Audio {
                            frame: extract_generated(&item.getattr("frame")?)?,
                            start_ms: item.getattr("start_ms")?.extract()?,
                        })
                    });
                    match ev {
                        Ok(ev) => {
                            let _ = tx.send(Some(ev));
                        }
                        Err(e) => break Some(Python::attach(|py| describe(py, &e))),
                    }
                }
                Ok(None) => break None,
                Err(e) => break Some(Python::attach(|py| describe(py, &e))),
            }
        };
        let _ = tx.send(Some(DuplexEvent::OutputEnded { error }));
    }

    /// The configuration and its updates, each applied once the previous one has. A session that
    /// cannot carry its startup configuration is unusable: it reports the error and closes.
    async fn run_lane(
        view: Arc<CancellingHandle>,
        mut lane: mpsc::UnboundedReceiver<DuplexCommand>,
        label: String,
        tx: mpsc::UnboundedSender<Option<DuplexEvent>>,
    ) {
        while let Some(cmd) = lane.recv().await {
            let configure = matches!(cmd, DuplexCommand::Configure { .. });
            let applied = view.await_call(|s| s.call_method1("_core_update", (command_to_py(s.py(), cmd)?,))).await;
            let Err(e) = applied else { continue };
            let message = Python::attach(|py| e.value(py).to_string());
            if !configure {
                log::error!(target: "livekit.agents", error = message.as_str(); "failed to update the duplex session");
                continue;
            }
            let error = RealtimeModelError { label: label.clone(), message, recoverable: false, ..Default::default() };
            let _ = tx.send(Some(DuplexEvent::Session(Event::HostError(HostRealtimeError { error, timestamp: wall_now() }))));
            let _ = view.await_call(|s| s.call_method0("_core_close_plugin")).await;
        }
    }

    /// A reply the plugin refused is reported, so the turn waiting for it fails.
    async fn generate(view: Arc<CancellingHandle>, cmd: DuplexCommand, req_id: String, tx: mpsc::UnboundedSender<Option<DuplexEvent>>) {
        let asked = view.await_call(|s| s.call_method1("_core_generate", (command_to_py(s.py(), cmd)?,))).await;
        if let Err(e) = asked {
            let error = Python::attach(|py| <RealtimeError as FromHostErr>::from_host(py, &e));
            let _ = tx.send(Some(DuplexEvent::Session(Event::GenerateReplyFailed(GenerateReplyFailed { req_id, error }))));
        }
    }
}

#[async_trait::async_trait]
impl DuplexHostSession for PyDuplexSession {
    fn send(&self, cmd: DuplexCommand) {
        match cmd {
            DuplexCommand::PushAudio(frame) => {
                let sent = Python::attach(|py| -> PyResult<()> {
                    let cmd = (&PushAudio { frame }).into_pyobject(py)?;
                    let push = self.view.object().bind(py).getattr("_core_push_audio")?;
                    self.view.task_locals().event_loop(py).call_method1("call_soon_threadsafe", (push, cmd))?;
                    Ok(())
                });
                if let Err(e) = sent {
                    Python::attach(|py| e.print(py));
                }
            }
            DuplexCommand::GenerateReply { ref req_id, .. } => {
                let req_id = req_id.clone();
                self.spawn(Self::generate(Arc::clone(&self.view), cmd, req_id, self.tx.clone()));
            }
            DuplexCommand::UpdateInstructions(_)
            | DuplexCommand::AppendItems(_)
            | DuplexCommand::UpdateTools(_)
            | DuplexCommand::Configure { .. } => {
                let _ = self.lane.send(cmd);
            }
        }
    }

    async fn next_event(&self) -> Option<DuplexEvent> {
        self.events.lock().await.recv().await.flatten()
    }

    async fn aclose(&self) {
        for task in std::mem::take(&mut *self.tasks.lock().unwrap_or_else(|p| p.into_inner())) {
            task.abort();
        }
        let _ = self.view.await_call(|s| s.call_method0("_core_aclose")).await;
    }
}

// ------------------------------------------------------------------------------------ the model

/// `DuplexRealtimeAdapter`'s engine, the session's realtime model when handed to it.
#[pyclass(frozen, name = "DuplexRealtimeModel", module = "livekit.agents._core")]
pub struct PyDuplexRealtimeModel {
    inner: Arc<DuplexModel>,
}

register_class!(PyDuplexRealtimeModel);

lk_agents_abi::python::inventory::submit! {
    lk_agents_abi::python::HostNative {
        trait_id: || std::any::TypeId::of::<dyn Realtime>(),
        read: |obj| {
            let n = obj.extract::<PyRef<'_, PyDuplexRealtimeModel>>().ok()?;
            let v: Arc<dyn Realtime> = n.inner.clone();
            Some(Box::new(v) as Box<dyn std::any::Any>)
        },
    }
}

#[pymethods]
impl PyDuplexRealtimeModel {
    /// `host` is the adapter's `_core_view.Duplex`; `audio_timeout` how long to wait for the next
    /// frame before the output counts as finished.
    #[new]
    #[pyo3(signature = (host, *, audio_timeout = 0.8))]
    fn new(py: Python<'_>, host: Bound<'_, PyAny>, audio_timeout: f64) -> PyResult<Self> {
        check_pid()?;
        let label: String = host.call_method0("_core_label")?.extract()?;
        let caps = host.call_method0("_core_capabilities")?;
        let get = |k: &str| -> PyResult<bool> { caps.get_item(k)?.extract() };
        let caps = DuplexCapabilities {
            user_transcription: get("user_transcription")?,
            auto_tool_reply_generation: get("auto_tool_reply_generation")?,
            mutable_chat_context: get("mutable_chat_context")?,
            mutable_instructions: get("mutable_instructions")?,
            mutable_tools: get("mutable_tools")?,
        };
        let view = CancellingHandle::new(py, host.unbind())?;
        let spawner = <Arc<dyn Spawner> as lk_agents_abi::python::Ambient>::ambient();
        let host = Arc::new(PyDuplexHost { view, label, caps, spawner });
        let clock = <Arc<dyn lk_agents_abi::Clock> as lk_agents_abi::python::Ambient>::ambient();
        let inner = DuplexModel::new(host, Duration::from_secs_f64(audio_timeout.max(0.0)), clock);
        Ok(PyDuplexRealtimeModel { inner: Arc::new(inner) })
    }

    /// The realtime capabilities the model runs under.
    fn capabilities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Realtime::capabilities(&*self.inner).into_pyobject(py).map(Bound::into_any)
    }
}
