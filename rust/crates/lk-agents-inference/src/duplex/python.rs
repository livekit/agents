//! `_core.DuplexRealtimeModel`, `_core.FixedGate`, `_core.AdaptiveNoiseGate`. Hand-written: the
//! model's host is a `_core_view.Duplex` view, and a gate is either one of these classes or any
//! object with `update(frame) -> bool` and `deactivate()`.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use lk_agents_abi::llm::realtime::{Event, GenerateReply, PushAudio};
use lk_agents_abi::python::{CancellingHandle, HostBridge, check_pid, extract_generated};
use lk_agents_abi::{AudioFrame, Realtime};
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
        Python::attach(|py| -> PyResult<_> {
            let (session, gate): (Bound<'_, PyAny>, Bound<'_, PyAny>) = pair.bind(py).extract()?;
            let session = Arc::new(PyDuplexSession(self.view.child(session.unbind()))) as Arc<dyn DuplexHostSession>;
            Ok((session, gate_from_py(&gate)))
        })
        .map_err(host_error)
    }
}

/// `_core_view._DuplexSession`: the plugin's session, flattened into kind-tagged events.
struct PyDuplexSession(CancellingHandle);

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
        "output_audio" => DuplexEvent::Audio {
            frame: extract_generated(&d.get_item("frame")?)?,
            start_ms: d.get_item("start_ms")?.extract()?,
        },
        "transcript_delta" => DuplexEvent::Transcript(Fragment {
            text: d.get_item("text")?.extract()?,
            start_ms: d.get_item("start_ms")?.extract()?,
            end_ms: d.get_item("end_ms")?.extract()?,
        }),
        "output_ended" => DuplexEvent::OutputEnded { error: d.get_item("error")?.extract()? },
        _ => DuplexEvent::Session(d.extract::<Event>()?),
    })
}

#[async_trait::async_trait]
impl DuplexHostSession for PyDuplexSession {
    fn send(&self, cmd: DuplexCommand) {
        let sent = Python::attach(|py| -> PyResult<()> {
            let cmd = command_to_py(py, cmd)?;
            let send = self.0.object().bind(py).getattr("_core_send")?;
            self.0.task_locals().event_loop(py).call_method1("call_soon_threadsafe", (send, cmd))?;
            Ok(())
        });
        if let Err(e) = sent {
            Python::attach(|py| e.print(py));
        }
    }

    async fn next_event(&self) -> Option<DuplexEvent> {
        loop {
            let ev = self.0.await_call(|s| s.call_method0("_core_next_event")).await.ok()?;
            let parsed = Python::attach(|py| -> PyResult<Option<DuplexEvent>> {
                let ev = ev.bind(py);
                if ev.is_none() { Ok(None) } else { event_from_py(ev).map(Some) }
            });
            match parsed {
                Ok(ev) => return ev,
                // a malformed event is the view's bug; the session goes on
                Err(e) => log::warn!(target: "livekit.agents", error = e.to_string().as_str(); "duplex event refused"),
            }
        }
    }

    async fn aclose(&self) {
        let _ = self.0.await_call(|s| s.call_method0("_core_aclose")).await;
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
        let host = Arc::new(PyDuplexHost { view, label, caps });
        let clock = <Arc<dyn lk_agents_abi::Clock> as lk_agents_abi::python::Ambient>::ambient();
        let inner = DuplexModel::new(host, Duration::from_secs_f64(audio_timeout.max(0.0)), clock);
        Ok(PyDuplexRealtimeModel { inner: Arc::new(inner) })
    }

    /// The realtime capabilities the model runs under.
    fn capabilities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Realtime::capabilities(&*self.inner).into_pyobject(py).map(Bound::into_any)
    }
}
