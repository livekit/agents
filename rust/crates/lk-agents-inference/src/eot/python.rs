//! `_core.EotStream` and `_core.EotThresholdOptions`: `inference/eot.py`'s stream and its threshold
//! table. Hand-written because all three share one threshold table
//! (`Arc<Mutex<ThresholdOptions>>`) that a cloud->local fallback rescales under all of them.

use std::collections::BTreeMap;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::{Arc, Mutex};

use lk_agents_abi::tokenize::language::{LanguageCode, ThresholdOptions, ThresholdOverride, TurnDetectorModel};
use lk_agents_abi::python::{check_pid, extract_generated};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{
    CloudOptions, CloudTransport, EotRequestId, EotSession, EotStream, EotTransport, LocalEotModel, LocalTransport,
    MetricsCallback, TransportFactory, TransportShell, TungsteniteEotConnector,
};
use lk_agents_abi::turn::eot::MetricsEvent;
use crate::python::{clear, header_provider, register_class};

fn model_from_str(model: &str) -> PyResult<TurnDetectorModel> {
    model.parse::<TurnDetectorModel>().map_err(|e| PyValueError::new_err(e.to_string()))
}

/// `NOT_GIVEN` arrives as `None`; a float replaces (`All`); a mapping merges (`PerLanguage`).
fn override_from_py(v: Option<&Bound<'_, PyAny>>) -> PyResult<Option<ThresholdOverride>> {
    match v.filter(|v| !v.is_none()) {
        None => Ok(None),
        Some(d) if d.cast::<PyDict>().is_ok() => Ok(map_from_py(Some(d))?.map(ThresholdOverride::PerLanguage)),
        Some(v) => v
            .extract::<f64>()
            .map(|f| Some(ThresholdOverride::All(f)))
            .map_err(|_| PyTypeError::new_err("threshold overrides must be a float or a mapping of language to float")),
    }
}

fn override_to_py<'py>(py: Python<'py>, o: Option<&ThresholdOverride>) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(match o {
        None => None,
        Some(ThresholdOverride::All(f)) => Some(f.into_pyobject(py)?.into_any()),
        Some(ThresholdOverride::PerLanguage(m)) => Some(m.into_pyobject(py)?.into_any()),
    })
}

fn map_from_py(d: Option<&Bound<'_, PyAny>>) -> PyResult<Option<BTreeMap<String, f64>>> {
    let Some(d) = d.filter(|d| !d.is_none()) else { return Ok(None) };
    let entries = d.cast::<PyDict>()?.iter().map(|(k, v)| Ok((k.str()?.to_string(), v.extract::<f64>()?)));
    entries.collect::<PyResult<_>>().map(Some)
}

/// `eot/languages.py`'s `ThresholdOptions`.
///
/// A shared-mutable `#[pyclass]`: the detector and its stream hold the same instance, so a
/// cloud->local fallback in the stream is visible from the detector.
#[pyclass(name = "EotThresholdOptions", module = "livekit.agents._core")]
pub struct PyEotThresholdOptions {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    inner: Arc<Mutex<ThresholdOptions>>,
}

register_class!(PyEotThresholdOptions);

impl PyEotThresholdOptions {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    fn shared(&self) -> Arc<Mutex<ThresholdOptions>> {
        Arc::clone(&self.inner)
    }

    fn table(&self) -> PyResult<std::sync::MutexGuard<'_, ThresholdOptions>> {
        check_pid()?;
        Ok(self.inner.lock().expect("thresholds"))
    }
}

#[pymethods]
impl PyEotThresholdOptions {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    #[new]
    #[pyo3(signature = (model, overrides = None, backchannel_overrides = None))]
    fn new(model: &str, overrides: Option<&Bound<'_, PyAny>>, backchannel_overrides: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        check_pid()?;
        Ok(PyEotThresholdOptions {
            inner: Arc::new(Mutex::new(ThresholdOptions::new(
                model_from_str(model)?,
                override_from_py(overrides)?,
                override_from_py(backchannel_overrides)?,
            ))),
        })
    }

    #[getter]
    fn model(&self) -> PyResult<&'static str> {
        Ok(self.table()?.model().as_str())
    }

    /// `None` is `NOT_GIVEN`; the host re-wraps it.
    #[getter]
    fn overrides<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let overrides = self.table()?.overrides().cloned();
        override_to_py(py, overrides.as_ref())
    }

    #[getter]
    fn backchannel_overrides<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let overrides = self.table()?.backchannel_overrides().cloned();
        override_to_py(py, overrides.as_ref())
    }

    #[getter]
    fn thresholds(&self) -> PyResult<BTreeMap<String, f64>> {
        Ok(self.table()?.thresholds().clone())
    }

    #[getter]
    fn default_threshold(&self) -> PyResult<Option<f64>> {
        Ok(self.table()?.default_threshold())
    }

    /// `language=None` scores as English, not the global default.
    #[pyo3(signature = (language = None))]
    fn lookup(&self, language: Option<&str>) -> PyResult<Option<f64>> {
        Ok(self.table()?.lookup(language.map(LanguageCode::new).as_ref()))
    }

    /// `None` disables backchannel detection; a `0.0` threshold also means off.
    #[pyo3(signature = (language = None))]
    fn lookup_backchannel(&self, language: Option<&str>) -> PyResult<Option<f64>> {
        Ok(self.table()?.lookup_backchannel(language.map(LanguageCode::new).as_ref()))
    }

    #[pyo3(signature = (language = None))]
    fn supports(&self, language: Option<&str>) -> PyResult<bool> {
        Ok(self.table()?.supports(language.map(LanguageCode::new).as_ref()))
    }

    #[pyo3(signature = (overrides = None))]
    fn update_overrides(&self, overrides: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let o = override_from_py(overrides)?;
        self.table()?.update_overrides(o);
        Ok(())
    }

    #[pyo3(signature = (overrides = None))]
    fn update_backchannel_overrides(&self, overrides: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let o = override_from_py(overrides)?;
        self.table()?.update_backchannel_overrides(o);
        Ok(())
    }

    /// Install the defaults a `SessionCreated` carried. Raises `ValueError` for an empty map or a
    /// non-positive default; the host turns that into the reference's non-retryable `APIError`.
    #[pyo3(signature = (
        server_thresholds,
        server_default,
        backchannel_thresholds = None,
        backchannel_default = 0.0,
    ))]
    fn update_defaults(
        &self,
        server_thresholds: &Bound<'_, PyAny>,
        server_default: f64,
        backchannel_thresholds: Option<&Bound<'_, PyAny>>,
        backchannel_default: f64,
    ) -> PyResult<()> {
        let server = map_from_py(Some(server_thresholds))?.unwrap_or_default();
        let bc = map_from_py(backchannel_thresholds)?;
        let result = self.table()?.update_defaults(
            &server,
            server_default,
            bc.as_ref(),
            backchannel_default,
        );
        result.map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// One-way and sticky: already-mini is a no-op, so a second cloud failure cannot reset the
    /// rescaled values back to stock.
    fn to_local_fallback(&self) -> PyResult<()> {
        self.table()?.to_local_fallback();
        Ok(())
    }
}

/// The mini model as the host object that owns the weights; no ONNX runtime here.
struct PyLocalEotModel {
    model: Py<PyAny>,
}

impl LocalEotModel for PyLocalEotModel {
    /// A raise becomes a warning and a 0.0 score (the reference's), not 1.0, which would cut the
    /// user off at every pause.
    fn predict(&self, pcm: &[i16]) -> Result<f64, String> {
        Python::attach(|py| {
            self.model.bind(py).call_method1("predict", (pcm.to_vec(),)).and_then(|out| out.extract::<f64>()).map_err(|e| {
                let message = e.to_string();
                // Cleared: on a blocking thread a live error indicator would surface on the next
                // unrelated GIL call.
                clear(py, e);
                message
            })
        })
    }
}

/// `TurnDetectorStream` (`inference/eot.py`): one stream of the user's audio, answering
/// `predict()` at each pause. The session's own detector when handed to it (`EotDetector`).
#[pyclass(frozen, name = "EotStream", module = "livekit.agents._core")]
pub struct PyEotStream {
    inner: Arc<EotStream>,
}

register_class!(PyEotStream);

lk_agents_abi::python::inventory::submit! {
    lk_agents_abi::python::HostNative {
        trait_id: || std::any::TypeId::of::<dyn lk_agents_abi::EotDetector>(),
        read: |obj| {
            use pyo3::types::PyAnyMethods as _;
            let n = obj.extract::<PyRef<'_, PyEotStream>>().ok()?;
            let v: Arc<dyn lk_agents_abi::EotDetector> = n.inner.clone();
            Some(Box::new(v) as Box<dyn std::any::Any>)
        },
    }
}

#[pymethods]
impl PyEotStream {
    /// `model` is the detector's starting model; `turn-detector-v1` connects to the gateway
    /// (`base_url`, credentials, `headers` called per connection, `timeout`/`max_retry`), the mini
    /// model runs natively when the build links it, else `livekit.local_inference.EOT()`, also
    /// after a cloud failure when
    /// `local_fallback`. `on_metrics(dict)` hears each `EOTInferenceMetrics`.
    #[new]
    #[pyo3(signature = (
        model,
        thresholds,
        *,
        local_fallback,
        base_url,
        api_key,
        api_secret,
        headers,
        sample_rate,
        timeout,
        max_retry,
        on_metrics = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: &str,
        thresholds: PyRef<'_, PyEotThresholdOptions>,
        local_fallback: bool,
        base_url: String,
        api_key: String,
        api_secret: String,
        headers: Py<PyAny>,
        sample_rate: u32,
        timeout: f64,
        max_retry: u32,
        on_metrics: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        check_pid()?;
        let shared = thresholds.shared();
        let session = EotSession::new(model_from_str(model)?, shared.clone(), local_fallback);
        let clock = <Arc<dyn lk_agents_abi::Clock> as lk_agents_abi::python::Ambient>::ambient();
        let spawner = <Arc<dyn lk_agents_abi::Spawner> as lk_agents_abi::python::Ambient>::ambient();
        let headers = Arc::new(headers);
        let make: TransportFactory = Arc::new(move |model| {
            if model == TurnDetectorModel::V1 {
                let opts = CloudOptions {
                    base_url: base_url.clone(),
                    api_key: api_key.clone(),
                    api_secret: api_secret.clone(),
                    conn_options: lk_agents_abi::ApiConnectOptions {
                        timeout,
                        max_retry,
                        ..lk_agents_abi::DEFAULT_API_CONNECT_OPTIONS
                    },
                    sample_rate,
                    headers: header_provider(Python::attach(|py| headers.clone_ref(py))),
                };
                let clock = clock.clone();
                return TransportShell::mount(model.as_str(), shared.clone(), move |sink| {
                    let transport = CloudTransport::new(opts, sink, Arc::new(TungsteniteEotConnector)).with_clock(clock);
                    Arc::new(transport) as Arc<dyn EotTransport>
                });
            }
            // The linked native model, else the host's.
            let local = match crate::local_models::native() {
                Some(native) => Ok((native.eot)()),
                None => Python::attach(|py| -> PyResult<Arc<dyn LocalEotModel>> {
                    let model = py.import("livekit.local_inference")?.getattr("EOT")?.call0()?.unbind();
                    Ok(Arc::new(PyLocalEotModel { model }) as Arc<dyn LocalEotModel>)
                }),
            };
            TransportShell::mount(model.as_str(), shared.clone(), move |sink| match local {
                Ok(model) => Arc::new(LocalTransport::new(model, sink)) as Arc<dyn EotTransport>,
                Err(e) => Arc::new(FailedTransport(e.to_string())) as Arc<dyn EotTransport>,
            })
        });
        let on_metrics = on_metrics.map(|cb| {
            Arc::new(move |ev: MetricsEvent| {
                Python::attach(|py| {
                    if let Err(e) = ev.into_pyobject(py).and_then(|d| cb.bind(py).call1((d,))) {
                        e.print(py);
                    }
                })
            }) as MetricsCallback
        });
        Ok(PyEotStream { inner: EotStream::open(session, sample_rate, make, on_metrics, spawner.as_ref()) })
    }

    /// The active model, which changes under the caller after a fallback.
    #[getter]
    fn model(&self) -> &'static str {
        self.inner.model()
    }

    /// It hears the user's audio (the session's `turn_detector_streaming`).
    #[getter]
    fn streaming(&self) -> bool {
        true
    }

    #[getter]
    fn is_fallback(&self) -> bool {
        self.inner.is_fallback()
    }

    /// True once the cloud transport is gone with no local model to replace it.
    #[getter]
    fn is_degraded(&self) -> bool {
        self.inner.is_degraded()
    }

    /// The prediction over the audio so far (an `EotPrediction` dict); supersedes any in flight.
    fn predict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let rx = self.inner.predict();
        lk_agents_abi::python::future_into_py(py, async move {
            let p = rx.await.map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("turn detector stream closed"))?;
            Python::attach(|py| -> PyResult<Py<PyDict>> {
                let d = PyDict::new(py);
                d.set_item("type", "eot_prediction")?;
                d.set_item("end_of_turn_probability", p.end_of_turn_probability)?;
                d.set_item("last_speaking_time", p.last_speaking_time)?;
                d.set_item("detection_delay", p.detection_delay)?;
                d.set_item("inference_duration", p.inference_duration)?;
                d.set_item("backchannel_probability", p.backchannel_probability)?;
                Ok(d.unbind())
            })
        })
    }

    #[pyo3(signature = (*, timed_out = false))]
    fn cancel_inference(&self, timed_out: bool) {
        self.inner.cancel_inference(timed_out);
    }

    #[pyo3(signature = (reason = None))]
    fn flush(&self, reason: Option<&str>) {
        let _ = reason;
        self.inner.flush();
    }

    fn push_audio(&self, frame: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.push_audio(extract_generated(frame)?);
        Ok(())
    }

    fn end_input(&self) {
        self.inner.end_input();
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.aclose();
        let stream = self.inner.clone();
        lk_agents_abi::python::future_into_py(py, async move {
            stream.wait_closed().await;
            Ok(())
        })
    }
}

/// A local model that could not be loaded: its `run()` fails, which the session handles as a
/// local failure (warned once, every prediction 1.0).
struct FailedTransport(String);

#[async_trait::async_trait]
impl EotTransport for FailedTransport {
    fn session_id(&self) -> Option<String> {
        None
    }
    fn run_inference(&self, _request_id: &EotRequestId) {}
    fn push_frame(&self, _frame: &lk_agents_abi::AudioFrame) {}
    fn flush(&self) {}
    fn end_input(&self) {}
    fn detach(&self) {}
    async fn run(&self) -> Result<(), lk_agents_abi::errors::ApiError> {
        Err(lk_agents_abi::errors::ApiError::new(self.0.clone()))
    }
}
