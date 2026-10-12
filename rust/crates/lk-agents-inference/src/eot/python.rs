//! `_core.EotSession`, `_core.EotTransport`, `_core.EotThresholdOptions`: the decision half of
//! `inference/eot/base.py`. Hand-written because all three share one threshold table
//! (`Arc<Mutex<ThresholdOptions>>`) that a cloud->local fallback rescales under all of them.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use lk_agents_abi::tokenize::language::{LanguageCode, ThresholdOptions, ThresholdOverride, TurnDetectorModel};
use lk_agents_abi::python::{check_pid, extract_generated, provider_error_to_py};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{
    CancelReason, CloudOptions, CloudTransport, EotAction, EotRequestId, EotSession, EotTransport, EotWarning,
    LocalEotModel, LocalTransport, SinkEvent, TransportPrediction, TransportShell, TungsteniteEotConnector,
};
use crate::python::{clear, header_provider, register_class, session_clock};

fn model_from_str(model: &str) -> PyResult<TurnDetectorModel> {
    model.parse::<TurnDetectorModel>().map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The sentences stay in Python; the tag and values are [`EotWarning::wire`]'s.
fn warning_to_py<'py>(py: Python<'py>, warning: &EotWarning) -> PyResult<Bound<'py, PyDict>> {
    warning.wire().into_pyobject(py)
}

fn action_to_py<'py>(py: Python<'py>, action: &EotAction) -> PyResult<Bound<'py, PyDict>> {
    match action {
        EotAction::Resolve(v) => v.into_pyobject(py),
        EotAction::StartInference(v) => v.into_pyobject(py),
        EotAction::Flush(v) => v.into_pyobject(py),
        EotAction::SwapToLocal(v) => v.into_pyobject(py),
        EotAction::Degrade(v) => v.into_pyobject(py),
        EotAction::Warn(warning) => warning_to_py(py, warning),
    }
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
    inner: Arc<Mutex<ThresholdOptions>>,
}

register_class!(PyEotThresholdOptions);

impl PyEotThresholdOptions {
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
        override_to_py(py, self.table()?.overrides())
    }

    #[getter]
    fn backchannel_overrides<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        override_to_py(py, self.table()?.backchannel_overrides())
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
        self.table()?
            .update_defaults(&server, server_default, bc.as_ref(), backchannel_default)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// One-way and sticky: already-mini is a no-op, so a second cloud failure cannot reset the
    /// rescaled values back to stock.
    fn to_local_fallback(&self) -> PyResult<()> {
        self.table()?.to_local_fallback();
        Ok(())
    }
}

/// `_BaseStreamingTurnDetectorStream`'s decisions. One per stream.
///
/// `frozen` with a `Mutex`: a transport callback inside an action loop would otherwise hit pyo3's
/// borrow flag and get a `RuntimeError` instead of a decision.
#[pyclass(frozen, name = "EotSession", module = "livekit.agents._core")]
pub struct PyEotSession {
    inner: Mutex<EotSession>,
}

register_class!(PyEotSession);

impl PyEotSession {
    fn lock(&self) -> std::sync::MutexGuard<'_, EotSession> {
        self.inner.lock().expect("eot session")
    }

    fn act(&self, py: Python<'_>, decide: impl FnOnce(&mut EotSession) -> Vec<EotAction>) -> PyResult<Vec<Py<PyDict>>> {
        check_pid()?;
        let actions = decide(&mut self.lock());
        actions.iter().map(|a| Ok(action_to_py(py, a)?.unbind())).collect()
    }
}

#[pymethods]
impl PyEotSession {
    /// `thresholds` is shared, not copied, so the cloud->local rescale is visible from
    /// `TurnDetector.unlikely_threshold`.
    #[new]
    #[pyo3(signature = (model, thresholds, *, local_fallback = true))]
    fn new(model: &str, thresholds: PyRef<'_, PyEotThresholdOptions>, local_fallback: bool) -> PyResult<Self> {
        check_pid()?;
        Ok(PyEotSession { inner: Mutex::new(EotSession::new(model_from_str(model)?, thresholds.shared(), local_fallback)) })
    }

    /// The active model, which changes under the caller after a fallback.
    #[getter]
    fn model(&self) -> PyResult<&'static str> {
        check_pid()?;
        Ok(self.lock().model().as_str())
    }

    #[getter]
    fn is_fallback(&self) -> PyResult<bool> {
        check_pid()?;
        Ok(self.lock().is_fallback())
    }

    /// True once the cloud transport is gone with no local model: turn detection is off for the
    /// rest of the stream and turns commit on the endpointing delay.
    #[getter]
    fn is_degraded(&self) -> PyResult<bool> {
        check_pid()?;
        Ok(self.lock().is_degraded())
    }

    /// `predict()`: supersede whatever is in flight (resolving it 0.0), then start `request_id`. On
    /// a closed stream the answer is 1.0 immediately, so the waiter does not hang.
    fn start(&self, py: Python<'_>, now: f64, request_id: String) -> PyResult<Vec<Py<PyDict>>> {
        self.act(py, |s| s.start(now, EotRequestId::from_string(request_id)))
    }

    /// `_resolve_prediction`: a transport's answer. An answer for a request no longer in flight is
    /// dropped.
    #[pyo3(signature = (
        now,
        request_id,
        probability,
        *,
        detection_delay = None,
        inference_duration = None,
        backchannel_probability = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn resolve(
        &self,
        py: Python<'_>,
        now: f64,
        request_id: &str,
        probability: f64,
        detection_delay: Option<f64>,
        inference_duration: Option<f64>,
        backchannel_probability: Option<f64>,
    ) -> PyResult<Vec<Py<PyDict>>> {
        let prediction = TransportPrediction { probability, detection_delay, inference_duration, backchannel_probability };
        self.act(py, |s| s.resolve(now, request_id, prediction))
    }

    /// `cancel_inference(timed_out=)`. `request_id=None` cancels whatever is current. A timed-out
    /// cancel swaps to the local model even with nothing in flight.
    #[pyo3(signature = (now, *, timed_out = false, request_id = None))]
    fn cancel(&self, py: Python<'_>, now: f64, timed_out: bool, request_id: Option<String>) -> PyResult<Vec<Py<PyDict>>> {
        let reason = if timed_out { CancelReason::TimedOut } else { CancelReason::Superseded };
        let id = request_id.map(EotRequestId::from_string);
        self.act(py, |s| s.cancel(now, id.as_ref(), reason))
    }

    /// `flush()`. Idempotent; resolves any in-flight request with 0.0.
    fn flush(&self, py: Python<'_>, now: f64) -> PyResult<Vec<Py<PyDict>>> {
        self.act(py, |s| s.flush(now))
    }

    /// `end_input()`: flush, then close for good. A later `start` answers 1.0 without a transport.
    fn end_input(&self, py: Python<'_>, now: f64) -> PyResult<Vec<Py<PyDict>>> {
        self.act(py, |s| s.end_input(now))
    }

    /// The mounted transport died. The arm depends on the current model, so a local transport
    /// failing after a fallback warns once, answers 1.0 and stays open.
    fn transport_failed(&self, py: Python<'_>, now: f64, reason: &str) -> PyResult<Vec<Py<PyDict>>> {
        self.act(py, |s| s.transport_failed(now, reason))
    }
}

fn sink_event_to_py<'py>(py: Python<'py>, event: &SinkEvent) -> PyResult<Option<Bound<'py, PyDict>>> {
    Ok(match event {
        // Not `"resolve"`: answers go through `EotSession.resolve` so a stale one is dropped.
        SinkEvent::Prediction(v) => Some(v.into_pyobject(py)?),
        SinkEvent::Metrics(v) => Some((&**v).into_pyobject(py)?),
        SinkEvent::Warn(warning) => Some(warning_to_py(py, warning)?),
        SinkEvent::End => None,
    })
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

/// `_StreamingTurnDetectionTransport` (`base.py:41`), over
/// `lk_agents_inference::eot::{CloudTransport, LocalTransport}`. One class with two constructors,
/// so the shared protocol cannot drift between backends.
#[pyclass(frozen, name = "EotTransport", module = "livekit.agents._core")]
pub struct PyEotTransport {
    shell: Arc<TransportShell>,
}

register_class!(PyEotTransport);

#[pymethods]
impl PyEotTransport {
    /// `_CloudTransport` (`transports.py:74`): `turn-detector-v1` over the gateway WebSocket.
    ///
    /// `headers` is a callable, called per connection, so each reconnect gets a fresh
    /// `X-LiveKit-Session-ID`. It must not include `Authorization`; the crate mints that.
    #[staticmethod]
    #[pyo3(signature = (
        *,
        base_url,
        api_key,
        api_secret,
        thresholds,
        headers,
        sample_rate,
        timeout,
        max_retry,
        clock = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn cloud(
        base_url: String,
        api_key: String,
        api_secret: String,
        thresholds: PyRef<'_, PyEotThresholdOptions>,
        headers: Py<PyAny>,
        sample_rate: u32,
        timeout: f64,
        max_retry: u32,
        clock: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let clock = clock.map(|c| session_clock(c, "clock")).transpose()?;
        check_pid()?;
        let opts = CloudOptions {
            base_url,
            api_key,
            api_secret,
            conn_options: lk_agents_abi::ApiConnectOptions { timeout, max_retry, ..lk_agents_abi::DEFAULT_API_CONNECT_OPTIONS },
            sample_rate,
            headers: header_provider(headers),
        };
        // The session's clock, so handshake and retry pacing run in session time; absent, the
        // ambient one (a loop on virtual time).
        let clock = clock.unwrap_or_else(<Arc<dyn lk_agents_abi::Clock> as lk_agents_abi::python::Ambient>::ambient);
        let shell = TransportShell::mount("turn-detector-v1", thresholds.shared(), |sink| {
            let transport = CloudTransport::new(opts, sink, Arc::new(TungsteniteEotConnector)).with_clock(clock);
            Arc::new(transport) as Arc<dyn EotTransport>
        });
        Ok(PyEotTransport { shell })
    }

    /// `_LocalTransport` (`transports.py:398`): `turn-detector-v1-mini`, in process.
    ///
    /// `model` is any object with `predict(list[int]) -> float` over 16 kHz mono PCM, in practice
    /// `livekit.local_inference.EOT()`, one per stream.
    #[staticmethod]
    #[pyo3(signature = (model, thresholds))]
    fn local(model: Py<PyAny>, thresholds: PyRef<'_, PyEotThresholdOptions>) -> PyResult<Self> {
        check_pid()?;
        let model = Arc::new(PyLocalEotModel { model }) as Arc<dyn LocalEotModel>;
        let shell = TransportShell::mount("turn-detector-v1-mini", thresholds.shared(), |sink| {
            Arc::new(LocalTransport::new(model, sink)) as Arc<dyn EotTransport>
        });
        Ok(PyEotTransport { shell })
    }

    /// The gateway's `X-LiveKit-Session-ID` for this connection, or `None` (local, or not yet
    /// connected).
    #[getter]
    fn session_id(&self) -> PyResult<Option<String>> {
        check_pid()?;
        Ok(self.shell.transport.session_id())
    }

    /// `run_inference` (`base.py:52`): ask for a prediction under this id.
    fn run_inference(&self, request_id: &str) -> PyResult<()> {
        check_pid()?;
        self.shell.transport.run_inference(&EotRequestId::from_string(request_id.to_owned()));
        Ok(())
    }

    /// One frame of already-resampled audio (the host resamples, as in the reference).
    fn push_frame(&self, frame: &Bound<'_, PyAny>) -> PyResult<()> {
        check_pid()?;
        let frame: lk_agents_abi::AudioFrame = extract_generated(frame)?;
        self.shell.transport.push_frame(&frame);
        Ok(())
    }

    /// The turn ended: `session_flush` on the wire, an emptied ring buffer locally.
    fn flush(&self) -> PyResult<()> {
        check_pid()?;
        self.shell.transport.flush();
        Ok(())
    }

    /// No more audio is coming. `run()` then returns normally, which the session tells apart from
    /// a failure.
    fn end_input(&self) -> PyResult<()> {
        check_pid()?;
        self.shell.transport.end_input();
        Ok(())
    }

    /// `detach` (`base.py:57`): stop, without waiting for anything in flight.
    fn detach(&self) -> PyResult<()> {
        check_pid()?;
        self.shell.detach();
        Ok(())
    }

    /// Run the transport for its lifetime. Returning means the audio ended; raising means the
    /// transport died, and the host passes that to `EotSession.transport_failed`.
    fn run<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let shell = Arc::clone(&self.shell);
        lk_agents_abi::python::future_into_py(py, async move {
            shell.run().await.map_err(|e| provider_error_to_py(lk_agents_abi::errors::ProviderError::Local(e)))
        })
    }

    /// The next transport event, or `None` once it stops reporting. `None` is [`SinkEvent::End`],
    /// not a channel close, so everything queued before it is delivered.
    fn recv_event<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let shell = Arc::clone(&self.shell);
        lk_agents_abi::python::future_into_py(py, async move {
            match shell.next_event().await {
                None => Ok(None),
                Some(event) => Python::attach(|py| Ok(sink_event_to_py(py, &event)?.map(|d| d.unbind()))),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eot::{EotDegrade, EotFlush, EotResolve, EotStartInference, EotSwapToLocal, PROVIDER};
    use lk_agents_abi::turn::eot::{MetricsEvent, PredictionEvent};
    use lk_agents_abi::metrics::base::Metadata;
    use lk_agents_abi::{EotInferenceMetrics, EotPrediction};
    use pyo3::types::PyDictMethods;

    /// `Python::attach`, initialising the interpreter first (each nextest test is its own process).
    fn attach<T>(f: impl FnOnce(Python<'_>) -> T) -> T {
        Python::initialize();
        Python::attach(f)
    }

    fn kind(d: &Bound<'_, PyDict>) -> String {
        d.get_item("type").unwrap().unwrap().extract().unwrap()
    }

    /// Every key in the dict, sorted and compared whole, so an extra key fails too.
    fn keys(d: &Bound<'_, PyDict>) -> Vec<String> {
        let mut out: Vec<String> = d.keys().iter().map(|k| k.extract().unwrap()).collect();
        out.sort();
        out
    }

    fn prediction() -> EotPrediction {
        EotPrediction {
            end_of_turn_probability: 0.9,
            last_speaking_time: 1_700_000_000.0,
            detection_delay: Some(0.1),
            inference_duration: Some(0.02),
            backchannel_probability: Some(0.3),
        }
    }

    fn id() -> EotRequestId {
        EotRequestId::from_string("sess_eot_1".to_owned())
    }

    // Catches: an action key added, dropped or renamed. `_apply` subscripts them unvalidated; the
    // length guard makes a new `EotAction` variant fail here.
    #[test]
    fn every_action_kind_emits_exactly_the_keys_the_host_reads() {
        attach(|py| {
            let cases = vec![
                EotAction::Resolve(EotResolve::new(id(), prediction())),
                EotAction::StartInference(EotStartInference { id: id() }),
                EotAction::Flush(EotFlush {}),
                EotAction::SwapToLocal(EotSwapToLocal {}),
                EotAction::Degrade(EotDegrade {}),
                EotAction::Warn(EotWarning::UnexpectedServerMessage),
            ];
            let expected: Vec<(&str, Vec<&str>)> = vec![
                (
                    "resolve",
                    vec![
                        "backchannel_probability",
                        "detection_delay",
                        "end_of_turn_probability",
                        "id",
                        "inference_duration",
                        "last_speaking_time",
                        "type",
                    ],
                ),
                ("start_inference", vec!["id", "type"]),
                ("flush", vec!["type"]),
                ("swap_to_local", vec!["type"]),
                ("degrade", vec!["type"]),
                ("warn", vec!["reason", "type", "warning"]),
            ];

            for (action, (want_kind, want_keys)) in cases.iter().zip(expected) {
                let d = action_to_py(py, action).unwrap();
                assert_eq!(kind(&d), want_kind);
                assert_eq!(keys(&d), want_keys, "{want_kind} does not carry the keys _apply reads");
            }
        });
    }

    // Catches: a warning that carries a number arriving without it in `reason`.
    #[test]
    fn the_warnings_that_carry_numbers_carry_them_into_the_reason() {
        attach(|py| {
            let reason = |warning: EotWarning| -> Option<String> {
                let d = warning_to_py(py, &warning).unwrap();
                d.get_item("reason").unwrap().unwrap().extract().unwrap()
            };
            assert_eq!(reason(EotWarning::TransportLatency { millis: 900 }).as_deref(), Some("900 ms behind"));
            // Present as `None`, not absent: `_warn` reads it unconditionally.
            assert_eq!(reason(EotWarning::UnexpectedServerMessage), None);
        });
    }

    // Catches: a transport answer tagged `"resolve"` (skipping `EotSession.resolve`'s id check), a
    // key drifting on either event (absent metadata included), or the pump's `None` sentinel lost.
    // `total_duration` and `detection_delay` are different clocks of the same type, so their values
    // are checked too.
    #[test]
    fn the_two_transport_events_are_tagged_and_keyed_as_apply_reads_them() {
        attach(|py| {
            let d = sink_event_to_py(
                py,
                &SinkEvent::Prediction(PredictionEvent {
                    id: "sess_eot_1".to_owned(),
                    probability: 0.8,
                    detection_delay: Some(0.1),
                    inference_duration: Some(0.02),
                    backchannel_probability: Some(0.0),
                }),
            )
            .unwrap()
            .expect("a prediction is an event");
            assert_eq!(kind(&d), "prediction");
            assert_eq!(
                keys(&d),
                vec!["backchannel_probability", "detection_delay", "id", "inference_duration", "probability", "type"]
            );
            assert_eq!(d.get_item("id").unwrap().unwrap().extract::<String>().unwrap(), "sess_eot_1");

            let metrics = |metadata: Option<Metadata>| {
                SinkEvent::Metrics(Box::new(MetricsEvent::of(&EotInferenceMetrics {
                    timestamp: 1_700_000_000.0,
                    total_duration: 0.4,
                    detection_delay: 0.2,
                    prediction_duration: 0.05,
                    num_requests: EotInferenceMetrics::DEFAULT_NUM_REQUESTS,
                    metadata,
                })))
            };
            let d = sink_event_to_py(
                py,
                &metrics(Some(Metadata {
                    model_name: Some("turn-detector-v1".to_owned()),
                    model_provider: Some(PROVIDER.to_owned()),
                })),
            )
            .unwrap()
            .expect("metrics are an event");
            assert_eq!(kind(&d), "metrics");
            assert_eq!(
                keys(&d),
                vec![
                    "detection_delay",
                    "model_name",
                    "model_provider",
                    "num_requests",
                    "prediction_duration",
                    "timestamp",
                    "total_duration",
                    "type",
                ]
            );
            assert_eq!(d.get_item("total_duration").unwrap().unwrap().extract::<f64>().unwrap(), 0.4);
            assert_eq!(d.get_item("detection_delay").unwrap().unwrap().extract::<f64>().unwrap(), 0.2);

            // `Metadata(..)` raises on a missing key, so both are written even when absent.
            let d = sink_event_to_py(py, &metrics(None)).unwrap().expect("metrics are an event");
            assert!(d.get_item("model_name").unwrap().unwrap().is_none());
            assert!(d.get_item("model_provider").unwrap().unwrap().is_none());

            // The host's event pump stops only on `End`'s `None`.
            assert!(sink_event_to_py(py, &SinkEvent::End).unwrap().is_none());
        });
    }
}
