//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `turn` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/turn/inference.rs.
type InferenceEotImpl = <crate::Selected as super::Implementations>::InferenceEot;

fn _extract_inference_eot_new_sample_rate(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    ::lk_agents_abi::python::number_from_py::<u32>(&obj, "sample_rate")
}

/// The detector behind `inference.TurnDetector`: the user's threshold overrides over the model's
/// defaults, shared by every stream it opens, so a stream's cloud-to-local fallback rescales the
/// table the detector answers from.
#[pyclass(frozen, name = "InferenceEot", module = "livekit.agents._core")]
pub struct InferenceEot(::std::sync::Arc<InferenceEotImpl>);

#[pymethods]
impl InferenceEot {
    /// `turn-detector-v1` connects to the gateway at `base_url`; `local_fallback` lets it degrade
    /// to the mini model, which `local_model` builds when the build links none.
    #[new]
    #[pyo3(signature = (model, overrides, backchannel_overrides, *, local_fallback, base_url, api_key, api_secret, sample_rate, local_model, on_metrics = None))]
    fn new(model: Bound<'_, PyAny>, overrides: Option<Bound<'_, PyAny>>, backchannel_overrides: Option<Bound<'_, PyAny>>, local_fallback: bool, base_url: ::std::string::String, api_key: ::std::string::String, api_secret: ::std::string::String, #[pyo3(from_py_with = _extract_inference_eot_new_sample_rate)] sample_rate: u32, local_model: Bound<'_, PyAny>, on_metrics: Option<Bound<'_, PyAny>>) -> PyResult<InferenceEot> {
        ::lk_agents_abi::python::check_pid()?;
        let env = crate::process::env();
        let model = crate::generated::records::turn_events::turn_detector_model_from_py(&model)?;
        let overrides = overrides.map(|v| crate::generated::records::turn_events::threshold_override_from_py(&v)).transpose()?;
        let backchannel_overrides = backchannel_overrides.map(|v| crate::generated::records::turn_events::threshold_override_from_py(&v)).transpose()?;
        let local_model = crate::generated::turn::EotModelsProxy::from_host(&local_model)?;
        let on_metrics = on_metrics.map(|v| crate::generated::turn::EotMetricsObserverProxy::from_host(&v)).transpose()?;
        let out = <InferenceEotImpl as ::lk_agents_abi::turn::inference::InferenceEot>::new(&env, model, overrides, backchannel_overrides, local_fallback, base_url, api_key, api_secret, sample_rate, local_model, on_metrics);
        Ok(InferenceEot(::std::sync::Arc::new(out)))
    }

    /// The active model, which a stream's fallback changes.
    #[getter]
    fn model<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <InferenceEotImpl as ::lk_agents_abi::turn::inference::InferenceEot>::model(&self.0);
        Ok(crate::generated::records::turn_events::turn_detector_model_to_py(py, out)?)
    }

    /// `None` is `NOT_GIVEN`.
    #[getter]
    fn overrides<'py>(&self, py: Python<'py>) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <InferenceEotImpl as ::lk_agents_abi::turn::inference::InferenceEot>::overrides(&self.0);
        Ok(out.map(|value| Ok::<_, ::pyo3::PyErr>(crate::generated::records::turn_events::threshold_override_to_py(py, value)?)).transpose()?)
    }

    #[getter]
    fn backchannel_overrides<'py>(&self, py: Python<'py>) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <InferenceEotImpl as ::lk_agents_abi::turn::inference::InferenceEot>::backchannel_overrides(&self.0);
        Ok(out.map(|value| Ok::<_, ::pyo3::PyErr>(crate::generated::records::turn_events::threshold_override_to_py(py, value)?)).transpose()?)
    }

    /// `language=None` scores as English, not the global default.
    #[pyo3(signature = (language = None))]
    fn lookup(&self, language: ::std::option::Option<::lk_agents_abi::LanguageCode>) -> PyResult<::std::option::Option<f64>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <InferenceEotImpl as ::lk_agents_abi::turn::inference::InferenceEot>::lookup(&self.0, language);
        Ok(out)
    }

    /// `None` disables backchannel detection; a `0.0` threshold also means off.
    #[pyo3(signature = (language = None))]
    fn lookup_backchannel(&self, language: ::std::option::Option<::lk_agents_abi::LanguageCode>) -> PyResult<::std::option::Option<f64>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <InferenceEotImpl as ::lk_agents_abi::turn::inference::InferenceEot>::lookup_backchannel(&self.0, language);
        Ok(out)
    }

    #[pyo3(signature = (language = None))]
    fn supports(&self, language: ::std::option::Option<::lk_agents_abi::LanguageCode>) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <InferenceEotImpl as ::lk_agents_abi::turn::inference::InferenceEot>::supports(&self.0, language);
        Ok(out)
    }

    #[pyo3(signature = (overrides = None))]
    fn update_overrides(&self, overrides: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        let overrides = overrides.map(|v| crate::generated::records::turn_events::threshold_override_from_py(&v)).transpose()?;
        <InferenceEotImpl as ::lk_agents_abi::turn::inference::InferenceEot>::update_overrides(&self.0, overrides);
        Ok(())
    }

    #[pyo3(signature = (overrides = None))]
    fn update_backchannel_overrides(&self, overrides: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        let overrides = overrides.map(|v| crate::generated::records::turn_events::threshold_override_from_py(&v)).transpose()?;
        <InferenceEotImpl as ::lk_agents_abi::turn::inference::InferenceEot>::update_backchannel_overrides(&self.0, overrides);
        Ok(())
    }

    /// One stream of the user's audio. `headers` (`get_inference_headers()`) identify the job on
    /// each gateway connection.
    #[pyo3(signature = (headers, conn_options))]
    fn stream(&self, headers: ::std::vec::Vec<(::std::string::String, ::std::string::String)>, conn_options: ::lk_agents_abi::ApiConnectOptions) -> PyResult<super::turn::EotStream> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <InferenceEotImpl as ::lk_agents_abi::turn::inference::InferenceEot>::stream(&self.0, headers, conn_options).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(super::turn::EotStream(::std::sync::Arc::new(out)))
    }
}

// Declared in rust/crates/lk-agents-abi/src/turn/inference.rs.
type EotStreamImpl = <crate::Selected as super::Implementations>::EotStream;

/// `TurnDetectorStream`: one stream of the user's audio, answering `predict()` at each pause. The
/// session's own detector when handed to it.
#[pyclass(frozen, name = "EotStream", module = "livekit.agents._core")]
pub struct EotStream(::std::sync::Arc<EotStreamImpl>);

#[pymethods]
impl EotStream {
    /// The active model, which changes under the caller after a fallback.
    #[getter]
    fn model<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <EotStreamImpl as ::lk_agents_abi::turn::inference::EotStream>::model(&self.0);
        Ok(crate::generated::records::turn_events::turn_detector_model_to_py(py, out)?)
    }

    #[getter]
    fn is_fallback(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <EotStreamImpl as ::lk_agents_abi::turn::inference::EotStream>::is_fallback(&self.0);
        Ok(out)
    }

    /// True once the cloud transport is gone with no local model to replace it.
    #[getter]
    fn is_degraded(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <EotStreamImpl as ::lk_agents_abi::turn::inference::EotStream>::is_degraded(&self.0);
        Ok(out)
    }

    /// The prediction over the audio so far; supersedes any in flight.
    fn predict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <EotStreamImpl as ::lk_agents_abi::turn::inference::EotStream>::predict(&this, )
                .await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(crate::generated::records::turn_events::eot_prediction_to_py(py, out)?, py))
        })
    }

    #[pyo3(signature = (*, timed_out = false))]
    fn cancel_inference(&self, timed_out: bool) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <EotStreamImpl as ::lk_agents_abi::turn::inference::EotStream>::cancel_inference(&self.0, timed_out);
        Ok(())
    }

    /// A segment boundary, not a close.
    #[pyo3(signature = (reason = None))]
    fn flush(&self, reason: ::std::option::Option<::std::string::String>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <EotStreamImpl as ::lk_agents_abi::turn::inference::EotStream>::flush(&self.0, reason);
        Ok(())
    }

    #[pyo3(signature = (frame))]
    fn push_audio(&self, frame: ::lk_agents_abi::AudioFrame) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <EotStreamImpl as ::lk_agents_abi::turn::inference::EotStream>::push_audio(&self.0, frame);
        Ok(())
    }

    fn end_input(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <EotStreamImpl as ::lk_agents_abi::turn::inference::EotStream>::end_input(&self.0);
        Ok(())
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <EotStreamImpl as ::lk_agents_abi::turn::inference::EotStream>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::EotDetector`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::EotDetector>(),
        read: |obj| {
            let this = obj.cast::<EotStream>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::EotDetector> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/turn/inference.rs.
type InterruptionDetectorImpl = <crate::Selected as super::Implementations>::InterruptionDetector;

fn _extract_interruption_detector_new_threshold(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(&obj, "threshold")).map(Some) }
}

fn _extract_interruption_detector_new_min_interruption_duration(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "min_interruption_duration")
}

fn _extract_interruption_detector_new_max_audio_duration(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "max_audio_duration")
}

fn _extract_interruption_detector_new_audio_prefix_duration(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "audio_prefix_duration")
}

fn _extract_interruption_detector_new_detection_interval(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "detection_interval")
}

fn _extract_interruption_detector_new_inference_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "inference_timeout")
}

fn _extract_interruption_detector_new_token_ttl_secs(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "token_ttl_secs")
}

fn _extract_interruption_detector_update_options_threshold(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(&obj, "threshold")).map(Some) }
}

fn _extract_interruption_detector_update_options_min_interruption_duration(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(&obj, "min_interruption_duration")).map(Some) }
}

/// `inference.AdaptiveInterruptionDetector`: the gateway model that tells an interruption from a
/// backchannel while the agent speaks. Natively the session's interruption source.
#[pyclass(frozen, name = "InterruptionDetector", module = "livekit.agents._core")]
pub struct InterruptionDetector(::std::sync::Arc<InterruptionDetectorImpl>);

#[pymethods]
impl InterruptionDetector {
    /// Empty `base_url`, `api_key` or `api_secret` fall back to the environment.
    #[new]
    #[pyo3(signature = (*, threshold = None, min_interruption_duration = 0.05, max_audio_duration = 3.0, audio_prefix_duration = 1.0, detection_interval = 0.1, inference_timeout = 0.7, base_url = None, api_key = None, api_secret = None, conn_options = None, token_ttl_secs = 600))]
    fn new(#[pyo3(from_py_with = _extract_interruption_detector_new_threshold)] threshold: ::std::option::Option<f64>, #[pyo3(from_py_with = _extract_interruption_detector_new_min_interruption_duration)] min_interruption_duration: f64, #[pyo3(from_py_with = _extract_interruption_detector_new_max_audio_duration)] max_audio_duration: f64, #[pyo3(from_py_with = _extract_interruption_detector_new_audio_prefix_duration)] audio_prefix_duration: f64, #[pyo3(from_py_with = _extract_interruption_detector_new_detection_interval)] detection_interval: f64, #[pyo3(from_py_with = _extract_interruption_detector_new_inference_timeout)] inference_timeout: f64, base_url: ::std::option::Option<::std::string::String>, api_key: ::std::option::Option<::std::string::String>, api_secret: ::std::option::Option<::std::string::String>, conn_options: ::std::option::Option<::lk_agents_abi::ApiConnectOptions>, #[pyo3(from_py_with = _extract_interruption_detector_new_token_ttl_secs)] token_ttl_secs: u64) -> PyResult<InterruptionDetector> {
        ::lk_agents_abi::python::check_pid()?;
        let env = crate::process::env();
        let out = <InterruptionDetectorImpl as ::lk_agents_abi::turn::inference::InterruptionDetector>::new(&env, threshold, min_interruption_duration, max_audio_duration, audio_prefix_duration, detection_interval, inference_timeout, base_url, api_key, api_secret, conn_options, token_ttl_secs).map_err(|e: ::lk_agents_abi::inference::InvalidOption| <::lk_agents_abi::inference::InvalidOption as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(InterruptionDetector(::std::sync::Arc::new(out)))
    }

    /// Shared with live streams, which reconnect to pick the options up.
    #[pyo3(signature = (*, threshold = None, min_interruption_duration = None))]
    fn update_options(&self, #[pyo3(from_py_with = _extract_interruption_detector_update_options_threshold)] threshold: ::std::option::Option<f64>, #[pyo3(from_py_with = _extract_interruption_detector_update_options_min_interruption_duration)] min_interruption_duration: ::std::option::Option<f64>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <InterruptionDetectorImpl as ::lk_agents_abi::turn::inference::InterruptionDetector>::update_options(&self.0, threshold, min_interruption_duration);
        Ok(())
    }
}

// Wherever the host passes a `::lk_agents_abi::turn::interruption::InterruptionSource`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::turn::interruption::InterruptionSource>(),
        read: |obj| {
            let this = obj.cast::<InterruptionDetector>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::turn::interruption::InterruptionSource> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/turn/detector.rs.

/// [`::lk_agents_abi::EotDetector`] over a host `livekit.agents._core_view.TurnDetector`.
#[derive(Clone)]
pub struct EotDetectorProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: crate::host::Notifier,
    id: ::lk_agents_abi::ProviderId,
}

impl EotDetectorProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::EotDetector>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::EotDetector>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::EotDetector>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::EotDetector>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::EotDetector>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        let id = {
            let v = crate::host::read(obj, "_core_id")?;
            v.extract::<::lk_agents_abi::ProviderId>()?
        };
        Ok(EotDetectorProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, notifier: ::std::default::Default::default(), id })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::EotDetector>(|obj| EotDetectorProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::EotDetector for EotDetectorProxy {
    fn id(&self) -> ::lk_agents_abi::ProviderId {
        ::std::clone::Clone::clone(&self.id)
    }

    fn push_audio(&self, frame: ::lk_agents_abi::AudioFrame) {
        let host = ::std::sync::Arc::clone(&self.host);
        let frame_arg = frame;
        ::lk_agents_runtime::host::notify(&self.notifier, "EotDetector._core_push_audio", &self.context, &[], true, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(frame_arg, py)?];
                Ok((host.bind(py).getattr("_core_push_audio")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn cancel(&self, timed_out: bool) {
        let host = ::std::sync::Arc::clone(&self.host);
        let timed_out_arg = timed_out;
        ::lk_agents_runtime::host::notify(&self.notifier, "EotDetector._core_cancel", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(timed_out_arg, py)?];
                Ok((host.bind(py).getattr("_core_cancel")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn flush(&self, reason: &str) {
        let host = ::std::sync::Arc::clone(&self.host);
        let reason_arg = reason.to_owned();
        ::lk_agents_runtime::host::notify(&self.notifier, "EotDetector._core_flush", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(reason_arg, py)?];
                Ok((host.bind(py).getattr("_core_flush")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn predict(&self, timeout: ::std::option::Option<::std::time::Duration>) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::lk_agents_abi::EotPrediction, ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let timeout_arg = timeout;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "EotDetector._core_predict", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(timeout_arg.map(|d| d.as_secs_f64()), py)?];
                Ok((host.bind(py).getattr("_core_predict")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (crate::generated::records::turn_events::eot_prediction_from_py(&o)).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn aclose(&self) -> ::lk_agents_abi::BoxFuture<'_, ()> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "EotDetector._core_aclose", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_aclose")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        if let Err(e) = started.await {
            ::lk_agents_abi::python::log::warn!("EotDetector._core_aclose failed: {e}");
        }
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/turn/inference.rs.

/// [`::lk_agents_abi::turn::inference::EotModels`] over a host `collections.abc.Callable`.
#[derive(Clone)]
pub struct EotModelsProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
}

impl EotModelsProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::turn::inference::EotModels>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::turn::inference::EotModels>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::turn::inference::EotModels>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::turn::inference::EotModels>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::turn::inference::EotModels>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let _ = context;
        Ok(EotModelsProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::turn::inference::EotModels>(|obj| EotModelsProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::turn::inference::EotModels for EotModelsProxy {
    fn create(&self) -> ::std::result::Result<::std::sync::Arc<dyn ::lk_agents_abi::turn::inference::EotModel>, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("__call__")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (crate::generated::turn::EotModelProxy::from_host(&o)).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/turn/inference.rs.

/// [`::lk_agents_abi::turn::inference::EotModel`] over a host `livekit.local_inference.EOT`.
#[derive(Clone)]
pub struct EotModelProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
}

impl EotModelProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::turn::inference::EotModel>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::turn::inference::EotModel>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::turn::inference::EotModel>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::turn::inference::EotModel>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::turn::inference::EotModel>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let _ = context;
        Ok(EotModelProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::turn::inference::EotModel>(|obj| EotModelProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::turn::inference::EotModel for EotModelProxy {
    fn predict(&self, pcm: ::std::vec::Vec<i16>) -> ::std::result::Result<f64, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let pcm_arg = pcm;
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(pcm_arg, py)?];
                Ok((host.bind(py).getattr("predict")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (::lk_agents_abi::python::number_from_py::<f64>(&o, "EotModel.predict")).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/turn/inference.rs.

/// [`::lk_agents_abi::turn::inference::EotMetricsObserver`] over a host `collections.abc.Callable`.
#[derive(Clone)]
pub struct EotMetricsObserverProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
}

impl EotMetricsObserverProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::turn::inference::EotMetricsObserver>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::turn::inference::EotMetricsObserver>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::turn::inference::EotMetricsObserver>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::turn::inference::EotMetricsObserver>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::turn::inference::EotMetricsObserver>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let _ = context;
        Ok(EotMetricsObserverProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::turn::inference::EotMetricsObserver>(|obj| EotMetricsObserverProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::turn::inference::EotMetricsObserver for EotMetricsObserverProxy {
    fn collected(&self, metrics: ::lk_agents_abi::EotInferenceMetrics) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let metrics_arg = metrics;
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(metrics_arg, py)?];
                Ok((host.bind(py).getattr("__call__")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}
