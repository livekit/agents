//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `turn` classes.

#![allow(clippy::too_many_arguments, reason = "a host signature takes the parameters it declares")]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/turn/inference.rs.
type InferenceEotImpl =
    <crate::generated::implementations::Selected as super::Implementations>::InferenceEot;

fn _extract_inference_eot_new_sample_rate(
    obj: &Bound<'_, PyAny>,
) -> PyResult<InferenceEotNewSampleRateInput> {
    crate::py_values::number_from_py::<u32>(obj.as_any(), "sample_rate")
}
type InferenceEotNewSampleRateInput = u32;

/// The detector behind `inference.TurnDetector`: the user's threshold overrides over the model's
/// defaults, shared by every stream it opens, so a stream's cloud-to-local fallback rescales the
/// table the detector answers from.
#[pyclass(frozen, name = "InferenceEot", module = "livekit.agents._core")]
pub struct InferenceEot(pub(crate) ::std::sync::Arc<InferenceEotImpl>);

#[pymethods]
impl InferenceEot {
    /// `turn-detector-v1` connects to the gateway at `base_url`; `local_fallback` lets it degrade
    /// to the mini model, which `local_model` builds when the build links none.
    #[new]
    #[pyo3(signature = (model, overrides, backchannel_overrides, *, local_fallback, base_url, api_key, api_secret, sample_rate, local_model, on_metrics = None))]
    fn new(
        py: Python<'_>,
        model: Bound<'_, PyAny>,
        overrides: Option<Bound<'_, PyAny>>,
        backchannel_overrides: Option<Bound<'_, PyAny>>,
        local_fallback: bool,
        base_url: ::std::string::String,
        api_key: ::std::string::String,
        api_secret: ::std::string::String,
        #[pyo3(from_py_with = _extract_inference_eot_new_sample_rate)]
        sample_rate: InferenceEotNewSampleRateInput,
        local_model: Bound<'_, PyAny>,
        on_metrics: Option<Bound<'_, PyAny>>,
    ) -> PyResult<InferenceEot> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let model =
            crate::generated::records::turn_events::turn_detector_model_from_py(model.as_any())?;
        let overrides = overrides
            .map(|v| crate::generated::records::turn_events::threshold_override_from_py(v.as_any()))
            .transpose()?;
        let backchannel_overrides = backchannel_overrides
            .map(|v| crate::generated::records::turn_events::threshold_override_from_py(v.as_any()))
            .transpose()?;
        let local_model =
            crate::generated::turn::EotModelFactoryProxy::from_host(local_model.as_any())?;
        let on_metrics = on_metrics
            .map(|v| crate::generated::turn::EotMetricsObserverProxy::from_host(v.as_any()))
            .transpose()?;
        let out = py.detach(|| {
            <InferenceEotImpl as ::lk_agents_abi::turn::inference::InferenceEot>::new(
                &env,
                model,
                overrides,
                backchannel_overrides,
                local_fallback,
                base_url,
                api_key,
                api_secret,
                sample_rate,
                local_model,
                on_metrics,
            )
        });
        ::pyo3::PyResult::Ok(InferenceEot(::std::sync::Arc::new(out)))
    }

    /// The active model, which a stream's fallback changes.
    #[getter]
    fn model<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <InferenceEotImpl as ::lk_agents_abi::turn::inference::InferenceEot>::model(&self.0)
        });
        crate::generated::records::turn_events::turn_detector_model_to_py(py, out)
    }

    /// `None` is `NOT_GIVEN`.
    #[getter]
    fn overrides<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <InferenceEotImpl as ::lk_agents_abi::turn::inference::InferenceEot>::overrides(&self.0)
        });
        out.map(|value| crate::generated::records::turn_events::threshold_override_to_py(py, value))
            .transpose()
    }

    #[getter]
    fn backchannel_overrides<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <InferenceEotImpl as ::lk_agents_abi::turn::inference::InferenceEot>::backchannel_overrides(&self.0));
        out.map(|value| crate::generated::records::turn_events::threshold_override_to_py(py, value))
            .transpose()
    }

    #[pyo3(signature = (overrides = None))]
    fn update_overrides(
        &self,
        py: Python<'_>,
        overrides: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let overrides = overrides
            .map(|v| crate::generated::records::turn_events::threshold_override_from_py(v.as_any()))
            .transpose()?;
        py.detach(|| {
            <InferenceEotImpl as ::lk_agents_abi::turn::inference::InferenceEot>::update_overrides(
                &self.0, overrides,
            )
        });
        Ok(())
    }

    #[pyo3(signature = (overrides = None))]
    fn update_backchannel_overrides(
        &self,
        py: Python<'_>,
        overrides: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let overrides = overrides
            .map(|v| crate::generated::records::turn_events::threshold_override_from_py(v.as_any()))
            .transpose()?;
        py.detach(|| <InferenceEotImpl as ::lk_agents_abi::turn::inference::InferenceEot>::update_backchannel_overrides(&self.0, overrides));
        Ok(())
    }

    /// One stream of the user's audio; each gateway connection carries the identity of the job
    /// `cx` runs for.
    #[pyo3(signature = (conn_options))]
    fn stream(
        &self,
        py: Python<'_>,
        conn_options: Bound<'_, PyAny>,
    ) -> PyResult<super::turn::TurnDetectorStream> {
        crate::py_calls::check_pid()?;
        let cx = crate::py_calls::entry_cx()?;
        let conn_options =
            crate::generated::records::connect::api_connect_options_from_py(conn_options.as_any())?;
        let out = py.detach(|| <InferenceEotImpl as ::lk_agents_abi::turn::inference::InferenceEot>::stream(&self.0, cx, conn_options)).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(super::turn::TurnDetectorStream(::std::sync::Arc::new(out)))
    }
}

// Declared in rust/crates/lk-agents-abi/src/turn/inference.rs.
type TurnDetectorStreamImpl =
    <crate::generated::implementations::Selected as super::Implementations>::EotStream;

/// `TurnDetectorStream`: one stream of the user's audio, answering `predict()` at each pause. The
/// session's own detector when handed to it, which implements its stream operations.
#[pyclass(frozen, name = "TurnDetectorStream", module = "livekit.agents.inference")]
pub struct TurnDetectorStream(pub(crate) ::std::sync::Arc<TurnDetectorStreamImpl>);

#[pymethods]
impl TurnDetectorStream {
    /// The active model, which changes under the caller after a fallback.
    #[getter]
    fn model<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <TurnDetectorStreamImpl as ::lk_agents_abi::turn::inference::EotStream>::model(&self.0)
        });
        crate::generated::records::turn_events::turn_detector_model_to_py(py, out)
    }

    #[getter]
    fn is_fallback(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <TurnDetectorStreamImpl as ::lk_agents_abi::turn::inference::EotStream>::is_fallback(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// True once the cloud transport is gone with no local model to replace it.
    #[getter]
    fn is_degraded(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <TurnDetectorStreamImpl as ::lk_agents_abi::turn::inference::EotStream>::is_degraded(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// The prediction over the audio so far; supersedes any in flight.
    fn predict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <TurnDetectorStreamImpl as ::lk_agents_abi::turn::inference::EotStream>::predict(&this, )
                .await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    crate::generated::records::turn_events::eot_prediction_to_py(py, out)?,
                    py,
                )
            })
        })
    }

    #[pyo3(signature = (*, timed_out = false))]
    fn cancel_inference(&self, py: Python<'_>, timed_out: bool) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <TurnDetectorStreamImpl as ::lk_agents_abi::turn::inference::EotStream>::cancel_inference(&self.0, timed_out));
        Ok(())
    }

    /// A segment boundary, not a close.
    #[pyo3(signature = (reason = None))]
    fn flush(&self, py: Python<'_>, reason: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let reason = reason
            .map(|v| crate::generated::records::turn_events::flush_reason_from_py(v.as_any()))
            .transpose()?;
        py.detach(|| {
            <TurnDetectorStreamImpl as ::lk_agents_abi::turn::inference::EotStream>::flush(
                &self.0, reason,
            )
        });
        Ok(())
    }

    #[pyo3(signature = (frame))]
    fn push_audio(&self, py: Python<'_>, frame: Bound<'_, PyAny>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let frame = crate::generated::records::rtc_frames::audio_frame_from_py(frame.as_any())?;
        py.detach(|| {
            <TurnDetectorStreamImpl as ::lk_agents_abi::turn::inference::EotStream>::push_audio(
                &self.0, frame,
            )
        });
        Ok(())
    }

    fn end_input(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <TurnDetectorStreamImpl as ::lk_agents_abi::turn::inference::EotStream>::end_input(
                &self.0,
            )
        });
        Ok(())
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () =
                <TurnDetectorStreamImpl as ::lk_agents_abi::turn::inference::EotStream>::aclose(
                    &this,
                )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::EotDetector`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::EotDetector>(),
        read: |obj| match obj.cast::<TurnDetectorStream>() {
            Ok(this) => {
                let native: ::std::sync::Arc<dyn ::lk_agents_abi::EotDetector> = this.get().0.clone();
                Ok(crate::py_objects::Taken::Native(::std::boxed::Box::new(native)))
            }
            Err(_) => Ok(crate::py_objects::Taken::Other),
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/turn/inference.rs.
type AdaptiveInterruptionDetectorImpl =
    <crate::generated::implementations::Selected as super::Implementations>::InterruptionDetector;

fn _extract_interruption_detector_new_threshold(
    obj: &Bound<'_, PyAny>,
) -> PyResult<InterruptionDetectorNewThresholdInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<f64>(obj.as_any(), "threshold")).map(Some)
    }
}
type InterruptionDetectorNewThresholdInput = ::std::option::Option<f64>;

fn _extract_interruption_detector_new_min_interruption_duration(
    obj: &Bound<'_, PyAny>,
) -> PyResult<InterruptionDetectorNewMinInterruptionDurationInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "min_interruption_duration")
}
type InterruptionDetectorNewMinInterruptionDurationInput = f64;

fn _extract_interruption_detector_new_max_audio_duration(
    obj: &Bound<'_, PyAny>,
) -> PyResult<InterruptionDetectorNewMaxAudioDurationInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "max_audio_duration")
}
type InterruptionDetectorNewMaxAudioDurationInput = f64;

fn _extract_interruption_detector_new_audio_prefix_duration(
    obj: &Bound<'_, PyAny>,
) -> PyResult<InterruptionDetectorNewAudioPrefixDurationInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "audio_prefix_duration")
}
type InterruptionDetectorNewAudioPrefixDurationInput = f64;

fn _extract_interruption_detector_new_detection_interval(
    obj: &Bound<'_, PyAny>,
) -> PyResult<InterruptionDetectorNewDetectionIntervalInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "detection_interval")
}
type InterruptionDetectorNewDetectionIntervalInput = f64;

fn _extract_interruption_detector_new_inference_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<InterruptionDetectorNewInferenceTimeoutInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "inference_timeout")
}
type InterruptionDetectorNewInferenceTimeoutInput = f64;

fn _extract_interruption_detector_new_token_ttl_secs(
    obj: &Bound<'_, PyAny>,
) -> PyResult<InterruptionDetectorNewTokenTtlSecsInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "token_ttl_secs")
}
type InterruptionDetectorNewTokenTtlSecsInput = u64;

fn _extract_interruption_detector_update_options_threshold(
    obj: &Bound<'_, PyAny>,
) -> PyResult<InterruptionDetectorUpdateOptionsThresholdInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<f64>(obj.as_any(), "threshold")).map(Some)
    }
}
type InterruptionDetectorUpdateOptionsThresholdInput = ::std::option::Option<f64>;

fn _extract_interruption_detector_update_options_min_interruption_duration(
    obj: &Bound<'_, PyAny>,
) -> PyResult<InterruptionDetectorUpdateOptionsMinInterruptionDurationInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<f64>(obj.as_any(), "min_interruption_duration"))
            .map(Some)
    }
}
type InterruptionDetectorUpdateOptionsMinInterruptionDurationInput = ::std::option::Option<f64>;

/// `inference.AdaptiveInterruptionDetector`: the gateway model that tells an interruption from a
/// backchannel while the agent speaks. Natively the session's interruption source.
#[pyclass(
    frozen,
    subclass,
    name = "AdaptiveInterruptionDetector",
    module = "livekit.agents.inference"
)]
pub struct AdaptiveInterruptionDetector(
    pub(crate) ::std::sync::Arc<AdaptiveInterruptionDetectorImpl>,
);

#[pymethods]
impl AdaptiveInterruptionDetector {
    /// `base_url`, `api_key` and `api_secret` default to the inference ladders over the server's
    /// settings (`inference_credentials`).
    #[new]
    #[pyo3(signature = (*, threshold = None, min_interruption_duration = 0.05, max_audio_duration = 3.0, audio_prefix_duration = 1.0, detection_interval = 0.1, inference_timeout = 0.7, base_url = None, api_key = None, api_secret = None, conn_options = None, token_ttl_secs = 600))]
    fn new(
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_interruption_detector_new_threshold)]
        threshold: InterruptionDetectorNewThresholdInput,
        #[pyo3(from_py_with = _extract_interruption_detector_new_min_interruption_duration)]
        min_interruption_duration: InterruptionDetectorNewMinInterruptionDurationInput,
        #[pyo3(from_py_with = _extract_interruption_detector_new_max_audio_duration)]
        max_audio_duration: InterruptionDetectorNewMaxAudioDurationInput,
        #[pyo3(from_py_with = _extract_interruption_detector_new_audio_prefix_duration)]
        audio_prefix_duration: InterruptionDetectorNewAudioPrefixDurationInput,
        #[pyo3(from_py_with = _extract_interruption_detector_new_detection_interval)]
        detection_interval: InterruptionDetectorNewDetectionIntervalInput,
        #[pyo3(from_py_with = _extract_interruption_detector_new_inference_timeout)]
        inference_timeout: InterruptionDetectorNewInferenceTimeoutInput,
        base_url: ::std::option::Option<::std::string::String>,
        api_key: ::std::option::Option<::std::string::String>,
        api_secret: ::std::option::Option<::std::string::String>,
        conn_options: Option<Bound<'_, PyAny>>,
        #[pyo3(from_py_with = _extract_interruption_detector_new_token_ttl_secs)]
        token_ttl_secs: InterruptionDetectorNewTokenTtlSecsInput,
    ) -> PyResult<AdaptiveInterruptionDetector> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let conn_options = conn_options
            .map(|v| crate::generated::records::connect::api_connect_options_from_py(v.as_any()))
            .transpose()?;
        let out = py.detach(|| <AdaptiveInterruptionDetectorImpl as ::lk_agents_abi::turn::inference::InterruptionDetector>::new(&env, threshold, min_interruption_duration, max_audio_duration, audio_prefix_duration, detection_interval, inference_timeout, base_url, api_key, api_secret, conn_options, token_ttl_secs)).map_err(|e: ::lk_agents_abi::errors::Invalid| <::lk_agents_abi::errors::Invalid as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(AdaptiveInterruptionDetector(::std::sync::Arc::new(out)))
    }

    /// Shared with live streams, which reconnect to pick the options up.
    #[pyo3(signature = (*, threshold = None, min_interruption_duration = None))]
    fn update_options(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_interruption_detector_update_options_threshold)]
        threshold: InterruptionDetectorUpdateOptionsThresholdInput,
        #[pyo3(from_py_with = _extract_interruption_detector_update_options_min_interruption_duration)] min_interruption_duration: InterruptionDetectorUpdateOptionsMinInterruptionDurationInput,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <AdaptiveInterruptionDetectorImpl as ::lk_agents_abi::turn::inference::InterruptionDetector>::update_options(&self.0, threshold, min_interruption_duration));
        Ok(())
    }
}

// Wherever the host passes a `::lk_agents_abi::turn::interruption::InterruptionSource`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::turn::interruption::InterruptionSource>(),
        read: |obj| match obj.cast::<AdaptiveInterruptionDetector>() {
            Ok(this) => {
                let native: ::std::sync::Arc<dyn ::lk_agents_abi::turn::interruption::InterruptionSource> = this.get().0.clone();
                Ok(crate::py_objects::Taken::Native(::std::boxed::Box::new(native)))
            }
            Err(_) => Ok(crate::py_objects::Taken::Other),
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/turn/keyterms.rs.
type KeytermFunctionsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::KeytermFunctions;

/// `_resolve_detection`: the defaults, then whatever keys the caller set. `turn_interval`
/// comes back clamped to at least 1, and an empty `instructions` resolves to the default prompt.
#[pyfunction]
#[pyo3(signature = (config))]
pub fn resolve_keyterm_detection<'py>(
    py: Python<'py>,
    config: Option<Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let config = config
        .map(|v| crate::generated::records::keyterms::keyterm_detection_options_from_py(v.as_any()))
        .transpose()?;
    let out = py.detach(|| <KeytermFunctionsImpl as ::lk_agents_abi::turn::keyterms::KeytermFunctions>::resolve_keyterm_detection(config));
    crate::generated::records::keyterms::resolved_keyterm_detection_to_py(py, out)
}

// Declared in rust/crates/lk-agents-abi/src/turn/inference.rs.

/// [`::lk_agents_abi::turn::inference::EotModelFactory`] over a host `collections.abc.Callable`.
#[derive(Clone)]
pub struct EotModelFactoryProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
}

impl EotModelFactoryProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::turn::inference::EotModelFactory>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::turn::inference::EotModelFactory>(obj)?
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::turn::inference::EotModelFactory>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::turn::inference::EotModelFactory>(obj)?
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::turn::inference::EotModelFactory>> {
        if crate::py_objects::native::<dyn ::lk_agents_abi::turn::inference::EotModelFactory>(obj)?
            .is_some()
        {
            return Err(::pyo3::exceptions::PyTypeError::new_err(
                "a native `EotModelFactory` is shared, never handed over to be owned",
            ));
        }
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let _ = context;
        Ok(EotModelFactoryProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

impl ::lk_agents_abi::turn::inference::EotModelFactory for EotModelFactoryProxy {
    fn create(
        &self,
    ) -> ::std::result::Result<
        ::std::sync::Arc<dyn ::lk_agents_abi::turn::inference::EotModel>,
        ::lk_agents_abi::errors::HostError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("__call__")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let out = (crate::generated::turn::EotModelProxy::from_host(o.as_any())).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::turn::inference::EotModel>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::turn::inference::EotModel>(obj)?
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::turn::inference::EotModel>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::turn::inference::EotModel>(obj)?
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::turn::inference::EotModel>> {
        if crate::py_objects::native::<dyn ::lk_agents_abi::turn::inference::EotModel>(obj)?
            .is_some()
        {
            return Err(::pyo3::exceptions::PyTypeError::new_err(
                "a native `EotModel` is shared, never handed over to be owned",
            ));
        }
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let _ = context;
        Ok(EotModelProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

impl ::lk_agents_abi::turn::inference::EotModel for EotModelProxy {
    fn predict(
        &self,
        pcm: ::std::vec::Vec<i16>,
    ) -> ::std::result::Result<f64, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let pcm_arg = pcm;
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(pcm_arg, py)?];
                Ok((
                    host.bind(py).getattr("predict")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let out = (crate::py_values::number_from_py::<f64>(o.as_any(), "EotModel.predict")).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::turn::inference::EotMetricsObserver>> {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::turn::inference::EotMetricsObserver,
        >(obj)?
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::turn::inference::EotMetricsObserver>> {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::turn::inference::EotMetricsObserver,
        >(obj)?
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::turn::inference::EotMetricsObserver>> {
        if crate::py_objects::native::<dyn ::lk_agents_abi::turn::inference::EotMetricsObserver>(
            obj,
        )?
        .is_some()
        {
            return Err(::pyo3::exceptions::PyTypeError::new_err(
                "a native `EotMetricsObserver` is shared, never handed over to be owned",
            ));
        }
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let _ = context;
        Ok(EotMetricsObserverProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

impl ::lk_agents_abi::turn::inference::EotMetricsObserver for EotMetricsObserverProxy {
    fn collected(
        &self,
        metrics: ::lk_agents_abi::EotInferenceMetrics,
    ) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let metrics_arg = metrics;
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::metrics::eot_inference_metrics_to_py(
                        py,
                        metrics_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("__call__")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let () = ({
                let _ = o;
                ::pyo3::PyResult::Ok(())
            })
            .map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(())
        })
    }
}
