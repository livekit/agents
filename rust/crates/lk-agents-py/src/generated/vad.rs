//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `vad` classes.

#![allow(clippy::too_many_arguments, reason = "a host signature takes the parameters it declares")]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/vad/inference.rs.
type InferenceVadImpl =
    <crate::generated::implementations::Selected as super::Implementations>::InferenceVad;

fn _extract_inference_vad_new_min_speech_duration(
    obj: &Bound<'_, PyAny>,
) -> PyResult<InferenceVadNewMinSpeechDurationInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "min_speech_duration")
}
type InferenceVadNewMinSpeechDurationInput = f64;

fn _extract_inference_vad_new_min_silence_duration(
    obj: &Bound<'_, PyAny>,
) -> PyResult<InferenceVadNewMinSilenceDurationInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "min_silence_duration")
}
type InferenceVadNewMinSilenceDurationInput = f64;

fn _extract_inference_vad_new_prefix_padding_duration(
    obj: &Bound<'_, PyAny>,
) -> PyResult<InferenceVadNewPrefixPaddingDurationInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "prefix_padding_duration")
}
type InferenceVadNewPrefixPaddingDurationInput = f64;

fn _extract_inference_vad_new_max_buffered_speech(
    obj: &Bound<'_, PyAny>,
) -> PyResult<InferenceVadNewMaxBufferedSpeechInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "max_buffered_speech")
}
type InferenceVadNewMaxBufferedSpeechInput = f64;

fn _extract_inference_vad_new_activation_threshold(
    obj: &Bound<'_, PyAny>,
) -> PyResult<InferenceVadNewActivationThresholdInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "activation_threshold")
}
type InferenceVadNewActivationThresholdInput = f64;

fn _extract_inference_vad_new_deactivation_threshold(
    obj: &Bound<'_, PyAny>,
) -> PyResult<InferenceVadNewDeactivationThresholdInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<f64>(obj.as_any(), "deactivation_threshold")).map(Some)
    }
}
type InferenceVadNewDeactivationThresholdInput = ::std::option::Option<f64>;

/// The Silero VAD behind `inference.VAD`.
#[pyclass(frozen, weakref, subclass, name = "InferenceVad", module = "livekit.agents._core")]
pub struct InferenceVad(pub(crate) ::std::sync::OnceLock<::std::sync::Arc<InferenceVadImpl>>);

#[pymethods]
impl InferenceVad {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(
        _args: &Bound<'_, ::pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, ::pyo3::types::PyDict>>,
    ) -> Self {
        Self(::std::sync::OnceLock::new())
    }

    /// `model_factory` builds the host's model when the build links no native one;
    /// `deactivation_threshold` is `max(activation_threshold - 0.15, 0.01)` when `None`. Only the
    /// `"silero"` model exists.
    #[pyo3(signature = (model_factory, *, model = None, min_speech_duration = 0.05, min_silence_duration = 0.25, prefix_padding_duration = 0.5, max_buffered_speech = 60.0, activation_threshold = 0.5, deactivation_threshold = None))]
    fn __init__(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        model_factory: Bound<'_, PyAny>,
        model: Option<Bound<'_, PyAny>>,
        #[pyo3(from_py_with = _extract_inference_vad_new_min_speech_duration)] min_speech_duration: InferenceVadNewMinSpeechDurationInput,
        #[pyo3(from_py_with = _extract_inference_vad_new_min_silence_duration)]
        min_silence_duration: InferenceVadNewMinSilenceDurationInput,
        #[pyo3(from_py_with = _extract_inference_vad_new_prefix_padding_duration)]
        prefix_padding_duration: InferenceVadNewPrefixPaddingDurationInput,
        #[pyo3(from_py_with = _extract_inference_vad_new_max_buffered_speech)] max_buffered_speech: InferenceVadNewMaxBufferedSpeechInput,
        #[pyo3(from_py_with = _extract_inference_vad_new_activation_threshold)]
        activation_threshold: InferenceVadNewActivationThresholdInput,
        #[pyo3(from_py_with = _extract_inference_vad_new_deactivation_threshold)]
        deactivation_threshold: InferenceVadNewDeactivationThresholdInput,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let once =
            || ::pyo3::exceptions::PyRuntimeError::new_err("InferenceVad.__init__ runs once");
        if slf.get().0.get().is_some() {
            return Err(once());
        }
        let env = crate::process::env();
        let label = crate::py_objects::class_label(slf.as_any())?;
        let model_factory =
            crate::generated::vad::VadModelsProxy::from_host(model_factory.as_any())?;
        let model = match model {
            Some(v) => crate::generated::records::inference_models::vad_models_from_py(v.as_any())?,
            None => "silero".parse::<::lk_agents_abi::inference::models::VadModelName>().map_err(
                |e| {
                    ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(
                        &e,
                    ))
                },
            )?,
        };
        let out = py
            .detach(|| {
                <InferenceVadImpl as ::lk_agents_abi::vad::inference::InferenceVad>::new(
                    &env,
                    label,
                    model_factory,
                    model,
                    min_speech_duration,
                    min_silence_duration,
                    prefix_padding_duration,
                    max_buffered_speech,
                    activation_threshold,
                    deactivation_threshold,
                )
            })
            .map_err(|e: ::lk_agents_abi::errors::Invalid| {
                <::lk_agents_abi::errors::Invalid as crate::py_errors::HostException>::to_py_err(&e)
            })?;
        slf.get().0.set(::std::sync::Arc::new(out)).map_err(|_| once())
    }

    #[getter]
    fn model<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let engine = self.engine()?;
        let out = py.detach(|| {
            <InferenceVadImpl as ::lk_agents_abi::vad::inference::InferenceVad>::model(engine)
        });
        crate::generated::records::inference_models::vad_models_to_py(py, out)
    }

    #[getter]
    fn capabilities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let engine = self.engine()?;
        let out = py.detach(|| {
            <InferenceVadImpl as ::lk_agents_abi::vad::inference::InferenceVad>::capabilities(
                engine,
            )
        });
        crate::generated::records::vad::vad_capabilities_to_py(py, out)
    }

    /// The given options, applied by each running stream before its next frame.
    #[pyo3(name = "_core_update_options")]
    #[pyo3(signature = (*, min_speech_duration = crate::py_values::Omittable::NOT_GIVEN, min_silence_duration = crate::py_values::Omittable::NOT_GIVEN, prefix_padding_duration = crate::py_values::Omittable::NOT_GIVEN, max_buffered_speech = crate::py_values::Omittable::NOT_GIVEN, activation_threshold = crate::py_values::Omittable::NOT_GIVEN, deactivation_threshold = crate::py_values::Omittable::NOT_GIVEN))]
    fn update_options(
        &self,
        py: Python<'_>,
        min_speech_duration: crate::py_values::Omittable<'_>,
        min_silence_duration: crate::py_values::Omittable<'_>,
        prefix_padding_duration: crate::py_values::Omittable<'_>,
        max_buffered_speech: crate::py_values::Omittable<'_>,
        activation_threshold: crate::py_values::Omittable<'_>,
        deactivation_threshold: crate::py_values::Omittable<'_>,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let engine = self.engine()?;
        let min_speech_duration = min_speech_duration.map(|value| {
            crate::py_values::number_from_py::<f64>(value.as_any(), "min_speech_duration")
        })?;
        let min_silence_duration = min_silence_duration.map(|value| {
            crate::py_values::number_from_py::<f64>(value.as_any(), "min_silence_duration")
        })?;
        let prefix_padding_duration = prefix_padding_duration.map(|value| {
            crate::py_values::number_from_py::<f64>(value.as_any(), "prefix_padding_duration")
        })?;
        let max_buffered_speech = max_buffered_speech.map(|value| {
            crate::py_values::number_from_py::<f64>(value.as_any(), "max_buffered_speech")
        })?;
        let activation_threshold = activation_threshold.map(|value| {
            crate::py_values::number_from_py::<f64>(value.as_any(), "activation_threshold")
        })?;
        let deactivation_threshold = deactivation_threshold.map(|value| {
            crate::py_values::number_from_py::<f64>(value.as_any(), "deactivation_threshold")
        })?;
        py.detach(|| {
            <InferenceVadImpl as ::lk_agents_abi::vad::inference::InferenceVad>::update_options(
                engine,
                min_speech_duration,
                min_silence_duration,
                prefix_padding_duration,
                max_buffered_speech,
                activation_threshold,
                deactivation_threshold,
            )
        });
        Ok(())
    }

    #[getter]
    fn min_silence_duration(&self, py: Python<'_>) -> PyResult<f64> {
        crate::py_calls::check_pid()?;
        let engine = self.engine()?;
        let out = py.detach(|| <InferenceVadImpl as ::lk_agents_abi::vad::inference::InferenceVad>::min_silence_duration(engine));
        ::pyo3::PyResult::Ok(out)
    }

    /// One detection; each event carries the audio it refers to.
    #[pyo3(name = "_core_native_stream")]
    fn stream(&self, py: Python<'_>) -> PyResult<InferenceVadStream> {
        crate::py_calls::check_pid()?;
        let engine = self.engine()?;
        let out = py.detach(|| <InferenceVadImpl as ::lk_agents_abi::vad::inference::InferenceVad>::stream(engine)).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok({
            let (sink, source) = out.split();
            InferenceVadStream(
                ::std::boxed::Box::new(sink),
                ::lk_agents_runtime::net::source::SharedStream::new(
                    ::std::boxed::Box::new(source),
                    ::lk_agents_runtime::process::host_clock(),
                ),
            )
        })
    }
}

impl InferenceVad {
    /// The implementation its `__init__` built.
    pub(crate) fn engine(&self) -> PyResult<&::std::sync::Arc<InferenceVadImpl>> {
        self.0.get().ok_or_else(|| crate::py_objects::unbuilt("InferenceVad"))
    }
}

// Wherever the host passes a `::lk_agents_abi::Vad`, an instance of this class is itself once built.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::Vad>(),
        read: |obj| match obj.cast::<InferenceVad>() {
            Ok(this) => {
                let engine = this.get().0.get().cloned().map(|engine| -> ::std::sync::Arc<dyn ::lk_agents_abi::Vad> { engine });
                crate::py_objects::built(obj, engine, "InferenceVad")
            }
            Err(_) => Ok(crate::py_objects::Taken::Other),
        },
    }
}

type InferenceVadStreamImpl = dyn ::lk_agents_abi::StreamInput<::lk_agents_abi::AudioFrame>;

/// One duplex stream: `push()` input, `recv()` resolves to the next output, `None` at the end.
#[pyclass(frozen, name = "InferenceVadStream", module = "livekit.agents._core")]
pub struct InferenceVadStream(
    ::std::boxed::Box<InferenceVadStreamImpl>,
    ::lk_agents_runtime::net::source::SharedStream<::lk_agents_abi::VadEvent>,
);

#[pymethods]
impl InferenceVadStream {
    fn push(&self, py: Python<'_>, item: Bound<'_, PyAny>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let item = crate::generated::records::rtc_frames::audio_frame_from_py(item.as_any())?;
        self.1.admit_input().map_err(|e: ::lk_agents_abi::errors::ProviderError| {
            <::lk_agents_abi::errors::ProviderError as crate::py_errors::HostException>::to_py_err(
                &e,
            )
        })?;
        py.detach(|| ::lk_agents_abi::StreamInput::push(&*self.0, item)).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::py_errors::HostException>::to_py_err(&e))
    }

    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        self.1.admit_input().map_err(|e: ::lk_agents_abi::errors::ProviderError| {
            <::lk_agents_abi::errors::ProviderError as crate::py_errors::HostException>::to_py_err(
                &e,
            )
        })?;
        py.detach(|| ::lk_agents_abi::StreamInput::flush(&*self.0)).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::py_errors::HostException>::to_py_err(&e))
    }

    fn end_input(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        self.1.input_failure().map_or(Ok(()), Err).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::py_errors::HostException>::to_py_err(&e))?;
        py.detach(|| ::lk_agents_abi::StreamInput::end_input(&*self.0));
        Ok(())
    }

    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        crate::py_streams::recv_native(&self.1, py, |py, item| {
            Ok(crate::generated::records::vad::vad_event_to_py(py, item)?.unbind())
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        crate::py_streams::close_native(&self.1, py)
    }
}

// Declared in rust/crates/lk-agents-abi/src/vad/inference.rs.

/// [`::lk_agents_abi::vad::inference::VadModels`] over a host `collections.abc.Callable`.
#[derive(Clone)]
pub struct VadModelsProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
}

impl VadModelsProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::vad::inference::VadModels>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::vad::inference::VadModels>(obj)?
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::vad::inference::VadModels>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::vad::inference::VadModels>(obj)?
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::vad::inference::VadModels>> {
        if crate::py_objects::native::<dyn ::lk_agents_abi::vad::inference::VadModels>(obj)?
            .is_some()
        {
            return Err(::pyo3::exceptions::PyTypeError::new_err(
                "a native `VadModels` is shared, never handed over to be owned",
            ));
        }
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let _ = context;
        Ok(VadModelsProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

impl ::lk_agents_abi::vad::inference::VadModels for VadModelsProxy {
    fn create(
        &self,
    ) -> ::std::result::Result<
        ::std::sync::Arc<dyn ::lk_agents_abi::vad::inference::VadModel>,
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
            let out = (crate::generated::vad::VadModelProxy::from_host(o.as_any())).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/vad/inference.rs.

/// [`::lk_agents_abi::vad::inference::VadModel`] over a host `livekit.local_inference.VAD`.
#[derive(Clone)]
pub struct VadModelProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
}

impl VadModelProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::vad::inference::VadModel>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::vad::inference::VadModel>(obj)?
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::vad::inference::VadModel>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::vad::inference::VadModel>(obj)?
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::vad::inference::VadModel>> {
        if crate::py_objects::native::<dyn ::lk_agents_abi::vad::inference::VadModel>(obj)?
            .is_some()
        {
            return Err(::pyo3::exceptions::PyTypeError::new_err(
                "a native `VadModel` is shared, never handed over to be owned",
            ));
        }
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let _ = context;
        Ok(VadModelProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

impl ::lk_agents_abi::vad::inference::VadModel for VadModelProxy {
    fn predict(
        &self,
        window: ::std::vec::Vec<i16>,
    ) -> ::std::result::Result<f64, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let window_arg = window;
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(window_arg, py)?];
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
            let out = (crate::py_values::number_from_py::<f64>(o.as_any(), "VadModel.predict")).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn reset(&self) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("reset")?, ::pyo3::types::PyTuple::new(py, args)?, None))
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

// Declared in rust/crates/lk-agents-abi/src/vad/vad.rs.

/// [`::lk_agents_abi::Vad`] over a host `livekit.agents.vad.VAD`.
#[derive(Clone)]
pub struct VadProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken;
    /// none when taken with no running loop: each call then runs in its driving operation's.
    context: ::std::option::Option<::lk_agents_abi::utils::host_ref::HostRef>,
    /// The `notify` calls, delivered in order.
    notifier: crate::py_calls::Notifier,
    label: ::std::string::String,
    capabilities: ::lk_agents_abi::VadCapabilities,
}

impl VadProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::Vad>> {
        if let Some(native) = crate::py_objects::native::<dyn ::lk_agents_abi::Vad>(obj)? {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::Vad>> {
        if let Some(native) = crate::py_objects::native::<dyn ::lk_agents_abi::Vad>(obj)? {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::Vad>> {
        if crate::py_objects::native::<dyn ::lk_agents_abi::Vad>(obj)?.is_some() {
            return Err(::pyo3::exceptions::PyTypeError::new_err(
                "a native `Vad` is shared, never handed over to be owned",
            ));
        }
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => Some(::std::clone::Clone::clone(context)),
            None => crate::py_calls::proxy_context(obj.py()),
        };
        let label = {
            let v = crate::py_calls::read(obj, "_core_label")?;
            v.extract::<::std::string::String>()?
        };
        let capabilities = {
            let v = crate::py_calls::read(obj, "_core_capabilities")?;
            crate::generated::records::vad::vad_capabilities_from_py(v.as_any())?
        };
        Ok(VadProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            context,
            notifier: ::std::default::Default::default(),
            label,
            capabilities,
        })
    }
}

impl ::lk_agents_abi::Vad for VadProxy {
    fn label(&self) -> ::std::string::String {
        ::std::clone::Clone::clone(&self.label)
    }

    fn capabilities(&self) -> ::lk_agents_abi::VadCapabilities {
        ::std::clone::Clone::clone(&self.capabilities)
    }

    fn read_id(
        &self,
    ) -> ::std::result::Result<::lk_agents_abi::ProviderId, ::lk_agents_abi::errors::HostError>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::lk_agents_abi::agent::HostContext::host_context(
            &crate::py_calls::calling_context(&self.context),
        )
        .cloned();
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_id")?,
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
            let out = (context
                .as_ref()
                .map(|context| out.returned_context(py, context))
                .transpose()
                .and_then(|context| {
                    crate::py_calls::returning(context, || {
                        crate::generated::records::types::provider_id_from_py(o.as_any())
                    })
                }))
            .map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn prewarm(&self, _cx: &::lk_agents_abi::telemetry::Cx) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "Vad._core_prewarm",
            &crate::py_calls::calling_context(&self.context),
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_prewarm")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
    }

    fn stream(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::lk_agents_abi::DuplexStream<
                ::std::boxed::Box<dyn ::lk_agents_abi::StreamInput<::lk_agents_abi::AudioFrame>>,
                ::std::boxed::Box<dyn ::lk_agents_abi::ProviderStream<::lk_agents_abi::VadEvent>>,
            >,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_streams::StreamCall,
            "Vad._core_stream",
            &crate::py_calls::calling_context(&self.context),
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_stream")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        let context = ::lk_agents_abi::agent::HostContext::host_context(
            &crate::py_calls::calling_context(&self.context),
        )
        .cloned();
        let wrap_cx = ::std::clone::Clone::clone(cx);
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            let out = ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("Vad._core_stream: the call captured no host context")));
                };
                let context = &out.returned_context(py, context).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                let o = out.value.bind(py);
                let out = (crate::py_streams::HostStream::duplex(py, o, context, crate::generated::records::rtc_frames::audio_frame_to_py, |o| crate::generated::records::vad::vad_event_from_py(o.as_any()), ::lk_agents_runtime::process::host_clock()).map(|stream| { let (sink, source) = stream.split(); ::lk_agents_abi::DuplexStream::new(::std::boxed::Box::new(sink) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamInput<::lk_agents_abi::AudioFrame>>, ::std::boxed::Box::new(source) as ::std::boxed::Box<dyn ::lk_agents_abi::ProviderStream<::lk_agents_abi::VadEvent>>) })).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })?;
            Ok(::lk_agents_runtime::inference::vad::plugin::PluginInput::wrap(
                out,
                &wrap_cx,
                ::lk_agents_runtime::process::host_clock(),
            ))
        })
    }
}
