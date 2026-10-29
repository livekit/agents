//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `vad` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/vad/inference.rs.
type InferenceVadImpl = <crate::Selected as super::Implementations>::InferenceVad;

fn _extract_inference_vad_new_min_speech_duration(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "min_speech_duration")
}

fn _extract_inference_vad_new_min_silence_duration(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "min_silence_duration")
}

fn _extract_inference_vad_new_prefix_padding_duration(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "prefix_padding_duration")
}

fn _extract_inference_vad_new_max_buffered_speech(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "max_buffered_speech")
}

fn _extract_inference_vad_new_activation_threshold(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "activation_threshold")
}

fn _extract_inference_vad_new_deactivation_threshold(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "deactivation_threshold")
}

fn _extract_inference_vad_update_options_min_speech_duration(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "min_speech_duration")
}

fn _extract_inference_vad_update_options_min_silence_duration(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "min_silence_duration")
}

fn _extract_inference_vad_update_options_prefix_padding_duration(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "prefix_padding_duration")
}

fn _extract_inference_vad_update_options_max_buffered_speech(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "max_buffered_speech")
}

fn _extract_inference_vad_update_options_activation_threshold(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "activation_threshold")
}

fn _extract_inference_vad_update_options_deactivation_threshold(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "deactivation_threshold")
}

/// The Silero VAD behind `inference.VAD`.
#[pyclass(frozen, name = "InferenceVad", module = "livekit.agents._core")]
pub struct InferenceVad(pub(crate) ::std::sync::Arc<InferenceVadImpl>);

#[pymethods]
impl InferenceVad {
    #[new]
    #[pyo3(signature = (model_factory, *, min_speech_duration, min_silence_duration, prefix_padding_duration, max_buffered_speech, activation_threshold, deactivation_threshold))]
    fn new(model_factory: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_inference_vad_new_min_speech_duration)] min_speech_duration: f64, #[pyo3(from_py_with = _extract_inference_vad_new_min_silence_duration)] min_silence_duration: f64, #[pyo3(from_py_with = _extract_inference_vad_new_prefix_padding_duration)] prefix_padding_duration: f64, #[pyo3(from_py_with = _extract_inference_vad_new_max_buffered_speech)] max_buffered_speech: f64, #[pyo3(from_py_with = _extract_inference_vad_new_activation_threshold)] activation_threshold: f64, #[pyo3(from_py_with = _extract_inference_vad_new_deactivation_threshold)] deactivation_threshold: f64) -> PyResult<InferenceVad> {
        crate::host::check_pid()?;
        let env = crate::process::env();
        let model_factory = crate::generated::vad::VadModelsProxy::from_host(&model_factory)?;
        let out = <InferenceVadImpl as ::lk_agents_abi::vad::inference::InferenceVad>::new(&env, model_factory, min_speech_duration, min_silence_duration, prefix_padding_duration, max_buffered_speech, activation_threshold, deactivation_threshold);
        Ok(InferenceVad(::std::sync::Arc::new(out)))
    }

    /// Every option, applied by each running stream before its next frame.
    #[pyo3(signature = (*, min_speech_duration, min_silence_duration, prefix_padding_duration, max_buffered_speech, activation_threshold, deactivation_threshold))]
    fn update_options(&self, #[pyo3(from_py_with = _extract_inference_vad_update_options_min_speech_duration)] min_speech_duration: f64, #[pyo3(from_py_with = _extract_inference_vad_update_options_min_silence_duration)] min_silence_duration: f64, #[pyo3(from_py_with = _extract_inference_vad_update_options_prefix_padding_duration)] prefix_padding_duration: f64, #[pyo3(from_py_with = _extract_inference_vad_update_options_max_buffered_speech)] max_buffered_speech: f64, #[pyo3(from_py_with = _extract_inference_vad_update_options_activation_threshold)] activation_threshold: f64, #[pyo3(from_py_with = _extract_inference_vad_update_options_deactivation_threshold)] deactivation_threshold: f64) -> PyResult<()> {
        crate::host::check_pid()?;
        <InferenceVadImpl as ::lk_agents_abi::vad::inference::InferenceVad>::update_options(&self.0, min_speech_duration, min_silence_duration, prefix_padding_duration, max_buffered_speech, activation_threshold, deactivation_threshold);
        Ok(())
    }

    /// One detection; each event carries the audio it refers to.
    fn stream(&self) -> PyResult<InferenceVadStream> {
        crate::host::check_pid()?;
        let out = <InferenceVadImpl as ::lk_agents_abi::vad::inference::InferenceVad>::stream(&self.0).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok({ let (sink, source) = out; InferenceVadStream(::std::boxed::Box::new(sink), ::lk_agents_runtime::net::source::NativeSource::new(::std::boxed::Box::new(source))) })
    }
}

// Wherever the host passes a `::lk_agents_abi::Vad`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::Vad>(),
        read: |obj| {
            let this = obj.cast::<InferenceVad>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::Vad> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

type InferenceVadStreamImpl = dyn ::lk_agents_abi::StreamSink<::lk_agents_abi::AudioFrame>;

/// One duplex stream: `push()` input, `recv()` resolves to the next output, `None` at the end.
#[pyclass(frozen, name = "InferenceVadStream", module = "livekit.agents._core")]
pub struct InferenceVadStream(::std::boxed::Box<InferenceVadStreamImpl>, ::lk_agents_runtime::net::source::NativeSource<::lk_agents_abi::VadEvent>);

#[pymethods]
impl InferenceVadStream {
    fn push(&self, item: ::lk_agents_abi::AudioFrame) -> PyResult<()> {
        crate::host::check_pid()?;
        ::lk_agents_abi::StreamSink::push(&*self.0, item).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))
    }

    fn flush(&self) -> PyResult<()> {
        crate::host::check_pid()?;
        ::lk_agents_abi::StreamSink::flush(&*self.0).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))
    }

    fn end_input(&self) -> PyResult<()> {
        crate::host::check_pid()?;
        ::lk_agents_abi::StreamSink::end_input(&*self.0);
        Ok(())
    }

    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        crate::host::source::recv(&self.1, py, |py, item| ::pyo3::IntoPyObjectExt::into_py_any(item, py))
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        ::lk_agents_abi::StreamSink::end_input(&*self.0);
        crate::host::source::aclose(&self.1, py)
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
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::vad::inference::VadModels>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::vad::inference::VadModels>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::vad::inference::VadModels>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::vad::inference::VadModels>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::vad::inference::VadModels>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let _ = context;
        Ok(VadModelsProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

impl ::lk_agents_abi::vad::inference::VadModels for VadModelsProxy {
    fn create(&self) -> ::std::result::Result<::std::sync::Arc<dyn ::lk_agents_abi::vad::inference::VadModel>, ::lk_agents_abi::errors::HostError> {
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
            let out = (crate::generated::vad::VadModelProxy::from_host(&o)).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
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
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::vad::inference::VadModel>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::vad::inference::VadModel>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::vad::inference::VadModel>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::vad::inference::VadModel>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::vad::inference::VadModel>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let _ = context;
        Ok(VadModelProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

impl ::lk_agents_abi::vad::inference::VadModel for VadModelProxy {
    fn predict(&self, window: ::std::vec::Vec<i16>) -> ::std::result::Result<f64, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let window_arg = window;
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(window_arg, py)?];
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
            let out = (::lk_agents_abi::python::number_from_py::<f64>(&o, "VadModel.predict")).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn reset(&self) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("reset")?, ::pyo3::types::PyTuple::new(py, args)?))
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

// Declared in rust/crates/lk-agents-abi/src/vad/vad.rs.

/// [`::lk_agents_abi::Vad`] over a host `livekit.agents.VAD`.
#[derive(Clone)]
pub struct VadProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: crate::host::Notifier,
    label: ::std::string::String,
    capabilities: ::lk_agents_abi::VadCapabilities,
    id: ::lk_agents_abi::ProviderId,
}

impl VadProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::Vad>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::Vad>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::Vad>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::Vad>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::Vad>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        let label = {
            let v = crate::host::read(obj, "_core_label")?;
            v.extract::<::std::string::String>()?
        };
        let capabilities = {
            let v = crate::host::read(obj, "_core_capabilities")?;
            v.extract::<::lk_agents_abi::VadCapabilities>()?
        };
        let id = {
            let v = crate::host::read(obj, "_core_id")?;
            v.extract::<::lk_agents_abi::ProviderId>()?
        };
        Ok(VadProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, notifier: ::std::default::Default::default(), label, capabilities, id })
    }
}

impl ::lk_agents_abi::Vad for VadProxy {
    fn label(&self) -> ::std::string::String {
        ::std::clone::Clone::clone(&self.label)
    }

    fn capabilities(&self) -> ::lk_agents_abi::VadCapabilities {
        ::std::clone::Clone::clone(&self.capabilities)
    }

    fn id(&self) -> ::lk_agents_abi::ProviderId {
        ::std::clone::Clone::clone(&self.id)
    }

    fn prewarm(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(&self.notifier, "Vad._core_prewarm", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_prewarm")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn stream(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(::std::boxed::Box<dyn ::lk_agents_abi::StreamSink<::lk_agents_abi::AudioFrame>>, ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::VadEvent>>), ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "Vad._core_stream", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_stream")?, ::pyo3::types::PyTuple::new(py, args)?))
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
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("Vad._core_stream: the call captured no host context")));
                };
            let o = out.bind(py);
            let out = (crate::host::stream::HostStream::duplex(py, o, context, |py, v| Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(v, py)?), |o| Ok(o.extract::<::lk_agents_abi::VadEvent>()?), ::lk_agents_runtime::process::host_clock()).map(|(sink, source)| (::std::boxed::Box::new(sink) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSink<::lk_agents_abi::AudioFrame>>, ::std::boxed::Box::new(source) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::VadEvent>>))).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }
}
