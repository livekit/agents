//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `stt` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/stt/fallback.rs.
type FallbackSttImpl = <crate::Selected as super::Implementations>::FallbackStt;

fn _extract_fallback_stt_new_attempt_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "attempt_timeout")
}

fn _extract_fallback_stt_new_max_retry_per_stt(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    ::lk_agents_abi::python::number_from_py::<u32>(&obj, "max_retry_per_stt")
}

fn _extract_fallback_stt_new_retry_interval(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "retry_interval")
}

fn _extract_fallback_stt_stream_start_time_offset(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(&obj, "start_time_offset")).map(Some) }
}

/// `stt.FallbackAdapter`'s engine as the host builds it; the public class stays Python.
#[pyclass(frozen, name = "FallbackStt", module = "livekit.agents._core")]
pub struct FallbackStt(pub(crate) ::std::sync::Arc<FallbackSttImpl>);

#[pymethods]
impl FallbackStt {
    /// `stts` in order, each already streaming (the host wraps a recognize-only entry in its
    /// `StreamAdapter`); the keywords are `FallbackAdapter.__init__`'s.
    #[new]
    #[pyo3(signature = (stts, *, attempt_timeout = 10.0, max_retry_per_stt = 1, retry_interval = 5.0, on_availability = None))]
    fn new(stts: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_fallback_stt_new_attempt_timeout)] attempt_timeout: f64, #[pyo3(from_py_with = _extract_fallback_stt_new_max_retry_per_stt)] max_retry_per_stt: u32, #[pyo3(from_py_with = _extract_fallback_stt_new_retry_interval)] retry_interval: f64, on_availability: Option<Bound<'_, PyAny>>) -> PyResult<FallbackStt> {
        ::lk_agents_abi::python::check_pid()?;
        let env = crate::process::env();
        let stts = stts.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::stt::SttProxy::from_host(&item) }).collect::<PyResult<Vec<_>>>())?;
        let on_availability = on_availability.map(|v| crate::generated::llm::ChainObserverProxy::from_host(&v)).transpose()?;
        let out = <FallbackSttImpl as ::lk_agents_abi::stt::fallback::FallbackStt>::new(&env, stts, attempt_timeout, max_retry_per_stt, retry_interval, on_availability).map_err(|e: ::lk_agents_abi::llm::inference::InvalidPayload| <::lk_agents_abi::llm::inference::InvalidPayload as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(FallbackStt(::std::sync::Arc::new(out)))
    }

    /// Which entry most recently served, as a position in the host's list.
    #[getter]
    fn active_index(&self) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <FallbackSttImpl as ::lk_agents_abi::stt::fallback::FallbackStt>::active_index(&self.0);
        Ok(out)
    }

    /// The merged capabilities, not the primary's: a caller must not wait for interim results a
    /// secondary never sends.
    #[getter]
    fn capabilities(&self) -> PyResult<::lk_agents_abi::SttCapabilities> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <FallbackSttImpl as ::lk_agents_abi::stt::fallback::FallbackStt>::capabilities(&self.0);
        Ok(out)
    }

    /// Every recovery probe in flight closes its stream, awaited (bounded).
    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <FallbackSttImpl as ::lk_agents_abi::stt::fallback::FallbackStt>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// `_recognize_impl`: the batch path, over whichever entry answers.
    #[pyo3(signature = (frames, language = None, conn_options = None))]
    fn recognize<'py>(&self, py: Python<'py>, frames: ::std::vec::Vec<::lk_agents_abi::AudioFrame>, language: ::std::option::Option<::lk_agents_abi::LanguageCode>, conn_options: ::std::option::Option<::lk_agents_abi::ApiConnectOptions>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <FallbackSttImpl as ::lk_agents_abi::stt::fallback::FallbackStt>::recognize(&this, frames, language, conn_options)
                .await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// Audio in, tee'd to the serving stream and every recovery probe; events out.
    #[pyo3(signature = (language = None, start_time_offset = None, conn_options = None))]
    fn stream<'py>(&self, py: Python<'py>, language: ::std::option::Option<::lk_agents_abi::LanguageCode>, #[pyo3(from_py_with = _extract_fallback_stt_stream_start_time_offset)] start_time_offset: ::std::option::Option<f64>, conn_options: ::std::option::Option<::lk_agents_abi::ApiConnectOptions>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <FallbackSttImpl as ::lk_agents_abi::stt::fallback::FallbackStt>::stream(&this, language, start_time_offset, conn_options)
                .await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any({ let (sink, source) = out; FallbackRecognizeStream(::std::boxed::Box::new(sink), ::lk_agents_runtime::net::source::NativeSource::new(::std::boxed::Box::new(source))) }, py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::Stt`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::Stt>(),
        read: |obj| {
            let this = obj.cast::<FallbackStt>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::Stt> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

type FallbackRecognizeStreamImpl = dyn ::lk_agents_abi::StreamSink<::lk_agents_abi::AudioFrame>;

/// One duplex stream: `push()` input, `recv()` resolves to the next output, `None` at the end.
#[pyclass(frozen, name = "FallbackRecognizeStream", module = "livekit.agents._core")]
pub struct FallbackRecognizeStream(::std::boxed::Box<FallbackRecognizeStreamImpl>, ::lk_agents_runtime::net::source::NativeSource<::lk_agents_abi::SpeechEvent>);

#[pymethods]
impl FallbackRecognizeStream {
    fn push(&self, item: ::lk_agents_abi::AudioFrame) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        ::lk_agents_abi::StreamSink::push(&*self.0, item).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))
    }

    fn flush(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        ::lk_agents_abi::StreamSink::flush(&*self.0).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))
    }

    fn end_input(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        ::lk_agents_abi::StreamSink::end_input(&*self.0);
        Ok(())
    }

    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        crate::host::source::recv(&self.1, py, |py, item| ::pyo3::IntoPyObjectExt::into_py_any(item, py))
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        ::lk_agents_abi::StreamSink::end_input(&*self.0);
        crate::host::source::aclose(&self.1, py)
    }
}

// Declared in rust/crates/lk-agents-abi/src/stt/inference.rs.
type InferenceSttImpl = <crate::Selected as super::Implementations>::InferenceStt;

fn _extract_inference_stt_new_sample_rate(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    ::lk_agents_abi::python::number_from_py::<u32>(&obj, "sample_rate")
}

fn _extract_inference_stt_stream_start_time_offset(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(&obj, "start_time_offset")).map(Some) }
}

/// The gateway STT behind `inference.STT`. Its options, keyterms and open streams are shared by
/// every view (`identified`); each view's connections carry its own job's identification. As the
/// session's STT it implements the keyterm and conversation hooks.
#[pyclass(frozen, name = "InferenceStt", module = "livekit.agents._core")]
pub struct InferenceStt(pub(crate) ::std::sync::Arc<InferenceSttImpl>);

#[pymethods]
impl InferenceStt {
    /// A client of `model` on the gateway at `base_url`. Starts nothing. `vad`'s end of speech
    /// finalizes the transcript (Speechmatics only).
    #[new]
    #[pyo3(signature = (*, model, language, encoding, sample_rate, base_url, api_key, api_secret, extra_kwargs, fallback = None, conn_options, vad = None))]
    fn new(model: ::std::option::Option<::std::string::String>, language: ::std::option::Option<::std::string::String>, encoding: ::std::string::String, #[pyo3(from_py_with = _extract_inference_stt_new_sample_rate)] sample_rate: u32, base_url: ::std::string::String, api_key: ::std::string::String, api_secret: ::std::string::String, extra_kwargs: Bound<'_, PyAny>, fallback: Option<Bound<'_, PyAny>>, conn_options: ::lk_agents_abi::ApiConnectOptions, vad: Option<Bound<'_, PyAny>>) -> PyResult<InferenceStt> {
        ::lk_agents_abi::python::check_pid()?;
        let env = crate::process::env();
        let extra_kwargs = match ::lk_agents_abi::python::value_from_py(&extra_kwargs, "extra_kwargs")? { ::serde_json::Value::Object(map) => Ok(map), value => Err(::pyo3::exceptions::PyValueError::new_err(format!("expected a dict of extra kwargs, got {value}"))) }?;
        let fallback = fallback.map(|v| v.cast::<::pyo3::types::PyList>().map_err(::pyo3::PyErr::from).and_then(|items| items.iter().map(|item| { if !item.is_instance_of::<::pyo3::types::PyDict>() { return Err(::pyo3::exceptions::PyValueError::new_err("each fallback entry must be a dict")); } crate::generated::records::inference_stt::fallback_model_from_py(&item) }).collect::<PyResult<Vec<_>>>())).transpose()?;
        let vad = vad.map(|v| crate::generated::vad::VadProxy::from_host(&v)).transpose()?;
        let out = <InferenceSttImpl as ::lk_agents_abi::stt::inference::InferenceStt>::new(&env, model, language, encoding, sample_rate, base_url, api_key, api_secret, extra_kwargs, fallback, conn_options, vad).map_err(|e: ::lk_agents_abi::errors::ApiError| <::lk_agents_abi::errors::ApiError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(InferenceStt(::std::sync::Arc::new(out)))
    }

    /// The model's VAD as the host resolved it on `update_options(model=)`; `None` drops it.
    /// Later streams use it.
    #[pyo3(signature = (vad = None))]
    fn set_vad(&self, vad: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        let vad = vad.map(|v| crate::generated::vad::VadProxy::from_host(&v)).transpose()?;
        <InferenceSttImpl as ::lk_agents_abi::stt::inference::InferenceStt>::set_vad(&self.0, vad);
        Ok(())
    }

    /// This client as a session of the job `headers` (`get_inference_headers()`) identifies holds
    /// it: the same options, keyterms and open streams, its connections carrying that job.
    /// `reporter` is the host `stt.STT` whose `error` events report this view's failures.
    #[pyo3(signature = (headers, reporter = None))]
    fn identified(&self, headers: ::std::vec::Vec<(::std::string::String, ::std::string::String)>, reporter: Option<Bound<'_, PyAny>>) -> PyResult<InferenceStt> {
        ::lk_agents_abi::python::check_pid()?;
        let reporter = reporter.map(|v| crate::generated::stt::SttProxy::from_host(&v)).transpose()?;
        let out = <InferenceSttImpl as ::lk_agents_abi::stt::inference::InferenceStt>::identified(&self.0, headers, reporter);
        Ok(InferenceStt(::std::sync::Arc::new(out)))
    }

    /// Read by later streams, and sent to the open ones as a mid-stream `session.update`.
    #[pyo3(signature = (*, model = None, language = None, extra = None))]
    fn update_options(&self, model: ::std::option::Option<::std::string::String>, language: ::std::option::Option<::std::string::String>, extra: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        let extra = extra.map(|v| match ::lk_agents_abi::python::value_from_py(&v, "extra")? { ::serde_json::Value::Object(map) => Ok(map), value => Err(::pyo3::exceptions::PyValueError::new_err(format!("expected a dict of extra kwargs, got {value}"))) }).transpose()?;
        <InferenceSttImpl as ::lk_agents_abi::stt::inference::InferenceStt>::update_options(&self.0, model, language, extra);
        Ok(())
    }

    /// The session's keyterms: a warning once for a model without keyterm prompting, else every
    /// open stream applies them (now if the user is silent, at end of speech if not).
    #[pyo3(signature = (keyterms))]
    fn update_keyterms(&self, keyterms: ::std::vec::Vec<::std::string::String>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <InferenceSttImpl as ::lk_agents_abi::stt::inference::InferenceStt>::update_keyterms(&self.0, keyterms);
        Ok(())
    }

    /// An assistant turn's text, its tail, biases recognition as `agent_context` on a model that
    /// carries it.
    #[pyo3(signature = (role, text))]
    fn push_conversation_item(&self, role: ::std::string::String, text: ::std::string::String) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <InferenceSttImpl as ::lk_agents_abi::stt::inference::InferenceStt>::push_conversation_item(&self.0, &role, &text);
        Ok(())
    }

    /// Opens a stream; its pump opens the socket. `start_time_offset` shifts its timestamps onto
    /// the caller's input timeline.
    #[pyo3(signature = (*, language = None, start_time_offset = None, conn_options = None))]
    fn stream(&self, language: ::std::option::Option<::std::string::String>, #[pyo3(from_py_with = _extract_inference_stt_stream_start_time_offset)] start_time_offset: ::std::option::Option<f64>, conn_options: ::std::option::Option<::lk_agents_abi::ApiConnectOptions>) -> PyResult<InferenceSttStream> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <InferenceSttImpl as ::lk_agents_abi::stt::inference::InferenceStt>::stream(&self.0, language, start_time_offset, conn_options).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok({ let (sink, source) = out; InferenceSttStream(::std::boxed::Box::new(sink), ::lk_agents_runtime::net::source::NativeSource::new(::std::boxed::Box::new(source))) })
    }
}

// Wherever the host passes a `::lk_agents_abi::Stt`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::Stt>(),
        read: |obj| {
            let this = obj.cast::<InferenceStt>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::Stt> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

type InferenceSttStreamImpl = dyn ::lk_agents_abi::stt::inference::InferenceSttAudio;

/// One duplex stream: `push()` input, `recv()` resolves to the next output, `None` at the end.
#[pyclass(frozen, name = "InferenceSttStream", module = "livekit.agents._core")]
pub struct InferenceSttStream(::std::boxed::Box<InferenceSttStreamImpl>, ::lk_agents_runtime::net::source::NativeSource<::lk_agents_abi::SpeechEvent>);

#[pymethods]
impl InferenceSttStream {
    fn push(&self, item: ::lk_agents_abi::AudioFrame) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        ::lk_agents_abi::StreamSink::push(&*self.0, item).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))
    }

    fn flush(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        ::lk_agents_abi::StreamSink::flush(&*self.0).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))
    }

    fn end_input(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        ::lk_agents_abi::StreamSink::end_input(&*self.0);
        Ok(())
    }

    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        crate::host::source::recv(&self.1, py, |py, item| ::pyo3::IntoPyObjectExt::into_py_any(item, py))
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        ::lk_agents_abi::StreamSink::end_input(&*self.0);
        crate::host::source::aclose(&self.1, py)
    }

    /// A mid-stream `session.update`, for the providers that take one (AssemblyAI, Deepgram Flux).
    #[pyo3(signature = (*, model = None, language = None, extra = None))]
    fn update_options(&self, model: ::std::option::Option<::std::string::String>, language: ::std::option::Option<::std::string::String>, extra: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        let extra = extra.map(|v| match ::lk_agents_abi::python::value_from_py(&v, "extra")? { ::serde_json::Value::Object(map) => Ok(map), value => Err(::pyo3::exceptions::PyValueError::new_err(format!("expected a dict of extra kwargs, got {value}"))) }).transpose()?;
        <InferenceSttStreamImpl as ::lk_agents_abi::stt::inference::InferenceSttAudio>::update_options(&*self.0, model, language, extra);
        Ok(())
    }
}

// Declared in rust/crates/lk-agents-abi/src/stt/stream_adapter.rs.
type SttStreamAdapterImpl = <crate::Selected as super::Implementations>::SttStreamAdapter;

fn _extract_stt_stream_adapter_stream_start_time_offset(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(&obj, "start_time_offset")).map(Some) }
}

/// `stt.StreamAdapter`'s engine as the host builds it; the public class stays Python.
#[pyclass(frozen, name = "SttStreamAdapter", module = "livekit.agents._core")]
pub struct SttStreamAdapter(pub(crate) ::std::sync::Arc<SttStreamAdapterImpl>);

#[pymethods]
impl SttStreamAdapter {
    /// The wrapped STT and VAD; each stream binds its own recognition policy.
    #[new]
    #[pyo3(signature = (stt, vad))]
    fn new(stt: Bound<'_, PyAny>, vad: Bound<'_, PyAny>) -> PyResult<SttStreamAdapter> {
        ::lk_agents_abi::python::check_pid()?;
        let env = crate::process::env();
        let stt = crate::generated::stt::SttProxy::from_host(&stt)?;
        let vad = crate::generated::vad::VadProxy::from_host(&vad)?;
        let out = <SttStreamAdapterImpl as ::lk_agents_abi::stt::stream_adapter::SttStreamAdapter>::new(&env, stt, vad);
        Ok(SttStreamAdapter(::std::sync::Arc::new(out)))
    }

    /// `_recognize_impl`: the wrapped STT's batch path.
    #[pyo3(signature = (frames, language = None, conn_options = None))]
    fn recognize<'py>(&self, py: Python<'py>, frames: ::std::vec::Vec<::lk_agents_abi::AudioFrame>, language: ::std::option::Option<::lk_agents_abi::LanguageCode>, conn_options: ::std::option::Option<::lk_agents_abi::ApiConnectOptions>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <SttStreamAdapterImpl as ::lk_agents_abi::stt::stream_adapter::SttStreamAdapter>::recognize(&this, frames, language, conn_options)
                .await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// Audio into the VAD, its utterances recognized.
    #[pyo3(signature = (language = None, start_time_offset = None, conn_options = None))]
    fn stream<'py>(&self, py: Python<'py>, language: ::std::option::Option<::lk_agents_abi::LanguageCode>, #[pyo3(from_py_with = _extract_stt_stream_adapter_stream_start_time_offset)] start_time_offset: ::std::option::Option<f64>, conn_options: ::std::option::Option<::lk_agents_abi::ApiConnectOptions>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <SttStreamAdapterImpl as ::lk_agents_abi::stt::stream_adapter::SttStreamAdapter>::stream(&this, language, start_time_offset, conn_options)
                .await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any({ let (sink, source) = out; SttAdapterStream(::std::boxed::Box::new(sink), ::lk_agents_runtime::net::source::NativeSource::new(::std::boxed::Box::new(source))) }, py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::Stt`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::Stt>(),
        read: |obj| {
            let this = obj.cast::<SttStreamAdapter>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::Stt> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

type SttAdapterStreamImpl = dyn ::lk_agents_abi::StreamSink<::lk_agents_abi::AudioFrame>;

/// One duplex stream: `push()` input, `recv()` resolves to the next output, `None` at the end.
#[pyclass(frozen, name = "SttAdapterStream", module = "livekit.agents._core")]
pub struct SttAdapterStream(::std::boxed::Box<SttAdapterStreamImpl>, ::lk_agents_runtime::net::source::NativeSource<::lk_agents_abi::SpeechEvent>);

#[pymethods]
impl SttAdapterStream {
    fn push(&self, item: ::lk_agents_abi::AudioFrame) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        ::lk_agents_abi::StreamSink::push(&*self.0, item).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))
    }

    fn flush(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        ::lk_agents_abi::StreamSink::flush(&*self.0).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))
    }

    fn end_input(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        ::lk_agents_abi::StreamSink::end_input(&*self.0);
        Ok(())
    }

    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        crate::host::source::recv(&self.1, py, |py, item| ::pyo3::IntoPyObjectExt::into_py_any(item, py))
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        ::lk_agents_abi::StreamSink::end_input(&*self.0);
        crate::host::source::aclose(&self.1, py)
    }
}

// Declared in rust/crates/lk-agents-abi/src/stt/stt.rs.

/// [`::lk_agents_abi::Stt`] over a host `livekit.agents.stt.STT`.
#[derive(Clone)]
pub struct SttProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: crate::host::Notifier,
    instance_id: ::std::option::Option<usize>,
    label: ::std::string::String,
    capabilities: ::lk_agents_abi::SttCapabilities,
    id: ::lk_agents_abi::ProviderId,
    conn_options: ::std::option::Option<::lk_agents_abi::ApiConnectOptions>,
    /// Appended to every awaited call once bound.
    conn: ::std::option::Option<::lk_agents_abi::ApiConnectOptions>,
}

impl SttProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::Stt>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::Stt>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::Stt>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::Stt>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::Stt>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        let instance_id = {
            let v = crate::host::read(obj, "_core_instance_id")?;
            if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<usize>(&v, "Stt._core_instance_id")).map(Some) }?
        };
        let label = {
            let v = crate::host::read(obj, "_core_label")?;
            v.extract::<::std::string::String>()?
        };
        let capabilities = {
            let v = crate::host::read(obj, "_core_capabilities")?;
            v.extract::<::lk_agents_abi::SttCapabilities>()?
        };
        let id = {
            let v = crate::host::read(obj, "_core_id")?;
            v.extract::<::lk_agents_abi::ProviderId>()?
        };
        let conn_options = {
            let v = crate::host::read(obj, "_core_conn_options")?;
            if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::ApiConnectOptions>()).map(Some) }?
        };
        Ok(SttProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, notifier: ::std::default::Default::default(), instance_id, label, capabilities, id, conn_options, conn: None })
    }
}

impl ::lk_agents_abi::Stt for SttProxy {
    fn instance_id(&self) -> ::std::option::Option<usize> {
        ::std::clone::Clone::clone(&self.instance_id)
    }

    fn label(&self) -> ::std::string::String {
        ::std::clone::Clone::clone(&self.label)
    }

    fn capabilities(&self) -> ::lk_agents_abi::SttCapabilities {
        ::std::clone::Clone::clone(&self.capabilities)
    }

    fn id(&self) -> ::lk_agents_abi::ProviderId {
        ::std::clone::Clone::clone(&self.id)
    }

    fn prewarm(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(&self.notifier, "Stt._core_prewarm", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_prewarm")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn update_keyterms(&self, keyterms: &[::std::string::String]) {
        let host = ::std::sync::Arc::clone(&self.host);
        let keyterms_arg = keyterms.to_vec();
        ::lk_agents_runtime::host::notify(&self.notifier, "Stt._core_update_keyterms", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(keyterms_arg, py)?];
                Ok((host.bind(py).getattr("_core_update_keyterms")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn push_conversation_item(&self, role: &str, text: &str) {
        let host = ::std::sync::Arc::clone(&self.host);
        let role_arg = role.to_owned();
        let text_arg = text.to_owned();
        ::lk_agents_runtime::host::notify(&self.notifier, "Stt._core_push_conversation_item", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(role_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(text_arg, py)?];
                Ok((host.bind(py).getattr("_core_push_conversation_item")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn failed(&self, error: ::lk_agents_abi::errors::ProviderError, recoverable: bool) {
        let host = ::std::sync::Arc::clone(&self.host);
        let error_arg = error;
        let error_arg = match ::pyo3::Python::attach(|py| -> ::pyo3::PyResult<::pyo3::Py<::pyo3::PyAny>> { Ok((::pyo3::IntoPyObjectExt::into_bound_py_any(error_arg, py)?).unbind()) }) {
            Ok(value) => value,
            Err(error) => {
                ::lk_agents_abi::python::log::warn!("Stt._core_failed failed: {error}");
                return;
            }
        };
        let recoverable_arg = recoverable;
        ::lk_agents_runtime::host::notify(&self.notifier, "Stt._core_failed", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [error_arg.into_bound(py), ::pyo3::IntoPyObjectExt::into_bound_py_any(recoverable_arg, py)?];
                Ok((host.bind(py).getattr("_core_failed")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn with_conn_options(&self, conn: &::lk_agents_abi::ApiConnectOptions) -> ::std::option::Option<::std::sync::Arc<dyn ::lk_agents_abi::Stt>> {
        Some(::lk_agents_runtime::inference::stt::retrying::RetryingStt::bound(::std::sync::Arc::new(Self { conn: Some(::std::clone::Clone::clone(conn)), ..::std::clone::Clone::clone(self) }), conn, ::lk_agents_runtime::process::host_clock()))
    }

    fn conn_options(&self) -> ::std::option::Option<::lk_agents_abi::ApiConnectOptions> {
        ::std::clone::Clone::clone(&self.conn_options)
    }

    fn recognize(&self, buffer: &[::lk_agents_abi::AudioFrame], language: Option<&::lk_agents_abi::LanguageCode>) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::lk_agents_abi::SpeechEvent, ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let buffer_arg = buffer.to_vec();
        let language_arg = language.map(::std::borrow::ToOwned::to_owned);
        let bound = ::std::clone::Clone::clone(&self.conn);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "Stt._core_recognize", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(buffer_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(language_arg, py)?];
                let mut args = args.to_vec();
                if let Some(bound) = bound {
                    args.push(crate::host::construct(&py.import("livekit.agents.types")?.getattr("APIConnectOptions")?, (), Some(&::pyo3::IntoPyObject::into_pyobject(bound, py)?.cast_into::<::pyo3::types::PyDict>()?))?);
                }
                Ok((host.bind(py).getattr("_core_recognize")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = (o.extract::<::lk_agents_abi::SpeechEvent>()).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn stream(&self, language: Option<&::lk_agents_abi::LanguageCode>, start_time_offset: ::std::option::Option<f64>) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::std::option::Option<(::std::boxed::Box<dyn ::lk_agents_abi::StreamSink<::lk_agents_abi::AudioFrame>>, ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::SpeechEvent>>)>, ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let language_arg = language.map(::std::borrow::ToOwned::to_owned);
        let start_time_offset_arg = start_time_offset;
        let bound = ::std::clone::Clone::clone(&self.conn);
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "Stt._core_stream", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(language_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(start_time_offset_arg, py)?];
                let mut args = args.to_vec();
                if let Some(bound) = bound {
                    args.push(crate::host::construct(&py.import("livekit.agents.types")?.getattr("APIConnectOptions")?, (), Some(&::pyo3::IntoPyObject::into_pyobject(bound, py)?.cast_into::<::pyo3::types::PyDict>()?))?);
                }
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
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("Stt._core_stream: the call captured no host context")));
                };
            let o = out.bind(py);
            let out = (if o.is_none() { ::pyo3::PyResult::Ok(None) } else { (crate::host::stream::HostStream::duplex(py, o, context, |py, v| Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(v, py)?), |o| Ok(o.extract::<::lk_agents_abi::SpeechEvent>()?), ::lk_agents_runtime::process::host_clock()).map(|(sink, source)| (::std::boxed::Box::new(sink) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSink<::lk_agents_abi::AudioFrame>>, ::std::boxed::Box::new(source) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::SpeechEvent>>))).map(Some) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }
}
