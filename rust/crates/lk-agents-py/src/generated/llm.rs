//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `llm` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/llm/duplex.rs.
type FixedGateImpl = <crate::Selected as super::Implementations>::FixedGate;

fn _extract_fixed_gate_new_silence(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "silence")
}

fn _extract_fixed_gate_new_activation_ratio(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "activation_ratio")
}

fn _extract_fixed_gate_new_deactivation_ratio(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "deactivation_ratio")
}

fn _extract_fixed_gate_new_min_silence_duration(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "min_silence_duration")
}

/// Opens on output that stands out from a silence the plugin already knows.
#[pyclass(frozen, name = "FixedGate", module = "livekit.agents._core")]
pub struct FixedGate(::std::sync::Arc<FixedGateImpl>);

#[pymethods]
impl FixedGate {
    #[new]
    #[pyo3(signature = (silence, *, activation_ratio = 3.0, deactivation_ratio = 1.8, min_silence_duration = 0.5))]
    fn new(#[pyo3(from_py_with = _extract_fixed_gate_new_silence)] silence: f64, #[pyo3(from_py_with = _extract_fixed_gate_new_activation_ratio)] activation_ratio: f64, #[pyo3(from_py_with = _extract_fixed_gate_new_deactivation_ratio)] deactivation_ratio: f64, #[pyo3(from_py_with = _extract_fixed_gate_new_min_silence_duration)] min_silence_duration: f64) -> PyResult<FixedGate> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <FixedGateImpl as ::lk_agents_abi::llm::duplex::FixedGate>::new(silence, activation_ratio, deactivation_ratio, min_silence_duration);
        Ok(FixedGate(::std::sync::Arc::new(out)))
    }

    /// True while the frame belongs to an open burst of output.
    #[pyo3(signature = (frame))]
    fn update(&self, frame: ::lk_agents_abi::AudioFrame) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <FixedGateImpl as ::lk_agents_abi::llm::duplex::FixedGate>::update(&self.0, frame).map_err(|e: ::lk_agents_abi::llm::inference::InvalidPayload| <::lk_agents_abi::llm::inference::InvalidPayload as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    /// End the open burst, keeping whatever has been learned about the model.
    fn deactivate(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <FixedGateImpl as ::lk_agents_abi::llm::duplex::FixedGate>::deactivate(&self.0);
        Ok(())
    }
}

// Wherever the host passes a `::lk_agents_abi::llm::duplex::AudioGate`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::llm::duplex::AudioGate>(),
        read: |obj| {
            let this = obj.cast::<FixedGate>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::llm::duplex::AudioGate> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/duplex.rs.
type AdaptiveNoiseGateImpl = <crate::Selected as super::Implementations>::AdaptiveNoiseGate;

fn _extract_adaptive_noise_gate_new_activation_ratio(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "activation_ratio")
}

fn _extract_adaptive_noise_gate_new_deactivation_ratio(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "deactivation_ratio")
}

fn _extract_adaptive_noise_gate_new_min_silence_duration(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "min_silence_duration")
}

fn _extract_adaptive_noise_gate_new_window(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "window")
}

/// Opens on output that stands out from the model's own silence: the floor is the quietest
/// `min_silence_duration` stretch it produced while not speaking, within `window`.
#[pyclass(frozen, name = "AdaptiveNoiseGate", module = "livekit.agents._core")]
pub struct AdaptiveNoiseGate(::std::sync::Arc<AdaptiveNoiseGateImpl>);

#[pymethods]
impl AdaptiveNoiseGate {
    #[new]
    #[pyo3(signature = (*, activation_ratio = 3.0, deactivation_ratio = 1.8, min_silence_duration = 0.5, window = 10.0))]
    fn new(#[pyo3(from_py_with = _extract_adaptive_noise_gate_new_activation_ratio)] activation_ratio: f64, #[pyo3(from_py_with = _extract_adaptive_noise_gate_new_deactivation_ratio)] deactivation_ratio: f64, #[pyo3(from_py_with = _extract_adaptive_noise_gate_new_min_silence_duration)] min_silence_duration: f64, #[pyo3(from_py_with = _extract_adaptive_noise_gate_new_window)] window: f64) -> PyResult<AdaptiveNoiseGate> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <AdaptiveNoiseGateImpl as ::lk_agents_abi::llm::duplex::AdaptiveNoiseGate>::new(activation_ratio, deactivation_ratio, min_silence_duration, window);
        Ok(AdaptiveNoiseGate(::std::sync::Arc::new(out)))
    }

    /// True while the frame belongs to an open burst of output.
    #[pyo3(signature = (frame))]
    fn update(&self, frame: ::lk_agents_abi::AudioFrame) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <AdaptiveNoiseGateImpl as ::lk_agents_abi::llm::duplex::AdaptiveNoiseGate>::update(&self.0, frame).map_err(|e: ::lk_agents_abi::llm::inference::InvalidPayload| <::lk_agents_abi::llm::inference::InvalidPayload as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    /// End the open burst, keeping whatever has been learned about the model.
    fn deactivate(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <AdaptiveNoiseGateImpl as ::lk_agents_abi::llm::duplex::AdaptiveNoiseGate>::deactivate(&self.0);
        Ok(())
    }
}

// Wherever the host passes a `::lk_agents_abi::llm::duplex::AudioGate`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::llm::duplex::AudioGate>(),
        read: |obj| {
            let this = obj.cast::<AdaptiveNoiseGate>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::llm::duplex::AudioGate> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/duplex.rs.
type DuplexRealtimeModelImpl = <crate::Selected as super::Implementations>::DuplexRealtimeModel;

fn _extract_duplex_realtime_model_new_audio_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "audio_timeout")
}

/// `DuplexRealtimeAdapter`'s engine, the session's realtime model when handed to it.
#[pyclass(frozen, name = "DuplexRealtimeModel", module = "livekit.agents._core")]
pub struct DuplexRealtimeModel(::std::sync::Arc<DuplexRealtimeModelImpl>);

#[pymethods]
impl DuplexRealtimeModel {
    /// `audio_timeout`: how long to wait for the next frame before the output counts as finished.
    #[new]
    #[pyo3(signature = (host, *, audio_timeout = 0.8))]
    fn new(host: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_duplex_realtime_model_new_audio_timeout)] audio_timeout: f64) -> PyResult<DuplexRealtimeModel> {
        ::lk_agents_abi::python::check_pid()?;
        let host = crate::generated::llm::DuplexHostProxy::from_host(&host)?;
        let out = <DuplexRealtimeModelImpl as ::lk_agents_abi::llm::duplex::DuplexRealtimeModel>::new(host, audio_timeout).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(DuplexRealtimeModel(::std::sync::Arc::new(out)))
    }

    /// The realtime capabilities the model runs under.
    fn capabilities(&self) -> PyResult<::lk_agents_abi::llm::realtime::RealtimeCapabilities> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <DuplexRealtimeModelImpl as ::lk_agents_abi::llm::duplex::DuplexRealtimeModel>::capabilities(&self.0);
        Ok(out)
    }
}

// Wherever the host passes a `::lk_agents_abi::Realtime`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::Realtime>(),
        read: |obj| {
            let this = obj.cast::<DuplexRealtimeModel>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::Realtime> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/fallback.rs.
type FallbackLlmImpl = <crate::Selected as super::Implementations>::FallbackLlm;

fn _extract_fallback_llm_new_attempt_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "attempt_timeout")
}

fn _extract_fallback_llm_new_max_retry_per_llm(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    ::lk_agents_abi::python::number_from_py::<u32>(&obj, "max_retry_per_llm")
}

fn _extract_fallback_llm_new_retry_interval(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "retry_interval")
}

/// The native LLM chain behind `llm.FallbackAdapter`. Runtime owns its live providers and tasks.
#[pyclass(frozen, name = "FallbackLlm", module = "livekit.agents._core")]
pub struct FallbackLlm(::std::sync::Arc<FallbackLlmImpl>);

#[pymethods]
impl FallbackLlm {
    #[new]
    #[pyo3(signature = (llms, *, attempt_timeout = 5.0, max_retry_per_llm = 0, retry_interval = 0.5, sticky = false, on_availability = None))]
    fn new(llms: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_fallback_llm_new_attempt_timeout)] attempt_timeout: f64, #[pyo3(from_py_with = _extract_fallback_llm_new_max_retry_per_llm)] max_retry_per_llm: u32, #[pyo3(from_py_with = _extract_fallback_llm_new_retry_interval)] retry_interval: f64, sticky: bool, on_availability: Option<Bound<'_, PyAny>>) -> PyResult<FallbackLlm> {
        ::lk_agents_abi::python::check_pid()?;
        let llms = llms.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::llm::LlmProxy::from_host(&item) }).collect::<PyResult<Vec<_>>>())?;
        let on_availability = on_availability.map(|v| crate::generated::llm::ChainObserverProxy::from_host(&v)).transpose()?;
        let out = <FallbackLlmImpl as ::lk_agents_abi::llm::fallback::FallbackLlm>::new(llms, attempt_timeout, max_retry_per_llm, retry_interval, sticky, on_availability).map_err(|e: ::lk_agents_abi::llm::inference::InvalidPayload| <::lk_agents_abi::llm::inference::InvalidPayload as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(FallbackLlm(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn active_index(&self) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <FallbackLlmImpl as ::lk_agents_abi::llm::fallback::FallbackLlm>::active_index(&self.0);
        Ok(out)
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <FallbackLlmImpl as ::lk_agents_abi::llm::fallback::FallbackLlm>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[pyo3(signature = (chat_ctx_host, *, tools, tools_host = None, parallel_tool_calls = None, tool_choice = None, extra_kwargs = None, conn_options = None))]
    fn chat<'py>(&self, py: Python<'py>, chat_ctx_host: ::lk_agents_abi::HostRef, tools: ::std::vec::Vec<::lk_agents_abi::ToolSpec>, tools_host: ::std::option::Option<::lk_agents_abi::HostRef>, parallel_tool_calls: ::std::option::Option<bool>, tool_choice: Option<Bound<'_, PyAny>>, extra_kwargs: Option<Bound<'_, PyAny>>, conn_options: ::std::option::Option<::lk_agents_abi::ApiConnectOptions>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let tool_choice = tool_choice.map(|v| ::lk_agents_abi::python::value_from_py(&v, "tool_choice")).transpose()?;
        let extra_kwargs = extra_kwargs.map(|v| match ::lk_agents_abi::python::value_from_py(&v, "extra_kwargs")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("extra_kwargs", ": expected a dict"))) }).transpose()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <FallbackLlmImpl as ::lk_agents_abi::llm::fallback::FallbackLlm>::chat(&this, chat_ctx_host, tools, tools_host, parallel_tool_calls, tool_choice, extra_kwargs, conn_options)
                .await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(FallbackLlmChunks(::lk_agents_runtime::source::NativeSource::new(::std::boxed::Box::new(out))), py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::Llm`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::Llm>(),
        read: |obj| {
            let this = obj.cast::<FallbackLlm>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::Llm> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

/// One stream of `::lk_agents_abi::LlmChunk` items: `recv()` resolves to the next, `None` at the end.
#[pyclass(frozen, name = "FallbackLlmChunks", module = "livekit.agents._core")]
pub struct FallbackLlmChunks(::lk_agents_runtime::source::NativeSource<::lk_agents_abi::LlmChunk>);

#[pymethods]
impl FallbackLlmChunks {
    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        crate::host::source::recv(&self.0, py, |py, item| ::pyo3::IntoPyObjectExt::into_py_any(item, py))
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        crate::host::source::aclose(&self.0, py)
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/fallback.rs.
type FallbackTtsImpl = <crate::Selected as super::Implementations>::FallbackTts;

fn _extract_fallback_tts_new_max_retry_per_tts(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    ::lk_agents_abi::python::number_from_py::<u32>(&obj, "max_retry_per_tts")
}

fn _extract_fallback_tts_new_sample_rate(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<u32>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<u32>(&obj, "sample_rate")).map(Some) }
}

/// `tts.FallbackAdapter`'s engine as the host builds it; the public class stays Python.
#[pyclass(frozen, name = "FallbackTts", module = "livekit.agents._core")]
pub struct FallbackTts(::std::sync::Arc<FallbackTtsImpl>);

#[pymethods]
impl FallbackTts {
    #[new]
    #[pyo3(signature = (ttss, *, max_retry_per_tts = 2, sample_rate = None, on_availability = None))]
    fn new(ttss: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_fallback_tts_new_max_retry_per_tts)] max_retry_per_tts: u32, #[pyo3(from_py_with = _extract_fallback_tts_new_sample_rate)] sample_rate: ::std::option::Option<u32>, on_availability: Option<Bound<'_, PyAny>>) -> PyResult<FallbackTts> {
        ::lk_agents_abi::python::check_pid()?;
        let ttss = ttss.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::tts::TtsProxy::from_host(&item) }).collect::<PyResult<Vec<_>>>())?;
        let on_availability = on_availability.map(|v| crate::generated::llm::ChainObserverProxy::from_host(&v)).transpose()?;
        let out = <FallbackTtsImpl as ::lk_agents_abi::llm::fallback::FallbackTts>::new(ttss, max_retry_per_tts, sample_rate, on_availability).map_err(|e: ::lk_agents_abi::llm::inference::InvalidPayload| <::lk_agents_abi::llm::inference::InvalidPayload as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(FallbackTts(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn active_index(&self) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <FallbackTtsImpl as ::lk_agents_abi::llm::fallback::FallbackTts>::active_index(&self.0);
        Ok(out)
    }

    #[getter]
    fn capabilities(&self) -> PyResult<::lk_agents_abi::TtsCapabilities> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <FallbackTtsImpl as ::lk_agents_abi::llm::fallback::FallbackTts>::capabilities(&self.0);
        Ok(out)
    }

    #[getter]
    fn sample_rate(&self) -> PyResult<u32> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <FallbackTtsImpl as ::lk_agents_abi::llm::fallback::FallbackTts>::sample_rate(&self.0);
        Ok(out)
    }

    #[getter]
    fn num_channels(&self) -> PyResult<u16> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <FallbackTtsImpl as ::lk_agents_abi::llm::fallback::FallbackTts>::num_channels(&self.0);
        Ok(out)
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <FallbackTtsImpl as ::lk_agents_abi::llm::fallback::FallbackTts>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[pyo3(signature = (text, conn_options = None))]
    fn synthesize<'py>(&self, py: Python<'py>, text: ::std::string::String, conn_options: ::std::option::Option<::lk_agents_abi::ApiConnectOptions>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <FallbackTtsImpl as ::lk_agents_abi::llm::fallback::FallbackTts>::synthesize(&this, text, conn_options)
                .await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(super::tts::FallbackTtsStream(::std::sync::Arc::new(out)), py))
        })
    }

    #[pyo3(signature = (conn_options = None))]
    fn stream<'py>(&self, py: Python<'py>, conn_options: ::std::option::Option<::lk_agents_abi::ApiConnectOptions>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <FallbackTtsImpl as ::lk_agents_abi::llm::fallback::FallbackTts>::stream(&this, conn_options)
                .await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(super::tts::FallbackTtsStream(::std::sync::Arc::new(out)), py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::Tts`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::Tts>(),
        read: |obj| {
            let this = obj.cast::<FallbackTts>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::Tts> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/fallback.rs.
type TtsTokenLogImpl = <crate::Selected as super::Implementations>::TtsTokenLog;

/// The pushed-token log as the host's `SynthesizeStream` keeps it (`_core.TtsTokenLog`): the text
/// a recovering stream replays.
///
/// `self._pushed_tokens` plus the live child sink. The text leg may not drop (§1.2d), so nothing
/// here is lossy or bounded.
#[pyclass(frozen, name = "TtsTokenLog", module = "livekit.agents._core")]
pub struct TtsTokenLog(::std::sync::Arc<TtsTokenLogImpl>);

#[pymethods]
impl TtsTokenLog {
    #[new]
    fn new() -> PyResult<TtsTokenLog> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <TtsTokenLogImpl as ::lk_agents_abi::llm::fallback::TtsTokenLog>::new();
        Ok(TtsTokenLog(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (text))]
    fn push(&self, text: ::std::string::String) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <TtsTokenLogImpl as ::lk_agents_abi::llm::fallback::TtsTokenLog>::push(&self.0, text);
        Ok(())
    }

    fn flush(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <TtsTokenLogImpl as ::lk_agents_abi::llm::fallback::TtsTokenLog>::flush(&self.0);
        Ok(())
    }

    fn end(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <TtsTokenLogImpl as ::lk_agents_abi::llm::fallback::TtsTokenLog>::end(&self.0);
        Ok(())
    }

    fn snapshot(&self) -> PyResult<(::std::vec::Vec<::std::option::Option<::std::string::String>>, bool)> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <TtsTokenLogImpl as ::lk_agents_abi::llm::fallback::TtsTokenLog>::snapshot(&self.0);
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/inference.rs.
type InferenceLlmImpl = <crate::Selected as super::Implementations>::InferenceLlm;

/// The gateway LLM behind `inference.LLM`. Its options are shared by every view
/// (`identified`); each view's requests carry its own job's identification.
#[pyclass(frozen, name = "InferenceLlm", module = "livekit.agents._core")]
pub struct InferenceLlm(::std::sync::Arc<InferenceLlmImpl>);

#[pymethods]
impl InferenceLlm {
    /// A client of `model` on the gateway at `base_url`. Starts nothing.
    #[new]
    #[pyo3(signature = (model, *, provider = None, base_url, api_key, api_secret, inference_class = None, extra_kwargs = None))]
    fn new(model: ::std::string::String, provider: ::std::option::Option<::std::string::String>, base_url: ::std::string::String, api_key: ::std::string::String, api_secret: ::std::string::String, inference_class: ::std::option::Option<::std::string::String>, extra_kwargs: Option<Bound<'_, PyAny>>) -> PyResult<InferenceLlm> {
        ::lk_agents_abi::python::check_pid()?;
        let extra_kwargs = extra_kwargs.map(|v| match ::lk_agents_abi::python::value_from_py(&v, "extra_kwargs")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("extra_kwargs", ": expected a dict"))) }).transpose()?;
        let out = <InferenceLlmImpl as ::lk_agents_abi::llm::inference::InferenceLlm>::new(model, provider, base_url, api_key, api_secret, inference_class, extra_kwargs).map_err(|e: ::lk_agents_abi::errors::ApiError| <::lk_agents_abi::errors::ApiError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(InferenceLlm(::std::sync::Arc::new(out)))
    }

    /// This client as a session of the job `headers` (`get_inference_headers()`) identifies
    /// holds it: the same options, its requests carrying that job. `reporter` is the host
    /// `llm.LLM` whose `error` events report the attempts this view retries.
    #[pyo3(signature = (headers, reporter = None))]
    fn identified(&self, headers: ::std::vec::Vec<(::std::string::String, ::std::string::String)>, reporter: Option<Bound<'_, PyAny>>) -> PyResult<InferenceLlm> {
        ::lk_agents_abi::python::check_pid()?;
        let reporter = reporter.map(|v| crate::generated::llm::LlmProxy::from_host(&v)).transpose()?;
        let out = <InferenceLlmImpl as ::lk_agents_abi::llm::inference::InferenceLlm>::identified(&self.0, headers, reporter);
        Ok(InferenceLlm(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn model(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <InferenceLlmImpl as ::lk_agents_abi::llm::inference::InferenceLlm>::model(&self.0);
        Ok(out)
    }

    /// The persistent request options, a copy.
    #[getter]
    fn extra_kwargs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <InferenceLlmImpl as ::lk_agents_abi::llm::inference::InferenceLlm>::extra_kwargs(&self.0);
        Ok(::lk_agents_abi::python::extra_to_py(py, &out)?.into_any())
    }

    /// Read by the next `chat()`. `extra_kwargs` replaces the persistent options.
    #[pyo3(signature = (*, model = None, extra_kwargs = None))]
    fn update_options(&self, model: ::std::option::Option<::std::string::String>, extra_kwargs: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        let extra_kwargs = extra_kwargs.map(|v| match ::lk_agents_abi::python::value_from_py(&v, "extra_kwargs")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("extra_kwargs", ": expected a dict"))) }).transpose()?;
        <InferenceLlmImpl as ::lk_agents_abi::llm::inference::InferenceLlm>::update_options(&self.0, model, extra_kwargs);
        Ok(())
    }

    /// Opens a completion of `items`; no I/O until the first pull. `headers` identify the
    /// caller's job, read in its context.
    #[pyo3(signature = (items, *, tools, conn_options, parallel_tool_calls = None, tool_choice = None, response_format = None, extra_kwargs = None, headers = None))]
    fn chat(&self, items: ::std::vec::Vec<::lk_agents_abi::ChatItem>, tools: ::std::vec::Vec<::lk_agents_abi::ToolSpec>, conn_options: ::lk_agents_abi::ApiConnectOptions, parallel_tool_calls: ::std::option::Option<bool>, tool_choice: ::std::option::Option<::lk_agents_abi::ToolChoice>, response_format: Option<Bound<'_, PyAny>>, extra_kwargs: Option<Bound<'_, PyAny>>, headers: ::std::option::Option<::std::vec::Vec<(::std::string::String, ::std::string::String)>>) -> PyResult<InferenceLlmChunks> {
        ::lk_agents_abi::python::check_pid()?;
        let response_format = response_format.map(|v| ::lk_agents_abi::python::value_from_py(&v, "response_format")).transpose()?;
        let extra_kwargs = extra_kwargs.map(|v| match ::lk_agents_abi::python::value_from_py(&v, "extra_kwargs")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("extra_kwargs", ": expected a dict"))) }).transpose()?;
        let out = <InferenceLlmImpl as ::lk_agents_abi::llm::inference::InferenceLlm>::chat(&self.0, items, tools, conn_options, parallel_tool_calls, tool_choice, response_format, extra_kwargs, headers).map_err(|e: ::lk_agents_abi::errors::ApiError| <::lk_agents_abi::errors::ApiError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(InferenceLlmChunks(::lk_agents_runtime::source::NativeSource::new(::std::boxed::Box::new(out))))
    }
}

// Wherever the host passes a `::lk_agents_abi::Llm`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::Llm>(),
        read: |obj| {
            let this = obj.cast::<InferenceLlm>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::Llm> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

/// One stream of `::lk_agents_abi::LlmChunk` items: `recv()` resolves to the next, `None` at the end.
#[pyclass(frozen, name = "InferenceLlmChunks", module = "livekit.agents._core")]
pub struct InferenceLlmChunks(::lk_agents_runtime::source::NativeSource<::lk_agents_abi::LlmChunk>);

#[pymethods]
impl InferenceLlmChunks {
    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        crate::host::source::recv(&self.0, py, |py, item| ::pyo3::IntoPyObjectExt::into_py_any(item, py))
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        crate::host::source::aclose(&self.0, py)
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/inference.rs.
type LlmChoiceParserImpl = <crate::Selected as super::Implementations>::ChoiceParser;

/// Accumulates the streamed choices of one OpenAI-compatible completion into chunks, filtering
/// the model's thinking tags: `inference.LLMStream` over a plugin's own client.
#[pyclass(name = "LlmChoiceParser", module = "livekit.agents._core")]
pub struct LlmChoiceParser(LlmChoiceParserImpl);

#[pymethods]
impl LlmChoiceParser {
    #[new]
    #[pyo3(signature = (model))]
    fn new(model: ::std::string::String) -> PyResult<LlmChoiceParser> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <LlmChoiceParserImpl as ::lk_agents_abi::llm::inference::ChoiceParser>::new(model);
        Ok(LlmChoiceParser(out))
    }

    /// The chunk `choice` (one streamed choice of completion `id`) completes, if any.
    #[pyo3(signature = (id, choice))]
    fn parse(&mut self, id: ::std::string::String, choice: Bound<'_, PyAny>) -> PyResult<::std::option::Option<::lk_agents_abi::LlmChunk>> {
        ::lk_agents_abi::python::check_pid()?;
        let choice = ::lk_agents_abi::python::value_from_py(&choice, "choice")?;
        let out = <LlmChoiceParserImpl as ::lk_agents_abi::llm::inference::ChoiceParser>::parse(&mut self.0, id, choice).map_err(|e: ::lk_agents_abi::llm::inference::InvalidPayload| <::lk_agents_abi::llm::inference::InvalidPayload as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/inference.rs.
type CompletionWireImpl = <crate::Selected as super::Implementations>::CompletionWire;

/// The usage chunk of a streamed completion chunk, when it reports usage.
#[pyfunction]
#[pyo3(signature = (chunk))]
pub fn llm_usage_chunk(chunk: Bound<'_, PyAny>) -> PyResult<::std::option::Option<::lk_agents_abi::LlmChunk>> {
    ::lk_agents_abi::python::check_pid()?;
    let chunk = ::lk_agents_abi::python::value_from_py(&chunk, "chunk")?;
    let out = <CompletionWireImpl as ::lk_agents_abi::llm::inference::CompletionWire>::llm_usage_chunk(chunk).map_err(|e: ::lk_agents_abi::llm::inference::InvalidPayload| <::lk_agents_abi::llm::inference::InvalidPayload as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/llm/mcp.rs.
type McpSessionImpl = <crate::Selected as super::Implementations>::McpSession;

fn _extract_mcp_session_new_client_session_timeout_seconds(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "client_session_timeout_seconds")
}

fn _extract_mcp_session_connect_http_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "timeout")
}

fn _extract_mcp_session_connect_http_sse_read_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "sse_read_timeout")
}

/// An SDK session whose owner cancels and joins independently of request borrowers.
#[pyclass(frozen, name = "McpSession", module = "livekit.agents._core")]
pub struct McpSession(::std::sync::Arc<McpSessionImpl>);

#[pymethods]
impl McpSession {
    #[new]
    #[pyo3(signature = (client_session_timeout_seconds))]
    fn new(#[pyo3(from_py_with = _extract_mcp_session_new_client_session_timeout_seconds)] client_session_timeout_seconds: f64) -> PyResult<McpSession> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <McpSessionImpl as ::lk_agents_abi::llm::mcp::McpSession>::new(client_session_timeout_seconds).map_err(|e: ::lk_agents_abi::llm::mcp::McpError| <::lk_agents_abi::llm::mcp::McpError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(McpSession(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn connected(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <McpSessionImpl as ::lk_agents_abi::llm::mcp::McpSession>::connected(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (allowed))]
    fn set_allowed_tools(&self, allowed: ::std::vec::Vec<::std::string::String>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <McpSessionImpl as ::lk_agents_abi::llm::mcp::McpSession>::set_allowed_tools(&self.0, allowed);
        Ok(())
    }

    fn invalidate_cache(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <McpSessionImpl as ::lk_agents_abi::llm::mcp::McpSession>::invalidate_cache(&self.0);
        Ok(())
    }

    #[pyo3(signature = (url, headers, transport, timeout, sse_read_timeout))]
    fn connect_http<'py>(&self, py: Python<'py>, url: ::std::string::String, headers: ::std::vec::Vec<(::std::string::String, ::std::string::String)>, transport: ::std::string::String, #[pyo3(from_py_with = _extract_mcp_session_connect_http_timeout)] timeout: f64, #[pyo3(from_py_with = _extract_mcp_session_connect_http_sse_read_timeout)] sse_read_timeout: f64) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <McpSessionImpl as ::lk_agents_abi::llm::mcp::McpSession>::connect_http(&this, url, headers, transport, timeout, sse_read_timeout)
                .await.map_err(|e: ::lk_agents_abi::llm::mcp::McpError| <::lk_agents_abi::llm::mcp::McpError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[pyo3(signature = (command, args, env, cwd))]
    fn connect_stdio<'py>(&self, py: Python<'py>, command: ::std::string::String, args: ::std::vec::Vec<::std::string::String>, env: ::std::vec::Vec<(::std::string::String, ::std::string::String)>, cwd: ::std::option::Option<::std::string::String>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <McpSessionImpl as ::lk_agents_abi::llm::mcp::McpSession>::connect_stdio(&this, command, args, env, cwd)
                .await.map_err(|e: ::lk_agents_abi::llm::mcp::McpError| <::lk_agents_abi::llm::mcp::McpError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[pyo3(signature = (transport, host_context))]
    fn connect_host<'py>(&self, py: Python<'py>, transport: Bound<'_, PyAny>, host_context: ::std::option::Option<::lk_agents_abi::HostRef>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let transport = crate::generated::llm::McpTransportProxy::from_host(&transport)?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <McpSessionImpl as ::lk_agents_abi::llm::mcp::McpSession>::connect_host(&this, transport, host_context)
                .await.map_err(|e: ::lk_agents_abi::llm::mcp::McpError| <::lk_agents_abi::llm::mcp::McpError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    fn list_tools<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <McpSessionImpl as ::lk_agents_abi::llm::mcp::McpSession>::list_tools(&this, )
                .await.map_err(|e: ::lk_agents_abi::llm::mcp::McpError| <::lk_agents_abi::llm::mcp::McpError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(::lk_agents_abi::python::value_to_py(py, &out)?, py))
        })
    }

    #[pyo3(signature = (name, arguments, progress, host_context))]
    fn call_tool<'py>(&self, py: Python<'py>, name: ::std::string::String, arguments: Bound<'_, PyAny>, progress: Option<Bound<'_, PyAny>>, host_context: ::std::option::Option<::lk_agents_abi::HostRef>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let arguments = match ::lk_agents_abi::python::value_from_py(&arguments, "arguments")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("arguments", ": expected a dict"))) }?;
        let progress = progress.map(|v| crate::generated::llm::McpProgressProxy::from_host(&v)).transpose()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <McpSessionImpl as ::lk_agents_abi::llm::mcp::McpSession>::call_tool(&this, name, arguments, progress, host_context)
                .await.map_err(|e: ::lk_agents_abi::llm::mcp::McpError| <::lk_agents_abi::llm::mcp::McpError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(::lk_agents_abi::python::value_to_py(py, &out)?, py))
        })
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <McpSessionImpl as ::lk_agents_abi::llm::mcp::McpSession>::close(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[getter]
    fn cleanup<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <McpSessionImpl as ::lk_agents_abi::llm::mcp::McpSession>::cleanup(&self.0);
        Ok(crate::generated::records::mcp_session::mcp_cleanup_to_py(py, out)?)
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/mcp.rs.
type McpToolsImpl = <crate::Selected as super::Implementations>::McpTools;

#[pyfunction]
#[pyo3(signature = (config))]
pub fn resolve_mcp_tool_options(config: ::std::option::Option<::lk_agents_abi::llm::mcp::McpToolOptionsPatch>) -> PyResult<::lk_agents_abi::llm::mcp::McpToolOptions> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <McpToolsImpl as ::lk_agents_abi::llm::mcp::McpTools>::resolve_tool_options(config);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (name, description, input_schema, meta))]
pub fn mcp_raw_schema<'py>(py: Python<'py>, name: ::std::string::String, description: ::std::option::Option<::std::string::String>, input_schema: Bound<'_, PyAny>, meta: Option<Bound<'_, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    ::lk_agents_abi::python::check_pid()?;
    let input_schema = match ::lk_agents_abi::python::value_from_py(&input_schema, "input_schema")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("input_schema", ": expected a dict"))) }?;
    let meta = meta.map(|v| match ::lk_agents_abi::python::value_from_py(&v, "meta")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("meta", ": expected a dict"))) }).transpose()?;
    let out = <McpToolsImpl as ::lk_agents_abi::llm::mcp::McpTools>::raw_schema(name, description, input_schema, meta);
    Ok(::lk_agents_abi::python::value_to_py(py, &out)?)
}

#[pyfunction]
#[pyo3(signature = (reload, toolset_initialized, server_connected))]
pub fn mcp_plan_setup(reload: bool, toolset_initialized: bool, server_connected: bool) -> PyResult<::std::string::String> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <McpToolsImpl as ::lk_agents_abi::llm::mcp::McpTools>::plan_setup(reload, toolset_initialized, server_connected);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (url, transport_type))]
pub fn mcp_http_transport(url: ::std::string::String, transport_type: ::std::option::Option<::std::string::String>) -> PyResult<::std::string::String> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <McpToolsImpl as ::lk_agents_abi::llm::mcp::McpTools>::http_transport(url, transport_type).map_err(|e: ::lk_agents_abi::llm::mcp::McpError| <::lk_agents_abi::llm::mcp::McpError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (tool_name, result))]
pub fn mcp_render_result(tool_name: ::std::string::String, result: Bound<'_, PyAny>) -> PyResult<::std::string::String> {
    ::lk_agents_abi::python::check_pid()?;
    let result = ::lk_agents_abi::python::value_from_py(&result, "result")?;
    let out = <McpToolsImpl as ::lk_agents_abi::llm::mcp::McpTools>::render_result(tool_name, result).map_err(|e: ::lk_agents_abi::llm::mcp::McpError| <::lk_agents_abi::llm::mcp::McpError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (result))]
pub fn mcp_check_result(result: Bound<'_, PyAny>) -> PyResult<()> {
    ::lk_agents_abi::python::check_pid()?;
    let result = ::lk_agents_abi::python::value_from_py(&result, "result")?;
    <McpToolsImpl as ::lk_agents_abi::llm::mcp::McpTools>::check_result(result).map_err(|e: ::lk_agents_abi::llm::mcp::McpError| <::lk_agents_abi::llm::mcp::McpError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(())
}

// Declared in rust/crates/lk-agents-abi/src/llm/provider_format/host.rs.
type LlmAlgorithmsImpl = <crate::Selected as super::Implementations>::LlmAlgorithms;

fn _extract_llm_algorithms_prepare_provider_format_thought_signatures(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<::std::collections::BTreeMap<::std::string::String, ::std::vec::Vec<u8>>>> {
    if obj.is_none() { Ok(None) } else { (obj.cast::<::pyo3::types::PyDict>().map_err(::pyo3::PyErr::from).and_then(|dict| dict.iter().map(|(key, item)| Ok((key.extract::<::std::string::String>()?, { if item.is_instance_of::<::pyo3::types::PyBytes>() || item.is_instance_of::<::pyo3::types::PyByteArray>() || item.is_instance_of::<::pyo3::types::PyMemoryView>() { ::lk_agents_abi::python::owned_bytes(&item, "thought_signatures") } else { item.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "thought_signatures") }).collect::<PyResult<Vec<_>>>()) } }?))).collect::<PyResult<::std::collections::BTreeMap<_, _>>>())).map(Some) }
}

#[pyfunction]
#[pyo3(signature = (format, items, *, images = None, inject_dummy_user_message = true, inject_trailing_user_message = false, thought_signatures = None))]
pub fn to_provider_format<'py>(py: Python<'py>, format: ::std::string::String, items: ::std::vec::Vec<::lk_agents_abi::ChatItem>, images: ::std::option::Option<::std::collections::BTreeMap<::std::string::String, ::lk_agents_abi::llm::provider_format::SerializedImage>>, inject_dummy_user_message: bool, inject_trailing_user_message: bool, #[pyo3(from_py_with = _extract_llm_algorithms_prepare_provider_format_thought_signatures)] thought_signatures: ::std::option::Option<::std::collections::BTreeMap<::std::string::String, ::std::vec::Vec<u8>>>) -> PyResult<Bound<'py, PyAny>> {
    ::lk_agents_abi::python::check_pid()?;
    crate::host::future_into_py(py, async move {
        let out = <LlmAlgorithmsImpl as ::lk_agents_abi::llm::provider_format::host::LlmAlgorithms>::prepare_provider_format(format, items, images, inject_dummy_user_message, inject_trailing_user_message, thought_signatures).await.map_err(|e: ::lk_agents_abi::llm::provider_format::host::LlmAlgorithmError| <::lk_agents_abi::llm::provider_format::host::LlmAlgorithmError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any({ let (x0, x1,) = out; ::pyo3::types::PyTuple::new(py, [::pyo3::types::PyList::new(py, x0.into_iter().map(|x| PyResult::Ok(::lk_agents_abi::python::value_to_py(py, &x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any(), ::lk_agents_abi::python::value_to_py(py, &x1)?])?.into_any() }, py))
    })
}

#[pyfunction]
#[pyo3(signature = (items, *, role, template))]
pub fn convert_mid_conversation_instructions(items: ::std::vec::Vec<::lk_agents_abi::ChatItem>, role: ::lk_agents_abi::Role, template: ::std::string::String) -> PyResult<::std::vec::Vec<(usize, ::std::option::Option<::lk_agents_abi::ChatItem>)>> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <LlmAlgorithmsImpl as ::lk_agents_abi::llm::provider_format::host::LlmAlgorithms>::convert_mid_conversation_instructions(items, role, &template);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (format, tools, *, tool_behavior = None, use_parameters_json_schema = false))]
pub fn to_fnc_ctx<'py>(py: Python<'py>, format: ::std::string::String, tools: Bound<'_, PyAny>, tool_behavior: ::std::option::Option<::std::string::String>, use_parameters_json_schema: bool) -> PyResult<Bound<'py, PyAny>> {
    ::lk_agents_abi::python::check_pid()?;
    let tools = tools.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::value_from_py(&item, "tools") }).collect::<PyResult<Vec<_>>>())?;
    let out = <LlmAlgorithmsImpl as ::lk_agents_abi::llm::provider_format::host::LlmAlgorithms>::to_fnc_ctx(&format, tools, tool_behavior, use_parameters_json_schema).map_err(|e: ::lk_agents_abi::llm::provider_format::host::LlmAlgorithmError| <::lk_agents_abi::llm::provider_format::host::LlmAlgorithmError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(::pyo3::types::PyList::new(py, out.into_iter().map(|x| PyResult::Ok(::lk_agents_abi::python::value_to_py(py, &x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())
}

#[pyfunction]
#[pyo3(signature = (schema, *, null_sentinel_for_defaults = false))]
pub fn to_strict_json_schema<'py>(py: Python<'py>, schema: Bound<'_, PyAny>, null_sentinel_for_defaults: bool) -> PyResult<Bound<'py, PyAny>> {
    let schema = ::lk_agents_abi::python::value_from_py(&schema, "schema")?;
    let out = <LlmAlgorithmsImpl as ::lk_agents_abi::llm::provider_format::host::LlmAlgorithms>::to_strict_json_schema(&schema, null_sentinel_for_defaults).map_err(|e: ::lk_agents_abi::llm::provider_format::StrictError| <::lk_agents_abi::llm::provider_format::StrictError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(::lk_agents_abi::python::extra_to_py(py, &out)?.into_any())
}

#[pyfunction]
#[pyo3(signature = (*, root, r#ref))]
pub fn resolve_ref<'py>(py: Python<'py>, root: Bound<'_, PyAny>, r#ref: ::std::string::String) -> PyResult<Bound<'py, PyAny>> {
    let root = match ::lk_agents_abi::python::value_from_py(&root, "root")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("root", ": expected a dict"))) }?;
    let out = <LlmAlgorithmsImpl as ::lk_agents_abi::llm::provider_format::host::LlmAlgorithms>::resolve_ref(&root, &r#ref).map_err(|e: ::lk_agents_abi::llm::provider_format::StrictError| <::lk_agents_abi::llm::provider_format::StrictError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(::lk_agents_abi::python::extra_to_py(py, &out)?.into_any())
}

#[pyfunction]
#[pyo3(signature = (value, schema, root = None))]
pub fn inject_schema_defaults<'py>(py: Python<'py>, value: Bound<'_, PyAny>, schema: Bound<'_, PyAny>, root: Option<Bound<'_, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let value = ::lk_agents_abi::python::value_from_py(&value, "value")?;
    let schema = ::lk_agents_abi::python::value_from_py(&schema, "schema")?;
    let root = root.map(|v| ::lk_agents_abi::python::value_from_py(&v, "root")).transpose()?;
    let out = <LlmAlgorithmsImpl as ::lk_agents_abi::llm::provider_format::host::LlmAlgorithms>::inject_schema_defaults(&value, &schema, root).map_err(|e: ::lk_agents_abi::llm::schema::SchemaError| <::lk_agents_abi::llm::schema::SchemaError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(::lk_agents_abi::python::value_to_py(py, &out)?)
}

#[pyfunction]
#[pyo3(signature = (name, description, parameters, *, internally_tagged = false))]
pub fn build_legacy_openai_schema<'py>(py: Python<'py>, name: ::std::string::String, description: ::std::option::Option<::std::string::String>, parameters: Bound<'_, PyAny>, internally_tagged: bool) -> PyResult<Bound<'py, PyAny>> {
    let parameters = ::lk_agents_abi::python::value_from_py(&parameters, "parameters")?;
    let out = <LlmAlgorithmsImpl as ::lk_agents_abi::llm::provider_format::host::LlmAlgorithms>::build_legacy_openai_schema(&name, description, parameters, internally_tagged);
    Ok(::lk_agents_abi::python::extra_to_py(py, &out)?.into_any())
}

#[pyfunction]
#[pyo3(signature = (name, description, parameters))]
pub fn build_strict_openai_schema<'py>(py: Python<'py>, name: ::std::string::String, description: ::std::option::Option<::std::string::String>, parameters: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let parameters = ::lk_agents_abi::python::value_from_py(&parameters, "parameters")?;
    let out = <LlmAlgorithmsImpl as ::lk_agents_abi::llm::provider_format::host::LlmAlgorithms>::build_strict_openai_schema(&name, description, &parameters).map_err(|e: ::lk_agents_abi::llm::schema::SchemaError| <::lk_agents_abi::llm::schema::SchemaError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(::lk_agents_abi::python::extra_to_py(py, &out)?.into_any())
}

#[pyfunction]
#[pyo3(signature = (name, schema))]
pub fn to_openai_response_format<'py>(py: Python<'py>, name: ::std::string::String, schema: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let schema = ::lk_agents_abi::python::value_from_py(&schema, "schema")?;
    let out = <LlmAlgorithmsImpl as ::lk_agents_abi::llm::provider_format::host::LlmAlgorithms>::to_openai_response_format(&name, &schema).map_err(|e: ::lk_agents_abi::llm::schema::SchemaError| <::lk_agents_abi::llm::schema::SchemaError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(::lk_agents_abi::python::extra_to_py(py, &out)?.into_any())
}

#[pyfunction]
#[pyo3(signature = (raw))]
pub fn parse_function_arguments<'py>(py: Python<'py>, raw: ::std::string::String) -> PyResult<Bound<'py, PyAny>> {
    let out = <LlmAlgorithmsImpl as ::lk_agents_abi::llm::provider_format::host::LlmAlgorithms>::parse_function_arguments(&raw).map_err(|e: ::lk_agents_abi::llm::provider_format::host::LlmAlgorithmError| <::lk_agents_abi::llm::provider_format::host::LlmAlgorithmError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(::lk_agents_abi::python::extra_to_py(py, &out)?.into_any())
}

#[pyfunction]
#[pyo3(signature = (name, args, scope))]
pub fn duplicate_key(name: ::std::string::String, args: Option<Bound<'_, PyAny>>, scope: ::lk_agents_abi::llm::mcp::DuplicateScope) -> PyResult<(::std::string::String, ::std::option::Option<::std::string::String>)> {
    ::lk_agents_abi::python::check_pid()?;
    let args = args.map(|v| ::lk_agents_abi::python::value_from_py(&v, "args")).transpose()?;
    let out = <LlmAlgorithmsImpl as ::lk_agents_abi::llm::provider_format::host::LlmAlgorithms>::duplicate_key(&name, args, scope);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (name, mode, confirm_duplicate, in_flight))]
pub fn decide_duplicate<'py>(py: Python<'py>, name: ::std::string::String, mode: ::lk_agents_abi::llm::mcp::DuplicateMode, confirm_duplicate: ::std::option::Option<bool>, in_flight: ::std::vec::Vec<(::std::string::String, bool)>) -> PyResult<Bound<'py, PyAny>> {
    let out = <LlmAlgorithmsImpl as ::lk_agents_abi::llm::provider_format::host::LlmAlgorithms>::decide_duplicate_wire(&name, mode, confirm_duplicate, in_flight);
    Ok({ let (x0, x1,) = out; ::pyo3::types::PyTuple::new(py, [::pyo3::IntoPyObjectExt::into_bound_py_any(x0, py)?, match x1 { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::value_to_py(py, &x)? }])?.into_any() })
}

#[pyfunction]
#[pyo3(signature = (parameters))]
pub fn inject_confirm_duplicate<'py>(py: Python<'py>, parameters: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let parameters = ::lk_agents_abi::python::value_from_py(&parameters, "parameters")?;
    let out = <LlmAlgorithmsImpl as ::lk_agents_abi::llm::provider_format::host::LlmAlgorithms>::inject_confirm_duplicate(&parameters);
    Ok(::lk_agents_abi::python::extra_to_py(py, &out)?.into_any())
}

#[pyfunction]
#[pyo3(signature = (content, start_tag, end_tag, buffer, end_marker, is_final))]
pub fn strip_thinking(content: ::std::option::Option<::std::string::String>, start_tag: ::std::string::String, end_tag: ::std::string::String, buffer: ::std::string::String, end_marker: ::std::option::Option<::std::string::String>, is_final: bool) -> PyResult<(::std::option::Option<::std::string::String>, ::std::string::String, ::std::option::Option<::std::string::String>)> {
    let out = <LlmAlgorithmsImpl as ::lk_agents_abi::llm::provider_format::host::LlmAlgorithms>::strip_thinking(content, start_tag, end_tag, buffer, end_marker, is_final);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (parts))]
pub fn flatten_delta_text(parts: ::std::vec::Vec<::std::string::String>) -> PyResult<::std::option::Option<::std::string::String>> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <LlmAlgorithmsImpl as ::lk_agents_abi::llm::provider_format::host::LlmAlgorithms>::flatten_delta_text(parts);
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.
type HostRealtimeModelImpl = <crate::Selected as super::Implementations>::HostRealtimeModel;

/// A plugin's realtime model as the core's own: its session opens when the driver first asks and
/// is the same when asked again.
#[pyclass(frozen, name = "HostRealtimeModel", module = "livekit.agents._core")]
pub struct HostRealtimeModel(::std::sync::Arc<HostRealtimeModelImpl>);

#[pymethods]
impl HostRealtimeModel {
    #[new]
    #[pyo3(signature = (model))]
    fn new(model: Bound<'_, PyAny>) -> PyResult<HostRealtimeModel> {
        ::lk_agents_abi::python::check_pid()?;
        let model = crate::generated::llm::HostRealtimeProxy::from_host(&model)?;
        let out = <HostRealtimeModelImpl as ::lk_agents_abi::llm::realtime::HostRealtimeModel>::new(model);
        Ok(HostRealtimeModel(::std::sync::Arc::new(out)))
    }
}

// Wherever the host passes a `::lk_agents_abi::Realtime`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::Realtime>(),
        read: |obj| {
            let this = obj.cast::<HostRealtimeModel>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::Realtime> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/realtime_native.rs.
type NativeRealtimeModelImpl = <crate::Selected as super::Implementations>::NativeRealtimeModel;

#[pyclass(frozen, name = "NativeRealtimeModel", module = "livekit.agents._core")]
pub struct NativeRealtimeModel(::std::sync::Arc<NativeRealtimeModelImpl>);

#[pymethods]
impl NativeRealtimeModel {
    #[new]
    #[pyo3(signature = (config))]
    fn new(config: Bound<'_, PyAny>) -> PyResult<NativeRealtimeModel> {
        ::lk_agents_abi::python::check_pid()?;
        let config = crate::generated::records::realtime_options::native_realtime_config_from_py(&config)?;
        let out = <NativeRealtimeModelImpl as ::lk_agents_abi::llm::realtime_native::NativeRealtimeModel>::new(config).map_err(|e: ::lk_agents_abi::llm::realtime_native::RealtimeConfigurationError| <::lk_agents_abi::llm::realtime_native::RealtimeConfigurationError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(NativeRealtimeModel(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn provider(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <NativeRealtimeModelImpl as ::lk_agents_abi::llm::realtime_native::NativeRealtimeModel>::provider(&self.0);
        Ok(out)
    }

    #[getter]
    fn model(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <NativeRealtimeModelImpl as ::lk_agents_abi::llm::realtime_native::NativeRealtimeModel>::model(&self.0);
        Ok(out)
    }

    #[getter]
    fn label(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <NativeRealtimeModelImpl as ::lk_agents_abi::llm::realtime_native::NativeRealtimeModel>::label(&self.0);
        Ok(out)
    }

    fn capabilities(&self) -> PyResult<::lk_agents_abi::llm::realtime::RealtimeCapabilities> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <NativeRealtimeModelImpl as ::lk_agents_abi::llm::realtime_native::NativeRealtimeModel>::capabilities(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (options))]
    fn update_options(&self, options: Bound<'_, PyAny>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        let options = crate::generated::records::realtime_options::realtime_options_from_py(&options)?;
        <NativeRealtimeModelImpl as ::lk_agents_abi::llm::realtime_native::NativeRealtimeModel>::update_options(&self.0, options);
        Ok(())
    }

    #[pyo3(signature = (headers))]
    fn bind_identification(&self, headers: ::std::vec::Vec<(::std::string::String, ::std::string::String)>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <NativeRealtimeModelImpl as ::lk_agents_abi::llm::realtime_native::NativeRealtimeModel>::bind_identification(&self.0, headers);
        Ok(())
    }
}

// Wherever the host passes a `::lk_agents_abi::Realtime`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::Realtime>(),
        read: |obj| {
            let this = obj.cast::<NativeRealtimeModel>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::Realtime> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/realtime_native.rs.
type RealtimeFallbackModelImpl = <crate::Selected as super::Implementations>::RealtimeFallbackModel;

fn _extract_realtime_fallback_model_new_cooldown(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "cooldown")
}

/// The generated facade holds native resources; host provider objects use the shared bridge.
#[pyclass(frozen, name = "RealtimeFallbackModel", module = "livekit.agents._core")]
pub struct RealtimeFallbackModel(::std::sync::Arc<RealtimeFallbackModelImpl>);

#[pymethods]
impl RealtimeFallbackModel {
    #[new]
    #[pyo3(signature = (models, *, cooldown = 10.0, regenerate_on_swap = true, on_availability = None))]
    fn new(models: ::std::vec::Vec<::lk_agents_abi::HostRef>, #[pyo3(from_py_with = _extract_realtime_fallback_model_new_cooldown)] cooldown: f64, regenerate_on_swap: bool, on_availability: ::std::option::Option<::lk_agents_abi::HostRef>) -> PyResult<RealtimeFallbackModel> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <RealtimeFallbackModelImpl as ::lk_agents_abi::llm::realtime_native::RealtimeFallbackModel>::new(models, cooldown, regenerate_on_swap, on_availability).map_err(|e: ::lk_agents_abi::llm::realtime_native::RealtimeConfigurationError| <::lk_agents_abi::llm::realtime_native::RealtimeConfigurationError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(RealtimeFallbackModel(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn active_index(&self) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <RealtimeFallbackModelImpl as ::lk_agents_abi::llm::realtime_native::RealtimeFallbackModel>::active_index(&self.0);
        Ok(out)
    }

    fn capabilities(&self) -> PyResult<::lk_agents_abi::llm::realtime::RealtimeCapabilities> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <RealtimeFallbackModelImpl as ::lk_agents_abi::llm::realtime_native::RealtimeFallbackModel>::capabilities(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (*, switch_model = false))]
    fn restart<'py>(&self, py: Python<'py>, switch_model: bool) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <RealtimeFallbackModelImpl as ::lk_agents_abi::llm::realtime_native::RealtimeFallbackModel>::restart(&this, switch_model)
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::Realtime`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::Realtime>(),
        read: |obj| {
            let this = obj.cast::<RealtimeFallbackModel>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::Realtime> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/duplex.rs.

/// [`::lk_agents_abi::llm::duplex::AudioGate`] over a host `livekit.agents.llm.AudioGate`.
#[derive(Clone)]
pub struct AudioGateProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
}

impl AudioGateProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::llm::duplex::AudioGate>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::llm::duplex::AudioGate>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::llm::duplex::AudioGate>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::llm::duplex::AudioGate>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::llm::duplex::AudioGate>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let _ = context;
        Ok(AudioGateProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::llm::duplex::AudioGate>(|obj| AudioGateProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::llm::duplex::AudioGate for AudioGateProxy {
    fn update(&self, frame: ::lk_agents_abi::AudioFrame) -> ::std::result::Result<bool, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let frame_arg = frame;
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(frame_arg, py)?];
                Ok((host.bind(py).getattr("update")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<bool>()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn deactivate(&self) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("deactivate")?, ::pyo3::types::PyTuple::new(py, args)?))
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

// Declared in rust/crates/lk-agents-abi/src/llm/duplex.rs.

/// [`::lk_agents_abi::llm::duplex::DuplexHost`] over a host `livekit.agents._core_view.Duplex`.
#[derive(Clone)]
pub struct DuplexHostProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
}

impl DuplexHostProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::llm::duplex::DuplexHost>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::llm::duplex::DuplexHost>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::llm::duplex::DuplexHost>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::llm::duplex::DuplexHost>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::llm::duplex::DuplexHost>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(DuplexHostProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::llm::duplex::DuplexHost>(|obj| DuplexHostProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::llm::duplex::DuplexHost for DuplexHostProxy {
    fn label(&self) -> ::std::result::Result<::std::string::String, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_label")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::std::string::String>()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn capabilities(&self) -> ::std::result::Result<::lk_agents_abi::llm::duplex::DuplexCapabilities, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_capabilities")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (crate::generated::records::duplex::duplex_capabilities_from_py(&o)).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn gate(&self) -> ::std::result::Result<::std::option::Option<::std::sync::Arc<dyn ::lk_agents_abi::llm::duplex::AudioGate>>, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_gate")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (if o.is_none() { Ok(None) } else { (crate::generated::llm::AudioGateProxy::from_host(&o)).map(Some) }).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn session(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::std::sync::Arc<dyn ::lk_agents_abi::llm::duplex::DuplexSession>, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "DuplexHost._core_session", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_session")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("DuplexHost._core_session: the call captured no host context")));
                };
            let o = out.bind(py);
            let out = (super::llm::DuplexSessionProxy::in_context(o, context)).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/duplex.rs.

/// [`::lk_agents_abi::llm::duplex::DuplexSession`] over a host `livekit.agents._core_view._DuplexSession`.
#[derive(Clone)]
pub struct DuplexSessionProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: crate::host::Notifier,
}

impl DuplexSessionProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::llm::duplex::DuplexSession>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::llm::duplex::DuplexSession>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::llm::duplex::DuplexSession>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::llm::duplex::DuplexSession>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::llm::duplex::DuplexSession>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(DuplexSessionProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, notifier: ::std::default::Default::default() })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::llm::duplex::DuplexSession>(|obj| DuplexSessionProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::llm::duplex::DuplexSession for DuplexSessionProxy {
    fn push_audio(&self, frame: ::lk_agents_abi::AudioFrame) {
        let host = ::std::sync::Arc::clone(&self.host);
        let frame_arg = frame;
        ::lk_agents_runtime::host::notify(&self.notifier, "DuplexSession._core_push_audio", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(frame_arg, py)?];
                Ok((host.bind(py).getattr("_core_push_audio")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn configure(&self, instructions: ::std::option::Option<::std::string::String>, items: ::std::option::Option<::std::vec::Vec<::lk_agents_abi::ChatItem>>, names: ::std::vec::Vec<::std::string::String>) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let instructions_arg = instructions;
        let items_arg = items;
        let names_arg = names;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "DuplexSession._core_configure", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 3] = [::pyo3::IntoPyObjectExt::into_bound_py_any(instructions_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(items_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(names_arg, py)?];
                Ok((host.bind(py).getattr("_core_configure")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
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
        })
    }

    fn update_instructions(&self, instructions: ::std::string::String) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let instructions_arg = instructions;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "DuplexSession._core_update_instructions", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(instructions_arg, py)?];
                Ok((host.bind(py).getattr("_core_update_instructions")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
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
        })
    }

    fn append_items(&self, items: ::std::vec::Vec<::lk_agents_abi::ChatItem>) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let items_arg = items;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "DuplexSession._core_append_items", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(items_arg, py)?];
                Ok((host.bind(py).getattr("_core_append_items")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
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
        })
    }

    fn update_tools(&self, names: ::std::vec::Vec<::std::string::String>) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let names_arg = names;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "DuplexSession._core_update_tools", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(names_arg, py)?];
                Ok((host.bind(py).getattr("_core_update_tools")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
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
        })
    }

    fn generate(&self, instructions: ::std::option::Option<::std::string::String>, tool_choice: ::std::option::Option<::lk_agents_abi::ToolChoice>) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let instructions_arg = instructions;
        let tool_choice_arg = tool_choice;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "DuplexSession._core_generate", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(instructions_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(tool_choice_arg, py)?];
                Ok((host.bind(py).getattr("_core_generate")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
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
        })
    }

    fn next_audio(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::std::option::Option<::lk_agents_abi::llm::duplex::DuplexAudio>, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "DuplexSession._core_next_audio", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_next_audio")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (if o.is_none() { Ok(None) } else { (crate::generated::records::duplex::duplex_audio_from_py(&o)).map(Some) }).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn next_event(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::std::option::Option<::lk_agents_abi::llm::duplex::DuplexEvent>, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "DuplexSession._core_next_event", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_next_event")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (if o.is_none() { Ok(None) } else { (crate::generated::records::duplex::duplex_event_from_py(&o)).map(Some) }).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn close_plugin(&self) -> ::lk_agents_abi::BoxFuture<'_, ()> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "DuplexSession._core_close_plugin", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_close_plugin")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        if let Err(e) = started.await {
            ::lk_agents_abi::python::log::warn!("DuplexSession._core_close_plugin failed: {e}");
        }
        })
    }

    fn aclose(&self) -> ::lk_agents_abi::BoxFuture<'_, ()> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "DuplexSession._core_aclose", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_aclose")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        if let Err(e) = started.await {
            ::lk_agents_abi::python::log::warn!("DuplexSession._core_aclose failed: {e}");
        }
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/fallback.rs.

/// [`::lk_agents_abi::llm::fallback::ChainObserver`] over a host `livekit.agents._core_view.AvailabilityObserver`.
#[derive(Clone)]
pub struct ChainObserverProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: crate::host::Notifier,
}

impl ChainObserverProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::llm::fallback::ChainObserver>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::llm::fallback::ChainObserver>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::llm::fallback::ChainObserver>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::llm::fallback::ChainObserver>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::llm::fallback::ChainObserver>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(ChainObserverProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, notifier: ::std::default::Default::default() })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::llm::fallback::ChainObserver>(|obj| ChainObserverProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::llm::fallback::ChainObserver for ChainObserverProxy {
    fn availability_changed(&self, ev: ::lk_agents_abi::llm::fallback::AvailabilityChanged) {
        let host = ::std::sync::Arc::clone(&self.host);
        let ev_arg = ev;
        ::lk_agents_runtime::host::notify(&self.notifier, "ChainObserver._core_availability_changed", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [crate::generated::records::llm_fallback::availability_changed_to_py(py, ev_arg)?];
                Ok((host.bind(py).getattr("_core_availability_changed")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn all_unavailable(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(&self.notifier, "ChainObserver._core_all_unavailable", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_all_unavailable")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/llm.rs.

/// [`::lk_agents_abi::Llm`] over a host `livekit.agents.llm.LLM`.
#[derive(Clone)]
pub struct LlmProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: crate::host::Notifier,
    label: ::std::string::String,
    model: ::std::string::String,
    id: ::lk_agents_abi::ProviderId,
    /// Appended to every awaited call once bound.
    conn: ::std::option::Option<::lk_agents_abi::ApiConnectOptions>,
}

impl LlmProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::Llm>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::Llm>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::Llm>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::Llm>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::Llm>> {
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
        let model = {
            let v = crate::host::read(obj, "_core_model")?;
            v.extract::<::std::string::String>()?
        };
        let id = {
            let v = crate::host::read(obj, "_core_id")?;
            v.extract::<::lk_agents_abi::ProviderId>()?
        };
        Ok(LlmProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, notifier: ::std::default::Default::default(), label, model, id, conn: None })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::Llm>(|obj| LlmProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::Llm for LlmProxy {
    fn label(&self) -> ::std::string::String {
        ::std::clone::Clone::clone(&self.label)
    }

    fn model(&self) -> ::std::string::String {
        ::std::clone::Clone::clone(&self.model)
    }

    fn id(&self) -> ::lk_agents_abi::ProviderId {
        ::std::clone::Clone::clone(&self.id)
    }

    fn prewarm(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(&self.notifier, "Llm._core_prewarm", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_prewarm")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn with_conn_options(&self, conn: &::lk_agents_abi::ApiConnectOptions) -> ::std::option::Option<::std::sync::Arc<dyn ::lk_agents_abi::Llm>> {
        Some(::std::sync::Arc::new(Self { conn: Some(::std::clone::Clone::clone(conn)), ..::std::clone::Clone::clone(self) }))
    }

    fn failed(&self, error: ::lk_agents_abi::errors::ProviderError, recoverable: bool) {
        let host = ::std::sync::Arc::clone(&self.host);
        let error_arg = error;
        let error_arg = match ::pyo3::Python::attach(|py| -> ::pyo3::PyResult<::pyo3::Py<::pyo3::PyAny>> { Ok((::pyo3::IntoPyObjectExt::into_bound_py_any(error_arg, py)?).unbind()) }) {
            Ok(value) => value,
            Err(error) => {
                ::lk_agents_abi::python::log::warn!("Llm._core_failed failed: {error}");
                return;
            }
        };
        let recoverable_arg = recoverable;
        ::lk_agents_runtime::host::notify(&self.notifier, "Llm._core_failed", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [error_arg.into_bound(py), ::pyo3::IntoPyObjectExt::into_bound_py_any(recoverable_arg, py)?];
                Ok((host.bind(py).getattr("_core_failed")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn chat<'a>(&'a self, req: &::lk_agents_abi::ChatRequest) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::LlmChunk>>, ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let req_arg = ::std::clone::Clone::clone(req);
        let bound = ::std::clone::Clone::clone(&self.conn);
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "Llm._core_chat", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(req_arg, py)?];
                let mut args = args.to_vec();
                if let Some(bound) = bound {
                    args.push(crate::host::construct(&py.import("livekit.agents.types")?.getattr("APIConnectOptions")?, (), Some(&::pyo3::IntoPyObject::into_pyobject(bound, py)?.cast_into::<::pyo3::types::PyDict>()?))?);
                }
                Ok((host.bind(py).getattr("_core_chat")?, ::pyo3::types::PyTuple::new(py, args)?))
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
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("Llm._core_chat: the call captured no host context")));
                };
            let o = out.bind(py);
            let out = (crate::host::stream::HostStream::open(py, o, context, |o| Ok(o.extract::<::lk_agents_abi::LlmChunk>()?), ::lk_agents_runtime::process::host_clock()).map(|s| ::std::boxed::Box::new(s) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::LlmChunk>>)).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/mcp.rs.

/// [`::lk_agents_abi::llm::mcp::McpTransport`] over a host `livekit.agents.llm.mcp._Streams`.
#[derive(Clone)]
pub struct McpTransportProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
}

impl McpTransportProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::llm::mcp::McpTransport>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::llm::mcp::McpTransport>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::llm::mcp::McpTransport>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::llm::mcp::McpTransport>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::llm::mcp::McpTransport>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let _ = context;
        Ok(McpTransportProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::llm::mcp::McpTransport>(|obj| McpTransportProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::llm::mcp::McpTransport for McpTransportProxy {
    fn start<'a>(&'a self, invocation: &::lk_agents_abi::llm::mcp::McpInvocation) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "McpTransport._core_start", invocation, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [crate::generated::records::mcp_session::mcp_invocation_to_py(py, invocation_arg)?];
                Ok((host.bind(py).getattr("_core_start")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
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
        })
    }

    fn send<'a>(&'a self, invocation: &::lk_agents_abi::llm::mcp::McpInvocation, message: &::serde_json::Value) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let message_arg = ::std::clone::Clone::clone(message);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "McpTransport._core_send", invocation, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [crate::generated::records::mcp_session::mcp_invocation_to_py(py, invocation_arg)?, ::lk_agents_abi::python::value_to_py(py, &message_arg)?];
                Ok((host.bind(py).getattr("_core_send")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
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
        })
    }

    fn receive<'a>(&'a self, invocation: &::lk_agents_abi::llm::mcp::McpInvocation) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<::serde_json::Value, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "McpTransport._core_receive", invocation, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [crate::generated::records::mcp_session::mcp_invocation_to_py(py, invocation_arg)?];
                Ok((host.bind(py).getattr("_core_receive")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (::lk_agents_abi::python::value_from_py(&o, "McpTransport._core_receive")).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn close<'a>(&'a self, invocation: &::lk_agents_abi::llm::mcp::McpInvocation) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "McpTransport._core_close", invocation, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [crate::generated::records::mcp_session::mcp_invocation_to_py(py, invocation_arg)?];
                Ok((host.bind(py).getattr("_core_close")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
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
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/mcp.rs.

/// [`::lk_agents_abi::llm::mcp::McpProgress`] over a host `livekit.agents.llm.mcp._Progress`.
#[derive(Clone)]
pub struct McpProgressProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
}

impl McpProgressProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::llm::mcp::McpProgress>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::llm::mcp::McpProgress>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::llm::mcp::McpProgress>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::llm::mcp::McpProgress>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::llm::mcp::McpProgress>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let _ = context;
        Ok(McpProgressProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::llm::mcp::McpProgress>(|obj| McpProgressProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::llm::mcp::McpProgress for McpProgressProxy {
    fn update<'a>(&'a self, invocation: &::lk_agents_abi::llm::mcp::McpInvocation, message: &str) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let message_arg = message.to_owned();
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "McpProgress._core_update", invocation, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [crate::generated::records::mcp_session::mcp_invocation_to_py(py, invocation_arg)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(message_arg, py)?];
                Ok((host.bind(py).getattr("_core_update")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
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
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.

/// [`::lk_agents_abi::llm::realtime::HostRealtime`] over a host `livekit.agents.llm.realtime.RealtimeModel`.
#[derive(Clone)]
pub struct HostRealtimeProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    label: ::std::string::String,
    capabilities: ::lk_agents_abi::llm::realtime::RealtimeCapabilities,
    id: ::lk_agents_abi::ProviderId,
}

impl HostRealtimeProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::llm::realtime::HostRealtime>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::llm::realtime::HostRealtime>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::llm::realtime::HostRealtime>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::llm::realtime::HostRealtime>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::llm::realtime::HostRealtime>> {
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
            v.extract::<::lk_agents_abi::llm::realtime::RealtimeCapabilities>()?
        };
        let id = {
            let v = crate::host::read(obj, "_core_id")?;
            v.extract::<::lk_agents_abi::ProviderId>()?
        };
        Ok(HostRealtimeProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, label, capabilities, id })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::llm::realtime::HostRealtime>(|obj| HostRealtimeProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::llm::realtime::HostRealtime for HostRealtimeProxy {
    fn label(&self) -> ::std::string::String {
        ::std::clone::Clone::clone(&self.label)
    }

    fn capabilities(&self) -> ::lk_agents_abi::llm::realtime::RealtimeCapabilities {
        ::std::clone::Clone::clone(&self.capabilities)
    }

    fn id(&self) -> ::lk_agents_abi::ProviderId {
        ::std::clone::Clone::clone(&self.id)
    }

    fn session(&self, turn_detection_disabled: bool) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::std::sync::Arc<dyn ::lk_agents_abi::llm::realtime::HostRealtimeSession>, ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let turn_detection_disabled_arg = turn_detection_disabled;
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "HostRealtime._core_session", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(turn_detection_disabled_arg, py)?];
                Ok((host.bind(py).getattr("_core_session")?, ::pyo3::types::PyTuple::new(py, args)?))
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
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("HostRealtime._core_session: the call captured no host context")));
                };
            let o = out.bind(py);
            let out = (super::llm::HostRealtimeSessionProxy::in_context(o, context)).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/realtime.rs.

/// [`::lk_agents_abi::llm::realtime::HostRealtimeSession`] over a host `livekit.agents.llm.realtime.RealtimeSession`.
#[derive(Clone)]
pub struct HostRealtimeSessionProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: crate::host::Notifier,
}

impl HostRealtimeSessionProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::llm::realtime::HostRealtimeSession>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::llm::realtime::HostRealtimeSession>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::llm::realtime::HostRealtimeSession>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::llm::realtime::HostRealtimeSession>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::llm::realtime::HostRealtimeSession>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(HostRealtimeSessionProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, notifier: ::std::default::Default::default() })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::llm::realtime::HostRealtimeSession>(|obj| HostRealtimeSessionProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::llm::realtime::HostRealtimeSession for HostRealtimeSessionProxy {
    fn send(&self, cmd: ::lk_agents_abi::RealtimeCommand) {
        let host = ::std::sync::Arc::clone(&self.host);
        let cmd_arg = cmd;
        ::lk_agents_runtime::host::notify(&self.notifier, "HostRealtimeSession._core_send", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(cmd_arg, py)?];
                Ok((host.bind(py).getattr("_core_send")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn update(&self, cmd: ::lk_agents_abi::RealtimeCommand) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let cmd_arg = cmd;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "HostRealtimeSession._core_update", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(cmd_arg, py)?];
                Ok((host.bind(py).getattr("_core_update")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
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
        })
    }

    fn request(&self, cmd: ::lk_agents_abi::RealtimeCommand) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::lk_agents_abi::HostRef, ::lk_agents_abi::llm::realtime::RealtimeError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let cmd_arg = cmd;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "HostRealtimeSession._core_request", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(cmd_arg, py)?];
                Ok((host.bind(py).getattr("_core_request")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::llm::realtime::RealtimeError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::lk_agents_abi::HostRef>()).map_err(|e| <::lk_agents_abi::llm::realtime::RealtimeError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn generation(&self, request: ::lk_agents_abi::HostRef) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::lk_agents_abi::HostRef, ::lk_agents_abi::llm::realtime::RealtimeError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let request_arg = request;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "HostRealtimeSession._core_generation", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(request_arg, py)?];
                Ok((host.bind(py).getattr("_core_generation")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::llm::realtime::RealtimeError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::lk_agents_abi::HostRef>()).map_err(|e| <::lk_agents_abi::llm::realtime::RealtimeError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn cancel(&self, request: ::lk_agents_abi::HostRef) -> ::lk_agents_abi::BoxFuture<'_, ()> {
        let host = ::std::sync::Arc::clone(&self.host);
        let request_arg = request;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "HostRealtimeSession._core_cancel", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(request_arg, py)?];
                Ok((host.bind(py).getattr("_core_cancel")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        if let Err(e) = started.await {
            ::lk_agents_abi::python::log::warn!("HostRealtimeSession._core_cancel failed: {e}");
        }
        })
    }

    fn drain(&self, generation: ::lk_agents_abi::HostRef, req_id: ::std::option::Option<::std::string::String>) -> ::lk_agents_abi::BoxFuture<'_, ()> {
        let host = ::std::sync::Arc::clone(&self.host);
        let generation_arg = generation;
        let req_id_arg = req_id;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "HostRealtimeSession._core_drain", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(generation_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(req_id_arg, py)?];
                Ok((host.bind(py).getattr("_core_drain")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        if let Err(e) = started.await {
            ::lk_agents_abi::python::log::warn!("HostRealtimeSession._core_drain failed: {e}");
        }
        })
    }

    fn next_event(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::std::option::Option<::lk_agents_abi::llm::realtime::Event>, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "HostRealtimeSession._core_next_event", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_next_event")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (if o.is_none() { Ok(None) } else { (o.extract::<::lk_agents_abi::llm::realtime::Event>()).map(Some) }).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn aclose(&self) -> ::lk_agents_abi::BoxFuture<'_, ()> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "HostRealtimeSession._core_aclose", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_aclose")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        if let Err(e) = started.await {
            ::lk_agents_abi::python::log::warn!("HostRealtimeSession._core_aclose failed: {e}");
        }
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/tool_executor.rs.

/// [`::lk_agents_abi::llm::tool_executor::HostToolRunner`] over a host `livekit.agents.voice.tool_executor.ToolRunner`.
#[derive(Clone)]
pub struct HostToolRunnerProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: crate::host::Notifier,
}

impl HostToolRunnerProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::llm::tool_executor::HostToolRunner>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::llm::tool_executor::HostToolRunner>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::llm::tool_executor::HostToolRunner>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::llm::tool_executor::HostToolRunner>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::llm::tool_executor::HostToolRunner>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(HostToolRunnerProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, notifier: ::std::default::Default::default() })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::llm::tool_executor::HostToolRunner>(|obj| HostToolRunnerProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::llm::tool_executor::HostToolRunner for HostToolRunnerProxy {
    fn describe(&self, name: &str) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::std::option::Option<::lk_agents_abi::agent::FunctionToolInfo>, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let name_arg = name.to_owned();
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "HostToolRunner._core_describe", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(name_arg, py)?];
                Ok((host.bind(py).getattr("_core_describe")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (if o.is_none() { Ok(None) } else { (crate::generated::records::agent_run::function_tool_info_from_py(&o)).map(Some) }).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn prepare(&self, call: &::lk_agents_abi::FunctionCall, arguments: ::serde_json::Map<::std::string::String, ::serde_json::Value>, speech: &::lk_agents_abi::CoreObject) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::std::result::Result<::lk_agents_abi::agent::PreparedArguments, ::lk_agents_abi::agent::ToolArgumentError>, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let call_arg = ::std::clone::Clone::clone(call);
        let arguments_arg = arguments;
        let speech_arg = ::std::clone::Clone::clone(speech);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "HostToolRunner._core_prepare", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 3] = [::pyo3::IntoPyObjectExt::into_bound_py_any(call_arg, py)?, ::lk_agents_abi::python::extra_to_py(py, &arguments_arg)?.into_any(), ::pyo3::IntoPyObjectExt::into_bound_py_any(speech_arg, py)?];
                Ok((host.bind(py).getattr("_core_prepare")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    if e.is_instance(py, &py.import("livekit.agents.voice.agent_host").and_then(|m| m.getattr("ToolArgumentError")).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?) {
                        return Ok(Err(<::lk_agents_abi::agent::ToolArgumentError as ::std::convert::From<String>>::from(e.value(py).to_string())));
                    }
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (crate::generated::records::agent_run::prepared_arguments_from_py(&o)).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(Ok(out))
        })
        })
    }

    fn read_back(&self, prepared: &::lk_agents_abi::HostRef) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::lk_agents_abi::agent::ReadBackAnswer, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let prepared_arg = ::std::clone::Clone::clone(prepared);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "HostToolRunner._core_read_back", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(prepared_arg, py)?];
                Ok((host.bind(py).getattr("_core_read_back")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (crate::generated::records::agent_run::read_back_answer_from_py(&o)).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn run(&self, prepared: &::lk_agents_abi::HostRef, call: &::lk_agents_abi::FunctionCall, task: bool) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::lk_agents_abi::llm::tool_executor::ToolReport, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let prepared_arg = ::std::clone::Clone::clone(prepared);
        let call_arg = ::std::clone::Clone::clone(call);
        let task_arg = task;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "HostToolRunner._core_run", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 3] = [::pyo3::IntoPyObjectExt::into_bound_py_any(prepared_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(call_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(task_arg, py)?];
                Ok((host.bind(py).getattr("_core_run")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::lk_agents_abi::llm::tool_executor::ToolReport>()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn cancel(&self, call_id: &str) {
        let host = ::std::sync::Arc::clone(&self.host);
        let call_id_arg = call_id.to_owned();
        ::lk_agents_runtime::host::notify(&self.notifier, "HostToolRunner._core_cancel", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(call_id_arg, py)?];
                Ok((host.bind(py).getattr("_core_cancel")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }
}
