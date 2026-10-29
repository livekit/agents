//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `tts` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/tts/markup/bindings.rs.
type MarkupFunctionsImpl = <crate::Selected as super::Implementations>::MarkupFunctions;

fn _extract_markup_functions_chat_ctx_summarize_plan_keep_last_turns(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "keep_last_turns")
}

#[pyfunction]
#[pyo3(signature = (parts, assistant))]
pub fn chat_message_text_content(parts: ::std::vec::Vec<::std::string::String>, assistant: bool) -> PyResult<::std::option::Option<::std::string::String>> {
    crate::host::check_pid()?;
    let out = <MarkupFunctionsImpl as ::lk_agents_abi::tts::markup::bindings::MarkupFunctions>::chat_message_text_content(parts, assistant);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (provider))]
pub fn supported_nonverbals<'py>(py: Python<'py>, provider: ::std::string::String) -> PyResult<Bound<'py, PyAny>> {
    let out = <MarkupFunctionsImpl as ::lk_agents_abi::tts::markup::bindings::MarkupFunctions>::supported_nonverbals(provider);
    Ok({ let d = ::pyo3::types::PyDict::new(py); for (k, x) in out { d.set_item(::pyo3::IntoPyObjectExt::into_bound_py_any(k.as_str(), py)?, ::pyo3::types::PyList::new(py, x.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?; } d.into_any() })
}

#[pyfunction]
#[pyo3(signature = (text))]
pub fn split_all_markup(text: ::std::string::String) -> PyResult<(::std::string::String, ::std::vec::Vec<::lk_agents_abi::tts::markup::ExpressiveTag>)> {
    let out = <MarkupFunctionsImpl as ::lk_agents_abi::tts::markup::bindings::MarkupFunctions>::split_all_markup(text);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (text))]
pub fn strip_all_markup(text: ::std::string::String) -> PyResult<::std::string::String> {
    let out = <MarkupFunctionsImpl as ::lk_agents_abi::tts::markup::bindings::MarkupFunctions>::strip_all_markup(text);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (text))]
pub fn strip_expr_markup(text: ::std::string::String) -> PyResult<::std::string::String> {
    let out = <MarkupFunctionsImpl as ::lk_agents_abi::tts::markup::bindings::MarkupFunctions>::strip_expr_markup(text);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (text))]
pub fn split_expr_markup(text: ::std::string::String) -> PyResult<(::std::string::String, ::std::vec::Vec<::lk_agents_abi::tts::markup::ExpressiveTag>)> {
    let out = <MarkupFunctionsImpl as ::lk_agents_abi::tts::markup::bindings::MarkupFunctions>::split_expr_markup(text);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (tag_name, content = None, attrs = None))]
pub fn to_xml(tag_name: ::std::string::String, content: ::std::option::Option<::std::string::String>, attrs: ::std::option::Option<::std::vec::Vec<(::std::string::String, ::std::string::String)>>) -> PyResult<::std::string::String> {
    let out = <MarkupFunctionsImpl as ::lk_agents_abi::tts::markup::bindings::MarkupFunctions>::to_xml(tag_name, content, attrs);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (summary))]
pub fn chat_history_summary_xml(summary: ::std::string::String) -> PyResult<::std::string::String> {
    let out = <MarkupFunctionsImpl as ::lk_agents_abi::tts::markup::bindings::MarkupFunctions>::chat_history_summary_xml(summary);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (items, keep_last_turns))]
pub fn chat_ctx_summarize_plan(items: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_markup_functions_chat_ctx_summarize_plan_keep_last_turns)] keep_last_turns: usize) -> PyResult<::std::option::Option<(usize, ::std::vec::Vec<usize>, ::std::string::String, ::std::string::String, f64)>> {
    let items = items.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::chat_items::chat_item_from_py(&item) }).collect::<PyResult<Vec<_>>>())?;
    let out = <MarkupFunctionsImpl as ::lk_agents_abi::tts::markup::bindings::MarkupFunctions>::chat_ctx_summarize_plan(items, keep_last_turns);
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/tts/markup/bindings.rs.
type MarkupImpl = <crate::Selected as super::Implementations>::MarkupClass;

/// The markup dialect a TTS speaks, for the expressive pipeline: what the LLM is taught to write
/// and how it is normalized and lowered before synthesis.
///
/// Plugins opt in by overriding `_provider_key`.
#[pyclass(frozen, name = "Markup", module = "livekit.agents._core")]
pub struct Markup(pub(crate) ::std::sync::Arc<MarkupImpl>);

#[pymethods]
impl Markup {
    #[new]
    #[pyo3(signature = (provider_key))]
    fn new(provider_key: ::std::string::String) -> PyResult<Markup> {
        let out = <MarkupImpl as ::lk_agents_abi::tts::markup::bindings::MarkupClass>::from_provider_key(provider_key);
        Ok(Markup(::std::sync::Arc::new(out)))
    }

    /// The provider's key into the markup tables, or `""` for no markup.
    #[pyo3(name = "_provider_key")]
    fn provider_key(&self) -> PyResult<::std::string::String> {
        let out = <MarkupImpl as ::lk_agents_abi::tts::markup::bindings::MarkupClass>::provider_key(&self.0);
        Ok(out)
    }

    /// The instructions teaching the LLM this voice's markup, `None` without markup.
    #[pyo3(signature = (*, speech_steering = None))]
    fn llm_instructions(&self, speech_steering: ::std::option::Option<::lk_agents_abi::tts::markup::SpeechSteering>) -> PyResult<::std::option::Option<::std::string::String>> {
        let out = <MarkupImpl as ::lk_agents_abi::tts::markup::bindings::MarkupClass>::llm_instructions(&self.0, speech_steering);
        Ok(out)
    }

    /// Fix common LLM markup mistakes (e.g. unclosed self-closing tags).
    #[pyo3(signature = (text))]
    fn normalize(&self, text: ::std::string::String) -> PyResult<::std::string::String> {
        let out = <MarkupImpl as ::lk_agents_abi::tts::markup::bindings::MarkupClass>::normalize(&self.0, text);
        Ok(out)
    }

    /// Lower the framework's markup to the provider's native syntax.
    #[pyo3(signature = (text))]
    fn convert(&self, text: ::std::string::String) -> PyResult<::std::string::String> {
        let out = <MarkupImpl as ::lk_agents_abi::tts::markup::bindings::MarkupClass>::convert(&self.0, text);
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/tts/markup/bindings.rs.
type TranscriptMarkupStripperImpl = <crate::Selected as super::Implementations>::TranscriptStripper;

/// Stateful, provider-agnostic markup stripper for one transcript segment.
#[pyclass(name = "TranscriptMarkupStripper", module = "livekit.agents._core")]
pub struct TranscriptMarkupStripper(pub(crate) TranscriptMarkupStripperImpl);

#[pymethods]
impl TranscriptMarkupStripper {
    #[new]
    fn new() -> PyResult<TranscriptMarkupStripper> {
        let out = <TranscriptMarkupStripperImpl as ::lk_agents_abi::tts::markup::bindings::TranscriptStripper>::new();
        Ok(TranscriptMarkupStripper(out))
    }

    #[pyo3(signature = (text))]
    fn push(&mut self, text: ::std::string::String) -> PyResult<::std::string::String> {
        let out = <TranscriptMarkupStripperImpl as ::lk_agents_abi::tts::markup::bindings::TranscriptStripper>::push(&mut self.0, text);
        Ok(out)
    }

    fn flush(&mut self) -> PyResult<::std::string::String> {
        let out = <TranscriptMarkupStripperImpl as ::lk_agents_abi::tts::markup::bindings::TranscriptStripper>::flush(&mut self.0);
        Ok(out)
    }

    #[getter]
    fn tags(&self) -> PyResult<::std::vec::Vec<::lk_agents_abi::tts::markup::ExpressiveTag>> {
        let out = <TranscriptMarkupStripperImpl as ::lk_agents_abi::tts::markup::bindings::TranscriptStripper>::tags(&self.0);
        Ok(out)
    }

    fn expression_attribute(&self) -> PyResult<::std::option::Option<::std::collections::BTreeMap<::std::string::String, ::std::string::String>>> {
        let out = <TranscriptMarkupStripperImpl as ::lk_agents_abi::tts::markup::bindings::TranscriptStripper>::expression_attribute(&self.0);
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/tts/stream_adapter.rs.
type StreamAdapterFunctionsImpl = <crate::Selected as super::Implementations>::StreamAdapterFunctions;

#[pyfunction]
#[pyo3(signature = (provider, conn_options))]
pub fn tts_adapter_stream<'py>(py: Python<'py>, provider: Bound<'_, PyAny>, conn_options: ::lk_agents_abi::ApiConnectOptions) -> PyResult<Bound<'py, PyAny>> {
    crate::host::check_pid()?;
    let env = crate::process::env();
    let provider = crate::generated::tts::TtsProxy::from_host(&provider)?;
    crate::host::future_into_py(py, async move {
        let out = <StreamAdapterFunctionsImpl as ::lk_agents_abi::StreamAdapterFunctions>::tts_adapter_stream(&env, provider, conn_options).await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(super::tts::TtsStream(::std::sync::Arc::new(out)), py))
    })
}

// Declared in rust/crates/lk-agents-abi/src/tts/stream_pacer.rs.
type SentencePacerImpl = <crate::Selected as super::Implementations>::SentencePacer;

fn _extract_sentence_pacer_new_minimum_audio(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "minimum_audio")
}

fn _extract_sentence_pacer_new_maximum_text(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "maximum_text")
}

fn _extract_sentence_pacer_poll_now(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "now")
}

fn _extract_sentence_pacer_poll_audio(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "audio")
}

/// Paces a synthesis's sentences: the first at once, the rest in batches once the audio already
/// synthesized is nearly played.
#[pyclass(name = "SentencePacer", module = "livekit.agents._core")]
pub struct SentencePacer(pub(crate) SentencePacerImpl);

#[pymethods]
impl SentencePacer {
    #[new]
    #[pyo3(signature = (minimum_audio, maximum_text))]
    fn new(#[pyo3(from_py_with = _extract_sentence_pacer_new_minimum_audio)] minimum_audio: f64, #[pyo3(from_py_with = _extract_sentence_pacer_new_maximum_text)] maximum_text: usize) -> PyResult<SentencePacer> {
        crate::host::check_pid()?;
        let out = <SentencePacerImpl as ::lk_agents_abi::tts::stream_pacer::SentencePacer>::new(minimum_audio, maximum_text);
        Ok(SentencePacer(out))
    }

    #[pyo3(signature = (text))]
    fn push(&mut self, text: ::std::string::String) -> PyResult<()> {
        crate::host::check_pid()?;
        <SentencePacerImpl as ::lk_agents_abi::tts::stream_pacer::SentencePacer>::push(&mut self.0, text);
        Ok(())
    }

    fn end(&mut self) -> PyResult<()> {
        crate::host::check_pid()?;
        <SentencePacerImpl as ::lk_agents_abi::tts::stream_pacer::SentencePacer>::end(&mut self.0);
        Ok(())
    }

    /// Text to emit, remaining audio for diagnostics, and next wakeup (None means done).
    #[pyo3(signature = (now, audio))]
    fn poll(&mut self, #[pyo3(from_py_with = _extract_sentence_pacer_poll_now)] now: f64, #[pyo3(from_py_with = _extract_sentence_pacer_poll_audio)] audio: f64) -> PyResult<(::std::option::Option<::std::string::String>, f64, ::std::option::Option<f64>)> {
        crate::host::check_pid()?;
        let out = <SentencePacerImpl as ::lk_agents_abi::tts::stream_pacer::SentencePacer>::poll(&mut self.0, now, audio);
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/tts/streams.rs.
type TtsStreamImpl = <crate::Selected as super::Implementations>::TtsStream;

/// One synthesis, `SynthesizeStream`'s shape: text in through `push_text`, `flush` and
/// `end_input`, audio out through `recv`. `aclose` cancels it; it does not end the input.
#[pyclass(frozen, name = "TtsStream", module = "livekit.agents._core")]
pub struct TtsStream(pub(crate) ::std::sync::Arc<TtsStreamImpl>);

#[pymethods]
impl TtsStream {
    #[pyo3(signature = (text))]
    fn push_text(&self, text: ::std::string::String) -> PyResult<()> {
        crate::host::check_pid()?;
        <TtsStreamImpl as ::lk_agents_abi::tts::streams::TtsStream>::push_text(&self.0, text).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn flush(&self) -> PyResult<()> {
        crate::host::check_pid()?;
        <TtsStreamImpl as ::lk_agents_abi::tts::streams::TtsStream>::flush(&self.0).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn end_input(&self) -> PyResult<()> {
        crate::host::check_pid()?;
        <TtsStreamImpl as ::lk_agents_abi::tts::streams::TtsStream>::end_input(&self.0).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <TtsStreamImpl as ::lk_agents_abi::tts::streams::TtsStream>::recv(&this, )
                .await.map_err(|e: ::lk_agents_abi::errors::PullError| <::lk_agents_abi::errors::PullError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <TtsStreamImpl as ::lk_agents_abi::tts::streams::TtsStream>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/tts/streams.rs.
type SynthesisChecksImpl = <crate::Selected as super::Implementations>::SynthesisChecks;

fn _extract_synthesis_checks_validate_synthesis_duration(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "duration")
}

fn _extract_synthesis_checks_validate_synthesis_segments(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<(usize, usize)>> {
    if obj.is_none() { Ok(None) } else { ((|| -> ::pyo3::PyResult<_> { let tuple = obj.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "segments"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((::lk_agents_abi::python::number_from_py::<usize>(&item_0, "segments"))?, (::lk_agents_abi::python::number_from_py::<usize>(&item_1, "segments"))?,)) })()).map(Some) }
}

/// Why a synthesis of ``text`` that pushed ``duration`` seconds of audio (and, streamed,
/// ``segments`` as ``(expected, actual)``) failed, or ``None``.
#[pyfunction]
#[pyo3(signature = (text, duration, segments))]
pub fn validate_synthesis(text: ::std::string::String, #[pyo3(from_py_with = _extract_synthesis_checks_validate_synthesis_duration)] duration: f64, #[pyo3(from_py_with = _extract_synthesis_checks_validate_synthesis_segments)] segments: ::std::option::Option<(usize, usize)>) -> PyResult<::std::option::Option<::std::string::String>> {
    let out = <SynthesisChecksImpl as ::lk_agents_abi::tts::streams::SynthesisChecks>::validate_synthesis(&text, duration, segments);
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/tts/tts.rs.

/// [`::lk_agents_abi::Tts`] over a host `livekit.agents.tts.TTS`.
#[derive(Clone)]
pub struct TtsProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: crate::host::Notifier,
    instance_id: ::std::option::Option<usize>,
    label: ::std::string::String,
    capabilities: ::lk_agents_abi::TtsCapabilities,
    id: ::lk_agents_abi::ProviderId,
    sample_rate: u32,
    num_channels: u16,
    markup_provider: ::std::string::String,
    lowers_markup: bool,
    conn_options: ::std::option::Option<::lk_agents_abi::ApiConnectOptions>,
    /// Appended to every awaited call once bound.
    conn: ::std::option::Option<::lk_agents_abi::ApiConnectOptions>,
}

impl TtsProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::Tts>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::Tts>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::Tts>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::Tts>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::Tts>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        let instance_id = {
            let v = crate::host::read(obj, "_core_instance_id")?;
            if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<usize>(&v, "Tts._core_instance_id")).map(Some) }?
        };
        let label = {
            let v = crate::host::read(obj, "_core_label")?;
            v.extract::<::std::string::String>()?
        };
        let capabilities = {
            let v = crate::host::read(obj, "_core_capabilities")?;
            v.extract::<::lk_agents_abi::TtsCapabilities>()?
        };
        let id = {
            let v = crate::host::read(obj, "_core_id")?;
            v.extract::<::lk_agents_abi::ProviderId>()?
        };
        let sample_rate = {
            let v = crate::host::read(obj, "_core_sample_rate")?;
            ::lk_agents_abi::python::number_from_py::<u32>(&v, "Tts._core_sample_rate")?
        };
        let num_channels = {
            let v = crate::host::read(obj, "_core_num_channels")?;
            ::lk_agents_abi::python::number_from_py::<u16>(&v, "Tts._core_num_channels")?
        };
        let markup_provider = {
            let v = crate::host::read(obj, "_core_markup_provider")?;
            v.extract::<::std::string::String>()?
        };
        let lowers_markup = {
            let v = crate::host::read(obj, "_core_lowers_markup")?;
            v.extract::<bool>()?
        };
        let conn_options = {
            let v = crate::host::read(obj, "_core_conn_options")?;
            if v.is_none() { Ok(None) } else { (v.extract::<::lk_agents_abi::ApiConnectOptions>()).map(Some) }?
        };
        Ok(TtsProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, notifier: ::std::default::Default::default(), instance_id, label, capabilities, id, sample_rate, num_channels, markup_provider, lowers_markup, conn_options, conn: None })
    }
}

impl ::lk_agents_abi::Tts for TtsProxy {
    fn instance_id(&self) -> ::std::option::Option<usize> {
        ::std::clone::Clone::clone(&self.instance_id)
    }

    fn label(&self) -> ::std::string::String {
        ::std::clone::Clone::clone(&self.label)
    }

    fn capabilities(&self) -> ::lk_agents_abi::TtsCapabilities {
        ::std::clone::Clone::clone(&self.capabilities)
    }

    fn id(&self) -> ::lk_agents_abi::ProviderId {
        ::std::clone::Clone::clone(&self.id)
    }

    fn prewarm(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(&self.notifier, "Tts._core_prewarm", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_prewarm")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn sample_rate(&self) -> u32 {
        ::std::clone::Clone::clone(&self.sample_rate)
    }

    fn num_channels(&self) -> u16 {
        ::std::clone::Clone::clone(&self.num_channels)
    }

    fn markup_provider(&self) -> ::std::string::String {
        ::std::clone::Clone::clone(&self.markup_provider)
    }

    fn lowers_markup(&self) -> bool {
        ::std::clone::Clone::clone(&self.lowers_markup)
    }

    fn with_conn_options(&self, conn: &::lk_agents_abi::ApiConnectOptions) -> ::std::option::Option<::std::sync::Arc<dyn ::lk_agents_abi::Tts>> {
        Some(::std::sync::Arc::new(Self { conn: Some(::std::clone::Clone::clone(conn)), ..::std::clone::Clone::clone(self) }))
    }

    fn conn_options(&self) -> ::std::option::Option<::lk_agents_abi::ApiConnectOptions> {
        ::std::clone::Clone::clone(&self.conn_options)
    }

    fn failed(&self, error: ::lk_agents_abi::errors::ProviderError, recoverable: bool) {
        let host = ::std::sync::Arc::clone(&self.host);
        let error_arg = error;
        let error_arg = match ::pyo3::Python::attach(|py| -> ::pyo3::PyResult<::pyo3::Py<::pyo3::PyAny>> { Ok((::pyo3::IntoPyObjectExt::into_bound_py_any(error_arg, py)?).unbind()) }) {
            Ok(value) => value,
            Err(error) => {
                ::lk_agents_abi::python::log::warn!("Tts._core_failed failed: {error}");
                return;
            }
        };
        let recoverable_arg = recoverable;
        ::lk_agents_runtime::host::notify(&self.notifier, "Tts._core_failed", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [error_arg.into_bound(py), ::pyo3::IntoPyObjectExt::into_bound_py_any(recoverable_arg, py)?];
                Ok((host.bind(py).getattr("_core_failed")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn synthesize<'a>(&'a self, text: &str) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::SynthesizedAudio>>, ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = text.to_owned();
        let bound = ::std::clone::Clone::clone(&self.conn);
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "Tts._core_synthesize", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(text_arg, py)?];
                let mut args = args.to_vec();
                if let Some(bound) = bound {
                    args.push(crate::host::construct(&py.import("livekit.agents.types")?.getattr("APIConnectOptions")?, (), Some(&::pyo3::IntoPyObject::into_pyobject(bound, py)?.cast_into::<::pyo3::types::PyDict>()?))?);
                }
                Ok((host.bind(py).getattr("_core_synthesize")?, ::pyo3::types::PyTuple::new(py, args)?))
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
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("Tts._core_synthesize: the call captured no host context")));
                };
            let o = out.bind(py);
            let out = (crate::host::stream::HostStream::open(py, o, context, |o| Ok(o.extract::<::lk_agents_abi::SynthesizedAudio>()?), ::lk_agents_runtime::process::host_clock()).map(|s| ::std::boxed::Box::new(s) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::SynthesizedAudio>>)).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn stream(&self, expressive: bool) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(::std::boxed::Box<dyn ::lk_agents_abi::StreamSink<::std::string::String>>, ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::SynthesizedAudio>>), ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let expressive_arg = expressive;
        let bound = ::std::clone::Clone::clone(&self.conn);
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "Tts._core_stream", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(expressive_arg, py)?];
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
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("Tts._core_stream: the call captured no host context")));
                };
            let o = out.bind(py);
            let out = (crate::host::stream::HostStream::duplex(py, o, context, |py, v| Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(v, py)?), |o| Ok(o.extract::<::lk_agents_abi::SynthesizedAudio>()?), ::lk_agents_runtime::process::host_clock()).map(|(sink, source)| (::std::boxed::Box::new(sink) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSink<::std::string::String>>, ::std::boxed::Box::new(source) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::SynthesizedAudio>>))).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }
}
