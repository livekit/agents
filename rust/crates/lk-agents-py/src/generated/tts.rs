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
    ::lk_agents_abi::python::check_pid()?;
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
#[pyo3(signature = (provider))]
pub fn max_input_len(provider: ::std::string::String) -> PyResult<::std::option::Option<usize>> {
    let out = <MarkupFunctionsImpl as ::lk_agents_abi::tts::markup::bindings::MarkupFunctions>::max_input_len(provider);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (provider, *, expressive))]
pub fn sentence_tokenizer_config(provider: ::std::string::String, expressive: bool) -> PyResult<(::std::option::Option<usize>, ::std::option::Option<usize>, bool)> {
    let out = <MarkupFunctionsImpl as ::lk_agents_abi::tts::markup::bindings::MarkupFunctions>::sentence_tokenizer_config(provider, expressive);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (provider, steering))]
pub fn steering_instructions(provider: ::std::string::String, steering: ::lk_agents_abi::tts::markup::SpeechSteering) -> PyResult<::std::string::String> {
    let out = <MarkupFunctionsImpl as ::lk_agents_abi::tts::markup::bindings::MarkupFunctions>::steering_instructions(provider, steering);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (text))]
pub fn convert_expression_tags(text: ::std::string::String) -> PyResult<::std::string::String> {
    let out = <MarkupFunctionsImpl as ::lk_agents_abi::tts::markup::bindings::MarkupFunctions>::convert_expression_tags(text);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (text, xml_tags))]
pub fn extract_and_strip(text: ::std::string::String, xml_tags: ::std::vec::Vec<::std::string::String>) -> PyResult<(::std::string::String, ::std::vec::Vec<(::std::string::String, ::std::string::String)>)> {
    let out = <MarkupFunctionsImpl as ::lk_agents_abi::tts::markup::bindings::MarkupFunctions>::extract_and_strip(text, xml_tags);
    Ok(out)
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
#[pyo3(signature = (tags))]
pub fn expression_attribute(tags: ::std::vec::Vec<::lk_agents_abi::tts::markup::ExpressiveTag>) -> PyResult<::std::option::Option<::std::collections::BTreeMap<::std::string::String, ::std::string::String>>> {
    let out = <MarkupFunctionsImpl as ::lk_agents_abi::tts::markup::bindings::MarkupFunctions>::expression_attribute(tags);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (texts, end_of_stream))]
pub fn drop_bracket_cues(texts: ::std::vec::Vec<::std::string::String>, end_of_stream: bool) -> PyResult<(::std::vec::Vec<(usize, ::std::option::Option<::std::string::String>)>, ::std::vec::Vec<(usize, ::std::option::Option<::std::string::String>)>)> {
    let out = <MarkupFunctionsImpl as ::lk_agents_abi::tts::markup::bindings::MarkupFunctions>::drop_bracket_cues(texts, end_of_stream);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (label, fallback = None))]
pub fn match_mood(label: ::std::string::String, fallback: ::std::option::Option<::lk_agents_abi::tts::markup::Mood>) -> PyResult<::std::option::Option<::lk_agents_abi::tts::markup::Mood>> {
    let out = <MarkupFunctionsImpl as ::lk_agents_abi::tts::markup::bindings::MarkupFunctions>::match_mood(label, fallback);
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
pub fn chat_ctx_summarize_plan(items: ::std::vec::Vec<::lk_agents_abi::ChatItem>, #[pyo3(from_py_with = _extract_markup_functions_chat_ctx_summarize_plan_keep_last_turns)] keep_last_turns: usize) -> PyResult<::std::option::Option<(usize, ::std::vec::Vec<usize>, ::std::string::String, ::std::string::String, f64)>> {
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
pub struct Markup(::std::sync::Arc<MarkupImpl>);

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
pub struct TranscriptMarkupStripper(TranscriptMarkupStripperImpl);

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
    ::lk_agents_abi::python::check_pid()?;
    let provider = crate::generated::tts::TtsProxy::from_host(&provider)?;
    crate::host::future_into_py(py, async move {
        let out = <StreamAdapterFunctionsImpl as ::lk_agents_abi::StreamAdapterFunctions>::tts_adapter_stream(provider, conn_options).await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(super::tts::FallbackTtsStream(::std::sync::Arc::new(out)), py))
    })
}

// Declared in rust/crates/lk-agents-abi/src/tts/streams.rs.
type InferenceTtsStreamImpl = <crate::Selected as super::Implementations>::InferenceTtsStream;

/// One live synthesis.
#[pyclass(frozen, name = "InferenceTtsStream", module = "livekit.agents._core")]
pub struct InferenceTtsStream(pub(crate) ::std::sync::Arc<InferenceTtsStreamImpl>);

#[pymethods]
impl InferenceTtsStream {
    #[pyo3(signature = (text))]
    fn push(&self, text: ::std::string::String) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <InferenceTtsStreamImpl as ::lk_agents_abi::tts::streams::InferenceTtsStream>::push(&self.0, text).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn flush(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <InferenceTtsStreamImpl as ::lk_agents_abi::tts::streams::InferenceTtsStream>::flush(&self.0).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn end_input(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <InferenceTtsStreamImpl as ::lk_agents_abi::tts::streams::InferenceTtsStream>::end_input(&self.0).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <InferenceTtsStreamImpl as ::lk_agents_abi::tts::streams::InferenceTtsStream>::recv(&this, )
                .await.map_err(|e: ::lk_agents_abi::errors::PullError| <::lk_agents_abi::errors::PullError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <InferenceTtsStreamImpl as ::lk_agents_abi::tts::streams::InferenceTtsStream>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/tts/streams.rs.
type FallbackTtsStreamImpl = <crate::Selected as super::Implementations>::FallbackTtsStream;

/// One synthesis, served by whichever entry the chain picks.
#[pyclass(frozen, name = "FallbackTtsStream", module = "livekit.agents._core")]
pub struct FallbackTtsStream(pub(crate) ::std::sync::Arc<FallbackTtsStreamImpl>);

#[pymethods]
impl FallbackTtsStream {
    #[pyo3(signature = (token))]
    fn push_text(&self, token: ::std::string::String) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <FallbackTtsStreamImpl as ::lk_agents_abi::tts::streams::FallbackTtsStream>::push_text(&self.0, token).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn flush(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <FallbackTtsStreamImpl as ::lk_agents_abi::tts::streams::FallbackTtsStream>::flush(&self.0).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn end_input(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <FallbackTtsStreamImpl as ::lk_agents_abi::tts::streams::FallbackTtsStream>::end_input(&self.0).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <FallbackTtsStreamImpl as ::lk_agents_abi::tts::streams::FallbackTtsStream>::recv(&this, )
                .await.map_err(|e: ::lk_agents_abi::errors::PullError| <::lk_agents_abi::errors::PullError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <FallbackTtsStreamImpl as ::lk_agents_abi::tts::streams::FallbackTtsStream>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
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
        Ok(TtsProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, notifier: ::std::default::Default::default(), instance_id, label, capabilities, id, sample_rate, num_channels, markup_provider, lowers_markup, conn: None })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::Tts>(|obj| TtsProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
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

    fn set_expressive(&self, enabled: bool) {
        let host = ::std::sync::Arc::clone(&self.host);
        let enabled_arg = enabled;
        ::lk_agents_runtime::host::notify(&self.notifier, "Tts._core_set_expressive", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(enabled_arg, py)?];
                Ok((host.bind(py).getattr("_core_set_expressive")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn with_conn_options(&self, conn: &::lk_agents_abi::ApiConnectOptions) -> ::std::option::Option<::std::sync::Arc<dyn ::lk_agents_abi::Tts>> {
        Some(::std::sync::Arc::new(Self { conn: Some(::std::clone::Clone::clone(conn)), ..::std::clone::Clone::clone(self) }))
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

    fn stream(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(::std::boxed::Box<dyn ::lk_agents_abi::StreamSink<::std::string::String>>, ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::SynthesizedAudio>>), ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let bound = ::std::clone::Clone::clone(&self.conn);
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "Tts._core_stream", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
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
