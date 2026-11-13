//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `tts` classes.

#![allow(clippy::too_many_arguments, reason = "a host signature takes the parameters it declares")]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/tts/markup.rs.
type MarkupFunctionsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::MarkupFunctions;
type MarkupFunctionsChatMessageTextContentOutput = ::std::option::Option<::std::string::String>;

#[pyfunction]
#[pyo3(signature = (parts, assistant))]
pub fn chat_message_text_content(
    py: Python<'_>,
    parts: ::std::vec::Vec<::std::string::String>,
    assistant: bool,
) -> PyResult<MarkupFunctionsChatMessageTextContentOutput> {
    crate::py_calls::check_pid()?;
    let out = py.detach(|| <MarkupFunctionsImpl as ::lk_agents_abi::tts::markup::MarkupFunctions>::chat_message_text_content(parts, assistant));
    ::pyo3::PyResult::Ok(out)
}

#[pyfunction]
#[pyo3(signature = (provider))]
pub fn supported_nonverbals<'py>(
    py: Python<'py>,
    provider: ::std::string::String,
) -> PyResult<Bound<'py, PyAny>> {
    let out = py.detach(|| <MarkupFunctionsImpl as ::lk_agents_abi::tts::markup::MarkupFunctions>::supported_nonverbals(provider));
    crate::generated::records::tts::markup_info_to_py(py, out)
}

#[pyfunction]
#[pyo3(signature = (text))]
pub fn strip_expr_markup(
    py: Python<'_>,
    text: ::std::string::String,
) -> PyResult<::std::string::String> {
    let out = py.detach(|| {
        <MarkupFunctionsImpl as ::lk_agents_abi::tts::markup::MarkupFunctions>::strip_expr_markup(
            text,
        )
    });
    ::pyo3::PyResult::Ok(out)
}

/// Strip only the `<expr/>` dialect, returning the clean text and its markers; the
/// provider-native tags stay in the words.
#[pyfunction]
#[pyo3(signature = (text))]
pub fn split_expr_markup<'py>(
    py: Python<'py>,
    text: ::std::string::String,
) -> PyResult<Bound<'py, PyAny>> {
    let out = py.detach(|| {
        <MarkupFunctionsImpl as ::lk_agents_abi::tts::markup::MarkupFunctions>::split_expr_markup(
            text,
        )
    });
    crate::generated::records::tts::markup_split_to_py(py, out)
}

#[pyfunction]
#[pyo3(signature = (tag_name, content = None, attrs = None))]
pub fn to_xml(
    py: Python<'_>,
    tag_name: ::std::string::String,
    content: ::std::option::Option<::std::string::String>,
    attrs: ::std::option::Option<::std::vec::Vec<(::std::string::String, ::std::string::String)>>,
) -> PyResult<::std::string::String> {
    let out = py.detach(|| {
        <MarkupFunctionsImpl as ::lk_agents_abi::tts::markup::MarkupFunctions>::to_xml(
            tag_name, content, attrs,
        )
    });
    ::pyo3::PyResult::Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/tts/markup.rs.
type MarkupImpl =
    <crate::generated::implementations::Selected as super::Implementations>::MarkupClass;
type MarkupClassLlmInstructionsOutput = ::std::option::Option<::std::string::String>;

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
    fn new(py: Python<'_>, provider_key: ::std::string::String) -> PyResult<Markup> {
        let out = py.detach(|| {
            <MarkupImpl as ::lk_agents_abi::tts::markup::MarkupClass>::from_provider_key(
                provider_key,
            )
        });
        ::pyo3::PyResult::Ok(Markup(::std::sync::Arc::new(out)))
    }

    /// The provider's key into the markup tables, or `""` for no markup.
    #[pyo3(name = "_provider_key")]
    fn provider_key(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        let out = py.detach(|| {
            <MarkupImpl as ::lk_agents_abi::tts::markup::MarkupClass>::provider_key(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// The instructions teaching the LLM this voice's markup, `None` without markup.
    #[pyo3(signature = (*, speech_steering = None))]
    fn llm_instructions(
        &self,
        py: Python<'_>,
        speech_steering: Option<Bound<'_, PyAny>>,
    ) -> PyResult<MarkupClassLlmInstructionsOutput> {
        let speech_steering = speech_steering
            .map(|v| {
                crate::generated::records::expressive::speech_steering_options_from_py(v.as_any())
            })
            .transpose()?;
        let out = py.detach(|| {
            <MarkupImpl as ::lk_agents_abi::tts::markup::MarkupClass>::llm_instructions(
                &self.0,
                speech_steering,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/tts/stream_adapter.rs.
type TtsStreamAdapterImpl =
    <crate::generated::implementations::Selected as super::Implementations>::TtsStreamAdapter;

/// `tts.StreamAdapter`'s engine: a non-streaming TTS as a streaming one, each chunk of its input
/// synthesized on its own.
#[pyclass(frozen, weakref, subclass, name = "TtsStreamAdapter", module = "livekit.agents._core")]
pub struct TtsStreamAdapter(
    pub(crate) ::std::sync::OnceLock<::std::sync::Arc<TtsStreamAdapterImpl>>,
);

#[pymethods]
impl TtsStreamAdapter {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(
        _args: &Bound<'_, ::pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, ::pyo3::types::PyDict>>,
    ) -> Self {
        Self(::std::sync::OnceLock::new())
    }

    /// `tts` synthesizes each chunk of the text pushed.
    #[pyo3(signature = (*, tts))]
    fn __init__(slf: &Bound<'_, Self>, py: Python<'_>, tts: Bound<'_, PyAny>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let once =
            || ::pyo3::exceptions::PyRuntimeError::new_err("TtsStreamAdapter.__init__ runs once");
        if slf.get().0.get().is_some() {
            return Err(once());
        }
        let env = crate::process::env();
        let label = crate::py_objects::class_label(slf.as_any())?;
        let events = crate::generated::types::ProviderEventsProxy::weak(slf.as_any())?;
        let tts = crate::generated::tts::TtsProxy::from_host(tts.as_any())?;
        let out = py.detach(|| {
            <TtsStreamAdapterImpl as ::lk_agents_abi::tts::stream_adapter::TtsStreamAdapter>::new(
                &env, label, events, tts,
            )
        });
        slf.get().0.set(::std::sync::Arc::new(out)).map_err(|_| once())
    }

    /// Streaming, with word alignment.
    #[getter]
    fn capabilities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let engine = self.engine()?;
        let out = py.detach(|| <TtsStreamAdapterImpl as ::lk_agents_abi::tts::stream_adapter::TtsStreamAdapter>::capabilities(engine));
        crate::generated::records::tts::tts_capabilities_to_py(py, out)
    }

    #[getter]
    fn sample_rate(&self, py: Python<'_>) -> PyResult<u32> {
        crate::py_calls::check_pid()?;
        let engine = self.engine()?;
        let out = py.detach(|| <TtsStreamAdapterImpl as ::lk_agents_abi::tts::stream_adapter::TtsStreamAdapter>::sample_rate(engine)).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn num_channels(&self, py: Python<'_>) -> PyResult<u16> {
        crate::py_calls::check_pid()?;
        let engine = self.engine()?;
        let out = py.detach(|| <TtsStreamAdapterImpl as ::lk_agents_abi::tts::stream_adapter::TtsStreamAdapter>::num_channels(engine));
        ::pyo3::PyResult::Ok(out)
    }
}

impl TtsStreamAdapter {
    /// The implementation its `__init__` built.
    pub(crate) fn engine(&self) -> PyResult<&::std::sync::Arc<TtsStreamAdapterImpl>> {
        self.0.get().ok_or_else(|| crate::py_objects::unbuilt("TtsStreamAdapter"))
    }
}

// Wherever the host passes a `::lk_agents_abi::Tts`, an instance of this class is itself once built.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::Tts>(),
        read: |obj| match obj.cast::<TtsStreamAdapter>() {
            Ok(this) => {
                let engine = this.get().0.get().cloned().map(|engine| -> ::std::sync::Arc<dyn ::lk_agents_abi::Tts> { engine });
                crate::py_objects::built(obj, engine, "TtsStreamAdapter")
            }
            Err(_) => Ok(crate::py_objects::Taken::Other),
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/tts/streams.rs.
type TtsStreamImpl =
    <crate::generated::implementations::Selected as super::Implementations>::TtsStream;

/// One synthesis, `SynthesizeStream`'s shape: text in through `push_text`, `flush` and
/// `end_input`, audio out through `recv`. `aclose` cancels it; it does not end the input.
#[pyclass(frozen, name = "TtsStream", module = "livekit.agents._core")]
pub struct TtsStream(pub(crate) ::std::sync::Arc<TtsStreamImpl>);

#[pymethods]
impl TtsStream {
    /// A synthesis by `tts` of `text`, or, without it, of the text pushed. Its text goes through
    /// the TTS's stage (split, paced, lowered to its syntax) and its attempts are retried within
    /// `conn_options`; the first opens when its audio is first read. Its requests carry the
    /// identity of the job `cx` runs for.
    #[new]
    #[pyo3(signature = (*, tts, text = None, conn_options = None))]
    fn new(
        py: Python<'_>,
        tts: Bound<'_, PyAny>,
        text: ::std::option::Option<::std::string::String>,
        conn_options: Option<Bound<'_, PyAny>>,
    ) -> PyResult<TtsStream> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let cx = crate::py_calls::entry_cx()?;
        let tts = crate::generated::tts::TtsProxy::from_host(tts.as_any())?;
        let conn_options = conn_options
            .map(|v| crate::generated::records::connect::api_connect_options_from_py(v.as_any()))
            .transpose()?;
        let out = py.detach(|| {
            <TtsStreamImpl as ::lk_agents_abi::tts::streams::TtsStream>::new(
                &env,
                cx,
                tts,
                text,
                conn_options,
            )
        });
        ::pyo3::PyResult::Ok(TtsStream(::std::sync::Arc::new(out)))
    }

    /// Whether its TTS runs natively; a host plugin's own attempts report their metrics.
    #[getter]
    fn native(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <TtsStreamImpl as ::lk_agents_abi::tts::streams::TtsStream>::native(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (text))]
    fn push_text(&self, py: Python<'_>, text: ::std::string::String) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <TtsStreamImpl as ::lk_agents_abi::tts::streams::TtsStream>::push_text(&self.0, text)
        })
        .map_err(|e: ::lk_agents_abi::errors::ProviderError| {
            <::lk_agents_abi::errors::ProviderError as crate::py_errors::HostException>::to_py_err(
                &e,
            )
        })?;
        Ok(())
    }

    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <TtsStreamImpl as ::lk_agents_abi::tts::streams::TtsStream>::flush(&self.0))
            .map_err(|e: ::lk_agents_abi::errors::ProviderError| {
            <::lk_agents_abi::errors::ProviderError as crate::py_errors::HostException>::to_py_err(
                &e,
            )
        })?;
        Ok(())
    }

    fn end_input(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <TtsStreamImpl as ::lk_agents_abi::tts::streams::TtsStream>::end_input(&self.0)
        })
        .map_err(|e: ::lk_agents_abi::errors::ProviderError| {
            <::lk_agents_abi::errors::ProviderError as crate::py_errors::HostException>::to_py_err(
                &e,
            )
        })?;
        Ok(())
    }

    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <TtsStreamImpl as ::lk_agents_abi::tts::streams::TtsStream>::recv(&this, )
                .await.map_err(|e: ::lk_agents_abi::errors::PullError| <::lk_agents_abi::errors::PullError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    out.map(|value| {
                        crate::generated::records::tts::synthesized_audio_to_py(py, value)
                    })
                    .transpose()?,
                    py,
                )
            })
        })
    }

    /// Every audio it synthesizes into `emitter`, until it ends: its frames, the words they
    /// cover and the usage reported, in one segment when `stream`.
    #[pyo3(signature = (emitter, *, stream))]
    fn drain_into<'py>(
        &self,
        py: Python<'py>,
        emitter: Bound<'_, PyAny>,
        stream: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let emitter = crate::py_objects::native::<
            dyn ::lk_agents_abi::tts::stream_pacer::PushedAudio,
        >(emitter.as_any())
        .and_then(|native| {
            native.ok_or_else(|| {
                ::pyo3::exceptions::PyTypeError::new_err(concat!(
                    "emitter",
                    ": expected a tts.AudioEmitter"
                ))
            })
        })?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <TtsStreamImpl as ::lk_agents_abi::tts::streams::TtsStream>::drain_into(&this, emitter, stream)
                .await.map_err(|e: ::lk_agents_abi::tts::streams::DrainError| <::lk_agents_abi::tts::streams::DrainError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () =
                <TtsStreamImpl as ::lk_agents_abi::tts::streams::TtsStream>::aclose(&this).await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/tts/streams.rs.
type SynthesisChecksImpl =
    <crate::generated::implementations::Selected as super::Implementations>::SynthesisChecks;
type SynthesisChecksValidateSynthesisOutput = ::std::option::Option<::std::string::String>;

fn _extract_synthesis_checks_validate_synthesis_duration(
    obj: &Bound<'_, PyAny>,
) -> PyResult<SynthesisChecksValidateSynthesisDurationInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "duration")
}
type SynthesisChecksValidateSynthesisDurationInput = f64;

/// Why a synthesis of ``text`` that pushed ``duration`` seconds of audio (and, streamed,
/// ``segments``) failed, or ``None``.
#[pyfunction]
#[pyo3(signature = (text, duration, segments))]
pub fn validate_synthesis(
    py: Python<'_>,
    text: ::std::string::String,
    #[pyo3(from_py_with = _extract_synthesis_checks_validate_synthesis_duration)]
    duration: SynthesisChecksValidateSynthesisDurationInput,
    segments: Option<Bound<'_, PyAny>>,
) -> PyResult<SynthesisChecksValidateSynthesisOutput> {
    let segments = segments
        .map(|v| crate::generated::records::tts::segment_count_from_py(v.as_any()))
        .transpose()?;
    let out = py.detach(|| {
        <SynthesisChecksImpl as ::lk_agents_abi::tts::streams::SynthesisChecks>::validate_synthesis(
            &text, duration, segments,
        )
    });
    ::pyo3::PyResult::Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/tts/tts.rs.
type TtsOptionsCellImpl =
    <crate::generated::implementations::Selected as super::Implementations>::TtsOptionsCell;

/// `_core.TtsOptionsCell`: the options a `tts.TTS` owns (`tts.options`), shared with every
/// synthesis over it. A synthesis reads them when its stage opens; one already open keeps them.
#[pyclass(frozen, name = "TtsOptionsCell", module = "livekit.agents._core")]
pub struct TtsOptionsCell(pub(crate) ::std::sync::Arc<TtsOptionsCellImpl>);

#[pymethods]
impl TtsOptionsCell {
    /// Split by `sentence_tokenizer`, paced as a TTS with `capabilities` is by default.
    #[new]
    #[pyo3(signature = (sentence_tokenizer, capabilities))]
    fn new(
        py: Python<'_>,
        sentence_tokenizer: Bound<'_, PyAny>,
        capabilities: Bound<'_, PyAny>,
    ) -> PyResult<TtsOptionsCell> {
        crate::py_calls::check_pid()?;
        let sentence_tokenizer =
            crate::generated::records::tts::options_tokenizer_from_py(sentence_tokenizer.as_any())?;
        let capabilities =
            crate::generated::records::tts::tts_capabilities_from_py(capabilities.as_any())?;
        let out = py.detach(|| {
            <TtsOptionsCellImpl as ::lk_agents_abi::tts::types::TtsOptionsCell>::new(
                sentence_tokenizer,
                capabilities,
            )
        });
        ::pyo3::PyResult::Ok(TtsOptionsCell(::std::sync::Arc::new(out)))
    }

    /// The tokenizer as it was given; `None` only for the options of a TTS no host built.
    #[getter]
    fn sentence_tokenizer<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <TtsOptionsCellImpl as ::lk_agents_abi::tts::types::TtsOptionsCell>::sentence_tokenizer(
                &self.0,
            )
        });
        out.map(|value| crate::py_objects::object_ref_to_py(py, value)).transpose()
    }

    #[getter]
    fn pacing<'py>(&self, py: Python<'py>) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <TtsOptionsCellImpl as ::lk_agents_abi::tts::types::TtsOptionsCell>::pacing(&self.0)
        });
        out.map(|value| {
            crate::generated::records::stream_pacer::sentence_stream_pacer_to_py(py, value)
        })
        .transpose()
    }

    /// Replace both at once.
    #[pyo3(signature = (sentence_tokenizer, pacing))]
    fn set(
        &self,
        py: Python<'_>,
        sentence_tokenizer: Bound<'_, PyAny>,
        pacing: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let sentence_tokenizer =
            crate::generated::records::tts::options_tokenizer_from_py(sentence_tokenizer.as_any())?;
        let pacing = pacing
            .map(|v| {
                crate::generated::records::stream_pacer::sentence_stream_pacer_from_py(v.as_any())
            })
            .transpose()?;
        py.detach(|| {
            <TtsOptionsCellImpl as ::lk_agents_abi::tts::types::TtsOptionsCell>::set(
                &self.0,
                sentence_tokenizer,
                pacing,
            )
        });
        Ok(())
    }
}

// Wherever the host passes a `::lk_agents_abi::tts::types::SharedTtsOptions`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::tts::types::SharedTtsOptions>(),
        read: |obj| match obj.cast::<TtsOptionsCell>() {
            Ok(this) => {
                let native: ::std::sync::Arc<dyn ::lk_agents_abi::tts::types::SharedTtsOptions> = this.get().0.clone();
                Ok(crate::py_objects::Taken::Native(::std::boxed::Box::new(native)))
            }
            Err(_) => Ok(crate::py_objects::Taken::Other),
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/tts/stream_adapter.rs.

/// [`::lk_agents_abi::tts::stream_adapter::HostSentenceTokenizer`] over a host `livekit.agents.tokenize.SentenceTokenizer`.
#[derive(Clone)]
pub struct HostSentenceTokenizerProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken;
    /// none when taken with no running loop: each call then runs in its driving operation's.
    context: ::std::option::Option<::lk_agents_abi::utils::host_ref::HostRef>,
}

impl HostSentenceTokenizerProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::tts::stream_adapter::HostSentenceTokenizer>>
    {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::tts::stream_adapter::HostSentenceTokenizer,
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
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::tts::stream_adapter::HostSentenceTokenizer>>
    {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::tts::stream_adapter::HostSentenceTokenizer,
        >(obj)?
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::tts::stream_adapter::HostSentenceTokenizer>>
    {
        if crate::py_objects::native::<
            dyn ::lk_agents_abi::tts::stream_adapter::HostSentenceTokenizer,
        >(obj)?
        .is_some()
        {
            return Err(::pyo3::exceptions::PyTypeError::new_err(
                "a native `HostSentenceTokenizer` is shared, never handed over to be owned",
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
        Ok(HostSentenceTokenizerProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            context,
        })
    }
}

impl ::lk_agents_abi::tts::stream_adapter::HostSentenceTokenizer for HostSentenceTokenizerProxy {
    fn stream(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::sync::Arc<dyn ::lk_agents_abi::tts::stream_pacer::HostSentenceStream>,
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "HostSentenceTokenizer.stream",
            &crate::py_calls::calling_context(&self.context),
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("stream")?, ::pyo3::types::PyTuple::new(py, args)?, None))
            }),
        );
        let context = ::lk_agents_abi::agent::HostContext::host_context(
            &crate::py_calls::calling_context(&self.context),
        )
        .cloned();
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("HostSentenceTokenizer.stream: the call captured no host context")));
                };
                let context = &out.returned_context(py, context).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                let o = out.value.bind(py);
                let out = (super::tts::HostSentenceStreamProxy::in_context(o.as_any(), context)).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/tts/stream_pacer.rs.

/// [`::lk_agents_abi::tts::stream_pacer::HostSentenceStream`] over a host `livekit.agents.tokenize.SentenceStream`.
#[derive(Clone)]
pub struct HostSentenceStreamProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken;
    /// none when taken with no running loop: each call then runs in its driving operation's.
    context: ::std::option::Option<::lk_agents_abi::utils::host_ref::HostRef>,
    /// The `notify` calls, delivered in order.
    notifier: crate::py_calls::Notifier,
    /// The host iterator `next` pulls, opened by its first pull.
    next_iter: ::std::sync::Arc<::pyo3::sync::PyOnceLock<Py<PyAny>>>,
}

impl HostSentenceStreamProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::tts::stream_pacer::HostSentenceStream>>
    {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::tts::stream_pacer::HostSentenceStream,
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
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::tts::stream_pacer::HostSentenceStream>>
    {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::tts::stream_pacer::HostSentenceStream,
        >(obj)?
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::tts::stream_pacer::HostSentenceStream>>
    {
        if crate::py_objects::native::<dyn ::lk_agents_abi::tts::stream_pacer::HostSentenceStream>(
            obj,
        )?
        .is_some()
        {
            return Err(::pyo3::exceptions::PyTypeError::new_err(
                "a native `HostSentenceStream` is shared, never handed over to be owned",
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
        Ok(HostSentenceStreamProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            context,
            notifier: ::std::default::Default::default(),
            next_iter: ::std::sync::Arc::new(::pyo3::sync::PyOnceLock::new()),
        })
    }
}

impl ::lk_agents_abi::tts::stream_pacer::HostSentenceStream for HostSentenceStreamProxy {
    fn push_text(&self, text: ::std::string::String) {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = text;
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "HostSentenceStream.push_text",
            &crate::py_calls::calling_context(&self.context),
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(text_arg, py)?];
                Ok((
                    host.bind(py).getattr("push_text")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
    }

    fn flush(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "HostSentenceStream.flush",
            &crate::py_calls::calling_context(&self.context),
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("flush")?, ::pyo3::types::PyTuple::new(py, args)?, None))
            }),
        );
    }

    fn end_input(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "HostSentenceStream.end_input",
            &crate::py_calls::calling_context(&self.context),
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("end_input")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
    }

    fn next(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::option::Option<::lk_agents_abi::tokenize::TokenData>,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let iterator = ::std::sync::Arc::clone(&self.next_iter);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "HostSentenceStream.__anext__",
            &crate::py_calls::calling_context(&self.context),
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    iterator
                        .get_or_try_init(py, || {
                            crate::py_calls::call(
                                &py.import("builtins")?.getattr("aiter")?,
                                (host.bind(py).clone(),),
                                None,
                            )
                            .map(::pyo3::Bound::unbind)
                        })?
                        .bind(py)
                        .clone(),
                    py.None().into_bound(py),
                ];
                Ok((
                    py.import("builtins")?.getattr("anext")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        let context = ::lk_agents_abi::agent::HostContext::host_context(
            &crate::py_calls::calling_context(&self.context),
        )
        .cloned();
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let out = (context.as_ref().map(|context| out.returned_context(py, context)).transpose().and_then(|context| crate::py_calls::returning(context, || if o.is_none() { Ok(None) } else { (crate::generated::records::tokenizer::token_data_from_py(o.as_any())).map(Some) }))).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn aclose(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "HostSentenceStream.aclose",
            &crate::py_calls::calling_context(&self.context),
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("aclose")?, ::pyo3::types::PyTuple::new(py, args)?, None))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/tts/tts.rs.

/// [`::lk_agents_abi::Tts`] over a host `livekit.agents.tts.TTS`.
#[derive(Clone)]
pub struct TtsProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken;
    /// none when taken with no running loop: each call then runs in its driving operation's.
    context: ::std::option::Option<::lk_agents_abi::utils::host_ref::HostRef>,
    /// The `notify` calls, delivered in order.
    notifier: crate::py_calls::Notifier,
    instance_id: ::std::option::Option<usize>,
    label: ::std::string::String,
    capabilities: ::lk_agents_abi::TtsCapabilities,
    num_channels: u16,
    retries_itself: bool,
    options: ::std::sync::Arc<dyn ::lk_agents_abi::tts::types::SharedTtsOptions>,
    conn_options: ::std::option::Option<::lk_agents_abi::ApiConnectOptions>,
    /// Appended to every awaited call once bound.
    conn: ::std::option::Option<::lk_agents_abi::ApiConnectOptions>,
}

impl TtsProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::Tts>> {
        if let Some(native) = crate::py_objects::native::<dyn ::lk_agents_abi::Tts>(obj)? {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::Tts>> {
        if let Some(native) = crate::py_objects::native::<dyn ::lk_agents_abi::Tts>(obj)? {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::Tts>> {
        if crate::py_objects::native::<dyn ::lk_agents_abi::Tts>(obj)?.is_some() {
            return Err(::pyo3::exceptions::PyTypeError::new_err(
                "a native `Tts` is shared, never handed over to be owned",
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
        let instance_id = {
            let v = crate::py_calls::read(obj, "_core_instance_id")?;
            if v.is_none() {
                Ok(None)
            } else {
                (crate::py_values::number_from_py::<usize>(v.as_any(), "Tts._core_instance_id"))
                    .map(Some)
            }?
        };
        let label = {
            let v = crate::py_calls::read(obj, "_core_label")?;
            v.extract::<::std::string::String>()?
        };
        let capabilities = {
            let v = crate::py_calls::read(obj, "_core_capabilities")?;
            crate::generated::records::tts::tts_capabilities_from_py(v.as_any())?
        };
        let num_channels = {
            let v = crate::py_calls::read(obj, "_core_num_channels")?;
            crate::py_values::number_from_py::<u16>(v.as_any(), "Tts._core_num_channels")?
        };
        let retries_itself = {
            let v = crate::py_calls::read(obj, "_core_retries_itself")?;
            v.extract::<bool>()?
        };
        let options = {
            let v = crate::py_calls::read(obj, "_core_options")?;
            crate::py_objects::native::<dyn ::lk_agents_abi::tts::types::SharedTtsOptions>(
                v.as_any(),
            )
            .and_then(|native| {
                native.ok_or_else(|| {
                    ::pyo3::exceptions::PyTypeError::new_err(concat!(
                        "Tts._core_options",
                        ": expected a `_core.TtsOptionsCell`"
                    ))
                })
            })?
        };
        let conn_options = {
            let v = crate::py_calls::read(obj, "_core_conn_options")?;
            if v.is_none() {
                Ok(None)
            } else {
                (crate::generated::records::connect::api_connect_options_from_py(v.as_any()))
                    .map(Some)
            }?
        };
        Ok(TtsProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            context,
            notifier: ::std::default::Default::default(),
            instance_id,
            label,
            capabilities,
            num_channels,
            retries_itself,
            options,
            conn_options,
            conn: None,
        })
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
            "Tts._core_prewarm",
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

    fn sample_rate(&self) -> ::std::result::Result<u32, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_sample_rate")?,
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
            let out = (crate::py_values::number_from_py::<u32>(
                o.as_any(),
                "Tts._core_sample_rate",
            ))
            .map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn num_channels(&self) -> u16 {
        ::std::clone::Clone::clone(&self.num_channels)
    }

    fn read_markup_provider(
        &self,
    ) -> ::std::result::Result<::std::string::String, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_markup_provider")?,
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
            let out = (o.extract::<::std::string::String>()).map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn retries_itself(&self) -> bool {
        ::std::clone::Clone::clone(&self.retries_itself)
    }

    fn options(&self) -> ::std::sync::Arc<dyn ::lk_agents_abi::tts::types::SharedTtsOptions> {
        ::std::clone::Clone::clone(&self.options)
    }

    fn with_conn_options(
        &self,
        conn: &::lk_agents_abi::ApiConnectOptions,
    ) -> ::std::option::Option<::std::sync::Arc<dyn ::lk_agents_abi::Tts>> {
        Some(::std::sync::Arc::new(Self {
            conn: Some(::std::clone::Clone::clone(conn)),
            conn_options: Some(::std::clone::Clone::clone(conn)),
            ..::std::clone::Clone::clone(self)
        }))
    }

    fn conn_options(&self) -> ::std::option::Option<::lk_agents_abi::ApiConnectOptions> {
        ::std::clone::Clone::clone(&self.conn_options)
    }

    fn failed(&self, error: ::lk_agents_abi::errors::ProviderError, recoverable: bool) {
        let host = ::std::sync::Arc::clone(&self.host);
        let error_arg = error;
        let error_arg = match ::pyo3::Python::attach(
            |py| -> ::pyo3::PyResult<::pyo3::Py<::pyo3::PyAny>> {
                Ok((crate::py_errors::ExceptionValue::to_exception(error_arg, py)?).unbind())
            },
        ) {
            Ok(value) => value,
            Err(error) => {
                log::warn!(target: ::lk_agents_abi::LOG_TARGET, "Tts._core_failed failed: {error}");
                return;
            }
        };
        let recoverable_arg = recoverable;
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "Tts._core_failed",
            &crate::py_calls::calling_context(&self.context),
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    error_arg.into_bound(py),
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(recoverable_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_failed")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
    }

    fn synthesize<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        chunk: &::lk_agents_abi::SpeechChunk,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::std::boxed::Box<
                dyn ::lk_agents_abi::ProviderStream<::lk_agents_abi::SynthesizedAudio>,
            >,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let chunk_arg = ::std::clone::Clone::clone(chunk);
        let bound = ::std::clone::Clone::clone(&self.conn);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_streams::StreamCall,
            "Tts._core_synthesize",
            &crate::py_calls::calling_context(&self.context),
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::tts::speech_chunk_to_py(py, chunk_arg)?];
                let mut args = args.to_vec();
                if let Some(bound) = bound {
                    args.push(crate::generated::records::connect::api_connect_options_to_py(
                        py, bound,
                    )?);
                }
                Ok((
                    host.bind(py).getattr("_core_synthesize")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        let context = ::lk_agents_abi::agent::HostContext::host_context(
            &crate::py_calls::calling_context(&self.context),
        )
        .cloned();
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("Tts._core_synthesize: the call captured no host context")));
                };
                let context = &out.returned_context(py, context).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                let o = out.value.bind(py);
                let out = (crate::py_streams::HostStream::open(py, o, context, |o| crate::generated::records::tts::synthesized_audio_from_py(o.as_any()), ::lk_agents_runtime::process::host_clock()).map(|s| ::std::boxed::Box::new(s) as ::std::boxed::Box<dyn ::lk_agents_abi::ProviderStream<::lk_agents_abi::SynthesizedAudio>>)).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn stream(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::lk_agents_abi::DuplexStream<
                ::std::boxed::Box<dyn ::lk_agents_abi::StreamInput<::lk_agents_abi::SpeechChunk>>,
                ::std::boxed::Box<
                    dyn ::lk_agents_abi::ProviderStream<::lk_agents_abi::SynthesizedAudio>,
                >,
            >,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let bound = ::std::clone::Clone::clone(&self.conn);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_streams::StreamCall,
            "Tts._core_stream",
            &crate::py_calls::calling_context(&self.context),
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                let mut args = args.to_vec();
                if let Some(bound) = bound {
                    args.push(crate::generated::records::connect::api_connect_options_to_py(
                        py, bound,
                    )?);
                }
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
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("Tts._core_stream: the call captured no host context")));
                };
                let context = &out.returned_context(py, context).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                let o = out.value.bind(py);
                let out = (crate::py_streams::HostStream::duplex(py, o, context, crate::generated::records::tts::speech_chunk_to_py, |o| crate::generated::records::tts::synthesized_audio_from_py(o.as_any()), ::lk_agents_runtime::process::host_clock()).map(|stream| { let (sink, source) = stream.split(); ::lk_agents_abi::DuplexStream::new(::std::boxed::Box::new(sink) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamInput<::lk_agents_abi::SpeechChunk>>, ::std::boxed::Box::new(source) as ::std::boxed::Box<dyn ::lk_agents_abi::ProviderStream<::lk_agents_abi::SynthesizedAudio>>) })).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::tts::streams::DrainError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        match self {
            Self::Pull(inner) => crate::py_errors::HostException::to_py_err(inner),
            Self::Emitter(inner) => crate::py_errors::HostException::to_py_err(inner),
        }
    }
}
