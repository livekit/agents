//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `inference` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/inference/mod.rs.
type InferenceCredentialsImpl = <crate::Selected as super::Implementations>::InferenceCredentials;

/// `explicit`, else `inference` (`LIVEKIT_INFERENCE_API_KEY`), else `server` (the server's
/// `LIVEKIT_API_KEY`); a set but empty `inference` shadows `server`. The host reads the
/// environment.
#[pyfunction]
#[pyo3(signature = (explicit = None, inference = None, server = None))]
pub fn inference_resolve_api_key(
    py: Python<'_>,
    explicit: ::std::option::Option<::std::string::String>,
    inference: ::std::option::Option<::std::string::String>,
    server: ::std::option::Option<::std::string::String>,
) -> PyResult<::std::string::String> {
    crate::host::check_pid()?;
    let out = py.detach(|| <InferenceCredentialsImpl as ::lk_agents_abi::inference::InferenceCredentials>::inference_resolve_api_key(explicit, inference, server));
    ::pyo3::PyResult::Ok(out)
}

/// The same ladder for the secret.
#[pyfunction]
#[pyo3(signature = (explicit = None, inference = None, server = None))]
pub fn inference_resolve_api_secret(
    py: Python<'_>,
    explicit: ::std::option::Option<::std::string::String>,
    inference: ::std::option::Option<::std::string::String>,
    server: ::std::option::Option<::std::string::String>,
) -> PyResult<::std::string::String> {
    crate::host::check_pid()?;
    let out = py.detach(|| <InferenceCredentialsImpl as ::lk_agents_abi::inference::InferenceCredentials>::inference_resolve_api_secret(explicit, inference, server));
    ::pyo3::PyResult::Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/inference/mod.rs.
type InferenceModelsImpl = <crate::Selected as super::Implementations>::InferenceModels;
type InferenceModelsInferenceQuotaUsageOutput =
    ::std::vec::Vec<(::std::string::String, ::std::string::String)>;
type InferenceModelsTtsParseModelStringOutput =
    (::std::string::String, ::std::option::Option<::std::string::String>);
type InferenceModelsSttParseModelStringOutput =
    (::std::string::String, ::std::option::Option<::std::string::String>);
type InferenceModelsMinReasoningEffortOutput = ::std::option::Option<::std::string::String>;
type InferenceModelsLlmUnsupportedParamsOutput = ::std::vec::Vec<::std::string::String>;

/// The gateway's quota dimensions among a response's header pairs (names case-insensitive);
/// an empty value is not reported.
#[pyfunction]
#[pyo3(signature = (headers))]
pub fn inference_quota_usage(
    py: Python<'_>,
    headers: ::std::vec::Vec<(::std::string::String, ::std::string::String)>,
) -> PyResult<InferenceModelsInferenceQuotaUsageOutput> {
    let out = py.detach(|| {
        <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::inference_quota_usage(
            headers,
        )
    });
    ::pyo3::PyResult::Ok(out)
}

/// `inference_url` (`LIVEKIT_INFERENCE_URL`), then staging if the server's `LIVEKIT_URL` names
/// the staging cloud, then production. The host reads the environment.
#[pyfunction]
#[pyo3(signature = (inference_url = None, server_url = None))]
pub fn inference_default_url(
    py: Python<'_>,
    inference_url: ::std::option::Option<::std::string::String>,
    server_url: ::std::option::Option<::std::string::String>,
) -> PyResult<::std::string::String> {
    let out = py.detach(|| {
        <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::inference_default_url(
            inference_url,
            server_url,
        )
    });
    ::pyo3::PyResult::Ok(out)
}

/// Loads the native models' weights; `false` when no provider is linked.
#[pyfunction]
#[pyo3(signature = (vad, eot))]
pub fn local_models_init(py: Python<'_>, vad: bool, eot: bool) -> PyResult<bool> {
    let out = py.detach(|| {
        <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::local_models_init(
            vad, eot,
        )
    });
    ::pyo3::PyResult::Ok(out)
}

/// `"provider/model:voice"` split into the model and its voice.
#[pyfunction]
#[pyo3(signature = (model))]
pub fn tts_parse_model_string(
    py: Python<'_>,
    model: ::std::string::String,
) -> PyResult<InferenceModelsTtsParseModelStringOutput> {
    let out = py.detach(|| <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::tts_parse_model_string(model));
    ::pyo3::PyResult::Ok(out)
}

/// Whether the model streams word timings for its audio under these options.
#[pyfunction]
#[pyo3(signature = (model, extra_kwargs))]
pub fn tts_has_aligned_transcript(
    py: Python<'_>,
    model: ::std::string::String,
    extra_kwargs: Bound<'_, PyAny>,
) -> PyResult<bool> {
    let extra_kwargs = match crate::python::value_from_py(extra_kwargs.as_any(), "extra_kwargs")? {
        ::serde_json::Value::Object(map) => Ok(map),
        _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!(
            "extra_kwargs",
            ": expected a dict"
        ))),
    }?;
    let out = py.detach(|| <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::tts_has_aligned_transcript(model, extra_kwargs));
    ::pyo3::PyResult::Ok(out)
}

/// The speech-markup dialect the model's provider reads.
#[pyfunction]
#[pyo3(signature = (model))]
pub fn tts_markup_provider_key(
    py: Python<'_>,
    model: ::std::string::String,
) -> PyResult<::std::string::String> {
    let out = py.detach(|| <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::tts_markup_provider_key(model));
    ::pyo3::PyResult::Ok(out)
}

/// `"provider/model:language"` split into the model and its language.
#[pyfunction]
#[pyo3(signature = (model))]
pub fn stt_parse_model_string(
    py: Python<'_>,
    model: ::std::string::String,
) -> PyResult<InferenceModelsSttParseModelStringOutput> {
    let out = py.detach(|| <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::stt_parse_model_string(model));
    ::pyo3::PyResult::Ok(out)
}

/// The lowest reasoning effort the model supports, or `None` if it has no such control.
#[pyfunction]
#[pyo3(signature = (model))]
pub fn min_reasoning_effort(
    py: Python<'_>,
    model: ::std::string::String,
) -> PyResult<InferenceModelsMinReasoningEffortOutput> {
    let out = py.detach(|| {
        <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::min_reasoning_effort(
            model,
        )
    });
    ::pyo3::PyResult::Ok(out)
}

/// Whether AMD may fill in its own LLM and STT from the gateway (`detector.py:184-194`).
#[pyfunction]
#[pyo3(signature = (livekit_url, api_key, api_secret))]
pub fn amd_auto_select_models(
    py: Python<'_>,
    livekit_url: ::std::string::String,
    api_key: ::std::string::String,
    api_secret: ::std::string::String,
) -> PyResult<bool> {
    let out = py.detach(|| <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::amd_auto_select_models(livekit_url, api_key, api_secret));
    ::pyo3::PyResult::Ok(out)
}

/// Whether `model` is in AMD's evaluated `"llm"` or `"stt"` table.
#[pyfunction]
#[pyo3(signature = (model, kind))]
pub fn amd_is_evaluated_model(
    py: Python<'_>,
    model: ::std::string::String,
    kind: ::std::string::String,
) -> PyResult<bool> {
    let out = py.detach(|| <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::amd_is_evaluated_model(model, kind)).map_err(|e: ::lk_agents_abi::inference::InvalidOption| <::lk_agents_abi::inference::InvalidOption as crate::python::HostException>::to_py_err(&e))?;
    ::pyo3::PyResult::Ok(out)
}

/// The parameters the given model refuses. `has_tools` is `bool(tools)`; `reasoning_effort`
/// is the request's `reasoning_effort` and `reasoning` the effort of its Responses
/// `reasoning`.
#[pyfunction]
#[pyo3(signature = (model, *, has_tools = false, reasoning_effort = None, reasoning = None))]
pub fn llm_unsupported_params(
    py: Python<'_>,
    model: ::std::string::String,
    has_tools: bool,
    reasoning_effort: ::std::option::Option<::std::string::String>,
    reasoning: ::std::option::Option<::std::string::String>,
) -> PyResult<InferenceModelsLlmUnsupportedParamsOutput> {
    let out = py.detach(|| <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::llm_unsupported_params(model, has_tools, reasoning_effort, reasoning));
    ::pyo3::PyResult::Ok(out)
}

/// The turn detector a host constructs: `(model, cloud, warnings)`.
#[pyfunction]
#[pyo3(signature = (version, hosted, dev, url, key, secret, local_fallback))]
pub fn eot_select<'py>(
    py: Python<'py>,
    version: ::std::option::Option<::std::string::String>,
    hosted: bool,
    dev: bool,
    url: ::std::string::String,
    key: ::std::string::String,
    secret: ::std::string::String,
    local_fallback: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let out = py
        .detach(|| {
            <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::eot_select(
                version,
                hosted,
                dev,
                url,
                key,
                secret,
                local_fallback,
            )
        })
        .map_err(|e: ::lk_agents_abi::inference::InvalidOption| {
            <::lk_agents_abi::inference::InvalidOption as crate::python::HostException>::to_py_err(
                &e,
            )
        })?;
    ::pyo3::PyResult::Ok({
        let (x0, x1, x2) = out;
        ::pyo3::types::PyTuple::new(
            py,
            [
                crate::generated::records::turn_events::turn_detector_model_to_py(py, x0)?,
                ::pyo3::IntoPyObjectExt::into_bound_py_any(x1, py)?,
                ::pyo3::types::PyList::new(
                    py,
                    x2.into_iter()
                        .map(|x| ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py))
                        .collect::<PyResult<Vec<_>>>()?,
                )?
                .into_any(),
            ],
        )?
        .into_any()
    })
}

/// `_resolve_vad_for_model`: what the host does with the caller's VAD for this model.
#[pyfunction]
#[pyo3(signature = (model = None, *, has_vad = false))]
pub fn stt_resolve_vad_for_model<'py>(
    py: Python<'py>,
    model: ::std::option::Option<::std::string::String>,
    has_vad: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let out = py.detach(|| <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::stt_resolve_vad_for_model(model, has_vad));
    ::pyo3::PyResult::Ok(crate::python::str_to_py(py, &out))
}

/// `_normalize_fallback`'s string arm for an STT chain.
#[pyfunction]
#[pyo3(signature = (model))]
pub fn stt_normalize_fallback_str<'py>(
    py: Python<'py>,
    model: ::std::string::String,
) -> PyResult<Bound<'py, PyAny>> {
    let out = py.detach(|| <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::stt_normalize_fallback_str(model));
    crate::generated::records::inference_stt::fallback_model_to_py(py, out)
}

/// `_normalize_fallback`'s string arm for a TTS chain.
#[pyfunction]
#[pyo3(signature = (model))]
pub fn tts_normalize_fallback_str<'py>(
    py: Python<'py>,
    model: ::std::string::String,
) -> PyResult<Bound<'py, PyAny>> {
    let out = py.detach(|| <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::tts_normalize_fallback_str(model));
    crate::generated::records::inference_tts::fallback_model_to_py(py, out)
}

/// `inference.STT`'s capabilities. Word alignment holds only when the model and every
/// `fallback` model align: any turn may be served by a fallback, and a false claim disables
/// the fast VAD interruption path.
#[pyfunction]
#[pyo3(signature = (model = None, *, fallback = None, extra_kwargs = None))]
pub fn stt_capabilities<'py>(
    py: Python<'py>,
    model: ::std::option::Option<::std::string::String>,
    fallback: ::std::option::Option<::std::vec::Vec<::std::string::String>>,
    extra_kwargs: Option<Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let extra_kwargs = extra_kwargs
        .map(|v| match crate::python::value_from_py(v.as_any(), "extra_kwargs")? {
            ::serde_json::Value::Object(map) => Ok(map),
            _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!(
                "extra_kwargs",
                ": expected a dict"
            ))),
        })
        .transpose()?;
    let out = py.detach(|| {
        <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::stt_capabilities(
            model,
            fallback,
            extra_kwargs,
        )
    });
    crate::generated::records::stt::stt_capabilities_to_py(py, out)
}

/// `stt.FallbackAdapter`'s capabilities, merged from its members' before any chain is built.
#[pyfunction]
#[pyo3(signature = (entries))]
pub fn stt_fallback_capabilities<'py>(
    py: Python<'py>,
    entries: Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let entries = entries.try_iter().and_then(|items| {
        items
            .map(|item| {
                let item = item?;
                crate::generated::records::stt::stt_capabilities_from_py(item.as_any())
            })
            .collect::<PyResult<Vec<_>>>()
    })?;
    let out = py.detach(|| <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::stt_fallback_capabilities(entries));
    crate::generated::records::stt::stt_capabilities_to_py(py, out)
}

/// `tts.FallbackAdapter`'s capabilities, merged from its members' before any chain is built.
#[pyfunction]
#[pyo3(signature = (entries))]
pub fn tts_fallback_capabilities<'py>(
    py: Python<'py>,
    entries: Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let entries = entries.try_iter().and_then(|items| {
        items
            .map(|item| {
                let item = item?;
                crate::generated::records::tts::tts_capabilities_from_py(item.as_any())
            })
            .collect::<PyResult<Vec<_>>>()
    })?;
    let out = py.detach(|| <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::tts_fallback_capabilities(entries));
    crate::generated::records::tts::tts_capabilities_to_py(py, out)
}

/// The realtime fallback's capabilities: `audio_output` and `turn_detection` must agree (a
/// `ValueError` naming the capability and both values), the rest are ANDed.
#[pyfunction]
#[pyo3(signature = (caps))]
pub fn merge_realtime_capabilities<'py>(
    py: Python<'py>,
    caps: Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let caps = caps.try_iter().and_then(|items| {
        items
            .map(|item| {
                let item = item?;
                crate::generated::records::realtime::realtime_capabilities_from_py(item.as_any())
            })
            .collect::<PyResult<Vec<_>>>()
    })?;
    let out = py.detach(|| <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::merge_realtime_capabilities(caps)).map_err(|e: ::lk_agents_abi::inference::InvalidOption| <::lk_agents_abi::inference::InvalidOption as crate::python::HostException>::to_py_err(&e))?;
    crate::generated::records::realtime::realtime_capabilities_to_py(py, out)
}

// Declared in rust/crates/lk-agents-abi/src/inference/avatar_session.rs.
type InferenceAvatarImpl = <crate::Selected as super::Implementations>::InferenceAvatar;
type InferenceAvatarSessionIdOutput = ::std::option::Option<::std::string::String>;
type InferenceAvatarProviderSessionIdOutput = ::std::option::Option<::std::string::String>;

/// One avatar provisioned through LiveKit Inference: at most one provider session per instance,
/// created under one operation identity, held until `aclose()` terminates it.
#[pyclass(frozen, name = "InferenceAvatar", module = "livekit.agents._core")]
pub struct InferenceAvatar(pub(crate) ::std::sync::Arc<InferenceAvatarImpl>);

#[pymethods]
impl InferenceAvatar {
    /// `model` is `"provider"` or `"provider/<avatar_id>"`. The identity defaults to
    /// `"<provider>-inference-avatar"`, the name to the identity. Starts nothing.
    #[new]
    #[pyo3(signature = (model, *, avatar_identity = None, avatar_name = None, extra_kwargs = None, base_url, api_key, api_secret, conn_options))]
    fn new(
        py: Python<'_>,
        model: ::std::string::String,
        avatar_identity: ::std::option::Option<::std::string::String>,
        avatar_name: ::std::option::Option<::std::string::String>,
        extra_kwargs: Option<Bound<'_, PyAny>>,
        base_url: ::std::string::String,
        api_key: ::std::string::String,
        api_secret: ::std::string::String,
        conn_options: Bound<'_, PyAny>,
    ) -> PyResult<InferenceAvatar> {
        crate::host::check_pid()?;
        let env = crate::process::env();
        let extra_kwargs = extra_kwargs
            .map(|v| match crate::python::value_from_py(v.as_any(), "extra_kwargs")? {
                ::serde_json::Value::Object(map) => Ok(map),
                _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!(
                    "extra_kwargs",
                    ": expected a dict"
                ))),
            })
            .transpose()?;
        let conn_options = crate::conversions::api_connect_options_from_py(conn_options.as_any())?;
        let out = py.detach(|| <InferenceAvatarImpl as ::lk_agents_abi::inference::avatar_session::InferenceAvatar>::new(&env, model, avatar_identity, avatar_name, extra_kwargs, base_url, api_key, api_secret, conn_options)).map_err(|e: ::lk_agents_abi::inference::avatar_session::AvatarError| <::lk_agents_abi::inference::avatar_session::AvatarError as crate::python::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(InferenceAvatar(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn provider(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::host::check_pid()?;
        let out = py.detach(|| <InferenceAvatarImpl as ::lk_agents_abi::inference::avatar_session::InferenceAvatar>::provider(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn avatar_identity(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::host::check_pid()?;
        let out = py.detach(|| <InferenceAvatarImpl as ::lk_agents_abi::inference::avatar_session::InferenceAvatar>::avatar_identity(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    /// The gateway-generated avatar session id, once created.
    #[getter]
    fn session_id(&self, py: Python<'_>) -> PyResult<InferenceAvatarSessionIdOutput> {
        crate::host::check_pid()?;
        let out = py.detach(|| <InferenceAvatarImpl as ::lk_agents_abi::inference::avatar_session::InferenceAvatar>::session_id(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    /// The provider's own session id while the session is held.
    #[getter]
    fn provider_session_id(
        &self,
        py: Python<'_>,
    ) -> PyResult<InferenceAvatarProviderSessionIdOutput> {
        crate::host::check_pid()?;
        let out = py.detach(|| <InferenceAvatarImpl as ::lk_agents_abi::inference::avatar_session::InferenceAvatar>::provider_session_id(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    /// Creates the provider session for the agent `agent_identity` in `room_name` and returns the
    /// audio sample rate the avatar takes. A second call while one runs, or once a session was
    /// created, is refused. Cancelling the call stops creating; a session created anyway is
    /// terminated. `livekit_url` defaults to `LIVEKIT_URL`; `headers` are `get_inference_headers()`.
    #[pyo3(signature = (*, room_name, room_sid, livekit_url = None, agent_identity, headers))]
    fn create<'py>(
        &self,
        py: Python<'py>,
        room_name: ::std::string::String,
        room_sid: ::std::string::String,
        livekit_url: ::std::option::Option<::std::string::String>,
        agent_identity: ::std::string::String,
        headers: ::std::vec::Vec<(::std::string::String, ::std::string::String)>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <InferenceAvatarImpl as ::lk_agents_abi::inference::avatar_session::InferenceAvatar>::create(&this, room_name, room_sid, livekit_url, agent_identity, headers)
                .await.map_err(|e: ::lk_agents_abi::inference::avatar_session::AvatarError| <::lk_agents_abi::inference::avatar_session::AvatarError as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// Stops creating and terminates every held session so it stops billing now. Returns once
    /// that cleanup settled or its deadline passed (then logged); a failed terminate keeps the
    /// session held, so a second call retries.
    #[pyo3(signature = (headers))]
    fn aclose<'py>(
        &self,
        py: Python<'py>,
        headers: ::std::vec::Vec<(::std::string::String, ::std::string::String)>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let () = <InferenceAvatarImpl as ::lk_agents_abi::inference::avatar_session::InferenceAvatar>::aclose(&this, headers)
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/inference/tts_client.rs.
type InferenceTtsImpl = <crate::Selected as super::Implementations>::InferenceTts;

fn _extract_inference_tts_new_sample_rate(
    obj: &Bound<'_, PyAny>,
) -> PyResult<InferenceTtsNewSampleRateInput> {
    crate::python::number_from_py::<u32>(obj.as_any(), "sample_rate")
}
type InferenceTtsNewSampleRateInput = u32;

/// `inference/tts.py`'s WebSocket, over `lk-agents-inference::tts::GatewayTts`.
#[pyclass(frozen, name = "InferenceTts", module = "livekit.agents._core")]
pub struct InferenceTts(pub(crate) ::std::sync::Arc<InferenceTtsImpl>);

#[pymethods]
impl InferenceTts {
    #[new]
    #[pyo3(signature = (model, *, base_url, api_key, api_secret, conn_options, voice = None, language = None, encoding = "pcm_s16le".to_owned(), sample_rate = 24000, extra_kwargs = None, fallback = None))]
    fn new(
        py: Python<'_>,
        model: ::std::string::String,
        base_url: ::std::string::String,
        api_key: ::std::string::String,
        api_secret: ::std::string::String,
        conn_options: Bound<'_, PyAny>,
        voice: ::std::option::Option<::std::string::String>,
        language: ::std::option::Option<::std::string::String>,
        encoding: ::std::string::String,
        #[pyo3(from_py_with = _extract_inference_tts_new_sample_rate)]
        sample_rate: InferenceTtsNewSampleRateInput,
        extra_kwargs: Option<Bound<'_, PyAny>>,
        fallback: Option<Bound<'_, PyAny>>,
    ) -> PyResult<InferenceTts> {
        crate::host::check_pid()?;
        let env = crate::process::env();
        let conn_options = crate::conversions::api_connect_options_from_py(conn_options.as_any())?;
        let extra_kwargs = extra_kwargs
            .map(|v| match crate::python::value_from_py(v.as_any(), "extra_kwargs")? {
                ::serde_json::Value::Object(map) => Ok(map),
                value => Err(::pyo3::exceptions::PyValueError::new_err(format!(
                    "expected a dict of extra kwargs, got {value}"
                ))),
            })
            .transpose()?;
        let fallback = fallback
            .map(|v| {
                v.cast::<::pyo3::types::PyList>().map_err(::pyo3::PyErr::from).and_then(|items| {
                    items
                        .iter()
                        .map(|item| {
                            if !item.is_instance_of::<::pyo3::types::PyDict>() {
                                return Err(::pyo3::exceptions::PyValueError::new_err(
                                    "each fallback entry must be a dict",
                                ));
                            }
                            crate::generated::records::inference_tts::fallback_model_from_py(
                                item.as_any(),
                            )
                        })
                        .collect::<PyResult<Vec<_>>>()
                })
            })
            .transpose()?;
        let out = py.detach(|| <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::new(&env, model, base_url, api_key, api_secret, conn_options, voice, language, encoding, sample_rate, extra_kwargs, fallback)).map_err(|e: ::lk_agents_abi::llm::inference::InvalidPayload| <::lk_agents_abi::llm::inference::InvalidPayload as crate::python::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(InferenceTts(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn sample_rate(&self, py: Python<'_>) -> PyResult<u32> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::sample_rate(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn aligned_transcript(&self, py: Python<'_>) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = py.detach(|| <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::aligned_transcript(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (*, headers = None))]
    fn prewarm(
        &self,
        py: Python<'_>,
        headers: ::std::option::Option<
            ::std::vec::Vec<(::std::string::String, ::std::string::String)>,
        >,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| {
            <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::prewarm(
                &self.0, headers,
            )
        });
        Ok(())
    }

    #[pyo3(signature = (headers, reporter = None))]
    fn identified(
        &self,
        py: Python<'_>,
        headers: ::std::vec::Vec<(::std::string::String, ::std::string::String)>,
        reporter: Option<Bound<'_, PyAny>>,
    ) -> PyResult<InferenceTts> {
        crate::host::check_pid()?;
        let reporter =
            reporter.map(|v| crate::generated::tts::TtsProxy::from_host(v.as_any())).transpose()?;
        let out = py.detach(|| {
            <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::identified(
                &self.0, headers, reporter,
            )
        });
        ::pyo3::PyResult::Ok(InferenceTts(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (*, model = None, voice = None, language = None, extra = None))]
    fn update_options(
        &self,
        py: Python<'_>,
        model: ::std::option::Option<::std::string::String>,
        voice: ::std::option::Option<::std::string::String>,
        language: ::std::option::Option<::std::string::String>,
        extra: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        let extra = extra
            .map(|v| match crate::python::value_from_py(v.as_any(), "extra")? {
                ::serde_json::Value::Object(map) => Ok(map),
                value => Err(::pyo3::exceptions::PyValueError::new_err(format!(
                    "expected a dict of extra kwargs, got {value}"
                ))),
            })
            .transpose()?;
        py.detach(|| <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::update_options(&self.0, model, voice, language, extra));
        Ok(())
    }

    #[pyo3(signature = (*, headers = None, conn_options = None, expressive = false))]
    fn stream<'py>(
        &self,
        py: Python<'py>,
        headers: ::std::option::Option<
            ::std::vec::Vec<(::std::string::String, ::std::string::String)>,
        >,
        conn_options: Option<Bound<'_, PyAny>>,
        expressive: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let conn_options = conn_options
            .map(|v| crate::conversions::api_connect_options_from_py(v.as_any()))
            .transpose()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::stream(&this, headers, conn_options, expressive)
                .await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    super::tts::TtsStream(::std::sync::Arc::new(out)),
                    py,
                )
            })
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::Tts`, an instance of this class is itself.
inventory::submit! {
    crate::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::Tts>(),
        read: |obj| {
            let this = obj.cast::<InferenceTts>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::Tts> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

impl crate::python::HostException for ::lk_agents_abi::inference::InvalidOption {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::python::HostException for ::lk_agents_abi::inference::avatar_session::AvatarError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::InvalidModel(..) => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
            Self::CatalogIdAndImage(..) => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
            Self::MissingUrl => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
            Self::Api(inner) => crate::python::HostException::to_py_err(inner),
            _ => ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(
                self,
            )),
        }
    }
}
