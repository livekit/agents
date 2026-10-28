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

/// `explicit`, else `LIVEKIT_INFERENCE_API_KEY`, else `LIVEKIT_API_KEY`; a set but empty
/// `LIVEKIT_INFERENCE_API_KEY` shadows `LIVEKIT_API_KEY` (`os.getenv(A, os.getenv(B, ""))`).
#[pyfunction]
#[pyo3(signature = (explicit = None))]
pub fn inference_resolve_api_key(explicit: ::std::option::Option<::std::string::String>) -> PyResult<::std::string::String> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <InferenceCredentialsImpl as ::lk_agents_abi::inference::InferenceCredentials>::inference_resolve_api_key(explicit);
    Ok(out)
}

/// The same ladder for the secret.
#[pyfunction]
#[pyo3(signature = (explicit = None))]
pub fn inference_resolve_api_secret(explicit: ::std::option::Option<::std::string::String>) -> PyResult<::std::string::String> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <InferenceCredentialsImpl as ::lk_agents_abi::inference::InferenceCredentials>::inference_resolve_api_secret(explicit);
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/inference/mod.rs.
type InferenceModelsImpl = <crate::Selected as super::Implementations>::InferenceModels;

fn _extract_inference_models_interruption_min_frames_min_interruption_duration(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "min_interruption_duration")
}

fn _extract_inference_models_interruption_estimate_probability_probabilities(obj: &Bound<'_, PyAny>) -> PyResult<::std::vec::Vec<f64>> {
    obj.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<f64>(&item, "probabilities") }).collect::<PyResult<Vec<_>>>())
}

/// The gateway's quota dimensions among a response's header pairs (names case-insensitive);
/// an empty value is not reported.
#[pyfunction]
#[pyo3(signature = (headers))]
pub fn inference_quota_usage(headers: ::std::vec::Vec<(::std::string::String, ::std::string::String)>) -> PyResult<::std::vec::Vec<(::std::string::String, ::std::string::String)>> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::inference_quota_usage(headers);
    Ok(out)
}

/// `LIVEKIT_INFERENCE_URL`, then staging if `LIVEKIT_URL` names the staging cloud, then
/// production.
#[pyfunction]
pub fn inference_default_url() -> PyResult<::std::string::String> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::inference_default_url();
    Ok(out)
}

/// Whether the build links a native provider of the local models.
#[pyfunction]
pub fn local_models_native() -> PyResult<bool> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::local_models_native();
    Ok(out)
}

/// Loads the native models' weights; `false` when no provider is linked.
#[pyfunction]
#[pyo3(signature = (vad, eot))]
pub fn local_models_init(vad: bool, eot: bool) -> PyResult<bool> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::local_models_init(vad, eot);
    Ok(out)
}

/// The frames an interruption must span: `min_interruption_duration` at the detector's rate.
#[pyfunction]
#[pyo3(signature = (min_interruption_duration))]
pub fn interruption_min_frames(#[pyo3(from_py_with = _extract_inference_models_interruption_min_frames_min_interruption_duration)] min_interruption_duration: f64) -> PyResult<u32> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::interruption_min_frames(min_interruption_duration);
    Ok(out)
}

/// The interruption probability a window of frame probabilities reports.
#[pyfunction]
#[pyo3(signature = (probabilities))]
pub fn interruption_estimate_probability(#[pyo3(from_py_with = _extract_inference_models_interruption_estimate_probability_probabilities)] probabilities: ::std::vec::Vec<f64>) -> PyResult<f64> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::interruption_estimate_probability(probabilities);
    Ok(out)
}

/// `"provider/model:voice"` split into the model and its voice.
#[pyfunction]
#[pyo3(signature = (model))]
pub fn tts_parse_model_string(model: ::std::string::String) -> PyResult<(::std::string::String, ::std::option::Option<::std::string::String>)> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::tts_parse_model_string(model);
    Ok(out)
}

/// Whether the model streams word timings for its audio under these options.
#[pyfunction]
#[pyo3(signature = (model, extra_kwargs))]
pub fn tts_has_aligned_transcript(model: ::std::string::String, extra_kwargs: Bound<'_, PyAny>) -> PyResult<bool> {
    let extra_kwargs = match ::lk_agents_abi::python::value_from_py(&extra_kwargs, "extra_kwargs")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("extra_kwargs", ": expected a dict"))) }?;
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::tts_has_aligned_transcript(model, extra_kwargs);
    Ok(out)
}

/// The speech-markup dialect the model's provider reads.
#[pyfunction]
#[pyo3(signature = (model))]
pub fn tts_markup_provider_key(model: ::std::string::String) -> PyResult<::std::string::String> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::tts_markup_provider_key(model);
    Ok(out)
}

/// `"provider/model:language"` split into the model and its language.
#[pyfunction]
#[pyo3(signature = (model))]
pub fn stt_parse_model_string(model: ::std::string::String) -> PyResult<(::std::string::String, ::std::option::Option<::std::string::String>)> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::stt_parse_model_string(model);
    Ok(out)
}

/// Whether these options ask the provider for speaker diarization.
#[pyfunction]
#[pyo3(signature = (extra_kwargs = None))]
pub fn stt_diarization_enabled(extra_kwargs: Option<Bound<'_, PyAny>>) -> PyResult<bool> {
    let extra_kwargs = extra_kwargs.map(|v| match ::lk_agents_abi::python::value_from_py(&v, "extra_kwargs")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("extra_kwargs", ": expected a dict"))) }).transpose()?;
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::stt_diarization_enabled(extra_kwargs);
    Ok(out)
}

/// Whether the model carries the agent's context across recognitions.
#[pyfunction]
#[pyo3(signature = (model = None))]
pub fn stt_supports_agent_context_carryover(model: ::std::option::Option<::std::string::String>) -> PyResult<bool> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::stt_supports_agent_context_carryover(model);
    Ok(out)
}

/// Whether the model reports word timings by default.
#[pyfunction]
#[pyo3(signature = (model = None))]
pub fn stt_aligned_transcript_for_model(model: ::std::option::Option<::std::string::String>) -> PyResult<bool> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::stt_aligned_transcript_for_model(model);
    Ok(out)
}

/// The lowest reasoning effort the model supports, or `None` if it has no such control.
#[pyfunction]
#[pyo3(signature = (model))]
pub fn min_reasoning_effort(model: ::std::string::String) -> PyResult<::std::option::Option<::std::string::String>> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::min_reasoning_effort(model);
    Ok(out)
}

/// Whether AMD may fill in its own LLM and STT from the gateway (`detector.py:184-194`).
#[pyfunction]
#[pyo3(signature = (livekit_url, api_key, api_secret))]
pub fn amd_auto_select_models(livekit_url: ::std::string::String, api_key: ::std::string::String, api_secret: ::std::string::String) -> PyResult<bool> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::amd_auto_select_models(livekit_url, api_key, api_secret);
    Ok(out)
}

/// Whether `model` is in AMD's evaluated `"llm"` or `"stt"` table.
#[pyfunction]
#[pyo3(signature = (model, kind))]
pub fn amd_is_evaluated_model(model: ::std::string::String, kind: ::std::string::String) -> PyResult<bool> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::amd_is_evaluated_model(model, kind).map_err(|e: ::lk_agents_abi::inference::InvalidOption| <::lk_agents_abi::inference::InvalidOption as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

/// `AMDPredictionEvent.is_human` for a category's wire value.
#[pyfunction]
#[pyo3(signature = (category))]
pub fn amd_category_is_human(category: ::std::string::String) -> PyResult<bool> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::amd_category_is_human(category).map_err(|e: ::lk_agents_abi::inference::InvalidOption| <::lk_agents_abi::inference::InvalidOption as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

/// `AMDPredictionEvent.is_machine` for a category's wire value.
#[pyfunction]
#[pyo3(signature = (category))]
pub fn amd_category_is_machine(category: ::std::string::String) -> PyResult<bool> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::amd_category_is_machine(category).map_err(|e: ::lk_agents_abi::inference::InvalidOption| <::lk_agents_abi::inference::InvalidOption as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

/// The parameters the given model refuses. `has_tools` is `bool(tools)`; `reasoning_effort`
/// is the request's `reasoning_effort` and `reasoning` the effort of its Responses
/// `reasoning`.
#[pyfunction]
#[pyo3(signature = (model, *, has_tools = false, reasoning_effort = None, reasoning = None))]
pub fn llm_unsupported_params(model: ::std::string::String, has_tools: bool, reasoning_effort: ::std::option::Option<::std::string::String>, reasoning: ::std::option::Option<::std::string::String>) -> PyResult<::std::vec::Vec<::std::string::String>> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::llm_unsupported_params(model, has_tools, reasoning_effort, reasoning);
    Ok(out)
}

/// The turn detector a host constructs: `(model, cloud, warnings)`.
#[pyfunction]
#[pyo3(signature = (version, hosted, dev, url, key, secret, local_fallback))]
pub fn eot_select(version: ::std::option::Option<::std::string::String>, hosted: bool, dev: bool, url: ::std::string::String, key: ::std::string::String, secret: ::std::string::String, local_fallback: bool) -> PyResult<(::std::string::String, bool, ::std::vec::Vec<::std::string::String>)> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::eot_select(version, hosted, dev, url, key, secret, local_fallback).map_err(|e: ::lk_agents_abi::inference::InvalidOption| <::lk_agents_abi::inference::InvalidOption as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

/// `_resolve_vad_for_model`: what the host does with the caller's VAD for this model.
#[pyfunction]
#[pyo3(signature = (model = None, *, has_vad = false))]
pub fn stt_resolve_vad_for_model(model: ::std::option::Option<::std::string::String>, has_vad: bool) -> PyResult<::lk_agents_abi::inference::VadResolution> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::stt_resolve_vad_for_model(model, has_vad);
    Ok(out)
}

/// `_normalize_fallback`'s string arm for an STT chain.
#[pyfunction]
#[pyo3(signature = (model))]
pub fn stt_normalize_fallback_str(model: ::std::string::String) -> PyResult<::lk_agents_abi::inference::fallback::SttFallbackModel> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::stt_normalize_fallback_str(model);
    Ok(out)
}

/// `_normalize_fallback`'s string arm for a TTS chain.
#[pyfunction]
#[pyo3(signature = (model))]
pub fn tts_normalize_fallback_str(model: ::std::string::String) -> PyResult<::lk_agents_abi::inference::fallback::TtsFallbackModel> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::tts_normalize_fallback_str(model);
    Ok(out)
}

/// `inference.STT`'s capabilities. Word alignment holds only when the model and every
/// `fallback` model align: any turn may be served by a fallback, and a false claim disables
/// the fast VAD interruption path.
#[pyfunction]
#[pyo3(signature = (model = None, *, fallback = None, extra_kwargs = None))]
pub fn stt_capabilities(model: ::std::option::Option<::std::string::String>, fallback: ::std::option::Option<::std::vec::Vec<::std::string::String>>, extra_kwargs: Option<Bound<'_, PyAny>>) -> PyResult<::lk_agents_abi::SttCapabilities> {
    let extra_kwargs = extra_kwargs.map(|v| match ::lk_agents_abi::python::value_from_py(&v, "extra_kwargs")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("extra_kwargs", ": expected a dict"))) }).transpose()?;
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::stt_capabilities(model, fallback, extra_kwargs);
    Ok(out)
}

/// `stt.FallbackAdapter`'s capabilities, merged from its members' before any chain is built.
#[pyfunction]
#[pyo3(signature = (entries))]
pub fn stt_fallback_capabilities(entries: ::std::vec::Vec<::lk_agents_abi::SttCapabilities>) -> PyResult<::lk_agents_abi::SttCapabilities> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::stt_fallback_capabilities(entries);
    Ok(out)
}

/// `tts.FallbackAdapter`'s capabilities, merged from its members' before any chain is built.
#[pyfunction]
#[pyo3(signature = (entries))]
pub fn tts_fallback_capabilities(entries: ::std::vec::Vec<::lk_agents_abi::TtsCapabilities>) -> PyResult<::lk_agents_abi::TtsCapabilities> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::tts_fallback_capabilities(entries);
    Ok(out)
}

/// The realtime fallback's capabilities: `audio_output` and `turn_detection` must agree (a
/// `ValueError` naming the capability and both values), the rest are ANDed.
#[pyfunction]
#[pyo3(signature = (caps))]
pub fn merge_realtime_capabilities(caps: ::std::vec::Vec<::lk_agents_abi::llm::realtime::RealtimeCapabilities>) -> PyResult<::lk_agents_abi::llm::realtime::RealtimeCapabilities> {
    let out = <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::merge_realtime_capabilities(caps).map_err(|e: ::lk_agents_abi::inference::InvalidOption| <::lk_agents_abi::inference::InvalidOption as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/inference/avatar_session.rs.
type InferenceAvatarImpl = <crate::Selected as super::Implementations>::InferenceAvatar;

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
    fn new(model: ::std::string::String, avatar_identity: ::std::option::Option<::std::string::String>, avatar_name: ::std::option::Option<::std::string::String>, extra_kwargs: Option<Bound<'_, PyAny>>, base_url: ::std::string::String, api_key: ::std::string::String, api_secret: ::std::string::String, conn_options: ::lk_agents_abi::ApiConnectOptions) -> PyResult<InferenceAvatar> {
        ::lk_agents_abi::python::check_pid()?;
        let extra_kwargs = extra_kwargs.map(|v| match ::lk_agents_abi::python::value_from_py(&v, "extra_kwargs")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("extra_kwargs", ": expected a dict"))) }).transpose()?;
        let out = <InferenceAvatarImpl as ::lk_agents_abi::inference::avatar_session::InferenceAvatar>::new(model, avatar_identity, avatar_name, extra_kwargs, base_url, api_key, api_secret, conn_options).map_err(|e: ::lk_agents_abi::inference::avatar_session::AvatarError| <::lk_agents_abi::inference::avatar_session::AvatarError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(InferenceAvatar(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn provider(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <InferenceAvatarImpl as ::lk_agents_abi::inference::avatar_session::InferenceAvatar>::provider(&self.0);
        Ok(out)
    }

    #[getter]
    fn avatar_identity(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <InferenceAvatarImpl as ::lk_agents_abi::inference::avatar_session::InferenceAvatar>::avatar_identity(&self.0);
        Ok(out)
    }

    /// The gateway-generated avatar session id, once created.
    #[getter]
    fn session_id(&self) -> PyResult<::std::option::Option<::std::string::String>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <InferenceAvatarImpl as ::lk_agents_abi::inference::avatar_session::InferenceAvatar>::session_id(&self.0);
        Ok(out)
    }

    /// The provider's own session id while the session is held.
    #[getter]
    fn provider_session_id(&self) -> PyResult<::std::option::Option<::std::string::String>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <InferenceAvatarImpl as ::lk_agents_abi::inference::avatar_session::InferenceAvatar>::provider_session_id(&self.0);
        Ok(out)
    }

    /// Creates the provider session for the agent `agent_identity` in `room_name` and returns the
    /// audio sample rate the avatar takes. A second call while one runs, or once a session was
    /// created, is refused. Cancelling the call stops creating; a session created anyway is
    /// terminated. `livekit_url` defaults to `LIVEKIT_URL`; `headers` are `get_inference_headers()`.
    #[pyo3(signature = (*, room_name, room_sid, livekit_url = None, agent_identity, headers))]
    fn create<'py>(&self, py: Python<'py>, room_name: ::std::string::String, room_sid: ::std::string::String, livekit_url: ::std::option::Option<::std::string::String>, agent_identity: ::std::string::String, headers: ::std::vec::Vec<(::std::string::String, ::std::string::String)>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <InferenceAvatarImpl as ::lk_agents_abi::inference::avatar_session::InferenceAvatar>::create(&this, room_name, room_sid, livekit_url, agent_identity, headers)
                .await.map_err(|e: ::lk_agents_abi::inference::avatar_session::AvatarError| <::lk_agents_abi::inference::avatar_session::AvatarError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// Stops creating and terminates every held session so it stops billing now. Returns once
    /// that cleanup settled or its deadline passed (then logged); a failed terminate keeps the
    /// session held, so a second call retries.
    #[pyo3(signature = (headers))]
    fn aclose<'py>(&self, py: Python<'py>, headers: ::std::vec::Vec<(::std::string::String, ::std::string::String)>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <InferenceAvatarImpl as ::lk_agents_abi::inference::avatar_session::InferenceAvatar>::aclose(&this, headers)
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/inference/tts_client.rs.
type InferenceTtsImpl = <crate::Selected as super::Implementations>::InferenceTts;

fn _extract_inference_tts_new_sample_rate(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    ::lk_agents_abi::python::number_from_py::<u32>(&obj, "sample_rate")
}

/// `inference/tts.py`'s WebSocket, over `lk-agents-inference::tts::NativeTts`.
#[pyclass(frozen, name = "InferenceTts", module = "livekit.agents._core")]
pub struct InferenceTts(pub(crate) ::std::sync::Arc<InferenceTtsImpl>);

#[pymethods]
impl InferenceTts {
    #[new]
    #[pyo3(signature = (model, *, base_url, api_key, api_secret, conn_options, voice = None, language = None, encoding = "pcm_s16le".to_owned(), sample_rate = 24000, extra_kwargs = None, fallback = None))]
    fn new(model: ::std::string::String, base_url: ::std::string::String, api_key: ::std::string::String, api_secret: ::std::string::String, conn_options: ::lk_agents_abi::ApiConnectOptions, voice: ::std::option::Option<::std::string::String>, language: ::std::option::Option<::std::string::String>, encoding: ::std::string::String, #[pyo3(from_py_with = _extract_inference_tts_new_sample_rate)] sample_rate: u32, extra_kwargs: Option<Bound<'_, PyAny>>, fallback: Option<Bound<'_, PyAny>>) -> PyResult<InferenceTts> {
        ::lk_agents_abi::python::check_pid()?;
        let extra_kwargs = extra_kwargs.map(|v| match ::lk_agents_abi::python::value_from_py(&v, "extra_kwargs")? { ::serde_json::Value::Object(map) => Ok(map), value => Err(::pyo3::exceptions::PyValueError::new_err(format!("expected a dict of extra kwargs, got {value}"))) }).transpose()?;
        let fallback = fallback.map(|v| v.cast::<::pyo3::types::PyList>().map_err(::pyo3::PyErr::from).and_then(|items| items.iter().map(|item| { if !item.is_instance_of::<::pyo3::types::PyDict>() { return Err(::pyo3::exceptions::PyValueError::new_err("each fallback entry must be a dict")); } item.extract::<::lk_agents_abi::inference::fallback::TtsFallbackModel>() }).collect::<PyResult<Vec<_>>>())).transpose()?;
        let out = <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::new(model, base_url, api_key, api_secret, conn_options, voice, language, encoding, sample_rate, extra_kwargs, fallback).map_err(|e: ::lk_agents_abi::llm::inference::InvalidPayload| <::lk_agents_abi::llm::inference::InvalidPayload as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(InferenceTts(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (expressive))]
    fn set_expressive(&self, expressive: bool) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::set_expressive(&self.0, expressive);
        Ok(())
    }

    #[getter]
    fn sample_rate(&self) -> PyResult<u32> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::sample_rate(&self.0);
        Ok(out)
    }

    #[getter]
    fn aligned_transcript(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::aligned_transcript(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (clock))]
    fn bind_clock(&self, clock: Bound<'_, PyAny>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        let clock = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::Clock>(&clock).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("clock", ": expected native livekit.agents._core.SessionClock")))?;
        <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::bind_clock(&self.0, clock);
        Ok(())
    }

    #[pyo3(signature = (*, headers = None))]
    fn prewarm(&self, headers: ::std::option::Option<::std::vec::Vec<(::std::string::String, ::std::string::String)>>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::prewarm(&self.0, headers);
        Ok(())
    }

    #[pyo3(signature = (headers, reporter = None))]
    fn identified(&self, headers: ::std::vec::Vec<(::std::string::String, ::std::string::String)>, reporter: Option<Bound<'_, PyAny>>) -> PyResult<InferenceTts> {
        ::lk_agents_abi::python::check_pid()?;
        let reporter = reporter.map(|v| crate::generated::tts::TtsProxy::from_host(&v)).transpose()?;
        let out = <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::identified(&self.0, headers, reporter);
        Ok(InferenceTts(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (*, model = None, voice = None, language = None, extra = None))]
    fn update_options(&self, model: ::std::option::Option<::std::string::String>, voice: ::std::option::Option<::std::string::String>, language: ::std::option::Option<::std::string::String>, extra: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        let extra = extra.map(|v| match ::lk_agents_abi::python::value_from_py(&v, "extra")? { ::serde_json::Value::Object(map) => Ok(map), value => Err(::pyo3::exceptions::PyValueError::new_err(format!("expected a dict of extra kwargs, got {value}"))) }).transpose()?;
        <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::update_options(&self.0, model, voice, language, extra);
        Ok(())
    }

    #[pyo3(signature = (*, headers = None, conn_options = None))]
    fn stream<'py>(&self, py: Python<'py>, headers: ::std::option::Option<::std::vec::Vec<(::std::string::String, ::std::string::String)>>, conn_options: ::std::option::Option<::lk_agents_abi::ApiConnectOptions>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::stream(&this, headers, conn_options)
                .await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(super::tts::InferenceTtsStream(::std::sync::Arc::new(out)), py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::Tts`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::Tts>(),
        read: |obj| {
            let this = obj.cast::<InferenceTts>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::Tts> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// A `::lk_agents_abi::Tts` this class implements is handed to the host as an instance of it.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNativeClass {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::Tts>(),
        write: |py, native| {
            let native = native.downcast::<InferenceTtsImpl>().ok()?;
            Some(::pyo3::Py::new(py, InferenceTts(native)).map(::pyo3::Py::into_any))
        },
    }
}
