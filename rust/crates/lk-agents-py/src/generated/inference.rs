//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `inference` classes.

#![allow(clippy::too_many_arguments, reason = "a host signature takes the parameters it declares")]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/inference/mod.rs.
type InferenceCredentialsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::InferenceCredentials;

/// `explicit`, else `LIVEKIT_INFERENCE_API_KEY`, else the server's `LIVEKIT_API_KEY`; `""`
/// when nothing resolves. A set but empty `LIVEKIT_INFERENCE_API_KEY` shadows the server's.
#[pyfunction]
#[pyo3(signature = (explicit = None))]
pub fn inference_api_key(
    py: Python<'_>,
    explicit: ::std::option::Option<::std::string::String>,
) -> PyResult<::std::string::String> {
    crate::py_calls::check_pid()?;
    let out = py.detach(|| <InferenceCredentialsImpl as ::lk_agents_abi::inference::InferenceCredentials>::inference_api_key(explicit));
    ::pyo3::PyResult::Ok(out)
}

/// The same ladder for the secret, over `LIVEKIT_INFERENCE_API_SECRET`.
#[pyfunction]
#[pyo3(signature = (explicit = None))]
pub fn inference_api_secret(
    py: Python<'_>,
    explicit: ::std::option::Option<::std::string::String>,
) -> PyResult<::std::string::String> {
    crate::py_calls::check_pid()?;
    let out = py.detach(|| <InferenceCredentialsImpl as ::lk_agents_abi::inference::InferenceCredentials>::inference_api_secret(explicit));
    ::pyo3::PyResult::Ok(out)
}

/// A nonempty `LIVEKIT_INFERENCE_URL`, then staging if the server's `LIVEKIT_URL` names the
/// staging cloud, then production.
#[pyfunction]
pub fn inference_url(py: Python<'_>) -> PyResult<::std::string::String> {
    crate::py_calls::check_pid()?;
    let out = py.detach(<InferenceCredentialsImpl as ::lk_agents_abi::inference::InferenceCredentials>::inference_url);
    ::pyo3::PyResult::Ok(out)
}

/// The gateway's URL (a nonempty `base_url`, else [`Self::inference_url`]) and key pair (the
/// ladders above); a `ValueError` when the key or the secret does not resolve.
#[pyfunction]
#[pyo3(signature = (base_url = None, api_key = None, api_secret = None))]
pub fn inference_credentials<'py>(
    py: Python<'py>,
    base_url: ::std::option::Option<::std::string::String>,
    api_key: ::std::option::Option<::std::string::String>,
    api_secret: ::std::option::Option<::std::string::String>,
) -> PyResult<Bound<'py, PyAny>> {
    crate::py_calls::check_pid()?;
    let out = py.detach(|| <InferenceCredentialsImpl as ::lk_agents_abi::inference::InferenceCredentials>::inference_credentials(base_url, api_key, api_secret)).map_err(|e: ::lk_agents_abi::errors::Invalid| <::lk_agents_abi::errors::Invalid as crate::py_errors::HostException>::to_py_err(&e))?;
    crate::generated::records::inference_credentials::credentials_to_py(py, out)
}

/// Names the host SDK on every inference request this process makes (`User-Agent`); the
/// first name given holds.
#[pyfunction]
#[pyo3(signature = (user_agent))]
pub fn identify_host(py: Python<'_>, user_agent: ::std::string::String) -> PyResult<()> {
    crate::py_calls::check_pid()?;
    py.detach(|| <InferenceCredentialsImpl as ::lk_agents_abi::inference::InferenceCredentials>::identify_host(user_agent));
    Ok(())
}

/// The identification headers an inference request made here carries, in order: this
/// process's `User-Agent`, the priority of a nonempty `inference_class`, and in a job its
/// nonempty room, job and agent ids, `LIVEKIT_WORKER_TOKEN` when set and the job's own
/// headers (overriding any of these).
#[pyfunction]
#[pyo3(signature = (inference_class = None))]
pub fn inference_identity_headers<'py>(
    py: Python<'py>,
    inference_class: ::std::option::Option<::std::string::String>,
) -> PyResult<Bound<'py, PyAny>> {
    crate::py_calls::check_pid()?;
    let cx = crate::py_calls::entry_cx()?;
    let out = py.detach(|| <InferenceCredentialsImpl as ::lk_agents_abi::inference::InferenceCredentials>::inference_identity_headers(cx, inference_class));
    ::pyo3::PyResult::Ok({
        let scalar: ::std::vec::Vec<(::std::string::String, ::std::string::String)> = out.into();
        ::pyo3::types::PyList::new(
            py,
            scalar
                .into_iter()
                .map(|x| {
                    ::pyo3::PyResult::Ok({
                        let (x0, x1) = x;
                        ::pyo3::types::PyTuple::new(
                            py,
                            [
                                ::pyo3::IntoPyObjectExt::into_bound_py_any(x0, py)?,
                                ::pyo3::IntoPyObjectExt::into_bound_py_any(x1, py)?,
                            ],
                        )?
                        .into_any()
                    })
                })
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .into_any()
    })
}

// Declared in rust/crates/lk-agents-abi/src/inference/mod.rs.
type InferenceModelsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::InferenceModels;
type InferenceModelsInferenceQuotaUsageOutput =
    ::std::vec::Vec<(::std::string::String, ::std::string::String)>;
type InferenceModelsMinReasoningEffortOutput = ::std::option::Option<::std::string::String>;
type InferenceModelsLlmUnsupportedParamsOutput = ::std::vec::Vec<::std::string::String>;

/// The gateway's quota dimensions among a response's header pairs (names case-insensitive);
/// an empty value is not reported.
#[pyfunction]
#[pyo3(signature = (headers))]
pub fn inference_quota_usage(
    py: Python<'_>,
    headers: Bound<'_, PyAny>,
) -> PyResult<InferenceModelsInferenceQuotaUsageOutput> {
    let headers = crate::generated::records::from_representation::<
        ::lk_agents_abi::Headers,
        ::std::vec::Vec<(::std::string::String, ::std::string::String)>,
    >(if headers.as_any().is_instance_of::<::pyo3::types::PyList>()
        || headers.as_any().is_instance_of::<::pyo3::types::PyTuple>()
    {
        headers
            .as_any()
            .try_iter()?
            .map(|item| {
                let item = &item?;
                (|| -> ::pyo3::PyResult<_> {
                    let tuple = item.cast::<::pyo3::types::PyTuple>()?;
                    if tuple.len() != 2 {
                        return Err(::pyo3::exceptions::PyTypeError::new_err(format!(
                            "{}: expected a tuple of 2 items",
                            "headers"
                        )));
                    }
                    let item_0 = tuple.get_item(0)?;
                    let item_1 = tuple.get_item(1)?;
                    Ok((
                        (item_0.extract::<::std::string::String>())?,
                        (item_1.extract::<::std::string::String>())?,
                    ))
                })()
            })
            .collect::<PyResult<Vec<_>>>()
    } else {
        Err(::pyo3::exceptions::PyTypeError::new_err("headers: expected a list or tuple"))
    }?)?;
    let out = py.detach(|| {
        <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::inference_quota_usage(
            headers,
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
    let out = py.detach(|| <InferenceModelsImpl as ::lk_agents_abi::inference::InferenceModels>::amd_is_evaluated_model(model, kind)).map_err(|e: ::lk_agents_abi::errors::Invalid| <::lk_agents_abi::errors::Invalid as crate::py_errors::HostException>::to_py_err(&e))?;
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

/// The turn detector a host constructs.
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
        .map_err(|e: ::lk_agents_abi::errors::Invalid| {
            <::lk_agents_abi::errors::Invalid as crate::py_errors::HostException>::to_py_err(&e)
        })?;
    crate::generated::records::inference_models::eot_selection_to_py(py, out)
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

// Declared in rust/crates/lk-agents-abi/src/inference/avatar_session.rs.
type InferenceAvatarImpl =
    <crate::generated::implementations::Selected as super::Implementations>::InferenceAvatar;
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
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let extra_kwargs = extra_kwargs
            .map(|v| match crate::py_values::value_from_py(v.as_any(), "extra_kwargs")? {
                ::serde_json::Value::Object(map) => Ok(map),
                _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!(
                    "extra_kwargs",
                    ": expected a dict"
                ))),
            })
            .transpose()?;
        let conn_options =
            crate::generated::records::connect::api_connect_options_from_py(conn_options.as_any())?;
        let out = py.detach(|| <InferenceAvatarImpl as ::lk_agents_abi::inference::avatar_session::InferenceAvatar>::new(&env, model, avatar_identity, avatar_name, extra_kwargs, base_url, api_key, api_secret, conn_options)).map_err(|e: ::lk_agents_abi::inference::avatar_session::AvatarError| <::lk_agents_abi::inference::avatar_session::AvatarError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(InferenceAvatar(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn provider(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <InferenceAvatarImpl as ::lk_agents_abi::inference::avatar_session::InferenceAvatar>::provider(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn avatar_identity(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <InferenceAvatarImpl as ::lk_agents_abi::inference::avatar_session::InferenceAvatar>::avatar_identity(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    /// The gateway-generated avatar session id, once created.
    #[getter]
    fn session_id(&self, py: Python<'_>) -> PyResult<InferenceAvatarSessionIdOutput> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <InferenceAvatarImpl as ::lk_agents_abi::inference::avatar_session::InferenceAvatar>::session_id(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    /// The provider's own session id while the session is held.
    #[getter]
    fn provider_session_id(
        &self,
        py: Python<'_>,
    ) -> PyResult<InferenceAvatarProviderSessionIdOutput> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <InferenceAvatarImpl as ::lk_agents_abi::inference::avatar_session::InferenceAvatar>::provider_session_id(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    /// Creates the provider session for the agent `agent_identity` in `room_name` and returns the
    /// audio sample rate the avatar takes. A second call while one runs, or once a session was
    /// created, is refused. Cancelling the call stops creating; a session created anyway is
    /// terminated. `livekit_url` defaults to `LIVEKIT_URL`; its requests carry the identity of the
    /// job `cx` runs for.
    #[pyo3(signature = (*, room_name, room_sid, livekit_url = None, agent_identity))]
    fn create<'py>(
        &self,
        py: Python<'py>,
        room_name: ::std::string::String,
        room_sid: ::std::string::String,
        livekit_url: ::std::option::Option<::std::string::String>,
        agent_identity: ::std::string::String,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let cx = crate::py_calls::entry_cx()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <InferenceAvatarImpl as ::lk_agents_abi::inference::avatar_session::InferenceAvatar>::create(&this, room_name, room_sid, livekit_url, agent_identity, cx)
                .await.map_err(|e: ::lk_agents_abi::inference::avatar_session::AvatarError| <::lk_agents_abi::inference::avatar_session::AvatarError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// Stops creating and terminates every held session so it stops billing now. Returns once
    /// that cleanup settled or its deadline passed (then logged); a failed terminate keeps the
    /// session held, so a second call retries.
    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let cx = crate::py_calls::entry_cx()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <InferenceAvatarImpl as ::lk_agents_abi::inference::avatar_session::InferenceAvatar>::aclose(&this, cx)
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/inference/tts_client.rs.
type InferenceTtsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::InferenceTts;

fn _extract_inference_tts_new_sample_rate(
    obj: &Bound<'_, PyAny>,
) -> PyResult<InferenceTtsNewSampleRateInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<u32>(obj.as_any(), "sample_rate")).map(Some)
    }
}
type InferenceTtsNewSampleRateInput = ::std::option::Option<u32>;

/// `inference/tts.py`'s WebSocket, over `lk-agents-inference::tts::GatewayTts`. One instance
/// serves every job it is handed to; each request carries the identity of the job it runs for.
#[pyclass(frozen, weakref, subclass, name = "InferenceTts", module = "livekit.agents._core")]
pub struct InferenceTts(pub(crate) ::std::sync::OnceLock<::std::sync::Arc<InferenceTtsImpl>>);

#[pymethods]
impl InferenceTts {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(
        _args: &Bound<'_, ::pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, ::pyo3::types::PyDict>>,
    ) -> Self {
        Self(::std::sync::OnceLock::new())
    }

    /// A client of `model` (``"provider/model[:voice]"``) on the gateway. Starts nothing. The
    /// gateway's URL and key pair resolve from the given values, the environment and the server's
    /// settings. Its text is split by `sentence_tokenizer` until `update_options` changes it.
    #[pyo3(signature = (sentence_tokenizer, model, *, voice = None, language = None, encoding = None, sample_rate = None, base_url = None, api_key = None, api_secret = None, extra_kwargs = None, fallback = None, conn_options = None))]
    fn __init__(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        sentence_tokenizer: Bound<'_, PyAny>,
        model: ::std::string::String,
        voice: ::std::option::Option<::std::string::String>,
        language: ::std::option::Option<::std::string::String>,
        encoding: ::std::option::Option<::std::string::String>,
        #[pyo3(from_py_with = _extract_inference_tts_new_sample_rate)]
        sample_rate: InferenceTtsNewSampleRateInput,
        base_url: ::std::option::Option<::std::string::String>,
        api_key: ::std::option::Option<::std::string::String>,
        api_secret: ::std::option::Option<::std::string::String>,
        extra_kwargs: Option<Bound<'_, PyAny>>,
        fallback: Option<Bound<'_, PyAny>>,
        conn_options: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let once =
            || ::pyo3::exceptions::PyRuntimeError::new_err("InferenceTts.__init__ runs once");
        if slf.get().0.get().is_some() {
            return Err(once());
        }
        let env = crate::process::env();
        let label = crate::py_objects::class_label(slf.as_any())?;
        let events = crate::generated::types::ProviderEventsProxy::weak(slf.as_any())?;
        let sentence_tokenizer =
            crate::generated::records::tts::options_tokenizer_from_py(sentence_tokenizer.as_any())?;
        let extra_kwargs = extra_kwargs
            .map(|v| match crate::py_values::value_from_py(v.as_any(), "extra_kwargs")? {
                ::serde_json::Value::Object(map) => Ok(map),
                value => Err(::pyo3::exceptions::PyValueError::new_err(format!(
                    "expected a dict of extra kwargs, got {value}"
                ))),
            })
            .transpose()?;
        let fallback = fallback
            .map(|v| crate::generated::records::inference_tts::fallback_models_from_py(v.as_any()))
            .transpose()?;
        let conn_options = conn_options
            .map(|v| crate::generated::records::connect::api_connect_options_from_py(v.as_any()))
            .transpose()?;
        let out = py
            .detach(|| {
                <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::new(
                    &env,
                    label,
                    events,
                    sentence_tokenizer,
                    model,
                    voice,
                    language,
                    encoding,
                    sample_rate,
                    base_url,
                    api_key,
                    api_secret,
                    extra_kwargs,
                    fallback,
                    conn_options,
                )
            })
            .map_err(|e: ::lk_agents_abi::errors::Invalid| {
                <::lk_agents_abi::errors::Invalid as crate::py_errors::HostException>::to_py_err(&e)
            })?;
        slf.get().0.set(::std::sync::Arc::new(out)).map_err(|_| once())
    }

    #[getter]
    fn model(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let engine = self.engine()?;
        let out = py.detach(|| {
            <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::model(
                engine,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// Streaming, its alignment read from the live model and extra kwargs.
    #[getter]
    fn capabilities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let engine = self.engine()?;
        let out = py.detach(|| <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::capabilities(engine));
        crate::generated::records::tts::tts_capabilities_to_py(py, out)
    }

    #[getter]
    fn num_channels(&self, py: Python<'_>) -> PyResult<u16> {
        crate::py_calls::check_pid()?;
        let engine = self.engine()?;
        let out = py.detach(|| <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::num_channels(engine));
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn sample_rate(&self, py: Python<'_>) -> PyResult<u32> {
        crate::py_calls::check_pid()?;
        let engine = self.engine()?;
        let out = py.detach(|| {
            <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::sample_rate(
                engine,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// The policy this client was given, which a session or run that supplies none keeps.
    #[pyo3(name = "_core_conn_options")]
    fn conn_options<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let engine = self.engine()?;
        let out = py.detach(|| <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::conn_options(engine));
        crate::generated::records::connect::api_connect_options_to_py(py, out)
    }

    /// The options it owns, which every synthesis over it reads.
    #[pyo3(name = "_core_options")]
    fn options(&self, py: Python<'_>) -> PyResult<super::tts::TtsOptionsCell> {
        crate::py_calls::check_pid()?;
        let engine = self.engine()?;
        let out = py.detach(|| {
            <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::options(
                engine,
            )
        });
        ::pyo3::PyResult::Ok(super::tts::TtsOptionsCell(out))
    }

    /// A configured socket waits in the pool for the first synthesis.
    fn prewarm(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let engine = self.engine()?;
        let cx = crate::py_calls::entry_cx()?;
        py.detach(|| {
            <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::prewarm(
                engine, cx,
            )
        });
        Ok(())
    }

    /// Read by the next synthesis; ``extra_kwargs`` merges into the persistent kwargs.
    #[pyo3(name = "_core_update_options")]
    #[pyo3(signature = (*, voice = None, model = None, language = None, extra_kwargs = None))]
    fn update_options(
        &self,
        py: Python<'_>,
        voice: ::std::option::Option<::std::string::String>,
        model: ::std::option::Option<::std::string::String>,
        language: ::std::option::Option<::std::string::String>,
        extra_kwargs: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let engine = self.engine()?;
        let extra_kwargs = extra_kwargs
            .map(|v| match crate::py_values::value_from_py(v.as_any(), "extra_kwargs")? {
                ::serde_json::Value::Object(map) => Ok(map),
                value => Err(::pyo3::exceptions::PyValueError::new_err(format!(
                    "expected a dict of extra kwargs, got {value}"
                ))),
            })
            .transpose()?;
        py.detach(|| <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::update_options(engine, voice, model, language, extra_kwargs));
        Ok(())
    }

    /// Closes the pooled sockets; a synthesis still running finishes, its socket closed instead
    /// of pooled.
    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let engine = self.engine()?;
        let this = ::std::sync::Arc::clone(engine);
        crate::py_calls::future_into_py(py, async move {
            let () =
                <InferenceTtsImpl as ::lk_agents_abi::inference::tts_client::InferenceTts>::aclose(
                    &this,
                )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

impl InferenceTts {
    /// The implementation its `__init__` built.
    pub(crate) fn engine(&self) -> PyResult<&::std::sync::Arc<InferenceTtsImpl>> {
        self.0.get().ok_or_else(|| crate::py_objects::unbuilt("InferenceTts"))
    }
}

// Wherever the host passes a `::lk_agents_abi::Tts`, an instance of this class is itself once built.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::Tts>(),
        read: |obj| match obj.cast::<InferenceTts>() {
            Ok(this) => {
                let engine = this.get().0.get().cloned().map(|engine| -> ::std::sync::Arc<dyn ::lk_agents_abi::Tts> { engine });
                crate::py_objects::built(obj, engine, "InferenceTts")
            }
            Err(_) => Ok(crate::py_objects::Taken::Other),
        },
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::inference::avatar_session::AvatarError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        match self {
            Self::InvalidModel(..) => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
            Self::CatalogIdAndImage(..) => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
            Self::InvalidOption(..) => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
            Self::MissingUrl => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
            Self::Api(inner) => crate::py_errors::HostException::to_py_err(inner),
            _ => ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(
                self,
            )),
        }
    }
}
