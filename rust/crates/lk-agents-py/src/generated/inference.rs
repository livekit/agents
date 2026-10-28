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

// Declared in rust/crates/lk-agents-abi/src/inference/avatar_session.rs.
type InferenceAvatarImpl = <crate::Selected as super::Implementations>::InferenceAvatar;

/// One avatar provisioned through LiveKit Inference: at most one provider session per instance,
/// created under one operation identity, held until `aclose()` terminates it.
#[pyclass(frozen, name = "InferenceAvatar", module = "livekit.agents._core")]
pub struct InferenceAvatar(::std::sync::Arc<InferenceAvatarImpl>);

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
        ::lk_agents_abi::python::future_into_py(py, async move {
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
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <InferenceAvatarImpl as ::lk_agents_abi::inference::avatar_session::InferenceAvatar>::aclose(&this, headers)
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}
