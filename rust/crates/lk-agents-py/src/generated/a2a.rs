//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `a2a` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/a2a/auth.rs.
type _core_CredentialsImpl = <crate::Selected as super::Implementations>::Credentials;

fn _extract_credentials_project_token_ttl_seconds(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "ttl_seconds")
}

fn _extract_credentials_thread_grant_token_ttl_seconds(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "ttl_seconds")
}

#[pyclass(frozen, name = "_core_Credentials", module = "livekit.agents._core")]
pub struct _core_Credentials(pub(crate) ::std::sync::Arc<_core_CredentialsImpl>);

#[pymethods]
impl _core_Credentials {
    #[new]
    #[pyo3(signature = (api_key, api_secret))]
    fn new(api_key: ::std::string::String, api_secret: ::std::string::String) -> PyResult<_core_Credentials> {
        crate::host::check_pid()?;
        let out = <_core_CredentialsImpl as ::lk_agents_abi::a2a::auth::Credentials>::new(api_key, api_secret);
        Ok(_core_Credentials(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (token, *, agent_name = "".to_owned(), deployment = "".to_owned()))]
    fn verify<'py>(&self, py: Python<'py>, token: ::std::string::String, agent_name: ::std::string::String, deployment: ::std::string::String) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let out = <_core_CredentialsImpl as ::lk_agents_abi::a2a::auth::Credentials>::verify(&self.0, token, agent_name, deployment).map_err(|e: ::lk_agents_abi::a2a::auth::AuthError| <::lk_agents_abi::a2a::auth::AuthError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(crate::generated::records::a2a_auth::auth_principal_to_py(py, out)?)
    }

    #[pyo3(signature = (authorization, thread_grant, node_authorized, agent_name, deployment))]
    fn principal<'py>(&self, py: Python<'py>, authorization: ::std::option::Option<::std::string::String>, thread_grant: ::std::option::Option<::std::string::String>, node_authorized: bool, agent_name: ::std::string::String, deployment: ::std::string::String) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let out = <_core_CredentialsImpl as ::lk_agents_abi::a2a::auth::Credentials>::principal(&self.0, authorization, thread_grant, node_authorized, agent_name, deployment).map_err(|e: ::lk_agents_abi::a2a::auth::AuthError| <::lk_agents_abi::a2a::auth::AuthError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(crate::generated::records::a2a_auth::auth_principal_to_py(py, out)?)
    }

    #[pyo3(signature = (*, identity = "worker".to_owned(), ttl_seconds = 3600))]
    fn project_token(&self, identity: ::std::string::String, #[pyo3(from_py_with = _extract_credentials_project_token_ttl_seconds)] ttl_seconds: u64) -> PyResult<::std::string::String> {
        crate::host::check_pid()?;
        let out = <_core_CredentialsImpl as ::lk_agents_abi::a2a::auth::Credentials>::project_token(&self.0, identity, ttl_seconds).map_err(|e: ::lk_agents_abi::a2a::auth::AuthError| <::lk_agents_abi::a2a::auth::AuthError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    #[pyo3(signature = (grant, *, identity = "".to_owned(), ttl_seconds = 3600))]
    fn thread_grant_token(&self, grant: Bound<'_, PyAny>, identity: ::std::string::String, #[pyo3(from_py_with = _extract_credentials_thread_grant_token_ttl_seconds)] ttl_seconds: u64) -> PyResult<::std::string::String> {
        crate::host::check_pid()?;
        let grant = crate::generated::records::a2a_auth::auth_grant_from_py(&grant)?;
        let out = <_core_CredentialsImpl as ::lk_agents_abi::a2a::auth::Credentials>::thread_grant_token(&self.0, grant, identity, ttl_seconds).map_err(|e: ::lk_agents_abi::a2a::auth::AuthError| <::lk_agents_abi::a2a::auth::AuthError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/auth.rs.
type AuthPermissionsImpl = <crate::Selected as super::Implementations>::AuthPermissions;

#[pyfunction]
#[pyo3(signature = (grant, operation, thread_id, scope))]
pub fn _core_auth_grant_allows(grant: Bound<'_, PyAny>, operation: Bound<'_, PyAny>, thread_id: ::std::string::String, scope: ::std::option::Option<::std::string::String>) -> PyResult<bool> {
    crate::host::check_pid()?;
    let grant = crate::generated::records::a2a_auth::auth_grant_from_py(&grant)?;
    let operation = crate::generated::records::a2a_auth::auth_operation_from_py(&operation)?;
    let out = <AuthPermissionsImpl as ::lk_agents_abi::a2a::auth::AuthPermissions>::_core_auth_grant_allows(grant, operation, thread_id, scope);
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/a2a/card.rs.
type EndpointCardImpl = <crate::Selected as super::Implementations>::EndpointCard;

/// Echo the supported capabilities selected by the optional extension header.
#[pyfunction]
#[pyo3(signature = (requested))]
pub fn _core_a2a_extensions_header(requested: ::std::option::Option<::std::string::String>) -> PyResult<::std::option::Option<::std::string::String>> {
    crate::host::check_pid()?;
    let out = <EndpointCardImpl as ::lk_agents_abi::a2a::card::EndpointCard>::_core_a2a_extensions_header(requested);
    Ok(out)
}

/// Serialize the upstream A2A card; the host only adapts it to its HTTP/SDK representation.
#[pyfunction]
#[pyo3(signature = (descriptor))]
pub fn _core_a2a_agent_card(descriptor: Bound<'_, PyAny>) -> PyResult<::std::string::String> {
    crate::host::check_pid()?;
    let descriptor = crate::generated::records::a2a_card::endpoint_descriptor_from_py(&descriptor)?;
    let out = <EndpointCardImpl as ::lk_agents_abi::a2a::card::EndpointCard>::_core_a2a_agent_card(descriptor).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/a2a/client.rs.
type MessageContentImpl = <crate::Selected as super::Implementations>::MessageContent;

/// The text parts of a generated A2A message, joined by newlines.
#[pyfunction]
#[pyo3(signature = (message))]
pub fn _core_a2a_message_text(message: Bound<'_, PyAny>) -> PyResult<::std::string::String> {
    crate::host::check_pid()?;
    let message = crate::generated::records::a2a_client::message_from_py(&message)?;
    let out = <MessageContentImpl as ::lk_agents_abi::a2a::client::MessageContent>::_core_a2a_message_text(message);
    Ok(out)
}

/// The first object data part of a generated A2A message.
#[pyfunction]
#[pyo3(signature = (message))]
pub fn _core_a2a_message_data<'py>(py: Python<'py>, message: Bound<'_, PyAny>) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
    crate::host::check_pid()?;
    let message = crate::generated::records::a2a_client::message_from_py(&message)?;
    let out = <MessageContentImpl as ::lk_agents_abi::a2a::client::MessageContent>::_core_a2a_message_data(message);
    Ok(out.map(|value| Ok::<_, ::pyo3::PyErr>(::lk_agents_abi::python::extra_to_py(py, &value)?.into_any())).transpose()?)
}

// Declared in rust/crates/lk-agents-abi/src/a2a/client.rs.
type ReplyImpl = <crate::Selected as super::Implementations>::Reply;

#[pyclass(frozen, name = "Reply", module = "livekit.agents._core")]
pub struct Reply(pub(crate) ::std::sync::Arc<ReplyImpl>);

#[pymethods]
impl Reply {
    #[getter]
    fn task_id(&self) -> PyResult<::std::string::String> {
        crate::host::check_pid()?;
        let out = <ReplyImpl as ::lk_agents_abi::a2a::client::Reply>::task_id(&self.0);
        Ok(out)
    }

    #[getter]
    fn thread_id(&self) -> PyResult<::std::string::String> {
        crate::host::check_pid()?;
        let out = <ReplyImpl as ::lk_agents_abi::a2a::client::Reply>::thread_id(&self.0);
        Ok(out)
    }

    fn __aiter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            match <ReplyImpl as ::lk_agents_abi::a2a::client::Reply>::next(&this).await.map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as ::lk_agents_abi::python::HostException>::to_py_err(&e))? {
                Some(out) => Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(crate::generated::records::a2a_client::task_update_to_py(py, out)?, py)),
                None => Err(::pyo3::exceptions::PyStopAsyncIteration::new_err(())),
            }
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <ReplyImpl as ::lk_agents_abi::a2a::client::Reply>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/client.rs.
type AgentClientImpl = <crate::Selected as super::Implementations>::AgentClient;

#[pyclass(frozen, name = "AgentClient", module = "livekit.agents._core")]
pub struct AgentClient(pub(crate) ::std::sync::Arc<AgentClientImpl>);

#[pymethods]
impl AgentClient {
    #[new]
    #[pyo3(signature = (endpoint_url, *, token = None, conn_options = None))]
    fn new(endpoint_url: ::std::string::String, token: ::std::option::Option<::std::string::String>, conn_options: ::std::option::Option<::lk_agents_abi::ApiConnectOptions>) -> PyResult<AgentClient> {
        crate::host::check_pid()?;
        let env = crate::process::env();
        let out = <AgentClientImpl as ::lk_agents_abi::a2a::client::AgentClient>::new(&env, endpoint_url, token, conn_options).map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(AgentClient(::std::sync::Arc::new(out)))
    }

    fn connect<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <AgentClientImpl as ::lk_agents_abi::a2a::client::AgentClient>::connect(&this, )
                .await.map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(crate::generated::records::a2a_client::agent_card_to_py(py, out)?, py))
        })
    }

    #[getter]
    fn offers_extension(&self) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = <AgentClientImpl as ::lk_agents_abi::a2a::client::AgentClient>::offers_extension(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (message, *, thread_id = None, task_id = None, metadata = None, chat_ctx = None))]
    fn send<'py>(&self, py: Python<'py>, message: Bound<'_, PyAny>, thread_id: ::std::option::Option<::std::string::String>, task_id: ::std::option::Option<::std::string::String>, metadata: Option<Bound<'_, PyAny>>, chat_ctx: Option<Bound<'_, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let message = crate::generated::records::a2a_client::message_input_from_py(&message)?;
        let metadata = metadata.map(|v| crate::generated::records::a2a_client::message_metadata_from_py(&v)).transpose()?;
        let chat_ctx = chat_ctx.map(|v| v.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::chat_items::chat_item_from_py(&item) }).collect::<PyResult<Vec<_>>>())).transpose()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <AgentClientImpl as ::lk_agents_abi::a2a::client::AgentClient>::send(&this, message, thread_id, task_id, metadata, chat_ctx)
                .await.map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(super::a2a::Reply(::std::sync::Arc::new(out)), py))
        })
    }

    #[pyo3(signature = (task_id, thread_id))]
    fn _core_subscribe<'py>(&self, py: Python<'py>, task_id: ::std::string::String, thread_id: ::std::string::String) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <AgentClientImpl as ::lk_agents_abi::a2a::client::AgentClient>::_core_subscribe(&this, task_id, thread_id)
                .await.map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(super::a2a::Reply(::std::sync::Arc::new(out)), py))
        })
    }

    #[pyo3(signature = (task_id, *, thread_id = None, reason = None))]
    fn cancel<'py>(&self, py: Python<'py>, task_id: ::std::string::String, thread_id: ::std::option::Option<::std::string::String>, reason: ::std::option::Option<::std::string::String>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <AgentClientImpl as ::lk_agents_abi::a2a::client::AgentClient>::cancel(&this, task_id, thread_id, reason)
                .await.map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(crate::generated::records::a2a_client::task_update_to_py(py, out)?, py))
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <AgentClientImpl as ::lk_agents_abi::a2a::client::AgentClient>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/door.rs.
type _core_A2ADoorImpl = <crate::Selected as super::Implementations>::A2ADoor;

fn _extract_a2_a_door_dispatch_body(obj: &Bound<'_, PyAny>) -> PyResult<::std::vec::Vec<u8>> {
    { if obj.is_instance_of::<::pyo3::types::PyBytes>() || obj.is_instance_of::<::pyo3::types::PyByteArray>() || obj.is_instance_of::<::pyo3::types::PyMemoryView>() { ::lk_agents_abi::python::owned_bytes(&obj, "body") } else { obj.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "body") }).collect::<PyResult<Vec<_>>>()) } }
}

fn _extract_a2_a_door_dispatch_body_size(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<usize>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<usize>(&obj, "body_size")).map(Some) }
}

#[pyclass(frozen, name = "_core_A2ADoor", module = "livekit.agents._core")]
pub struct _core_A2ADoor(pub(crate) ::std::sync::Arc<_core_A2ADoorImpl>);

#[pymethods]
impl _core_A2ADoor {
    /// The door over `storage`'s namespace, which opens its backend on first use; accepted work
    /// is settled by the storage's `aclose`.
    #[new]
    #[pyo3(signature = (storage, config, host))]
    fn new(storage: Bound<'_, PyAny>, config: Bound<'_, PyAny>, host: Bound<'_, PyAny>) -> PyResult<_core_A2ADoor> {
        crate::host::check_pid()?;
        let env = crate::process::env();
        let storage = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::storage::StorageNamespace>(&storage).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("storage", ": expected a native Storage")))?;
        let config = crate::generated::records::a2a_door::door_config_from_py(&config)?;
        let host = crate::generated::a2a::TurnHostProxy::from_host(&host)?;
        let out = <_core_A2ADoorImpl as ::lk_agents_abi::a2a::door::A2ADoor>::new(&env, storage, config, host).map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(_core_A2ADoor(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn max_payload_bytes(&self) -> PyResult<usize> {
        crate::host::check_pid()?;
        let out = <_core_A2ADoorImpl as ::lk_agents_abi::a2a::door::A2ADoor>::max_payload_bytes(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (body, principal, *, version = None, offset = None, body_size = None, host_context = None))]
    fn dispatch<'py>(&self, py: Python<'py>, #[pyo3(from_py_with = _extract_a2_a_door_dispatch_body)] body: ::std::vec::Vec<u8>, principal: Bound<'_, PyAny>, version: ::std::option::Option<::std::string::String>, offset: ::std::option::Option<::std::string::String>, #[pyo3(from_py_with = _extract_a2_a_door_dispatch_body_size)] body_size: ::std::option::Option<usize>, host_context: ::std::option::Option<::lk_agents_abi::HostRef>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let principal = crate::generated::records::a2a_auth::auth_principal_from_py(&principal)?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <_core_A2ADoorImpl as ::lk_agents_abi::a2a::door::A2ADoor>::dispatch(&this, body, principal, version, offset, body_size, host_context)
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(super::a2a::_core_A2AResponse(::std::sync::Arc::new(out)), py))
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <_core_A2ADoorImpl as ::lk_agents_abi::a2a::door::A2ADoor>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/door.rs.
type _core_A2AResponseImpl = <crate::Selected as super::Implementations>::A2AResponse;

#[pyclass(frozen, name = "_core_A2AResponse", module = "livekit.agents._core")]
pub struct _core_A2AResponse(pub(crate) ::std::sync::Arc<_core_A2AResponseImpl>);

#[pymethods]
impl _core_A2AResponse {
    #[getter]
    fn streaming(&self) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = <_core_A2AResponseImpl as ::lk_agents_abi::a2a::door::A2AResponse>::streaming(&self.0);
        Ok(out)
    }

    fn next<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <_core_A2AResponseImpl as ::lk_agents_abi::a2a::door::A2AResponse>::next(&this, )
                .await.map_err(|e: ::lk_agents_abi::a2a::door::DoorError| <::lk_agents_abi::a2a::door::DoorError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out.map(|value| Ok::<_, ::pyo3::PyErr>(crate::generated::records::a2a_door::door_frame_to_py(py, value)?)).transpose()?, py))
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <_core_A2AResponseImpl as ::lk_agents_abi::a2a::door::A2AResponse>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/remote.rs.
type _core_RemoteInteractionImpl = <crate::Selected as super::Implementations>::RemoteInteraction;

#[pyclass(frozen, name = "_core_RemoteInteraction", module = "livekit.agents._core")]
pub struct _core_RemoteInteraction(pub(crate) ::std::sync::Arc<_core_RemoteInteractionImpl>);

#[pymethods]
impl _core_RemoteInteraction {
    /// `project_url`, `api_key` and `api_secret` are the server's LiveKit project, which grants
    /// and verifies a shared thread; empty when there is none.
    #[new]
    #[pyo3(signature = (endpoint, tool_id, function_call, history, project_url, api_key, api_secret))]
    fn new(endpoint: ::std::string::String, tool_id: ::std::string::String, function_call: Bound<'_, PyAny>, history: Bound<'_, PyAny>, project_url: ::std::string::String, api_key: ::std::string::String, api_secret: ::std::string::String) -> PyResult<_core_RemoteInteraction> {
        crate::host::check_pid()?;
        let function_call = crate::generated::records::chat_items::chat_item_from_py(&function_call)?;
        let history = history.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::chat_items::chat_item_from_py(&item) }).collect::<PyResult<Vec<_>>>())?;
        let out = <_core_RemoteInteractionImpl as ::lk_agents_abi::a2a::remote::RemoteInteraction>::new(endpoint, tool_id, function_call, history, project_url, api_key, api_secret).map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(_core_RemoteInteraction(::std::sync::Arc::new(out)))
    }

    /// The call's message; `thread_id` shares the caller's thread under `token`'s grant for it.
    #[pyo3(signature = (arguments, structured, offers_extension, history, thread_id = None, token = None))]
    fn message<'py>(&self, py: Python<'py>, arguments: Bound<'_, PyAny>, structured: bool, offers_extension: bool, history: Bound<'_, PyAny>, thread_id: ::std::option::Option<::std::string::String>, token: ::std::option::Option<::std::string::String>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let arguments = ::lk_agents_abi::python::value_from_py(&arguments, "arguments")?;
        let history = history.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::chat_items::chat_item_from_py(&item) }).collect::<PyResult<Vec<_>>>())?;
        let out = <_core_RemoteInteractionImpl as ::lk_agents_abi::a2a::remote::RemoteInteraction>::message(&self.0, arguments, structured, offers_extension, history, thread_id, token).map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(crate::generated::records::a2a_client::message_to_py(py, out)?)
    }

    /// The person's answer to the retained input request `request_id`, on its task; refused
    /// once that request was settled or superseded.
    #[pyo3(signature = (request_id, text))]
    fn answer<'py>(&self, py: Python<'py>, request_id: ::std::string::String, text: ::std::string::String) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let out = <_core_RemoteInteractionImpl as ::lk_agents_abi::a2a::remote::RemoteInteraction>::answer(&self.0, request_id, text).map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(crate::generated::records::a2a_client::message_to_py(py, out)?)
    }

    #[pyo3(signature = (update))]
    fn update<'py>(&self, py: Python<'py>, update: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let update = crate::generated::records::a2a_client::task_update_from_py(&update)?;
        let out = <_core_RemoteInteractionImpl as ::lk_agents_abi::a2a::remote::RemoteInteraction>::update(&self.0, update).map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(crate::generated::records::a2a_remote::remote_event_to_py(py, out)?)
    }

    fn cancelled(&self) -> PyResult<()> {
        crate::host::check_pid()?;
        <_core_RemoteInteractionImpl as ::lk_agents_abi::a2a::remote::RemoteInteraction>::cancelled(&self.0).map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[getter]
    fn extra<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let out = <_core_RemoteInteractionImpl as ::lk_agents_abi::a2a::remote::RemoteInteraction>::extra(&self.0).map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(::lk_agents_abi::python::extra_to_py(py, &out)?.into_any())
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/remote.rs.
type RemoteConfigurationImpl = <crate::Selected as super::Implementations>::RemoteConfiguration;

/// Resolve an endpoint credential: `token` as given; else, for an endpoint on the server's
/// project's origin (`project_url`, signed with `api_key`/`api_secret`; empty when there is
/// none), a grant for the caller's `thread_id`, or a project credential without one.
#[pyfunction]
#[pyo3(signature = (endpoint, project_url, api_key, api_secret, token = None, thread_id = None))]
pub fn _core_remote_token(endpoint: ::std::string::String, project_url: ::std::string::String, api_key: ::std::string::String, api_secret: ::std::string::String, token: ::std::option::Option<::std::string::String>, thread_id: ::std::option::Option<::std::string::String>) -> PyResult<::std::option::Option<::std::string::String>> {
    crate::host::check_pid()?;
    let out = <RemoteConfigurationImpl as ::lk_agents_abi::a2a::remote::RemoteConfiguration>::_core_remote_token(endpoint, project_url, api_key, api_secret, token, thread_id).map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/a2a/turn.rs.

/// [`::lk_agents_abi::a2a::turn::TurnHost`] over a host `livekit.agents.storage.context._TurnHost`.
#[derive(Clone)]
pub struct TurnHostProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
}

impl TurnHostProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::a2a::turn::TurnHost>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::a2a::turn::TurnHost>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::a2a::turn::TurnHost>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::a2a::turn::TurnHost>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::a2a::turn::TurnHost>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let _ = context;
        Ok(TurnHostProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

impl ::lk_agents_abi::a2a::turn::TurnHost for TurnHostProxy {
    fn handle<'a>(&'a self, ctx: &::lk_agents_abi::a2a::turn::TurnContext) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(::std::boxed::Box<dyn ::lk_agents_abi::StreamSink<::lk_agents_abi::a2a::turn::Verdict>>, ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::a2a::turn::TurnEvent>>), ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let ctx_arg = ::std::clone::Clone::clone(ctx);
        let context = ::lk_agents_abi::agent::HostContext::host_context(ctx).cloned();
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "TurnHost._core_handle", ctx, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [crate::generated::records::storage_turn::turn_context_to_py(py, ctx_arg)?];
                Ok((host.bind(py).getattr("_core_handle")?, ::pyo3::types::PyTuple::new(py, args)?))
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
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("TurnHost._core_handle: the call captured no host context")));
                };
            let o = out.bind(py);
            let out = (crate::host::stream::HostStream::duplex(py, o, context, |py, v| Ok(crate::generated::records::storage_turn::verdict_to_py(py, v)?), |o| Ok(crate::generated::records::storage_turn::turn_event_from_py(&o)?), ::lk_agents_runtime::process::host_clock()).map(|(sink, source)| (::std::boxed::Box::new(sink) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSink<::lk_agents_abi::a2a::turn::Verdict>>, ::std::boxed::Box::new(source) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::a2a::turn::TurnEvent>>))).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }
}
