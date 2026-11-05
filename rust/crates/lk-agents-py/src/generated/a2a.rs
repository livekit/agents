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
type CoreCredentialsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::Credentials;

fn _extract_credentials_project_token_ttl_seconds(
    obj: &Bound<'_, PyAny>,
) -> PyResult<CredentialsProjectTokenTtlSecondsInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "ttl_seconds")
}
type CredentialsProjectTokenTtlSecondsInput = u64;

fn _extract_credentials_thread_grant_token_ttl_seconds(
    obj: &Bound<'_, PyAny>,
) -> PyResult<CredentialsThreadGrantTokenTtlSecondsInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "ttl_seconds")
}
type CredentialsThreadGrantTokenTtlSecondsInput = u64;

#[pyclass(frozen, name = "_core_Credentials", module = "livekit.agents._core")]
pub struct CoreCredentials(pub(crate) ::std::sync::Arc<CoreCredentialsImpl>);

#[pymethods]
impl CoreCredentials {
    #[new]
    #[pyo3(signature = (api_key, api_secret))]
    fn new(
        py: Python<'_>,
        api_key: ::std::string::String,
        api_secret: ::std::string::String,
    ) -> PyResult<CoreCredentials> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <CoreCredentialsImpl as ::lk_agents_abi::a2a::auth::Credentials>::new(
                api_key, api_secret,
            )
        });
        ::pyo3::PyResult::Ok(CoreCredentials(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (token, *, agent_name = "".to_owned(), deployment = "".to_owned()))]
    fn verify<'py>(
        &self,
        py: Python<'py>,
        token: ::std::string::String,
        agent_name: ::std::string::String,
        deployment: ::std::string::String,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreCredentialsImpl as ::lk_agents_abi::a2a::auth::Credentials>::verify(&self.0, token, agent_name, deployment)).map_err(|e: ::lk_agents_abi::a2a::auth::AuthError| <::lk_agents_abi::a2a::auth::AuthError as crate::py_errors::HostException>::to_py_err(&e))?;
        crate::generated::records::a2a_auth::principal_to_py(py, out)
    }

    #[pyo3(signature = (authorization, thread_grant, node_authorized, agent_name, deployment))]
    fn principal<'py>(
        &self,
        py: Python<'py>,
        authorization: ::std::option::Option<::std::string::String>,
        thread_grant: ::std::option::Option<::std::string::String>,
        node_authorized: bool,
        agent_name: ::std::string::String,
        deployment: ::std::string::String,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreCredentialsImpl as ::lk_agents_abi::a2a::auth::Credentials>::principal(&self.0, authorization, thread_grant, node_authorized, agent_name, deployment)).map_err(|e: ::lk_agents_abi::a2a::auth::AuthError| <::lk_agents_abi::a2a::auth::AuthError as crate::py_errors::HostException>::to_py_err(&e))?;
        crate::generated::records::a2a_auth::principal_to_py(py, out)
    }

    #[pyo3(signature = (*, identity = "worker".to_owned(), ttl_seconds = 3600))]
    fn project_token(
        &self,
        py: Python<'_>,
        identity: ::std::string::String,
        #[pyo3(from_py_with = _extract_credentials_project_token_ttl_seconds)]
        ttl_seconds: CredentialsProjectTokenTtlSecondsInput,
    ) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreCredentialsImpl as ::lk_agents_abi::a2a::auth::Credentials>::project_token(&self.0, identity, ttl_seconds)).map_err(|e: ::lk_agents_abi::a2a::auth::AuthError| <::lk_agents_abi::a2a::auth::AuthError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (grant, *, identity = "".to_owned(), ttl_seconds = 3600))]
    fn thread_grant_token(
        &self,
        py: Python<'_>,
        grant: Bound<'_, PyAny>,
        identity: ::std::string::String,
        #[pyo3(from_py_with = _extract_credentials_thread_grant_token_ttl_seconds)]
        ttl_seconds: CredentialsThreadGrantTokenTtlSecondsInput,
    ) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let grant =
            crate::generated::records::a2a_auth::agent_thread_grant_from_py(grant.as_any())?;
        let out = py.detach(|| <CoreCredentialsImpl as ::lk_agents_abi::a2a::auth::Credentials>::thread_grant_token(&self.0, grant, identity, ttl_seconds)).map_err(|e: ::lk_agents_abi::a2a::auth::AuthError| <::lk_agents_abi::a2a::auth::AuthError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/auth.rs.
type AuthPermissionsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::AuthPermissions;

#[pyfunction]
#[pyo3(signature = (grant, operation, thread_id, scope))]
pub fn _core_auth_grant_allows(
    py: Python<'_>,
    grant: Bound<'_, PyAny>,
    operation: Bound<'_, PyAny>,
    thread_id: ::std::string::String,
    scope: ::std::option::Option<::std::string::String>,
) -> PyResult<bool> {
    crate::py_calls::check_pid()?;
    let grant = crate::generated::records::a2a_auth::agent_thread_grant_from_py(grant.as_any())?;
    let operation =
        crate::generated::records::a2a_auth::auth_operation_from_py(operation.as_any())?;
    let out = py.detach(|| <AuthPermissionsImpl as ::lk_agents_abi::a2a::auth::AuthPermissions>::_core_auth_grant_allows(grant, operation, thread_id, scope));
    ::pyo3::PyResult::Ok(out)
}

/// The caller of the console's unauthenticated listener: every thread permission, threads
/// created in the `console` scope.
#[pyfunction]
pub fn _core_unauthenticated_principal<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    crate::py_calls::check_pid()?;
    let out = py.detach(<AuthPermissionsImpl as ::lk_agents_abi::a2a::auth::AuthPermissions>::_core_unauthenticated_principal);
    crate::generated::records::a2a_auth::principal_to_py(py, out)
}

// Declared in rust/crates/lk-agents-abi/src/a2a/card.rs.
type EndpointCardImpl =
    <crate::generated::implementations::Selected as super::Implementations>::EndpointCard;

/// The upstream A2A card's JSON, as the endpoint serves it.
#[pyfunction]
#[pyo3(signature = (descriptor))]
pub fn _core_a2a_agent_card(
    py: Python<'_>,
    descriptor: Bound<'_, PyAny>,
) -> PyResult<::std::string::String> {
    crate::py_calls::check_pid()?;
    let descriptor =
        crate::generated::records::a2a_card::endpoint_descriptor_from_py(descriptor.as_any())?;
    let out = py
        .detach(|| {
            <EndpointCardImpl as ::lk_agents_abi::a2a::card::EndpointCard>::_core_a2a_agent_card(
                descriptor,
            )
        })
        .map_err(|e: ::lk_agents_abi::errors::HostError| {
            <::lk_agents_abi::errors::HostError as crate::py_errors::HostException>::to_py_err(&e)
        })?;
    ::pyo3::PyResult::Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/a2a/client.rs.
type MessageContentImpl =
    <crate::generated::implementations::Selected as super::Implementations>::MessageContent;

/// The text parts of a generated A2A message, joined by newlines.
#[pyfunction]
#[pyo3(signature = (message))]
pub fn _core_a2a_message_text(
    py: Python<'_>,
    message: Bound<'_, PyAny>,
) -> PyResult<::std::string::String> {
    crate::py_calls::check_pid()?;
    let message = crate::generated::records::a2a_client::message_from_py(message.as_any())?;
    let out = py.detach(|| <MessageContentImpl as ::lk_agents_abi::a2a::client::MessageContent>::_core_a2a_message_text(message));
    ::pyo3::PyResult::Ok(out)
}

/// The first object data part of a generated A2A message.
#[pyfunction]
#[pyo3(signature = (message))]
pub fn _core_a2a_message_data<'py>(
    py: Python<'py>,
    message: Bound<'_, PyAny>,
) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
    crate::py_calls::check_pid()?;
    let message = crate::generated::records::a2a_client::message_from_py(message.as_any())?;
    let out = py.detach(|| <MessageContentImpl as ::lk_agents_abi::a2a::client::MessageContent>::_core_a2a_message_data(message));
    out.map(|value| ::pyo3::PyResult::Ok(crate::py_values::extra_to_py(py, &value)?.into_any()))
        .transpose()
}

// Declared in rust/crates/lk-agents-abi/src/a2a/client.rs.
type ReplyImpl = <crate::generated::implementations::Selected as super::Implementations>::Reply;

#[pyclass(frozen, name = "Reply", module = "livekit.agents._core")]
pub struct Reply(pub(crate) ::std::sync::Arc<ReplyImpl>);

#[pymethods]
impl Reply {
    #[getter]
    fn task_id(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out =
            py.detach(|| <ReplyImpl as ::lk_agents_abi::a2a::client::Reply>::task_id(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn thread_id(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out =
            py.detach(|| <ReplyImpl as ::lk_agents_abi::a2a::client::Reply>::thread_id(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    fn __aiter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            match <ReplyImpl as ::lk_agents_abi::a2a::client::Reply>::next(&this).await.map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as crate::py_errors::HostException>::to_py_err(&e))? {
                Some(out) => Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(crate::generated::records::a2a_client::task_update_to_py(py, out)?, py)),
                None => Err(::pyo3::exceptions::PyStopAsyncIteration::new_err(())),
            }
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <ReplyImpl as ::lk_agents_abi::a2a::client::Reply>::aclose(&this).await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/client.rs.
type AgentClientImpl =
    <crate::generated::implementations::Selected as super::Implementations>::AgentClient;

#[pyclass(frozen, name = "AgentClient", module = "livekit.agents._core")]
pub struct AgentClient(pub(crate) ::std::sync::Arc<AgentClientImpl>);

#[pymethods]
impl AgentClient {
    #[new]
    #[pyo3(signature = (endpoint_url, *, token = None, conn_options = None))]
    fn new(
        py: Python<'_>,
        endpoint_url: ::std::string::String,
        token: ::std::option::Option<::std::string::String>,
        conn_options: Option<Bound<'_, PyAny>>,
    ) -> PyResult<AgentClient> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let conn_options = conn_options
            .map(|v| crate::generated::records::connect::api_connect_options_from_py(v.as_any()))
            .transpose()?;
        let out = py.detach(|| <AgentClientImpl as ::lk_agents_abi::a2a::client::AgentClient>::new(&env, endpoint_url, token, conn_options)).map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(AgentClient(::std::sync::Arc::new(out)))
    }

    fn connect<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <AgentClientImpl as ::lk_agents_abi::a2a::client::AgentClient>::connect(&this, )
                .await.map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    crate::generated::records::a2a_client::agent_card_to_py(py, out)?,
                    py,
                )
            })
        })
    }

    #[getter]
    fn offers_extension(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <AgentClientImpl as ::lk_agents_abi::a2a::client::AgentClient>::offers_extension(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (message, *, thread_id = None, task_id = None, metadata = None, chat_ctx = None))]
    fn send<'py>(
        &self,
        py: Python<'py>,
        message: Bound<'_, PyAny>,
        thread_id: ::std::option::Option<::std::string::String>,
        task_id: ::std::option::Option<::std::string::String>,
        metadata: Option<Bound<'_, PyAny>>,
        chat_ctx: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let message =
            crate::generated::records::a2a_client::message_input_from_py(message.as_any())?;
        let metadata = metadata
            .map(|v| crate::generated::records::a2a_client::message_metadata_from_py(v.as_any()))
            .transpose()?;
        let chat_ctx = chat_ctx
            .map(|v| {
                v.try_iter().and_then(|items| {
                    items
                        .map(|item| {
                            let item = item?;
                            crate::generated::records::chat_items::chat_item_from_py(item.as_any())
                        })
                        .collect::<PyResult<Vec<_>>>()
                })
            })
            .transpose()?;
        let cx = crate::py_calls::entry_cx();
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <AgentClientImpl as ::lk_agents_abi::a2a::client::AgentClient>::send(&this, message, thread_id, task_id, metadata, chat_ctx, cx)
                .await.map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    super::a2a::Reply(::std::sync::Arc::new(out)),
                    py,
                )
            })
        })
    }

    #[pyo3(signature = (task_id, thread_id))]
    fn _core_subscribe<'py>(
        &self,
        py: Python<'py>,
        task_id: ::std::string::String,
        thread_id: ::std::string::String,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let cx = crate::py_calls::entry_cx();
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <AgentClientImpl as ::lk_agents_abi::a2a::client::AgentClient>::_core_subscribe(&this, task_id, thread_id, cx)
                .await.map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    super::a2a::Reply(::std::sync::Arc::new(out)),
                    py,
                )
            })
        })
    }

    #[pyo3(signature = (task_id, *, thread_id = None, reason = None))]
    fn cancel<'py>(
        &self,
        py: Python<'py>,
        task_id: ::std::string::String,
        thread_id: ::std::option::Option<::std::string::String>,
        reason: ::std::option::Option<::std::string::String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <AgentClientImpl as ::lk_agents_abi::a2a::client::AgentClient>::cancel(&this, task_id, thread_id, reason)
                .await.map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    crate::generated::records::a2a_client::task_update_to_py(py, out)?,
                    py,
                )
            })
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () =
                <AgentClientImpl as ::lk_agents_abi::a2a::client::AgentClient>::aclose(&this).await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/door.rs.
type CoreA2ADoorImpl =
    <crate::generated::implementations::Selected as super::Implementations>::A2ADoor;

#[pyclass(frozen, name = "_core_A2ADoor", module = "livekit.agents._core")]
pub struct CoreA2ADoor(pub(crate) ::std::sync::Arc<CoreA2ADoorImpl>);

#[pymethods]
impl CoreA2ADoor {
    /// The door over `storage`'s namespace, which opens its backend on first use; accepted work
    /// is settled by the storage's `aclose`.
    #[new]
    #[pyo3(signature = (storage, config, host))]
    fn new(
        py: Python<'_>,
        storage: Bound<'_, PyAny>,
        config: Bound<'_, PyAny>,
        host: Bound<'_, PyAny>,
    ) -> PyResult<CoreA2ADoor> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let storage = crate::py_objects::native::<dyn ::lk_agents_abi::storage::StorageNamespace>(
            storage.as_any(),
        )
        .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "storage",
                ": expected a native Storage"
            ))
        })?;
        let config = crate::generated::records::a2a_door::door_config_from_py(config.as_any())?;
        let host = crate::generated::a2a::TurnHostProxy::from_host(host.as_any())?;
        let out = py.detach(|| <CoreA2ADoorImpl as ::lk_agents_abi::a2a::door::A2ADoor>::new(&env, storage, config, host)).map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(CoreA2ADoor(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn max_payload_bytes(&self, py: Python<'_>) -> PyResult<usize> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <CoreA2ADoorImpl as ::lk_agents_abi::a2a::door::A2ADoor>::max_payload_bytes(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <CoreA2ADoorImpl as ::lk_agents_abi::a2a::door::A2ADoor>::aclose(&this).await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::a2a::door::TextEndpoint`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::a2a::door::TextEndpoint>(),
        read: |obj| {
            let this = obj.cast::<CoreA2ADoor>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::a2a::door::TextEndpoint> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/remote.rs.
type CoreRemoteInteractionImpl =
    <crate::generated::implementations::Selected as super::Implementations>::RemoteInteraction;

/// One delegation of a tool call to a remote agent (docs/agents.md "Remote agents").
#[pyclass(frozen, name = "_core_RemoteInteraction", module = "livekit.agents._core")]
pub struct CoreRemoteInteraction(pub(crate) ::std::sync::Arc<CoreRemoteInteractionImpl>);

#[pymethods]
impl CoreRemoteInteraction {
    /// `project_url`, `api_key` and `api_secret` are the server's LiveKit project, which grants
    /// and verifies a shared thread; empty when there is none.
    #[new]
    #[pyo3(signature = (endpoint, tool_id, function_call, history, project_url, api_key, api_secret))]
    fn new(
        py: Python<'_>,
        endpoint: ::std::string::String,
        tool_id: ::std::string::String,
        function_call: Bound<'_, PyAny>,
        history: Bound<'_, PyAny>,
        project_url: ::std::string::String,
        api_key: ::std::string::String,
        api_secret: ::std::string::String,
    ) -> PyResult<CoreRemoteInteraction> {
        crate::py_calls::check_pid()?;
        let function_call =
            crate::generated::records::chat_items::chat_item_from_py(function_call.as_any())?;
        let history = history.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::generated::records::chat_items::chat_item_from_py(item.as_any())
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        let out = py.detach(|| <CoreRemoteInteractionImpl as ::lk_agents_abi::a2a::remote::RemoteInteraction>::new(endpoint, tool_id, function_call, history, project_url, api_key, api_secret)).map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(CoreRemoteInteraction(::std::sync::Arc::new(out)))
    }

    /// The call's message; `thread_id` shares the caller's thread under `token`'s grant for it.
    #[pyo3(signature = (arguments, structured, offers_extension, history, thread_id = None, token = None))]
    fn message<'py>(
        &self,
        py: Python<'py>,
        arguments: Bound<'_, PyAny>,
        structured: bool,
        offers_extension: bool,
        history: Bound<'_, PyAny>,
        thread_id: ::std::option::Option<::std::string::String>,
        token: ::std::option::Option<::std::string::String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let arguments = crate::py_values::value_from_py(arguments.as_any(), "arguments")?;
        let history = history.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::generated::records::chat_items::chat_item_from_py(item.as_any())
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        let out = py.detach(|| <CoreRemoteInteractionImpl as ::lk_agents_abi::a2a::remote::RemoteInteraction>::message(&self.0, arguments, structured, offers_extension, history, thread_id, token)).map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as crate::py_errors::HostException>::to_py_err(&e))?;
        crate::generated::records::a2a_client::message_to_py(py, out)
    }

    /// Run the delegation on the endpoint: connect with `token`, send the call's message
    /// (`thread_id` shares the caller's thread under `token`'s grant), then follow its updates:
    /// each reaches `host.progress`, a moved task is followed, and on a call (`live`) the
    /// request is announced through `host.delegated` and each question answered through
    /// `host.ask`. Returns the outcome, or, off a call, the expert's question. The client and
    /// its reply are closed before it returns.
    #[pyo3(signature = (host, arguments, structured, history, token, thread_id, conn_options, live))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        host: Bound<'_, PyAny>,
        arguments: Bound<'_, PyAny>,
        structured: bool,
        history: Bound<'_, PyAny>,
        token: ::std::option::Option<::std::string::String>,
        thread_id: ::std::option::Option<::std::string::String>,
        conn_options: Option<Bound<'_, PyAny>>,
        live: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let host = crate::generated::a2a::DelegationHostProxy::from_host(host.as_any())?;
        let arguments = crate::py_values::value_from_py(arguments.as_any(), "arguments")?;
        let history = history.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::generated::records::chat_items::chat_item_from_py(item.as_any())
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        let conn_options = conn_options
            .map(|v| crate::generated::records::connect::api_connect_options_from_py(v.as_any()))
            .transpose()?;
        let context = crate::py_calls::entered();
        let cx = crate::py_calls::entry_cx();
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <CoreRemoteInteractionImpl as ::lk_agents_abi::a2a::remote::RemoteInteraction>::run(&this, &env, host, arguments, structured, history, token, thread_id, conn_options, live, context, cx)
                .await.map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    crate::generated::records::a2a_remote::remote_event_to_py(py, out)?,
                    py,
                )
            })
        })
    }

    /// The tool call was cancelled: the interaction records it and a running delegation stops,
    /// closing its reply (which cancels the remote task) and client; `settled()` waits for that.
    fn cancelled(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <CoreRemoteInteractionImpl as ::lk_agents_abi::a2a::remote::RemoteInteraction>::cancelled(&self.0)).map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// After `cancelled()`: the running delegation's cleanup, bounded by the teardown limit.
    fn settled<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <CoreRemoteInteractionImpl as ::lk_agents_abi::a2a::remote::RemoteInteraction>::settled(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    #[getter]
    fn extra<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreRemoteInteractionImpl as ::lk_agents_abi::a2a::remote::RemoteInteraction>::extra(&self.0)).map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(crate::py_values::extra_to_py(py, &out)?.into_any())
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/remote.rs.
type RemoteConfigurationImpl =
    <crate::generated::implementations::Selected as super::Implementations>::RemoteConfiguration;
type RemoteConfigurationCoreRemoteTokenOutput = ::std::option::Option<::std::string::String>;

/// The delegation tool declared in code: `name` and `description` are required, and the tool
/// takes the request, stated in full. Nothing is fetched from the endpoint.
#[pyfunction]
#[pyo3(signature = (name, description))]
pub fn _core_delegation_tool<'py>(
    py: Python<'py>,
    name: ::std::string::String,
    description: ::std::string::String,
) -> PyResult<Bound<'py, PyAny>> {
    crate::py_calls::check_pid()?;
    let out = py.detach(|| <RemoteConfigurationImpl as ::lk_agents_abi::a2a::remote::RemoteConfiguration>::_core_delegation_tool(name, description)).map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as crate::py_errors::HostException>::to_py_err(&e))?;
    crate::generated::records::a2a_client::agent_card_to_py(py, out)
}

/// Resolve an endpoint credential: `token` as given; else, for an endpoint on the server's
/// project's origin (`project_url`, signed with `api_key`/`api_secret`; empty when there is
/// none), a grant for the caller's `thread_id`, or a project credential without one.
#[pyfunction]
#[pyo3(signature = (endpoint, project_url, api_key, api_secret, token = None, thread_id = None))]
pub fn _core_remote_token(
    py: Python<'_>,
    endpoint: ::std::string::String,
    project_url: ::std::string::String,
    api_key: ::std::string::String,
    api_secret: ::std::string::String,
    token: ::std::option::Option<::std::string::String>,
    thread_id: ::std::option::Option<::std::string::String>,
) -> PyResult<RemoteConfigurationCoreRemoteTokenOutput> {
    crate::py_calls::check_pid()?;
    let out = py.detach(|| <RemoteConfigurationImpl as ::lk_agents_abi::a2a::remote::RemoteConfiguration>::_core_remote_token(endpoint, project_url, api_key, api_secret, token, thread_id)).map_err(|e: ::lk_agents_abi::a2a::client::AgentClientError| <::lk_agents_abi::a2a::client::AgentClientError as crate::py_errors::HostException>::to_py_err(&e))?;
    ::pyo3::PyResult::Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/a2a/turn.rs.
type CoreHandlerTurnImpl =
    <crate::generated::implementations::Selected as super::Implementations>::HandlerTurn;

/// One turn as its handler sees it: the handler awaits each report, and the turn reads them in
/// that order, publishing progress and tokens and answering checkpoints.
///
/// A report waits while the turn holds 64 unread ones, so a handler never runs ahead of the
/// turn. `checkpoint()` returns `True` when the run should yield (a newer message waits under
/// `interrupt`, or the client cancelled the task). Once the turn ended, or was closed, reports
/// are dropped and `checkpoint()` returns `True`.
#[pyclass(frozen, name = "_core_HandlerTurn", module = "livekit.agents._core")]
pub struct CoreHandlerTurn(pub(crate) ::std::sync::Arc<CoreHandlerTurnImpl>);

#[pymethods]
impl CoreHandlerTurn {
    fn checkpoint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out =
                <CoreHandlerTurnImpl as ::lk_agents_abi::a2a::turn::HandlerTurn>::checkpoint(&this)
                    .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[pyo3(signature = (text))]
    fn progress<'py>(
        &self,
        py: Python<'py>,
        text: ::std::string::String,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <CoreHandlerTurnImpl as ::lk_agents_abi::a2a::turn::HandlerTurn>::progress(
                &this, text,
            )
            .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    #[pyo3(signature = (delta))]
    fn token<'py>(
        &self,
        py: Python<'py>,
        delta: ::std::string::String,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <CoreHandlerTurnImpl as ::lk_agents_abi::a2a::turn::HandlerTurn>::token(
                &this, delta,
            )
            .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::a2a::turn::HandlerPort`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::a2a::turn::HandlerPort>(),
        read: |obj| {
            let this = obj.cast::<CoreHandlerTurn>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::a2a::turn::HandlerPort> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/turn.rs.
type TurnImpl = <crate::generated::implementations::Selected as super::Implementations>::Turn;

#[pyclass(frozen, name = "Turn", module = "livekit.agents._core")]
pub struct Turn(pub(crate) ::std::sync::Arc<TurnImpl>);

#[pymethods]
impl Turn {
    #[new]
    #[pyo3(signature = (thread, lane, host, options = None))]
    fn new(
        py: Python<'_>,
        thread: Bound<'_, PyAny>,
        lane: ::std::string::String,
        host: Bound<'_, PyAny>,
        options: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Turn> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let thread = thread
            .cast::<super::storage::Thread>()
            .map(|value| ::std::sync::Arc::clone(&value.get().0))
            .map_err(::pyo3::PyErr::from)?;
        let host = crate::generated::a2a::TurnHostProxy::from_host(host.as_any())?;
        let options = options
            .map(|v| crate::generated::records::storage_options::turn_options_from_py(v.as_any()))
            .transpose()?;
        let out = py.detach(|| {
            <TurnImpl as ::lk_agents_abi::a2a::turn::Turn>::new(&env, thread, lane, host, options)
        });
        ::pyo3::PyResult::Ok(Turn(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn lane(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <TurnImpl as ::lk_agents_abi::a2a::turn::Turn>::lane(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (channel))]
    fn policy_for(
        &self,
        py: Python<'_>,
        channel: ::std::string::String,
    ) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <TurnImpl as ::lk_agents_abi::a2a::turn::Turn>::policy_for(&self.0, channel)
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (message, task_id = None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        message: Bound<'_, PyAny>,
        task_id: ::std::option::Option<::std::string::String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let message = crate::py_values::value_from_py(message.as_any(), "message")?;
        let context = crate::py_calls::entered();
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <TurnImpl as ::lk_agents_abi::a2a::turn::Turn>::run(&this, message, task_id, context)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    crate::generated::records::storage_turn::turn_result_to_py(py, out)?,
                    py,
                )
            })
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/turn.rs.
type HandlerOutcomesImpl =
    <crate::generated::implementations::Selected as super::Implementations>::HandlerOutcomes;

#[pyfunction]
#[pyo3(signature = (output, outcome, runs))]
pub fn normalize_handler_output<'py>(
    py: Python<'py>,
    output: Bound<'_, PyAny>,
    outcome: Bound<'_, PyAny>,
    runs: Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    crate::py_calls::check_pid()?;
    let output = crate::generated::records::storage_turn::handler_output_from_py(output.as_any())?;
    let outcome =
        crate::generated::records::storage_handler::handler_outcome_from_py(outcome.as_any())?;
    let runs = runs.try_iter().and_then(|items| {
        items
            .map(|item| {
                let item = item?;
                crate::generated::records::storage_handler::handler_run_from_py(item.as_any())
            })
            .collect::<PyResult<Vec<_>>>()
    })?;
    let out = py.detach(|| <HandlerOutcomesImpl as ::lk_agents_abi::a2a::turn::HandlerOutcomes>::normalize_handler_output(output, outcome, runs)).map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
    crate::generated::records::storage_turn::handler_output_to_py(py, out)
}

// Declared in rust/crates/lk-agents-abi/src/a2a/remote.rs.

/// [`::lk_agents_abi::a2a::remote::DelegationHost`] over a host `livekit.agents.a2a._Delegation`.
#[derive(Clone)]
pub struct DelegationHostProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
}

impl DelegationHostProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::a2a::remote::DelegationHost>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::a2a::remote::DelegationHost>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::a2a::remote::DelegationHost>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::a2a::remote::DelegationHost>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::a2a::remote::DelegationHost>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let _ = context;
        Ok(DelegationHostProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

impl ::lk_agents_abi::a2a::remote::DelegationHost for DelegationHostProxy {
    fn progress<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        context: &::lk_agents_abi::invoke::CallContext,
        text: ::std::option::Option<::std::string::String>,
        extra: ::serde_json::Map<::std::string::String, ::serde_json::Value>,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = text;
        let extra_arg = extra;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "DelegationHost._core_progress",
            context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(text_arg, py)?,
                    crate::py_values::extra_to_py(py, &extra_arg)?.into_any(),
                ];
                Ok((
                    host.bind(py).getattr("_core_progress")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
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

    fn delegated<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        context: &::lk_agents_abi::invoke::CallContext,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "DelegationHost._core_delegated",
            context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_delegated")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
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

    fn ask<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        context: &::lk_agents_abi::invoke::CallContext,
        question: ::std::string::String,
        verbatim: bool,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<::std::string::String, ::lk_agents_abi::errors::HostError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let question_arg = question;
        let verbatim_arg = verbatim;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "DelegationHost._core_ask",
            context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(question_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(verbatim_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_ask")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
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
                let out = (o.extract::<::std::string::String>()).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/turn.rs.

/// [`::lk_agents_abi::a2a::turn::TurnHost`] over a host `livekit.agents.storage.session._TurnHost`.
#[derive(Clone)]
pub struct TurnHostProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
}

impl TurnHostProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::a2a::turn::TurnHost>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::a2a::turn::TurnHost>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::a2a::turn::TurnHost>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::a2a::turn::TurnHost>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::a2a::turn::TurnHost>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let _ = context;
        Ok(TurnHostProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

impl ::lk_agents_abi::a2a::turn::TurnHost for TurnHostProxy {
    fn handle<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        ctx: &::lk_agents_abi::a2a::turn::TurnContext,
        turn: ::std::sync::Arc<dyn ::lk_agents_abi::a2a::turn::HandlerPort>,
        cancel: ::std::sync::Arc<dyn ::lk_agents_abi::invoke::Cancellation>,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::lk_agents_abi::a2a::turn::HandlerOutput,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let ctx_arg = ::std::clone::Clone::clone(ctx);
        let turn_arg = turn;
        let started = ::lk_agents_runtime::host::call_cancellable(
            &crate::py_calls::Entry,
            "TurnHost._core_handle",
            ctx,
            &[],
            cx,
            cancel,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    crate::generated::records::storage_turn::turn_context_to_py(py, ctx_arg)?,
                    <dyn ::lk_agents_abi::a2a::turn::HandlerPort as super::NativeToHost>::to_host(
                        py, turn_arg,
                    )?,
                ];
                Ok((
                    host.bind(py).getattr("_core_handle")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
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
                let out = (crate::generated::records::storage_turn::handler_output_from_py(o.as_any())).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::a2a::auth::AuthError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Unauthorized(..) => crate::py_errors::host_exception(
                "livekit.agents.a2a",
                "Unauthorized",
                ::std::string::ToString::to_string(self),
            ),
            Self::Forbidden(..) => crate::py_errors::host_exception(
                "livekit.agents.a2a",
                "Forbidden",
                ::std::string::ToString::to_string(self),
            ),
            _ => ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(
                self,
            )),
        }
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::a2a::client::AgentClientError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::InvalidInput(..) => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
            Self::Overlap => crate::py_errors::host_exception(
                "livekit.agents._bindings",
                "StreamStateError",
                ::std::string::ToString::to_string(self),
            ),
            Self::Provider(inner) => crate::py_errors::HostException::to_py_err(inner),
            _ => ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(
                self,
            )),
        }
    }
}
