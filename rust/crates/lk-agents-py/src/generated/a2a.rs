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

#[pyclass(frozen, name = "_core_Credentials", module = "livekit.agents._core")]
pub struct _core_Credentials(::std::sync::Arc<_core_CredentialsImpl>);

#[pymethods]
impl _core_Credentials {
    #[new]
    #[pyo3(signature = (api_key, api_secret))]
    fn new(api_key: ::std::string::String, api_secret: ::std::string::String) -> PyResult<_core_Credentials> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_CredentialsImpl as ::lk_agents_abi::a2a::auth::Credentials>::new(api_key, api_secret);
        Ok(_core_Credentials(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (token, *, agent_name = "".to_owned(), deployment = "".to_owned()))]
    fn verify(&self, token: ::std::string::String, agent_name: ::std::string::String, deployment: ::std::string::String) -> PyResult<::lk_agents_abi::a2a::auth::AuthPrincipal> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_CredentialsImpl as ::lk_agents_abi::a2a::auth::Credentials>::verify(&self.0, token, agent_name, deployment).map_err(|e: ::lk_agents_abi::a2a::auth::AuthError| <::lk_agents_abi::a2a::auth::AuthError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    #[pyo3(signature = (authorization, thread_grant, node_authorized, agent_name, deployment))]
    fn principal(&self, authorization: ::std::option::Option<::std::string::String>, thread_grant: ::std::option::Option<::std::string::String>, node_authorized: bool, agent_name: ::std::string::String, deployment: ::std::string::String) -> PyResult<::lk_agents_abi::a2a::auth::AuthPrincipal> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_CredentialsImpl as ::lk_agents_abi::a2a::auth::Credentials>::principal(&self.0, authorization, thread_grant, node_authorized, agent_name, deployment).map_err(|e: ::lk_agents_abi::a2a::auth::AuthError| <::lk_agents_abi::a2a::auth::AuthError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    #[pyo3(signature = (*, identity = "worker".to_owned(), ttl_seconds = 3600))]
    fn project_token(&self, identity: ::std::string::String, ttl_seconds: u64) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_CredentialsImpl as ::lk_agents_abi::a2a::auth::Credentials>::project_token(&self.0, identity, ttl_seconds).map_err(|e: ::lk_agents_abi::a2a::auth::AuthError| <::lk_agents_abi::a2a::auth::AuthError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    #[pyo3(signature = (grant, *, identity = "".to_owned(), ttl_seconds = 3600))]
    fn thread_grant_token(&self, grant: ::lk_agents_abi::a2a::auth::AuthGrant, identity: ::std::string::String, ttl_seconds: u64) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_CredentialsImpl as ::lk_agents_abi::a2a::auth::Credentials>::thread_grant_token(&self.0, grant, identity, ttl_seconds).map_err(|e: ::lk_agents_abi::a2a::auth::AuthError| <::lk_agents_abi::a2a::auth::AuthError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/a2a/auth.rs.
type AuthPermissionsImpl = <crate::Selected as super::Implementations>::AuthPermissions;

#[pyfunction]
#[pyo3(signature = (grant, operation, thread_id, scope))]
pub fn _core_auth_grant_allows(grant: ::lk_agents_abi::a2a::auth::AuthGrant, operation: ::lk_agents_abi::a2a::auth::AuthOperation, thread_id: ::std::string::String, scope: ::std::option::Option<::std::string::String>) -> PyResult<bool> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <AuthPermissionsImpl as ::lk_agents_abi::a2a::auth::AuthPermissions>::_core_auth_grant_allows(grant, operation, thread_id, scope);
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/a2a/turn.rs.

/// [`::lk_agents_abi::a2a::turn::TurnHost`] over a host `livekit.agents.storage.context.TurnHost`.
#[derive(Clone)]
pub struct TurnHostProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
}

impl TurnHostProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::a2a::turn::TurnHost>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::a2a::turn::TurnHost>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(TurnHostProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: ::lk_agents_runtime::host::python::capture_context(obj.py())? }))
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::a2a::turn::TurnHost>(|obj| TurnHostProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

#[::async_trait::async_trait]
impl ::lk_agents_abi::a2a::turn::TurnHost for TurnHostProxy {
    async fn handle(&self, ctx: ::lk_agents_abi::a2a::turn::TurnContext) -> ::std::result::Result<(::std::boxed::Box<dyn ::lk_agents_abi::StreamSink<::lk_agents_abi::a2a::turn::Verdict>>, ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::a2a::turn::TurnEvent>>), ::lk_agents_abi::errors::ProviderError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let ctx_arg = ctx;
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(ctx_arg, py)?];
                Ok((host.bind(py).getattr("_core_handle")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (::lk_agents_runtime::host::stream::HostStream::duplex(py, o, context, |py, v| Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(v, py)?), |o| Ok(o.extract::<::lk_agents_abi::a2a::turn::TurnEvent>()?), ::lk_agents_runtime::process::host_clock()).map(|(sink, source)| (::std::boxed::Box::new(sink) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSink<::lk_agents_abi::a2a::turn::Verdict>>, ::std::boxed::Box::new(source) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::a2a::turn::TurnEvent>>))).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}
