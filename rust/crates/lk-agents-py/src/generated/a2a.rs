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
