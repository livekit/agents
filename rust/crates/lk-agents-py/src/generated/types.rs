//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `types` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/types/provider.rs.
type ProviderAttemptsImpl = <crate::Selected as super::Implementations>::ProviderAttempts;

/// Host callbacks execute attempts authorized by the shared inference request machine.
#[pyclass(name = "ProviderAttempts", module = "livekit.agents._core")]
pub struct ProviderAttempts(ProviderAttemptsImpl);

#[pymethods]
impl ProviderAttempts {
    #[new]
    #[pyo3(signature = (kind, conn_options, now_ns))]
    fn new(kind: ::lk_agents_abi::types::retry::RequestKind, conn_options: ::lk_agents_abi::ApiConnectOptions, now_ns: u64) -> PyResult<ProviderAttempts> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::new(kind, conn_options, now_ns).map_err(|e: ::lk_agents_abi::ApiConnectOptionsError| <::lk_agents_abi::ApiConnectOptionsError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(ProviderAttempts(out))
    }

    fn external_retry_owner(&mut self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::external_retry_owner(&mut self.0);
        Ok(())
    }

    #[pyo3(signature = (now_ns))]
    fn ready(&mut self, now_ns: u64) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::ready(&mut self.0, now_ns).map_err(|e: ::lk_agents_abi::errors::ApiError| <::lk_agents_abi::errors::ApiError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    #[pyo3(signature = (retryable, status, produced_output, now_ns))]
    fn failure(&mut self, retryable: bool, status: ::std::option::Option<i64>, produced_output: bool, now_ns: u64) -> PyResult<(::std::string::String, f64, ::std::option::Option<::std::string::String>)> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::failure(&mut self.0, retryable, status, produced_output, now_ns).map_err(|e: ::lk_agents_abi::errors::ApiError| <::lk_agents_abi::errors::ApiError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    fn wake_at(&self) -> PyResult<::std::option::Option<u64>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::wake_at(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (now_ns))]
    fn expect_response(&mut self, now_ns: u64) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::expect_response(&mut self.0, now_ns);
        Ok(())
    }

    #[pyo3(signature = (now_ns))]
    fn input_ended(&mut self, now_ns: u64) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::input_ended(&mut self.0, now_ns);
        Ok(())
    }

    #[pyo3(signature = (event, now_ns))]
    fn observe(&mut self, event: ::lk_agents_abi::SpeechEvent, now_ns: u64) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::observe(&mut self.0, event, now_ns);
        Ok(())
    }

    #[pyo3(signature = (now_ns))]
    fn succeeded(&mut self, now_ns: u64) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::succeeded(&mut self.0, now_ns).map_err(|e: ::lk_agents_abi::errors::ApiError| <::lk_agents_abi::errors::ApiError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[getter]
    fn cleanup_timeout(&self) -> PyResult<f64> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::cleanup_timeout(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (now_ns))]
    fn cancel(&mut self, now_ns: u64) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::cancel(&mut self.0, now_ns);
        Ok(())
    }
}
