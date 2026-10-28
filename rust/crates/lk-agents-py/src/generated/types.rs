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
type _core_ProviderAttemptsImpl = <crate::Selected as super::Implementations>::ProviderAttempts;

/// Host callbacks execute attempts authorized by the shared inference request machine.
#[pyclass(name = "_core_ProviderAttempts", module = "livekit.agents._core")]
pub struct _core_ProviderAttempts(_core_ProviderAttemptsImpl);

#[pymethods]
impl _core_ProviderAttempts {
    #[new]
    #[pyo3(signature = (kind, conn_options, now_ns))]
    fn new(kind: ::lk_agents_abi::types::retry::RequestKind, conn_options: ::lk_agents_abi::ApiConnectOptions, now_ns: u64) -> PyResult<_core_ProviderAttempts> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::new(kind, conn_options, now_ns).map_err(|e: ::lk_agents_abi::ApiConnectOptionsError| <::lk_agents_abi::ApiConnectOptionsError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(_core_ProviderAttempts(out))
    }

    fn external_retry_owner(&mut self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::external_retry_owner(&mut self.0);
        Ok(())
    }

    #[pyo3(signature = (now_ns))]
    fn ready(&mut self, now_ns: u64) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::ready(&mut self.0, now_ns).map_err(|e: ::lk_agents_abi::errors::ApiError| <::lk_agents_abi::errors::ApiError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    #[pyo3(signature = (error, produced_output, now_ns))]
    fn failure<'py>(&mut self, py: Python<'py>, error: ::lk_agents_abi::errors::ProviderError, produced_output: bool, now_ns: u64) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::failure(&mut self.0, error, produced_output, now_ns).map_err(|e: ::lk_agents_abi::errors::ApiError| <::lk_agents_abi::errors::ApiError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(crate::generated::records::_retry::_core__retry_decision_to_py(py, out)?)
    }

    fn wake_at(&self) -> PyResult<::std::option::Option<u64>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::wake_at(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (input, now_ns))]
    fn capacity_exceeded(&mut self, input: bool, now_ns: u64) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::capacity_exceeded(&mut self.0, input, now_ns).map_err(|e: ::lk_agents_abi::errors::ApiError| <::lk_agents_abi::errors::ApiError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (now_ns))]
    fn expect_response(&mut self, now_ns: u64) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::expect_response(&mut self.0, now_ns);
        Ok(())
    }

    #[pyo3(signature = (now_ns))]
    fn input_ended(&mut self, now_ns: u64) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::input_ended(&mut self.0, now_ns);
        Ok(())
    }

    #[pyo3(signature = (event, now_ns))]
    fn observe(&mut self, event: ::lk_agents_abi::SpeechEvent, now_ns: u64) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::observe(&mut self.0, event, now_ns);
        Ok(())
    }

    #[pyo3(signature = (now_ns))]
    fn succeeded(&mut self, now_ns: u64) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::succeeded(&mut self.0, now_ns).map_err(|e: ::lk_agents_abi::errors::ApiError| <::lk_agents_abi::errors::ApiError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[getter]
    fn cleanup_timeout(&self) -> PyResult<f64> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::cleanup_timeout(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (now_ns))]
    fn cancel(&mut self, now_ns: u64) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::cancel(&mut self.0, now_ns);
        Ok(())
    }
}
