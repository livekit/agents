//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `types` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/types/connect.rs.
type ConnectOptionsFunctionsImpl = <crate::Selected as super::Implementations>::ConnectOptionsFunctions;

fn _extract_connect_options_functions_api_connect_options_check_max_retry(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    ::lk_agents_abi::python::number_from_py::<i64>(&obj, "max_retry")
}

fn _extract_connect_options_functions_api_connect_options_check_retry_interval(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "retry_interval")
}

fn _extract_connect_options_functions_api_connect_options_check_pool_timeout(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(&obj, "pool_timeout")).map(Some) }
}

fn _extract_connect_options_functions_api_connect_options_check_connect_timeout(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(&obj, "connect_timeout")).map(Some) }
}

fn _extract_connect_options_functions_api_connect_options_check_write_timeout(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(&obj, "write_timeout")).map(Some) }
}

fn _extract_connect_options_functions_api_connect_options_check_read_timeout(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(&obj, "read_timeout")).map(Some) }
}

fn _extract_connect_options_functions_api_connect_options_check_total_timeout(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(&obj, "total_timeout")).map(Some) }
}

fn _extract_connect_options_functions_api_retry_interval_retry_interval(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "retry_interval")
}

fn _extract_connect_options_functions_api_retry_interval_num_retries(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    ::lk_agents_abi::python::number_from_py::<u32>(&obj, "num_retries")
}

/// The record as the host built it, checked at the boundary; the error names the field.
#[pyfunction]
#[pyo3(signature = (max_retry, retry_interval, pool_timeout, connect_timeout, write_timeout, read_timeout, total_timeout))]
pub fn api_connect_options_check(#[pyo3(from_py_with = _extract_connect_options_functions_api_connect_options_check_max_retry)] max_retry: i64, #[pyo3(from_py_with = _extract_connect_options_functions_api_connect_options_check_retry_interval)] retry_interval: f64, #[pyo3(from_py_with = _extract_connect_options_functions_api_connect_options_check_pool_timeout)] pool_timeout: ::std::option::Option<f64>, #[pyo3(from_py_with = _extract_connect_options_functions_api_connect_options_check_connect_timeout)] connect_timeout: ::std::option::Option<f64>, #[pyo3(from_py_with = _extract_connect_options_functions_api_connect_options_check_write_timeout)] write_timeout: ::std::option::Option<f64>, #[pyo3(from_py_with = _extract_connect_options_functions_api_connect_options_check_read_timeout)] read_timeout: ::std::option::Option<f64>, #[pyo3(from_py_with = _extract_connect_options_functions_api_connect_options_check_total_timeout)] total_timeout: ::std::option::Option<f64>) -> PyResult<()> {
    <ConnectOptionsFunctionsImpl as ::lk_agents_abi::types::connect::ConnectOptionsFunctions>::api_connect_options_check(max_retry, retry_interval, pool_timeout, connect_timeout, write_timeout, read_timeout, total_timeout).map_err(|e: ::lk_agents_abi::ApiConnectOptionsError| <::lk_agents_abi::ApiConnectOptionsError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(())
}

/// `APIConnectOptions.interval_for_retry` for the host's retry loops; only `retry_interval`
/// affects the answer.
#[pyfunction]
#[pyo3(signature = (retry_interval, num_retries))]
pub fn api_retry_interval(#[pyo3(from_py_with = _extract_connect_options_functions_api_retry_interval_retry_interval)] retry_interval: f64, #[pyo3(from_py_with = _extract_connect_options_functions_api_retry_interval_num_retries)] num_retries: u32) -> PyResult<f64> {
    let out = <ConnectOptionsFunctionsImpl as ::lk_agents_abi::types::connect::ConnectOptionsFunctions>::api_retry_interval(retry_interval, num_retries);
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/types/provider.rs.
type _core_ProviderAttemptsImpl = <crate::Selected as super::Implementations>::ProviderAttempts;

fn _extract_provider_attempts_new_now_ns(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "now_ns")
}

fn _extract_provider_attempts_ready_now_ns(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "now_ns")
}

fn _extract_provider_attempts_failure_now_ns(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "now_ns")
}

fn _extract_provider_attempts_capacity_exceeded_now_ns(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "now_ns")
}

fn _extract_provider_attempts_expect_response_now_ns(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "now_ns")
}

fn _extract_provider_attempts_input_ended_now_ns(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "now_ns")
}

fn _extract_provider_attempts_observe_now_ns(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "now_ns")
}

fn _extract_provider_attempts_observe_response_now_ns(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "now_ns")
}

fn _extract_provider_attempts_succeeded_now_ns(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "now_ns")
}

fn _extract_provider_attempts_cancel_now_ns(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "now_ns")
}

/// Host callbacks execute attempts authorized by the shared inference request machine.
#[pyclass(name = "_core_ProviderAttempts", module = "livekit.agents._core")]
pub struct _core_ProviderAttempts(pub(crate) _core_ProviderAttemptsImpl);

#[pymethods]
impl _core_ProviderAttempts {
    #[new]
    #[pyo3(signature = (kind, conn_options, now_ns))]
    fn new(kind: ::lk_agents_abi::types::retry::RequestKind, conn_options: ::lk_agents_abi::ApiConnectOptions, #[pyo3(from_py_with = _extract_provider_attempts_new_now_ns)] now_ns: u64) -> PyResult<_core_ProviderAttempts> {
        crate::host::check_pid()?;
        let out = <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::new(kind, conn_options, now_ns).map_err(|e: ::lk_agents_abi::ApiConnectOptionsError| <::lk_agents_abi::ApiConnectOptionsError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(_core_ProviderAttempts(out))
    }

    fn external_retry_owner(&mut self) -> PyResult<()> {
        crate::host::check_pid()?;
        <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::external_retry_owner(&mut self.0);
        Ok(())
    }

    #[pyo3(signature = (now_ns))]
    fn ready(&mut self, #[pyo3(from_py_with = _extract_provider_attempts_ready_now_ns)] now_ns: u64) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::ready(&mut self.0, now_ns).map_err(|e: ::lk_agents_abi::errors::ApiError| <::lk_agents_abi::errors::ApiError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    #[pyo3(signature = (error, produced_output, now_ns))]
    fn failure<'py>(&mut self, py: Python<'py>, error: ::lk_agents_abi::errors::ProviderError, produced_output: bool, #[pyo3(from_py_with = _extract_provider_attempts_failure_now_ns)] now_ns: u64) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let out = <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::failure(&mut self.0, error, produced_output, now_ns).map_err(|e: ::lk_agents_abi::errors::ApiError| <::lk_agents_abi::errors::ApiError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(crate::generated::records::_retry::_core__retry_decision_to_py(py, out)?)
    }

    fn wake_at(&self) -> PyResult<::std::option::Option<u64>> {
        crate::host::check_pid()?;
        let out = <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::wake_at(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (input, now_ns))]
    fn capacity_exceeded(&mut self, input: bool, #[pyo3(from_py_with = _extract_provider_attempts_capacity_exceeded_now_ns)] now_ns: u64) -> PyResult<()> {
        crate::host::check_pid()?;
        <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::capacity_exceeded(&mut self.0, input, now_ns).map_err(|e: ::lk_agents_abi::errors::ApiError| <::lk_agents_abi::errors::ApiError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (now_ns))]
    fn expect_response(&mut self, #[pyo3(from_py_with = _extract_provider_attempts_expect_response_now_ns)] now_ns: u64) -> PyResult<()> {
        crate::host::check_pid()?;
        <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::expect_response(&mut self.0, now_ns);
        Ok(())
    }

    #[pyo3(signature = (now_ns))]
    fn input_ended(&mut self, #[pyo3(from_py_with = _extract_provider_attempts_input_ended_now_ns)] now_ns: u64) -> PyResult<()> {
        crate::host::check_pid()?;
        <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::input_ended(&mut self.0, now_ns);
        Ok(())
    }

    #[pyo3(signature = (event, now_ns))]
    fn observe(&mut self, event: ::lk_agents_abi::SpeechEvent, #[pyo3(from_py_with = _extract_provider_attempts_observe_now_ns)] now_ns: u64) -> PyResult<()> {
        crate::host::check_pid()?;
        <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::observe(&mut self.0, event, now_ns);
        Ok(())
    }

    #[pyo3(signature = (observation, now_ns))]
    fn observe_response(&mut self, observation: ::lk_agents_abi::types::provider::ResponseObservation, #[pyo3(from_py_with = _extract_provider_attempts_observe_response_now_ns)] now_ns: u64) -> PyResult<()> {
        crate::host::check_pid()?;
        <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::observe_response(&mut self.0, observation, now_ns);
        Ok(())
    }

    #[pyo3(signature = (now_ns))]
    fn succeeded(&mut self, #[pyo3(from_py_with = _extract_provider_attempts_succeeded_now_ns)] now_ns: u64) -> PyResult<()> {
        crate::host::check_pid()?;
        <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::succeeded(&mut self.0, now_ns).map_err(|e: ::lk_agents_abi::errors::ApiError| <::lk_agents_abi::errors::ApiError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[getter]
    fn cleanup_timeout(&self) -> PyResult<f64> {
        crate::host::check_pid()?;
        let out = <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::cleanup_timeout(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (now_ns))]
    fn cancel(&mut self, #[pyo3(from_py_with = _extract_provider_attempts_cancel_now_ns)] now_ns: u64) -> PyResult<()> {
        crate::host::check_pid()?;
        <_core_ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::cancel(&mut self.0, now_ns);
        Ok(())
    }
}
