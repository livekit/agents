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
type ConnectOptionsFunctionsImpl =
    <crate::Selected as super::Implementations>::ConnectOptionsFunctions;

fn _extract_connect_options_functions_api_connect_options_check_max_retry(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectOptionsFunctionsApiConnectOptionsCheckMaxRetryInput> {
    crate::python::number_from_py::<i64>(obj.as_any(), "max_retry")
}
type ConnectOptionsFunctionsApiConnectOptionsCheckMaxRetryInput = i64;

fn _extract_connect_options_functions_api_connect_options_check_retry_interval(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectOptionsFunctionsApiConnectOptionsCheckRetryIntervalInput> {
    crate::python::number_from_py::<f64>(obj.as_any(), "retry_interval")
}
type ConnectOptionsFunctionsApiConnectOptionsCheckRetryIntervalInput = f64;

fn _extract_connect_options_functions_api_connect_options_check_pool_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectOptionsFunctionsApiConnectOptionsCheckPoolTimeoutInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::python::number_from_py::<f64>(obj.as_any(), "pool_timeout")).map(Some)
    }
}
type ConnectOptionsFunctionsApiConnectOptionsCheckPoolTimeoutInput = ::std::option::Option<f64>;

fn _extract_connect_options_functions_api_connect_options_check_connect_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectOptionsFunctionsApiConnectOptionsCheckConnectTimeoutInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::python::number_from_py::<f64>(obj.as_any(), "connect_timeout")).map(Some)
    }
}
type ConnectOptionsFunctionsApiConnectOptionsCheckConnectTimeoutInput = ::std::option::Option<f64>;

fn _extract_connect_options_functions_api_connect_options_check_write_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectOptionsFunctionsApiConnectOptionsCheckWriteTimeoutInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::python::number_from_py::<f64>(obj.as_any(), "write_timeout")).map(Some)
    }
}
type ConnectOptionsFunctionsApiConnectOptionsCheckWriteTimeoutInput = ::std::option::Option<f64>;

fn _extract_connect_options_functions_api_connect_options_check_read_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectOptionsFunctionsApiConnectOptionsCheckReadTimeoutInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::python::number_from_py::<f64>(obj.as_any(), "read_timeout")).map(Some)
    }
}
type ConnectOptionsFunctionsApiConnectOptionsCheckReadTimeoutInput = ::std::option::Option<f64>;

fn _extract_connect_options_functions_api_connect_options_check_total_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectOptionsFunctionsApiConnectOptionsCheckTotalTimeoutInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::python::number_from_py::<f64>(obj.as_any(), "total_timeout")).map(Some)
    }
}
type ConnectOptionsFunctionsApiConnectOptionsCheckTotalTimeoutInput = ::std::option::Option<f64>;

fn _extract_connect_options_functions_api_retry_interval_retry_interval(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectOptionsFunctionsApiRetryIntervalRetryIntervalInput> {
    crate::python::number_from_py::<f64>(obj.as_any(), "retry_interval")
}
type ConnectOptionsFunctionsApiRetryIntervalRetryIntervalInput = f64;

fn _extract_connect_options_functions_api_retry_interval_num_retries(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectOptionsFunctionsApiRetryIntervalNumRetriesInput> {
    crate::python::number_from_py::<u32>(obj.as_any(), "num_retries")
}
type ConnectOptionsFunctionsApiRetryIntervalNumRetriesInput = u32;

/// The record as the host built it, checked at the boundary; the error names the field.
#[pyfunction]
#[pyo3(signature = (max_retry, retry_interval, pool_timeout, connect_timeout, write_timeout, read_timeout, total_timeout))]
pub fn api_connect_options_check(
    py: Python<'_>,
    #[pyo3(from_py_with = _extract_connect_options_functions_api_connect_options_check_max_retry)]
    max_retry: ConnectOptionsFunctionsApiConnectOptionsCheckMaxRetryInput,
    #[pyo3(from_py_with = _extract_connect_options_functions_api_connect_options_check_retry_interval)]
    retry_interval: ConnectOptionsFunctionsApiConnectOptionsCheckRetryIntervalInput,
    #[pyo3(from_py_with = _extract_connect_options_functions_api_connect_options_check_pool_timeout)]
    pool_timeout: ConnectOptionsFunctionsApiConnectOptionsCheckPoolTimeoutInput,
    #[pyo3(from_py_with = _extract_connect_options_functions_api_connect_options_check_connect_timeout)]
    connect_timeout: ConnectOptionsFunctionsApiConnectOptionsCheckConnectTimeoutInput,
    #[pyo3(from_py_with = _extract_connect_options_functions_api_connect_options_check_write_timeout)]
    write_timeout: ConnectOptionsFunctionsApiConnectOptionsCheckWriteTimeoutInput,
    #[pyo3(from_py_with = _extract_connect_options_functions_api_connect_options_check_read_timeout)]
    read_timeout: ConnectOptionsFunctionsApiConnectOptionsCheckReadTimeoutInput,
    #[pyo3(from_py_with = _extract_connect_options_functions_api_connect_options_check_total_timeout)]
    total_timeout: ConnectOptionsFunctionsApiConnectOptionsCheckTotalTimeoutInput,
) -> PyResult<()> {
    py.detach(|| <ConnectOptionsFunctionsImpl as ::lk_agents_abi::types::connect::ConnectOptionsFunctions>::api_connect_options_check(max_retry, retry_interval, pool_timeout, connect_timeout, write_timeout, read_timeout, total_timeout)).map_err(|e: ::lk_agents_abi::ApiConnectOptionsError| <::lk_agents_abi::ApiConnectOptionsError as crate::python::HostException>::to_py_err(&e))?;
    Ok(())
}

/// `APIConnectOptions.interval_for_retry` for the host's retry loops; only `retry_interval`
/// affects the answer.
#[pyfunction]
#[pyo3(signature = (retry_interval, num_retries))]
pub fn api_retry_interval(
    py: Python<'_>,
    #[pyo3(from_py_with = _extract_connect_options_functions_api_retry_interval_retry_interval)]
    retry_interval: ConnectOptionsFunctionsApiRetryIntervalRetryIntervalInput,
    #[pyo3(from_py_with = _extract_connect_options_functions_api_retry_interval_num_retries)]
    num_retries: ConnectOptionsFunctionsApiRetryIntervalNumRetriesInput,
) -> PyResult<f64> {
    let out = py.detach(|| <ConnectOptionsFunctionsImpl as ::lk_agents_abi::types::connect::ConnectOptionsFunctions>::api_retry_interval(retry_interval, num_retries));
    ::pyo3::PyResult::Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/types/provider.rs.
type CoreProviderAttemptsImpl = <crate::Selected as super::Implementations>::ProviderAttempts;

fn _extract_provider_attempts_new_now_ns(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ProviderAttemptsNewNowNsInput> {
    crate::python::number_from_py::<u64>(obj.as_any(), "now_ns")
}
type ProviderAttemptsNewNowNsInput = u64;

fn _extract_provider_attempts_ready_now_ns(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ProviderAttemptsReadyNowNsInput> {
    crate::python::number_from_py::<u64>(obj.as_any(), "now_ns")
}
type ProviderAttemptsReadyNowNsInput = u64;

fn _extract_provider_attempts_failure_now_ns(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ProviderAttemptsFailureNowNsInput> {
    crate::python::number_from_py::<u64>(obj.as_any(), "now_ns")
}
type ProviderAttemptsFailureNowNsInput = u64;
type ProviderAttemptsWakeAtOutput = ::std::option::Option<u64>;

fn _extract_provider_attempts_capacity_exceeded_now_ns(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ProviderAttemptsCapacityExceededNowNsInput> {
    crate::python::number_from_py::<u64>(obj.as_any(), "now_ns")
}
type ProviderAttemptsCapacityExceededNowNsInput = u64;

fn _extract_provider_attempts_expect_response_now_ns(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ProviderAttemptsExpectResponseNowNsInput> {
    crate::python::number_from_py::<u64>(obj.as_any(), "now_ns")
}
type ProviderAttemptsExpectResponseNowNsInput = u64;

fn _extract_provider_attempts_input_ended_now_ns(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ProviderAttemptsInputEndedNowNsInput> {
    crate::python::number_from_py::<u64>(obj.as_any(), "now_ns")
}
type ProviderAttemptsInputEndedNowNsInput = u64;

fn _extract_provider_attempts_observe_now_ns(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ProviderAttemptsObserveNowNsInput> {
    crate::python::number_from_py::<u64>(obj.as_any(), "now_ns")
}
type ProviderAttemptsObserveNowNsInput = u64;

fn _extract_provider_attempts_observe_response_now_ns(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ProviderAttemptsObserveResponseNowNsInput> {
    crate::python::number_from_py::<u64>(obj.as_any(), "now_ns")
}
type ProviderAttemptsObserveResponseNowNsInput = u64;

fn _extract_provider_attempts_succeeded_now_ns(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ProviderAttemptsSucceededNowNsInput> {
    crate::python::number_from_py::<u64>(obj.as_any(), "now_ns")
}
type ProviderAttemptsSucceededNowNsInput = u64;

fn _extract_provider_attempts_cancel_now_ns(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ProviderAttemptsCancelNowNsInput> {
    crate::python::number_from_py::<u64>(obj.as_any(), "now_ns")
}
type ProviderAttemptsCancelNowNsInput = u64;

/// Host callbacks execute attempts authorized by the shared inference request machine.
#[pyclass(name = "_core_ProviderAttempts", module = "livekit.agents._core")]
pub struct CoreProviderAttempts(pub(crate) CoreProviderAttemptsImpl);

#[pymethods]
impl CoreProviderAttempts {
    #[new]
    #[pyo3(signature = (kind, conn_options, now_ns))]
    fn new(
        py: Python<'_>,
        kind: Bound<'_, PyAny>,
        conn_options: Bound<'_, PyAny>,
        #[pyo3(from_py_with = _extract_provider_attempts_new_now_ns)]
        now_ns: ProviderAttemptsNewNowNsInput,
    ) -> PyResult<CoreProviderAttempts> {
        crate::host::check_pid()?;
        let kind = crate::python::str_from_py(kind.as_any())?;
        let conn_options = crate::conversions::api_connect_options_from_py(conn_options.as_any())?;
        let out = py.detach(|| <CoreProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::new(kind, conn_options, now_ns)).map_err(|e: ::lk_agents_abi::ApiConnectOptionsError| <::lk_agents_abi::ApiConnectOptionsError as crate::python::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(CoreProviderAttempts(out))
    }

    fn external_retry_owner(&mut self) -> PyResult<()> {
        crate::host::check_pid()?;
        <CoreProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::external_retry_owner(&mut self.0);
        Ok(())
    }

    #[pyo3(signature = (now_ns))]
    fn ready(
        &mut self,
        #[pyo3(from_py_with = _extract_provider_attempts_ready_now_ns)]
        now_ns: ProviderAttemptsReadyNowNsInput,
    ) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = <CoreProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::ready(&mut self.0, now_ns).map_err(|e: ::lk_agents_abi::errors::ApiError| <::lk_agents_abi::errors::ApiError as crate::python::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (error, produced_output, now_ns))]
    fn failure<'py>(
        &mut self,
        py: Python<'py>,
        error: Bound<'_, PyAny>,
        produced_output: bool,
        #[pyo3(from_py_with = _extract_provider_attempts_failure_now_ns)]
        now_ns: ProviderAttemptsFailureNowNsInput,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let error = PyResult::Ok(crate::python::capture_value(&error))?;
        let out = <CoreProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::failure(&mut self.0, error, produced_output, now_ns).map_err(|e: ::lk_agents_abi::errors::ApiError| <::lk_agents_abi::errors::ApiError as crate::python::HostException>::to_py_err(&e))?;
        crate::generated::records::_retry::core_retry_decision_to_py(py, out)
    }

    fn wake_at(&self) -> PyResult<ProviderAttemptsWakeAtOutput> {
        crate::host::check_pid()?;
        let out = <CoreProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::wake_at(&self.0);
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (input, now_ns))]
    fn capacity_exceeded(
        &mut self,
        input: bool,
        #[pyo3(from_py_with = _extract_provider_attempts_capacity_exceeded_now_ns)]
        now_ns: ProviderAttemptsCapacityExceededNowNsInput,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        <CoreProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::capacity_exceeded(&mut self.0, input, now_ns).map_err(|e: ::lk_agents_abi::errors::ApiError| <::lk_agents_abi::errors::ApiError as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (now_ns))]
    fn expect_response(
        &mut self,
        #[pyo3(from_py_with = _extract_provider_attempts_expect_response_now_ns)]
        now_ns: ProviderAttemptsExpectResponseNowNsInput,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        <CoreProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::expect_response(&mut self.0, now_ns);
        Ok(())
    }

    #[pyo3(signature = (now_ns))]
    fn input_ended(
        &mut self,
        #[pyo3(from_py_with = _extract_provider_attempts_input_ended_now_ns)]
        now_ns: ProviderAttemptsInputEndedNowNsInput,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        <CoreProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::input_ended(&mut self.0, now_ns);
        Ok(())
    }

    #[pyo3(signature = (event, now_ns))]
    fn observe(
        &mut self,
        event: Bound<'_, PyAny>,
        #[pyo3(from_py_with = _extract_provider_attempts_observe_now_ns)]
        now_ns: ProviderAttemptsObserveNowNsInput,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        let event = crate::generated::records::stt::speech_event_from_py(event.as_any())?;
        <CoreProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::observe(
            &mut self.0,
            event,
            now_ns,
        );
        Ok(())
    }

    #[pyo3(signature = (observation, now_ns))]
    fn observe_response(
        &mut self,
        observation: Bound<'_, PyAny>,
        #[pyo3(from_py_with = _extract_provider_attempts_observe_response_now_ns)]
        now_ns: ProviderAttemptsObserveResponseNowNsInput,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        let observation = crate::python::str_from_py(observation.as_any())?;
        <CoreProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::observe_response(&mut self.0, observation, now_ns);
        Ok(())
    }

    #[pyo3(signature = (now_ns))]
    fn succeeded(
        &mut self,
        #[pyo3(from_py_with = _extract_provider_attempts_succeeded_now_ns)]
        now_ns: ProviderAttemptsSucceededNowNsInput,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        <CoreProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::succeeded(&mut self.0, now_ns).map_err(|e: ::lk_agents_abi::errors::ApiError| <::lk_agents_abi::errors::ApiError as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[getter]
    fn cleanup_timeout(&self) -> PyResult<f64> {
        crate::host::check_pid()?;
        let out = <CoreProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::cleanup_timeout(&self.0);
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (now_ns))]
    fn cancel(
        &mut self,
        #[pyo3(from_py_with = _extract_provider_attempts_cancel_now_ns)]
        now_ns: ProviderAttemptsCancelNowNsInput,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        <CoreProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::cancel(
            &mut self.0,
            now_ns,
        );
        Ok(())
    }
}

impl crate::python::HostException for ::lk_agents_abi::ApiConnectOptionsError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
    }
}
