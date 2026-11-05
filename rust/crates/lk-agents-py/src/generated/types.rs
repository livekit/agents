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
type ConnectOptionsFunctionsImpl = <crate::generated::implementations::Selected as super::Implementations>::ConnectOptionsFunctions;

fn _extract_connect_options_functions_api_connect_options_check_max_retry(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectOptionsFunctionsApiConnectOptionsCheckMaxRetryInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "max_retry")
}
type ConnectOptionsFunctionsApiConnectOptionsCheckMaxRetryInput = i64;

fn _extract_connect_options_functions_api_connect_options_check_retry_interval(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectOptionsFunctionsApiConnectOptionsCheckRetryIntervalInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "retry_interval")
}
type ConnectOptionsFunctionsApiConnectOptionsCheckRetryIntervalInput = f64;

fn _extract_connect_options_functions_api_connect_options_check_pool_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectOptionsFunctionsApiConnectOptionsCheckPoolTimeoutInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<f64>(obj.as_any(), "pool_timeout")).map(Some)
    }
}
type ConnectOptionsFunctionsApiConnectOptionsCheckPoolTimeoutInput = ::std::option::Option<f64>;

fn _extract_connect_options_functions_api_connect_options_check_connect_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectOptionsFunctionsApiConnectOptionsCheckConnectTimeoutInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<f64>(obj.as_any(), "connect_timeout")).map(Some)
    }
}
type ConnectOptionsFunctionsApiConnectOptionsCheckConnectTimeoutInput = ::std::option::Option<f64>;

fn _extract_connect_options_functions_api_connect_options_check_write_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectOptionsFunctionsApiConnectOptionsCheckWriteTimeoutInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<f64>(obj.as_any(), "write_timeout")).map(Some)
    }
}
type ConnectOptionsFunctionsApiConnectOptionsCheckWriteTimeoutInput = ::std::option::Option<f64>;

fn _extract_connect_options_functions_api_connect_options_check_read_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectOptionsFunctionsApiConnectOptionsCheckReadTimeoutInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<f64>(obj.as_any(), "read_timeout")).map(Some)
    }
}
type ConnectOptionsFunctionsApiConnectOptionsCheckReadTimeoutInput = ::std::option::Option<f64>;

fn _extract_connect_options_functions_api_connect_options_check_total_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectOptionsFunctionsApiConnectOptionsCheckTotalTimeoutInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<f64>(obj.as_any(), "total_timeout")).map(Some)
    }
}
type ConnectOptionsFunctionsApiConnectOptionsCheckTotalTimeoutInput = ::std::option::Option<f64>;

fn _extract_connect_options_functions_api_retry_interval_retry_interval(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectOptionsFunctionsApiRetryIntervalRetryIntervalInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "retry_interval")
}
type ConnectOptionsFunctionsApiRetryIntervalRetryIntervalInput = f64;

fn _extract_connect_options_functions_api_retry_interval_num_retries(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectOptionsFunctionsApiRetryIntervalNumRetriesInput> {
    crate::py_values::number_from_py::<u32>(obj.as_any(), "num_retries")
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
    py.detach(|| <ConnectOptionsFunctionsImpl as ::lk_agents_abi::types::connect::ConnectOptionsFunctions>::api_connect_options_check(max_retry, retry_interval, pool_timeout, connect_timeout, write_timeout, read_timeout, total_timeout)).map_err(|e: ::lk_agents_abi::ApiConnectOptionsError| <::lk_agents_abi::ApiConnectOptionsError as crate::py_errors::HostException>::to_py_err(&e))?;
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
type ProviderAttemptsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::ProviderAttempts;

/// The attempts of one host provider request (`livekit.agents.ProviderAttempts`).
///
/// `run()` drives them: it waits until an attempt may start, runs the host's attempt, races it
/// against the read and total deadlines, and on failure retries, raises or stops as the shared
/// attempt policy decides. Inside a session the session owns retries (`core = True`): `run()`
/// then runs exactly one attempt, still bounded by its deadlines.
///
/// The observation methods feed the read deadline while an attempt runs: a provider calls
/// `expect_response()` when it sent something that owes an answer, `progress()` when response
/// data arrived (heartbeats are not progress), `audio_received()` for synthesized audio and
/// `complete_response()` when nothing is owed any more.
#[pyclass(frozen, name = "ProviderAttempts", module = "livekit.agents._core")]
pub struct ProviderAttempts(pub(crate) ::std::sync::Arc<ProviderAttemptsImpl>);

#[pymethods]
impl ProviderAttempts {
    #[new]
    #[pyo3(signature = (kind, conn_options, host))]
    fn new(
        py: Python<'_>,
        kind: Bound<'_, PyAny>,
        conn_options: Bound<'_, PyAny>,
        host: Bound<'_, PyAny>,
    ) -> PyResult<ProviderAttempts> {
        crate::py_calls::check_pid()?;
        let kind = crate::py_values::str_from_py(kind.as_any())?;
        let conn_options =
            crate::generated::records::connect::api_connect_options_from_py(conn_options.as_any())?;
        let host = crate::generated::types::AttemptHostProxy::from_host(host.as_any())?;
        let context = crate::py_calls::entered();
        let out = py.detach(|| <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::new(kind, conn_options, host, context)).map_err(|e: ::lk_agents_abi::ApiConnectOptionsError| <::lk_agents_abi::ApiConnectOptionsError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(ProviderAttempts(::std::sync::Arc::new(out)))
    }

    /// Whether the session owns retries; set before the request starts.
    #[getter]
    fn core(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::core(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[setter(core)]
    fn set_core(&self, py: Python<'_>, core: bool) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::set_core(
                &self.0, core,
            )
        });
        Ok(())
    }

    /// Run the request's attempts. Returns the successful attempt's value (`None` when the
    /// provider stopped the request, a 499); raises the attempt's own error when it is not
    /// retried, or `APIConnectionError` from the last error once the retries are exhausted.
    /// Each attempt runs under the caller's current span.
    fn run<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let cx = crate::py_calls::entry_cx();
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::run(&this, cx)
                .await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    out.map(|value| crate::py_objects::object_ref_to_py(py, value)).transpose()?,
                    py,
                )
            })
        })
    }

    /// A request was sent that owes a response; `final`: it was the last input.
    #[pyo3(signature = (*, r#final = false))]
    fn expect_response(&self, py: Python<'_>, r#final: bool) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::expect_response(&self.0, r#final));
        Ok(())
    }

    /// The input ended: its final response is owed.
    fn input_ended(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::input_ended(&self.0));
        Ok(())
    }

    /// Response data arrived: the read deadline restarts.
    fn progress(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::progress(
                &self.0,
            )
        });
        Ok(())
    }

    /// Synthesized audio arrived: it settles the first response, and after the final input it
    /// is progress.
    fn audio_received(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::audio_received(&self.0));
        Ok(())
    }

    /// No response is outstanding: the read deadline disarms.
    fn complete_response(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::complete_response(&self.0));
        Ok(())
    }

    /// A recognition event, for the stream's read deadline and transcript completion.
    #[pyo3(signature = (event))]
    fn observe(&self, py: Python<'_>, event: Bound<'_, PyAny>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let event = crate::generated::records::stt::speech_event_from_py(event.as_any())?;
        py.detach(|| {
            <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::observe(
                &self.0, event,
            )
        });
        Ok(())
    }

    /// Whether retries were exhausted: `run()` then raised `APIConnectionError` from the last
    /// attempt's error, which the request's ending `error` event carries.
    #[getter]
    fn exhausted(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::exhausted(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// After `run()` was cancelled and the host cancelled its running attempt: wait until that
    /// attempt finished its cleanup, at most `cleanup_timeout` (a stalled cleanup is logged and
    /// keeps running). `_bindings.run_attempts` does both.
    fn settled<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::settled(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// How long a cancelled attempt's cleanup may take before it is reported as stalled.
    #[getter]
    fn cleanup_timeout(&self, py: Python<'_>) -> PyResult<f64> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <ProviderAttemptsImpl as ::lk_agents_abi::types::provider::ProviderAttempts>::cleanup_timeout(&self.0));
        ::pyo3::PyResult::Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/types/provider.rs.

/// [`::lk_agents_abi::types::provider::AttemptHost`] over a host `livekit.agents._bindings.AttemptHost`.
#[derive(Clone)]
pub struct AttemptHostProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: crate::py_calls::Notifier,
}

impl AttemptHostProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::types::provider::AttemptHost>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::types::provider::AttemptHost>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::types::provider::AttemptHost>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::types::provider::AttemptHost>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::types::provider::AttemptHost>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::py_calls::capture_context(obj.py())?,
        };
        Ok(AttemptHostProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            context,
            notifier: ::std::default::Default::default(),
        })
    }
}

impl ::lk_agents_abi::types::provider::AttemptHost for AttemptHostProxy {
    fn attempt<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        context: &::lk_agents_abi::invoke::CallContext,
        cancel: ::std::sync::Arc<dyn ::lk_agents_abi::invoke::Cancellation>,
        retry: bool,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::std::option::Option<::lk_agents_abi::HostRef>,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let retry_arg = retry;
        let started = ::lk_agents_runtime::host::call_cancellable(
            &crate::py_calls::Entry,
            "AttemptHost._core_attempt",
            context,
            &[],
            cx,
            cancel,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(retry_arg, py)?];
                Ok((
                    host.bind(py).getattr("_core_attempt")?,
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
                let out = (if o.is_none() { Ok(None) } else { (crate::py_objects::object_ref_from_py(o.as_any())).map(Some) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn produced(&self) -> ::std::result::Result<bool, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_produced")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let out = (o.extract::<bool>()).map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn retrying(&self, error: ::lk_agents_abi::errors::ProviderError) {
        let host = ::std::sync::Arc::clone(&self.host);
        let error_arg = error;
        let error_arg =
            match ::pyo3::Python::attach(|py| -> ::pyo3::PyResult<::pyo3::Py<::pyo3::PyAny>> {
                Ok((crate::py_errors::ExceptionValue::to_exception(error_arg, py)?).unbind())
            }) {
                Ok(value) => value,
                Err(error) => {
                    log::warn!("AttemptHost._core_retrying failed: {error}");
                    return;
                }
            };
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "AttemptHost._core_retrying",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [error_arg.into_bound(py)];
                Ok((
                    host.bind(py).getattr("_core_retrying")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::ApiConnectOptionsError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
    }
}
