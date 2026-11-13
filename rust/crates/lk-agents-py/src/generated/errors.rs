//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `errors` classes.

#![allow(clippy::too_many_arguments, reason = "a host signature takes the parameters it declares")]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/errors/api.rs.
type ApiErrorFunctionsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::ApiErrorFunctions;

fn _extract_api_error_functions_http_error_message_status(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ApiErrorFunctionsHttpErrorMessageStatusInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "status")
}
type ApiErrorFunctionsHttpErrorMessageStatusInput = i64;

fn _extract_api_error_functions_retryable_for_status_status_code(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ApiErrorFunctionsRetryableForStatusStatusCodeInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "status_code")
}
type ApiErrorFunctionsRetryableForStatusStatusCodeInput = i64;

/// The message of an `APIStatusError` built from an HTTP status: the message and the reason
/// when the message adds detail, else just the reason.
#[pyfunction]
#[pyo3(signature = (message, status, reason))]
pub fn http_error_display(
    py: Python<'_>,
    message: ::std::string::String,
    #[pyo3(from_py_with = _extract_api_error_functions_http_error_message_status)]
    status: ApiErrorFunctionsHttpErrorMessageStatusInput,
    reason: ::std::option::Option<::std::string::String>,
) -> PyResult<::std::string::String> {
    let out = py.detach(|| {
        <ApiErrorFunctionsImpl as ::lk_agents_abi::errors::ApiErrorFunctions>::http_error_message(
            &message, status, reason,
        )
    });
    ::pyo3::PyResult::Ok(out)
}

/// `APIStatusError`'s `retryable`: a non-transient 4xx status, 499 among them, never is.
#[pyfunction]
#[pyo3(signature = (status_code, retryable))]
pub fn retryable_for_status(
    py: Python<'_>,
    #[pyo3(from_py_with = _extract_api_error_functions_retryable_for_status_status_code)]
    status_code: ApiErrorFunctionsRetryableForStatusStatusCodeInput,
    retryable: ::std::option::Option<bool>,
) -> PyResult<bool> {
    let out = py.detach(|| {
        <ApiErrorFunctionsImpl as ::lk_agents_abi::errors::ApiErrorFunctions>::retryable_for_status(
            status_code,
            retryable,
        )
    });
    ::pyo3::PyResult::Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/errors/api.rs.
type WebSocketFunctionsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::WebSocketFunctions;

fn _extract_web_socket_functions_websocket_close_kind_code(
    obj: &Bound<'_, PyAny>,
) -> PyResult<WebSocketFunctionsWebsocketCloseKindCodeInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<u16>(obj.as_any(), "code")).map(Some)
    }
}
type WebSocketFunctionsWebsocketCloseKindCodeInput = ::std::option::Option<u16>;

/// What a close code says about the failure: `http.WebSocket.close_error`'s retryability.
#[pyfunction]
#[pyo3(signature = (code))]
pub fn websocket_close_kind<'py>(
    py: Python<'py>,
    #[pyo3(from_py_with = _extract_web_socket_functions_websocket_close_kind_code)]
    code: WebSocketFunctionsWebsocketCloseKindCodeInput,
) -> PyResult<Bound<'py, PyAny>> {
    let out = py.detach(|| <WebSocketFunctionsImpl as ::lk_agents_abi::errors::WebSocketFunctions>::websocket_close_kind(code));
    crate::generated::records::types::close_kind_to_py(py, out)
}

impl crate::py_errors::HostException for ::lk_agents_abi::errors::Invalid {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::errors::ApiError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        let err = match self.kind {
            <::lk_agents_abi::errors::ApiErrorKind>::Api => crate::py_errors::host_exception_with(
                "livekit.agents._bindings",
                "APIError",
                ::std::string::ToString::to_string(self),
                |kwargs| {
                    let py = kwargs.py();
                    ::pyo3::types::PyDictMethods::set_item(
                        kwargs,
                        "retryable",
                        ::pyo3::IntoPyObjectExt::into_bound_py_any(
                            ::std::clone::Clone::clone(&self.retryable),
                            py,
                        )?,
                    )?;
                    ::std::result::Result::Ok(())
                },
            ),
            <::lk_agents_abi::errors::ApiErrorKind>::Status => {
                crate::py_errors::host_exception_with(
                    "livekit.agents._bindings",
                    "APIStatusError",
                    ::std::string::ToString::to_string(self),
                    |kwargs| {
                        let py = kwargs.py();
                        ::pyo3::types::PyDictMethods::set_item(
                            kwargs,
                            "retryable",
                            ::pyo3::IntoPyObjectExt::into_bound_py_any(
                                ::std::clone::Clone::clone(&self.retryable),
                                py,
                            )?,
                        )?;
                        ::pyo3::types::PyDictMethods::set_item(
                            kwargs,
                            "status_code",
                            match ::std::clone::Clone::clone(&self.status_code) {
                                None => py.None().into_bound(py),
                                Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?,
                            },
                        )?;
                        ::pyo3::types::PyDictMethods::set_item(
                            kwargs,
                            "request_id",
                            match ::std::clone::Clone::clone(&self.request_id) {
                                None => py.None().into_bound(py),
                                Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?,
                            },
                        )?;
                        ::std::result::Result::Ok(())
                    },
                )
            }
            <::lk_agents_abi::errors::ApiErrorKind>::Cancelled => {
                crate::py_errors::host_exception_with(
                    "livekit.agents._bindings",
                    "APIStatusError",
                    ::std::string::ToString::to_string(self),
                    |kwargs| {
                        let py = kwargs.py();
                        ::pyo3::types::PyDictMethods::set_item(
                            kwargs,
                            "retryable",
                            ::pyo3::IntoPyObjectExt::into_bound_py_any(
                                ::std::clone::Clone::clone(&self.retryable),
                                py,
                            )?,
                        )?;
                        ::pyo3::types::PyDictMethods::set_item(
                            kwargs,
                            "status_code",
                            match ::std::clone::Clone::clone(&self.status_code) {
                                None => py.None().into_bound(py),
                                Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?,
                            },
                        )?;
                        ::pyo3::types::PyDictMethods::set_item(
                            kwargs,
                            "request_id",
                            match ::std::clone::Clone::clone(&self.request_id) {
                                None => py.None().into_bound(py),
                                Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?,
                            },
                        )?;
                        ::std::result::Result::Ok(())
                    },
                )
            }
            <::lk_agents_abi::errors::ApiErrorKind>::Connection => {
                crate::py_errors::host_exception_with(
                    "livekit.agents._bindings",
                    "APIConnectionError",
                    ::std::string::ToString::to_string(self),
                    |kwargs| {
                        let py = kwargs.py();
                        ::pyo3::types::PyDictMethods::set_item(
                            kwargs,
                            "retryable",
                            ::pyo3::IntoPyObjectExt::into_bound_py_any(
                                ::std::clone::Clone::clone(&self.retryable),
                                py,
                            )?,
                        )?;
                        ::std::result::Result::Ok(())
                    },
                )
            }
            <::lk_agents_abi::errors::ApiErrorKind>::Timeout => {
                crate::py_errors::host_exception_with(
                    "livekit.agents._bindings",
                    "APITimeoutError",
                    ::std::string::ToString::to_string(self),
                    |kwargs| {
                        let py = kwargs.py();
                        ::pyo3::types::PyDictMethods::set_item(
                            kwargs,
                            "retryable",
                            ::pyo3::IntoPyObjectExt::into_bound_py_any(
                                ::std::clone::Clone::clone(&self.retryable),
                                py,
                            )?,
                        )?;
                        ::pyo3::types::PyDictMethods::set_item(
                            kwargs,
                            "timeout",
                            match ::std::clone::Clone::clone(&self.timeout) {
                                None => py.None().into_bound(py),
                                Some(x) => crate::generated::records::types::timeout_to_py(py, x)?,
                            },
                        )?;
                        ::std::result::Result::Ok(())
                    },
                )
            }
        };
        if let ::std::option::Option::Some(cause) = &self.cause {
            ::pyo3::Python::attach(|py| {
                err.set_cause(
                    py,
                    ::std::option::Option::Some(crate::py_errors::HostException::to_py_err(
                        &**cause,
                    )),
                )
            });
        }
        err
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::errors::ProviderError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        match self {
            Self::Host { error, .. } => crate::py_errors::HostException::to_py_err(error),
            Self::Local(inner) => crate::py_errors::HostException::to_py_err(inner),
        }
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::errors::PullError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        match self {
            Self::Overlap => crate::py_errors::host_exception(
                "livekit.agents._bindings",
                "StreamStateError",
                ::std::string::ToString::to_string(self),
            ),
            Self::Failed(inner) => crate::py_errors::HostException::to_py_err(inner),
        }
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::errors::HostError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        crate::py_errors::raise_host_error(::std::clone::Clone::clone(self))
    }
}
