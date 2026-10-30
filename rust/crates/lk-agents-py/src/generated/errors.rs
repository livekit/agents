//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `errors` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/errors/api.rs.
type ApiErrorFunctionsImpl = <crate::Selected as super::Implementations>::ApiErrorFunctions;

fn _extract_api_error_functions_http_error_message_status(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    crate::python::number_from_py::<i64>(&obj, "status")
}

/// The message of an `APIStatusError` built from an HTTP status: the message and the reason
/// when the message adds detail, else just the reason.
#[pyfunction]
#[pyo3(signature = (message, status, reason))]
pub fn http_error_display(
    py: Python<'_>,
    message: ::std::string::String,
    #[pyo3(from_py_with = _extract_api_error_functions_http_error_message_status)] status: i64,
    reason: ::std::option::Option<::std::string::String>,
) -> PyResult<::std::string::String> {
    let out = py.detach(|| {
        <ApiErrorFunctionsImpl as ::lk_agents_abi::errors::ApiErrorFunctions>::http_error_message(
            &message, status, reason,
        )
    });
    Ok(out)
}

impl crate::python::HostException for ::lk_agents_abi::errors::ApiError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        let err = match self.kind {
            <::lk_agents_abi::errors::ApiErrorKind>::Api => crate::python::host_exception_with(
                "livekit.agents._exceptions",
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
            <::lk_agents_abi::errors::ApiErrorKind>::Status => crate::python::host_exception_with(
                "livekit.agents._exceptions",
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
            ),
            <::lk_agents_abi::errors::ApiErrorKind>::Cancelled => {
                crate::python::host_exception_with(
                    "livekit.agents._exceptions",
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
                crate::python::host_exception_with(
                    "livekit.agents._exceptions",
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
            <::lk_agents_abi::errors::ApiErrorKind>::Timeout => crate::python::host_exception_with(
                "livekit.agents._exceptions",
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
                        "phase",
                        match ::std::clone::Clone::clone(&self.phase) {
                            None => py.None().into_bound(py),
                            Some(x) => crate::python::str_to_py(py, &x),
                        },
                    )?;
                    ::std::result::Result::Ok(())
                },
            ),
        };
        if let ::std::option::Option::Some(cause) = &self.cause {
            ::pyo3::Python::attach(|py| {
                err.set_cause(
                    py,
                    ::std::option::Option::Some(crate::python::HostException::to_py_err(&**cause)),
                )
            });
        }
        err
    }
}

impl crate::python::HostException for ::lk_agents_abi::errors::ProviderError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Host { error, .. } => crate::python::HostException::to_py_err(error),
            Self::Local(inner) => crate::python::HostException::to_py_err(inner),
            _ => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
        }
    }
}

impl crate::python::HostException for ::lk_agents_abi::errors::PullError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Overlap => crate::python::host_exception(
                "livekit.agents._exceptions",
                "StreamStateError",
                ::std::string::ToString::to_string(self),
            ),
            Self::Failed(inner) => crate::python::HostException::to_py_err(inner),
            _ => ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(
                self,
            )),
        }
    }
}

impl crate::python::HostException for ::lk_agents_abi::errors::HostError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        crate::python::raise_host_error(::std::clone::Clone::clone(self))
    }
}
