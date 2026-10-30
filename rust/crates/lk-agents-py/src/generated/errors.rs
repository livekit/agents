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
    ::lk_agents_abi::python::number_from_py::<i64>(&obj, "status")
}

/// The message of an `APIStatusError` built from an HTTP status: the message and the reason
/// when the message adds detail, else just the reason.
#[pyfunction]
#[pyo3(signature = (message, status, reason))]
pub fn http_error_display(py: Python<'_>, message: ::std::string::String, #[pyo3(from_py_with = _extract_api_error_functions_http_error_message_status)] status: i64, reason: ::std::option::Option<::std::string::String>) -> PyResult<::std::string::String> {
    let out = py.detach(|| <ApiErrorFunctionsImpl as ::lk_agents_abi::errors::ApiErrorFunctions>::http_error_message(&message, status, reason));
    Ok(out)
}
