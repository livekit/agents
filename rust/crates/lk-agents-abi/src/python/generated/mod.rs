//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(record)]` declarations of `lk-agents-abi`.
//!
//! Each record crosses as an instance of its generated model.

mod agent;
mod llm;
mod storage;
mod telemetry;

/// `obj` is not an instance of the model(s) `expected` names.
fn not_a(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, expected: &str) -> ::pyo3::PyErr {
    use ::pyo3::types::{PyAnyMethods as _, PyTypeMethods as _};
    let got = obj.get_type().name().map(|n| n.to_string()).unwrap_or_default();
    ::pyo3::exceptions::PyTypeError::new_err(format!("expected {expected}, got {got}"))
}
