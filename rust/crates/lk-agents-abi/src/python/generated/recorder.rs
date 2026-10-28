//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(record)]` declarations of `lk-agents-abi`.
//!
//! The `recorder` records.

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/recorder.rs.
static RECORDING_SUMMARY: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::recorder::RecordingSummary {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { bytes_written, started_at, last_flushed, dropped_input_samples, dropped_output_samples } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("bytes_written", ::pyo3::IntoPyObjectExt::into_bound_py_any(bytes_written, py)?)?;
        kwargs.set_item("started_at", match started_at { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("last_flushed", match last_flushed { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("dropped_input_samples", ::pyo3::IntoPyObjectExt::into_bound_py_any(dropped_input_samples, py)?)?;
        kwargs.set_item("dropped_output_samples", ::pyo3::IntoPyObjectExt::into_bound_py_any(dropped_output_samples, py)?)?;
        RECORDING_SUMMARY.import(py, "livekit.agents._generated.recorder", "RecordingSummary")?.call((), Some(&kwargs))
    }
}
