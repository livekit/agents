//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(record)]` declarations of `lk-agents-abi`.
//!
//! The `telemetry` proxy bridges and in-place records.

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/telemetry/report.rs.
static RECORDING_OPTIONS: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::telemetry::report::RecordingOptions {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(RECORDING_OPTIONS.import(py, "livekit.agents._generated.telemetry", "RecordingOptions")?)? {
            return Err(super::not_a(obj, "RecordingOptions"));
        }
        Ok(Self {
            audio: ::lk_agents_abi::python::attr_with(obj, "RecordingOptions", "audio", |v| v.extract::<bool>())?,
            traces: ::lk_agents_abi::python::attr_with(obj, "RecordingOptions", "traces", |v| v.extract::<bool>())?,
            logs: ::lk_agents_abi::python::attr_with(obj, "RecordingOptions", "logs", |v| v.extract::<bool>())?,
            transcript: ::lk_agents_abi::python::attr_with(obj, "RecordingOptions", "transcript", |v| v.extract::<bool>())?,
            redaction: ::lk_agents_abi::python::attr_with(obj, "RecordingOptions", "redaction", |v| v.extract::<bool>())?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::telemetry::report::RecordingOptions {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { audio, traces, logs, transcript, redaction } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("audio", ::pyo3::IntoPyObjectExt::into_bound_py_any(audio, py)?)?;
        kwargs.set_item("traces", ::pyo3::IntoPyObjectExt::into_bound_py_any(traces, py)?)?;
        kwargs.set_item("logs", ::pyo3::IntoPyObjectExt::into_bound_py_any(logs, py)?)?;
        kwargs.set_item("transcript", ::pyo3::IntoPyObjectExt::into_bound_py_any(transcript, py)?)?;
        kwargs.set_item("redaction", ::pyo3::IntoPyObjectExt::into_bound_py_any(redaction, py)?)?;
        RECORDING_OPTIONS.import(py, "livekit.agents._generated.telemetry", "RecordingOptions")?.call((), Some(&kwargs))
    }
}
