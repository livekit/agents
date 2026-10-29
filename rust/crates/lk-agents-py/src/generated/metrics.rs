//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `metrics` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/metrics/mod.rs.
type MetricsRecordsImpl = <crate::Selected as super::Implementations>::MetricsRecords;

/// `log_metrics(metrics)` minus the logging: `(message, extra)`, or `None` for do not log
/// (`VADMetrics` and `EOTInferenceMetrics` have no branch in the reference).
#[pyfunction]
#[pyo3(signature = (metrics))]
pub fn metrics_log_fields<'py>(py: Python<'py>, metrics: ::lk_agents_abi::metrics::AgentMetrics) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <MetricsRecordsImpl as ::lk_agents_abi::metrics::MetricsRecords>::metrics_log_fields(metrics);
    Ok(out.map(|value| Ok::<_, ::pyo3::PyErr>({ let (x0, x1,) = value; ::pyo3::types::PyTuple::new(py, [::pyo3::IntoPyObjectExt::into_bound_py_any(x0, py)?, ::lk_agents_abi::python::extra_to_py(py, &x1)?.into_any()])?.into_any() })).transpose()?)
}

/// `_record_turn_metrics(report)` minus the recording. `None` and `{}` yield nothing; a
/// present `0.0` yields a record.
#[pyfunction]
#[pyo3(signature = (report))]
pub fn otel_turn_records<'py>(py: Python<'py>, report: ::std::option::Option<::lk_agents_abi::MetricsReport>) -> PyResult<Bound<'py, PyAny>> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <MetricsRecordsImpl as ::lk_agents_abi::metrics::MetricsRecords>::otel_turn_records(report);
    Ok(::pyo3::types::PyList::new(py, out.into_iter().map(|x| PyResult::Ok({ let (x0, x1, x2, x3,) = x; ::pyo3::types::PyTuple::new(py, [::pyo3::IntoPyObjectExt::into_bound_py_any(x0, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(x1, py)?, crate::generated::records::telemetry::otel_value_to_py(py, x2)?, { let d = ::pyo3::types::PyDict::new(py); for (k, x) in x3 { d.set_item(::pyo3::IntoPyObjectExt::into_bound_py_any(k, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?; } d.into_any() }])?.into_any() })).collect::<PyResult<Vec<_>>>()?)?.into_any())
}

/// `collect_usage(ev)` minus the recording: the counter increments it would make. Empty for
/// unhandled types, unknown types and honest zeros alike (the reference gates on truthiness).
#[pyfunction]
#[pyo3(signature = (metrics))]
pub fn otel_usage_records<'py>(py: Python<'py>, metrics: ::lk_agents_abi::metrics::AgentMetrics) -> PyResult<Bound<'py, PyAny>> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <MetricsRecordsImpl as ::lk_agents_abi::metrics::MetricsRecords>::otel_usage_records(metrics);
    Ok(::pyo3::types::PyList::new(py, out.into_iter().map(|x| PyResult::Ok({ let (x0, x1, x2, x3,) = x; ::pyo3::types::PyTuple::new(py, [::pyo3::IntoPyObjectExt::into_bound_py_any(x0, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(x1, py)?, crate::generated::records::telemetry::otel_value_to_py(py, x2)?, { let d = ::pyo3::types::PyDict::new(py); for (k, x) in x3 { d.set_item(::pyo3::IntoPyObjectExt::into_bound_py_any(k, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)?; } d.into_any() }])?.into_any() })).collect::<PyResult<Vec<_>>>()?)?.into_any())
}
