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
type RuntimeLoadImpl =
    <crate::generated::implementations::Selected as super::Implementations>::RuntimeLoad;
type RuntimeLoadRuntimeBusySecondsOutput = ::std::vec::Vec<f64>;

/// Each async worker's total busy seconds so far; empty before the runtime started.
#[pyfunction]
pub fn runtime_busy_seconds(py: Python<'_>) -> PyResult<RuntimeLoadRuntimeBusySecondsOutput> {
    let out =
        py.detach(<RuntimeLoadImpl as ::lk_agents_abi::metrics::RuntimeLoad>::runtime_busy_seconds);
    ::pyo3::PyResult::Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/metrics/mod.rs.
type MetricsRecordsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::MetricsRecords;

/// `log_metrics(metrics)` minus the logging: `(message, extra)`, or `None` for do not log
/// (`VADMetrics` and `EOTInferenceMetrics` have no branch in the reference).
#[pyfunction]
#[pyo3(signature = (metrics))]
pub fn metrics_log_fields<'py>(
    py: Python<'py>,
    metrics: Bound<'_, PyAny>,
) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
    crate::py_calls::check_pid()?;
    let metrics = crate::generated::records::metrics::agent_metrics_from_py(metrics.as_any())?;
    let out = py.detach(|| {
        <MetricsRecordsImpl as ::lk_agents_abi::metrics::MetricsRecords>::metrics_log_fields(
            metrics,
        )
    });
    out.map(|value| {
        ::pyo3::PyResult::Ok({
            let (x0, x1) = value;
            ::pyo3::types::PyTuple::new(
                py,
                [
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(x0, py)?,
                    crate::py_values::extra_to_py(py, &x1)?.into_any(),
                ],
            )?
            .into_any()
        })
    })
    .transpose()
}
