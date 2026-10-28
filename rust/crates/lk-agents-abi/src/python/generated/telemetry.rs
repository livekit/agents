//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(record)]` declarations of `lk-agents-abi`.
//!
//! The `telemetry` proxy bridges and in-place records.

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/telemetry/report/records.rs.
static LOG_RECORD_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::LogRecordWire {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { logger, body, timestamp_nanos, severity_number, severity_text, attributes } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("logger", ::pyo3::IntoPyObjectExt::into_bound_py_any(logger, py)?)?;
        kwargs.set_item("body", ::pyo3::IntoPyObjectExt::into_bound_py_any(body, py)?)?;
        kwargs.set_item("timestamp_nanos", ::pyo3::IntoPyObjectExt::into_bound_py_any(timestamp_nanos, py)?)?;
        kwargs.set_item("severity_number", ::pyo3::IntoPyObjectExt::into_bound_py_any(severity_number, py)?)?;
        kwargs.set_item("severity_text", ::pyo3::IntoPyObjectExt::into_bound_py_any(severity_text, py)?)?;
        kwargs.set_item("attributes", ::lk_agents_abi::python::extra_to_py(py, &attributes)?.into_any())?;
        LOG_RECORD_WIRE.import(py, "livekit.agents._generated.telemetry", "LogRecordWire")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/loop_monitor.rs.

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::telemetry::loop_monitor::LoopBlockSeverity {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let value = match self {
            Self::Warning => "warning",
            Self::Error => "error",
        };
        Ok(::pyo3::types::PyString::new(py, value).into_any())
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/loop_monitor.rs.
static BLOCKED_REPORT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::telemetry::loop_monitor::BlockedReport {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { duration, started_at, warn_threshold, severity, gc_time, cpu_time, watchdog_gap, process_descheduled, task_name, stacks, importing, parent_span_context, location } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(duration, py)?)?;
        kwargs.set_item("started_at", ::pyo3::IntoPyObjectExt::into_bound_py_any(started_at, py)?)?;
        kwargs.set_item("warn_threshold", ::pyo3::IntoPyObjectExt::into_bound_py_any(warn_threshold, py)?)?;
        kwargs.set_item("severity", ::pyo3::IntoPyObjectExt::into_bound_py_any(severity, py)?)?;
        kwargs.set_item("gc_time", ::pyo3::IntoPyObjectExt::into_bound_py_any(gc_time, py)?)?;
        kwargs.set_item("cpu_time", ::pyo3::IntoPyObjectExt::into_bound_py_any(cpu_time, py)?)?;
        kwargs.set_item("watchdog_gap", ::pyo3::IntoPyObjectExt::into_bound_py_any(watchdog_gap, py)?)?;
        kwargs.set_item("process_descheduled", ::pyo3::IntoPyObjectExt::into_bound_py_any(process_descheduled, py)?)?;
        kwargs.set_item("task_name", match task_name { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("stacks", ::pyo3::types::PyList::new(py, stacks.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())?;
        kwargs.set_item("importing", match importing { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("parent_span_context", match parent_span_context { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("location", match location { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        BLOCKED_REPORT.import(py, "livekit.agents._generated.telemetry", "BlockedReport")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/loop_monitor.rs.
static LOOP_MONITOR_SPAN: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::telemetry::loop_monitor::LoopMonitorSpan {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { attributes, error } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("attributes", ::lk_agents_abi::python::extra_to_py(py, &attributes)?.into_any())?;
        kwargs.set_item("error", match error { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        LOOP_MONITOR_SPAN.import(py, "livekit.agents._generated.telemetry", "LoopMonitorSpan")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/loop_monitor.rs.
static LOOP_MONITOR_LOG: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::telemetry::loop_monitor::LoopMonitorLog {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { message, extra } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("message", ::pyo3::IntoPyObjectExt::into_bound_py_any(message, py)?)?;
        kwargs.set_item("extra", ::lk_agents_abi::python::extra_to_py(py, &extra)?.into_any())?;
        LOOP_MONITOR_LOG.import(py, "livekit.agents._generated.telemetry", "LoopMonitorLog")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/loop_monitor.rs.
static LOOP_MONITOR_EMISSION: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::telemetry::loop_monitor::LoopMonitorEmission {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { report, span, log, reported } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("report", ::pyo3::IntoPyObjectExt::into_bound_py_any(report, py)?)?;
        kwargs.set_item("span", match span { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("log", match log { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        kwargs.set_item("reported", ::pyo3::IntoPyObjectExt::into_bound_py_any(reported, py)?)?;
        LOOP_MONITOR_EMISSION.import(py, "livekit.agents._generated.telemetry", "LoopMonitorEmission")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/transport.rs.
static TRACE_CARRIER: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::telemetry::transport::TraceCarrier {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(TRACE_CARRIER.import(py, "livekit.agents._generated.telemetry_transport", "TraceCarrier")?)? {
            return Err(super::not_a(obj, "TraceCarrier"));
        }
        Ok(Self {
            traceparent: ::lk_agents_abi::python::attr_with(obj, "TraceCarrier", "traceparent", |v| if v.is_none() { Ok(None) } else { (v.extract::<::std::string::String>()).map(Some) })?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::telemetry::transport::TraceCarrier {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { traceparent } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("traceparent", match traceparent { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        TRACE_CARRIER.import(py, "livekit.agents._generated.telemetry_transport", "TraceCarrier")?.call((), Some(&kwargs))
    }
}
