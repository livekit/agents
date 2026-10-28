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

// Declared in rust/crates/lk-agents-abi/src/telemetry/report.rs.
static EVALUATION_RECORD: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::EvaluationRecord {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { name, tag, verdict, reasoning, instructions } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("name", ::pyo3::IntoPyObjectExt::into_bound_py_any(name, py)?)?;
        kwargs.set_item("tag", ::pyo3::IntoPyObjectExt::into_bound_py_any(tag, py)?)?;
        kwargs.set_item("verdict", ::pyo3::IntoPyObjectExt::into_bound_py_any(verdict, py)?)?;
        kwargs.set_item("reasoning", ::pyo3::IntoPyObjectExt::into_bound_py_any(reasoning, py)?)?;
        kwargs.set_item("instructions", ::pyo3::IntoPyObjectExt::into_bound_py_any(instructions, py)?)?;
        EVALUATION_RECORD.import(py, "livekit.agents._generated.telemetry", "EvaluationRecord")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/report.rs.
static SPAN_UPDATE_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::telemetry::report::SpanUpdateWire {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { span, attrs } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("span", ::pyo3::IntoPyObjectExt::into_bound_py_any(span, py)?)?;
        kwargs.set_item("attrs", ::pyo3::IntoPyObjectExt::into_bound_py_any(attrs, py)?)?;
        SPAN_UPDATE_WIRE.import(py, "livekit.agents._generated.telemetry", "SpanUpdateWire")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/report.rs.
static SPAN_UPDATE_ATTRS_WIRE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::telemetry::report::SpanUpdateAttrsWire {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { is_interruption, probability, total_duration, prediction_duration, detection_delay } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("is_interruption", ::pyo3::IntoPyObjectExt::into_bound_py_any(is_interruption, py)?)?;
        kwargs.set_item("probability", ::pyo3::IntoPyObjectExt::into_bound_py_any(probability, py)?)?;
        kwargs.set_item("total_duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(total_duration, py)?)?;
        kwargs.set_item("prediction_duration", ::pyo3::IntoPyObjectExt::into_bound_py_any(prediction_duration, py)?)?;
        kwargs.set_item("detection_delay", ::pyo3::IntoPyObjectExt::into_bound_py_any(detection_delay, py)?)?;
        SPAN_UPDATE_ATTRS_WIRE.import(py, "livekit.agents._generated.telemetry", "SpanUpdateAttrsWire")?.call((), Some(&kwargs))
    }
}

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
