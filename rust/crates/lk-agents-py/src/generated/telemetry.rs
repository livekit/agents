//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `telemetry` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/telemetry/policy.rs.
type _core_TelemetryPolicyImpl = <crate::Selected as super::Implementations>::TelemetryPolicy;

fn _extract_telemetry_policy_export_action_held_count(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "held_count")
}

fn _extract_telemetry_policy_value_bytes_length(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "length")
}

fn _extract_telemetry_policy_value_bytes_depth(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "depth")
}

fn _extract_telemetry_policy_value_bytes_children(obj: &Bound<'_, PyAny>) -> PyResult<::std::vec::Vec<usize>> {
    obj.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<usize>(&item, "children") }).collect::<PyResult<Vec<_>>>())
}

fn _extract_telemetry_policy_record_bytes_fields(obj: &Bound<'_, PyAny>) -> PyResult<::std::vec::Vec<usize>> {
    obj.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<usize>(&item, "fields") }).collect::<PyResult<Vec<_>>>())
}

fn _extract_telemetry_policy_record_bytes_event_count(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "event_count")
}

fn _extract_telemetry_policy_record_bytes_link_count(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "link_count")
}

/// The host supplies SDK fields and lengths; shared policy decides what may be retained/exported.
#[pyclass(frozen, name = "_core_TelemetryPolicy", module = "livekit.agents._core")]
pub struct _core_TelemetryPolicy(pub(crate) ::std::sync::Arc<_core_TelemetryPolicyImpl>);

#[pymethods]
impl _core_TelemetryPolicy {
    #[new]
    #[pyo3(signature = (allow_pii, capture_content, project_redaction))]
    fn new(allow_pii: bool, capture_content: bool, project_redaction: bool) -> PyResult<_core_TelemetryPolicy> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::new(allow_pii, capture_content, project_redaction);
        Ok(_core_TelemetryPolicy(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn redacted(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::redacted(&self.0);
        Ok(out)
    }

    #[getter]
    fn max_depth(&self) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::max_depth(&self.0);
        Ok(out)
    }

    #[getter]
    fn max_collection(&self) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::max_collection(&self.0);
        Ok(out)
    }

    #[getter]
    fn max_events(&self) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::max_events(&self.0);
        Ok(out)
    }

    #[getter]
    fn max_record_bytes(&self) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::max_record_bytes(&self.0);
        Ok(out)
    }

    #[getter]
    fn max_pending(&self) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::max_pending(&self.0);
        Ok(out)
    }

    #[getter]
    fn max_records(&self) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::max_records(&self.0);
        Ok(out)
    }

    #[getter]
    fn batch_items(&self) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::batch_items(&self.0);
        Ok(out)
    }

    #[getter]
    fn schedule_delay_millis(&self) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::schedule_delay_millis(&self.0);
        Ok(out)
    }

    #[getter]
    fn export_timeout_millis(&self) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::export_timeout_millis(&self.0);
        Ok(out)
    }

    #[getter]
    fn max_providers(&self) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::max_providers(&self.0);
        Ok(out)
    }

    #[getter]
    fn shutdown_timeout_seconds(&self) -> PyResult<f64> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::shutdown_timeout_seconds(&self.0);
        Ok(out)
    }

    #[getter]
    fn report_retained_bytes(&self) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::report_retained_bytes(&self.0);
        Ok(out)
    }

    #[getter]
    fn upload_timeout_seconds(&self) -> PyResult<f64> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::upload_timeout_seconds(&self.0);
        Ok(out)
    }

    /// The destination may export, hold, drop, refuse capacity or stop after recording is disabled.
    #[pyo3(signature = (registered, signal_enabled, job_open, held_count, recording_disabled))]
    fn export_action<'py>(&self, py: Python<'py>, registered: bool, signal_enabled: bool, job_open: bool, #[pyo3(from_py_with = _extract_telemetry_policy_export_action_held_count)] held_count: usize, recording_disabled: bool) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::export_action(&self.0, registered, signal_enabled, job_open, held_count, recording_disabled);
        Ok(crate::generated::records::telemetry::telemetry_export_action_to_py(py, out)?)
    }

    #[pyo3(signature = (key))]
    fn is_pii_attribute(&self, key: ::std::string::String) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::is_pii_attribute(&self.0, key);
        Ok(out)
    }

    /// Whether to keep the field and, if present, the replacement for its value.
    #[pyo3(signature = (parent, key, destination))]
    fn attribute_action(&self, parent: ::std::option::Option<::std::string::String>, key: ::std::string::String, destination: bool) -> PyResult<(bool, ::std::option::Option<::std::string::String>)> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::attribute_action(&self.0, parent, key, destination);
        Ok(out)
    }

    #[pyo3(signature = (name))]
    fn keep_event(&self, name: ::std::string::String) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::keep_event(&self.0, name);
        Ok(out)
    }

    /// A replacement body, or None to snapshot the original SDK value.
    #[pyo3(signature = (safe_body, body))]
    fn log_body(&self, safe_body: bool, body: ::std::option::Option<::std::string::String>) -> PyResult<::std::option::Option<::std::string::String>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::log_body(&self.0, safe_body, body);
        Ok(out)
    }

    #[pyo3(signature = (is_error, description))]
    fn status_description(&self, is_error: bool, description: ::std::option::Option<::std::string::String>) -> PyResult<::std::option::Option<::std::string::String>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::status_description(&self.0, is_error, description);
        Ok(out)
    }

    /// Conservative retained bytes for an SDK scalar or collection. Child charges are already
    /// bounded verdicts; unsupported kinds, depth and collection overflow refuse the record.
    #[pyo3(signature = (kind, length, depth, children))]
    fn value_bytes(&self, kind: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_telemetry_policy_value_bytes_length)] length: usize, #[pyo3(from_py_with = _extract_telemetry_policy_value_bytes_depth)] depth: usize, #[pyo3(from_py_with = _extract_telemetry_policy_value_bytes_children)] children: ::std::vec::Vec<usize>) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let kind = crate::generated::records::telemetry::telemetry_value_kind_from_py(&kind)?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::value_bytes(&self.0, kind, length, depth, children);
        Ok(out)
    }

    /// SDK record overhead plus field charges. Counts are supplied before traversing events/links.
    #[pyo3(signature = (kind, fields, event_count, link_count))]
    fn record_bytes(&self, kind: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_telemetry_policy_record_bytes_fields)] fields: ::std::vec::Vec<usize>, #[pyo3(from_py_with = _extract_telemetry_policy_record_bytes_event_count)] event_count: usize, #[pyo3(from_py_with = _extract_telemetry_policy_record_bytes_link_count)] link_count: usize) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let kind = crate::generated::records::telemetry::telemetry_record_kind_from_py(&kind)?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::record_bytes(&self.0, kind, fields, event_count, link_count);
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/policy.rs.
type _core_MetricSeriesImpl = <crate::Selected as super::Implementations>::MetricSeries;

/// The framework's metric series. Subject folds decide measurements; this refuses unbounded labels
/// and series before the host SDK creates a new one.
#[pyclass(name = "_core_MetricSeries", module = "livekit.agents._core")]
pub struct _core_MetricSeries(pub(crate) _core_MetricSeriesImpl);

#[pymethods]
impl _core_MetricSeries {
    #[new]
    fn new() -> PyResult<_core_MetricSeries> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_MetricSeriesImpl as ::lk_agents_abi::telemetry::policy::MetricSeries>::new();
        Ok(_core_MetricSeries(out))
    }

    #[pyo3(signature = (name, attributes))]
    fn admit(&mut self, name: ::std::string::String, attributes: ::std::vec::Vec<(::std::string::String, ::std::string::String)>) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_MetricSeriesImpl as ::lk_agents_abi::telemetry::policy::MetricSeries>::admit(&mut self.0, name, attributes);
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/report/projection.rs.
type _core_SessionReportProjectionImpl = <crate::Selected as super::Implementations>::SessionReportProjection;

fn _extract_session_report_projection_event_limit_items(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "items")
}

fn _extract_session_report_projection_accepts_event_retained_bytes(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "retained_bytes")
}

fn _extract_session_report_projection_accepts_event_event_bytes(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "event_bytes")
}

fn _extract_session_report_projection_project_event_retained_bytes(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "retained_bytes")
}

fn _extract_session_report_projection_event_record_timestamp(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "timestamp")
}

#[pyclass(frozen, name = "_core_SessionReportProjection", module = "livekit.agents._core")]
pub struct _core_SessionReportProjection(pub(crate) ::std::sync::Arc<_core_SessionReportProjectionImpl>);

#[pymethods]
impl _core_SessionReportProjection {
    #[new]
    #[pyo3(signature = (capture_content, options, project_redaction))]
    fn new(capture_content: bool, options: ::lk_agents_abi::telemetry::report::RecordingOptions, project_redaction: bool) -> PyResult<_core_SessionReportProjection> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_SessionReportProjectionImpl as ::lk_agents_abi::telemetry::report::projection::SessionReportProjection>::new(capture_content, &options, project_redaction);
        Ok(_core_SessionReportProjection(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn content(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_SessionReportProjectionImpl as ::lk_agents_abi::telemetry::report::projection::SessionReportProjection>::content(&self.0);
        Ok(out)
    }

    #[getter]
    fn transcript(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_SessionReportProjectionImpl as ::lk_agents_abi::telemetry::report::projection::SessionReportProjection>::transcript(&self.0);
        Ok(out)
    }

    #[getter]
    fn audio(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_SessionReportProjectionImpl as ::lk_agents_abi::telemetry::report::projection::SessionReportProjection>::audio(&self.0);
        Ok(out)
    }

    /// Optional observations beyond this prefix are omitted without failing the transcript.
    #[pyo3(signature = (items))]
    fn event_limit(&self, #[pyo3(from_py_with = _extract_session_report_projection_event_limit_items)] items: usize) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_SessionReportProjectionImpl as ::lk_agents_abi::telemetry::report::projection::SessionReportProjection>::event_limit(&self.0, items);
        Ok(out)
    }

    /// Admission precedes host SDK serialization and native JSON extraction.
    #[pyo3(signature = (retained_bytes, event_bytes))]
    fn accepts_event(&self, #[pyo3(from_py_with = _extract_session_report_projection_accepts_event_retained_bytes)] retained_bytes: usize, #[pyo3(from_py_with = _extract_session_report_projection_accepts_event_event_bytes)] event_bytes: usize) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_SessionReportProjectionImpl as ::lk_agents_abi::telemetry::report::projection::SessionReportProjection>::accepts_event(&self.0, retained_bytes, event_bytes);
        Ok(out)
    }

    #[pyo3(signature = (report))]
    fn project<'py>(&self, py: Python<'py>, report: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let report = crate::generated::records::telemetry::session_report_snapshot_from_py(&report)?;
        let out = <_core_SessionReportProjectionImpl as ::lk_agents_abi::telemetry::report::projection::SessionReportProjection>::project(&self.0, report);
        Ok(crate::generated::records::telemetry::session_report_snapshot_to_py(py, out)?)
    }

    #[pyo3(signature = (event, retained_bytes))]
    fn project_event<'py>(&self, py: Python<'py>, event: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_session_report_projection_project_event_retained_bytes)] retained_bytes: usize) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        ::lk_agents_abi::python::check_pid()?;
        let event = ::lk_agents_abi::python::value_from_py(&event, "event")?;
        let out = <_core_SessionReportProjectionImpl as ::lk_agents_abi::telemetry::report::projection::SessionReportProjection>::project_event(&self.0, event, retained_bytes);
        Ok(out.map(|value| Ok::<_, ::pyo3::PyErr>(crate::generated::records::telemetry::projected_report_event_to_py(py, value)?)).transpose()?)
    }

    #[pyo3(signature = (event, identity, timestamp, disabled))]
    fn event_record(&self, event: Bound<'_, PyAny>, identity: (::std::string::String, ::std::string::String, ::std::string::String), #[pyo3(from_py_with = _extract_session_report_projection_event_record_timestamp)] timestamp: f64, disabled: bool) -> PyResult<::std::option::Option<::lk_agents_abi::LogRecordWire>> {
        ::lk_agents_abi::python::check_pid()?;
        let event = crate::generated::records::telemetry::projected_report_event_from_py(&event)?;
        let out = <_core_SessionReportProjectionImpl as ::lk_agents_abi::telemetry::report::projection::SessionReportProjection>::event_record(&self.0, event, identity, timestamp, disabled);
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/loop_monitor.rs.
type LoopMonitorImpl = <crate::Selected as super::Implementations>::LoopMonitor;

fn _extract_loop_monitor_new_warn_threshold(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "warn_threshold")
}

fn _extract_loop_monitor_new_error_threshold(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "error_threshold")
}

fn _extract_loop_monitor_new_tick_interval(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(&obj, "tick_interval")).map(Some) }
}

/// A loop's watchdog: runtime owns thresholds, timing, admission and cleanup, while the host
/// supplies asyncio scheduling, stack samples and OpenTelemetry emission.
#[pyclass(frozen, name = "LoopMonitor", module = "livekit.agents._core")]
pub struct LoopMonitor(pub(crate) ::std::sync::Arc<LoopMonitorImpl>);

#[pymethods]
impl LoopMonitor {
    #[new]
    #[pyo3(signature = (event_loop, *, warn_threshold, error_threshold, tick_interval, name, emit_spans, report, span_of))]
    fn new(event_loop: ::lk_agents_abi::HostRef, #[pyo3(from_py_with = _extract_loop_monitor_new_warn_threshold)] warn_threshold: f64, #[pyo3(from_py_with = _extract_loop_monitor_new_error_threshold)] error_threshold: f64, #[pyo3(from_py_with = _extract_loop_monitor_new_tick_interval)] tick_interval: ::std::option::Option<f64>, name: ::std::string::String, emit_spans: bool, report: ::lk_agents_abi::HostRef, span_of: ::lk_agents_abi::HostRef) -> PyResult<LoopMonitor> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <LoopMonitorImpl as ::lk_agents_abi::telemetry::loop_monitor::LoopMonitor>::new(event_loop, warn_threshold, error_threshold, tick_interval, name, emit_spans, report, span_of).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(LoopMonitor(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn warn_threshold(&self) -> PyResult<f64> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <LoopMonitorImpl as ::lk_agents_abi::telemetry::loop_monitor::LoopMonitor>::warn_threshold(&self.0);
        Ok(out)
    }

    #[getter]
    fn error_threshold(&self) -> PyResult<f64> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <LoopMonitorImpl as ::lk_agents_abi::telemetry::loop_monitor::LoopMonitor>::error_threshold(&self.0);
        Ok(out)
    }

    fn start(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <LoopMonitorImpl as ::lk_agents_abi::telemetry::loop_monitor::LoopMonitor>::start(&self.0).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn stop(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <LoopMonitorImpl as ::lk_agents_abi::telemetry::loop_monitor::LoopMonitor>::stop(&self.0).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/loop_monitor.rs.
type LoopMonitorThresholdsImpl = <crate::Selected as super::Implementations>::LoopMonitorThresholds;

#[pyfunction]
#[pyo3(signature = (warn_ms, error_ms))]
pub fn _core_loop_monitor_thresholds(warn_ms: ::std::option::Option<::std::string::String>, error_ms: ::std::option::Option<::std::string::String>) -> PyResult<::std::option::Option<(f64, f64)>> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <LoopMonitorThresholdsImpl as ::lk_agents_abi::telemetry::loop_monitor::LoopMonitorThresholds>::_core_loop_monitor_thresholds(warn_ms, error_ms);
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/observation.rs.
type _core_EventRetentionImpl = <crate::Selected as super::Implementations>::EventRetention;

fn _extract_event_retention_admit_retained_bytes(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "retained_bytes")
}

fn _extract_event_retention_admit_now_ns(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "now_ns")
}

fn _extract_event_retention_expire_now_ns(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "now_ns")
}

/// Bounded retention of a session report's host event snapshots. The native policy retains only
/// byte charges and supplied monotonic timestamps; the adapter owns independent model copies.
#[pyclass(name = "_core_EventRetention", module = "livekit.agents._core")]
pub struct _core_EventRetention(pub(crate) _core_EventRetentionImpl);

#[pymethods]
impl _core_EventRetention {
    #[new]
    #[pyo3(signature = (capture_content))]
    fn new(capture_content: bool) -> PyResult<_core_EventRetention> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_EventRetentionImpl as ::lk_agents_abi::telemetry::observation::EventRetention>::new(capture_content).map_err(|e: ::lk_agents_abi::telemetry::upload::UploadError| <::lk_agents_abi::telemetry::upload::UploadError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(_core_EventRetention(out))
    }

    /// Bind report access to the first job that starts the session. Ownership cannot transfer.
    #[pyo3(signature = (job_id))]
    fn claim(&mut self, job_id: ::std::string::String) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <_core_EventRetentionImpl as ::lk_agents_abi::telemetry::observation::EventRetention>::claim(&mut self.0, job_id);
        Ok(())
    }

    /// Sessions started outside a job retain their existing application-owned access contract.
    #[pyo3(signature = (job_id))]
    fn allows_read(&self, job_id: ::std::string::String) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_EventRetentionImpl as ::lk_agents_abi::telemetry::observation::EventRetention>::allows_read(&self.0, job_id);
        Ok(out)
    }

    /// Immutable consent captured when this session's report retention was created.
    #[getter]
    fn capture_content(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_EventRetentionImpl as ::lk_agents_abi::telemetry::observation::EventRetention>::capture_content(&self.0);
        Ok(out)
    }

    /// Reject content without creation-time consent, unknown schemas and live-object events.
    #[pyo3(signature = (kind, redaction))]
    fn accepts(&self, kind: ::std::string::String, redaction: bool) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_EventRetentionImpl as ::lk_agents_abi::telemetry::observation::EventRetention>::accepts(&self.0, kind, redaction);
        Ok(out)
    }

    /// Expire old entries, then admit only within the item and byte budgets. The adapter removes
    /// exactly the returned prefix before appending the newly admitted snapshot.
    #[pyo3(signature = (retained_bytes, now_ns))]
    fn admit(&mut self, #[pyo3(from_py_with = _extract_event_retention_admit_retained_bytes)] retained_bytes: usize, #[pyo3(from_py_with = _extract_event_retention_admit_now_ns)] now_ns: u64) -> PyResult<(usize, bool)> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_EventRetentionImpl as ::lk_agents_abi::telemetry::observation::EventRetention>::admit(&mut self.0, retained_bytes, now_ns);
        Ok(out)
    }

    #[pyo3(signature = (now_ns))]
    fn expire(&mut self, #[pyo3(from_py_with = _extract_event_retention_expire_now_ns)] now_ns: u64) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_EventRetentionImpl as ::lk_agents_abi::telemetry::observation::EventRetention>::expire(&mut self.0, now_ns);
        Ok(out)
    }

    #[getter]
    fn diagnostics(&self) -> PyResult<::std::vec::Vec<(::std::string::String, u64)>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_EventRetentionImpl as ::lk_agents_abi::telemetry::observation::EventRetention>::diagnostics(&self.0);
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/upload.rs.
type _core_SessionReportUploadImpl = <crate::Selected as super::Implementations>::SessionReportUpload;

fn _extract_session_report_upload_upload_start_time_millis(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    ::lk_agents_abi::python::number_from_py::<i64>(&obj, "start_time_millis")
}

/// Credentials for the session recording endpoint and host OTLP exporters.
#[pyclass(frozen, name = "_core_SessionReportUpload", module = "livekit.agents._core")]
pub struct _core_SessionReportUpload(pub(crate) ::std::sync::Arc<_core_SessionReportUploadImpl>);

#[pymethods]
impl _core_SessionReportUpload {
    #[new]
    #[pyo3(signature = (api_key, api_secret))]
    fn new(api_key: ::std::string::String, api_secret: ::std::string::String) -> PyResult<_core_SessionReportUpload> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_SessionReportUploadImpl as ::lk_agents_abi::telemetry::upload::SessionReportUpload>::new(api_key, api_secret).map_err(|e: ::lk_agents_abi::telemetry::upload::UploadError| <::lk_agents_abi::telemetry::upload::UploadError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(_core_SessionReportUpload(::std::sync::Arc::new(out)))
    }

    /// A fresh LiveKit observability credential. No token policy lives in the host adapter.
    fn bearer(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_SessionReportUploadImpl as ::lk_agents_abi::telemetry::upload::SessionReportUpload>::bearer(&self.0).map_err(|e: ::lk_agents_abi::telemetry::upload::UploadError| <::lk_agents_abi::telemetry::upload::UploadError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    /// Recording options are explicit consent for audio; transcript also requires GenAI content
    /// capture. Project redaction excludes both raw content formats.
    #[pyo3(signature = (transcript_requested, audio_requested, capture_content, redaction_enabled))]
    fn content_policy(&self, transcript_requested: bool, audio_requested: bool, capture_content: bool, redaction_enabled: bool) -> PyResult<(bool, bool)> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_SessionReportUploadImpl as ::lk_agents_abi::telemetry::upload::SessionReportUpload>::content_policy(&self.0, transcript_requested, audio_requested, capture_content, redaction_enabled);
        Ok(out)
    }

    /// Returns false only when the project disabled recording. Export failures are reported once
    /// to the caller, which decides where to surface them without failing the conversation.
    #[pyo3(signature = (base_url, room_id, job_id, simulated, redaction_enabled, start_time_millis, transcript, audio_path, disabled))]
    fn upload<'py>(&self, py: Python<'py>, base_url: ::std::string::String, room_id: ::std::string::String, job_id: ::std::string::String, simulated: bool, redaction_enabled: bool, #[pyo3(from_py_with = _extract_session_report_upload_upload_start_time_millis)] start_time_millis: i64, transcript: ::std::option::Option<::std::string::String>, audio_path: ::std::option::Option<::std::string::String>, disabled: bool) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <_core_SessionReportUploadImpl as ::lk_agents_abi::telemetry::upload::SessionReportUpload>::upload(&this, base_url, room_id, job_id, simulated, redaction_enabled, start_time_millis, transcript, audio_path, disabled)
                .await.map_err(|e: ::lk_agents_abi::telemetry::upload::UploadError| <::lk_agents_abi::telemetry::upload::UploadError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/upload.rs.
type _core_PreparedTranscriptImpl = <crate::Selected as super::Implementations>::PreparedTranscript;

fn _extract_prepared_transcript_check_input_items(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "items")
}

fn _extract_prepared_transcript_check_input_item_bytes(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "item_bytes")
}

/// Incremental report preparation. Host adapters admit one item's shape before conversion and
/// append it off their event loop; no whole-history extraction crosses a binding call.
#[pyclass(frozen, name = "_core_PreparedTranscript", module = "livekit.agents._core")]
pub struct _core_PreparedTranscript(pub(crate) ::std::sync::Arc<_core_PreparedTranscriptImpl>);

#[pymethods]
impl _core_PreparedTranscript {
    #[new]
    fn new() -> PyResult<_core_PreparedTranscript> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_PreparedTranscriptImpl as ::lk_agents_abi::telemetry::upload::PreparedTranscript>::new().map_err(|e: ::lk_agents_abi::telemetry::upload::UploadError| <::lk_agents_abi::telemetry::upload::UploadError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(_core_PreparedTranscript(::std::sync::Arc::new(out)))
    }

    /// Maximum items per append batch before the adapter returns control to its event loop.
    #[getter]
    fn input_batch_size(&self) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_PreparedTranscriptImpl as ::lk_agents_abi::telemetry::upload::PreparedTranscript>::input_batch_size(&self.0);
        Ok(out)
    }

    /// Which media parts belong to the transcript's native projection (image, audio).
    fn content_policy(&self) -> PyResult<(bool, bool)> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_PreparedTranscriptImpl as ::lk_agents_abi::telemetry::upload::PreparedTranscript>::content_policy(&self.0);
        Ok(out)
    }

    /// Validate host collection lengths and the shared retained-byte verdict before converting an
    /// item's fields. These scalar facts cannot allocate an unbounded native item.
    #[pyo3(signature = (items, item_bytes))]
    fn check_input(&self, #[pyo3(from_py_with = _extract_prepared_transcript_check_input_items)] items: usize, #[pyo3(from_py_with = _extract_prepared_transcript_check_input_item_bytes)] item_bytes: usize) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <_core_PreparedTranscriptImpl as ::lk_agents_abi::telemetry::upload::PreparedTranscript>::check_input(&self.0, items, item_bytes).map_err(|e: ::lk_agents_abi::telemetry::upload::UploadError| <::lk_agents_abi::telemetry::upload::UploadError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (item))]
    fn append(&self, item: ::lk_agents_abi::ChatItem) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <_core_PreparedTranscriptImpl as ::lk_agents_abi::telemetry::upload::PreparedTranscript>::append(&self.0, item).map_err(|e: ::lk_agents_abi::telemetry::upload::UploadError| <::lk_agents_abi::telemetry::upload::UploadError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn finish(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_PreparedTranscriptImpl as ::lk_agents_abi::telemetry::upload::PreparedTranscript>::finish(&self.0).map_err(|e: ::lk_agents_abi::telemetry::upload::UploadError| <::lk_agents_abi::telemetry::upload::UploadError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    /// Append one already projected event to the same bounded recording preparation.
    #[pyo3(signature = (event))]
    fn append_event(&self, event: Bound<'_, PyAny>) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let event = ::lk_agents_abi::python::value_from_py(&event, "event")?;
        let out = <_core_PreparedTranscriptImpl as ::lk_agents_abi::telemetry::upload::PreparedTranscript>::append_event(&self.0, event).map_err(|e: ::lk_agents_abi::telemetry::upload::UploadError| <::lk_agents_abi::telemetry::upload::UploadError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    /// Serialize and write a console report without converting its history back into host objects.
    #[pyo3(signature = (report, path))]
    fn finish_report<'py>(&self, py: Python<'py>, report: Bound<'_, PyAny>, path: ::std::string::String) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let report = crate::generated::records::telemetry::session_report_snapshot_from_py(&report)?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <_core_PreparedTranscriptImpl as ::lk_agents_abi::telemetry::upload::PreparedTranscript>::finish_report(&this, report, path)
                .await.map_err(|e: ::lk_agents_abi::telemetry::upload::UploadError| <::lk_agents_abi::telemetry::upload::UploadError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}
