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

/// The host supplies SDK fields and lengths; shared policy decides what may be retained/exported.
#[pyclass(frozen, name = "_core_TelemetryPolicy", module = "livekit.agents._core")]
pub struct _core_TelemetryPolicy(::std::sync::Arc<_core_TelemetryPolicyImpl>);

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
    fn export_action(&self, registered: bool, signal_enabled: bool, job_open: bool, held_count: usize, recording_disabled: bool) -> PyResult<::lk_agents_abi::telemetry::policy::TelemetryExportAction> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::export_action(&self.0, registered, signal_enabled, job_open, held_count, recording_disabled);
        Ok(out)
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
    fn value_bytes(&self, kind: ::lk_agents_abi::telemetry::policy::TelemetryValueKind, length: usize, depth: usize, children: ::std::vec::Vec<usize>) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::value_bytes(&self.0, kind, length, depth, children);
        Ok(out)
    }

    /// SDK record overhead plus field charges. Counts are supplied before traversing events/links.
    #[pyo3(signature = (kind, fields, event_count, link_count))]
    fn record_bytes(&self, kind: ::lk_agents_abi::telemetry::policy::TelemetryRecordKind, fields: ::std::vec::Vec<usize>, event_count: usize, link_count: usize) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::record_bytes(&self.0, kind, fields, event_count, link_count);
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/loop_monitor.rs.
type LoopMonitorImpl = <crate::Selected as super::Implementations>::LoopMonitor;

/// A loop's watchdog: runtime owns thresholds, timing, admission and cleanup, while the host
/// supplies asyncio scheduling, stack samples and OpenTelemetry emission.
#[pyclass(frozen, name = "LoopMonitor", module = "livekit.agents._core")]
pub struct LoopMonitor(::std::sync::Arc<LoopMonitorImpl>);

#[pymethods]
impl LoopMonitor {
    #[new]
    #[pyo3(signature = (event_loop, *, warn_threshold, error_threshold, tick_interval, name, emit_spans, report, span_of))]
    fn new(event_loop: ::lk_agents_abi::HostRef, warn_threshold: f64, error_threshold: f64, tick_interval: ::std::option::Option<f64>, name: ::std::string::String, emit_spans: bool, report: ::lk_agents_abi::HostRef, span_of: ::lk_agents_abi::HostRef) -> PyResult<LoopMonitor> {
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
pub fn _core_loop_monitor_thresholds_from_env() -> PyResult<::std::option::Option<(f64, f64)>> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <LoopMonitorThresholdsImpl as ::lk_agents_abi::telemetry::loop_monitor::LoopMonitorThresholds>::_core_loop_monitor_thresholds_from_env();
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/observation.rs.
type _core_EventRetentionImpl = <crate::Selected as super::Implementations>::EventRetention;

/// Bounded retention of a session report's host event snapshots. The native policy retains only
/// byte charges and supplied monotonic timestamps; the adapter owns independent model copies.
#[pyclass(frozen, name = "_core_EventRetention", module = "livekit.agents._core")]
pub struct _core_EventRetention(::std::sync::Arc<_core_EventRetentionImpl>);

#[pymethods]
impl _core_EventRetention {
    #[new]
    #[pyo3(signature = (capture_content))]
    fn new(capture_content: bool) -> PyResult<_core_EventRetention> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_EventRetentionImpl as ::lk_agents_abi::telemetry::observation::EventRetention>::new(capture_content);
        Ok(_core_EventRetention(::std::sync::Arc::new(out)))
    }

    /// Bind report access to the first job that starts the session. Ownership cannot transfer.
    #[pyo3(signature = (job_id))]
    fn claim(&self, job_id: ::std::string::String) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <_core_EventRetentionImpl as ::lk_agents_abi::telemetry::observation::EventRetention>::claim(&self.0, job_id);
        Ok(())
    }

    /// Sessions started outside a job retain their existing application-owned access contract.
    #[pyo3(signature = (job_id))]
    fn allows_read(&self, job_id: ::std::string::String) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_EventRetentionImpl as ::lk_agents_abi::telemetry::observation::EventRetention>::allows_read(&self.0, job_id);
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
    fn admit(&self, retained_bytes: usize, now_ns: u64) -> PyResult<(usize, bool)> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_EventRetentionImpl as ::lk_agents_abi::telemetry::observation::EventRetention>::admit(&self.0, retained_bytes, now_ns);
        Ok(out)
    }

    #[pyo3(signature = (now_ns))]
    fn expire(&self, now_ns: u64) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_EventRetentionImpl as ::lk_agents_abi::telemetry::observation::EventRetention>::expire(&self.0, now_ns);
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

/// Credentials for the session recording endpoint and host OTLP exporters.
#[pyclass(frozen, name = "_core_SessionReportUpload", module = "livekit.agents._core")]
pub struct _core_SessionReportUpload(::std::sync::Arc<_core_SessionReportUploadImpl>);

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
    fn upload<'py>(&self, py: Python<'py>, base_url: ::std::string::String, room_id: ::std::string::String, job_id: ::std::string::String, simulated: bool, redaction_enabled: bool, start_time_millis: i64, transcript: ::std::option::Option<::std::string::String>, audio_path: ::std::option::Option<::std::string::String>, disabled: bool) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <_core_SessionReportUploadImpl as ::lk_agents_abi::telemetry::upload::SessionReportUpload>::upload(&this, base_url, room_id, job_id, simulated, redaction_enabled, start_time_millis, transcript, audio_path, disabled)
                .await.map_err(|e: ::lk_agents_abi::telemetry::upload::UploadError| <::lk_agents_abi::telemetry::upload::UploadError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/upload.rs.
type _core_PreparedTranscriptImpl = <crate::Selected as super::Implementations>::PreparedTranscript;

/// Incremental report preparation. Host adapters admit one item's shape before conversion and
/// append it from admitted SDK work; no whole-history extraction crosses a binding call.
#[pyclass(frozen, name = "_core_PreparedTranscript", module = "livekit.agents._core")]
pub struct _core_PreparedTranscript(::std::sync::Arc<_core_PreparedTranscriptImpl>);

#[pymethods]
impl _core_PreparedTranscript {
    #[new]
    fn new() -> PyResult<_core_PreparedTranscript> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_PreparedTranscriptImpl as ::lk_agents_abi::telemetry::upload::PreparedTranscript>::new().map_err(|e: ::lk_agents_abi::telemetry::upload::UploadError| <::lk_agents_abi::telemetry::upload::UploadError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(_core_PreparedTranscript(::std::sync::Arc::new(out)))
    }

    /// Maximum items per SDK callback before the adapter returns control to its event loop.
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
    fn check_input(&self, items: usize, item_bytes: usize) -> PyResult<()> {
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
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/work.rs.
type _core_TelemetryWorkerImpl = <crate::Selected as super::Implementations>::TelemetryWorker;

/// Process-owned SDK work. Constructing another handle shares the same queue and worker ceiling.
#[pyclass(frozen, name = "_core_TelemetryWorker", module = "livekit.agents._core")]
pub struct _core_TelemetryWorker(::std::sync::Arc<_core_TelemetryWorkerImpl>);

#[pymethods]
impl _core_TelemetryWorker {
    #[new]
    fn new() -> PyResult<_core_TelemetryWorker> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryWorkerImpl as ::lk_agents_abi::telemetry::work::TelemetryWorker>::new();
        Ok(_core_TelemetryWorker(::std::sync::Arc::new(out)))
    }

    /// Refuses overload or a duplicate live key before retaining the callback. `retained_bytes`
    /// includes the immutable SDK batch captured by the callback; no producer waits for capacity.
    #[pyo3(signature = (key, callback, retained_bytes))]
    fn submit(&self, key: ::std::string::String, callback: ::lk_agents_abi::HostRef, retained_bytes: usize) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryWorkerImpl as ::lk_agents_abi::telemetry::work::TelemetryWorker>::submit(&self.0, key, callback, retained_bytes);
        Ok(out)
    }

    /// Retention admission for an SDK batch queue, independent of cleanup callback slots.
    #[pyo3(signature = (key, retained_bytes))]
    fn reserve(&self, key: ::std::string::String, retained_bytes: usize) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryWorkerImpl as ::lk_agents_abi::telemetry::work::TelemetryWorker>::reserve(&self.0, key, retained_bytes);
        Ok(out)
    }

    #[pyo3(signature = (key))]
    fn release(&self, key: ::std::string::String) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <_core_TelemetryWorkerImpl as ::lk_agents_abi::telemetry::work::TelemetryWorker>::release(&self.0, key);
        Ok(())
    }

    /// `kind` is one of `dropped`, `failed`, `conflict`; unknown categories are refused.
    #[pyo3(signature = (kind, count))]
    fn record_drop(&self, kind: ::std::string::String, count: u64) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <_core_TelemetryWorkerImpl as ::lk_agents_abi::telemetry::work::TelemetryWorker>::record_drop(&self.0, kind, count);
        Ok(())
    }

    /// Refuse unbounded labels/series before host SDK instrument creation.
    #[pyo3(signature = (name, attributes))]
    fn admit_metric(&self, name: ::std::string::String, attributes: ::std::vec::Vec<(::std::string::String, ::std::string::String)>) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryWorkerImpl as ::lk_agents_abi::telemetry::work::TelemetryWorker>::admit_metric(&self.0, name, attributes);
        Ok(out)
    }

    /// Off-loop cleanup only. A deadline never detaches work or releases its capacity early.
    #[pyo3(signature = (key, timeout))]
    fn wait(&self, key: ::std::string::String, timeout: f64) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryWorkerImpl as ::lk_agents_abi::telemetry::work::TelemetryWorker>::wait(&self.0, key, timeout);
        Ok(out)
    }

    #[pyo3(signature = (key, timeout))]
    fn wait_async<'py>(&self, py: Python<'py>, key: ::std::string::String, timeout: f64) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <_core_TelemetryWorkerImpl as ::lk_agents_abi::telemetry::work::TelemetryWorker>::wait_async(&this, key, timeout)
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// Drain bounded lifecycle observations; timed-out work keeps its captured key and owner.
    fn drain_observations(&self) -> PyResult<::std::vec::Vec<::lk_agents_abi::telemetry::work::TelemetryWorkObservation>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryWorkerImpl as ::lk_agents_abi::telemetry::work::TelemetryWorker>::drain_observations(&self.0);
        Ok(out)
    }

    /// Nonrecursive counters, read without exporting/logging them through this same pipeline.
    #[getter]
    fn diagnostics(&self) -> PyResult<::std::vec::Vec<(::std::string::String, u64)>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryWorkerImpl as ::lk_agents_abi::telemetry::work::TelemetryWorker>::diagnostics(&self.0);
        Ok(out)
    }
}
