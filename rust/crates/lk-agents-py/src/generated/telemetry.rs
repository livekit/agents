//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `telemetry` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

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

    /// Project and serialize a native history on bounded runtime workers, with no host objects
    /// or Python JSON allocation on the session loop.
    #[pyo3(signature = (items))]
    fn transcript<'py>(&self, py: Python<'py>, items: ::std::vec::Vec<::lk_agents_abi::ChatItem>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <_core_SessionReportUploadImpl as ::lk_agents_abi::telemetry::upload::SessionReportUpload>::transcript(&this, items)
                .await.map_err(|e: ::lk_agents_abi::telemetry::upload::UploadError| <::lk_agents_abi::telemetry::upload::UploadError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
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
