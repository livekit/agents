//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `recorder` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/recorder.rs.
type SessionRecorderImpl = <crate::Selected as super::Implementations>::SessionRecorder;

fn _extract_session_recorder_new_sample_rate(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    ::lk_agents_abi::python::number_from_py::<u32>(&obj, "sample_rate")
}

fn _extract_session_recorder_new_t0(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(&obj, "t0")).map(Some) }
}

fn _extract_session_recorder_new_bitrate(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    ::lk_agents_abi::python::number_from_py::<u32>(&obj, "bitrate")
}

fn _extract_session_recorder_flush_now(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "now")
}

fn _extract_session_recorder_finish_now(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(&obj, "now")).map(Some) }
}

/// One admitted runtime worker owns opening, encoding, writes and finalization.
#[pyclass(frozen, name = "SessionRecorder", module = "livekit.agents._core")]
pub struct SessionRecorder(pub(crate) ::std::sync::Arc<SessionRecorderImpl>);

#[pymethods]
impl SessionRecorder {
    #[new]
    #[pyo3(signature = (path, *, sample_rate = 48000, t0 = None, bitrate = 64000))]
    fn new(path: ::std::string::String, #[pyo3(from_py_with = _extract_session_recorder_new_sample_rate)] sample_rate: u32, #[pyo3(from_py_with = _extract_session_recorder_new_t0)] t0: ::std::option::Option<f64>, #[pyo3(from_py_with = _extract_session_recorder_new_bitrate)] bitrate: u32) -> PyResult<SessionRecorder> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <SessionRecorderImpl as ::lk_agents_abi::recorder::SessionRecorder>::new(path, sample_rate, t0, bitrate).map_err(|e: ::lk_agents_abi::recorder::RecordingFailure| <::lk_agents_abi::recorder::RecordingFailure as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(SessionRecorder(::std::sync::Arc::new(out)))
    }

    /// Wait until the native file and encoder are open before attaching the tap.
    fn ready<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <SessionRecorderImpl as ::lk_agents_abi::recorder::SessionRecorder>::ready(&this, )
                .await.map_err(|e: ::lk_agents_abi::recorder::RecordingFailure| <::lk_agents_abi::recorder::RecordingFailure as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[getter]
    fn bytes_written(&self) -> PyResult<u64> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <SessionRecorderImpl as ::lk_agents_abi::recorder::SessionRecorder>::bytes_written(&self.0);
        Ok(out)
    }

    #[getter]
    fn started_at(&self) -> PyResult<::std::option::Option<f64>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <SessionRecorderImpl as ::lk_agents_abi::recorder::SessionRecorder>::started_at(&self.0);
        Ok(out)
    }

    #[getter]
    fn dropped_samples(&self) -> PyResult<(u64, u64)> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <SessionRecorderImpl as ::lk_agents_abi::recorder::SessionRecorder>::dropped_samples(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (now))]
    fn flush<'py>(&self, py: Python<'py>, #[pyo3(from_py_with = _extract_session_recorder_flush_now)] now: f64) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <SessionRecorderImpl as ::lk_agents_abi::recorder::SessionRecorder>::flush(&this, now)
                .await.map_err(|e: ::lk_agents_abi::recorder::RecordingFailure| <::lk_agents_abi::recorder::RecordingFailure as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(crate::generated::records::recorder::recording_summary_to_py(py, out)?, py))
        })
    }

    #[pyo3(signature = (now = None))]
    fn finish<'py>(&self, py: Python<'py>, #[pyo3(from_py_with = _extract_session_recorder_finish_now)] now: ::std::option::Option<f64>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <SessionRecorderImpl as ::lk_agents_abi::recorder::SessionRecorder>::finish(&this, now)
                .await.map_err(|e: ::lk_agents_abi::recorder::RecordingFailure| <::lk_agents_abi::recorder::RecordingFailure as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::recorder::AudioRecorder`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::recorder::AudioRecorder>(),
        read: |obj| {
            let this = obj.cast::<SessionRecorder>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::recorder::AudioRecorder> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// A `::lk_agents_abi::recorder::AudioRecorder` this class implements is handed to the host as an instance of it.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNativeClass {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::recorder::AudioRecorder>(),
        write: |py, native| {
            let native = native.downcast::<SessionRecorderImpl>().ok()?;
            Some(::pyo3::Py::new(py, SessionRecorder(native)).map(::pyo3::Py::into_any))
        },
    }
}
