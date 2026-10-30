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

fn _extract_session_recorder_new_sample_rate(
    obj: &Bound<'_, PyAny>,
) -> PyResult<SessionRecorderNewSampleRateInput> {
    crate::python::number_from_py::<u32>(obj.as_any(), "sample_rate")
}
type SessionRecorderNewSampleRateInput = u32;

fn _extract_session_recorder_new_t0(obj: &Bound<'_, PyAny>) -> PyResult<SessionRecorderNewT0Input> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::python::number_from_py::<f64>(obj.as_any(), "t0")).map(Some)
    }
}
type SessionRecorderNewT0Input = ::std::option::Option<f64>;

fn _extract_session_recorder_new_bitrate(
    obj: &Bound<'_, PyAny>,
) -> PyResult<SessionRecorderNewBitrateInput> {
    crate::python::number_from_py::<u32>(obj.as_any(), "bitrate")
}
type SessionRecorderNewBitrateInput = u32;
type SessionRecorderStartedAtOutput = ::std::option::Option<f64>;
type SessionRecorderDroppedSamplesOutput = (u64, u64);

fn _extract_session_recorder_flush_now(
    obj: &Bound<'_, PyAny>,
) -> PyResult<SessionRecorderFlushNowInput> {
    crate::python::number_from_py::<f64>(obj.as_any(), "now")
}
type SessionRecorderFlushNowInput = f64;

fn _extract_session_recorder_finish_now(
    obj: &Bound<'_, PyAny>,
) -> PyResult<SessionRecorderFinishNowInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::python::number_from_py::<f64>(obj.as_any(), "now")).map(Some)
    }
}
type SessionRecorderFinishNowInput = ::std::option::Option<f64>;

/// One admitted runtime worker owns opening, encoding, writes and finalization.
#[pyclass(frozen, name = "SessionRecorder", module = "livekit.agents._core")]
pub struct SessionRecorder(pub(crate) ::std::sync::Arc<SessionRecorderImpl>);

#[pymethods]
impl SessionRecorder {
    #[new]
    #[pyo3(signature = (path, *, sample_rate = 48000, t0 = None, bitrate = 64000))]
    fn new(
        py: Python<'_>,
        path: ::std::string::String,
        #[pyo3(from_py_with = _extract_session_recorder_new_sample_rate)]
        sample_rate: SessionRecorderNewSampleRateInput,
        #[pyo3(from_py_with = _extract_session_recorder_new_t0)] t0: SessionRecorderNewT0Input,
        #[pyo3(from_py_with = _extract_session_recorder_new_bitrate)]
        bitrate: SessionRecorderNewBitrateInput,
    ) -> PyResult<SessionRecorder> {
        crate::host::check_pid()?;
        let env = crate::process::env();
        let out = py.detach(|| <SessionRecorderImpl as ::lk_agents_abi::recorder::SessionRecorder>::new(&env, path, sample_rate, t0, bitrate)).map_err(|e: ::lk_agents_abi::recorder::RecordingFailure| <::lk_agents_abi::recorder::RecordingFailure as crate::python::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(SessionRecorder(::std::sync::Arc::new(out)))
    }

    /// Wait until the native file and encoder are open before attaching the tap.
    fn ready<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let () = <SessionRecorderImpl as ::lk_agents_abi::recorder::SessionRecorder>::ready(&this, )
                .await.map_err(|e: ::lk_agents_abi::recorder::RecordingFailure| <::lk_agents_abi::recorder::RecordingFailure as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    #[getter]
    fn bytes_written(&self, py: Python<'_>) -> PyResult<u64> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <SessionRecorderImpl as ::lk_agents_abi::recorder::SessionRecorder>::bytes_written(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn started_at(&self, py: Python<'_>) -> PyResult<SessionRecorderStartedAtOutput> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <SessionRecorderImpl as ::lk_agents_abi::recorder::SessionRecorder>::started_at(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn dropped_samples(&self, py: Python<'_>) -> PyResult<SessionRecorderDroppedSamplesOutput> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <SessionRecorderImpl as ::lk_agents_abi::recorder::SessionRecorder>::dropped_samples(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (now))]
    fn flush<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = _extract_session_recorder_flush_now)]
        now: SessionRecorderFlushNowInput,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <SessionRecorderImpl as ::lk_agents_abi::recorder::SessionRecorder>::flush(&this, now)
                .await.map_err(|e: ::lk_agents_abi::recorder::RecordingFailure| <::lk_agents_abi::recorder::RecordingFailure as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    crate::generated::records::recorder::recording_summary_to_py(py, out)?,
                    py,
                )
            })
        })
    }

    #[pyo3(signature = (now = None))]
    fn finish<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = _extract_session_recorder_finish_now)]
        now: SessionRecorderFinishNowInput,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let () = <SessionRecorderImpl as ::lk_agents_abi::recorder::SessionRecorder>::finish(&this, now)
                .await.map_err(|e: ::lk_agents_abi::recorder::RecordingFailure| <::lk_agents_abi::recorder::RecordingFailure as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::recorder::AudioRecorder`, an instance of this class is itself.
inventory::submit! {
    crate::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::recorder::AudioRecorder>(),
        read: |obj| {
            let this = obj.cast::<SessionRecorder>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::recorder::AudioRecorder> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

impl crate::python::HostException for ::lk_agents_abi::recorder::RecordingFailure {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Io(..) => {
                ::pyo3::exceptions::PyOSError::new_err(::std::string::ToString::to_string(self))
            }
            _ => ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(
                self,
            )),
        }
    }
}
