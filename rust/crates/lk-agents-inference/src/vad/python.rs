//! The host's Silero model behind [`VadModelFactory`], and a [`VadOutput`] as the host's
//! `VADEvent` dict.

use std::sync::Arc;

use lk_agents_abi::python::{HostHandle, HostProxy};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::VadModelFactory;
use super::machine::{VadModel, VadOutput};
use crate::python::clear;

/// One host model, called from the machine's task; `None` when the factory raised, a silent model
/// rather than a panic on a tokio worker (never selected on a healthy path).
struct PyVadModel {
    model: Option<Py<PyAny>>,
}

impl VadModel for PyVadModel {
    /// `(probability, inference_time)`. A bare `float` (timed here, as the reference does) or a
    /// pair are both accepted; a failed call answers `(0.0, 0.0)`.
    fn predict(&mut self, window: &[i16]) -> (f32, f64) {
        let Some(model) = &self.model else { return (0.0, 0.0) };
        Python::attach(|py| {
            let started = std::time::Instant::now();
            let out = model.bind(py).call_method1("predict", (window.to_vec(),));
            let elapsed = started.elapsed().as_secs_f64();
            match out.and_then(|v| v.extract::<f32>().map(|p| (p, elapsed)).or_else(|_| v.extract::<(f32, f64)>())) {
                Ok(pair) => pair,
                Err(e) => {
                    clear(py, e);
                    (0.0, 0.0)
                }
            }
        })
    }

    fn reset(&mut self) {
        let Some(model) = &self.model else { return };
        Python::attach(|py| {
            if let Err(e) = model.bind(py).call_method0("reset") {
                clear(py, e);
            }
        });
    }
}

/// `model_factory`: a Python callable that returns a fresh model per stream.
struct PyVadFactory {
    factory: Py<PyAny>,
}

impl VadModelFactory for PyVadFactory {
    fn create(&self) -> Box<dyn VadModel> {
        let model = Python::attach(|py| self.factory.bind(py).call0().map(Bound::unbind).ok());
        Box::new(PyVadModel { model })
    }
}

/// Any callable is a factory: called per stream, under the GIL, from the machine's task.
impl HostProxy for dyn VadModelFactory {
    fn from_host(obj: &Bound<'_, PyAny>, _clock: Option<Arc<dyn lk_agents_abi::Clock>>) -> PyResult<Arc<Self>> {
        Ok(Arc::new(PyVadFactory { factory: obj.clone().unbind() }))
    }

    fn view(handle: HostHandle, _clock: Option<Arc<dyn lk_agents_abi::Clock>>) -> PyResult<Box<Self>> {
        Python::attach(|py| Ok(Box::new(PyVadFactory { factory: handle.bind(py).clone().unbind() }) as Box<Self>))
    }
}

/// `VadOutput` as the host's `VADEvent` dict: the event's fields plus `frames` (decoded audio) and
/// the two slow-inference diagnostics, which belong to the output, not the event.
impl<'py> IntoPyObject<'py> for VadOutput {
    type Target = PyDict;
    type Output = Bound<'py, PyDict>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = (&self.event).into_pyobject(py)?;
        let frames = PyList::empty(py);
        for frame in &self.frames {
            frames.append(frame.into_pyobject(py)?)?;
        }
        d.set_item("frames", frames)?;
        d.set_item("extra_inference_time", self.extra_inference_time)?;
        d.set_item("slow_inference", self.slow_inference)?;
        Ok(d)
    }
}
