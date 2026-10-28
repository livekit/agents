//! The storage surface of `_core` the generated `Storage` class does not declare yet: the
//! response streams handle (`Storage.streams()`) and the helpers `turn.rs` crosses its turn
//! records with, the pydantic models `lk-agents-codegen` writes from the A2A crate's
//! `#[derive(Host)]` records (`livekit.agents._generated.storage_turn`).

use std::sync::Arc;

use lk_agents_abi::python::{check_pid, pyo3_async_runtimes, value_from_py, value_to_py};
use lk_agents_runtime::storage::{QueuedStreams, StreamRead};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

/// A storage failure as the `livekit.agents.storage` exception the ABI declares for it.
pub(crate) use lk_agents_abi::storage::storage_error_to_py as to_py_err;

/// A request into the crate's serde type: a pydantic model (its JSON dump) or a dict in the
/// model's shape; a malformed one is a `ValueError` naming the field.
pub(crate) fn parse_request<T: DeserializeOwned>(
    obj: &Bound<'_, PyAny>,
    what: &str,
) -> PyResult<T> {
    let dumped;
    let obj = if obj.hasattr("model_dump")? {
        let kwargs = PyDict::new(obj.py());
        kwargs.set_item("mode", "json")?;
        dumped = obj.call_method("model_dump", (), Some(&kwargs))?;
        &dumped
    } else {
        obj
    };
    let v = value_from_py(obj, what)?;
    serde_json::from_value(v).map_err(|e| PyValueError::new_err(format!("{what}: {e}")))
}

/// The generated pydantic class `livekit.agents._generated.<module>.<name>`.
fn model_class<'py>(py: Python<'py>, module: &str, name: &str) -> PyResult<Bound<'py, PyAny>> {
    py.import(format!("livekit.agents._generated.{module}"))?.getattr(name)
}

/// `class.model_validate(value)`: the JSON the crate serialised, as an instance of its model.
pub(crate) fn validate(py: Python<'_>, class: &Bound<'_, PyAny>, v: &Value) -> PyResult<Py<PyAny>> {
    Ok(class.call_method1("model_validate", (value_to_py(py, v)?,))?.unbind())
}

/// A record as an instance of its generated model.
pub(crate) fn model<T: Serialize>(
    py: Python<'_>,
    module: &str,
    name: &str,
    v: &T,
) -> PyResult<Py<PyAny>> {
    let v = serde_json::to_value(v).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    validate(py, &model_class(py, module, name)?, &v)
}

/// The response streams of one storage (`Storage.streams()`): a task's frames, kept for its
/// subscribers on any worker.
#[pyclass(name = "Streams", module = "livekit.agents._core", frozen)]
pub struct PyStreams {
    inner: Arc<QueuedStreams>,
}

impl From<Arc<QueuedStreams>> for PyStreams {
    fn from(inner: Arc<QueuedStreams>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyStreams {
    /// A new stream: a `StreamInfo`. `ttl_seconds=None` is the default, a fixed day.
    #[pyo3(signature = (ttl_seconds=None))]
    fn create<'py>(
        &self,
        py: Python<'py>,
        ttl_seconds: Option<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let streams = self.inner.clone();
        let ttl = ttl_seconds.map(std::time::Duration::from_secs);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let info = streams.create(ttl).await.map_err(to_py_err)?;
            Python::attach(|py| info.into_py_any(py))
        })
    }

    /// Queues one frame: kept in order; a failure is logged and does not fail the turn.
    fn append(&self, stream_id: &str, record: &[u8]) -> PyResult<()> {
        check_pid()?;
        self.inner.append(stream_id, record.to_vec());
        Ok(())
    }

    /// Closes the stream once its queued frames are kept: the terminal frame.
    fn close<'py>(&self, py: Python<'py>, stream_id: String) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let streams = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            streams.close(&stream_id).await.map_err(to_py_err)
        })
    }

    fn delete<'py>(&self, py: Python<'py>, stream_id: String) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let streams = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            streams.delete(&stream_id).await.map_err(to_py_err)
        })
    }

    /// A reader from `after` (an offset; `None` is the beginning). `live` follows the tail until
    /// the stream closes.
    #[pyo3(signature = (stream_id, after=None, live=false))]
    fn read(&self, stream_id: &str, after: Option<String>, live: bool) -> PyResult<PyStreamReader> {
        check_pid()?;
        let reader = self.inner.read(stream_id, after.as_deref(), live);
        Ok(PyStreamReader { inner: Arc::new(tokio::sync::Mutex::new(reader)) })
    }
}

/// A read position over one stream; `next()` yields `bytes`, `None` at the end.
#[pyclass(name = "StreamReader", module = "livekit.agents._core", frozen)]
pub struct PyStreamReader {
    inner: Arc<tokio::sync::Mutex<Box<dyn StreamRead>>>,
}

#[pymethods]
impl PyStreamReader {
    /// The next frame, or `None`: the tail of a catch-up read, the close of a live one.
    fn next<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let reader = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let record = reader.lock().await.next().await.map_err(to_py_err)?;
            Python::attach(|py| Ok(record.map(|r| PyBytes::new(py, &r).into_any().unbind())))
        })
    }

    /// The offset to resume from after the frames returned so far. Awaitable.
    fn next_offset<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let reader = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            Ok(reader.lock().await.next_offset().to_owned())
        })
    }
}

lk_agents_abi::register_host!(class PyStreams);
lk_agents_abi::register_host!(class PyStreamReader);

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a request that does not deserialize raising the storage error instead of a
    // `ValueError` that names the field, a model crossing as its object rather than its dump, or
    // a result handed to the class under the wrong name.
    #[test]
    fn requests_and_results_cross_as_models() {
        Python::initialize();
        Python::attach(|py| {
            let d = py.eval(c"{'id': 'i'}", None, None).unwrap();
            let e = parse_request::<lk_agents_storage::run::ItemRow>(&d, "item").unwrap_err();
            assert!(e.is_instance_of::<PyValueError>(py), "{e}");
            assert!(e.to_string().contains("payload"), "{e}");

            // a pydantic model is whatever `model_dump(mode="json")` says it is
            let ns = PyDict::new(py);
            py.run(
                c"class M:\n    def model_dump(self, mode):\n        assert mode == 'json'\n        return {'id': 'i', 'payload': {'message': {'id': 'i'}}}\n",
                None,
                Some(&ns),
            )
            .unwrap();
            let m = ns.get_item("M").unwrap().unwrap().call0().unwrap();
            let item: lk_agents_storage::run::ItemRow = parse_request(&m, "item").unwrap();
            assert_eq!((item.id.as_str(), item.message_id.as_deref()), ("i", None));

            // a result reaches `model_validate` as the crate's JSON
            py.run(
                c"class Row:\n    @classmethod\n    def model_validate(cls, d):\n        return ('validated', d)\n",
                None,
                Some(&ns),
            )
            .unwrap();
            let class = ns.get_item("Row").unwrap().unwrap();
            let out = validate(py, &class, &serde_json::to_value(&item).unwrap()).unwrap();
            let (tag, d): (String, Bound<'_, PyDict>) = out.extract(py).unwrap();
            assert_eq!(tag, "validated");
            assert_eq!(d.get_item("id").unwrap().unwrap().extract::<String>().unwrap(), "i");
            assert!(d.get_item("message_id").unwrap().unwrap().is_none());
        });
    }
}
