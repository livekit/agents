//! The storage surface of `_core` the generated `Storage` class does not declare yet: the
//! response streams handle (`Storage.streams()`) and the helpers `turn.rs` crosses its turn
//! records with, the pydantic models `lk-agents-codegen` writes from the A2A crate's
//! `#[derive(Host)]` records (`livekit.agents._generated.storage_turn`).

use std::sync::Arc;

use lk_agents_abi::python::{check_pid, pyo3_async_runtimes, value_from_py};
use lk_agents_runtime::storage::{QueuedStreams, StreamRead};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use serde::de::DeserializeOwned;

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
            Python::attach(|py| Ok(crate::generated::records::storage_class::stream_info_to_py(py, info)?.unbind()))
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
}
