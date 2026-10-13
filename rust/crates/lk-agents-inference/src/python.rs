//! What this crate's hand-written Python classes share; anything a declaration can say is a
//! `#[host(..)]` beside its item.

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use serde_json::{Map, Value};

use crate::native::ReqwestTransport;

/// One HTTPS client for the process, built on first use: nothing may start at import.
pub(crate) fn transport() -> PyResult<Arc<ReqwestTransport>> {
    static TRANSPORT: std::sync::OnceLock<Result<Arc<ReqwestTransport>, String>> = std::sync::OnceLock::new();
    TRANSPORT
        .get_or_init(|| ReqwestTransport::new().map(Arc::new).map_err(|e| e.to_string()))
        .clone()
        .map_err(|e| PyRuntimeError::new_err(format!("could not build the HTTPS client: {e}")))
}

/// The per-request connect options for every native client here. `max_retry = 0`: the host's base
/// classes own the retry loop, and two multiply.
pub(crate) fn conn_options(timeout: f64) -> lk_agents_abi::ApiConnectOptions {
    lk_agents_abi::ApiConnectOptions { timeout, max_retry: 0, ..lk_agents_abi::DEFAULT_API_CONNECT_OPTIONS }
}

pub(crate) fn json_from_py(value: Option<&Bound<'_, PyAny>>) -> PyResult<Value> {
    match value {
        None => Ok(Value::Null),
        Some(v) if v.is_none() => Ok(Value::Null),
        Some(v) => lk_agents_abi::python::value_from_py(v, "inference"),
    }
}

pub(crate) fn json_map(d: Option<&Bound<'_, PyAny>>) -> PyResult<Map<String, Value>> {
    let Some(d) = d else { return Ok(Map::new()) };
    if d.is_none() {
        return Ok(Map::new());
    }
    match json_from_py(Some(d))? {
        Value::Object(m) => Ok(m),
        // A non-dict is a caller bug; treating it as empty would drop the user's `extra_kwargs`.
        other => Err(PyTypeError::new_err(format!("extra_kwargs must be a mapping, got {other}"))),
    }
}

/// `clock=` (`_core.SessionClock`), read through the class the voice crate registers for `dyn Clock`.
pub(crate) fn session_clock(obj: &Bound<'_, PyAny>, arg: &str) -> PyResult<Arc<dyn lk_agents_abi::Clock>> {
    if let Some(clock) = lk_agents_abi::python::native::<dyn lk_agents_abi::Clock>(obj) {
        return Ok(clock);
    }
    let py = obj.py();
    let class = py.import("livekit.agents._core")?.getattr("SessionClock")?;
    let err = PyErr::from(pyo3::CastError::new(obj.as_borrowed(), class));
    Err(pyo3::impl_::extract_argument::argument_extraction_error(py, arg, err))
}

/// Cleared, so a host callback's error on a worker thread does not surface on the next GIL call.
pub(crate) fn clear(py: Python<'_>, e: PyErr) {
    e.restore(py);
    PyErr::fetch(py);
}

/// A host callable returning `get_inference_headers()`'s pairs, called once per connection.
pub(crate) fn header_provider(headers: Py<PyAny>) -> crate::headers::HeaderProvider {
    Arc::new(move || {
        Python::attach(|py| match headers.bind(py).call0() {
            Ok(items) => items.extract::<Vec<(String, String)>>().unwrap_or_default(),
            // Empty rather than a panic on a tokio worker; the key pair still authenticates.
            Err(e) => {
                clear(py, e);
                Vec::new()
            }
        })
    })
}

/// Register a `#[pyfunction]` of this crate on `_core`.
macro_rules! register_fn {
    ($($f:path),* $(,)?) => {
        $(
            ::lk_agents_abi::python::inventory::submit! {
                ::lk_agents_abi::python::Registration(|m| {
                    use ::pyo3::types::PyModuleMethods as _;
                    m.add_function(::pyo3::wrap_pyfunction!($f, m)?)
                })
            }
        )*
    };
}
pub(crate) use register_fn;

/// Register a hand-written `#[pyclass]` of this crate on `_core`.
macro_rules! register_class {
    ($($c:ty),* $(,)?) => {
        $(
            ::lk_agents_abi::python::inventory::submit! {
                ::lk_agents_abi::python::Registration(|m| {
                    use ::pyo3::types::PyModuleMethods as _;
                    m.add_class::<$c>()
                })
            }
        )*
    };
}
pub(crate) use register_class;
