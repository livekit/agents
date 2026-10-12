//! Host objects as side-table keys, and host failures as core errors re-raised as the original.

use std::sync::OnceLock;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyType};

use crate::errors::{ApiErrorKind, CapturedError, PayloadTable, ProviderError, TaxonomyMatches, host_retryable};
use crate::{HostRef, MediaRef};

// A host exception goes into the side table and only its key plus scalar `retryable`/`status_code`/
// kind cross, so it re-raises with traceback and subclass intact. `raise_original` takes it once.

// --------------------------------------------------------------------- the side table

fn insert(obj: Py<PyAny>) -> CapturedError {
    let t = errors();
    CapturedError::new(t.insert(obj), t.registry())
}

/// Peek without taking.
pub fn resolve_captured(py: Python<'_>, host_ref: CapturedError) -> Option<Py<PyAny>> {
    errors().resolve(host_ref.id(), |obj| obj.clone_ref(py))
}

/// Every clone of the key answers `None` afterwards and releases nothing on drop. Idempotent.
pub fn release(host_ref: CapturedError) -> Option<Py<PyAny>> {
    host_ref.take().and_then(|id| errors().take(id))
}

// --------------------------------------------------------------------- capture / re-raise

/// Resolved once: captures happen on the retry path.
struct ErrorTypes {
    api: Py<PyType>,
    status: Py<PyType>,
    connection: Py<PyType>,
    timeout: Py<PyType>,
}

fn error_types(py: Python<'_>) -> Option<&'static ErrorTypes> {
    static TYPES: PyOnceLock<ErrorTypes> = PyOnceLock::new();
    // `get_or_try_init`: a failed import (teardown, Rust unit tests) must not be cached, or nothing
    // would ever be retried again.
    TYPES
        .get_or_try_init(py, || -> PyResult<ErrorTypes> {
            // Lazy: `_exceptions` imports `_core`, so resolving at module init would be a cycle.
            let module = py.import("livekit.agents._exceptions")?;
            let get = |name: &str| -> PyResult<Py<PyType>> {
                Ok(module.getattr(name)?.cast_into::<PyType>()?.unbind())
            };
            Ok(ErrorTypes {
                api: get("APIError")?,
                status: get("APIStatusError")?,
                connection: get("APIConnectionError")?,
                timeout: get("APITimeoutError")?,
            })
        })
        .ok()
}

/// By class, not name: plugins subclass, and `llm/llm.py:288` tests `isinstance`.
fn taxonomy_matches(value: &Bound<'_, PyAny>) -> Option<TaxonomyMatches> {
    let types = error_types(value.py())?;
    let is = |t: &Py<PyType>| value.is_instance(t.bind(value.py())).unwrap_or(false);
    Some(TaxonomyMatches {
        timeout: is(&types.timeout),
        connection: is(&types.connection),
        status: is(&types.status),
        api: is(&types.api),
    })
}

pub fn capture_value(value: &Bound<'_, PyAny>) -> ProviderError {
    let status_code = value.getattr("status_code").ok().and_then(|v| v.extract::<i64>().ok());
    let matches = taxonomy_matches(value);
    let api_kind = matches.and_then(|m| ApiErrorKind::from_matches(m, status_code));
    let declared = value.getattr("retryable").ok().and_then(|v| v.extract::<bool>().ok());
    let retryable = host_retryable(api_kind, matches.is_some(), declared);

    ProviderError::Host { error: capture_host_value(value), api_kind, retryable, status_code }
}

/// Without inventing retry or fatality policy.
pub fn capture_host(py: Python<'_>, err: &PyErr) -> crate::errors::HostError {
    capture_host_value(err.value(py).as_any())
}

fn capture_host_value(value: &Bound<'_, PyAny>) -> crate::errors::HostError {
    let kind =
        value.get_type().name().map(|n| n.to_string()).unwrap_or_else(|_| "BaseException".into());
    // An exception's __str__ can itself raise. Preserve the original even in that case.
    let message = value.str().map(|s| s.to_string()).unwrap_or_default();
    crate::errors::HostError::captured(kind, message, insert(value.clone().unbind()))
}

pub fn raise_host(py: Python<'_>, error: crate::errors::HostError) -> PyErr {
    match error.host_ref {
        Some(reference) => raise_original(py, reference),
        None => PyRuntimeError::new_err(error.to_string()),
    }
}

pub fn provider_error_to_py(err: ProviderError) -> PyErr {
    Python::attach(|py| raise_provider(py, err))
}

pub fn capture(py: Python<'_>, err: &PyErr) -> ProviderError {
    capture_value(&err.value(py).clone().into_any())
}

/// `retryable` must go to the constructor: its `__init__` default is `True`, so setting it
/// afterwards is too late for a plugin reading it.
pub fn raise_provider(py: Python<'_>, err: ProviderError) -> PyErr {
    let api = match err {
        ProviderError::Host { error, .. } => return raise_host(py, error),
        ProviderError::Local(api) => api,
    };
    let Some(types) = error_types(py) else {
        // No package around us (unit test or teardown).
        return PyRuntimeError::new_err(api.message);
    };
    let class = match api.kind {
        ApiErrorKind::Api => &types.api,
        ApiErrorKind::Status => &types.status,
        ApiErrorKind::Connection => &types.connection,
        ApiErrorKind::Timeout => &types.timeout,
        // The reference detects cancellation as `APIStatusError` with status 499.
        ApiErrorKind::Cancelled => &types.status,
    };
    let kwargs = PyDict::new(py);
    let build = || -> PyResult<PyErr> {
        kwargs.set_item("retryable", api.retryable)?;
        if matches!(api.kind, ApiErrorKind::Status | ApiErrorKind::Cancelled) {
            kwargs.set_item("status_code", api.status_code.unwrap_or(0))?;
            kwargs.set_item("request_id", api.request_id.clone())?;
            kwargs.set_item("body", py.None())?;
        }
        let instance = class.bind(py).call((api.message.clone(),), Some(&kwargs))?;
        Ok(PyErr::from_value(instance))
    };
    build().unwrap_or_else(|e| e)
}

/// Without releasing the side-table entry, for events that report an error rather than raise it.
pub fn to_py_exception(py: Python<'_>, err: &ProviderError) -> PyResult<Py<PyAny>> {
    if let ProviderError::Host { error, .. } = err {
        return Ok(match error.host_ref.clone().and_then(|r| resolve_captured(py, r)) {
            Some(obj) => obj,
            // Released already (re-raised once): the message is what is left of it.
            None => PyRuntimeError::new_err(error.to_string()).into_value(py).into_any(),
        });
    }
    Ok(raise_provider(py, err.clone()).into_value(py).into_any())
}

/// For a `#[host(error)]` kind that names one. A failed import is the error instead.
pub fn host_exception(module: &str, class: &str, message: String) -> PyErr {
    Python::attach(|py| {
        let class = py.import(module).and_then(|m| m.getattr(class));
        class.and_then(|c| c.call1((message,))).map_or_else(|e| e, PyErr::from_value)
    })
}

/// Works once across all clones; a second call is a `RuntimeError`.
pub fn raise_original(py: Python<'_>, host_ref: CapturedError) -> PyErr {
    let id = host_ref.id();
    match release(host_ref) {
        Some(obj) => PyErr::from_value(obj.into_bound(py)),
        None => PyRuntimeError::new_err(format!(
            "host exception {id} is no longer in the side table (already re-raised or released)"
        )),
    }
}

/// A `RuntimeError` for a diagnostic that had no original.
pub fn raise_host_error(e: crate::errors::HostError) -> PyErr {
    Python::attach(|py| raise_host(py, e))
}

/// The `map_err` of every bridge await.
pub fn captured(err: PyErr) -> ProviderError {
    Python::attach(|py| capture(py, &err))
}

// Two tables, counted separately so neither hides a leak in the other. Dropping a `Py<PyAny>`
// needs the GIL and there is no drain thread, so they drain at every GIL-holding entry point.

/// Both live for the process, so keys never outlive a registry.
pub type Table = PayloadTable<Py<PyAny>>;

pub fn media() -> &'static Table {
    static MEDIA: OnceLock<Table> = OnceLock::new();
    MEDIA.get_or_init(Table::new)
}

pub fn errors() -> &'static Table {
    static ERRORS: OnceLock<Table> = OnceLock::new();
    ERRORS.get_or_init(Table::new)
}

/// A host media object: `rtc.VideoFrame`, or the whole `list[rtc.AudioFrame]`.
pub fn intern_media(obj: Py<PyAny>) -> MediaRef {
    let t = media();
    MediaRef::new(t.insert(obj), t.registry())
}

pub fn intern_host(obj: Py<PyAny>) -> HostRef {
    let t = media();
    HostRef::new(t.insert(obj), t.registry())
}

pub fn resolve(py: Python<'_>, id: u64) -> Option<Py<PyAny>> {
    media().resolve(id, |obj| obj.clone_ref(py))
}

/// A miss means the table is corrupt; raise rather than panic.
pub fn resolve_key(py: Python<'_>, id: u64) -> PyResult<Py<PyAny>> {
    resolve(py, id).ok_or_else(|| {
        PyValueError::new_err(format!("host media ref {id} is no longer in the side table"))
    })
}

/// Resolve a `HostRef` id back to the object the host interned.
///
/// Read, not released: the entry belongs to whatever core value still holds the id.
#[pyfunction]
pub fn resolve_host_ref(py: Python<'_>, id: u64) -> Option<Py<PyAny>> {
    resolve(py, id)
}

crate::register_host!(fn self::resolve_host_ref);
