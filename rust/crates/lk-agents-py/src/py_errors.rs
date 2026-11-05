//! Python exception capture, classification and restoration.

/// Interpreter interface namespaces. Generated inherent methods perform the declared
/// imports and member access; they accept Python objects without asserting their runtime type.
pub mod interfaces {
    #[lk_agents_abi::host(interface(
        host = "builtins.BaseException",
        attrs(add_note, traceback(name = "__traceback__"))
    ))]
    pub struct BaseException;

    #[lk_agents_abi::host(interface(host = "livekit.agents._bindings.APIError"))]
    pub struct ApiError;

    #[lk_agents_abi::host(interface(host = "livekit.agents._bindings.APIStatusError"))]
    pub struct ApiStatusError;

    #[lk_agents_abi::host(interface(host = "livekit.agents._bindings.APIConnectionError"))]
    pub struct ApiConnectionError;

    #[lk_agents_abi::host(interface(host = "livekit.agents._bindings.APITimeoutError"))]
    pub struct ApiTimeoutError;
}

use crate::generated::accessors as access;
use crate::py_objects::exception_table;
use lk_agents_abi::errors::ApiErrorKind;
use lk_agents_abi::errors::CapturedError;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::errors::TaxonomyMatches;
use lk_agents_abi::errors::host_retryable;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;
use pyo3::types::PyType;

// A host exception goes into the side table and only its key plus scalar `retryable`/`status_code`/
// kind cross, so it re-raises with traceback and subclass intact. `raise_original` takes it once.

// --------------------------------------------------------------------- the side table

fn insert(obj: Py<PyAny>) -> CapturedError {
    let t = exception_table();
    CapturedError::new(t.insert(obj), t.registry())
}

/// Peek without taking.
pub fn resolve_captured(py: Python<'_>, host_ref: CapturedError) -> Option<Py<PyAny>> {
    exception_table().resolve(&host_ref, |obj| obj.clone_ref(py))
}

/// Every clone of the key answers `None` afterwards and releases nothing on drop. Idempotent.
pub fn release(host_ref: CapturedError) -> Option<Py<PyAny>> {
    exception_table().take(&host_ref)
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
            Ok(ErrorTypes {
                api: access::ApiError::import(py)?.cast_into::<PyType>()?.unbind(),
                status: access::ApiStatusError::import(py)?.cast_into::<PyType>()?.unbind(),
                connection: access::ApiConnectionError::import(py)?.cast_into::<PyType>()?.unbind(),
                timeout: access::ApiTimeoutError::import(py)?.cast_into::<PyType>()?.unbind(),
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
    let metadata = crate::generated::records::host_errors::provider_error_metadata_from_py(value)
        .unwrap_or_default();
    let status_code = metadata.status_code;
    let matches = taxonomy_matches(value);
    let api_kind = matches.and_then(|m| ApiErrorKind::from_matches(m, status_code));
    let declared = metadata.retryable;
    let retryable = host_retryable(api_kind, matches.is_some(), declared);

    ProviderError::Host { error: capture_host_value(value), api_kind, retryable, status_code }
}

/// Without inventing retry or fatality policy.
pub fn capture_host(py: Python<'_>, err: &PyErr) -> lk_agents_abi::errors::HostError {
    capture_host_value(err.value(py).as_any())
}

fn capture_host_value(value: &Bound<'_, PyAny>) -> lk_agents_abi::errors::HostError {
    let kind =
        value.get_type().name().map(|n| n.to_string()).unwrap_or_else(|_| "BaseException".into());
    // An exception's __str__ can itself raise. Preserve the original even in that case.
    let message = value.str().map(|s| s.to_string()).unwrap_or_default();
    lk_agents_abi::errors::HostError::captured(kind, message, insert(value.clone().unbind()))
        .with_traceback(crate::py_calls::traceback(value))
}

pub fn raise_host(py: Python<'_>, error: lk_agents_abi::errors::HostError) -> PyErr {
    match error.host_ref {
        Some(reference) => raise_original(py, reference),
        None => PyRuntimeError::new_err(error.to_string()),
    }
}

pub fn capture(py: Python<'_>, err: &PyErr) -> ProviderError {
    capture_value(&err.value(py).clone().into_any())
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
    Ok(crate::py_errors::HostException::to_py_err(err).into_value(py).into_any())
}

/// The exception object an observer or event receives for a failure.
pub trait ExceptionValue {
    fn to_exception<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>>;
}

/// The host's own exception (resolved, not released), else an `APIError` from the engine's message.
impl ExceptionValue for ProviderError {
    fn to_exception<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(to_py_exception(py, &self)?.into_bound(py))
    }
}

/// The host's own exception while it is retained (resolved, not released), else a `RuntimeError`
/// carrying the diagnostic: what an observer records or logs.
impl ExceptionValue for lk_agents_abi::errors::HostError {
    fn to_exception<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self.host_ref.clone().and_then(|r| resolve_captured(py, r)) {
            Some(original) => original.into_bound(py),
            None => {
                PyRuntimeError::new_err(self.to_string()).into_value(py).into_bound(py).into_any()
            }
        })
    }
}

/// For a `#[host(kind = "..", kwargs)]` variant: the class called with the message and the
/// keyword arguments `kwargs` sets. A failed import or call is the error instead.
pub fn host_exception_with(
    module: &str,
    class: &str,
    message: String,
    kwargs: impl FnOnce(&Bound<'_, PyDict>) -> PyResult<()>,
) -> PyErr {
    Python::attach(|py| {
        let build = || -> PyResult<PyErr> {
            let arguments = PyDict::new(py);
            kwargs(&arguments)?;
            let class = py.import(module)?.getattr(class)?;
            Ok(PyErr::from_value(crate::py_calls::call(&class, (message,), Some(&arguments))?))
        };
        build().unwrap_or_else(|e| e)
    })
}

/// For a `#[host(error)]` kind that names one. A failed import is the error instead.
pub fn host_exception(module: &str, class: &str, message: String) -> PyErr {
    Python::attach(|py| {
        let class = py.import(module).and_then(|m| m.getattr(class));
        class
            .and_then(|c| crate::py_calls::call(&c, (message,), None))
            .map_or_else(|e| e, PyErr::from_value)
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
pub fn raise_host_error(e: lk_agents_abi::errors::HostError) -> PyErr {
    Python::attach(|py| raise_host(py, e))
}

/// The `map_err` of every bridge await.
pub fn captured(err: PyErr) -> ProviderError {
    Python::attach(|py| capture(py, &err))
}

/// The host exception a declared error raises (`#[host(error)]`, generated).
pub trait HostException {
    fn to_py_err(&self) -> PyErr;
}

/// Implemented beside each error type a proxied trait returns.
pub trait FromHostErr {
    fn from_host(py: Python<'_>, err: &PyErr) -> Self;
}

impl FromHostErr for ProviderError {
    fn from_host(py: Python<'_>, err: &PyErr) -> Self {
        capture(py, err)
    }
}

/// Keeps the original exception, so re-raising it raises what the host raised.
impl FromHostErr for lk_agents_abi::errors::HostError {
    fn from_host(py: Python<'_>, err: &PyErr) -> Self {
        crate::py_errors::capture_host(py, err)
    }
}

/// The host's exception as the failure a reply request reports: its message and its `code`.
impl FromHostErr for lk_agents_abi::llm::realtime::RealtimeError {
    fn from_host(py: Python<'_>, err: &PyErr) -> Self {
        let value = err.value(py);
        let metadata =
            crate::generated::records::host_errors::realtime_error_metadata_from_py(value.as_any())
                .unwrap_or_default();
        Self { message: metadata.message, code: metadata.code }
    }
}
