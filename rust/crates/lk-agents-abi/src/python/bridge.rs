//! The binding's async runtime the in-place expansions await on, host failures as the core's
//! errors, and the native classes a generated proxy takes as themselves.

use pyo3::prelude::*;

use crate::errors::ProviderError;

use super::host_errors::capture;

/// The binding's async runtime, submitted by `lk-agents-py`: the in-place expansions await on it.
pub struct HostRuntime {
    /// The runtime's fork guard, as a host exception.
    pub check_pid: fn() -> PyResult<()>,
    /// A future resolving to a host value as the awaitable a `_core` coroutine method returns.
    pub future_into_py: for<'py> fn(
        Python<'py>,
        crate::BoxFuture<'static, PyResult<Py<PyAny>>>,
    ) -> PyResult<Bound<'py, PyAny>>,
}

inventory::collect!(HostRuntime);

fn linked() -> Option<&'static HostRuntime> {
    inventory::iter::<HostRuntime>.into_iter().next()
}

pub(super) fn host_runtime() -> &'static HostRuntime {
    linked().expect("no host runtime is linked (lk-agents-py)")
}

/// A Rust future resolving to a host value as the awaitable a `_core` coroutine method returns, on
/// [`HostRuntime`].
pub fn future_into_py(
    py: Python<'_>,
    fut: impl std::future::Future<Output = PyResult<Py<PyAny>>> + Send + 'static,
) -> PyResult<Bound<'_, PyAny>> {
    (host_runtime().future_into_py)(py, Box::pin(fut))
}

// --------------------------------------------------------------------- proxies (`#[host(proxy)]`)

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
impl FromHostErr for crate::errors::HostError {
    fn from_host(py: Python<'_>, err: &PyErr) -> Self {
        super::host_errors::capture_host(py, err)
    }
}

/// A `_core` class whose value the core uses as it is rather than proxying; submitted
/// (`inventory`) by the owning crate and tried first by a generated proxy's `from_host`.
pub struct HostNative {
    /// A fn: a `TypeId` is not const-constructible.
    pub trait_id: fn() -> std::any::TypeId,
    pub read: fn(&Bound<'_, PyAny>) -> Option<Box<dyn std::any::Any>>,
}

inventory::collect!(HostNative);

pub fn native<T: ?Sized + 'static>(obj: &Bound<'_, PyAny>) -> Option<std::sync::Arc<T>> {
    let id = std::any::TypeId::of::<T>();
    for n in inventory::iter::<HostNative> {
        if (n.trait_id)() != id {
            continue;
        }
        if let Some(b) = (n.read)(obj)
            && let Ok(arc) = b.downcast::<std::sync::Arc<T>>()
        {
            return Some(*arc);
        }
    }
    None
}
