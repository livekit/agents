//! Host failures as the core's errors, and the native classes a generated proxy takes as themselves.

use pyo3::prelude::*;

use lk_agents_abi::errors::ProviderError;

use super::host_errors::capture;

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
impl FromHostErr for lk_agents_abi::errors::HostError {
    fn from_host(py: Python<'_>, err: &PyErr) -> Self {
        super::host_errors::capture_host(py, err)
    }
}

/// The host's exception as the failure a reply request reports: its message and its `code`.
impl FromHostErr for lk_agents_abi::llm::realtime::RealtimeError {
    fn from_host(py: Python<'_>, err: &PyErr) -> Self {
        let value = err.value(py);
        let code = value.getattr("code").ok().and_then(|c| c.extract::<String>().ok());
        Self { message: value.to_string(), code }
    }
}

/// Keeps the original exception, so re-raising it raises what the host raised.
impl FromHostErr for lk_agents_abi::voice::rtc::RtcError {
    fn from_host(py: Python<'_>, err: &PyErr) -> Self {
        Self::Host(super::host_errors::capture_host(py, err))
    }
}

/// Keeps the original exception, so re-raising it raises what the host raised.
impl FromHostErr for lk_agents_abi::voice::io::IoError {
    fn from_host(py: Python<'_>, err: &PyErr) -> Self {
        Self::Host(super::host_errors::capture_host(py, err))
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
