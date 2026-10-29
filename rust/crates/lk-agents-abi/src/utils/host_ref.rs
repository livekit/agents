//! Opaque host payloads Rust moves and never inspects. Released by refcount, not per generation:
//! committed history outlives any one generation's snapshot.

pub use crate::errors::RefRegistry;
use crate::errors::side_table::side_table_ref;

/// A key into the binding's side table for an opaque host object (`extra`,
/// `AgentConfigUpdate._tools`).
#[crate::host(handwritten(extract, emit, host = "object"))]
#[derive(Clone)]
pub struct HostRef(std::sync::Arc<crate::errors::side_table::Entry>);
side_table_ref!(HostRef);

/// A key into the binding's side table for a host media object (`rtc.VideoFrame`,
/// `list[rtc.AudioFrame]`).
#[crate::host(handwritten(extract, emit, host = "object"))]
#[derive(Clone)]
pub struct MediaRef(std::sync::Arc<crate::errors::side_table::Entry>);
side_table_ref!(MediaRef);

/// A handle whose originating host no longer owns a live entry.
#[crate::host(error(kind = "StaleHandle", declare = "RuntimeError"))]
#[derive(Debug, Clone, thiserror::Error)]
#[error("host reference {id} is stale or belongs to another runtime")]
pub struct StaleHostRef {
    pub id: u64,
}

#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for HostRef {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, pyo3::PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        (&self).into_pyobject(py)
    }
}

#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for &HostRef {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, pyo3::PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        let owned = self.belongs_to(crate::python::media().registry());
        Ok(resolve_owned(py, self.id(), owned)?.into_bound(py))
    }
}

/// Interned; the key keeps it alive.
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for HostRef {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        Ok(crate::python::intern_host(obj.to_owned().unbind()))
    }
}

/// Callers compare by identity; a missing entry raises.
#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for MediaRef {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, pyo3::PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        (&self).into_pyobject(py)
    }
}

#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for &MediaRef {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, pyo3::PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        let owned = self.belongs_to(crate::python::media().registry());
        Ok(resolve_owned(py, self.id(), owned)?.into_bound(py))
    }
}

#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for MediaRef {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        Ok(crate::python::intern_media(obj.to_owned().unbind()))
    }
}

impl MediaRef {
    /// Validate every audio frame a host list holds; a stale reference or invalid frame fails.
    pub fn audio_frames(&self) -> Result<Vec<crate::AudioFrame>, crate::errors::ProviderError> {
        #[cfg(feature = "python")]
        {
            use pyo3::types::PyAnyMethods as _;
            pyo3::Python::attach(|py| {
                let owned = self.belongs_to(crate::python::media().registry());
                let list = resolve_owned(py, self.id(), owned).map_err(crate::python::captured)?;
                let mut frames = Vec::new();
                for (index, item) in
                    list.bind(py).try_iter().map_err(crate::python::captured)?.enumerate()
                {
                    let frame = item
                        .and_then(|item| item.extract::<crate::AudioFrame>())
                        .map_err(|e| crate::python::at_path(py, e, &format!("frames[{index}]")));
                    frames.push(frame.map_err(crate::python::captured)?);
                }
                Ok(frames)
            })
        }
        #[cfg(not(feature = "python"))]
        Err(crate::errors::ApiError {
            retryable: false,
            ..crate::errors::ApiError::new("host media cannot be read without a host runtime")
        }
        .into())
    }
}

/// The host object `id` names, provided its handle was issued by this runtime's registry.
#[cfg(feature = "python")]
fn resolve_owned(
    py: pyo3::Python<'_>,
    id: u64,
    owned: bool,
) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
    if !owned {
        return Err(crate::python::HostException::to_py_err(&StaleHostRef { id }));
    }
    crate::python::resolve_key(py, id)
}

#[cfg(test)]
mod tests {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    use std::sync::{Arc, Mutex};

    use super::*;

    /// Stand-in for the binding's table: records what it was asked to forget.
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    struct Table(Arc<Mutex<Vec<u64>>>);

    impl RefRegistry for Table {
        fn release(&self, id: u64) {
            self.0.lock().unwrap().push(id);
        }
    }

    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    fn table() -> (Arc<dyn RefRegistry>, Arc<Mutex<Vec<u64>>>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        (Arc::new(Table(log.clone())), log)
    }

    // Catches: releasing per clone, or never releasing.
    #[test]
    fn release_fires_once_when_the_last_clone_drops() {
        let (registry, log) = table();
        let a = HostRef::new(7, &registry);
        let b = a.clone();
        let c = a.clone();

        drop(a);
        drop(b);
        assert!(log.lock().unwrap().is_empty(), "released while a clone was still live");

        drop(c);
        assert_eq!(*log.lock().unwrap(), vec![7]);
    }

    // Catches: holding the registry by `Arc` (keeps host objects alive past shutdown), or
    // unwrapping the upgrade inside `Drop`.
    #[test]
    fn release_after_the_registry_is_gone_is_a_no_op() {
        let (registry, log) = table();
        let r = HostRef::new(9, &registry);

        drop(registry);
        drop(r);

        assert!(log.lock().unwrap().is_empty());
    }

}
