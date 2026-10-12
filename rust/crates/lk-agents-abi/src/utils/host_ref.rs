//! Opaque host payloads Rust moves and never inspects. Released by refcount, not per generation:
//! committed history outlives any one generation's snapshot.

pub use crate::errors::RefRegistry;
use crate::errors::side_table::side_table_ref;

side_table_ref!(HostRef, "an opaque host object (`extra`, `AgentConfigUpdate._tools`)");
side_table_ref!(MediaRef, "a host media object (`rtc.VideoFrame`, `list[rtc.AudioFrame]`)");

/// The reverse of [`HostRef`]: a core object the host receives as a live handle.
#[derive(Clone)]
pub struct CoreObject(pub std::sync::Arc<dyn std::any::Any + Send + Sync>);

impl std::fmt::Debug for CoreObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CoreObject(..)")
    }
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
        Ok(match crate::python::resolve(py, self.id()) {
            Some(obj) => obj.into_bound(py),
            None => py.None().into_bound(py),
        })
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
        Ok(crate::python::resolve_key(py, self.id())?.into_bound(py))
    }
}

#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for MediaRef {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        Ok(crate::python::intern_media(obj.to_owned().unbind()))
    }
}

/// Submitted (`inventory`) beside the class, so [`CoreObject`]'s `IntoPyObject` needs no table of
/// every crate's types.
#[cfg(feature = "python")]
pub struct CoreObjectClass {
    pub to_host: for<'py> fn(pyo3::Python<'py>, &std::sync::Arc<dyn std::any::Any + Send + Sync>) -> Option<pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>>>,
}

#[cfg(feature = "python")]
inventory::collect!(CoreObjectClass);

#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for &CoreObject {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, pyo3::PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        for class in inventory::iter::<CoreObjectClass> {
            if let Some(out) = (class.to_host)(py, &self.0) {
                return out;
            }
        }
        Err(pyo3::exceptions::PyTypeError::new_err("a core object with no host class"))
    }
}

#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for CoreObject {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, pyo3::PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        (&self).into_pyobject(py)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    /// Stand-in for the binding's table: records what it was asked to forget.
    struct Table(Arc<Mutex<Vec<u64>>>);

    impl RefRegistry for Table {
        fn release(&self, id: u64) {
            self.0.lock().unwrap().push(id);
        }
    }

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

    // Catches: a per-generation snapshot freeing an `extra` that committed history still holds.
    #[test]
    fn a_cloned_snapshot_keeps_the_payload_alive() {
        let (registry, log) = table();
        let history = vec![HostRef::new(1, &registry), HostRef::new(2, &registry)];

        let snapshot = history.clone();
        drop(snapshot);
        assert!(log.lock().unwrap().is_empty(), "a discarded snapshot released the host's data");

        drop(history);
        assert_eq!(*log.lock().unwrap(), vec![1, 2]);
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

    // Catches: `MediaRef` and `HostRef` collapsed into one type or one id space.
    #[test]
    fn a_media_ref_and_a_host_ref_with_the_same_id_are_separate_entries() {
        let (registry, log) = table();
        let h = HostRef::new(4, &registry);
        let m = MediaRef::new(4, &registry);
        assert_eq!(h.id(), m.id());

        drop(h);
        drop(m);
        assert_eq!(*log.lock().unwrap(), vec![4, 4]);
    }
}
