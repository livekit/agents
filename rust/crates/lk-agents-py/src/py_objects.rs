//! Retained Python objects, reference lifetimes and native capability lookup.

use lk_agents_abi::errors::CapturedError;
use lk_agents_abi::errors::RefRegistry;
use lk_agents_abi::utils::host_ref::{ObjectRef, StaleHostRef};
use parking_lot::Mutex;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

struct Store<T> {
    entries: Mutex<HashMap<u64, T>>,
    /// Separate so `release` never contends with a lookup.
    pending: Mutex<Vec<u64>>,
    /// So "nothing to drain" is one atomic load.
    pending_len: AtomicUsize,
    /// Never reused; shared by every key type into this table.
    next_id: AtomicU64,
}

impl<T: Send + 'static> RefRegistry for Store<T> {
    fn release(&self, id: u64) {
        let mut queue = self.pending.lock();
        queue.push(id);
        self.pending_len.store(queue.len(), Ordering::Release);
    }

    /// Does not drain; `pending` is a subset of `entries`.
    fn outstanding(&self) -> usize {
        let pending = self.pending.lock();
        let entries = self.entries.lock();
        entries.len().saturating_sub(pending.len())
    }
}

/// Keep it for the process: keys hold a `Weak` to its registry.
pub struct PayloadTable<T> {
    store: Arc<Store<T>>,
    registry: Arc<dyn RefRegistry>,
}

impl<T: Send + 'static> Default for PayloadTable<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Send + 'static> PayloadTable<T> {
    pub fn new() -> Self {
        let store = Arc::new(Store {
            entries: Mutex::new(HashMap::new()),
            pending: Mutex::new(Vec::new()),
            pending_len: AtomicUsize::new(0),
            // Ids start at 1 so that 0 is never a live entry.
            next_id: AtomicU64::new(1),
        });
        let registry: Arc<dyn RefRegistry> = Arc::<Store<T>>::clone(&store);
        PayloadTable { store, registry }
    }

    pub fn registry(&self) -> &Arc<dyn RefRegistry> {
        &self.registry
    }

    /// One entry per call.
    pub fn insert(&self, payload: T) -> u64 {
        self.drain_if_pending();
        let id = self.store.next_id.fetch_add(1, Ordering::Relaxed);
        self.store.entries.lock().insert(id, payload);
        id
    }

    /// Look up a live key from this table; another table's numeric ID is never sufficient.
    pub fn resolve<R>(
        &self,
        key: &impl AsRef<lk_agents_abi::utils::host_ref::RefEntry>,
        f: impl FnOnce(&T) -> R,
    ) -> Option<R> {
        let entry = key.as_ref();
        if !entry.is_from_registry(self.registry()) {
            return None;
        }
        self.drain_if_pending();
        self.store.entries.lock().get(&entry.id()).map(f)
    }

    /// The binding's half of `CapturedError::take`.
    pub fn take(&self, key: &CapturedError) -> Option<T> {
        if !key.as_ref().is_from_registry(self.registry()) {
            return None;
        }
        self.drain_if_pending();
        let id = key.take()?;
        self.store.entries.lock().remove(&id)
    }

    fn drain_if_pending(&self) {
        if self.store.pending_len.load(Ordering::Acquire) > 0 {
            self.drain();
        }
    }

    /// Call where the payload's `Drop` is legal (with the GIL, for Python).
    pub fn drain(&self) -> usize {
        let store = &self.store;
        let ids = {
            let mut queue = store.pending.lock();
            store.pending_len.store(0, Ordering::Release);
            std::mem::take(&mut *queue)
        };

        // Drop after the guard: a finaliser can run host code that interns into `entries`.
        let removed: Vec<T> = {
            let mut entries = store.entries.lock();
            ids.iter().filter_map(|id| entries.remove(id)).collect()
        };
        let n = removed.len();
        drop(removed);
        n
    }

    pub fn outstanding(&self) -> usize {
        self.store.outstanding()
    }

    pub fn len_after_drain(&self) -> usize {
        self.drain();
        self.store.entries.lock().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use lk_agents_abi::HostRef;

    // Catches: release removing the entry inline instead of queueing, or the queue never draining.
    #[test]
    fn a_dropped_key_queues_its_entry_and_the_next_drain_removes_it() {
        let table = PayloadTable::<String>::new();
        let id = table.insert("frame".to_owned());
        let key = HostRef::new(id, table.registry());
        assert_eq!(table.outstanding(), 1);

        std::thread::spawn(move || drop(key)).join().unwrap();
        assert_eq!(table.outstanding(), 0, "released, even before the drain");
        assert_eq!(table.len_after_drain(), 0);
        assert_eq!(table.drain(), 0, "a second drain finds nothing");
    }

    // Catches: `take` leaving the entry, or the key's drop after a take releasing the id again.
    #[test]
    fn take_moves_the_payload_out_once_and_ids_are_never_reused() {
        let table = PayloadTable::<u8>::new();
        let first = table.insert(1);
        let key = lk_agents_abi::errors::CapturedError::new(first, table.registry());
        assert_eq!(table.take(&key), Some(1));
        assert_eq!(table.take(&key), None, "taken means gone");
        drop(key);
        let second = table.insert(2);
        assert_ne!(second, first);
        assert_eq!(table.len_after_drain(), 1, "the taken key's drop released nothing");
        let second_key = HostRef::new(second, table.registry());
        assert_eq!(table.resolve(&second_key, |v| *v), Some(2));
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

// Two tables, counted separately so neither hides a leak in the other. Dropping a `Py<PyAny>`
// needs the GIL and there is no drain thread, so they drain at every GIL-holding entry point.

/// Both live for the process, so keys never outlive a registry.
pub type Table = PayloadTable<Py<PyAny>>;

pub fn object_table() -> &'static Table {
    static OBJECTS: OnceLock<Table> = OnceLock::new();
    OBJECTS.get_or_init(Table::new)
}

pub fn exception_table() -> &'static Table {
    static ERRORS: OnceLock<Table> = OnceLock::new();
    ERRORS.get_or_init(Table::new)
}

/// Retain one opaque Python object under its ABI-declared reference type.
pub fn retain<Kind>(obj: Py<PyAny>) -> ObjectRef<Kind> {
    let table = object_table();
    ObjectRef::new(table.insert(obj), table.registry())
}

/// Resolve a typed reference only in the table that issued it, preserving Python identity.
pub fn resolve<Kind>(py: Python<'_>, reference: &ObjectRef<Kind>) -> PyResult<Py<PyAny>> {
    object_table().resolve(reference, |obj| obj.clone_ref(py)).ok_or_else(|| {
        crate::py_errors::HostException::to_py_err(&StaleHostRef { id: reference.id() })
    })
}

/// The shared primitive selected by retained-object ABI declarations.
pub fn object_ref_from_py<Kind>(obj: &Bound<'_, PyAny>) -> PyResult<ObjectRef<Kind>> {
    Ok(retain(obj.clone().unbind()))
}

pub fn object_ref_to_py<Kind>(
    py: Python<'_>,
    value: ObjectRef<Kind>,
) -> PyResult<Bound<'_, PyAny>> {
    Ok(resolve(py, &value)?.into_bound(py))
}

/// The host's own exception, released: `None` when another clone already took it.
pub fn captured_error_to_py<'py>(
    py: Python<'py>,
    value: CapturedError,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(crate::py_errors::release(value)
        .map_or_else(|| py.None().into_bound(py), |obj| obj.into_bound(py)))
}
