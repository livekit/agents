//! The binding-owned side table. `release` only queues the id; `drain` removes entries where
//! dropping a payload is legal, opportunistically on insert, resolve and take.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use super::RefRegistry;

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
        // Called from `Drop`: recover a poisoned lock rather than panic (which would abort).
        let mut queue = lock(&self.pending);
        queue.push(id);
        self.pending_len.store(queue.len(), Ordering::Release);
    }

    /// Does not drain; `pending` is a subset of `entries`.
    fn outstanding(&self) -> usize {
        let pending = lock(&self.pending);
        let entries = lock(&self.entries);
        entries.len().saturating_sub(pending.len())
    }
}

/// The critical sections cannot leave the containers inconsistent.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
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
        lock(&self.store.entries).insert(id, payload);
        id
    }

    pub fn resolve<R>(&self, id: u64, f: impl FnOnce(&T) -> R) -> Option<R> {
        self.drain_if_pending();
        lock(&self.store.entries).get(&id).map(f)
    }

    /// The binding's half of `CapturedError::take`.
    pub fn take(&self, id: u64) -> Option<T> {
        self.drain_if_pending();
        lock(&self.store.entries).remove(&id)
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
            let mut queue = lock(&store.pending);
            store.pending_len.store(0, Ordering::Release);
            std::mem::take(&mut *queue)
        };

        // Drop after the guard: a finaliser can run host code that interns into `entries`.
        let removed: Vec<T> = {
            let mut entries = lock(&store.entries);
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
        lock(&self.store.entries).len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HostRef;

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
        let key = crate::errors::CapturedError::new(first, table.registry());
        assert_eq!(key.take().and_then(|id| table.take(id)), Some(1));
        assert_eq!(table.take(first), None, "taken means gone");
        drop(key);
        let second = table.insert(2);
        assert_ne!(second, first);
        assert_eq!(table.len_after_drain(), 1, "the taken key's drop released nothing");
        assert_eq!(table.resolve(second, |v| *v), Some(2));
    }
}
