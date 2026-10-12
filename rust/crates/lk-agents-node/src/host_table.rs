//! §1.11's opaque-payload seam, with the table on the **other** side of the boundary.
//!
//! The rule is unchanged: *"opaque host payloads travel as a `u64` into a table the binding owns.
//! Rust moves the key and never inspects the payload."* What changes for Node is who holds the
//! map, and it changes for the same reason [`crate::errors`] does -- a `napi_ref` is thread-affine
//! and a `Py<PyAny>` is not. So:
//!
//! | | `lk-agents-py` | `lk-agents-node` |
//! |---|---|---|
//! | who allocates the id | Rust (`next_id`) | **JS** (it owns the map) |
//! | where the payload lives | Rust `HashMap<u64, Py<PyAny>>` | **JS `Map<number, unknown>`** |
//! | who drains releases | Rust, at any GIL-holding entry point | **JS**, via [`drain_released`] |
//! | `RefRegistry::release` | push onto a queue | push onto a queue -- *identical* |
//!
//! The last row is the point. `lk_agents_abi::{RefRegistry, HostRef, MediaRef}` needed **no
//! change at all** to serve a host with the opposite ownership: the trait asks exactly one thing
//! ("forget the entry") and explicitly forbids resolution, which is what leaves the table's
//! location free. That is direct evidence the core is host-neutral where it matters most.

use std::sync::{Arc, Mutex, OnceLock};

use lk_agents_abi::{HostRef, MediaRef, RefRegistry};

/// The one queue. Ids whose last `Arc` has dropped, waiting for the JS thread to delete them.
static RELEASED: Mutex<Vec<u64>> = Mutex::new(Vec::new());

/// Zero-sized forwarder, so the `'static` `Arc` owns no data of its own -- exactly
/// `lk-agents-py`'s `TableHandle`.
struct Handle;

impl RefRegistry for Handle {
    fn release(&self, id: u64) {
        // From `Drop`, on whatever thread dropped the last clone: no panic, no blocking, no JS.
        // The only lock taken guards a `Vec::push`.
        RELEASED.lock().unwrap_or_else(|p| p.into_inner()).push(id);
    }
}

fn registry() -> &'static Arc<dyn RefRegistry> {
    static REGISTRY: OnceLock<Arc<dyn RefRegistry>> = OnceLock::new();
    // Leaked for the life of the process, as in `lk-agents-py`: `HostRef` holds a `Weak`, and a
    // registry that could die while chat items are alive would silently stop releasing.
    REGISTRY.get_or_init(|| Arc::new(Handle))
}

/// Take ownership of a payload the shim has already put in its map under `id`.
///
/// The shim allocates `id`; Rust only promises to say when it may go. Handing the same id to two
/// `intern_*` calls would release it twice -- the same contract `HostRef::new` documents.
pub fn intern_media(id: u64) -> MediaRef {
    MediaRef::new(id, registry())
}

/// The `AgentConfigUpdate._tools` case: live host callables Rust routes but never calls directly.
pub fn intern_host(id: u64) -> HostRef {
    HostRef::new(id, registry())
}

/// Ids the shim may now `map.delete(...)`. Pull, never push.
pub fn drain_released() -> Vec<u64> {
    std::mem::take(&mut *RELEASED.lock().unwrap_or_else(|p| p.into_inner()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One release queue per process, so the tests that assert on its contents must not overlap.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn serial() -> std::sync::MutexGuard<'static, ()> {
        SERIAL.lock().unwrap_or_else(|p| p.into_inner())
    }

    #[test]
    fn a_clone_shares_the_entry_and_only_the_last_drop_releases() {
        let _serial = serial();
        let _ = drain_released();
        let a = intern_media(101);
        let b = a.clone();
        assert_eq!(a.id(), b.id());
        drop(a);
        assert!(
            drain_released().is_empty(),
            "a discarded snapshot must not free an image committed history still points at"
        );
        drop(b);
        assert_eq!(drain_released(), vec![101]);
    }

    #[test]
    fn host_and_media_refs_share_one_id_space_and_one_queue() {
        let _serial = serial();
        let _ = drain_released();
        drop(intern_host(202));
        drop(intern_media(203));
        let mut ids = drain_released();
        ids.sort_unstable();
        assert_eq!(ids, vec![202, 203]);
    }

    #[test]
    fn releasing_from_another_thread_never_blocks_and_never_takes_the_js_thread() {
        let _serial = serial();
        let _ = drain_released();
        let r = intern_media(303);
        std::thread::spawn(move || drop(r)).join().unwrap();
        assert_eq!(drain_released(), vec![303], "release must work from any thread");
    }
}
