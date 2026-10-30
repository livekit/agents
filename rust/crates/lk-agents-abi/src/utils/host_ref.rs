//! Opaque host payloads Rust moves and never inspects. Released by refcount, not per generation:
//! committed history outlives any one generation's snapshot.

pub use crate::errors::RefRegistry;
use crate::errors::side_table::side_table_ref;

/// A key into the binding's side table for an opaque host object (`extra`,
/// `AgentConfigUpdate._tools`).
#[crate::host(handwritten(extract, emit, host = "object", py))]
#[derive(Clone)]
pub struct HostRef(std::sync::Arc<crate::errors::side_table::Entry>);
side_table_ref!(HostRef);

/// A key into the binding's side table for a host media object (`rtc.VideoFrame`,
/// `list[rtc.AudioFrame]`).
#[crate::host(handwritten(extract, emit, host = "object", py))]
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

#[cfg(test)]
mod tests {
    use std::sync::{Arc, OnceLock};

    use super::*;

    /// Stand-in for the binding's table: records the one id it was asked to forget.
    struct Table(Arc<OnceLock<u64>>);

    impl RefRegistry for Table {
        fn release(&self, id: u64) {
            assert!(self.0.set(id).is_ok(), "released twice");
        }
    }

    fn table() -> (Arc<dyn RefRegistry>, Arc<OnceLock<u64>>) {
        let released = Arc::new(OnceLock::new());
        (Arc::new(Table(released.clone())), released)
    }

    // Catches: releasing per clone, or never releasing.
    #[test]
    fn release_fires_once_when_the_last_clone_drops() {
        let (registry, released) = table();
        let a = HostRef::new(7, &registry);
        let b = a.clone();
        let c = a.clone();

        drop(a);
        drop(b);
        assert!(released.get().is_none(), "released while a clone was still live");

        drop(c);
        assert_eq!(released.get(), Some(&7));
    }

    // Catches: holding the registry by `Arc` (keeps host objects alive past shutdown), or
    // unwrapping the upgrade inside `Drop`.
    #[test]
    fn release_after_the_registry_is_gone_is_a_no_op() {
        let (registry, released) = table();
        let r = HostRef::new(9, &registry);

        drop(registry);
        drop(r);

        assert!(released.get().is_none());
    }
}
