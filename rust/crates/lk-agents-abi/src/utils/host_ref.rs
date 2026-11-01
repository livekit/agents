//! Opaque host payloads Rust moves and never inspects. Released by refcount, not per generation:
//! committed history outlives any one generation's snapshot.

pub use crate::errors::RefRegistry;
pub use crate::errors::side_table::Entry as RefEntry;
use crate::errors::side_table::Entry;
use std::marker::PhantomData;
use std::sync::Arc;

/// A retained object with a compile-time role; all roles share one ownership implementation.
pub struct ObjectRef<Kind> {
    entry: Arc<Entry>,
    kind: PhantomData<fn() -> Kind>,
}

impl<Kind> ObjectRef<Kind> {
    /// `id` must already be in `registry` and must not be passed to another `new` call.
    pub fn new(id: u64, registry: &Arc<dyn RefRegistry>) -> Self {
        Self { entry: Arc::new(Entry::new(id, registry)), kind: PhantomData }
    }

    pub fn id(&self) -> u64 {
        self.entry.id()
    }
}

impl<Kind> Clone for ObjectRef<Kind> {
    fn clone(&self) -> Self {
        Self { entry: self.entry.clone(), kind: PhantomData }
    }
}

impl<Kind> AsRef<RefEntry> for ObjectRef<Kind> {
    fn as_ref(&self) -> &RefEntry {
        &self.entry
    }
}

impl<Kind> PartialEq for ObjectRef<Kind> {
    fn eq(&self, other: &Self) -> bool {
        self.entry.id() == other.entry.id() && self.entry.same_registry(&other.entry)
    }
}

impl<Kind> Eq for ObjectRef<Kind> {}

impl<Kind> std::fmt::Debug for ObjectRef<Kind> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ObjectRef").field(&self.id()).finish()
    }
}

/// Distinguishes an arbitrary opaque host object from media in Rust's type system.
pub enum HostObject {}
/// Distinguishes retained media from arbitrary host configuration objects.
pub enum MediaObject {}

/// A key into the binding's side table for an opaque host object (`extra`,
/// `AgentConfigUpdate._tools`).
#[crate::host(codec(extract, emit, host = "object", py = "py_objects::object_ref"))]
pub type HostRef = ObjectRef<HostObject>;

/// A key into the binding's side table for a host media object (`rtc.VideoFrame`,
/// `list[rtc.AudioFrame]`).
#[crate::host(codec(extract, emit, host = "object", py = "py_objects::object_ref"))]
pub type MediaRef = ObjectRef<MediaObject>;

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
