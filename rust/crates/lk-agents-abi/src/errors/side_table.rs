//! The refcounted side-table keys. The registry is held by [`Weak`] so keys never keep the host
//! table alive; `release` may run on any thread, so it must not block or take the GIL.

#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

/// No resolution here: a pure crate must not read host memory.
pub trait RefRegistry: Send + Sync + 'static {
    /// Called from `Drop` on any thread: no panics, no blocking, no GIL.
    fn release(&self, id: u64);

    /// The leak detector. The default `0` suits a registry that owns no payloads.
    fn outstanding(&self) -> usize {
        0
    }
}

/// Shared reference metadata, exported as `RefEntry` for checked host-table access.
pub struct Entry {
    id: u64,
    registry: Weak<dyn RefRegistry>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// So `Drop` does not release a moved payload.
    taken: AtomicBool,
}

impl Entry {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    pub fn new(id: u64, registry: &Arc<dyn RefRegistry>) -> Self {
        Entry { id, registry: Arc::downgrade(registry), taken: AtomicBool::new(false) }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn is_from_registry(&self, registry: &Arc<dyn RefRegistry>) -> bool {
        self.registry.ptr_eq(&Arc::downgrade(registry))
    }

    pub(crate) fn same_registry(&self, other: &Self) -> bool {
        self.registry.ptr_eq(&other.registry)
    }

    /// `Some` once, then `None` from any clone.
    pub(crate) fn take(&self) -> Option<u64> {
        if self.taken.swap(true, Ordering::AcqRel) { None } else { Some(self.id) }
    }
}

impl Drop for Entry {
    fn drop(&mut self) {
        if self.taken.load(Ordering::Acquire) {
            return;
        }
        if let Some(registry) = self.registry.upgrade() {
            registry.release(self.id);
        }
    }
}

/// The behavior of a key `pub struct Name(Arc<Entry>)`, written out where it is declared so its
/// declaration (its host attributes) is the codegen's to read. `Clone` shares the entry, `Drop` of
/// the last clone releases. `take` is added only for exceptions.
macro_rules! side_table_ref {
    ($name:ident) => {
        impl $name {
            /// `id` must already be in `registry` and never passed to a second `new`.
            pub fn new(
                id: u64,
                registry: &::std::sync::Arc<dyn $crate::errors::RefRegistry>,
            ) -> Self {
                $name(::std::sync::Arc::new($crate::errors::side_table::Entry::new(id, registry)))
            }

            /// Nothing in the core reads it.
            pub fn id(self: &Self) -> u64 {
                self.0.id()
            }
        }

        impl AsRef<$crate::errors::side_table::Entry> for $name {
            fn as_ref(&self) -> &$crate::errors::side_table::Entry {
                &self.0
            }
        }

        /// By id: the payload is not inspectable here.
        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.0.id() == other.0.id() && self.0.same_registry(&other.0)
            }
        }

        impl Eq for $name {}

        impl ::std::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                write!(f, concat!(stringify!($name), "({})"), self.0.id())
            }
        }
    };
}

pub(crate) use side_table_ref;

#[cfg(test)]
pub(crate) mod testkit {
    //! A stand-in for the binding's table, shared by the `HostRef` and `CapturedError` tests.

    use std::collections::BTreeSet;
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    use std::sync::{Arc, Mutex};

    use super::RefRegistry;

    /// Holds ids the way the binding does, and answers `outstanding` from what is left.
    pub struct Table {
        #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
        live: Mutex<BTreeSet<u64>>,
    }

    impl Table {
        #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
        /// A table already holding `ids`, as if the binding had captured them.
        pub fn holding(ids: impl IntoIterator<Item = u64>) -> Arc<Table> {
            Arc::new(Table { live: Mutex::new(ids.into_iter().collect()) })
        }

        pub fn as_registry(self: &Arc<Self>) -> Arc<dyn RefRegistry> {
            Arc::clone(self) as Arc<dyn RefRegistry>
        }
    }

    impl RefRegistry for Table {
        fn release(&self, id: u64) {
            self.live.lock().unwrap().remove(&id);
        }

        fn outstanding(&self) -> usize {
            self.live.lock().unwrap().len()
        }
    }
}
