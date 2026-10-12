//! The refcounted side-table keys. The registry is held by [`Weak`] so keys never keep the host
//! table alive; `release` may run on any thread, so it must not block or take the GIL.

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

/// `pub` only for `side_table_ref!`.
pub struct Entry {
    id: u64,
    registry: Weak<dyn RefRegistry>,
    /// So `Drop` does not release a moved payload.
    taken: AtomicBool,
}

impl Entry {
    pub fn new(id: u64, registry: &Arc<dyn RefRegistry>) -> Self {
        Entry { id, registry: Arc::downgrade(registry), taken: AtomicBool::new(false) }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    /// `Some` once, then `None` from any clone.
    pub fn take(&self) -> Option<u64> {
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

/// `Clone` shares the entry, `Drop` of the last clone releases. `take` is added only for exceptions.
macro_rules! side_table_ref {
    ($name:ident, $what:literal) => {
        #[doc = concat!("A key into the binding's side table for ", $what, ".")]
        #[derive(Clone)]
        pub struct $name(::std::sync::Arc<$crate::errors::side_table::Entry>);

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

        /// By id: the payload is not inspectable here.
        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.0.id() == other.0.id()
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
    use std::sync::{Arc, Mutex};

    use super::RefRegistry;

    /// Holds ids the way the binding does, and answers `outstanding` from what is left.
    pub struct Table {
        live: Mutex<BTreeSet<u64>>,
    }

    impl Table {
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
