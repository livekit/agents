//! The runtime utilities every crate above the purity line shares. Nothing here performs I/O.
pub mod audio;

#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
/// A poisoned lock is taken anyway: the guarded state is kept consistent by its critical sections.
pub fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}
pub mod bound_clock;
pub mod bounded_lossy;
pub mod byte_stream;
pub mod clock;
pub mod connection_pool;
pub mod dates;
pub mod filters;
pub mod frame;
pub mod host_ref;
pub mod maybe;
pub mod monotone;
pub mod resample;
pub mod spawner;
pub mod stream;
pub mod strings;
pub mod vocabulary;
pub mod timer;
pub mod video;

