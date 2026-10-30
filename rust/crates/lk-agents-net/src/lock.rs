//! The I/O crates' blocking locks. A thread holding the host's interpreter lock (Python's GIL)
//! never blocks on one of them while holding it: a contended lock is waited for with the host lock
//! released, as the host installs ([`set_host_wait`]). So a host call made under one of these locks
//! (a log line, a proxy, an emit) cannot deadlock against a host thread that wants the same lock
//! (review13). Neither poisons.
#![allow(clippy::disallowed_types, reason = "the I/O crates' one wrapper over parking_lot's locks")]

use std::ops::{Deref, DerefMut};
use std::sync::OnceLock;

static HOST_WAIT: OnceLock<fn(&mut dyn FnMut())> = OnceLock::new();

/// How a contended wait runs: the host releases its own lock around `wait` when the calling
/// thread holds it. Installed once, by the host binding at import.
pub fn set_host_wait(wait: fn(&mut dyn FnMut())) {
    let _ = HOST_WAIT.set(wait);
}

fn contended<G>(acquire: impl FnOnce() -> G) -> G {
    let Some(host_wait) = HOST_WAIT.get() else { return acquire() };
    let (mut acquire, mut guard) = (Some(acquire), None);
    host_wait(&mut || guard = acquire.take().map(|acquire| acquire()));
    guard.expect("the host wait ran the acquisition")
}

// Debug builds count the guards a thread holds, so a host entry can refuse to run under one
// (`assert_unlocked`).
#[cfg(debug_assertions)]
thread_local! {
    static HELD: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

struct Held;

impl Held {
    fn new() -> Self {
        #[cfg(debug_assertions)]
        HELD.with(|held| held.set(held.get() + 1));
        Held
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        HELD.with(|held| held.set(held.get() - 1));
    }
}

/// A host call (`entry`) must not run while this thread holds one of these locks: in debug and
/// test builds it panics. Log lines are exempt; they never wait for the host.
#[track_caller]
pub fn assert_unlocked(entry: &str) {
    #[cfg(debug_assertions)]
    {
        let held = HELD.with(std::cell::Cell::get);
        assert!(held == 0, "{entry} called while this thread holds {held} lock guard(s)");
    }
    let _ = entry;
}

macro_rules! guard {
    ($name:ident, $inner:ident, $($mut:ident)?) => {
        pub struct $name<'a, T: ?Sized>(parking_lot::$inner<'a, T>, Held);

        impl<T: ?Sized> Deref for $name<'_, T> {
            type Target = T;
            fn deref(&self) -> &T {
                &self.0
            }
        }

        $(
            impl<T: ?Sized> $mut for $name<'_, T> {
                fn deref_mut(&mut self) -> &mut T {
                    &mut self.0
                }
            }
        )?

        impl<T: ?Sized + std::fmt::Debug> std::fmt::Debug for $name<'_, T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}

guard!(MutexGuard, MutexGuard, DerefMut);
guard!(RwLockReadGuard, RwLockReadGuard,);
guard!(RwLockWriteGuard, RwLockWriteGuard, DerefMut);

#[derive(Default)]
pub struct Mutex<T: ?Sized>(parking_lot::Mutex<T>);

impl<T> Mutex<T> {
    pub const fn new(value: T) -> Self {
        Self(parking_lot::Mutex::new(value))
    }

    pub fn into_inner(self) -> T {
        self.0.into_inner()
    }
}

impl<T: ?Sized> Mutex<T> {
    pub fn lock(&self) -> MutexGuard<'_, T> {
        let guard = self.0.try_lock().unwrap_or_else(|| contended(|| self.0.lock()));
        MutexGuard(guard, Held::new())
    }

    pub fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
        self.0.try_lock().map(|guard| MutexGuard(guard, Held::new()))
    }

    pub fn get_mut(&mut self) -> &mut T {
        self.0.get_mut()
    }
}

impl<T: ?Sized + std::fmt::Debug> std::fmt::Debug for Mutex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Default)]
pub struct RwLock<T: ?Sized>(parking_lot::RwLock<T>);

impl<T> RwLock<T> {
    pub const fn new(value: T) -> Self {
        Self(parking_lot::RwLock::new(value))
    }
}

impl<T: ?Sized> RwLock<T> {
    pub fn read(&self) -> RwLockReadGuard<'_, T> {
        let guard = self.0.try_read().unwrap_or_else(|| contended(|| self.0.read()));
        RwLockReadGuard(guard, Held::new())
    }

    pub fn write(&self) -> RwLockWriteGuard<'_, T> {
        let guard = self.0.try_write().unwrap_or_else(|| contended(|| self.0.write()));
        RwLockWriteGuard(guard, Held::new())
    }
}

impl<T: ?Sized + std::fmt::Debug> std::fmt::Debug for RwLock<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static WAITS: AtomicUsize = AtomicUsize::new(0);

    fn counting(wait: &mut dyn FnMut()) {
        WAITS.fetch_add(1, Ordering::SeqCst);
        wait();
    }

    // Catches: a contended lock waited for while the host lock is held; an uncontended one paying
    // for the host's release.
    // Catches: a host entry running under a guard unnoticed in tests.
    #[test]
    #[should_panic(expected = "while this thread holds 1 lock guard")]
    fn a_host_entry_under_a_guard_panics_in_tests() {
        let lock = Mutex::new(());
        let _guard = lock.lock();
        assert_unlocked("a host call");
    }

    #[test]
    fn only_a_contended_lock_waits_through_the_host() {
        set_host_wait(counting);
        let lock = Arc::new(Mutex::new(0));
        *lock.lock() += 1;
        assert_eq!(WAITS.load(Ordering::SeqCst), 0, "uncontended");
        let held = lock.lock();
        let waiter = std::thread::spawn({
            let lock = lock.clone();
            move || *lock.lock() += 1
        });
        while WAITS.load(Ordering::SeqCst) == 0 {
            std::thread::yield_now();
        }
        drop(held);
        waiter.join().unwrap();
        assert_eq!(*lock.lock(), 2);
    }
}
