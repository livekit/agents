//! The exception seam, and the one place where Node is **not** shaped like Python.
//!
//! §5b's requirement is language-neutral: "a plugin's `APIStatusError(429)` must reach
//! `FallbackChain` with `.retryable` intact and be re-raised **as the same object**". The
//! mechanism §1.11 prescribes is a binding-owned side table keyed by a `CapturedError`, and
//! `lk-agents-py` builds exactly that -- a `Py<PyAny>` map, `isinstance` classification against
//! four cached `PyType`s, and a `capture`/`raise_original` protocol over an RAII key.
//!
//! # Node needs the same protocol and the opposite ownership
//!
//! Two facts decide it, and they point the same way:
//!
//! * **A JS value is thread-affine; a `Py<PyAny>` is not.** With the GIL you may hold a Python
//!   object anywhere and touch it wherever you next take the lock. A `napi_ref` may only be
//!   created, read and deleted on its owning JS thread -- napi routes even the *release* back
//!   through a custom-GC threadsafe function to guarantee it. A Rust-side map of JS values would
//!   therefore be a map Rust can never read.
//! * **`FallbackChain` decides `retryable` on a tokio worker.** Classification is an `instanceof`
//!   chain, which is JS-thread work. Python can do it lazily inside `capture` because `capture`
//!   already holds the GIL; Node cannot do it lazily at all.
//!
//! So the table lives **in JavaScript** -- `node/@livekit-agents/src/errors.ts`'s `Map<number,
//! unknown>` -- and the shim classifies at the catch site, where it is already on the JS thread
//! and already has the exception in hand. Rust receives a [`JsProviderError`]: a `hostRef` into
//! that map plus the same scalar copy of `retryable`/`statusCode`/kind the Python path carries.
//! `ProviderError::Host` is unchanged and `FallbackChain` cannot tell the two hosts apart, which
//! is the property that matters.
//!
//! **This is a finding about `lk-agents-py`, not about the core.** `CapturedError` is exactly the
//! right abstraction for both; only the side it is indexed from differs. Nothing in
//! `lk-agents-abi::errors` had to change.
//!
//! # What napi gives for free, and why it is still not enough
//!
//! `napi::Error` already retains the original thrown value behind a refcounted `ErrorRef`, releases
//! it on the owning thread, and re-throws the *same object* -- identity preservation, for nothing.
//! It is not sufficient here for one reason: it exposes `reason` and `cause` as data and nothing
//! else, so `retryable` and `status_code` are unreachable from the worker where the chain needs
//! them. Worth knowing anyway: for any path that only has to propagate a failure (not decide on
//! it), letting `napi::Error` carry it is strictly better than a table entry.
//!
//! # Release
//!
//! The key is RAII (`lk_agents_abi::errors::side_table`), the same shape as
//! [`crate::host_table`]: the last clone's drop moves the id from `LIVE` to `RELEASED`, and the
//! shim pulls `RELEASED` through [`drain_released`] to `map.delete(...)`. [`release`] is the
//! explicit early form for a caller that has just re-raised or discarded the failure and wants the
//! JS entry offered back now rather than when the last clone happens to drop; it takes the key,
//! so the drop that follows releases nothing.

use std::sync::{Arc, Mutex, OnceLock};

use lk_agents_abi::errors::{ApiErrorKind, CapturedError, ProviderError, RefRegistry};
use napi_derive::napi;

/// A host failure, classified on the JS thread where the exception actually is.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct JsProviderError {
    /// The key in the shim's error map. `0` means "not stashed" -- a failure the shim could not or
    /// chose not to retain, which re-raises as a fresh `Error` built from `message`.
    pub host_ref: i64,
    /// The constructor name, e.g. `"APIStatusError"`, or a plugin's own subclass of it. Carried
    /// separately from `apiKind` for the same reason as in Python: the subclass name is what a
    /// user sees, the taxonomy is what the chain branches on.
    pub kind: String,
    /// `"api" | "status" | "connection" | "timeout" | "cancelled"`, or absent for an exception
    /// outside the taxonomy -- which, as in the reference, means "never retried".
    pub api_kind: Option<String>,
    pub message: String,
    pub retryable: bool,
    pub status_code: Option<i64>,
}

fn api_kind(name: Option<&str>) -> Option<ApiErrorKind> {
    match name? {
        "api" => Some(ApiErrorKind::Api),
        "status" => Some(ApiErrorKind::Status),
        "connection" => Some(ApiErrorKind::Connection),
        "timeout" => Some(ApiErrorKind::Timeout),
        "cancelled" => Some(ApiErrorKind::Cancelled),
        // Unknown discriminants are "outside the taxonomy", not a panic: the shim is
        // hand-written and a future kind must degrade to not-retried rather than crash a session.
        _ => None,
    }
}

impl From<JsProviderError> for ProviderError {
    fn from(e: JsProviderError) -> Self {
        ProviderError::Host {
            error: lk_agents_abi::errors::HostError::captured(
                e.kind,
                e.message,
                intern(e.host_ref),
            ),
            api_kind: api_kind(e.api_kind.as_deref()),
            retryable: e.retryable,
            status_code: e.status_code,
        }
    }
}

/// The scalars a caller needs to re-raise, handed back out. The shim looks `hostRef` up in its own
/// map and throws the original; if it is gone it builds an `Error` from `message`.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct JsErrorOut {
    pub host_ref: i64,
    pub kind: String,
    pub message: String,
    pub retryable: bool,
    pub status_code: Option<i64>,
}

impl From<&ProviderError> for JsErrorOut {
    fn from(e: &ProviderError) -> Self {
        JsErrorOut {
            host_ref: e.host_ref().map(|r| r.id() as i64).unwrap_or(0),
            kind: match e {
                ProviderError::Host { error, .. } => error.kind.clone(),
                ProviderError::Local(err) => format!("{:?}", err.kind),
            },
            message: e.message().to_owned(),
            retryable: e.retryable(),
            status_code: e.status_code(),
        }
    }
}

// ------------------------------------------------------------------ the release protocol

/// IdMint whose `ProviderError` is still alive somewhere and whose JS entry must stay.
static LIVE: Mutex<Vec<i64>> = Mutex::new(Vec::new());
/// IdMint whose `ProviderError` has been consumed and whose JS entry may go.
///
/// The same shape as `lk-agents-py`'s `host_table` release queue and for the same reason: the drop
/// happens on whatever tokio worker last held the error, which is never the JS thread. Pushing an
/// id costs a `Vec::push`; the deletion happens in JS on the next drain.
static RELEASED: Mutex<Vec<i64>> = Mutex::new(Vec::new());

fn lock(m: &Mutex<Vec<i64>>) -> std::sync::MutexGuard<'_, Vec<i64>> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// Move `id` from `LIVE` to `RELEASED`. `0` is "not stashed": nothing to offer back.
fn forget(id: i64) {
    if id == 0 {
        return;
    }
    let mut live = lock(&LIVE);
    if let Some(pos) = live.iter().position(|x| *x == id) {
        live.swap_remove(pos);
    }
    lock(&RELEASED).push(id);
}

/// Zero-sized forwarder onto the two queues -- `host_table.rs`'s `Handle`, for exceptions.
struct Handle;

impl RefRegistry for Handle {
    fn release(&self, id: u64) {
        // From `Drop`, on whatever thread dropped the last clone: no panic, no blocking, no JS.
        forget(id as i64);
    }

    fn outstanding(&self) -> usize {
        lock(&LIVE).len()
    }
}

fn registry() -> &'static Arc<dyn RefRegistry> {
    static REGISTRY: OnceLock<Arc<dyn RefRegistry>> = OnceLock::new();
    // Leaked for the life of the process, as in `host_table.rs`: the key holds a `Weak`, and a
    // registry that could die while errors are alive would silently stop releasing.
    REGISTRY.get_or_init(|| Arc::new(Handle))
}

/// Take ownership of a failure the shim has already put in its map under `id`. `0` is tracked by
/// nothing and released to nothing.
fn intern(id: i64) -> CapturedError {
    if id != 0 {
        lock(&LIVE).push(id);
    }
    CapturedError::new(id as u64, registry())
}

/// Give the JS entry back now. Called once the failure has been re-raised or discarded; the key is
/// taken, so the `ProviderError`'s own drop then releases nothing. Idempotent, and a no-op for a
/// native error or a `hostRef` of 0.
pub fn release(err: &ProviderError) {
    if let Some(id) = err.host_ref().and_then(|r| r.take()) {
        forget(id as i64);
    }
}

/// IdMint the shim may now delete from its map. Drained by the host, never pushed to it -- same rule
/// as the telemetry drain.
pub fn drain_released() -> Vec<i64> {
    std::mem::take(&mut *lock(&RELEASED))
}

/// How many captured exceptions the shim is still holding for us. The leak detector: an entry that
/// is neither re-raised nor released is one retained JS `Error` per failure, for the session.
pub fn live_count() -> usize {
    registry().outstanding()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `LIVE`/`RELEASED` are process-wide by design -- they are the release protocol for a table
    /// that lives in the host, and there is one host per process. Cargo runs tests in threads, so
    /// any test that reads a *count* has to be serialised against any test that changes one.
    /// Without this, `live_count()` is a race and the failure is intermittent.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn serial() -> std::sync::MutexGuard<'static, ()> {
        SERIAL.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn js_err(host_ref: i64, api_kind: Option<&str>, retryable: bool) -> JsProviderError {
        JsProviderError {
            host_ref,
            kind: "MyPluginRateLimit".to_owned(),
            api_kind: api_kind.map(str::to_owned),
            message: "429 slow down".to_owned(),
            retryable,
            status_code: Some(429),
        }
    }

    #[test]
    fn a_subclass_keeps_its_own_name_and_still_classifies_as_the_base() {
        let _serial = serial();
        let e: ProviderError = js_err(7, Some("status"), true).into();
        assert_eq!(
            e.api_kind(),
            Some(ApiErrorKind::Status),
            "the taxonomy is what the chain reads"
        );
        match &e {
            ProviderError::Host { error, .. } => {
                assert_eq!(&error.kind, "MyPluginRateLimit", "the subclass name must survive")
            }
            ProviderError::Local(other) => panic!("expected Host, got {other:?}"),
        }
        assert!(e.retryable());
        assert_eq!(e.status_code(), Some(429));
        release(&e);
    }

    #[test]
    fn an_unknown_api_kind_is_outside_the_taxonomy_not_a_panic() {
        let _serial = serial();
        let e: ProviderError = js_err(0, Some("teleported"), false).into();
        assert_eq!(e.api_kind(), None, "unknown means never retried, as in the reference");
        assert!(!e.retryable());
    }

    #[test]
    fn a_captured_error_is_tracked_until_released_and_then_offered_to_the_host() {
        let _serial = serial();
        let before = live_count();
        let e: ProviderError = js_err(4242, Some("status"), true).into();
        assert_eq!(live_count(), before + 1, "the table is a leak detector");
        release(&e);
        assert_eq!(live_count(), before);
        assert!(drain_released().contains(&4242));
        drop(e);
        assert!(
            !drain_released().contains(&4242),
            "the drop after an explicit release must not offer the id a second time"
        );
    }

    // Catches: the leak the RAII key removes (design 10, Smells 12-19). A `Host` error that is
    // merely dropped -- `let _ = tx.send(Err(e))`, an overwritten `last_error` -- must offer its
    // JS entry back by itself, and only once the *last* clone is gone.
    #[test]
    fn dropping_the_last_clone_offers_the_entry_to_the_host() {
        let _serial = serial();
        let _ = drain_released();
        let before = live_count();
        let e: ProviderError = js_err(5150, Some("status"), true).into();
        let twin = e.clone();
        drop(e);
        assert_eq!(live_count(), before + 1, "released while a clone was still live");
        assert!(!drain_released().contains(&5150));
        drop(twin);
        assert_eq!(live_count(), before, "the last drop is the release");
        assert_eq!(drain_released(), vec![5150]);
    }

    #[test]
    fn a_failure_with_no_stashed_object_is_not_tracked() {
        let _serial = serial();
        let before = live_count();
        let e: ProviderError = js_err(0, None, false).into();
        assert_eq!(live_count(), before, "hostRef 0 has nothing to release");
        release(&e);
        drop(e);
        assert!(!drain_released().contains(&0));
    }

    #[test]
    fn the_scalars_survive_the_round_trip_back_out() {
        let _serial = serial();
        let e: ProviderError = js_err(9, Some("status"), true).into();
        let out = JsErrorOut::from(&e);
        assert_eq!(out.host_ref, 9);
        assert_eq!(out.kind, "MyPluginRateLimit");
        assert_eq!(out.status_code, Some(429));
        assert!(out.retryable);
        release(&e);
    }
}
