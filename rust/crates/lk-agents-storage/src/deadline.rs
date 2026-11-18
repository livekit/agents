//! One deadline per storage operation: every request and every resend inside it share the
//! caller's, so no layer waits on a budget of its own. A transport waits for its socket until
//! that deadline, and for a sent request's answer [`REQUEST_WAIT`] from its send; the one retry
//! owner ([`crate::thread::Retry`]) sends again only while a whole [`REQUEST_TIMEOUT`] fits
//! before it.

use std::future::Future;
use std::time::Duration;

use lk_agents_abi::{Clock, Ts};

/// agent-db's request timeout (`agentdb.request_timeout` in production): the server answers a
/// request it read by then, `timed_out` at the latest.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

/// The deadline of an operation whose caller set none: past [`REQUEST_TIMEOUT`], which ends the
/// request on the server first, by a margin for its answer's way back. A sent request waits as
/// long for its answer.
pub const REQUEST_WAIT: Duration = Duration::from_secs(15);

tokio::task_local! {
    static DEADLINE: Ts;
}

/// The deadline of the operation this task runs, if one was set.
pub fn current() -> Option<Ts> {
    DEADLINE.try_with(|deadline| *deadline).ok()
}

/// The deadline in scope, else [`REQUEST_WAIT`] from now on `clock`.
pub fn or_default(clock: &dyn Clock) -> Ts {
    current().unwrap_or_else(|| clock.now().saturating_add(REQUEST_WAIT))
}

/// `fut` under `deadline`, never later than the deadline already in scope: an inner operation
/// cannot outlive its caller's.
pub async fn within<F: Future>(deadline: Ts, fut: F) -> F::Output {
    let deadline = current().map_or(deadline, |outer| outer.min(deadline));
    DEADLINE.scope(deadline, fut).await
}
