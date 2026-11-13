use std::time::Duration;

use lk_agents_abi::Ts;
use lk_agents_abi::errors::ApiError;

/// What the owner performs or reports, in order.
#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    /// Start this attempt's exchange.
    Launch(u32),
    /// Stop this attempt's exchange; its release is still reported.
    Stop(u32),
    /// The attempt failed and the request will retry after its backoff.
    Retry(Backoff),
    /// The request failed: no attempt follows.
    GiveUp(ApiError),
    /// A retiring request's outcome was still unknown at its deadline; no attempt follows.
    Unresolved,
    /// The attempt missed its cleanup deadline; the runtime still owns it.
    CleanupIncomplete(u32),
}

/// The retry decision for a failed, fully released provider attempt.
#[derive(Debug, Clone, PartialEq)]
pub enum Retry {
    At(Backoff),
    GiveUp(ApiError),
}

/// A retry the policy decided. Its owner learns when it runs only by announcing it, which logs it
/// once, or gives it up with [`Backoff::abandon`]: no retry is applied silently (`rust/check`
/// rejects an arm that neither announces, abandons nor hands it on).
#[derive(Debug, Clone, PartialEq)]
#[must_use = "a retry is applied through `announce`, which logs it"]
pub struct Backoff {
    /// The failed attempt's id.
    pub(super) attempt: u32,
    pub(super) at: Ts,
    pub(super) delay: Duration,
    pub(super) error: ApiError,
}

impl Backoff {
    /// The failure being retried.
    pub fn error(&self) -> &ApiError {
        &self.error
    }

    /// The owner does not apply this retry (its replay would lose or repeat output): the failure
    /// it was decided for, which the owner raises instead.
    pub fn abandon(self) -> ApiError {
        self.error.log_detail();
        self.error
    }

    /// Logs the retry of `subject` after `error` (the failure as its owner raised it) and returns
    /// when the next attempt runs.
    pub fn announce(self, subject: &str, error: &dyn std::fmt::Display) -> Ts {
        self.error.log_detail();
        let attempt = self.attempt.saturating_add(1);
        lk_agents_abi::types::retry::retrying(
            log::Level::Warn,
            subject,
            attempt,
            self.delay,
            error,
        );
        self.at
    }
}
