use std::collections::BTreeMap;

use super::{AttemptFailure, Backoff, Policy, Retry, Step};
use lk_agents_abi::errors::{ApiError, ApiErrorKind, Timeout};
use lk_agents_abi::{ApiConnectOptions, HOST_TEARDOWN_TIMEOUT, Ts};

// Phase timeouts (connect, write, read) restart on transport progress; total runs from acceptance
// and never restarts. A failure is retried only while its error allows it, attempts remain and
// the owner has not marked the request unsafe to replay. A request the server may have received
// but no answer resolved may still take effect, so when no caller takes its outcome it is retired:
// it keeps asking under its identity until an attempt answers or the retirement deadline passes.

/// A retiring request's phases are bounded so it asks at least this many times inside its window:
/// a silent attempt never holds the whole window, even when the caller set no limits.
const RETIREMENT_ATTEMPTS: u32 = 3;

#[derive(Debug, Clone, Copy)]
enum Phase {
    /// Backing off before `attempt`.
    Waiting {
        attempt: u32,
        at: Ts,
    },
    Attempting {
        attempt: u32,
        stage: Stage,
    },
    /// The outcome is decided; launched attempts may still owe their release.
    Settled,
}

#[derive(Debug, Clone, Copy)]
enum Stage {
    Connecting { since: Ts },
    Writing { since: Ts },
    Reading { last_progress: Ts },
}

#[derive(Debug)]
pub struct Request {
    policy: Policy,
    total_deadline: Option<Ts>,
    phase: Phase,
    /// The outcome would repeat something visible: no further attempt.
    replay_unsafe: bool,
    /// A sent attempt ended without an answer: the request may have taken effect.
    unresolved: bool,
    /// No caller takes the outcome: retrying until an answer or the total deadline.
    retiring: bool,
    next_attempt: u32,
    failures: u32,
    /// Launched attempts not yet closed, with their cleanup deadline once asked to stop or done.
    open: BTreeMap<u32, Option<Ts>>,
}

impl Request {
    /// Accepts the request at `now` and launches its first attempt.
    pub fn start(policy: Policy, now: Ts, steps: &mut Vec<Step>) -> Self {
        let mut attempts = Request {
            policy,
            total_deadline: policy.total_timeout.map(|t| now.saturating_add(t)),
            phase: Phase::Settled,
            replay_unsafe: false,
            unresolved: false,
            retiring: false,
            next_attempt: 0,
            failures: 0,
            open: BTreeMap::new(),
        };
        if attempts.total_deadline.is_some_and(|deadline| deadline <= now) {
            steps.push(Step::GiveUp(ApiError::timed_out(Timeout::Total).with_retryable(false)));
        } else {
            attempts.launch(0, now, steps);
        }
        attempts
    }

    /// Whole requests retried on the options' allowance and backoff alone, by an owner whose
    /// requests enforce their own connection and total limits (a voice reply's model and synthesis).
    /// The first attempt is running.
    pub fn retrying(options: &ApiConnectOptions, now: Ts) -> Self {
        let policy = Policy {
            connect_timeout: None,
            write_timeout: None,
            read_timeout: None,
            total_timeout: None,
            ..Policy::from_options(options)
        };
        Request::start(policy, now, &mut Vec::new())
    }

    /// The provider finished this attempt and released it. Cancellation never retries, even
    /// when a provider incorrectly marked its error retryable.
    pub fn attempt_failed(&mut self, error: ApiError, now: Ts) -> Retry {
        let Some(attempt) = self.current() else { return Retry::GiveUp(error) };
        let mut steps = Vec::new();
        self.failed_error(attempt, error, now, &mut steps);
        self.closed(attempt, now, &mut steps);
        for step in steps {
            match step {
                // the total deadline may come first: its wake ends the request
                Step::Retry(backoff) => {
                    let at = self.wake_at(None).expect("retry wake");
                    return Retry::At(Backoff { at, ..backoff });
                }
                Step::GiveUp(error) => return Retry::GiveUp(error),
                _ => {}
            }
        }
        unreachable!("a current attempt's failure decides whether to retry")
    }

    /// The attempt whose outcome is the request's, while one runs.
    pub fn current(&self) -> Option<u32> {
        match self.phase {
            Phase::Attempting { attempt, .. } => Some(attempt),
            _ => None,
        }
    }

    /// The request's outcome is decided (succeeded, given up or cancelled).
    pub fn is_decided(&self) -> bool {
        matches!(self.phase, Phase::Settled)
    }

    /// Decided, and no attempt is left to clean up.
    pub fn is_settled(&self) -> bool {
        self.is_decided() && self.open.is_empty()
    }

    /// The request may have taken effect on the server: an attempt is sent and unanswered, or
    /// one ended so. Retiring it learns the outcome.
    pub fn may_have_taken_effect(&self) -> bool {
        self.unresolved || self.sent()
    }

    /// Visible output reached the consumer: a replay would repeat it.
    pub fn delivered(&mut self) {
        self.replay_unsafe = true;
    }

    /// The earliest instant [`Request::wake`] has work to do; `expecting` is since when the
    /// owner expects a response (`None`: not now, so no read timer runs).
    pub fn wake_at(&self, expecting: Option<Ts>) -> Option<Ts> {
        let phase = match self.phase {
            Phase::Waiting { at, .. } => Some(at),
            Phase::Attempting { stage, .. } => {
                self.stage_deadline(stage, expecting).map(|(at, _)| at)
            }
            Phase::Settled => None,
        };
        let total = self.total_deadline.filter(|_| !self.is_decided());
        [phase, total, self.open.values().flatten().min().copied()].into_iter().flatten().min()
    }

    pub fn connected(&mut self, attempt: u32, now: Ts) {
        self.advance(attempt, |stage| {
            matches!(stage, Stage::Connecting { .. }).then_some(Stage::Writing { since: now })
        });
    }

    pub fn written(&mut self, attempt: u32, now: Ts) {
        self.advance(attempt, |stage| match stage {
            Stage::Connecting { .. } | Stage::Writing { .. } => {
                Some(Stage::Reading { last_progress: now })
            }
            Stage::Reading { .. } => None,
        });
    }

    /// Response bytes prove the request was sent, whatever the transport reported before.
    pub fn progress(&mut self, attempt: u32, now: Ts) {
        self.advance(attempt, |_| Some(Stage::Reading { last_progress: now }));
    }

    /// `attempt` produced the request's answer; `false` when it is not the current attempt.
    pub fn succeeded(&mut self, attempt: u32, now: Ts) -> bool {
        if self.current() != Some(attempt) {
            return false;
        }
        self.unresolved = false;
        self.close_attempt(attempt, now);
        self.phase = Phase::Settled;
        true
    }

    /// `attempt` failed; only the current attempt's failure decides anything.
    pub fn failed(
        &mut self,
        attempt: u32,
        failure: AttemptFailure,
        now: Ts,
        steps: &mut Vec<Step>,
    ) {
        if self.current() == Some(attempt) {
            // A final rejection answers the operation (a replay gets it too); a retryable one
            // leaves an earlier attempt's effect unknown; a cut or unreadable answer is none.
            let rejected = matches!(failure, AttemptFailure::Rejected { .. });
            let error = failure.into_error();
            if !rejected {
                self.unresolved |= self.sent();
            } else if !error.retryable {
                self.unresolved = false;
            }
            self.failed_error(attempt, error, now, steps);
        }
    }

    /// A provider already adapted its failure at the boundary. Preserve its retryability and
    /// phase rather than converting it back through a transport failure. A retry waits at least
    /// the `retry_after` the server named (`Retry-After`, `google.rpc.RetryInfo`).
    pub fn failed_error(&mut self, attempt: u32, error: ApiError, now: Ts, steps: &mut Vec<Step>) {
        if self.current() != Some(attempt) {
            return;
        }
        self.close_attempt(attempt, now);
        self.fail(attempt, error, now, steps);
    }

    /// A long-lived recognition completed one utterance. Its next failure gets a fresh retry
    /// allowance; the operation's acceptance deadline and attempt identities stay unchanged.
    pub fn reset_retries(&mut self) {
        self.failures = 0;
    }

    /// An enclosing request owns reopen decisions. Phase and cleanup observations stay local.
    pub fn disable_retries(&mut self) {
        self.policy.max_retry = 0;
    }

    /// Consecutive retries consumed by the current recognition allowance.
    pub fn retries(&self) -> u32 {
        self.failures
    }

    /// A parent operation may only shorten the deadline inherited by this request.
    pub fn constrain_deadline(&mut self, deadline: Option<Ts>) {
        self.total_deadline = match (self.total_deadline, deadline) {
            (Some(own), Some(parent)) => Some(own.min(parent)),
            (own, parent) => own.or(parent),
        };
    }

    /// The attempt's transport resources are released. Closed before reporting an outcome: the
    /// transport lost it.
    pub fn closed(&mut self, attempt: u32, now: Ts, steps: &mut Vec<Step>) {
        self.open.remove(&attempt);
        if self.current() == Some(attempt) {
            self.unresolved |= self.sent();
            let failure = AttemptFailure::Connection("the response ended unexpectedly".into());
            self.fail(attempt, failure.into_error(), now, steps);
        }
    }

    /// The consumer gave up: the running attempt is stopped and no outcome follows.
    pub fn cancel(&mut self, now: Ts, steps: &mut Vec<Step>) {
        self.unresolved |= self.sent();
        if let Some(attempt) = self.current() {
            self.stop(attempt, now, steps);
        }
        self.phase = Phase::Settled;
    }

    /// No caller takes the outcome any more. A request that may have taken effect keeps asking
    /// under its identity, retrying every retryable failure, until an attempt answers or
    /// `deadline` ([`Step::Unresolved`]); any other request stops now. Every phase,
    /// including the running attempt's, is bounded to a share of the window.
    pub fn retire(&mut self, now: Ts, deadline: Ts, steps: &mut Vec<Step>) {
        if self.retiring {
            return;
        }
        if !self.may_have_taken_effect() {
            self.cancel(now, steps);
            return;
        }
        self.retiring = true;
        self.replay_unsafe = false;
        self.policy.max_retry = u32::MAX;
        self.total_deadline = Some(deadline);
        let attempt_limit = deadline.saturating_duration_since(now) / RETIREMENT_ATTEMPTS;
        let p = &mut self.policy;
        for limit in [&mut p.connect_timeout, &mut p.write_timeout, &mut p.read_timeout] {
            *limit = Some(limit.map_or(attempt_limit, |t| t.min(attempt_limit)));
        }
        if self.is_decided() {
            self.phase = Phase::Waiting {
                attempt: self.next_attempt,
                at: now.saturating_add(self.policy.retry_interval),
            };
        }
    }

    pub fn wake(&mut self, now: Ts, expecting: Option<Ts>, steps: &mut Vec<Step>) {
        let expired: Vec<u32> = self
            .open
            .iter()
            .filter(|(_, deadline)| deadline.is_some_and(|d| d <= now))
            .map(|(a, _)| *a)
            .collect();
        for attempt in expired {
            // Reported once; the runtime keeps owning the attempt.
            self.open.remove(&attempt);
            steps.push(Step::CleanupIncomplete(attempt));
        }
        if self.is_decided() {
            return;
        }
        if self.total_deadline.is_some_and(|d| d <= now) {
            self.cancel(now, steps);
            steps.push(if self.retiring {
                Step::Unresolved
            } else {
                Step::GiveUp(ApiError::timed_out(Timeout::Total).with_retryable(false))
            });
            return;
        }
        match self.phase {
            Phase::Waiting { attempt, at } if at <= now => self.launch(attempt, now, steps),
            Phase::Attempting { attempt, stage } => {
                if let Some((at, limit)) = self.stage_deadline(stage, expecting)
                    && at <= now
                {
                    self.unresolved |= self.sent();
                    self.stop(attempt, now, steps);
                    self.fail(attempt, ApiError::timed_out(limit), now, steps);
                }
            }
            _ => {}
        }
    }

    fn launch(&mut self, attempt: u32, now: Ts, steps: &mut Vec<Step>) {
        self.open.insert(attempt, None);
        self.next_attempt = attempt + 1;
        self.phase = Phase::Attempting { attempt, stage: Stage::Connecting { since: now } };
        steps.push(Step::Launch(attempt));
    }

    /// The attempt's transport task ends on its own after a terminal report or a stop; it still
    /// owes the report that its resources are released.
    fn close_attempt(&mut self, attempt: u32, now: Ts) {
        if let Some(deadline) = self.open.get_mut(&attempt) {
            deadline.get_or_insert(now.saturating_add(HOST_TEARDOWN_TIMEOUT));
        }
    }

    fn stop(&mut self, attempt: u32, now: Ts, steps: &mut Vec<Step>) {
        if self.open.contains_key(&attempt) {
            self.close_attempt(attempt, now);
            steps.push(Step::Stop(attempt));
        }
    }

    /// The current attempt's request may have reached the server.
    fn sent(&self) -> bool {
        matches!(
            self.phase,
            Phase::Attempting { stage: Stage::Writing { .. } | Stage::Reading { .. }, .. }
        )
    }

    fn advance(&mut self, attempt: u32, next: impl FnOnce(Stage) -> Option<Stage>) {
        if let Phase::Attempting { attempt: current, stage } = &mut self.phase
            && *current == attempt
            && let Some(next) = next(*stage)
        {
            *stage = next;
        }
    }

    /// Retries while the error allows it, a replay is safe and attempts remain; the backoff
    /// never outlives the total deadline, whose own wake ends the request.
    fn fail(&mut self, attempt: u32, error: ApiError, now: Ts, steps: &mut Vec<Step>) {
        let retryable =
            error.retryable && error.kind != ApiErrorKind::Cancelled && !self.replay_unsafe;
        if let Some(delay) = self.policy.retry_delay(self.failures, retryable) {
            let delay = delay.max(error.retry_after().unwrap_or_default());
            self.failures += 1;
            let at = now.saturating_add(delay);
            self.phase = Phase::Waiting { attempt: attempt + 1, at };
            steps.push(Step::Retry(Backoff { attempt, at, delay, error }));
        } else {
            self.phase = Phase::Settled;
            error.log_detail();
            steps.push(Step::GiveUp(error.with_retryable(retryable)));
        }
    }

    fn stage_deadline(&self, stage: Stage, expecting: Option<Ts>) -> Option<(Ts, Timeout)> {
        let p = &self.policy;
        match stage {
            Stage::Connecting { since } => {
                p.connect_timeout.map(|t| (since.saturating_add(t), Timeout::Connect))
            }
            Stage::Writing { since } => {
                p.write_timeout.map(|t| (since.saturating_add(t), Timeout::Write))
            }
            Stage::Reading { last_progress } => {
                let waiting_since = expecting?.max(last_progress);
                p.read_timeout.map(|t| (waiting_since.saturating_add(t), Timeout::Read))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_kind_stops_even_when_marked_retryable() {
        let mut request = Request::retrying(&ApiConnectOptions::default(), Ts::ZERO);
        let error = ApiError::cancelled("cancelled").with_retryable(true);
        assert!(matches!(
            request.attempt_failed(error, Ts::ZERO),
            Retry::GiveUp(ApiError { kind: ApiErrorKind::Cancelled, retryable: false, .. })
        ));
        assert!(request.is_settled());
    }

    #[test]
    fn a_delivered_request_cannot_replay() {
        let mut request = Request::retrying(&ApiConnectOptions::default(), Ts::ZERO);
        request.delivered();
        assert!(matches!(
            request.attempt_failed(ApiError::connection(None), Ts::ZERO),
            Retry::GiveUp(ApiError { retryable: false, .. })
        ));
        assert!(request.is_settled());
    }

    // Catches: a provider's Retry-After ignored (retried at the policy's interval into another
    // 429), or the policy's own longer backoff cut short by it.
    #[test]
    fn a_retry_waits_at_least_the_retry_after_the_provider_named() {
        // the default policy retries the first failure after 0.1 s
        let options = ApiConnectOptions::default();
        for (asked, want) in [(3.0, 3.0), (0.01, 0.1)] {
            let mut request = Request::retrying(&options, Ts::ZERO);
            let error = ApiError {
                said: lk_agents_abi::errors::ProviderSaid::boxed(
                    None,
                    Some(std::time::Duration::from_secs_f64(asked)),
                ),
                ..ApiError::status("busy", 429, None, None)
            };
            let Retry::At(backoff) = request.attempt_failed(error, Ts::ZERO) else {
                panic!("a 429 is retried")
            };
            let at = backoff.announce("test", &"busy");
            assert_eq!(at, Ts::ZERO.saturating_add(std::time::Duration::from_secs_f64(want)));
        }
    }
}
