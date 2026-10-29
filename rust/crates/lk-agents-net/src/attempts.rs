//! The attempts of one gateway request, as a synchronous machine on supplied time: the attempt
//! currently running and its phase deadline, the retry decision and its backoff, the total
//! deadline and the cleanup every launched attempt still owes. Its owner (a request lifecycle)
//! turns the [`AttemptStep`]s into its own resource's effects and reports the transport's
//! observations back by attempt.
//!
//! Timeouts follow `APIConnectOptions`: connect runs from launch until the transport has a
//! connection, write until the request body is handed over, read while a response is expected
//! (the owner says since when), each restarted by transport progress; total runs from acceptance
//! and never restarts. A failure is retried only while its error allows it, attempts remain and
//! the owner has not marked the request unsafe to replay.
//!
//! A request the server may have received and that no answer resolved (timed out, stopped or cut
//! after it was sent) may still take effect. When no caller takes its outcome, its owner retires
//! it: the request keeps asking under its identity until an attempt answers or the retirement
//! deadline passes, which is reported.

use std::collections::BTreeMap;
use std::time::Duration;

use lk_agents_abi::errors::{ApiError, ApiErrorKind, TimeoutPhase};
use lk_agents_abi::types::retry::{RequestKind, RetryDecision};
use lk_agents_abi::{ApiConnectOptions, ApiConnectOptionsError, HOST_TEARDOWN_TIMEOUT, Ts};

/// The host's `APIConnectOptions` rules: its boundary check and its retry interval.
pub struct ConnectOptionsRules;

impl lk_agents_abi::types::connect::ConnectOptionsFunctions for ConnectOptionsRules {
    fn api_connect_options_check(
        max_retry: i64,
        retry_interval: f64,
        pool_timeout: Option<f64>,
        connect_timeout: Option<f64>,
        write_timeout: Option<f64>,
        read_timeout: Option<f64>,
        total_timeout: Option<f64>,
    ) -> Result<(), ApiConnectOptionsError> {
        let max_retry = u32::try_from(max_retry).map_err(|_| ApiConnectOptionsError {
            field: "max_retry",
            requirement: "must be an integer >= 0",
        })?;
        ApiConnectOptions {
            max_retry,
            retry_interval,
            pool_timeout,
            connect_timeout,
            write_timeout,
            read_timeout,
            total_timeout,
        }
        .validate()
    }

    fn api_retry_interval(retry_interval: f64, num_retries: u32) -> f64 {
        ApiConnectOptions { retry_interval, ..ApiConnectOptions::default() }
            .interval_for_retry(num_retries)
    }
}

/// A retiring request's phases are bounded so it asks at least this many times inside its window:
/// a silent attempt never holds the whole window, even when the caller set no limits.
const RETIREMENT_ATTEMPTS: u32 = 3;

/// The resolved per-request policy. `None` disables a limit. There is no pool limit: the
/// gateway transport has no pool capacity, so a request never waits for a connection slot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RequestPolicy {
    pub connect_timeout: Option<Duration>,
    pub write_timeout: Option<Duration>,
    pub read_timeout: Option<Duration>,
    pub total_timeout: Option<Duration>,
    /// Attempts after the first.
    pub max_retry: u32,
    pub retry_interval: Duration,
}

impl RequestPolicy {
    /// The shared retry allowance and delay; the owner supplies whether replay is eligible.
    pub fn retry_delay(&self, retries: u32, retryable: bool) -> Option<Duration> {
        if !retryable || retries >= self.max_retry {
            return None;
        }
        Some(if retries == 0 {
            Duration::from_secs_f64(lk_agents_abi::types::connect::FIRST_RETRY_INTERVAL)
        } else {
            self.retry_interval
        })
    }

    /// The phase limits of validated options; `pool_timeout` bounds no wait of this transport and
    /// the legacy `timeout` is not read.
    pub fn from_options(options: &ApiConnectOptions) -> Self {
        let limit = |seconds: Option<f64>| seconds.map(Duration::from_secs_f64);
        RequestPolicy {
            connect_timeout: limit(options.connect_timeout),
            write_timeout: limit(options.write_timeout),
            read_timeout: limit(options.read_timeout),
            total_timeout: limit(options.total_timeout),
            max_retry: options.max_retry,
            retry_interval: Duration::from_secs_f64(options.retry_interval),
        }
    }
}

/// The attempt owner gave up. Timeouts keep their class/phase and caller-owned cause; other
/// exhausted recognition/completion allowances carry one shared diagnostic.
pub fn terminal_failure(
    kind: RequestKind,
    error: &ApiError,
    max_retry: u32,
    retries: u32,
) -> RetryDecision {
    if max_retry == 0 || !error.retryable || error.kind == ApiErrorKind::Timeout {
        return RetryDecision::Raise;
    }
    let action = match kind {
        RequestKind::Llm => "generate LLM completion",
        RequestKind::SttBatch | RequestKind::SttStream => "recognize speech",
        RequestKind::Interruption => "detect interruption",
        RequestKind::TtsBatch | RequestKind::TtsStream | RequestKind::Avatar => {
            return RetryDecision::Raise;
        }
    };
    RetryDecision::Exhausted(format!(
        "failed to {action} after {} attempts",
        u64::from(retries) + 1
    ))
}

/// Why an attempt failed, as the transport saw it.
#[derive(Debug, Clone, PartialEq)]
pub enum AttemptFailure {
    /// A non-2xx response, its message read from the body.
    Rejected { status: i64, message: String, request_id: Option<String> },
    /// Launch, connection or mid-body transport failure.
    Connection(String),
    /// The response is not the declared wire shape.
    Malformed(String),
}

impl AttemptFailure {
    pub fn into_error(self) -> ApiError {
        match self {
            AttemptFailure::Rejected { status, message, request_id } => {
                ApiError::from_http(&message, status, request_id)
            }
            AttemptFailure::Connection(message) => ApiError::connection(Some(message)),
            AttemptFailure::Malformed(message) => {
                ApiError::connection(Some(format!("malformed response: {message}")))
            }
        }
    }
}

/// What the owner performs or reports, in order.
#[derive(Debug, Clone, PartialEq)]
pub enum AttemptStep {
    /// Start this attempt's exchange.
    Launch(u32),
    /// Stop this attempt's exchange; its release is still reported.
    Stop(u32),
    /// The attempt failed and the request will retry after its backoff.
    Retry { attempt: u32, error: ApiError },
    /// The request failed: no attempt follows.
    GiveUp(ApiError),
    /// A retiring request's outcome was still unknown at its deadline; no attempt follows.
    Unresolved,
    /// The attempt missed its cleanup deadline; the runtime still owns it.
    CleanupIncomplete(u32),
}

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
pub struct Attempts {
    policy: RequestPolicy,
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

impl Attempts {
    /// Accepts the request at `now` and launches its first attempt.
    pub fn start(policy: RequestPolicy, now: Ts, steps: &mut Vec<AttemptStep>) -> Self {
        let mut attempts = Attempts {
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
            steps.push(AttemptStep::GiveUp(
                ApiError::timed_out(TimeoutPhase::Total).with_retryable(false),
            ));
        } else {
            attempts.launch(0, now, steps);
        }
        attempts
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

    /// The earliest instant [`Attempts::wake`] has work to do; `expecting` is since when the
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
        steps: &mut Vec<AttemptStep>,
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
    /// phase rather than converting it back through a transport failure.
    pub fn failed_error(
        &mut self,
        attempt: u32,
        error: ApiError,
        now: Ts,
        steps: &mut Vec<AttemptStep>,
    ) {
        if self.current() != Some(attempt) {
            return;
        }
        self.close_attempt(attempt, now);
        self.fail(attempt, error, Duration::ZERO, now, steps);
    }

    /// The server named when to retry (`Retry-After`, `google.rpc.RetryInfo`): a retry waits at
    /// least `retry_after`.
    pub fn failed_retry_after(
        &mut self,
        attempt: u32,
        error: ApiError,
        retry_after: Duration,
        now: Ts,
        steps: &mut Vec<AttemptStep>,
    ) {
        if self.current() != Some(attempt) {
            return;
        }
        self.close_attempt(attempt, now);
        self.fail(attempt, error, retry_after, now, steps);
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
    pub fn closed(&mut self, attempt: u32, now: Ts, steps: &mut Vec<AttemptStep>) {
        self.open.remove(&attempt);
        if self.current() == Some(attempt) {
            self.unresolved |= self.sent();
            let failure = AttemptFailure::Connection("the response ended unexpectedly".into());
            self.fail(attempt, failure.into_error(), Duration::ZERO, now, steps);
        }
    }

    /// The consumer gave up: the running attempt is stopped and no outcome follows.
    pub fn cancel(&mut self, now: Ts, steps: &mut Vec<AttemptStep>) {
        self.unresolved |= self.sent();
        if let Some(attempt) = self.current() {
            self.stop(attempt, now, steps);
        }
        self.phase = Phase::Settled;
    }

    /// No caller takes the outcome any more. A request that may have taken effect keeps asking
    /// under its identity, retrying every retryable failure, until an attempt answers or
    /// `deadline` ([`AttemptStep::Unresolved`]); any other request stops now. Every phase,
    /// including the running attempt's, is bounded to a share of the window.
    pub fn retire(&mut self, now: Ts, deadline: Ts, steps: &mut Vec<AttemptStep>) {
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

    pub fn wake(&mut self, now: Ts, expecting: Option<Ts>, steps: &mut Vec<AttemptStep>) {
        let expired: Vec<u32> = self
            .open
            .iter()
            .filter(|(_, deadline)| deadline.is_some_and(|d| d <= now))
            .map(|(a, _)| *a)
            .collect();
        for attempt in expired {
            // Reported once; the runtime keeps owning the attempt.
            self.open.remove(&attempt);
            steps.push(AttemptStep::CleanupIncomplete(attempt));
        }
        if self.is_decided() {
            return;
        }
        if self.total_deadline.is_some_and(|d| d <= now) {
            self.cancel(now, steps);
            steps.push(if self.retiring {
                AttemptStep::Unresolved
            } else {
                AttemptStep::GiveUp(ApiError::timed_out(TimeoutPhase::Total).with_retryable(false))
            });
            return;
        }
        match self.phase {
            Phase::Waiting { attempt, at } if at <= now => self.launch(attempt, now, steps),
            Phase::Attempting { attempt, stage } => {
                if let Some((at, phase)) = self.stage_deadline(stage, expecting)
                    && at <= now
                {
                    self.unresolved |= self.sent();
                    self.stop(attempt, now, steps);
                    self.fail(attempt, ApiError::timed_out(phase), Duration::ZERO, now, steps);
                }
            }
            _ => {}
        }
    }

    fn launch(&mut self, attempt: u32, now: Ts, steps: &mut Vec<AttemptStep>) {
        self.open.insert(attempt, None);
        self.next_attempt = attempt + 1;
        self.phase = Phase::Attempting { attempt, stage: Stage::Connecting { since: now } };
        steps.push(AttemptStep::Launch(attempt));
    }

    /// The attempt's transport task ends on its own after a terminal report or a stop; it still
    /// owes the report that its resources are released.
    fn close_attempt(&mut self, attempt: u32, now: Ts) {
        if let Some(deadline) = self.open.get_mut(&attempt) {
            deadline.get_or_insert(now.saturating_add(HOST_TEARDOWN_TIMEOUT));
        }
    }

    fn stop(&mut self, attempt: u32, now: Ts, steps: &mut Vec<AttemptStep>) {
        if self.open.contains_key(&attempt) {
            self.close_attempt(attempt, now);
            steps.push(AttemptStep::Stop(attempt));
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
    fn fail(
        &mut self,
        attempt: u32,
        error: ApiError,
        at_least: Duration,
        now: Ts,
        steps: &mut Vec<AttemptStep>,
    ) {
        let retryable = error.retryable && !self.replay_unsafe;
        if let Some(delay) = self.policy.retry_delay(self.failures, retryable) {
            let delay = delay.max(at_least);
            self.failures += 1;
            self.phase = Phase::Waiting { attempt: attempt + 1, at: now.saturating_add(delay) };
            steps.push(AttemptStep::Retry { attempt, error });
        } else {
            self.phase = Phase::Settled;
            steps.push(AttemptStep::GiveUp(error.with_retryable(retryable)));
        }
    }

    fn stage_deadline(&self, stage: Stage, expecting: Option<Ts>) -> Option<(Ts, TimeoutPhase)> {
        let p = &self.policy;
        match stage {
            Stage::Connecting { since } => {
                p.connect_timeout.map(|t| (since.saturating_add(t), TimeoutPhase::Connect))
            }
            Stage::Writing { since } => {
                p.write_timeout.map(|t| (since.saturating_add(t), TimeoutPhase::Write))
            }
            Stage::Reading { last_progress } => {
                let waiting_since = expecting?.max(last_progress);
                p.read_timeout.map(|t| (waiting_since.saturating_add(t), TimeoutPhase::Read))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use lk_agents_abi::types::connect::ConnectOptionsFunctions as _;

    use super::ConnectOptionsRules;

    // Catches: a negative retry count wrapping into a huge budget instead of naming its field.
    #[test]
    fn a_negative_max_retry_is_refused_by_name() {
        let err =
            ConnectOptionsRules::api_connect_options_check(-1, 2.0, None, None, None, None, None)
                .unwrap_err();
        assert_eq!(err.to_string(), "conn_options.max_retry must be an integer >= 0");
    }
}
