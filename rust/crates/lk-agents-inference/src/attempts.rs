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

use std::collections::BTreeMap;
use std::time::Duration;

use lk_agents_abi::errors::{ApiError, TimeoutPhase};
use lk_agents_abi::{ApiConnectOptions, Ts};

/// How long an attempt's transport task may take to settle once asked to stop. Dropping an HTTP
/// exchange is immediate, so a miss is a runtime defect, reported rather than waited on.
pub const ATTEMPT_CLEANUP_TIMEOUT: Duration = Duration::from_secs(5);

/// The first retry waits this long, the next ones `retry_interval` (`APIConnectOptions`).
const FIRST_RETRY_INTERVAL: Duration = Duration::from_millis(100);

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
            AttemptFailure::Rejected { status, message, request_id } => ApiError::from_http(&message, status, request_id),
            AttemptFailure::Connection(message) => ApiError::connection(Some(message)),
            AttemptFailure::Malformed(message) => ApiError::connection(Some(format!("malformed response: {message}"))),
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
    /// The attempt missed its cleanup deadline; the runtime still owns it (decision 14).
    CleanupIncomplete(u32),
}

#[derive(Debug, Clone, Copy)]
enum Phase {
    /// Backing off before `attempt`.
    Waiting { attempt: u32, at: Ts },
    Attempting { attempt: u32, stage: Stage },
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
            open: BTreeMap::new(),
        };
        attempts.launch(0, now, steps);
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

    /// Visible output reached the consumer: a replay would repeat it.
    pub fn delivered(&mut self) {
        self.replay_unsafe = true;
    }

    /// The earliest instant [`Attempts::wake`] has work to do; `expecting` is since when the
    /// owner expects a response (`None`: not now, so no read timer runs).
    pub fn wake_at(&self, expecting: Option<Ts>) -> Option<Ts> {
        let phase = match self.phase {
            Phase::Waiting { at, .. } => Some(at),
            Phase::Attempting { stage, .. } => self.stage_deadline(stage, expecting).map(|(at, _)| at),
            Phase::Settled => None,
        };
        let total = self.total_deadline.filter(|_| !self.is_decided());
        [phase, total, self.open.values().flatten().min().copied()].into_iter().flatten().min()
    }

    pub fn connected(&mut self, attempt: u32, now: Ts) {
        self.advance(attempt, |stage| matches!(stage, Stage::Connecting { .. }).then_some(Stage::Writing { since: now }));
    }

    pub fn written(&mut self, attempt: u32, now: Ts) {
        self.advance(attempt, |stage| match stage {
            Stage::Connecting { .. } | Stage::Writing { .. } => Some(Stage::Reading { last_progress: now }),
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
        self.close_attempt(attempt, now);
        self.phase = Phase::Settled;
        true
    }

    /// `attempt` failed; only the current attempt's failure decides anything.
    pub fn failed(&mut self, attempt: u32, failure: AttemptFailure, now: Ts, steps: &mut Vec<AttemptStep>) {
        if self.current() == Some(attempt) {
            self.close_attempt(attempt, now);
            self.fail(attempt, failure.into_error(), now, steps);
        }
    }

    /// The attempt's transport resources are released. Closed before reporting an outcome: the
    /// transport lost it.
    pub fn closed(&mut self, attempt: u32, now: Ts, steps: &mut Vec<AttemptStep>) {
        self.open.remove(&attempt);
        if self.current() == Some(attempt) {
            let failure = AttemptFailure::Connection("the response ended unexpectedly".into());
            self.fail(attempt, failure.into_error(), now, steps);
        }
    }

    /// The consumer gave up: the running attempt is stopped and no outcome follows.
    pub fn cancel(&mut self, now: Ts, steps: &mut Vec<AttemptStep>) {
        if let Some(attempt) = self.current() {
            self.stop(attempt, now, steps);
        }
        self.phase = Phase::Settled;
    }

    /// No further attempt and no total limit, but the running attempt is left to answer within
    /// the cleanup deadline: an answer may hold a resource that must be released.
    pub fn drain(&mut self, now: Ts) {
        match self.phase {
            Phase::Waiting { .. } => self.phase = Phase::Settled,
            Phase::Attempting { attempt, .. } => {
                self.policy.max_retry = attempt;
                self.total_deadline = None;
                self.close_attempt(attempt, now);
            }
            Phase::Settled => {}
        }
    }

    pub fn wake(&mut self, now: Ts, expecting: Option<Ts>, steps: &mut Vec<AttemptStep>) {
        let expired: Vec<u32> =
            self.open.iter().filter(|(_, deadline)| deadline.is_some_and(|d| d <= now)).map(|(a, _)| *a).collect();
        for attempt in expired {
            // Reported once; the runtime keeps owning the attempt.
            self.open.remove(&attempt);
            if self.current() == Some(attempt) {
                steps.push(AttemptStep::Stop(attempt));
                self.phase = Phase::Settled;
            }
            steps.push(AttemptStep::CleanupIncomplete(attempt));
        }
        if self.is_decided() {
            return;
        }
        if self.total_deadline.is_some_and(|d| d <= now) {
            self.cancel(now, steps);
            steps.push(AttemptStep::GiveUp(ApiError::timed_out(TimeoutPhase::Total).with_retryable(false)));
            return;
        }
        match self.phase {
            Phase::Waiting { attempt, at } if at <= now => self.launch(attempt, now, steps),
            Phase::Attempting { attempt, stage } => {
                if let Some((at, phase)) = self.stage_deadline(stage, expecting)
                    && at <= now
                {
                    self.stop(attempt, now, steps);
                    self.fail(attempt, ApiError::timed_out(phase), now, steps);
                }
            }
            _ => {}
        }
    }

    fn launch(&mut self, attempt: u32, now: Ts, steps: &mut Vec<AttemptStep>) {
        self.open.insert(attempt, None);
        self.phase = Phase::Attempting { attempt, stage: Stage::Connecting { since: now } };
        steps.push(AttemptStep::Launch(attempt));
    }

    /// The attempt's transport task ends on its own after a terminal report or a stop; it still
    /// owes the report that its resources are released.
    fn close_attempt(&mut self, attempt: u32, now: Ts) {
        if let Some(deadline) = self.open.get_mut(&attempt) {
            deadline.get_or_insert(now.saturating_add(ATTEMPT_CLEANUP_TIMEOUT));
        }
    }

    fn stop(&mut self, attempt: u32, now: Ts, steps: &mut Vec<AttemptStep>) {
        if self.open.contains_key(&attempt) {
            self.close_attempt(attempt, now);
            steps.push(AttemptStep::Stop(attempt));
        }
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
    fn fail(&mut self, attempt: u32, error: ApiError, now: Ts, steps: &mut Vec<AttemptStep>) {
        let retryable = error.retryable && !self.replay_unsafe;
        if retryable && attempt < self.policy.max_retry {
            let delay = if attempt == 0 { FIRST_RETRY_INTERVAL } else { self.policy.retry_interval };
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
            Stage::Connecting { since } => p.connect_timeout.map(|t| (since.saturating_add(t), TimeoutPhase::Connect)),
            Stage::Writing { since } => p.write_timeout.map(|t| (since.saturating_add(t), TimeoutPhase::Write)),
            Stage::Reading { last_progress } => {
                let waiting_since = expecting?.max(last_progress);
                p.read_timeout.map(|t| (waiting_since.saturating_add(t), TimeoutPhase::Read))
            }
        }
    }
}
