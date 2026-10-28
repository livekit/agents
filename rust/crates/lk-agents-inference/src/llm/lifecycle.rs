//! One chat completion request as a synchronous machine: accepted at a supplied instant, it owns
//! the attempts, their phase deadlines, the retry decision, the response assembly and the
//! cleanup owed by every attempt it launched. The runtime performs the HTTP exchange an
//! [`LlmRequestEffect::LlmRequestStart`] asks for and reports what the transport observed.
//!
//! Timeouts follow `APIConnectOptions`: connect runs from launch until the transport has a
//! connection, write until the request body is handed over, read while a response is expected
//! and the consumer is waiting for it (downstream backpressure is not remote inactivity), each
//! restarted by transport progress; total runs from acceptance and never restarts. A failure is
//! retried only while nothing visible was delivered, since a replay would repeat it.

use std::collections::BTreeMap;
use std::time::Duration;

use lk_agents_abi::errors::{ApiError, TimeoutPhase};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::{LlmChunk, Ts};

use super::parse::{ChoiceAccumulator, TagFilter, WireChunk, usage_chunk};

/// How long an attempt's transport task may take to settle once asked to stop. Dropping an HTTP
/// exchange is immediate, so a miss is a runtime defect, reported rather than waited on.
pub const ATTEMPT_CLEANUP_TIMEOUT: Duration = Duration::from_secs(5);

/// The first retry waits this long, the next ones `retry_interval` (`APIConnectOptions`).
const FIRST_RETRY_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LlmRequestId(pub u64);

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

/// What the transport reports. Every attempt-scoped event names its attempt; one from an
/// attempt that is no longer current only settles that attempt's cleanup.
#[derive(Debug, Clone)]
pub enum LlmRequestEvent {
    /// A deadline from [`LlmRequest::wake_at`] is due.
    Wake,
    /// The consumer started or stopped waiting for the next chunk.
    DemandStarted,
    DemandEnded,
    Connected { attempt: u32 },
    RequestWritten { attempt: u32 },
    /// Response bytes that completed no event.
    Progress { attempt: u32 },
    /// One decoded completion chunk.
    Chunk { attempt: u32, chunk: WireChunk },
    /// The `[DONE]` sentinel, or the body's clean end.
    Ended { attempt: u32 },
    Failed { attempt: u32, failure: AttemptFailure },
    /// The attempt's transport resources are released.
    AttemptClosed { attempt: u32 },
    /// The consumer gave up on the request.
    Cancel,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AttemptFailure {
    /// A non-2xx response, its message read from the body.
    Rejected { status: i64, message: String, request_id: Option<String> },
    /// Launch, connection or mid-body transport failure.
    Connection(String),
    /// The body is not a well-formed completion stream.
    Malformed(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum LlmRequestEffect {
    LlmRequestStart { request_id: LlmRequestId, attempt: u32, llm: ResourceId },
    LlmRequestCancel { request_id: LlmRequestId, attempt: u32 },
    Deliver(LlmChunk),
    /// A failed attempt that will be retried.
    AttemptFailed { attempt: u32, error: ApiError },
    /// The request's single outcome. Cancellation has none: the consumer asked for it.
    Finished(Result<(), ApiError>),
    /// An attempt missed its cleanup deadline; the runtime still owns it.
    CleanupIncomplete { request_id: LlmRequestId, attempt: u32 },
}

#[derive(Debug)]
enum Phase {
    /// Backing off before `attempt`.
    Waiting { attempt: u32, at: Ts },
    Attempting { attempt: u32, stage: Stage, assembly: Assembly },
    Settled,
}

#[derive(Debug, Clone, Copy)]
enum Stage {
    Connecting { since: Ts },
    Writing { since: Ts },
    Reading { last_progress: Ts },
}

/// One attempt's response assembly; a retry starts over.
#[derive(Debug)]
struct Assembly {
    choices: ChoiceAccumulator,
    filter: TagFilter,
}

#[derive(Debug)]
pub struct LlmRequest {
    id: LlmRequestId,
    llm: ResourceId,
    policy: RequestPolicy,
    think_tags: (&'static str, &'static str),
    total_deadline: Option<Ts>,
    phase: Phase,
    /// Since when the consumer has been waiting, if it is.
    demand: Option<Ts>,
    /// Visible output was delivered: a replay would repeat it.
    delivered: bool,
    /// Launched attempts not yet closed, with their cleanup deadline once asked to stop.
    open: BTreeMap<u32, Option<Ts>>,
}

impl LlmRequest {
    /// Accepts the request at `now` and launches its first attempt.
    pub fn start(
        id: LlmRequestId,
        llm: ResourceId,
        policy: RequestPolicy,
        think_tags: (&'static str, &'static str),
        now: Ts,
    ) -> (Self, Vec<LlmRequestEffect>) {
        let mut request = LlmRequest {
            id,
            llm,
            policy,
            think_tags,
            total_deadline: policy.total_timeout.map(|t| now.saturating_add(t)),
            phase: Phase::Settled,
            demand: None,
            delivered: false,
            open: BTreeMap::new(),
        };
        let mut effects = Vec::new();
        request.launch(0, now, &mut effects);
        (request, effects)
    }

    /// The earliest instant a [`LlmRequestEvent::Wake`] has work to do.
    pub fn wake_at(&self) -> Option<Ts> {
        let phase = match &self.phase {
            Phase::Waiting { at, .. } => Some(*at),
            Phase::Attempting { stage, .. } => self.stage_deadline(*stage).map(|(at, _)| at),
            Phase::Settled => None,
        };
        let settling = !matches!(self.phase, Phase::Settled);
        [phase, self.total_deadline.filter(|_| settling), self.open.values().flatten().min().copied()]
            .into_iter()
            .flatten()
            .min()
    }

    /// No outcome is owed and no attempt is left to clean up.
    pub fn is_settled(&self) -> bool {
        matches!(self.phase, Phase::Settled) && self.open.is_empty()
    }

    pub fn handle(&mut self, event: LlmRequestEvent, now: Ts) -> Vec<LlmRequestEffect> {
        let mut effects = Vec::new();
        match event {
            LlmRequestEvent::Wake => self.wake(now, &mut effects),
            LlmRequestEvent::DemandStarted => self.demand = Some(now),
            LlmRequestEvent::DemandEnded => self.demand = None,
            LlmRequestEvent::Cancel => {
                if let Some(attempt) = self.current_attempt() {
                    self.stop(attempt, now, &mut effects);
                }
                self.phase = Phase::Settled;
            }
            LlmRequestEvent::AttemptClosed { attempt } => {
                self.open.remove(&attempt);
                // Closed before reporting an outcome: the transport lost it.
                if self.current_attempt() == Some(attempt) {
                    let failure = AttemptFailure::Connection("the response ended unexpectedly".into());
                    self.fail(attempt, failure_error(failure), now, &mut effects);
                }
            }
            LlmRequestEvent::Connected { attempt } => {
                self.advance(attempt, |stage| matches!(stage, Stage::Connecting { .. }).then_some(Stage::Writing { since: now }));
            }
            LlmRequestEvent::RequestWritten { attempt } => {
                self.advance(attempt, |stage| match stage {
                    Stage::Connecting { .. } | Stage::Writing { .. } => Some(Stage::Reading { last_progress: now }),
                    Stage::Reading { .. } => None,
                });
            }
            LlmRequestEvent::Progress { attempt } => self.progress(attempt, now),
            LlmRequestEvent::Chunk { attempt, chunk } => {
                self.progress(attempt, now);
                self.assemble(attempt, &chunk, &mut effects);
            }
            LlmRequestEvent::Ended { attempt } => {
                if self.current_attempt() == Some(attempt) {
                    self.close_attempt(attempt, now);
                    self.phase = Phase::Settled;
                    effects.push(LlmRequestEffect::Finished(Ok(())));
                }
            }
            LlmRequestEvent::Failed { attempt, failure } => {
                if self.current_attempt() == Some(attempt) {
                    self.close_attempt(attempt, now);
                    self.fail(attempt, failure_error(failure), now, &mut effects);
                }
            }
        }
        effects
    }

    fn current_attempt(&self) -> Option<u32> {
        match self.phase {
            Phase::Attempting { attempt, .. } => Some(attempt),
            _ => None,
        }
    }

    fn launch(&mut self, attempt: u32, now: Ts, effects: &mut Vec<LlmRequestEffect>) {
        self.open.insert(attempt, None);
        self.phase = Phase::Attempting {
            attempt,
            stage: Stage::Connecting { since: now },
            assembly: Assembly {
                choices: ChoiceAccumulator::new(),
                filter: TagFilter::new(self.think_tags.0, self.think_tags.1),
            },
        };
        effects.push(LlmRequestEffect::LlmRequestStart { request_id: self.id, attempt, llm: self.llm });
    }

    /// The attempt's transport task ends on its own after a terminal report; it still owes the
    /// report that its resources are released.
    fn close_attempt(&mut self, attempt: u32, now: Ts) {
        if let Some(deadline) = self.open.get_mut(&attempt) {
            deadline.get_or_insert(now.saturating_add(ATTEMPT_CLEANUP_TIMEOUT));
        }
    }

    fn stop(&mut self, attempt: u32, now: Ts, effects: &mut Vec<LlmRequestEffect>) {
        if self.open.contains_key(&attempt) {
            self.close_attempt(attempt, now);
            effects.push(LlmRequestEffect::LlmRequestCancel { request_id: self.id, attempt });
        }
    }

    fn advance(&mut self, attempt: u32, next: impl FnOnce(Stage) -> Option<Stage>) {
        if let Phase::Attempting { attempt: current, stage, .. } = &mut self.phase
            && *current == attempt
            && let Some(next) = next(*stage)
        {
            *stage = next;
        }
    }

    /// Response bytes prove the request was sent, whatever the transport reported before.
    fn progress(&mut self, attempt: u32, now: Ts) {
        self.advance(attempt, |_| Some(Stage::Reading { last_progress: now }));
    }

    fn assemble(&mut self, attempt: u32, chunk: &WireChunk, effects: &mut Vec<LlmRequestEffect>) {
        let Phase::Attempting { attempt: current, assembly, .. } = &mut self.phase else { return };
        if *current != attempt {
            return;
        }
        for choice in &chunk.choices {
            if let Some(parsed) = assembly.choices.parse_choice(&chunk.id, choice, &mut assembly.filter) {
                self.delivered |= parsed.has_response();
                effects.push(LlmRequestEffect::Deliver(parsed));
            }
        }
        if let Some(usage) = usage_chunk(chunk) {
            effects.push(LlmRequestEffect::Deliver(usage));
        }
    }

    /// Retries while the error allows it, nothing visible was delivered and attempts remain;
    /// the backoff never outlives the total deadline, whose own wake ends the request.
    fn fail(&mut self, attempt: u32, error: ApiError, now: Ts, effects: &mut Vec<LlmRequestEffect>) {
        let retryable = error.retryable && !self.delivered;
        if retryable && attempt < self.policy.max_retry {
            let delay = if attempt == 0 { FIRST_RETRY_INTERVAL } else { self.policy.retry_interval };
            self.phase = Phase::Waiting { attempt: attempt + 1, at: now.saturating_add(delay) };
            effects.push(LlmRequestEffect::AttemptFailed { attempt, error });
        } else {
            self.phase = Phase::Settled;
            effects.push(LlmRequestEffect::Finished(Err(error.with_retryable(retryable))));
        }
    }

    fn stage_deadline(&self, stage: Stage) -> Option<(Ts, TimeoutPhase)> {
        let p = &self.policy;
        match stage {
            Stage::Connecting { since } => p.connect_timeout.map(|t| (since.saturating_add(t), TimeoutPhase::Connect)),
            Stage::Writing { since } => p.write_timeout.map(|t| (since.saturating_add(t), TimeoutPhase::Write)),
            Stage::Reading { last_progress } => {
                let waiting_since = self.demand?.max(last_progress);
                p.read_timeout.map(|t| (waiting_since.saturating_add(t), TimeoutPhase::Read))
            }
        }
    }

    fn wake(&mut self, now: Ts, effects: &mut Vec<LlmRequestEffect>) {
        let expired: Vec<u32> = self
            .open
            .iter()
            .filter(|(_, deadline)| deadline.is_some_and(|d| d <= now))
            .map(|(attempt, _)| *attempt)
            .collect();
        for attempt in expired {
            // Reported once; the runtime keeps owning the attempt (decision 14).
            self.open.remove(&attempt);
            effects.push(LlmRequestEffect::CleanupIncomplete { request_id: self.id, attempt });
        }
        if matches!(self.phase, Phase::Settled) {
            return;
        }
        if self.total_deadline.is_some_and(|d| d <= now) {
            if let Some(attempt) = self.current_attempt() {
                self.stop(attempt, now, effects);
            }
            self.phase = Phase::Settled;
            let error = ApiError::timed_out(TimeoutPhase::Total).with_retryable(false);
            effects.push(LlmRequestEffect::Finished(Err(error)));
            return;
        }
        match self.phase {
            Phase::Waiting { attempt, at } if at <= now => self.launch(attempt, now, effects),
            Phase::Attempting { attempt, stage, .. } => {
                if let Some((at, phase)) = self.stage_deadline(stage)
                    && at <= now
                {
                    self.stop(attempt, now, effects);
                    self.fail(attempt, ApiError::timed_out(phase), now, effects);
                }
            }
            _ => {}
        }
    }
}

fn failure_error(failure: AttemptFailure) -> ApiError {
    match failure {
        AttemptFailure::Rejected { status, message, request_id } => ApiError::from_http(&message, status, request_id),
        AttemptFailure::Connection(message) => ApiError::connection(Some(message)),
        AttemptFailure::Malformed(message) => {
            ApiError::connection(Some(format!("malformed completion stream: {message}")))
        }
    }
}

#[cfg(test)]
mod tests;
