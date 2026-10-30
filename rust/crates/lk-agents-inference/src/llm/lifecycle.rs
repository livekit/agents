//! One chat completion request as a synchronous machine: accepted at a supplied instant, it owns
//! its [`Attempts`], the response assembly and the replay guard. The runtime performs the HTTP
//! exchange an [`LlmRequestEffect::LlmRequestStart`] asks for and reports what the transport
//! observed.
//!
//! The read limit runs only while the consumer is waiting for the next chunk: downstream
//! backpressure is not remote inactivity. A failure is retried only while nothing visible was
//! delivered, since a replay would repeat it.

use lk_agents_abi::errors::ApiError;
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::{LlmChunk, Ts};

pub use lk_agents_core::attempts::{AttemptFailure, RequestPolicy};
use lk_agents_core::attempts::{AttemptStep, Attempts};

use super::parse::{ChoiceAccumulator, TagFilter, WireChunk, usage_chunk};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LlmRequestId(pub u64);

/// What the transport reports. Every attempt-scoped event names its attempt; one from an
/// attempt that is no longer current only settles that attempt's cleanup.
#[derive(Debug, Clone)]
pub enum LlmRequestEvent {
    /// A deadline from [`LlmRequest::wake_at`] is due.
    Wake,
    /// The consumer started or stopped waiting for the next chunk.
    DemandStarted,
    DemandEnded,
    Connected {
        attempt: u32,
    },
    RequestWritten {
        attempt: u32,
    },
    /// Response bytes that completed no event.
    Progress {
        attempt: u32,
    },
    /// One decoded completion chunk.
    Chunk {
        attempt: u32,
        chunk: WireChunk,
    },
    /// The `[DONE]` sentinel.
    Ended {
        attempt: u32,
    },
    Failed {
        attempt: u32,
        failure: AttemptFailure,
    },
    /// The attempt's transport resources are released.
    AttemptClosed {
        attempt: u32,
    },
    /// The consumer gave up on the request.
    Cancel,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LlmRequestEffect {
    LlmRequestStart {
        request_id: LlmRequestId,
        attempt: u32,
        llm: ResourceId,
    },
    LlmRequestCancel {
        request_id: LlmRequestId,
        attempt: u32,
    },
    Deliver(LlmChunk),
    /// A failed attempt that will be retried.
    AttemptFailed {
        attempt: u32,
        error: ApiError,
    },
    /// The request's single outcome. Cancellation has none: the consumer asked for it.
    Finished(Result<(), ApiError>),
    /// An attempt missed its cleanup deadline; the runtime still owns it.
    CleanupIncomplete {
        request_id: LlmRequestId,
        attempt: u32,
    },
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
    think_tags: (&'static str, &'static str),
    attempts: Attempts,
    assembly: Assembly,
    /// Since when the consumer has been waiting, if it is.
    demand: Option<Ts>,
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
        let mut steps = Vec::new();
        let attempts = Attempts::start(policy, now, &mut steps);
        let mut request = LlmRequest {
            id,
            llm,
            think_tags,
            attempts,
            assembly: Assembly {
                choices: ChoiceAccumulator::new(),
                filter: TagFilter::new(think_tags.0, think_tags.1),
            },
            demand: None,
        };
        let mut effects = Vec::new();
        request.apply(steps, &mut effects);
        (request, effects)
    }

    /// The earliest instant a [`LlmRequestEvent::Wake`] has work to do.
    pub fn wake_at(&self) -> Option<Ts> {
        self.attempts.wake_at(self.demand)
    }

    /// No outcome is owed and no attempt is left to clean up.
    pub fn is_settled(&self) -> bool {
        self.attempts.is_settled()
    }

    pub fn handle(&mut self, event: LlmRequestEvent, now: Ts) -> Vec<LlmRequestEffect> {
        let mut effects = Vec::new();
        let mut steps = Vec::new();
        match event {
            LlmRequestEvent::Wake => self.attempts.wake(now, self.demand, &mut steps),
            LlmRequestEvent::DemandStarted => self.demand = Some(now),
            LlmRequestEvent::DemandEnded => self.demand = None,
            LlmRequestEvent::Cancel => self.attempts.cancel(now, &mut steps),
            LlmRequestEvent::AttemptClosed { attempt } => {
                self.attempts.closed(attempt, now, &mut steps)
            }
            LlmRequestEvent::Connected { attempt } => self.attempts.connected(attempt, now),
            LlmRequestEvent::RequestWritten { attempt } => self.attempts.written(attempt, now),
            LlmRequestEvent::Progress { attempt } => self.attempts.progress(attempt, now),
            LlmRequestEvent::Chunk { attempt, chunk } => {
                self.attempts.progress(attempt, now);
                if self.attempts.current() == Some(attempt) {
                    self.assemble(&chunk, &mut effects);
                }
            }
            LlmRequestEvent::Ended { attempt } => {
                if self.attempts.succeeded(attempt, now) {
                    effects.push(LlmRequestEffect::Finished(Ok(())));
                }
            }
            LlmRequestEvent::Failed { attempt, failure } => {
                self.attempts.failed(attempt, failure, now, &mut steps)
            }
        }
        self.apply(steps, &mut effects);
        effects
    }

    fn apply(&mut self, steps: Vec<AttemptStep>, effects: &mut Vec<LlmRequestEffect>) {
        let request_id = self.id;
        for step in steps {
            effects.push(match step {
                AttemptStep::Launch(attempt) => {
                    let (open, close) = self.think_tags;
                    self.assembly = Assembly {
                        choices: ChoiceAccumulator::new(),
                        filter: TagFilter::new(open, close),
                    };
                    LlmRequestEffect::LlmRequestStart { request_id, attempt, llm: self.llm }
                }
                AttemptStep::Stop(attempt) => {
                    LlmRequestEffect::LlmRequestCancel { request_id, attempt }
                }
                AttemptStep::Retry { attempt, error } => {
                    LlmRequestEffect::AttemptFailed { attempt, error }
                }
                AttemptStep::GiveUp(error) => LlmRequestEffect::Finished(Err(error)),
                AttemptStep::CleanupIncomplete(attempt) => {
                    LlmRequestEffect::CleanupIncomplete { request_id, attempt }
                }
                // A generation is never retired: its consumer's cancel ends it.
                AttemptStep::Unresolved => continue,
            });
        }
    }

    fn assemble(&mut self, chunk: &WireChunk, effects: &mut Vec<LlmRequestEffect>) {
        let assembly = &mut self.assembly;
        for choice in &chunk.choices {
            if let Some(parsed) =
                assembly.choices.parse_choice(&chunk.id, choice, &mut assembly.filter)
            {
                if parsed.has_response() {
                    self.attempts.delivered();
                }
                effects.push(LlmRequestEffect::Deliver(parsed));
            }
        }
        if let Some(usage) = usage_chunk(chunk) {
            effects.push(LlmRequestEffect::Deliver(usage));
        }
    }
}

#[cfg(test)]
mod tests;
