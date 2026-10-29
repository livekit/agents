//! The frontend expert event queue (voice-session.md "Expert events and frontend presentation"):
//! one frontend conversation's delegations to remote experts, their queued events, and the expert
//! question that owns the person's next reply. Decisions only; the driver presents, forwards and
//! resolves. Owned by the session, so replacing the speaking agent keeps it.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use lk_agents_abi::llm::tool_executor::ToolOutcome;
use serde::{Deserialize, Serialize};

/// A remote tool's configured identity: its endpoint and tool name, never a similar
/// model-visible name alone.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ExpertTool {
    pub endpoint: String,
    pub name: String,
}

impl ExpertTool {
    pub fn new(endpoint: &str, name: &str) -> Self {
        Self { endpoint: endpoint.trim_end_matches('/').to_owned(), name: name.to_owned() }
    }
}

/// What the session allows when the queue decides.
#[derive(Debug, Clone, Copy)]
pub struct Floor<'a> {
    /// The expert tools the active agent exposes.
    pub tools: &'a [ExpertTool],
    /// No speech, hold or other reply occupies the conversation.
    pub free: bool,
    /// A `VoiceTask` owns the live interaction: the person's turns answer it.
    pub task: bool,
}

impl Floor<'_> {
    fn reaches(&self, tool: &ExpertTool) -> bool {
        !self.task && self.tools.contains(tool)
    }
}

/// One event to present, chosen when the floor permits.
#[derive(Debug, Clone, PartialEq)]
pub enum Presentation {
    /// The delegation's latest progress; older progress was superseded.
    Progress { call_id: String, text: String },
    /// The question now owns the person's next reply (re-presented after a suspension).
    Question { call_id: String, request_id: String, question: String, verbatim: bool },
    /// The originating call's result.
    Outcome { call_id: String, outcome: ToolOutcome },
}

/// The pending input request a person's reply is collected for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingRequest {
    pub call_id: String,
    pub request_id: String,
}

/// How an input request entered the queue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Asked {
    /// Queued; `superseded` is the same expert's earlier request that can no longer be answered.
    Queued { superseded: Option<String> },
    /// The same request again: neither queued nor asked twice.
    Duplicate,
    /// No open delegation of this call.
    Refused,
}

/// How an outcome entered the queue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Settled {
    /// Queued to resolve the call; `superseded` is a request that can no longer be answered.
    Queued { superseded: Option<String> },
    /// The call is already resolved: never resolved twice.
    Duplicate,
    /// Not a delegation of this queue: an ordinary tool end.
    NotOwned,
}

/// An answer for a request that is no longer the one waiting: settled, superseded or never asked.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("the expert is no longer waiting for this answer")]
pub struct Stale;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Delegation {
    id: String,
    tool: ExpertTool,
    /// The last progress occurrence taken: an older or repeated one is a duplicate.
    progress_seen: u64,
    /// Every input request taken.
    requests: BTreeSet<String>,
    settled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Event {
    Progress(String),
    Question { request_id: String, question: String, verbatim: bool },
    Outcome(ToolOutcome),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Queued {
    call_id: String,
    event: Event,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Foreground {
    call_id: String,
    request_id: String,
    question: String,
    verbatim: bool,
    /// Presented, then the active agent lost the expert's tool (or a task took the
    /// interaction): replies are not routed until it is presented again.
    suspended: bool,
}

/// The queue's records: serializable delegation references, queued events in order, and the
/// foreground request. Saved under the frontend conversation's identity when the application
/// persists frontend delivery; never a live await.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Experts {
    /// By the originating call's id.
    delegations: BTreeMap<String, Delegation>,
    queue: VecDeque<Queued>,
    foreground: Option<Foreground>,
}

impl Experts {
    /// The adapter started a delegation for `call_id`. A call opens one delegation.
    pub fn open(&mut self, call_id: &str, delegation_id: &str, tool: ExpertTool) -> bool {
        if self.delegations.contains_key(call_id) {
            return false;
        }
        let delegation = Delegation {
            id: delegation_id.to_owned(),
            tool,
            progress_seen: 0,
            requests: BTreeSet::new(),
            settled: false,
        };
        self.delegations.insert(call_id.to_owned(), delegation);
        true
    }

    /// The call's delegation is open and unresolved.
    pub fn owns(&self, call_id: &str) -> bool {
        self.delegations.get(call_id).is_some_and(|d| !d.settled)
    }

    /// The delegation id of the call's delegation.
    pub fn delegation_id(&self, call_id: &str) -> Option<&str> {
        self.delegations.get(call_id).map(|d| d.id.as_str())
    }

    /// Progress `occurrence` (from 1, increasing per delegation). Queued progress of the same
    /// delegation is superseded in place.
    pub fn progress(&mut self, call_id: &str, occurrence: u64, text: String) -> bool {
        let Some(d) = self.delegations.get_mut(call_id).filter(|d| !d.settled) else {
            return false;
        };
        if occurrence <= d.progress_seen {
            return false;
        }
        d.progress_seen = occurrence;
        let queued = self.queue.iter_mut().find(|q| q.call_id == call_id);
        match queued {
            Some(Queued { event: Event::Progress(old), .. }) => *old = text,
            Some(_) | None => self
                .queue
                .push_back(Queued { call_id: call_id.into(), event: Event::Progress(text) }),
        }
        true
    }

    /// The expert asks. Its own earlier request and progress are superseded: only the latest
    /// question of a delegation can be answered.
    pub fn ask(
        &mut self,
        call_id: &str,
        request_id: &str,
        question: String,
        verbatim: bool,
    ) -> Asked {
        let Some(d) = self.delegations.get_mut(call_id).filter(|d| !d.settled) else {
            return Asked::Refused;
        };
        if !d.requests.insert(request_id.to_owned()) {
            return Asked::Duplicate;
        }
        let superseded = self.withdraw(call_id);
        let event = Event::Question { request_id: request_id.into(), question, verbatim };
        self.queue.push_back(Queued { call_id: call_id.into(), event });
        Asked::Queued { superseded }
    }

    /// The delegation ended with `outcome`, which resolves its originating call once presented.
    pub fn outcome(&mut self, call_id: &str, outcome: ToolOutcome) -> Settled {
        match self.delegations.get_mut(call_id) {
            None => return Settled::NotOwned,
            Some(d) if d.settled => return Settled::Duplicate,
            Some(d) => d.settled = true,
        }
        let superseded = self.withdraw(call_id);
        self.queue.push_back(Queued { call_id: call_id.into(), event: Event::Outcome(outcome) });
        Settled::Queued { superseded }
    }

    /// The application settled or cancelled the interaction: nothing of it is presented, and its
    /// pending request (returned) can no longer be answered.
    pub fn cancel(&mut self, call_id: &str) -> Option<String> {
        let d = self.delegations.get_mut(call_id).filter(|d| !d.settled)?;
        d.settled = true;
        let superseded = self.withdraw(call_id);
        self.queue.retain(|q| q.call_id != call_id);
        superseded
    }

    /// Drop the call's queued progress and its question, queued or in the foreground: the
    /// returned request can no longer be answered.
    fn withdraw(&mut self, call_id: &str) -> Option<String> {
        let mut superseded =
            self.foreground.take_if(|f| f.call_id == call_id).map(|f| f.request_id);
        self.queue.retain(|q| match &q.event {
            _ if q.call_id != call_id => true,
            Event::Outcome(_) => true,
            Event::Question { request_id, .. } => {
                superseded = Some(request_id.clone());
                false
            }
            Event::Progress(_) => false,
        });
        superseded
    }

    /// Suspend a presented question the active agent can no longer interact with.
    pub fn observe(&mut self, floor: Floor<'_>) {
        let Some(fg) = &mut self.foreground else { return };
        let tool = &self.delegations[&fg.call_id].tool;
        if !floor.reaches(tool) {
            fg.suspended = true;
        }
    }

    /// The next event to present, in arrival order among those the active agent's tools reach:
    /// a suspended question first, once reachable again. A question waits while another owns
    /// the person's reply.
    pub fn next(&mut self, floor: Floor<'_>) -> Option<Presentation> {
        self.observe(floor);
        if !floor.free {
            return None;
        }
        if let Some(fg) = self.foreground.as_mut().filter(|f| f.suspended)
            && floor.reaches(&self.delegations[&fg.call_id].tool)
        {
            fg.suspended = false;
            return Some(Presentation::Question {
                call_id: fg.call_id.clone(),
                request_id: fg.request_id.clone(),
                question: fg.question.clone(),
                verbatim: fg.verbatim,
            });
        }
        let at = self.queue.iter().position(|q| {
            floor.reaches(&self.delegations[&q.call_id].tool)
                && !(matches!(q.event, Event::Question { .. }) && self.foreground.is_some())
        })?;
        let Queued { call_id, event } = self.queue.remove(at)?;
        Some(match event {
            Event::Progress(text) => Presentation::Progress { call_id, text },
            Event::Question { request_id, question, verbatim } => {
                self.foreground = Some(Foreground {
                    call_id: call_id.clone(),
                    request_id: request_id.clone(),
                    question: question.clone(),
                    verbatim,
                    suspended: false,
                });
                Presentation::Question { call_id, request_id, question, verbatim }
            }
            Event::Outcome(outcome) => Presentation::Outcome { call_id, outcome },
        })
    }

    /// Collection: the request a person's reply answers now, if any. Nothing is forwarded while
    /// the question is suspended or a task owns the interaction.
    pub fn collect(&mut self, floor: Floor<'_>) -> Option<PendingRequest> {
        self.observe(floor);
        let fg = self.foreground.as_ref().filter(|f| !f.suspended)?;
        Some(PendingRequest { call_id: fg.call_id.clone(), request_id: fg.request_id.clone() })
    }

    /// Delivery: an answer to exactly this pending request, which it consumes.
    pub fn answer(&mut self, request: &PendingRequest) -> Result<(), Stale> {
        self.foreground
            .take_if(|f| f.call_id == request.call_id && f.request_id == request.request_id)
            .map(|_| ())
            .ok_or(Stale)
    }
}

#[cfg(test)]
mod tests;
