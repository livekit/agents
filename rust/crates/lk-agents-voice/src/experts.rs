use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::agent_session::machine::activity::scheduling::SpeechId;
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::session::InputRequestRef;
use serde::{Deserialize, Serialize};

// Owned by the session, not the agent: replacing the speaking agent keeps the queue.

/// Remote identity survives a model-facing tool rename. A local body is eligible only
/// through the same configured child/tool pair; strings from those domains never alias.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
enum ExpertKey {
    RemoteId(String),
    LocalTool { agent: String, tool: String },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ExpertTool {
    key: ExpertKey,
    /// Used to select a newly detached call from the current installed model tools.
    pub name: String,
}

impl ExpertTool {
    pub fn remote(remote_id: &str, name: &str) -> Self {
        Self { key: ExpertKey::RemoteId(remote_id.into()), name: name.into() }
    }

    pub fn local(agent: &str, name: &str) -> Self {
        Self {
            key: ExpertKey::LocalTool { agent: agent.into(), tool: name.into() },
            name: name.into(),
        }
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
        !self.task && self.tools.iter().any(|candidate| candidate.key == tool.key)
    }
}

/// One event to present, chosen when the floor permits.
#[derive(Debug, Clone, PartialEq)]
pub enum Presentation {
    /// The delegation's latest progress; older progress was superseded.
    Progress { output_id: String, text: String },
    /// Exact text is never coalesced, rephrased or withdrawn by a later question or result.
    Exact { output_id: String, text: String, call: Option<crate::agent_session::SpeechCall> },
    /// A question selected for presentation. Native response admission supplies its answers;
    /// presentation priority alone never selects a remote reply destination.
    Question { output_id: String, request_id: String, question: String, verbatim: bool },
    /// The originating call's result.
    Outcome { output_id: String, outcome: ToolOutcome },
}

/// The pending input request a person's reply is collected for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingRequest {
    pub output_id: String,
    pub request_id: String,
}

/// An answer for a request that is no longer the one waiting: settled, superseded or never asked.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("the expert is no longer waiting for this answer")]
pub struct Stale;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Delegation {
    tool: ExpertTool,
    /// The last progress occurrence taken: an older or repeated one is a duplicate.
    progress_seen: u64,
    /// Every input request taken.
    requests: BTreeMap<String, Option<InputRequestRef>>,
    /// The latest live input occurrence. Older requests retain only receipt provenance.
    waiting: Option<String>,
    /// The originating call's pending output is on the record: its outcome can replace it.
    recorded: bool,
    settled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum ExpertInboxEvent {
    Progress(String),
    Exact {
        text: String,
        /// A live speech promise is never recovered as a new caller's authority.
        #[serde(skip)]
        call: Option<crate::agent_session::SpeechCall>,
    },
    Question {
        request_id: String,
        question: String,
        verbatim: bool,
    },
    Outcome(ToolOutcome),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Queued {
    output_id: String,
    event: ExpertInboxEvent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Foreground {
    output_id: String,
    request_id: String,
    question: String,
    verbatim: bool,
    /// The active agent lost this tool or a task took the interaction. Re-offering the prompt
    /// does not invalidate its earlier native delivery or a response already recognized.
    suspended: bool,
}

/// The queue's records: serializable delegation references, queued events in order, and the
/// foreground request. Saved under the frontend conversation's identity when the application
/// persists frontend delivery; never a live await.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Experts {
    /// By the originating call's canonical output item ID, never its provider call ID.
    delegations: BTreeMap<String, Delegation>,
    queue: VecDeque<Queued>,
    foreground: Option<Foreground>,
    /// Live output receipts have no meaning after a call ends; communication owns recovery.
    #[serde(skip)]
    presentations: BTreeMap<SpeechId, PendingRequest>,
    /// Admitted human responses awaiting an eligible live continuation; never persisted here.
    #[serde(skip)]
    responses: BTreeMap<InputRequestRef, lk_agents_abi::ChatMessage>,
    /// Wake the frontend once per held occurrence, including after its tool becomes eligible.
    #[serde(skip)]
    prompted_responses: BTreeSet<InputRequestRef>,
}

impl Experts {
    /// A detached call delegates through its canonical output item ID.
    pub fn open(&mut self, output_id: &str, tool: ExpertTool) -> bool {
        if self.delegations.contains_key(output_id) {
            return false;
        }
        let delegation = Delegation {
            tool,
            progress_seen: 0,
            requests: BTreeMap::new(),
            waiting: None,
            recorded: false,
            settled: false,
        };
        self.delegations.insert(output_id.to_owned(), delegation);
        true
    }

    /// The originating call's pending output was committed to the record.
    pub fn recorded(&mut self, output_id: &str) {
        if let Some(d) = self.delegations.get_mut(output_id) {
            d.recorded = true;
        }
    }

    /// The call's delegation is open and unresolved.
    pub fn owns(&self, output_id: &str) -> bool {
        self.delegations.get(output_id).is_some_and(|d| !d.settled)
    }

    /// Progress `occurrence` (from 1, increasing per delegation). Queued progress of the same
    /// delegation is superseded in place.
    pub fn progress(&mut self, output_id: &str, occurrence: u64, text: String) -> bool {
        let Some(d) = self.delegations.get_mut(output_id).filter(|d| !d.settled) else {
            return false;
        };
        if occurrence <= d.progress_seen {
            return false;
        }
        d.progress_seen = occurrence;
        if text.trim().is_empty() {
            return false;
        }
        let queued = self.queue.iter_mut().rfind(|q| q.output_id == output_id);
        match queued {
            Some(Queued { event: ExpertInboxEvent::Progress(old), .. }) => *old = text,
            Some(_) | None => self.queue.push_back(Queued {
                output_id: output_id.into(),
                event: ExpertInboxEvent::Progress(text),
            }),
        }
        true
    }

    /// Exact output keeps its place even when later progress or a result supersedes ordinary
    /// announcements. Its native call is only a live delivery receipt, never business control.
    pub fn exact(
        &mut self,
        output_id: &str,
        occurrence: u64,
        text: String,
        call: Option<crate::agent_session::SpeechCall>,
    ) -> bool {
        let Some(d) = self.delegations.get_mut(output_id).filter(|d| !d.settled) else {
            return false;
        };
        if occurrence <= d.progress_seen {
            return false;
        }
        d.progress_seen = occurrence;
        self.queue.push_back(Queued {
            output_id: output_id.into(),
            event: ExpertInboxEvent::Exact { text, call },
        });
        true
    }

    /// The expert asks. Its own earlier request and progress are superseded: only the latest
    /// question of a delegation can be answered. A repeated request is not asked twice.
    pub fn ask(
        &mut self,
        output_id: &str,
        request_id: &str,
        question: String,
        verbatim: bool,
        input_request: Option<InputRequestRef>,
    ) -> bool {
        let Some(d) = self.delegations.get_mut(output_id).filter(|d| !d.settled) else {
            return false;
        };
        if d.requests.contains_key(request_id) {
            return false;
        }
        d.requests.insert(request_id.to_owned(), input_request);
        d.waiting = Some(request_id.to_owned());
        self.withdraw(output_id);
        let event =
            ExpertInboxEvent::Question { request_id: request_id.into(), question, verbatim };
        self.queue.push_back(Queued { output_id: output_id.into(), event });
        true
    }

    /// The delegation ended with `outcome`, which resolves its originating call once presented;
    /// never twice.
    pub fn outcome(&mut self, output_id: &str, outcome: ToolOutcome) -> bool {
        let Some(d) = self.delegations.get_mut(output_id).filter(|d| !d.settled) else {
            return false;
        };
        d.settled = true;
        d.waiting = None;
        self.withdraw(output_id);
        self.queue.push_back(Queued {
            output_id: output_id.into(),
            event: ExpertInboxEvent::Outcome(outcome),
        });
        true
    }

    /// The application settled or cancelled the interaction: nothing of it is presented, and its
    /// pending request can no longer be answered.
    pub fn cancel(&mut self, output_id: &str) -> bool {
        let Some(d) = self.delegations.get_mut(output_id).filter(|d| !d.settled) else {
            return false;
        };
        d.settled = true;
        d.waiting = None;
        self.withdraw(output_id);
        self.queue.retain(|q| {
            q.output_id != output_id || matches!(q.event, ExpertInboxEvent::Exact { .. })
        });
        true
    }

    /// Drop the call's queued progress and its question, queued or in the foreground.
    fn withdraw(&mut self, output_id: &str) {
        if let Some(delegation) = self.delegations.get(output_id) {
            let current = delegation
                .waiting
                .as_ref()
                .and_then(|id| delegation.requests.get(id))
                .and_then(Option::as_ref);
            self.responses.retain(|reference, _| {
                !delegation.requests.values().any(|known| known.as_ref() == Some(reference))
                    || Some(reference) == current
            });
            self.prompted_responses.retain(|reference| self.responses.contains_key(reference));
        }
        self.foreground.take_if(|f| f.output_id == output_id);
        self.queue.retain(|q| {
            q.output_id != output_id
                || matches!(q.event, ExpertInboxEvent::Outcome(_) | ExpertInboxEvent::Exact { .. })
        });
    }

    /// Suspend a presented question the active agent can no longer interact with.
    pub fn observe(&mut self, floor: Floor<'_>) {
        let Some(fg) = &mut self.foreground else {
            return;
        };
        let tool = &self.delegations[&fg.output_id].tool;
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
            && floor.reaches(&self.delegations[&fg.output_id].tool)
        {
            fg.suspended = false;
            return Some(Presentation::Question {
                output_id: fg.output_id.clone(),
                request_id: fg.request_id.clone(),
                question: fg.question.clone(),
                verbatim: fg.verbatim,
            });
        }
        let at = self.queue.iter().position(|q| {
            let d = &self.delegations[&q.output_id];
            floor.reaches(&d.tool)
                && match q.event {
                    ExpertInboxEvent::Question { .. } => self.foreground.is_none(),
                    ExpertInboxEvent::Outcome(_) | ExpertInboxEvent::Exact { .. } => d.recorded,
                    ExpertInboxEvent::Progress(_) => true,
                }
        })?;
        let Queued { output_id, event } = self.queue.remove(at)?;
        Some(match event {
            ExpertInboxEvent::Progress(text) => Presentation::Progress { output_id, text },
            ExpertInboxEvent::Exact { text, call } => Presentation::Exact { output_id, text, call },
            ExpertInboxEvent::Question { request_id, question, verbatim } => {
                self.foreground = Some(Foreground {
                    output_id: output_id.clone(),
                    request_id: request_id.clone(),
                    question: question.clone(),
                    verbatim,
                    suspended: false,
                });
                Presentation::Question { output_id, request_id, question, verbatim }
            }
            ExpertInboxEvent::Outcome(outcome) => Presentation::Outcome { output_id, outcome },
        })
    }

    /// Scheduling refused this exact foreground question before any presentation began.
    /// Keep its source and native reference; another eligible delivery can retry it later.
    pub(crate) fn defer_question(&mut self, request: &PendingRequest) {
        if let Some(foreground) = self.foreground.as_mut().filter(|foreground| {
            foreground.output_id == request.output_id && foreground.request_id == request.request_id
        }) {
            foreground.suspended = true;
        }
    }

    /// Collection: the request a person's reply answers now, if any. Nothing is forwarded while
    /// the question is suspended or a task owns the interaction.
    pub fn collect(&mut self, floor: Floor<'_>) -> Option<PendingRequest> {
        self.observe(floor);
        let fg = self.foreground.as_ref().filter(|f| !f.suspended)?;
        Some(PendingRequest { output_id: fg.output_id.clone(), request_id: fg.request_id.clone() })
    }

    /// Delivery to exactly this live continuation, which it consumes. It need not be the
    /// foreground prompt: a native reservation can answer an earlier presented question.
    pub fn answer(&mut self, request: &PendingRequest) -> Result<(), Stale> {
        let delegation = self
            .delegations
            .get_mut(&request.output_id)
            .filter(|delegation| !delegation.settled)
            .ok_or(Stale)?;
        if delegation.waiting.as_deref() != Some(request.request_id.as_str()) {
            return Err(Stale);
        }
        delegation.waiting = None;
        if let Some(Some(reference)) = delegation.requests.get(&request.request_id) {
            self.responses.remove(reference);
            self.prompted_responses.remove(reference);
        }
        self.queue.retain(|queued| {
            queued.output_id != request.output_id
                || !matches!(&queued.event, ExpertInboxEvent::Question { request_id, .. }
                    if request_id == &request.request_id)
        });
        self.foreground
            .take_if(|f| f.output_id == request.output_id && f.request_id == request.request_id);
        Ok(())
    }

    /// Registration supplies the native local reference before its presentation can be admitted.
    pub fn reference_retained(
        &mut self,
        output_id: &str,
        request_id: &str,
        reference: InputRequestRef,
    ) {
        if let Some(request) = self
            .delegations
            .get_mut(output_id)
            .and_then(|delegation| delegation.requests.get_mut(request_id))
        {
            *request = Some(reference);
        }
    }

    pub fn waits_for(&self, reference: &InputRequestRef) -> bool {
        self.delegations.values().any(|delegation| {
            !delegation.settled
                && delegation
                    .waiting
                    .as_ref()
                    .and_then(|request| delegation.requests.get(request))
                    .and_then(Option::as_ref)
                    == Some(reference)
        })
    }

    pub fn accepts(&self, reference: &InputRequestRef, floor: Floor<'_>) -> bool {
        self.delegations.values().any(|delegation| {
            !delegation.settled
                && floor.reaches(&delegation.tool)
                && delegation
                    .waiting
                    .as_ref()
                    .and_then(|request| delegation.requests.get(request))
                    .and_then(Option::as_ref)
                    == Some(reference)
        })
    }

    pub fn reference(&self, request: &PendingRequest) -> Option<&InputRequestRef> {
        self.delegations.get(&request.output_id)?.requests.get(&request.request_id)?.as_ref()
    }

    pub fn reserve(&mut self, reference: InputRequestRef, message: lk_agents_abi::ChatMessage) {
        if self.delegations.values().any(|delegation| {
            !delegation.settled
                && delegation
                    .waiting
                    .as_ref()
                    .and_then(|id| delegation.requests.get(id))
                    .and_then(Option::as_ref)
                    == Some(&reference)
        }) {
            self.prompted_responses.remove(&reference);
            self.responses.insert(reference, message);
        }
    }

    /// A checkpointed outgoing operation owns this exact response. It is no longer a
    /// promptable question; this says nothing about peer acceptance or output delivery.
    pub fn response_queued(&mut self, reference: &InputRequestRef) -> bool {
        let waiting: Vec<_> = self
            .delegations
            .iter()
            .filter_map(|(output_id, delegation)| {
                let request_id = delegation.waiting.as_ref()?;
                (!delegation.settled
                    && delegation.requests.get(request_id).and_then(Option::as_ref)
                        == Some(reference))
                .then(|| PendingRequest {
                    output_id: output_id.clone(),
                    request_id: request_id.clone(),
                })
            })
            .collect();
        let consumed = !waiting.is_empty();
        for request in waiting {
            self.answer(&request)
                .expect("the exact waiting occurrence was checked in this transition");
        }
        self.responses.remove(reference);
        self.prompted_responses.remove(reference);
        consumed
    }

    /// An eligible agent gets one normal planning turn for responses retained while it was away.
    pub fn prompt_responses(&mut self, floor: Floor<'_>) -> bool {
        if !floor.free {
            return false;
        }
        let mut prompt = false;
        for (request, _) in self.responses(floor) {
            if let Some(reference) = self.reference(&request).cloned() {
                prompt |= self.prompted_responses.insert(reference);
            }
        }
        prompt
    }

    /// Recognition is independent of the speaking agent; delivery still requires its tool.
    pub fn responses(&self, floor: Floor<'_>) -> Vec<(PendingRequest, lk_agents_abi::ChatMessage)> {
        self.delegations
            .iter()
            .filter_map(|(output_id, delegation)| {
                if delegation.settled || !floor.reaches(&delegation.tool) {
                    return None;
                }
                let request_id = delegation.waiting.as_ref()?;
                let reference = delegation.requests.get(request_id)?.as_ref()?;
                reference.remote()?;
                let message = self.responses.get(reference)?.clone();
                Some((
                    PendingRequest { output_id: output_id.clone(), request_id: request_id.clone() },
                    message,
                ))
            })
            .collect()
    }

    /// Speculation cannot authorize a response to an existing remote input occurrence.
    pub fn needs_input_routing(&self) -> bool {
        self.delegations.values().any(|delegation| {
            !delegation.settled
                && delegation
                    .waiting
                    .as_ref()
                    .and_then(|id| delegation.requests.get(id))
                    .is_some_and(Option::is_some)
        })
    }

    /// A scheduled question has no delivery authority until that speech reports actual words.
    pub fn presenting(&mut self, speech: SpeechId, request: PendingRequest) {
        self.presentations.insert(speech, request);
    }

    pub fn delivered(&mut self, speech: SpeechId) -> Option<InputRequestRef> {
        let request = self.presentations.remove(&speech)?;
        // A late receipt belongs to the words actually delivered, even if a newer input has
        // superseded that occurrence. Shared input state decides response eligibility.
        self.delegations.get(&request.output_id)?.requests.get(&request.request_id)?.clone()
    }

    /// An unspoken scheduled question is offered again when the floor permits.
    pub fn presentation_ended(&mut self, speech: SpeechId) {
        if let Some(request) = self.presentations.remove(&speech)
            && let Some(foreground) = self.foreground.as_mut()
            && foreground.output_id == request.output_id
            && foreground.request_id == request.request_id
        {
            foreground.suspended = true;
        }
    }
}

#[cfg(test)]
#[path = "experts_tests.rs"]
mod tests;
