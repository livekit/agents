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

/// A delegation's latest progress; older progress was superseded.
#[derive(Debug, Clone, PartialEq)]
pub struct Update {
    pub output_id: String,
    pub text: String,
}

/// A delegation's live question. Native response admission supplies its answers; presentation
/// alone never selects a remote reply destination.
#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub output_id: String,
    pub request_id: String,
    pub question: String,
    pub verbatim: bool,
}

/// A delegation's result, resolving its originating call.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolution {
    pub output_id: String,
    pub outcome: ToolOutcome,
}

/// The most questions the person hears before their next turn: five waiting questions are not
/// one five-part utterance.
const MAX_ASKED_TOGETHER: usize = 2;

/// One presentation opportunity, chosen when the floor permits.
#[derive(Debug, Clone, PartialEq)]
pub enum Presentation {
    /// Every reachable delegation's latest progress, summarized together.
    Progress(Vec<Update>),
    /// Exact text is never coalesced, rephrased or withdrawn by a later question or result.
    Exact { output_id: String, text: String, call: Option<crate::agent_session::SpeechCall> },
    /// Questions asked together; each remains its own request. A verbatim question is asked
    /// alone, word for word.
    Questions(Vec<Question>),
    /// Results ready together, presented in one reply; each resolves its own originating call.
    /// A verbatim result is presented alone, like exact output.
    Outcomes(Vec<Resolution>),
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
        /// Back in line after the person passed it, at this prompt count: it is not asked
        /// again until another prompt was asked, and only this once.
        #[serde(default)]
        passed: Option<u64>,
    },
    Outcome(ToolOutcome),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Queued {
    output_id: String,
    event: ExpertInboxEvent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Asked {
    request_id: String,
    question: String,
    verbatim: bool,
    /// Unheard, or the active agent lost this tool or a task took the interaction. Re-offering
    /// the prompt does not invalidate its earlier native delivery or a recognized response.
    suspended: bool,
    /// The person's turn count when it was asked: until they speak, the next prompt waits.
    #[serde(default)]
    turn: u64,
    /// The person took a turn without answering it. It stays open for a later answer.
    #[serde(default)]
    passed: bool,
    /// Asked again after a pass: passing it again leaves it open without asking a third time.
    #[serde(default)]
    repeated: bool,
}

/// The queue's records: serializable delegation references, queued events in order, and the
/// asked questions. Saved under the frontend conversation's identity when the application
/// persists frontend delivery; never a live await.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Experts {
    /// By the originating call's canonical output item ID, never its provider call ID.
    delegations: BTreeMap<String, Delegation>,
    queue: VecDeque<Queued>,
    /// Questions presented and still waiting, by delegation: several can be open at once.
    asked: BTreeMap<String, Asked>,
    /// The person's turns so far.
    #[serde(default)]
    turns: u64,
    /// Question prompts presented so far.
    #[serde(default)]
    prompts: u64,
    /// Progress announcements scheduled and not ended, by speech.
    #[serde(skip)]
    announcing: BTreeMap<SpeechId, Vec<Update>>,
    /// Delegations answered and not heard from since: other tasks' questions wait for their
    /// next event, and a follow-up question of theirs goes ahead, once per answer.
    #[serde(skip)]
    answered: BTreeSet<String>,
    /// The follow-up queued ahead, and the delegation whose follow-up led the last prompt: it
    /// does not get a second head start in a row.
    #[serde(skip)]
    follow_up: (Option<String>, Option<String>),
    /// Live output receipts have no meaning after a call ends; communication owns recovery.
    #[serde(skip)]
    presentations: BTreeMap<SpeechId, Vec<PendingRequest>>,
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

    /// The remote task that asked this question, as its native occurrence names it.
    pub fn asking_task(&self, output_id: &str, request_id: &str) -> Option<&str> {
        let reference = self.delegations.get(output_id)?.requests.get(request_id)?.as_ref()?;
        reference.remote().map(|(_, task, _)| task.task_id.as_str())
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
        self.answered.remove(output_id);
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
        self.answered.remove(output_id);
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
        let event = ExpertInboxEvent::Question {
            request_id: request_id.into(),
            question,
            verbatim,
            passed: None,
        };
        let queued = Queued { output_id: output_id.into(), event };
        let (ahead, led) = &mut self.follow_up;
        if self.answered.remove(output_id) && led.as_deref() != Some(output_id) {
            *ahead = Some(output_id.into());
            self.queue.push_front(queued);
        } else {
            self.queue.push_back(queued);
        }
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
        self.answered.remove(output_id);
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
        self.answered.remove(output_id);
        self.withdraw(output_id);
        self.queue.retain(|q| {
            q.output_id != output_id || matches!(q.event, ExpertInboxEvent::Exact { .. })
        });
        true
    }

    /// Drop the call's queued progress and its question, queued or asked.
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
        self.asked.remove(output_id);
        self.queue.retain(|q| {
            q.output_id != output_id
                || matches!(q.event, ExpertInboxEvent::Outcome(_) | ExpertInboxEvent::Exact { .. })
        });
    }

    /// Suspend asked questions the active agent can no longer interact with.
    pub fn observe(&mut self, floor: Floor<'_>) {
        for (output_id, asked) in &mut self.asked {
            if !floor.reaches(&self.delegations[output_id].tool) {
                asked.suspended = true;
            }
        }
    }

    /// The person took a turn: whatever was asked before it has had its chance.
    pub fn person_turn(&mut self) {
        self.turns += 1;
        self.answered.clear();
    }

    /// The next presentation among events the active agent's tools reach.
    ///
    /// Pacing: a prompt asks the line's first question with the other waiting questions of the
    /// same expert (related work), at most `MAX_ASKED_TOGETHER`; other experts' questions wait
    /// for later prompts. After a prompt is heard, until the person's next turn only a late
    /// question of an expert already asking may follow, up to that same limit; other questions
    /// and routine progress wait, results and exact output do not. Questions come first when a
    /// suspended one is reachable again or one leads the queue.
    ///
    /// A prompt only asks: progress and results are always presentations of their own.
    ///
    /// Fairness: waiting questions are asked in arrival order, so a newer task never overtakes
    /// an older one, except a bounded follow-up: until a task the person just answered is heard
    /// from again (or the person speaks again), other questions wait, and if it asks a
    /// follow-up that goes first, once per answer and never twice in a row. A question the person passed (a turn without answering it) stays open and
    /// goes back in line behind the questions already waiting; it is asked once more only after
    /// another prompt, never in the prompt right after it was passed.
    ///
    /// Otherwise the next event in arrival order: progress takes every reachable delegation's
    /// latest update with it, and a result takes every other result ready before the next
    /// ordered output.
    pub fn next(&mut self, floor: Floor<'_>) -> Option<Presentation> {
        self.observe(floor);
        if !floor.free {
            return None;
        }
        self.pass();
        let awaiting = self.awaiting_person();
        let reaches = |output_id: &str| floor.reaches(&self.delegations[output_id].tool);
        let eligible = |q: &Queued| {
            reaches(&q.output_id)
                && match &q.event {
                    ExpertInboxEvent::Outcome(_) | ExpertInboxEvent::Exact { .. } => {
                        self.delegations[&q.output_id].recorded
                    }
                    ExpertInboxEvent::Question { passed, .. } => {
                        self.due(*passed)
                            && Self::room(&awaiting, self.expert(&q.output_id)) > 0
                            && self.follows_up(&q.output_id)
                    }
                    ExpertInboxEvent::Progress(_) => awaiting.is_empty(),
                }
        };
        let reoffer = awaiting.is_empty()
            && self.answered.is_empty()
            && self.asked.iter().any(|(id, asked)| asked.suspended && reaches(id));
        let at = self.queue.iter().position(eligible);
        let leads =
            at.is_some_and(|at| matches!(self.queue[at].event, ExpertInboxEvent::Question { .. }));
        if reoffer || leads {
            return Some(Presentation::Questions(self.take_questions(floor, &awaiting)));
        }
        let at = at?;
        if matches!(&self.queue[at].event,
            ExpertInboxEvent::Outcome(outcome) if !Self::exact_outcome(outcome))
        {
            return Some(Presentation::Outcomes(self.take_outcomes(floor)));
        }
        let Queued { output_id, event } = self.queue.remove(at)?;
        Some(match event {
            ExpertInboxEvent::Progress(text) => {
                let mut updates = vec![Update { output_id, text }];
                updates.extend(self.take_progress(floor));
                Presentation::Progress(updates)
            }
            ExpertInboxEvent::Exact { text, call } => Presentation::Exact { output_id, text, call },
            ExpertInboxEvent::Question { .. } => unreachable!("a leading question is taken above"),
            ExpertInboxEvent::Outcome(outcome) => {
                Presentation::Outcomes(vec![Resolution { output_id, outcome }])
            }
        })
    }

    /// Heard questions the person has not had a turn to answer yet, counted by expert.
    fn awaiting_person(&self) -> BTreeMap<ExpertKey, usize> {
        let mut awaiting = BTreeMap::new();
        for (output_id, asked) in &self.asked {
            if !asked.suspended && asked.turn == self.turns {
                *awaiting.entry(self.expert(output_id).clone()).or_default() += 1;
            }
        }
        awaiting
    }

    fn expert(&self, output_id: &str) -> &ExpertKey {
        &self.delegations[output_id].tool.key
    }

    /// How many more of `expert`'s questions the person can hear before their next turn.
    fn room(awaiting: &BTreeMap<ExpertKey, usize>, expert: &ExpertKey) -> usize {
        if awaiting.is_empty() {
            return MAX_ASKED_TOGETHER;
        }
        if !awaiting.contains_key(expert) {
            return 0;
        }
        MAX_ASKED_TOGETHER.saturating_sub(awaiting.values().sum())
    }

    /// Until each task the person just answered was heard from again, only its own follow-up
    /// can be asked: the follow-up is not overtaken by a question that merely arrived first.
    fn follows_up(&self, output_id: &str) -> bool {
        self.answered.is_empty() || self.follow_up.0.as_deref() == Some(output_id)
    }

    /// A passed question is due again once another prompt was asked after its pass.
    fn due(&self, passed: Option<u64>) -> bool {
        passed.is_none_or(|at| self.prompts > at)
    }

    /// A verbatim result keeps its own delivery, like exact output.
    fn exact_outcome(outcome: &ToolOutcome) -> bool {
        lk_agents_core::tool_outcome::verbatim(outcome).is_some()
    }

    /// Heard questions the person's turn left unanswered release their place: each stays open
    /// and, the first time, goes back in line behind the questions already waiting.
    fn pass(&mut self) {
        for (output_id, asked) in &mut self.asked {
            if asked.suspended || asked.passed || asked.turn == self.turns {
                continue;
            }
            asked.passed = true;
            if asked.repeated {
                continue;
            }
            let event = ExpertInboxEvent::Question {
                request_id: asked.request_id.clone(),
                question: asked.question.clone(),
                verbatim: asked.verbatim,
                passed: Some(self.prompts),
            };
            self.queue.push_back(Queued { output_id: output_id.clone(), event });
        }
    }

    /// One prompt: reachable suspended questions, then due queued ones in arrival order. The
    /// first leads; the others of its expert join it. A verbatim question is asked alone.
    fn take_questions(
        &mut self,
        floor: Floor<'_>,
        awaiting: &BTreeMap<ExpertKey, usize>,
    ) -> Vec<Question> {
        let reaches = |output_id: &str| floor.reaches(&self.delegations[output_id].tool);
        let suspended =
            self.asked.iter().filter(|(id, a)| a.suspended && reaches(id)).map(|(output_id, a)| {
                let question = Question {
                    output_id: output_id.clone(),
                    request_id: a.request_id.clone(),
                    question: a.question.clone(),
                    verbatim: a.verbatim,
                };
                (question, a.repeated)
            });
        let queued = self.queue.iter().filter(|q| reaches(&q.output_id)).filter_map(|q| {
            let ExpertInboxEvent::Question { request_id, question, verbatim, passed } = &q.event
            else {
                return None;
            };
            let question = Question {
                output_id: q.output_id.clone(),
                request_id: request_id.clone(),
                question: question.clone(),
                verbatim: *verbatim,
            };
            self.due(*passed).then_some((question, passed.is_some()))
        });
        let mut sources = BTreeSet::new();
        let candidates: Vec<_> = suspended
            .chain(queued)
            .filter(|(q, _)| self.follows_up(&q.output_id))
            .filter(|(q, _)| awaiting.is_empty() || !q.verbatim)
            .filter(|(q, _)| Self::room(awaiting, self.expert(&q.output_id)) > 0)
            .filter(|(q, _)| sources.insert(q.output_id.clone()))
            .collect();
        let Some((first, _)) = candidates.first() else {
            return Vec::new();
        };
        let leader = first.output_id.clone();
        let expert = self.expert(&leader);
        let chosen: Vec<_> = if first.verbatim {
            candidates.into_iter().take(1).collect()
        } else {
            let room = Self::room(awaiting, expert);
            candidates
                .into_iter()
                .filter(|(q, _)| !q.verbatim && self.expert(&q.output_id) == expert)
                .take(room)
                .collect()
        };
        self.prompts += 1;
        // A follow-up's head start lasts until the next prompt.
        self.answered.clear();
        let ahead = self.follow_up.0.take();
        self.follow_up.1 = ahead.filter(|id| *id == leader);
        let mut questions = Vec::with_capacity(chosen.len());
        for (q, repeated) in chosen {
            self.queue.retain(|queued| {
                queued.output_id != q.output_id
                    || !matches!(queued.event, ExpertInboxEvent::Question { .. })
            });
            let asked = Asked {
                request_id: q.request_id.clone(),
                question: q.question.clone(),
                verbatim: q.verbatim,
                suspended: false,
                turn: self.turns,
                passed: false,
                repeated,
            };
            self.asked.insert(q.output_id.clone(), asked);
            questions.push(q);
        }
        questions
    }

    /// Every presentable result up to the next ordered output (exact text or a verbatim
    /// result), which keeps its place.
    fn take_outcomes(&mut self, floor: Floor<'_>) -> Vec<Resolution> {
        let mut taken = Vec::new();
        let mut ordered = false;
        let mut rest = VecDeque::new();
        for queued in self.queue.drain(..) {
            let ready = !ordered
                && floor.reaches(&self.delegations[&queued.output_id].tool)
                && self.delegations[&queued.output_id].recorded;
            match queued.event {
                ExpertInboxEvent::Outcome(outcome) if ready && !Self::exact_outcome(&outcome) => {
                    taken.push(Resolution { output_id: queued.output_id, outcome });
                }
                ExpertInboxEvent::Outcome(_) | ExpertInboxEvent::Exact { .. } if ready => {
                    ordered = true;
                    rest.push_back(queued);
                }
                _ => rest.push_back(queued),
            }
        }
        self.queue = rest;
        taken
    }

    /// Queued progress of other reachable delegations, unless it waits behind that
    /// delegation's own exact output, question or result.
    fn take_progress(&mut self, floor: Floor<'_>) -> Vec<Update> {
        let mut behind = BTreeSet::new();
        let mut updates = Vec::new();
        let mut rest = VecDeque::new();
        for queued in self.queue.drain(..) {
            match queued.event {
                ExpertInboxEvent::Progress(text)
                    if !behind.contains(&queued.output_id)
                        && floor.reaches(&self.delegations[&queued.output_id].tool) =>
                {
                    updates.push(Update { output_id: queued.output_id, text });
                }
                _ => {
                    behind.insert(queued.output_id.clone());
                    rest.push_back(queued);
                }
            }
        }
        self.queue = rest;
        updates
    }

    /// A progress announcement was scheduled as `speech`.
    pub fn announcing(&mut self, speech: SpeechId, updates: Vec<Update>) {
        self.announcing.insert(speech, updates);
    }

    /// Scheduled announcements that carry progress of `output_id`, which a newer question or
    /// result of that delegation has made stale.
    pub fn overtaken(&self, output_id: &str) -> Vec<SpeechId> {
        self.announcing
            .iter()
            .filter(|(_, updates)| updates.iter().any(|u| u.output_id == output_id))
            .map(|(speech, _)| *speech)
            .collect()
    }

    /// A stale announcement is withdrawn before it was heard. Its other delegations' updates
    /// go back in line unless something newer of theirs is queued or they no longer run.
    pub fn withdraw_announcement(&mut self, speech: SpeechId) {
        for update in self.announcing.remove(&speech).unwrap_or_default() {
            let live = self
                .delegations
                .get(&update.output_id)
                .is_some_and(|d| !d.settled && d.waiting.is_none());
            let newer = self.queue.iter().any(|q| q.output_id == update.output_id);
            if live && !newer {
                let event = ExpertInboxEvent::Progress(update.text);
                self.queue.push_back(Queued { output_id: update.output_id, event });
            }
        }
    }

    /// Scheduling refused this exact asked question before any presentation began.
    /// Keep its source and native reference; another eligible delivery can retry it later.
    pub(crate) fn defer_question(&mut self, request: &PendingRequest) {
        if let Some(asked) = self
            .asked
            .get_mut(&request.output_id)
            .filter(|asked| asked.request_id == request.request_id)
        {
            asked.suspended = true;
        }
    }

    /// Collection: the requests a person's reply may answer now. Nothing is forwarded to a
    /// suspended question or while a task owns the interaction.
    pub fn collect(&mut self, floor: Floor<'_>) -> Vec<PendingRequest> {
        self.observe(floor);
        self.asked
            .iter()
            .filter(|(_, asked)| !asked.suspended)
            .map(|(output_id, asked)| PendingRequest {
                output_id: output_id.clone(),
                request_id: asked.request_id.clone(),
            })
            .collect()
    }

    /// Delivery to exactly this live continuation, which it consumes. It need not be the
    /// latest asked prompt: a native reservation can answer an earlier presented question.
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
        self.asked.remove(&request.output_id);
        self.answered.insert(request.output_id.clone());
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

    /// Scheduled questions have no delivery authority until that speech reports actual words.
    pub fn presenting(&mut self, speech: SpeechId, requests: Vec<PendingRequest>) {
        self.presentations.insert(speech, requests);
    }

    pub fn delivered(&mut self, speech: SpeechId) -> Vec<InputRequestRef> {
        let requests = self.presentations.remove(&speech).unwrap_or_default();
        // A late receipt belongs to the words actually delivered, even if a newer input has
        // superseded that occurrence. Shared input state decides response eligibility.
        requests
            .iter()
            .filter_map(|request| {
                self.delegations.get(&request.output_id)?.requests.get(&request.request_id)?.clone()
            })
            .collect()
    }

    /// Unspoken scheduled questions are offered again when the floor permits.
    pub fn presentation_ended(&mut self, speech: SpeechId) {
        self.announcing.remove(&speech);
        for request in self.presentations.remove(&speech).unwrap_or_default() {
            self.defer_question(&request);
        }
    }
}

#[cfg(test)]
#[path = "experts_tests.rs"]
mod tests;
