//! The AMD reply gate as a run's observations show it: which runs attached, which turns committed
//! under a gate, what the scripted answerer answered, and which model requests and audio answered
//! each turn. The AMD rules read only this view and the public recording.
//!
//! A turn is gated when the host heard it committed for a reply (`UserTurnCompleted` after an
//! `EndOfTurn` that replies) while a run was attached: from the accepted `AmdAttach` to the
//! session's `Detached` for that run, or to the end of the run when the session never sent one.
//! That window does not depend on the session's own account of its gate, so a session that
//! ignores the gate is still judged.

use lk_agents_abi::Role;
use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechKind;
use lk_agents_voice::amd::{AmdObservation, AmdRunId};
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::trace::{Step, TurnStep};

use super::{Detail, Kind, Observed, Recording};
use crate::input::{AmdOpening, AmdVerdict, Input};

/// One message of a model request, as the fake model received it.
#[derive(Clone, Debug, PartialEq)]
pub struct RequestMessage {
    pub id: String,
    pub role: Role,
    pub text: String,
}

/// One observation, stamped, and the run it was for.
pub type Seen = (Observed, AmdRunId, AmdObservation);

impl Recording {
    pub fn record_amd(&mut self, at: Observed, run: AmdRunId, observation: AmdObservation) {
        self.amd.push((at, run, observation));
    }

    /// What the attached AMD runs observed, in order.
    pub fn amd(&self) -> &[Seen] {
        &self.amd
    }
}

/// One AMD run, from its attach.
#[derive(Clone, Debug, PartialEq)]
pub struct Attachment {
    pub run: u64,
    pub attached: Observed,
    /// The session's `Detached` for this run.
    pub detached: Option<Observed>,
    /// The first `AmdDetach` the session accepted for this run.
    pub detach_asked: Option<Observed>,
    /// The first accepted `AmdActivate`, and its opening.
    pub activated: Option<(Observed, Option<AmdOpening>)>,
}

impl Attachment {
    /// Until the session detached, or the end of the run.
    pub fn until(&self) -> Observed {
        self.detached.unwrap_or(Observed::END)
    }
}

/// A turn committed for a reply while a run was attached.
#[derive(Clone, Debug, PartialEq)]
pub struct GatedTurn {
    pub run: u64,
    pub turn: u64,
    pub words: String,
    pub committed: Observed,
    /// The user item the session saved for it, if it saved one.
    pub item_id: Option<String>,
    /// The next turn committed under the same run, which supersedes this one.
    pub superseded: Option<Observed>,
}

/// One answer the scripted AMD sent and the session accepted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Answer {
    Decide { at: Observed, run: u64, turn: u64, verdict: AmdVerdict },
    Detach { at: Observed, run: u64, category: AmdVerdict, forbid: bool },
}

impl Answer {
    pub fn at(&self) -> Observed {
        match self {
            Answer::Decide { at, .. } | Answer::Detach { at, .. } => *at,
        }
    }

    /// Whether this answer lets the turn reply, as `fakes::amd` builds the decision.
    pub fn allows(&self) -> bool {
        match self {
            Answer::Decide { verdict, .. } => {
                !matches!(verdict, AmdVerdict::Deny | AmdVerdict::Wait)
            }
            Answer::Detach { category, forbid, .. } => !forbid && *category != AmdVerdict::Wait,
        }
    }
}

/// One model request, and the speech it was made for.
#[derive(Clone, Debug)]
pub struct Request<'a> {
    pub chat: u32,
    pub opened: Observed,
    pub speech: Option<u64>,
    pub messages: &'a [RequestMessage],
    pub tools: &'a [String],
}

impl Request<'_> {
    /// The last user message: the words the request answers.
    pub fn last_user(&self) -> Option<&RequestMessage> {
        self.messages.iter().rev().find(|m| m.role == Role::User)
    }

    pub fn carries(&self, text: &str) -> bool {
        self.messages.iter().any(|m| m.text.contains(text))
    }
}

/// One response request the session sent a realtime socket (`GenerateReply`).
#[derive(Clone, Debug, PartialEq)]
pub struct RtResponse {
    pub socket: u32,
    pub at: Observed,
    pub req_id: String,
    /// The per-response instructions it carried.
    pub instructions: Option<String>,
    /// The session's tools on the socket when it was sent: the last `UpdateTools` before it, or
    /// `None` when none was sent.
    pub tools: Option<Vec<String>>,
}

impl RtResponse {
    pub fn offers(&self, tool: &str) -> bool {
        self.tools.as_ref().is_some_and(|tools| tools.iter().any(|t| t == tool))
    }
}

pub struct AmdView<'a>(&'a Recording);

impl Recording {
    pub fn amd_view(&self) -> AmdView<'_> {
        AmdView(self)
    }
}

impl<'a> AmdView<'a> {
    /// Each run, numbered as the scripted answerer numbers them: every `AmdAttach` it sent.
    pub fn attachments(&self) -> Vec<Attachment> {
        let mut out: Vec<Attachment> = Vec::new();
        let mut run = 0;
        for applied in self.0.applied() {
            match &applied.input {
                Input::AmdAttach {} => {
                    run += 1;
                    // A refused attach (a handoff in flight, a task running, a realtime model the
                    // gate cannot hold, the close) installs no gate (spec §5.1 entry checks).
                    if applied.accepted && !self.attach_refused(run) {
                        let detached = self.0.amd().iter().find_map(|(at, _, o)| match o {
                            AmdObservation::Detached { run: r } if r.0 == run => Some(*at),
                            _ => None,
                        });
                        out.push(Attachment {
                            run,
                            attached: applied.at,
                            detached,
                            detach_asked: None,
                            activated: None,
                        });
                    }
                }
                Input::AmdDetach { .. } if applied.accepted => {
                    if let Some(a) = out.iter_mut().rev().find(|a| a.run == run)
                        && a.detach_asked.is_none()
                    {
                        a.detach_asked = Some(applied.at);
                    }
                }
                Input::AmdActivate { opening } if applied.accepted => {
                    if let Some(a) = out.iter_mut().rev().find(|a| a.run == run)
                        && a.activated.is_none()
                    {
                        a.activated = Some((applied.at, *opening));
                    }
                }
                _ => {}
            }
        }
        out.extend(self.executor_attachments());
        out
    }

    /// The runs AMD's executor attached (`Input::AmdEnter`), from the gate's attach, with the
    /// first `Detach` and `Activated` the session took from each.
    fn executor_attachments(&self) -> Vec<Attachment> {
        use lk_agents_voice::amd::{AmdAnswer, Opening};
        use lk_agents_voice::trace::GateStep;
        let view = self.0.amd_run();
        let mut out = Vec::new();
        for (_, run) in view.entered() {
            // The gate's window opens where it attached: a run it never attached gates nothing.
            let attached = self.0.journal().iter().find_map(|(at, step)| {
                matches!(step, Step::Gate(GateStep::Attached { run: r }) if *r == run)
                    .then_some(*at)
            });
            let Some(attached) = attached else { continue };
            let answers = view.answers(run);
            let detach_asked = answers
                .iter()
                .find_map(|(at, a)| matches!(a, AmdAnswer::Detach { .. }).then_some(*at));
            let activated = answers.iter().find_map(|(at, a)| match a {
                AmdAnswer::Activated { opening, .. } => Some((
                    *at,
                    opening.as_ref().map(|o| match o {
                        Opening::Text(_) => AmdOpening::Text,
                        Opening::Instructions(_) => AmdOpening::Instructions,
                    }),
                )),
                _ => None,
            });
            let detached = self.0.amd().iter().find_map(|(at, _, o)| match o {
                AmdObservation::Detached { run: r } if *r == run => Some(*at),
                _ => None,
            });
            out.push(Attachment { run: run.0, attached, detached, detach_asked, activated });
        }
        out
    }

    /// The answers AMD's executor sent, as the scripted answerer's verdicts would have said them:
    /// a decision by what it asks of the reply, `Detach` by its category (`wait` when it denies
    /// and keeps replies) and whether it forbids replies.
    fn executor_answers(&self) -> Vec<Answer> {
        use lk_agents_voice::amd::{AmdAnswer, DetachCancel};
        let view = self.0.amd_run();
        let mut out = Vec::new();
        for (_, run) in view.entered() {
            for (at, answer) in view.answers(run) {
                match answer {
                    AmdAnswer::ReplyDecided { turn, decision, .. } => {
                        let staged = decision.instructions.is_some();
                        let silence = decision.silence.is_some_and(|d| !d.is_zero());
                        let verdict = match (decision.allow, staged, decision.dtmf, silence) {
                            (false, ..) => AmdVerdict::Deny,
                            (true, true, true, _) => AmdVerdict::Ivr,
                            (true, false, true, _) => AmdVerdict::HeldIvr,
                            (true, true, false, true) => AmdVerdict::Silence,
                            (true, true, false, false) => AmdVerdict::Stage,
                            (true, false, false, true) => AmdVerdict::Held,
                            (true, false, false, false) => AmdVerdict::Allow,
                        };
                        out.push(Answer::Decide { at, run: run.0, turn: turn.0, verdict });
                    }
                    AmdAnswer::Detach { pending, cancel, .. } => {
                        let forbid = cancel == DetachCancel::ForbidReplies;
                        let category = if !pending.allow && !forbid {
                            AmdVerdict::Wait
                        } else {
                            AmdVerdict::Allow
                        };
                        out.push(Answer::Detach { at, run: run.0, category, forbid });
                    }
                    AmdAnswer::Activated { .. } | AmdAnswer::InsertUserItem { .. } => {}
                }
            }
        }
        out
    }

    /// The decision an answer stands for: the executor's own, or the one the scripted answerer
    /// builds from its verdict (`fakes::amd`).
    pub fn decision(&self, answer: &Answer) -> lk_agents_voice::amd::ReplyDecision {
        use lk_agents_voice::amd::AmdAnswer;
        let view = self.0.amd_run();
        for (_, run) in view.entered() {
            for (at, sent) in view.answers(run) {
                if at != answer.at() {
                    continue;
                }
                match (sent, answer) {
                    (
                        AmdAnswer::ReplyDecided { turn, decision, .. },
                        Answer::Decide { turn: t, .. },
                    ) if turn.0 == *t => {
                        return decision;
                    }
                    (AmdAnswer::Detach { pending, .. }, Answer::Detach { .. }) => return pending,
                    _ => {}
                }
            }
        }
        match *answer {
            Answer::Decide { verdict, .. } => crate::fakes::amd::decision(verdict),
            Answer::Detach { category, forbid, .. } => crate::fakes::amd::pending(category, forbid),
        }
    }

    /// Every `InsertUserItem` the session took, with its run, turn and words.
    pub fn inserted_items(&self) -> Vec<(Observed, u64, u64, String)> {
        use lk_agents_voice::amd::AmdAnswer;
        let view = self.0.amd_run();
        let mut out = Vec::new();
        for (_, run) in view.entered() {
            for (at, answer) in view.answers(run) {
                if let AmdAnswer::InsertUserItem { run, turn, text } = answer {
                    out.push((at, run.0, turn.0, text));
                }
            }
        }
        out.sort_by_key(|(at, ..)| *at);
        out
    }

    /// Every response request the session sent a realtime socket, in order.
    pub fn rt_responses(&self) -> Vec<RtResponse> {
        use super::RtSent;
        let mut tools: std::collections::BTreeMap<u32, Vec<String>> = Default::default();
        let mut out = Vec::new();
        for (at, socket, sent) in self.0.rt_sent() {
            match sent {
                RtSent::Tools(set) => {
                    tools.insert(*socket, set.clone());
                }
                RtSent::Response { req_id, instructions } => out.push(RtResponse {
                    socket: *socket,
                    at: *at,
                    req_id: req_id.clone(),
                    instructions: instructions.clone(),
                    tools: tools.get(socket).cloned(),
                }),
                _ => {}
            }
        }
        out
    }

    /// When the session committed a realtime socket's input buffer, in order.
    pub fn rt_commits(&self) -> Vec<Observed> {
        (self.0.rt_sent().iter())
            .filter(|(_, _, sent)| *sent == super::RtSent::CommitAudio)
            .map(|(at, ..)| *at)
            .collect()
    }

    /// Where `turn`'s realtime reply may be asked for: from its commit to the next turn the host
    /// heard committed, a newer gated turn, or a host `say()` or `generate_reply()`.
    fn rt_window(&self, turn: &GatedTurn) -> Observed {
        let next_commit = self.0.host_view().events().into_iter().find_map(|(at, e)| match e {
            SessionEvent::UserTurnCompleted(done)
                if at > turn.committed && done.turn.0 != turn.turn =>
            {
                Some(at)
            }
            _ => None,
        });
        let host_speech = self.0.applied().iter().find_map(|a| {
            (a.accepted
                && a.at > turn.committed
                && matches!(a.input, Input::Say { .. } | Input::GenerateReply { .. }))
            .then_some(a.at)
        });
        [turn.superseded, next_commit, host_speech]
            .into_iter()
            .flatten()
            .min()
            .unwrap_or(Observed::END)
    }

    /// The response requests that answer `turn` on a realtime model: every one asked in its
    /// window, the opening's aside (a realtime response carries no turn of its own).
    pub fn rt_responses_answering(&self, turn: &GatedTurn) -> Vec<RtResponse> {
        let until = self.rt_window(turn);
        self.rt_responses()
            .into_iter()
            .filter(|q| q.at > turn.committed && q.at < until)
            .filter(|q| q.instructions.as_deref() != Some(crate::fakes::amd::OPENING_INSTRUCTIONS))
            .collect()
    }

    /// Whether the session ran on a realtime model: it sent a realtime socket anything.
    pub fn realtime(&self) -> bool {
        !self.0.rt_sent().is_empty()
    }

    /// Whether the session refused run `run`'s attach.
    fn attach_refused(&self, run: u64) -> bool {
        use lk_agents_voice::trace::GateStep;
        self.0.journal().iter().any(|(_, step)| {
            matches!(step, Step::Gate(GateStep::AttachRefused { run: r, .. }) if r.0 == run)
        })
    }

    /// The run attached at `at`, if any.
    pub fn attached_at(&self, at: Observed) -> Option<Attachment> {
        self.attachments().into_iter().rev().find(|a| a.attached < at && at < a.until())
    }

    /// Every turn committed for a reply under a gate, in order.
    pub fn gated_turns(&self) -> Vec<GatedTurn> {
        let host = self.0.host_view().events();
        let mut out: Vec<GatedTurn> = Vec::new();
        for (at, event) in &host {
            let SessionEvent::UserTurnCompleted(done) = event else { continue };
            if !self.replies(*at) {
                continue;
            }
            let Some(attachment) = self.attached_at(*at) else { continue };
            // The gate's own commit, which the host hears after: AMD may answer in between.
            let at = &self.gate_committed(attachment.run, done.turn.0, *at);
            let item_id = host.iter().find_map(|(_, e)| match e {
                SessionEvent::ConversationItemAdded {
                    item,
                    source: lk_agents_voice::agent::Source::UserTurn { turn },
                    ..
                } if *turn == done.turn => item.as_message().map(|m| m.id.clone()),
                _ => None,
            });
            if let Some(previous) = out.iter_mut().rev().find(|t| t.run == attachment.run)
                && previous.superseded.is_none()
            {
                previous.superseded = Some(*at);
            }
            out.push(GatedTurn {
                run: attachment.run,
                turn: done.turn.0,
                words: done.new_transcript.clone(),
                committed: *at,
                item_id,
                superseded: None,
            });
        }
        out
    }

    /// When the gate committed `turn` of `run`, which the host heard at `heard`: the session
    /// reports AMD the commit in its own transition, so AMD may answer before the host hears it.
    fn gate_committed(&self, run: u64, turn: u64, heard: Observed) -> Observed {
        use lk_agents_voice::trace::GateStep;
        (self.0.journal().iter())
            .filter(|(at, _)| *at <= heard)
            .rev()
            .find_map(|(at, step)| match step {
                Step::Gate(GateStep::Committed { run: r, turn: t })
                    if r.0 == run && t.0 == turn =>
                {
                    Some(*at)
                }
                _ => None,
            })
            .unwrap_or(heard)
    }

    /// Whether the end of turn the host heard at `at` asked for a reply: the verdict journalled
    /// just before it.
    pub fn replies(&self, at: Observed) -> bool {
        self.0
            .journal()
            .iter()
            .rev()
            .filter(|(seen, _)| *seen <= at)
            .find_map(|(_, step)| match step {
                Step::Turn(TurnStep::EndOfTurn { verdict, .. }) => Some(verdict.replies()),
                _ => None,
            })
            .unwrap_or(true)
    }

    /// Every answer the session accepted, in order, with the run and turn a decision named.
    pub fn answers(&self) -> Vec<Answer> {
        let mut run = 0;
        let mut out = Vec::new();
        for applied in self.0.applied() {
            match &applied.input {
                Input::AmdAttach {} => run += 1,
                Input::AmdDecide { turn, verdict } if applied.accepted => {
                    if let Some(Detail::AmdTurn { run, turn }) =
                        self.0.resource(Kind::AmdTurn, turn.nth).map(|r| r.detail().clone())
                    {
                        out.push(Answer::Decide { at: applied.at, run, turn, verdict: *verdict });
                    }
                }
                Input::AmdDecideAhead { verdict } if applied.accepted => {
                    let turn = self.turn_ahead(run, applied.at);
                    out.push(Answer::Decide { at: applied.at, run, turn, verdict: *verdict });
                }
                Input::AmdDetach { category, forbid } if applied.accepted => {
                    out.push(Answer::Detach {
                        at: applied.at,
                        run,
                        category: *category,
                        forbid: *forbid,
                    });
                }
                _ => {}
            }
        }
        out.extend(self.executor_answers());
        out.sort_by_key(Answer::at);
        out
    }

    /// The turn after the latest one `run` observed before `before`: the one
    /// [`Input::AmdDecideAhead`] names.
    pub fn turn_ahead(&self, run: u64, before: Observed) -> u64 {
        let latest = self
            .0
            .resources(Kind::AmdTurn)
            .filter(|t| t.opened_at() < before)
            .filter_map(|t| match t.detail() {
                Detail::AmdTurn { run: r, turn } if *r == run => Some(*turn),
                _ => None,
            })
            .max();
        latest.unwrap_or(0) + 1
    }

    /// The answers that could decide `turn`: a decision naming it, or its run's detach, after
    /// its commit and before a newer turn superseded it.
    pub fn answers_for(&self, turn: &GatedTurn) -> Vec<Answer> {
        let until = turn.superseded.unwrap_or(Observed::END);
        self.answers()
            .into_iter()
            .filter(|a| a.at() > turn.committed && a.at() < until)
            .filter(|a| match a {
                Answer::Decide { run, turn: t, .. } => *run == turn.run && *t == turn.turn,
                Answer::Detach { run, .. } => *run == turn.run,
            })
            .collect()
    }

    /// The first answer that lets `turn` reply.
    pub fn allowed_at(&self, turn: &GatedTurn) -> Option<Answer> {
        self.answers_for(turn).into_iter().find(Answer::allows)
    }

    /// When the customer hook of `turn` returned: its commit when no hook ran for it, `None`
    /// while it runs.
    pub fn hook_returned(&self, turn: &GatedTurn) -> Option<Observed> {
        self.hook_returned_for(turn.turn, turn.committed)
    }

    /// [`Self::hook_returned`] for any turn, gated or not, committed at `committed`: the end of
    /// its own hook, or, when it never got one, of every older turn's hook, which it waited
    /// behind (an older turn's hook can be issued after this commit). `None` while one runs.
    pub fn hook_returned_for(&self, turn: u64, committed: Observed) -> Option<Observed> {
        use lk_agents_voice::agent_session::HookKind;
        use lk_agents_voice::trace::HandoffStep;
        let mut own = None;
        let mut older: Vec<lk_agents_voice::agent_session::HookId> = Vec::new();
        let mut finished = std::collections::BTreeMap::new();
        for (at, step) in self.0.journal() {
            match step {
                Step::Handoff(HandoffStep::HookFinished { hook, .. }) => {
                    finished.entry(*hook).or_insert(*at);
                }
                Step::Handoff(HandoffStep::HookIssued {
                    hook,
                    kind: HookKind::OnUserTurnCompleted,
                    turn: Some(t),
                    ..
                }) if own.is_none() => {
                    if t.0 == turn {
                        own = Some(*hook);
                    } else if t.0 < turn {
                        older.push(*hook);
                    }
                }
                _ => {}
            }
        }
        if let Some(own) = own {
            return finished.get(&own).copied();
        }
        let mut until = committed;
        for hook in older {
            match finished.get(&hook) {
                Some(at) if *at < committed => {}
                Some(at) => until = until.max(*at),
                None => return None,
            }
        }
        Some(until)
    }

    /// The `on_user_turn_completed` call `step` issues for `turn`, if it is one.
    fn turn_hook(&self, step: &Step, turn: u64) -> Option<lk_agents_voice::agent_session::HookId> {
        use lk_agents_voice::agent_session::HookKind;
        use lk_agents_voice::audio_recognition::TurnId;
        use lk_agents_voice::trace::HandoffStep;
        match step {
            Step::Handoff(HandoffStep::HookIssued {
                hook,
                kind: HookKind::OnUserTurnCompleted,
                turn: Some(t),
                ..
            }) if *t == TurnId(turn) => Some(*hook),
            _ => None,
        }
    }

    /// Whether the session committed the realtime server's audio buffer for `turn` at its commit:
    /// the server's transcript of that audio is the turn's item, not one the session saves.
    pub fn audio_committed(&self, turn: &GatedTurn) -> bool {
        use lk_agents_abi::RealtimeCommandKind;
        use lk_agents_voice::trace::RealtimeStep;
        let until = turn.superseded.unwrap_or(Observed::END);
        self.0.journal().iter().any(|(at, step)| {
            *at >= turn.committed
                && at.at_ms == turn.committed.at_ms
                && *at < until
                && matches!(
                    step,
                    Step::Realtime(RealtimeStep::Sent {
                        kind: RealtimeCommandKind::CommitAudio,
                        ..
                    })
                )
        })
    }

    /// Whether the host answered `turn`'s customer hook with `StopResponse` or an error. Python
    /// ignores such a turn: it keeps neither a reply nor its words (`agent_activity.py`, "ignore
    /// this turn", and the `except Exception` after it).
    pub fn stopped(&self, turn: &GatedTurn) -> bool {
        self.stopped_turn(turn.turn)
    }

    /// [`Self::stopped`] for any turn, gated or not.
    pub fn stopped_turn(&self, turn: u64) -> bool {
        use crate::input::HookAnswer;
        let Some(hook) = self.0.journal().iter().find_map(|(_, step)| self.turn_hook(step, turn))
        else {
            return false;
        };
        let Some(nth) = self.0.resources(Kind::Hook).find(|o| o.id() == super::Id::Hook(hook))
        else {
            return false;
        };
        let nth = nth.nth();
        self.0.applied().iter().any(|a| {
            a.accepted
                && matches!(&a.input, Input::HookDone {
                    hook, answer: HookAnswer::TurnStop | HookAnswer::Failed, ..
                } if hook.nth == nth)
        })
    }

    /// The answer that decided `turn`, which rules about a decision's payload (stage, silence,
    /// DTMF) must read rather than the first one sent: its first decision, applied once both it
    /// and the hook's return came, unless its run's detach came first and settled the turn with
    /// `pending`, which replaces a decision not applied yet.
    pub fn deciding(&self, turn: &GatedTurn) -> Option<Answer> {
        let answers = self.answers_for(turn);
        let decide = answers.iter().find(|a| matches!(a, Answer::Decide { .. })).copied();
        let detach = answers.iter().find(|a| matches!(a, Answer::Detach { .. })).copied();
        let hook = self.hook_returned(turn);
        let applied = decide.and_then(|d| hook.map(|h| d.at().max(h)));
        match (applied, detach) {
            (Some(applied), Some(detach)) if detach.at() < applied => Some(detach),
            (None, Some(detach)) => Some(detach),
            _ => decide,
        }
    }

    /// When an agent first declared a tool named `name` (its `AgentConfigUpdate`): from then on
    /// the tool in a request is the agent's, not one AMD added.
    pub fn agent_declared_at(&self, name: &str) -> Option<Observed> {
        self.0.host_view().events().iter().find_map(|(at, e)| match e {
            SessionEvent::ConversationItemAdded { item, .. } => match &**item {
                lk_agents_abi::ChatItem::AgentConfigUpdate(update) => {
                    update.tools_added.iter().flatten().any(|t| t == name).then_some(*at)
                }
                _ => None,
            },
            _ => None,
        })
    }

    /// The deciding answer of `turn` and when the session could apply it: once both it and the
    /// turn's customer hook were in. `None` while the hook runs, with no deciding answer, or when
    /// a newer turn superseded the turn first.
    pub fn applied(&self, turn: &GatedTurn) -> Option<(Answer, Observed)> {
        let answer = self.deciding(turn)?;
        let at = answer.at().max(self.hook_returned(turn)?);
        turn.superseded.is_none_or(|s| at < s).then_some((answer, at))
    }

    /// Whether the run was listening when `turn` committed: only then does its decision move the
    /// hold.
    pub fn bound(&self, turn: &GatedTurn) -> bool {
        self.attachments()
            .into_iter()
            .find(|a| a.run == turn.run)
            .and_then(|a| a.activated)
            .is_some_and(|(activated, _)| activated < turn.committed)
    }

    /// The gated turn a request for `speech` answers, if any.
    pub fn turn_of_speech(&self, speech: u64) -> Option<GatedTurn> {
        self.gated_turns()
            .into_iter()
            .find(|turn| self.requests_answering(turn).iter().any(|q| q.speech == Some(speech)))
    }

    /// Every model request, in order.
    pub fn requests(&self) -> Vec<Request<'a>> {
        self.0
            .resources(Kind::Chat)
            .filter_map(|chat| match chat.detail() {
                Detail::Chat { speech, messages, tools, .. } => Some(Request {
                    chat: chat.nth(),
                    opened: chat.opened_at(),
                    speech: *speech,
                    messages,
                    tools,
                }),
                _ => None,
            })
            .collect()
    }

    /// The requests that answer `turn`: the newest user turn item in them is its saved item, or,
    /// for a preemptive generation made before the commit, their last user message is its words.
    // The newest turn item, not the last user message: a storage partition's text can follow it.
    pub fn requests_answering(&self, turn: &GatedTurn) -> Vec<Request<'a>> {
        let previous = self
            .gated_turns()
            .into_iter()
            .filter(|t| t.run == turn.run && t.committed < turn.committed)
            .map(|t| t.committed)
            .next_back()
            .unwrap_or(Observed::at(0, 0));
        let turn_items = self.turn_items();
        self.requests()
            .into_iter()
            .filter(|r| {
                // A host `generate_reply` or the opening answers no turn, even when the turn's
                // words are the newest in its context.
                let kind = r.speech.and_then(|s| self.0.host_view().speech(s).kind());
                if kind
                    .is_some_and(|k| !matches!(k, SpeechKind::TurnReply | SpeechKind::Preemptive))
                {
                    return false;
                }
                let Some(user) = r.last_user() else { return false };
                // A preemptive generation's own words are not a turn item yet: older turns'
                // items before them say nothing about which turn it answers.
                if self.preemptive(r) {
                    return user.text == turn.words
                        && r.opened > previous
                        && r.opened < turn.committed;
                }
                let newest_turn = r.messages.iter().rev().find(|m| turn_items.contains(&m.id));
                newest_turn.is_some_and(|m| turn.item_id.as_deref() == Some(m.id.as_str()))
            })
            .collect()
    }

    /// Whether the session dropped `turn` with no reply, behind a speech it could not interrupt or
    /// while scheduling was paused: its diagnostic names the turn's words.
    pub fn dropped_unanswered(&self, turn: &GatedTurn) -> bool {
        self.dropped_words(&turn.words, turn.committed)
    }

    /// [`Self::dropped_unanswered`] for any turn's `words`, committed at `committed`.
    pub fn dropped_words(&self, words: &str, committed: Observed) -> bool {
        use lk_agents_voice::events::diagnostics;
        self.0.host_view().events().iter().any(|(at, e)| match e {
            SessionEvent::Diagnostic(d) => {
                *at > committed
                    && [diagnostics::UNINTERRUPTIBLE_SPEECH, diagnostics::SCHEDULING_PAUSED]
                        .contains(&d.message)
                    && d.fields.iter().any(|(_, value)| *value == words)
            }
            _ => false,
        })
    }

    /// Whether the session's end of turn `turn` asked for no reply (`skip_reply`), which its
    /// `EndOfTurn` verdict does not show.
    pub fn skips_reply(&self, turn: u64) -> bool {
        use lk_agents_voice::audio_recognition::observation::RecognitionObservation;
        self.0.journal().iter().any(|(_, step)| {
            matches!(step, Step::Decided(RecognitionObservation::EndOfTurn(info))
                if info.turn_id.0 == turn && info.skip_reply)
        })
    }

    /// The id of every user item a committed turn saved.
    fn turn_items(&self) -> std::collections::BTreeSet<String> {
        (self.0.host_view().events().iter())
            .filter_map(|(_, e)| match e {
                SessionEvent::ConversationItemAdded {
                    item,
                    source: lk_agents_voice::agent::Source::UserTurn { .. },
                    ..
                } => item.as_message().map(|m| m.id.clone()),
                _ => None,
            })
            .collect()
    }

    /// Whether the request belongs to a preemptive generation.
    pub fn preemptive(&self, request: &Request<'_>) -> bool {
        use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechKind;
        request
            .speech
            .is_some_and(|s| self.0.host_view().speech(s).kind() == Some(SpeechKind::Preemptive))
    }

    /// When the speech's first audio reached the sink.
    pub fn first_audio(&self, speech: u64) -> Option<Observed> {
        self.0.internal_view().speech(speech).first_sink_push_at()
    }

    /// What `run` observed, in order.
    pub fn observed(&self, run: u64) -> Vec<(Observed, &'a AmdObservation)> {
        self.0
            .amd()
            .iter()
            .filter(|(_, r, _)| *r == AmdRunId(run))
            .map(|(at, _, o)| (*at, o))
            .collect()
    }

    /// The fake VAD's ground truth: each boundary it reported, as `(speaking, boundary_ms,
    /// reported)`, the boundary backdated by the speech or silence the case said it measured.
    pub fn vad_boundaries(&self) -> Vec<(bool, u64, Observed)> {
        self.0
            .applied()
            .iter()
            .filter(|a| a.accepted)
            .filter_map(|a| match &a.input {
                Input::VadStart { speech_ms, .. } => {
                    Some((true, a.at.at_ms.saturating_sub(speech_ms.0), a.at))
                }
                Input::VadEnd { silence_ms, .. } => {
                    Some((false, a.at.at_ms.saturating_sub(silence_ms.0), a.at))
                }
                _ => None,
            })
            .collect()
    }

    /// Whether the participant was speaking at `at` by the fake VAD, and since when they were
    /// silent if not.
    pub fn silent_since(&self, at: Observed) -> Option<u64> {
        match self.vad_boundaries().into_iter().rfind(|(_, _, seen)| *seen < at) {
            Some((true, _, _)) => None,
            Some((false, boundary, _)) => Some(boundary),
            None => Some(0),
        }
    }
}
