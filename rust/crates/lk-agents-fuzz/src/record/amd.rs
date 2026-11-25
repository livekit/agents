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
                    if applied.accepted {
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
        out
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

    /// Whether the end of turn the host heard at `at` asked for a reply: the verdict journalled
    /// just before it.
    fn replies(&self, at: Observed) -> bool {
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
        out
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

    /// The requests that answer `turn`: their last user message is its saved item, or, for a
    /// preemptive generation made before the commit, its words.
    pub fn requests_answering(&self, turn: &GatedTurn) -> Vec<Request<'a>> {
        let previous = self
            .gated_turns()
            .into_iter()
            .filter(|t| t.run == turn.run && t.committed < turn.committed)
            .map(|t| t.committed)
            .next_back()
            .unwrap_or(Observed::at(0, 0));
        self.requests()
            .into_iter()
            .filter(|r| {
                let Some(user) = r.last_user() else { return false };
                if turn.item_id.as_deref() == Some(user.id.as_str()) {
                    return true;
                }
                self.preemptive(r) && user.text == turn.words && r.opened > previous
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
