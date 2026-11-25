//! The runs entered through AMD's executor (`lk_agents_runtime::voice::amd`): what each told the
//! host (its events and its `execute()` outcome), what it asked its fake models, and what the case
//! answered them. The executor's own account is its `AmdStep` journal; a rule holds that account
//! against these outcomes, so a run that journals nothing proves nothing.

use lk_agents_voice::amd::AmdRunId;
use lk_agents_voice::amd::effects::ReqId;
use lk_agents_voice::amd::events::{AmdEvent, AmdReason, Category, Outcome, Prediction, Stage};
use lk_agents_voice::audio_recognition::TurnId;
use lk_agents_voice::trace::{AmdStep, Step};

use super::amd::RequestMessage;
use super::{Detail, Kind, Observed, Recording};
use crate::input::AmdSays;

/// Which of AMD's fake models a request went to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AskKind {
    /// The classifier LLM, for a category.
    Classify,
    /// The classifier LLM, for a menu.
    Menu,
    /// The decision model, for one choice.
    Choice,
}

/// One request AMD's executor made, as the fake model received it.
#[derive(Clone, Debug, PartialEq)]
pub struct Asked {
    pub kind: AskKind,
    /// An LLM request's messages, in order.
    pub messages: Vec<RequestMessage>,
    /// The tools it offered, by name.
    pub tools: Vec<String>,
    /// `tool_choice="required"`.
    pub required: bool,
    /// It carried AMD's client feature header.
    pub tagged: bool,
    /// A decision model's `input`.
    pub input: Option<String>,
    /// A decision model's options.
    pub options: Vec<String>,
}

impl Asked {
    pub fn carries(&self, text: &str) -> bool {
        self.messages.iter().any(|m| m.text.contains(text))
            || self.input.as_deref().is_some_and(|input| input.contains(text))
    }

    /// The state JSON a classification sends (its second system message, or the decision
    /// model's `Request:` line).
    pub fn state(&self) -> Option<String> {
        match self.kind {
            AskKind::Classify => self.messages.get(1).map(|m| m.text.clone()),
            AskKind::Choice => self.input.as_deref().and_then(|input| {
                let line = input.lines().next()?;
                line.strip_prefix("Request: ").map(str::to_owned)
            }),
            AskKind::Menu => None,
        }
    }
}

/// What the case answered one ask, and whether the ask's model call still took it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reply {
    pub at: Observed,
    pub ask: u32,
    pub says: AmdSays,
    pub accepted: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Executed {
    /// Each `AmdEnter`'s answer: the run it attached, or why none did.
    entered: Vec<(Observed, Result<AmdRunId, String>)>,
    /// Each event a run handed the host, in order.
    events: Vec<(Observed, AmdRunId, AmdEvent)>,
    /// Each `execute()` that returned, with the outcome it returned.
    outcomes: Vec<(Observed, AmdRunId, Outcome)>,
    replies: Vec<Reply>,
    /// Every AMD answer the session's loop took, scripted or the executor's, as it took it.
    answers: Vec<(Observed, lk_agents_voice::amd::AmdAnswer)>,
}

impl Recording {
    pub fn record_amd_entered(&mut self, at: Observed, entered: Result<AmdRunId, String>) {
        self.amd_run.entered.push((at, entered));
    }

    pub fn record_amd_event(&mut self, at: Observed, run: AmdRunId, event: AmdEvent) {
        self.amd_run.events.push((at, run, event));
    }

    pub fn record_amd_outcome(&mut self, at: Observed, run: AmdRunId, outcome: Outcome) {
        self.amd_run.outcomes.push((at, run, outcome));
    }

    pub fn record_amd_answer(&mut self, at: Observed, answer: lk_agents_voice::amd::AmdAnswer) {
        self.amd_run.answers.push((at, answer));
    }

    pub fn record_amd_reply(&mut self, reply: Reply) {
        self.amd_run.replies.push(reply);
    }

    pub fn amd_run(&self) -> AmdRunView<'_> {
        AmdRunView(self)
    }
}

/// A prediction as the executor journalled it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Predicted {
    pub at: Observed,
    pub run: AmdRunId,
    pub turn: TurnId,
    pub turn_id: u64,
    pub source_req: Option<ReqId>,
    pub raw: Option<Category>,
    pub category: Category,
    pub stage: Stage,
    pub reason: AmdReason,
    pub likely_human: bool,
}

pub struct AmdRunView<'a>(&'a Recording);

impl<'a> AmdRunView<'a> {
    /// The runs that attached through `AmdEnter`, in order.
    pub fn entered(&self) -> Vec<(Observed, AmdRunId)> {
        (self.0.amd_run.entered.iter())
            .filter_map(|(at, e)| Some((*at, *e.as_ref().ok()?)))
            .collect()
    }

    /// Each `AmdEnter` answer, refusals included.
    pub fn enter_answers(&self) -> &[(Observed, Result<AmdRunId, String>)] {
        &self.0.amd_run.entered
    }

    pub fn events(&self, run: AmdRunId) -> Vec<(Observed, AmdEvent)> {
        (self.0.amd_run.events.iter())
            .filter(|(_, r, _)| *r == run)
            .map(|(at, _, e)| (*at, e.clone()))
            .collect()
    }

    /// The run's `amd_prediction` events, in order.
    pub fn predictions(&self, run: AmdRunId) -> Vec<(Observed, Prediction)> {
        (self.events(run).into_iter())
            .filter_map(|(at, e)| match e {
                AmdEvent::Prediction(p) => Some((at, p)),
                _ => None,
            })
            .collect()
    }

    /// The run's `amd_completed` events.
    pub fn completed(&self, run: AmdRunId) -> Vec<(Observed, Outcome)> {
        (self.events(run).into_iter())
            .filter_map(|(at, e)| match e {
                AmdEvent::Completed(o) => Some((at, o)),
                _ => None,
            })
            .collect()
    }

    /// The run's `amd_menu_observed` events.
    pub fn menus(&self, run: AmdRunId) -> usize {
        (self.events(run).iter()).filter(|(_, e)| matches!(e, AmdEvent::MenuObserved(_))).count()
    }

    /// What the run's `execute()` returned, each time it returned.
    pub fn outcomes(&self, run: AmdRunId) -> Vec<(Observed, Outcome)> {
        (self.0.amd_run.outcomes.iter())
            .filter(|(_, r, _)| *r == run)
            .map(|(at, _, o)| (*at, o.clone()))
            .collect()
    }

    /// Every request the fake models received, by `Ref<AmdAsk>` number, with when it opened.
    pub fn asks(&self) -> Vec<(u32, Observed, Asked)> {
        (self.0.resources(Kind::AmdAsk))
            .filter_map(|r| match r.detail() {
                Detail::AmdAsk(asked) => Some((r.nth(), r.opened_at(), (**asked).clone())),
                _ => None,
            })
            .collect()
    }

    pub fn asks_of(&self, kind: AskKind) -> Vec<(u32, Observed, Asked)> {
        self.asks().into_iter().filter(|(_, _, a)| a.kind == kind).collect()
    }

    /// The answers the session's loop took from `run`, in order.
    pub fn answers(&self, run: AmdRunId) -> Vec<(Observed, lk_agents_voice::amd::AmdAnswer)> {
        use lk_agents_voice::amd::AmdAnswer;
        (self.0.amd_run.answers.iter())
            .filter(|(_, answer)| match answer {
                AmdAnswer::Activated { run: r, .. }
                | AmdAnswer::ReplyDecided { run: r, .. }
                | AmdAnswer::InsertUserItem { run: r, .. }
                | AmdAnswer::Detach { run: r, .. } => *r == run,
            })
            .cloned()
            .collect()
    }

    /// What the case answered, in order.
    pub fn replies(&self) -> &[Reply] {
        &self.0.amd_run.replies
    }

    /// The executor's own steps, in order.
    pub fn steps(&self) -> Vec<(Observed, AmdStep)> {
        (self.0.journal().iter())
            .filter_map(|(at, step)| match step {
                Step::Amd(step) => Some((*at, step.clone())),
                _ => None,
            })
            .collect()
    }

    pub fn predicted(&self) -> Vec<Predicted> {
        (self.steps().into_iter())
            .filter_map(|(at, step)| match step {
                AmdStep::Predicted {
                    run,
                    turn,
                    turn_id,
                    source_req,
                    raw,
                    category,
                    stage,
                    reason,
                    likely_human,
                } => Some(Predicted {
                    at,
                    run,
                    turn,
                    turn_id,
                    source_req,
                    raw,
                    category,
                    stage,
                    reason,
                    likely_human,
                }),
                _ => None,
            })
            .collect()
    }

    /// When the session reported `Detached` for `run`.
    pub fn detached(&self, run: AmdRunId) -> Option<Observed> {
        use lk_agents_voice::amd::AmdObservation;
        self.0.amd().iter().find_map(|(at, r, o)| {
            matches!(o, AmdObservation::Detached { run: d } if *d == run && *r == run)
                .then_some(*at)
        })
    }
}

/// A classifier's category for an answer the case gives, as `record_result` carries it; `None`
/// for an answer that names none.
pub fn says_category(says: AmdSays) -> Option<Category> {
    Some(match says {
        AmdSays::Human => Category::Human,
        AmdSays::Ivr => Category::MachineIvr,
        AmdSays::Screening => Category::MachineScreening,
        AmdSays::Voicemail => Category::MachineVm,
        AmdSays::Unavailable => Category::MachineUnavailable,
        AmdSays::Wait => Category::Wait,
        AmdSays::Uncertain => Category::Uncertain,
        AmdSays::Menu | AmdSays::NoMenu | AmdSays::Garbled | AmdSays::Failed | AmdSays::Crashed => {
            return None;
        }
    })
}
