//! An invocation a coordinator (VoiceSession) composes: it owns the record, plans each request
//! and supplies voice constraints after handling canonical tool results.

use std::collections::{BTreeMap, BTreeSet};

use lk_agents_abi::{ChatItem, ChatMessage, Ts};

use super::{AgentRun, Effect, Next, Phase, StepId};

/// How an invocation opens, and who owns its record and its requests.
#[derive(Debug, Clone)]
pub enum Opening {
    /// Direct runs and TextSession: the incoming message, answered against the history, and every
    /// request planned here.
    Message(Box<ChatMessage>),
    /// VoiceSession: the coordinator owns the record and plans each request
    /// ([`Effect::PlanRequest`]), and handles tool results ([`Effect::HandleToolResults`]).
    /// `approval_answers`: the read-backs the person's latest message answered.
    Coordinated { approval_answers: BTreeSet<String> },
}

impl Opening {
    pub fn message(&self) -> Option<&ChatMessage> {
        match self {
            Opening::Message(message) => Some(message),
            Opening::Coordinated { .. } => None,
        }
    }

    pub(super) fn coordinated(&self) -> bool {
        matches!(self, Opening::Coordinated { .. })
    }
}

/// Constraints supplied after the coordinator handles the run's canonical tool results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentRunContinuation {
    /// The run goes on as it decided; a next model step offers its tools but may not call them
    /// when `tools_forbidden`.
    Continue { tools_forbidden: bool },
    /// No further model step: a run that would continue completes with nothing.
    Stop,
    /// The person cut the reply: the run ends interrupted, unless it already failed or accepted
    /// its output.
    Interrupted,
}

impl AgentRun {
    /// The person's latest answers: what a step that has not called yet grants with. The person
    /// moved on from the answers a step already calling read: they grant nothing more.
    pub(super) fn answered(&mut self, approval_answers: BTreeSet<String>) {
        if let Some(acting) = self.phase.acting() {
            let answers =
                if acting.calls.is_empty() { approval_answers.clone() } else { BTreeSet::new() };
            acting.calls.answered(answers);
        }
        self.approval_answers = approval_answers;
    }

    /// Apply permitted result edits and continuation constraints. Neither may undo the run
    /// failure, accepted output, or explicit terminal selection.
    pub(super) fn tool_results_handled(
        &mut self,
        step: StepId,
        continuation: AgentRunContinuation,
        reply_required: BTreeMap<String, bool>,
        now: Ts,
        fx: &mut Vec<Effect>,
    ) {
        if !matches!(&self.phase, Phase::Settling { step: s, .. } if *s == step) {
            return;
        }
        let Phase::Settling { mut next, calls, editable_continuation, .. } =
            std::mem::replace(&mut self.phase, Phase::Idle)
        else {
            unreachable!()
        };
        // Only outputs belonging to this correlated step may be edited. Keep the canonical
        // record and the coordinator's published view in agreement without choosing terminal
        // outcomes again or accepting replacement text/history from the coordinator.
        let mut replies = false;
        for item in &mut self.history {
            let ChatItem::FunctionCallOutput(output) = item else { continue };
            if !calls.iter().any(|call| {
                call.item.call_id == output.call_id && output.id == format!("{}_out", call.item.id)
            }) {
                continue;
            }
            if let Some(required) = reply_required.get(&output.call_id) {
                output.reply_required = *required;
            }
            replies |= output.reply_required;
        }
        if editable_continuation {
            next = if replies { Next::Continue } else { Next::Stop };
        }
        match (continuation, next) {
            (_, next @ (Next::Fail(_) | Next::Output(_))) => self.proceed(next, now, fx),
            (AgentRunContinuation::Interrupted, _) => self.complete_interrupted(fx),
            (AgentRunContinuation::Stop, Next::Continue) => self.end_without_output(fx),
            (AgentRunContinuation::Continue { tools_forbidden }, Next::Continue) => {
                self.tools_forbidden = tools_forbidden;
                self.next_step(now, fx)
            }
            (_, next) => self.proceed(next, now, fx),
        }
    }
}
