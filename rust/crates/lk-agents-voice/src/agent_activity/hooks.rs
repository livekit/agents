//! The developer's callbacks the driver awaits (`on_enter`, `on_exit`, `on_user_turn_completed`,
//! the agent a tool hands off to, a tool reply's rendering, a filler's source, the
//! `function_tools_executed` handlers), and how long each is waited on.

use crate::agent::TurnEdit;
use crate::agent_activity::handoff::HandoffEvent;
use crate::agent_session::effects::Out;
use crate::agent_session::signals::Signal;
use crate::agent_session::{Driver, HookCall, HookId, HookKind, HookOutcome, HostEvent};
use crate::audio_recognition::PendingTurn;
use crate::run_result::Owned;
use crate::trace::{CoreStep, HandoffStep, HookEnd};
use lk_agents_abi::Ts;
use std::collections::BTreeMap;
use std::time::Duration;

/// How long the driver waits on a callback whose answer it needs when the host named no bound.
pub const HOOK_BOUND: Duration = crate::speech_handle::INTERRUPTION_TIMEOUT;

/// The callbacks in flight, by id.
#[derive(Default)]
pub(crate) struct Hooks {
    pending: BTreeMap<HookId, Pending>,
    next: u64,
}

struct Pending {
    kind: HookKind,
    /// When the driver stops waiting; `None` when nothing waits.
    deadline: Option<Ts>,
    /// A `DescribeAgent` asked by the driving agent. A tool's handoff blocks the agent whose tool
    /// returned the new one, not one a handoff installs meanwhile (cleared at the swap).
    describing: bool,
}

impl Hooks {
    pub(crate) fn mint(&mut self) -> HookId {
        self.next += 1;
        HookId(self.next)
    }

    /// The host was asked `hook`: waited on until `deadline` (`None`: nothing waits).
    pub(crate) fn wait_for(&mut self, hook: HookId, kind: HookKind, deadline: Option<Ts>) {
        let describing = kind == HookKind::DescribeAgent;
        self.pending.insert(hook, Pending { kind, deadline, describing });
    }

    /// The callback is answered or given up on: its kind, if it was waited on.
    pub(crate) fn finish(&mut self, hook: HookId) -> Option<HookKind> {
        self.pending.remove(&hook).map(|h| h.kind)
    }

    pub(crate) fn is_pending(&self, hook: HookId) -> bool {
        self.pending.contains_key(&hook)
    }

    /// The pending callbacks of one kind, in id order.
    pub(crate) fn pending_of(&self, kind: HookKind) -> Vec<HookId> {
        self.pending.iter().filter(|(_, h)| h.kind == kind).map(|(id, _)| *id).collect()
    }

    pub(crate) fn next_deadline(&self) -> Option<Ts> {
        self.pending.values().filter_map(|h| h.deadline).min()
    }

    /// A tool's handoff waits for the host to describe the new agent.
    pub(crate) fn describing(&self) -> bool {
        self.pending.values().any(|h| h.describing)
    }

    /// The agent that asked a pending describe has left: the arriving one is not blocked by it.
    pub(crate) fn stop_describing(&mut self) {
        self.pending.values_mut().for_each(|h| h.describing = false);
    }

    /// Every callback still waited on, taken, in id order.
    pub(crate) fn take_all(&mut self) -> Vec<(HookId, HookKind)> {
        std::mem::take(&mut self.pending).into_iter().map(|(id, h)| (id, h.kind)).collect()
    }
}

impl Driver {
    /// Journalled where decided, sent to the host when one runs callbacks, else finished in place
    /// (`false`). `wait` bounds the wait when an answer is needed.
    pub(crate) fn issue_hook(
        &mut self,
        now: Ts,
        call: HookCall,
        wait: Option<Duration>,
        out: &mut Out,
    ) -> bool {
        let id = call.id();
        let kind = call.kind();
        let (agent, speech, turn, at_tail) = match &call {
            HookCall::OnEnter { agent, .. } | HookCall::OnExit { agent, .. } => {
                (Some(agent.clone()), None, None, None)
            }
            HookCall::OnUserTurnCompleted { turn, .. } => (None, None, Some(*turn), None),
            HookCall::DescribeAgent { speech, .. } => (None, Some(*speech), None, None),
            HookCall::RenderToolReply { at_tail, .. } => (None, None, None, Some(*at_tail)),
            HookCall::FillerSource { .. } => (None, None, None, None),
            HookCall::FunctionToolsExecuted { speech, .. } => (None, Some(*speech), None, None),
        };
        out.trace(HandoffStep::HookIssued { hook: id, kind, agent, speech, turn, at_tail });
        let hosted =
            self.activity.hook_bound.is_some() && self.events.send(HostEvent::Hook(call)).is_ok();
        if !hosted {
            out.trace(HandoffStep::HookFinished { hook: id, kind, how: HookEnd::NoHost });
            return false;
        }
        let deadline = wait.map(|d| now.saturating_add(d));
        self.activity.hooks.wait_for(id, kind, deadline);
        // An open run owns the agent's `on_enter` and the describe a tool's handoff waits on.
        match kind {
            HookKind::OnEnter | HookKind::DescribeAgent => self.run_own(Owned::Hook(id), out),
            HookKind::OnExit
            | HookKind::OnUserTurnCompleted
            | HookKind::RenderToolReply
            | HookKind::FillerSource
            | HookKind::FunctionToolsExecuted => {}
        }
        true
    }

    pub(crate) fn hook_wait(&self) -> Duration {
        self.activity.hook_bound.unwrap_or(HOOK_BOUND)
    }

    /// Nothing waits on it.
    pub(crate) fn issue_on_enter(&mut self, now: Ts, out: &mut Out) {
        let id = self.activity.hooks.mint();
        let agent = self.activity.agent.clone();
        self.issue_hook(now, HookCall::OnEnter { id, agent }, None, out);
    }

    /// The turn parks on it until answer or bound.
    pub(crate) fn issue_turn_hook(
        &mut self,
        now: Ts,
        info: crate::audio_recognition::EndOfTurnInfo,
        bound: Duration,
        out: &mut Out,
    ) {
        let id = self.activity.hooks.mint();
        let call = HookCall::OnUserTurnCompleted {
            id,
            turn: info.turn_id,
            new_transcript: info.new_transcript.clone(),
            transcript_confidence: info.transcript_confidence,
        };
        if self.issue_hook(now, call, None, out) {
            self.activity.pending_turn = Some(PendingTurn {
                info,
                issued_at: now,
                deadline: now.saturating_add(bound),
                hook: Some(id),
            });
        } else {
            self.finish_turn(now, info, Vec::new(), out);
        }
    }

    pub(crate) fn on_hook_done(
        &mut self,
        now: Ts,
        id: HookId,
        outcome: HookOutcome,
        out: &mut Out,
    ) {
        let how = match &outcome {
            HookOutcome::Failed => HookEnd::Failed,
            HookOutcome::Turn(edits) => HookEnd::Returned { edits: edits.len() },
            HookOutcome::Returned
            | HookOutcome::Described(_)
            | HookOutcome::Rendered(_)
            | HookOutcome::Filler { .. }
            | HookOutcome::FunctionToolsExecuted { .. } => HookEnd::Returned { edits: 0 },
        };
        // An id nobody waits on (a late answer) is dropped.
        let Some(kind) = self.finish_hook(id, how, out) else { return };
        self.hook_ended(now, kind, id, outcome, out);
    }

    /// The callback left `pending`, journalled and let go by the open run; `None` if not pending.
    fn finish_hook(&mut self, id: HookId, how: HookEnd, out: &mut Out) -> Option<HookKind> {
        let kind = self.activity.hooks.finish(id)?;
        out.trace(HandoffStep::HookFinished { hook: id, kind, how });
        self.run_release(Owned::Hook(id), out);
        Some(kind)
    }

    /// What waited on the callback goes on. `Failed` is also one that will not be answered (its
    /// bound passed): an exit proceeds, a describe is dropped, a render or a filler source falls
    /// back to its default. A turn callback's bound is `pending_turn`'s. A mismatched answer
    /// counts as unanswered.
    fn hook_ended(
        &mut self,
        now: Ts,
        kind: HookKind,
        id: HookId,
        outcome: HookOutcome,
        out: &mut Out,
    ) {
        use HookOutcome::{
            Described, Failed, Filler, FunctionToolsExecuted, Rendered, Returned, Turn,
        };
        let fits = matches!(
            (kind, &outcome),
            (HookKind::OnExit | HookKind::OnEnter, Returned | Failed)
                | (HookKind::DescribeAgent, Described(_) | Failed)
                | (HookKind::OnUserTurnCompleted, Turn(_) | Returned | Failed)
                | (HookKind::RenderToolReply, Rendered(_) | Returned | Failed)
                | (HookKind::FillerSource, Filler { .. } | Returned | Failed)
                | (
                    HookKind::FunctionToolsExecuted,
                    FunctionToolsExecuted { .. } | Returned | Failed
                )
        );
        // The sequence must not hang on a malformed answer: it goes on as if unanswered.
        if !fits {
            out.trace(CoreStep::PortFailure("hook outcome does not match the call it answers"));
        }
        match kind {
            HookKind::OnExit => out.signal(Signal::OnExitEnded(id)),
            HookKind::OnEnter => {}
            HookKind::DescribeAgent => match outcome {
                Described(update) => self.handoff_step(HandoffEvent::Described(id, update), out),
                Failed
                | Returned
                | Turn(_)
                | Rendered(_)
                | Filler { .. }
                | FunctionToolsExecuted { .. } => {
                    self.handoff_step(HandoffEvent::DescribeDropped(id), out)
                }
            },
            // "error occurred during on_user_turn_completed": return, no reply.
            HookKind::OnUserTurnCompleted => {
                let edits = match outcome {
                    Turn(edits) => edits,
                    Failed if fits => vec![TurnEdit::StopResponse],
                    Failed
                    | Returned
                    | Described(_)
                    | Rendered(_)
                    | Filler { .. }
                    | FunctionToolsExecuted { .. } => Vec::new(),
                };
                // Matched to its parked turn; a late answer is dropped.
                if let Some(pending) = self.activity.pending_turn.take_if(|p| p.hook == Some(id)) {
                    self.finish_turn(now, pending.finished(now), edits, out)
                }
            }
            // The deferred tool reply's instructions, or the defaults when rendering failed.
            HookKind::RenderToolReply => {
                let text = match outcome {
                    Rendered(text) => Some(text),
                    Failed
                    | Returned
                    | Turn(_)
                    | Described(_)
                    | Filler { .. }
                    | FunctionToolsExecuted { .. } => None,
                };
                self.on_tool_reply_rendered(now, id, text, out)
            }
            // The filler's callable: text, a speech it made itself, or nothing; raising skips.
            HookKind::FillerSource => {
                let (text, spoke) = match outcome {
                    Filler { text, spoke } => (text, spoke),
                    Failed
                    | Returned
                    | Turn(_)
                    | Described(_)
                    | Rendered(_)
                    | FunctionToolsExecuted { .. } => (None, false),
                };
                self.on_filler_source(now, id, text, spoke, out)
            }
            // What the handlers left; unanswered, the event stands as issued.
            HookKind::FunctionToolsExecuted => {
                let left = match outcome {
                    FunctionToolsExecuted { reply_required, handoff_required } => {
                        Some((reply_required, handoff_required))
                    }
                    Failed | Returned | Turn(_) | Described(_) | Rendered(_) | Filler { .. } => {
                        None
                    }
                };
                self.on_tools_executed_handled(id, left, out)
            }
        }
    }

    /// Expired bounds: an unreturned `on_exit` lets the drain go on; an unanswered describe drops
    /// the handoff.
    pub(crate) fn expire_hooks(&mut self, now: Ts, out: &mut Out) {
        let due: Vec<_> = (self.activity.hooks.pending.iter())
            .filter(|(_, h)| h.deadline.is_some_and(|d| d <= now))
            .map(|(id, _)| *id)
            .collect();
        for id in due {
            if let Some(kind) = self.finish_hook(id, HookEnd::TimedOut, out) {
                self.hook_ended(now, kind, id, HookOutcome::Failed, out);
            }
        }
    }
}
