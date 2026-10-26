//! The developer's callbacks the driver awaits (`on_enter`, `on_exit`, `on_user_turn_completed`,
//! a tool reply's rendering, a filler's source, the `function_tools_executed` handlers), and how
//! long each is waited on.

use crate::agent::TurnEdit;
use crate::agent_session::effects::{Out, Signal, Timer};
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
    pending: BTreeMap<HookId, HookKind>,
    next: u64,
}

impl Hooks {
    pub(crate) fn mint(&mut self) -> HookId {
        self.next += 1;
        HookId(self.next)
    }

    /// The host was asked `hook`.
    pub(crate) fn wait_for(&mut self, hook: HookId, kind: HookKind) {
        self.pending.insert(hook, kind);
    }

    /// The callback is answered or given up on: its kind, if it was waited on.
    pub(crate) fn finish(&mut self, hook: HookId) -> Option<HookKind> {
        self.pending.remove(&hook)
    }

    /// Every callback still waited on, taken, in id order.
    pub(crate) fn take_all(&mut self) -> Vec<(HookId, HookKind)> {
        std::mem::take(&mut self.pending).into_iter().collect()
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
        self.activity.hooks.wait_for(id, kind);
        if let Some(wait) = wait {
            out.machine().arm(Timer::Hook(id), now.saturating_add(wait));
        }
        // An open run owns the agent's `on_enter`.
        match kind {
            HookKind::OnEnter => self.run_own(Owned::Hook(id), out),
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
            self.activity.pending_turn = Some(PendingTurn { info, issued_at: now, hook: Some(id) });
            out.machine().arm(Timer::UserTurnCompleted, now.saturating_add(bound));
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
        self.disarm(Timer::Hook(id), out);
        out.trace(HandoffStep::HookFinished { hook: id, kind, how });
        self.run_release(Owned::Hook(id), out);
        Some(kind)
    }

    /// What waited on the callback goes on. `Failed` is also one that will not be answered (its
    /// bound passed): an exit proceeds, a render or a filler source falls back to its default. A
    /// turn callback's bound is `pending_turn`'s. A mismatched answer counts as unanswered.
    fn hook_ended(
        &mut self,
        now: Ts,
        kind: HookKind,
        id: HookId,
        outcome: HookOutcome,
        out: &mut Out,
    ) {
        use HookOutcome::{Failed, Filler, FunctionToolsExecuted, Rendered, Returned, Turn};
        let fits = matches!(
            (kind, &outcome),
            (HookKind::OnExit | HookKind::OnEnter, Returned | Failed)
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
            // "error occurred during on_user_turn_completed": return, no reply.
            HookKind::OnUserTurnCompleted => {
                let edits = match outcome {
                    Turn(edits) => edits,
                    Failed if fits => vec![TurnEdit::StopResponse],
                    Failed
                    | Returned
                    | Rendered(_)
                    | Filler { .. }
                    | FunctionToolsExecuted { .. } => Vec::new(),
                };
                // Matched to its parked turn; a late answer is dropped.
                if let Some(pending) = self.activity.pending_turn.take_if(|p| p.hook == Some(id)) {
                    self.disarm(Timer::UserTurnCompleted, out);
                    self.finish_turn(now, pending.finished(now), edits, out)
                }
            }
            // The deferred tool reply's instructions, or the defaults when rendering failed.
            HookKind::RenderToolReply => {
                let text = match outcome {
                    Rendered(text) => Some(text),
                    Failed | Returned | Turn(_) | Filler { .. } | FunctionToolsExecuted { .. } => {
                        None
                    }
                };
                self.on_tool_reply_rendered(now, id, text, out)
            }
            // The filler's callable: text, a speech it made itself, or nothing; `None` when it
            // raised or its bound passed.
            HookKind::FillerSource => {
                let answered = match outcome {
                    Filler { text, spoke } => Some((text, spoke)),
                    Failed => None,
                    Returned | Turn(_) | Rendered(_) | FunctionToolsExecuted { .. } => {
                        Some((None, false))
                    }
                };
                self.on_filler_source(now, id, answered, out)
            }
            // What the handlers left; unanswered, the event stands as issued.
            HookKind::FunctionToolsExecuted => {
                let left = match outcome {
                    FunctionToolsExecuted { reply_required, handoff_required } => {
                        Some((reply_required, handoff_required))
                    }
                    Failed | Returned | Turn(_) | Rendered(_) | Filler { .. } => None,
                };
                self.on_tools_executed_handled(id, left, out)
            }
        }
    }

    /// The bound passed: an unreturned `on_exit` lets the drain go on.
    pub(crate) fn hook_timed_out(&mut self, now: Ts, id: HookId, out: &mut Out) {
        if let Some(kind) = self.finish_hook(id, HookEnd::TimedOut, out) {
            self.hook_ended(now, kind, id, HookOutcome::Failed, out);
        }
    }
}

/// The driving agent's `on_exit`, and the ones claimed but not yet asked of the host.
#[derive(Default)]
pub(crate) struct OnExit {
    current: State,
    /// Claimed and not yet asked of the host, with the agent each is for.
    claimed: Vec<(HookId, String)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum State {
    #[default]
    Running,
    Exiting(HookId),
    /// The close stopped waiting for it.
    Abandoned(HookId),
    Exited,
}

/// The `on_exit` a leaving agent owes: issued now, already running, or none (it ran).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Exit {
    Issue(HookId),
    Wait(HookId),
    Done,
}

impl Exit {
    pub(crate) fn hook(self) -> Option<HookId> {
        match self {
            Exit::Issue(hook) | Exit::Wait(hook) => Some(hook),
            Exit::Done => None,
        }
    }
}

impl OnExit {
    /// The driving agent's `on_exit` running now.
    pub(crate) fn exiting(&self) -> Option<HookId> {
        match self.current {
            State::Exiting(hook) | State::Abandoned(hook) => Some(hook),
            State::Running | State::Exited => None,
        }
    }

    pub(crate) fn take_claimed(&mut self) -> Vec<(HookId, String)> {
        std::mem::take(&mut self.claimed)
    }

    pub(crate) fn ended(&mut self, hook: HookId) {
        if self.current == State::Exiting(hook) {
            self.current = State::Exited;
        }
    }

    /// The close: an exit still running is no longer waited on.
    pub(crate) fn abandon(&mut self) {
        if let State::Exiting(hook) = self.current {
            self.current = State::Abandoned(hook);
        }
    }

    /// The swap: the arriving agent owes its own `on_exit`.
    pub(crate) fn swapped(&mut self) {
        self.current = State::Running;
    }
}

/// The driving agent being left: its `on_exit`, claimed by the first to leave it.
pub(crate) struct Leave<'a> {
    pub(crate) on_exit: &'a mut OnExit,
    pub(crate) hooks: &'a mut Hooks,
    pub(crate) agent: &'a str,
}

impl Leave<'_> {
    /// The exit owed, running from now on when it is issued here: asked of the host as soon as
    /// the claimant returns, before anything it raised.
    pub(crate) fn claim(&mut self) -> Exit {
        match self.on_exit.current {
            State::Running => {
                let hook = self.hooks.mint();
                self.on_exit.current = State::Exiting(hook);
                self.on_exit.claimed.push((hook, self.agent.to_owned()));
                Exit::Issue(hook)
            }
            State::Exiting(hook) => Exit::Wait(hook),
            State::Abandoned(_) | State::Exited => Exit::Done,
        }
    }
}
