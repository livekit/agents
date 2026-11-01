//! The developer's callbacks the driver awaits (`on_enter`, `on_exit`, `on_user_turn_completed`,
//! a tool reply's rendering, a filler's source, the `function_tools_executed` handlers), and how
//! long each is waited on.

use crate::agent::TurnEdit;
use crate::agent_session::effects::{Out, Signal, Timer};
use crate::agent_session::{Driver, HookCall, HookId, HookKind, HookOutcome, HostEvent};
use crate::audio_recognition::PendingTurn;
use crate::run_result::Owned;
use crate::trace::{CoreStep, HandoffStep, HookEnd, TurnStep};
use lk_agents_abi::HOST_TEARDOWN_TIMEOUT;
use lk_agents_abi::Ts;
use lk_agents_abi::voice::events::AgentState;
use std::collections::BTreeMap;
use std::time::Duration;

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
        let (agent, speech, turn) = match &call {
            HookCall::OnEnter { agent, .. }
            | HookCall::OnExit { agent, .. }
            | HookCall::OnUserTurnExceeded { agent, .. } => (Some(agent.clone()), None, None),
            HookCall::OnUserTurnCompleted { turn, .. } => (None, None, Some(*turn)),
            HookCall::FillerSource { .. } => (None, None, None),
            HookCall::FunctionToolsExecuted { speech, .. } => (None, Some(*speech), None),
        };
        out.trace(HandoffStep::HookIssued { hook: id, kind, agent, speech, turn });
        let hosted = self.activity.hook_bound.is_some();
        if !hosted {
            out.trace(HandoffStep::HookFinished { hook: id, kind, how: HookEnd::NoHost });
            return false;
        }
        out.machine().push(crate::agent_session::effects::Effect::Host(HostEvent::Hook(call)));
        self.activity.hooks.wait_for(id, kind);
        let mut spans = Vec::new();
        let wall = self.step.wall_secs_at(now);
        self.session.trace.hook_issued(wall, id, kind, &self.activity.agent, &mut spans);
        spans.into_iter().for_each(|ev| out.machine().emit(ev));
        if let Some(wait) = wait {
            out.machine().arm(Timer::Hook(id), now.saturating_add(wait));
        }
        // An open run owns the agent's `on_enter`.
        match kind {
            HookKind::OnEnter => self.run_own(Owned::Hook(id), out),
            HookKind::OnExit
            | HookKind::OnUserTurnCompleted
            | HookKind::OnUserTurnExceeded
            | HookKind::FillerSource
            | HookKind::FunctionToolsExecuted => {}
        }
        true
    }

    /// Nothing waits on it.
    pub(crate) fn issue_on_enter(&mut self, now: Ts, out: &mut Out) {
        let id = self.activity.hooks.mint();
        let agent = self.activity.agent.clone();
        self.issue_hook(
            now,
            HookCall::OnEnter { id, agent, scope: self.activity.scope() },
            None,
            out,
        );
    }

    /// The turn parks on it until answered, as long as that takes (`await on_user_turn_completed`).
    pub(crate) fn issue_turn_hook(
        &mut self,
        now: Ts,
        info: crate::audio_recognition::EndOfTurnInfo,
        out: &mut Out,
    ) {
        let id = self.activity.hooks.mint();
        let call = HookCall::OnUserTurnCompleted {
            id,
            agent: self.activity.agent.clone(),
            scope: self.activity.scope(),
            chat_ctx: self.session.chat_ctx.agent_chat_ctx().items,
            turn: info.turn_id,
            new_transcript: info.new_transcript.clone(),
            transcript_confidence: info.transcript_confidence,
        };
        if self.issue_hook(now, call, None, out) {
            self.activity.pending_turn = Some(PendingTurn { info, issued_at: now, hook: Some(id) });
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
        let mut spans = Vec::new();
        let wall = self.step.wall_secs_at(self.step.now);
        self.session.trace.hook_finished(wall, id, matches!(how, HookEnd::Failed), &mut spans);
        spans.into_iter().for_each(|ev| out.machine().emit(ev));
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
        use HookOutcome::{Failed, Filler, FunctionToolsExecuted, Returned, Turn};
        let fits = matches!(
            (kind, &outcome),
            (
                HookKind::OnExit | HookKind::OnEnter | HookKind::OnUserTurnExceeded,
                Returned | Failed
            ) | (HookKind::OnUserTurnCompleted, Turn(_) | Returned | Failed)
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
            HookKind::OnUserTurnExceeded => {
                if self.activity.turn_limit == TurnLimit::Running(id) {
                    self.activity.turn_limit = TurnLimit::Idle;
                }
            }
            // "error occurred during on_user_turn_completed": return, no reply.
            HookKind::OnUserTurnCompleted => {
                let edits = match outcome {
                    Turn(edits) => edits,
                    Failed if fits => vec![TurnEdit::StopResponse],
                    Failed | Returned | Filler { .. } | FunctionToolsExecuted { .. } => Vec::new(),
                };
                // Matched to its parked turn; a late answer is dropped.
                if let Some(pending) = self.activity.pending_turn.take_if(|p| p.hook == Some(id)) {
                    self.finish_turn(now, pending.finished(now), edits, out);
                    self.next_turn(now, out);
                }
            }
            // The filler's callable: text, a speech it made itself, or nothing; `None` when it
            // raised or its bound passed.
            HookKind::FillerSource => {
                let answered = match outcome {
                    Filler { text, spoke } => Some((text, spoke)),
                    Failed => None,
                    Returned | Turn(_) | FunctionToolsExecuted { .. } => Some((None, false)),
                };
                self.on_filler_source(now, id, answered, out)
            }
            // What the handlers left; unanswered, the event stands as issued.
            HookKind::FunctionToolsExecuted => {
                let left = match outcome {
                    FunctionToolsExecuted { reply_required } => Some(reply_required),
                    Failed | Returned | Turn(_) | Filler { .. } => None,
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
    claimed: Vec<(HookId, String, u64)>,
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

    pub(crate) fn take_claimed(&mut self) -> Vec<(HookId, String, u64)> {
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
    pub(crate) scope: u64,
}

impl Leave<'_> {
    /// The exit owed, running from now on when it is issued here: asked of the host as soon as
    /// the claimant returns, before anything it raised.
    pub(crate) fn claim(&mut self) -> Exit {
        match self.on_exit.current {
            State::Running => {
                let hook = self.hooks.mint();
                self.on_exit.current = State::Exiting(hook);
                self.on_exit.claimed.push((hook, self.agent.to_owned(), self.scope));
                Exit::Issue(hook)
            }
            State::Exiting(hook) => Exit::Wait(hook),
            State::Abandoned(_) | State::Exited => Exit::Done,
        }
    }
}

/// `on_user_turn_exceeded` (`_user_turn_exceeded_task`): waiting for the reply the end of turn may
/// bring, or the developer's callback running.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) enum TurnLimit {
    #[default]
    Idle,
    Waiting(lk_agents_abi::voice::events::UserTurnExceededEvent),
    Running(HookId),
}

/// Why a user turn past its limit gets no `on_user_turn_exceeded`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, lk_agents_abi::Vocabulary)]
pub enum TurnLimitSkip {
    /// Scheduling is paused, a handoff blocks new turns, or the session closes.
    Paused,
    /// The callback is still running.
    Busy,
    /// The agent started speaking: the end of turn replied.
    AgentSpeaking,
}

impl Driver {
    /// `on_user_turn_exceeded`: a new crossing replaces the wait still running.
    pub(crate) fn on_user_turn_exceeded(
        &mut self,
        ev: &lk_agents_abi::voice::events::UserTurnExceededEvent,
        out: &mut Out,
    ) {
        let skip = if self.turn_limit_blocked() {
            Some(TurnLimitSkip::Paused)
        } else if matches!(self.activity.turn_limit, TurnLimit::Running(_)) {
            return out.trace(TurnStep::TurnLimitSkipped(TurnLimitSkip::Busy));
        } else if self.session.states.agent_state() == AgentState::Speaking {
            Some(TurnLimitSkip::AgentSpeaking)
        } else {
            None
        };
        self.activity.turn_limit = match skip {
            Some(skip) => {
                out.trace(TurnStep::TurnLimitSkipped(skip));
                TurnLimit::Idle
            }
            None => TurnLimit::Waiting(ev.clone()),
        };
    }

    /// The wait ends once no reply is decided, queued or speaking: the callback runs, unless a
    /// handoff or a pause began meanwhile.
    pub(crate) fn settle_turn_limit(&mut self, now: Ts, out: &mut Out) {
        if !matches!(self.activity.turn_limit, TurnLimit::Waiting(_)) {
            return;
        }
        if self.turn_limit_blocked() {
            self.activity.turn_limit = TurnLimit::Idle;
            return out.trace(TurnStep::TurnLimitSkipped(TurnLimitSkip::Paused));
        }
        if !self.activity.is_idle() {
            return;
        }
        let TurnLimit::Waiting(event) = std::mem::take(&mut self.activity.turn_limit) else {
            return;
        };
        let id = self.activity.hooks.mint();
        let agent = self.activity.agent.clone();
        if self.issue_hook(
            now,
            HookCall::OnUserTurnExceeded { id, agent, event, scope: self.activity.scope() },
            None,
            out,
        ) {
            self.activity.turn_limit = TurnLimit::Running(id);
        }
    }

    /// `_scheduling_paused or _new_turns_blocked`, or the activity closing.
    fn turn_limit_blocked(&self) -> bool {
        self.activity.speech_scheduler.is_paused()
            || self.activity.in_window()
            || self.session.closing.is_some()
    }
}

impl super::AgentActivity {
    pub(crate) fn hook_wait(&self) -> Duration {
        self.hook_bound.unwrap_or(HOST_TEARDOWN_TIMEOUT)
    }

    /// The agent started speaking: the wait ends with no callback.
    pub(crate) fn turn_limit_agent_speaking(&mut self, out: &mut Out) {
        if matches!(self.turn_limit, TurnLimit::Waiting(_)) {
            self.turn_limit = TurnLimit::Idle;
            out.trace(TurnStep::TurnLimitSkipped(TurnLimitSkip::AgentSpeaking));
        }
    }

    /// The activity ends (a swap, the close's edge): the wait raises `ActivityClosedError`, a
    /// running callback is abandoned.
    pub(crate) fn refuse_turn_limit(&mut self, out: &mut Out) {
        if matches!(std::mem::take(&mut self.turn_limit), TurnLimit::Waiting(_)) {
            out.trace(TurnStep::TurnLimitSkipped(TurnLimitSkip::Paused));
        }
    }
}
