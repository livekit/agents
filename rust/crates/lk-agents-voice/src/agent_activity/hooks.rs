//! The developer's callbacks the driver awaits (`on_enter`, `on_exit`, `on_user_turn_completed`,
//! a tool reply's rendering, a filler's source, the `function_tools_executed` handlers), and how
//! long each is waited on.

use crate::agent::TurnEdit;
use crate::agent_session::effects::Timer;
use crate::agent_session::machine::coordination::VoiceCallback;
use crate::agent_session::{HookCall, HookId, HookKind, HookOutcome, VoiceDriver, VoiceHostEvent};
use crate::audio_recognition::PendingTurn;
use crate::trace::{CoreStep, HandoffStep, HookEnd, TurnStep};
use lk_agents_abi::HOST_TEARDOWN_TIMEOUT;
use lk_agents_abi::Ts;
use lk_agents_abi::voice::events::AgentState;
use std::collections::BTreeMap;
use std::time::Duration;

/// Callback identity stays live until its accepted completion has been dispatched. An exit's
/// continuation is deferred so its caller can install dependent close/handoff state first.
#[derive(Default)]
pub(crate) struct Hooks {
    pending: BTreeMap<HookId, HookRecord>,
    next: u64,
}

#[derive(Debug)]
struct HookRecord {
    kind: HookKind,
    hosted: bool,
    run: Option<crate::events::RunId>,
    completion: Option<(HookEnd, HookOutcome)>,
}

impl Hooks {
    pub(crate) fn has_run_work(&self, run: crate::events::RunId) -> bool {
        self.pending.values().any(|record| record.run == Some(run))
    }

    pub(crate) fn mint(&mut self) -> HookId {
        self.next += 1;
        HookId(self.next)
    }

    fn wait_for(&mut self, hook: HookId, kind: HookKind, hosted: bool) {
        self.pending.insert(hook, HookRecord { kind, hosted, run: None, completion: None });
    }

    /// First accepted result wins. The record remains live while its continuation is queued.
    fn accept(&mut self, hook: HookId, how: HookEnd, outcome: HookOutcome) -> Option<HookKind> {
        let record = self.pending.get_mut(&hook)?;
        if record.completion.is_some() {
            return None;
        }
        record.completion = Some((how, outcome));
        Some(record.kind)
    }

    fn take_finished(&mut self, hook: HookId) -> Option<HookRecord> {
        self.pending.get(&hook)?.completion.as_ref()?;
        self.pending.remove(&hook)
    }

    fn abandon(&mut self, hook: HookId) -> Option<HookRecord> {
        self.pending.remove(&hook)
    }

    /// Invalidate every callback before any abandonment fallback can admit more work.
    fn take_all(&mut self) -> Vec<(HookId, HookRecord)> {
        std::mem::take(&mut self.pending).into_iter().collect()
    }
}

impl VoiceDriver {
    /// Issue through the common lifecycle. Local outcomes need no host call or timeout; exit
    /// continuation is deferred until its caller has installed the state that waits for it.
    pub(crate) fn issue_hook(&mut self, now: Ts, mut call: HookCall, wait: Option<Duration>) {
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
        self.record_trace(HandoffStep::HookIssued { hook: id, kind, agent, speech, turn });
        let hosted = self.activity.hook_bound.is_some();
        self.activity.hooks.wait_for(id, kind, hosted);
        if kind == HookKind::OnEnter
            && let Some(run) = self.session.runs.current()
        {
            self.activity.hooks.pending.get_mut(&id).expect("hook just admitted").run =
                Some(run.id);
            self.session.runs.participated();
        }
        if !hosted {
            // A missing filler callable is a refusal, not a decline that can reschedule now.
            let outcome = if kind == HookKind::FillerSource {
                HookOutcome::Failed
            } else {
                HookOutcome::Returned
            };
            self.accept_hook_completion(now, id, HookEnd::NoHost, outcome);
            return;
        }
        // Its span opens first: the host runs the hook under it.
        let arriving = self.activity.handoff.arriving();
        let llm = self.session.llm.as_ref().map(|llm| &llm.provider);
        let agent = &self.activity.agent;
        call.set_span(self.session.spans.hook_issued(now, id, kind, agent, turn, arriving, llm));
        self.record_effect(crate::agent_session::effects::VoiceEffect::Host(VoiceHostEvent::Hook(
            call,
        )));
        if let Some(wait) = wait {
            self.set_deadline(Timer::Hook(id), now.saturating_add(wait));
        }
    }

    /// Issue enter without blocking activation; an open voice run still tracks its completion.
    pub(crate) fn issue_on_enter(&mut self, now: Ts) {
        let id = self.activity.hooks.mint();
        let agent = self.activity.agent.clone();
        let scope = self.activity.scope();
        self.issue_hook(now, HookCall::OnEnter { id, agent, scope, span: None }, None);
    }

    /// The turn parks on it until answered, as long as that takes (`await on_user_turn_completed`).
    pub(crate) fn issue_turn_hook(
        &mut self,
        now: Ts,
        info: crate::audio_recognition::EndOfTurnInfo,
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
            span: None,
        };
        self.activity.pending_turn = Some(PendingTurn { info, issued_at: now, hook: Some(id) });
        self.issue_hook(now, call, None);
    }

    pub(crate) fn on_hook_done(&mut self, now: Ts, id: HookId, outcome: HookOutcome) {
        let how = match &outcome {
            HookOutcome::Failed | HookOutcome::Raised(_) => HookEnd::Failed,
            HookOutcome::Turn(edits) => HookEnd::Returned { edits: edits.len() },
            HookOutcome::Returned
            | HookOutcome::Filler { .. }
            | HookOutcome::FunctionToolsExecuted { .. } => HookEnd::Returned { edits: 0 },
        };
        self.accept_hook_completion(now, id, how, outcome);
    }

    fn accept_hook_completion(&mut self, now: Ts, id: HookId, how: HookEnd, outcome: HookOutcome) {
        let Some(kind) = self.activity.hooks.accept(id, how, outcome) else { return };
        self.clear_deadline_if_set(Timer::Hook(id));
        if kind == HookKind::OnExit {
            self.defer_callback(VoiceCallback::HookFinished(id));
        } else {
            // These consumers were already synchronous; keep turn edits, filler refusal and
            // tools-executed handling ahead of any earlier queued cross-component callbacks.
            self.dispatch_hook_completion(now, id);
        }
    }

    /// One terminal record, one observation/release, then kind-specific continuation. A queued
    /// completion whose owner was abandoned is absent and cannot revive the old activity.
    pub(crate) fn dispatch_hook_completion(&mut self, now: Ts, id: HookId) {
        let Some(mut record) = self.activity.hooks.take_finished(id) else { return };
        let (how, outcome) = record.completion.take().expect("finished record has an outcome");
        self.account_hook_completion(id, &record, how, Some(&outcome));
        self.hook_ended(now, record.kind, id, outcome, how == HookEnd::NoHost);
    }

    fn account_hook_completion(
        &mut self,
        id: HookId,
        record: &HookRecord,
        how: HookEnd,
        outcome: Option<&HookOutcome>,
    ) {
        if record.hosted {
            use crate::telemetry;
            let failure = match (how, outcome) {
                (_, Some(HookOutcome::Raised(error))) => {
                    Some(telemetry::HookFailure::Raised(error))
                }
                (HookEnd::TimedOut, _) => Some(telemetry::HookFailure::TimedOut),
                (HookEnd::Failed, _) => Some(telemetry::HookFailure::Failed),
                _ => None,
            };
            let stop = matches!(outcome, Some(HookOutcome::Turn(edits))
                if edits.iter().any(|e| matches!(e, TurnEdit::StopResponse)));
            self.session.spans.hook_finished(self.step.now, id, failure, stop);
        }
        self.clear_deadline_if_set(Timer::Hook(id));
        self.record_trace(HandoffStep::HookFinished { hook: id, kind: record.kind, how });
    }

    /// Suppress continuation without overwriting a result already accepted for this hook.
    pub(crate) fn abandon_hook(&mut self, id: HookId) {
        let Some(record) = self.activity.hooks.abandon(id) else { return };
        self.account_abandoned_hook(id, record);
    }

    fn account_abandoned_hook(&mut self, id: HookId, record: HookRecord) {
        let how = record.completion.as_ref().map_or(HookEnd::Abandoned, |(how, _)| *how);
        let outcome = record.completion.as_ref().map(|(_, outcome)| outcome);
        self.account_hook_completion(id, &record, how, outcome);
    }

    /// All records are invalidated before releasing participation or running the one required
    /// tools-executed fallback. Accepted terminal provenance is retained even when abandoned.
    pub(crate) fn abandon_hooks(&mut self) {
        let records = self.activity.hooks.take_all();
        for (id, record) in &records {
            let how = record.completion.as_ref().map_or(HookEnd::Abandoned, |(how, _)| *how);
            let outcome = record.completion.as_ref().map(|(_, outcome)| outcome);
            self.account_hook_completion(*id, record, how, outcome);
        }
        for (id, record) in records {
            if record.kind == HookKind::FunctionToolsExecuted {
                self.on_tools_executed_handled(id, None);
            }
        }
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
        local: bool,
    ) {
        use HookOutcome::{Failed, Filler, FunctionToolsExecuted, Raised, Returned, Turn};
        let outcome = match outcome {
            HookOutcome::Raised(_) => Failed,
            outcome => outcome,
        };
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
            self.record_trace(CoreStep::PortFailure(
                "hook outcome does not match the call it answers",
            ));
        }
        match kind {
            HookKind::OnExit => {
                self.activity.exit_ended(id);
                if let Some(callback) = self.activity.handoff_exit_finished(id) {
                    self.defer_callback(callback);
                }
                self.issue_claimed_exits();
                let outcomes =
                    self.activity.drain_exit_finished(id, self.session.closing.is_some());
                self.record_drain_outcomes(outcomes);
                self.issue_claimed_exits();
                self.close_exit_done(now, id);
            }
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
                    Failed
                    | Returned
                    | Filler { .. }
                    | FunctionToolsExecuted { .. }
                    | Raised(_) => Vec::new(),
                };
                // Matched to its parked turn; a late answer is dropped.
                if let Some(pending) = self.activity.pending_turn.take_if(|p| p.hook == Some(id)) {
                    let info = if local { pending.info } else { pending.finished(now) };
                    self.finish_turn(now, info, edits);
                    if !local {
                        self.next_turn(now);
                    }
                }
            }
            // The filler's callable: text, a speech it made itself, or nothing; `None` when it
            // raised or its bound passed.
            HookKind::FillerSource => {
                let answered = match outcome {
                    Filler { text, spoke } => Some((text, spoke)),
                    Failed | Raised(_) => None,
                    Returned | Turn(_) | FunctionToolsExecuted { .. } => Some((None, false)),
                };
                self.on_filler_source(now, id, answered)
            }
            // What the handlers left; unanswered, the event stands as issued.
            HookKind::FunctionToolsExecuted => {
                let left = match outcome {
                    FunctionToolsExecuted { reply_required } => Some(reply_required),
                    Failed | Raised(_) | Returned | Turn(_) | Filler { .. } => None,
                };
                self.on_tools_executed_handled(id, left)
            }
        }
    }

    /// The bound passed: an unreturned `on_exit` lets the drain go on.
    pub(crate) fn hook_timed_out(&mut self, now: Ts, id: HookId) {
        self.accept_hook_completion(now, id, HookEnd::TimedOut, HookOutcome::Failed);
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

impl VoiceDriver {
    /// `on_user_turn_exceeded`: a new crossing replaces the wait still running.
    pub(crate) fn on_user_turn_exceeded(
        &mut self,
        ev: &lk_agents_abi::voice::events::UserTurnExceededEvent,
    ) {
        let skip = if self.turn_limit_blocked() {
            Some(TurnLimitSkip::Paused)
        } else if matches!(self.activity.turn_limit, TurnLimit::Running(_)) {
            return self.record_trace(TurnStep::TurnLimitSkipped(TurnLimitSkip::Busy));
        } else if self.session.states.agent_state() == AgentState::Speaking {
            Some(TurnLimitSkip::AgentSpeaking)
        } else {
            None
        };
        self.activity.turn_limit = match skip {
            Some(skip) => {
                self.record_trace(TurnStep::TurnLimitSkipped(skip));
                TurnLimit::Idle
            }
            None => TurnLimit::Waiting(ev.clone()),
        };
    }

    /// The wait ends once no reply is decided, queued or speaking: the callback runs, unless a
    /// handoff or a pause began meanwhile.
    pub(crate) fn settle_turn_limit(&mut self, now: Ts) {
        if !matches!(self.activity.turn_limit, TurnLimit::Waiting(_)) {
            return;
        }
        if self.turn_limit_blocked() {
            self.activity.turn_limit = TurnLimit::Idle;
            return self.record_trace(TurnStep::TurnLimitSkipped(TurnLimitSkip::Paused));
        }
        if !self.activity.is_idle() {
            return;
        }
        let TurnLimit::Waiting(event) = std::mem::take(&mut self.activity.turn_limit) else {
            return;
        };
        let id = self.activity.hooks.mint();
        let agent = self.activity.agent.clone();
        self.activity.turn_limit = TurnLimit::Running(id);
        self.issue_hook(
            now,
            HookCall::OnUserTurnExceeded { id, agent, event, scope: self.activity.scope() },
            None,
        );
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
    pub(crate) fn turn_limit_agent_speaking(&mut self) -> Option<TurnLimitSkip> {
        if matches!(self.turn_limit, TurnLimit::Waiting(_)) {
            self.turn_limit = TurnLimit::Idle;
            return Some(TurnLimitSkip::AgentSpeaking);
        }
        None
    }

    /// The activity ends (a swap, the close's edge): the wait raises `ActivityClosedError`, a
    /// running callback is abandoned.
    pub(crate) fn refuse_turn_limit(&mut self) -> Option<TurnLimitSkip> {
        if matches!(std::mem::take(&mut self.turn_limit), TurnLimit::Waiting(_)) {
            return Some(TurnLimitSkip::Paused);
        }
        None
    }
}

#[cfg(test)]
#[path = "hooks_tests.rs"]
mod tests;
