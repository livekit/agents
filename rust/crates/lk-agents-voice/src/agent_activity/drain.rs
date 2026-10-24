//! `drain()`: the host waits for the driving agent to say its goodbye (`on_exit`) and for
//! scheduling to pause with nothing left to say; the session stays open.

use crate::agent_activity::handoff::FrameExit;
use crate::agent_activity::scheduling::PauseCause;
use crate::agent_session::effects::Out;
use crate::agent_session::{Answer, DrainRefused, Driver, HookId, PreviousActivity, RequestId};
use crate::trace::HandoffStep;
use lk_agents_abi::Ts;

/// The driving agent's drain, and the callers whose agent was left before theirs finished.
#[derive(Default)]
pub(crate) struct Drain {
    current: Option<AgentDrain>,
    /// Answered `Ok` once no speech is open and, for an agent an inline task paused
    /// (`Some(hook)`), once its `on_exit` ended.
    left: Vec<(Option<HookId>, RequestId)>,
}

struct AgentDrain {
    phase: Phase,
    /// Every caller still waiting; answered together.
    waiters: Vec<RequestId>,
}

enum Phase {
    /// The agent may still speak its goodbye (`None`: it owes no `on_exit`).
    Exiting { hook: Option<HookId> },
    /// A handoff in progress is leaving the agent: callers wait for its swap, as `drain()` waits
    /// on the lock the switch holds.
    Joined,
    /// `settled` answers when nothing is outstanding, or closes when refused.
    Pausing { settled: tokio::sync::oneshot::Receiver<()> },
    /// The pause settled with a speech blocked on an inline task still open: wait for its end.
    /// (The reference's drain deadlocks here on the task's `pause()`.)
    Settling,
    /// Exited and paused: the agent stays, idle, until a close or a new agent.
    Drained,
}

/// Where the drain's pause is.
enum Poll {
    Waiting,
    /// Settled with nothing open: answered `Ok`.
    Settled,
    /// Refused: answered with the reason.
    Refused,
}

impl Drain {
    /// Every caller still waiting, taken: a left agent's first, then the driving agent's (whose
    /// drain is forgotten).
    pub(crate) fn take_waiters(&mut self) -> Vec<RequestId> {
        let mut waiters: Vec<_> =
            std::mem::take(&mut self.left).into_iter().map(|(_, w)| w).collect();
        if let Some(drain) = self.current.take() {
            waiters.extend(drain.waiters);
        }
        waiters
    }

    /// The drain waits on its agent's `on_exit` still.
    pub(crate) fn exiting(&self) -> bool {
        matches!(self.current, Some(AgentDrain { phase: Phase::Exiting { .. }, .. }))
    }

    /// The drain answers: its callers. `None` leaves the agent `Drained`; a refusal forgets it.
    fn finish(&mut self, refused: Option<DrainRefused>) -> Option<Vec<RequestId>> {
        let drain = self.current.as_mut()?;
        let waiters = std::mem::take(&mut drain.waiters);
        match refused {
            None => drain.phase = Phase::Drained,
            Some(_) => self.current = None,
        }
        Some(waiters)
    }

    /// Has the pause settled, with nothing open (`quiet`)?
    fn poll(&mut self, quiet: bool) -> Poll {
        let Some(drain) = &mut self.current else { return Poll::Waiting };
        let settled = match &mut drain.phase {
            Phase::Settling if quiet => return Poll::Settled,
            Phase::Pausing { settled } => settled,
            Phase::Exiting { .. } | Phase::Joined | Phase::Settling | Phase::Drained => {
                return Poll::Waiting;
            }
        };
        match settled.try_recv() {
            Ok(()) if quiet => Poll::Settled,
            Ok(()) => {
                drain.phase = Phase::Settling;
                Poll::Waiting
            }
            Err(tokio::sync::oneshot::error::TryRecvError::Empty) => Poll::Waiting,
            Err(tokio::sync::oneshot::error::TryRecvError::Closed) => Poll::Refused,
        }
    }

    /// Callers of a left agent waiting on nothing but quiet (none while any waits on an
    /// `on_exit`), taken.
    fn take_left_ready(&mut self) -> Vec<RequestId> {
        let (ready, waiting) =
            std::mem::take(&mut self.left).into_iter().partition(|(h, _)| h.is_none());
        self.left = waiting;
        ready.into_iter().map(|(_, call)| call).collect()
    }
}

impl Driver {
    /// `AgentActivity.drain()`, for `waiters` (one request, or the ones a handoff handed on).
    /// Never refused while running; it waits where the reference's `_lock` would: on the close's
    /// drain, a handoff's swap, or the agent's `on_exit` then the pause.
    pub(crate) fn drain(&mut self, now: Ts, waiters: Vec<RequestId>, out: &mut Out) {
        if self.session.closing.is_some() {
            return self.join_close_drain(waiters, out);
        }
        let quiet = self.activity.speech_scheduler.is_empty();
        let switching = self.activity.handoff.switching();
        let Some(drain) = &mut self.activity.drain.current else {
            // A handoff in progress: these callers wait for its `on_exit` and pause.
            if switching {
                let exit_hook = self.activity.handoff.exit.exiting();
                self.activity.drain.current = Some(AgentDrain { phase: Phase::Joined, waiters });
                out.trace(HandoffStep::DrainStarted { exit_hook });
                return;
            }
            // Resumed after its `on_exit` ran (`Exited`): only the pause is left.
            let exit = self.frame_exit();
            let phase = Phase::Exiting { hook: exit.hook() };
            self.activity.drain.current = Some(AgentDrain { phase, waiters });
            out.trace(HandoffStep::DrainStarted { exit_hook: exit.hook() });
            if self.issue_exit(now, exit, out) {
                self.drain_pause(now, out);
            }
            return;
        };
        match drain.phase {
            // Already exited and paused: answered now, or once refused speeches are released;
            // under a handoff, at the swap.
            Phase::Drained if !switching && quiet => {
                for call in waiters {
                    self.answer_drain(Some(call), None, out);
                }
            }
            Phase::Drained if !switching => {
                drain.phase = Phase::Settling;
                drain.waiters.extend(waiters);
            }
            _ => drain.waiters.extend(waiters),
        }
    }

    /// Left for good (`Close`): answered `Ok` once nothing it said is open. Paused under an inline
    /// task: the paused agent's `on_exit` runs now, once, before the task's `on_enter`, as the
    /// reference does.
    pub(crate) fn settle_drain_at_swap(
        &mut self,
        now: Ts,
        previous: PreviousActivity,
        out: &mut Out,
    ) {
        let Some(drain) = self.activity.drain.current.take() else { return };
        if drain.waiters.is_empty() {
            return;
        }
        // The `on_exit` the callers still wait for, if any.
        let waiting_on = match (previous, self.activity.handoff.exit) {
            (PreviousActivity::Close, _) | (PreviousActivity::Pause, FrameExit::Exited) => None,
            (PreviousActivity::Pause, FrameExit::Exiting(hook)) => Some(hook),
            (PreviousActivity::Pause, FrameExit::Running) => {
                let hook = self.activity.hooks.mint();
                out.trace(HandoffStep::DrainStarted { exit_hook: Some(hook) });
                (!self.issue_on_exit(now, hook, out)).then_some(hook)
            }
        };
        self.activity.drain.left.extend(drain.waiters.into_iter().map(|w| (waiting_on, w)));
        self.left_drains_ready(out);
    }

    /// A joined handoff ended without a swap: the callers' drain runs on this agent now.
    pub(crate) fn rebind_joined_drain(&mut self, now: Ts, out: &mut Out) {
        if self.activity.handoff.switching()
            || !matches!(self.activity.drain.current, Some(AgentDrain { phase: Phase::Joined, .. }))
        {
            return;
        }
        let Some(drain) = self.activity.drain.current.take() else { return };
        if !drain.waiters.is_empty() {
            self.drain(now, drain.waiters, out);
        }
    }

    /// Callers whose agent was left for good, answered `Ok` once no speech is open.
    fn left_drains_ready(&mut self, out: &mut Out) {
        if !self.activity.speech_scheduler.is_empty() {
            return;
        }
        let ready = self.activity.drain.take_left_ready();
        if ready.is_empty() {
            return;
        }
        out.trace(HandoffStep::Drained { refused: None });
        for call in ready {
            self.answer(Some(call), Answer::Drain(Ok(())));
        }
    }

    /// The `on_exit` `hook` ended: a drain waiting on it pauses scheduling, and callers bound to
    /// the paused agent wait only for nothing to be open.
    pub(crate) fn drain_exit_done(&mut self, now: Ts, hook: HookId, out: &mut Out) {
        if let Some(AgentDrain { phase: Phase::Exiting { hook: Some(h) }, .. }) =
            self.activity.drain.current
            && h == hook
        {
            self.drain_pause(now, out);
        }
        for (waits_on, _) in &mut self.activity.drain.left {
            if *waits_on == Some(hook) {
                *waits_on = None;
            }
        }
        self.left_drains_ready(out);
    }

    /// `_pause_scheduling_task()`, after `on_exit`, so a goodbye it scheduled is waited for.
    fn drain_pause(&mut self, now: Ts, out: &mut Out) {
        let Some(drain) = &mut self.activity.drain.current else { return };
        let (done, settled) = tokio::sync::oneshot::channel();
        drain.phase = Phase::Pausing { settled };
        self.pause_scheduling(now, PauseCause::Drain, done, out);
    }

    /// Has the drain's pause settled? Settled: waiters answered `Ok`; refused: with the reason.
    pub(crate) fn drain_ready(&mut self, out: &mut Out) {
        self.left_drains_ready(out);
        let quiet = self.activity.speech_scheduler.is_empty();
        match self.activity.drain.poll(quiet) {
            Poll::Settled => self.finish_drain(None, out),
            Poll::Waiting => {}
            Poll::Refused => {
                let refused = if self.session.closing.is_some() {
                    DrainRefused::Closing
                } else {
                    DrainRefused::FrameChanging
                };
                self.finish_drain(Some(refused), out);
            }
        }
    }

    /// Answer every waiter. `None` leaves the agent `Drained`; a refusal forgets the drain.
    pub(crate) fn finish_drain(&mut self, refused: Option<DrainRefused>, out: &mut Out) {
        let Some(waiters) = self.activity.drain.finish(refused) else { return };
        out.trace(HandoffStep::Drained { refused });
        for call in waiters {
            self.answer(Some(call), Answer::Drain(refused.map_or(Ok(()), Err)));
        }
    }

    /// One caller answered on its own: a refusal at the request, or an agent already drained.
    pub(crate) fn answer_drain(
        &mut self,
        call: Option<RequestId>,
        refused: Option<DrainRefused>,
        out: &mut Out,
    ) {
        out.trace(HandoffStep::Drained { refused });
        self.answer(call, Answer::Drain(refused.map_or(Ok(()), Err)));
    }
}
