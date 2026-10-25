//! `drain()`: the host waits for the driving agent to say its goodbye (`on_exit`) and for
//! scheduling to pause with nothing left to say; the session stays open.

use crate::agent_activity::on_exit::Exit;
use crate::agent_activity::scheduling::PauseCause;
use crate::agent_session::effects::Out;
use crate::agent_session::signals::{Cx, PauseResult, Signal};
use crate::agent_session::{Answer, DrainRefused, HookId, PreviousActivity, RequestId};
use crate::trace::HandoffStep;

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
    /// The agent may still speak its goodbye.
    Exiting {
        hook: HookId,
    },
    /// A handoff in progress is leaving the agent: callers wait for its swap, as `drain()` waits
    /// on the lock the switch holds.
    Joined,
    Pausing,
    /// The pause settled with a speech blocked on an inline task still open: wait for its end.
    /// (The reference's drain deadlocks here on the task's `pause()`.)
    Settling,
    /// Exited and paused: the agent stays, idle, until a close or a new agent.
    Drained,
}

pub(crate) enum DrainEvent {
    /// `drain()`, for these callers (one request, or the ones a handoff handed back).
    Asked(Vec<RequestId>),
    OnExitEnded(HookId),
    Paused(PauseResult),
    /// No speech is open any more.
    Quiet,
    /// The swap leaves the agent: paused under an inline task, or for good.
    Leaving(PreviousActivity),
    /// The handoff it joined ended without a swap: its callers' drain runs on this agent.
    AgentStays,
    /// The close: a drain still waiting on its `on_exit` is refused.
    Closing,
}

impl Drain {
    pub(crate) fn handle(&mut self, ev: DrainEvent, mut cx: Cx<'_>, out: &mut Out) {
        match ev {
            DrainEvent::Asked(waiters) => self.asked(waiters, &mut cx, out),
            DrainEvent::OnExitEnded(hook) => {
                // `_pause_scheduling_task()` after `on_exit`, so a goodbye it scheduled is waited
                // for.
                if let Some(AgentDrain { phase: phase @ Phase::Exiting { .. }, .. }) =
                    &mut self.current
                    && matches!(phase, Phase::Exiting { hook: h } if *h == hook)
                {
                    *phase = Phase::Pausing;
                    out.signal(Signal::PauseScheduling(PauseCause::Drain));
                }
                for (waits_on, _) in &mut self.left {
                    if *waits_on == Some(hook) {
                        *waits_on = None;
                    }
                }
                self.left_ready(cx.quiet, out);
            }
            DrainEvent::Paused(result) => {
                let Some(AgentDrain { phase: phase @ Phase::Pausing, .. }) = &mut self.current
                else {
                    return;
                };
                match result {
                    PauseResult::Settled if cx.quiet => self.finish(None, out),
                    PauseResult::Settled => *phase = Phase::Settling,
                    PauseResult::Refused => {
                        let refused = if cx.closing {
                            DrainRefused::Closing
                        } else {
                            DrainRefused::ActivityChanging
                        };
                        self.finish(Some(refused), out);
                    }
                }
            }
            DrainEvent::Quiet => {
                self.left_ready(true, out);
                if let Some(AgentDrain { phase: Phase::Settling, .. }) = self.current {
                    self.finish(None, out);
                }
            }
            DrainEvent::Leaving(previous) => self.leaving(previous, &mut cx, out),
            DrainEvent::AgentStays => {
                if cx.switching
                    || !matches!(self.current, Some(AgentDrain { phase: Phase::Joined, .. }))
                {
                    return;
                }
                let Some(drain) = self.current.take() else { return };
                if !drain.waiters.is_empty() {
                    self.asked(drain.waiters, &mut cx, out);
                }
            }
            DrainEvent::Closing => {
                if let Some(AgentDrain { phase: Phase::Exiting { .. }, .. }) = self.current {
                    self.finish(Some(DrainRefused::Closing), out);
                }
            }
        }
    }

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

    /// Never refused while running; it waits where the reference's `_lock` would: on a handoff's
    /// swap, or the agent's `on_exit` then the pause.
    fn asked(&mut self, waiters: Vec<RequestId>, cx: &mut Cx<'_>, out: &mut Out) {
        let Some(drain) = &mut self.current else {
            if cx.switching {
                out.trace(HandoffStep::DrainStarted { exit_hook: cx.leave.on_exit.exiting() });
                self.current = Some(AgentDrain { phase: Phase::Joined, waiters });
                return;
            }
            let exit = cx.leave.claim();
            out.trace(HandoffStep::DrainStarted { exit_hook: exit.hook() });
            let phase = match exit.hook() {
                Some(hook) => Phase::Exiting { hook },
                None => Phase::Pausing,
            };
            self.current = Some(AgentDrain { phase, waiters });
            if exit == Exit::Done {
                out.signal(Signal::PauseScheduling(PauseCause::Drain));
            }
            return;
        };
        match drain.phase {
            // Already exited and paused: answered now, or once refused speeches are released;
            // under a handoff, at the swap.
            Phase::Drained if !cx.switching && cx.quiet => {
                for call in waiters {
                    out.trace(HandoffStep::Drained { refused: None });
                    out.answer(Some(call), Answer::Drain(Ok(())));
                }
            }
            Phase::Drained if !cx.switching => {
                drain.phase = Phase::Settling;
                drain.waiters.extend(waiters);
            }
            _ => drain.waiters.extend(waiters),
        }
    }

    /// Left for good (`Close`): answered `Ok` once nothing it said is open. Paused under an inline
    /// task: the paused agent's `on_exit` runs now, once, before the task's `on_enter`, as the
    /// reference does.
    fn leaving(&mut self, previous: PreviousActivity, cx: &mut Cx<'_>, out: &mut Out) {
        let Some(drain) = self.current.take() else { return };
        if drain.waiters.is_empty() {
            return;
        }
        let waiting_on = match previous {
            PreviousActivity::Close => None,
            PreviousActivity::Pause => match cx.leave.claim() {
                Exit::Issue(hook) => {
                    out.trace(HandoffStep::DrainStarted { exit_hook: Some(hook) });
                    Some(hook)
                }
                Exit::Wait(hook) => Some(hook),
                Exit::Done => None,
            },
        };
        self.left.extend(drain.waiters.into_iter().map(|w| (waiting_on, w)));
        self.left_ready(cx.quiet, out);
    }

    /// Callers whose agent was left, answered `Ok` once nothing is open and no `on_exit` is
    /// awaited.
    fn left_ready(&mut self, quiet: bool, out: &mut Out) {
        if !quiet {
            return;
        }
        let (ready, waiting): (Vec<_>, _) =
            std::mem::take(&mut self.left).into_iter().partition(|(h, _)| h.is_none());
        self.left = waiting;
        if ready.is_empty() {
            return;
        }
        out.trace(HandoffStep::Drained { refused: None });
        for (_, call) in ready {
            out.answer(Some(call), Answer::Drain(Ok(())));
        }
    }

    /// Every waiter answered. `None` leaves the agent `Drained`; a refusal forgets the drain.
    fn finish(&mut self, refused: Option<DrainRefused>, out: &mut Out) {
        let Some(drain) = self.current.as_mut() else { return };
        let waiters = std::mem::take(&mut drain.waiters);
        match refused {
            None => drain.phase = Phase::Drained,
            Some(_) => self.current = None,
        }
        out.trace(HandoffStep::Drained { refused });
        for call in waiters {
            out.answer(Some(call), Answer::Drain(refused.map_or(Ok(()), Err)));
        }
    }
}
