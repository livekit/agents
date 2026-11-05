//! `drain()`: the host waits for the driving agent to say its goodbye (`on_exit`) and for
//! scheduling to pause with nothing left to say; the session stays open.

use super::Cx;
use crate::agent_activity::hooks::Exit;
use crate::agent_session::machine::coordination::PauseResult;
use crate::agent_session::{DrainRefused, HookId, RequestId};

/// The driving agent's drain, and the callers whose agent was left before theirs finished.
#[derive(Default)]
pub(crate) struct Drain {
    current: Option<AgentDrain>,
    /// Answered `Ok` once no speech is open.
    left: Vec<RequestId>,
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
    /// The pause settled with a speech still open: wait for its end.
    Settling,
    /// Exited and paused: the agent stays, idle, until a close or a new agent.
    Drained,
}

pub(crate) enum DrainEvent {
    /// `drain()`, for these callers (one request, or the ones a handoff handed back).
    Asked(Vec<RequestId>),
    Paused(PauseResult),
    /// No speech is open any more.
    Quiet,
    /// The swap leaves the agent.
    Leaving,
    /// The handoff it joined ended without a swap: its callers' drain runs on this agent.
    AgentStays,
    /// The close: a drain still waiting on its `on_exit` is refused.
    Closing,
}

/// Milestones produced by the drain owner. The driver records them in this order before
/// dispatching deferred scheduling work; the drain never writes to the runtime outbox.
pub(crate) enum DrainOutcome {
    Started { exit_hook: Option<HookId> },
    Finished { waiters: Vec<RequestId>, refused: Option<DrainRefused> },
    PauseScheduling,
}

impl Drain {
    #[must_use]
    pub(super) fn handle(&mut self, ev: DrainEvent, cx: Cx<'_>) -> Vec<DrainOutcome> {
        let mut outcomes = Vec::new();
        self.handle_outcomes(ev, cx, &mut outcomes);
        outcomes
    }

    fn handle_outcomes(&mut self, ev: DrainEvent, mut cx: Cx<'_>, out: &mut Vec<DrainOutcome>) {
        match ev {
            DrainEvent::Asked(waiters) => self.asked(waiters, &mut cx, out),
            DrainEvent::Paused(result) => self.paused(result, &cx, out),
            DrainEvent::Quiet => {
                self.left_ready(true, out);
                if let Some(AgentDrain { phase: Phase::Settling, .. }) = self.current {
                    self.finish(None, out);
                }
            }
            DrainEvent::Leaving => self.leaving(&mut cx, out),
            DrainEvent::AgentStays => self.agent_stays(&mut cx, out),
            DrainEvent::Closing => {
                if let Some(AgentDrain { phase: Phase::Exiting { .. }, .. }) = self.current {
                    self.finish(Some(DrainRefused::Closing), out);
                }
            }
        }
    }

    /// Scheduling paused: a quiet settle finishes the drain, one with a speech open waits for
    /// its end, and a refusal ends it refused.
    fn paused(&mut self, result: PauseResult, cx: &Cx<'_>, out: &mut Vec<DrainOutcome>) {
        let Some(AgentDrain { phase: phase @ Phase::Pausing, .. }) = &mut self.current else {
            return;
        };
        match result {
            PauseResult::Settled if cx.quiet => self.finish(None, out),
            PauseResult::Settled => *phase = Phase::Settling,
            PauseResult::Refused => {
                let refused =
                    if cx.closing { DrainRefused::Closing } else { DrainRefused::ActivityChanging };
                self.finish(Some(refused), out);
            }
        }
    }

    /// The handoff this drain joined ended without a swap: its callers' drain runs here.
    fn agent_stays(&mut self, cx: &mut Cx<'_>, out: &mut Vec<DrainOutcome>) {
        if cx.switching || !matches!(self.current, Some(AgentDrain { phase: Phase::Joined, .. })) {
            return;
        }
        let Some(drain) = self.current.take() else { return };
        if !drain.waiters.is_empty() {
            self.asked(drain.waiters, cx, out);
        }
    }

    /// The common hook owner finished the exit this drain waits on.
    #[must_use]
    pub(super) fn exit_hook_finished(&mut self, hook: HookId, quiet: bool) -> Vec<DrainOutcome> {
        let mut outcomes = Vec::new();
        self.exit_finished_outcomes(hook, quiet, &mut outcomes);
        outcomes
    }

    fn exit_finished_outcomes(&mut self, hook: HookId, quiet: bool, out: &mut Vec<DrainOutcome>) {
        // Pause after on_exit so any goodbye it scheduled is included in the drain.
        if let Some(AgentDrain { phase: phase @ Phase::Exiting { .. }, .. }) = &mut self.current
            && matches!(phase, Phase::Exiting { hook: h } if *h == hook)
        {
            *phase = Phase::Pausing;
            out.push(DrainOutcome::PauseScheduling);
        }
        self.left_ready(quiet, out);
    }

    /// A drain is waiting or done: the agent stays idle until a close or a new agent.
    pub(crate) fn active(&self) -> bool {
        self.current.is_some()
    }

    /// Every caller still waiting, taken: a left agent's first, then the driving agent's (whose
    /// drain is forgotten).
    pub(crate) fn take_waiters(&mut self) -> Vec<RequestId> {
        let mut waiters = std::mem::take(&mut self.left);
        if let Some(drain) = self.current.take() {
            waiters.extend(drain.waiters);
        }
        waiters
    }

    /// Never refused while running; it waits where the reference's `_lock` would: on a handoff's
    /// swap, or the agent's `on_exit` then the pause.
    fn asked(&mut self, waiters: Vec<RequestId>, cx: &mut Cx<'_>, out: &mut Vec<DrainOutcome>) {
        let Some(drain) = &mut self.current else {
            if cx.switching {
                out.push(DrainOutcome::Started { exit_hook: cx.leave.on_exit.exiting() });
                self.current = Some(AgentDrain { phase: Phase::Joined, waiters });
                return;
            }
            let exit = cx.leave.claim();
            out.push(DrainOutcome::Started { exit_hook: exit.hook() });
            let phase = match exit.hook() {
                Some(hook) => Phase::Exiting { hook },
                None => Phase::Pausing,
            };
            self.current = Some(AgentDrain { phase, waiters });
            if exit == Exit::Done {
                out.push(DrainOutcome::PauseScheduling);
            }
            return;
        };
        match drain.phase {
            // Already exited and paused: answered now, or once refused speeches are released;
            // under a handoff, at the swap.
            Phase::Drained if !cx.switching && cx.quiet => {
                for call in waiters {
                    out.push(DrainOutcome::Finished { waiters: vec![call], refused: None });
                }
            }
            Phase::Drained if !cx.switching => {
                drain.phase = Phase::Settling;
                drain.waiters.extend(waiters);
            }
            _ => drain.waiters.extend(waiters),
        }
    }

    /// The agent is left: its callers are answered `Ok` once nothing it said is open.
    fn leaving(&mut self, cx: &mut Cx<'_>, out: &mut Vec<DrainOutcome>) {
        let Some(drain) = self.current.take() else { return };
        if drain.waiters.is_empty() {
            return;
        }
        self.left.extend(drain.waiters);
        self.left_ready(cx.quiet, out);
    }

    /// Callers whose agent was left, answered `Ok` once nothing is open.
    fn left_ready(&mut self, quiet: bool, out: &mut Vec<DrainOutcome>) {
        if !quiet || self.left.is_empty() {
            return;
        }
        out.push(DrainOutcome::Finished { waiters: std::mem::take(&mut self.left), refused: None });
    }

    /// Every waiter answered. `None` leaves the agent `Drained`; a refusal forgets the drain.
    fn finish(&mut self, refused: Option<DrainRefused>, out: &mut Vec<DrainOutcome>) {
        let Some(drain) = self.current.as_mut() else { return };
        let waiters = std::mem::take(&mut drain.waiters);
        match refused {
            None => drain.phase = Phase::Drained,
            Some(_) => self.current = None,
        }
        out.push(DrainOutcome::Finished { waiters, refused });
    }
}
