//! What one part of the session tells another within a step: raised into [`Out`], dispatched in
//! the order raised until none is left, never seen outside the step.

use crate::agent_activity::drain::{Drain, DrainEvent};
use crate::agent_activity::handoff::{Handoff, HandoffEvent, HandoffId};
use crate::agent_activity::on_exit::Leave;
use crate::agent_activity::scheduling::PauseCause;
use crate::agent_session::effects::Out;
use crate::agent_session::{
    AgentUpdate, ClosePhase, Driver, HandoffRefused, HookCall, HookId, RequestId,
};
use crate::run_result::Owned;

pub(crate) enum Signal {
    /// `_pause_scheduling_task` for `cause`, answered with `Paused`.
    PauseScheduling(PauseCause),
    Paused(PauseCause, PauseResult),
    /// The last speech held was released.
    Quiet,
    /// An `on_exit` ended: answered, timed out, or with nobody to ask.
    OnExitEnded(HookId),
    RunOwn(Owned),
    /// A switch begins: an inline task entering captures its parent now.
    HandoffStarted,
    /// The switch's drain settled: the swap is due.
    Swap,
    RefuseHandoff {
        id: Option<HandoffId>,
        update: Box<AgentUpdate>,
        cause: HandoffRefused,
    },
    /// The switch in progress ended without a swap.
    AgentStays,
    /// Outside a switch, a held user turn reaches the agent.
    DeliverHeldTurn,
    NextHandoff,
    InterruptForDtmf,
    /// The digits read back for the caller to confirm.
    ReadBackDtmf(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PauseResult {
    Settled,
    Refused,
}

/// What a machine reads of the rest of the session, and the agent it may leave.
pub(crate) struct Cx<'a> {
    /// No speech is held.
    pub(crate) quiet: bool,
    pub(crate) closing: bool,
    pub(crate) switching: bool,
    pub(crate) leave: Leave<'a>,
}

impl Driver {
    pub(crate) fn dispatch(&mut self, signal: Signal, out: &mut Out) {
        let now = self.step.now;
        match signal {
            Signal::PauseScheduling(cause) => self.pause_scheduling(now, cause, out),
            Signal::Paused(PauseCause::Handoff, result) => {
                self.handoff_step(HandoffEvent::Paused(result), out)
            }
            Signal::Paused(PauseCause::Drain, result) => {
                self.drain_step(DrainEvent::Paused(result), out)
            }
            // Settled or refused, the close's drain is over.
            Signal::Paused(PauseCause::Close, _) => {
                if matches!(self.session.close_phase(), Some(ClosePhase::Draining { .. })) {
                    self.close_edge(now, out);
                }
            }
            Signal::Quiet => self.drain_step(DrainEvent::Quiet, out),
            Signal::OnExitEnded(hook) => {
                self.activity.on_exit.ended(hook);
                self.handoff_step(HandoffEvent::OnExitEnded(hook), out);
                self.drain_step(DrainEvent::OnExitEnded(hook), out);
                self.close_exit_done(now, hook, out);
            }
            Signal::RunOwn(what) => self.run_own(what, out),
            Signal::HandoffStarted => self.capture_task_parent(),
            Signal::Swap => self.perform_handoff(now, out),
            Signal::RefuseHandoff { id, update, cause } => {
                self.refuse_handoff(id, update, cause, out)
            }
            Signal::AgentStays => self.drain_step(DrainEvent::AgentStays, out),
            Signal::DeliverHeldTurn => self.deliver_held_turn(now, out),
            Signal::NextHandoff => {
                self.handoff_step(HandoffEvent::Next, out);
                self.pump_schedule(now, out);
            }
            Signal::InterruptForDtmf => self.interrupt_for_dtmf(now, out),
            Signal::ReadBackDtmf(digits) => self.read_back_dtmf(now, digits, out),
        }
    }

    /// `drain()`: under a close, its callers wait on the close's own drain.
    pub(crate) fn drain(&mut self, waiters: Vec<RequestId>, out: &mut Out) {
        if self.session.closing.is_some() {
            return self.join_close_drain(waiters, out);
        }
        self.drain_step(DrainEvent::Asked(waiters), out);
    }

    pub(crate) fn handoff_step(&mut self, ev: HandoffEvent, out: &mut Out) {
        let (cx, handoff, _) = self.machines();
        handoff.handle(ev, cx, out);
        self.issue_claimed_exits(out);
    }

    pub(crate) fn drain_step(&mut self, ev: DrainEvent, out: &mut Out) {
        let (cx, _, drain) = self.machines();
        drain.handle(ev, cx, out);
        self.issue_claimed_exits(out);
    }

    /// The `on_exit`s claimed in the last step, asked of the host; with none, over at once.
    pub(crate) fn issue_claimed_exits(&mut self, out: &mut Out) {
        let now = self.step.now;
        for (hook, agent) in self.activity.on_exit.take_claimed() {
            let wait = self.hook_wait();
            if !self.issue_hook(now, HookCall::OnExit { id: hook, agent }, Some(wait), out) {
                out.signal(Signal::OnExitEnded(hook));
            }
        }
    }

    fn machines(&mut self) -> (Cx<'_>, &mut Handoff, &mut Drain) {
        let closing = self.session.closing.is_some();
        let a = &mut self.activity;
        let cx = Cx {
            quiet: a.speech_scheduler.is_empty(),
            closing,
            switching: a.handoff.switching(),
            leave: Leave { on_exit: &mut a.on_exit, hooks: &mut a.hooks, agent: &a.agent },
        };
        (cx, &mut a.handoff, &mut a.drain)
    }

    /// The driving agent, for the first to leave it to claim its `on_exit`.
    pub(crate) fn leave(&mut self) -> Leave<'_> {
        let a = &mut self.activity;
        Leave { on_exit: &mut a.on_exit, hooks: &mut a.hooks, agent: &a.agent }
    }
}
