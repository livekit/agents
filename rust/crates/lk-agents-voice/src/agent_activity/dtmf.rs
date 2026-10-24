//! `GetDtmfTask`: collecting the digits the caller presses on the keypad, debounced, with an
//! optional read-back to confirm them.

use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::cycle::{Cycle, TurnTimer};
use crate::agent_session::{Answer, Driver, DtmfCommand, RequestId};
use crate::events::SessionEvent;
use crate::trace::{InterruptCause, InterruptStep};
use crate::workflows::builtins::dtmf::{DtmfAction, DtmfCollector, DtmfOutcome, DtmfReply};
use lk_agents_abi::Ts;

/// The collection while it runs.
pub(crate) struct DtmfCollection {
    pub(crate) collector: DtmfCollector,
    /// Answered once with the first outcome, or `Stopped`.
    pub(crate) call: Option<RequestId>,
    /// Whether `TurnTimer::DtmfDebounce` is armed, so a cancel is never a cancel of nothing.
    pub(crate) armed: bool,
    /// The confirm arm's read-back: the collector's guard is held until this speech is done.
    confirm: Option<SpeechId>,
}

impl DtmfCollection {
    /// A state edge the collection reads; the read-back's end releases its guard.
    fn on_event(&mut self, ev: &SessionEvent, now: Ts) -> Option<DtmfAction> {
        match ev {
            SessionEvent::AgentStateChanged { new, .. } => {
                Some(self.collector.on_agent_state_changed(new.as_str(), now))
            }
            SessionEvent::UserStateChanged { new, .. } => {
                Some(self.collector.on_user_state_changed(new.as_str(), now))
            }
            SessionEvent::SpeechDone { speech, .. } if self.confirm == Some(*speech) => {
                self.confirm = None;
                self.collector.finish_reply();
                None
            }
            _ => None,
        }
    }
}

impl Driver {
    pub(crate) fn on_dtmf(&mut self, now: Ts, cmd: DtmfCommand, cycle: &mut Cycle) {
        match cmd {
            DtmfCommand::Start { collector, call } => {
                self.dtmf_stop(cycle);
                self.activity.dtmf =
                    Some(DtmfCollection { collector, call, armed: false, confirm: None });
            }
            DtmfCommand::Digit { digit } => {
                let Some(dtmf) = &mut self.activity.dtmf else { return };
                let action = dtmf.collector.on_sip_dtmf(&digit, now);
                self.dtmf_perform(now, action, cycle);
            }
            DtmfCommand::Stop => self.dtmf_stop(cycle),
        }
    }

    /// The collection stops: its guard released (the host reading `dtmf_reply_running()` must not
    /// see a guard nobody will release), its debounce cancelled, an owed answer `Stopped`.
    pub(crate) fn dtmf_stop(&mut self, cycle: &mut Cycle) {
        let Some(mut dtmf) = self.activity.dtmf.take() else { return };
        dtmf.collector.finish_reply();
        if dtmf.armed {
            cycle.machine().cancel(TurnTimer::DtmfDebounce);
        }
        self.answer(dtmf.call, Answer::Dtmf(DtmfOutcome::Stopped));
    }

    /// A state edge on its way to the host: a speaker cancels the debounce, both quiet restarts
    /// it, and the read-back's end releases the guard.
    pub(crate) fn dtmf_on_event(&mut self, now: Ts, ev: &SessionEvent, cycle: &mut Cycle) {
        let Some(action) = self.activity.dtmf.as_mut().and_then(|d| d.on_event(ev, now)) else {
            return;
        };
        self.dtmf_perform(now, action, cycle);
    }

    fn dtmf_perform(&mut self, now: Ts, action: DtmfAction, cycle: &mut Cycle) {
        let Some(dtmf) = &mut self.activity.dtmf else { return };
        match action {
            DtmfAction::Idle | DtmfAction::Ignored => {}
            // `schedule()` replaces a pending timer: one arm, which the timer set replaces.
            DtmfAction::Schedule { at } => {
                dtmf.armed = true;
                cycle.machine().arm(TurnTimer::DtmfDebounce, at);
            }
            // `Debounced.run()`: the pending timer cancelled, the body now.
            DtmfAction::Cancel | DtmfAction::RunNow => {
                if std::mem::take(&mut dtmf.armed) {
                    cycle.machine().cancel(TurnTimer::DtmfDebounce);
                }
                if matches!(action, DtmfAction::RunNow) {
                    self.dtmf_reply(now, cycle);
                }
            }
        }
    }

    pub(crate) fn on_dtmf_debounce(&mut self, now: Ts, cycle: &mut Cycle) {
        let Some(dtmf) = &mut self.activity.dtmf else { return };
        dtmf.armed = false;
        self.dtmf_reply(now, cycle);
    }

    /// `_generate_dtmf_reply`: interrupt, then the collector's decision; the confirm arm holds the
    /// guard until the read-back is done.
    fn dtmf_reply(&mut self, now: Ts, cycle: &mut Cycle) {
        if self.activity.dtmf.is_none() {
            return;
        }
        // `self.session.interrupt()`: the host's non-forced interrupt, from the task's code.
        cycle.trace(InterruptStep::HostInterrupt { speech: None, force: false });
        if let Ok(speeches) = self.interrupt_all(now, false, InterruptCause::Host, cycle) {
            let waiter = crate::agent_activity::pause::InterruptWaiter { speeches, call: None };
            self.activity.interrupt_waiters.push(waiter);
        }
        let Some(dtmf) = &mut self.activity.dtmf else { return };
        let outcome = match dtmf.collector.generate_reply() {
            DtmfReply::Incomplete(error) => DtmfOutcome::Incomplete { message: error.message },
            DtmfReply::Complete(result) => DtmfOutcome::Complete(result),
            DtmfReply::Confirm { user_input } => return self.dtmf_confirm(now, user_input, cycle),
        };
        // The guard released; the call answered once (`if not self.done()`: a task already
        // completed takes no second result).
        dtmf.collector.finish_reply();
        let call = dtmf.call.take();
        self.answer(call, Answer::Dtmf(outcome));
    }

    /// The confirm arm's read-back, which holds the collector's guard until it is done; one
    /// refused at creation (a draining or closing queue) releases it now.
    fn dtmf_confirm(&mut self, now: Ts, user_input: String, cycle: &mut Cycle) {
        let h = self.generate_reply_speech(
            now,
            Some(user_input),
            None,
            None,
            None,
            None,
            None,
            false,
            cycle,
        );
        let queued = self.activity.speech_scheduler.contains(h.id());
        let Some(dtmf) = &mut self.activity.dtmf else { return };
        if queued {
            dtmf.confirm = Some(h.id());
        } else {
            dtmf.collector.finish_reply();
        }
    }
}
