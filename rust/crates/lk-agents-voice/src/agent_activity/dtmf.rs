//! `GetDtmfTask`: collecting the digits the caller presses on the keypad, debounced, with an
//! optional read-back to confirm them.

use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::effects::{Out, TurnTimer};
use crate::agent_session::signals::Signal;
use crate::agent_session::{Answer, Driver, DtmfCommand, RequestId};
use crate::events::SessionEvent;
use crate::trace::{InterruptCause, InterruptStep};
use crate::workflows::builtins::dtmf::{DtmfAction, DtmfCollector, DtmfOutcome, DtmfReply};
use lk_agents_abi::Ts;
use lk_agents_abi::voice::events::{AgentState, UserState};

/// The collection while it runs.
#[derive(Default)]
pub(crate) struct Dtmf {
    running: Option<Collection>,
}

struct Collection {
    collector: DtmfCollector,
    /// Answered once with the first outcome, or `Stopped`.
    call: Option<RequestId>,
    /// Whether `TurnTimer::DtmfDebounce` is armed, so a cancel is never a cancel of nothing.
    armed: bool,
    /// The confirm arm's read-back: the collector's guard is held until this speech is done.
    confirm: Option<SpeechId>,
}

pub(crate) enum DtmfTaskEvent {
    Start {
        collector: DtmfCollector,
        call: Option<RequestId>,
    },
    Digit(String),
    /// `GetDtmfTask` stopped, or the session closes: an owed answer is `Stopped`.
    Stop,
    AgentState(AgentState),
    UserState(UserState),
    SpeechDone(SpeechId),
    Debounced,
    /// The read-back asked for: the speech holding the guard, or `None` when it was refused.
    ReadBack(Option<SpeechId>),
}

impl From<DtmfCommand> for DtmfTaskEvent {
    fn from(cmd: DtmfCommand) -> Self {
        match cmd {
            DtmfCommand::Start { collector, call } => DtmfTaskEvent::Start { collector, call },
            DtmfCommand::Digit { digit } => DtmfTaskEvent::Digit(digit),
            DtmfCommand::Stop => DtmfTaskEvent::Stop,
        }
    }
}

impl Dtmf {
    pub(crate) fn handle(&mut self, ev: DtmfTaskEvent, now: Ts, out: &mut Out) {
        let action = match ev {
            DtmfTaskEvent::Start { collector, call } => {
                self.stop(out);
                self.running = Some(Collection { collector, call, armed: false, confirm: None });
                return;
            }
            DtmfTaskEvent::Stop => return self.stop(out),
            DtmfTaskEvent::Debounced => {
                let Some(dtmf) = &mut self.running else { return };
                dtmf.armed = false;
                return self.reply(out);
            }
            DtmfTaskEvent::ReadBack(speech) => {
                let Some(dtmf) = &mut self.running else { return };
                match speech {
                    Some(speech) => dtmf.confirm = Some(speech),
                    None => dtmf.collector.finish_reply(),
                }
                return;
            }
            ev => {
                let Some(dtmf) = &mut self.running else { return };
                match ev {
                    DtmfTaskEvent::Digit(digit) => dtmf.collector.on_sip_dtmf(&digit, now),
                    DtmfTaskEvent::AgentState(new) => {
                        dtmf.collector.on_agent_state_changed(new.as_str(), now)
                    }
                    DtmfTaskEvent::UserState(new) => {
                        dtmf.collector.on_user_state_changed(new.as_str(), now)
                    }
                    // The read-back's end releases its guard.
                    DtmfTaskEvent::SpeechDone(speech) if dtmf.confirm == Some(speech) => {
                        dtmf.confirm = None;
                        dtmf.collector.finish_reply();
                        return;
                    }
                    _ => return,
                }
            }
        };
        self.perform(action, out);
    }

    /// The call a collection still owes, when the loop ends.
    pub(crate) fn take_call(&mut self) -> Option<RequestId> {
        self.running.take().and_then(|d| d.call)
    }

    /// The collection stops: its guard released (the host reading `dtmf_reply_running()` must not
    /// see a guard nobody will release), its debounce cancelled, an owed answer `Stopped`.
    fn stop(&mut self, out: &mut Out) {
        let Some(mut dtmf) = self.running.take() else { return };
        dtmf.collector.finish_reply();
        if dtmf.armed {
            out.machine().cancel(TurnTimer::DtmfDebounce);
        }
        out.answer(dtmf.call, Answer::Dtmf(DtmfOutcome::Stopped));
    }

    fn perform(&mut self, action: DtmfAction, out: &mut Out) {
        let Some(dtmf) = &mut self.running else { return };
        match action {
            DtmfAction::Idle | DtmfAction::Ignored => {}
            // `schedule()` replaces a pending timer: one arm, which the timer set replaces.
            DtmfAction::Schedule { at } => {
                dtmf.armed = true;
                out.machine().arm(TurnTimer::DtmfDebounce, at);
            }
            // `Debounced.run()`: the pending timer cancelled, the body now.
            DtmfAction::Cancel | DtmfAction::RunNow => {
                if std::mem::take(&mut dtmf.armed) {
                    out.machine().cancel(TurnTimer::DtmfDebounce);
                }
                if matches!(action, DtmfAction::RunNow) {
                    self.reply(out);
                }
            }
        }
    }

    /// `_generate_dtmf_reply`: interrupt, then the collector's decision; the confirm arm holds the
    /// guard until the read-back is done.
    fn reply(&mut self, out: &mut Out) {
        let Some(dtmf) = &mut self.running else { return };
        out.signal(Signal::InterruptForDtmf);
        let outcome = match dtmf.collector.generate_reply() {
            DtmfReply::Incomplete(error) => DtmfOutcome::Incomplete { message: error.message },
            DtmfReply::Complete(result) => DtmfOutcome::Complete(result),
            DtmfReply::Confirm { user_input } => {
                return out.signal(Signal::ReadBackDtmf(user_input));
            }
        };
        // The guard released; the call answered once (`if not self.done()`: a task already
        // completed takes no second result).
        dtmf.collector.finish_reply();
        out.answer(dtmf.call.take(), Answer::Dtmf(outcome));
    }
}

impl Driver {
    pub(crate) fn dtmf_step(&mut self, ev: DtmfTaskEvent, out: &mut Out) {
        let now = self.step.now;
        self.activity.dtmf.handle(ev, now, out);
    }

    /// The edges a collection reads, off the events on their way to the host.
    pub(crate) fn dtmf_on_event(&mut self, ev: &SessionEvent, out: &mut Out) {
        let ev = match ev {
            SessionEvent::AgentStateChanged { new, .. } => DtmfTaskEvent::AgentState(*new),
            SessionEvent::UserStateChanged { new, .. } => DtmfTaskEvent::UserState(*new),
            SessionEvent::SpeechDone { speech, .. } => DtmfTaskEvent::SpeechDone(*speech),
            _ => return,
        };
        self.dtmf_step(ev, out);
    }

    /// `self.session.interrupt()`: the host's non-forced interrupt, from the task's code.
    pub(crate) fn interrupt_for_dtmf(&mut self, now: Ts, out: &mut Out) {
        out.trace(InterruptStep::HostInterrupt { speech: None, force: false });
        if let Ok(speeches) = self.interrupt_all(now, false, InterruptCause::Host, out) {
            let waiter =
                crate::agent_activity::false_interruption::InterruptWaiter { speeches, call: None };
            self.activity.interrupt_waiters.push(waiter);
        }
    }

    /// The confirm arm's read-back; one refused at creation (a draining or closing queue)
    /// releases the guard now.
    pub(crate) fn read_back_dtmf(&mut self, now: Ts, user_input: String, out: &mut Out) {
        let h = self.generate_reply_speech(
            now,
            Some(user_input),
            None,
            None,
            None,
            None,
            None,
            false,
            out,
        );
        let queued = self.activity.speech_scheduler.contains(h.id());
        self.dtmf_step(DtmfTaskEvent::ReadBack(queued.then(|| h.id())), out);
    }
}
