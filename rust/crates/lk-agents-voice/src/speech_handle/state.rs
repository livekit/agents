//! Accepted interruption and completion state. Live handles supply observations and publish effects.

use super::SpeechError;
use lk_agents_abi::errors::ProviderError;

#[derive(Clone)]
pub struct SpeechView {
    pub allow_interruptions: bool,
    pub interrupted: bool,
    pub done: bool,
    pub error: Option<ProviderError>,
}

pub struct SpeechState {
    view: SpeechView,
    holds: u32,
    restore: bool,
    watchdog_armed: bool,
    interrupted_at_done: bool,
    completing_after_cancel: bool,
}

pub enum Input {
    ResetInterruptions { allow: bool },
    Interrupt { force: bool },
    SetAllowInterruptions { allow: bool },
    HoldInterruptions,
    ReleaseInterruptions,
    Complete { error: Option<ProviderError>, interrupt_requested: bool },
    HostRequest { force: bool, created: bool, already_requested: bool },
    ValidateHostSetter { allow: bool, interrupt_requested: bool },
    ValidateHostHold,
    InterruptRefused,
    TakeWatchdog,
    WatchdogFired,
    TasksCancelled { interrupt_requested: bool },
    TakeInterruptedAtDone,
}

pub enum Effect {
    PublishAcceptedView(SpeechView),
    PublishRequestFlag(bool),
    CancelTasks,
    GenerationDone,
}

pub enum Reply {
    Unit,
    Flag(bool),
}

impl Reply {
    pub fn flag(self) -> bool {
        match self {
            Self::Flag(value) => value,
            Self::Unit => unreachable!("the speech transition returns a flag"),
        }
    }
}

pub struct Transition {
    pub reply: Reply,
    pub effects: Vec<Effect>,
}

impl SpeechState {
    pub fn new(allow: bool) -> Self {
        Self {
            view: SpeechView {
                allow_interruptions: allow,
                interrupted: false,
                done: false,
                error: None,
            },
            holds: 0,
            restore: allow,
            watchdog_armed: false,
            interrupted_at_done: false,
            completing_after_cancel: false,
        }
    }

    pub fn handle(&mut self, input: Input) -> Result<Transition, SpeechError> {
        let mut reply = Reply::Unit;
        let mut effects = Vec::new();
        match input {
            Input::ResetInterruptions { allow } => {
                self.view.allow_interruptions = allow;
                self.holds = 0;
                self.restore = allow;
                effects.push(self.publish());
            }
            Input::Interrupt { force } => {
                let landed = !self.view.interrupted && !self.view.done;
                if landed {
                    if !force && !self.view.allow_interruptions {
                        return Err(SpeechError::InterruptionsNotAllowed);
                    }
                    self.view.interrupted = true;
                    self.watchdog_armed = true;
                    effects.push(self.publish());
                }
                reply = Reply::Flag(landed);
            }
            Input::SetAllowInterruptions { allow } => {
                if self.view.interrupted && !allow {
                    return Err(SpeechError::AlreadyInterrupted);
                }
                self.view.allow_interruptions = allow;
                effects.push(self.publish());
            }
            Input::HoldInterruptions => {
                if self.holds == 0 {
                    if self.view.interrupted {
                        return Err(SpeechError::AlreadyInterrupted);
                    }
                    self.restore = self.view.allow_interruptions;
                    self.view.allow_interruptions = false;
                }
                self.holds += 1;
                effects.push(self.publish());
            }
            Input::ReleaseInterruptions => {
                if self.holds > 0 {
                    self.holds -= 1;
                    if self.holds == 0 && !(self.view.interrupted && !self.restore) {
                        self.view.allow_interruptions = self.restore;
                    }
                    effects.push(self.publish());
                }
            }
            Input::Complete { error, interrupt_requested } => {
                self.complete(error, interrupt_requested, &mut effects);
            }
            Input::HostRequest { force, created, already_requested } => {
                let requested =
                    !(already_requested || (created && (self.view.interrupted || self.view.done)));
                if requested {
                    if !force && !self.view.allow_interruptions {
                        return Err(SpeechError::InterruptionsNotAllowed);
                    }
                    effects.push(Effect::PublishRequestFlag(true));
                }
                reply = Reply::Flag(requested);
            }
            Input::ValidateHostSetter { allow, interrupt_requested } => {
                if !allow && (interrupt_requested || self.view.interrupted) {
                    return Err(SpeechError::AlreadyInterrupted);
                }
            }
            Input::ValidateHostHold => {
                if self.view.interrupted {
                    return Err(SpeechError::AlreadyInterrupted);
                }
            }
            Input::InterruptRefused => effects.push(Effect::PublishRequestFlag(false)),
            Input::TakeWatchdog => reply = Reply::Flag(std::mem::take(&mut self.watchdog_armed)),
            Input::WatchdogFired => {
                let fired = std::mem::take(&mut self.watchdog_armed);
                if fired {
                    self.completing_after_cancel = true;
                    effects.push(Effect::CancelTasks);
                }
                reply = Reply::Flag(fired);
            }
            Input::TasksCancelled { interrupt_requested } => {
                if std::mem::take(&mut self.completing_after_cancel) {
                    self.complete(None, interrupt_requested, &mut effects);
                }
            }
            Input::TakeInterruptedAtDone => {
                reply = Reply::Flag(std::mem::take(&mut self.interrupted_at_done))
            }
        }
        Ok(Transition { reply, effects })
    }

    fn complete(
        &mut self,
        error: Option<ProviderError>,
        interrupt_requested: bool,
        effects: &mut Vec<Effect>,
    ) {
        if !self.view.done {
            if interrupt_requested && !self.view.interrupted {
                self.view.interrupted = true;
                self.interrupted_at_done = true;
            }
            if error.is_some() {
                self.view.error = error;
            }
            self.view.done = true;
        }
        self.watchdog_armed = false;
        effects.push(self.publish());
        effects.push(Effect::GenerationDone);
    }

    fn publish(&self) -> Effect {
        Effect::PublishAcceptedView(self.view.clone())
    }

    pub fn exception(&self) -> Result<Option<ProviderError>, SpeechError> {
        if self.view.done { Ok(self.view.error.clone()) } else { Err(SpeechError::NotDoneYet) }
    }

    pub fn outcome(&self) -> crate::events::SpeechOutcome {
        use crate::events::SpeechOutcome;
        if self.view.error.is_some() && self.view.done {
            SpeechOutcome::Failed
        } else if self.view.interrupted {
            SpeechOutcome::Interrupted
        } else {
            SpeechOutcome::Completed
        }
    }
}
