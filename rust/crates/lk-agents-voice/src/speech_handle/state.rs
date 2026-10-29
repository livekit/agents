//! Accepted interruption and completion state. Live handles supply observations and publish effects.

use super::{CreationError, SpeechError};
use crate::agent_activity::scheduling::SpeechId;
use lk_agents_abi::errors::ProviderError;

/// Whether the host's call was answered: `say()` and `generate_reply()` return before the loop
/// creates the speech, and a handle constructed directly never gets one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Creation {
    Pending,
    Created,
    /// The session ended before the loop created the speech.
    Lost,
    Detached,
}

/// What an accepted host interruption does before or after speech creation.
pub(super) enum CreationInterrupt {
    Send(SpeechId),
    Ignore,
    Wait,
}

impl Creation {
    /// Only the pending promise can settle; repeated answers never replace its outcome.
    pub(super) fn answered(&mut self, created: bool) -> bool {
        if *self != Self::Pending {
            return false;
        }
        *self = if created { Self::Created } else { Self::Lost };
        true
    }

    pub(super) fn settled(&self) -> bool {
        *self != Self::Pending
    }

    pub(super) fn result(self) -> Result<(), CreationError> {
        match self {
            Self::Created => Ok(()),
            Self::Lost => Err(CreationError::Ended),
            Self::Pending | Self::Detached => Err(CreationError::Consumed),
        }
    }

    pub(super) fn interrupt(self, created: Option<SpeechId>) -> CreationInterrupt {
        match (created, self) {
            (Some(speech), _) => CreationInterrupt::Send(speech),
            (None, Self::Detached) => CreationInterrupt::Ignore,
            (None, _) => CreationInterrupt::Wait,
        }
    }
}

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
    authorized: bool,
    num_steps: u32,
    generations: Vec<bool>,
}

pub enum Input {
    AdvanceStep,
    AuthorizeGeneration,
    ClearAuthorization,
    MarkGenerationDone,
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
    TakeInterruptedAtDone,
}

pub enum Effect {
    PublishAcceptedView(SpeechView),
    PublishRequestFlag(bool),
    NewGeneration,
    PublishAuthorization(bool),
    PublishStep(u32),
    GenerationDone { step: usize },
}

pub enum Reply {
    Unit,
    Flag(bool),
    Step(u32),
}

impl Reply {
    pub fn flag(self) -> bool {
        match self {
            Self::Flag(value) => value,
            Self::Unit | Self::Step(_) => unreachable!("the speech transition returns a flag"),
        }
    }

    pub fn step(self) -> u32 {
        match self {
            Self::Step(value) => value,
            Self::Unit | Self::Flag(_) => unreachable!("the speech transition returns a step"),
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
            authorized: false,
            num_steps: 1,
            generations: Vec::new(),
        }
    }

    pub fn handle(&mut self, input: Input) -> Result<Transition, SpeechError> {
        let mut reply = Reply::Unit;
        let mut effects = Vec::new();
        match input {
            Input::AdvanceStep => {
                self.num_steps += 1;
                effects.push(Effect::PublishStep(self.num_steps));
                reply = Reply::Step(self.num_steps);
            }
            Input::AuthorizeGeneration => {
                self.generations.push(false);
                self.authorized = true;
                effects.push(Effect::NewGeneration);
                effects.push(Effect::PublishAuthorization(true));
            }
            Input::ClearAuthorization => {
                self.authorized = false;
                effects.push(Effect::PublishAuthorization(false));
            }
            Input::MarkGenerationDone => {
                let step = self.finish_latest_generation()?;
                effects.push(Effect::GenerationDone { step });
            }
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
        if let Ok(step) = self.finish_latest_generation() {
            effects.push(Effect::GenerationDone { step });
        }
    }

    fn finish_latest_generation(&mut self) -> Result<usize, SpeechError> {
        let step = self.generations.len().checked_sub(1).ok_or(SpeechError::NoGenerationToMark)?;
        self.generations[step] = true;
        Ok(step)
    }

    pub fn interruptible(&self) -> bool {
        !self.view.interrupted && self.view.allow_interruptions
    }

    pub fn authorized(&self) -> bool {
        self.authorized
    }

    pub fn step_authorized(&self, step: u32) -> bool {
        self.generations.len() >= step as usize
    }

    pub fn generation(&self, step: usize) -> Result<usize, SpeechError> {
        let len = self.generations.len();
        if len == 0 {
            return Err(SpeechError::NoActiveGeneration);
        }
        if step >= len {
            return Err(SpeechError::NoSuchGeneration { step, len });
        }
        Ok(step)
    }

    pub fn latest_generation(&self) -> Result<usize, SpeechError> {
        self.generations.len().checked_sub(1).ok_or(SpeechError::NoActiveGeneration)
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
