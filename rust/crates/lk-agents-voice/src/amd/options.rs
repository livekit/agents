//! AMD's options, their defaults and validation (spec §4.1).

use super::Opening;
use super::events::Stage;
use std::time::Duration;

pub const DEFAULT_SCREENING_INSTRUCTIONS: &str = "Call state: automated call screening. Answer the screening assistant's latest prompt briefly. Use what you know about who you are and why you are calling. Then wait for its next prompt.";
pub const DEFAULT_VOICEMAIL_INSTRUCTIONS: &str = "Call state: voicemail. Deliver one concise, self-contained message. State who you are, why you are calling, and the next step for the recipient.";
pub const DEFAULT_IVR_INSTRUCTIONS: &str = "Call state: an automated phone menu. Use the participant's actual prompt and the purpose of this call to choose the next step. Use send_dtmf_events for an explicit keypad choice, or give a short spoken answer when requested. Do not invent a menu option. After the action, wait for the next prompt.";
pub const DEFAULT_HUMAN_INSTRUCTIONS: &str = "Call state: a human has answered. The latest participant turn is from the human, not the automated system. Earlier automated prompts no longer apply. Respond to the human's latest message and continue the call normally. Do not resume a response to an earlier automated prompt.";

/// Appended to the IVR instructions once the voicemail message played (spec §10.2).
pub const VOICEMAIL_PLAYED_NOTE: &str = "\nThe voicemail message already played locally.";

/// `AMD(...)`'s behavior options. Model and participant selection belong to the host and legs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmdOptions {
    /// Reply instructions in `machine-screening`; `None` makes it a stage the application handles.
    pub screening_instructions: Option<String>,
    /// Instructions for the voicemail message; `None` makes `machine-vm` a stage the application
    /// handles.
    pub voicemail_instructions: Option<String>,
    /// Reply instructions in `machine-ivr`; `None` makes it a stage the application handles.
    pub ivr_instructions: Option<String>,
    /// Instructions for the reply to the turn whose `human` prediction completes AMD after a
    /// machine stage. `None` keeps the agent's own.
    pub human_instructions: Option<String>,
    /// Fixed text the agent says first.
    pub opening_text: Option<String>,
    /// Instructions for a reply the agent generates and says first.
    pub opening_instructions: Option<String>,
    /// Discard SIP audio until the call is answered.
    pub wait_until_answered: bool,
    /// Silent, inactive time before completion outside `machine-vm`.
    pub idle_timeout: Duration,
    /// Silent, inactive time in `machine-vm`.
    pub voicemail_idle_timeout: Duration,
    /// Overall limit from the start of listening.
    pub timeout: Duration,
    /// Deadline for each classifier request.
    pub inference_timeout: Duration,
    /// Participant silence before a machine-stage reply plays; zero disables the silence hold.
    pub machine_silence_threshold: Duration,
    /// Consecutive `uncertain` predictions before completion.
    pub max_uncertain_turns: u32,
    /// Request timeouts before completion.
    pub max_inference_timeouts: u32,
}

impl Default for AmdOptions {
    fn default() -> Self {
        AmdOptions {
            screening_instructions: Some(DEFAULT_SCREENING_INSTRUCTIONS.into()),
            voicemail_instructions: Some(DEFAULT_VOICEMAIL_INSTRUCTIONS.into()),
            ivr_instructions: Some(DEFAULT_IVR_INSTRUCTIONS.into()),
            human_instructions: Some(DEFAULT_HUMAN_INSTRUCTIONS.into()),
            opening_text: None,
            opening_instructions: None,
            wait_until_answered: true,
            idle_timeout: Duration::from_secs(10),
            voicemail_idle_timeout: Duration::from_secs(10),
            timeout: Duration::from_secs(60),
            inference_timeout: Duration::from_secs(2),
            machine_silence_threshold: Duration::from_millis(1500),
            max_uncertain_turns: 3,
            max_inference_timeouts: 3,
        }
    }
}

/// Why options were refused: `AMD.__init__`'s `ValueError`s, in its words.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AmdOptionsError {
    #[error("amd timeouts and max_uncertain_turns must be positive")]
    NotPositive,
    #[error("max_inference_timeouts must be positive")]
    MaxInferenceTimeouts,
    #[error("set either opening_text or opening_instructions, not both")]
    BothOpenings,
    #[error("opening_text must not be empty")]
    EmptyOpeningText,
    #[error("opening_instructions must not be empty")]
    EmptyOpeningInstructions,
}

impl AmdOptions {
    /// The constructor's checks, in its order. A negative `machine_silence_threshold` cannot be
    /// a `Duration`; the host refuses it when converting.
    pub fn validate(&self) -> Result<(), AmdOptionsError> {
        let timeouts =
            [self.idle_timeout, self.voicemail_idle_timeout, self.timeout, self.inference_timeout];
        if timeouts.contains(&Duration::ZERO) || self.max_uncertain_turns < 1 {
            return Err(AmdOptionsError::NotPositive);
        }
        if self.max_inference_timeouts < 1 {
            return Err(AmdOptionsError::MaxInferenceTimeouts);
        }
        match (&self.opening_text, &self.opening_instructions) {
            (Some(_), Some(_)) => Err(AmdOptionsError::BothOpenings),
            (Some(text), None) if text.trim().is_empty() => Err(AmdOptionsError::EmptyOpeningText),
            (None, Some(text)) if text.trim().is_empty() => {
                Err(AmdOptionsError::EmptyOpeningInstructions)
            }
            _ => Ok(()),
        }
    }

    /// The opening `Activated` carries, if any.
    pub fn opening(&self) -> Option<Opening> {
        match (&self.opening_text, &self.opening_instructions) {
            (Some(text), _) => Some(Opening::Text(text.clone())),
            (None, Some(instructions)) => Some(Opening::Instructions(instructions.clone())),
            (None, None) => None,
        }
    }

    /// A stage's reply instructions as configured; `None` for a stage without any.
    pub fn instructions(&self, stage: Stage) -> Option<&str> {
        match stage {
            Stage::MachineScreening => self.screening_instructions.as_deref(),
            Stage::MachineVm => self.voicemail_instructions.as_deref(),
            Stage::MachineIvr => self.ivr_instructions.as_deref(),
            Stage::Human => self.human_instructions.as_deref(),
            Stage::Uncertain | Stage::MachineUnavailable => None,
        }
    }

    /// `_returns_early`: entering this machine stage completes AMD, because its instructions are
    /// `None` (spec §9.5).
    pub fn returns_early(&self, stage: Stage) -> bool {
        matches!(stage, Stage::MachineScreening | Stage::MachineVm | Stage::MachineIvr)
            && self.instructions(stage).is_none()
    }
}
