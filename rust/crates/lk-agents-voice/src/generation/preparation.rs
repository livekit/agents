//! Generation-step preparation and playback authorization share one owner.
//!
//! Preparation can admit speculative synthesis before playback. Once observed, authorization
//! belongs to the entire generation step and is reused by every segment, including text-only
//! segments. The runtime only waits for scheduling/authorization facts and supplies their result.

use std::time::SystemTime;

/// Playback permission, not a command to a physical audio sink.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackAuthorization {
    Pending,
    Authorized,
    Interrupted,
}

/// Facts used to prepare a generation step. `authorization_available` means the speech handle
/// has published authorization; it has not necessarily been observed/consumed by this owner.
pub enum GenerationPreparationEvent {
    Begin {
        has_tts: bool,
        audio_enabled: bool,
        has_audio_output: bool,
        llm_reply: bool,
        preemptive_tts: bool,
        authorization_available: bool,
    },
    ReadinessResolved {
        ready: bool,
        authorization_available: bool,
    },
    PlaybackResolved {
        authorization: PlaybackAuthorization,
        authorization_available: bool,
    },
}

pub enum GenerationPreparationEffect {
    WaitForReadiness,
    WaitForPlaybackAuthorization,
    Start,
    Interrupted,
}

#[derive(Default)]
enum PreparationPhase {
    #[default]
    Initial,
    Readiness,
    InitialPlayback,
    FinalPlayback,
    Started,
    Interrupted,
}

/// One generation step's synthesis admission and reusable playback authorization.
/// This is separate from AudioPlayback's per-segment source/receipt lifecycle and launch hold.
#[derive(Default)]
pub struct GenerationPreparation {
    phase: PreparationPhase,
    authorized_at: Option<SystemTime>,
}

impl GenerationPreparation {
    /// Whether authorization is published but still needs to be consumed by this step.
    pub fn pending_authorization(&self, available: bool) -> bool {
        self.authorized_at.is_none() && available
    }

    pub fn authorized_at(&self) -> Option<SystemTime> {
        self.authorized_at
    }

    /// Already-authorized segments may reuse the permission; interruption still wins.
    pub fn authorization(&self, interrupted: bool) -> PlaybackAuthorization {
        if interrupted {
            PlaybackAuthorization::Interrupted
        } else if self.authorized_at.is_some() {
            PlaybackAuthorization::Authorized
        } else {
            PlaybackAuthorization::Pending
        }
    }

    /// Runtime observed the step's authorization at this supplied instant. Retain the first
    /// observation, so multiple segments cannot restamp reply-start metrics.
    pub fn authorize(&mut self, at: SystemTime) -> PlaybackAuthorization {
        self.authorized_at.get_or_insert(at);
        PlaybackAuthorization::Authorized
    }

    pub fn handle(&mut self, event: GenerationPreparationEvent) -> GenerationPreparationEffect {
        match (&self.phase, event) {
            (
                PreparationPhase::Initial,
                GenerationPreparationEvent::Begin {
                    has_tts,
                    audio_enabled,
                    has_audio_output,
                    llm_reply,
                    preemptive_tts,
                    authorization_available,
                },
            ) => {
                if has_tts && audio_enabled && has_audio_output {
                    if llm_reply && preemptive_tts {
                        self.check_authorization(authorization_available)
                    } else {
                        self.phase = PreparationPhase::Readiness;
                        GenerationPreparationEffect::WaitForReadiness
                    }
                } else {
                    self.phase = PreparationPhase::InitialPlayback;
                    GenerationPreparationEffect::WaitForPlaybackAuthorization
                }
            }
            (
                PreparationPhase::Readiness,
                GenerationPreparationEvent::ReadinessResolved {
                    ready: true,
                    authorization_available,
                },
            )
            | (
                PreparationPhase::InitialPlayback,
                GenerationPreparationEvent::PlaybackResolved {
                    authorization: PlaybackAuthorization::Authorized,
                    authorization_available,
                },
            ) => self.check_authorization(authorization_available),
            (
                PreparationPhase::FinalPlayback,
                GenerationPreparationEvent::PlaybackResolved {
                    authorization: PlaybackAuthorization::Authorized,
                    ..
                },
            ) => {
                self.phase = PreparationPhase::Started;
                GenerationPreparationEffect::Start
            }
            (PreparationPhase::Started, _) => GenerationPreparationEffect::Start,
            _ => {
                self.phase = PreparationPhase::Interrupted;
                GenerationPreparationEffect::Interrupted
            }
        }
    }

    fn check_authorization(&mut self, available: bool) -> GenerationPreparationEffect {
        if self.pending_authorization(available) {
            self.phase = PreparationPhase::FinalPlayback;
            GenerationPreparationEffect::WaitForPlaybackAuthorization
        } else {
            self.phase = PreparationPhase::Started;
            GenerationPreparationEffect::Start
        }
    }
}

#[cfg(test)]
#[path = "preparation_tests.rs"]
mod tests;
