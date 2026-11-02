//! Work requested by the avatar runner state machine.
use super::machine::runner::{AvatarIoOperation, CallId, RunnerError};

#[derive(Debug, Clone, PartialEq)]
pub enum AvatarEffect {
    /// Run this; its answer comes back as an [`AvatarEvent`](super::machine::runner::AvatarEvent).
    Io(AvatarIoOperation),
    /// Cancel every [`AvatarIoOperation`] in flight and wait for them to be gone, then [`AvatarEvent::IoCancelled`](super::machine::runner::AvatarEvent::IoCancelled).
    CancelIo,
    Answer {
        call: CallId,
        result: Result<(), RunnerError>,
    },
}
