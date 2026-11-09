use super::machine::runner::{AvatarIoOperation, CallId, RunnerError};

#[derive(Debug, Clone, PartialEq)]
pub enum AvatarEffect {
    /// Run this; its answer comes back as an [`AvatarEvent`](super::machine::runner::AvatarEvent).
    Io(AvatarIoOperation),
    /// Cancel every in-flight [`AvatarIoOperation`], then answer
    /// [`AvatarEvent::IoCancelled`](super::machine::runner::AvatarEvent::IoCancelled).
    CancelIo,
    Answer {
        call: CallId,
        result: Result<(), RunnerError>,
    },
}
