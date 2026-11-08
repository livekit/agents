mod audio;
mod core;
mod metrics;
mod video;

pub use audio::{AudioReceiver, AudioSender, audio};
pub use video::{VideoReceiver, VideoSender, video};

use std::time::Duration;

// Events travel in the media at the point they apply to and are never dropped: the bound drops
// only media queued after the last event, and `flush` is the one way to discard media.

/// The longest frame merging builds: the frame size every consumer is tested against.
pub const MERGED_FRAME: Duration = Duration::from_millis(50);

/// Events pending past which the consumer counts as stuck.
pub const MAX_EVENTS: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Item<M, E> {
    Media(M),
    /// Media dropped by the bound, as the duration it covered (audio only).
    Gap(Duration),
    Event(E),
}

/// What an audio queue keeps when its consumer falls behind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Keep up to this much queued audio; drop the oldest beyond it.
    CatchUp(Duration),
    /// Once the consumer lags more than this, drop to the newest audio.
    Now(Duration),
}

/// The head audio a receiver has drained.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Head {
    /// Received so far, monotone.
    pub received: Duration,
    /// Head audio is still queued.
    pub pending: bool,
}

/// More than [`MAX_EVENTS`] events are pending: the consumer stopped draining.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("media consumer stopped draining its events")]
pub struct Stuck;
