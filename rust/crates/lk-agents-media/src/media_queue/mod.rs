//! Live media from a producer that never blocks to a consumer that may fall behind, after
//! GStreamer's leaky queue and its serialized events.
//!
//! - [`audio`] is bounded by its queued duration under a [`Policy`]; past it the oldest frames are
//!   dropped as one [`Item::Gap`]. A backlog is merged into frames of up to [`MERGED_FRAME`].
//! - [`video`] is bounded by frame count and keeps the newest; it never merges and never reports a
//!   gap.
//! - Events travel in the media at the point they apply to and are never dropped. The bound drops
//!   only media queued after the last event; `flush` is the one way to discard media on purpose.
//!
//! One producer that never blocks and one async consumer share the queue: the lock is held only
//! for the queue's own short work, the receiver's waker is registered under it and woken after it
//! is released, and a merged frame is built outside it. No thread, no spawn, no clock.

mod audio;
mod core;
mod video;

pub use audio::{AudioReceiver, AudioSender, audio};
pub use video::{VideoReceiver, VideoSender, video};

use std::time::Duration;

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

/// More than [`MAX_EVENTS`] events are pending: the consumer stopped draining.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("media consumer stopped draining its events")]
pub struct Stuck;
