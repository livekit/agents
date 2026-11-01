//! Queue-backed audio's generated boundary; segment admission and report correlation are native.

use super::avatar::AudioReceiverObservation;
use super::io::{AvatarPlaybackSegment, PlaybackSegment};
use crate::AudioFrame;

#[crate::host(record(module = "avatar_queue", dataclass, emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct QueueFinished {
    pub segment: PlaybackSegment,
    pub playback_position: f64,
    pub interrupted: bool,
}

#[crate::host(record(module = "avatar_queue", dataclass, emit))]
#[derive(Debug, Clone, PartialEq, Default)]
pub struct QueueEffects {
    pub started: Vec<PlaybackSegment>,
    pub finished: Vec<QueueFinished>,
    pub clear: bool,
    pub flushed: bool,
    pub reset: bool,
}

#[crate::host(error(kind = "livekit.agents.utils.aio.ChanClosed"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("the audio queue is closed")]
pub struct QueueClosed;

#[crate::host(class(name = "_core_QueueAudio"))]
pub trait QueueAudio: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(wait_playback_start: bool) -> Self;
    #[host(call)]
    fn check_open(&self) -> Result<(), QueueClosed>;
    #[host(call)]
    fn capture(
        &self,
        frame: AudioFrame,
        segment: PlaybackSegment,
        epoch: u64,
    ) -> Result<QueueEffects, QueueClosed>;
    #[host(call)]
    fn flush(&self) -> Result<(), QueueClosed>;
    #[host(call)]
    fn clear(&self) -> QueueEffects;
    #[host(call)]
    fn playback_finished(
        &self,
        playback_position: f64,
        interrupted: bool,
        segment: Option<AvatarPlaybackSegment>,
    ) -> QueueEffects;
    #[host(call)]
    fn playback_started(&self, segment: Option<AvatarPlaybackSegment>) -> QueueEffects;
    #[host(await)]
    async fn next_audio(&self) -> AudioReceiverObservation;
    #[host(call)]
    fn close(&self) -> QueueEffects;
}
