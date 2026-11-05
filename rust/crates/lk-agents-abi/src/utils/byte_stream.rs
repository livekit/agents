//! The host's audio chunker; the implementation is `lk-agents-media`'s.

use crate::AudioFrame;

/// Buffers bytes and emits whole frames: fixed-size, or progressive (20 ms first, doubling to
/// the target) to cut TTS time-to-first-audio.
#[crate::host(class(name = "AudioByteStream", module = "livekit.agents.utils.media", pid = false))]
pub trait AudioByteStream: Send + Sized + 'static {
    /// ``samples_per_channel=None`` means 100 ms, as in the reference.
    #[host(new)]
    fn new(
        sample_rate: u32,
        num_channels: u16,
        #[host(default = None)] samples_per_channel: Option<u32>,
        #[host(default = false)] progressive: bool,
    ) -> Self;

    /// Buffer ``data`` and emit every whole frame it completes.
    #[host(call)]
    fn push(&mut self, #[host(pcm)] data: Vec<u8>) -> Vec<AudioFrame>;

    /// Emit the buffered whole samples, short of a frame or not; a trailing partial sample stays
    /// for the next push.
    #[host(call)]
    fn flush(&mut self) -> Vec<AudioFrame>;

    /// Seconds of audio waiting for the next frame.
    #[host(getter)]
    fn buffered_duration(&self) -> f64;

    /// Restart progressive sizing, keeping the buffered audio: after a mid-stream flush, the
    /// next burst starts with small frames without losing a partial sample.
    #[host(call)]
    fn reset_progressive(&mut self);

    /// Drop everything buffered and restart progressive sizing (called on interruption, so the
    /// next word is not a full frame late).
    #[host(call)]
    fn clear(&mut self);
}
