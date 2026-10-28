//! A caller-supplied background audio stream.

use crate::AudioFrame;
use crate::errors::ProviderError;

/// A caller-supplied stream. Cancellation must reach any pending host-side pull. A host
/// `AsyncIterator[rtc.AudioFrame]` is this trait through `background_audio.py`'s `_FrameStream`.
#[crate::host(proxy(host = "livekit.agents.voice.background_audio._FrameStream"))]
#[async_trait::async_trait]
pub trait FrameStream: Send + Sync {
    /// The next frame, `None` once exhausted (or cancelled by [`cancel_pending`](Self::cancel_pending)).
    #[host(await, name = "_core_next")]
    async fn next(&self) -> Result<Option<AudioFrame>, ProviderError>;
    #[host(notify, name = "_core_cancel_pending")]
    fn cancel_pending(&self);
    #[host(await, name = "_core_close")]
    async fn close(&self);
}
