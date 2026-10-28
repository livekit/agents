//! The avatar's host contracts: its audio receiver, video generator and media, and what the
//! receiver tells its host.

use crate::errors::ProviderError;
use crate::utils::video::{AvatarFrame, MediaFrame};

/// The `clear_buffer` event `DataStreamAudioReceiver` emits when the agent clears its audio
/// (`AudioReceiver.emit`).
#[crate::host(proxy(host = "livekit.agents.voice.avatar.DataStreamAudioReceiver"))]
pub trait ClearBufferListener: Send + Sync {
    #[host(notify, name = "_core_clear_buffer")]
    fn clear_buffer(&self);
}

/// `AudioReceiver`: the agent's audio as the avatar worker receives it, and where its playback is
/// reported. A host `AudioReceiver` carries this surface (`voice/avatar.py`).
#[crate::host(proxy(host = "livekit.agents.voice.avatar.AudioReceiver"))]
#[async_trait::async_trait]
pub trait AvatarAudioReceiver: Send + Sync {
    /// Begin iterating (`__aiter__`).
    #[host(await, name = "_core_open_audio")]
    async fn open_audio(&self) -> Result<(), ProviderError>;
    /// The next frame or segment end; `None` once the receiver is exhausted.
    #[host(await, name = "_core_next_audio")]
    async fn next_audio(&self) -> Result<Option<AvatarFrame>, ProviderError>;
    #[host(await, name = "_core_notify_playback_started")]
    async fn notify_playback_started(&self) -> Result<(), ProviderError>;
    #[host(await, name = "_core_notify_playback_finished")]
    async fn notify_playback_finished(
        &self,
        playback_position: f64,
        interrupted: bool,
    ) -> Result<(), ProviderError>;
    #[host(await, name = "_core_aclose")]
    async fn aclose(&self) -> Result<(), ProviderError>;
}

/// `VideoGenerator`: the avatar's model, fed the audio and yielding the frames to publish. A host
/// `VideoGenerator` carries this surface (`voice/avatar.py`).
#[crate::host(proxy(host = "livekit.agents.voice.avatar.VideoGenerator"))]
#[async_trait::async_trait]
pub trait AvatarVideoGenerator: Send + Sync {
    /// An audio frame, or `None` for `AudioSegmentEnd`.
    #[host(await, name = "_core_push_audio")]
    async fn push_audio(&self, frame: Option<MediaFrame>) -> Result<(), ProviderError>;
    #[host(await, name = "_core_clear_buffer")]
    async fn clear_buffer(&self) -> Result<(), ProviderError>;
    /// Begin iterating (`__aiter__`).
    #[host(await, name = "_core_open_frames")]
    async fn open_frames(&self) -> Result<(), ProviderError>;
    /// The next frame or segment end; `None` once the generator is exhausted.
    #[host(await, name = "_core_next_frame")]
    async fn next_frame(&self) -> Result<Option<AvatarFrame>, ProviderError>;
}

/// The SDK's half: the avatar's tracks in the room and the `rtc.AVSynchronizer` pacing what is
/// pushed to them. Python and Node each have their own.
#[crate::host(proxy(host = "livekit.agents.voice.avatar._AvatarMedia"))]
#[async_trait::async_trait]
pub trait AvatarMedia: Send + Sync {
    /// Publish the audio track, wait for its subscription, then publish the video track.
    #[host(await, name = "_core_publish")]
    async fn publish(&self) -> Result<(), ProviderError>;
    /// One frame into the synchronizer, which paces it out.
    #[host(await, name = "_core_push")]
    async fn push(&self, frame: MediaFrame) -> Result<(), ProviderError>;
    #[host(await, name = "_core_aclose")]
    async fn aclose(&self) -> Result<(), ProviderError>;
}
