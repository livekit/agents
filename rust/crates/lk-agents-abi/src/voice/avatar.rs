//! The avatar's host contracts: its audio receiver, video generator and media, and what the
//! receiver tells its host.

use crate::errors::ProviderError;
use crate::utils::video::{AudioReceiverFrame, AvatarFrame, MediaFrame};
use crate::voice::io::AvatarPlaybackSegment;

/// The `clear_buffer` event `DataStreamAudioReceiver` emits when the agent clears its audio
/// (`AudioReceiver.emit`).
#[crate::host(proxy(host = "livekit.agents.voice.avatar.AudioReceiver"))]
pub trait ClearBufferListener: Send + Sync {
    #[host(notify, name = "_core_clear_buffer")]
    fn clear_buffer(&self);
}

/// `AudioReceiver`: the agent's audio as the avatar worker receives it, and where its playback is
/// reported. A host `AudioReceiver` carries this surface (`voice/avatar.py`).
#[crate::host(proxy(host = "livekit.agents.voice.avatar.AudioReceiver"))]
pub trait AvatarAudioReceiver: Send + Sync {
    /// Begin iterating (`__aiter__`).
    #[host(await, name = "_core_open_audio")]
    fn open_audio(&self) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
    /// One observation: receiver clear epoch, whether this first frame opened a segment,
    /// and the frame (None at EOF). The receiver stamps it before crossing the async boundary.
    #[host(await, name = "_core_next_audio")]
    fn next_audio(
        &self,
    ) -> crate::BoxFuture<
        '_,
        Result<
            (u64, bool, Option<AvatarPlaybackSegment>, Option<AudioReceiverFrame>),
            ProviderError,
        >,
    >;
    #[host(await, name = "_core_notify_playback_started")]
    fn notify_playback_started(
        &self,
        segment: AvatarPlaybackSegment,
    ) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
    #[host(await, name = "_core_notify_playback_finished")]
    fn notify_playback_finished(
        &self,
        segment: AvatarPlaybackSegment,
        playback_position: f64,
        interrupted: bool,
    ) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
    #[host(await, name = "_core_aclose")]
    fn aclose(&self) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
}

/// `VideoGenerator`: the avatar's model, fed the audio and yielding the frames to publish. A host
/// `VideoGenerator` carries this surface (`voice/avatar.py`).
#[crate::host(proxy(host = "livekit.agents.voice.avatar.VideoGenerator"))]
pub trait AvatarVideoGenerator: Send + Sync {
    /// An audio frame, or `None` for `AudioSegmentEnd`.
    #[host(await, name = "_core_push_audio")]
    fn push_audio(
        &self,
        frame: Option<MediaFrame>,
    ) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
    #[host(await, name = "_core_clear_buffer")]
    fn clear_buffer(&self) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
    /// Begin iterating (`__aiter__`).
    #[host(await, name = "_core_open_frames")]
    fn open_frames(&self) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
    /// The next frame or segment end; `None` once the generator is exhausted.
    #[host(await, name = "_core_next_frame")]
    fn next_frame(&self) -> crate::BoxFuture<'_, Result<Option<AvatarFrame>, ProviderError>>;
}

/// The SDK's half: the avatar's tracks in the room and the `rtc.AVSynchronizer` pacing what is
/// pushed to them. Python and Node each have their own.
#[crate::host(proxy(host = "livekit.agents.voice.avatar._AvatarMedia"))]
pub trait AvatarMedia: Send + Sync {
    /// Publish the audio track, wait for its subscription, then publish the video track.
    #[host(await, name = "_core_publish")]
    fn publish(&self) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
    /// One frame into the synchronizer, which paces it out.
    #[host(await, name = "_core_push")]
    fn push(&self, frame: MediaFrame) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
    #[host(await, name = "_core_aclose")]
    fn aclose(&self) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
}

/// The agent's native audio sink, sending one byte stream per avatar speech segment.
#[crate::host(class(name = "DataStreamAudioOutput", native = crate::voice::io::AudioOutput))]
pub trait DataStreamAudioOutput: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(
        env: &Self::Env,
        room: crate::HostRef,
        #[host(kw)] destination_identity: String,
        #[host(default = None)] sample_rate: Option<u32>,
        #[host(default = None)] wait_remote_track: Option<i32>,
        #[host(default = 2.0)] clear_buffer_timeout: Option<f64>,
        #[host(default = false)] wait_playback_start: bool,
    ) -> Result<Self, ProviderError>;

    #[host(getter)]
    fn label(&self) -> String;
    #[host(getter)]
    fn sample_rate(&self) -> Option<u32>;

    /// Stop admitting audio, settle startup and pending writes, close streams and retire RPC routes.
    #[host(await)]
    async fn aclose(&self);
}

/// The worker's receiver keeps stream identity with each delivered frame and playback report.
#[crate::host(class(name = "DataStreamAudioReceiver", native = AvatarAudioReceiver))]
pub trait DataStreamAudioReceiver: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(
        env: &Self::Env,
        room: std::sync::Arc<dyn crate::voice::rtc::RtcRoom>,
        #[host(kw, default = None)] sender_identity: Option<String>,
        #[host(default = 100)] frame_size_ms: u32,
        #[host(default = 3)] rpc_max_retries: u32,
        #[host(default = None)] on_clear_buffer: Option<std::sync::Arc<dyn ClearBufferListener>>,
    ) -> Result<Self, ProviderError>;
    #[host(await)]
    async fn start(&self) -> Result<(), ProviderError>;
    #[host(await)]
    async fn next_audio(
        &self,
    ) -> Result<(u64, bool, Option<AvatarPlaybackSegment>, Option<AudioReceiverFrame>), ProviderError>;
    #[host(call)]
    fn notify_playback_started(&self, segment: AvatarPlaybackSegment) -> Result<(), ProviderError>;
    #[host(call)]
    fn notify_playback_finished(
        &self,
        segment: AvatarPlaybackSegment,
        playback_position: f64,
        interrupted: bool,
    ) -> Result<(), ProviderError>;
    #[host(call)]
    fn aclose(&self) -> Result<(), ProviderError>;
}

/// Why the runner refused a call, or what its loops ended with (`wait_for_complete`).
#[crate::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RunnerError {
    #[error("AvatarRunner not started")]
    NotStarted,
    #[host(kind = "asyncio.CancelledError")]
    #[error("the avatar runner was closed")]
    Closed,
    /// What the receiver, the generator or the synchronizer raised.
    #[host(delegate)]
    #[error(transparent)]
    Failed(ProviderError),
    #[error(transparent)]
    Rtc(#[from] crate::voice::rtc::RtcError),
}

#[crate::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReceiverEnd {
    #[host(kind = "StopAsyncIteration")]
    #[error("the receiver is closed")]
    Closed,
    /// A stream without `sample_rate`/`num_channels`: the reference's `ValueError`.
    #[host(kind = "ValueError")]
    #[error("{0}")]
    BadStream(String),
    #[error("{0}")]
    Failed(String),
}

/// The avatar worker's native queue owner. Its decisions remain in RunnerCore.
#[crate::host(class(name = "AvatarRunner"))]
pub trait AvatarRunner: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(
        env: &Self::Env,
        room: crate::voice::room_owner::RoomSession,
        #[host(kw)] receiver: std::sync::Arc<dyn AvatarAudioReceiver>,
        generator: std::sync::Arc<dyn AvatarVideoGenerator>,
        media: std::sync::Arc<dyn AvatarMedia>,
        lazy_publish: bool,
    ) -> Result<Self, crate::voice::rtc::RtcError>;
    #[host(await)]
    async fn start(&self) -> Result<(), RunnerError>;
    #[host(await)]
    async fn wait_for_complete(&self) -> Result<(), RunnerError>;
    #[host(call)]
    fn clear_buffer(&self, epoch: u64);
    #[host(await)]
    async fn aclose(&self) -> Result<(), RunnerError>;
}
