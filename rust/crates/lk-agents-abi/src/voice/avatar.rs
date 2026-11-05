//! The avatar's host contracts: its audio receiver, video generator and media, and what the
//! receiver tells its host.

/// A clear epoch, segment boundary, playback segment and optional audio frame.
pub type AudioReceiverObservation =
    (u64, bool, Option<AvatarPlaybackSegment>, Option<AudioReceiverFrame>);

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
    fn open_audio(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
    /// One observation: receiver clear epoch, whether this first frame opened a segment,
    /// and the frame (None at EOF). The receiver stamps it before crossing the async boundary.
    #[host(await, name = "_core_next_audio")]
    fn next_audio(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<AudioReceiverObservation, ProviderError>>;
    #[host(await, name = "_core_notify_playback_started")]
    fn notify_playback_started(
        &self,
        cx: &crate::telemetry::Cx,
        segment: AvatarPlaybackSegment,
    ) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
    #[host(await, name = "_core_notify_playback_finished")]
    fn notify_playback_finished(
        &self,
        cx: &crate::telemetry::Cx,
        segment: AvatarPlaybackSegment,
        playback_position: f64,
        interrupted: bool,
    ) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
    #[host(await, name = "_core_aclose")]
    fn aclose(&self, cx: &crate::telemetry::Cx) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
}

/// `VideoGenerator`: the avatar's model, fed the audio and yielding the frames to publish. A host
/// `VideoGenerator` carries this surface (`voice/avatar.py`).
#[crate::host(proxy(host = "livekit.agents.voice.avatar.VideoGenerator"))]
pub trait AvatarVideoGenerator: Send + Sync {
    /// An audio frame, or `None` for `AudioSegmentEnd`.
    #[host(await, name = "_core_push_audio")]
    fn push_audio(
        &self,
        cx: &crate::telemetry::Cx,
        frame: Option<MediaFrame>,
    ) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
    #[host(await, name = "_core_clear_buffer")]
    fn clear_buffer(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
    /// Begin iterating (`__aiter__`).
    #[host(await, name = "_core_open_frames")]
    fn open_frames(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
    /// The next frame or segment end; `None` once the generator is exhausted.
    #[host(await, name = "_core_next_frame")]
    fn next_frame(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<Option<AvatarFrame>, ProviderError>>;
}

/// Native media execution over the shared RTC source and publication capabilities.
pub trait AvatarMedia: Send + Sync {
    /// Publish the audio track, wait for its subscription, then publish the video track.
    fn publish(&self) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
    /// One frame into the synchronizer, which paces it out.
    fn push(&self, frame: MediaFrame) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
    fn aclose(&self) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
}

/// Invalid avatar sink configuration, or a failure to acquire its process resources.
#[derive(Debug, Clone, thiserror::Error)]
#[crate::host(error(kind = "ValueError"))]
pub enum AvatarOutputError {
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    #[host(delegate)]
    Provider(#[from] ProviderError),
}

/// The agent's native audio sink, sending one byte stream per avatar speech segment.
#[crate::host(class(name = "DataStreamAudioOutput", native = crate::voice::io::AudioOutput, subclass))]
pub trait DataStreamAudioOutput: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(
        env: &Self::Env,
        room: std::sync::Arc<dyn crate::voice::rtc::RtcRoom>,
        #[host(kw)] destination_identity: String,
        #[host(default = None)] sample_rate: Option<u32>,
        #[host(default = None)] wait_remote_track: Option<i32>,
        #[host(default = 2.0)] clear_buffer_timeout: Option<f64>,
        #[host(default = false)] wait_playback_start: bool,
    ) -> Result<Self, AvatarOutputError>;

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
    #[host(class(name = "AVSynchronizer"))]
    type Synchronizer;

    #[host(new)]
    fn new(
        env: &Self::Env,
        room: std::sync::Arc<dyn crate::voice::room_owner::SessionRoom>,
        #[host(kw)] receiver: std::sync::Arc<dyn AvatarAudioReceiver>,
        generator: std::sync::Arc<dyn AvatarVideoGenerator>,
        options: AvatarOptions,
        queue_size_ms: u32,
        lazy_publish: bool,
    ) -> Result<Self, crate::voice::rtc::RtcError>;
    #[host(getter)]
    fn av_sync(&self) -> std::sync::Arc<Self::Synchronizer>;
    #[host(await)]
    async fn start(&self) -> Result<(), RunnerError>;
    #[host(await)]
    async fn wait_for_complete(&self) -> Result<(), RunnerError>;
    #[host(call)]
    fn clear_buffer(&self, epoch: u64);
    #[host(await)]
    async fn aclose(&self) -> Result<(), RunnerError>;
}

#[crate::host(record(
    module = "avatar_options",
    path = "livekit.agents.voice.avatar",
    dataclass,
    extract,
    emit
))]
#[derive(Debug, Clone, PartialEq)]
pub struct AvatarOptions {
    pub video_width: u32,
    pub video_height: u32,
    pub video_fps: f64,
    pub audio_sample_rate: u32,
    pub audio_channels: u16,
}

/// The agent's `AvatarSession`, as its lifecycle reaches it.
#[crate::host(proxy(host = "livekit.agents.voice.avatar.AvatarSession"))]
pub trait AvatarHost: Send + Sync {
    /// The avatar's `metrics_collected` event.
    #[host(notify, name = "_core_metrics_collected")]
    fn metrics_collected(&self, metrics: crate::metrics::base::AvatarMetrics);
    /// The avatar's `aclose` (a plugin's own), which the job's shutdown runs.
    #[host(call, getter = "aclose")]
    fn close(&self) -> Result<crate::HostRef, crate::errors::HostError>;
}

/// Why `wait_for_join` returned without the avatar.
#[crate::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AvatarJoinError {
    /// The avatar had not published its video within the waiter's timeout. The join wait goes
    /// on; a later `wait_for_join` can still see it.
    #[host(kind = "TimeoutError")]
    #[error("the avatar did not join within {0}s")]
    Timeout(f64),
    /// The avatar closed before it joined, or while it was awaited.
    #[host(kind = "asyncio.CancelledError")]
    #[error("the avatar session was closed")]
    Closed,
    /// The room failed or disconnected before the avatar joined.
    #[error(transparent)]
    #[host(delegate)]
    Rtc(#[from] crate::voice::rtc::RtcError),
}

/// `_core_AvatarLifecycle`: an agent's avatar from `AvatarSession.start()` to its close. Created,
/// it waits in the background for the avatar participant to join `room` and publish video (once
/// `room` is connected) and reports the join as `AvatarMetrics` (`session_started_time` when the
/// room was connected, `avatar_joined_time` when the video was published); a session's assistant
/// message added with a `playback_latency` is reported as `AvatarMetrics(playback_latency=..)`
/// until the close. Each metric reaches the avatar's `metrics_collected` and the session's
/// metrics once it runs. Inside a job, the job's shutdown runs the avatar's `aclose`.
#[crate::host(class(name = "_core_AvatarLifecycle"))]
pub trait AvatarLifecycle: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;
    #[host(class(name = "_core_SessionOwner"))]
    type Owner: Send + Sync + 'static;

    #[host(new)]
    fn new(
        env: &Self::Env,
        owner: std::sync::Arc<Self::Owner>,
        room: std::sync::Arc<dyn crate::voice::rtc::RtcRoom>,
        #[host(kw)] identity: String,
        provider: String,
        avatar: std::sync::Arc<dyn AvatarHost>,
        #[host(default = None)] job: Option<std::sync::Arc<dyn crate::worker::job::SessionJob>>,
    ) -> Result<Self, crate::errors::HostError>;

    /// Until the avatar joined; `timeout` seconds at most (`None` waits as long as it takes).
    #[host(await)]
    async fn wait_for_join(&self, timeout: Option<f64>) -> Result<(), AvatarJoinError>;

    /// Stops the join wait and the metrics, then, inside a job and while `room` is connected,
    /// removes the avatar participant from the room through the job's RoomService
    /// (`RemoveParticipant`): a participant already gone (`not_found`) is skipped, any other
    /// failure logged. Once, however many callers: each waits for the same close.
    #[host(await)]
    async fn aclose(&self);
}
