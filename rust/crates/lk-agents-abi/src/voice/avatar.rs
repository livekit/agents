use crate::errors::ProviderError;
use crate::utils::video::{AudioReceiverFrame, AvatarFrame, MediaFrame};
use crate::voice::io::{
    AvatarPlaybackSegment, IoError, OutputAlreadyChained, PlaybackFinishedEvent, PlaybackListener,
    PlaybackSegment,
};

/// What the avatar runner reads from its receiver: the clear epoch the frame was captured in,
/// whether this first frame opened a segment, the segment, and the frame (`None` at the end).
#[crate::host(record(module = "avatar", extract, emit))]
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AudioReceiverObservation {
    pub epoch: u64,
    pub starts_segment: bool,
    pub segment: Option<AvatarPlaybackSegment>,
    pub frame: Option<AudioReceiverFrame>,
}

/// The `clear_buffer` event `DataStreamAudioReceiver` emits when the agent clears its audio.
#[crate::host(proxy(host = "livekit.agents.voice.avatar.AudioReceiver"))]
pub trait ClearBufferListener: Send + Sync {
    #[host(notify, emit = "clear_buffer")]
    fn clear_buffer(&self);
}

/// Audio the avatar runner reads natively: each item stamped with the clear epoch it was
/// captured in, and the clears themselves.
#[crate::host(native(refused = "expected a native avatar audio source"))]
pub trait AvatarAudio: std::any::Any + Send + Sync {
    /// The next observation; its `frame` is `None` at the end.
    fn next_audio(&self) -> crate::BoxFuture<'_, Result<AudioReceiverObservation, ProviderError>>;
    /// The epoch of the receiver's latest clear once it is past `after`; `None` once it closed.
    fn next_clear(&self, after: u64) -> crate::BoxFuture<'_, Option<u64>>;
}

/// `AudioReceiver`: the agent's audio as the avatar worker receives it, and where its playback is
/// reported: `DataStreamAudioReceiver`, or `HostAudioReceiver` over any other.
#[crate::host(native(refused = "expected a DataStreamAudioReceiver or a HostAudioReceiver"))]
pub trait AvatarAudioReceiver: AvatarAudio {
    fn notify_playback_started(
        &self,
        segment: AvatarPlaybackSegment,
    ) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
    fn notify_playback_finished(
        &self,
        playback_position: f64,
        interrupted: bool,
        segment: AvatarPlaybackSegment,
    ) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
    fn aclose(&self) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
}

/// A host `AudioReceiver` (`voice/avatar.py`): its audio iterated, and its playback reports.
#[crate::host(proxy(host = "livekit.agents.voice.avatar.AudioReceiver"))]
pub trait AudioReceiverHost: Send + Sync {
    /// The next frame or segment end; `None` once the receiver is exhausted.
    #[host(await, iter)]
    fn next_frame(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<Option<AudioReceiverFrame>, ProviderError>>;
    #[host(await)]
    fn notify_playback_started(
        &self,
        cx: &crate::telemetry::Cx,
        #[host(kw)] segment: AvatarPlaybackSegment,
    ) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
    #[host(await)]
    fn notify_playback_finished(
        &self,
        cx: &crate::telemetry::Cx,
        playback_position: f64,
        interrupted: bool,
        #[host(kw)] segment: AvatarPlaybackSegment,
    ) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
    #[host(await)]
    fn aclose(&self, cx: &crate::telemetry::Cx) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
}

/// The avatar runner's receiver over a host `AudioReceiver`: its audio read from `audio` when
/// given (`QueueAudioOutput`'s native queue), else from the host's iterator into a queue that
/// drops its oldest audio only when far behind (measured) and is flushed by each `clear_buffer`.
/// Playback reports reach the host receiver. Its host calls run under the span it was made in.
#[crate::host(class(name = "HostAudioReceiver", native = AvatarAudioReceiver))]
pub trait HostAudioReceiver: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(
        env: &Self::Env,
        receiver: std::sync::Arc<dyn AudioReceiverHost>,
        #[host(kw, default = None)] audio: Option<std::sync::Arc<dyn AvatarAudio>>,
        cx: crate::telemetry::Cx,
    ) -> Result<Self, ProviderError>;
    /// The host receiver's `clear_buffer` event: the audio queued so far is dropped.
    #[host(call)]
    fn clear_buffer(&self);
}

/// `VideoGenerator`: the avatar's model, fed the audio and yielding the frames to publish. A host
/// `VideoGenerator` carries this surface (`voice/avatar.py`).
#[crate::host(proxy(host = "livekit.agents.voice.avatar.VideoGenerator"))]
pub trait AvatarVideoGenerator: Send + Sync {
    /// An audio frame, or `AudioSegmentEnd`.
    #[host(await)]
    fn push_audio(
        &self,
        cx: &crate::telemetry::Cx,
        frame: AudioReceiverFrame,
    ) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
    #[host(await)]
    fn clear_buffer(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<(), ProviderError>>;
    /// Begin generating. A host generator opens its iterator with its first read.
    fn open_frames(
        &self,
        _cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async { Ok(()) })
    }
    /// The next frame or segment end; `None` once the generator is exhausted.
    #[host(await, iter)]
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
#[crate::host(class(
    name = "DataStreamAudioOutput",
    native = crate::voice::io::AudioOutput,
    module = "livekit.agents.voice.avatar",
    subclass
))]
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

    // `io.AudioOutput`'s surface, so a host stage wraps this output (`next_in_chain=`) as it
    // wraps a host sink: the stage forwards its calls here and this output reports to it.

    /// Admit the key before capturing; raises when another segment is still capturing.
    #[host(call)]
    fn begin_segment(&self, segment: PlaybackSegment) -> Result<(), IoError>;
    /// A capture the stage refused or cancelled before any audio was taken: the segment it
    /// opened is withdrawn here too (`io.AudioOutput._withdraw_capture`).
    #[host(call, name = "_withdraw_capture")]
    fn withdraw_capture(&self) -> Result<(), IoError>;
    /// The admitted key, latched by the stage before it passes audio on.
    #[host(getter)]
    fn playback_segment(&self) -> Option<PlaybackSegment>;
    /// Waits for the avatar to join, then streams the frame into the segment's byte stream.
    #[host(await)]
    async fn capture_frame(&self, frame: crate::AudioFrame) -> Result<(), IoError>;
    /// Ends the segment: its stream's trailer ends it on the avatar.
    #[host(call)]
    fn flush(&self);
    /// Asks the avatar to drop what it holds of the segment (`lk.clear_buffer`).
    #[host(call)]
    fn clear_buffer(&self);
    /// False: the avatar cannot pause.
    #[host(getter)]
    fn can_pause(&self) -> bool;
    #[host(call)]
    fn pause(&self);
    #[host(call)]
    fn resume(&self);
    #[host(call)]
    fn on_attached(&self);
    #[host(call)]
    fn on_detached(&self);
    /// Until the segments captured so far played out; the last one's event.
    #[host(await)]
    async fn wait_for_playout(&self) -> PlaybackFinishedEvent;
    /// The stage built over this output (`next_in_chain=`): each playback report reaches its
    /// `on_playback_*`, in order, on its loop, until the close. Registering the same stage again
    /// does nothing; another one raises.
    #[host(call, name = "_set_parent")]
    fn set_parent(
        &self,
        parent: std::sync::Arc<dyn PlaybackListener>,
    ) -> Result<(), OutputAlreadyChained>;

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
    async fn next_audio(&self) -> Result<AudioReceiverObservation, ProviderError>;
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
    #[host(notify, emit = "metrics_collected")]
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

/// How long an avatar has to join after `AvatarSession.start()`: `wait_for_join`'s default
/// timeout.
pub const JOIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// `AvatarError`: the avatar session failed while the call went on (it did not join in time).
#[crate::host(record(
    module = "avatar_errors",
    path = "livekit.agents.voice.avatar",
    pydantic,
    arbitrary_types,
    constant(type = "avatar_error")
))]
pub struct AvatarError {
    #[host(default_factory = "time.time")]
    pub timestamp: f64,
    #[host(default = "avatar")]
    pub label: String,
    #[host(annotation = "typing.Annotated[Exception, pydantic.Field(exclude=True)]")]
    pub error: crate::HostRef,
    #[host(default = true)]
    pub recoverable: bool,
}

/// `_core_AvatarLifecycle`: an agent's avatar from `AvatarSession.start()` to its close. Created,
/// it waits in the background for the avatar participant to join `room` and publish video (once
/// `room` is connected) and reports the join as `AvatarMetrics` (`session_started_time` when the
/// room was connected, `avatar_joined_time` when the video was published); a session's assistant
/// message added with a `playback_latency` is reported as `AvatarMetrics(playback_latency=..)`
/// until the close. Each metric reaches the avatar's `metrics_collected` and the session's
/// metrics once it runs. Inside a job, the job's shutdown runs the avatar's `aclose`. An avatar
/// not joined [`JOIN_TIMEOUT`] after its start is the session's one recoverable `avatar_error`,
/// and its attached output stops waiting for it; the join wait goes on.
#[crate::host(class(name = "_core_AvatarLifecycle"))]
pub trait AvatarLifecycle: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;
    #[host(class(name = "_core_SessionOwner"))]
    type Owner: Send + Sync + 'static;
    #[host(class(name = "DataStreamAudioOutput"))]
    type Output: Send + Sync + 'static;

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

    /// The avatar's audio output: once the avatar has not joined within [`JOIN_TIMEOUT`], its
    /// audio fails at once with that error instead of waiting, until the avatar joins.
    #[host(call)]
    fn attach_output(&self, output: std::sync::Arc<Self::Output>);

    /// The avatar failed after it started (its audio can no longer reach it): the session's
    /// `avatar_error`. The avatar closes itself next.
    #[host(call)]
    fn fail(&self, error: crate::errors::ProviderError);

    /// Stops the join wait and the metrics, then, inside a job and while `room` is connected,
    /// removes the avatar participant from the room through the job's RoomService
    /// (`RemoveParticipant`): a participant already gone (`not_found`) is skipped, any other
    /// failure logged. Once, however many callers: each waits for the same close.
    #[host(await)]
    async fn aclose(&self);
}
