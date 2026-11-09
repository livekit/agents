use crate::AudioFrame;
use crate::errors::ProviderError;
use std::sync::Arc;

/// A caller-supplied stream: a host `AsyncIterator[rtc.AudioFrame]`.
#[crate::host(proxy(host = "collections.abc.AsyncIterator"))]
pub trait FrameStream: Send + Sync {
    /// The next frame, `None` once exhausted. `cancel` reaches the host's pending pull, which
    /// settles before the read does.
    #[host(await, iter)]
    fn next(
        &self,
        cx: &crate::telemetry::Cx,
        #[host(cancel)] cancel: Arc<dyn crate::invoke::Cancellation>,
    ) -> crate::BoxFuture<'_, Result<Option<AudioFrame>, ProviderError>>;
}

/// `play(loop=True)` with an iterator source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopingStreamError;

impl std::fmt::Display for LoopingStreamError {
    /// The message is public API, so it is reproduced exactly.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            "Looping sound via AsyncIterator is not supported. Use a string file path or your own 'infinite' AsyncIterator with loop=False",
        )
    }
}

impl std::error::Error for LoopingStreamError {}

/// A sound packaged with livekit-agents.
#[crate::host(record(
    module = "background_audio_wire",
    path = "livekit.agents.voice.background_audio",
    enum
))]
#[python(method(
    name = "path",
    ret = "str",
    hook = "builtin_audio_clip_path",
    doc = "The packaged file, on disk for the life of the process."
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinAudioClip {
    #[host(rename = "city-ambience.ogg")]
    CityAmbience,
    #[host(rename = "forest-ambience.ogg")]
    ForestAmbience,
    #[host(rename = "office-ambience.ogg")]
    OfficeAmbience,
    #[host(rename = "crowded-room.ogg")]
    CrowdedRoom,
    #[host(rename = "keyboard-typing.ogg")]
    KeyboardTyping,
    #[host(rename = "keyboard-typing2.ogg")]
    KeyboardTyping2,
    #[host(rename = "hold_music.ogg")]
    HoldMusic,
}

/// What a sound plays: a builtin clip (read as its packaged file), a file path, which can loop,
/// or a host `AsyncIterator[rtc.AudioFrame]`.
#[crate::host(record(
    module = "background_audio_wire",
    path = "livekit.agents.voice.background_audio",
    name = "AudioSource",
    transparent,
    extract
))]
#[derive(Clone)]
pub enum SoundSource {
    #[host(
        host_type = "livekit.agents.voice.background_audio.BuiltinAudioClip",
        extract_with = "py_values::builtin_clip_path"
    )]
    Clip(String),
    File(String),
    #[host(annotation = "collections.abc.AsyncIterator[livekit.rtc.AudioFrame]")]
    Stream(Arc<dyn FrameStream>),
}

/// The audio to be played in the background.
#[crate::host(record(
    module = "background_audio_wire",
    path = "livekit.agents.voice.background_audio",
    name = "AudioConfig",
    dataclass,
    frozen,
    slots,
    extract,
    default
))]
#[derive(Clone)]
pub struct Sound {
    pub source: SoundSource,
    /// The volume of the audio (0.0-1.0).
    #[host(default = 1.0)]
    pub volume: f64,
    /// The chance this one is picked when a list of sounds is offered (0.0-1.0).
    #[host(default = 1.0)]
    pub probability: f64,
    /// Seconds to ramp from 0 up to `volume` when playback starts. `0` starts at full volume.
    #[host(default = 0.0)]
    pub fade_in: f64,
    /// Seconds to ramp back down to 0 when `PlayHandle.stop()` is called. `0` cuts immediately.
    #[host(default = 0.0)]
    pub fade_out: f64,
}

/// One sound, or a list of sounds one is drawn from by probability (a total below 1.0 may draw
/// none: silence).
#[crate::host(record(
    module = "background_audio_wire",
    path = "livekit.agents.voice.background_audio",
    transparent,
    extract
))]
#[derive(Clone)]
pub enum Sounds {
    Choice(Vec<Sound>),
    One(Sound),
    #[host(otherwise)]
    Source(SoundSource),
}

impl Sounds {
    /// The candidates, and whether one is drawn by probability (a list).
    pub fn candidates(self) -> (Vec<Sound>, bool) {
        match self {
            Sounds::Choice(sounds) => (sounds, true),
            Sounds::One(sound) => (vec![sound], false),
            Sounds::Source(source) => (vec![Sound::new(source)], false),
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[crate::host(error(kind = "RuntimeError"))]
pub enum PlaybackError {
    #[error("BackgroundAudio is not started")]
    NotStarted,
    #[error("BackgroundAudio is closed")]
    Closed,
    #[error(transparent)]
    #[host(kind = "ValueError")]
    LoopingStream(#[from] LoopingStreamError),
}

/// A sound's monotone stop and completion handle.
#[crate::host(class(name = "PlayState"))]
pub trait PlayState: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(#[host(default = 0.0)] fade_out: f64) -> Self;
    #[host(getter)]
    fn done(&self) -> bool;
    #[host(getter)]
    fn stop_requested(&self) -> bool;
    #[host(call)]
    fn stop(&self) -> bool;
    #[host(call, name = "mark_playout_done")]
    fn finish(&self) -> bool;
    #[host(await, name = "wait_for_playout")]
    async fn wait(&self);
}

/// Runtime-owned mixer and published background track.
#[crate::host(class(name = "BackgroundAudioPlayer"))]
pub trait BackgroundAudioPlayer: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(class(name = "PlayState"))]
    type State: PlayState;

    #[host(new)]
    fn new(
        env: &Self::Env,
        #[host(default = 48_000)] sample_rate: u32,
        #[host(default = 1)] num_channels: u16,
        #[host(kw, default = 4800)] blocksize: usize,
        #[host(default = 200)] stream_timeout_ms: u64,
    ) -> Result<Self, ProviderError>;

    #[host(await)]
    async fn start(
        &self,
        room: std::sync::Arc<dyn crate::voice::room_owner::SessionRoom>,
        draw: f64,
        #[host(kw)] ambient: Option<Sounds>,
        #[host(default = None, sdk)] source: Option<crate::voice::rtc::TrackSource>,
    ) -> Result<crate::voice::rtc::TrackSid, crate::voice::rtc::RtcError>;

    #[host(call)]
    fn play(
        &self,
        sounds: Sounds,
        draw: f64,
        #[host(kw, default = false)] looping: bool,
        #[host(default = false)] ambient: bool,
    ) -> Result<Self::State, PlaybackError>;

    #[host(call)]
    fn on_agent_state(
        &self,
        sounds: Option<Sounds>,
        draw: f64,
        thinking: bool,
    ) -> Result<(), PlaybackError>;

    #[host(getter)]
    fn closed(&self) -> bool;
    #[host(await)]
    async fn aclose(&self);
}

/// `voice.PlayHandle`: one sound `BackgroundAudioPlayer.play()` started; awaiting it waits for
/// its playout.
#[crate::host(facade(
    path = "livekit.agents.voice.background_audio",
    name = "PlayHandle",
    native = PlayState,
    from_native,
    extended
))]
pub trait PlayHandleFacade {
    /// Whether the sound finished playing.
    #[host(native = done)]
    fn done(&self) -> bool;
    /// Stop the sound. A source started with a `fade_out` fades out first and stays not done
    /// until its tail played; with none, playback is cut at once.
    fn stop(&self);
    /// Wait for the sound to finish playing.
    async fn wait_for_playout(&self);
}
