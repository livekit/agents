//! A caller-supplied background audio stream.

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

/// What a sound plays: a file (a builtin clip arrives as its path) or a host stream.
#[crate::host(record(module = "background_audio_wire", extract))]
#[derive(Clone)]
pub enum SoundSource {
    #[host(field = "path")]
    File(String),
    #[host(field = "stream")]
    Stream(Arc<dyn FrameStream>),
}

/// One candidate sound, as the host hands it over, and how to play it.
#[crate::host(record(module = "background_audio_wire", extract))]
#[derive(Clone)]
pub struct Sound {
    pub source: SoundSource,
    /// 0.0-1.0.
    pub volume: f64,
    /// The chance this one is picked when several are offered. 0.0-1.0.
    pub probability: f64,
    /// Seconds to ramp from 0 up to `volume` when playback starts. `0` starts at full volume.
    pub fade_in: f64,
    /// Seconds to ramp back down to 0 when `stop()` is called. `0` cuts immediately.
    pub fade_out: f64,
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
        #[host(kw)] ambient: Vec<Sound>,
        choose: bool,
        #[host(default = None, sdk)] source: Option<crate::voice::rtc::TrackSource>,
    ) -> Result<crate::voice::rtc::TrackSid, crate::voice::rtc::RtcError>;

    #[host(call)]
    fn play(
        &self,
        sounds: Vec<Sound>,
        draw: f64,
        #[host(kw, default = false)] looping: bool,
        #[host(default = false)] ambient: bool,
        #[host(default = true)] choose: bool,
    ) -> Result<Self::State, PlaybackError>;

    #[host(call)]
    fn on_agent_state(
        &self,
        sounds: Vec<Sound>,
        draw: f64,
        thinking: bool,
        #[host(kw, default = true)] choose: bool,
    ) -> Result<(), PlaybackError>;

    #[host(getter)]
    fn closed(&self) -> bool;
    #[host(await)]
    async fn aclose(&self);
}
