use async_trait::async_trait;
use lk_agents_abi::AudioFrame;
use lk_agents_abi::errors::ProviderError;
use lk_agents_media::{Mixer, StreamId};
use lk_agents_rtc::{
    AudioSource, LocalParticipant, RtcError, TrackPublication, TrackPublishOptions, TrackSource,
};
use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

// Kept small because `clear_queue()` would cut ambient sounds abruptly: a sound removed from the
// mixer drains out within this window instead.
pub const AUDIO_SOURCE_BUFFER_MS: u32 = 400;

/// The published track's name; `aclose` resolves by name since a reconnect makes the cached SID
/// stale (#5698).
pub const TRACK_NAME: &str = "background_audio";

/// Resolved by name: `BuiltinAudioClip.path()` is public API, and the files are git-LFS objects
/// that `include_bytes!` could silently embed as pointers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinAudioClip {
    CityAmbience,
    ForestAmbience,
    OfficeAmbience,
    CrowdedRoom,
    KeyboardTyping,
    KeyboardTyping2,
    HoldMusic,
}

impl BuiltinAudioClip {
    /// The enum's Python value, the file name (public API).
    pub fn file_name(self) -> &'static str {
        match self {
            BuiltinAudioClip::CityAmbience => "city-ambience.ogg",
            BuiltinAudioClip::ForestAmbience => "forest-ambience.ogg",
            BuiltinAudioClip::OfficeAmbience => "office-ambience.ogg",
            BuiltinAudioClip::CrowdedRoom => "crowded-room.ogg",
            BuiltinAudioClip::KeyboardTyping => "keyboard-typing.ogg",
            BuiltinAudioClip::KeyboardTyping2 => "keyboard-typing2.ogg",
            // The one shipped name with an underscore; not a typo.
            BuiltinAudioClip::HoldMusic => "hold_music.ogg",
        }
    }
}

/// `AudioSource = AsyncIterator[rtc.AudioFrame] | str | BuiltinAudioClip`; a builtin clip arrives
/// as its file path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SoundSource {
    File(String),
    /// A host-owned `AsyncIterator[rtc.AudioFrame]`; the binding pulls it and applies [`Playback`].
    Stream,
}

/// `AudioConfig` -- one candidate sound with its playback parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioConfig {
    pub source: SoundSource,
    /// 0.0-1.0.
    pub volume: f64,
    /// The chance this one is picked when several are offered. 0.0-1.0.
    pub probability: f64,
    /// Seconds to ramp from 0 up to `volume` when playback starts. `0` starts at full volume.
    pub fade_in: f64,
    /// Seconds to ramp back down to 0 when [`PlaybackHandle::stop`] is called. `0` cuts immediately.
    pub fade_out: f64,
}

/// `_select_sound_from_list`: pick one config by probability, or none. `rand` supplies the draws in
/// order; the silence roll happens only when the probabilities sum below one.
pub fn select_sound(sounds: &[AudioConfig], rand: &mut dyn FnMut() -> f64) -> Option<usize> {
    let total: f64 = sounds.iter().map(|s| s.probability).sum();
    if total <= 0.0 {
        return None;
    }
    // A total below 1.0 is a chance of silence, by design.
    if total < 1.0 && rand() > total {
        return None;
    }

    let normalize_factor = if total <= 1.0 { 1.0 } else { total };
    let r = rand() * total.min(1.0);
    let mut cumulative = 0.0;
    for (i, sound) in sounds.iter().enumerate() {
        if sound.probability <= 0.0 {
            continue;
        }
        cumulative += sound.probability / normalize_factor;
        if r <= cumulative {
            return Some(i);
        }
    }
    // Float error can leave `r` above the final sum: the last entry wins.
    Some(sounds.len() - 1)
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

/// `None` when no change is needed, else one value per sample with volume and the equal-power fade
/// baked in. Mixed-width as numpy: `f64` ramp, `f32` accumulator.
pub fn frame_gain(
    t: u64,
    n: usize,
    stop_t: Option<u64>,
    fade_in: f64,
    fade_out: f64,
    sample_rate: u32,
    volume: f64,
) -> Option<Vec<f32>> {
    let sr = f64::from(sample_rate);
    let fade_in_n = if fade_in > 0.0 { (fade_in * sr) as u64 } else { 0 };
    let fade_out_n = if fade_out > 0.0 { (fade_out * sr) as u64 } else { 0 };
    let needs_fade_in = fade_in_n > 0 && t < fade_in_n;
    let needs_fade_out = fade_out_n > 0 && stop_t.is_some();
    if !needs_fade_in && !needs_fade_out && volume == 1.0 {
        return None;
    }

    let mut gain = vec![volume as f32; n];
    if needs_fade_in {
        for (i, g) in gain.iter_mut().enumerate() {
            let phase = ((t + i as u64) as f64 / fade_in_n as f64).clamp(0.0, 1.0);
            *g *= (phase * (PI / 2.0)).sin() as f32;
        }
    }
    if let Some(stop_t) = stop_t
        && fade_out_n > 0
    {
        for (i, g) in gain.iter_mut().enumerate() {
            let idx = (t + i as u64) as f64;
            let phase = ((idx - stop_t as f64) / fade_out_n as f64).clamp(0.0, 1.0);
            *g *= (phase * (PI / 2.0)).cos() as f32;
        }
    }
    Some(gain)
}

/// Apply a per-sample gain to an interleaved i16 frame (`np.repeat(gain, num_channels)`).
pub fn apply_gain(pcm: &[i16], gain: &[f32], num_channels: u16) -> Vec<i16> {
    let ch = num_channels.max(1) as usize;
    pcm.iter()
        .enumerate()
        .map(|(i, &s)| {
            let g = gain.get(i / ch).copied().unwrap_or(1.0);
            // `np.clip(...)` then `.astype(np.int16)`: saturate, then truncate toward zero.
            (f32::from(s) * g).clamp(-32768.0, 32767.0) as i16
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrameStep {
    /// `None`: pass the frame through untouched.
    pub gain: Option<Vec<f32>>,
    /// The fade-out has run its length; stop pulling frames after this one.
    pub last: bool,
}

/// Per-sound playback cursor (`_gen_wrapper`'s `t`, `stop_t`). The stop instant is latched once, at
/// the frame the stop was first seen.
#[derive(Debug, Clone, PartialEq)]
pub struct Playback {
    volume: f64,
    fade_in: f64,
    fade_out: f64,
    /// Cumulative samples (per channel) emitted so far.
    t: u64,
    stop_t: Option<u64>,
}

impl Playback {
    pub fn new(cfg: &AudioConfig) -> Self {
        Playback {
            volume: cfg.volume,
            fade_in: cfg.fade_in,
            fade_out: cfg.fade_out,
            t: 0,
            stop_t: None,
        }
    }

    /// Advance by one frame. With `fade_out == 0` nothing is latched; the driver drops the stream.
    pub fn on_frame(
        &mut self,
        samples_per_channel: usize,
        sample_rate: u32,
        stop_requested: bool,
    ) -> FrameStep {
        if self.stop_t.is_none() && self.fade_out > 0.0 && stop_requested {
            self.stop_t = Some(self.t);
        }
        let gain = frame_gain(
            self.t,
            samples_per_channel,
            self.stop_t,
            self.fade_in,
            self.fade_out,
            sample_rate,
            self.volume,
        );
        self.t += samples_per_channel as u64;
        let last = match self.stop_t {
            Some(stop_t) => self.t - stop_t >= (self.fade_out * f64::from(sample_rate)) as u64,
            None => false,
        };
        FrameStep { gain, last }
    }
}

/// A caller-supplied stream. Cancellation must reach any pending host-side pull. A host
/// `AsyncIterator[rtc.AudioFrame]` is this trait through `background_audio.py`'s `_FrameStream`.
#[lk_agents_abi::host(proxy(host = "livekit.agents.voice.background_audio._FrameStream"))]
#[async_trait]
pub trait FrameStream: Send + Sync {
    async fn next(&self) -> Result<Option<AudioFrame>, ProviderError>;
    fn cancel_pending(&self);
    async fn close(&self);
}

#[derive(Clone)]
pub enum PlaybackInput {
    File(String),
    Stream(Arc<dyn FrameStream>),
}

/// `_core.Sound`: one candidate, as the host hands it over -- a file path (a builtin clip arrives
/// as its path) or a host stream.
#[lk_agents_abi::class(from_py_object)]
#[derive(Clone)]
pub struct Sound {
    pub config: AudioConfig,
    pub input: PlaybackInput,
}

#[lk_agents_abi::host(object)]
impl Sound {
    #[host(new(defaults(volume = 1.0, probability = 1.0, fade_in = 0.0, fade_out = 0.0)))]
    pub fn for_host(
        #[host(required)] path: Option<String>,
        stream: Option<Arc<dyn FrameStream>>,
        volume: f64,
        probability: f64,
        fade_in: f64,
        fade_out: f64,
    ) -> Self {
        let (source, input) = match (path, stream) {
            (_, Some(stream)) => (SoundSource::Stream, PlaybackInput::Stream(stream)),
            (path, None) => {
                let path = path.unwrap_or_default();
                (SoundSource::File(path.clone()), PlaybackInput::File(path))
            }
        };
        Sound { config: AudioConfig { source, volume, probability, fade_in, fade_out }, input }
    }
}

/// Uniform draws in `[0, 1)` for [`select_sound`]: the host's `random.random`, so a seeded host
/// run picks what the reference picks, one draw at a time.
#[lk_agents_abi::host(proxy(host = "livekit.agents.voice.background_audio._Random", live))]
pub trait RandomDraw: Send + Sync {
    fn draw(&self) -> f64;
}

/// Completion and stop notification owned by voice, shared with the caller's handle
/// (`PlayHandle`'s two write-once futures).
pub struct PlaybackHandle {
    fade_out: f64,
    stopped: AtomicBool,
    done: tokio::sync::watch::Sender<bool>,
}

/// `_core.PlayState`: what `PlayHandle` wraps.
#[lk_agents_abi::host(object(name = "PlayState", cell = "arc"))]
impl PlaybackHandle {
    pub fn new(fade_out: f64) -> Arc<Self> {
        Arc::new(Self::with(fade_out, false))
    }
    /// A handle nothing plays into yet (`PlayHandle()` built by hand).
    #[host(new(defaults(fade_out = 0.0)))]
    pub fn for_host(fade_out: f64) -> Self {
        Self::with(fade_out, false)
    }
    /// A sound that was never going to play (the probability draw selected silence) is born done.
    pub fn already_done() -> Arc<Self> {
        Arc::new(Self::with(0.0, true))
    }
    fn with(fade_out: f64, done: bool) -> Self {
        let (done, _) = tokio::sync::watch::channel(done);
        Self { fade_out, stopped: AtomicBool::new(false), done }
    }
    #[host(getter)]
    pub fn done(&self) -> bool {
        *self.done.borrow()
    }
    #[host(getter)]
    pub fn stop_requested(&self) -> bool {
        self.stopped.load(Ordering::Acquire)
    }
    /// Ask playback to stop; `true` if done now. With a fade-out the handle is not done yet, or
    /// the generator would be closed mid-fade.
    #[host(method)]
    pub fn stop(&self) -> bool {
        self.done.send_if_modified(|done| {
            if *done {
                return false;
            }
            self.stopped.store(true, Ordering::Release);
            *done = self.fade_out <= 0.0;
            *done
        })
    }
    /// `_mark_playout_done`. Idempotent, and called from several places for exactly that reason.
    #[host(method(name = "mark_playout_done"))]
    pub fn finish(&self) -> bool {
        self.done.send_if_modified(|done| !std::mem::replace(done, true))
    }
    #[host(method(name = "wait_for_playout"))]
    pub async fn wait(&self) {
        let mut done = self.done.subscribe();
        while !*done.borrow_and_update() {
            if done.changed().await.is_err() {
                break;
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[lk_agents_abi::host(error(kind = "RuntimeError"))]
pub enum PlaybackError {
    #[error("BackgroundAudio is not started")]
    NotStarted,
    #[error("BackgroundAudio is closed")]
    Closed,
    #[error(transparent)]
    #[host(kind = "ValueError")]
    LoopingStream(#[from] LoopingStreamError),
}

pub struct BackgroundAudioPlayer {
    state: tokio::sync::Mutex<MixerState>,
    queued: Arc<Mutex<Vec<Source>>>,
    output: tokio::sync::Mutex<Option<PublishedOutput>>,
    thinking: Mutex<Option<Arc<PlaybackHandle>>>,
    closed: AtomicBool,
    started: AtomicBool,
    pub sample_rate: u32,
    pub num_channels: u16,
    pub blocksize: usize,
    timeout: Duration,
}

struct PublishedOutput {
    participant: Arc<dyn LocalParticipant>,
    publication: Arc<dyn TrackPublication>,
    source: Arc<dyn AudioSource>,
    stop: tokio::sync::watch::Sender<bool>,
    pump: tokio::task::JoinHandle<()>,
}

/// `_core.BackgroundAudioPlayer`: the mixer and the published track, under `background_audio.py`.
#[lk_agents_abi::host(object(cell = "arc"))]
impl BackgroundAudioPlayer {
    #[host(new(
        kw_only = "blocksize",
        defaults(sample_rate = 48_000, num_channels = 1, blocksize = 4800, stream_timeout_ms = 200)
    ))]
    pub fn for_host(sample_rate: u32, num_channels: u16, blocksize: usize, stream_timeout_ms: u64) -> Self {
        Self::build(sample_rate, num_channels, blocksize, Duration::from_millis(stream_timeout_ms))
    }

    fn build(sample_rate: u32, num_channels: u16, blocksize: usize, timeout: Duration) -> Self {
        let mixer = Mixer::new(sample_rate, num_channels, blocksize);
        let blocksize = mixer.blocksize();
        let queued = Arc::new(Mutex::new(Vec::new()));
        Self {
            state: tokio::sync::Mutex::new(MixerState {
                mixer,
                sources: BTreeMap::new(),
                retired: Vec::new(),
                queued: queued.clone(),
            }),
            queued,
            output: tokio::sync::Mutex::new(None),
            thinking: Mutex::new(None),
            closed: AtomicBool::new(false),
            started: AtomicBool::new(false),
            sample_rate,
            num_channels,
            blocksize,
            timeout,
        }
    }

    /// [`start`](Self::start), for a host: the publication's sid, which the host resolves to its
    /// own `rtc.LocalTrackPublication`.
    #[host(method(name = "start"))]
    pub async fn start_for_host(
        self: &Arc<Self>,
        participant: Arc<dyn LocalParticipant>,
    ) -> Result<lk_agents_rtc::TrackSid, RtcError> {
        // Publish the track and own the pump. If the caller cancels, close waits and releases the
        // track.
        let this = self.clone();
        tokio::spawn(async move {
            let mut output = this.output.lock().await;
            if this.closed.load(Ordering::Acquire) {
                return Err(RtcError::Closed("background audio"));
            }
            if let Some(output) = output.as_ref() {
                return Ok(output.publication.sid().clone());
            }
            let (source, publication) = participant
                .publish_audio_track(TrackPublishOptions {
                    name: TRACK_NAME.into(),
                    source: TrackSource::Unknown,
                    sample_rate: this.sample_rate,
                    num_channels: this.num_channels,
                    queue_size_ms: AUDIO_SOURCE_BUFFER_MS,
                })
                .await?;
            let (stop, stop_rx) = tokio::sync::watch::channel(false);
            let engine = this.clone();
            let sink = source.clone();
            let pump = tokio::spawn(async move {
                if let Err(err) = engine.pump(sink, stop_rx).await {
                    log::error!("background audio capture failed: {err}");
                }
            });
            *output = Some(PublishedOutput {
                participant,
                source,
                publication: publication.clone(),
                stop,
                pump,
            });
            this.started.store(true, Ordering::Release);
            Ok(publication.sid().clone())
        })
        .await
        .map_err(|err| RtcError::host(err.to_string()))?
    }

    /// [`play`](Self::play), drawing from the host's `random.random`.
    #[host(method(name = "play", kw_only = "looping", defaults(looping = false, ambient = false, choose = true)))]
    pub fn play_for_host(
        &self,
        sounds: Vec<Sound>,
        rand: Arc<dyn RandomDraw>,
        looping: bool,
        ambient: bool,
        choose: bool,
    ) -> Result<Arc<PlaybackHandle>, PlaybackError> {
        self.play(sounds, choose, looping, ambient, &mut || rand.draw())
    }

    /// [`on_agent_state`](Self::on_agent_state), drawing from the host's `random.random`.
    #[host(method(name = "on_agent_state", kw_only = "choose", defaults(choose = true)))]
    pub fn on_agent_state_for_host(
        &self,
        sounds: Vec<Sound>,
        rand: Arc<dyn RandomDraw>,
        thinking: bool,
        choose: bool,
    ) -> Result<(), PlaybackError> {
        // `_agent_state_changed`. Re-entering "thinking" while the clip still plays must not layer
        // a second copy; `stop()` on a finished handle is a no-op.
        if sounds.is_empty() {
            return Ok(());
        }
        let mut handle = self.thinking.lock().unwrap();
        match (thinking, handle.as_ref()) {
            (true, Some(h)) if !h.done() => {}
            (true, _) => *handle = Some(self.play(sounds, choose, false, false, &mut || rand.draw())?),
            (false, Some(h)) => {
                h.stop();
            }
            (false, None) => {}
        }
        Ok(())
    }

    pub fn play(
        &self,
        sounds: Vec<Sound>,
        choose: bool,
        looping: bool,
        ambient: bool,
        rand: &mut dyn FnMut() -> f64,
    ) -> Result<Arc<PlaybackHandle>, PlaybackError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(PlaybackError::Closed);
        }
        if !self.started.load(Ordering::Acquire) {
            return Err(PlaybackError::NotStarted);
        }
        let configs: Vec<_> = sounds.iter().map(|sound| sound.config.clone()).collect();
        let index =
            if choose { select_sound(&configs, rand) } else { (!sounds.is_empty()).then_some(0) };
        let Some(index) = index else {
            return Ok(PlaybackHandle::already_done());
        };
        let sound = sounds.into_iter().nth(index).unwrap();
        // `loop_ambient = isinstance(cfg.source, str)`: files and builtins loop, iterators don't.
        let is_stream = sound.config.source == SoundSource::Stream;
        let looping = looping || (ambient && !is_stream);
        if looping && is_stream {
            return Err(LoopingStreamError.into());
        }
        let handle = PlaybackHandle::new(sound.config.fade_out);
        let input = match sound.input {
            PlaybackInput::File(path) => {
                Input::File(FileInput { path, looping, receiver: None, worker: None })
            }
            PlaybackInput::Stream(stream) => Input::Stream(stream),
        };
        self.queued.lock().unwrap().push(Source {
            input,
            pending: None,
            playback: Playback::new(&sound.config),
            handle: handle.clone(),
        });
        Ok(handle)
    }

    pub async fn next_block(&self) -> Option<Vec<i16>> {
        while !self.closed.load(Ordering::Acquire) {
            if let Some(pcm) = self.state.lock().await.round(self.timeout).await {
                return Some(pcm);
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        None
    }
    async fn pump(
        &self,
        source: Arc<dyn AudioSource>,
        mut stop: tokio::sync::watch::Receiver<bool>,
    ) -> Result<(), RtcError> {
        loop {
            let pcm = tokio::select! { biased; _ = stop.changed() => return Ok(()), pcm = self.next_block() => pcm };
            let Some(pcm) = pcm else {
                return Ok(());
            };
            let data: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
            let frame =
                AudioFrame::new(data, self.sample_rate, self.num_channels, self.blocksize as u32);
            tokio::select! { biased; _ = stop.changed() => return Ok(()), result = source.capture(&frame) => { result?; } }
        }
    }

    /// `aclose()`: stops the output and every playing sound.
    #[host(method(name = "aclose"))]
    pub async fn close(self: &Arc<Self>) {
        self.closed.store(true, Ordering::Release);
        let this = self.clone();
        let _ = tokio::spawn(async move {
            let mut output = this.output.lock().await;
            if let Some(output) = output.take() {
                let _ = output.stop.send(true);
                let _ = output.pump.await;
                let _ = output.source.close().await;
                // The SDK can replace the publication on reconnect; resolve the current SID.
                if let Some(publication) = output
                    .participant
                    .track_publications()
                    .into_iter()
                    .find(|p| p.name() == TRACK_NAME)
                {
                    let _ = output.participant.unpublish_track(publication.sid()).await;
                }
            }
            let mut state = this.state.lock().await;
            let mut sources: Vec<_> = std::mem::take(&mut state.sources).into_values().collect();
            sources.extend(this.queued.lock().unwrap().drain(..));
            let retired = std::mem::take(&mut state.retired);
            for id in state.mixer.stream_ids() {
                state.mixer.remove_stream(id);
            }
            drop(state);
            for task in retired {
                let _ = task.await;
            }
            for source in sources {
                source.close().await;
            }
        })
        .await;
    }
}

struct MixerState {
    mixer: Mixer,
    sources: BTreeMap<StreamId, Source>,
    retired: Vec<tokio::task::JoinHandle<()>>,
    queued: Arc<Mutex<Vec<Source>>>,
}

type PendingFrame = Pin<Box<dyn Future<Output = Result<Option<AudioFrame>, ProviderError>> + Send>>;

struct Source {
    input: Input,
    pending: Option<PendingFrame>,
    playback: Playback,
    handle: Arc<PlaybackHandle>,
}

enum Input {
    Stream(Arc<dyn FrameStream>),
    File(FileInput),
    Closed,
}

impl MixerState {
    async fn retire(&mut self, mut source: Source) {
        // Retain cleanup in mixer state so cancelling the pump cannot orphan a stream close.
        self.retired.retain(|task| !task.is_finished());
        self.retired.push(source.teardown());
        let _ = self.retired.last_mut().unwrap().await;
        self.retired.pop();
    }

    async fn round(&mut self, timeout: Duration) -> Option<Vec<i16>> {
        for source in self.queued.lock().unwrap().drain(..) {
            let id = self.mixer.add_stream();
            self.sources.insert(id, source);
        }
        for id in self.mixer.stream_ids() {
            if self.sources.get(&id).is_some_and(|s| s.handle.done()) {
                self.mixer.remove_stream(id);
                if let Some(source) = self.sources.remove(&id) {
                    self.retire(source).await;
                }
                continue;
            }
            while self.mixer.wants_more(id) {
                let Some(source) = self.sources.get_mut(&id) else {
                    break;
                };
                let frame = match &mut source.input {
                    Input::File(file) => match tokio::time::timeout(
                        timeout,
                        file.next(self.mixer.sample_rate(), self.mixer.num_channels()),
                    )
                    .await
                    {
                        Ok(Some(Ok(pcm))) => {
                            Some((pcm, self.mixer.sample_rate(), self.mixer.num_channels()))
                        }
                        Err(_) => {
                            self.mixer.note_timeout(id);
                            break;
                        }
                        other => {
                            if let Ok(Some(Err(err))) = other {
                                log::error!("background audio decode failed: {err}");
                            }
                            None
                        }
                    },
                    Input::Stream(stream) => {
                        if source.pending.is_none() {
                            let stream = stream.clone();
                            source.pending = Some(Box::pin(async move { stream.next().await }));
                        }
                        match tokio::time::timeout(timeout, source.pending.as_mut().unwrap()).await
                        {
                            Ok(result) => {
                                source.pending = None;
                                match result {
                                    Ok(Some(frame)) => Some((
                                        frame
                                            .data
                                            .chunks_exact(2)
                                            .map(|b| i16::from_le_bytes([b[0], b[1]]))
                                            .collect(),
                                        frame.sample_rate,
                                        frame.num_channels,
                                    )),
                                    Ok(None) => None,
                                    Err(err) => {
                                        log::error!("background audio stream failed: {err}");
                                        None
                                    }
                                }
                            }
                            Err(_) => {
                                self.mixer.note_timeout(id);
                                break;
                            }
                        }
                    }
                    Input::Closed => None,
                };
                let Some((pcm, rate, channels)) = frame else {
                    self.mixer.mark_exhausted(id);
                    break;
                };
                let (pcm, last) = source.process(pcm, rate, channels);
                self.mixer.push(id, &pcm);
                if last {
                    self.mixer.mark_exhausted(id);
                    break;
                }
            }
        }
        let block = self.mixer.mix();
        for id in self.mixer.drain_removals() {
            if let Some(source) = self.sources.remove(&id) {
                self.retire(source).await;
            }
        }
        block
    }
}

impl Source {
    fn process(&mut self, pcm: Vec<i16>, rate: u32, channels: u16) -> (Vec<i16>, bool) {
        let step = self.playback.on_frame(
            pcm.len() / usize::from(channels.max(1)),
            rate,
            self.handle.stop_requested(),
        );
        let pcm = match step.gain {
            Some(gain) => apply_gain(&pcm, &gain, channels),
            None => pcm,
        };
        (pcm, step.last)
    }
    async fn close(mut self) {
        let _ = self.teardown().await;
    }
    fn teardown(&mut self) -> tokio::task::JoinHandle<()> {
        if let Input::Stream(stream) = &self.input {
            stream.cancel_pending();
        }
        let pending = self.pending.take();
        let input = std::mem::replace(&mut self.input, Input::Closed);
        tokio::spawn(async move {
            if let Some(pending) = pending {
                let _ = pending.await;
            }
            match input {
                Input::Stream(stream) => stream.close().await,
                Input::File(mut file) => {
                    file.receiver.take();
                    if let Some(worker) = file.worker.take() {
                        let _ = worker.await;
                    }
                }
                Input::Closed => {}
            }
        })
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        self.handle.finish();
        if !matches!(self.input, Input::Closed) {
            self.teardown();
        }
    }
}

struct FileInput {
    path: String,
    looping: bool,
    receiver: Option<tokio::sync::mpsc::Receiver<Result<Vec<i16>, String>>>,
    worker: Option<tokio::task::JoinHandle<()>>,
}

impl FileInput {
    async fn next(
        &mut self,
        sample_rate: u32,
        num_channels: u16,
    ) -> Option<Result<Vec<i16>, String>> {
        if self.receiver.is_none() {
            let (tx, rx) = tokio::sync::mpsc::channel(2);
            self.receiver = Some(rx);
            let path = self.path.clone();
            let looping = self.looping;
            // Decode off the event loop; dropping the receiver stops the worker.
            self.worker = Some(tokio::task::spawn_blocking(move || {
                let decode = || -> Result<(), String> {
                    use std::io::Read;
                    loop {
                        let mut file = std::fs::File::open(&path).map_err(|err| err.to_string())?;
                        let mut decoder =
                            lk_agents_media::Decoder::new("", sample_rate, num_channels)
                                .map_err(|err| err.to_string())?;
                        let mut buffer = [0; 4096];
                        let mut emitted = false;
                        loop {
                            let size = file.read(&mut buffer).map_err(|err| err.to_string())?;
                            let pcm = if size == 0 {
                                decoder.flush()
                            } else {
                                decoder.push(&buffer[..size])
                            }
                            .map_err(|err| err.to_string())?;
                            if !pcm.is_empty() {
                                emitted = true;
                                for frame in pcm.chunks(
                                    (sample_rate as usize / 100).max(1) * usize::from(num_channels),
                                ) {
                                    if tx.blocking_send(Ok(frame.to_vec())).is_err() {
                                        return Ok(());
                                    }
                                }
                            }
                            if size == 0 {
                                break;
                            }
                        }
                        if !looping || !emitted || tx.is_closed() {
                            return Ok(());
                        }
                    }
                };
                if let Err(err) = decode() {
                    let _ = tx.blocking_send(Err(err));
                }
            }));
        }
        self.receiver.as_mut().unwrap().recv().await
    }
}

