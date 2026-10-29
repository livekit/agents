//! Background sound stream execution and room track ownership.

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::rtc::{
    AudioSource, LocalParticipant, RtcError, TrackPublication, TrackPublishOptions, TrackSource,
};
use lk_agents_abi::{AudioFrame, Clock, Spawner};
use lk_agents_inference::codecs::{AudioDecoder, DecoderError, decoder_limits};
use lk_agents_media::pool::MediaPool;
use lk_agents_media::{Mixer, StreamId};
use lk_agents_net::env::{ProcessEnv, MediaPoolSource};
use lk_agents_net::task::TaskSet;
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub use lk_agents_voice::background_audio::{
    AUDIO_SOURCE_BUFFER_MS, BuiltinAudioClip, LoopingStreamError, Playback,
    PlayerPhase, TRACK_NAME, ThinkingSound, apply_gain, plan_play, select_sound, thinking_sound,
};

pub use lk_agents_abi::voice::background_audio::{
    FrameStream, PlaybackError, Sound, SoundSource,
};

/// Completion and stop notification owned by voice, shared with the caller's handle
/// (`PlayHandle`'s two write-once futures).
pub struct PlaybackDone {
    fade_out: f64,

    stopped: AtomicBool,
    done: tokio::sync::watch::Sender<bool>,
}

/// `_core.PlayState`: what `PlayHandle` wraps.
impl PlaybackDone {
    pub fn new(fade_out: f64) -> Arc<Self> {
        Arc::new(Self::with(fade_out, false))
    }
    /// A handle nothing plays into yet (`PlayHandle()` built by hand).
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
    pub fn done(&self) -> bool {
        *self.done.borrow()
    }
    pub fn stop_requested(&self) -> bool {
        self.stopped.load(Ordering::Acquire)
    }
    /// Ask playback to stop; `true` if done now. With a fade-out the handle is not done yet, or
    /// the generator would be closed mid-fade.
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
    pub fn finish(&self) -> bool {
        self.done.send_if_modified(|done| !std::mem::replace(done, true))
    }
    pub async fn wait(&self) {
        let mut done = self.done.subscribe();
        while !*done.borrow_and_update() {
            if done.changed().await.is_err() {
                break;
            }
        }
    }
}

pub struct BackgroundAudioPlayer {
    clock: Arc<dyn Clock>,
    spawner: Arc<dyn Spawner>,
    /// Pumps, stops and stream closes, which outlive a cancelled waiter.
    tasks: Arc<TaskSet>,
    /// The media pool file sounds decode on.
    media_pool: MediaPoolSource,
    state: tokio::sync::Mutex<MixerState>,

    queued: Arc<Mutex<Vec<Source>>>,

    output: tokio::sync::Mutex<Option<PublishedOutput>>,

    thinking: Mutex<Option<Arc<PlaybackDone>>>,

    closed: AtomicBool,

    /// `aclose()` is stopping the sounds: agent states no longer start a thinking sound. Cleared
    /// if that close is cancelled, as the reference re-attaches its listener.
    closing: AtomicBool,

    started: AtomicBool,
    /// Woken by `play` and `close`: an idle mixer waits on it rather than polling.
    wake: tokio::sync::Notify,
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
    pump: tokio::sync::oneshot::Receiver<()>,
}

/// `_core.BackgroundAudioPlayer`: the mixer and the published track, under `background_audio.py`.
impl BackgroundAudioPlayer {
    pub fn for_host(
        sample_rate: u32,
        num_channels: u16,
        blocksize: usize,
        stream_timeout_ms: u64,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        media_pool: MediaPoolSource,
    ) -> Self {
        Self::build(
            sample_rate,
            num_channels,
            blocksize,
            Duration::from_millis(stream_timeout_ms),
            clock,
            spawner,
            media_pool,
        )
    }

    fn build(
        sample_rate: u32,
        num_channels: u16,
        blocksize: usize,
        timeout: Duration,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        media_pool: MediaPoolSource,
    ) -> Self {
        let mixer = Mixer::new(sample_rate, num_channels, blocksize);
        let blocksize = mixer.blocksize();
        let queued = Arc::new(Mutex::new(Vec::new()));
        Self {
            clock,
            spawner,
            tasks: Arc::default(),
            media_pool,
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
            closing: AtomicBool::new(false),
            started: AtomicBool::new(false),
            wake: tokio::sync::Notify::new(),
            sample_rate,
            num_channels,
            blocksize,
            timeout,
        }
    }

    /// `start(room)`: publishes the track on the room's local participant, starts the `ambient`
    /// sound (drawn by probability with `choose`; a file loops), and returns the track's sid,
    /// which the host resolves to its own `rtc.LocalTrackPublication`. A second start returns the
    /// same sid. `source` is `track_publish_options.source`.
    pub async fn start_for_host(
        self: &Arc<Self>,
        room: Arc<dyn lk_agents_abi::voice::room_owner::SessionRoom>,
        draw: f64,
        ambient: Vec<Sound>,
        choose: bool,
        source: Option<TrackSource>,
    ) -> Result<lk_agents_abi::voice::rtc::TrackSid, RtcError> {
        let participant = room.rtc().ok_or(RtcError::NotConnected)?.local().clone();
        // Publish the track and own the pump. If the caller cancels, close waits and releases the
        // track.
        let this = self.clone();
        spawn_owned((&self.tasks, self.spawner.as_ref()), async move {
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
                    source: source.unwrap_or_default(),
                    sample_rate: this.sample_rate,
                    num_channels: this.num_channels,
                    queue_size_ms: AUDIO_SOURCE_BUFFER_MS,
                })
                .await?;
            let (stop, stop_rx) = tokio::sync::watch::channel(false);
            let engine = this.clone();
            let sink = source.clone();
            let pump = spawn_owned((&this.tasks, this.spawner.as_ref()), async move {
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
            if !ambient.is_empty() {
                // Open and started: `play` cannot refuse.
                let _ = this.play(ambient, choose, false, true, draw);
            }
            Ok(publication.sid().clone())
        })
        .await
        .map_err(|_| RtcError::host("background audio publication task stopped"))?
    }

    /// [`play`](Self::play), choosing by the host's `draw` (`random.random()`).
    pub fn play_for_host(
        &self,
        sounds: Vec<Sound>,
        draw: f64,
        looping: bool,
        ambient: bool,
        choose: bool,
    ) -> Result<Arc<PlaybackDone>, PlaybackError> {
        self.play(sounds, choose, looping, ambient, draw)
    }

    /// [`on_agent_state`](Self::on_agent_state), choosing by the host's `draw` (`random.random()`).
    pub fn on_agent_state_for_host(
        &self,
        sounds: Vec<Sound>,
        draw: f64,
        thinking: bool,
        choose: bool,
    ) -> Result<(), PlaybackError> {
        let mut handle = self.thinking.lock().unwrap();
        let last_done = handle.as_ref().map(|h| h.done());
        match thinking_sound(self.phase(), !sounds.is_empty(), thinking, last_done) {
            ThinkingSound::Keep => {}
            ThinkingSound::Start => *handle = Some(self.play(sounds, choose, false, false, draw)?),
            ThinkingSound::Stop => {
                if let Some(h) = handle.as_ref() {
                    h.stop();
                }
            }
        }
        Ok(())
    }

    fn phase(&self) -> PlayerPhase {
        PlayerPhase {
            started: self.started.load(Ordering::Acquire),
            closing: self.closing.load(Ordering::Acquire),
            closed: self.closed.load(Ordering::Acquire),
        }
    }

    pub fn play(
        &self,
        sounds: Vec<Sound>,
        choose: bool,
        looping: bool,
        ambient: bool,
        draw: f64,
    ) -> Result<Arc<PlaybackDone>, PlaybackError> {
        let Some((index, looping)) =
            plan_play(self.phase(), &sounds, choose, looping, ambient, draw)?
        else {
            return Ok(PlaybackDone::already_done());
        };
        let sound = sounds.into_iter().nth(index).expect("the plan names one of the sounds");
        let handle = PlaybackDone::new(sound.fade_out);
        let playback = Playback::new(&sound);
        let input = match sound.source {
            SoundSource::Stream(stream) => Input::Stream(stream),
            SoundSource::File(path) => Input::File(FileInput {
                path,
                looping,
                clock: self.clock.clone(),
                media_pool: self.media_pool,
                decoder: None,
            }),
        };
        self.queued.lock().unwrap().push(Source {
            input,
            pending: None,
            playback,
            handle: handle.clone(),
            spawner: self.spawner.clone(),
            tasks: self.tasks.clone(),
        });
        self.wake.notify_one();
        Ok(handle)
    }

    pub async fn next_block(&self) -> Option<Vec<i16>> {
        while !self.closed.load(Ordering::Acquire) {
            let woken = self.wake.notified();
            let idle = {
                let mut state = self.state.lock().await;
                if let Some(pcm) = state.round(self.timeout, self.clock.as_ref()).await {
                    return Some(pcm);
                }
                state.sources.is_empty() && self.queued.lock().unwrap().is_empty()
            };
            // Nothing to mix: wait for the next sound instead of polling. A play or close since
            // `woken` was taken has left its permit.
            if idle {
                woken.await;
            } else {
                self.clock
                    .wait_until(self.clock.now().saturating_add(Duration::from_millis(1)))
                    .await;
            }
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

    /// `closed`: `aclose()` got past stopping the sounds; the output is gone or going.
    pub fn closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    /// `aclose()`: stops every playing sound, then the output. Cancelled while the sounds stop,
    /// the player stays open with its output running (`closed` stays false), as the reference
    /// keeps its mixer; past that, the close runs to its end.
    pub async fn close(self: &Arc<Self>) {
        // Never started (or already closed): nothing to close, and a later `start()` still works.
        // A start in flight holds the output: its outcome is waited for.
        if self.closed.load(Ordering::Acquire) || self.output.lock().await.is_none() {
            return;
        }
        /// Clears `closing` unless the close got past stopping the sounds.

        struct Reopen<'a>(&'a AtomicBool, bool);
        impl Drop for Reopen<'_> {
            fn drop(&mut self) {
                if !self.1 {
                    self.0.store(false, Ordering::Release);
                }
            }
        }
        self.closing.store(true, Ordering::Release);
        let mut reopen = Reopen(&self.closing, false);
        let this = self.clone();
        let _ =
            spawn_owned(
                (&self.tasks, self.spawner.as_ref()),
                async move { this.stop_sounds().await },
            )
            .await;
        reopen.1 = true;
        self.closed.store(true, Ordering::Release);
        self.wake.notify_one();
        let this = self.clone();
        let _ = spawn_owned((&self.tasks, self.spawner.as_ref()), async move {
            let mut output = this.output.lock().await;
            if let Some(output) = output.take() {
                let _ = output.stop.send(true);
                let _ = output.pump.await;
                let _ = output.source.close().await;
                // The SDK can replace the publication on reconnect; resolve the current SID.
                match output.participant.track_publications() {
                    Ok(publications) => {
                        if let Some(publication) = publications.into_iter().find(|p| p.name() == TRACK_NAME) {
                            if let Err(error) = output.participant.unpublish_track(publication.sid()).await {
                                log::error!(target: "livekit.agents", "background audio publication cleanup failed: {error}");
                            }
                        }
                    }
                    Err(error) => log::error!(target: "livekit.agents", "background audio publication lookup failed: {error}"),
                }
            }
            drop(output);
            this.stop_sounds().await;
        })
        .await;
    }

    async fn stop_sounds(&self) {
        let mut state = self.state.lock().await;
        let mut sources: Vec<_> = std::mem::take(&mut state.sources).into_values().collect();
        sources.extend(self.queued.lock().unwrap().drain(..));
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
    }
}

/// The player's task set owns the work even if its waiter is cancelled.
fn spawn_owned<T: Send + 'static>(
    (tasks, spawner): (&Arc<TaskSet>, &dyn Spawner),
    future: impl Future<Output = T> + Send + 'static,
) -> tokio::sync::oneshot::Receiver<T> {
    let (done, completed) = tokio::sync::oneshot::channel();
    tasks.spawn(spawner, "background audio", async move {
        let _ = done.send(future.await);
    });
    completed
}

struct MixerState {
    mixer: Mixer,
    sources: BTreeMap<StreamId, Source>,
    retired: Vec<tokio::sync::oneshot::Receiver<()>>,

    queued: Arc<Mutex<Vec<Source>>>,
}

type PendingFrame = Pin<Box<dyn Future<Output = Result<Option<AudioFrame>, ProviderError>> + Send>>;

struct Source {
    input: Input,
    pending: Option<PendingFrame>,
    playback: Playback,
    handle: Arc<PlaybackDone>,
    spawner: Arc<dyn Spawner>,
    tasks: Arc<TaskSet>,
}

enum Input {
    Stream(Arc<dyn FrameStream>),
    File(FileInput),
    Closed,
}

impl MixerState {
    async fn retire(&mut self, mut source: Source) {
        // Retain cleanup in mixer state so cancelling the pump cannot orphan a stream close.
        self.retired.retain_mut(|task| {
            matches!(task.try_recv(), Err(tokio::sync::oneshot::error::TryRecvError::Empty))
        });
        self.retired.push(source.teardown());
        let _ = self.retired.last_mut().unwrap().await;
        self.retired.pop();
    }

    async fn round(&mut self, timeout: Duration, clock: &dyn Clock) -> Option<Vec<i16>> {
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
                    Input::File(file) => match lk_agents_abi::timeout(
                        clock,
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
                        match lk_agents_abi::timeout(
                            clock,
                            timeout,
                            source.pending.as_mut().unwrap(),
                        )
                        .await
                        {
                            Ok(result) => {
                                source.pending = None;
                                match result {
                                    Ok(Some(frame)) => Some((
                                        lk_agents_media::pcm_samples(&frame.data),
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

    fn teardown(&mut self) -> tokio::sync::oneshot::Receiver<()> {
        if let Input::Stream(stream) = &self.input {
            stream.cancel_pending();
        }
        let pending = self.pending.take();
        let input = std::mem::replace(&mut self.input, Input::Closed);
        spawn_owned((&self.tasks, self.spawner.as_ref()), async move {
            if let Some(pending) = pending {
                let _ = pending.await;
            }
            match input {
                Input::Stream(stream) => stream.close().await,
                Input::File(mut file) => {
                    if let Err(error) = file.close().await {
                        log::warn!("background audio cleanup incomplete: {error}");
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
            let _ = self.teardown();
        }
    }
}

struct FileInput {
    path: String,
    looping: bool,
    clock: Arc<dyn Clock>,
    media_pool: MediaPoolSource,
    decoder: Option<AudioDecoder>,
}

impl FileInput {
    async fn next(
        &mut self,
        sample_rate: u32,
        num_channels: u16,
    ) -> Option<Result<Vec<i16>, String>> {
        if self.decoder.is_none() {
            let opened = (self.media_pool)().map(MediaPool::from).and_then(|media_pool| {
                let limits = decoder_limits(sample_rate, num_channels);
                let path = self.path.clone();
                AudioDecoder::open_file(
                    &media_pool,
                    path,
                    self.looping,
                    sample_rate,
                    num_channels,
                    limits,
                )
                .map_err(|e| e.to_string())
            });
            match opened {
                Ok(decoder) => self.decoder = Some(decoder),
                Err(error) => return Some(Err(error)),
            }
        }
        match self.decoder.as_mut().unwrap().next().await {
            Ok(Some(frame)) => Some(Ok(lk_agents_media::pcm_samples(&frame.data))),
            Ok(None) => None,
            Err(error) => Some(Err(error.to_string())),
        }
    }

    /// Observes the decoder's native settlement, bounded by the host teardown budget.
    async fn close(&mut self) -> Result<(), DecoderError> {
        let Some(decoder) = self.decoder.as_mut() else { return Ok(()) };
        let limit = lk_agents_abi::HOST_TEARDOWN_TIMEOUT;
        lk_agents_abi::utils::clock::timeout(self.clock.as_ref(), limit, decoder.close())
            .await
            .unwrap_or(Err(DecoderError::IncompleteCleanup))
    }
}

#[cfg(test)]
mod tests;

/// The host class holds the same playback handle the mixer settles.
pub struct PlayState(pub Arc<PlaybackDone>);

impl lk_agents_abi::voice::background_audio::PlayState for PlayState {
    fn new(fade_out: f64) -> Self {
        Self(PlaybackDone::new(fade_out))
    }
    fn done(&self) -> bool {
        self.0.done()
    }
    fn stop_requested(&self) -> bool {
        self.0.stop_requested()
    }
    fn stop(&self) -> bool {
        self.0.stop()
    }
    fn finish(&self) -> bool {
        self.0.finish()
    }
    async fn wait(&self) {
        self.0.wait().await;
    }
}

/// Native class execution over the ABI-owned room capability.
pub struct BackgroundAudio(Arc<BackgroundAudioPlayer>);

impl lk_agents_abi::voice::background_audio::BackgroundAudioPlayer for BackgroundAudio {
    type State = PlayState;
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        sample_rate: u32,
        num_channels: u16,
        blocksize: usize,
        stream_timeout_ms: u64,
    ) -> Result<Self, ProviderError> {
        let spawner = (env.spawner)().map_err(lk_agents_abi::errors::ApiError::new)?;
        Ok(Self(Arc::new(BackgroundAudioPlayer::for_host(
            sample_rate,
            num_channels,
            blocksize,
            stream_timeout_ms,
            env.clock.clone(),
            spawner,
            env.media_pool,
        ))))
    }

    async fn start(
        &self,
        room: lk_agents_abi::voice::room_owner::RoomSession,
        draw: f64,
        ambient: Vec<Sound>,
        choose: bool,
        source: Option<TrackSource>,
    ) -> Result<lk_agents_abi::voice::rtc::TrackSid, RtcError> {
        self.0.start_for_host(room.0, draw, ambient, choose, source).await
    }

    fn play(
        &self,
        sounds: Vec<Sound>,
        draw: f64,
        looping: bool,
        ambient: bool,
        choose: bool,
    ) -> Result<Self::State, PlaybackError> {
        self.0.play_for_host(sounds, draw, looping, ambient, choose).map(PlayState)
    }

    fn on_agent_state(
        &self,
        sounds: Vec<Sound>,
        draw: f64,
        thinking: bool,
        choose: bool,
    ) -> Result<(), PlaybackError> {
        self.0.on_agent_state_for_host(sounds, draw, thinking, choose)
    }

    fn closed(&self) -> bool {
        self.0.closed()
    }
    async fn aclose(&self) {
        self.0.close().await;
    }
}
