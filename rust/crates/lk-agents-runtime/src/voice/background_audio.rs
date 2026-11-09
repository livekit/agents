#![allow(clippy::disallowed_types, clippy::disallowed_methods, reason = "migrating to signal")]
use crate::telemetry;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::rtc::{
    AudioSource, LocalParticipant, RtcError, TrackPublication, TrackPublishOptions, TrackSource,
};
use lk_agents_abi::{AudioFrame, Clock, Spawner};
use lk_agents_inference::codecs::{AudioDecoder, DecoderError, decoder_limits};
use lk_agents_media::media_queue::{self, AudioReceiver, AudioSender, Item, Policy};
use lk_agents_media::{Mixer, StreamId};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::task::TaskSet;
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

pub use lk_agents_voice::background_audio::{
    AUDIO_SOURCE_BUFFER_MS, BuiltinAudioClip, LoopingStreamError, Playback, PlayerPhase,
    TRACK_NAME, ThinkingSound, apply_gain, plan_play, select_sound, thinking_sound,
};

pub use lk_agents_abi::voice::background_audio::{
    FrameStream, PlaybackError, Sound, SoundSource, Sounds,
};

/// Completion and stop notification owned by voice, shared with the caller's handle
/// (`PlayHandle`'s two write-once futures).
pub struct PlaybackDone {
    fade_out: f64,

    stopped: CancellationToken,
    done: tokio::sync::watch::Sender<bool>,
}

/// `_core.PlayState`: what `PlayHandle` wraps.
impl PlaybackDone {
    pub fn new(fade_out: f64) -> Arc<Self> {
        Arc::new(Self::with(fade_out, false))
    }
    /// A sound that was never going to play (the probability draw selected silence) is born done.
    pub fn already_done() -> Arc<Self> {
        Arc::new(Self::with(0.0, true))
    }

    fn with(fade_out: f64, done: bool) -> Self {
        let (done, _) = tokio::sync::watch::channel(done);
        Self { fade_out, stopped: CancellationToken::new(), done }
    }
    pub fn done(&self) -> bool {
        *self.done.borrow()
    }
    pub fn stop_requested(&self) -> bool {
        self.stopped.is_cancelled()
    }
    /// Ask playback to stop; `true` if done now. With a fade-out the handle is not done yet, or
    /// the generator would be closed mid-fade.
    pub fn stop(&self) -> bool {
        self.done.send_if_modified(|done| {
            if *done {
                return false;
            }
            self.stopped.cancel();
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
    state: Mutex<MixerState>,

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
    stop: CancellationToken,
    pump: tokio::sync::oneshot::Receiver<()>,
}

impl PublishedOutput {
    /// Stops the pump, closes the source and unpublishes the track.
    async fn retire(self) {
        self.stop.cancel();
        let _ = self.pump.await;
        if let Err(error) = self.source.close(&telemetry::Cx::NONE).await {
            log::warn!(target: lk_agents_abi::LOG_TARGET, error:% = error; "background audio source close failed");
        }
        // The SDK can replace the publication on reconnect; resolve the current SID.
        let publications = match self.participant.track_publications() {
            Ok(publications) => publications,
            Err(error) => {
                log::error!(target: lk_agents_abi::LOG_TARGET, "background audio publication lookup failed: {error}");
                return;
            }
        };
        let Some(publication) = publications.into_iter().find(|p| p.name() == TRACK_NAME) else {
            return;
        };
        if let Err(error) =
            self.participant.unpublish_track(&telemetry::Cx::NONE, publication.sid()).await
        {
            log::error!(target: lk_agents_abi::LOG_TARGET, "background audio publication cleanup failed: {error}");
        }
    }
}

/// `_core.BackgroundAudioPlayer`: the mixer and the published track, under `background_audio.py`.
impl BackgroundAudioPlayer {
    pub fn new(
        sample_rate: u32,
        num_channels: u16,
        blocksize: usize,
        stream_timeout_ms: u64,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
    ) -> Self {
        let mixer = Mixer::new(sample_rate, num_channels, blocksize);
        let blocksize = mixer.blocksize();
        let queued = Arc::new(Mutex::new(Vec::new()));
        Self {
            clock,
            spawner,
            tasks: Arc::default(),
            state: Mutex::new(MixerState { mixer, sources: BTreeMap::new(), retired: Vec::new() }),
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
            timeout: Duration::from_millis(stream_timeout_ms),
        }
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
            SoundSource::Clip(path) | SoundSource::File(path) => {
                Input::File(FileInput { path, looping, clock: self.clock.clone(), decoder: None })
            }
        };
        self.queued.lock().push(Source {
            input,
            pending: None,
            cancel: None,
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
            let queued = std::mem::take(&mut *self.queued.lock());
            let sources: Vec<_> =
                queued.into_iter().map(|source| self.start_source(source)).collect();
            {
                let mut state = self.state.lock();
                for source in sources {
                    let id = state.mixer.add_stream();
                    state.sources.insert(id, source);
                }
            }
            let round = std::future::poll_fn(|cx| self.state.lock().poll_round(cx));
            let block = tokio::select! {
                biased;
                _ = woken => continue,
                block = round => block,
            };
            std::future::poll_fn(|cx| self.state.lock().poll_retired(cx)).await;
            if let Some(block) = block {
                return Some(block);
            }
        }
        None
    }

    fn start_source(&self, mut source: Source) -> SourceReader {
        let (audio, frames) = media_queue::audio(
            Policy::CatchUp(self.timeout),
            telemetry::AudioConsumer::BackgroundAudio,
            self.clock.clone(),
        );
        let request = Arc::new(tokio::sync::Notify::new());
        let stop = CancellationToken::new();
        let (requested, stopped) = (request.clone(), stop.clone());
        let clock = self.clock.clone();
        let timeout = self.timeout;
        let (rate, channels) = (self.sample_rate, self.num_channels);
        let handle = source.handle.clone();
        let playing = handle.clone();
        let done = spawn_owned((&self.tasks, self.spawner.as_ref()), async move {
            loop {
                tokio::select! {
                    biased;
                    _ = stopped.cancelled() => break,
                    _ = playing.wait() => break,
                    _ = requested.notified() => {},
                }
                let fetched = tokio::select! {
                    biased;
                    _ = stopped.cancelled() => break,
                    _ = playing.wait() => break,
                    fetched = source.next_pcm(rate, channels, timeout, clock.as_ref()) => fetched,
                };
                if !source.emit(fetched, &audio) {
                    break;
                }
            }
            source.close().await;
        });
        SourceReader {
            frames,
            request,
            cancel_on_drop: stop.clone().drop_guard(),
            stop,
            done,
            handle,
            phase: ReadPhase::Idle,
        }
    }

    async fn pump(
        &self,
        source: Arc<dyn AudioSource>,
        stop: CancellationToken,
    ) -> Result<(), RtcError> {
        loop {
            let pcm = tokio::select! { biased; _ = stop.cancelled() => return Ok(()), pcm = self.next_block() => pcm };
            let Some(pcm) = pcm else {
                return Ok(());
            };
            let data: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
            let frame =
                AudioFrame::new(data, self.sample_rate, self.num_channels, self.blocksize as u32);
            tokio::select! { biased; _ = stop.cancelled() => return Ok(()), result = source.capture(&telemetry::Cx::NONE, &frame) => { result?; } }
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
                output.retire().await;
            }
            drop(output);
            this.stop_sounds().await;
        })
        .await;
    }

    async fn stop_sounds(&self) {
        let retired = {
            let mut state = self.state.lock();
            for (_, source) in std::mem::take(&mut state.sources) {
                state.retire(source);
            }
            for id in state.mixer.stream_ids() {
                state.mixer.remove_stream(id);
            }
            std::mem::take(&mut state.retired)
        };
        let sources = std::mem::take(&mut *self.queued.lock());
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
    sources: BTreeMap<StreamId, SourceReader>,
    retired: Vec<tokio::sync::oneshot::Receiver<()>>,
}

struct SourceReader {
    frames: AudioReceiver<FetchedState>,
    request: Arc<tokio::sync::Notify>,
    stop: CancellationToken,
    cancel_on_drop: tokio_util::sync::DropGuard,
    done: tokio::sync::oneshot::Receiver<()>,
    handle: Arc<PlaybackDone>,
    phase: ReadPhase,
}

enum ReadPhase {
    Idle,
    Reading,
    Ready,
}

enum FetchedState {
    More,
    Ended,
    TimedOut,
}

type PendingFrame = Pin<Box<dyn Future<Output = Result<Option<AudioFrame>, ProviderError>> + Send>>;

struct Source {
    input: Input,
    pending: Option<PendingFrame>,
    /// Reaches the pending read's host pull when the source is torn down.
    cancel: Option<Arc<crate::host::Cancellation>>,
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
    fn poll_retired(&mut self, cx: &mut Context<'_>) -> Poll<()> {
        self.retired.retain_mut(|task| Pin::new(task).poll(cx).is_pending());
        if self.retired.is_empty() { Poll::Ready(()) } else { Poll::Pending }
    }

    fn retire(&mut self, source: SourceReader) {
        // Retain cleanup in mixer state so cancelling the pump cannot orphan a stream close.
        self.retired.retain_mut(|task| {
            matches!(task.try_recv(), Err(tokio::sync::oneshot::error::TryRecvError::Empty))
        });
        source.stop.cancel();
        drop(source.cancel_on_drop);
        self.retired.push(source.done);
    }

    fn poll_round(&mut self, cx: &mut Context<'_>) -> Poll<Option<Vec<i16>>> {
        if self.sources.is_empty() {
            return Poll::Pending;
        }
        let mut ready = true;
        for id in self.mixer.stream_ids() {
            if self.sources.get(&id).is_some_and(|s| s.handle.done()) {
                self.mixer.remove_stream(id);
                if let Some(source) = self.sources.remove(&id) {
                    self.retire(source);
                }
                continue;
            }
            ready &= self.poll_source(id, cx);
        }
        if !ready {
            return Poll::Pending;
        }
        let block = self.mixer.mix();
        for id in self.mixer.drain_removals() {
            if let Some(source) = self.sources.remove(&id) {
                self.retire(source);
            }
        }
        for source in self.sources.values_mut() {
            source.phase = ReadPhase::Idle;
        }
        Poll::Ready(block)
    }

    fn poll_source(&mut self, id: StreamId, cx: &mut Context<'_>) -> bool {
        let Some(source) = self.sources.get_mut(&id) else { return true };
        while !matches!(source.phase, ReadPhase::Ready) {
            if matches!(source.phase, ReadPhase::Idle) {
                if !self.mixer.wants_more(id) {
                    source.phase = ReadPhase::Ready;
                    break;
                }
                source.request.notify_one();
                source.phase = ReadPhase::Reading;
            }
            let received = std::pin::pin!(source.frames.recv()).as_mut().poll(cx);
            match received {
                Poll::Pending => return false,
                Poll::Ready(Some(Item::Media(frame))) => {
                    self.mixer.push(id, &lk_agents_media::pcm_samples(&frame.data));
                }
                Poll::Ready(Some(Item::Event(FetchedState::More))) => {
                    source.phase = ReadPhase::Idle
                }
                Poll::Ready(Some(Item::Event(FetchedState::TimedOut))) => {
                    self.mixer.note_timeout(id);
                    source.phase = ReadPhase::Ready;
                }
                Poll::Ready(None | Some(Item::Event(FetchedState::Ended))) => {
                    self.mixer.mark_exhausted(id);
                    source.phase = ReadPhase::Ready;
                }
                Poll::Ready(Some(Item::Gap(_))) => {}
            }
        }
        true
    }
}

/// One read from a sound's input, bounded by the stream timeout.
enum Fetched {
    Pcm(Vec<i16>, u32, u16),
    Ended,
    TimedOut,
}

impl Source {
    fn emit(&mut self, fetched: Fetched, audio: &AudioSender<FetchedState>) -> bool {
        let state = match fetched {
            Fetched::Pcm(pcm, rate, channels) => {
                let (pcm, last) = self.process(pcm, rate, channels);
                let samples = pcm.len() / usize::from(channels.max(1));
                let data: Vec<u8> = pcm.iter().flat_map(|sample| sample.to_le_bytes()).collect();
                audio.push(AudioFrame::new(data, rate, channels, samples as u32));
                if last { FetchedState::Ended } else { FetchedState::More }
            }
            Fetched::Ended => FetchedState::Ended,
            Fetched::TimedOut => FetchedState::TimedOut,
        };
        audio.event(state).is_ok() && !audio.is_closed()
    }

    /// The next PCM block, at the mixer's format for a file and at the frame's own for a stream.
    async fn next_pcm(
        &mut self,
        rate: u32,
        channels: u16,
        timeout: Duration,
        clock: &dyn Clock,
    ) -> Fetched {
        let stream = match &mut self.input {
            Input::File(file) => return file.next_within(rate, channels, timeout, clock).await,
            Input::Stream(stream) => stream,
            Input::Closed => return Fetched::Ended,
        };
        // A read that times out stays pending for the next round, so no frame is lost.
        let cancel = &mut self.cancel;
        let pending = self.pending.get_or_insert_with(|| {
            let (stream, cancel) = (stream.clone(), cancel.insert(Arc::default()).clone());
            Box::pin(async move { stream.next(&telemetry::Cx::NONE, cancel).await })
        });
        let Ok(result) = lk_agents_abi::timeout(clock, timeout, pending).await else {
            return Fetched::TimedOut;
        };
        self.pending = None;
        self.cancel = None;
        match result {
            Ok(Some(frame)) => Fetched::Pcm(
                lk_agents_media::pcm_samples(&frame.data),
                frame.sample_rate,
                frame.num_channels,
            ),
            Ok(None) => Fetched::Ended,
            Err(err) => {
                log::error!(target: lk_agents_abi::LOG_TARGET, "background audio stream failed: {err}");
                Fetched::Ended
            }
        }
    }

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
        if let Some(cancel) = self.cancel.take() {
            cancel.cancel();
        }
        let pending = self.pending.take();
        let input = std::mem::replace(&mut self.input, Input::Closed);
        spawn_owned((&self.tasks, self.spawner.as_ref()), async move {
            if let Some(pending) = pending {
                let _ = pending.await;
            }
            match input {
                Input::File(mut file) => {
                    if let Err(error) = file.close().await {
                        log::warn!(target: lk_agents_abi::LOG_TARGET, "background audio cleanup incomplete: {error}");
                    }
                }
                Input::Stream(_) | Input::Closed => {}
            }
        })
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        self.handle.finish();
        if !matches!(self.input, Input::Closed) {
            drop(self.teardown());
        }
    }
}

struct FileInput {
    path: String,
    looping: bool,
    clock: Arc<dyn Clock>,
    decoder: Option<AudioDecoder>,
}

impl FileInput {
    /// The next decoded block within `timeout`. A decode failure ends the sound.
    async fn next_within(
        &mut self,
        rate: u32,
        channels: u16,
        timeout: Duration,
        clock: &dyn Clock,
    ) -> Fetched {
        match lk_agents_abi::timeout(clock, timeout, self.next(rate, channels)).await {
            Ok(Some(Ok(pcm))) => Fetched::Pcm(pcm, rate, channels),
            Err(_) => Fetched::TimedOut,
            Ok(Some(Err(err))) => {
                log::error!(target: lk_agents_abi::LOG_TARGET, "background audio decode failed: {err}");
                Fetched::Ended
            }
            Ok(None) => Fetched::Ended,
        }
    }

    async fn next(
        &mut self,
        sample_rate: u32,
        num_channels: u16,
    ) -> Option<Result<Vec<i16>, DecoderError>> {
        if self.decoder.is_none() {
            let opened = AudioDecoder::open_file(
                &lk_agents_media::pool::media_pool(),
                self.path.clone(),
                self.looping,
                sample_rate,
                num_channels,
                decoder_limits(sample_rate, num_channels),
            );
            match opened {
                Ok(decoder) => self.decoder = Some(decoder),
                Err(error) => return Some(Err(error)),
            }
        }
        match self.decoder.as_mut().expect("the decoder was opened above").next().await {
            Ok(Some(frame)) => Some(Ok(lk_agents_media::pcm_samples(&frame.data))),
            Ok(None) => None,
            Err(error) => Some(Err(error)),
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
pub struct BackgroundAudio(pub Arc<BackgroundAudioPlayer>);

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
        let spawner =
            (env.spawner)().map_err(|e| lk_agents_abi::errors::ApiError::new(e.to_string()))?;
        Ok(Self(Arc::new(BackgroundAudioPlayer::new(
            sample_rate,
            num_channels,
            blocksize,
            stream_timeout_ms,
            env.clock.clone(),
            spawner,
        ))))
    }

    /// `start(room)`: publishes the track on the room's local participant, starts the `ambient`
    /// sound (drawn by probability from a list; a file loops), and returns the track's sid,
    /// which the host resolves to its own `rtc.LocalTrackPublication`. A second start returns the
    /// same sid. `source` is `track_publish_options.source`.
    async fn start(
        &self,
        room: Arc<dyn lk_agents_abi::voice::room_owner::SessionRoom>,
        draw: f64,
        ambient: Option<Sounds>,
        source: Option<TrackSource>,
    ) -> Result<lk_agents_abi::voice::rtc::TrackSid, RtcError> {
        let participant = room.rtc().ok_or(RtcError::NotConnected)?.local().clone();
        // Publish the track and own the pump. If the caller cancels, close waits and releases the
        // track.
        let this = self.0.clone();
        spawn_owned((&self.0.tasks, self.0.spawner.as_ref()), async move {
            let mut output = this.output.lock().await;
            if this.closed.load(Ordering::Acquire) {
                return Err(RtcError::Closed("background audio"));
            }
            if let Some(output) = output.as_ref() {
                return Ok(output.publication.sid().clone());
            }
            let lk_agents_abi::voice::rtc::PublishedAudioTrack { source, publication } = participant
                .publish_audio_track(
                    &telemetry::Cx::NONE,
                    TrackPublishOptions {
                        name: TRACK_NAME.into(),
                        source: source.unwrap_or_default(),
                        sample_rate: this.sample_rate,
                        num_channels: this.num_channels,
                        queue_size_ms: AUDIO_SOURCE_BUFFER_MS,
                    },
                )
                .await?;
            let stop = CancellationToken::new();
            let stop_rx = stop.clone();
            let engine = this.clone();
            let sink = source.clone();
            let pump = spawn_owned((&this.tasks, this.spawner.as_ref()), async move {
                if let Err(err) = engine.pump(sink, stop_rx).await {
                    log::error!(target: lk_agents_abi::LOG_TARGET, "background audio capture failed: {err}");
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
            let (sounds, choose) = ambient.map_or((Vec::new(), false), Sounds::candidates);
            if !sounds.is_empty() {
                // Open and started: `play` cannot refuse.
                let _ = this.play(sounds, choose, false, true, draw);
            }
            Ok(publication.sid().clone())
        })
        .await
        .map_err(|_| RtcError::host("background audio publication task stopped"))?
    }

    fn play(
        &self,
        sounds: Sounds,
        draw: f64,
        looping: bool,
        ambient: bool,
    ) -> Result<Self::State, PlaybackError> {
        let (sounds, choose) = sounds.candidates();
        self.0.play(sounds, choose, looping, ambient, draw).map(PlayState)
    }

    fn on_agent_state(
        &self,
        sounds: Option<Sounds>,
        draw: f64,
        thinking: bool,
    ) -> Result<(), PlaybackError> {
        let (sounds, choose) = sounds.map_or((Vec::new(), false), Sounds::candidates);
        let mut handle = self.0.thinking.lock();
        let last_done = handle.as_ref().map(|h| h.done());
        match thinking_sound(self.0.phase(), !sounds.is_empty(), thinking, last_done) {
            ThinkingSound::Keep => {}
            ThinkingSound::Start => {
                *handle = Some(self.0.play(sounds, choose, false, false, draw)?)
            }
            ThinkingSound::Stop => {
                if let Some(h) = handle.as_ref() {
                    h.stop();
                }
            }
        }
        Ok(())
    }

    fn closed(&self) -> bool {
        self.0.closed()
    }
    async fn aclose(&self) {
        self.0.close().await;
    }
}

#[cfg(test)]
#[path = "background_audio_tests.rs"]
mod tests;
