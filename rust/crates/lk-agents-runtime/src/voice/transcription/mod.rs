use crate::telemetry;
use lk_agents_abi::voice::io::{
    AudioOutput, AudioOutputCapabilities, AudioOutputExt, PlaybackEvent, PlaybackFinished,
    PlaybackObserver, PlaybackSegment, PlaybackTracker, Result, TranscriptionOutput,
};
use lk_agents_abi::{AudioFrame, Clock, OwnedTask, Spawner, SpawnerExt, TranscriptChunk};
use lk_agents_net::signal;
use lk_agents_voice::transcription::Heard;
use lk_agents_voice::transcription::effects::TranscriptSyncEffect;
use lk_agents_voice::transcription::machine::lifecycle::{
    TranscriptOutput, TranscriptPacerOptions, TranscriptSync, TranscriptSyncEvent,
};
use lk_agents_voice::transcription::machine::synchronizer::Hyphenator;
use parking_lot::Mutex;
use std::sync::{Arc, Weak};
use tokio::sync::mpsc;

/// The transcript synchronizer both synced outputs and the facade report to: voice's
/// `TranscriptSync` deciding, this executing. Nothing holds the facade, so no reference cycle.
struct Synchronizer {
    clock: Arc<dyn Clock>,
    spawner: Arc<dyn Spawner>,
    me: Weak<Synchronizer>,
    /// The tick the decisions asked for next; replacing it cancels the one it replaces.
    pacing: Mutex<Option<OwnedTask<()>>>,
    /// The playback last reported started or moving and not yet finished: the words a pause or a
    /// tick shows are its.
    playing: Mutex<Option<PlaybackSegment>>,
    /// The synchronizer's decisions.
    state: Mutex<TranscriptSync>,
    /// `None` once closed: dropping the sender is what ends the forwarder.
    tx: Mutex<Option<mpsc::UnboundedSender<TranscriptOutput>>>,
    /// Shared with the forwarder task, which must not keep the synchronizer alive (or it never
    /// ends).
    /// The forwarder task's: it ends with the task, however it ends.
    forwarder_done: signal::Cell<bool>,
    /// The audio node's tracker; released words are raised on it as `transcript_shown`.
    tracker: Arc<dyn PlaybackTracker>,
}

impl Synchronizer {
    /// Step the pure lifecycle and perform its ordered synchronous effects. The words released
    /// are `reported`'s, the playback whose report this is, if any.
    fn apply(
        &self,
        event: TranscriptSyncEvent,
        reported: Option<PlaybackSegment>,
    ) -> Option<Heard> {
        let mut heard = None;
        let effects = self.state.lock().handle(event);
        for effect in effects {
            if let Some(report) = self.perform(effect, reported) {
                heard = Some(report);
            }
        }
        heard
    }

    fn perform(
        &self,
        effect: TranscriptSyncEffect,
        reported: Option<PlaybackSegment>,
    ) -> Option<Heard> {
        match effect {
            TranscriptSyncEffect::PublishWord { chunk, start_time, end_time } => {
                let shown = TranscriptChunk {
                    start_time: Some(start_time),
                    end_time: Some(end_time),
                    ..chunk
                };
                self.tracker.transcript_shown(reported, &shown);
                self.send(TranscriptOutput::Word(TranscriptChunk { start_time: None, ..shown }));
            }
            TranscriptSyncEffect::FlushTranscript => self.send(TranscriptOutput::Flush),
            TranscriptSyncEffect::ReportHeard(heard) => return Some(heard),
            TranscriptSyncEffect::ReportCleared { segment, heard } => {
                self.tracker.transcript_cleared(segment, &heard.text)
            }
            TranscriptSyncEffect::Warn(warning) => {
                log::warn!(target: lk_agents_abi::LOG_TARGET, "{}", warning.message());
            }
            TranscriptSyncEffect::Queue(output) => {
                if let Some(tx) = self.tx.lock().as_ref() {
                    let _ = tx.send(output);
                }
            }
            TranscriptSyncEffect::RetireText { admitted } => {
                self.apply(TranscriptSyncEvent::TextRetired { admitted }, reported);
            }
            TranscriptSyncEffect::TickAt(at) => {
                let (me, clock) = (self.me.clone(), Arc::clone(&self.clock));
                let tick = self.spawner.spawn(async move {
                    clock.wait_until(at).await;
                    if let Some(sync) = me.upgrade() {
                        let playing = *sync.playing.lock();
                        sync.apply(TranscriptSyncEvent::Tick(sync.clock.now()), playing);
                    }
                });
                let replaced = self.pacing.lock().replace(tick);
                drop(replaced);
            }
            _ => unreachable!("a chain operation must be performed by its owning output"),
        }
        None
    }

    fn send(&self, out: TranscriptOutput) {
        self.apply(TranscriptSyncEvent::Output(out), None);
    }

    /// The physical attachment callback completed; only then does the gate change.
    fn on_attachment_changed(&self, audio: Option<bool>, text: Option<bool>) {
        self.apply(TranscriptSyncEvent::AttachmentChanged { audio, text }, None);
    }

    /// The close's wait: give the forwarder until the decided deadline to send its words.
    async fn wait_for_forwarded_words(&self) {
        let effects = self.state.lock().handle(TranscriptSyncEvent::WaitStarted(self.clock.now()));
        for wait in effects {
            let TranscriptSyncEffect::WaitUntil(at) = wait else {
                unreachable!("close only waits on its forwarded words");
            };
            let left = at.saturating_duration_since(self.clock.now());
            let forwarded = self.forwarder_done.wait_for(|set| *set);
            if lk_agents_abi::timeout(&*self.clock, left, forwarded).await.is_err() {
                log::warn!(target: lk_agents_abi::LOG_TARGET, timeout_ms = left.as_millis() as u64; "transcript forwarder drain timed out");
            }
            self.apply(TranscriptSyncEvent::WaitFinished, None);
        }
    }

    /// The forwarder: the only task and the only place the text sink is awaited.
    async fn forward(
        next: Option<Arc<dyn TranscriptionOutput>>,
        mut rx: mpsc::UnboundedReceiver<TranscriptOutput>,
    ) {
        while let Some(out) = rx.recv().await {
            let Some(next) = &next else { continue };
            match out {
                // A failure is the sink's. Dropping it keeps the rest of the transcript flowing.
                TranscriptOutput::Word(t) => {
                    let _ = next.capture_timed(&telemetry::Cx::NONE, &t).await;
                }
                TranscriptOutput::Flush => next.flush(),
                TranscriptOutput::Detached => next.on_detached(),
            }
        }
    }
}

/// `_SyncedAudioOutput`.
pub struct SyncedAudioOutput {
    sync: Arc<Synchronizer>,
    tracker: Arc<dyn PlaybackTracker>,
    next: Arc<dyn AudioOutput>,
    observer: Mutex<Option<Arc<dyn PlaybackObserver>>>,
}

/// [`Weak`] so the tracker doesn't keep the chain alive.
struct AudioEvents(Weak<SyncedAudioOutput>);

impl PlaybackObserver for AudioEvents {
    fn on_playback_event(&self, ev: &PlaybackEvent) {
        if let Some(node) = self.0.upgrade() {
            node.handle_playback(ev);
        }
    }

    fn on_playback_reset(&self) {
        if let Some(node) = self.0.upgrade() {
            node.sync.apply(TranscriptSyncEvent::PlaybackReset, None);
            node.tracker.on_playback_reset();
        }
    }
}

impl SyncedAudioOutput {
    fn new(sync: Arc<Synchronizer>, next: Arc<dyn AudioOutput>) -> Arc<Self> {
        let node = Arc::new(SyncedAudioOutput {
            tracker: Arc::clone(&sync.tracker),
            sync,
            next: next.clone(),
            observer: Mutex::new(None),
        });
        let observer: Arc<dyn PlaybackObserver> = Arc::new(AudioEvents(Arc::downgrade(&node)));
        next.tracker().observe(observer.clone());
        *node.observer.lock() = Some(observer);
        node
    }

    /// Synchronous: a wrapper's counter must move before the leaf's emit returns, or
    /// `wait_for_playout` races the event meant to release it.
    fn handle_playback(&self, ev: &PlaybackEvent) {
        match ev {
            PlaybackEvent::Started(e) => {
                if self.tracker.playback_started(e.segment, e.created_at) {
                    *self.sync.playing.lock() = Some(e.segment);
                    let at = self.sync.clock.now();
                    self.sync.apply(TranscriptSyncEvent::PlaybackStarted { at }, Some(e.segment));
                }
            }
            PlaybackEvent::Progressed(e) => {
                if self.tracker.playback_progressed(e.segment, e.started_at, e.offset, e.duration) {
                    *self.sync.playing.lock() = Some(e.segment);
                    let (position, at) = (e.offset + e.duration, self.sync.clock.now());
                    self.sync.apply(
                        TranscriptSyncEvent::PlaybackProgressed { position, at },
                        Some(e.segment),
                    );
                }
            }
            PlaybackEvent::Finished { segment, event: e } => {
                if !self.tracker.accepts_report(*segment) {
                    return;
                }
                self.sync.playing.lock().take_if(|playing| playing == segment);
                let position = e.playback_position;
                let finished =
                    TranscriptSyncEvent::PlaybackFinished { position, interrupted: e.interrupted };
                let heard = self.sync.apply(finished, Some(*segment));
                let synchronized_transcript =
                    heard.map(|h| h.text).or(e.synchronized_transcript.clone());
                self.tracker.playback_finished(
                    *segment,
                    PlaybackFinished { synchronized_transcript, ..e.clone() },
                );
            }
        }
    }
}

impl AudioOutput for SyncedAudioOutput {
    fn label(&self) -> &str {
        "TranscriptSynchronizer"
    }

    fn tracker(&self) -> &Arc<dyn PlaybackTracker> {
        &self.tracker
    }

    fn capabilities(&self) -> AudioOutputCapabilities {
        AudioOutputCapabilities { pause: true }
    }

    fn sample_rate(&self) -> Option<u32> {
        self.next.sample_rate()
    }

    fn next_in_chain(&self) -> Option<Arc<dyn AudioOutput>> {
        Some(self.next.clone())
    }

    fn capture<'a>(
        &'a self,
        _cx: &telemetry::Cx,
        frame: &'a AudioFrame,
    ) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            // Counted before it passes: the sink below may report its playback from inside the
            // capture, and that playback is this segment's.
            let secs = lk_agents_abi::voice::rtc::frame_duration(frame).as_secs_f64();
            let segment = self.tracker.current_segment();
            self.sync.apply(TranscriptSyncEvent::AudioCaptureStarted { secs, segment }, None);
            self.next.capture_frame(frame).await
        })
    }

    fn flush_segment(&self) {
        self.next.flush();
        self.sync.apply(TranscriptSyncEvent::AudioEnded, None);
    }

    fn clear_buffer(&self) {
        self.next.clear_buffer();
        // Words the clear still shows are the playback's that was moving.
        let (playing, at) = (*self.sync.playing.lock(), self.sync.clock.now());
        self.sync.apply(TranscriptSyncEvent::AudioCleared { at }, playing);
    }

    /// The refused frame was counted as its segment's audio, and none of it will play: the
    /// segment ends cut, with nothing heard, so the next speech's audio opens its own.
    fn withdraw_capture(&self) -> Result<()> {
        let below = self.next.withdraw_capture();
        if let Some(segment) = self.tracker.withdraw_capture() {
            self.sync.apply(TranscriptSyncEvent::AudioEnded, None);
            let finished = TranscriptSyncEvent::PlaybackFinished {
                position: std::time::Duration::ZERO,
                interrupted: true,
            };
            self.sync.apply(finished, Some(segment));
        }
        below
    }

    fn on_attached(&self) {
        self.next.on_attached();
        self.sync.on_attachment_changed(Some(true), None);
    }

    fn on_detached(&self) {
        self.next.on_detached();
        self.sync.on_attachment_changed(Some(false), None);
    }

    fn pause(&self) {
        self.next.pause();
        let playing = *self.sync.playing.lock();
        self.sync.apply(TranscriptSyncEvent::PlaybackPaused { at: self.sync.clock.now() }, playing);
    }

    fn resume(&self) {
        self.next.resume();
        let playing = *self.sync.playing.lock();
        self.sync
            .apply(TranscriptSyncEvent::PlaybackResumed { at: self.sync.clock.now() }, playing);
    }
}

/// `_SyncedTranscriptionOutput`.
pub struct SyncedTranscriptionOutput {
    sync: Arc<Synchronizer>,
    next: Option<Arc<dyn TranscriptionOutput>>,
    host: parking_lot::Mutex<lk_agents_voice::transcription::HostText>,
}

impl SyncedTranscriptionOutput {
    /// `aligned` records whether the caller used the timed API.
    async fn capture(&self, text: &TranscriptChunk, aligned: bool) -> Result<()> {
        let effects = self
            .sync
            .state
            .lock()
            .handle(TranscriptSyncEvent::Capture { text: text.clone(), aligned });
        for effect in effects {
            match effect {
                TranscriptSyncEffect::Capture { text, aligned: true } => {
                    if let Some(next) = &self.next {
                        next.capture_timed(&telemetry::Cx::NONE, &text).await?;
                    }
                }
                // a settled piece passes on whole: whether it is spoken, and its delivery
                TranscriptSyncEffect::Capture { text, aligned: false } => {
                    if let Some(next) = &self.next {
                        next.capture_timed(&telemetry::Cx::NONE, &text).await?;
                    }
                }
                effect => {
                    self.sync.perform(effect, None);
                }
            }
        }
        Ok(())
    }
}

impl TranscriptionOutput for SyncedTranscriptionOutput {
    fn label(&self) -> &str {
        "TranscriptSynchronizer"
    }

    fn next_in_chain(&self) -> Option<Arc<dyn TranscriptionOutput>> {
        self.next.clone()
    }

    fn capture_timed<'a>(
        &'a self,
        _cx: &telemetry::Cx,
        text: &'a TranscriptChunk,
    ) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move { self.capture(text, true).await })
    }

    fn capture_text<'a>(
        &'a self,
        _cx: &telemetry::Cx,
        text: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        // a host's plain string: read for markup, which no voice reading took out
        let chunk = self.host.lock().push(text);
        Box::pin(async move { self.capture(&chunk, false).await })
    }

    fn flush(&self) {
        self.host.lock().reset();
        let effects = self.sync.state.lock().handle(TranscriptSyncEvent::FlushText);
        for effect in effects {
            match effect {
                TranscriptSyncEffect::FlushText => {
                    if let Some(next) = &self.next {
                        next.flush();
                    }
                }
                effect => {
                    self.sync.perform(effect, None);
                }
            }
        }
    }

    fn on_attached(&self) {
        let effects = self.sync.state.lock().handle(TranscriptSyncEvent::AttachText);
        for effect in effects {
            match effect {
                TranscriptSyncEffect::AttachText => {
                    if let Some(next) = &self.next {
                        next.on_attached();
                    }
                    self.sync.on_attachment_changed(None, Some(true));
                }
                effect => {
                    self.sync.perform(effect, None);
                }
            }
        }
    }

    fn on_detached(&self) {
        self.sync.apply(TranscriptSyncEvent::DetachText, None);
    }
}

/// Everything `TranscriptSynchronizer.__init__` takes, plus the two ports Rust needs explicitly.
pub struct TranscriptSyncConfig {
    /// The audio sink below; a bare leaf is wrapped in an [`AudioSinkProxy`].
    pub audio_next: Arc<dyn AudioOutput>,
    /// The transcript sink below this node. `None` for the console path, which prints locally.
    pub text_next: Option<Arc<dyn TranscriptionOutput>>,
    pub speed: f64,
    pub hyphenator: Arc<dyn Hyphenator>,
    pub clock: Arc<dyn Clock>,
    pub spawner: Arc<dyn Spawner>,
}

pub struct TranscriptSynchronizer {
    sync: Arc<Synchronizer>,
    audio: Arc<SyncedAudioOutput>,
    text: Arc<SyncedTranscriptionOutput>,
    /// Forwards released words to the text sink; ends once the close drops its sender.
    _forwarder: OwnedTask<()>,
}

impl TranscriptSynchronizer {
    pub fn new(config: TranscriptSyncConfig) -> Arc<Self> {
        Arc::new(Self::build(config))
    }

    fn build(config: TranscriptSyncConfig) -> Self {
        let TranscriptSyncConfig { audio_next, text_next, speed, hyphenator, clock, spawner } =
            config;

        // Wrapped so an avatar hand-off can replace the leaf later.
        let audio_next = if audio_next.next_in_chain().is_none() {
            AudioSinkProxy::new(audio_next) as Arc<dyn AudioOutput>
        } else {
            audio_next
        };

        let opts = TranscriptPacerOptions { speed, hyphenator };
        let (tx, rx) = mpsc::unbounded_channel();
        let (done, forwarder_done) = signal::signal(false);
        let forwarder_spawner = Arc::clone(&spawner);
        let sync = Arc::new_cyclic(|me| Synchronizer {
            state: Mutex::new(TranscriptSync::new(opts)),
            clock,
            spawner,
            me: me.clone(),
            pacing: Mutex::new(None),
            playing: Mutex::new(None),
            tx: Mutex::new(Some(tx)),
            forwarder_done,
            tracker: Arc::new(crate::voice::playback::Tracker::new()),
        });
        // The task holds only the done flag, so it never keeps the synchronizer alive.
        let sink = text_next.clone();
        let forwarder = forwarder_spawner.spawn(async move {
            Synchronizer::forward(sink, rx).await;
            done.set(true);
        });

        let audio = SyncedAudioOutput::new(sync.clone(), audio_next);
        let text = Arc::new(SyncedTranscriptionOutput {
            sync: sync.clone(),
            next: text_next,
            host: Default::default(),
        });
        TranscriptSynchronizer { sync, audio, text, _forwarder: forwarder }
    }

    /// The node to install as `session.output.audio`.
    pub fn audio_output(&self) -> Arc<dyn AudioOutput> {
        self.audio.clone()
    }

    /// The node to install as `session.output.transcription`.
    pub fn transcription_output(&self) -> Arc<dyn TranscriptionOutput> {
        self.text.clone()
    }

    /// Close: end the live segment and give the forwarder [`lk_agents_abi::HOST_TEARDOWN_TIMEOUT`] to drain.
    /// `RoomIO.aclose` must close this before the transcription outputs it feeds.
    pub async fn aclose(&self) {
        let effects = self.sync.state.lock().handle(TranscriptSyncEvent::Close);
        for effect in effects {
            match effect {
                TranscriptSyncEffect::CloseSender => {
                    self.sync.tx.lock().take();
                    // Nothing more is forwarded, so nothing more is paced.
                    let pacing = self.sync.pacing.lock().take();
                    drop(pacing);
                }
                TranscriptSyncEffect::BeginWait => self.sync.wait_for_forwarded_words().await,
                effect => {
                    self.sync.perform(effect, None);
                }
            }
        }
    }
}

/// A stable node beneath an audio wrapper chain, with its own segment tracker.
pub struct AudioSinkProxy {
    tracker: Arc<dyn PlaybackTracker>,
    next: Arc<dyn AudioOutput>,
}

impl AudioSinkProxy {
    pub fn new(next: Arc<dyn AudioOutput>) -> Arc<Self> {
        let tracker = Arc::new(crate::voice::playback::Tracker::new());
        next.tracker().observe(tracker.clone());
        Arc::new(AudioSinkProxy { tracker, next })
    }
}

impl AudioOutput for AudioSinkProxy {
    fn label(&self) -> &str {
        "AudioSinkProxy"
    }

    fn tracker(&self) -> &Arc<dyn PlaybackTracker> {
        &self.tracker
    }

    /// The proxy itself imposes nothing; the answer is whatever is below it.
    fn capabilities(&self) -> AudioOutputCapabilities {
        AudioOutputCapabilities { pause: true }
    }

    fn sample_rate(&self) -> Option<u32> {
        self.next.sample_rate()
    }

    fn next_in_chain(&self) -> Option<Arc<dyn AudioOutput>> {
        Some(self.next.clone())
    }

    fn capture<'a>(
        &'a self,
        _cx: &telemetry::Cx,
        frame: &'a AudioFrame,
    ) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move { self.next.capture_frame(frame).await })
    }

    fn flush_segment(&self) {
        self.next.flush();
    }

    fn clear_buffer(&self) {
        self.next.clear_buffer();
    }

    fn on_attached(&self) {
        self.next.on_attached();
    }

    fn on_detached(&self) {
        self.next.on_detached();
    }
}

#[cfg(test)]
#[path = "transcription_tests.rs"]
mod tests;
