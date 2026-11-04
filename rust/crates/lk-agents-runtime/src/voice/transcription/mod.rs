//! `TranscriptSynchronizer`'s live chain: the audio and text nodes it installs and the task that
//! forwards released words. Its decisions are voice's `TranscriptSync`.

use crate::monotone::Monotone;
use crate::telemetry;
use lk_agents_abi::voice::io::{
    AudioOutput, AudioOutputCapabilities, AudioOutputExt, PlaybackEvent, PlaybackFinished,
    PlaybackObserver, PlaybackSegment, PlaybackTracker, Result, TextOutput,
};
use lk_agents_abi::{AudioFrame, Clock, OwnedTask, Spawner, SpawnerExt, TranscriptChunk};
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
    /// The synchronizer's decisions.
    state: Mutex<TranscriptSync>,
    /// `None` once closed: dropping the sender is what ends the forwarder.
    tx: Mutex<Option<mpsc::UnboundedSender<TranscriptOutput>>>,
    /// Shared with the forwarder task, which must not keep the synchronizer alive (or it never
    /// ends).
    forwarder_done: Arc<Monotone<bool>>,
    /// The audio node's tracker; released words are raised on it as `transcript_shown`.
    tracker: Arc<dyn PlaybackTracker>,
}

impl Synchronizer {
    fn decide(&self, event: TranscriptSyncEvent) -> Vec<TranscriptSyncEffect> {
        self.state.lock().handle(event)
    }

    /// Step the pure lifecycle and perform its ordered synchronous effects. The words released
    /// are `reported`'s, the playback whose report this is, if any.
    fn apply(
        &self,
        event: TranscriptSyncEvent,
        reported: Option<PlaybackSegment>,
    ) -> Option<Heard> {
        let mut heard = None;
        for effect in self.decide(event) {
            if let Some(report) = self.perform(effect, reported) {
                heard = Some(report);
            }
        }
        heard
    }

    fn step(&self, event: TranscriptSyncEvent) -> Option<Heard> {
        self.apply(event, None)
    }

    /// `input` from `segment`'s playback report.
    fn step_reported(&self, segment: PlaybackSegment, event: TranscriptSyncEvent) -> Option<Heard> {
        self.apply(event, Some(segment))
    }

    fn perform(
        &self,
        effect: TranscriptSyncEffect,
        reported: Option<PlaybackSegment>,
    ) -> Option<Heard> {
        match effect {
            TranscriptSyncEffect::PublishWord { word, start_time, end_time } => {
                let shown = TranscriptChunk {
                    start_time: Some(start_time),
                    end_time: Some(end_time),
                    ..TranscriptChunk::new(word)
                };
                self.tracker.transcript_shown(reported, &shown);
                self.send(TranscriptOutput::Word(TranscriptChunk { start_time: None, ..shown }));
            }
            TranscriptSyncEffect::FlushTranscript => self.send(TranscriptOutput::Flush),
            TranscriptSyncEffect::ReportHeard(heard) => return Some(heard),
            TranscriptSyncEffect::Warn(warning) => {
                log::warn!(target: "livekit.agents", "{}", warning.message());
            }
            TranscriptSyncEffect::Queue(output) => {
                if let Some(tx) = self.tx.lock().as_ref() {
                    let _ = tx.send(output);
                }
            }
            TranscriptSyncEffect::RetireText { admitted } => {
                self.apply(TranscriptSyncEvent::TextRetired { admitted }, reported);
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

    /// The forwarder: the only task and the only place the text sink is awaited.
    async fn forward(
        next: Option<Arc<dyn TextOutput>>,
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
            node.sync.step(TranscriptSyncEvent::PlaybackReset);
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
                    self.sync.step_reported(e.segment, TranscriptSyncEvent::PlaybackStarted);
                }
            }
            PlaybackEvent::Progressed(e) => {
                if self.tracker.playback_progressed(e.segment, e.started_at, e.offset, e.duration) {
                    self.sync.step_reported(
                        e.segment,
                        TranscriptSyncEvent::PlaybackProgressed(e.offset + e.duration),
                    );
                }
            }
            PlaybackEvent::Finished { segment, event: e } => {
                if !self.tracker.accepts_report(*segment) {
                    return;
                }
                let position = e.playback_position;
                let finished =
                    TranscriptSyncEvent::PlaybackFinished { position, interrupted: e.interrupted };
                let heard = self.sync.step_reported(*segment, finished);
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
            self.sync.step(TranscriptSyncEvent::AudioCaptureStarted { secs });
            self.next.capture_frame(frame).await
        })
    }

    fn flush_segment(&self) {
        self.next.flush();
        self.sync.step(TranscriptSyncEvent::AudioEnded);
    }

    fn clear_buffer(&self) {
        self.next.clear_buffer();
        self.sync.step(TranscriptSyncEvent::AudioCleared);
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
        self.sync.step(TranscriptSyncEvent::PlaybackPaused);
    }

    fn resume(&self) {
        self.next.resume();
    }
}

/// `_SyncedTextOutput`.
pub struct SyncedTextOutput {
    sync: Arc<Synchronizer>,
    next: Option<Arc<dyn TextOutput>>,
}

impl SyncedTextOutput {
    /// `aligned` records whether the caller used the timed API.
    async fn capture(&self, text: &TranscriptChunk, aligned: bool) -> Result<()> {
        for effect in self.sync.decide(TranscriptSyncEvent::Capture { text: text.clone(), aligned })
        {
            match effect {
                TranscriptSyncEffect::Capture { text, aligned } => {
                    if let Some(next) = &self.next {
                        if aligned {
                            next.capture_timed(&telemetry::Cx::NONE, &text).await?;
                        } else {
                            next.capture_text(&telemetry::Cx::NONE, &text.text).await?;
                        }
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

impl TextOutput for SyncedTextOutput {
    fn label(&self) -> &str {
        "TranscriptSynchronizer"
    }

    fn next_in_chain(&self) -> Option<Arc<dyn TextOutput>> {
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
        Box::pin(async move { self.capture(&TranscriptChunk::new(text), false).await })
    }

    fn flush(&self) {
        for effect in self.sync.decide(TranscriptSyncEvent::FlushText) {
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
        for effect in self.sync.decide(TranscriptSyncEvent::AttachText) {
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
    pub text_next: Option<Arc<dyn TextOutput>>,
    pub speed: f64,
    pub hyphenator: Arc<dyn Hyphenator>,
    pub clock: Arc<dyn Clock>,
    pub spawner: Arc<dyn Spawner>,
}

pub struct TranscriptSynchronizer {
    sync: Arc<Synchronizer>,
    audio: Arc<SyncedAudioOutput>,
    text: Arc<SyncedTextOutput>,
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

        // `io.AudioOutput.__init__`'s auto-wrap, so an avatar hand-off can replace the leaf later.
        let audio_next = if audio_next.next_in_chain().is_none() {
            AudioSinkProxy::new(audio_next) as Arc<dyn AudioOutput>
        } else {
            audio_next
        };

        let opts = TranscriptPacerOptions { speed, hyphenator };
        let (tx, rx) = mpsc::unbounded_channel();
        let sync = Arc::new(Synchronizer {
            state: Mutex::new(TranscriptSync::new(opts)),
            clock,
            tx: Mutex::new(Some(tx)),
            forwarder_done: Arc::new(Monotone::new(false)),
            tracker: Arc::new(crate::voice::playback::Tracker::new()),
        });
        // The task holds only the done flag, so it never keeps the synchronizer alive.
        let forwarder_done = Arc::clone(&sync.forwarder_done);
        let sink = text_next.clone();
        let forwarder = spawner.spawn(async move {
            Synchronizer::forward(sink, rx).await;
            forwarder_done.set(true);
        });

        let audio = SyncedAudioOutput::new(sync.clone(), audio_next);
        let text = Arc::new(SyncedTextOutput { sync: sync.clone(), next: text_next });
        TranscriptSynchronizer { sync, audio, text, _forwarder: forwarder }
    }

    /// The node to install as `session.output.audio`.
    pub fn audio_output(&self) -> Arc<dyn AudioOutput> {
        self.audio.clone()
    }

    /// The node to install as `session.output.transcription`.
    pub fn text_output(&self) -> Arc<dyn TextOutput> {
        self.text.clone()
    }

    /// Close: end the live segment and give the forwarder [`lk_agents_abi::HOST_TEARDOWN_TIMEOUT`] to drain.
    /// `RoomIO.aclose` must close this before the transcription outputs it feeds.
    pub async fn aclose(&self) {
        for effect in self.sync.decide(TranscriptSyncEvent::Close) {
            match effect {
                TranscriptSyncEffect::CloseSender => {
                    self.sync.tx.lock().take();
                }
                TranscriptSyncEffect::BeginWait => {
                    for wait in
                        self.sync.decide(TranscriptSyncEvent::WaitStarted(self.sync.clock.now()))
                    {
                        let TranscriptSyncEffect::WaitUntil(at) = wait else {
                            unreachable!("close only waits on its forwarded words");
                        };
                        let left = at.saturating_duration_since(self.sync.clock.now());
                        let _ = lk_agents_abi::timeout(
                            &*self.sync.clock,
                            left,
                            self.sync.forwarder_done.wait_for(|set| *set),
                        )
                        .await;
                        self.sync.apply(TranscriptSyncEvent::WaitFinished, None);
                    }
                }
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
#[path = "synchronizer_tests.rs"]
mod tests;
