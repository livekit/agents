//! `TranscriptSynchronizer`'s live chain: the audio and text nodes it installs and the task that
//! forwards released words. Its decisions are voice's `TranscriptSync`.

use crate::monotone::Monotone;
use lk_agents_abi::voice::io::{
    AudioOutput, AudioOutputCapabilities, AudioOutputExt, PlaybackEvent, PlaybackFinished,
    PlaybackObserver, PlaybackSegment, PlaybackTracker, Result, TextOutput,
};
use lk_agents_abi::{AudioFrame, Clock, OwnedTask, Spawner, SpawnerExt, TimedString};
use lk_agents_net::lock::Mutex;
use lk_agents_voice::transcription::Heard;
use lk_agents_voice::transcription::lifecycle::{
    TranscriptOutput as Out, TranscriptSync, TranscriptSyncEffect, TranscriptSyncEvent,
};
use lk_agents_voice::transcription::pacer::{PaceEffect, PacerOptions};
use lk_agents_voice::transcription::synchronizer::{Hyphenator, TranscriptSyncInput};
use std::sync::{Arc, Weak};
use tokio::sync::mpsc;

/// What both chain nodes and the facade share; nothing holds the facade, so no reference cycle.
struct TranscriptChain {
    clock: Arc<dyn Clock>,
    /// The synchronizer's decisions.
    state: Mutex<TranscriptSync>,
    /// `None` once closed: dropping the sender is what ends the forwarder.
    tx: Mutex<Option<mpsc::UnboundedSender<Out>>>,
    /// Shared with the forwarder task, which must not keep the core alive (or it never ends).
    forwarder_done: Arc<Monotone<bool>>,
    /// The audio node's tracker; released words are raised on it as `transcript_shown`.
    tracker: Arc<PlaybackTracker>,
}

impl TranscriptChain {
    fn enabled(&self) -> bool {
        self.state.lock().segments.enabled()
    }

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

    fn step(&self, input: TranscriptSyncInput) -> Option<Heard> {
        self.apply(TranscriptSyncEvent::Input(input), None)
    }

    /// `input` from `segment`'s playback report.
    fn step_reported(&self, segment: PlaybackSegment, input: TranscriptSyncInput) -> Option<Heard> {
        self.apply(TranscriptSyncEvent::Input(input), Some(segment))
    }

    fn perform(
        &self,
        effect: TranscriptSyncEffect,
        reported: Option<PlaybackSegment>,
    ) -> Option<Heard> {
        match effect {
            TranscriptSyncEffect::Pace(PaceEffect::Forward { word, start_time, end_time }) => {
                let shown = TimedString {
                    start_time: Some(start_time),
                    end_time: Some(end_time),
                    ..TimedString::new(word)
                };
                self.tracker.transcript_shown(reported, &shown);
                self.send(Out::Word(TimedString { start_time: None, ..shown }));
            }
            TranscriptSyncEffect::Pace(PaceEffect::Flush) => self.send(Out::Flush),
            TranscriptSyncEffect::Pace(PaceEffect::Heard(heard)) => return Some(heard),
            TranscriptSyncEffect::Pace(PaceEffect::Warn(warning)) => {
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

    fn send(&self, out: Out) {
        self.apply(TranscriptSyncEvent::Output(out), None);
    }

    /// The physical attachment callback completed; only then does the gate change.
    fn on_attachment_changed(&self, audio: Option<bool>, text: Option<bool>) {
        self.apply(TranscriptSyncEvent::AttachmentChanged { audio, text }, None);
    }

    /// The forwarder: the only task and the only place the text sink is awaited.
    async fn forward(next: Option<Arc<dyn TextOutput>>, mut rx: mpsc::UnboundedReceiver<Out>) {
        while let Some(out) = rx.recv().await {
            let Some(next) = &next else { continue };
            match out {
                // A failure is the sink's. Dropping it keeps the rest of the transcript flowing.
                Out::Word(t) => {
                    let _ = next.capture_timed(&t).await;
                }
                Out::Flush => next.flush(),
                Out::Detached => next.on_detached(),
            }
        }
    }
}

/// `_SyncedAudioOutput`.
pub struct SyncedAudioOutput {
    core: Arc<TranscriptChain>,
    tracker: Arc<PlaybackTracker>,
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
            node.core.step(TranscriptSyncInput::Reset);
            node.tracker.on_playback_reset();
        }
    }
}

impl SyncedAudioOutput {
    fn new(core: Arc<TranscriptChain>, next: Arc<dyn AudioOutput>) -> Arc<Self> {
        let node = Arc::new(SyncedAudioOutput {
            tracker: Arc::clone(&core.tracker),
            core,
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
                    self.core.step_reported(e.segment, TranscriptSyncInput::Started);
                }
            }
            PlaybackEvent::Progressed(e) => {
                if self.tracker.playback_progressed(e.segment, e.started_at, e.offset, e.duration) {
                    self.core.step_reported(
                        e.segment,
                        TranscriptSyncInput::Progressed(e.offset + e.duration),
                    );
                }
            }
            PlaybackEvent::Finished { segment, event: e } => {
                if !self.tracker.accepts_report(*segment) {
                    return;
                }
                let position = e.playback_position;
                let finished =
                    TranscriptSyncInput::Finished { position, interrupted: e.interrupted };
                let heard = self.core.step_reported(*segment, finished);
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

    fn tracker(&self) -> &Arc<PlaybackTracker> {
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

    fn capture<'a>(&'a self, frame: &'a AudioFrame) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            // Counted before it passes: the sink below may report its playback from inside the
            // capture, and that playback is this segment's.
            let secs = lk_agents_abi::voice::rtc::frame_duration(frame).as_secs_f64();
            self.core.step(TranscriptSyncInput::Audio { secs });
            self.next.capture_frame(frame).await
        })
    }

    fn flush_segment(&self) {
        self.next.flush();
        self.core.step(TranscriptSyncInput::AudioEnd);
    }

    fn clear_buffer(&self) {
        self.next.clear_buffer();
        self.core.step(TranscriptSyncInput::Clear);
    }

    fn on_attached(&self) {
        self.next.on_attached();
        self.core.on_attachment_changed(Some(true), None);
    }

    fn on_detached(&self) {
        self.next.on_detached();
        self.core.on_attachment_changed(Some(false), None);
    }

    fn pause(&self) {
        self.next.pause();
        self.core.step(TranscriptSyncInput::Paused);
    }

    fn resume(&self) {
        self.next.resume();
    }
}

/// `_SyncedTextOutput`.
pub struct SyncedTextOutput {
    core: Arc<TranscriptChain>,
    next: Option<Arc<dyn TextOutput>>,
}

impl SyncedTextOutput {
    /// `aligned` records whether the caller used the timed API.
    async fn capture(&self, text: &TimedString, aligned: bool) -> Result<()> {
        for effect in self.core.decide(TranscriptSyncEvent::Capture { text: text.clone(), aligned })
        {
            match effect {
                TranscriptSyncEffect::Capture { text, aligned } => {
                    if let Some(next) = &self.next {
                        if aligned {
                            next.capture_timed(&text).await?;
                        } else {
                            next.capture_text(&text.text).await?;
                        }
                    }
                }
                effect => {
                    self.core.perform(effect, None);
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
        text: &'a TimedString,
    ) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move { self.capture(text, true).await })
    }

    fn capture_text<'a>(&'a self, text: &'a str) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move { self.capture(&TimedString::new(text), false).await })
    }

    fn flush(&self) {
        for effect in self.core.decide(TranscriptSyncEvent::FlushText) {
            match effect {
                TranscriptSyncEffect::FlushText => {
                    if let Some(next) = &self.next {
                        next.flush();
                    }
                }
                effect => {
                    self.core.perform(effect, None);
                }
            }
        }
    }

    fn on_attached(&self) {
        for effect in self.core.decide(TranscriptSyncEvent::AttachText) {
            match effect {
                TranscriptSyncEffect::AttachText => {
                    if let Some(next) = &self.next {
                        next.on_attached();
                    }
                    self.core.on_attachment_changed(None, Some(true));
                }
                effect => {
                    self.core.perform(effect, None);
                }
            }
        }
    }

    fn on_detached(&self) {
        self.core.apply(TranscriptSyncEvent::DetachText, None);
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
    core: Arc<TranscriptChain>,
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

        let opts = PacerOptions { speed, hyphenator };
        let (tx, rx) = mpsc::unbounded_channel();
        let core = Arc::new(TranscriptChain {
            state: Mutex::new(TranscriptSync::new(opts)),
            clock,
            tx: Mutex::new(Some(tx)),
            forwarder_done: Arc::new(Monotone::new(false)),
            tracker: Arc::new(PlaybackTracker::new()),
        });
        // The task holds only the done flag, so it never keeps the core alive.
        let forwarder_done = Arc::clone(&core.forwarder_done);
        let sink = text_next.clone();
        let forwarder = spawner.spawn(async move {
            TranscriptChain::forward(sink, rx).await;
            forwarder_done.set(true);
        });

        let audio = SyncedAudioOutput::new(core.clone(), audio_next);
        let text = Arc::new(SyncedTextOutput { core: core.clone(), next: text_next });
        TranscriptSynchronizer { core, audio, text, _forwarder: forwarder }
    }

    /// The node to install as `session.output.audio`.
    pub fn audio_output(&self) -> Arc<dyn AudioOutput> {
        self.audio.clone()
    }

    /// The node to install as `session.output.transcription`.
    pub fn text_output(&self) -> Arc<dyn TextOutput> {
        self.text.clone()
    }

    pub fn enabled(&self) -> bool {
        self.core.enabled()
    }

    pub fn set_enabled(&self, enabled: bool) {
        self.core.step(TranscriptSyncInput::Enabled(enabled));
    }

    /// End the live segment now and start a fresh one.
    pub fn rotate_segment(&self) {
        self.core.apply(TranscriptSyncEvent::Rotate, None);
    }

    /// Close: end the live segment and give the forwarder [`lk_agents_abi::HOST_TEARDOWN_TIMEOUT`] to drain.
    /// `RoomIO.aclose` must close this before the transcription outputs it feeds.
    pub async fn aclose(&self) {
        for effect in self.core.decide(TranscriptSyncEvent::Close) {
            match effect {
                TranscriptSyncEffect::CloseSender => {
                    self.core.tx.lock().take();
                }
                TranscriptSyncEffect::BeginWait => {
                    for wait in
                        self.core.decide(TranscriptSyncEvent::WaitStarted(self.core.clock.now()))
                    {
                        let TranscriptSyncEffect::WaitUntil(at) = wait else {
                            unreachable!("close only waits on its forwarded words");
                        };
                        let left = at.saturating_duration_since(self.core.clock.now());
                        let _ = lk_agents_abi::timeout(
                            &*self.core.clock,
                            left,
                            self.core.forwarder_done.wait_for(|set| *set),
                        )
                        .await;
                        self.core.apply(TranscriptSyncEvent::WaitFinished, None);
                    }
                }
                effect => {
                    self.core.perform(effect, None);
                }
            }
        }
    }
}

/// A stable node beneath an audio wrapper chain, with its own segment tracker.
pub struct AudioSinkProxy {
    tracker: Arc<PlaybackTracker>,
    next: Arc<dyn AudioOutput>,
}

impl AudioSinkProxy {
    pub fn new(next: Arc<dyn AudioOutput>) -> Arc<Self> {
        let tracker = Arc::new(PlaybackTracker::new());
        next.tracker().observe(tracker.clone());
        Arc::new(AudioSinkProxy { tracker, next })
    }
}

impl AudioOutput for AudioSinkProxy {
    fn label(&self) -> &str {
        "AudioSinkProxy"
    }

    fn tracker(&self) -> &Arc<PlaybackTracker> {
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

    fn capture<'a>(&'a self, frame: &'a AudioFrame) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
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
#[path = "transcript_sync_tests.rs"]
mod tests;
