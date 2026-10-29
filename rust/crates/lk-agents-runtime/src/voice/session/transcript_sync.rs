//! `TranscriptSynchronizer`'s live chain: the audio and text nodes it installs and the task that
//! forwards released words. Its decisions are voice's `SyncLifecycle`.

use lk_agents_abi::voice::io::{
    AudioOutput, AudioOutputExt, PlaybackEvent, PlaybackFinished, PlaybackObserver,
    PlaybackTracker, Result, TextOutput,
};
use lk_agents_abi::voice::rtc::AudioOutputCapabilities;
use lk_agents_abi::{AudioFrame, Clock, Monotone, OwnedTask, Spawner, SpawnerExt, TimedString};
use lk_agents_voice::io::AudioSinkProxy;
use lk_agents_voice::transcription::Heard;
use lk_agents_voice::transcription::lifecycle::{
    SyncEffect, SyncEvent, SyncLifecycle, TranscriptOutput as Out,
};
use lk_agents_voice::transcription::pacer::{PaceEffect, PacerOptions};
use lk_agents_voice::transcription::synchronizer::{Hyphenator, SyncInput};
use std::sync::{Arc, Mutex, Weak};
use tokio::sync::mpsc;

/// The actor's locked state. Its decisions belong to the synchronous lifecycle.
struct SyncState {
    lifecycle: SyncLifecycle,
}

/// What both chain nodes and the facade share; nothing holds the facade, so no reference cycle.
struct SyncCore {
    clock: Arc<dyn Clock>,
    state: Mutex<SyncState>,
    /// `None` once closed: dropping the sender is what ends the forwarder.
    tx: Mutex<Option<mpsc::UnboundedSender<Out>>>,
    /// Shared with the forwarder task, which must not keep the core alive (or it never ends).
    forwarder_done: Arc<Monotone<bool>>,
    /// The audio node's tracker; released words are raised on it as `transcript_shown`.
    tracker: Arc<PlaybackTracker>,
}

impl SyncCore {
    fn enabled(&self) -> bool {
        self.state.lock().unwrap().lifecycle.segments.enabled()
    }

    fn decide(&self, event: SyncEvent) -> Vec<SyncEffect> {
        self.state.lock().unwrap().lifecycle.handle(event)
    }

    /// Step the pure lifecycle and perform its ordered synchronous effects.
    fn apply(&self, event: SyncEvent) -> Option<Heard> {
        let mut heard = None;
        for effect in self.decide(event) {
            if let Some(report) = self.perform(effect) {
                heard = Some(report);
            }
        }
        heard
    }

    fn step(&self, input: SyncInput) -> Option<Heard> {
        self.apply(SyncEvent::Input(input))
    }

    fn perform(&self, effect: SyncEffect) -> Option<Heard> {
        match effect {
            SyncEffect::Pace(PaceEffect::Forward { word, start_time, end_time }) => {
                let shown = TimedString {
                    start_time: Some(start_time),
                    end_time: Some(end_time),
                    ..TimedString::new(word)
                };
                self.tracker.transcript_shown(&shown);
                self.send(Out::Word(TimedString { start_time: None, ..shown }));
            }
            SyncEffect::Pace(PaceEffect::Flush) => self.send(Out::Flush),
            SyncEffect::Pace(PaceEffect::Heard(heard)) => return Some(heard),
            SyncEffect::Pace(PaceEffect::Warn(warning)) => {
                log::warn!(target: "livekit.agents", "{}", warning.message());
            }
            SyncEffect::Queue(output) => {
                if let Some(tx) = self.tx.lock().unwrap().as_ref() {
                    let _ = tx.send(output);
                }
            }
            SyncEffect::RetireText { admitted } => {
                self.apply(SyncEvent::TextRetired { admitted });
            }
            _ => unreachable!("a chain operation must be performed by its owning output"),
        }
        None
    }

    fn send(&self, out: Out) {
        self.apply(SyncEvent::Output(out));
    }

    /// The physical attachment callback completed; only then does the gate change.
    fn on_attachment_changed(&self, audio: Option<bool>, text: Option<bool>) {
        self.apply(SyncEvent::AttachmentChanged { audio, text });
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
    core: Arc<SyncCore>,
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
            node.core.step(SyncInput::Reset);
            node.tracker.on_playback_reset();
        }
    }
}

impl SyncedAudioOutput {
    fn new(core: Arc<SyncCore>, next: Arc<dyn AudioOutput>) -> Arc<Self> {
        let node = Arc::new(SyncedAudioOutput {
            tracker: Arc::clone(&core.tracker),
            core,
            next: next.clone(),
            observer: Mutex::new(None),
        });
        let observer: Arc<dyn PlaybackObserver> = Arc::new(AudioEvents(Arc::downgrade(&node)));
        next.tracker().observe(observer.clone());
        *node.observer.lock().unwrap() = Some(observer);
        node
    }

    /// Synchronous: a wrapper's counter must move before the leaf's emit returns, or
    /// `wait_for_playout` races the event meant to release it.
    fn handle_playback(&self, ev: &PlaybackEvent) {
        match ev {
            PlaybackEvent::Started(e) => {
                if self.tracker.playback_started(e.segment, e.created_at) {
                    self.core.step(SyncInput::Started);
                }
            }
            PlaybackEvent::Progressed(e) => {
                if self.tracker.playback_progressed(e.segment, e.started_at, e.offset, e.duration) {
                    self.core.step(SyncInput::Progressed(e.offset + e.duration));
                }
            }
            PlaybackEvent::Finished { segment, event: e } => {
                if !self.tracker.accepts_report(*segment) {
                    return;
                }
                let position = e.playback_position;
                let heard =
                    self.core.step(SyncInput::Finished { position, interrupted: e.interrupted });
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
            self.core.step(SyncInput::Audio { secs });
            self.next.capture_frame(frame).await
        })
    }

    fn flush_segment(&self) {
        self.next.flush();
        self.core.step(SyncInput::AudioEnd);
    }

    fn clear_buffer(&self) {
        self.next.clear_buffer();
        self.core.step(SyncInput::Clear);
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
        self.core.step(SyncInput::Paused);
    }

    fn resume(&self) {
        self.next.resume();
    }
}

/// `_SyncedTextOutput`.
pub struct SyncedTextOutput {
    core: Arc<SyncCore>,
    next: Option<Arc<dyn TextOutput>>,
}

impl SyncedTextOutput {
    /// `aligned` records whether the caller used the timed API.
    async fn capture(&self, text: &TimedString, aligned: bool) -> Result<()> {
        for effect in self.core.decide(SyncEvent::Capture { text: text.clone(), aligned }) {
            match effect {
                SyncEffect::Capture { text, aligned } => {
                    if let Some(next) = &self.next {
                        if aligned {
                            next.capture_timed(&text).await?;
                        } else {
                            next.capture_text(&text.text).await?;
                        }
                    }
                }
                effect => {
                    self.core.perform(effect);
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
        for effect in self.core.decide(SyncEvent::FlushText) {
            match effect {
                SyncEffect::FlushText => {
                    if let Some(next) = &self.next {
                        next.flush();
                    }
                }
                effect => {
                    self.core.perform(effect);
                }
            }
        }
    }

    fn on_attached(&self) {
        for effect in self.core.decide(SyncEvent::AttachText) {
            match effect {
                SyncEffect::AttachText => {
                    if let Some(next) = &self.next {
                        next.on_attached();
                    }
                    self.core.on_attachment_changed(None, Some(true));
                }
                effect => {
                    self.core.perform(effect);
                }
            }
        }
    }

    fn on_detached(&self) {
        self.core.apply(SyncEvent::DetachText);
    }
}

/// Everything `TranscriptSynchronizer.__init__` takes, plus the two ports Rust needs explicitly.
pub struct SyncConfig {
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
    core: Arc<SyncCore>,
    audio: Arc<SyncedAudioOutput>,
    text: Arc<SyncedTextOutput>,
    /// Forwards released words to the text sink; ends once the close drops its sender.
    _forwarder: OwnedTask<()>,
}

impl TranscriptSynchronizer {
    pub fn new(config: SyncConfig) -> Arc<Self> {
        Arc::new(Self::build(config))
    }

    fn build(config: SyncConfig) -> Self {
        let SyncConfig { audio_next, text_next, speed, hyphenator, clock, spawner } = config;

        // `io.AudioOutput.__init__`'s auto-wrap, so an avatar hand-off can replace the leaf later.
        let audio_next = if audio_next.next_in_chain().is_none() {
            AudioSinkProxy::new(audio_next) as Arc<dyn AudioOutput>
        } else {
            audio_next
        };

        let opts = PacerOptions { speed, hyphenator };
        let (tx, rx) = mpsc::unbounded_channel();
        let core = Arc::new(SyncCore {
            state: Mutex::new(SyncState { lifecycle: SyncLifecycle::new(opts) }),
            clock,
            tx: Mutex::new(Some(tx)),
            forwarder_done: Arc::new(Monotone::new(false)),
            tracker: Arc::new(PlaybackTracker::new()),
        });
        // The task holds only the done flag, so it never keeps the core alive.
        let forwarder_done = Arc::clone(&core.forwarder_done);
        let sink = text_next.clone();
        let forwarder = spawner.spawn(async move {
            SyncCore::forward(sink, rx).await;
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
        self.core.step(SyncInput::Enabled(enabled));
    }

    /// End the live segment now and start a fresh one.
    pub fn rotate_segment(&self) {
        self.core.apply(SyncEvent::Rotate);
    }

    /// Close: end the live segment and give the forwarder [`lk_agents_abi::HOST_TEARDOWN_TIMEOUT`] to drain.
    /// `RoomIO.aclose` must close this before the transcription outputs it feeds.
    pub async fn aclose(&self) {
        for effect in self.core.decide(SyncEvent::Close) {
            match effect {
                SyncEffect::CloseSender => {
                    self.core.tx.lock().unwrap().take();
                }
                SyncEffect::BeginWait => {
                    for wait in self.core.decide(SyncEvent::WaitStarted(self.core.clock.now())) {
                        let SyncEffect::WaitUntil(at) = wait else {
                            unreachable!("close only waits on its forwarded words");
                        };
                        let left = at.saturating_duration_since(self.core.clock.now());
                        let _ = lk_agents_abi::timeout(
                            &*self.core.clock,
                            left,
                            self.core.forwarder_done.wait_for(|set| *set),
                        )
                        .await;
                        self.core.apply(SyncEvent::WaitFinished);
                    }
                }
                effect => {
                    self.core.perform(effect);
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "transcript_sync_tests.rs"]
mod tests;
