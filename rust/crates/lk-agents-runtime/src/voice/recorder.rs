use crate::voice::session::SessionShared;
use lk_agents_abi::recorder::AudioRecorder;
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::voice::io::{PlaybackEvent, PlaybackObserver, PlaybackSegment};
use lk_agents_abi::{AudioFrame, Clock, Ts};
use lk_agents_voice::unplayed_audio::{Replay, UnplayedAudio};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Default)]
pub(super) struct RecorderOutputs(BTreeMap<ResourceId, Arc<dyn PlaybackObserver>>);

impl RecorderOutputs {
    pub(super) fn detach(&mut self, shared: &SessionShared, output: Option<ResourceId>) {
        let removed: Vec<_> = match output {
            Some(id) => self.0.remove(&id).map(|observer| vec![(id, observer)]).unwrap_or_default(),
            None => std::mem::take(&mut self.0).into_iter().collect(),
        };
        for (id, observer) in removed {
            if let Some(output) = shared.resources.audio_output(id) {
                output.tracker().unobserve(&observer);
            }
        }
    }

    /// Watched with no recorder too: what the output still plays is kept for one that attaches.
    pub(super) fn attach(&mut self, shared: &SessionShared, id: ResourceId) {
        let Some(output) = shared.resources.audio_output(id) else { return };
        let observer = RecorderPlaybackObserver::new(Arc::clone(&shared.recorder), id);
        let observer: Arc<dyn PlaybackObserver> = Arc::new(observer);
        if let Some(old) = self.0.insert(id, observer.clone()) {
            output.tracker().unobserve(&old);
        }
        output.tracker().observe(observer);
    }
}

/// A recorder, and the monotonic-to-wall anchor its stamps are converted against.
#[derive(Clone)]
pub struct RecorderTap {
    pub recorder: Arc<dyn AudioRecorder>,
    clock: Arc<dyn Clock>,
    /// `(wall seconds, the `Ts` it was read at)`, read once at construction.
    anchor: (f64, Ts),
}

impl RecorderTap {
    pub fn new(recorder: Arc<dyn AudioRecorder>, clock: Arc<dyn Clock>) -> Self {
        let wall = lk_agents_abi::unix_secs(clock.wall());
        let anchor = (wall, clock.now());
        RecorderTap { recorder, clock, anchor }
    }

    /// `ts` as wall seconds, saturating below the anchor.
    pub fn at(&self, ts: Ts) -> f64 {
        let (wall, origin) = self.anchor;
        wall + ts.saturating_duration_since(origin).as_secs_f64()
    }

    pub fn now(&self) -> f64 {
        self.at(self.clock.now())
    }
}

/// The agent channel's one feed: what the outputs still play, and the recorder that hears it.
/// Captures, playback reports, an attach and a detach are applied one at a time, each with its
/// recorder calls, so an attach lands between two captures and every frame reaches the recording
/// exactly once, in order.
#[derive(Default)]
pub struct RecorderFeed(Mutex<Feed>);

#[derive(Default)]
struct Feed {
    unplayed: UnplayedAudio,
    tap: Option<RecorderTap>,
}

impl RecorderFeed {
    /// The attached recorder, for the microphone's frames and the flushes.
    pub fn tap(&self) -> Option<RecorderTap> {
        self.0.lock().tap.clone()
    }

    /// A frame the output took for `segment`.
    pub fn captured(&self, segment: Option<PlaybackSegment>, frame: &AudioFrame) {
        let mut feed = self.0.lock();
        if let Some(segment) = segment {
            feed.unplayed.captured(segment, frame);
        }
        if let Some(tap) = &feed.tap {
            tap.recorder.capture_frame(tap.now(), frame);
        }
    }

    /// `tap` attaches: it is told what the outputs still play, from where each is, then hears
    /// what follows.
    pub fn attach(&self, tap: RecorderTap, settled: &[PlaybackSegment]) {
        let mut feed = self.0.lock();
        for segment in settled {
            feed.unplayed.retain(segment.attachment, |s| s != *segment);
        }
        for step in feed.unplayed.join(tap.clock.now()) {
            match step {
                Replay::Started(at) => tap.recorder.on_playback_started(tap.at(at)),
                Replay::Capture(frame) => tap.recorder.capture_frame(tap.now(), &frame),
            }
        }
        feed.tap = Some(tap);
    }

    pub fn detach(&self) {
        self.0.lock().tap = None;
    }

    /// The segments kept, oldest first.
    fn segments(&self) -> Vec<PlaybackSegment> {
        self.0.lock().unplayed.segments()
    }

    fn played(&self, ev: &PlaybackEvent) {
        let mut feed = self.0.lock();
        let told = feed.unplayed.told(ev);
        feed.unplayed.observe(ev);
        let Some(tap) = &feed.tap else { return };
        let r = &tap.recorder;
        for ev in told {
            match ev {
                PlaybackEvent::Started(e) => r.on_playback_started(tap.at(e.created_at)),
                PlaybackEvent::Progressed(e) => r.on_playback_progressed(
                    tap.at(e.started_at),
                    e.offset.as_secs_f64(),
                    e.duration.as_secs_f64(),
                ),
                PlaybackEvent::Finished { event: e, .. } => {
                    r.on_playback_finished(tap.now(), e.playback_position.as_secs_f64())
                }
            }
        }
    }

    fn reset(&self, attachment: u64) {
        self.0.lock().unplayed.reset(attachment);
    }
}

/// Attaches `tap` to the feed: what the active outputs still play is replayed to it first.
pub(super) fn join(shared: &SessionShared, tap: RecorderTap, outputs: &[ResourceId]) {
    // The trackers are asked with no lock held: a segment settled meanwhile was reported, so the
    // feed has already let it go.
    let kept = shared.recorder.segments();
    let settled: Vec<_> = outputs
        .iter()
        .filter_map(|id| shared.resources.audio_output(*id).map(|output| (*id, output)))
        .flat_map(|(id, output)| {
            let tracker = output.tracker();
            kept.iter()
                .filter(|s| s.attachment == id.0 && !tracker.has_pending_segment(**s))
                .copied()
                .collect::<Vec<_>>()
        })
        .collect();
    shared.recorder.attach(tap, &settled);
}

/// Forwards one attachment's playback events to the feed. Several attachments may share the same
/// physical tracker while their previously admitted operations settle.
pub struct RecorderPlaybackObserver {
    feed: Arc<RecorderFeed>,
    attachment: ResourceId,
}

impl RecorderPlaybackObserver {
    pub fn new(feed: Arc<RecorderFeed>, attachment: ResourceId) -> Self {
        Self { feed, attachment }
    }
}

impl PlaybackObserver for RecorderPlaybackObserver {
    fn on_playback_event(&self, ev: &PlaybackEvent) {
        let attachment = match ev {
            PlaybackEvent::Started(e) => e.segment.attachment,
            PlaybackEvent::Progressed(e) => e.segment.attachment,
            PlaybackEvent::Finished { segment, .. } => segment.attachment,
        };
        if attachment == self.attachment.0 {
            self.feed.played(ev);
        }
    }

    fn on_playback_reset(&self) {
        self.feed.reset(self.attachment.0);
    }
}

#[cfg(test)]
#[path = "recorder_tests.rs"]
mod tests;
