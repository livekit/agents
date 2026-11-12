use crate::voice::session::SessionShared;
use lk_agents_abi::recorder::AudioRecorder;
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::voice::io::{PlaybackEvent, PlaybackObserver};
use lk_agents_abi::{Clock, Ts};
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
        let observer = RecorderPlaybackObserver::new(shared.recorder_tap(), id, &shared.unplayed);
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

/// Replays to a recorder attached at `now` what the active outputs still play, from where each
/// is: the person hears it, so the recording holds it.
pub(super) fn join(shared: &SessionShared, tap: &RecorderTap, outputs: &[ResourceId]) {
    // The trackers are asked with no lock held.
    let kept = shared.unplayed.lock().segments();
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
    let replay = {
        let mut unplayed = shared.unplayed.lock();
        for segment in settled {
            unplayed.retain(segment.attachment, |s| s != segment);
        }
        unplayed.join(tap.clock.now())
    };
    for step in replay {
        match step {
            Replay::Started(at) => tap.recorder.on_playback_started(tap.at(at)),
            Replay::Capture(frame) => tap.recorder.capture_frame(tap.now(), &frame),
        }
    }
}

/// Forwards one attachment's playback events to a recorder, and keeps what it still plays for
/// one attached later. Several attachments may share the same physical tracker while their
/// previously admitted operations settle.
pub struct RecorderPlaybackObserver {
    tap: Option<RecorderTap>,
    attachment: lk_agents_abi::resources::ResourceId,
    unplayed: Arc<Mutex<UnplayedAudio>>,
}

impl RecorderPlaybackObserver {
    pub fn new(
        tap: Option<RecorderTap>,
        attachment: lk_agents_abi::resources::ResourceId,
        unplayed: &Arc<Mutex<UnplayedAudio>>,
    ) -> Self {
        Self { tap, attachment, unplayed: Arc::clone(unplayed) }
    }
}

impl PlaybackObserver for RecorderPlaybackObserver {
    fn on_playback_event(&self, ev: &PlaybackEvent) {
        let attachment = match ev {
            PlaybackEvent::Started(e) => e.segment.attachment,
            PlaybackEvent::Progressed(e) => e.segment.attachment,
            PlaybackEvent::Finished { segment, .. } => segment.attachment,
        };
        if attachment != self.attachment.0 {
            return;
        }
        let told = {
            let mut unplayed = self.unplayed.lock();
            let told = unplayed.told(ev);
            unplayed.observe(ev);
            told
        };
        let (Some(tap), Some(ev)) = (&self.tap, told) else { return };
        let r = &tap.recorder;
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

    fn on_playback_reset(&self) {
        self.unplayed.lock().reset(self.attachment.0);
    }
}

#[cfg(test)]
#[path = "recorder_tests.rs"]
mod tests;
