use crate::io::{PlaybackEvent, PlaybackObserver};
use lk_agents_abi::{Clock, Ts};
use lk_agents_media::recorder::AudioRecorder;
use std::sync::Arc;
use std::time::UNIX_EPOCH;

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
        let wall = clock.wall().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64();
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

/// Forwards one [`crate::io::PlaybackTracker`]'s events to a recorder. Via the tracker because it
/// suppresses over-reports, which would close a segment still being filled.
pub struct RecorderPlaybackObserver(pub RecorderTap);

impl PlaybackObserver for RecorderPlaybackObserver {
    fn on_playback_event(&self, ev: &PlaybackEvent) {
        let tap = &self.0;
        let r = &tap.recorder;
        match ev {
            PlaybackEvent::Started(e) => r.on_playback_started(tap.at(e.created_at)),
            PlaybackEvent::Progressed(e) => r.on_playback_progressed(
                tap.at(e.started_at),
                e.offset.as_secs_f64(),
                e.duration.as_secs_f64(),
            ),
            PlaybackEvent::Finished(e) => {
                r.on_playback_finished(tap.now(), e.playback_position.as_secs_f64())
            }
        }
    }
}
