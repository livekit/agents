use crate::agent_session::Driver;
use crate::agent_session::cycle::{Cycle, TurnTimer};
use crate::io::{AudioOutput, PlaybackEvent, PlaybackObserver};
use crate::room_io::RECORDER_FLUSH;
use crate::trace::PlayoutStep;
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

/// The recorder tap as the driver runs it: attached, observing the audio output, flushed on a
/// timer, detached with one last flush.
impl Driver {
    /// A second attach replaces the first.
    pub(crate) fn attach_recorder(
        &mut self,
        now: Ts,
        recorder: Arc<dyn lk_agents_media::recorder::AudioRecorder>,
        cycle: &mut Cycle,
    ) {
        let tap = crate::recorder_io::RecorderTap::new(recorder, self.shared.time.clock());
        *self.shared.recorder.write().unwrap_or_else(|p| p.into_inner()) = Some(tap);
        self.observe_recorder(None);
        cycle.machine().arm(TurnTimer::RecorderFlush, now.saturating_add(RECORDER_FLUSH));
    }

    /// One last flush, observer off, timer cancelled.
    pub(crate) fn detach_recorder(&mut self, now: Ts, cycle: &mut Cycle) {
        if self.shared.recorder_tap().is_none() {
            return;
        }
        self.flush_recorder(now, false, cycle);
        if let Some(observer) =
            self.shared.recorder_observer.lock().unwrap_or_else(|p| p.into_inner()).take()
            && let Some(out) = self.shared.output.audio()
        {
            out.tracker().unobserve(&observer);
        }
        *self.shared.recorder.write().unwrap_or_else(|p| p.into_inner()) = None;
        cycle.machine().cancel(TurnTimer::RecorderFlush);
        cycle.trace(PlayoutStep::RecorderDetached);
    }

    /// Off `previous` first.
    pub(crate) fn observe_recorder(&self, previous: Option<&dyn AudioOutput>) {
        let Some(tap) = self.shared.recorder_tap() else { return };
        let mut slot = self.shared.recorder_observer.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(old) = slot.take()
            && let Some(prev) = previous
        {
            prev.tracker().unobserve(&old);
        }
        if let Some(out) = self.shared.output.audio() {
            let observer: Arc<dyn crate::io::PlaybackObserver> =
                Arc::new(crate::recorder_io::RecorderPlaybackObserver(tap));
            out.tracker().observe(observer.clone());
            *slot = Some(observer);
        }
    }

    pub(crate) fn flush_recorder(&mut self, now: Ts, rearm: bool, cycle: &mut Cycle) {
        let Some(tap) = self.shared.recorder_tap() else { return };
        let ok = tap.recorder.flush(tap.now()).is_ok();
        cycle.trace(PlayoutStep::RecorderFlushed { ok });
        if rearm && !self.session.closing_past_edge() {
            cycle.machine().arm(TurnTimer::RecorderFlush, now.saturating_add(RECORDER_FLUSH));
        }
    }
}
