use crate::agent_session::Driver;
use crate::agent_session::effects::{Effect, Out, Timer};
use crate::room_io::RECORDER_FLUSH;
use crate::trace::PlayoutStep;
use lk_agents_abi::recorder::AudioRecorder;
use lk_agents_abi::voice::io::{PlaybackEvent, PlaybackObserver};
use lk_agents_abi::{Clock, Ts};
use std::sync::Arc;

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

/// Forwards one attachment's playback events to a recorder. Several attachments may share the
/// same physical tracker while their previously admitted operations settle.
pub struct RecorderPlaybackObserver {
    tap: RecorderTap,
    attachment: lk_agents_abi::resources::ResourceId,
}

impl RecorderPlaybackObserver {
    pub fn new(tap: RecorderTap, attachment: lk_agents_abi::resources::ResourceId) -> Self {
        Self { tap, attachment }
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
        let tap = &self.tap;
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
}

/// The recorder tap as the driver runs it: attached, observing the audio output, flushed on a
/// timer, detached with one last flush.
impl Driver {
    /// A second attach replaces the first.
    pub(crate) fn attach_recorder(
        &mut self,
        now: Ts,
        recorder: Arc<dyn lk_agents_abi::recorder::AudioRecorder>,
        out: &mut Out,
    ) {
        self.session.recorder_attached = true;
        out.machine().push(Effect::AttachRecorder {
            recorder: crate::agent_session::commands::HostRecorder(recorder),
            output: self.session.outputs.audio.as_ref().map(|o| o.id),
        });
        out.machine().arm(Timer::RecorderFlush, now.saturating_add(RECORDER_FLUSH));
    }

    /// One last flush, observer off, timer cancelled.
    pub(crate) fn detach_recorder(&mut self, now: Ts, out: &mut Out) {
        if !std::mem::replace(&mut self.session.recorder_attached, false) {
            return;
        }
        out.machine().push(Effect::FlushRecorder(self.step.wall_secs_at(now)));
        out.machine()
            .push(Effect::DetachRecorder(self.session.outputs.audio.as_ref().map(|o| o.id)));
        out.machine().cancel(Timer::RecorderFlush);
        out.trace(PlayoutStep::RecorderDetached);
    }

    pub(crate) fn flush_recorder(&mut self, now: Ts, rearm: bool, out: &mut Out) {
        if !self.session.recorder_attached {
            return;
        }
        out.machine().push(Effect::FlushRecorder(self.step.wall_secs_at(now)));
        if rearm && !self.session.closing_past_edge() {
            out.machine().arm(Timer::RecorderFlush, now.saturating_add(RECORDER_FLUSH));
        }
    }
}
