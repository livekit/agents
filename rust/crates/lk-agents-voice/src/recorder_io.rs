//! RecorderIO's attachment and periodic flush decisions. Runtime owns physical callbacks.

use crate::agent_session::effects::{Effect, Out, Timer};
use crate::trace::PlayoutStep;
use lk_agents_abi::Ts;
use lk_agents_abi::resources::ResourceId;
use std::time::Duration;

const RECORDER_FLUSH: Duration = Duration::from_millis(2500);

/// The recording lifetime of one session. All clocks and resource identities are supplied.
pub(crate) struct RecorderIo {
    initial: Option<ResourceId>,
    attached: Option<ResourceId>,
}

impl RecorderIo {
    pub fn new(initial: Option<ResourceId>) -> Self {
        Self { initial, attached: None }
    }
    pub fn take_initial(&mut self) -> Option<ResourceId> {
        self.initial.take()
    }
    pub fn resource(&self) -> Option<ResourceId> {
        self.attached.or(self.initial)
    }

    pub fn attach(
        &mut self,
        now: Ts,
        recorder: ResourceId,
        output: Option<ResourceId>,
        out: &mut Out,
    ) {
        self.attached = Some(recorder);
        out.machine().push(Effect::AttachRecorder {
            recorder: crate::agent_session::commands::RecorderHandle(recorder),
            output,
        });
        out.machine().arm(Timer::RecorderFlush, now.saturating_add(RECORDER_FLUSH));
    }

    pub fn detach(&mut self, wall: f64, output: Option<ResourceId>, out: &mut Out) {
        if self.attached.take().is_none() {
            return;
        }
        out.machine().push(Effect::FlushRecorder(wall));
        out.machine().push(Effect::DetachRecorder(output));
        out.machine().cancel(Timer::RecorderFlush);
        out.trace(PlayoutStep::RecorderDetached);
    }

    pub fn flush(&self, now: Ts, wall: f64, rearm: bool, out: &mut Out) {
        if self.attached.is_none() {
            return;
        }
        out.machine().push(Effect::FlushRecorder(wall));
        if rearm {
            out.machine().arm(Timer::RecorderFlush, now.saturating_add(RECORDER_FLUSH));
        }
    }
}
