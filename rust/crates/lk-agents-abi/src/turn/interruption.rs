//! The adaptive-interruption seam: interruption or backchannel? One trait so a half-wired session is
//! unrepresentable. `audio` gets the real frames at pipeline rate; the implementor resamples.

use std::sync::Arc;
use std::time::Duration;

use crate::turn::detector::{DetectorLeg, LegEvent, ReqId};
use crate::{AudioFrame, StreamSink, Ts};

/// The reference's four sentinels, the overlap ones carrying its timestamps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptionSignal {
    AgentSpeechStarted,
    AgentSpeechEnded,
    OverlapStarted { started_at: Ts, speech_duration: Duration },
    OverlapEnded { ended_at: Ts, agent_ended: bool },
}

/// `inference.OverlappingSpeechEvent`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverlapVerdict {
    pub is_interruption: bool,
    /// Force-ended because the agent finished speaking: inconclusive, not a confirmed backchannel.
    pub agent_ended: bool,
    /// `ev.overlap_started_at or ev.detected_at`: a zero `overlap_started_at` falls through.
    pub ignore_user_transcript_until: Ts,
}

#[async_trait::async_trait]
pub trait InterruptionPort: Send + Sync {
    /// Installed once and pushed to per frame.
    fn audio(&self) -> Arc<dyn StreamSink<AudioFrame>>;

    /// Non-blocking: called from the session loop.
    fn signal(&self, signal: InterruptionSignal);

    /// The default drops the id; its verdicts come back with `req: None` and apply to the open
    /// overlap.
    fn signal_overlap(&self, overlap: ReqId, signal: InterruptionSignal) {
        let _ = overlap;
        self.signal(signal);
    }

    /// One per overlap, not per frame. `Closed` ends the pump; the session carries on with VAD-only
    /// interruption.
    async fn next_event(&self) -> LegEvent<OverlapVerdict>;
}

/// `signal(Some(req), OverlapStarted)` is `signal_overlap`; every other sentinel opens no request.
#[async_trait::async_trait]
impl DetectorLeg for dyn InterruptionPort {
    type Signal = InterruptionSignal;
    type Verdict = OverlapVerdict;

    fn audio(&self) -> Option<Arc<dyn StreamSink<AudioFrame>>> {
        Some(InterruptionPort::audio(self))
    }

    fn signal(&self, req: Option<ReqId>, signal: Self::Signal) {
        match req {
            Some(overlap) => self.signal_overlap(overlap, signal),
            None => InterruptionPort::signal(self, signal),
        }
    }

    async fn next(&self) -> LegEvent<Self::Verdict> {
        self.next_event().await
    }
}

/// Opened when the session starts (opening spawns), stamped on the session's anchor.
pub trait InterruptionSource: Send + Sync {
    fn open(&self, anchor: crate::WallAnchor) -> Arc<dyn InterruptionPort>;
}

/// Only a native detector crosses: there is no host protocol to proxy.
#[cfg(feature = "python")]
impl crate::python::HostProxy for dyn InterruptionSource {
    fn from_host(
        obj: &pyo3::Bound<'_, pyo3::PyAny>,
        _clock: Option<Arc<dyn crate::Clock>>,
    ) -> pyo3::PyResult<Arc<Self>> {
        crate::python::native::<dyn InterruptionSource>(obj).ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "interruption_detector must be a `_core.InterruptionDetector` -- the host's \
                 `inference.AdaptiveInterruptionDetector` wraps one and that is the object to \
                 pass (`detector._core`).",
            )
        })
    }

    fn view(
        _handle: crate::python::HostHandle,
        _clock: Option<Arc<dyn crate::Clock>>,
    ) -> pyo3::PyResult<Box<Self>> {
        Err(pyo3::exceptions::PyTypeError::new_err("an interruption detector is native only"))
    }
}
