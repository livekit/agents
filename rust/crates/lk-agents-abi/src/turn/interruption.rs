//! The adaptive-interruption seam: interruption or backchannel? One trait so a half-wired session is
//! unrepresentable. `audio` gets the real frames at pipeline rate; the implementor resamples.

use std::sync::Arc;
use std::time::Duration;

use crate::turn::detector::{LegEvent, ReqId};
use crate::{AudioFrame, StreamInput, Ts};

/// The reference's four sentinels, the overlap ones carrying its timestamps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptionSignal {
    AgentSpeechStarted,
    AgentSpeechEnded,
    OverlapStarted { started_at: Ts, speech_duration: Duration },
    OverlapEnded { ended_at: Ts, agent_ended: bool },
}

/// `inference.OverlappingSpeechEvent`.
#[derive(Debug, Clone, PartialEq)]
pub struct OverlapVerdict {
    pub is_interruption: bool,
    /// Force-ended because the agent finished speaking: inconclusive, not a confirmed backchannel.
    pub agent_ended: bool,
    /// `ev.overlap_started_at or ev.detected_at`: a zero `overlap_started_at` falls through.
    pub ignore_user_transcript_until: Ts,
    /// The detector's event, emitted as `overlapping_speech`; `None` from a detector without one.
    pub reported: Option<Arc<crate::voice::events::OverlappingSpeechEvent>>,
    /// The detector's `metrics_collected` for this verdict; `None` from a detector without one.
    pub metrics: Option<crate::metrics::base::InterruptionMetrics>,
}

pub trait InterruptionPort: Send + Sync {
    /// Installed once and pushed to per frame.
    fn audio(&self) -> Arc<dyn StreamInput<AudioFrame>>;

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
    fn next_event(&self) -> crate::BoxFuture<'_, LegEvent<OverlapVerdict>>;
}

/// Opened when the session starts (opening spawns), stamped on the session's anchor. Only a
/// native detector crosses: there is no host protocol to proxy.
#[crate::host(native(
    refused = "interruption_detector must be an `inference.AdaptiveInterruptionDetector`"
))]
pub trait InterruptionSource: Send + Sync {
    /// Audio drops are measured under `cx`'s job.
    fn open(
        &self,
        anchor: crate::WallAnchor,
        cx: &crate::telemetry::Cx,
    ) -> Arc<dyn InterruptionPort>;
}

// Public data contracts. Python behavior remains in its adapters.

#[crate::host(record(
    module = "interruption_errors",
    path = "livekit.agents.inference",
    pydantic,
    arbitrary_types,
    constant(type = "interruption_detection_error")
))]
pub struct InterruptionDetectionError {
    #[host(default_factory = "time.time")]
    pub timestamp: f64,
    pub label: String,
    #[host(annotation = "typing.Annotated[Exception, pydantic.Field(exclude=True)]")]
    pub error: crate::HostRef,
    pub recoverable: bool,
}
