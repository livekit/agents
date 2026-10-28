//! Publication ownership, independent of the SDK operation that creates the track.

use lk_agents_abi::voice::io::{IoError, Result};
use lk_agents_abi::voice::rtc::TrackSid;

#[derive(Default)]
enum Phase {
    #[default]
    Idle,
    Creating,
    Subscribing(TrackSid),
    Ready(TrackSid),
    Failed {
        error: IoError,
        sid: Option<TrackSid>,
    },
    Closed {
        creating: bool,
        sid: Option<TrackSid>,
    },
}

#[derive(Default)]
pub(super) struct Publication {
    phase: Phase,
}

impl Publication {
    pub(super) fn begin(&mut self) -> bool {
        if !matches!(self.phase, Phase::Idle) {
            return false;
        }
        self.phase = Phase::Creating;
        true
    }

    /// Install the source in the same transition, or retain it for retirement.
    pub(super) fn adopt(&mut self, sid: TrackSid) -> bool {
        if !matches!(self.phase, Phase::Creating) {
            return false;
        }
        self.phase = Phase::Subscribing(sid);
        true
    }

    pub(super) fn creation_retired(&mut self) {
        if let Phase::Closed { creating, .. } = &mut self.phase {
            *creating = false;
        }
    }

    pub(super) fn creating(&self) -> bool {
        matches!(self.phase, Phase::Creating | Phase::Closed { creating: true, .. })
    }

    pub(super) fn ready(&mut self) {
        if let Phase::Subscribing(sid) = &self.phase {
            self.phase = Phase::Ready(sid.clone());
        }
    }

    pub(super) fn fail(&mut self, error: IoError) {
        self.phase = match std::mem::take(&mut self.phase) {
            Phase::Creating => Phase::Failed { error, sid: None },
            Phase::Subscribing(sid) => Phase::Failed { error, sid: Some(sid) },
            Phase::Closed { sid, .. } => Phase::Closed { creating: false, sid },
            other => other,
        };
    }

    pub(super) fn outcome(&self) -> Option<Result<()>> {
        match &self.phase {
            Phase::Ready(_) => Some(Ok(())),
            Phase::Failed { error, .. } => Some(Err(error.clone())),
            Phase::Closed { .. } => Some(Err(IoError::Rtc("room audio output is closed".into()))),
            Phase::Idle | Phase::Creating | Phase::Subscribing(_) => None,
        }
    }

    pub(super) fn close(&mut self) {
        self.phase = match std::mem::take(&mut self.phase) {
            Phase::Creating => Phase::Closed { creating: true, sid: None },
            Phase::Subscribing(sid) | Phase::Ready(sid) => {
                Phase::Closed { creating: false, sid: Some(sid) }
            }
            Phase::Failed { sid, .. } => Phase::Closed { creating: false, sid },
            Phase::Idle => Phase::Closed { creating: false, sid: None },
            closed @ Phase::Closed { .. } => closed,
        };
    }

    pub(super) fn retire(&mut self) -> Option<TrackSid> {
        match &mut self.phase {
            Phase::Closed { sid, .. } => sid.take(),
            _ => None,
        }
    }
}
