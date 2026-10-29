//! Session lifecycle decisions over an observed phase; execution retains the phase's resources.

use crate::agent_activity::options::{RECORDING_ALL_OFF, RECORDING_ALL_ON, records_anything};
use lk_agents_abi::voice::session::{OpenRefused, SessionRefused};
use lk_agents_abi::voice::session_effects::StartPlan;

#[derive(Clone, Copy)]
pub enum SessionLifecycle {
    Ready,
    Opening,
    Running,
    Closed,
}

pub enum OpenAction {
    Begin,
    InstallRunning,
}

pub enum RecorderTarget {
    Builder,
    Running,
    Ignore,
}

pub enum CloseAction {
    Running,
    Opening,
    Closed,
    None,
}

impl SessionLifecycle {
    pub fn open(self) -> Result<OpenAction, OpenRefused> {
        match self {
            Self::Ready => Ok(OpenAction::Begin),
            Self::Opening | Self::Running => Err(OpenRefused::AlreadyStarted),
            Self::Closed => Err(OpenRefused::Closed),
        }
    }

    pub fn opened(self) -> OpenAction {
        OpenAction::InstallRunning
    }

    pub fn configure_ready(self) -> bool {
        matches!(self, Self::Ready)
    }

    pub fn running(self) -> Result<(), SessionRefused> {
        match self {
            Self::Running => Ok(()),
            Self::Ready | Self::Opening => Err(SessionRefused::NotStarted),
            Self::Closed => Err(SessionRefused::Closed),
        }
    }

    pub fn recorder_target(self) -> RecorderTarget {
        match self {
            Self::Ready => RecorderTarget::Builder,
            Self::Running => RecorderTarget::Running,
            Self::Opening | Self::Closed => RecorderTarget::Ignore,
        }
    }

    /// Running access remains available until the executor reports the close complete.
    pub fn close(self) -> CloseAction {
        match self {
            Self::Running => CloseAction::Running,
            Self::Opening => CloseAction::Opening,
            Self::Ready | Self::Closed => CloseAction::None,
        }
    }

    pub fn closed(self) -> CloseAction {
        CloseAction::Closed
    }
}

/// The session's start planning (`_core.plan_session_start`).
pub struct SessionStartPlans;

impl lk_agents_abi::voice::session_effects::SessionStart for SessionStartPlans {
    /// The job's recording setting, held by its primary session only: a second session in the
    /// job neither records nor hosts.
    fn plan_session_start(job_default: bool, another_session_is_primary: bool) -> StartPlan {
        let is_primary = !another_session_is_primary;
        let recording =
            if job_default && is_primary { RECORDING_ALL_ON } else { RECORDING_ALL_OFF };
        StartPlan {
            recording,
            is_primary,
            hosting: is_primary,
            records: records_anything(&recording),
        }
    }
}
