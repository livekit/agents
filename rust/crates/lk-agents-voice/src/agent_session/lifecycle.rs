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

pub enum RecorderTarget {
    Builder,
    Running,
    Ignore,
}

pub enum CloseAction {
    Ready,
    Running,
    Opening,
    None,
}

impl SessionLifecycle {
    pub fn open(self) -> Result<(), OpenRefused> {
        match self {
            Self::Ready => Ok(()),
            Self::Opening | Self::Running => Err(OpenRefused::AlreadyStarted),
            Self::Closed => Err(OpenRefused::Closed),
        }
    }

    /// Startup may finish only while its admission still owns the opening phase.
    pub fn accepts_opened(self) -> bool {
        matches!(self, Self::Opening)
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
            Self::Ready => CloseAction::Ready,
            Self::Closed => CloseAction::None,
        }
    }
}

/// The session's start planning (`_core.plan_session_start`).
pub struct SessionStartPlans;

impl lk_agents_abi::voice::session_effects::SessionStart for SessionStartPlans {
    fn default_session_models(
        vad_supplied: bool,
        detector_supplied: bool,
        realtime: bool,
    ) -> lk_agents_abi::voice::session_effects::DefaultModelPlan {
        lk_agents_abi::voice::session_effects::DefaultModelPlan {
            vad_model: (!vad_supplied)
                .then_some(lk_agents_abi::voice::session_effects::DefaultVadModel::Silero),
            turn_detector: !detector_supplied && !realtime,
        }
    }

    fn user_turn_exceeded_reply() -> lk_agents_abi::voice::session_effects::ExceededTurnReply {
        lk_agents_abi::voice::session_effects::ExceededTurnReply {
            instructions: "The user has been speaking too long without giving a chance to reply. Politely cut in with a short reply or notice. Keep it short since the user cannot interrupt it.".into(),
            allow_interruptions: false,
            tool_choice: lk_agents_abi::ToolChoice::Forbidden,
        }
    }
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

pub struct StorageStart {
    configured_lane: bool,
}

pub enum StorageSource<T> {
    Lane(T),
    Ephemeral,
}

pub use lk_agents_core::resource_lifecycle::{CloseOwner, OwnerLifecycle, StartOwner};

/// The latest requested agent is exposed until that handoff settles. An older handoff cannot
/// clear a newer selection, even when both requests name the same agent.
#[derive(Default)]
pub struct AgentSelection {
    next: u64,
    pending: Option<u64>,
}

impl AgentSelection {
    pub fn begin(&mut self) -> u64 {
        self.next += 1;
        self.pending = Some(self.next);
        self.next
    }
    pub fn settled(&mut self, id: u64) -> bool {
        if self.pending == Some(id) {
            self.pending = None;
            true
        } else {
            false
        }
    }
}

impl StorageStart {
    pub fn new(configured_lane: bool) -> Self {
        Self { configured_lane }
    }
    pub fn load_lane(&self) -> bool {
        self.configured_lane
    }
    /// Refine the supplied load result without retaining its payload in decision state.
    /// A Lane outcome therefore always carries the successful result it selected.
    pub fn after_load<T>(&self, loaded: Option<T>) -> StorageSource<T> {
        match loaded {
            Some(lane) if self.configured_lane => StorageSource::Lane(lane),
            Some(_) | None => StorageSource::Ephemeral,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_older_handoff_cannot_clear_the_latest_selection() {
        let mut selection = AgentSelection::default();
        let first = selection.begin();
        let latest = selection.begin();
        assert!(!selection.settled(first));
        assert!(selection.settled(latest));
        assert!(!selection.settled(latest));
    }

    #[test]
    fn only_a_successfully_loaded_configured_lane_is_durable() {
        assert!(!StorageStart::new(false).load_lane());
        assert!(StorageStart::new(true).load_lane());
        assert!(matches!(StorageStart::new(true).after_load(Some(())), StorageSource::Lane(())));
        assert!(matches!(StorageStart::new(true).after_load::<()>(None), StorageSource::Ephemeral));
        assert!(matches!(
            StorageStart::new(false).after_load::<()>(None),
            StorageSource::Ephemeral
        ));
        assert!(matches!(StorageStart::new(false).after_load(Some(())), StorageSource::Ephemeral));
        assert!(matches!(
            StorageStart::new(true).after_load(Some("loaded lane")),
            StorageSource::Lane("loaded lane")
        ));
    }
}
