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
            Self::Ready => CloseAction::Ready,
            Self::Closed => CloseAction::None,
        }
    }
}

/// The session's start planning (`_core.plan_session_start`).
pub struct SessionStartPlans;

impl lk_agents_abi::voice::session_effects::SessionStart for SessionStartPlans {
    fn reply_tool_indices(tools: Vec<lk_agents_abi::ToolSpec>, names: Vec<String>) -> Vec<usize> {
        tools
            .into_iter()
            .enumerate()
            .filter_map(|(index, tool)| {
                (tool.provider_tool.is_some() || names.contains(&tool.name)).then_some(index)
            })
            .collect()
    }
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

#[cfg(test)]
mod plans_tests {
    use super::SessionStartPlans;
    use lk_agents_abi::voice::session_effects::SessionStart as _;

    #[test]
    fn narrowing_local_tools_keeps_provider_tools_and_declaration_order() {
        let tool = |name: &str, provider| lk_agents_abi::ToolSpec {
            name: name.into(),
            description: String::new(),
            json_schema: serde_json::Value::Null,
            flags: Default::default(),
            provider_tool: provider,
        };
        let provider = lk_agents_abi::llm::types::ProviderToolDeclaration {
            provider: None,
            payload: serde_json::Value::Null,
        };
        let tools = vec![tool("first", None), tool("hosted", Some(provider)), tool("last", None)];
        assert_eq!(
            SessionStartPlans::reply_tool_indices(tools.clone(), vec!["last".into()]),
            vec![1, 2]
        );
        assert_eq!(SessionStartPlans::reply_tool_indices(tools, Vec::new()), vec![1]);
    }
}
