//! Ephemeral proposed-call classification; captured input occurrences remain native records.

use super::remote::RemoteTaskRef;
use crate::session::{InputCandidate, InputRequestRef};

/// The model-facing action vocabulary; internal task/input references never appear on its wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallAction {
    New,
    Continue,
    Respond,
    Unclear,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallCandidate {
    pub task: RemoteTaskRef,
    pub state: crate::a2a_protocol::TaskState,
    /// Observed communication supplies the task's intent even while it is still working.
    pub messages: Vec<crate::a2a_protocol::Message>,
    /// Only occurrences offered to this frontend model turn may supply held responses.
    pub responses: Vec<InputCandidate>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallRoutingRequest {
    pub remote_id: String,
    pub endpoint: String,
    /// Business arguments after the framework's task_id has been removed.
    pub proposed: serde_json::Value,
    /// An explicit selection restricts routing to this task; it does not select an answer.
    pub selected_task: Option<RemoteTaskRef>,
    pub history: Vec<crate::ChatMessage>,
    pub candidates: Vec<CallCandidate>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CallRoute {
    New,
    Continue(RemoteTaskRef),
    Respond(InputRequestRef),
    Unclear,
}

/// Native admission is distinct from a model verdict and is rechecked before dispatch.
#[derive(Debug, Clone, PartialEq)]
pub enum CallAdmission {
    New,
    Continue(RemoteTaskRef),
    Respond {
        request: InputRequestRef,
        message: Box<crate::ChatMessage>,
    },
    /// The original outgoing operation owns this response, including an uncertain send.
    AlreadyForwarded {
        task: RemoteTaskRef,
    },
}
