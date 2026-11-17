use lk_agents_abi::a2a_protocol as a2a;
pub mod journal;
pub mod retained;

use a2a::{Message, Part, Role, TaskState};
use lk_agents_abi::a2a::metadata::KIND_KEY;
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use serde::{Deserialize, Serialize};

use crate::a2a::caller_history::CallerHistory;
use crate::a2a::client::TaskUpdate;
use crate::a2a::rpc::TaskRef;

// Credentials are verified in runtime before this owner receives `VerifiedAuthority`.

/// A local verifier's authority for this exact destination and caller thread. Never restored
/// from conversation state or inferred from peer-controlled card/metadata fields.
#[derive(Debug, Clone)]
pub struct VerifiedAuthority {
    endpoint: String,
    thread_id: String,
    caller_identity: String,
}

impl VerifiedAuthority {
    /// Only the credential/trust adapter calls this, after verification. Plain clients do not
    /// need this proof: their access is checked by the destination endpoint.
    pub fn from_local_verifier(
        endpoint: String,
        thread_id: String,
        caller_identity: String,
    ) -> Result<Self, DelegationError> {
        if endpoint.is_empty() || thread_id.is_empty() || caller_identity.is_empty() {
            return Err(DelegationError::InvalidAuthority);
        }
        Ok(Self { endpoint: endpoint.trim_end_matches('/').into(), thread_id, caller_identity })
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DelegationError {
    #[error("the authority does not cover this delegation's destination, thread and caller")]
    InvalidAuthority,
    #[error("the delegation is not waiting for a reply")]
    NotWaiting,
    #[error("the remote update belongs to another task or context")]
    WrongTask,
    #[error("the local interaction is already running or settled")]
    Settled,
    #[error("the remote task ended without a result")]
    MissingOutcome,
    #[error("the retained interaction does not match its tool, call or destination")]
    InvalidState,
    #[error(transparent)]
    History(#[from] crate::a2a::caller_history::HistoryError),
}

/// The original parent call and remote destination, retained independently of presentation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InteractionId {
    pub delegation_id: String,
    pub tool_id: String,
    pub parent_call_id: String,
    pub endpoint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Local observation lifetime. Remote task status and cancellation belong to task state.
pub enum InteractionPhase {
    Ready,
    Running,
    InputRequired,
    Settled,
}

/// Only data is saved. No queue, transport, credential proof or live await is serialized.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InteractionState {
    pub id: InteractionId,
    pub context_id: Option<String>,
    pub pending_task_id: Option<String>,
    pub phase: InteractionPhase,
    pub request_id: Option<String>,
    /// The checkpointed input occurrence this presentation asks the person to answer.
    pub input_request: Option<lk_agents_abi::delegation::remote::InputRequestRef>,
    pub question_sequence: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DelegationUpdate {
    Progress(String),
    InputRequired {
        request_id: String,
        question: String,
        verbatim: Option<String>,
    },
    Completed(ToolOutcome),
    Failed(ToolOutcome),
    Cancelled,
    /// A merged task has no answer of its own; follow its actual replacement.
    Follow(TaskRef),
    Pending,
}

/// One interaction's decision owner. A reusable RemoteAgent holds none of this state.
#[derive(Debug)]
pub struct RemoteInteraction {
    state: InteractionState,
}

impl RemoteInteraction {
    pub fn new(id: InteractionId) -> Result<Self, DelegationError> {
        if id.delegation_id.is_empty()
            || id.tool_id.is_empty()
            || id.parent_call_id.is_empty()
            || id.endpoint.is_empty()
        {
            return Err(DelegationError::InvalidState);
        }
        Ok(Self {
            state: InteractionState {
                id,
                context_id: None,
                pending_task_id: None,
                phase: InteractionPhase::Ready,
                request_id: None,
                input_request: None,
                question_sequence: 0,
            },
        })
    }

    pub fn restore(state: InteractionState, id: &InteractionId) -> Result<Self, DelegationError> {
        Self::new(state.id.clone())?;
        if &state.id != id
            || state.context_id.as_deref() == Some("")
            || state.pending_task_id.as_deref() == Some("")
            || state.pending_task_id.is_some() && state.context_id.is_none()
            || state.phase == InteractionPhase::InputRequired
                && (state.pending_task_id.is_none() || state.request_id.is_none())
            || state.input_request.as_ref().is_some_and(|reference| {
                reference.remote().is_none_or(|(owner, task, occurrence)| {
                    owner.is_empty()
                        || occurrence.is_empty()
                        || Some(&task.task_id) != state.pending_task_id.as_ref()
                        || Some(&task.context_id) != state.context_id.as_ref()
                })
            })
        {
            return Err(DelegationError::InvalidState);
        }
        Ok(Self { state })
    }

    pub fn snapshot(&self) -> InteractionState {
        self.state.clone()
    }

    /// Builds every initial/subsequent/answer message through the same policy. Caller history
    /// here is explicitly supplied and already validated; implicit thread history requires the
    /// local proof. A capability only controls understanding of the explicit data part.
    #[allow(
        clippy::too_many_arguments,
        reason = "each input is an independent part of the message policy"
    )]
    pub fn message(
        &mut self,
        text: String,
        message_id: String,
        caller_thread: Option<&str>,
        caller_identity: &str,
        authority: Option<&VerifiedAuthority>,
        offers_extension: bool,
        history: Option<&CallerHistory>,
        default_context: Option<&str>,
    ) -> Result<Message, DelegationError> {
        if matches!(self.state.phase, InteractionPhase::Settled | InteractionPhase::Running) {
            return Err(DelegationError::Settled);
        }
        if message_id.is_empty() || default_context == Some("") {
            return Err(DelegationError::InvalidState);
        }
        let shared = match authority {
            Some(proof)
                if proof.endpoint == self.state.id.endpoint.trim_end_matches('/')
                    && Some(proof.thread_id.as_str()) == caller_thread
                    && proof.caller_identity == caller_identity =>
            {
                Some(proof.thread_id.clone())
            }
            Some(_) => return Err(DelegationError::InvalidAuthority),
            None => None,
        };
        if let (Some(shared), Some(context)) = (&shared, &self.state.context_id)
            && shared != context
        {
            return Err(DelegationError::InvalidAuthority);
        }
        let livekit = offers_extension || shared.is_some();
        let mut parts = vec![Part::text(text)];
        if offers_extension
            && shared.is_none()
            && let Some(history) = history
        {
            parts.push(history.to_part()?);
        }
        let message = Message {
            message_id,
            // A task continuation keeps its context; verified caller sharing is authoritative.
            // The saved default only addresses new work and never implies a pending task.
            context_id: shared
                .or_else(|| self.state.context_id.clone())
                .or_else(|| default_context.map(str::to_owned)),
            task_id: self.state.pending_task_id.clone(),
            role: Role::User,
            parts,
            metadata: livekit.then(|| [(KIND_KEY.to_owned(), "delegation".into())].into()),
            extensions: None,
            reference_task_ids: None,
        };
        self.state.context_id = message.context_id.clone();
        self.state.phase = InteractionPhase::Running;
        self.state.request_id = None;
        Ok(message)
    }

    /// An answer addresses the exact retained input request; presentation cannot change it.
    pub fn accepts_reply(&self, request_id: &str) -> Result<(), DelegationError> {
        if self.state.phase != InteractionPhase::InputRequired
            || self.state.request_id.as_deref() != Some(request_id)
        {
            return Err(DelegationError::NotWaiting);
        }
        Ok(())
    }

    /// Bind the observation's committed occurrence before presenting it. A later status may
    /// supersede it; sending this reference then fails instead of answering another question.
    pub fn retain_input_request(
        &mut self,
        reference: Option<lk_agents_abi::delegation::remote::InputRequestRef>,
    ) -> Result<(), DelegationError> {
        if self.state.phase != InteractionPhase::InputRequired
            || reference.as_ref().is_some_and(|reference| {
                reference.remote().is_none_or(|(owner, task, occurrence)| {
                    owner.is_empty()
                        || occurrence.is_empty()
                        || Some(&task.task_id) != self.state.pending_task_id.as_ref()
                        || Some(&task.context_id) != self.state.context_id.as_ref()
                })
            })
        {
            return Err(DelegationError::InvalidState);
        }
        self.state.input_request = reference;
        Ok(())
    }

    pub fn update(&mut self, update: &TaskUpdate) -> Result<DelegationUpdate, DelegationError> {
        if self.state.phase == InteractionPhase::Settled {
            return Err(DelegationError::Settled);
        }
        self.identify(update)?;
        self.state.input_request = None;
        let exact = update.metadata.get("livekit/verbatim").and_then(|v| v.as_str());
        let outcome = match update.state {
            TaskState::Unspecified | TaskState::Submitted | TaskState::Working => {
                if update.state != TaskState::Working || update.text.is_empty() || !update.progress
                {
                    DelegationUpdate::Pending
                } else {
                    DelegationUpdate::Progress(update.text.clone())
                }
            }
            TaskState::InputRequired => {
                if update.task_id.is_empty() || update.context_id.is_empty() {
                    return Err(DelegationError::InvalidState);
                }
                self.state.question_sequence = self
                    .state
                    .question_sequence
                    .checked_add(1)
                    .ok_or(DelegationError::InvalidState)?;
                let request_id = format!(
                    "{}:{}:{}",
                    self.state.id.delegation_id, update.task_id, self.state.question_sequence
                );
                self.state.request_id = Some(request_id.clone());
                self.state.phase = InteractionPhase::InputRequired;
                DelegationUpdate::InputRequired {
                    request_id,
                    question: update.text.clone(),
                    verbatim: exact
                        .filter(|s| !s.is_empty() && update.text.starts_with(s))
                        .map(str::to_owned),
                }
            }
            TaskState::Completed => {
                if let Some(task) = &update.follow_task {
                    self.state.pending_task_id = Some(task.id.clone());
                    return Ok(DelegationUpdate::Follow(task.clone()));
                }
                self.state.phase = InteractionPhase::Settled;
                DelegationUpdate::Completed(answer(&update.text, exact))
            }
            TaskState::Failed | TaskState::Rejected => {
                self.state.phase = InteractionPhase::Settled;
                DelegationUpdate::Failed(ToolOutcome::ToolError(if update.text.is_empty() {
                    "the remote task failed".into()
                } else {
                    update.text.clone()
                }))
            }
            TaskState::Canceled => {
                self.state.phase = InteractionPhase::Settled;
                DelegationUpdate::Cancelled
            }
            TaskState::AuthRequired => {
                // Authentication is an interrupted remote task, not a human input request.
                // Report the tool's inability to continue without changing that task state.
                self.state.phase = InteractionPhase::Settled;
                DelegationUpdate::Failed(ToolOutcome::ToolError(if update.text.is_empty() {
                    "the remote task requires additional authentication".into()
                } else {
                    update.text.clone()
                }))
            }
        };
        Ok(outcome)
    }

    /// Reconciliation is an observation of the original call, never a new message. Its saved
    /// owner has already checked the destination and origin; stale presentation state cannot
    /// redirect that observation. Working tasks continue through the ordinary follow path.
    pub fn reconcile(
        &mut self,
        response: a2a::StreamResponse,
    ) -> Result<DelegationUpdate, DelegationError> {
        let update = crate::a2a::client::ReplyState::default()
            .accept(response)
            .map_err(|_| DelegationError::InvalidState)?
            .ok_or(DelegationError::InvalidState)?;
        let mut restored = Self::new(self.state.id.clone())?;
        restored.state.question_sequence = self.state.question_sequence;
        let event = restored.update(&update)?;
        self.state = restored.state;
        Ok(match event {
            DelegationUpdate::Pending | DelegationUpdate::Progress(_) => {
                DelegationUpdate::Follow(TaskRef {
                    id: update.task_id,
                    context_id: update.context_id,
                })
            }
            event => event,
        })
    }

    fn identify(&mut self, update: &TaskUpdate) -> Result<(), DelegationError> {
        if self.state.context_id.as_ref().is_some_and(|id| id != &update.context_id)
            || self.state.pending_task_id.as_ref().is_some_and(|id| id != &update.task_id)
        {
            return Err(DelegationError::WrongTask);
        }
        if !update.context_id.is_empty() {
            self.state.context_id = Some(update.context_id.clone());
        }
        if !update.task_id.is_empty() {
            self.state.pending_task_id = Some(update.task_id.clone());
        }
        Ok(())
    }

    pub fn ended(&self) -> Result<(), DelegationError> {
        match self.state.phase {
            InteractionPhase::Settled | InteractionPhase::InputRequired => Ok(()),
            _ => Err(DelegationError::MissingOutcome),
        }
    }
}

pub(crate) fn answer(text: &str, exact: Option<&str>) -> ToolOutcome {
    if let Some(exact) = exact.filter(|s| !s.is_empty() && text.starts_with(s)) {
        let rest = text[exact.len()..].trim();
        ToolOutcome::Value {
            text: exact.into(),
            verbatim: true,
            reply_required: !rest.is_empty(),
            more: (!rest.is_empty()).then(|| rest.to_owned()),
        }
    } else {
        ToolOutcome::Value { text: text.into(), verbatim: false, reply_required: true, more: None }
    }
}

/// The correlation written by the Rust owner to the parent's call, never sent to a peer.
pub fn record_delegation(
    call: &mut lk_agents_abi::FunctionCall,
    id: &InteractionId,
    task: &TaskRef,
    livekit: bool,
) {
    for (key, value) in [
        ("lk.delegation.task_id", task.id.as_str()),
        ("lk.delegation.endpoint", id.endpoint.as_str()),
        ("lk.delegation.transport", "JSONRPC"),
        ("lk.delegation.profile", if livekit { "livekit" } else { "a2a" }),
    ] {
        call.extra.insert(key.into(), value.into());
    }
}

#[cfg(test)]
#[path = "delegation_tests.rs"]
mod tests;

/// Decode a bounded caller chain before admission or outbound delegation.
pub fn partition_chain(
    metadata: &serde_json::Map<String, serde_json::Value>,
) -> Result<
    Vec<lk_agents_abi::a2a::remote::DelegationPartition>,
    lk_agents_abi::a2a::client::AgentClientError,
> {
    use lk_agents_abi::a2a::client::AgentClientError;
    let Some(value) = metadata.get(lk_agents_abi::a2a::metadata::PARTITION_CHAIN_KEY) else {
        return Ok(Vec::new());
    };
    let chain: Vec<lk_agents_abi::a2a::remote::DelegationPartition> =
        serde_json::from_value(value.clone()).map_err(|error| {
            AgentClientError::InvalidInput(format!("invalid delegation partition chain: {error}"))
        })?;
    if chain.len() > 64
        || chain.iter().any(|entry| {
            entry.storage_location.is_empty()
                || entry.thread_id.is_empty()
                || entry.partition_key.is_empty()
        })
    {
        return Err(AgentClientError::InvalidInput("invalid delegation partition chain".into()));
    }
    Ok(chain)
}

/// Different endpoints cannot delegate into a partition held by their active caller chain.
pub fn validate_partition_delegation(
    metadata: &serde_json::Map<String, serde_json::Value>,
    target: &lk_agents_abi::a2a::remote::DelegationPartition,
) -> Result<(), lk_agents_abi::storage::StorageError> {
    let chain = partition_chain(metadata)
        .map_err(|error| lk_agents_abi::storage::StorageError::Invalid(error.to_string()))?;
    if chain.contains(target) {
        return Err(lk_agents_abi::storage::StorageError::Invalid(format!(
            "delegation cannot re-enter active partition_key {:?} in thread {}; use a separate partition_key for the delegated conversation",
            target.partition_key, target.thread_id
        )));
    }
    Ok(())
}
