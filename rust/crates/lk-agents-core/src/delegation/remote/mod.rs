use lk_agents_abi::a2a_protocol as a2a;
pub mod cancel;
pub mod journal;
mod result;
pub mod retained;
pub mod routing;

use a2a::{Message, Part, Role, TaskState};
use lk_agents_abi::a2a::metadata::KIND_KEY;
use lk_agents_abi::llm::tool_executor::ToolOutcome;

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
    #[error("the remote update belongs to another task or context")]
    WrongTask,
    #[error("the local interaction is already running or settled")]
    Settled,
    #[error("the remote task ended without a result")]
    MissingOutcome,
    #[error("the remote interaction or captured input identity is invalid")]
    InvalidState,
    #[error(transparent)]
    History(#[from] crate::a2a::caller_history::HistoryError),
}

/// The original parent call and remote destination, retained independently of presentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InteractionId {
    /// Stable configured remote identity; the endpoint remains independently pinned.
    pub remote_id: String,
    /// Canonical originating FunctionCall.id; provider wire IDs can repeat.
    pub parent_call_id: String,
    pub endpoint: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Local observation lifetime. Remote task status and cancellation belong to task state.
pub enum InteractionPhase {
    Ready,
    Running,
    InputRequired,
    Settled,
}

/// Transient observation state. Durable tasks and exact input occurrences live in the journal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InteractionState {
    pub id: InteractionId,
    pub context_id: Option<String>,
    pub pending_task_id: Option<String>,
    pub phase: InteractionPhase,
    /// The checkpointed input occurrence this presentation asks the person to answer.
    pub input_request: Option<lk_agents_abi::delegation::remote::InputRequestRef>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DelegationUpdate {
    Progress(String),
    /// Exact working output has no business-terminal meaning.
    Exact(String),
    InputRequired {
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

/// How a lost observation is recovered. Reading a known task is safe; resending the
/// operation that created it never is.
#[derive(Debug, Clone, PartialEq)]
pub enum ObservationLoss {
    /// Follow the same task on a fresh stream.
    Resubscribe(TaskRef),
    /// Read the task once: the peer cannot stream it, or refused the resubscription.
    Snapshot(TaskRef),
    /// No task identity was observed, or recovery made no progress within the connection
    /// policy: the outcome stays unknown.
    Unknown,
}

impl ObservationLoss {
    /// A peer that answered the resubscription (refused it, or the task already ended) can
    /// still be read once; a transport failure leaves the outcome unknown.
    pub fn resubscribe_failed(task: TaskRef, error: &crate::a2a::client::ClientError) -> Self {
        match error {
            crate::a2a::client::ClientError::Rpc { .. } => Self::Snapshot(task),
            _ => Self::Unknown,
        }
    }
}

/// One interaction's decision owner. A reusable RemoteAgent holds none of this state.
#[derive(Debug)]
pub struct RemoteInteraction {
    state: InteractionState,
    /// Consecutive lost observations that delivered nothing after their opening snapshot.
    losses: u32,
    /// Updates consumed since the last lost observation.
    since_loss: u32,
}

impl RemoteInteraction {
    pub fn new(id: InteractionId) -> Result<Self, DelegationError> {
        if id.remote_id.trim().is_empty() || id.parent_call_id.is_empty() || id.endpoint.is_empty()
        {
            return Err(DelegationError::InvalidState);
        }
        Ok(Self {
            state: InteractionState {
                id,
                context_id: None,
                pending_task_id: None,
                phase: InteractionPhase::Ready,
                input_request: None,
            },
            losses: 0,
            since_loss: 0,
        })
    }

    pub fn snapshot(&self) -> InteractionState {
        self.state.clone()
    }

    /// Task addressing also supports clarification and correction without an input request.
    /// The conversation owner validates this reference against its saved remote observations.
    pub fn target_task(
        &mut self,
        task: lk_agents_abi::delegation::remote::RemoteTaskRef,
    ) -> Result<(), DelegationError> {
        if self.state.phase != InteractionPhase::Ready
            || task.task_id.is_empty()
            || task.context_id.is_empty()
        {
            return Err(DelegationError::InvalidState);
        }
        self.state.context_id = Some(task.context_id);
        self.state.pending_task_id = Some(task.task_id);
        self.state.input_request = None;
        Ok(())
    }

    /// The conversation owner selected this held response. A replacement agent need not inherit
    /// the original agent's tool history to continue the exact same task and input occurrence.
    pub fn respond_to(
        &mut self,
        reference: lk_agents_abi::session::InputRequestRef,
    ) -> Result<(), DelegationError> {
        let Some((remote_id, task, request_id)) = reference.remote() else {
            return Err(DelegationError::InvalidState);
        };
        if remote_id != self.state.id.remote_id || request_id.is_empty() {
            return Err(DelegationError::InvalidState);
        }
        self.target_task(task.clone())?;
        self.state.input_request = Some(reference);
        Ok(())
    }

    /// Build another message for this live task without replacing its observer or parent call.
    /// Clarifications leave a pending question intact; an answer must carry the exact retained
    /// occurrence. The peer's next observation, rather than this send, changes task status.
    pub fn task_message(
        &self,
        text: String,
        message_id: String,
        input_request: Option<lk_agents_abi::session::InputRequestRef>,
        offers_extension: bool,
    ) -> Result<Message, DelegationError> {
        if self.state.phase == InteractionPhase::Settled {
            return Err(DelegationError::Settled);
        }
        if input_request
            .as_ref()
            .is_some_and(|request| self.state.input_request.as_ref() != Some(request))
        {
            return Err(DelegationError::InvalidState);
        }
        let task = lk_agents_abi::delegation::remote::RemoteTaskRef {
            task_id: self.state.pending_task_id.clone().ok_or(DelegationError::WrongTask)?,
            context_id: self.state.context_id.clone().ok_or(DelegationError::WrongTask)?,
        };
        let mut sender = Self::new(self.state.id.clone())?;
        sender.target_task(task)?;
        sender.message(text, message_id, None, "", None, offers_extension, None, None)
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
        Ok(message)
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
                    owner != self.state.id.remote_id
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
        // A resubscription opens with the task snapshot; only a later update is progress.
        self.since_loss = self.since_loss.saturating_add(1);
        if self.since_loss > 1 {
            self.losses = 0;
        }
        let exact = update.metadata.get("livekit/verbatim").and_then(|v| v.as_str());
        let outcome = match update.state {
            TaskState::Unspecified | TaskState::Submitted | TaskState::Working => {
                if update.state != TaskState::Working || update.text.is_empty() || !update.progress
                {
                    DelegationUpdate::Pending
                } else {
                    match exact {
                        Some(text) => DelegationUpdate::Exact(text.to_owned()),
                        None => DelegationUpdate::Progress(update.text.clone()),
                    }
                }
            }
            TaskState::InputRequired => {
                if update.task_id.is_empty() || update.context_id.is_empty() {
                    return Err(DelegationError::InvalidState);
                }
                self.state.phase = InteractionPhase::InputRequired;
                DelegationUpdate::InputRequired {
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
                DelegationUpdate::Completed(result::completed(update, &self.state.id.remote_id))
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
        let event = restored.update(&update)?;
        self.state = restored.state;
        Ok(match event {
            DelegationUpdate::Pending
            | DelegationUpdate::Progress(_)
            | DelegationUpdate::Exact(_) => DelegationUpdate::Follow(TaskRef {
                id: update.task_id,
                context_id: update.context_id,
            }),
            event => event,
        })
    }

    /// A snapshot read after a lost observation. A task still running cannot be observed
    /// further this way: `None` keeps its outcome unknown rather than polling or resending.
    /// A merged task is followed to its replacement.
    pub fn reconcile_lost(
        &mut self,
        response: a2a::StreamResponse,
    ) -> Result<Option<DelegationUpdate>, DelegationError> {
        let lost = self.state.pending_task_id.clone();
        Ok(match self.reconcile(response)? {
            DelegationUpdate::Follow(task) if lost.as_ref() == Some(&task.id) => None,
            event => Some(event),
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

    /// The observation stream ended or failed before an outcome. A known running task is
    /// read again (`streaming` is the peer's card capability); `max_retry` bounds consecutive
    /// recoveries that made no progress.
    pub fn observation_lost(&mut self, streaming: bool, max_retry: u32) -> ObservationLoss {
        let (Some(id), Some(context_id)) = (&self.state.pending_task_id, &self.state.context_id)
        else {
            return ObservationLoss::Unknown;
        };
        if self.state.phase == InteractionPhase::Settled || self.losses >= max_retry {
            return ObservationLoss::Unknown;
        }
        self.losses += 1;
        self.since_loss = 0;
        let task = TaskRef { id: id.clone(), context_id: context_id.clone() };
        if streaming { ObservationLoss::Resubscribe(task) } else { ObservationLoss::Snapshot(task) }
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
