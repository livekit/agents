use super::effects::TextSessionEffect;
use super::routing;
use crate::agent_run::{self, AgentRun, CallId, Opening, OutputVerdict, RunInput};
use crate::approval;
use crate::usage::Ledger;
use lk_agents_abi::Role;
use lk_agents_abi::agent::ArgumentIdentity;
use lk_agents_abi::agent::run::{
    Output, RunError, RunOutcome, RunResult, StreamEvent, ValidationId,
};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::usage::Usage;
use lk_agents_abi::{ChatItem, ChatMessage, FunctionCall, Ts, WallAnchor};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

mod continuation;
#[cfg(test)]
#[path = "machine_tests.rs"]
mod tests;

pub use lk_agents_abi::text_session::{
    ChildCallStatus, FrameId, ParentCall, PendingChildCall, SessionFrame, TextSessionError,
    TextSessionState, TextSessionTurn,
};
pub use routing::{ChatHistory, LocalAgentTool, RoutingError, SessionAgent, TextSessionConfig};

// Only a settled turn exposes a new state snapshot.

/// Runtime inputs: identities and saved state; no services nor an instruction to load/save them.
pub struct TextSessionInput {
    pub session_id: String,
    /// The random value this request's runs mint their chat items under: drawn afresh
    /// per request, so nothing about it is saved.
    pub id_seed: u64,
    pub execution_id: String,
    pub task_id: Option<String>,
    pub state: Option<TextSessionState>,
    pub wall: WallAnchor,
    pub cleanup_timeout: Duration,
    pub delegated: bool,
}

/// Every event belonging to an invocation carries its identity. Late receipts cannot mutate
/// a new invocation of the same frame, nor a new request reconstructed from the returned state.
pub enum TextSessionEvent {
    PersistenceFailed(lk_agents_abi::storage::StorageError),
    Start(ChatMessage),
    Run {
        invocation_id: String,
        event: agent_run::AgentRunEvent,
    },
    /// The host's accepted value serialized *after* validation/replacement. The runtime also
    /// retains the original typed host value for the public result.
    OutputValidated {
        invocation_id: String,
        validation: ValidationId,
        verdict: OutputVerdict,
        output: Option<Value>,
    },
    DialogueDelivered {
        invocation_id: String,
        message: ChatMessage,
    },
    /// All delivery receipts and invocation cleanup have been reported. A non-presenting
    /// consumer acknowledges immediately; opening a stream does not imply delivery. `delivered`:
    /// the text of each of the invocation's dialogue messages that reached the person, by id.
    InvocationReleased {
        invocation_id: String,
        delivered: BTreeMap<String, String>,
    },
    /// Return from a direct execution of the active frame, preserving its original parent call.
    /// As with AgentRun, the result must already have passed the agent's output validators.
    Reconcile {
        frame_id: FrameId,
        result: RunResult,
        output: Option<Value>,
    },
    Cancel,
    Wake,
}

struct PreparedChild {
    function_call: FunctionCall,
    arguments: Value,
    identity: String,
}

enum Child {
    Preparing(FunctionCall),
    Prepared(PreparedChild),
    Executing(PreparedChild),
}

enum InvocationRelease {
    Settled(Box<RunResult>, Option<Value>, Vec<PendingChildCall>),
    Parked,
}

struct Invocation {
    id: String,
    frame_id: FrameId,
    run: AgentRun,
    children: BTreeMap<CallId, Child>,
    outputs: BTreeMap<ValidationId, Value>,
}

enum Phase {
    NotStarted,
    Running(Invocation),
    Releasing(Invocation, InvocationRelease),
    Done,
}

impl Phase {
    fn invocation(&self) -> Option<&Invocation> {
        match self {
            Self::Running(invocation) | Self::Releasing(invocation, _) => Some(invocation),
            Self::NotStarted | Self::Done => None,
        }
    }

    fn invocation_mut(&mut self) -> Option<&mut Invocation> {
        match self {
            Self::Running(invocation) | Self::Releasing(invocation, _) => Some(invocation),
            Self::NotStarted | Self::Done => None,
        }
    }
}

pub struct TextSession {
    failure_input_requirements: Vec<lk_agents_abi::agent::run::RunInputRequirement>,
    config: TextSessionConfig,
    state: TextSessionState,
    id_seed: u64,
    wall: WallAnchor,
    cleanup_timeout: Duration,
    delegated: bool,
    checkpoints: bool,
    persistence_failure: Option<lk_agents_abi::storage::StorageError>,
    recording: bool,
    checkpoint_due: Option<Ts>,
    checkpoint_bytes: usize,
    phase: Phase,
    /// The turn's ended invocations: the live one is added to it when read.
    usage: Ledger,
    sequence: u64,
}

impl TextSession {
    pub fn new(config: TextSessionConfig, input: TextSessionInput) -> Result<Self, RoutingError> {
        config.validate()?;
        let state = match input.state {
            Some(mut state) => {
                if state.session_id != input.session_id {
                    return Err(RoutingError::InvalidState("session identity changed".into()));
                }
                let frame = state
                    .frames
                    .get(&state.active_frame_id)
                    .ok_or_else(|| RoutingError::InvalidState("missing active frame".into()))?;
                let pending = frame.parent.is_some()
                    || !frame.pending_calls.is_empty()
                    || agent_run::execution::pending(&frame.history).is_some();
                // A reply under a new task resumes the pending frame under that task; the door
                // completes the waiting task as merged into it.
                match input.task_id {
                    Some(task_id) => reidentify(&mut state, task_id),
                    None if !pending => state.execution_id = input.execution_id,
                    None => {}
                }
                state
            }
            None => {
                let root = FrameId { value: format!("{}/frame_0", input.session_id) };
                let frame = SessionFrame {
                    id: root.clone(),
                    agent_name: config.root_agent.clone(),
                    history: Vec::new(),
                    parent: None,
                    completed: false,
                    pending_calls: Vec::new(),
                    answer_tool: None,
                    visible_dialogue: Vec::new(),
                    starting_history_ids: BTreeSet::new(),
                    transferred_message_ids: BTreeSet::new(),
                    merge_chat_history: false,
                };
                TextSessionState {
                    session_id: input.session_id,
                    execution_id: input.task_id.unwrap_or(input.execution_id),
                    root_frame_id: root.clone(),
                    active_frame_id: root.clone(),
                    ancestors: Vec::new(),
                    frames: BTreeMap::from([(root, frame)]),
                    next_id: 0,
                }
            }
        };
        config.validate_state(&state)?;
        Ok(Self {
            failure_input_requirements: Vec::new(),
            config,
            state,
            id_seed: input.id_seed,
            wall: input.wall,
            cleanup_timeout: input.cleanup_timeout,
            delegated: input.delegated,
            checkpoints: false,
            persistence_failure: None,
            recording: false,
            checkpoint_due: None,
            checkpoint_bytes: 0,
            phase: Phase::NotStarted,
            usage: Ledger::default(),
            sequence: 0,
        })
    }

    /// Install the managed-turn admission and delivery capability for every frame.
    pub fn with_checkpoints(mut self) -> Self {
        self.checkpoints = true;
        self
    }

    fn mint(&mut self, kind: &str) -> String {
        self.state.next_id += 1;
        format!("{}/{kind}_{}", self.state.session_id, self.state.next_id)
    }

    pub fn with_persistence(mut self) -> Self {
        self.recording = true;
        self
    }

    pub fn wake_at(&self) -> Option<Ts> {
        match (self.phase.invocation().and_then(|i| i.run.wake_at()), self.checkpoint_due) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    pub fn handle(&mut self, event: TextSessionEvent, now: Ts) -> Vec<TextSessionEffect> {
        let mut effects = Vec::new();
        // A late host receipt still transfers ownership of its prepared-arguments resource.
        // Invocation identity suffices for runtime cleanup, even after this request settled.
        if let TextSessionEvent::Run {
            invocation_id,
            event: agent_run::AgentRunEvent::ToolPrepared { prepared, .. },
        } = &event
            && !self.is_current(invocation_id)
        {
            effects.push(TextSessionEffect::Run {
                invocation_id: invocation_id.clone(),
                effect: agent_run::AgentRunEffect::ToolArgumentsRelease { prepared: *prepared },
            });
            return effects;
        }
        if matches!(self.phase, Phase::Done) {
            return effects;
        }
        match event {
            TextSessionEvent::PersistenceFailed(error) => {
                self.persistence_failure = Some(error);
                return self.handle(TextSessionEvent::Cancel, now);
            }
            TextSessionEvent::Start(message) if matches!(self.phase, Phase::NotStarted) => {
                let frame = self.active_frame_mut();
                record_delivered(frame, message.clone());
                // An answered read-back is now ordinary history. AgentRun alone decides whether
                // a repeated validated call consumes it; a model argument cannot grant approval.
                frame.pending_calls.retain(|p| p.status != ChildCallStatus::Approval);
                self.invoke(Some(message), now, &mut effects);
            }
            TextSessionEvent::Run { invocation_id, event } if self.is_current(&invocation_id) => {
                self.run_event(event, now, &mut effects);
            }
            TextSessionEvent::OutputValidated { invocation_id, validation, verdict, output }
                if self.is_current(&invocation_id) =>
            {
                if matches!(verdict, OutputVerdict::Accepted { .. })
                    && let Some(output) = output
                    && let Some(invocation) = self.phase.invocation_mut()
                {
                    invocation.outputs.insert(validation, output);
                }
                self.run_event(
                    agent_run::AgentRunEvent::OutputValidated { validation, verdict },
                    now,
                    &mut effects,
                );
            }
            TextSessionEvent::DialogueDelivered { invocation_id, message }
                if self.is_current(&invocation_id) =>
            {
                let frame = self.active_frame_mut();
                record_delivered(frame, message);
            }
            TextSessionEvent::InvocationReleased { invocation_id, delivered } if matches!(&self.phase, Phase::Releasing(invocation, _) if invocation.id == invocation_id) =>
            {
                self.receipts(&delivered);
                let phase = std::mem::replace(&mut self.phase, Phase::Done);
                let Phase::Releasing(invocation, release) = phase else { return effects };
                self.usage.absorb(invocation.run.ledger());
                match release {
                    InvocationRelease::Settled(result, output, approvals) => {
                        self.settled(*result, output, approvals, now, &mut effects);
                    }
                    InvocationRelease::Parked => {
                        self.next_child(now, &mut effects);
                    }
                }
            }
            TextSessionEvent::Reconcile { frame_id, result, output }
                if matches!(self.phase, Phase::NotStarted)
                    && frame_id == self.state.active_frame_id =>
            {
                self.usage.add(&result.usage);
                self.settled(result, output, Vec::new(), now, &mut effects);
            }
            TextSessionEvent::Cancel => {
                if matches!(self.phase, Phase::Releasing(..)) {
                    self.end_invocation();
                    self.finish(Err(TextSessionError::Run(RunError::Cancelled)), &mut effects);
                } else if matches!(self.phase, Phase::Running(_)) {
                    self.run_event(agent_run::AgentRunEvent::Cancel, now, &mut effects);
                } else {
                    self.finish(Err(TextSessionError::Run(RunError::Cancelled)), &mut effects);
                }
            }
            TextSessionEvent::Wake => {
                self.run_event(agent_run::AgentRunEvent::Wake, now, &mut effects)
            }
            _ => {}
        }
        for effect in &effects {
            if !self.recording {
                break;
            }
            if let TextSessionEffect::Emit {
                event: lk_agents_abi::agent::run::StreamEvent::TextDelta { delta, .. },
                ..
            } = effect
            {
                self.checkpoint_bytes = self.checkpoint_bytes.saturating_add(delta.len());
                self.checkpoint_due
                    .get_or_insert(now.saturating_add(std::time::Duration::from_millis(250)));
            }
        }
        let flush_due =
            self.checkpoint_bytes >= 64 * 1024 || self.checkpoint_due.is_some_and(|due| due <= now);
        let failed = effects
            .iter()
            .any(|effect| matches!(effect, TextSessionEffect::Finished { result: Err(_), .. }));
        if failed {
            self.checkpoint_due = None;
            self.checkpoint_bytes = 0;
        }
        if self.recording
            && !failed
            && (flush_due
                || effects.iter().any(|effect| {
                    matches!(
                        effect,
                        TextSessionEffect::InvocationStarted { .. }
                            | TextSessionEffect::InvocationParked { .. }
                            | TextSessionEffect::InvocationSettled { .. }
                            | TextSessionEffect::Finished { result: Ok(_), .. }
                            | TextSessionEffect::Run {
                                effect: agent_run::AgentRunEffect::ToolExecute { .. }
                                    | agent_run::AgentRunEffect::HostCheckpoint { .. },
                                ..
                            }
                    )
                }))
            && self.persistence_failure.is_none()
        {
            self.checkpoint_due = None;
            self.checkpoint_bytes = 0;
            let mut state = self.state.clone();
            if let Some(invocation) = self.phase.invocation() {
                state.frames.get_mut(&invocation.frame_id).expect("invoked frame exists").history =
                    invocation.run.progress_history();
            }
            effects.insert(0, TextSessionEffect::Checkpoint(Box::new(state)));
        }
        effects
    }

    fn is_current(&self, invocation_id: &str) -> bool {
        self.phase.invocation().is_some_and(|i| i.id == invocation_id)
    }

    /// `new` checks the active frame exists and every transition keeps it.
    fn active_frame_mut(&mut self) -> &mut SessionFrame {
        self.state.frames.get_mut(&self.state.active_frame_id).expect("the active frame exists")
    }

    fn invoke(
        &mut self,
        message: Option<ChatMessage>,
        now: Ts,
        effects: &mut Vec<TextSessionEffect>,
    ) {
        let invocation_id = self.mint("invocation");
        let frame = &self.state.frames[&self.state.active_frame_id];
        let agent = &self.config.agents[&frame.agent_name];
        let resume = message.is_none();
        let mut message = message.unwrap_or_else(|| {
            ChatMessage::text(String::new(), lk_agents_abi::Role::User, "", std::time::UNIX_EPOCH)
        });
        agent_run::execution::bind(
            &mut message,
            &frame.history,
            Some(&self.state.execution_id),
            &invocation_id,
        );
        let input = RunInput {
            invocation_id: invocation_id.clone(),
            id_seed: lk_agents_abi::types::ids::splitmix64(self.id_seed, self.state.next_id),
            opening: Opening::Message(Box::new(message)),
            history: frame.history.clone(),
            answer_tool: frame.answer_tool.clone(),
            delegated: self.delegated,
            mode: crate::tool_outcome::Mode::Text,
            llm: agent.llm.clone(),
            wall: self.wall,
            cleanup_timeout: self.cleanup_timeout,
        };
        let history = if resume {
            input.history.clone()
        } else {
            let mut history = input.history.clone();
            history.extend(input.opening.message().cloned().map(ChatItem::Message));
            history
        };
        let mut run = match AgentRun::new(agent.definition.clone(), input) {
            Ok(run) => {
                if self.checkpoints {
                    run.with_checkpoints()
                } else {
                    run
                }
            }
            Err(error) => {
                self.finish(Err(TextSessionError::Routing(RoutingError::Agent(error))), effects);
                return;
            }
        };
        effects.push(TextSessionEffect::InvocationStarted {
            frame_id: frame.id.clone(),
            agent_name: frame.agent_name.clone(),
            invocation_id: invocation_id.clone(),
            execution_id: self.state.execution_id.clone(),
            history,
        });
        let run_effects = if resume {
            run.continue_from_history(now)
        } else {
            run.handle(agent_run::AgentRunEvent::Start, now)
        };
        self.phase = Phase::Running(Invocation {
            id: invocation_id,
            frame_id: frame.id.clone(),
            run,
            children: BTreeMap::new(),
            outputs: BTreeMap::new(),
        });
        self.run_effects(run_effects, now, effects);
    }

    fn run_event(
        &mut self,
        mut event: agent_run::AgentRunEvent,
        now: Ts,
        effects: &mut Vec<TextSessionEffect>,
    ) {
        let Some(invocation) = self.phase.invocation_mut() else {
            return;
        };
        if let agent_run::AgentRunEvent::ToolPrepared { call, prepared, identity } = &event
            && let Some(child) = invocation.children.get_mut(call)
            && let Child::Preparing(function_call) = child
        {
            match accept_child_arguments(function_call, identity) {
                Ok(prepared) => *child = Child::Prepared(prepared),
                Err(failure) => {
                    let effect =
                        agent_run::AgentRunEffect::ToolArgumentsRelease { prepared: *prepared };
                    effects.push(TextSessionEffect::Run {
                        invocation_id: invocation.id.clone(),
                        effect,
                    });
                    event = agent_run::AgentRunEvent::ToolPrepareFailed { call: *call, failure };
                }
            }
        }
        let run_effects = invocation.run.handle(event, now);
        self.run_effects(run_effects, now, effects);
    }

    fn run_effects(
        &mut self,
        run_effects: Vec<agent_run::AgentRunEffect>,
        now: Ts,
        effects: &mut Vec<TextSessionEffect>,
    ) {
        for effect in run_effects {
            let Some(invocation) = self.phase.invocation() else {
                break;
            };
            let frame_id = invocation.frame_id.clone();
            let invocation_id = invocation.id.clone();
            match effect {
                agent_run::AgentRunEffect::Emit(event) => {
                    self.emit(frame_id, invocation_id, event, effects)
                }
                agent_run::AgentRunEffect::ToolPrepare { call, ref function_call, .. } => {
                    self.prepare_child(call, function_call);
                    effects.push(TextSessionEffect::Run { invocation_id, effect });
                }
                agent_run::AgentRunEffect::ToolExecute {
                    call, prepared, function_call, ..
                } if invocation.children.contains_key(&call) => {
                    self.execute_child(call, function_call, invocation_id, prepared, effects)
                }
                agent_run::AgentRunEffect::ToolCancel { call }
                    if matches!(invocation.children.get(&call), Some(Child::Executing(_))) =>
                {
                    self.cancel_child(call, now, effects)
                }
                agent_run::AgentRunEffect::Finished { result, input_requirements } => self
                    .invocation_finished(
                        frame_id,
                        invocation_id,
                        *result,
                        input_requirements,
                        effects,
                    ),
                other => effects.push(TextSessionEffect::Run { invocation_id, effect: other }),
            }
        }
        self.park(now, effects);
    }

    fn emit(
        &mut self,
        frame_id: FrameId,
        invocation_id: String,
        event: StreamEvent,
        effects: &mut Vec<TextSessionEffect>,
    ) {
        self.sequence += 1;
        let agent_name = self.state.frames[&frame_id].agent_name.clone();
        effects.push(TextSessionEffect::Emit {
            sequence: self.sequence,
            frame_id,
            invocation_id,
            agent_name,
            event,
        });
    }

    /// A call of an agent tool becomes a child the session runs itself.
    fn prepare_child(&mut self, call: CallId, function_call: &FunctionCall) {
        let Some(invocation) = self.phase.invocation_mut() else {
            return;
        };
        let agent_name = &self.state.frames[&invocation.frame_id].agent_name;
        if !self.config.agents[agent_name].agent_tools.contains_key(&function_call.name) {
            return;
        }
        invocation.children.insert(call, Child::Preparing(function_call.clone()));
    }

    fn execute_child(
        &mut self,
        call: CallId,
        function_call: FunctionCall,
        invocation_id: String,
        prepared: ResourceId,
        effects: &mut Vec<TextSessionEffect>,
    ) {
        let Some(invocation) = self.phase.invocation_mut() else { return };
        let Some(child) = invocation.children.remove(&call) else { return };
        let child = match child {
            Child::Prepared(mut child) | Child::Executing(mut child) => {
                child.function_call = function_call;
                Child::Executing(child)
            }
            child @ Child::Preparing(_) => {
                invocation.children.insert(call, child);
                return;
            }
        };
        invocation.children.insert(call, child);
        effects.push(TextSessionEffect::Run {
            invocation_id,
            effect: agent_run::AgentRunEffect::ToolArgumentsRelease { prepared },
        });
    }

    fn cancel_child(&mut self, call: CallId, now: Ts, effects: &mut Vec<TextSessionEffect>) {
        let Some(invocation) = self.phase.invocation_mut() else { return };
        let Some(Child::Executing(child)) = invocation.children.remove(&call) else { return };
        invocation.children.insert(call, Child::Prepared(child));
        let settled = invocation.run.handle(
            agent_run::AgentRunEvent::ToolSettled {
                call,
                outcome: lk_agents_abi::llm::tool_executor::ToolOutcome::Cancelled,
            },
            now,
        );
        self.run_effects(settled, now, effects);
    }

    fn invocation_finished(
        &mut self,
        frame_id: FrameId,
        invocation_id: String,
        result: Result<RunResult, RunError>,
        input_requirements: Vec<lk_agents_abi::agent::run::RunInputRequirement>,
        effects: &mut Vec<TextSessionEffect>,
    ) {
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                self.failure_input_requirements = input_requirements;
                return self.finish(Err(TextSessionError::Run(error)), effects);
            }
        };
        let phase = std::mem::replace(&mut self.phase, Phase::Done);
        let (Phase::Running(invocation) | Phase::Releasing(invocation, _)) = phase else { return };
        let output = validated_output(&invocation, &result.outcome);
        let approvals = self.approvals(&invocation, &result);
        effects.push(TextSessionEffect::InvocationSettled {
            frame_id,
            invocation_id,
            result: result.clone(),
        });
        self.phase = Phase::Releasing(
            invocation,
            InvocationRelease::Settled(Box::new(result), output, approvals),
        );
    }

    fn pending(
        &self,
        frame_id: &FrameId,
        invocation_id: &str,
        child: &PreparedChild,
        status: ChildCallStatus,
    ) -> PendingChildCall {
        let frame = &self.state.frames[frame_id];
        let tool = &self.config.agents[&frame.agent_name].agent_tools[&child.function_call.name];
        PendingChildCall {
            parent: ParentCall {
                frame_id: frame_id.clone(),
                call_id: child.function_call.call_id.clone(),
            },
            invocation_id: invocation_id.into(),
            execution_id: self.state.execution_id.clone(),
            tool_name: child.function_call.name.clone(),
            agent_name: tool.agent_name.clone(),
            validated_arguments: child.arguments.clone(),
            identity: child.identity.clone(),
            status,
        }
    }

    fn approvals(&self, invocation: &Invocation, result: &RunResult) -> Vec<PendingChildCall> {
        invocation
            .children
            .values()
            .filter_map(|child| match child {
                Child::Prepared(child) | Child::Executing(child) => Some(child),
                Child::Preparing(_) => None,
            })
            .filter_map(|child| {
                result.all_messages.iter().find_map(|item| match item {
                    ChatItem::FunctionCall(call)
                        if call.call_id == child.function_call.call_id
                            && approval::mark(call)
                                == Some(lk_agents_abi::llm::tool_executor::Approval::Asked) =>
                    {
                        Some(self.pending(
                            &invocation.frame_id,
                            &invocation.id,
                            child,
                            ChildCallStatus::Approval,
                        ))
                    }
                    _ => None,
                })
            })
            .collect()
    }

    fn park(&mut self, now: Ts, effects: &mut Vec<TextSessionEffect>) {
        let Phase::Running(invocation) = &mut self.phase else {
            return;
        };
        let calls: Vec<_> = invocation
            .children
            .iter()
            .filter(|(_, c)| matches!(c, Child::Executing(_)))
            .map(|(id, _)| *id)
            .collect();
        let history = match invocation.run.delegation_checkpoint(&calls, now) {
            agent_run::DelegationCheckpoint::Waiting => return,
            agent_run::DelegationCheckpoint::ConflictingOutcome => {
                for call in &calls {
                    if let Some(Child::Executing(child)) = invocation.children.remove(call) {
                        invocation.children.insert(*call, Child::Prepared(child));
                    }
                }
                for call in calls {
                    self.run_event(
                        agent_run::AgentRunEvent::ToolSettled {
                            call,
                            outcome: lk_agents_abi::llm::tool_executor::ToolOutcome::Rejected(
                                "child was not started because a sibling ended this invocation"
                                    .into(),
                            ),
                        },
                        now,
                        effects,
                    );
                }
                return;
            }
            agent_run::DelegationCheckpoint::Ready { history } => history,
        };
        let Phase::Running(invocation) = &self.phase else { return };
        let mut pending: Vec<_> = invocation
            .children
            .values()
            .filter_map(|child| {
                let Child::Executing(child) = child else { return None };
                Some(self.pending(
                    &invocation.frame_id,
                    &invocation.id,
                    child,
                    ChildCallStatus::Ready,
                ))
            })
            .collect();
        // Validation/launch arrival order must not change the model's sibling order.
        pending.sort_by_key(|p| {
            history
                .iter()
                .position(|i| {
                    matches!(i,
            ChatItem::FunctionCall(c) if c.call_id == p.parent.call_id)
                })
                .unwrap_or(usize::MAX)
        });
        let frame =
            self.state.frames.get_mut(&invocation.frame_id).expect("the invoked frame exists");
        frame.history = history;
        frame.pending_calls = pending;
        effects.push(TextSessionEffect::InvocationParked {
            frame_id: invocation.frame_id.clone(),
            invocation_id: invocation.id.clone(),
        });
        if let Phase::Running(invocation) = std::mem::replace(&mut self.phase, Phase::Done) {
            self.phase = Phase::Releasing(invocation, InvocationRelease::Parked);
        }
    }

    /// The released invocation's dialogue as the person received it: a message cut short reads
    /// as interrupted, with the text that reached them.
    fn receipts(&mut self, delivered: &BTreeMap<String, String>) {
        let Phase::Releasing(invocation, release) = &self.phase else { return };
        let items = match release {
            InvocationRelease::Settled(result, ..) => &result.new_messages,
            InvocationRelease::Parked => &self.state.frames[&invocation.frame_id].history,
        };
        let received: Vec<ChatMessage> = (items.iter())
            .filter_map(|item| {
                let ChatItem::Message(message) = item else { return None };
                let text = delivered.get(&message.id)?;
                let mut message = message.clone();
                message.interrupted |= message.raw_text_content().as_deref() != Some(text.as_str());
                message.content = vec![lk_agents_abi::ChatContent::Text(text.clone())];
                Some(message)
            })
            .collect();
        let frame = self.active_frame_mut();
        for message in received {
            record_delivered(frame, message);
        }
    }

    /// The latest snapshot of the turn's usage: every local invocation it ran, once.
    pub fn usage(&self) -> Usage {
        match self.phase.invocation() {
            Some(invocation) => {
                let mut usage = self.usage.clone();
                usage.absorb(invocation.run.ledger());
                usage.snapshot()
            }
            None => self.usage.snapshot(),
        }
    }

    /// Changes whenever [`usage`](Self::usage) may have: ended invocations, then the live one.
    pub fn usage_revision(&self) -> (u64, u64) {
        (self.usage.revision(), self.phase.invocation().map_or(0, |i| i.run.ledger().revision()))
    }

    fn end_invocation(&mut self) {
        let phase = std::mem::replace(&mut self.phase, Phase::Done);
        if let Phase::Running(invocation) | Phase::Releasing(invocation, _) = phase {
            self.usage.absorb(invocation.run.ledger());
        }
    }

    fn finish(
        &mut self,
        result: Result<TextSessionTurn, TextSessionError>,
        effects: &mut Vec<TextSessionEffect>,
    ) {
        let mut result = match self.persistence_failure.take() {
            Some(error) => Err(TextSessionError::Storage(error)),
            None => result,
        };
        let invocation_id = match &result {
            Err(TextSessionError::Run(_)) => {
                self.phase.invocation().map(|invocation| invocation.id.clone())
            }
            _ => None,
        };
        self.end_invocation();
        self.usage.settle();
        if let Ok(turn) = &mut result {
            turn.result.usage = self.usage.snapshot();
        }
        effects.push(TextSessionEffect::Finished {
            invocation_id,
            result,
            input_requirements: std::mem::take(&mut self.failure_input_requirements),
        });
    }
}

/// The pending execution continues under `execution_id`: its waiting child calls move with it.
fn reidentify(state: &mut TextSessionState, execution_id: String) {
    for frame in state.frames.values_mut() {
        for pending in &mut frame.pending_calls {
            if pending.execution_id == state.execution_id {
                pending.execution_id = execution_id.clone();
            }
        }
    }
    state.execution_id = execution_id;
}

/// Records a child's validated arguments when they are a JSON object, else refuses them.
fn accept_child_arguments(
    function_call: &FunctionCall,
    identity: &ArgumentIdentity,
) -> Result<PreparedChild, crate::tool_calls::PrepareFailure> {
    match identity {
        ArgumentIdentity::Json(value) if value.is_object() => Ok(PreparedChild {
            function_call: function_call.clone(),
            arguments: value.clone(),
            identity: approval::canonical(identity),
        }),
        _ => Err(crate::tool_calls::PrepareFailure::Refused(
            "local agent arguments must have a serializable validated JSON object".into(),
        )),
    }
}

/// The output a completed run hands its parent: the validated value or the text.
fn validated_output(invocation: &Invocation, outcome: &RunOutcome) -> Option<Value> {
    match outcome {
        RunOutcome::Completed { output: Output::Validated(id) } => {
            invocation.outputs.get(id).cloned()
        }
        RunOutcome::Completed { output: Output::Text(text) } => Some(Value::String(text.clone())),
        _ => None,
    }
}

fn record_delivered(frame: &mut SessionFrame, message: ChatMessage) {
    if !matches!(message.role, Role::User | Role::Assistant)
        || frame.starting_history_ids.contains(&message.id)
        || frame.visible_dialogue.iter().any(|m| m.id == message.id)
    {
        return;
    }
    frame.visible_dialogue.push(message);
}

fn transfer(child: &mut SessionFrame, parent: &mut SessionFrame) {
    if !child.merge_chat_history {
        return;
    }
    for message in &child.visible_dialogue {
        if child.transferred_message_ids.insert(message.id.clone()) {
            if !parent.history.iter().any(|item| item.id() == message.id) {
                parent.history.push(ChatItem::Message(message.clone()));
            }
            record_delivered(parent, message.clone());
        }
    }
}
