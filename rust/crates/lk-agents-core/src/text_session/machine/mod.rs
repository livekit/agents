//! Request-scoped backend conversation coordination (text-session.md). `state` holds frames;
//! `invocation` composes one AgentRun at a time. Runtime executes its effects and supplies
//! validation and delivery receipts. Only a settled turn exposes a new state snapshot.
//!
//! - `state`: owned routing records for explicit application persistence.
//! - `routing`: definitions, context-scope references and restored-state validation.
//! - Private history helpers: one-time fresh/fork selection and delivered-dialogue transfer.
//! - `continuation`: child admission, completion and fresh parent invocations.

use super::effects::TextSessionEffect;
use super::routing;
use crate::agent_run::{self, AgentRun, CallId, Opening, OutputVerdict, RunInput};
use crate::approval;
use crate::llm::history::ChatContextExt;
use crate::usage::Ledger;
use lk_agents_abi::agent::ArgumentIdentity;
use lk_agents_abi::agent::run::{
    Output, RunError, RunOutcome, RunResult, StreamEvent, ValidationId,
};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::usage::Usage;
use lk_agents_abi::{ChatContext, CopyOptions, Role};
use lk_agents_abi::{ChatItem, ChatMessage, FunctionCall, Ts, WallAnchor};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

mod continuation;
#[cfg(test)]
#[path = "../../text_session_tests.rs"]
mod tests;

pub use lk_agents_abi::text_session::{
    ChildCallStatus, FrameId, ParentCall, PendingChildCall, SessionFrame, TextSessionError,
    TextSessionState, TextSessionTurn,
};
pub use routing::{ChatHistory, LocalAgentTool, RoutingError, SessionAgent, TextSessionConfig};

/// Runtime/application inputs. Scope references describe contexts the application has built;
/// this record contains neither services nor an instruction to load/save them.
pub struct TextSessionInput {
    pub session_id: String,
    pub execution_id: String,
    pub task_id: Option<String>,
    pub root_scope: String,
    pub available_scopes: BTreeSet<String>,
    pub separate_contexts: bool,
    pub state: Option<TextSessionState>,
    pub wall: WallAnchor,
    pub cleanup_timeout: Duration,
    pub delegated: bool,
}

/// Every event belonging to an invocation carries its identity. Late receipts cannot mutate
/// a new invocation of the same frame, nor a new request reconstructed from the returned state.
pub enum TextSessionEvent {
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
    arguments: Option<Value>,
    identity: Option<String>,
    executing: bool,
}

enum InvocationRelease {
    Settled(Box<RunResult>, Option<Value>, Vec<PendingChildCall>),
    Parked,
}

struct Invocation {
    id: String,
    frame_id: FrameId,
    run: AgentRun,
    children: BTreeMap<CallId, PreparedChild>,
    outputs: BTreeMap<ValidationId, Value>,
    release: Option<InvocationRelease>,
}

pub struct TextSession {
    failure_input_requirements: Vec<lk_agents_abi::agent::run::RunInputRequirement>,
    config: TextSessionConfig,
    state: TextSessionState,
    scopes: BTreeSet<String>,
    separate_contexts: bool,
    wall: WallAnchor,
    cleanup_timeout: Duration,
    delegated: bool,
    checkpoints: bool,
    invocation: Option<Invocation>,
    /// The turn's ended invocations: the live one is added to it when read.
    usage: Ledger,
    started: bool,
    done: bool,
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
                match &input.task_id {
                    Some(task_id) if pending && *task_id != state.execution_id => {
                        return Err(RoutingError::InvalidState(
                            "execution identity changed while awaiting input".into(),
                        ));
                    }
                    Some(task_id) => state.execution_id = task_id.clone(),
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
                    context_scope: input.root_scope,
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
        let mut scopes = input.available_scopes;
        if input.separate_contexts {
            scopes.extend(
                state
                    .frames
                    .values()
                    .filter(|&frame| {
                        frame.parent.is_some() && frame.context_scope == frame.id.value
                    })
                    .map(|frame| frame.context_scope.clone()),
            );
        }
        config.validate_state(&state, &scopes)?;
        Ok(Self {
            failure_input_requirements: Vec::new(),
            config,
            state,
            scopes,
            separate_contexts: input.separate_contexts,
            wall: input.wall,
            cleanup_timeout: input.cleanup_timeout,
            delegated: input.delegated,
            checkpoints: false,
            invocation: None,
            usage: Ledger::default(),
            started: false,
            done: false,
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

    pub fn wake_at(&self) -> Option<Ts> {
        self.invocation.as_ref().and_then(|i| i.run.wake_at())
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
        if self.done {
            return effects;
        }
        match event {
            TextSessionEvent::Start(message) if !self.started => {
                self.started = true;
                let frame = self.state.frames.get_mut(&self.state.active_frame_id).unwrap();
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
                {
                    self.invocation.as_mut().unwrap().outputs.insert(validation, output);
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
                let frame = self.state.frames.get_mut(&self.state.active_frame_id).unwrap();
                record_delivered(frame, message);
            }
            TextSessionEvent::InvocationReleased { invocation_id, delivered }
                if self.is_current(&invocation_id) =>
            {
                self.receipts(&delivered);
                match self.invocation.as_mut().unwrap().release.take() {
                    Some(InvocationRelease::Settled(result, output, approvals)) => {
                        self.end_invocation();
                        self.settled(*result, output, approvals, now, &mut effects);
                    }
                    Some(InvocationRelease::Parked) => {
                        self.end_invocation();
                        self.next_child(now, &mut effects);
                    }
                    None => {}
                }
            }
            TextSessionEvent::Reconcile { frame_id, result, output }
                if !self.started && frame_id == self.state.active_frame_id =>
            {
                self.started = true;
                self.usage.add(&result.usage);
                self.settled(result, output, Vec::new(), now, &mut effects);
            }
            TextSessionEvent::Cancel => {
                if self.invocation.as_ref().is_some_and(|i| i.release.is_some()) {
                    self.end_invocation();
                    self.finish(Err(TextSessionError::Run(RunError::Cancelled)), &mut effects);
                } else if self.invocation.is_some() {
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
        effects
    }

    fn is_current(&self, invocation_id: &str) -> bool {
        self.invocation.as_ref().is_some_and(|i| i.id == invocation_id)
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
                self.finish(
                    Err(TextSessionError::Routing(RoutingError::InvalidState(format!(
                        "{error:?}"
                    )))),
                    effects,
                );
                return;
            }
        };
        effects.push(TextSessionEffect::InvocationStarted {
            frame_id: frame.id.clone(),
            agent_name: frame.agent_name.clone(),
            invocation_id: invocation_id.clone(),
            context_scope: frame.context_scope.clone(),
            reads_context: !(self.separate_contexts && frame.context_scope == frame.id.value),
            execution_id: self.state.execution_id.clone(),
            history,
        });
        let run_effects = if resume {
            run.continue_from_history(now)
        } else {
            run.handle(agent_run::AgentRunEvent::Start, now)
        };
        self.invocation = Some(Invocation {
            id: invocation_id,
            frame_id: frame.id.clone(),
            run,
            children: BTreeMap::new(),
            outputs: BTreeMap::new(),
            release: None,
        });
        self.run_effects(run_effects, now, effects);
    }

    fn run_event(
        &mut self,
        mut event: agent_run::AgentRunEvent,
        now: Ts,
        effects: &mut Vec<TextSessionEffect>,
    ) {
        let Some(invocation) = self.invocation.as_mut() else {
            return;
        };
        if let agent_run::AgentRunEvent::ToolPrepared { call, prepared, identity } = &event
            && let Some(child) = invocation.children.get_mut(call)
            && child.identity.is_none()
            && let Err(failure) = accept_child_arguments(child, identity)
        {
            effects.push(TextSessionEffect::Run {
                invocation_id: invocation.id.clone(),
                effect: agent_run::AgentRunEffect::ToolArgumentsRelease { prepared: *prepared },
            });
            event = agent_run::AgentRunEvent::ToolPrepareFailed { call: *call, failure };
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
            let Some(invocation) = self.invocation.as_ref() else {
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
                    if invocation.children.get(&call).is_some_and(|c| c.executing) =>
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
        let Some(invocation) = self.invocation.as_mut() else {
            return;
        };
        let agent_name = &self.state.frames[&invocation.frame_id].agent_name;
        if !self.config.agents[agent_name].agent_tools.contains_key(&function_call.name) {
            return;
        }
        let child = PreparedChild {
            function_call: function_call.clone(),
            arguments: None,
            identity: None,
            executing: false,
        };
        invocation.children.insert(call, child);
    }

    fn execute_child(
        &mut self,
        call: CallId,
        function_call: FunctionCall,
        invocation_id: String,
        prepared: ResourceId,
        effects: &mut Vec<TextSessionEffect>,
    ) {
        let child = self
            .invocation
            .as_mut()
            .and_then(|invocation| invocation.children.get_mut(&call))
            .expect("an executing child was prepared in the current invocation");
        child.function_call = function_call;
        child.executing = true;
        effects.push(TextSessionEffect::Run {
            invocation_id,
            effect: agent_run::AgentRunEffect::ToolArgumentsRelease { prepared },
        });
    }

    fn cancel_child(&mut self, call: CallId, now: Ts, effects: &mut Vec<TextSessionEffect>) {
        let invocation =
            self.invocation.as_mut().expect("a canceled child runs in the current invocation");
        invocation
            .children
            .get_mut(&call)
            .expect("a canceled child was prepared in the current invocation")
            .executing = false;
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
        let invocation = self.invocation.as_ref().expect("a run finishes in its invocation");
        let output = validated_output(invocation, &result.outcome);
        let approvals = self.approvals(invocation, &result);
        effects.push(TextSessionEffect::InvocationSettled {
            frame_id,
            invocation_id,
            result: result.clone(),
        });
        self.invocation.as_mut().expect("a run finishes in its invocation").release =
            Some(InvocationRelease::Settled(Box::new(result), output, approvals));
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
            validated_arguments: child.arguments.clone().unwrap(),
            identity: child.identity.clone().unwrap(),
            status,
        }
    }

    fn approvals(&self, invocation: &Invocation, result: &RunResult) -> Vec<PendingChildCall> {
        invocation
            .children
            .values()
            .filter(|child| child.arguments.is_some())
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
        let Some(invocation) = &self.invocation else {
            return;
        };
        if invocation.release.is_some() {
            return;
        }
        let calls: Vec<_> =
            invocation.children.iter().filter(|(_, c)| c.executing).map(|(id, _)| *id).collect();
        let history = match invocation.run.delegation_checkpoint(&calls, now) {
            agent_run::DelegationCheckpoint::Waiting => return,
            agent_run::DelegationCheckpoint::ConflictingOutcome => {
                for id in &calls {
                    self.invocation.as_mut().unwrap().children.get_mut(id).unwrap().executing =
                        false;
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
        let mut pending: Vec<_> = calls
            .iter()
            .map(|id| {
                self.pending(
                    &invocation.frame_id,
                    &invocation.id,
                    &invocation.children[id],
                    ChildCallStatus::Ready,
                )
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
                .unwrap()
        });
        let frame = self.state.frames.get_mut(&invocation.frame_id).unwrap();
        frame.history = history;
        frame.pending_calls = pending;
        effects.push(TextSessionEffect::InvocationParked {
            frame_id: invocation.frame_id.clone(),
            invocation_id: invocation.id.clone(),
        });
        self.invocation.as_mut().unwrap().release = Some(InvocationRelease::Parked);
    }

    /// The released invocation's dialogue as the person received it: a message cut short reads
    /// as interrupted, with the text that reached them.
    fn receipts(&mut self, delivered: &BTreeMap<String, String>) {
        let Some(invocation) = &self.invocation else { return };
        let items = match &invocation.release {
            Some(InvocationRelease::Settled(result, ..)) => &result.new_messages,
            Some(InvocationRelease::Parked) => &self.state.frames[&invocation.frame_id].history,
            None => return,
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
        let frame = self.state.frames.get_mut(&self.state.active_frame_id).unwrap();
        for message in received {
            record_delivered(frame, message);
        }
    }

    /// The latest snapshot of the turn's usage: every local invocation it ran, once.
    pub fn usage(&self) -> Usage {
        match &self.invocation {
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
        (self.usage.revision(), self.invocation.as_ref().map_or(0, |i| i.run.ledger().revision()))
    }

    fn end_invocation(&mut self) {
        if let Some(invocation) = self.invocation.take() {
            self.usage.absorb(invocation.run.ledger());
        }
    }

    fn finish(
        &mut self,
        mut result: Result<TextSessionTurn, TextSessionError>,
        effects: &mut Vec<TextSessionEffect>,
    ) {
        self.done = true;
        let invocation_id = match &result {
            Err(TextSessionError::Run(_)) => {
                self.invocation.as_ref().map(|invocation| invocation.id.clone())
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

/// Records a child's validated arguments when they are a JSON object, else refuses them.
fn accept_child_arguments(
    child: &mut PreparedChild,
    identity: &ArgumentIdentity,
) -> Result<(), crate::tool_calls::PrepareFailure> {
    match identity {
        ArgumentIdentity::Json(value) if value.is_object() => {
            child.identity = Some(approval::canonical(identity));
            child.arguments = Some(value.clone());
            Ok(())
        }
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

fn initial(history: &[ChatItem], mode: ChatHistory) -> Vec<ChatItem> {
    match mode {
        ChatHistory::Fresh => Vec::new(),
        ChatHistory::Fork => {
            let calls: std::collections::BTreeSet<_> = history
                .iter()
                .filter_map(|item| match item {
                    ChatItem::FunctionCall(call) => Some(call.call_id.as_str()),
                    _ => None,
                })
                .collect();
            let answered: std::collections::BTreeSet<_> = history
                .iter()
                .filter_map(|item| match item {
                    ChatItem::FunctionCallOutput(output)
                        if calls.contains(output.call_id.as_str()) =>
                    {
                        Some(output.call_id.as_str())
                    }
                    _ => None,
                })
                .collect();
            let complete = history
                .iter()
                .filter(|item| match item {
                    ChatItem::FunctionCall(call) => answered.contains(call.call_id.as_str()),
                    ChatItem::FunctionCallOutput(output) => {
                        answered.contains(output.call_id.as_str())
                    }
                    _ => true,
                })
                .cloned()
                .collect();
            ChatContext::new(complete)
                .copy(&CopyOptions {
                    exclude_instructions: true,
                    exclude_handoff: true,
                    exclude_config_update: true,
                    ..Default::default()
                })
                .items
        }
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
