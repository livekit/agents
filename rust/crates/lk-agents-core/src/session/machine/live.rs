use super::*;
use crate::session::live::{FrozenHuman, TurnInput};
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::storage::StorageError;
use lk_agents_abi::text_session::LiveTurnOrigin;

impl Session {
    /// This capability is installed only by a configured native endpoint owner. Standalone
    /// runs retain their ordinary cleanup and result semantics.
    pub fn with_live(mut self, origin: LiveTurnOrigin) -> Self {
        self.state.execution_id = origin.execution_id.clone();
        self.live = Some(crate::session::live::Live::new(origin));
        self
    }

    pub fn live_idle(&self) -> bool {
        self.live.as_ref().is_some_and(|live| live.idle())
    }

    /// A released model turn may still have identified bodies retained by this owner. All
    /// other execution modes wait for the invocation's actual resource cleanup as before.
    pub fn allows_live_release(&self, invocation: &str) -> bool {
        self.live.is_some()
            && self
                .phase
                .invocation()
                .is_some_and(|current| current.id == invocation && current.recovery.is_none())
    }

    pub fn admit_live(
        &mut self,
        origin: LiveTurnOrigin,
        message: ChatMessage,
        now: Ts,
    ) -> Result<Vec<SessionEffect>, StorageError> {
        let mut effects = Vec::new();
        self.enqueue_live(origin, message, now, &mut effects)?;
        if self.recording && !effects.is_empty() {
            let mut state = self.state.clone();
            if let Some(invocation) = self.phase.invocation() {
                state.frames.get_mut(&invocation.frame_id).expect("invoked frame exists").history =
                    invocation.run.progress_history();
            }
            effects.insert(0, SessionEffect::Checkpoint(Box::new(state)));
        }
        Ok(effects)
    }

    pub(super) fn enqueue_live(
        &mut self,
        origin: LiveTurnOrigin,
        message: ChatMessage,
        now: Ts,
        effects: &mut Vec<SessionEffect>,
    ) -> Result<(), StorageError> {
        if self.closing || self.persistence_failure.is_some() {
            return Err(StorageError::Invalid("the conversation owner is closing".into()));
        }
        let human = FrozenHuman {
            message,
            candidates: self.input_candidates(),
            offered: crate::delegation::remote::routing::offered(&self.state),
            history: self.state.chat_history.clone(),
        };
        self.live
            .as_mut()
            .ok_or_else(|| {
                StorageError::Invalid("this conversation has no live endpoint owner".into())
            })?
            .enqueue(origin, human)?;
        self.next_live(now, effects);
        Ok(())
    }

    pub(super) fn next_live(&mut self, now: Ts, effects: &mut Vec<SessionEffect>) {
        if !matches!(self.phase, Phase::NotStarted) || self.live_dialogue_pending() {
            return;
        }
        let active = self.state.active_frame_id.clone();
        let completed_child = self
            .state
            .frames
            .get(&active)
            .is_some_and(|frame| frame.completed && frame.parent.is_some());
        let Some(turn) = self.live.as_mut().and_then(|live| live.next(&active, completed_child))
        else {
            return;
        };
        // Transport admission already selected this task. A delivery restores its original
        // identity instead of borrowing whichever newer request most recently had the floor.
        self.state.execution_id = turn.origin.execution_id.clone();
        match turn.input {
            TurnInput::Cancelled => {
                self.finish(Err(TextSessionError::Run(RunError::Cancelled)), effects)
            }
            TurnInput::Human(human) => {
                self.record_dialogue(human.message.clone());
                if let Err(error) = self.restore_completed(now, effects) {
                    self.finish(Err(TextSessionError::Routing(error)), effects);
                    return;
                }
                if let Some(resume) =
                    self.phase.invocation_mut().and_then(|invocation| invocation.recovery.as_mut())
                {
                    resume.incoming = Some(human);
                } else {
                    self.route_frozen_input(*human, now, effects);
                }
            }
            TurnInput::Delivery(keys) => {
                let outputs =
                    self.live.as_ref().expect("selected live turn").delivered_outputs(&keys);
                let mut requirements = Vec::new();
                let mut cancelled = self.live.as_ref().is_some_and(|live| live.current_cancelled());
                for delivery in outputs {
                    let Some(report) = delivery.report else {
                        self.active_frame_mut().completed = false;
                        self.finish(
                            Err(TextSessionError::Routing(RoutingError::InvalidState(
                                "a live tool ended without its final outcome".into(),
                            ))),
                            effects,
                        );
                        return;
                    };
                    let Some(frame) = self.state.frames.get_mut(&delivery.frame_id) else {
                        self.finish(
                            Err(TextSessionError::Routing(RoutingError::InvalidState(
                                "a live body lost its originating frame".into(),
                            ))),
                            effects,
                        );
                        return;
                    };
                    let shape = crate::tool_outcome::shape(
                        &report.outcome,
                        crate::tool_outcome::Mode::Text,
                    );
                    let Some(output) = frame.history.iter_mut().find_map(|item| match item {
                        ChatItem::FunctionCallOutput(output)
                            if output.id == delivery.call.output_id() =>
                        {
                            Some(output)
                        }
                        _ => None,
                    }) else {
                        self.finish(
                            Err(TextSessionError::Routing(RoutingError::InvalidState(
                                "a live body lost its canonical progress output".into(),
                            ))),
                            effects,
                        );
                        return;
                    };
                    output.output = shape.text;
                    output.is_error = shape.is_error;
                    output.reply_required = shape.reply_required;
                    for item in &mut frame.history {
                        if let ChatItem::FunctionCall(call) = item
                            && call.id == delivery.call.id
                        {
                            call.extra
                                .remove(lk_agents_abi::llm::tool_executor::LIVE_LOCAL_PENDING_KEY);
                            if crate::tool_calls::remote_report(call).is_some() {
                                crate::tool_calls::record_remote_report(call, &report.outcome);
                            }
                        }
                    }
                    if shape.is_error
                        || matches!(
                            report.outcome,
                            ToolOutcome::InputRequired { .. }
                                | ToolOutcome::Working(_)
                                | ToolOutcome::Cancelled
                        )
                    {
                        frame.completed = false;
                    }
                    cancelled |= matches!(report.outcome, ToolOutcome::Cancelled);
                    let settled = crate::tool_calls::Settled {
                        outcome: report.outcome,
                        input_request: report.input_request.map(|reference| *reference),
                        retry: false,
                    };
                    if let Some(requirement) =
                        settled.input_requirement(&delivery.invocation_id, &delivery.call)
                    {
                        requirements.push(requirement);
                    }
                }
                if cancelled {
                    self.active_frame_mut().completed = false;
                    self.finish(Err(TextSessionError::Run(RunError::Cancelled)), effects);
                } else if !requirements.is_empty() {
                    self.invoke_offered(None, None, requirements, now, effects);
                } else if self.state.frames[&active].completed
                    && self.state.frames[&active].parent.is_some()
                {
                    if self.live.as_ref().is_some_and(|live| live.pending_for_frame(&active))
                        || self.output_pending_for_frame(&active)
                    {
                        self.finish_live_pending(effects);
                    } else if let Err(error) = self.restore_completed(now, effects) {
                        self.finish(Err(TextSessionError::Routing(error)), effects);
                    } else if self
                        .phase
                        .invocation()
                        .is_none_or(|invocation| invocation.recovery.is_none())
                    {
                        self.next_child(now, effects);
                    }
                } else {
                    self.invoke(None, now, effects);
                }
            }
            TurnInput::Initial => {}
        }
    }

    pub(super) fn observe_live_event(
        &mut self,
        invocation: &str,
        event: &agent_run::AgentRunEvent,
    ) {
        let Some(live) = self.live.as_mut() else {
            return;
        };
        match event {
            agent_run::AgentRunEvent::ToolPrepared { call, identity, .. } => {
                live.prepared((invocation.to_owned(), *call), identity.clone());
            }
            agent_run::AgentRunEvent::ToolProgress { call, .. } => {
                live.detach(&(invocation.to_owned(), *call));
            }
            agent_run::AgentRunEvent::ToolSettled { call, report } => {
                if matches!(report.outcome, ToolOutcome::Update { .. }) {
                    live.detach(&(invocation.to_owned(), *call));
                } else {
                    live.settle(&(invocation.to_owned(), *call), report.clone());
                }
            }
            agent_run::AgentRunEvent::ToolLaunchFailed { call, error } => {
                live.settle(
                    &(invocation.to_owned(), *call),
                    crate::tool_outcome::host_failure(error).into(),
                );
            }
            agent_run::AgentRunEvent::CallLost { call, .. } => {
                live.lost(&(invocation.to_owned(), *call));
            }
            _ => {}
        }
    }

    pub(super) fn detached_live_event(
        &self,
        invocation: &str,
        event: &agent_run::AgentRunEvent,
    ) -> bool {
        let call = match event {
            agent_run::AgentRunEvent::ToolSettled { call, .. }
            | agent_run::AgentRunEvent::ToolLaunchFailed { call, .. }
            | agent_run::AgentRunEvent::CallLost { call, .. }
            | agent_run::AgentRunEvent::ToolProgress { call, .. } => *call,
            _ => return false,
        };
        self.live.as_ref().is_some_and(|live| live.detached(&(invocation.to_owned(), call)))
    }

    pub(super) fn settle_live_event(
        &mut self,
        invocation: &str,
        event: agent_run::AgentRunEvent,
        now: Ts,
        effects: &mut Vec<SessionEffect>,
    ) {
        let live = self.live.as_mut().expect("guarded live body");
        match event {
            agent_run::AgentRunEvent::ToolSettled { call, report } => {
                live.settle(&(invocation.to_owned(), call), report);
            }
            agent_run::AgentRunEvent::ToolLaunchFailed { call, error } => {
                live.settle(
                    &(invocation.to_owned(), call),
                    crate::tool_outcome::host_failure(&error).into(),
                );
            }
            agent_run::AgentRunEvent::CallLost { call, .. } => {
                live.lost(&(invocation.to_owned(), call));
            }
            _ => return,
        }
        self.next_live(now, effects);
    }

    /// A retained accepted child needs no new model decision while other body deliveries
    /// remain. This report commits only the current delivery; its business task stays Working.
    fn finish_live_pending(&mut self, effects: &mut Vec<SessionEffect>) {
        let result = lk_agents_abi::agent::run::RunResult {
            outcome: lk_agents_abi::agent::run::RunOutcome::NoOutput,
            interrupted: false,
            new_messages: Vec::new(),
            all_messages: self.state.frames[&self.state.active_frame_id].history.clone(),
            usage: self.usage.snapshot(),
            verbatim: None,
            pending_call: None,
            input_requirements: Vec::new(),
        };
        self.finish(Ok(TextSessionTurn { result, state: self.state.clone() }), effects);
    }

    pub(super) fn finish_live(
        &mut self,
        result: Result<TextSessionTurn, TextSessionError>,
        effects: &mut Vec<SessionEffect>,
    ) {
        self.end_invocation();
        self.phase = Phase::NotStarted;
        let Some((turn, origin, working)) =
            self.live.as_mut().and_then(|live| live.commit_started())
        else {
            return;
        };
        effects.push(SessionEffect::LiveTurnFinished {
            turn,
            origin,
            result: Box::new(result),
            working,
        });
    }
}
