use super::*;

mod continuation;
mod input;
#[cfg(test)]
#[path = "machine_tests.rs"]
mod tests;

impl Session {
    pub fn new(config: SessionConfig, input: SessionInput) -> Result<Self, RoutingError> {
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
                // Root completion describes the preceding invocation, not the lifetime of
                // the conversation. A new turn reopens that same agent context.
                if state.active_frame_id == state.root_frame_id {
                    let root = state
                        .frames
                        .get_mut(&state.root_frame_id)
                        .expect("the active root was checked above");
                    root.completed = false;
                    root.output = None;
                    root.error = None;
                }
                state
            }
            None => SessionSnapshot::new(
                input.session_id,
                input.task_id.unwrap_or(input.execution_id),
                config.root_agent.clone(),
            ),
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
            closing: false,
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

    /// Remote tools share this serialized owner with local agent execution. Runtime must
    /// checkpoint the resulting snapshot before acknowledging a command or starting I/O.
    pub fn remote(
        &mut self,
        command: lk_agents_abi::delegation::remote::RemoteCommand,
    ) -> Result<
        Option<lk_agents_abi::delegation::remote::InputRequestRef>,
        lk_agents_abi::storage::StorageError,
    > {
        crate::session::remote::admit(
            &command.input,
            self.closing || self.persistence_failure.is_some(),
            matches!(self.phase, Phase::Done),
        )?;
        crate::session::remote::apply(&mut self.state, command)
    }

    fn remote_state(
        &self,
        remote_id: &str,
        endpoint: &str,
    ) -> Result<
        Option<&lk_agents_abi::delegation::remote::RemoteAgentState>,
        lk_agents_abi::storage::StorageError,
    > {
        use lk_agents_abi::storage::StorageError;
        if self.closing || self.persistence_failure.is_some() || matches!(self.phase, Phase::Done) {
            return Err(StorageError::Invalid("the conversation owner is closing or ended".into()));
        }
        crate::session::remote::state(&self.state, remote_id, endpoint)
    }

    /// A saved context is routing data for this destination, never authority for another one.
    pub fn remote_context(
        &self,
        remote_id: &str,
        endpoint: &str,
    ) -> Result<Option<String>, lk_agents_abi::storage::StorageError> {
        Ok(self
            .remote_state(remote_id, endpoint)?
            .and_then(|state| state.default_context_id.clone()))
    }

    pub fn remote_task(
        &self,
        remote_id: &str,
        endpoint: &str,
        task_id: &str,
    ) -> Result<
        lk_agents_abi::delegation::remote::RemoteTaskRef,
        lk_agents_abi::storage::StorageError,
    > {
        let remote = self.remote_state(remote_id, endpoint)?.ok_or_else(|| {
            lk_agents_abi::storage::StorageError::Invalid("this remote has no known tasks".into())
        })?;
        crate::session::remote::task(remote, task_id)
    }

    /// Only the initiating local call can recover its previous send. Controls such as GetTask
    /// observe that send; they never replace it as the operation whose outcome is recovered.
    pub fn remote_recovery(
        &self,
        remote_id: &str,
        endpoint: &str,
        origin: &lk_agents_abi::session::ToolCallRef,
    ) -> Result<
        Option<lk_agents_abi::delegation::remote::RemoteRecovery>,
        lk_agents_abi::storage::StorageError,
    > {
        let Some(state) = self.remote_state(remote_id, endpoint)? else { return Ok(None) };
        crate::session::remote::recovery(state, origin)
    }

    pub fn snapshot(&self) -> SessionSnapshot {
        self.state.clone()
    }

    pub fn wake_at(&self) -> Option<Ts> {
        match (self.phase.invocation().and_then(|i| i.run.wake_at()), self.checkpoint_due) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    pub fn handle(&mut self, event: SessionEvent, now: Ts) -> Vec<SessionEffect> {
        let mut effects = Vec::new();
        if matches!(event, SessionEvent::Cancel) {
            self.closing = true;
        }
        // A late host receipt still transfers ownership of its prepared-arguments resource.
        // Invocation identity suffices for runtime cleanup, even after this request settled.
        if let SessionEvent::Run {
            invocation_id,
            event: agent_run::AgentRunEvent::ToolPrepared { prepared, .. },
        } = &event
            && !self.is_current(invocation_id)
        {
            effects.push(SessionEffect::Run {
                invocation_id: invocation_id.clone(),
                effect: agent_run::AgentRunEffect::ToolArgumentsRelease { prepared: *prepared },
            });
            return effects;
        }
        if matches!(self.phase, Phase::Done) {
            return effects;
        }
        match event {
            SessionEvent::PersistenceFailed(error) => {
                self.persistence_failure = Some(error);
                return self.handle(SessionEvent::Cancel, now);
            }
            SessionEvent::Start(message) if matches!(self.phase, Phase::NotStarted) => {
                self.record_dialogue(message.clone());
                // Receipt and human input precede classification. The detector never sees
                // a queued question or an agent's private model history.
                self.route_input(message, now, &mut effects);
            }
            SessionEvent::InputRouted { request_id, result, usage } => {
                self.input_routed(request_id, result, usage, now, &mut effects);
            }
            SessionEvent::Run { invocation_id, event } if self.is_current(&invocation_id) => {
                self.run_event(event, now, &mut effects);
            }
            SessionEvent::OutputValidated { invocation_id, validation, verdict, output }
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
            SessionEvent::DialogueDelivered { invocation_id, message }
                if self.is_current(&invocation_id) =>
            {
                self.record_dialogue(message);
            }
            SessionEvent::InvocationReleased { invocation_id, delivered } if matches!(&self.phase, Phase::Releasing(invocation, _) if invocation.id == invocation_id) =>
            {
                self.receipts(&delivered);
                // Release has ended this invocation, not the owner. Its delivery receipts
                // still admit presented input occurrences before settlement or continuation.
                let phase = std::mem::replace(&mut self.phase, Phase::NotStarted);
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
            SessionEvent::Reconcile { frame_id, result, output }
                if matches!(self.phase, Phase::NotStarted)
                    && frame_id == self.state.active_frame_id =>
            {
                self.usage.add(&result.usage);
                self.settled(result, output, Vec::new(), now, &mut effects);
            }
            SessionEvent::Cancel => {
                if matches!(self.phase, Phase::Releasing(..)) {
                    self.end_invocation();
                    self.finish(Err(TextSessionError::Run(RunError::Cancelled)), &mut effects);
                } else if matches!(self.phase, Phase::Running(_)) {
                    self.run_event(agent_run::AgentRunEvent::Cancel, now, &mut effects);
                } else {
                    self.finish(Err(TextSessionError::Run(RunError::Cancelled)), &mut effects);
                }
            }
            SessionEvent::Wake => self.run_event(agent_run::AgentRunEvent::Wake, now, &mut effects),
            _ => {}
        }
        for effect in &effects {
            if !self.recording {
                break;
            }
            if let SessionEffect::Emit {
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
            .any(|effect| matches!(effect, SessionEffect::Finished { result: Err(_), .. }));
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
                        SessionEffect::InvocationStarted { .. }
                            | SessionEffect::RouteInput(_)
                            | SessionEffect::InvocationParked { .. }
                            | SessionEffect::InvocationSettled { .. }
                            | SessionEffect::Finished { result: Ok(_), .. }
                            | SessionEffect::Run {
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
            effects.insert(0, SessionEffect::Checkpoint(Box::new(state)));
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

    fn invoke(&mut self, message: Option<ChatMessage>, now: Ts, effects: &mut Vec<SessionEffect>) {
        self.invoke_offered(message, None, now, effects);
    }

    fn invoke_offered(
        &mut self,
        message: Option<ChatMessage>,
        offered: Option<Vec<super::InputCandidate>>,
        now: Ts,
        effects: &mut Vec<SessionEffect>,
    ) {
        let invocation_id = self.mint("invocation");
        let mut answered = BTreeSet::new();
        let stamp = self.stamp(now);
        let mut notices = Vec::new();
        let frame = self.active_frame_mut();
        frame.completed = false;
        frame.output = None;
        frame.error = None;
        for request in frame.input_requests.values_mut() {
            if request.state != lk_agents_abi::session::InputRequestState::Responding {
                continue;
            }
            let Some(response) = &request.response_message else { continue };
            if let Some(origin) = &request.origin {
                answered.insert(origin.call_id.clone());
                if let Some(call) = frame.history.iter().find_map(|item| match item {
                    ChatItem::FunctionCall(call)
                        if call.id == origin.call_id
                            && approval::mark(call)
                                == Some(lk_agents_abi::llm::tool_executor::Approval::Asked) =>
                    {
                        Some(call)
                    }
                    _ => None,
                }) {
                    notices.push(
                        ChatMessage::text(
                            format!("{invocation_id}/approval/{}", origin.call_id),
                            Role::System,
                            approval::answered_text(
                                &call.name,
                                &response.raw_text_content().unwrap_or_default(),
                            ),
                            stamp,
                        )
                        .into(),
                    );
                }
            }
            if message.as_ref().is_none_or(|incoming| incoming.id != response.id)
                && !frame.history.iter().any(|item| item.id() == response.id)
            {
                frame.history.push(response.clone().into());
            }
            // Admission to the owning agent's history is local delivery. Its origin call is
            // provenance, not a directive to invoke that tool with the person's raw words.
            request.state = lk_agents_abi::session::InputRequestState::Responded;
        }
        // A side question leaves approval and its accepted arguments unresolved. Remove that
        // admission only when the owner delivered a response to the corresponding read-back.
        frame.pending_calls.retain(|call| {
            call.status != ChildCallStatus::Approval || !answered.contains(&call.parent.call_id)
        });
        frame.history.extend(notices);
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
            answer_tool: None,
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
                let run =
                    run.with_input_answers(answered).with_input_offers(offered.unwrap_or_else(
                        || crate::delegation::remote::routing::offered(&self.state),
                    ));
                if self.checkpoints { run.with_checkpoints() } else { run }
            }
            Err(error) => {
                self.finish(Err(TextSessionError::Routing(RoutingError::Agent(error))), effects);
                return;
            }
        };
        effects.push(SessionEffect::InvocationStarted {
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
        effects: &mut Vec<SessionEffect>,
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
                    effects
                        .push(SessionEffect::Run { invocation_id: invocation.id.clone(), effect });
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
        effects: &mut Vec<SessionEffect>,
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
                    effects.push(SessionEffect::Run { invocation_id, effect });
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
                        now,
                        effects,
                    ),
                other => effects.push(SessionEffect::Run { invocation_id, effect: other }),
            }
        }
        self.park(now, effects);
    }

    fn emit(
        &mut self,
        frame_id: FrameId,
        invocation_id: String,
        event: StreamEvent,
        effects: &mut Vec<SessionEffect>,
    ) {
        self.sequence += 1;
        let agent_name = self.state.frames[&frame_id].agent_name.clone();
        effects.push(SessionEffect::Emit {
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
        effects: &mut Vec<SessionEffect>,
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
        effects.push(SessionEffect::Run {
            invocation_id,
            effect: agent_run::AgentRunEffect::ToolArgumentsRelease { prepared },
        });
    }

    fn cancel_child(&mut self, call: CallId, now: Ts, effects: &mut Vec<SessionEffect>) {
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
        now: Ts,
        effects: &mut Vec<SessionEffect>,
    ) {
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                if let Some(invocation) = self.phase.invocation() {
                    let history = invocation.run.progress_history();
                    let mut approvals = self.approvals(invocation, &history);
                    let stamp = self.stamp(now);
                    let frame = self.active_frame_mut();
                    frame.history = history;
                    continuation::retain_approvals(&mut approvals, &frame.pending_calls);
                    frame.pending_calls = approvals;
                    for requirement in &input_requirements {
                        continuation::retain_input(frame, requirement, stamp);
                    }
                }
                self.failure_input_requirements = input_requirements;
                return self.finish(Err(TextSessionError::Run(error)), effects);
            }
        };
        let phase = std::mem::replace(&mut self.phase, Phase::Done);
        let (Phase::Running(invocation) | Phase::Releasing(invocation, _)) = phase else { return };
        let output = validated_output(&invocation, &result.outcome);
        let approvals = self.approvals(&invocation, &result.all_messages);
        effects.push(SessionEffect::InvocationSettled {
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
                call_id: child.function_call.id.clone(),
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

    fn approvals(&self, invocation: &Invocation, history: &[ChatItem]) -> Vec<PendingChildCall> {
        invocation
            .children
            .values()
            .filter_map(|child| match child {
                Child::Prepared(child) | Child::Executing(child) => Some(child),
                Child::Preparing(_) => None,
            })
            .filter_map(|child| {
                history.iter().find_map(|item| match item {
                    ChatItem::FunctionCall(call)
                        if call.id == child.function_call.id
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

    fn park(&mut self, now: Ts, effects: &mut Vec<SessionEffect>) {
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
        continuation::retain_approvals(
            &mut pending,
            &self.state.frames[&invocation.frame_id].pending_calls,
        );
        // Validation/launch arrival order must not change the model's sibling order.
        pending.sort_by_key(|p| {
            history
                .iter()
                .position(|i| {
                    matches!(i,
            ChatItem::FunctionCall(c) if c.id == p.parent.call_id)
                })
                .unwrap_or(usize::MAX)
        });
        let frame =
            self.state.frames.get_mut(&invocation.frame_id).expect("the invoked frame exists");
        frame.history = history;
        frame.pending_calls = pending;
        effects.push(SessionEffect::InvocationParked {
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
        for message in received {
            self.record_dialogue(message);
        }
    }

    fn record_dialogue(&mut self, message: ChatMessage) {
        let frame = self.active_frame_mut();
        if !matches!(message.role, Role::User | Role::Assistant)
            || frame.starting_history_ids.contains(&message.id)
        {
            return;
        }
        record_delivered(frame, message.clone());
        match self.state.chat_history.iter_mut().find(|item| item.id == message.id) {
            Some(delivered) => *delivered = message,
            None => self.state.chat_history.push(message),
        }
    }

    /// The latest snapshot of the turn's usage: every local invocation it ran, once.
    pub fn routing_usage(&mut self, usage: &Usage) {
        self.usage.add(usage);
    }

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
        effects: &mut Vec<SessionEffect>,
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
        if let Err(error) = &result {
            let history =
                self.phase.invocation().map(|invocation| invocation.run.progress_history());
            let frame = self.active_frame_mut();
            if let Some(history) = history {
                frame.history = history;
            }
            frame.completed = false;
            frame.output = None;
            frame.error = Some(error.to_string());
            if self.checkpoints && !matches!(error, TextSessionError::Storage(_)) {
                effects.push(SessionEffect::Checkpoint(Box::new(self.state.clone())));
            }
        }
        self.end_invocation();
        self.usage.settle();
        if let Ok(turn) = &mut result {
            turn.result.usage = self.usage.snapshot();
        }
        effects.push(SessionEffect::Finished {
            invocation_id,
            result,
            input_requirements: std::mem::take(&mut self.failure_input_requirements),
        });
    }
}

/// The pending execution continues under `execution_id`: its waiting child calls move with it.
fn reidentify(state: &mut SessionSnapshot, execution_id: String) {
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
    {
        return;
    }
    match frame.visible_dialogue.iter_mut().find(|delivered| delivered.id == message.id) {
        Some(delivered) => *delivered = message,
        None => frame.visible_dialogue.push(message),
    }
}

fn transfer(child: &mut SessionFrame, parent: &mut SessionFrame) {
    if !child.options.merge_chat_ctx {
        return;
    }
    for message in &child.visible_dialogue {
        if !parent.history.iter().any(|item| item.id() == message.id) {
            parent.history.push(ChatItem::Message(message.clone()));
        }
        record_delivered(parent, message.clone());
    }
}
