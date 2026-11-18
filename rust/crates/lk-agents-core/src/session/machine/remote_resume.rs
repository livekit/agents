//! A saved accepted child reconciles its exact remote origins before returning to its parent.

use super::*;
use lk_agents_abi::a2a_protocol as a2a;
use lk_agents_abi::delegation::remote::{A2AJournalPayload, A2ARequest, RemoteRecovery};
use lk_agents_abi::llm::tool_executor::{REMOTE_REPORT_KEY, RemoteReport};
use lk_agents_abi::session::ToolCallRef;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Proof {
    Completed,
    Failed,
    Cancelled,
    Pending,
}

fn invalid(message: &str) -> RoutingError {
    RoutingError::InvalidState(message.into())
}

impl Session {
    pub(super) fn resume_child_remotes(
        &mut self,
        now: Ts,
        effects: &mut Vec<SessionEffect>,
    ) -> Result<bool, RoutingError> {
        let frame = &self.state.frames[&self.state.active_frame_id];
        let mut calls = Vec::new();
        let mut origins = Vec::new();
        let mut invalidated = false;
        for item in &frame.history {
            let ChatItem::FunctionCall(call) = item else { continue };
            let origin = ToolCallRef { agent_id: frame.id.value.clone(), call_id: call.id.clone() };
            let sources: Vec<_> = self
                .state
                .remote_agents
                .iter()
                .filter(|(_, remote)| {
                    remote.requests.values().any(|request| {
                        request.origin.as_ref() == Some(&origin)
                            && remote.events.get(request.request_event as usize).is_some_and(
                                |event| {
                                    matches!(
                                        &event.payload,
                                        A2AJournalPayload::Request(A2ARequest::Send(_))
                                    )
                                },
                            )
                    })
                })
                .collect();
            let report = crate::tool_calls::remote_report(call);
            if sources.is_empty() {
                if report == Some(RemoteReport::Pending) {
                    return Err(invalid("the pending remote call has no provable journal origin"));
                }
                continue;
            }
            let [(remote_id, remote)] = sources.as_slice() else {
                return Err(invalid("the remote call has ambiguous saved origins"));
            };
            let Some(report) = report else {
                return Err(invalid(
                    "the saved remote call has no unambiguous native report provenance",
                ));
            };
            if report == RemoteReport::Completed {
                if !frame.history.iter().any(|item| matches!(item, ChatItem::FunctionCallOutput(output) if output.id == call.output_id())) {
                    return Err(invalid("the final remote report has no canonical output"));
                }
                continue;
            }
            let agent = &self.config.agents[&frame.agent_name];
            let configured = agent.definition.tools.iter().any(|tool| {
                tool.spec.name == call.name
                    && tool.remote.as_deref() == Some(remote_id.as_str())
                    && tool.local_agent.is_none()
            });
            if !configured || agent.agent_tools.contains_key(&call.name) {
                return Err(invalid(
                    "the configured remote identity does not expose the original callable name; recovery cannot substitute a host function",
                ));
            }
            let recovery = crate::session::remote::recovery(remote, &origin)
                .map_err(|error| RoutingError::InvalidState(error.to_string()))?;
            match recovery {
                Some(RemoteRecovery::Uncertain) | None => {
                    return Err(invalid(
                        "the original remote outcome is unprovable; inspect or reconcile it before continuing",
                    ));
                }
                Some(RemoteRecovery::Dispatch { request, .. })
                    if !matches!(request.as_ref(), A2ARequest::Send(_)) =>
                {
                    return Err(invalid("remote recovery has no original business request"));
                }
                _ => {}
            }
            invalidated |= report != RemoteReport::Pending;
            calls.push(call.clone());
            origins
                .push(RemoteOrigin { remote_id: (*remote_id).clone(), call_id: call.id.clone() });
        }
        if calls.is_empty() {
            return Ok(false);
        }
        self.start_remote_resume(
            calls,
            RemoteResume { origins, incoming: None, invalidated },
            now,
            effects,
        )?;
        Ok(true)
    }

    fn start_remote_resume(
        &mut self,
        calls: Vec<FunctionCall>,
        resume: RemoteResume,
        now: Ts,
        effects: &mut Vec<SessionEffect>,
    ) -> Result<(), RoutingError> {
        let invocation_id = self.mint("invocation");
        let frame = &self.state.frames[&self.state.active_frame_id];
        let agent = &self.config.agents[&frame.agent_name];
        let history = frame.history.clone();
        let input = RunInput {
            invocation_id: invocation_id.clone(),
            id_seed: lk_agents_abi::types::ids::splitmix64(self.id_seed, self.state.next_id),
            opening: Opening::Message(Box::new(ChatMessage::text(
                String::new(),
                Role::User,
                "",
                std::time::UNIX_EPOCH,
            ))),
            history: history.clone(),
            delegated: self.delegated,
            mode: crate::tool_outcome::Mode::Text,
            llm: agent.llm.clone(),
            wall: self.wall,
            cleanup_timeout: self.cleanup_timeout,
        };
        let mut run =
            AgentRun::new(agent.definition.clone(), input).map_err(RoutingError::Agent)?;
        let run_effects = run.resume_remote(calls, now);
        effects.push(SessionEffect::InvocationStarted {
            origin: self.live.as_ref().and_then(|live| live.origin().cloned()),
            frame_id: frame.id.clone(),
            agent_name: frame.agent_name.clone(),
            invocation_id: invocation_id.clone(),
            execution_id: self.state.execution_id.clone(),
            history,
        });
        self.phase = Phase::Running(Invocation {
            id: invocation_id,
            frame_id: frame.id.clone(),
            run,
            children: BTreeMap::new(),
            outputs: BTreeMap::new(),
            recovery: Some(resume),
        });
        self.run_effects(run_effects, now, effects);
        Ok(())
    }

    fn remote_proof(&self, frame_id: &FrameId, origin: &RemoteOrigin) -> Proof {
        let Some(remote) = self.state.remote_agents.get(&origin.remote_id) else {
            return Proof::Pending;
        };
        let source =
            ToolCallRef { agent_id: frame_id.value.clone(), call_id: origin.call_id.clone() };
        // Follow observations retain the same exact originating call. Controls have no origin,
        // and cannot supply a business result for this child.
        let Some((request_id, request)) = remote
            .requests
            .iter()
            .filter(|(_, request)| request.origin.as_ref() == Some(&source))
            .max_by_key(|(_, request)| request.request_event)
        else {
            return Proof::Pending;
        };
        let Ok(RemoteRecovery::Observed(response)) =
            crate::delegation::remote::journal::recovery(remote, request_id)
        else {
            return if request
                .task
                .as_ref()
                .and_then(|task| remote.tasks.get(&task.task_id))
                .is_some_and(|task| task.task.status.state == a2a::TaskState::AuthRequired)
            {
                Proof::Failed
            } else {
                Proof::Pending
            };
        };
        match *response {
            a2a::SendMessageResponse::Message(_) => Proof::Completed,
            a2a::SendMessageResponse::Task(task) => {
                let Ok(Some(update)) = crate::a2a::client::ReplyState::default()
                    .accept(a2a::StreamResponse::Task(task))
                else {
                    return Proof::Pending;
                };
                match update.state {
                    a2a::TaskState::Completed if update.follow_task.is_none() => Proof::Completed,
                    a2a::TaskState::Canceled => Proof::Cancelled,
                    a2a::TaskState::Failed
                    | a2a::TaskState::Rejected
                    | a2a::TaskState::AuthRequired => Proof::Failed,
                    _ => Proof::Pending,
                }
            }
        }
    }

    pub(super) fn remote_invocation_finished(
        &mut self,
        frame_id: FrameId,
        invocation_id: String,
        result: Result<RunResult, RunError>,
        requirements: Vec<lk_agents_abi::agent::run::RunInputRequirement>,
        _now: Ts,
        effects: &mut Vec<SessionEffect>,
    ) {
        let resume = self
            .phase
            .invocation()
            .and_then(|invocation| invocation.recovery.as_ref())
            .expect("guarded native recovery");
        let proofs: BTreeMap<_, _> = resume
            .origins
            .iter()
            .map(|origin| (origin.call_id.clone(), self.remote_proof(&frame_id, origin)))
            .collect();
        let mut result = match result {
            Ok(result) => result,
            Err(error) => {
                let history = self
                    .phase
                    .invocation()
                    .expect("the recovery invocation exists")
                    .run
                    .progress_history();
                let frame =
                    self.state.frames.get_mut(&frame_id).expect("the recovery frame exists");
                frame.history = history;
                if matches!(error, RunError::Cancelled)
                    || proofs
                        .values()
                        .any(|proof| matches!(proof, Proof::Failed | Proof::Cancelled))
                {
                    frame.completed = false;
                }
                // The incoming turn was already accepted as human dialogue. Preserve it in
                // this child's canonical context even when reconciliation itself must refuse.
                self.retain_recovery_human();
                self.failure_input_requirements = requirements;
                self.finish(Err(TextSessionError::Run(error)), effects);
                return;
            }
        };
        let reports: BTreeMap<_, _> = proofs
            .iter()
            .map(|(call, proof)| {
                let question = result
                    .input_requirements
                    .iter()
                    .any(|requirement| requirement.call.id == *call);
                let failed = result.all_messages.iter().any(|item| {
                    matches!(item, ChatItem::FunctionCallOutput(output)
                if output.id == format!("{call}_out") && output.is_error)
                });
                let report = if question {
                    RemoteReport::InputRequired
                } else if failed {
                    RemoteReport::Failed
                } else {
                    match proof {
                        Proof::Completed => RemoteReport::Completed,
                        Proof::Failed => RemoteReport::Failed,
                        Proof::Cancelled => RemoteReport::Cancelled,
                        Proof::Pending => RemoteReport::Pending,
                    }
                };
                (call.clone(), report)
            })
            .collect();
        for item in result.all_messages.iter_mut().chain(result.new_messages.iter_mut()) {
            if let ChatItem::FunctionCall(call) = item
                && let Some(report) = reports.get(&call.id)
            {
                call.extra.insert(
                    REMOTE_REPORT_KEY.into(),
                    serde_json::to_value(report).expect("native report provenance serializes"),
                );
            }
        }
        self.phase
            .invocation_mut()
            .expect("the recovery invocation exists")
            .run
            .commit_remote_reports(&reports);
        let frame = self.state.frames.get_mut(&frame_id).expect("the recovery frame exists");
        frame.history = result.all_messages.clone();
        if matches!(result.outcome, RunOutcome::InputRequired(_))
            || proofs.values().any(|proof| matches!(proof, Proof::Failed | Proof::Cancelled))
        {
            frame.completed = false;
        }
        let phase = std::mem::replace(&mut self.phase, Phase::Done);
        let Phase::Running(invocation) = phase else { return };
        effects.push(SessionEffect::InvocationSettled {
            frame_id,
            invocation_id,
            result: result.clone(),
        });
        self.phase = Phase::Releasing(invocation, InvocationRelease::Recovered(Box::new(result)));
    }

    fn retain_human_context(&mut self, message: ChatMessage) {
        if let Some(invocation) = self.phase.invocation_mut() {
            invocation.run.observe_context(std::slice::from_ref(&message));
        }
        let frame = self.active_frame_mut();
        if !frame.history.iter().any(|item| item.id() == message.id) {
            frame.history.push(message.into());
        }
    }

    fn retain_recovery_human(&mut self) {
        let human = self
            .phase
            .invocation_mut()
            .and_then(|invocation| invocation.recovery.as_mut())
            .and_then(|resume| resume.incoming.take());
        if let Some(human) = human {
            self.retain_human_context(human.message);
        }
    }

    fn continue_recovered_human(
        &mut self,
        result: &RunResult,
        human: crate::session::live::FrozenHuman,
        now: Ts,
        effects: &mut Vec<SessionEffect>,
    ) {
        if let Err(error) = self.retain_settled_inputs(result, now) {
            self.retain_human_context(human.message);
            self.finish(Err(TextSessionError::Storage(error)), effects);
            return;
        }
        // Use the candidates captured before recovery, not the question just observed.
        self.route_frozen_input(human, now, effects);
    }

    pub(super) fn remote_released(
        &mut self,
        result: RunResult,
        resume: RemoteResume,
        now: Ts,
        effects: &mut Vec<SessionEffect>,
    ) {
        let active = self.state.active_frame_id.clone();
        if result.interrupted {
            self.state.frames.get_mut(&active).expect("the active child exists").completed = false;
        }
        if matches!(result.outcome, RunOutcome::InputRequired(_)) {
            if let Some(human) = resume.incoming {
                self.continue_recovered_human(&result, *human, now, effects);
            } else {
                self.settled(result, None, Vec::new(), now, effects);
            }
            return;
        }
        let complete = resume
            .origins
            .iter()
            .all(|origin| self.remote_proof(&active, origin) == Proof::Completed);
        let failed_report = resume.origins.iter().any(|origin| result.all_messages.iter().any(|item|
            matches!(item, ChatItem::FunctionCallOutput(output) if output.id == format!("{}_out", origin.call_id) && output.is_error)));
        if !complete || failed_report || result.interrupted {
            if self.state.frames[&active].completed {
                if let Some(human) = resume.incoming {
                    self.retain_human_context(human.message);
                }
                self.finish(Err(TextSessionError::Routing(invalid("the original remote outcome remains unresolved; the accepted child is retained"))), effects);
            } else if let Some(human) = resume.incoming {
                self.continue_recovered_human(&result, *human, now, effects);
            } else {
                self.settled(result, None, Vec::new(), now, effects);
            }
            return;
        }
        if resume.invalidated {
            self.state.frames.get_mut(&active).expect("the active child exists").completed = false;
            match resume.incoming {
                Some(human) => self.route_frozen_input(*human, now, effects),
                None => self.invoke(None, now, effects),
            }
            return;
        }
        let frame = &self.state.frames[&active];
        let Some(parent) = frame.parent.clone() else {
            self.finish(
                Err(TextSessionError::Routing(invalid("remote recovery lost its parent call"))),
                effects,
            );
            return;
        };
        let Some(output) = frame.output.clone() else {
            self.finish(
                Err(TextSessionError::Routing(RoutingError::MissingAcceptedOutput)),
                effects,
            );
            return;
        };
        if let Err(error) = self.propagate_child(parent, output, now, effects) {
            self.finish(Err(TextSessionError::Routing(error)), effects);
            return;
        }
        match resume.incoming {
            Some(human) => self.route_frozen_input(*human, now, effects),
            None => self.next_child(now, effects),
        }
    }
}
