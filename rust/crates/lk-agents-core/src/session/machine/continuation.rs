use super::{
    ChildCallStatus, PendingChildCall, RoutingError, Session, SessionEffect, SessionFrame,
    TextSessionError, TextSessionTurn, transfer,
};
use lk_agents_abi::agent::run::{RunOutcome, RunResult};
use lk_agents_abi::{ChatItem, FunctionCallOutput, Ts};
use serde_json::Value;

// A parent is invoked only after every child call in its checkpoint has a result.

impl Session {
    pub(super) fn next_child(&mut self, now: Ts, effects: &mut Vec<SessionEffect>) {
        loop {
            let parent_id = self.state.active_frame_id.clone();
            let pending = self.state.frames[&parent_id]
                .pending_calls
                .iter()
                .find(|call| call.status != ChildCallStatus::Approval)
                .cloned();
            let Some(pending) = pending else {
                self.invoke(None, now, effects);
                return;
            };
            if pending.status != ChildCallStatus::Ready {
                self.finish(
                    Err(TextSessionError::Routing(RoutingError::InvalidState(
                        "active frame still has an active child".into(),
                    ))),
                    effects,
                );
                return;
            }
            if let Err(error) = self.admit_child(&pending) {
                effects.push(SessionEffect::RoutingRefused {
                    frame_id: parent_id.clone(),
                    call_id: pending.parent.call_id.clone(),
                    error: error.clone(),
                });
                if let Err(error) =
                    self.resolve_call(&pending, error.to_string(), true, now, effects)
                {
                    self.finish(Err(TextSessionError::Routing(error)), effects);
                    return;
                }
                continue;
            }
            let parent = &self.state.frames[&parent_id];
            let tool = &self.config.agents[&parent.agent_name].agent_tools[&pending.tool_name];
            let history = crate::delegation::local::opening(&parent.history, tool.chat_history);
            let merge_chat_ctx = tool.merge_chat_ctx;
            let frame_id = super::FrameId { value: self.mint("frame") };
            let task_id = self.mint("task");
            let mut starting_history_ids = history
                .iter()
                .map(|i| i.id().to_owned())
                .collect::<std::collections::BTreeSet<_>>();
            starting_history_ids.insert(task_id.clone());
            let task = crate::delegation::local::task(
                task_id,
                &pending.validated_arguments,
                self.stamp(now),
            );
            let frame = SessionFrame {
                id: frame_id.clone(),
                agent_name: pending.agent_name.clone(),
                history,
                parent: Some(pending.parent.clone()),
                completed: false,
                output: None,
                error: None,
                input_requests: Default::default(),
                pending_calls: Vec::new(),
                visible_dialogue: Vec::new(),
                starting_history_ids,
                options: lk_agents_abi::text_session::FrameOptions { merge_chat_ctx },
            };
            self.state
                .frames
                .get_mut(&parent_id)
                .expect("a pending call's parent frame exists")
                .pending_calls
                .iter_mut()
                .find(|call| call.parent == pending.parent)
                .expect("the selected child call remains in its parent")
                .status = ChildCallStatus::Running { frame_id: frame_id.clone() };
            self.state.frames.insert(frame_id.clone(), frame);
            self.state.ancestors.push(parent_id);
            self.state.active_frame_id = frame_id;
            self.invoke(Some(task), now, effects);
            return;
        }
    }

    fn admit_child(&self, pending: &PendingChildCall) -> Result<(), RoutingError> {
        if self.state.ancestors.len() + 1 >= self.config.max_depth {
            return Err(RoutingError::MaxDepth { max_depth: self.config.max_depth });
        }
        let parent = &self.state.frames[&pending.parent.frame_id];
        if !self.config.agents[&parent.agent_name].agent_tools.contains_key(&pending.tool_name) {
            return Err(RoutingError::UnknownTool(pending.tool_name.clone()));
        }
        if !self.config.agents.contains_key(&pending.agent_name) {
            return Err(RoutingError::UnknownAgent(pending.agent_name.clone()));
        }
        Ok(())
    }

    pub(super) fn settled(
        &mut self,
        result: RunResult,
        output: Option<Value>,
        approvals: Vec<PendingChildCall>,
        now: Ts,
        effects: &mut Vec<SessionEffect>,
    ) {
        let active_id = self.state.active_frame_id.clone();
        let stamp = self.stamp(now);
        let fallback = match &result.outcome {
            RunOutcome::InputRequired(_)
                if result.pending_call.as_ref().is_none_or(|call| {
                    !result.input_requirements.iter().any(|requirement| &requirement.call == call)
                }) =>
            {
                Some(self.mint("input"))
            }
            _ => None,
        };
        let mut presentations = Vec::new();
        let frame = self.active_frame_mut();
        frame.history = result.all_messages.clone();
        let mut pending = approvals;
        retain_approvals(&mut pending, &frame.pending_calls);
        frame.pending_calls = pending;
        frame.completed = matches!(result.outcome, RunOutcome::Completed { .. });
        frame.output = output.clone().or_else(|| match &result.outcome {
            RunOutcome::Completed { output: lk_agents_abi::agent::run::Output::Text(text) } => {
                Some(Value::String(text.clone()))
            }
            _ => None,
        });
        frame.error = None;
        let delivered_question = match &result.outcome {
            RunOutcome::InputRequired(question) => result.new_messages.iter().find_map(|item| {
                let message = item.as_message()?;
                (message.role == lk_agents_abi::Role::Assistant
                    && message.raw_text_content().as_deref() == Some(question.question.as_str()))
                .then(|| {
                    frame.visible_dialogue.iter().find(|delivered| {
                        delivered.id == message.id
                            && !delivered.interrupted
                            && delivered.raw_text_content() == message.raw_text_content()
                    })
                })
                .flatten()
                .map(|message| message.id.clone())
            }),
            _ => None,
        };
        for requirement in &result.input_requirements {
            let reference = retain_input(frame, requirement, stamp);
            if let Some(message_id) = requirement.presentation_message_id.as_ref().filter(|id| {
                frame.visible_dialogue.iter().any(|message| {
                    message.id == **id
                        && !message.interrupted
                        && message.role == lk_agents_abi::Role::Assistant
                        && message.raw_text_content().is_some_and(|text| !text.trim().is_empty())
                        && result.new_messages.iter().any(|item| {
                            item.as_message().is_some_and(|source| {
                                source.id == message.id
                                    && source.raw_text_content() == message.raw_text_content()
                            })
                        })
                })
            }) {
                presentations.push((reference, message_id.clone()));
            }
        }
        // Framework read-backs and a structured child's own question have no tool-body
        // requirement. They still own an input occurrence, under the same delivery rules.
        if let (Some(id), RunOutcome::InputRequired(question)) = (fallback, &result.outcome) {
            frame.input_requests.insert(
                id.clone(),
                lk_agents_abi::session::AgentInputRequest {
                    state: lk_agents_abi::session::InputRequestState::Open,
                    source_message: lk_agents_abi::ChatMessage::text(
                        &id,
                        lk_agents_abi::Role::Assistant,
                        &question.question,
                        stamp,
                    ),
                    response_message: None,
                    origin: result.pending_call.as_ref().map(|call| {
                        lk_agents_abi::session::ToolCallRef {
                            agent_id: active_id.value.clone(),
                            call_id: call.id.clone(),
                        }
                    }),
                    presented_message_id: None,
                },
            );
            if let Some(message_id) = &delivered_question {
                presentations.push((
                    lk_agents_abi::session::InputRequestRef::Agent {
                        agent_id: active_id.value.clone(),
                        request_id: id,
                    },
                    message_id.clone(),
                ));
            }
        }
        if frame.completed && frame.parent.is_some() {
            for request in frame.input_requests.values_mut() {
                if matches!(request.state, lk_agents_abi::session::InputRequestState::Open) {
                    request.state = lk_agents_abi::session::InputRequestState::Closed;
                }
            }
        }
        let parent_call = match frame.parent.clone() {
            Some(parent)
                if !result.interrupted
                    && !matches!(result.outcome, RunOutcome::InputRequired { .. }) =>
            {
                parent
            }
            _ => {
                for (reference, message_id) in presentations {
                    if let Err(error) =
                        self.input(lk_agents_abi::session::SessionInputCommand::Presented {
                            requests: vec![reference],
                            message_id,
                        })
                    {
                        self.finish(Err(TextSessionError::Storage(error)), effects);
                        return;
                    }
                }
                self.finish(Ok(TextSessionTurn { result, state: self.state.clone() }), effects);
                return;
            }
        };
        let output = output.or_else(|| match &result.outcome {
            RunOutcome::Completed { output: lk_agents_abi::agent::run::Output::Text(text) } => {
                Some(Value::String(text.clone()))
            }
            _ => None,
        });
        let Some(output) = output else {
            self.finish(
                Err(TextSessionError::Routing(RoutingError::MissingAcceptedOutput)),
                effects,
            );
            return;
        };
        let pending = self.state.frames[&parent_call.frame_id]
            .pending_calls
            .iter()
            .find(|p| p.parent == parent_call)
            .cloned();
        let Some(pending) = pending else {
            self.finish(
                Err(TextSessionError::Routing(RoutingError::InvalidState(
                    "child outcome has no pending parent call".into(),
                ))),
                effects,
            );
            return;
        };
        let text = match output {
            Value::String(text) => text,
            value => value.to_string(),
        };
        if let Err(error) = self.resolve_call(&pending, text, false, now, effects) {
            self.finish(Err(TextSessionError::Routing(error)), effects);
            return;
        }
        // Move, rather than alias, the two frames while transferring dialogue.
        let mut child = self.state.frames.remove(&active_id).expect("the active frame exists");
        child.completed = true;
        let parent =
            self.state.frames.get_mut(&parent_call.frame_id).expect("the parent frame exists");
        transfer(&mut child, parent);
        // The result and any selected dialogue are now in the parent. Retaining every
        // completed child would grow a long-lived conversation without bound.
        self.state.active_frame_id = parent_call.frame_id;
        self.state.ancestors.pop();
        self.next_child(now, effects);
    }

    fn resolve_call(
        &mut self,
        pending: &PendingChildCall,
        text: String,
        is_error: bool,
        now: Ts,
        effects: &mut Vec<SessionEffect>,
    ) -> Result<(), RoutingError> {
        let stamp = self.stamp(now);
        let parent =
            self.state.frames.get_mut(&pending.parent.frame_id).ok_or_else(|| {
                RoutingError::InvalidState("pending call has no parent frame".into())
            })?;
        let call = parent
            .history
            .iter()
            .find_map(|item| match item {
                ChatItem::FunctionCall(c) if c.id == pending.parent.call_id => Some(c),
                _ => None,
            })
            .ok_or_else(|| RoutingError::InvalidState("parent call is missing".into()))?;
        if parent
            .history
            .iter()
            .any(|item| matches!(item, ChatItem::FunctionCallOutput(o) if o.id == call.output_id()))
        {
            return Err(RoutingError::InvalidState("parent call already resolved".into()));
        }
        let mut output =
            FunctionCallOutput::new(call.output_id(), &call.call_id, text, is_error, stamp);
        output.name = call.name.clone();
        let event = lk_agents_abi::agent::run::StreamEvent::ToolResult {
            call_id: call.call_id.clone(),
            tool_name: call.name.clone(),
            status: if is_error {
                lk_agents_abi::agent::run::ToolResultStatus::Failed
            } else {
                lk_agents_abi::agent::run::ToolResultStatus::Completed
            },
            output: output.output.clone(),
        };
        parent.history.push(ChatItem::FunctionCallOutput(output));
        parent.pending_calls.retain(|p| p.parent != pending.parent);
        self.sequence += 1;
        effects.push(SessionEffect::Emit {
            sequence: self.sequence,
            frame_id: parent.id.clone(),
            agent_name: parent.agent_name.clone(),
            invocation_id: pending.invocation_id.clone(),
            event,
        });
        Ok(())
    }

    pub(super) fn stamp(&self, now: Ts) -> std::time::SystemTime {
        std::time::UNIX_EPOCH
            + std::time::Duration::try_from_secs_f64(self.wall.seconds(now).max(0.0))
                .unwrap_or_default()
    }
}

/// Retain the source even when phrasing or delivery fails. Presentation requires its own receipt.
pub(super) fn retain_input(
    frame: &mut SessionFrame,
    requirement: &lk_agents_abi::agent::run::RunInputRequirement,
    stamp: std::time::SystemTime,
) -> lk_agents_abi::session::InputRequestRef {
    let remote = frame.history.iter().find_map(|item| {
        let ChatItem::FunctionCall(call) = item else { return None };
        if call.id != requirement.call.id {
            return None;
        }
        let value = call.extra.get(crate::delegation::remote::retained::STATE_KEY)?;
        let state: crate::delegation::remote::InteractionState =
            serde_json::from_value(value.clone()).ok()?;
        state.input_request
    });
    remote.unwrap_or_else(|| {
        let id = format!("{}/input/{}", requirement.invocation_id, requirement.call.id);
        frame.input_requests.entry(id.clone()).or_insert_with(|| {
            lk_agents_abi::session::AgentInputRequest {
                state: lk_agents_abi::session::InputRequestState::Open,
                source_message: lk_agents_abi::ChatMessage::text(
                    id.clone(),
                    lk_agents_abi::Role::Assistant,
                    &requirement.question,
                    stamp,
                ),
                response_message: None,
                origin: Some(lk_agents_abi::session::ToolCallRef {
                    agent_id: frame.id.value.clone(),
                    call_id: requirement.call.id.clone(),
                }),
                presented_message_id: None,
            }
        });
        lk_agents_abi::session::InputRequestRef::Agent {
            agent_id: frame.id.value.clone(),
            request_id: id,
        }
    })
}

pub(super) fn retain_approvals(pending: &mut Vec<PendingChildCall>, previous: &[PendingChildCall]) {
    for call in previous {
        if call.status == ChildCallStatus::Approval
            && !pending.iter().any(|kept| kept.parent == call.parent)
        {
            pending.push(call.clone());
        }
    }
}
