//! Sequential child admission and once-only parent-call resolution. A parent is invoked only
//! after every child call in its checkpoint has a result.

use lk_agents_abi::agent::run::{RunOutcome, RunResult};
use lk_agents_abi::{ChatItem, ChatMessage, FunctionCallOutput, Role, Ts};
use serde_json::Value;

use super::{
    ChildCallStatus, PendingChildCall, RoutingError, SessionFrame, TextSession, TextSessionEffect,
    TextSessionError, TextSessionTurn, history,
};

impl TextSession {
    pub(super) fn next_child(&mut self, now: Ts, effects: &mut Vec<TextSessionEffect>) {
        loop {
            let parent_id = self.state.active_frame_id.clone();
            let pending = self.state.frames[&parent_id].pending_calls.first().cloned();
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
            let admission = self.admit_child(&pending);
            let scope = match admission {
                Ok(scope) => scope,
                Err(error) => {
                    effects.push(TextSessionEffect::RoutingRefused {
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
            };
            let parent = &self.state.frames[&parent_id];
            let tool = &self.config.agents[&parent.agent_name].agent_tools[&pending.tool_name];
            let history = history::initial(&parent.history, tool.chat_history);
            let merge_chat_history = tool.merge_chat_history;
            let separate_context = self.separate_contexts && tool.context_scope.is_none();
            let frame_id = super::FrameId { value: self.mint("frame") };
            let scope = if separate_context {
                self.scopes.insert(frame_id.value.clone());
                frame_id.value.clone()
            } else {
                scope
            };
            let task_id = self.mint("task");
            let mut starting_history_ids = history
                .iter()
                .map(|i| i.id().to_owned())
                .collect::<std::collections::BTreeSet<_>>();
            starting_history_ids.insert(task_id.clone());
            let task = ChatMessage::text(
                task_id,
                Role::User,
                pending.validated_arguments.to_string(),
                self.stamp(now),
            );
            let frame = SessionFrame {
                id: frame_id.clone(),
                agent_name: pending.agent_name.clone(),
                history,
                parent: Some(pending.parent.clone()),
                context_scope: scope,
                completed: false,
                pending_calls: Vec::new(),
                answer_tool: None,
                visible_dialogue: Vec::new(),
                starting_history_ids,
                transferred_message_ids: Default::default(),
                merge_chat_history,
            };
            self.state.frames.get_mut(&parent_id).unwrap().pending_calls[0].status =
                ChildCallStatus::Running { frame_id: frame_id.clone() };
            self.state.frames.insert(frame_id.clone(), frame);
            self.state.ancestors.push(parent_id);
            self.state.active_frame_id = frame_id;
            self.invoke(Some(task), now, effects);
            return;
        }
    }

    fn admit_child(&self, pending: &PendingChildCall) -> Result<String, RoutingError> {
        if self.state.ancestors.len() + 1 >= self.config.max_depth {
            return Err(RoutingError::MaxDepth { max_depth: self.config.max_depth });
        }
        let parent = &self.state.frames[&pending.parent.frame_id];
        let tool = self.config.agents[&parent.agent_name]
            .agent_tools
            .get(&pending.tool_name)
            .ok_or_else(|| RoutingError::UnknownTool(pending.tool_name.clone()))?;
        if !self.config.agents.contains_key(&pending.agent_name) {
            return Err(RoutingError::UnknownAgent(pending.agent_name.clone()));
        }
        let scope = tool.context_scope.as_ref().unwrap_or(&parent.context_scope);
        if !self.scopes.contains(scope) {
            return Err(RoutingError::UnknownScope(scope.clone()));
        }
        Ok(scope.clone())
    }

    pub(super) fn settled(
        &mut self,
        result: RunResult,
        output: Option<Value>,
        approvals: Vec<PendingChildCall>,
        now: Ts,
        effects: &mut Vec<TextSessionEffect>,
    ) {
        let active_id = self.state.active_frame_id.clone();
        let frame = self.state.frames.get_mut(&active_id).unwrap();
        frame.history = result.all_messages.clone();
        frame.pending_calls = approvals;
        frame.answer_tool = match &result.outcome {
            RunOutcome::InputRequired { answer_tool, .. } => answer_tool.clone(),
            _ => None,
        };
        if result.interrupted
            || matches!(result.outcome, RunOutcome::InputRequired { .. })
            || frame.parent.is_none()
        {
            self.finish(Ok(TextSessionTurn { result, state: self.state.clone() }), effects);
            return;
        }
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
        let parent_call = frame.parent.clone().unwrap();
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
        let mut child = self.state.frames.remove(&active_id).unwrap();
        child.completed = true;
        let parent = self.state.frames.get_mut(&parent_call.frame_id).unwrap();
        history::transfer(&mut child, parent);
        self.state.frames.insert(active_id, child);
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
        effects: &mut Vec<TextSessionEffect>,
    ) -> Result<(), RoutingError> {
        let stamp = self.stamp(now);
        let parent = self.state.frames.get_mut(&pending.parent.frame_id).unwrap();
        if parent.history.iter().any(|item| {
            matches!(item,
            ChatItem::FunctionCallOutput(o) if o.call_id == pending.parent.call_id)
        }) {
            return Err(RoutingError::InvalidState("parent call already resolved".into()));
        }
        let call = parent
            .history
            .iter()
            .find_map(|item| match item {
                ChatItem::FunctionCall(c) if c.call_id == pending.parent.call_id => Some(c),
                _ => None,
            })
            .ok_or_else(|| RoutingError::InvalidState("parent call is missing".into()))?;
        let mut output = FunctionCallOutput::new(
            format!("{}_out", call.id),
            &call.call_id,
            text,
            is_error,
            stamp,
        );
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
        effects.push(TextSessionEffect::Emit {
            sequence: self.sequence,
            frame_id: parent.id.clone(),
            agent_name: parent.agent_name.clone(),
            invocation_id: pending.invocation_id.clone(),
            event,
        });
        Ok(())
    }

    fn stamp(&self, now: Ts) -> std::time::SystemTime {
        std::time::UNIX_EPOCH
            + std::time::Duration::try_from_secs_f64(self.wall.seconds(now).max(0.0))
                .unwrap_or_default()
    }
}
