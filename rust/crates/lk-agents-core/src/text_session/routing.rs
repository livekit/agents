//! Definition and restored-state validation. Live context construction and save fencing belong
//! to the caller; the coordinator only checks the supplied names and scope references.

use std::collections::{BTreeMap, BTreeSet};

use lk_agents_abi::agent::AgentDefinition;
use lk_agents_abi::resources::LlmRef;

use super::{ChildCallStatus, TextSessionState};

pub use lk_agents_abi::text_session::{ChatHistory, LocalAgentTool, RoutingError};

#[derive(Debug, Clone)]
pub struct SessionAgent {
    pub definition: AgentDefinition,
    pub llm: Option<LlmRef>,
    /// Distinguishes explicit `output_type=str` from a standalone agent's implicit default.
    pub output_type_explicit: bool,
    /// Names index the existing FunctionToolInfo definitions, including schemas and approval.
    pub agent_tools: BTreeMap<String, LocalAgentTool>,
}

#[derive(Debug, Clone)]
pub struct TextSessionConfig {
    pub root_agent: String,
    pub agents: BTreeMap<String, SessionAgent>,
    pub max_depth: usize,
}

impl TextSessionConfig {
    pub fn new(root_agent: String, agents: BTreeMap<String, SessionAgent>) -> Self {
        Self { root_agent, agents, max_depth: 8 }
    }

    pub fn validate(&self) -> Result<(), RoutingError> {
        if self.max_depth == 0 {
            return Err(RoutingError::MaxDepth { max_depth: 0 });
        }
        if !self.agents.contains_key(&self.root_agent) {
            return Err(RoutingError::UnknownAgent(self.root_agent.clone()));
        }
        for (name, agent) in &self.agents {
            if name.is_empty() || *name != agent.definition.name {
                return Err(RoutingError::InvalidState("unstable agent name".into()));
            }
            crate::agent::validate(&agent.definition)
                .map_err(|e| RoutingError::InvalidState(format!("{e:?}")))?;
            for (tool_name, tool) in &agent.agent_tools {
                if !agent.definition.tools.iter().any(|t| t.spec.name == *tool_name) {
                    return Err(RoutingError::UnknownTool(tool_name.clone()));
                }
                let child = self
                    .agents
                    .get(&tool.agent_name)
                    .ok_or_else(|| RoutingError::UnknownAgent(tool.agent_name.clone()))?;
                if !child.output_type_explicit {
                    return Err(RoutingError::MissingOutputType(tool.agent_name.clone()));
                }
            }
        }
        Ok(())
    }

    pub(super) fn validate_state(
        &self,
        state: &TextSessionState,
        scopes: &BTreeSet<String>,
    ) -> Result<(), RoutingError> {
        let invalid = |reason: &str| RoutingError::InvalidState(reason.into());
        if state.session_id.is_empty() || state.execution_id.is_empty() {
            return Err(invalid("missing session or execution identity"));
        }
        if state.ancestors.len() + 1 > self.max_depth {
            return Err(RoutingError::MaxDepth { max_depth: self.max_depth });
        }
        let root = state.frames.get(&state.root_frame_id).ok_or_else(|| invalid("missing root"))?;
        if root.agent_name != self.root_agent || root.parent.is_some() {
            return Err(invalid("root definition or parent changed"));
        }
        let mut chain = state.ancestors.clone();
        chain.push(state.active_frame_id.clone());
        if chain.first() != Some(&state.root_frame_id)
            || chain.iter().collect::<BTreeSet<_>>().len() != chain.len()
        {
            return Err(invalid("invalid ancestor chain"));
        }
        for (id, frame) in &state.frames {
            if id != &frame.id {
                return Err(invalid("frame key differs from its identity"));
            }
            if !self.agents.contains_key(&frame.agent_name) {
                return Err(RoutingError::UnknownAgent(frame.agent_name.clone()));
            }
            if !scopes.contains(&frame.context_scope) {
                return Err(RoutingError::UnknownScope(frame.context_scope.clone()));
            }
            if let Some(parent) = &frame.parent {
                let parent_frame = state
                    .frames
                    .get(&parent.frame_id)
                    .ok_or_else(|| invalid("missing parent frame"))?;
                let call_exists = parent_frame.history.iter().any(|item| {
                    matches!(item,
                    lk_agents_abi::ChatItem::FunctionCall(c) if c.call_id == parent.call_id)
                });
                if !call_exists {
                    return Err(invalid("missing originating call"));
                }
                if !frame.completed
                    && !parent_frame.pending_calls.iter().any(|p| {
                        p.parent == *parent
                            && p.status == (ChildCallStatus::Running { frame_id: id.clone() })
                    })
                {
                    return Err(invalid("child has no pending parent call"));
                }
            }
            for pending in &frame.pending_calls {
                if pending.parent.frame_id != *id
                    || pending.execution_id != state.execution_id
                    || pending.invocation_id.is_empty()
                    || !pending.validated_arguments.is_object()
                {
                    return Err(invalid("invalid pending child identity or arguments"));
                }
                let tool = self.agents[&frame.agent_name]
                    .agent_tools
                    .get(&pending.tool_name)
                    .ok_or_else(|| RoutingError::UnknownTool(pending.tool_name.clone()))?;
                if tool.agent_name != pending.agent_name {
                    return Err(invalid("pending child definition changed"));
                }
                if pending.identity != crate::approval::canonical_json(&pending.validated_arguments)
                {
                    return Err(invalid("pending validated identity changed"));
                }
                let calls: Vec<_> = frame
                    .history
                    .iter()
                    .filter_map(|item| match item {
                        lk_agents_abi::ChatItem::FunctionCall(call)
                            if call.call_id == pending.parent.call_id =>
                        {
                            Some(call)
                        }
                        _ => None,
                    })
                    .collect();
                if calls.len() != 1 || calls[0].name != pending.tool_name {
                    return Err(invalid("pending child has no unique originating call"));
                }
                let answered = frame.history.iter().any(|item| matches!(item,
                    lk_agents_abi::ChatItem::FunctionCallOutput(output) if output.call_id == pending.parent.call_id));
                if answered != (pending.status == ChildCallStatus::Approval) {
                    return Err(invalid("pending child call has an inconsistent result"));
                }
            }
        }
        for (index, id) in chain.iter().enumerate() {
            let frame = state.frames.get(id).ok_or_else(|| invalid("missing active ancestor"))?;
            if frame.completed {
                return Err(invalid("completed frame in active chain"));
            }
            if index > 0 && frame.parent.as_ref().map(|p| &p.frame_id) != chain.get(index - 1) {
                return Err(invalid("ancestor does not match parent"));
            }
            if index + 1 == chain.len()
                && frame.pending_calls.iter().any(|p| p.status != ChildCallStatus::Approval)
            {
                return Err(invalid("active frame has an unresolved child"));
            }
        }
        Ok(())
    }
}
