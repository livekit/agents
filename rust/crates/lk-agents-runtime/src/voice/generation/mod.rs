//! Generation execution over the session's injected ports.

mod forward;
mod llm;
mod output;
pub mod pipeline_reply;
pub(crate) mod playout;
pub mod realtime_generation;
pub(crate) mod tools;
mod tts;

use crate::voice::resources::Resources;
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::{Llm, Tts};
use lk_agents_voice::generation::ReplyContext as Plan;
use std::sync::Arc;

/// A reply plan bound to the resources it was admitted with. Follow-ups retain this snapshot.
pub(crate) struct ReplyExecution {
    pub plan: Plan,
    pub resources: ReplyResources,
}

#[derive(Clone)]
pub(crate) struct ReplyResources {
    pub llm: Option<Arc<dyn Llm>>,
    pub tts: Option<Arc<dyn Tts>>,
    pub dynamic_instructions: Option<Arc<dyn lk_agents_abi::agent::HostInstructions>>,
    pub toolset: Arc<lk_agents_voice::tool_executor::Toolset>,
    pub tools_host_extra: Vec<lk_agents_abi::HostRef>,
    pub output_nodes: crate::voice::nodes::OutputNodes,
}

impl ReplyExecution {
    pub fn bind(
        plan: Plan,
        llm: Option<ResourceId>,
        tts: Option<ResourceId>,
        resources: &Resources,
    ) -> Result<Self, lk_agents_voice::agent_session::DriverError> {
        use lk_agents_voice::agent_session::DriverError;
        let snapshot = ReplyResources {
            llm: llm
                .map(|id| resources.llm(id).ok_or(DriverError::Missing("registered LLM")))
                .transpose()?,
            tts: tts
                .map(|id| resources.tts(id).ok_or(DriverError::Missing("registered TTS")))
                .transpose()?,
            dynamic_instructions: plan
                .providers
                .dynamic_instructions
                .map(|id| {
                    resources
                        .instructions(id)
                        .ok_or(DriverError::Missing("registered instructions callback"))
                })
                .transpose()?,
            toolset: plan.providers.toolset.clone(),
            tools_host_extra: plan.providers.tools_host_extra.clone(),
            output_nodes: crate::voice::nodes::OutputNodes {
                transcription: plan
                    .providers
                    .output_nodes
                    .transcription
                    .map(|id| {
                        resources
                            .transcription_node(id)
                            .ok_or(DriverError::Missing("registered transcription node"))
                    })
                    .transpose()?,
            },
        };
        Ok(Self { plan, resources: snapshot })
    }
}
