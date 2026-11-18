use lk_agents_abi::FunctionCall;
use serde_json::{Map, Value};

use super::RemoteInteraction;
use crate::a2a::rpc::TaskRef;

/// Transport correlation for observed work. Conversation continuation lives in native state.
pub fn call_extra(
    interaction: &RemoteInteraction,
    call: &FunctionCall,
    livekit: bool,
) -> Map<String, Value> {
    let state = interaction.snapshot();
    let mut call = call.clone();
    if let (Some(id), Some(context_id)) = (&state.pending_task_id, &state.context_id) {
        super::record_delegation(
            &mut call,
            &state.id,
            &TaskRef { id: id.clone(), context_id: context_id.clone() },
            livekit,
        );
    }
    call.extra
}
