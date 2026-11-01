//! `transcription_node`: the node an agent overrides on the way
//! out, as the decision core records them per agent.

use lk_agents_abi::resources::ResourceId;

/// The nodes an agent overrides on the way out, opened at each use from the agent a reply was
/// created under: `transcription_node` per spoken segment over the transcription output.
#[derive(Default, Clone)]
pub struct OutputNodes {
    pub transcription: Option<ResourceId>,
}
