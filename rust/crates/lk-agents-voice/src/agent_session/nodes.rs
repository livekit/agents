//! `transcription_node` and `realtime_audio_output_node`: the nodes an agent overrides on the way
//! out, as the decision core records them per agent.

use lk_agents_abi::voice::nodes::NodeFactory;
use std::sync::Arc;

/// The nodes an agent overrides on the way out, opened at each use from the agent a reply was
/// created under: `transcription_node` per spoken segment over the transcription output,
/// `realtime_audio_output_node` per realtime generation over the audio output.
#[derive(Default, Clone)]
pub struct OutputNodes {
    pub transcription: Option<Arc<dyn NodeFactory>>,
    pub realtime_audio: Option<Arc<dyn NodeFactory>>,
}
