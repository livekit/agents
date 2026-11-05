//! Microphone-routing work produced by `machine::AudioInputRouter`.
//! Runtime's voice executor delivers frames, closes inputs, and reports facts to VoiceDriver.

use crate::agent_session::VoiceEvent;
use lk_agents_abi::AudioFrame;
use lk_agents_abi::resources::ResourceId;

/// What the runtime performs for the forwarder.
pub enum AudioInputRouterEffect {
    PushFrame {
        sink: ResourceId,
        frame: AudioFrame,
        /// Pre-connect audio: the sink keeps it in full (`StreamInput::head`).
        head: bool,
    },
    /// Ending, not dropping: dropping a host stream doesn't close it.
    EndInput(ResourceId),
    /// An input for the decision core, through the queue.
    ReportToSession(Box<VoiceEvent>),
}
