use crate::agent_session::VoiceEvent;
use lk_agents_abi::AudioFrame;
use lk_agents_abi::resources::ResourceId;

/// How a sink takes a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delivery {
    /// Live audio (`StreamInput::push`).
    Live,
    /// Pre-connect audio: the sink keeps it in full (`StreamInput::head`).
    Head,
    /// Silence that stands in for nothing said: a bounded sink drops it first
    /// (`StreamInput::filler`).
    Filler,
}

/// What the runtime performs for the forwarder.
pub enum AudioInputRouterEffect {
    PushFrame {
        sink: ResourceId,
        frame: AudioFrame,
        delivery: Delivery,
    },
    /// Ending, not dropping: dropping a host stream doesn't close it.
    EndInput(ResourceId),
    /// Dropping without ending: the stream is closed before its input ends, so the provider is
    /// never asked to finalize what it heard.
    AbortInput(ResourceId),
    /// An input for the decision core, through the queue.
    ReportToSession(Box<VoiceEvent>),
}
