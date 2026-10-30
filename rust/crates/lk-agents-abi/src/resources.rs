//! The providers a session uses, by id. The host registers each one when it hands it over and
//! puts the typed reference in its command: the core decides with the reference (its id and what
//! the provider said about itself at registration) and names the id in the effects that reach it.
//! The runtime's table resolves an id to the provider for whoever performs those effects. The
//! microphone and the audio sinks it feeds are registered the same way, so the data plane names
//! them by id.

use crate::llm::realtime::RealtimeCapabilities;
use crate::voice::io::AudioOutputCapabilities;
use crate::{ProviderId, SttCapabilities, TtsCapabilities, VadCapabilities};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceId(pub u64);

#[derive(Debug, Clone, PartialEq)]
pub struct SttRef {
    pub id: ResourceId,
    pub provider: ProviderId,
    pub capabilities: SttCapabilities,
    /// The object registered: the same recogniser handed back under a new id is not reopened.
    pub instance: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VadRef {
    pub id: ResourceId,
    pub provider: ProviderId,
    pub capabilities: VadCapabilities,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlmRef {
    pub id: ResourceId,
    pub provider: ProviderId,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TtsRef {
    pub id: ResourceId,
    pub provider: ProviderId,
    pub capabilities: TtsCapabilities,
    pub sample_rate: u32,
    pub num_channels: u16,
    /// The markup dialect it speaks (`markup._provider_key()`).
    pub markup: crate::tts::markup::Markup,
    /// [`Tts::lowers_markup`].
    pub lowers_markup: bool,
}

/// The session a realtime model opened: what it said it can do when it opened.
#[derive(Debug, Clone, PartialEq)]
pub struct RealtimeRef {
    pub id: ResourceId,
    /// The model that opened it, which its responses are counted against.
    pub provider: ProviderId,
    pub capabilities: RealtimeCapabilities,
}

/// A video input, registered under a new id each time it is handed over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoInputRef {
    pub id: ResourceId,
    /// The object registered: the input already read, handed back, is not reattached.
    pub instance: usize,
}

/// The agent's audio output: what it said about itself when handed over.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioOutputRef {
    pub id: ResourceId,
    pub label: String,
    /// Taken over its whole chain: `pause` only if every link can pause.
    pub capabilities: AudioOutputCapabilities,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextOutputRef {
    pub id: ResourceId,
    pub label: String,
}
