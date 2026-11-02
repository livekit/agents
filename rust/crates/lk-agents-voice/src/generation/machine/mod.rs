//! Synchronous generation owners. Effects are declared in the sibling effects module.
use super::*;

pub mod audio_playback;
pub mod forward;
pub mod llm;
pub mod pipeline;
pub(in crate::generation) mod playout;
pub mod preparation;
pub mod realtime_generation;
pub mod tts;

pub use audio_playback::AudioPlayback;
pub use preparation::GenerationPreparation;
pub use realtime_generation::RealtimeGeneration;
