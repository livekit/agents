//! The inference speech clients composed with the voice core's speech text, as this process wires
//! them (`Process`): gateway and fallback synthesis, the sentence adapter and the judges' prompts.

use lk_agents_runtime::inference::fallback::FallbackTts;
use lk_agents_runtime::inference::tts::GatewayTts;

/// The voice core's speech text, as this process lends it.
fn speech() -> lk_agents_abi::tts::text::SpeechTextProcessors {
    lk_agents_runtime::process::env().speech_text
}

mod fallback;
mod judge;
mod gateway;
mod sentence_adapter;
