//! The inference speech clients with the core's speech text: gateway and fallback synthesis, the
//! sentence adapter and the judges' prompts.

use lk_agents_runtime::inference::fallback::FallbackTts;
use lk_agents_runtime::inference::tts::GatewayTts;

mod fallback;
mod judge;
mod gateway;
mod sentence_adapter;
