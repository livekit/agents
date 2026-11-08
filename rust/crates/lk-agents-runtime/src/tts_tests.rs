//! The inference speech clients with the core's speech text: gateway and fallback synthesis and
//! the sentence adapter.

use lk_agents_runtime::inference::fallback::FallbackTts;
use lk_agents_runtime::inference::tts::GatewayTts;

#[path = "tts_fallback_tests.rs"]
mod fallback;
#[path = "tts_gateway_tests.rs"]
mod gateway;
#[path = "tts_sentence_tests.rs"]
mod sentence_adapter;
