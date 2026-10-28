//! The inference speech clients composed with the voice core's speech text, as this process wires
//! them (`Process`): gateway and fallback synthesis, the sentence adapter and the judges' prompts.

type NativeTts = lk_agents_runtime::inference::tts::NativeTts<lk_agents_runtime::process::Process>;
type FallbackTts =
    lk_agents_runtime::inference::fallback::FallbackTts<lk_agents_runtime::process::Process>;

mod fallback;
mod judge;
mod native;
mod sentence_adapter;
