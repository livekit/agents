//! TTS PCM framing. `machine::AudioEmitter` emits `effects::AudioEmitterEffect`.
//! Inference supplies decoding, ordered delivery, and the scheduled-flush timer.

pub mod effects;
pub mod machine;

pub use effects::AudioEmitterEffect;
pub use machine::AudioEmitter;
