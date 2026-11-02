//! Microphone input routing: pure fan-out, resampling, and gates beside voice coordination.
//!
//! `machine::AudioInputRouter` produces `effects::AudioInputRouterEffect`; runtime performs I/O.
//! AudioInput remains the source interface, not this routing state machine.

pub mod effects;
pub mod machine;

pub use effects::AudioInputRouterEffect;
pub use machine::{
    AudioFanOut, AudioInputRouter, AudioInputRouterCommand, FrameShape, InputCensus,
};
