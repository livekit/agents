//! Transcript lifecycle and its private per-segment pacing state.
pub mod lifecycle;
pub(in crate::transcription) mod pacer;
pub mod synchronizer;

pub use lifecycle::TranscriptSync;
