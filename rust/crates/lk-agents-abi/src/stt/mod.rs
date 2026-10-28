//! `stt/`: the STT seam, its events and the VAD-driven stream adapter.

pub mod events;
pub mod stream_adapter;
pub mod stt;

/// Audio held across recognition reconnects: 2 seconds of 20 ms frames, oldest dropped first.
/// The retry wrapper and fallback chain use the same bound.
pub const HELD_FRAMES: usize = 100;
