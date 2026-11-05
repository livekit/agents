//! `stt/`: the STT seam, its events and the VAD-driven stream adapter.

pub mod events;
pub mod fallback;
pub mod inference;
pub mod stream_adapter;
#[path = "stt.rs"]
pub mod types;

/// Audio the fallback chain holds across reconnects: 2 seconds of 20 ms frames, oldest dropped
/// first.
pub const HELD_FRAMES: usize = 100;
