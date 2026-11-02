#![allow(
    clippy::disallowed_macros,
    reason = "Clippy checks macro bans at crate scope; exact legacy macro sites are ratcheted."
)]

//! Voice conversation decisions over supplied events and time; runtime executes their effects.
//!
//! Start with [`agent_session::machine::VoiceDriver`], the ordered session coordinator.
//! [`agent_activity`] holds the active agent's floor, hooks and reply coordination;
//! its replies compose the shared [`lk_agents_core::agent_run::AgentRun`].
//!
//! | Area | Read here |
//! | --- | --- |
//! | Session inputs, transitions and runtime work | [`agent_session`] and [`effects`] |
//! | Recognition and user-turn decisions | [`audio_recognition::machine::AudioRecognition`] |
//! | Generation, playback authorization and heard audio | [`generation`] and [`generation::machine`] |
//! | Transcript timing and publication | [`transcription`] and [`room_io`] |
//! | Frame routing and buffering | [`data_plane`] |
//! | Tool execution policy and run results | [`tool_executor`] and [`run_result`] |
//! | Realtime provider observations | [`agent_activity::realtime`] |
//!
//! The corresponding I/O owners live in `lk-agents-runtime/src/voice`, `room_io` and `avatar`.
//! Use `docs/internals/rust-code-map.md` for the Python-to-Rust reading paths and test commands.
//! Module size is not an ownership signal: effect declarations, port contracts and machine maps
//! deliberately remain separate even when small.

extern crate self as lk_agents_voice;

pub mod agent;
pub mod agent_activity;
pub mod agent_session;
pub mod amd;
pub mod audio;
pub mod audio_recognition;
pub mod av_sync;
pub mod avatar;
pub mod background_audio;
pub mod data_plane;
pub mod effects;
pub mod endpointing;
pub mod events;
pub mod experts;
pub mod filler_scheduler;
pub mod generation;
pub mod ivr;
pub mod keyterm_detection;
pub mod playback;
pub mod provider_layers;
pub mod remote_session;
pub mod room_io;
pub mod run_result;
pub mod speech_handle;
pub mod telemetry;
pub mod tool_executor;
pub mod trace;
pub mod transcription;
pub mod turn;
pub mod workflows;

pub mod tokenize;
