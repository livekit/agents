#![allow(
    clippy::disallowed_macros,
    reason = "Clippy checks macro bans at crate scope; exact legacy macro sites are ratcheted."
)]

extern crate self as lk_agents_voice;

pub mod background_audio;
pub mod agent;
pub mod agent_activity;
pub mod agent_session;
pub mod amd;
pub mod audio;
pub mod avatar;
pub mod avatar_runner;
pub mod audio_recognition;
pub mod data_plane;
pub mod endpointing;
pub mod events;
pub mod filler_scheduler;
pub mod generation;
pub mod io;
pub mod ivr;
pub mod keyterm_detection;
pub mod recorder_io;
pub mod remote_session;
pub mod room_io;
pub mod run_result;
pub mod runtime;
pub mod speech_handle;
pub mod telemetry;
pub mod tool_executor;
pub mod trace;
pub mod transcription;
pub mod turn;
pub mod workflows;

