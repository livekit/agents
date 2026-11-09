//! Voice conversation decisions over supplied events and time; the runtime executes their effects.
//! Start with [`agent_session::machine::VoiceDriver`], the ordered session coordinator.

extern crate self as lk_agents_voice;

pub mod agent;
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
