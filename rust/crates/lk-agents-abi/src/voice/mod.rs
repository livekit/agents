//! `voice/`: the session-facing boundary records. One `#[derive(Host)]` declaration per shape that
//! crosses the seam; `make generated` writes the pydantic and TypeScript twins. A declaration with
//! no Rust caller is not dead code.

pub mod amd;
pub mod avatar;
pub mod background_audio;
pub mod calls;
pub mod events;
pub mod io;
pub mod nodes;
pub mod room_io;
pub mod rtc;
pub mod session;
pub mod session_effects;
pub mod session_options;
pub mod speech;
pub mod workflows;

pub mod console;

pub mod room_owner;
