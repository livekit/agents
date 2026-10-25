//! `voice/`: the session-facing boundary records. One `#[derive(Host)]` declaration per shape that
//! crosses the seam; `make generated` writes the pydantic and TypeScript twins. A declaration with
//! no Rust caller is not dead code.

pub mod amd;
pub mod calls;
pub mod events;
pub mod host_policy;
pub mod io;
pub mod room_io;
pub mod rtc;
pub mod run_result;
pub mod session_effects;
pub mod session_options;
pub mod workflows;
