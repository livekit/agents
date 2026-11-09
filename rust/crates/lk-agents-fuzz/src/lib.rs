//! The voice deterministic simulation test: the production `VoiceSession` and runtime executor
//! over fake providers and a fake room, on paused virtual time and a seeded scheduler.
//! `rules/` holds the guarantees, checked once per run over the finished record.

#![allow(
    clippy::unwrap_used,
    clippy::print_stdout,
    reason = "the DST harness: a broken fake is a test failure, and runs report to stdout"
)]

pub mod authored;
mod calibration {
    include!(concat!(env!("OUT_DIR"), "/calibration.rs"));
}
pub mod boundary;
pub mod check;
pub mod drive;
pub mod fakes;
pub mod input;
pub mod record;
pub mod rules;
pub mod sched;
pub mod search;
