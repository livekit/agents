//! Voice session state-machine owners and ordered coordination.
//!
//! `VoiceDriver` consumes session inputs/time and produces `VoiceEffect`; runtime executes I/O.
//! Coordination keeps existing FIFO and pending-operation semantics private to this domain.

pub mod activity;
pub(crate) mod coordination;
pub mod driver;
pub mod output;

pub use driver::*;
