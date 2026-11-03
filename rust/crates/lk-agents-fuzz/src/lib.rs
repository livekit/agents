//! The voice behavioural suite: one table of inputs, one runner, one set of contracts.
//!
//! `input.rs` is everything the test can send, `record/` everything a run leaves behind,
//! `drive/` a real session it is applied to. `rules/` are the guarantees, one per file,
//! discovered by `build.rs`; `check.rs` is how one reports; `authored.rs` builds the recordings
//! their examples use. `search/` is the campaign and `fakes/` the controlled providers and room.
//!
//! What is under test is the voice core through the API its host calls (`VoiceSession`), and
//! the room is the fake room, every provider a fake. Fakes record calls and implement controlled
//! port behavior: cases deliver transcripts, detector results and faults; paced room, avatar,
//! console and realtime models also produce boundary events on the injected clock. They do not
//! decide framework scheduling, turn admission or interruption policy. A run is deterministic: paused virtual time, a seeded
//! scheduler the case carries, ids scoped to the session.
//!
//! Voice runs the production VoiceSession and runtime executor over controlled RTC/provider ports.
//! Arrival, BasicPOS and task-bridge scheduling explore event delivery and injected task execution.
//! The task bridge remains useful for I/O cancellation and cleanup races even when every framework
//! decision is synchronous. Room tests exercise production room I/O over the same fake RTC ports.
//! Backend A2A/TextSession execution, native provider transports, host bindings and actual
//! sockets/backends have separate component and conformance suites. `boundary.rs` states the
//! simulation boundary and every campaign prints it. See the crate README for commands and a map.
//!
//! The rules run once per run, over the finished record, and state properties of the whole
//! timeline. `cargo test` runs the rules' own examples, the determinism check and the regression
//! tests; the search lives behind `--ignored`: `src/survey_tests.rs` for reach and findings across
//! seeds, `src/soak_tests.rs` for what only forty turns show. A seed alone reproduces its run:
//! `VOICE_SEED=<n>` reruns one.

pub mod authored;
pub mod boundary;
pub mod check;
pub mod drive;
pub mod fakes;
pub mod input;
mod manual_spans;
pub mod record;
pub mod rules;
pub mod sched;
pub mod search;
