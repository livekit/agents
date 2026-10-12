//! The voice behavioural suite: one table of inputs, one runner, one set of contracts.
//!
//! `input.rs` is everything the test can send, `record.rs` everything a run leaves behind,
//! `drive.rs` a real session it is applied to. `rules/` are the guarantees, one per file,
//! discovered by `build.rs`; `check.rs` is how one reports; `authored.rs` builds the recordings
//! their examples use. `search.rs` is the campaign, `fakes/` the controlled providers, and
//! `replay.rs` the saved cases in `replay/`.
//!
//! What is under test is the driver and only the driver. The fakes are dumb: they deliver what
//! the case chose and record what they were given, and every event at a provider boundary -- a
//! transcript, a detector's tick, a playout report, a socket drop -- is an input the case sends,
//! never something a fake derives. A run is deterministic: paused virtual time, a seeded
//! scheduler the case carries, ids scoped to the session.
//!
//! The rules run once per run, over the finished record, and state properties of the whole
//! timeline. `cargo test` runs the rules' own examples, the determinism check and the saved
//! corpus; the search lives behind `--ignored`: `tests/survey.rs` for reach and findings across
//! seeds, `tests/soak.rs` for what only forty turns show.

pub mod authored;
pub mod check;
pub mod drive;
pub mod fakes;
pub mod input;
pub mod record;
pub mod replay;
pub mod rules;
pub mod search;
