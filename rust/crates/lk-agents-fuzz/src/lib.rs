//! The voice behavioural suite: one table of inputs, one runner, one set of contracts.
//!
//! `input.rs` is everything the test can send, `record.rs` everything a run leaves behind,
//! `drive.rs` a real session it is applied to. `rules/` are the guarantees, one per file,
//! discovered by `build.rs`; `check.rs` is how one reports; `authored.rs` builds the recordings
//! their examples use. `search.rs` is the campaign and `fakes/` the controlled providers.
//!
//! What is under test is the voice core through the API its host calls (`SessionApi`), and
//! only that: the room is the fake room, every provider a fake. The fakes are dumb: they deliver what
//! the case chose and record what they were given, and every event at a provider boundary -- a
//! transcript, a detector's tick, a playout report, a socket drop -- is an input the case sends,
//! never something a fake derives. A run is deterministic: paused virtual time, a seeded
//! scheduler the case carries, ids scoped to the session.
//!
//! Scope (owner decision 28): the DST exercises sans-IO cores with realistic fakes, BasicPOS (and
//! the arrival baseline) over the cores' event deliveries; it depends only on cores and the ABI in
//! its final form, and runs no deterministic executor or POS over runtime adapter tasks. Runtime
//! adapters are verified outside it (conformance with cancellation at every await point, stress,
//! real-peer tests, a regression per bug: K-30). The voice harness still runs the core's policy
//! tasks and some adapters, under the transitional task bridge, until they become core machines;
//! `boundary.rs` states what each campaign simulates and leaves outside, and every campaign prints
//! it. One crate, a module per subject: voice today; worker, storage and A2A to come (K-2..K-4).
//!
//! The rules run once per run, over the finished record, and state properties of the whole
//! timeline. `cargo test` runs the rules' own examples, the determinism check and the regression
//! tests; the search lives behind `--ignored`: `tests/survey.rs` for reach and findings across
//! seeds, `tests/soak.rs` for what only forty turns show. A seed alone reproduces its run:
//! `VOICE_SEED=<n>` reruns one.

pub mod authored;
pub mod boundary;
pub mod check;
pub mod corpus;
pub mod drive;
pub mod fakes;
pub mod inference;
pub mod input;
pub mod record;
pub mod rules;
pub mod sched;
pub mod search;
pub mod storage;
pub mod worker;
