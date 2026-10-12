//! The host-backed half of §1.0 for Node. `NodeTts` is built; `NodeLlm`/`NodeStt`/`NodeVad` are
//! not -- see the M7 report for why one provider was the brief.
//!
//! # What is the same as `lk-agents-py`, and it is most of it
//!
//! The seam is at the **factory method**, never at the inner `_run()` (§1.1): `TTS.synthesize()`
//! and `TTS.stream()` are wrapped, and the plugin's own stream object -- with its retry loop, its
//! metrics tee and its channel plumbing -- keeps running in JavaScript. That is forced, not
//! chosen: the factory is where the plugin constructs *its own* subclass, and Rust cannot
//! construct that generically in either language.
//!
//! `conn_options.maxRetry = 0` on every wrapped provider is enforced here for the same reason
//! (§1.1's no-double-retry rule), and the `TtsCapabilities::streaming` read is per **instance**,
//! at construction, because six plugins decide it from a constructor flag.
//!
//! # What is different, and why each difference is Node's and not the core's
//!
//! | `lk-agents-py` | `lk-agents-node` | cause |
//! |---|---|---|
//! | `HostHandle { obj, TaskLocals }` -- one Python object plus its loop | [`host::NodeHandle`] -- a bundle of `ThreadsafeFunction`s | there is no "call a method by name on a JS object from a tokio worker". A tsfn is the only legal crossing, and it is created from a *function*, so the shim hands over one function per operation. |
//! | `call_soon_threadsafe` for every sync push | `tsfn.call(..., NonBlocking)` | same hazard (§1.2d), different primitive. Both mean "handed to the loop", not "the host accepted it". |
//! | `TaskLocals::with_running_loop` captured at construction (§1.6 #5) | **nothing to capture** | a tsfn keeps the env alive by itself. This whole hazard class does not exist in Node -- see the report. |
//! | streams identified by a `Py<PyAny>` Rust holds | streams identified by an `i64` **JS** holds | thread affinity again, as in [`crate::host_table`]. |
//! | errors captured into a Rust side table | errors classified in JS, scalars + a `hostRef` cross | [`crate::errors`]. |
//!
//! None of these touched a core crate.

pub mod host;
mod tts;

pub use host::{NodeHandle, TtsCallbacks};
pub use tts::NodeTts;
