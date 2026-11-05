//! `lk-agents-net`: the shared async transport of the I/O crates.
//!
//! - [`clock`]: the process's clocks, on Tokio time.
//! - [`http`]: HTTPS exchanges reporting transport observations, under `lk_agents_core::attempts`.
//! - [`audio_drops`]: a media queue consumer's dropped audio, as the `audio_dropped` metric.
//! - [`source`]: a native source as one host consumer pulls and closes it.
//! - [`a2a_auth`]: endpoint and thread credentials over LiveKit tokens.
//! - [`a2a_client`]: the delegation A2A client, executing `lk_agents_core::a2a`'s decisions.
//! - [`telemetry`]: the process tracer, the span context and the spans scoped to async work; the
//!   OTLP exporter to LiveKit Cloud and the application's endpoint ([`telemetry::otlp`]); the
//!   LiveKit Cloud session report's records and recording upload ([`telemetry::session_report`]).
//! - [`blocking_io`]: the process's blocking I/O pool.
//! - [`dns`]: name resolution and TCP connects on the blocking I/O pool.
//! - [`env`]: what the hosting process lends a client.
//! - [`task`]: owned tasks with correlated cancellation and one observed outcome.
//! - [`sse`]: event-stream framing over response bytes.
//! - [`token`]: framework credentials signed with LiveKit's token crate.
//! - [`transport`]: the WebSocket seam, transport errors and waits bounded on a clock.
//! - [`tls`]: the shared native trust store and TLS configuration.
//! - [`websocket`]: the real WebSocket transport over `tokio-tungstenite`.

pub mod a2a_auth;
pub mod a2a_client;
pub mod audio_drops;
pub mod blocking_io;
pub mod clock;
pub mod dns;
pub mod env;
pub mod http;
pub mod source;
pub mod sse;
pub mod task;
pub mod telemetry;
pub mod tls;
pub mod token;
pub mod transport;
pub mod websocket;

pub mod connection_pool;

pub mod strings;

/// This crate, under the name `#[span]` functions expand to.
extern crate self as lk_agents_net;
