//! `lk-agents-net`: the shared async transport of the I/O crates.
//!
//! - [`clock`]: the process's clocks, on Tokio time.
//! - [`http`]: HTTPS exchanges reporting transport observations, under `lk_agents_core::attempts`.
//! - [`source`]: a native source as one host consumer pulls and closes it.
//! - [`a2a_auth`]: endpoint and thread credentials over LiveKit tokens.
//! - [`a2a_client`]: the delegation A2A client, executing `lk_agents_core::a2a`'s decisions.
//! - [`otlp`]: the OTLP exporter to LiveKit Cloud and the application's endpoint.
//! - [`session_report`]: the LiveKit Cloud session report's transcript and recording upload.
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
pub mod blocking_io;
pub mod clock;
pub mod dns;
pub mod env;
pub mod http;
pub mod otlp;
pub mod session_report;
pub mod source;
pub mod sse;
pub mod task;
pub mod tls;
pub mod token;
pub mod transport;
pub mod websocket;

pub mod connection_pool;

pub mod strings;
