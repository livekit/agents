//! `lk-agents-net`: the shared async transport of the I/O crates.
//!
//! - [`clock`]: the process's clocks, on Tokio time.
//! - [`attempts`]: the attempts of one request: phase limits, retries, total deadline, cleanup.
//! - [`monotone`]: a level-triggered cell whose value only moves forward.
//! - [`http`]: HTTPS exchanges reporting transport observations, under [`attempts`].
//! - [`source`]: a native source as one host consumer pulls and closes it.
//! - [`a2a_auth`]: endpoint and thread credentials over LiveKit tokens.
//! - [`a2a_client`]: the delegation A2A client, executing `lk_agents_core::a2a`'s decisions.
//! - [`session_report`]: the LiveKit Cloud session report's transcript and recording upload.
//! - [`blocking_io`]: the process's blocking I/O pool.
//! - [`dns`]: name resolution and TCP connects on the blocking I/O pool.
//! - [`env`]: what the hosting process lends a client.
//! - [`task`]: owned tasks with correlated cancellation and one observed outcome.
//! - [`sse`]: event-stream framing over response bytes.
//! - [`token`]: framework credentials signed with LiveKit's token crate.
//! - [`transport`]: the WebSocket seam, transport errors and waits bounded on a clock.
//! - [`websocket`]: the real WebSocket transport over `tokio-tungstenite`.
//! - [`webtransport`]: raw QUIC sessions with TLS and HTTP connect headers.

pub mod a2a_auth;
pub mod a2a_client;
pub mod attempts;
pub mod blocking_io;
pub mod clock;
pub mod credentials;
pub mod dns;
pub mod env;
pub mod http;
pub mod lock;
pub mod monotone;
pub mod session_report;
pub mod source;
pub mod sse;
pub mod task;
pub mod tcp;
pub mod token;
pub mod transport;
pub mod websocket;
pub mod webtransport;

pub mod connection_pool;

pub mod strings;
