//! `lk-agents-net`: the shared async transport of the I/O crates (decision 77).
//!
//! - [`task`]: owned tasks with correlated cancellation and one observed outcome.
//! - [`sse`]: event-stream framing over response bytes.
//! - [`token`]: framework credentials signed with LiveKit's token crate.
//! - [`transport`]: the WebSocket seam, transport errors and waits bounded on a clock.
//! - [`websocket`]: the real WebSocket transport over `tokio-tungstenite`.

pub mod sse;
pub mod task;
pub mod token;
pub mod transport;
pub mod websocket;
