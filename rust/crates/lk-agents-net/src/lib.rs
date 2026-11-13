//! The shared async transport of the I/O crates: HTTP, SSE, WebSocket, tokens, the delegation
//! A2A client and telemetry export.
#![cfg_attr(
    test,
    allow(
        clippy::disallowed_methods,
        reason = "tests drive Tokio's paused clock and spawn directly; the library is linted without"
    )
)]

pub mod a2a_auth;
pub mod a2a_client;
pub mod blocking_io;
pub mod clock;
pub mod dns;
pub mod env;
pub mod export;
pub mod http;
pub mod job_identity;
pub mod owned;
pub mod room_service;
pub mod signal;
pub mod source;
pub mod sse;
pub mod task;
pub mod tls;
pub mod token;
pub mod transport;
pub mod websocket;

pub mod connection_pool;

pub mod strings;

/// This crate, under the name `#[span]` functions expand to.
extern crate self as lk_agents_net;
