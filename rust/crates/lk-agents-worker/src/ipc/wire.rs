//! The job process protocol: what the worker says to a job process ([`ToChild`]) and what the job
//! process answers ([`ToParent`]). The [`channel`](super::channel) carries them and checks the
//! version.
//!
//! The parent speaks first, [`Hello`]; the child answers [`ToParent::Ready`]. Everything after is
//! one-way: a ping's pong echoes its timestamp, and the shutdown is acked by messages of its own.

use serde::{Deserialize, Serialize};

/// The longest free text a message carries (an error, a reason); longer text is cut.
pub const MAX_TEXT: usize = 64 << 10;

/// The worker's messages to a job process.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ToChild {
    /// The first message: the child initializes, then answers `Ready`.
    Hello(Hello),
    /// Answered with a `Pong` once the child's event loop ran a callback.
    Ping { sent_ms: u64 },
    /// The process's one job.
    StartJob { request_id: u64, job: Box<StartJob> },
    /// Acked with `ShutdownAck`; the job then shuts down with `reason` (the process exits at once
    /// without one).
    Shutdown { reason: String },
}

/// A job process's messages to the worker.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ToParent {
    /// The answer to `Hello`: initialized, or the error it failed with.
    Ready {
        error: Option<String>,
    },
    Pong {
        sent_ms: u64,
    },
    /// Confirms host acceptance of the correlated job, or its rejection cause.
    JobAccepted {
        request_id: u64,
        error: Option<String>,
    },
    ShutdownAck,
    /// The job is shutting down (or the process, without one).
    ShuttingDown,
    /// The job ended with `reason`; the process exits.
    Exiting {
        reason: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hello {
    /// The worker's event loop runs in debug mode: so does the job's.
    pub asyncio_debug: bool,
    pub http_proxy: Option<String>,
    /// Silence from the worker past this long means it is gone: the child stops.
    pub health_timeout_ms: Option<u64>,
}

pub use lk_agents_abi::worker::job::StartJobFields as StartJob;

/// `text`, cut to [`MAX_TEXT`] bytes with a marker, so a huge traceback cannot exceed a frame.
pub fn bounded(mut text: String) -> String {
    if text.len() > MAX_TEXT {
        let mut end = MAX_TEXT;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
        text.push_str("... [truncated]");
    }
    text
}
