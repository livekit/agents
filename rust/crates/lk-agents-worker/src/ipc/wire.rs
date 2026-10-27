//! The job process protocol: what the worker says to a job process ([`ToChild`]) and what the job
//! process answers ([`ToParent`]), each a typed message encoded with postcard, in frames of a
//! 4-byte big-endian length then the message, at most [`MAX_FRAME`] bytes.
//!
//! The parent speaks first, [`Hello`] with its [`PROTOCOL_VERSION`]; the child answers
//! [`ToParent::Ready`] with its own. Either side fails on a version it does not speak. Everything
//! after is one-way: a ping's pong echoes its timestamp, and the shutdown is acked by messages of
//! its own.

use std::io;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// The version both ends must speak; a change to a message is a new version.
pub const PROTOCOL_VERSION: u32 = 1;

/// The largest message a side reads: a job's `Job` protobuf is a few kilobytes.
pub const MAX_FRAME: usize = 4 << 20;

/// The worker's messages to a job process.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ToChild {
    /// The first message: the child initializes, then answers `Ready`.
    Hello(Hello),
    /// Answered with a `Pong` once the child's event loop ran a callback.
    Ping { sent_ms: u64 },
    /// The process's one job.
    StartJob(Box<StartJob>),
    /// Acked with `ShutdownAck`; the job then shuts down with `reason` (the process exits at once
    /// without one).
    Shutdown { reason: String },
    /// The child writes its stacks (where SIGUSR1 does not exist, Windows).
    DumpStacks,
}

/// A job process's messages to the worker.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ToParent {
    /// The answer to `Hello`: initialized, or the error it failed with.
    Ready {
        version: u32,
        error: Option<String>,
    },
    Pong {
        sent_ms: u64,
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
    pub version: u32,
    /// The worker's event loop runs in debug mode: so does the job's.
    pub asyncio_debug: bool,
    pub http_proxy: Option<String>,
    /// Silence from the worker past this long means it is gone: the child stops.
    pub health_timeout_ms: Option<u64>,
}

/// A job (`RunningJobInfo`): its `Job` protobuf, the accept arguments, where it connects, and the
/// dispatch timeline (unix seconds, 0 when unknown).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StartJob {
    pub job: Vec<u8>,
    pub name: String,
    pub identity: String,
    pub metadata: String,
    pub url: String,
    pub token: String,
    pub worker_id: String,
    pub fake_job: bool,
    pub received_at: f64,
    pub accepted_at: f64,
    pub assigned_at: f64,
    pub launched_at: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum WireError {
    #[error("the socket closed: {0}")]
    Io(#[from] io::Error),
    #[error("a {0}-byte frame is over the {MAX_FRAME}-byte limit")]
    TooLarge(usize),
    #[error("malformed message: {0}")]
    Malformed(#[from] postcard::Error),
    #[error("unexpected message: {0}")]
    Unexpected(String),
    #[error("the other side speaks protocol version {theirs}, this one {PROTOCOL_VERSION}")]
    Version { theirs: u32 },
}

/// A message's bytes.
pub fn encode<T: Serialize>(msg: &T) -> Vec<u8> {
    postcard::to_allocvec(msg).expect("the messages serialize")
}

/// A message from its bytes, all of them.
pub fn decode<T: DeserializeOwned>(data: &[u8]) -> Result<T, WireError> {
    let (msg, rest) = postcard::take_from_bytes(data)?;
    if !rest.is_empty() {
        return Err(postcard::Error::DeserializeBadEncoding.into());
    }
    Ok(msg)
}

/// The next message.
pub async fn read<T: DeserializeOwned, R: AsyncRead + Unpin + ?Sized>(
    r: &mut R,
) -> Result<T, WireError> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len).await?;
    let len = u32::from_be_bytes(len) as usize;
    if len > MAX_FRAME {
        return Err(WireError::TooLarge(len));
    }
    let mut buf = vec![0; len];
    r.read_exact(&mut buf).await?;
    decode(&buf)
}

/// Writes one message.
pub async fn write<T: Serialize, W: AsyncWrite + Unpin + ?Sized>(
    w: &mut W,
    msg: &T,
) -> io::Result<()> {
    let data = encode(msg);
    if data.len() > MAX_FRAME {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, WireError::TooLarge(data.len())));
    }
    let mut frame = Vec::with_capacity(4 + data.len());
    frame.extend_from_slice(&(data.len() as u32).to_be_bytes());
    frame.extend_from_slice(&data);
    w.write_all(&frame).await?;
    w.flush().await
}
