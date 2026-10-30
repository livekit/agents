//! Supervision outcomes, timing options and diagnostics.

use std::sync::Arc;
use std::time::Duration;

use super::super::channel::{ChannelError, VERSION};
use super::super::wire::StartJob;

/// How often the memory monitor samples the process.
pub const MEMORY_MONITOR_INTERVAL: Duration = Duration::from_secs(5);
/// The least time between two "high memory usage" warnings of one process.
pub const MEMORY_WARN_COOLDOWN: Duration = Duration::from_secs(120);
/// Warn again before the cooldown when usage grew this much since the last warning.
pub const MEMORY_WARN_RESET_DELTA_MB: f64 = 50.0;
/// Between SIGUSR1 (the child dumps its stacks) and SIGKILL.
pub const KILL_GRACE: Duration = Duration::from_millis(500);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Debug = 10,
    Info = 20,
    Warning = 30,
    Error = 40,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExtraValue {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

/// A log record's extra fields; the host adds the process's own after them.
pub type Extra = Vec<(&'static str, ExtraValue)>;

/// The fields as the host's log record takes them.
pub fn extra_fields(extra: Extra) -> serde_json::Map<String, serde_json::Value> {
    extra
        .into_iter()
        .map(|(key, value)| {
            let value = match value {
                ExtraValue::Int(v) => v.into(),
                ExtraValue::Float(v) => v.into(),
                ExtraValue::Str(v) => v.into(),
                ExtraValue::Bool(v) => v.into(),
            };
            (key.to_owned(), value)
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct SupervisorOptions {
    pub pid: u32,
    pub initialize_timeout: f64,
    pub close_timeout: f64,
    pub memory_warn_mb: f64,
    pub memory_limit_mb: f64,
    pub ping_interval: f64,
    pub ping_timeout: f64,
    pub high_ping_threshold: f64,
    /// A process can be killed; a job thread cannot: its late initialization, ack or exit is
    /// logged and waited for instead, and `kill` asks it to shut down.
    pub killable: bool,
}

impl SupervisorOptions {
    /// What the logs call the supervised runner.
    pub(super) fn runner(&self) -> &'static str {
        if self.killable { "process" } else { "job runner" }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum InitError {
    #[error("initialization already requested")]
    AlreadyInitializing,
    #[error("duplex closed")]
    Closed,
    #[error("first message must be Ready")]
    Unexpected,
    #[error("process initialization failed: {0}")]
    Failed(String),
    #[error("the job process speaks protocol version {0}, the worker {VERSION}")]
    Version(u32),
    #[error("IPC protocol failed: {0}")]
    Protocol(Arc<ProtocolError>),
}

/// Why the supervisor gave up on the connection: the channel failed, or the child said what the
/// protocol does not allow then.
#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error(transparent)]
    Channel(#[from] ChannelError),
    #[error("unexpected message: {0}")]
    Unexpected(String),
}

impl From<&ProtocolError> for lk_agents_abi::worker::ipc::IpcFailure {
    fn from(error: &ProtocolError) -> Self {
        let message = error.to_string();
        match error {
            ProtocolError::Channel(ChannelError::Closed | ChannelError::Truncated) => {
                Self::Io { kind: "UnexpectedEof".into(), os_error: None, message }
            }
            ProtocolError::Channel(ChannelError::Io { kind, os_error, .. }) => {
                Self::Io { kind: format!("{kind:?}"), os_error: *os_error, message }
            }
            ProtocolError::Channel(ChannelError::Malformed(cause)) => {
                Self::Malformed { message: cause.clone() }
            }
            ProtocolError::Channel(ChannelError::TooLarge(size)) => {
                Self::TooLarge { size: *size as u64 }
            }
            ProtocolError::Channel(ChannelError::Version(received)) => {
                Self::Version { received: *received, expected: VERSION }
            }
            ProtocolError::Unexpected(message) => Self::Unexpected { message: message.clone() },
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StartJobError {
    #[error("invalid job launch fields: {0}")]
    InvalidFields(String),
    #[error("the process is not initialized")]
    NotReady,
    #[error("the process already accepted a job or is accepting one")]
    AlreadyRunning,
    #[error("the IPC connection is closed")]
    Closed,
    #[error("IPC protocol failed: {0}")]
    Protocol(Arc<ProtocolError>),
    #[error("the child refused the job: {0}")]
    Rejected(String),
    #[error("the child did not acknowledge the job before its deadline")]
    Timeout,
}

pub fn validate_start_job(job: &StartJob) -> Result<(), StartJobError> {
    use prost::Message;
    livekit_protocol::Job::decode(job.job.as_slice())
        .map_err(|error| StartJobError::InvalidFields(format!("job protobuf: {error}")))?;
    for (field, value) in [
        ("received_at", job.received_at),
        ("accepted_at", job.accepted_at),
        ("assigned_at", job.assigned_at),
        ("launched_at", job.launched_at),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(StartJobError::InvalidFields(format!(
                "{field} must be finite and nonnegative"
            )));
        }
    }
    Ok(())
}

pub use lk_agents_abi::secs;

/// `round(x, digits)`: Python rounds the exact binary value half to even, as formatting does.
pub(super) fn round(x: f64, digits: usize) -> f64 {
    format!("{x:.digits$}").parse().unwrap_or(x)
}

/// A native termination request failed; exit/reaping ownership remains unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignalFailure {
    pub pid: u32,
    pub cause: String,
}

/// Cleanup missed its deadline; ownership stays with the driver until a reaping receipt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CleanupObservation {
    pub pid: u32,
    pub exited: bool,
    pub reaped: bool,
}

/// Where a graceful close is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Close {
    None,
    Ack(Duration),
    ShuttingDown(Duration),
    Exit(Duration),
    Waiting,
}

/// Rate-limits the high memory warning: once per cooldown, or sooner when usage grew.
#[derive(Default)]
pub struct WarnGate {
    last: Option<(Duration, f64)>,
}

impl WarnGate {
    pub fn should_warn(&mut self, memory_mb: f64, now: Duration) -> bool {
        let emit = match self.last {
            None => true,
            Some((at, mb)) => {
                now.saturating_sub(at) >= MEMORY_WARN_COOLDOWN
                    || memory_mb - mb >= MEMORY_WARN_RESET_DELTA_MB
            }
        };
        if emit {
            self.last = Some((now, memory_mb));
        }
        emit
    }
}
