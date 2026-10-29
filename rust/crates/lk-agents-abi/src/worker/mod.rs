//! `worker.py` and `job.py`: the worker records, the end of a job, and the remote-session records.

pub mod endpoint;
pub mod endpoint_host;
pub mod ipc;
pub mod job;
pub mod remote_session;
pub mod worker;

pub mod cli;

pub mod ipc_host;

pub mod server_host;

/// A token that could not be signed: missing credentials.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct TokenRefused(pub String);

/// The join tokens the host hands a participant it starts itself (a notebook's user and agent, a
/// simulated person). The runtime signs them.
#[crate::host(functions)]
pub trait JoinTokens {
    /// A token joining `room` as `identity`, publishing, subscribing and updating its own
    /// metadata; `agent` joins as an agent. `ttl` in seconds; the signer's default when `None`.
    #[host(function)]
    fn room_join_token(
        api_key: String,
        api_secret: String,
        identity: String,
        room: String,
        #[host(kw, default = false)] agent: bool,
        #[host(default = None)] ttl: Option<f64>,
    ) -> Result<String, TokenRefused>;
}
