pub mod http;
pub mod ipc;
pub mod job;
pub mod remote_session;
#[path = "worker.rs"]
pub mod types;

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

    /// A token for a participant other than the agent, such as an avatar. `claims` is JSON, the
    /// payload of an access token with the identity in `identity`.
    ///
    /// With `api_key` and `api_secret`, the token is signed here and lives `ttl` seconds (six
    /// hours when `None`). Without them, in token mode, the server issues it for the job
    /// `job_id` and sets its lifetime.
    #[host(function)]
    async fn participant_token(
        claims: String,
        api_key: String,
        api_secret: String,
        job_id: Option<String>,
        #[host(default = None)] ttl: Option<f64>,
    ) -> Result<String, TokenRefused>;
}

// Public data contracts. Python behavior remains in its adapters.

/// A pass/fail verdict for a scenario, with a human-readable reason.
#[crate::host(record(
    module = "simulation",
    path = "livekit.agents.evals",
    dataclass,
    extract,
    emit
))]
pub struct SimulationVerdict {
    pub success: bool,
    pub reason: String,
}
