//! The resolved worker configuration and the worker's protocol constants.
use super::WorkerPermissions;

use std::time::Duration;

/// Must be unique across processes, not unpredictable: the clock and the pid suffice.
pub(crate) fn new_instance_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    format!("{:x}-{:x}", nanos, std::process::id())
}

/// The wait for `JobAssignment` after the user accepts.
pub const ASSIGNMENT_TIMEOUT: Duration = Duration::from_millis(7_500);
/// How often a connected worker sends `UpdateWorkerStatus`. Per-connection.
pub const UPDATE_STATUS_INTERVAL: Duration = Duration::from_millis(2_500);
/// How often the load function is re-invoked. Lives for the whole worker, connected or not.
pub const UPDATE_LOAD_INTERVAL: Duration = Duration::from_millis(500);
/// Bounds connection establishment, registration, writes, and transport shutdown.
pub const CONNECTION_TIMEOUT: Duration = Duration::from_secs(10);
/// Default `drain()` deadline.
pub const DRAIN_TIMEOUT: Duration = Duration::from_secs(3600);
pub const WORKER_PROTOCOL_VERSION: u32 = 1;

/// Dev mode uses infinity, which disables the availability check and zeroes the idle estimate in
/// `effective_load`.
pub const DEFAULT_LOAD_THRESHOLD_PROD: f32 = 0.7;
/// `math.inf`.
pub const DEFAULT_LOAD_THRESHOLD_DEV: f32 = f32::INFINITY;

/// Resolved settings for [`crate::Worker`]; performs no environment reads.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub agent_name: String,
    pub deployment: String,
    pub version: String,
    pub permissions: WorkerPermissions,
    pub endpoints: Vec<livekit_protocol::agent_http::AgentEndpoint>,
    /// Generated once and reused across reconnects: a new one supersedes the previous epoch and
    /// closes its connections. The server rejects a manifest without one.
    pub instance_id: String,
    /// Fallback when `JobAssignment.url` is unset.
    pub ws_url: String,
    pub load_threshold: EnvOption<f32>,
    pub num_idle_processes: EnvOption<u32>,
    pub devmode: bool,
    /// A simulated worker bypasses capacity limits while it is not draining.
    pub simulation: bool,
    /// Used when a drain request supplies no timeout.
    pub drain_timeout: Option<Duration>,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        WorkerConfig {
            agent_name: String::new(),
            deployment: String::new(),
            version: "0.0.0".into(),
            permissions: WorkerPermissions::default(),
            endpoints: Vec::new(),
            instance_id: String::new(),
            ws_url: String::new(),
            load_threshold: EnvOption::Split {
                dev: DEFAULT_LOAD_THRESHOLD_DEV,
                prod: DEFAULT_LOAD_THRESHOLD_PROD,
            },
            num_idle_processes: EnvOption::Split { dev: 0, prod: 4 },
            devmode: false,
            simulation: false,
            drain_timeout: Some(DRAIN_TIMEOUT),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EnvOption<T> {
    Fixed(T),
    Split { dev: T, prod: T },
}

impl<T: Copy> EnvOption<T> {
    pub fn get(self, devmode: bool) -> T {
        match self {
            EnvOption::Fixed(v) => v,
            EnvOption::Split { dev, prod } => {
                if devmode {
                    dev
                } else {
                    prod
                }
            }
        }
    }
}
