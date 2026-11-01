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

/// Production leaves headroom before CPU saturation.
pub const DEFAULT_LOAD_THRESHOLD_PROD: f32 = 0.7;
/// One quarter of the default load budget, independent of prewarming configuration.
pub const DEFAULT_JOB_LOAD: f32 = DEFAULT_LOAD_THRESHOLD_PROD / 4.0;
/// Infinity disables capacity limits.
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

/// Time allowed for draining active jobs and completing process shutdown.
#[derive(Debug, Clone, Copy)]
pub struct DrainOptions {
    total: Duration,
    drain: Duration,
}

impl DrainOptions {
    pub fn new(total: f64, drain: f64) -> Result<Self, String> {
        let duration = |name: &str, seconds: f64| {
            Duration::try_from_secs_f64(seconds)
                .ok()
                .filter(|value| value.as_secs_f64() <= 3_155_760_000.0)
                .ok_or_else(|| format!("{name} must be finite and nonnegative"))
        };
        let total = duration("shutdown_timeout", total)?;
        let drain = duration("drain_timeout", drain)?;
        if total.is_zero() || drain >= total {
            return Err("shutdown_timeout must be positive and greater than drain_timeout".into());
        }
        Ok(Self { total, drain })
    }

    pub fn total(self) -> Duration {
        self.total
    }

    pub fn drain_remaining(self, remaining: Duration) -> Duration {
        let reserve = self.total - self.drain;
        self.drain.min(remaining.saturating_sub(reserve))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drain_keeps_the_cleanup_reserve_even_after_delays() {
        let options = DrainOptions::new(55.0, 45.0).unwrap();
        assert_eq!(options.drain_remaining(Duration::from_secs(55)), Duration::from_secs(45));
        assert_eq!(options.drain_remaining(Duration::from_secs(40)), Duration::from_secs(30));
        assert_eq!(options.drain_remaining(Duration::from_secs(5)), Duration::ZERO);
        assert!(DrainOptions::new(45.0, 45.0).is_err());
        assert!(DrainOptions::new(f64::INFINITY, 0.0).is_err());
    }
}
