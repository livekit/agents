//! Worker settings and protocol timing defaults.

use crate::{ServerType, WorkerPermissions};
use std::time::Duration;

/// The wait for `JobAssignment` after the user accepts.
pub const ASSIGNMENT_TIMEOUT: Duration = Duration::from_millis(7_500);
/// How often a connected worker sends `UpdateWorkerStatus`. Per-connection.
pub const UPDATE_STATUS_INTERVAL: Duration = Duration::from_millis(2_500);
/// How often the load function is re-invoked. Lives for the whole worker, connected or not.
#[lk_agents_abi::host(export(pid = false))]
pub const UPDATE_LOAD_INTERVAL: Duration = Duration::from_millis(500);
/// Default `drain()` deadline.
pub const DRAIN_TIMEOUT: Duration = Duration::from_secs(3600);
pub const WORKER_PROTOCOL_VERSION: u32 = 1;

/// Dev mode uses infinity, which disables the availability check and zeroes the idle estimate in
/// `effective_load`.
pub const DEFAULT_LOAD_THRESHOLD_PROD: f32 = 0.7;
/// `math.inf`.
pub const DEFAULT_LOAD_THRESHOLD_DEV: f32 = f32::INFINITY;

/// Optional strings fall back to the environment when `None`; `Some("")` clears a value.
/// Explicit values win; empty environment values are ignored. `Default` reads nothing.
pub struct WorkerOptions {
    /// Falls back to `LIVEKIT_URL`.
    pub ws_url: Option<String>,
    /// The `/agent` address when it is not `ws_url`: a development livekit-server serves
    /// WebTransport on its own UDP port (`webtransport.port`) while jobs still join rooms at
    /// `ws_url`. Falls back to `LIVEKIT_WORKER_URL`, then `ws_url`.
    pub worker_url: Option<String>,
    /// Falls back to `LIVEKIT_API_KEY`.
    pub api_key: Option<String>,
    /// Falls back to `LIVEKIT_API_SECRET`.
    pub api_secret: Option<String>,
    /// Falls back to `LIVEKIT_WORKER_TOKEN`.
    pub worker_token: Option<String>,
    /// Falls back to `LIVEKIT_AGENT_NAME_OVERRIDE`, then `LIVEKIT_AGENT_NAME`.
    pub agent_name: Option<String>,
    /// Falls back to `LIVEKIT_AGENT_DEPLOYMENT`.
    pub deployment: Option<String>,
    pub server_type: ServerType,
    pub version: String,
    pub permissions: WorkerPermissions,
    /// Routes served at `/agents/{agent_name}/{deployment}/{path}`; declaring any requests the
    /// endpoint data plane, which only WebTransport carries.
    pub endpoints: Vec<livekit_protocol::agent_http::AgentEndpoint>,
    pub load_threshold: EnvOption<f32>,
    /// Defaults to zero in development and `min(ceil(cpu_count), 4)` in production (CPU detection
    /// honors `NUM_CPUS` and container quotas).
    pub num_idle_processes: Option<EnvOption<u32>>,
    pub devmode: bool,
    pub simulation: bool,
    pub drain_timeout: Option<Duration>,
    pub transport: crate::TransportConfig,
    pub headers: Vec<(String, String)>,
    pub max_retry: u32,
    /// Bounds connection establishment, registration, writes, and transport shutdown.
    pub connection_timeout: Duration,
}

impl Default for WorkerOptions {
    fn default() -> Self {
        let worker = WorkerConfig::default();
        Self {
            ws_url: None,
            worker_url: None,
            api_key: None,
            api_secret: None,
            worker_token: None,
            agent_name: None,
            deployment: None,
            server_type: worker.server_type,
            version: env!("CARGO_PKG_VERSION").into(),
            permissions: worker.permissions,
            endpoints: Vec::new(),
            load_threshold: worker.load_threshold,
            num_idle_processes: None,
            devmode: worker.devmode,
            simulation: worker.simulation,
            drain_timeout: worker.drain_timeout,
            transport: crate::TransportConfig::WebSocket { heartbeat: crate::HEARTBEAT_INTERVAL },
            headers: Vec::new(),
            max_retry: 16,
            connection_timeout: Duration::from_secs(10),
        }
    }
}

impl WorkerOptions {
    pub(crate) fn resolve(self) -> (WorkerConfig, crate::RuntimeConfig, crate::TransportConfig) {
        self.resolve_with(|name| std::env::var(name).ok(), || crate::CpuMonitor::new().cpu_count())
    }

    fn resolve_with(
        self,
        mut read_env: impl FnMut(&str) -> Option<String>,
        cpu_count: impl FnOnce() -> f32,
    ) -> (WorkerConfig, crate::RuntimeConfig, crate::TransportConfig) {
        let mut env = |name: &str| read_env(name).filter(|value| !value.is_empty());
        let ws_url = self.ws_url.or_else(|| env("LIVEKIT_URL")).unwrap_or_default();
        let worker_url = self
            .worker_url
            .or_else(|| env("LIVEKIT_WORKER_URL"))
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| ws_url.clone());
        let api_key = self.api_key.or_else(|| env("LIVEKIT_API_KEY")).unwrap_or_default();
        let api_secret = self.api_secret.or_else(|| env("LIVEKIT_API_SECRET")).unwrap_or_default();
        let worker_token = self
            .worker_token
            .or_else(|| env("LIVEKIT_WORKER_TOKEN"))
            .filter(|value| !value.is_empty());
        let agent_name = self
            .agent_name
            .or_else(|| env("LIVEKIT_AGENT_NAME_OVERRIDE"))
            .or_else(|| env("LIVEKIT_AGENT_NAME"))
            .unwrap_or_default();
        let deployment =
            self.deployment.or_else(|| env("LIVEKIT_AGENT_DEPLOYMENT")).unwrap_or_default();
        let worker = WorkerConfig {
            ws_url: ws_url.clone(),
            agent_name,
            deployment,
            server_type: self.server_type,
            version: self.version,
            permissions: self.permissions,
            endpoints: self.endpoints,
            instance_id: new_instance_id(),
            load_threshold: self.load_threshold,
            num_idle_processes: self.num_idle_processes.unwrap_or_else(|| EnvOption::Split {
                dev: 0,
                prod: crate::hw::cpu::idle_process_target(cpu_count()),
            }),
            devmode: self.devmode,
            simulation: self.simulation,
            drain_timeout: self.drain_timeout,
        };
        let connection = crate::RuntimeConfig {
            server_url: worker_url,
            api_key,
            api_secret,
            worker_token,
            headers: self.headers,
            max_retry: self.max_retry,
            connection_timeout: self.connection_timeout,
        };
        (worker, connection, self.transport)
    }
}

/// Must be unique across processes, not unpredictable: the clock and the pid suffice.
fn new_instance_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    format!("{:x}-{:x}", nanos, std::process::id())
}

/// Resolved settings for [`crate::WorkerState`]; performs no environment reads.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub server_type: ServerType,
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
            server_type: ServerType::Room,
            agent_name: String::new(),
            deployment: String::new(),
            version: "0.0.0".into(),
            permissions: WorkerPermissions::default(),
            endpoints: Vec::new(),
            instance_id: new_instance_id(),
            ws_url: String::new(),
            load_threshold: EnvOption::Split {
                dev: DEFAULT_LOAD_THRESHOLD_DEV,
                prod: DEFAULT_LOAD_THRESHOLD_PROD,
            },
            num_idle_processes: EnvOption::Split {
                dev: 0,
                prod: crate::hw::cpu::idle_process_target(f32::INFINITY),
            },
            devmode: false,
            simulation: false,
            drain_timeout: Some(DRAIN_TIMEOUT),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EnvOption<T> {
    Fixed(T),
    Split {
        dev: T,
        prod: T,
    },
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

#[cfg(test)]
mod tests {
    use super::*;

    fn environment(name: &str) -> Option<String> {
        Some(
            match name {
                "LIVEKIT_URL" => "wss://environment.example",
                "LIVEKIT_WORKER_URL" => "https://environment.example:7883",
                "LIVEKIT_API_KEY" => "environment-key",
                "LIVEKIT_API_SECRET" => "environment-secret",
                "LIVEKIT_WORKER_TOKEN" => "environment-token",
                "LIVEKIT_AGENT_NAME_OVERRIDE" => "override-name",
                "LIVEKIT_AGENT_NAME" => "default-name",
                "LIVEKIT_AGENT_DEPLOYMENT" => "environment-deployment",
                _ => return None,
            }
            .into(),
        )
    }

    #[test]
    fn unspecified_options_use_environment_and_detected_capacity() {
        let (worker, connection, _) = WorkerOptions::default().resolve_with(environment, || 1.5);
        assert_eq!(worker.ws_url, "wss://environment.example");
        assert_eq!(connection.server_url, "https://environment.example:7883");
        assert_eq!(connection.api_key, "environment-key");
        assert_eq!(connection.api_secret, "environment-secret");
        assert_eq!(connection.worker_token.as_deref(), Some("environment-token"));
        assert_eq!(worker.agent_name, "override-name");
        assert_eq!(worker.deployment, "environment-deployment");
        assert_eq!(worker.num_idle_processes.get(false), 2);
        assert_eq!(worker.num_idle_processes.get(true), 0);
        assert_eq!(worker.version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn explicit_options_win_without_reading_environment_or_cpu() {
        let options = WorkerOptions {
            ws_url: Some("wss://explicit.example".into()),
            worker_url: Some("https://explicit.example:7883".into()),
            api_key: Some("explicit-key".into()),
            api_secret: Some("explicit-secret".into()),
            worker_token: Some("explicit-token".into()),
            agent_name: Some("explicit-name".into()),
            deployment: Some("explicit-deployment".into()),
            num_idle_processes: Some(EnvOption::Fixed(7)),
            ..Default::default()
        };
        let (worker, connection, _) = options.resolve_with(
            |_| panic!("explicit options must not consult the environment"),
            || panic!("explicit capacity must not probe CPUs"),
        );
        assert_eq!(worker.ws_url, "wss://explicit.example");
        assert_eq!(connection.server_url, "https://explicit.example:7883");
        assert_eq!(connection.api_key, "explicit-key");
        assert_eq!(connection.api_secret, "explicit-secret");
        assert_eq!(connection.worker_token.as_deref(), Some("explicit-token"));
        assert_eq!(worker.agent_name, "explicit-name");
        assert_eq!(worker.deployment, "explicit-deployment");
        assert_eq!(worker.num_idle_processes.get(false), 7);
        assert_eq!(worker.num_idle_processes.get(true), 7);
    }

    #[test]
    fn explicit_empty_values_disable_environment_fallbacks() {
        let options = WorkerOptions {
            ws_url: Some(String::new()),
            worker_url: Some(String::new()),
            api_key: Some(String::new()),
            api_secret: Some(String::new()),
            worker_token: Some(String::new()),
            agent_name: Some(String::new()),
            deployment: Some(String::new()),
            num_idle_processes: Some(EnvOption::Fixed(0)),
            ..Default::default()
        };
        let (worker, connection, _) = options.resolve_with(environment, || 8.0);
        assert!(worker.ws_url.is_empty());
        assert!(connection.server_url.is_empty(), "an empty worker_url follows ws_url, not the environment");
        assert!(connection.api_key.is_empty());
        assert!(connection.api_secret.is_empty());
        assert_eq!(connection.worker_token, None);
        assert!(worker.agent_name.is_empty());
        assert!(worker.deployment.is_empty());
        assert_eq!(worker.num_idle_processes.get(false), 0);
    }

    #[test]
    fn partial_options_keep_other_environment_fallbacks() {
        let options = WorkerOptions {
            ws_url: Some("wss://explicit.example".into()),
            agent_name: Some("explicit-name".into()),
            ..Default::default()
        };
        let (worker, connection, _) = options.resolve_with(environment, || 8.0);
        assert_eq!(worker.ws_url, "wss://explicit.example");
        assert_eq!(worker.agent_name, "explicit-name");
        assert_eq!(connection.api_key, "environment-key");
        assert_eq!(worker.deployment, "environment-deployment");
        assert_eq!(worker.num_idle_processes.get(false), 4);
    }

    #[test]
    fn absent_or_empty_environment_uses_local_defaults() {
        for empty in [false, true] {
            let (worker, connection, _) =
                WorkerOptions::default().resolve_with(|_| empty.then(String::new), || 0.5);
            assert!(worker.ws_url.is_empty());
            assert!(worker.agent_name.is_empty());
            assert!(worker.deployment.is_empty());
            assert!(connection.api_key.is_empty());
            assert!(connection.api_secret.is_empty());
            assert_eq!(connection.worker_token, None);
            assert_eq!(worker.num_idle_processes.get(false), 1);
        }
        let (worker, _, _) = WorkerOptions::default().resolve_with(
            |name| {
                if name == "LIVEKIT_AGENT_NAME_OVERRIDE" {
                    Some(String::new())
                } else {
                    environment(name)
                }
            },
            || 1.0,
        );
        assert_eq!(worker.agent_name, "default-name");
    }
}
