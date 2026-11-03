//! `AgentServer(...)`'s options, resolved once against the environment, and the environment
//! reads the host calls.

use livekit_protocol as proto;
use lk_agents_abi::logging::LogWriter;

use crate::{EnvOption, RuntimeConfig, WorkerConfig, WorkerPermissions};

/// `load_threshold`'s default, to which a hosted (Cloud) worker is held.
pub const DEFAULT_LOAD_THRESHOLD: EnvOption<f32> = EnvOption::Split {
    dev: crate::config::DEFAULT_LOAD_THRESHOLD_DEV,
    prod: crate::config::DEFAULT_LOAD_THRESHOLD_PROD,
};

/// `AgentServer(listeners=)`: each listener's address, a key left out taking its default.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Listeners {
    #[serde(default)]
    pub public: PublicListener,
    #[serde(default)]
    pub health: ListenerAddress,
    /// Off unless given.
    pub prometheus: Option<ListenerAddress>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ListenerAddress {
    pub host: Option<String>,
    pub port: Option<u16>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicListener {
    pub host: Option<String>,
    pub port: Option<u16>,
    /// Read the client address and scheme from `X-Forwarded-For` and `X-Forwarded-Proto`.
    #[serde(default)]
    pub proxy_headers: bool,
}

/// Where a listener binds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bind {
    pub host: String,
    pub port: u16,
}

impl Bind {
    /// A given value wins; otherwise every interface on the listener's port, or loopback on a
    /// free port in development and the console.
    fn resolve(host: &Option<String>, port: Option<u16>, local: bool, default_port: u16) -> Self {
        let fallback = if local { "127.0.0.1" } else { "0.0.0.0" };
        Bind {
            host: host.clone().unwrap_or_else(|| fallback.into()),
            port: port.unwrap_or(if local { 0 } else { default_port }),
        }
    }
}

impl Listeners {
    pub fn public(&self, local: bool) -> Bind {
        Bind::resolve(&self.public.host, self.public.port, local, 8080)
    }

    pub fn health(&self, local: bool) -> Bind {
        Bind::resolve(&self.health.host, self.health.port, local, 8081)
    }

    pub fn prometheus(&self, local: bool) -> Option<Bind> {
        let given = self.prometheus.as_ref()?;
        Some(Bind::resolve(&given.host, given.port, local, 9090))
    }
}

/// What `AgentServer(...)` was given, with the environment read where an option was left empty, as
/// `AgentServer.__init__` reads it. `update_options` writes over it until the server starts.
#[derive(Debug, Clone)]
pub struct ServerSettings {
    pub ws_url: String,
    pub api_key: String,
    pub api_secret: String,
    /// `LIVEKIT_WORKER_TOKEN`: set on a worker hosted on Cloud.
    pub worker_token: String,
    /// `LIVEKIT_WORKER_URL`: the `/agent` address when it is not `ws_url`. A development
    /// livekit-server serves WebTransport on its own UDP port (`webtransport.port`) while jobs
    /// still join rooms at `ws_url`.
    pub worker_url: String,
    pub deployment: String,
    pub agent_name: String,
    /// `LIVEKIT_AGENT_ENDPOINTS_URL`: where a job process delegates to its server's endpoints,
    /// resolved by the server before it starts them.
    pub endpoints_url: String,
    pub listeners: Listeners,
    pub load_threshold: EnvOption<f32>,
    pub num_idle_processes: Option<EnvOption<u32>>,
    /// Maximum drain time in seconds; zero skips waiting for natural completion.
    pub drain_timeout: f64,
    /// Overall launcher budget, including draining and host cleanup.
    pub shutdown_timeout: f64,
    pub max_retry: u32,
    pub permissions: WorkerPermissions,
    pub http_proxy: Option<String>,
    pub prometheus_multiproc_dir: Option<String>,
    /// Set by `--simulation`: load shedding off, and no health listener (runs share a host).
    pub simulation: bool,
    /// The SDK version the worker registers and `/worker` reports.
    pub version: String,
    /// The job processes' settings, which the host's pool takes as they are.
    pub job_executor_type: String,
    pub multiprocessing_context: String,
    pub(crate) multiprocessing_fallback: bool,
    pub job_memory_warn_mb: f64,
    pub job_memory_limit_mb: f64,
    pub shutdown_process_timeout: f64,
    pub session_end_timeout: f64,
    pub initialize_process_timeout: f64,
}

/// `AgentServer()`'s defaults, for the options a caller leaves out.
impl Default for ServerSettings {
    fn default() -> Self {
        ServerSettings {
            ws_url: String::new(),
            api_key: String::new(),
            api_secret: String::new(),
            worker_token: String::new(),
            worker_url: String::new(),
            deployment: String::new(),
            agent_name: String::new(),
            endpoints_url: String::new(),
            listeners: Listeners::default(),
            load_threshold: DEFAULT_LOAD_THRESHOLD,
            num_idle_processes: None,
            drain_timeout: crate::DRAIN_TIMEOUT.as_secs_f64(),
            shutdown_timeout: crate::DRAIN_TIMEOUT.as_secs_f64() + 10.0,
            max_retry: 16,
            permissions: WorkerPermissions::default(),
            http_proxy: None,
            prometheus_multiproc_dir: None,
            simulation: false,
            version: env!("CARGO_PKG_VERSION").into(),
            job_executor_type: "process".into(),
            multiprocessing_fallback: false,
            multiprocessing_context: if cfg!(target_os = "linux") { "forkserver" } else { "spawn" }
                .into(),
            job_memory_warn_mb: 1000.0,
            job_memory_limit_mb: 0.0,
            shutdown_process_timeout: 10.0,
            session_end_timeout: 300.0,
            initialize_process_timeout: 10.0,
        }
    }
}

impl ServerSettings {
    /// Validate the whole update before replacing the running configuration.
    pub fn apply(
        &mut self,
        options: lk_agents_abi::worker::types::ServerOptions,
    ) -> Result<(), String> {
        use lk_agents_abi::Maybe::Given;
        let mut next = self.clone();
        if let Given(value) = options.ws_url {
            next.ws_url = value;
        }
        if let Given(value) = options.api_key {
            next.api_key = value;
        }
        if let Given(value) = options.api_secret {
            next.api_secret = value;
        }
        if let Given(value) = options.agent_name {
            next.agent_name = value;
        }
        if let Given(value) = options.listeners {
            next.listeners =
                serde_json::from_value(value).map_err(|error| format!("listeners: {error}"))?;
        }
        if let Given(value) = options.max_retry {
            next.max_retry = value;
        }
        if let Given(value) = options.permissions {
            next.permissions = value;
        }
        if let Given(value) = options.http_proxy {
            next.http_proxy = value;
        }
        if let Given(value) = options.prometheus_multiproc_dir {
            next.prometheus_multiproc_dir = value;
        }
        if let Given(value) = options.version {
            next.version = value;
        }
        if let Given(value) = options.job_executor_type {
            next.job_executor_type = value;
        }
        if let Given(value) = options.multiprocessing_context {
            next.multiprocessing_context = value;
            next.multiprocessing_fallback = false;
        }
        if let Given(value) = options.job_memory_warn_mb {
            next.job_memory_warn_mb = value;
        }
        if let Given(value) = options.job_memory_limit_mb {
            next.job_memory_limit_mb = value;
        }
        if let Given(value) = options.shutdown_process_timeout {
            next.shutdown_process_timeout = value;
        }
        if let Given(value) = options.session_end_timeout {
            next.session_end_timeout = value;
        }
        if let Given(value) = options.initialize_process_timeout {
            next.initialize_process_timeout = value;
        }
        if let Given(value) = options.load_threshold {
            // The host's float; the load arithmetic is single precision.
            next.load_threshold = EnvOption::Fixed(value as f32);
        }
        if let Given(value) = options.num_idle_processes {
            next.num_idle_processes = Some(EnvOption::Fixed(value));
        }
        if let Given(value) = options.drain_timeout {
            next.drain_timeout = value;
        }
        if let Given(value) = options.shutdown_timeout {
            next.shutdown_timeout = value;
        }
        next.validate()?;
        if next.multiprocessing_context == "forkserver" && !cfg!(target_os = "linux") {
            next.multiprocessing_fallback = true;
            next.multiprocessing_context = "spawn".into();
        }
        *self = next;
        Ok(())
    }

    fn validate(&self) -> Result<(), String> {
        crate::config::DrainOptions::new(self.shutdown_timeout, self.drain_timeout)?;
        for (name, value) in [
            ("drain_timeout", self.drain_timeout),
            ("shutdown_timeout", self.shutdown_timeout),
            ("job_memory_warn_mb", self.job_memory_warn_mb),
            ("job_memory_limit_mb", self.job_memory_limit_mb),
            ("shutdown_process_timeout", self.shutdown_process_timeout),
            ("session_end_timeout", self.session_end_timeout),
            ("initialize_process_timeout", self.initialize_process_timeout),
        ] {
            if !value.is_finite() || value < 0.0 || value > 3_155_760_000.0 {
                return Err(format!("{name} must be finite and non-negative"));
            }
        }
        if self.load_threshold.get(false).is_nan() || self.load_threshold.get(false) < 0.0 {
            return Err("load_threshold must be non-negative".into());
        }
        if !matches!(self.job_executor_type.as_str(), "process" | "thread") {
            return Err("job_executor_type must be process or thread".into());
        }
        if !matches!(self.multiprocessing_context.as_str(), "spawn" | "forkserver") {
            return Err("multiprocessing_context must be spawn or forkserver".into());
        }
        Ok(())
    }

    /// The environment read where `__init__` reads it. An empty option or variable is unset;
    /// `LIVEKIT_AGENT_NAME_OVERRIDE` wins over `agent_name`, which wins over
    /// `LIVEKIT_AGENT_NAME`. The worker address, the worker token and the deployment are only the
    /// environment's.
    pub fn with_env(mut self, read_env: impl Fn(&str) -> Option<String>) -> Self {
        let env = |name: &str| read_env(name).filter(|value| !value.is_empty());
        let or_env = |given: &mut String, name: &str| {
            if given.is_empty() {
                *given = env(name).unwrap_or_default();
            }
        };
        or_env(&mut self.ws_url, "LIVEKIT_URL");
        or_env(&mut self.api_key, "LIVEKIT_API_KEY");
        or_env(&mut self.api_secret, "LIVEKIT_API_SECRET");
        match env("LIVEKIT_AGENT_NAME_OVERRIDE") {
            Some(name) => self.agent_name = name,
            None => or_env(&mut self.agent_name, "LIVEKIT_AGENT_NAME"),
        }
        self.worker_token = env("LIVEKIT_WORKER_TOKEN").unwrap_or_default();
        self.worker_url = env("LIVEKIT_WORKER_URL").unwrap_or_default();
        self.deployment = env("LIVEKIT_AGENT_DEPLOYMENT").unwrap_or_default();
        self.endpoints_url = env("LIVEKIT_AGENT_ENDPOINTS_URL").unwrap_or_default();
        self
    }

    /// `HTTPS_PROXY`, else `HTTP_PROXY`: the proxy when none was given.
    pub fn env_proxy(read_env: impl Fn(&str) -> Option<String>) -> Option<String> {
        read_env("HTTPS_PROXY").filter(|v| !v.is_empty()).or_else(|| read_env("HTTP_PROXY"))
    }

    /// Hosted on Cloud: the worker's load is the platform's to judge.
    pub fn cloud(&self) -> bool {
        !self.worker_token.is_empty()
    }

    /// What `run()` refuses before starting anything, in its order.
    pub fn check_routes(&self, has_routes: bool) -> Result<(), String> {
        if has_routes && self.agent_name.is_empty() {
            return Err(
                "server.http routes are served at /agents/{agent_name}/..., so they require \
                 an agent_name (AgentServer(agent_name=) or LIVEKIT_AGENT_NAME)"
                    .into(),
            );
        }
        Ok(())
    }

    pub fn check_registration(&self) -> Result<(), String> {
        if self.ws_url.is_empty() {
            return Err("ws_url is required, or set LIVEKIT_URL environment variable".into());
        }
        if self.api_key.is_empty() {
            return Err("api_key is required, or set LIVEKIT_API_KEY environment variable".into());
        }
        if self.api_secret.is_empty() {
            return Err(
                "api_secret is required, or set LIVEKIT_API_SECRET environment variable".into()
            );
        }
        Ok(())
    }

    /// The variables the job processes inherit: the resolved connection, and the Prometheus
    /// multiprocess directory when one is configured.
    pub fn child_env(&self) -> Vec<(&'static str, String)> {
        let mut env = Vec::new();
        for (name, value) in [
            ("LIVEKIT_URL", &self.ws_url),
            ("LIVEKIT_API_KEY", &self.api_key),
            ("LIVEKIT_API_SECRET", &self.api_secret),
            ("LIVEKIT_WORKER_TOKEN", &self.worker_token),
            ("LIVEKIT_AGENT_DEPLOYMENT", &self.deployment),
            ("LIVEKIT_AGENT_ENDPOINTS_URL", &self.endpoints_url),
        ] {
            if !value.is_empty() {
                env.push((name, value.clone()));
            }
        }
        if let Some(dir) = &self.prometheus_multiproc_dir {
            env.push(("PROMETHEUS_MULTIPROC_DIR", dir.clone()));
        }
        env
    }

    /// The worker's registration and the connection it is made on.
    pub(crate) fn worker(
        &self,
        devmode: bool,
        endpoints: Vec<proto::agent_http::AgentEndpoint>,
    ) -> (WorkerConfig, RuntimeConfig) {
        let worker = WorkerConfig {
            agent_name: self.agent_name.clone(),
            deployment: self.deployment.clone(),
            version: self.version.clone(),
            permissions: self.permissions.clone(),
            endpoints,
            instance_id: crate::config::new_instance_id(),
            ws_url: self.ws_url.clone(),
            load_threshold: EnvOption::Fixed(if self.simulation {
                f32::INFINITY
            } else {
                self.load_threshold.get(devmode)
            }),
            num_idle_processes: self.num_idle_processes.unwrap_or(EnvOption::Fixed(0)),
            devmode,
            drain_timeout: std::time::Duration::try_from_secs_f64(self.drain_timeout).ok(),
        };
        let server_url = if self.worker_url.is_empty() { &self.ws_url } else { &self.worker_url };
        let connection = RuntimeConfig {
            server_url: server_url.clone(),
            worker_token: Some(self.worker_token.clone()).filter(|token| !token.is_empty()),
            headers: Vec::new(),
            api_key: self.api_key.clone(),
            api_secret: self.api_secret.clone(),
            max_retry: self.max_retry,
            connection_timeout: crate::config::CONNECTION_TIMEOUT,
        };
        (worker, connection)
    }
}

/// The Prometheus multiprocess directory a run uses: the configured one, else the inherited
/// `PROMETHEUS_MULTIPROC_DIR`; created, and emptied of a previous run's files.
pub fn prepare_prometheus_dir(configured: Option<&str>, logging: LogWriter) -> Option<String> {
    let dir = configured
        .map(str::to_owned)
        .or_else(|| std::env::var("PROMETHEUS_MULTIPROC_DIR").ok().filter(|d| !d.is_empty()))?;
    if let Err(e) = std::fs::create_dir_all(&dir) {
        logging.emit(
            log::Level::Warn,
            format!("failed to create the prometheus directory path={dir} error={e}"),
        );
        return Some(dir);
    }
    logging
        .emit(log::Level::Debug, format!("cleaning prometheus multiprocess directory path={dir}"));
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for path in entries.flatten().map(|e| e.path()).filter(|p| p.is_file()) {
            if let Err(e) = std::fs::remove_file(&path) {
                logging.emit(log::Level::Warn, format!("failed to remove {}: {e}", path.display()));
            }
        }
    }
    Some(dir)
}

/// The agent server's environment reads, as the host calls them.
pub struct Environment;

impl lk_agents_abi::worker::types::ServerEnvironment for Environment {
    fn cpu_count() -> f64 {
        f64::from(crate::CpuMonitor::new().cpu_count())
    }
}

/// Where the agent's endpoints are reached, once the server knows: LiveKit Cloud routes them, or
/// the worker's public listener serves them at `base` (`http://<listener>/agents/<agent>/<dep>`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Serving {
    #[default]
    Unknown,
    Cloud,
    Local {
        base: String,
    },
}

impl ServerSettings {
    /// `/agents/<agent>/<deployment>`; a worker registered without a deployment is `default`.
    pub fn endpoint_prefix(&self) -> String {
        let deployment = if self.deployment.is_empty() { "default" } else { &self.deployment };
        format!("/agents/{}/{deployment}", self.agent_name)
    }

    /// Cloud routes the endpoints: the registration said so, or the worker is hosted there.
    fn cloud_routed(&self, serving: &Serving) -> bool {
        *serving == Serving::Cloud || (*serving == Serving::Unknown && self.cloud())
    }

    /// The URL `endpoint` publishes in its card: Cloud's; its path when the worker serves it,
    /// for the card to put on the address the request reached.
    pub fn endpoint_url(&self, serving: &Serving, endpoint: &str) -> Option<String> {
        if let Serving::Local { .. } = serving {
            return Some(format!("{}/{endpoint}", self.endpoint_prefix()));
        }
        if !self.cloud_routed(serving) {
            return None;
        }
        endpoint_url_base(&self.ws_url, &self.agent_name, &self.deployment)
            .map(|base| format!("{base}/{endpoint}"))
    }

    /// Where this worker's own delegations to `endpoint` go: Cloud, in the worker's deployment
    /// (so development calls never reach production), or the worker's public listener.
    pub fn delegation_url(&self, serving: &Serving, endpoint: &str) -> Result<String, String> {
        if let Serving::Local { base } = serving {
            return Ok(format!("{base}/{endpoint}"));
        }
        if *serving == Serving::Unknown && !self.endpoints_url.is_empty() {
            return Ok(format!("{}/{endpoint}", self.endpoints_url));
        }
        if !self.cloud_routed(serving) {
            return Err(format!(
                "endpoint {endpoint:?} is not reachable yet: the server has not started serving"
            ));
        }
        self.endpoint_url(serving, endpoint)
            .ok_or_else(|| "delegating to an endpoint needs LIVEKIT_URL and an agent_name".into())
    }
}

/// `https://<project>/agents/<agent>/<deployment>`, where the text doors are reached (plan 7.2),
/// when the URL and the agent are known; a worker registered without a deployment is `default`.
pub fn endpoint_url_base(ws_url: &str, agent_name: &str, deployment: &str) -> Option<String> {
    if ws_url.is_empty() || agent_name.is_empty() {
        return None;
    }
    let base = ws_url.trim_end_matches('/');
    let base = match (base.strip_prefix("wss://"), base.strip_prefix("ws://")) {
        (Some(rest), _) => format!("https://{rest}"),
        (None, Some(rest)) => format!("http://{rest}"),
        (None, None) => base.to_owned(),
    };
    let deployment = if deployment.is_empty() { "default" } else { deployment };
    Some(format!("{base}/agents/{agent_name}/{deployment}"))
}
