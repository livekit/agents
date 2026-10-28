//! `AgentServer` around the worker connection (`worker.py`): its options resolved once against the
//! environment, the health listener and its `/worker` record, the job `simulate_job` hands the
//! host, and the lifecycle (start, load, drain, close) the binding runs. The host keeps the job
//! processes and the user's callbacks.

use std::sync::Arc;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use livekit_protocol as proto;
use prost::Message as _;

use crate::{
    AcceptArguments, EnvOption, JobDecision, JobRequest, RunningJobInfo, ServerType,
    WORKER_PROTOCOL_VERSION, WorkerPermissions,
};

/// `load_threshold`'s default, to which a hosted (Cloud) worker is held.
pub const DEFAULT_LOAD_THRESHOLD: EnvOption<f32> = EnvOption::Split {
    dev: crate::config::DEFAULT_LOAD_THRESHOLD_DEV,
    prod: crate::config::DEFAULT_LOAD_THRESHOLD_PROD,
};

/// What `AgentServer(...)` was given, with the environment read where an option was left empty, as
/// `AgentServer.__init__` reads it. `update_options` writes over it until the server starts.
#[derive(Debug, Clone)]
pub struct ServerSettings {
    pub ws_url: String,
    pub api_key: String,
    pub api_secret: String,
    /// `LIVEKIT_WORKER_TOKEN`: set on a worker hosted on Cloud.
    pub worker_token: String,
    pub deployment: String,
    pub agent_name: String,
    /// The health listener's interface; empty is every one.
    pub host: String,
    pub port: EnvOption<u16>,
    pub load_threshold: EnvOption<f32>,
    pub num_idle_processes: Option<EnvOption<u32>>,
    /// `drain()`'s deadline in seconds; zero waits for every job.
    pub drain_timeout: f64,
    pub max_retry: u32,
    pub permissions: WorkerPermissions,
    pub http_proxy: Option<String>,
    pub prometheus_port: Option<u16>,
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
            deployment: String::new(),
            agent_name: String::new(),
            host: String::new(),
            port: EnvOption::Split { dev: 0, prod: 8081 },
            load_threshold: DEFAULT_LOAD_THRESHOLD,
            num_idle_processes: None,
            drain_timeout: crate::DRAIN_TIMEOUT.as_secs_f64(),
            max_retry: 16,
            permissions: WorkerPermissions::default(),
            http_proxy: None,
            prometheus_port: None,
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
        options: lk_agents_abi::worker::worker::ServerOptions,
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
        if let Given(value) = options.host {
            next.host = value;
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
        if let Given(value) = options.prometheus_port {
            next.prometheus_port = value;
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
        if let Given(value) = options.port {
            next.port = EnvOption::Fixed(value);
        }
        if let Given(value) = options.load_threshold {
            next.load_threshold = EnvOption::Fixed(value);
        }
        if let Given(value) = options.num_idle_processes {
            next.num_idle_processes = Some(EnvOption::Fixed(value));
        }
        if let Given(value) = options.drain_timeout {
            next.drain_timeout = value.unwrap_or(0.0);
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
        for (name, value) in [
            ("drain_timeout", self.drain_timeout),
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

    /// The environment read where `__init__` reads it: an empty URL or credential falls back to
    /// its variable, `LIVEKIT_AGENT_NAME_OVERRIDE` wins over `agent_name`, which wins over
    /// `LIVEKIT_AGENT_NAME`; the worker token and the deployment are only the environment's.
    pub fn with_env(mut self, read_env: impl Fn(&str) -> Option<String>) -> Self {
        let env = |name: &str| read_env(name).filter(|v| !v.is_empty());
        let or_env = |value: &mut String, name: &str| {
            if value.is_empty() {
                *value = env(name).unwrap_or_default();
            }
        };
        or_env(&mut self.ws_url, "LIVEKIT_URL");
        or_env(&mut self.api_key, "LIVEKIT_API_KEY");
        or_env(&mut self.api_secret, "LIVEKIT_API_SECRET");
        self.worker_token = env("LIVEKIT_WORKER_TOKEN").unwrap_or_default();
        self.deployment = env("LIVEKIT_AGENT_DEPLOYMENT").unwrap_or_default();
        if let Some(name) = env("LIVEKIT_AGENT_NAME_OVERRIDE") {
            self.agent_name = name;
        }
        or_env(&mut self.agent_name, "LIVEKIT_AGENT_NAME");
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
    pub fn check_registration(&self, has_routes: bool) -> Result<(), String> {
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
        if has_routes && self.agent_name.is_empty() {
            return Err(
                "server.http routes are served at /agents/{agent_name}/..., so they require \
                 an agent_name (AgentServer(agent_name=) or LIVEKIT_AGENT_NAME)"
                    .into(),
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

    /// The worker connection's options; strings are passed explicitly, so it reads no environment.
    pub fn worker_options(
        &self,
        devmode: bool,
        server_type: ServerType,
        endpoints: Vec<proto::agent_http::AgentEndpoint>,
        transport: crate::TransportConfig,
        worker_url: Option<String>,
    ) -> crate::WorkerOptions {
        crate::WorkerOptions {
            ws_url: Some(self.ws_url.clone()),
            worker_url: Some(worker_url.unwrap_or_default()),
            api_key: Some(self.api_key.clone()),
            api_secret: Some(self.api_secret.clone()),
            worker_token: Some(self.worker_token.clone()),
            agent_name: Some(self.agent_name.clone()),
            deployment: Some(self.deployment.clone()),
            server_type,
            version: self.version.clone(),
            permissions: self.permissions.clone(),
            endpoints,
            load_threshold: EnvOption::Fixed(self.load_threshold.get(devmode)),
            num_idle_processes: self.num_idle_processes,
            devmode,
            simulation: self.simulation,
            drain_timeout: std::time::Duration::try_from_secs_f64(self.drain_timeout).ok(),
            transport,
            max_retry: self.max_retry,
            ..crate::WorkerOptions::default()
        }
    }
}

#[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
/// The Prometheus multiprocess directory a run uses: the configured one, else the inherited
/// `PROMETHEUS_MULTIPROC_DIR`; created, and emptied of a previous run's files.
pub fn prepare_prometheus_dir(configured: Option<&str>) -> Option<String> {
    let dir = configured
        .map(str::to_owned)
        .or_else(|| std::env::var("PROMETHEUS_MULTIPROC_DIR").ok().filter(|d| !d.is_empty()))?;
    if let Err(e) = std::fs::create_dir_all(&dir) {
        crate::diagnostics::emit(
            log::Level::Warn,
            format!("failed to create the prometheus directory path={dir} error={e}"),
        );
        return Some(dir);
    }
    crate::diagnostics::emit(
        log::Level::Debug,
        format!("cleaning prometheus multiprocess directory path={dir}"),
    );
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for path in entries.flatten().map(|e| e.path()).filter(|p| p.is_file()) {
            if let Err(e) = std::fs::remove_file(&path) {
                crate::diagnostics::emit(
                    log::Level::Warn,
                    format!("failed to remove {}: {e}", path.display()),
                );
            }
        }
    }
    Some(dir)
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

// --------------------------------------------------------------------- simulated jobs

/// `simulate_job`'s job: a room job under `room` for `agent_identity` (read from `token` when one
/// is given), with a join token minted from the credentials unless given, or none for a fake job
/// without credentials.
#[allow(clippy::too_many_arguments)]
pub fn simulated_job(
    settings: &ServerSettings,
    worker_id: &str,
    room: &str,
    fake_job: bool,
    agent_identity: Option<String>,
    room_info: Option<proto::Room>,
    token: Option<String>,
) -> Result<RunningJobInfo, String> {
    let token = token.filter(|t| !t.is_empty());
    let agent_identity = match (&token, agent_identity) {
        (Some(token), _) => Some(crate::token_identity(token).map_err(|e| e.to_string())?),
        (None, identity) => identity,
    };
    let identity = match agent_identity {
        Some(identity) => identity,
        None if fake_job => lk_agents_abi::utils::strings::shortuuid("fake-agent-"),
        None => return Err("agent_identity is None but fake_job is False".into()),
    };
    let room_info = match room_info {
        Some(info) => info,
        None if fake_job => proto::Room {
            sid: lk_agents_abi::utils::strings::shortuuid("SRM_"),
            name: room.to_owned(),
            ..Default::default()
        },
        None => return Err("room_info is None but fake_job is False".into()),
    };
    let job = proto::Job {
        id: lk_agents_abi::utils::strings::shortuuid(if fake_job { "mock-job-" } else { "job-" }),
        room: Some(room_info),
        r#type: proto::JobType::JtRoom as i32,
        ..Default::default()
    };
    let token = match token {
        Some(token) => token,
        None if fake_job && (settings.api_key.is_empty() || settings.api_secret.is_empty()) => {
            String::new()
        }
        None => crate::agent_join_token(&settings.api_key, &settings.api_secret, &identity, room)
            .map_err(|e| e.to_string())?,
    };
    Ok(RunningJobInfo {
        accept_arguments: AcceptArguments {
            identity,
            name: String::new(),
            metadata: String::new(),
            attributes: Vec::new(),
        },
        job: crate::protocol::job_info(&job),
        url: settings.ws_url.clone(),
        token,
        worker_id: worker_id.to_owned(),
        fake_job,
        received_at: 0.0,
        accepted_at: 0.0,
        assigned_at: 0.0,
    })
}

// --------------------------------------------------------------------- job requests

/// The answer to a job request when the host's `on_request` is not needed: a server with no
/// voice door rejects it (without terminating, so another worker may take it) and one whose door
/// has no `on_request` accepts it as `JobRequest.accept()` does. `None`: ask the host.
pub fn own_decision(req: &JobRequest, has_rtc: bool, has_on_request: bool) -> Option<JobDecision> {
    if !has_rtc {
        crate::diagnostics::emit(
            log::Level::Warn,
            format!(
                "a job reached a server with no rtc_session; rejecting it job_id={} room={}",
                req.job.id,
                req.room_name()
            ),
        );
        return Some(JobDecision::Reject { terminate: false });
    }
    (!has_on_request).then(|| {
        JobDecision::Accept(AcceptArguments {
            identity: format!("agent-{}", req.job.id),
            ..AcceptArguments::default()
        })
    })
}

/// A room as `simulate_job` takes it, from its protobuf bytes.
pub fn decode_room(bytes: &[u8]) -> Result<proto::Room, String> {
    proto::Room::decode(bytes).map_err(|e| format!("room_info: {e}"))
}

// --------------------------------------------------------------------- the listeners

/// What the health and metrics listeners read from the host, beside the server's own state.
#[async_trait::async_trait]
pub trait HealthProbe: Send + Sync {
    /// The active jobs, for `/worker`.
    async fn health(&self) -> usize;
    /// The Prometheus exposition, `(content type, body)`; `None` when it cannot be rendered.
    async fn metrics(&self) -> Option<(String, String)>;
}

/// The server state the listener and `/worker` read.
pub struct ServerStatus {
    pub ready: tokio::sync::watch::Sender<bool>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    pub connection_failed: AtomicBool,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// `f32` bits.
    pub worker_load: AtomicU32,
}

impl Default for ServerStatus {
    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
    fn default() -> Self {
        Self {
            ready: tokio::sync::watch::channel(false).0,
            connection_failed: AtomicBool::new(false),
            worker_load: AtomicU32::new(0),
        }
    }
}

impl ServerStatus {
    pub fn load(&self) -> f32 {
        f32::from_bits(self.worker_load.load(Ordering::Relaxed))
    }

    pub fn set_load(&self, load: f32) {
        self.worker_load.store(load.to_bits(), Ordering::Relaxed);
    }
}

/// What `/worker` reports, as `MessageToJson(WorkerInfo, preserving_proto_field_name=True)` prints
/// it: field order, zero values left out, two-space indent.
pub fn worker_info_json(info: &proto::WorkerInfo) -> String {
    let mut m = serde_json::Map::new();
    let mut put = |k: &str, v: serde_json::Value, zero: bool| {
        if !zero {
            m.insert(k.to_owned(), v);
        }
    };
    put("worker_type", info.worker_type.clone().into(), info.worker_type.is_empty());
    put("agent_name", info.agent_name.clone().into(), info.agent_name.is_empty());
    put("active_jobs", info.active_jobs.into(), info.active_jobs == 0.0);
    put("sdk_version", info.sdk_version.clone().into(), info.sdk_version.is_empty());
    put("worker_load", info.worker_load.into(), info.worker_load == 0.0);
    put("protocol_version", info.protocol_version.into(), info.protocol_version == 0);
    serde_json::to_string_pretty(&serde_json::Value::Object(m)).unwrap_or_default()
}

/// One response: status line, content type, body.
type Response = (u16, &'static str, String, String);

const TEXT: &str = "text/plain; charset=utf-8";

fn text(code: u16, reason: &'static str, body: &str) -> Response {
    (code, reason, TEXT.into(), body.into())
}

/// `/` answers whether the worker is serving, `/worker` what it is.
async fn health_route(
    method: &str,
    path: &str,
    settings: &ServerSettings,
    server_type: ServerType,
    status: &ServerStatus,
    probe: &dyn HealthProbe,
) -> Response {
    let path = path.split('?').next().unwrap_or_default();
    if path != "/" && path != "/worker" {
        return text(404, "Not Found", "404: Not Found");
    }
    if method != "GET" && method != "HEAD" {
        return text(405, "Method Not Allowed", "405: Method Not Allowed");
    }
    if path == "/" {
        if !*status.ready.borrow() {
            return text(503, "Service Unavailable", "worker startup incomplete");
        }
        if status.connection_failed.load(Ordering::Relaxed) {
            return text(503, "Service Unavailable", "failed to connect to livekit");
        }
        return text(200, "OK", "OK");
    }
    let info = proto::WorkerInfo {
        worker_type: match server_type {
            ServerType::Room => "JT_ROOM",
            ServerType::Publisher => "JT_PUBLISHER",
        }
        .into(),
        agent_name: settings.agent_name.clone(),
        active_jobs: probe.health().await as f32,
        sdk_version: settings.version.clone(),
        worker_load: status.load(),
        protocol_version: WORKER_PROTOCOL_VERSION as i32,
    };
    (200, "OK", "application/json".into(), worker_info_json(&info))
}

/// `/metrics`: the host's Prometheus exposition.
async fn metrics_route(method: &str, path: &str, probe: &dyn HealthProbe) -> Response {
    if path.split('?').next().unwrap_or_default() != "/metrics" {
        return text(404, "Not Found", "404: Not Found");
    }
    if method != "GET" && method != "HEAD" {
        return text(405, "Method Not Allowed", "405: Method Not Allowed");
    }
    match probe.metrics().await {
        Some((content_type, body)) => (200, "OK", content_type, body),
        None => text(500, "Internal Server Error", "metrics unavailable"),
    }
}

/// The health listener on `listener` until dropped.
pub async fn serve_health(
    listener: tokio::net::TcpListener,
    settings: Arc<ServerSettings>,
    server_type: ServerType,
    status: Arc<ServerStatus>,
    probe: Arc<dyn HealthProbe>,
    stop: tokio::sync::watch::Receiver<bool>,
) -> Result<(), String> {
    serve(listener, stop, move |method, path| {
        let (settings, status, probe) = (settings.clone(), status.clone(), probe.clone());
        async move { health_route(&method, &path, &settings, server_type, &status, &*probe).await }
    })
    .await
}

/// The Prometheus listener (`prometheus_port`) on `listener` until dropped.
pub async fn serve_metrics(
    listener: tokio::net::TcpListener,
    probe: Arc<dyn HealthProbe>,
    stop: tokio::sync::watch::Receiver<bool>,
) -> Result<(), String> {
    serve(listener, stop, move |method, path| {
        let probe = probe.clone();
        async move { metrics_route(&method, &path, &*probe).await }
    })
    .await
}

#[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
/// Serves `listener` until dropped: one request per connection, answered by
/// `respond(method, path)`, then closed.
async fn serve<F, R>(
    listener: tokio::net::TcpListener,
    mut stop: tokio::sync::watch::Receiver<bool>,
    respond: F,
) -> Result<(), String>
where
    F: Fn(String, String) -> R + Send + Sync + 'static,
    R: std::future::Future<Output = Response> + Send + 'static,
{
    let respond = Arc::new(respond);
    let mut connections = tokio::task::JoinSet::new();
    let mut result = Ok(());
    loop {
        tokio::select! {
            biased;
            changed = stop.changed() => {
                if changed.is_err() || *stop.borrow_and_update() { break; }
            }
            joined = connections.join_next(), if !connections.is_empty() => {
                if let Some(Err(error)) = joined {
                    result = Err(format!("health/metrics connection: {error}"));
                    break;
                }
            }
            accepted = listener.accept(), if connections.len() < 64 => {
                match accepted {
                    Ok((socket, _)) => {
                        connections.spawn(serve_connection(socket, respond.clone()));
                    }
                    Err(error) => { result = Err(error.to_string()); break; }
                }
            }
        }
    }
    connections.abort_all();
    while let Some(joined) = connections.join_next().await {
        if let Err(error) = joined {
            if !error.is_cancelled() {
                result = Err(format!("health/metrics connection: {error}"));
            }
        }
    }
    result
}

async fn serve_connection<F, R>(mut socket: tokio::net::TcpStream, respond: Arc<F>)
where
    F: Fn(String, String) -> R + Send + Sync + 'static,
    R: std::future::Future<Output = Response> + Send + 'static,
{
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut buf = Vec::with_capacity(1024);
    let mut chunk = [0u8; 1024];
    let (method, path) = loop {
        let Ok(n) = socket.read(&mut chunk).await else { return };
        if n == 0 || buf.len() > 16 * 1024 {
            return;
        }
        buf.extend_from_slice(&chunk[..n]);
        let mut headers = [httparse::EMPTY_HEADER; 64];
        let mut req = httparse::Request::new(&mut headers);
        match req.parse(&buf) {
            Ok(httparse::Status::Complete(_)) => {
                break (
                    req.method.unwrap_or_default().to_owned(),
                    req.path.unwrap_or_default().to_owned(),
                );
            }
            Ok(httparse::Status::Partial) => continue,
            Err(_) => return,
        }
    };
    let (code, reason, content_type, body) = respond(method.clone(), path).await;
    let head = format!(
        "HTTP/1.1 {code} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
                 Connection: close\r\n\r\n",
        body.len()
    );
    let _ = socket.write_all(head.as_bytes()).await;
    if method != "HEAD" {
        let _ = socket.write_all(body.as_bytes()).await;
    }
    let _ = socket.shutdown().await;
}

/// A listener's socket on `host:port` (`host` empty: every interface) and the port it got.
pub async fn bind_listener(
    host: &str,
    port: u16,
) -> std::io::Result<(tokio::net::TcpListener, u16)> {
    let host = if host.is_empty() { "0.0.0.0" } else { host };
    let listener = tokio::net::TcpListener::bind((host, port)).await?;
    let port = listener.local_addr()?.port();
    Ok((listener, port))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> ServerSettings {
        ServerSettings { version: "2.0.0".into(), ..ServerSettings::default() }
    }

    fn env(vars: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| vars.iter().find(|(k, _)| *k == name).map(|(_, v)| (*v).to_owned())
    }

    // Catches: an argument losing to the environment, the override losing to the argument, or an
    // empty variable read as a value.
    #[test]
    fn options_fall_back_to_the_environment_and_the_override_wins() {
        let mut given = settings();
        given.ws_url = "wss://given".into();
        given.agent_name = "given-agent".into();
        let s = given.with_env(env(&[
            ("LIVEKIT_URL", "wss://env"),
            ("LIVEKIT_API_KEY", "key"),
            ("LIVEKIT_API_SECRET", ""),
            ("LIVEKIT_AGENT_NAME_OVERRIDE", "launcher"),
            ("LIVEKIT_WORKER_TOKEN", "hosted"),
        ]));
        assert_eq!(s.ws_url, "wss://given");
        assert_eq!(s.api_key, "key");
        assert_eq!(s.api_secret, "");
        assert_eq!(s.agent_name, "launcher");
        assert!(s.cloud());
        let s = settings().with_env(env(&[("LIVEKIT_AGENT_NAME", "from-env")]));
        assert_eq!(s.agent_name, "from-env");
        assert!(!s.cloud());
    }

    // Catches: a missing credential reported as another, or routes allowed without an agent name.
    #[test]
    fn registration_needs_the_url_the_credentials_and_a_name_for_routes() {
        let mut s = settings();
        assert!(s.check_registration(false).unwrap_err().starts_with("ws_url"));
        s.ws_url = "wss://x".into();
        assert!(s.check_registration(false).unwrap_err().starts_with("api_key"));
        s.api_key = "k".into();
        assert!(s.check_registration(false).unwrap_err().starts_with("api_secret"));
        s.api_secret = "s".into();
        assert!(s.check_registration(false).is_ok());
        assert!(s.check_registration(true).unwrap_err().contains("agent_name"));
    }

    #[test]
    fn invalid_server_update_cannot_partially_change_launch_configuration() {
        use lk_agents_abi::Maybe::Given;
        use lk_agents_abi::worker::worker::ServerOptions;
        let mut server = settings();
        let original_url = server.ws_url.clone();
        let error = server
            .apply(ServerOptions {
                ws_url: Given("wss://new.example".into()),
                initialize_process_timeout: Given(f64::NAN),
                ..Default::default()
            })
            .unwrap_err();
        assert!(error.contains("initialize_process_timeout"));
        assert_eq!(server.ws_url, original_url, "a refused update changes no option");
        server
            .apply(ServerOptions {
                ws_url: Given("wss://new.example".into()),
                initialize_process_timeout: Given(4.0),
                drain_timeout: Given(None),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(server.ws_url, "wss://new.example");
        assert_eq!(server.initialize_process_timeout, 4.0);
        assert_eq!(server.drain_timeout, 0.0, "explicit null disables the deadline");
        assert_eq!(server.shutdown_process_timeout, 10.0, "omission preserves other fields");
    }

    #[test]
    fn unsupported_start_method_is_refused_before_launch() {
        use lk_agents_abi::Maybe::Given;
        use lk_agents_abi::worker::worker::ServerOptions;
        let mut server = settings();
        let method = server.multiprocessing_context.clone();
        let error = server
            .apply(ServerOptions {
                multiprocessing_context: Given("fork".into()),
                ..Default::default()
            })
            .unwrap_err();
        assert!(error.contains("multiprocessing_context"));
        assert_eq!(server.multiprocessing_context, method);
    }

    struct Probe;

    #[async_trait::async_trait]
    impl HealthProbe for Probe {
        async fn health(&self) -> usize {
            0
        }

        async fn metrics(&self) -> Option<(String, String)> {
            Some(("text/plain; version=0.0.4".into(), "lk_agents_worker_load 0.5\n".into()))
        }
    }

    #[tokio::test]
    async fn endpoint_route_inventory_health_listener_serves_no_content() {
        let status = ServerStatus::default();
        assert_eq!(
            health_route("GET", "/", &settings(), ServerType::Room, &status, &Probe).await.0,
            503
        );
        status.ready.send_replace(true);
        assert_eq!(
            health_route("GET", "/", &settings(), ServerType::Room, &status, &Probe).await.0,
            200
        );
        for path in [
            "/threads",
            "/threads/AT_unknown/tasks",
            "/content/file.txt",
            "/.well-known/agent-card.json",
        ] {
            assert_eq!(
                health_route("GET", path, &settings(), ServerType::Room, &status, &Probe).await.0,
                404
            );
        }
    }

    // Catches: the metrics listener answering another path, or losing the host's content type.
    #[tokio::test]
    async fn the_metrics_listener_serves_the_host_exposition_at_metrics() {
        let (code, _, content_type, body) = metrics_route("GET", "/metrics", &Probe).await;
        assert_eq!((code, content_type.as_str()), (200, "text/plain; version=0.0.4"));
        assert_eq!(body, "lk_agents_worker_load 0.5\n");
        assert_eq!(metrics_route("GET", "/", &Probe).await.0, 404);
        assert_eq!(metrics_route("POST", "/metrics", &Probe).await.0, 405);
    }

    // Catches: `/worker` printing zero fields, out of order, or without the protobuf's indent.
    #[test]
    fn the_worker_record_prints_as_message_to_json() {
        let info = proto::WorkerInfo {
            worker_type: "JT_ROOM".into(),
            agent_name: String::new(),
            active_jobs: 1.0,
            sdk_version: "2.0.0".into(),
            worker_load: 0.0,
            protocol_version: 1,
        };
        assert_eq!(
            worker_info_json(&info),
            "{\n  \"worker_type\": \"JT_ROOM\",\n  \"active_jobs\": 1.0,\n  \"sdk_version\": \"2.0.0\",\n  \"protocol_version\": 1\n}"
        );
    }

    // Catches: a text-only server launching a process for a job, a default accept that differs
    // from `JobRequest.accept()`, or the host skipped when it has an `on_request`.
    #[test]
    fn a_server_answers_a_job_request_alone_unless_its_door_has_on_request() {
        let job = proto::Job { id: "AJ_1".into(), ..Default::default() };
        let req = JobRequest { job: crate::protocol::job_info(&job), resuming: false };
        assert_eq!(own_decision(&req, false, true), Some(JobDecision::Reject { terminate: false }));
        let Some(JobDecision::Accept(args)) = own_decision(&req, true, false) else {
            panic!("a door without on_request accepts")
        };
        assert_eq!((args.identity.as_str(), args.name.as_str()), ("agent-AJ_1", ""));
        assert_eq!(own_decision(&req, true, true), None);
    }

    // Catches: a fake job without credentials refused, a real one without its room accepted, or the
    // identity a given token names ignored.
    #[test]
    fn a_simulated_job_names_its_agent_and_room() {
        let s = settings();
        let fake =
            simulated_job(&s, "W1", "console-room", true, Some("console".into()), None, None)
                .unwrap();
        assert!(fake.job.id.starts_with("mock-job-"));
        assert_eq!(fake.job.room.as_deref(), Some("console-room"));
        assert_eq!((fake.accept_arguments.identity.as_str(), fake.token.as_str()), ("console", ""));
        assert_eq!(fake.worker_id, "W1");
        let real = simulated_job(&s, "W1", "r", false, Some("a".into()), None, None).unwrap_err();
        assert_eq!(real, "room_info is None but fake_job is False");
        let mut creds = settings();
        creds.api_key = "APIkey".into();
        creds.api_secret = "thiskeyistotallyunsafe".into();
        let minted = simulated_job(&creds, "W", "r", true, None, None, None).unwrap();
        assert!(minted.accept_arguments.identity.starts_with("fake-agent-"));
        let named =
            simulated_job(&creds, "W", "r", true, Some("x".into()), None, Some(minted.token))
                .unwrap();
        assert_eq!(named.accept_arguments.identity, minted.accept_arguments.identity);
    }
    #[allow(clippy::disallowed_methods, reason = "S0: runtime migration")]
    #[tokio::test]
    async fn server_shutdown_resources_joins_stalled_health_and_metrics_connections() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        for count in [1, 8, 32, 64] {
            let (listener, port) = bind_listener("127.0.0.1", 0).await.unwrap();
            let (stop, stopped) = tokio::sync::watch::channel(false);
            let serving = serve_metrics(listener, Arc::new(Probe), stopped);
            tokio::pin!(serving);
            let mut peers = Vec::new();
            for _ in 0..count {
                let mut peer = tokio::net::TcpStream::connect(("127.0.0.1", port)).await.unwrap();
                peer.write_all(b"GET /metrics HTTP/1.1\r\n").await.unwrap();
                peers.push(peer);
                assert!(futures_util::poll!(&mut serving).is_pending());
            }
            tokio::task::yield_now().await;
            stop.send_replace(true);
            tokio::time::timeout(std::time::Duration::from_secs(1), serving)
                .await
                .unwrap()
                .unwrap();
            for mut peer in peers {
                let result = tokio::time::timeout(
                    std::time::Duration::from_secs(1),
                    peer.read(&mut [0_u8; 1]),
                )
                .await
                .unwrap();
                assert!(
                    matches!(result, Ok(0)) || result.is_err(),
                    "every accepted connection must be closed"
                );
            }
        }
    }
}
