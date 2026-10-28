//! `AgentServer` around the worker connection (`worker.py`): its options resolved once against the
//! environment, the health listener and its `/worker` record, the job `simulate_job` hands the
//! host, and the lifecycle (start, load, drain, close) the binding runs. The host keeps the job
//! processes and the user's callbacks.

use crate::{
    Worker, WorkerCommand, WorkerHandle, WorkerHost, WorkerNotification, WorkerRegistered,
};
use async_trait::async_trait;
use futures_util::{FutureExt, StreamExt, future::BoxFuture, stream::FuturesUnordered};
use lifecycle::{ServerEffect, ServerInput, ServerLifecycle};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Duration;
use tokio::sync::{Mutex, watch};
mod health;
mod lifecycle;
pub use health::{
    HealthProbe, ServerStatus, bind_listener, serve_health, serve_metrics, worker_info_json,
};
#[cfg(test)]
use health::{health_route, metrics_route};
pub use lifecycle::{ServerPhase, ServerResource};

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

/// The agent server's environment reads, as the host calls them.
pub struct Environment;

impl lk_agents_abi::worker::worker::ServerEnvironment for Environment {
    fn cpu_count() -> f64 {
        f64::from(crate::CpuMonitor::new().cpu_count())
    }

    fn endpoint_url_base(ws_url: String, agent_name: String, deployment: String) -> Option<String> {
        endpoint_url_base(&ws_url, &agent_name, &deployment)
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
        None if fake_job => lk_agents_net::strings::shortuuid("fake-agent-"),
        None => return Err("agent_identity is None but fake_job is False".into()),
    };
    let room_info = match room_info {
        Some(info) => info,
        None if fake_job => proto::Room {
            sid: lk_agents_net::strings::shortuuid("SRM_"),
            name: room.to_owned(),
            ..Default::default()
        },
        None => return Err("room_info is None but fake_job is False".into()),
    };
    let job = proto::Job {
        id: lk_agents_net::strings::shortuuid(if fake_job { "mock-job-" } else { "job-" }),
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

#[async_trait]
pub trait ServerHost: WorkerHost + Clone + Send + Sync + 'static {
    fn configured(&self, has_rtc: bool, has_on_request: bool, user_load: bool) -> Self;
    async fn start(
        &self,
        settings: ServerSettings,
        devmode: bool,
        has_rtc: bool,
    ) -> Result<(), String>;
    async fn started(&self) -> Result<(), String>;
    async fn registered(&self, info: WorkerRegistered) -> Result<(), String>;
    async fn idle_processes(&self, target: u32) -> Result<(), String>;
    async fn worker_load(&self, load: f32) -> Result<(), String>;
    async fn close_websockets(&self) -> Result<(), String>;
    async fn join_jobs(&self) -> Result<(), String>;
    async fn close(&self) -> Result<(), String>;
    async fn active_jobs(&self) -> usize;
    async fn metrics(&self) -> Option<(String, String)>;
}

pub struct ServerStart {
    pub settings: ServerSettings,
    pub devmode: bool,
    pub unregistered: bool,
    pub has_rtc: bool,
    pub has_text_doors: bool,
    pub has_load_fnc: bool,
    pub has_on_request: bool,
    pub publisher: bool,
    pub endpoints: Vec<crate::protocol::AgentEndpoint>,
    pub transport: crate::TransportConfig,
    pub worker_url: Option<String>,
}

#[derive(Clone)]
pub struct ServerSnapshot {
    pub phase: ServerPhase,
    pub worker_id: String,
    pub http_port: u16,
    pub worker: Option<WorkerHandle>,
    pub has_rtc: bool,
    pub incomplete_cleanup: Vec<ServerResource>,
}

#[derive(Clone, Default)]
struct Control {
    close: bool,
    drain: Option<Option<Duration>>,
}

type Outcome = watch::Sender<Option<Result<(), String>>>;

#[derive(Clone)]
pub struct ServerRuntime<H: ServerHost> {
    host: H,
    workers: Option<crate::io::WorkerIo>,
    control: watch::Sender<Control>,
    receiver: Arc<Mutex<Option<watch::Receiver<Control>>>>,
    snapshot: watch::Sender<ServerSnapshot>,
    outcome: Outcome,
    cleanup_outcome: Outcome,
    drain_outcome: Outcome,
}

impl<H: ServerHost> ServerRuntime<H> {
    pub fn new(host: H, workers: Option<crate::io::WorkerIo>) -> Self {
        let (control, receiver) = watch::channel(Control::default());
        Self {
            host,
            workers,
            control,
            receiver: Arc::new(Mutex::new(Some(receiver))),
            snapshot: watch::channel(ServerSnapshot {
                phase: ServerPhase::Idle,
                worker_id: "unregistered".into(),
                http_port: 0,
                worker: None,
                has_rtc: false,
                incomplete_cleanup: Vec::new(),
            })
            .0,
            outcome: watch::channel(None).0,
            cleanup_outcome: watch::channel(None).0,
            drain_outcome: watch::channel(None).0,
        }
    }

    pub fn snapshot(&self) -> ServerSnapshot {
        self.snapshot.borrow().clone()
    }

    pub async fn run(&self, start: ServerStart) -> Result<(), String> {
        let receiver = self.receiver.lock().await.take().ok_or("worker is already running")?;
        let origin = tokio::time::Instant::now();
        let timeout = Duration::try_from_secs_f64(start.settings.shutdown_process_timeout)
            .map_err(|_| "shutdown timeout must be finite and nonnegative")?;
        let process_cleanup = timeout * 3 + crate::ipc::supervisor::KILL_GRACE;
        let connection_cleanup = if start.unregistered {
            Duration::ZERO
        } else {
            crate::WorkerOptions::default().connection_timeout
        };
        let mut state = ServerLifecycle::new(process_cleanup.max(connection_cleanup));
        let effects = state.handle(ServerInput::Start, Duration::ZERO);
        self.publish(&state, effects);
        self.snapshot.send_modify(|snapshot| snapshot.has_rtc = start.has_rtc);
        let runtime = self.clone();
        tokio::spawn(async move { runtime.serve(start, receiver, state, origin).await });
        wait_outcome(&self.outcome).await
    }

    pub async fn simulate_job(&self, info: crate::RunningJobInfo) -> Result<(), String> {
        if self.snapshot().phase != ServerPhase::Running {
            return Err("the server is not accepting jobs".into());
        }
        self.host.launch_job(info).await.map(|_| ()).map_err(|error| error.to_string())
    }

    pub async fn drain(&self, timeout: Option<Duration>) -> Result<(), String> {
        match self.snapshot().phase {
            ServerPhase::Idle | ServerPhase::Closed => return Ok(()),
            ServerPhase::Closing => return wait_outcome(&self.cleanup_outcome).await,
            _ => {}
        }
        self.control.send_if_modified(|control| {
            if control.drain.is_some() {
                return false;
            }
            control.drain = Some(timeout);
            true
        });
        wait_outcome(&self.drain_outcome).await
    }

    pub async fn close(&self) -> Result<(), String> {
        if self.snapshot().phase == ServerPhase::Idle {
            return Ok(());
        }
        self.control.send_if_modified(|control| {
            if control.close {
                return false;
            }
            control.close = true;
            true
        });
        wait_outcome(&self.cleanup_outcome).await
    }

    fn publish(&self, state: &ServerLifecycle, effects: Vec<ServerEffect>) {
        self.snapshot.send_modify(|snapshot| snapshot.phase = state.phase());
        for effect in effects {
            match effect {
                ServerEffect::DrainFinished(result) => {
                    self.drain_outcome.send_if_modified(|outcome| {
                        if outcome.is_some() {
                            return false;
                        }
                        *outcome = Some(result.clone());
                        true
                    });
                }
                ServerEffect::CloseFinished { run_result, cleanup_result } => {
                    self.cleanup_outcome.send_if_modified(|outcome| {
                        if outcome.is_some() {
                            return false;
                        }
                        *outcome = Some(cleanup_result.clone());
                        true
                    });
                    self.outcome.send_if_modified(|outcome| {
                        if outcome.is_some() {
                            return false;
                        }
                        *outcome = Some(run_result.clone());
                        true
                    });
                }
                ServerEffect::IncompleteCleanup { resources } => {
                    self.snapshot
                        .send_modify(|snapshot| snapshot.incomplete_cleanup = resources.clone());
                    crate::diagnostics::emit(
                        log::Level::Error,
                        format!(
                            "server cleanup deadline missed; runtime retains ownership resources={resources:?}"
                        ),
                    );
                }
                _ => {}
            }
        }
    }

    async fn serve(
        self,
        start: ServerStart,
        mut control: watch::Receiver<Control>,
        mut state: ServerLifecycle,
        origin: tokio::time::Instant,
    ) {
        let mut expected = vec![ServerResource::Setup, ServerResource::Host, ServerResource::Load];
        if !start.settings.simulation {
            expected.push(ServerResource::Health);
        }
        if start.settings.prometheus_port.is_some() {
            expected.push(ServerResource::Metrics);
        }
        if !start.unregistered {
            expected.push(ServerResource::Worker);
        }
        for resource in expected {
            state.handle(ServerInput::ResourceStarted(resource), origin.elapsed());
        }
        let (stop, _) = watch::channel(false);
        let mut resources = Resources::default();
        let started = {
            let startup = self.start(&start, &stop, &mut resources);
            tokio::pin!(startup);
            loop {
                tokio::select! {
                    biased;
                    changed = control.changed() => {
                        if changed.is_err() { break None; }
                        let request = control.borrow_and_update().clone();
                        if request.drain.is_some() {
                            let effects = state.handle(ServerInput::Drain, origin.elapsed());
                            self.publish(&state, effects);
                        }
                        if request.close {
                            let effects = state.handle(ServerInput::Close, origin.elapsed());
                            self.publish(&state, effects);
                            break None;
                        }
                    }
                    result = &mut startup => break Some(result),
                }
            }
        };
        let mut drain = None;
        let mut started_notification = None;
        if let Some(result) = started {
            let readiness = result.as_ref().ok().cloned();
            let effects =
                state.handle(ServerInput::StartupFinished(result.map(|_| ())), origin.elapsed());
            let notify_started = effects.contains(&ServerEffect::Started);
            if effects.contains(&ServerEffect::Drain) {
                drain =
                    Some(self.drain_resources(
                        resources.worker.clone(),
                        control.borrow().drain.flatten(),
                    ));
            }
            self.publish(&state, effects);
            if notify_started {
                started_notification =
                    Some((self.host.started().boxed(), readiness.expect("startup succeeded")));
            }
        }
        while !matches!(state.phase(), ServerPhase::Closing | ServerPhase::Closed) {
            tokio::select! {
                biased;
                changed = control.changed() => {
                    if changed.is_err() { break; }
                    let request = control.borrow_and_update().clone();
                    if request.close {
                        let effects = state.handle(ServerInput::Close, origin.elapsed());
                        self.publish(&state, effects);
                    } else if request.drain.is_some() {
                        let effects = state.handle(ServerInput::Drain, origin.elapsed());
                        if effects.contains(&ServerEffect::Drain) {
                            drain = Some(self.drain_resources(resources.worker.clone(), request.drain.flatten()));
                        }
                        self.publish(&state, effects);
                    }
                }
                result = async { started_notification.as_mut().expect("notification pending").0.as_mut().await }, if started_notification.is_some() => {
                    let (_, readiness) = started_notification.take().expect("notification pending");
                    match result {
                        Ok(()) => { readiness.send_replace(true); }
                        Err(error) => {
                            let effects = state.handle(ServerInput::Failed(error), origin.elapsed());
                            self.publish(&state, effects);
                        }
                    }
                }
                result = pending_drain(&mut drain), if drain.is_some() => {
                    drain = None;
                    let effects = state.handle(ServerInput::DrainFinished(result), origin.elapsed());
                    self.publish(&state, effects);
                }
                result = pending_worker(&mut resources.connection), if resources.connection.is_some() => {
                    resources.connection = None;
                    let result = result.unwrap_or_else(|error| Err(error.to_string()));
                    let error = result.clone().err().unwrap_or_else(|| "worker connection stopped".into());
                    state.handle(ServerInput::ResourceStopped { resource: ServerResource::Worker, result }, origin.elapsed());
                    let effects = state.handle(ServerInput::Failed(error), origin.elapsed());
                    self.publish(&state, effects);
                }
            }
        }
        drop(drain);
        let effects = state.handle(ServerInput::Close, origin.elapsed());
        self.publish(&state, effects);
        drop(started_notification);
        stop.send_replace(true);
        let mut closing = resources.close(self.host.clone());
        while !closing.is_empty() {
            let delay = state.wake_at().map(|at| at.saturating_sub(origin.elapsed()));
            tokio::select! {
                biased;
                _ = deadline(delay) => {
                    let effects = state.tick(origin.elapsed());
                    self.publish(&state, effects);
                }
                result = closing.next() => {
                    if let Some((resource, result)) = result {
                        if let Err(error) = &result {
                            crate::diagnostics::emit(log::Level::Error, format!("server resource cleanup failed resource={resource:?} error={error}"));
                        }
                        let effects = state.handle(ServerInput::ResourceStopped { resource, result }, origin.elapsed());
                        self.publish(&state, effects);
                    }
                }
            }
        }
    }

    async fn start(
        &self,
        start: &ServerStart,
        stop: &watch::Sender<bool>,
        resources: &mut Resources,
    ) -> Result<watch::Sender<bool>, String> {
        let mut settings = start.settings.clone();
        if !start.has_rtc && !start.has_text_doors {
            return Err("No RTC session entrypoint has been registered".into());
        }
        if !start.unregistered {
            settings.check_registration(!start.endpoints.is_empty())?;
        }
        if settings.cloud() {
            settings.load_threshold = DEFAULT_LOAD_THRESHOLD;
        }
        let host = self.host.configured(
            start.has_rtc,
            start.has_on_request,
            start.has_load_fnc && !settings.cloud(),
        );
        let configured = settings.prometheus_multiproc_dir.clone();
        let discover_cpus = settings.num_idle_processes.is_none();
        let workers = match &self.workers {
            Some(workers) => workers.clone(),
            None => crate::io::WorkerIo::new(tokio::runtime::Handle::current(), 4)
                .map_err(|error| error.to_string())?,
        };
        let admission_deadline =
            Duration::try_from_secs_f64(settings.initialize_process_timeout)
                .map_err(|_| "initialization timeout must be finite and nonnegative")?;
        let prepare = async {
            loop {
                match workers.try_admit() {
                    Ok(permit) => {
                        break Ok(workers.spawn_admitted(permit, move || {
                            crate::diagnostics::initialize()?;
                            Ok((
                                prepare_prometheus_dir(configured.as_deref()),
                                discover_cpus.then(|| crate::CpuMonitor::new().cpu_count()),
                            ))
                        }));
                    }
                    Err(crate::io::WorkerIoError::Exhausted) => {
                        tokio::time::sleep(crate::ipc::pool::REFILL_INTERVAL).await;
                    }
                    Err(error) => break Err(error.to_string()),
                }
            }
        };
        resources.setup = Some(
            tokio::time::timeout(admission_deadline, prepare)
                .await
                .map_err(|_| "server setup capacity wait timed out".to_owned())??,
        );
        let setup = resources.setup.as_mut().expect("setup owned before awaiting").await;
        resources.setup.take();
        let (directory, cpus) = setup.map_err(|error| error.to_string())??;
        settings.prometheus_multiproc_dir = directory;
        if let Some(cpus) = cpus {
            settings.num_idle_processes =
                Some(crate::EnvOption::Split { dev: 0, prod: cpus.ceil() as u32 });
        }
        if !start.has_rtc {
            settings.num_idle_processes = Some(crate::EnvOption::Fixed(0));
        }
        let server_type =
            if start.publisher { crate::ServerType::Publisher } else { crate::ServerType::Room };
        let status = Arc::new(ServerStatus::default());
        let probe = Arc::new(HostHealth(host.clone()));
        if !settings.simulation {
            let (listener, port) = bind_listener(&settings.host, settings.port.get(start.devmode))
                .await
                .map_err(|error| error.to_string())?;
            self.snapshot.send_modify(|snapshot| snapshot.http_port = port);
            resources.health = Some(tokio::spawn(serve_health(
                listener,
                Arc::new(settings.clone()),
                server_type,
                status.clone(),
                probe.clone(),
                stop.subscribe(),
            )));
        }
        if let Some(port) = settings.prometheus_port {
            let (listener, _) =
                bind_listener(&settings.host, port).await.map_err(|error| error.to_string())?;
            resources.metrics =
                Some(tokio::spawn(serve_metrics(listener, probe, stop.subscribe())));
        }
        if settings.multiprocessing_fallback {
            crate::diagnostics::emit(
                log::Level::Warn,
                "the forkserver is used on Linux only, job processes are spawned".into(),
            );
        }
        host.start(settings.clone(), start.devmode, start.has_rtc).await?;
        if !start.unregistered {
            let options = settings.worker_options(
                start.devmode,
                server_type,
                start.endpoints.clone(),
                match &start.transport {
                    crate::TransportConfig::WebSocket { heartbeat } => {
                        crate::TransportConfig::WebSocket { heartbeat: *heartbeat }
                    }
                    crate::TransportConfig::WebTransport { insecure } => {
                        crate::TransportConfig::WebTransport { insecure: *insecure }
                    }
                },
                start.worker_url.clone(),
            );
            let (mut worker, handle) = Worker::new(options, host.clone(), workers.clone());
            resources.worker = Some(handle.clone());
            self.snapshot.send_modify(|snapshot| snapshot.worker = Some(handle.clone()));
            let mut notifications = handle.subscribe();
            let (host, snapshots, failed) =
                (self.host.clone(), self.snapshot.clone(), status.clone());
            resources.connection = Some(tokio::spawn(async move {
                let forward = async {
                    while let Ok(event) = notifications.recv().await {
                        match event {
                            WorkerNotification::Registered(info) => {
                                snapshots.send_modify(|snapshot| {
                                    snapshot.worker_id = info.worker_id.clone()
                                });
                                host.registered(info).await?;
                            }
                            WorkerNotification::Stopped(_) => break,
                            _ => {}
                        }
                    }
                    Ok::<(), String>(())
                };
                let run = worker.run();
                tokio::pin!(run);
                let result = tokio::select! {
                    result = &mut run => result.map_err(|error| error.to_string()),
                    result = forward => match result {
                        Ok(()) => run.await.map_err(|error| error.to_string()),
                        Err(error) => Err(error),
                    },
                };
                if result.is_err() {
                    failed.connection_failed.store(true, std::sync::atomic::Ordering::Release);
                }
                result
            }));
        }
        let (host, worker) = (self.host.clone(), resources.worker.clone());
        let readiness = status.ready.clone();
        resources.load = Some(tokio::spawn(async move {
            let mut interval = tokio::time::interval(crate::UPDATE_LOAD_INTERVAL);
            loop {
                interval.tick().await;
                if let Some(worker) = &worker {
                    let snapshot = worker.snapshot();
                    status.set_load(snapshot.worker_load);
                    if let Some(target) = snapshot.target_idle_processes {
                        host.idle_processes(target).await?;
                    }
                }
                host.worker_load(status.load()).await?;
            }
        }));
        Ok(readiness)
    }

    fn drain_resources(
        &self,
        worker: Option<WorkerHandle>,
        timeout: Option<Duration>,
    ) -> BoxFuture<'static, Result<(), String>> {
        let host = self.host.clone();
        async move {
            let drain = async {
                host.close_websockets().await?;
                match worker {
                    Some(worker) => worker.drain(timeout).await,
                    None => host.join_jobs().await,
                }
            };
            match timeout {
                Some(timeout) => tokio::time::timeout(timeout, drain)
                    .await
                    .map_err(|_| "drain timed out".to_string())?,
                None => drain.await,
            }
        }
        .boxed()
    }
}

async fn wait_outcome(outcome: &Outcome) -> Result<(), String> {
    outcome
        .subscribe()
        .wait_for(Option::is_some)
        .await
        .map_err(|error| error.to_string())?
        .clone()
        .unwrap_or(Ok(()))
}

async fn deadline(delay: Option<Duration>) {
    match delay {
        Some(delay) => tokio::time::sleep(delay).await,
        None => std::future::pending().await,
    }
}

async fn pending_drain(
    drain: &mut Option<BoxFuture<'static, Result<(), String>>>,
) -> Result<(), String> {
    match drain {
        Some(drain) => drain.await,
        None => std::future::pending().await,
    }
}

async fn pending_worker(
    worker: &mut Option<tokio::task::JoinHandle<Result<(), String>>>,
) -> Result<Result<(), String>, tokio::task::JoinError> {
    match worker {
        Some(worker) => worker.await,
        None => std::future::pending().await,
    }
}

#[derive(Default)]
struct Resources {
    setup: Option<crate::io::WorkerIoTask<Result<(Option<String>, Option<f32>), String>>>,
    worker: Option<WorkerHandle>,
    connection: Option<tokio::task::JoinHandle<Result<(), String>>>,
    load: Option<tokio::task::JoinHandle<Result<(), String>>>,
    health: Option<tokio::task::JoinHandle<Result<(), String>>>,
    metrics: Option<tokio::task::JoinHandle<Result<(), String>>>,
}

impl Resources {
    fn close<H: ServerHost>(
        self,
        host: H,
    ) -> FuturesUnordered<BoxFuture<'static, (ServerResource, Result<(), String>)>> {
        let closing = FuturesUnordered::new();
        closing.push(async move { (ServerResource::Host, host.close().await) }.boxed());
        closing.push(
            async move {
                let result = match self.setup {
                    Some(setup) => setup
                        .await
                        .map_err(|error| error.to_string())
                        .and_then(|result| result.map(|_| ())),
                    None => Ok(()),
                };
                (ServerResource::Setup, result)
            }
            .boxed(),
        );
        let close_admission = self
            .worker
            .map(|worker| {
                worker.send(WorkerCommand::Close).map_err(|_| {
                    worker
                        .snapshot()
                        .finished
                        .unwrap_or_else(|| "worker close admission failed".into())
                })
            })
            .unwrap_or(Ok(()));
        for (resource, task) in [
            (ServerResource::Worker, self.connection),
            (ServerResource::Load, self.load),
            (ServerResource::Health, self.health),
            (ServerResource::Metrics, self.metrics),
        ] {
            let admission =
                if resource == ServerResource::Worker { close_admission.clone() } else { Ok(()) };
            if resource == ServerResource::Load
                && let Some(task) = &task
            {
                task.abort();
            }
            closing.push(
                async move {
                    let result = match task {
                        Some(task) => match task.await {
                            Ok(result) => result,
                            Err(error)
                                if resource == ServerResource::Load && error.is_cancelled() =>
                            {
                                Ok(())
                            }
                            Err(error) => Err(error.to_string()),
                        },
                        None => Ok(()),
                    };
                    (resource, admission.and(result))
                }
                .boxed(),
            );
        }
        closing
    }
}

struct HostHealth<H>(H);

#[async_trait]
impl<H: ServerHost> HealthProbe for HostHealth<H> {
    async fn health(&self) -> usize {
        self.0.active_jobs().await
    }
    async fn metrics(&self) -> Option<(String, String)> {
        self.0.metrics().await
    }
}

#[cfg(test)]
mod lifecycle_tests;
#[cfg(test)]
mod tests;
