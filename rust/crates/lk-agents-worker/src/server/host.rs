use crate::ipc::telemetry;
use std::any::Any;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::worker::http::{EndpointHandler, EndpointKind};

use crate::http::{EndpointServer, HttpServerLimits, TextDoors};
use lk_agents_abi::a2a::door::TextEndpoint;
use lk_agents_abi::invoke::CallContext;
use lk_agents_abi::worker::ipc::JobRunnerOptions;
use lk_agents_abi::worker::ipc_host::{JobLaunchers, WorkerError};
use lk_agents_abi::worker::job::JobExecutorType;
use lk_agents_abi::worker::server_host::{
    EndpointRoute, ServerConfig, ServerConfiguration, ServerHost as Host, ServerJobs, ServerRuntime,
};
use lk_agents_abi::worker::types::{ProcessOptions, ProcessStartOptions, ServerOptions};
use parking_lot::{Mutex, MutexGuard};

use crate::ipc::ProcessPool;
use crate::ipc::process_pool::JobObserver;
use crate::server::{self, HealthProbe, ServerPhase, ServerSettings, ServerStart};
use crate::transport::AgentEndpointKind;
use crate::{
    EnvOption, HEARTBEAT_INTERVAL, IncomingRequest, JobDecision, JobId, JobRequest, RunningJobInfo,
    TransportConfig, WorkerHost,
};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::task::TokioSpawner;

pub struct Configuration {
    settings: Arc<Mutex<ServerSettings>>,
    /// Where the endpoints are reached, written by the runtime once it knows.
    serving: Arc<Mutex<server::Serving>>,
    /// Seals the configuration once the server starts.
    running: Arc<AtomicBool>,
}

impl Configuration {
    fn get(&self) -> MutexGuard<'_, ServerSettings> {
        self.settings.lock()
    }
}

impl ServerConfiguration for Configuration {}

impl ServerConfig for Configuration {
    fn new(
        options: ServerOptions,
        process: ProcessOptions,
        version: String,
    ) -> Result<Self, WorkerError> {
        let env = |name: &str| std::env::var(name).ok();
        let mut settings = ServerSettings {
            http_proxy: ServerSettings::env_proxy(env),
            version,
            ..Default::default()
        };
        settings.apply(options, process)?;
        Ok(Self {
            settings: Arc::new(Mutex::new(settings.with_env(env))),
            serving: Arc::default(),
            running: Arc::default(),
        })
    }

    fn update(&self, options: ServerOptions, process: ProcessOptions) -> Result<(), WorkerError> {
        if self.running.load(Ordering::Acquire) {
            return Err(WorkerError::Failed(
                "cannot update options after starting the server".into(),
            ));
        }
        self.get().apply(options, process)
    }

    fn ws_url(&self) -> String {
        self.get().ws_url.clone()
    }

    fn api_key(&self) -> String {
        self.get().api_key.clone()
    }

    fn api_secret(&self) -> String {
        self.get().api_secret.clone()
    }

    fn agent_name(&self) -> String {
        self.get().agent_name.clone()
    }

    fn deployment(&self) -> String {
        self.get().deployment.clone()
    }

    fn card_version(&self) -> String {
        let deployment = self.deployment();
        if !deployment.is_empty() {
            return deployment;
        }
        std::env::var("LIVEKIT_AGENT_VERSION")
            .ok()
            .filter(|version| !version.is_empty())
            .unwrap_or_else(|| "1.0.0".into())
    }

    fn cloud(&self) -> bool {
        self.get().cloud()
    }

    fn drain_timeout(&self) -> f64 {
        self.get().drain_timeout
    }

    fn shutdown_timeout(&self) -> f64 {
        self.get().shutdown_timeout
    }

    fn simulation(&self) -> bool {
        self.get().simulation
    }

    fn set_simulation(&self, value: bool) {
        self.get().simulation = value;
    }

    fn endpoint_url(&self, endpoint: String) -> Option<String> {
        let serving = self.serving.lock().clone();
        self.get().endpoint_url(&serving, &endpoint)
    }

    fn delegation_url(&self, endpoint: String) -> Result<String, WorkerError> {
        let serving = self.serving.lock().clone();
        self.get().delegation_url(&serving, &endpoint)
    }
}

/// What serves the endpoint routes, once the server runs.
pub(crate) type Mounted = Arc<OnceLock<Arc<EndpointServer>>>;

/// The job processes as the server drives them: the native pool, or a test's fake.
#[async_trait]
pub trait JobPool: Send + Sync {
    /// Fill the pool for the first time.
    async fn start(&self) -> Result<(), HostError>;
    async fn launch(&self, info: RunningJobInfo) -> Result<(), HostError>;
    async fn terminate(&self, job_id: &str);
    fn set_target_idle(&self, target: usize);
    /// The jobs whose process runs them now.
    fn active_jobs(&self) -> Vec<RunningJobInfo>;
    /// Waits until every process a job was launched on exited (the drain).
    async fn join(&self);
    async fn close(&self);
}

/// What the server's pool is made of, once the host supplied its launchers.
pub struct PoolSetup {
    pub launchers: Arc<dyn JobLaunchers>,
    pub num_idle_processes: usize,
    pub options: JobRunnerOptions,
    /// The context the launchers' host calls run in.
    pub context: CallContext,
    pub observer: JobObserver,
}

pub type PoolFactory = Arc<dyn Fn(PoolSetup) -> Result<Arc<dyn JobPool>, HostError> + Send + Sync>;

/// The native pool of supervised job processes.
fn native_pools(env: ProcessEnv) -> PoolFactory {
    Arc::new(move |setup: PoolSetup| {
        let pool = ProcessPool::observed(
            &env,
            setup.launchers,
            setup.num_idle_processes,
            setup.options,
            setup.context,
            Some(setup.observer),
        )
        .map_err(|error| HostError::new(error.to_string()))?;
        Ok(Arc::new(pool) as Arc<dyn JobPool>)
    })
}

/// What the runtime hands the pool it starts.
pub(crate) struct PoolLaunch {
    pub context: CallContext,
    pub asyncio_debug: bool,
    pub observer: JobObserver,
}

/// The host server, the job processes the runtime owns, and what the runtime decides on its own.
#[derive(Clone)]
pub struct ServerCallbacks {
    pub(crate) proxy: Arc<dyn Host>,
    pools: PoolFactory,
    /// The job processes once the host started them; `None` when the server runs no jobs.
    pool: Arc<tokio::sync::SetOnce<Option<Arc<dyn JobPool>>>>,
    /// The user gave a `load_fnc` the worker may call (not on Cloud).
    user_load: bool,
    /// The server has a voice door.
    has_rtc: bool,
    pub(crate) mounted: Mounted,
}

impl ServerCallbacks {
    pub(crate) fn new(proxy: Arc<dyn Host>, pools: PoolFactory) -> Self {
        ServerCallbacks {
            proxy,
            pools,
            pool: Arc::default(),
            user_load: true,
            has_rtc: false,
            mounted: Mounted::default(),
        }
    }

    pub(crate) fn configured(&self, has_rtc: bool, user_load: bool) -> Self {
        Self { has_rtc, user_load, ..self.clone() }
    }

    fn pool(&self) -> Option<Arc<dyn JobPool>> {
        self.pool.get().cloned().flatten()
    }

    /// The host prepares the job processes and hands over their launchers; the pool fills.
    pub(crate) async fn start(
        &self,
        start: ServerSettings,
        devmode: bool,
        has_rtc: bool,
        launch: PoolLaunch,
    ) -> Result<(), HostError> {
        let options = ProcessStartOptions {
            job_processes: has_rtc,
            env: start.child_env().into_iter().map(|(key, value)| (key.into(), value)).collect(),
            multiprocessing_context: start.multiprocessing_context,
            session_end_timeout: start.session_end_timeout,
        };
        let launchers = match self.proxy.start(&telemetry::Cx::NONE, options).await {
            Ok(Some(launchers)) if has_rtc => launchers,
            started => {
                let _ = self.pool.set(None);
                return started.map(drop);
            }
        };
        let options = JobRunnerOptions {
            thread: start.job_executor_type == JobExecutorType::Thread,
            initialize_timeout: start.initialize_process_timeout,
            close_timeout: start.shutdown_process_timeout,
            memory_warn_mb: start.job_memory_warn_mb,
            memory_limit_mb: start.job_memory_limit_mb,
            ping_interval: 2.5,
            ping_timeout: start.ping_timeout,
            high_ping_threshold: 0.5,
            http_proxy: start.http_proxy.filter(|proxy| !proxy.is_empty()),
            asyncio_debug: launch.asyncio_debug,
        };
        let num_idle = start.num_idle_processes.unwrap_or(EnvOption::Fixed(0)).get(devmode);
        let pool = (self.pools)(PoolSetup {
            launchers,
            num_idle_processes: num_idle as usize,
            options,
            context: launch.context,
            observer: launch.observer,
        });
        let pool = match pool {
            Ok(pool) => pool,
            Err(error) => {
                let _ = self.pool.set(None);
                return Err(error);
            }
        };
        let _ = self.pool.set(Some(pool.clone()));
        pool.start().await
    }

    pub(crate) fn set_target_idle(&self, target: u32) {
        if let Some(pool) = self.pool() {
            pool.set_target_idle(target as usize);
        }
    }

    pub(crate) fn active_jobs(&self) -> Vec<RunningJobInfo> {
        self.pool().map(|pool| pool.active_jobs()).unwrap_or_default()
    }

    pub(crate) async fn join_jobs(&self) {
        if let Some(pool) = self.pool() {
            pool.join().await;
        }
    }

    /// Closes the job processes; a server that never started them releases the reloads waiting.
    pub(crate) async fn close_pool(&self) {
        let _ = self.pool.set(None);
        if let Some(pool) = self.pool() {
            pool.close().await;
        }
    }

    /// Jobs kept from before a reload, launched once the job processes started.
    pub(crate) async fn reload(&self, jobs: Vec<RunningJobInfo>) -> Result<(), WorkerError> {
        let Some(pool) = self.pool.wait().await.clone() else {
            return Err(WorkerError::Failed("a server with no rtc_session runs no job".into()));
        };
        for info in jobs {
            log::info!(target: lk_agents_abi::LOG_TARGET, job_id = info.job.id.as_str(); "reloading job");
            pool.launch(info).await?;
        }
        Ok(())
    }

    /// Admit no new exchange, close WebSockets and wait for the exchanges in flight.
    pub(crate) async fn drain_endpoints(&self) {
        if let Some(endpoints) = self.mounted.get() {
            endpoints.drain().await;
        }
    }
}

#[async_trait]
impl WorkerHost for ServerCallbacks {
    async fn on_job_request(&self, req: JobRequest) -> JobDecision {
        server::job_decision(&req, self.has_rtc)
    }

    async fn launch_job(&self, info: RunningJobInfo) -> Result<JobId, HostError> {
        let pool = self.pool().ok_or_else(|| HostError::new("the server runs no job processes"))?;
        let job_id = info.job.id.clone();
        pool.launch(info).await?;
        Ok(job_id)
    }

    async fn terminate_job(&self, id: JobId) -> Result<(), HostError> {
        if let Some(pool) = self.pool() {
            pool.terminate(&id).await;
        }
        Ok(())
    }

    /// A raising `load_fnc` must not read as idle (that over-schedules); `None` falls back to the
    /// CPU monitor.
    async fn user_load(&self) -> Option<f32> {
        if !self.user_load {
            return None;
        }
        Some(
            self.proxy.load(&telemetry::Cx::NONE).await.map(|load| load as f32).unwrap_or(f32::NAN),
        )
    }

    async fn serve_endpoint(&self, request: IncomingRequest) {
        crate::http::dispatch(request, self.mounted.get().map(Arc::as_ref)).await;
    }

    async fn stop_endpoints(&self) {
        if let Some(endpoints) = self.mounted.get() {
            endpoints.stop().await;
        }
    }
}

#[async_trait]
impl HealthProbe for ServerCallbacks {
    async fn health(&self) -> usize {
        self.active_jobs().len()
    }
}

fn env_flag(name: &str) -> bool {
    matches!(std::env::var(name).unwrap_or_default().to_lowercase().as_str(), "1" | "true" | "yes")
}

/// The transport the routes need: the endpoint data plane rides WebTransport; without routes the
/// WebSocket control connection suffices (`LIVEKIT_WORKER_WEBTRANSPORT` asks for it anyway).
fn transport_for(routes: bool) -> TransportConfig {
    if routes || env_flag("LIVEKIT_WORKER_WEBTRANSPORT") {
        TransportConfig::WebTransport { insecure: env_flag("LIVEKIT_WORKER_INSECURE") }
    } else {
        TransportConfig::WebSocket { heartbeat: HEARTBEAT_INTERVAL }
    }
}

/// A manifest route as the register message carries it.
/// A text door's route: the worker serves its card and JSON-RPC methods under its name.
fn door_route(name: &str) -> EndpointRoute {
    EndpointRoute {
        path: format!("/{name}/{{path:path}}"),
        methods: ["GET", "HEAD", "POST"].map(str::to_owned).to_vec(),
        kind: EndpointKind::Http,
        public: false,
    }
}

fn manifest_entry(route: EndpointRoute) -> livekit_protocol::agent_http::AgentEndpoint {
    let kind = match route.kind {
        EndpointKind::Http => AgentEndpointKind::AekHttp,
        EndpointKind::Text => AgentEndpointKind::AekText,
    };
    livekit_protocol::agent_http::AgentEndpoint {
        path: route.path,
        methods: route.methods,
        kind: kind as i32,
        public: route.public,
    }
}

pub struct AgentServerRuntime {
    settings: Arc<Mutex<ServerSettings>>,
    serving: Arc<Mutex<server::Serving>>,
    running: Arc<AtomicBool>,
    runtime: server::ServerRuntime,
    /// The runtime's host server, with the job processes the dev channel reads.
    host: ServerCallbacks,
    mounted: Mounted,
    spawner: TokioSpawner,
    /// The process's span recording, for the endpoint server's `http_request` spans.
    tracer: crate::ipc::telemetry::Tracer,
}

impl AgentServerRuntime {
    fn settings(&self) -> ServerSettings {
        self.settings.lock().clone()
    }
}

impl ServerRuntime for AgentServerRuntime {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        config: Arc<dyn ServerConfiguration>,
        host: Arc<dyn Host>,
    ) -> Result<Self, WorkerError> {
        let config: Arc<dyn Any + Send + Sync> = config;
        let config = config
            .downcast::<Configuration>()
            .map_err(|_| WorkerError::Invalid("expected a native ServerConfig".into()))?;
        let runtime = (env.runtime)().map_err(|error| WorkerError::Failed(error.to_string()))?;
        let spawner = TokioSpawner::new(runtime);
        let host = ServerCallbacks::new(host, native_pools(env.clone()));
        let mounted = host.mounted.clone();
        Ok(Self {
            settings: config.settings.clone(),
            serving: config.serving.clone(),
            running: config.running.clone(),
            runtime: server::ServerRuntime::new(host.clone(), None),
            host,
            mounted,
            spawner,
            tracer: env.tracer.clone(),
        })
    }

    fn worker_id(&self) -> String {
        self.runtime.snapshot().worker_id
    }

    fn draining(&self) -> bool {
        self.runtime.snapshot().phase == ServerPhase::Draining
    }

    fn http_port(&self) -> u16 {
        self.runtime.snapshot().http_port
    }

    fn active_jobs(&self) -> Vec<RunningJobInfo> {
        self.host.active_jobs()
    }

    async fn run(
        &self,
        devmode: bool,
        unregistered: bool,
        has_rtc: bool,
        has_text_doors: bool,
        routes: Vec<EndpointRoute>,
        has_load_fnc: bool,
        endpoints: Arc<dyn EndpointHandler>,
        text_doors: Vec<Arc<dyn TextEndpoint>>,
        asyncio_debug: bool,
        context: CallContext,
    ) -> Result<(), WorkerError> {
        let door_routes = text_doors.iter().map(|door| door_route(door.name()));
        let routes = door_routes.chain(routes).map(manifest_entry).collect::<Vec<_>>();
        let text = TextDoors {
            doors: text_doors,
            settings: self.settings.clone(),
            serving: self.serving.clone(),
            host_context: context.host_context.clone(),
        };
        let limits = HttpServerLimits::default();
        let server = EndpointServer::new(endpoints, limits, self.tracer.clone()).with_text(text);
        let _ = self.mounted.set(Arc::new(server));
        let start = ServerStart {
            settings: self.settings(),
            devmode,
            unregistered,
            has_rtc,
            has_text_doors,
            transport: transport_for(!routes.is_empty()),
            endpoints: routes,
            has_load_fnc,
            serving: self.serving.clone(),
            context,
            asyncio_debug,
        };
        self.running.store(true, Ordering::Release);
        let runtime = self.runtime.clone();
        let ran = self.spawner.run(async move { runtime.run(start).await }).await;
        ran.map_err(|error| WorkerError::Failed(error.to_string()))
    }

    async fn drain(&self, timeout: Option<f64>) -> Result<(), WorkerError> {
        let timeout = timeout.map(Duration::try_from_secs_f64).transpose().map_err(|_| {
            WorkerError::Invalid("drain timeout must be finite and nonnegative".into())
        })?;
        let runtime = self.runtime.clone();
        self.spawner
            .run(async move { runtime.drain(timeout).await })
            .await
            .map_err(WorkerError::from)
    }

    async fn aclose(&self) -> Result<(), WorkerError> {
        let runtime = self.runtime.clone();
        let closed = self.spawner.run(async move { runtime.close().await }).await;
        closed.map_err(|error| WorkerError::Failed(error.to_string()))
    }

    async fn simulate_job(
        &self,
        room: String,
        fake_job: bool,
        agent_identity: Option<String>,
        room_info: Option<Vec<u8>>,
        token: Option<String>,
    ) -> Result<(), WorkerError> {
        let snapshot = self.runtime.snapshot();
        if !snapshot.has_rtc {
            return Err(WorkerError::Failed("a server with no rtc_session runs no job".into()));
        }
        if snapshot.phase != ServerPhase::Running {
            return Err(WorkerError::Failed("the server is not accepting jobs".into()));
        }
        let room_info = room_info.map(|bytes| server::decode_room(&bytes)).transpose()?;
        let info = server::simulated_job(
            &self.settings(),
            &snapshot.worker_id,
            &room,
            server::SimulatedJobOptions { fake_job, agent_identity, room_info, token },
        )?;
        let runtime = self.runtime.clone();
        self.spawner
            .run(async move { runtime.simulate_job(info).await })
            .await
            .map_err(|error| WorkerError::Failed(error.to_string()))
    }
}

impl ServerJobs for AgentServerRuntime {
    fn running_jobs(&self) -> Vec<RunningJobInfo> {
        self.host.active_jobs()
    }

    fn reload_jobs(
        &self,
        jobs: Vec<RunningJobInfo>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(), WorkerError>> {
        let host = self.host.clone();
        Box::pin(async move { self.spawner.run(async move { host.reload(jobs).await }).await })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a manifest entry's kind or methods lost on the way to the register message, or an
    // HTTP routes registered as text routes.
    #[test]
    fn routes_reach_the_manifest_with_their_typed_kind() {
        let route = |path: &str, kind: EndpointKind| EndpointRoute {
            path: path.into(),
            methods: vec!["GET".into(), "HEAD".into()],
            kind,
            public: true,
        };
        let text = manifest_entry(route("/a2a", EndpointKind::Text));
        let http = manifest_entry(route("/items/{id}", EndpointKind::Http));
        assert_eq!(text.kind, AgentEndpointKind::AekText as i32);
        assert_eq!(http.kind, AgentEndpointKind::AekHttp as i32);
        assert_eq!(http.methods, vec!["GET", "HEAD"]);
        assert!(http.public);
        assert!(matches!(transport_for(true), TransportConfig::WebTransport { .. }));
    }

    // Catches: a text door unrouted, routed publicly, or without its POST.
    #[test]
    fn a_text_door_is_routed_privately_under_its_name() {
        let entry = manifest_entry(door_route("support"));
        assert_eq!(entry.path, "/support/{path:path}");
        assert_eq!(entry.methods, vec!["GET", "HEAD", "POST"]);
        assert!(!entry.public && entry.kind == AgentEndpointKind::AekHttp as i32);
    }
}
