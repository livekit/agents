//! The agent server as the host drives it: `_core.ServerConfig` (`AgentServer`'s options, the
//! environment read where one was left empty, writable until the server starts) and
//! `_core.ServerRuntime` (the server over the host's job processes and callbacks, which it
//! reaches through the declared `ServerHost`), with endpoint streams dispatched to the host.

use crate::ipc::telemetry;
use lk_agents_abi::logging::LogWriter;
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
use lk_agents_abi::worker::ipc_host::WorkerError;
use lk_agents_abi::worker::server_host::{
    EndpointRoute, ServerConfig, ServerConfiguration, ServerHost as Host, ServerRuntime,
};
use lk_agents_abi::worker::types::{ProcessPoolOptions, ProcessStartOptions, ServerOptions};
use parking_lot::{Mutex, MutexGuard};

use crate::server::{self, HealthProbe, ServerPhase, ServerSettings, ServerStart};
use crate::transport::AgentEndpointKind;
use crate::{
    EnvOption, HEARTBEAT_INTERVAL, IncomingRequest, JobDecision, JobId, JobRequest, RunningJobInfo,
    TransportConfig, WorkerCommand, WorkerHost,
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
    fn new(options: ServerOptions) -> Result<Self, WorkerError> {
        let env = |name: &str| std::env::var(name).ok();
        let mut settings =
            ServerSettings { http_proxy: ServerSettings::env_proxy(env), ..Default::default() };
        settings.apply(options).map_err(WorkerError::Invalid)?;
        Ok(Self {
            settings: Arc::new(Mutex::new(settings.with_env(env))),
            serving: Arc::default(),
            running: Arc::default(),
        })
    }

    fn update(&self, options: ServerOptions) -> Result<(), WorkerError> {
        if self.running.load(Ordering::Acquire) {
            return Err(WorkerError::Failed(
                "cannot update options after starting the server".into(),
            ));
        }
        self.get().apply(options).map_err(WorkerError::Invalid)
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
        self.get().delegation_url(&serving, &endpoint).map_err(WorkerError::Failed)
    }
}

/// What serves the endpoint routes, once the server runs.
pub(crate) type Mounted = Arc<OnceLock<Arc<EndpointServer>>>;

/// The host server with what the runtime decides on its own.
#[derive(Clone)]
pub struct ServerCallbacks {
    pub(crate) proxy: Arc<dyn Host>,
    /// The user gave a `load_fnc` the worker may call (not on Cloud).
    user_load: bool,
    /// The server has a voice door.
    has_rtc: bool,
    pub(crate) mounted: Mounted,
    pub(crate) logging: LogWriter,
}

impl ServerCallbacks {
    pub(crate) fn new(proxy: Arc<dyn Host>, logging: LogWriter) -> Self {
        ServerCallbacks {
            proxy,
            user_load: true,
            has_rtc: false,
            mounted: Mounted::default(),
            logging,
        }
    }

    pub(crate) fn configured(&self, has_rtc: bool, user_load: bool) -> Self {
        Self { has_rtc, user_load, ..self.clone() }
    }

    pub(crate) async fn start(
        &self,
        start: ServerSettings,
        devmode: bool,
        has_rtc: bool,
    ) -> Result<(), String> {
        let options = ProcessStartOptions {
            devmode,
            job_processes: has_rtc,
            env: start.child_env().into_iter().map(|(key, value)| (key.into(), value)).collect(),
            job_executor_type: start.job_executor_type.clone(),
            multiprocessing_context: start.multiprocessing_context.clone(),
            pool: ProcessPoolOptions {
                num_idle_processes: start
                    .num_idle_processes
                    .unwrap_or(EnvOption::Fixed(0))
                    .get(devmode),
                initialize_timeout: start.initialize_process_timeout,
                close_timeout: start.shutdown_process_timeout,
                session_end_timeout: start.session_end_timeout,
                memory_warn_mb: start.job_memory_warn_mb,
                memory_limit_mb: start.job_memory_limit_mb,
                http_proxy: start.http_proxy.filter(|proxy| !proxy.is_empty()),
            },
        };
        self.proxy.start(&telemetry::Cx::NONE, options).await.map_err(|error| error.to_string())
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
    fn logger(&self) -> LogWriter {
        self.logging
    }

    async fn on_job_request(&self, req: JobRequest) -> JobDecision {
        server::job_decision(&req, self.has_rtc, self.logging)
    }

    async fn launch_job(&self, info: RunningJobInfo) -> Result<JobId, HostError> {
        self.proxy.launch_job(&telemetry::Cx::NONE, info).await
    }

    async fn terminate_job(&self, id: JobId) -> Result<(), HostError> {
        self.proxy.terminate_job(&telemetry::Cx::NONE, id).await
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
        self.proxy.active_jobs(&telemetry::Cx::NONE).await.unwrap_or(0)
    }

    async fn metrics(&self) -> Option<(String, String)> {
        self.proxy.metrics(&telemetry::Cx::NONE).await.ok()
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
    mounted: Mounted,
    spawner: TokioSpawner,
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
        let spawner = TokioSpawner::new((env.runtime)().map_err(WorkerError::Failed)?);
        let host = ServerCallbacks::new(host, env.logging);
        let mounted = host.mounted.clone();
        Ok(Self {
            settings: config.settings.clone(),
            serving: config.serving.clone(),
            running: config.running.clone(),
            runtime: server::ServerRuntime::new(host, None),
            mounted,
            spawner,
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
        context: CallContext,
    ) -> Result<(), WorkerError> {
        let routes = routes.into_iter().map(manifest_entry).collect::<Vec<_>>();
        let text = TextDoors {
            doors: text_doors,
            settings: self.settings.clone(),
            serving: self.serving.clone(),
            host_context: context.host_context,
        };
        let server = EndpointServer::new(endpoints, HttpServerLimits::default()).with_text(text);
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
        };
        self.running.store(true, Ordering::Release);
        let runtime = self.runtime.clone();
        self.spawner.run(async move { runtime.run(start).await }).await.map_err(WorkerError::Failed)
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
        self.spawner.run(async move { runtime.close().await }).await.map_err(WorkerError::Failed)
    }

    fn job_status(&self, job_id: String, status: String) {
        let status = match status.as_str() {
            "failed" => livekit_protocol::JobStatus::JsFailed,
            "success" => livekit_protocol::JobStatus::JsSuccess,
            _ => livekit_protocol::JobStatus::JsRunning,
        };
        if let Some(worker) = self.runtime.snapshot().worker
            && worker.send(WorkerCommand::JobStatus { job_id: job_id.clone(), status }).is_err()
        {
            self.runtime.logger().emit(
                log::Level::Error,
                format!("worker job status admission failed job_id={job_id}"),
            );
        }
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
        let room_info = room_info
            .map(|bytes| server::decode_room(&bytes))
            .transpose()
            .map_err(WorkerError::Invalid)?;
        let info = server::simulated_job(
            &self.settings(),
            &snapshot.worker_id,
            &room,
            fake_job,
            agent_identity,
            room_info,
            token,
        )
        .map_err(WorkerError::Invalid)?;
        let runtime = self.runtime.clone();
        self.spawner
            .run(async move { runtime.simulate_job(info).await })
            .await
            .map_err(WorkerError::Failed)
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
}
