//! The agent server as the host drives it: its configuration, its runtime, and the host server
//! (`worker._ServerCallbacks`) that owns the job processes and the user's callbacks.

use std::any::Any;
use std::sync::Arc;

use crate::a2a::door::TextEndpoint;
use crate::errors::HostError;
use crate::invoke::CallContext;
use crate::worker::http::{EndpointHandler, EndpointKind};
use crate::worker::ipc_host::WorkerError;
use crate::worker::types::{ProcessStartOptions, RunningJobInfo, ServerOptions};

/// Preserve the native configuration identity shared by a server and its options surface.
#[crate::host(native(refused = "expected a native ServerConfig"))]
pub trait ServerConfiguration: Any + Send + Sync {}

#[crate::host(class(name = "ServerConfig", native = ServerConfiguration))]
pub trait ServerConfig: Send + Sync + Sized + 'static {
    /// `options` left empty read the environment (`http_proxy`: `HTTPS_PROXY`, else
    /// `HTTP_PROXY`).
    #[host(new)]
    fn new(options: ServerOptions) -> Result<Self, WorkerError>;

    /// Before the server starts only.
    #[host(call)]
    fn update(&self, options: ServerOptions) -> Result<(), WorkerError>;

    #[host(getter)]
    fn ws_url(&self) -> String;
    #[host(getter)]
    fn api_key(&self) -> String;
    #[host(getter)]
    fn api_secret(&self) -> String;
    #[host(getter)]
    fn agent_name(&self) -> String;
    #[host(getter)]
    fn deployment(&self) -> String;
    /// Hosted on Cloud (`LIVEKIT_WORKER_TOKEN` set).
    #[host(getter)]
    fn cloud(&self) -> bool;
    #[host(getter)]
    fn drain_timeout(&self) -> f64;
    #[host(getter)]
    fn shutdown_timeout(&self) -> f64;
    #[host(getter)]
    fn simulation(&self) -> bool;
    #[host(setter, name = "simulation")]
    fn set_simulation(&self, value: bool);

    /// The address `endpoint` publishes in its card: its Cloud URL when Cloud routes it; its
    /// path when this worker serves it, put on the address the request reached.
    #[host(call)]
    fn endpoint_url(&self, endpoint: String) -> Option<String>;

    /// Where this worker delegates to its own `endpoint`: Cloud, in the worker's deployment, or
    /// the worker's public listener. Raises before the server knows which.
    #[host(call)]
    fn delegation_url(&self, endpoint: String) -> Result<String, WorkerError>;
}

/// One route of the endpoint manifest the worker registers.
#[derive(Debug, Clone, PartialEq, Eq)]
#[crate::host(record(module = "worker_server", extract))]
pub struct EndpointRoute {
    pub path: String,
    pub methods: Vec<String>,
    /// `"http"`, or `"text"` for a session endpoint.
    pub kind: EndpointKind,
    pub public: bool,
}

/// The job processes and the user's callbacks, which the server runtime calls on the host's
/// loop. Which jobs to take, when, and whether job processes start at all are the runtime's.
#[crate::host(proxy(host = "livekit.agents.worker._ServerCallbacks"))]
pub trait ServerHost: Send + Sync {
    /// A job the runtime took, with the timeline it stamped; its LiveKit job id.
    #[host(await, name = "_core_launch_job")]
    fn launch_job(
        &self,
        cx: &crate::telemetry::Cx,
        info: RunningJobInfo,
    ) -> crate::BoxFuture<'_, Result<String, HostError>>;
    #[host(await, name = "_core_terminate_job")]
    fn terminate_job(
        &self,
        cx: &crate::telemetry::Cx,
        job_id: String,
    ) -> crate::BoxFuture<'_, Result<(), HostError>>;
    /// The user's `load_fnc`, asked only when there is one.
    #[host(await, name = "_core_load")]
    fn load(&self, cx: &crate::telemetry::Cx) -> crate::BoxFuture<'_, Result<f64, HostError>>;
    /// The job processes, when the runtime wants them.
    #[host(await, name = "_core_start")]
    fn start(
        &self,
        cx: &crate::telemetry::Cx,
        options: ProcessStartOptions,
    ) -> crate::BoxFuture<'_, Result<(), HostError>>;
    #[host(await, name = "_core_started")]
    fn started(&self, cx: &crate::telemetry::Cx) -> crate::BoxFuture<'_, Result<(), HostError>>;
    #[host(await, name = "_core_idle_processes")]
    fn idle_processes(
        &self,
        cx: &crate::telemetry::Cx,
        target: u32,
    ) -> crate::BoxFuture<'_, Result<(), HostError>>;
    #[host(await, name = "_core_worker_load")]
    fn worker_load(
        &self,
        cx: &crate::telemetry::Cx,
        load: f64,
    ) -> crate::BoxFuture<'_, Result<(), HostError>>;
    #[host(await, name = "_core_join_jobs")]
    fn join_jobs(&self, cx: &crate::telemetry::Cx) -> crate::BoxFuture<'_, Result<(), HostError>>;
    #[host(await, name = "_core_close")]
    fn close(&self, cx: &crate::telemetry::Cx) -> crate::BoxFuture<'_, Result<(), HostError>>;
    /// The jobs running, for the health check.
    #[host(await, name = "_core_health")]
    fn active_jobs(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<usize, HostError>>;
    /// The Prometheus exposition and its content type.
    #[host(await, name = "_core_metrics")]
    fn metrics(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<(String, String), HostError>>;
}

#[crate::host(class(name = "ServerRuntime"))]
pub trait ServerRuntime: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(
        env: &Self::Env,
        config: Arc<dyn ServerConfiguration>,
        host: Arc<dyn ServerHost>,
    ) -> Result<Self, WorkerError>;

    #[host(getter)]
    fn worker_id(&self) -> String;
    #[host(getter)]
    fn draining(&self) -> bool;
    #[host(getter)]
    fn http_port(&self) -> u16;

    /// Start the job processes and the health listener and, unless `unregistered`, register;
    /// return once closed. The worker serves each `text` route with its door from `text_doors`,
    /// whose handlers run in the context this call captured; `endpoints` receives every other
    /// stream for `routes`.
    #[host(await)]
    #[allow(clippy::too_many_arguments)]
    async fn run(
        &self,
        #[host(kw)] devmode: bool,
        unregistered: bool,
        has_rtc: bool,
        has_text_doors: bool,
        routes: Vec<EndpointRoute>,
        has_load_fnc: bool,
        endpoints: Arc<dyn EndpointHandler>,
        text_doors: Vec<Arc<dyn TextEndpoint>>,
        #[host(context)] context: CallContext,
    ) -> Result<(), WorkerError>;

    /// `None` (or `0`) waits without a deadline; past it raises `TimeoutError`.
    #[host(await)]
    async fn drain(&self, #[host(default = None)] timeout: Option<f64>) -> Result<(), WorkerError>;

    #[host(await)]
    async fn aclose(&self) -> Result<(), WorkerError>;

    /// `"running"`, `"failed"` or `"success"`.
    #[host(call)]
    fn job_status(&self, job_id: String, status: String);

    /// A job in `room` with no dispatch: the console's, a notebook's.
    #[host(await)]
    async fn simulate_job(
        &self,
        room: String,
        #[host(kw)] fake_job: bool,
        #[host(default = None)] agent_identity: Option<String>,
        #[host(default = None)] room_info: Option<Vec<u8>>,
        #[host(default = None)] token: Option<String>,
    ) -> Result<(), WorkerError>;
}
