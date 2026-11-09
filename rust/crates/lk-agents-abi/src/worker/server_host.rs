use std::any::Any;
use std::sync::Arc;

use crate::a2a::door::TextEndpoint;
use crate::errors::HostError;
use crate::invoke::CallContext;
use crate::worker::http::{EndpointHandler, EndpointKind};
use crate::worker::ipc_host::{JobLaunchers, WorkerError};
use crate::worker::job::RunningJobInfo;
use crate::worker::types::{ProcessOptions, ProcessStartOptions, ServerOptions};

/// Preserve the native configuration identity shared by a server and its options surface.
#[crate::host(native(refused = "expected a native ServerConfig"))]
pub trait ServerConfiguration: Any + Send + Sync {}

#[crate::host(class(name = "ServerConfig", native = ServerConfiguration))]
pub trait ServerConfig: Send + Sync + Sized + 'static {
    /// Options left empty read the environment (`http_proxy`: `HTTPS_PROXY`, else
    /// `HTTP_PROXY`). `version`: the SDK version the worker registers and `/worker` reports.
    #[host(new)]
    fn new(
        options: ServerOptions,
        process: ProcessOptions,
        version: String,
    ) -> Result<Self, WorkerError>;

    /// Before the server starts only.
    #[host(call)]
    fn update(&self, options: ServerOptions, process: ProcessOptions) -> Result<(), WorkerError>;

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
    /// An A2A card's default version: the deployment the worker registered under, else
    /// `LIVEKIT_AGENT_VERSION` for a local worker, else `"1.0.0"`.
    #[host(getter)]
    fn card_version(&self) -> String;
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

/// What the server runtime asks of the host, on the host's loop: the job processes' launcher and
/// the user's callbacks. The runtime owns the process pool and every job decision.
#[crate::host(proxy(host = "livekit.agents.worker._ServerCallbacks"))]
pub trait ServerHost: Send + Sync {
    /// The user's `load_fnc`, asked only when there is one.
    #[host(await, name = "_core_load")]
    fn load(&self, cx: &crate::telemetry::Cx) -> crate::BoxFuture<'_, Result<f64, HostError>>;
    /// Prepare the job processes (the forkserver's preloads) and return the factory of their
    /// launchers; `None` when the server runs no jobs.
    #[host(await, name = "_core_start")]
    fn start(
        &self,
        cx: &crate::telemetry::Cx,
        options: ProcessStartOptions,
    ) -> crate::BoxFuture<'_, Result<Option<Arc<dyn JobLaunchers>>, HostError>>;
    #[host(await, name = "_core_started")]
    fn started(&self, cx: &crate::telemetry::Cx) -> crate::BoxFuture<'_, Result<(), HostError>>;
    /// Answered by a responsive host on every load interval: `/` reads a silent one as stuck.
    #[host(await, name = "_core_heartbeat")]
    fn heartbeat(&self, cx: &crate::telemetry::Cx) -> crate::BoxFuture<'_, Result<(), HostError>>;
}

/// The server's jobs, as the dev channel reads and resumes them across a reload.
#[crate::host(native(refused = "expected a native ServerRuntime"))]
pub trait ServerJobs: Any + Send + Sync {
    /// The jobs running now.
    fn running_jobs(&self) -> Vec<RunningJobInfo>;
    /// Launch jobs kept from before a reload, once the job processes started.
    fn reload_jobs(
        &self,
        jobs: Vec<RunningJobInfo>,
    ) -> crate::BoxFuture<'_, Result<(), WorkerError>>;
}

#[crate::host(class(name = "ServerRuntime", native = ServerJobs))]
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
    /// The jobs running now.
    #[host(getter)]
    fn active_jobs(&self) -> Vec<RunningJobInfo>;

    /// Start the job processes and the health listener and, unless `unregistered`, register;
    /// return once closed. The worker routes `/<name>/...` of each of `text_doors` to that door
    /// (its card and JSON-RPC methods; private, before `routes`), and serves each `text` route
    /// with its door, whose handlers run in the context this call captured; `endpoints`
    /// receives every other stream for `routes`. `asyncio_debug`: the job runners' loops run in
    /// debug mode.
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
        #[host(kw, default = false)] asyncio_debug: bool,
        #[host(context)] context: CallContext,
    ) -> Result<(), WorkerError>;

    /// `None` (or `0`) waits without a deadline; past it raises `TimeoutError`.
    #[host(await)]
    async fn drain(&self, #[host(default = None)] timeout: Option<f64>) -> Result<(), WorkerError>;

    #[host(await)]
    async fn aclose(&self) -> Result<(), WorkerError>;

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

/// What the worker's `/` and `/ready` report, the body of a failed check.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthState {
    /// Not registered yet, or the job process pool has not filled for the first time.
    Starting,
    #[host(rename = "OK")]
    Ready,
    /// The connection dropped; the worker is registering again.
    Reconnecting,
    Draining,
    /// Reconnection gave up, or the pool cannot start.
    Failed,
    /// The host has not answered for the stuck deadline.
    Stuck,
}

impl HealthState {
    /// `/`: only what a restart fixes fails it, never a drain or a dropped connection.
    pub fn live(self) -> bool {
        matches!(self, HealthState::Ready | HealthState::Reconnecting | HealthState::Draining)
    }

    /// `/ready`: whether to receive traffic.
    pub fn ready(self) -> bool {
        self == HealthState::Ready
    }
}
