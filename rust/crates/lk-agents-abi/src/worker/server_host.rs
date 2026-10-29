//! The agent server as the host drives it: its configuration, its runtime, and the host server
//! (`worker._ServerHost`) that owns the job processes and the user's callbacks.

use std::any::Any;
use std::sync::Arc;

use crate::errors::HostError;
use crate::worker::endpoint_host::{AsgiApplication, EndpointService};
use crate::worker::ipc_host::WorkerError;
use crate::worker::worker::{
    JobRequest, ProcessStartOptions, RegisteredInfo, RunningJobInfo, ServerOptions,
};

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
    fn simulation(&self) -> bool;
    #[host(setter, name = "simulation")]
    fn set_simulation(&self, value: bool);
}

/// The host's answer to a job request.
#[derive(Debug, Clone, PartialEq, Eq)]
#[crate::host(record(module = "worker_server", extract))]
pub enum JobAnswer {
    Accept {
        name: String,
        identity: String,
        metadata: String,
        attributes: Vec<(String, String)>,
    },
    Reject,
    /// Reject and ask the server to end the job.
    Terminate,
    /// The callback returned without answering: reject without terminating.
    Unanswered,
}

/// One route of the endpoint manifest the worker registers.
#[derive(Debug, Clone, PartialEq, Eq)]
#[crate::host(record(module = "worker_server", extract))]
pub struct EndpointRoute {
    pub path: String,
    pub methods: Vec<String>,
    /// `"http"`, or `"text"` for a session endpoint.
    pub kind: String,
    pub public: bool,
}

/// The job processes and the user's callbacks, which the server runtime calls on the host's
/// loop. Which jobs to take, when, and whether job processes start at all are the runtime's.
#[crate::host(proxy(host = "livekit.agents.worker._ServerHost"))]
pub trait ServerHost: Send + Sync {
    /// The voice door's `on_request`, asked only when there is one.
    #[host(await, name = "_core_job_request")]
    fn job_request(
        &self,
        request: JobRequest,
    ) -> crate::BoxFuture<'_, Result<JobAnswer, HostError>>;
    /// A job the runtime took, with the timeline it stamped; its LiveKit job id.
    #[host(await, name = "_core_launch_job")]
    fn launch_job(&self, info: RunningJobInfo) -> crate::BoxFuture<'_, Result<String, HostError>>;
    #[host(await, name = "_core_terminate_job")]
    fn terminate_job(&self, job_id: String) -> crate::BoxFuture<'_, Result<(), HostError>>;
    /// The user's `load_fnc`, asked only when there is one.
    #[host(await, name = "_core_load")]
    fn load(&self) -> crate::BoxFuture<'_, Result<f64, HostError>>;
    /// The job processes, when the runtime wants them.
    #[host(await, name = "_core_start")]
    fn start(&self, options: ProcessStartOptions) -> crate::BoxFuture<'_, Result<(), HostError>>;
    #[host(await, name = "_core_started")]
    fn started(&self) -> crate::BoxFuture<'_, Result<(), HostError>>;
    #[host(await, name = "_core_registered")]
    fn registered(&self, info: RegisteredInfo) -> crate::BoxFuture<'_, Result<(), HostError>>;
    #[host(await, name = "_core_idle_processes")]
    fn idle_processes(&self, target: u32) -> crate::BoxFuture<'_, Result<(), HostError>>;
    #[host(await, name = "_core_worker_load")]
    fn worker_load(&self, load: f64) -> crate::BoxFuture<'_, Result<(), HostError>>;
    #[host(await, name = "_core_join_jobs")]
    fn join_jobs(&self) -> crate::BoxFuture<'_, Result<(), HostError>>;
    #[host(await, name = "_core_close")]
    fn close(&self) -> crate::BoxFuture<'_, Result<(), HostError>>;
    /// The jobs running, for the health check.
    #[host(await, name = "_core_health")]
    fn active_jobs(&self) -> crate::BoxFuture<'_, Result<usize, HostError>>;
    /// The Prometheus exposition and its content type.
    #[host(await, name = "_core_metrics")]
    fn metrics(&self) -> crate::BoxFuture<'_, Result<(String, String), HostError>>;
}

#[crate::host(class(name = "ServerRuntime"))]
pub trait ServerRuntime: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(
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
    /// return once closed. `endpoints` serves `routes` with `app`.
    #[host(await)]
    #[allow(clippy::too_many_arguments)]
    async fn run(
        &self,
        #[host(kw)] devmode: bool,
        unregistered: bool,
        has_rtc: bool,
        has_text_doors: bool,
        publisher: bool,
        routes: Vec<EndpointRoute>,
        has_load_fnc: bool,
        has_on_request: bool,
        endpoints: Arc<dyn EndpointService>,
        app: Arc<dyn AsgiApplication>,
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
