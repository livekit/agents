//! Native server configuration and the host-loop bootstrap adapter.

use std::any::Any;

use crate::HostRef;
use crate::errors::HostError;
use crate::worker::worker::ServerOptions;

/// Preserve the native configuration identity shared by a server and its options surface.
#[crate::host(native(refused = "expected a native ServerConfig"))]
pub trait ServerConfiguration: Any + Send + Sync {}

#[crate::host(class(name = "ServerConfig", native = ServerConfiguration))]
pub trait ServerConfig: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(options: ServerOptions) -> Result<Self, HostError>;

    #[host(call)]
    fn update(&self, options: ServerOptions) -> Result<(), HostError>;

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
    #[host(getter)]
    fn cloud(&self) -> bool;
    #[host(getter)]
    fn drain_timeout(&self) -> f64;
    #[host(getter)]
    fn simulation(&self) -> bool;
    #[host(setter, name = "simulation")]
    fn set_simulation(&self, value: bool);
}

/// Returned host references are futures owned by the bootstrap loop.
#[crate::host(class(name = "ServerRuntime"))]
pub trait ServerRuntime: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(config: HostRef, host: HostRef) -> Result<Self, HostError>;

    #[host(getter)]
    fn worker_id(&self) -> String;
    #[host(getter)]
    fn draining(&self) -> bool;
    #[host(getter)]
    fn http_port(&self) -> u16;

    #[host(call, future = ())]
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        #[host(kw)] devmode: bool,
        unregistered: bool,
        has_rtc: bool,
        has_text_doors: bool,
        publisher: bool,
        routes: HostRef,
        has_load_fnc: bool,
        has_on_request: bool,
    ) -> Result<HostRef, HostError>;

    #[host(call, future = ())]
    fn drain(&self, #[host(default = None)] timeout: Option<f64>) -> Result<HostRef, HostError>;

    #[host(call, future = ())]
    fn aclose(&self) -> Result<HostRef, HostError>;

    #[host(call)]
    fn job_status(&self, job_id: String, status: String);

    #[host(call, future = ())]
    fn simulate_job(
        &self,
        room: String,
        #[host(kw)] fake_job: bool,
        #[host(default = None)] agent_identity: Option<String>,
        #[host(default = None)] room_info: Option<Vec<u8>>,
        #[host(default = None)] token: Option<String>,
    ) -> Result<HostRef, HostError>;
}
