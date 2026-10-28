//! Native process bootstrap adapters used by the Python host.
use crate::worker::job::StartJobFields;
use crate::{HostRef, errors::HostError};
use std::collections::BTreeMap;

#[crate::host(class(name = "ProcSupervisor"))]
pub trait WorkerProcSupervisor: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(
        fd: i64,
        host: HostRef,
        #[host(kw)] pid: u32,
        initialize_timeout: f64,
        close_timeout: f64,
        memory_warn_mb: f64,
        memory_limit_mb: f64,
        ping_interval: f64,
        ping_timeout: f64,
        high_ping_threshold: f64,
        #[host(default = true)] killable: bool,
    ) -> Result<Self, HostError>;
    #[host(call, future = f64)]
    fn initialize(
        &self,
        #[host(kw)] asyncio_debug: bool,
        http_proxy: Option<String>,
    ) -> Result<HostRef, HostError>;
    #[host(getter)]
    fn initialized(&self) -> bool;
    #[host(call)]
    fn _core_launched_process(
        &self,
        pid: u32,
        sentinel: i64,
        method: String,
        owned: Vec<i64>,
    ) -> Result<(), HostError>;
    #[host(call, future = bool)]
    fn run(&self, joined: HostRef, host: HostRef) -> Result<HostRef, HostError>;
    #[host(call, future = bool)]
    fn _core_wait_exit(&self) -> Result<HostRef, HostError>;
    #[host(call, future = bool)]
    fn _core_join(&self) -> Result<HostRef, HostError>;
    #[host(call, future = ())]
    fn aclose(&self) -> Result<HostRef, HostError>;
    #[host(call, future = ())]
    fn kill(&self) -> Result<HostRef, HostError>;
    #[host(call, future = ())]
    fn start_job(&self, fields: StartJobFields) -> Result<HostRef, HostError>;
    #[host(getter)]
    fn killed(&self) -> bool;
}

#[crate::host(class(name = "IpcClient"))]
pub trait WorkerIpcClient: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(
        fd: i64,
        initialize: HostRef,
        #[host(kw, default = false)] process: bool,
    ) -> Result<Self, HostError>;
    #[host(getter)]
    fn asyncio_debug(&self) -> bool;
    #[host(call, future = ())]
    fn serve(&self, host: HostRef) -> Result<HostRef, HostError>;
    #[host(call)]
    fn stop(&self);
    #[host(call, future = ())]
    fn aclose(&self) -> Result<HostRef, HostError>;
}

#[crate::host(class(name = "ProcPool"))]
pub trait WorkerProcPool: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(
        host: HostRef,
        #[host(kw)] num_idle_processes: usize,
        initialize_timeout: f64,
        close_timeout: f64,
    ) -> Result<Self, HostError>;
    #[host(getter)]
    fn processes(&self) -> Vec<HostRef>;
    #[host(getter)]
    fn jobs_waiting(&self) -> usize;
    #[host(getter)]
    fn target_idle_processes(&self) -> usize;
    #[host(setter, name = "target_idle_processes")]
    fn set_target_idle_processes(&self, n: usize);
    #[host(call, future = ())]
    fn start(&self) -> Result<HostRef, HostError>;
    #[host(call, future = ())]
    fn launch_job(&self, info: HostRef) -> Result<HostRef, HostError>;
    #[host(call, future = ())]
    fn join_jobs(&self) -> Result<HostRef, HostError>;
    #[host(call, future = ())]
    fn aclose(&self) -> Result<HostRef, HostError>;
}

#[crate::host(class(name = "JobRun"))]
pub trait WorkerJobRun: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(
        #[host(kw)] fake_job: bool,
        session_end_timeout: f64,
        redaction: bool,
        #[host(default = "")] url: String,
        #[host(default = false)] recording: bool,
        #[host(default = None)] session_end: Option<HostRef>,
        #[host(default = 60.0)] session_aclose_timeout: f64,
    ) -> Result<Self, HostError>;
    #[host(call)]
    fn add_shutdown_callback(&self, callback: HostRef) -> Result<(), HostError>;
    #[host(getter)]
    fn observability_url(&self) -> Option<String>;
    #[host(getter)]
    fn buffer_logs(&self) -> bool;
    #[host(call)]
    fn shutdown(&self, reason: String, user_initiated: bool);
    #[host(call)]
    fn connected(&self);
    #[host(call, future = ())]
    fn connect(&self, connect: HostRef) -> Result<HostRef, HostError>;
    #[host(call)]
    fn add_participant_entrypoint(
        &self,
        entrypoint: HostRef,
        kinds: Vec<i32>,
    ) -> Result<(), HostError>;
    #[host(call)]
    fn participant_joined(
        &self,
        ctx: HostRef,
        participant: HostRef,
        identity: String,
        kind: i32,
    ) -> Result<(), HostError>;
    #[host(call)]
    fn participant_left(&self, attributes: BTreeMap<String, String>);
    #[host(call, future = HostRef)]
    fn api_call(
        &self,
        call: HostRef,
        name: String,
        console_result: HostRef,
    ) -> Result<HostRef, HostError>;
    #[host(call, future = ())]
    fn close_tasks(&self) -> Result<HostRef, HostError>;
    #[host(call)]
    fn init_recording(&self, options: BTreeMap<String, bool>) -> Result<Option<bool>, HostError>;
    #[host(call)]
    fn claim_session(&self, session: HostRef) -> Result<bool, HostError>;
    #[host(getter)]
    fn primary_session(&self) -> Option<HostRef>;
    #[host(getter)]
    fn recording_initialized(&self) -> bool;
    #[host(getter)]
    fn redaction(&self) -> bool;
    #[host(call, future = ())]
    fn run(&self, host: HostRef) -> Result<HostRef, HostError>;
}
