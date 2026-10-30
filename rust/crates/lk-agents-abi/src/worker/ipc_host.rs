//! The job processes as the host drives them: the parent's supervisor and pool of each process,
//! the child's channel to its parent and the run of its one job, with the host objects they call.
use std::any::Any;
use std::sync::Arc;

use crate::invoke::CallContext;
use crate::worker::ipc::ProcessCompletion;
use crate::worker::job::StartJobFields;
use crate::{HostRef, errors::HostError};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// A worker operation's failure, raised as the host exception its kind names.
#[derive(Debug, Clone, thiserror::Error)]
#[crate::host(error(kind = "RuntimeError"))]
pub enum WorkerError {
    #[error("{0}")]
    Failed(String),
    #[error("{0}")]
    #[host(kind = "ValueError")]
    Invalid(String),
    #[error("{0}")]
    #[host(kind = "TimeoutError")]
    TimedOut(String),
    #[error("the IPC socket is closed")]
    #[host(kind = "ConnectionError")]
    Closed,
    #[error("the job ended")]
    #[host(kind = "asyncio.CancelledError")]
    Cancelled,
    #[error(transparent)]
    #[host(delegate)]
    Host(#[from] HostError),
}

/// The executor of one job process (`ipc.JobExecutor`): it spawns the process, owns its
/// start-method bookkeeping and receives the supervisor's diagnostics.
#[crate::host(proxy(host = "livekit.agents.ipc.JobExecutor"))]
pub trait JobExecutor: Any + Send + Sync {
    /// Spawn the process on its socket; it reports its receipt through `_core_launched_process`.
    #[host(await, name = "_core_launch")]
    fn launch<'a>(
        &'a self,
        #[host(context)] context: &'a CallContext,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
    #[host(await, name = "_core_launched")]
    fn launched<'a>(
        &'a self,
        #[host(context)] context: &'a CallContext,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
    #[host(await, name = "_core_launch_failed")]
    fn launch_failed<'a>(
        &'a self,
        #[host(context)] context: &'a CallContext,
        cause: String,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
    /// The process exited and was reaped.
    #[host(await, name = "_core_completed")]
    fn completed<'a>(
        &'a self,
        #[host(context)] context: &'a CallContext,
        completion: ProcessCompletion,
        pid: u32,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
    /// One supervision diagnostic on the host's logger, with the failure that caused it.
    #[host(call, name = "_core_log")]
    fn log(
        &self,
        level: i32,
        message: String,
        extra: Map<String, Value>,
        error: Option<HostError>,
    ) -> Result<(), HostError>;
    /// The pool's steps: start the process (its supervisor), initialize it, hand it a job
    /// (`launched_at` is when it took it), close it and join it.
    #[host(await, name = "_core_start")]
    fn start<'a>(
        &'a self,
        #[host(context)] context: &'a CallContext,
    ) -> crate::BoxFuture<'a, Result<Arc<dyn SupervisedProcess>, HostError>>;
    #[host(await, name = "initialize")]
    fn initialize<'a>(
        &'a self,
        #[host(context)] context: &'a CallContext,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
    #[host(await, name = "_core_launch_job")]
    fn launch_job<'a>(
        &'a self,
        #[host(context)] context: &'a CallContext,
        info: HostRef,
        launched_at: f64,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
    #[host(await, name = "aclose")]
    fn close<'a>(
        &'a self,
        #[host(context)] context: &'a CallContext,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
    #[host(await, name = "_core_join")]
    fn join<'a>(
        &'a self,
        #[host(context)] context: &'a CallContext,
    ) -> crate::BoxFuture<'a, Result<bool, HostError>>;
}

/// A native `ProcSupervisor`, as its executor hands it to the pool.
#[crate::host(native(refused = "expected a native ProcSupervisor"))]
pub trait SupervisedProcess: Any + Send + Sync {}

#[crate::host(class(name = "ProcSupervisor", native = SupervisedProcess))]
pub trait ProcessSupervisor: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    #[allow(clippy::too_many_arguments)]
    fn new(
        env: &Self::Env,
        fd: i64,
        host: Arc<dyn JobExecutor>,
        #[host(kw)] pid: u32,
        initialize_timeout: f64,
        close_timeout: f64,
        memory_warn_mb: f64,
        memory_limit_mb: f64,
        ping_interval: f64,
        ping_timeout: f64,
        high_ping_threshold: f64,
        #[host(default = true)] killable: bool,
        #[host(context)] context: CallContext,
    ) -> Result<Self, WorkerError>;
    /// The seconds the process took to be ready.
    #[host(await)]
    async fn initialize(
        &self,
        #[host(kw)] asyncio_debug: bool,
        http_proxy: Option<String>,
    ) -> Result<f64, WorkerError>;
    #[host(getter)]
    fn initialized(&self) -> bool;
    /// The launch's receipt, reported on the launch worker right after the OS created the child.
    #[host(call)]
    fn _core_launched_process(
        &self,
        pid: u32,
        sentinel: i64,
        method: String,
        owned: Vec<i64>,
    ) -> Result<(), WorkerError>;
    /// Launch, supervise, observe the exit and reap; whether the process ran (an OS child was
    /// reaped). Cancelling the wait leaves the process owned.
    #[host(await)]
    async fn run(&self) -> bool;
    #[host(await)]
    async fn _core_wait_exit(&self) -> bool;
    #[host(await)]
    async fn _core_join(&self) -> bool;
    #[host(await)]
    async fn aclose(&self);
    #[host(await)]
    async fn kill(&self);
    #[host(await)]
    async fn start_job(&self, fields: StartJobFields) -> Result<(), WorkerError>;
    #[host(getter)]
    fn killed(&self) -> bool;
}

/// The process's setup, called with the parent's HTTP proxy before readiness.
#[crate::host(proxy(host = "collections.abc.Callable"))]
pub trait ChildInitialize: Send + Sync {
    #[host(call, name = "__call__")]
    fn initialize(&self, http_proxy: Option<String>) -> Result<(), HostError>;
}

/// The job process's runner (`ipc.child._JobProc`): it starts the job the parent hands over and
/// runs each step of the job's run, in the context the run captured.
#[crate::host(proxy(host = "livekit.agents.ipc.child._JobProc"))]
pub trait JobProcess: Send + Sync {
    /// Answers once its loop ran it: a pong measures the loop's responsiveness.
    #[host(await, name = "_core_ping")]
    fn ping(&self) -> crate::BoxFuture<'_, Result<(), HostError>>;
    #[host(await, name = "_core_start_job")]
    fn start_job(
        &self,
        fields: StartJobFields,
    ) -> crate::BoxFuture<'_, Result<Arc<dyn JobExecution>, HostError>>;

    /// The entrypoint's task.
    #[host(await, name = "_core_start_entrypoint")]
    fn start_entrypoint<'a>(
        &'a self,
        #[host(context)] context: &'a CallContext,
    ) -> crate::BoxFuture<'a, Result<HostRef, HostError>>;
    #[host(await, name = "_core_wait")]
    fn wait<'a>(
        &'a self,
        #[host(context)] context: &'a CallContext,
        task: HostRef,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
    #[host(await, name = "_core_cancel")]
    fn cancel<'a>(
        &'a self,
        #[host(context)] context: &'a CallContext,
        task: HostRef,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
    #[host(await, name = "_core_shutdown_began")]
    fn shutdown_began<'a>(
        &'a self,
        #[host(context)] context: &'a CallContext,
        reason: String,
        user_initiated: bool,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
    #[host(await, name = "_core_close_session")]
    fn close_session<'a>(
        &'a self,
        #[host(context)] context: &'a CallContext,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
    #[host(await, name = "_core_session_end")]
    fn session_end<'a>(
        &'a self,
        #[host(context)] context: &'a CallContext,
        span: HostRef,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
    #[host(await, name = "_core_upload")]
    fn upload<'a>(
        &'a self,
        #[host(context)] context: &'a CallContext,
        span: HostRef,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
    #[host(await, name = "_core_disconnect")]
    fn disconnect<'a>(
        &'a self,
        #[host(context)] context: &'a CallContext,
        span: HostRef,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
    /// A shutdown callback, given the reason when it takes one, in its span (the shutdown's own
    /// for a framework callback).
    #[host(await, name = "_core_shutdown_callback")]
    fn shutdown_callback<'a>(
        &'a self,
        #[host(context)] context: &'a CallContext,
        callback: HostRef,
        reason: Option<String>,
        span: Option<HostRef>,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
    /// A span of the shutdown's, labelled with the callback it runs.
    #[host(call, name = "_core_span")]
    fn span(&self, name: String, callback: Option<String>) -> Result<HostRef, HostError>;
    #[host(call, name = "_core_span_end")]
    fn span_end(
        &self,
        span: HostRef,
        error: Option<HostError>,
        timed_out: bool,
    ) -> Result<(), HostError>;
    #[host(call, name = "_core_log")]
    fn log(
        &self,
        level: i32,
        message: String,
        extra: Map<String, Value>,
        error: Option<HostError>,
    ) -> Result<(), HostError>;
}

/// An awaited application call of the job's (`connect`, `delete_room`, ...), run in its caller's
/// context.
#[crate::host(proxy(host = "collections.abc.Callable"))]
pub trait JobCall: Send + Sync {
    #[host(await, name = "__call__")]
    fn call<'a>(
        &'a self,
        #[host(context)] context: &'a CallContext,
    ) -> crate::BoxFuture<'a, Result<HostRef, HostError>>;
}

/// A participant entrypoint: `entrypoint(ctx, participant)`.
#[crate::host(proxy(host = "collections.abc.Callable"))]
pub trait ParticipantEntrypoint: Send + Sync {
    #[host(await, name = "__call__")]
    fn run<'a>(
        &'a self,
        #[host(context)] context: &'a CallContext,
        job: HostRef,
        participant: HostRef,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
}

#[crate::host(class(name = "IpcClient"))]
pub trait IpcClient: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    /// The handshake on the parent's socket, blocking before the process's loop exists.
    #[host(new, detach)]
    fn new(
        env: &Self::Env,
        fd: i64,
        initialize: Arc<dyn ChildInitialize>,
        #[host(kw, default = false)] process: bool,
    ) -> Result<Self, WorkerError>;
    #[host(getter)]
    fn asyncio_debug(&self) -> bool;
    /// Serve the parent's messages until it goes, its shutdown ends the process, or `stop()`.
    #[host(await)]
    async fn serve(&self, host: Arc<dyn JobProcess>) -> Result<(), WorkerError>;
    #[host(call)]
    fn stop(&self);
    #[host(await)]
    async fn aclose(&self);
}

#[crate::host(proxy(host = "livekit.agents.ipc.ProcPool"))]
pub trait PoolCallbacks: Send + Sync {
    /// A new executor, made on the host's loop.
    #[host(await, name = "_core_create")]
    fn create(&self) -> crate::BoxFuture<'_, Result<Arc<dyn JobExecutor>, HostError>>;
    #[host(notify, name = "emit")]
    fn emit(&self, event: String, process: Arc<dyn JobExecutor>);
}

#[crate::host(class(name = "ProcPool"))]
pub trait ProcessPool: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(
        env: &Self::Env,
        host: Arc<dyn PoolCallbacks>,
        #[host(kw)] num_idle_processes: usize,
        initialize_timeout: f64,
        close_timeout: f64,
        #[host(context)] context: CallContext,
    ) -> Result<Self, WorkerError>;
    #[host(getter)]
    fn processes(&self) -> Vec<Arc<dyn JobExecutor>>;
    #[host(getter)]
    fn jobs_waiting(&self) -> usize;
    #[host(getter)]
    fn target_idle_processes(&self) -> usize;
    #[host(setter, name = "target_idle_processes")]
    fn set_target_idle_processes(&self, n: usize);
    #[host(await)]
    async fn start(&self) -> Result<(), WorkerError>;
    #[host(await)]
    async fn launch_job(&self, job_id: String, info: HostRef) -> Result<(), WorkerError>;
    /// Waits until every process a job was launched on exited (the drain).
    #[host(await)]
    async fn join_jobs(&self);
    #[host(await)]
    async fn aclose(&self);
}

/// A native `JobRun`, as the runner hands it back when its job starts.
#[crate::host(native(refused = "expected a native JobRun"))]
pub trait JobExecution: Any + Send + Sync {}

/// A job's run: what ends the job (the first reason wins), the entrypoint's supervision, the
/// order and bounds of the shutdown, the participant entrypoints, the room connected once, the
/// application calls in flight and the recording's setup.
#[crate::host(class(name = "JobRun", native = JobExecution))]
pub trait JobRun: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    /// `session_end` names the door's `on_session_end`, when it has one. `observability_url`
    /// overrides where the job's telemetry and recording go (`LIVEKIT_OBSERVABILITY_URL`).
    #[host(new)]
    fn new(
        env: &Self::Env,
        #[host(kw)] fake_job: bool,
        session_end_timeout: f64,
        redaction: bool,
        #[host(default = "")] url: String,
        #[host(default = false)] recording: bool,
        #[host(default = None)] session_end: Option<String>,
        #[host(default = 60.0)] session_aclose_timeout: f64,
        #[host(default = None)] observability_url: Option<String>,
    ) -> Result<Self, WorkerError>;
    /// `callback(reason)` when it takes the reason, else `callback()`, as the job shuts down; a
    /// framework callback runs in the shutdown's span, any other in its own span named `name`.
    #[host(call)]
    fn add_shutdown_callback(
        &self,
        callback: HostRef,
        takes_reason: bool,
        framework: bool,
        name: String,
    );
    #[host(getter)]
    fn observability_url(&self) -> Option<String>;
    #[host(getter)]
    fn buffer_logs(&self) -> bool;
    #[host(call)]
    fn shutdown(&self, reason: String, user_initiated: bool);
    #[host(call)]
    fn connected(&self);
    /// Awaits `connect()` once: a call while it runs waits for it, a call after it connected
    /// returns at once, a failed one leaves the next call to try again.
    #[host(await)]
    async fn connect(
        &self,
        connect: Arc<dyn JobCall>,
        #[host(context)] context: CallContext,
    ) -> Result<(), WorkerError>;
    /// `entrypoint` (named `name`) runs for each joining participant of one of `kinds`.
    #[host(call)]
    fn add_participant_entrypoint(
        &self,
        entrypoint: Arc<dyn ParticipantEntrypoint>,
        kinds: Vec<i32>,
        name: String,
    );
    /// A participant joined, or was in the room when it connected; `job` is the `JobContext`.
    #[host(call)]
    fn participant_joined(
        &self,
        job: HostRef,
        participant: HostRef,
        identity: String,
        kind: i32,
        #[host(context)] context: CallContext,
    );
    #[host(call)]
    fn participant_left(&self, attributes: BTreeMap<String, String>);
    /// An application API call (`delete_room`, ...): its failure is logged whether or not the
    /// caller awaits it, and it is cancelled once the job ends. A console job makes no call: it
    /// resolves to `console_result`.
    #[host(await)]
    async fn api_call(
        &self,
        call: Arc<dyn JobCall>,
        name: String,
        console_result: HostRef,
        #[host(context)] context: CallContext,
    ) -> Result<HostRef, WorkerError>;
    /// Cancels the API calls still running and waits for them.
    #[host(await)]
    async fn close_tasks(&self);
    /// The redaction to record with when the cloud upload is to be set up now, `None` when not.
    #[host(call)]
    fn init_recording(&self, options: BTreeMap<String, bool>) -> Result<Option<bool>, WorkerError>;
    #[host(getter)]
    fn recording_initialized(&self) -> bool;
    #[host(getter)]
    fn redaction(&self) -> bool;
    /// The entrypoint, then the shutdown sequence once the job ends; `host` runs each step in the
    /// context this call captured.
    #[host(await)]
    async fn run(
        &self,
        host: Arc<dyn JobProcess>,
        #[host(context)] context: CallContext,
    ) -> Result<(), WorkerError>;
}
