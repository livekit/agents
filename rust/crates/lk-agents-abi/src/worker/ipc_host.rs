//! The job processes as the host drives them: the parent's supervisor and pool of each process,
//! the child's channel to its parent and the run of its one job, with the host objects they call.
use std::any::Any;
use std::sync::Arc;

use crate::invoke::CallContext;
use crate::worker::ipc::{JobCompletion, JobRunnerOptions, JobStatus};
use crate::worker::job::StartJobFields;
use crate::worker::types::RunningJobInfo;
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

/// The host's side of one job runner (`ipc.JobLauncher`): it starts the runner, finishes the start
/// method's bookkeeping once the runner ended, and writes the supervisor's diagnostics.
#[crate::host(proxy(host = "livekit.agents.ipc.JobLauncher"))]
pub trait JobLauncher: Send + Sync {
    /// Start the runner on the job's end of the channel: a thread of this process when
    /// `launch.thread`, else a process. A process reports its receipt through
    /// `launch.process_started`, a thread its return through `launch.thread_exited`.
    #[host(await, name = "_core_launch")]
    fn launch<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        #[host(context)] context: &'a CallContext,
        endpoint: i64,
        launch: Arc<dyn LaunchReport>,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
    /// The runner ended and its supervision is over.
    #[host(await, name = "_core_completed")]
    fn completed<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        #[host(context)] context: &'a CallContext,
        completion: JobCompletion,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
    /// One supervision diagnostic on the host's logger, with the failure that caused it. `extra`
    /// already names the process (`pid`) and its job (`job_id`, `room`, `room_id`).
    #[host(call, name = "_core_log")]
    fn log(
        &self,
        level: i32,
        message: String,
        extra: Map<String, Value>,
        error: Option<HostError>,
    ) -> Result<(), HostError>;
}

/// A native `JobLaunch`, as the supervisor hands it to its launcher.
#[crate::host(native(refused = "expected a native JobLaunch"))]
pub trait LaunchReport: Any + Send + Sync {}

/// One launch of a job runner, as the launcher reports it.
#[crate::host(class(name = "JobLaunch", native = LaunchReport))]
pub trait JobLaunch: Send + Sync + Sized + 'static {
    /// The runner is a thread of this process, else a process of the launcher's start method.
    #[host(getter)]
    fn thread(&self) -> bool;
    /// The process's receipt, reported on the launch worker right after the OS created it.
    #[host(call)]
    fn process_started(
        &self,
        pid: u32,
        sentinel: i64,
        method: String,
        owned: Vec<i64>,
    ) -> Result<(), WorkerError>;
    /// The job thread returned, reported from that thread.
    #[host(call)]
    fn thread_exited(&self) -> Result<(), WorkerError>;
}

/// A native `JobSupervisor`, as the pool hands it to the host.
#[crate::host(native(refused = "expected a native JobSupervisor"))]
pub trait SupervisedJob: Any + Send + Sync {}

/// One job runner, a process or (`options.thread`) a thread of this process, and its one job:
/// the worker launches it through `launcher`, initializes it, hands it the job, pings it,
/// samples its memory, closes it and waits for its end, whether or not the host still awaits.
///
/// `status` is `None` until a job was launched, then `RUNNING` until the runner ended, then
/// `SUCCESS` (exit code 0 and a clean control channel) or `FAILED`. `running_job` is set once
/// the runner took its job and is kept after the runner ended. `aclose()` asks an initialized
/// runner to shut down and stops it when late (`close_timeout`); `kill()` stops it at once. A
/// thread cannot be stopped: both wait for it to return, and it has no pong deadline nor memory
/// limit.
#[crate::host(class(name = "JobSupervisor", native = SupervisedJob))]
pub trait JobSupervisor: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(
        env: &Self::Env,
        launcher: Arc<dyn JobLauncher>,
        options: JobRunnerOptions,
        #[host(context)] context: CallContext,
    ) -> Result<Self, WorkerError>;
    /// `PCEXEC_...` for a process, `THEXEC_...` for a thread.
    #[host(getter)]
    fn id(&self) -> String;
    #[host(getter)]
    fn thread(&self) -> bool;
    /// `start()` was called.
    #[host(getter)]
    fn started(&self) -> bool;
    /// The runner answered its initialization.
    #[host(getter)]
    fn initialized(&self) -> bool;
    /// `aclose()` or `kill()` was called: it starts no runner and takes no job.
    #[host(getter)]
    fn closing(&self) -> bool;
    /// The runner was stopped by force (a process killed).
    #[host(getter)]
    fn killed(&self) -> bool;
    /// The process's id, once the OS created it; `None` for a thread, and for a forkserver child
    /// whose id the forkserver has not reported.
    #[host(getter)]
    fn pid(&self) -> Option<u32>;
    /// The runner's exit code once its supervision is over (negative: the signal that ended a
    /// Unix process).
    #[host(getter)]
    fn exitcode(&self) -> Option<i32>;
    #[host(getter)]
    fn status(&self) -> Option<JobStatus>;
    #[host(getter)]
    fn running_job(&self) -> Option<RunningJobInfo>;
    /// The seconds the runner took to be ready, once it was.
    #[host(getter)]
    fn initialize_time(&self) -> Option<f64>;
    /// Launch the runner; returns once it exists. Raises once closing, when already started, or
    /// with the launcher's own error when nothing was launched. A launch failing after the OS
    /// created the process is logged with its cause and the process is stopped.
    #[host(await)]
    async fn start(&self) -> Result<(), WorkerError>;
    /// Send the runner its `Hello` and wait for it to be ready: the seconds it took. A late runner
    /// is stopped (`TimeoutError`).
    #[host(await)]
    async fn initialize(&self) -> Result<f64, WorkerError>;
    /// Hand the ready runner its one job, stamped `launched_at` now; the job continues the
    /// caller's current span. Raises when it is not ready or already has a job.
    #[host(await)]
    async fn launch_job(
        &self,
        info: RunningJobInfo,
        #[host(context)] context: CallContext,
        cx: crate::telemetry::Cx,
    ) -> Result<(), WorkerError>;
    /// Close it gracefully (see the class) and wait for its cleanup; at once when not started.
    #[host(await)]
    async fn aclose(&self);
    /// Stop it at once and wait for its cleanup; at once when not started.
    #[host(await)]
    async fn kill(&self);
    /// Wait for the end of its supervision, its completion reported to the launcher: whether a
    /// runner ran. `False` at once when it was never started.
    #[host(await)]
    async fn join(&self) -> bool;
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
    fn ping(&self, cx: &crate::telemetry::Cx) -> crate::BoxFuture<'_, Result<(), HostError>>;
    #[host(await, name = "_core_start_job")]
    fn start_job(
        &self,
        cx: &crate::telemetry::Cx,
        fields: StartJobFields,
    ) -> crate::BoxFuture<'_, Result<Arc<dyn JobExecution>, HostError>>;

    /// The entrypoint's task.
    #[host(await, name = "_core_start_entrypoint")]
    fn start_entrypoint<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        #[host(context)] context: &'a CallContext,
    ) -> crate::BoxFuture<'a, Result<HostRef, HostError>>;
    #[host(await, name = "_core_wait")]
    fn wait<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        #[host(context)] context: &'a CallContext,
        task: HostRef,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
    #[host(await, name = "_core_cancel")]
    fn cancel<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        #[host(context)] context: &'a CallContext,
        task: HostRef,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
    #[host(await, name = "_core_close_session")]
    fn close_session<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        #[host(context)] context: &'a CallContext,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
    #[host(await, name = "_core_session_end")]
    fn session_end<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        #[host(context)] context: &'a CallContext,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
    /// The job's session report, once its session closed; `None` when the job ran no session.
    #[host(await, name = "_core_session_report")]
    fn session_report<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        #[host(context)] context: &'a CallContext,
    ) -> crate::BoxFuture<'a, Result<Option<crate::telemetry::report::SessionReport>, HostError>>;
    #[host(await, name = "_core_disconnect")]
    fn disconnect<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        #[host(context)] context: &'a CallContext,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
    /// A shutdown callback, given the reason when it takes one, in its span (the shutdown's own
    /// for a framework callback), current in `context`.
    #[host(await, name = "_core_shutdown_callback")]
    fn shutdown_callback<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        #[host(context)] context: &'a CallContext,
        callback: HostRef,
        reason: Option<String>,
    ) -> crate::BoxFuture<'a, Result<(), HostError>>;
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
        cx: &crate::telemetry::Cx,
        #[host(context)] context: &'a CallContext,
    ) -> crate::BoxFuture<'a, Result<HostRef, HostError>>;
}

/// A participant entrypoint: `entrypoint(ctx, participant)`.
#[crate::host(proxy(host = "collections.abc.Callable"))]
pub trait ParticipantEntrypoint: Send + Sync {
    #[host(await, name = "__call__")]
    fn run<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
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
    #[host(new)]
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
    /// The launcher of a new process, made on the host's loop.
    #[host(await, name = "_core_create")]
    fn create(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<Arc<dyn JobLauncher>, HostError>>;
    /// The pool's one notification of a process's step: `process_created`, `process_started`,
    /// `process_ready`, `process_job_launched`, `process_closed`, or a cleanup failure.
    #[host(notify, name = "emit")]
    fn emit(&self, event: String, process: Arc<dyn SupervisedJob>);
}

/// The warm job processes: the pool keeps `num_idle_processes` ready, hands each job the next one
/// and retries a failed spawn or launch. Each process is a `JobSupervisor` with `options`.
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
        options: JobRunnerOptions,
        #[host(context)] context: CallContext,
    ) -> Result<Self, WorkerError>;
    /// The processes the pool owns, from their creation until their cleanup.
    #[host(getter)]
    fn processes(&self) -> Vec<Arc<dyn SupervisedJob>>;
    /// The jobs whose process runs them now (`status` `RUNNING`).
    #[host(getter)]
    fn active_jobs(&self) -> Vec<RunningJobInfo>;
    /// The pool's process that took the job `job_id`.
    #[host(call)]
    fn get_by_job_id(&self, job_id: String) -> Option<Arc<dyn SupervisedJob>>;
    #[host(getter)]
    fn jobs_waiting(&self) -> usize;
    #[host(getter)]
    fn target_idle_processes(&self) -> usize;
    #[host(setter, name = "target_idle_processes")]
    fn set_target_idle_processes(&self, n: usize);
    #[host(await)]
    async fn start(&self) -> Result<(), WorkerError>;
    /// The job continues the caller's current span.
    #[host(await)]
    async fn launch_job(
        &self,
        info: RunningJobInfo,
        #[host(context)] context: CallContext,
        cx: crate::telemetry::Cx,
    ) -> Result<(), WorkerError>;
    /// Waits until every process a job was launched on exited (the drain).
    #[host(await)]
    async fn join_jobs(&self);
    #[host(await)]
    async fn aclose(&self);
}

/// A native `JobRun`, as the runner hands it back when its job starts and as the job's primary
/// session holds it.
#[crate::host(native(refused = "expected a native JobRun"))]
pub trait JobExecution: Any + Send + Sync {
    /// The job's primary session closed with `reason`: the job ends.
    fn session_closed(&self, reason: String);
}

/// A job's run: what ends the job (the first reason wins), the entrypoint's supervision, the
/// order and bounds of the shutdown, the participant entrypoints, the room connected once, the
/// application calls in flight and the recording's setup.
#[crate::host(class(name = "JobRun", native = JobExecution))]
pub trait JobRun: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    /// `session_end` names the door's `on_session_end`, when it has one. `observability_url`
    /// overrides where the job's telemetry and recording go (`LIVEKIT_OBSERVABILITY_URL`);
    /// `api_key` and `api_secret` sign its Cloud telemetry.
    #[host(new)]
    #[allow(clippy::too_many_arguments, reason = "Preserve the public Python signature")]
    fn new(
        env: &Self::Env,
        #[host(kw)] fake_job: bool,
        session_end_timeout: f64,
        redaction: bool,
        #[host(default = "")] url: String,
        #[host(default = None)] session_end: Option<String>,
        #[host(default = 60.0)] session_aclose_timeout: f64,
        #[host(default = None)] observability_url: Option<String>,
        #[host(default = "")] api_key: String,
        #[host(default = "")] api_secret: String,
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
    /// Opens the job's root span for the job the parent handed over, before its run starts.
    #[host(call)]
    fn start_trace(&self, fields: StartJobFields);
    /// The application ends the job.
    #[host(call)]
    fn shutdown(&self, reason: String);
    #[host(call)]
    fn room_disconnected(&self);
    #[host(call)]
    fn connected(&self);
    /// Awaits `connect()` once: a call while it runs waits for it, a call after it connected
    /// returns at once, a failed one leaves the next call to try again.
    #[host(await)]
    async fn connect(
        &self,
        connect: Arc<dyn JobCall>,
        #[host(context)] context: CallContext,
        cx: crate::telemetry::Cx,
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
        cx: crate::telemetry::Cx,
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
        cx: crate::telemetry::Cx,
    ) -> Result<HostRef, WorkerError>;
    /// Cancels the API calls still running and waits for them.
    #[host(await)]
    async fn close_tasks(&self);
    /// The session's recording options: the job registers with the exporter's Cloud destination
    /// once, at the first call.
    #[host(call)]
    fn init_recording(&self, options: BTreeMap<String, bool>) -> Result<(), WorkerError>;
    /// The entrypoint, then the shutdown sequence once the job ends; `host` runs each step in the
    /// context this call captured.
    #[host(await)]
    async fn run(
        &self,
        host: Arc<dyn JobProcess>,
        #[host(context)] context: CallContext,
    ) -> Result<(), WorkerError>;
}

/// Native process bootstrap operations, shared by language hosts.
#[crate::host(functions)]
pub trait ProcessBootstrap {
    #[host(function)]
    fn ipc_pair() -> Result<(i64, i64), crate::errors::HostError>;
}

/// Which process the host is initializing. Initialization itself starts no runtime.
#[crate::host(native_enum(name = "Role"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessRole {
    Worker,
    Job,
}

/// Redeclares process ownership after a fork; safe to call repeatedly.
#[crate::host(functions(pid = false))]
pub trait ProcessInitialization {
    #[host(function)]
    fn init(role: ProcessRole);
}
