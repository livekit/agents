//! The parent's side of one job process as its executor drives it (`_core.ProcSupervisor`): the
//! launch the host performs, then initialization, supervision, exit observation and reaping,
//! which continue whether or not the host still awaits them.

use lk_agents_net::task::{TaskSet, TokioSpawner};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::errors::HostError;
use lk_agents_abi::invoke::CallContext;
use lk_agents_abi::worker::ipc_host::{
    JobExecutor, ProcessSupervisor, SupervisedProcess, WorkerError,
};
use lk_agents_abi::worker::job::StartJobFields;
use tokio::sync::watch;

use super::process::{ProcessControl, ProcessLaunch, ProcessResource, StartMethod};
use super::supervisor::{CleanupObservation, SignalFailure};
use super::{
    Channel, Extra, ExtraValue, HEALTH_CHECK_GRACE, Hello, InitError, Level, MemoryError,
    ProcSupervisor, ProcessCallbacks, SupervisorOptions, extra_fields, secs,
};
use lk_agents_net::blocking_io::BlockingIo;
use lk_agents_net::env::ProcessEnv;

pub struct ProcessOwner {
    launch: watch::Sender<ProcessLaunch>,
    pub(crate) sup: ProcSupervisor,
    /// Taken by `run()`: the executor holds this object, so keeping it would be a cycle the
    /// host's collector cannot see.
    host: Mutex<Option<Arc<dyn JobExecutor>>>,
    /// What the executor's steps run in: the constructor's caller.
    context: CallContext,
    io: BlockingIo,
    /// The child's health timeout, from the supervision's ping settings.
    health_timeout_ms: Option<u64>,
    /// Whether the process initialized, once that settled (answered, failed, timed out, killed).
    initialized: watch::Sender<Option<bool>>,
    /// Completed launch: false without an OS child, true only with its reaping receipt.
    ran: watch::Sender<Option<bool>>,
    killable: bool,
    process: watch::Sender<Option<Arc<ProcessControl>>>,
    /// The launch and the ownership loop `run()` starts; each ends with the process, and
    /// dropping this object does not cancel them.
    tasks: Arc<TaskSet>,
}

impl SupervisedProcess for ProcessOwner {}

impl ProcessOwner {
    fn settle(&self, initialized: bool) {
        self.initialized.send_if_modified(|v| v.is_none() && v.replace(initialized).is_none());
    }

    async fn wait_ran(&self) -> bool {
        let mut ran = self.ran.subscribe();
        ran.wait_for(Option::is_some).await.map(|done| done.unwrap_or(false)).unwrap_or(false)
    }
}

impl ProcessSupervisor for ProcessOwner {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        fd: i64,
        host: Arc<dyn JobExecutor>,
        pid: u32,
        initialize_timeout: f64,
        close_timeout: f64,
        memory_warn_mb: f64,
        memory_limit_mb: f64,
        ping_interval: f64,
        ping_timeout: f64,
        high_ping_threshold: f64,
        killable: bool,
        context: CallContext,
    ) -> Result<Self, WorkerError> {
        // Thread runners and direct supervisors need diagnostics before any OS launch.
        crate::diagnostics::initialize().map_err(WorkerError::Failed)?;
        let io = (env.blocking_io)().map_err(WorkerError::Failed)?;
        let opts = SupervisorOptions {
            pid,
            initialize_timeout,
            close_timeout,
            memory_warn_mb,
            memory_limit_mb,
            ping_interval,
            ping_timeout,
            high_ping_threshold,
            killable,
        };
        // The child calls itself orphaned when the parent goes quiet past its pong deadline
        // (+10 s); a job thread's parent has none.
        let health_timeout_ms = (ping_interval > 0.0 && ping_timeout > 0.0)
            .then(|| secs(ping_timeout + HEALTH_CHECK_GRACE).as_millis() as u64);
        let (process, native) = watch::channel(None);
        let sup = {
            let _runtime = io.runtime_handle().enter();
            let channel = Channel::from_endpoint(fd)
                .map_err(|error| WorkerError::Invalid(error.to_string()))?;
            let diagnostics =
                ProcessDiagnostics { host: Mutex::new(Some(host.clone())), process: native };
            ProcSupervisor::new(channel, diagnostics, opts, io.clone())
        };
        Ok(Self {
            launch: watch::channel(ProcessLaunch::Pending).0,
            sup,
            host: Mutex::new(Some(host)),
            context,
            io,
            health_timeout_ms,
            initialized: watch::channel(None).0,
            ran: watch::channel(None).0,
            killable,
            process,
            tasks: Arc::default(),
        })
    }

    async fn initialize(
        &self,
        asyncio_debug: bool,
        http_proxy: Option<String>,
    ) -> Result<f64, WorkerError> {
        let hello = Hello {
            asyncio_debug,
            http_proxy: http_proxy.filter(|p| !p.is_empty()),
            health_timeout_ms: self.health_timeout_ms,
        };
        let sup = self.sup.clone();
        let outcome = self.io.run(async move { sup.initialize(hello).await }).await;
        self.initialized.send_if_modified(|v| {
            v.is_none() && v.replace(matches!(outcome, Ok(Some(_)))).is_none()
        });
        let runner = if self.killable { "process" } else { "runner" };
        match outcome {
            Ok(Some(elapsed)) => Ok(elapsed.as_secs_f64()),
            Ok(None) => Err(WorkerError::TimedOut(format!("{runner} initialization timed out"))),
            Err(InitError::Closed) => Err(WorkerError::Closed),
            Err(error) => Err(WorkerError::Failed(error.to_string())),
        }
    }

    fn initialized(&self) -> bool {
        *self.initialized.borrow() == Some(true)
    }

    fn _core_launched_process(
        &self,
        pid: u32,
        sentinel: i64,
        method: String,
        owned: Vec<i64>,
    ) -> Result<(), WorkerError> {
        if !matches!(*self.launch.borrow(), ProcessLaunch::Pending) {
            return Err(WorkerError::Failed("launch already reported".into()));
        }
        let failed = |error: std::io::Error| WorkerError::Failed(error.to_string());
        let method = StartMethod::parse(&method).map_err(failed)?;
        let resource = ProcessResource::new(pid, sentinel, method, owned).map_err(failed)?;
        self.launch.send_replace(ProcessLaunch::Started(resource));
        Ok(())
    }

    async fn run(&self) -> bool {
        let Some(host) = self.host.lock().take() else {
            return self.wait_ran().await;
        };
        let spawner = TokioSpawner::new(self.io.runtime_handle());
        self.tasks.spawn(
            &spawner,
            "process launch",
            launch(host.clone(), self.context.clone(), self.sup.clone(), self.launch.clone()),
        );
        self.tasks.spawn(
            &spawner,
            "process ownership",
            own(
                host,
                self.context.clone(),
                self.sup.clone(),
                self.launch.clone(),
                self.process.clone(),
                self.initialized.subscribe(),
                self.ran.clone(),
            ),
        );
        self.wait_ran().await
    }

    async fn _core_wait_exit(&self) -> bool {
        let sup = self.sup.clone();
        self.io.run(async move { sup.wait_exit().await }).await
    }

    async fn _core_join(&self) -> bool {
        self.wait_ran().await
    }

    async fn aclose(&self) {
        let initialized = self.initialized();
        self.settle(false);
        let sup = self.sup.clone();
        self.io
            .run(async move {
                if initialized {
                    sup.shutdown().await;
                } else {
                    sup.kill().await;
                    sup.wait_cleanup().await;
                }
            })
            .await;
    }

    async fn kill(&self) {
        self.settle(false);
        let sup = self.sup.clone();
        self.io
            .run(async move {
                sup.kill().await;
                sup.wait_cleanup().await;
            })
            .await;
    }

    async fn start_job(&self, fields: StartJobFields) -> Result<(), WorkerError> {
        let sup = self.sup.clone();
        self.io
            .run(async move { sup.start_job(fields).await })
            .await
            .map_err(|error| WorkerError::Failed(error.to_string()))
    }

    fn killed(&self) -> bool {
        self.sup.killed()
    }
}

/// The host's launch: on success the process's receipt was reported; a failure before that
/// receipt leaves no process to own.
async fn launch(
    host: Arc<dyn JobExecutor>,
    context: CallContext,
    sup: ProcSupervisor,
    launch: watch::Sender<ProcessLaunch>,
) {
    match host.launch(&context).await {
        Ok(()) => {
            let _ = host.launched(&context).await;
        }
        Err(error) => {
            let cause = error.to_string();
            let mut before_receipt = false;
            launch.send_if_modified(|state| {
                if matches!(state, ProcessLaunch::Pending) {
                    *state = ProcessLaunch::Failed(cause.clone());
                    before_receipt = true;
                    true
                } else {
                    false
                }
            });
            if !before_receipt {
                sup.launch_failed(cause.clone()).await;
            }
            let _ = host.launch_failed(&context, cause).await;
        }
    }
}

/// The launched process's owner: its identity, supervision once initialized, its exit and the
/// start method's reaping receipt, then the host's completion.
async fn own(
    host: Arc<dyn JobExecutor>,
    context: CallContext,
    sup: ProcSupervisor,
    launch: watch::Sender<ProcessLaunch>,
    process: watch::Sender<Option<Arc<ProcessControl>>>,
    initialized: watch::Receiver<Option<bool>>,
    ran: watch::Sender<Option<bool>>,
) {
    let mut launched = launch.subscribe();
    let _ = launched.wait_for(|state| !matches!(state, ProcessLaunch::Pending)).await;
    let mut owned = ProcessLaunch::Claimed;
    launch.send_modify(|state| std::mem::swap(state, &mut owned));
    let mut resource = match owned {
        ProcessLaunch::Started(resource) => resource,
        ProcessLaunch::Failed(cause) => {
            sup.not_launched(cause).await;
            ran.send_replace(Some(false));
            return;
        }
        _ => return,
    };
    if resource.control().is_none() {
        sup.launch_failed("forkserver bootstrap incomplete before PID receipt".into()).await;
    }
    let poll = Duration::from_millis(10);
    let mut identity_announced = false;
    let mut supervising = false;
    let mut observed_exit = false;
    let mut reported = false;
    loop {
        if !identity_announced {
            match resource.poll_identity() {
                Ok(true) => {
                    let control = resource.control().expect("identity observed").clone();
                    process.send_replace(Some(control.clone()));
                    sup.launched(control.pid).await;
                    identity_announced = true;
                }
                Ok(false) => {
                    tokio::time::sleep(poll).await;
                    continue;
                }
                Err(error) => {
                    if !reported {
                        reported = true;
                        sup.launch_failed(format!(
                            "forkserver process identity receipt failed: {error}"
                        ))
                        .await;
                    }
                    tokio::time::sleep(poll).await;
                    continue;
                }
            }
        }
        if !supervising && initialized.borrow().is_some() {
            supervising = true;
            sup.supervise().await;
        }
        let polled = if observed_exit {
            resource.poll_reaped().map(|code| (true, code))
        } else {
            resource.poll_exit().map(|exited| (exited, None))
        };
        match polled {
            Ok((true, _)) if !observed_exit => {
                observed_exit = true;
                process.send_replace(None);
                sup.exited(resource.exitcode()).await;
            }
            Ok((_, Some(code))) => {
                let pid = resource.control().expect("identity observed").pid;
                drop(resource);
                sup.reaped(code).await;
                let completion = sup.completion(i32::try_from(code).ok());
                let _ = host.completed(&context, completion, pid).await;
                ran.send_replace(Some(true));
                return;
            }
            Err(error) if !reported => {
                reported = true;
                crate::diagnostics::emit(
                    log::Level::Error,
                    format!(
                        "native process {} observation failed: {error}",
                        resource.control().expect("identity observed").pid
                    ),
                );
            }
            _ => {}
        }
        tokio::time::sleep(poll).await;
    }
}

/// The supervisor's diagnostics and process handle, until the process exited.
struct ProcessDiagnostics {
    host: Mutex<Option<Arc<dyn JobExecutor>>>,
    process: watch::Receiver<Option<Arc<ProcessControl>>>,
}

impl ProcessDiagnostics {
    fn emit(&self, level: Level, message: &str, extra: Extra, error: Option<HostError>) {
        if let Some(host) = self.host.lock().clone() {
            queue_diagnostic(host, level, message, extra, error);
        }
    }
}

/// One record for the executor's logger, written by the bounded diagnostics writer.
pub(crate) fn queue_diagnostic(
    host: Arc<dyn JobExecutor>,
    level: Level,
    message: &str,
    mut extra: Extra,
    error: Option<HostError>,
) {
    let message = message.to_owned();
    crate::diagnostics::submit_host(move |dropped| {
        if dropped > 0 {
            extra.push(("dropped_supervisor_logs", ExtraValue::Int(dropped as i64)));
        }
        if let Err(failure) = host.log(level as i32, message, extra_fields(extra), error) {
            log::warn!(target: crate::LOG, "process diagnostic failed: {failure}");
        }
    });
}

impl ProcessCallbacks for ProcessDiagnostics {
    fn log(&self, level: Level, msg: &str, extra: Extra) {
        self.emit(level, msg, extra, None);
    }

    fn log_exception(&self, msg: &str, err: HostError) {
        self.emit(Level::Error, msg, Vec::new(), Some(err));
    }

    fn is_alive(&self) -> bool {
        self.process.borrow().is_some()
    }

    fn kill(&self) -> Result<(), String> {
        let process = self.process.borrow();
        process.as_ref().map_or(Ok(()), |process| process.kill().map_err(|error| error.to_string()))
    }

    fn sample_memory(&self) -> Result<(f64, String), MemoryError> {
        let pid = self
            .process
            .borrow()
            .as_ref()
            .map(|process| process.pid)
            .ok_or_else(|| MemoryError::Unavailable("process exited".into()))?;
        super::process::sample_memory(pid)
            .map_err(|error| MemoryError::Unavailable(error.to_string()))
    }

    fn incomplete_cleanup(&self, observation: CleanupObservation) {
        self.emit(
            Level::Error,
            "process cleanup incomplete",
            vec![
                ("pid", ExtraValue::Int(i64::from(observation.pid))),
                ("exited", ExtraValue::Bool(observation.exited)),
                ("reaped", ExtraValue::Bool(observation.reaped)),
            ],
            None,
        );
    }

    fn signal_failed(&self, observation: SignalFailure) {
        self.emit(
            Level::Error,
            "native process signal failed",
            vec![
                ("pid", ExtraValue::Int(i64::from(observation.pid))),
                ("cause", ExtraValue::Str(observation.cause)),
            ],
            None,
        );
    }

    /// The executor is a host object: it drops after the guard.
    fn release(&self) {
        let host = self.host.lock().take();
        drop(host);
    }
}
