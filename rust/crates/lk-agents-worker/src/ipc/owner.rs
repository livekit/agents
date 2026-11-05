//! One job runner as the host and the pool see it (`_core.JobSupervisor`): its launch through the
//! host's launcher, its initialization, its one job, its close and its end, and the state they
//! leave (status, running job, pid, exit code). Supervision continues whether or not anyone still
//! awaits it.

use crate::ipc::telemetry;
use lk_agents_abi::logging::LogWriter;
use lk_agents_net::task::{TaskSet, TokioSpawner};
use parking_lot::Mutex;
use std::sync::Arc;

use lk_agents_abi::errors::HostError;
use lk_agents_abi::invoke::CallContext;
use lk_agents_abi::worker::ipc::{JobCompletion, JobRunnerOptions, JobStatus};
use lk_agents_abi::worker::ipc_host::{
    self, JobLauncher, LaunchReport, SupervisedJob, WorkerError,
};
use lk_agents_abi::worker::job::StartJobFields;
use lk_agents_abi::worker::types::RunningJobInfo;
use tokio::sync::watch;

use super::executor::{RunnerExecutor, ThreadExecutor};
use super::process::{ProcessExecutor, ProcessReceipt};
use super::{
    Extra, ExtraValue, HEALTH_CHECK_GRACE, Hello, InitError, Level, RunnerSupervisor,
    SupervisorCallbacks, SupervisorOptions, extra_fields, secs,
};
use lk_agents_net::env::ProcessEnv;

/// What the runner's supervision has established, read by the getters and stamped on its logs.
#[derive(Default)]
pub(crate) struct RunnerState {
    closing: bool,
    pub(crate) pid: Option<u32>,
    running_job: Option<RunningJobInfo>,
    /// Set once the supervision is over, before the launcher hears of it.
    completion: Option<JobCompletion>,
    initialize_time: Option<f64>,
}

pub struct JobSupervisor {
    id: String,
    options: JobRunnerOptions,
    launcher: Arc<dyn JobLauncher>,
    context: CallContext,
    logger: JobLogger,
    spawner: TokioSpawner,
    state: Arc<Mutex<RunnerState>>,
    /// Set by `start()`.
    sup: Mutex<Option<RunnerSupervisor>>,
    /// Whether the runner initialized, once that settled (answered, failed, timed out, killed).
    initialized: watch::Sender<Option<bool>>,
    /// Whether a runner ran, once its supervision is over and the launcher heard of it.
    ran: watch::Sender<Option<bool>>,
    /// The completion report; it ends with the runner, and dropping this object does not cancel
    /// it.
    tasks: Arc<TaskSet>,
}

impl SupervisedJob for JobSupervisor {}

impl JobSupervisor {
    fn sup(&self) -> Option<RunnerSupervisor> {
        self.sup.lock().clone()
    }

    fn settle(&self, initialized: bool) {
        self.initialized.send_if_modified(|v| v.is_none() && v.replace(initialized).is_none());
    }

    pub(crate) fn logger(&self) -> &JobLogger {
        &self.logger
    }

    /// Whether a launched runner ended; an error when none was started.
    pub(crate) async fn wait_exit(&self) -> Result<bool, HostError> {
        let sup = self.sup().ok_or_else(|| HostError::new("process not started"))?;
        Ok(self.spawner.run(async move { sup.wait_exit().await }).await)
    }

    fn supervision(&self) -> SupervisorOptions {
        let opts = &self.options;
        SupervisorOptions {
            runner: if opts.thread { "job thread" } else { "process" },
            initialize_timeout: opts.initialize_timeout,
            close_timeout: opts.close_timeout,
            memory_warn_mb: opts.memory_warn_mb,
            memory_limit_mb: opts.memory_limit_mb,
            ping_interval: opts.ping_interval,
            ping_timeout: opts.ping_timeout,
            high_ping_threshold: opts.high_ping_threshold,
        }
    }
}

impl ipc_host::JobSupervisor for JobSupervisor {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        launcher: Arc<dyn JobLauncher>,
        mut options: JobRunnerOptions,
        context: CallContext,
    ) -> Result<Self, WorkerError> {
        // Logging must be available before any launch.
        env.logging.initialize().map_err(WorkerError::Failed)?;
        if options.thread {
            // A thread cannot be stopped: it is never late nor too large.
            options.ping_timeout = 0.0;
            options.memory_warn_mb = 0.0;
            options.memory_limit_mb = 0.0;
        }
        let state = Arc::new(Mutex::new(RunnerState::default()));
        let logger =
            JobLogger { logging: env.logging, launcher: launcher.clone(), state: state.clone() };
        Ok(Self {
            id: lk_agents_net::strings::shortuuid(if options.thread {
                "THEXEC_"
            } else {
                "PCEXEC_"
            }),
            options,
            launcher,
            context,
            logger,
            spawner: TokioSpawner::new((env.runtime)().map_err(WorkerError::Failed)?),
            state,
            sup: Mutex::new(None),
            initialized: watch::channel(None).0,
            ran: watch::channel(None).0,
            tasks: Arc::new(TaskSet::default()),
        })
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn thread(&self) -> bool {
        self.options.thread
    }

    fn started(&self) -> bool {
        self.sup.lock().is_some()
    }

    fn initialized(&self) -> bool {
        *self.initialized.borrow() == Some(true)
    }

    fn closing(&self) -> bool {
        self.state.lock().closing
    }

    fn killed(&self) -> bool {
        self.sup().is_some_and(|sup| sup.killed())
    }

    fn pid(&self) -> Option<u32> {
        self.state.lock().pid
    }

    fn exitcode(&self) -> Option<i32> {
        self.state.lock().completion.as_ref().and_then(|completion| completion.exit_code)
    }

    fn status(&self) -> Option<JobStatus> {
        let state = self.state.lock();
        state.running_job.as_ref()?;
        Some(match &state.completion {
            None => JobStatus::Running,
            Some(JobCompletion { exit_code: Some(0), failure: None }) => JobStatus::Success,
            Some(_) => JobStatus::Failed,
        })
    }

    fn running_job(&self) -> Option<RunningJobInfo> {
        self.state.lock().running_job.clone()
    }

    fn initialize_time(&self) -> Option<f64> {
        self.state.lock().initialize_time
    }

    async fn start(&self) -> Result<(), WorkerError> {
        let sup = {
            let mut slot = self.sup.lock();
            if self.state.lock().closing {
                return Err(WorkerError::Failed("process is closed".into()));
            }
            if slot.is_some() {
                return Err(WorkerError::Failed("process already started".into()));
            }
            let (launcher, context) = (self.launcher.clone(), self.context.clone());
            let _runtime = self.spawner.handle().enter();
            let executor: Arc<dyn RunnerExecutor> = if self.options.thread {
                Arc::new(ThreadExecutor::new(launcher, context))
            } else {
                Arc::new(ProcessExecutor::new(launcher, context, self.logger.clone()))
            };
            let sup = RunnerSupervisor::new(
                executor,
                self.logger.clone(),
                self.supervision(),
                lk_agents_net::blocking_io::blocking_io(),
            );
            *slot = Some(sup.clone());
            sup
        };
        let (supervised, state, ran) = (sup.clone(), self.state.clone(), self.ran.clone());
        let (launcher, context) = (self.launcher.clone(), self.context.clone());
        self.tasks.spawn(&self.spawner, "job runner completion", async move {
            let launched = supervised.wait_cleaned().await;
            if launched {
                let completion = supervised.completion();
                state.lock().completion = Some(completion.clone());
                let _ = launcher.completed(&telemetry::Cx::NONE, &context, completion).await;
            }
            ran.send_replace(Some(launched));
        });
        // The launcher's own failure is raised as it was.
        self.spawner.run(async move { sup.launched().await }).await.map_err(WorkerError::Host)
    }

    async fn initialize(&self) -> Result<f64, WorkerError> {
        let sup = self.sup().ok_or_else(|| WorkerError::Failed("process not started".into()))?;
        let opts = &self.options;
        // The child calls itself orphaned when the parent goes quiet past its pong deadline
        // (+10 s); without a pong deadline it has none.
        let health_timeout_ms = (opts.ping_interval > 0.0 && opts.ping_timeout > 0.0)
            .then(|| secs(opts.ping_timeout + HEALTH_CHECK_GRACE).as_millis() as u64);
        let hello = Hello {
            asyncio_debug: opts.asyncio_debug,
            http_proxy: opts.http_proxy.clone().filter(|p| !p.is_empty()),
            health_timeout_ms,
        };
        let outcome = self
            .spawner
            .run(async move {
                let outcome = sup.initialize(hello).await;
                sup.supervise().await;
                outcome
            })
            .await;
        self.settle(matches!(outcome, Ok(Some(_))));
        match outcome {
            Ok(Some(elapsed)) => {
                let elapsed = elapsed.as_secs_f64();
                self.state.lock().initialize_time = Some(elapsed);
                Ok(elapsed)
            }
            Ok(None) => Err(WorkerError::TimedOut("initialization timed out".into())),
            Err(InitError::Closed) => Err(WorkerError::Closed),
            Err(error) => Err(WorkerError::Failed(error.to_string())),
        }
    }

    async fn launch_job(
        &self,
        info: RunningJobInfo,
        _context: CallContext,
        cx: telemetry::Cx,
    ) -> Result<(), WorkerError> {
        let sup =
            self.sup().ok_or_else(|| WorkerError::Failed("process not initialized".into()))?;
        let fields = StartJobFields {
            job: info.job.proto.clone(),
            name: info.accept_arguments.name.clone(),
            identity: info.accept_arguments.identity.clone(),
            metadata: info.accept_arguments.metadata.clone(),
            url: info.url.clone(),
            token: info.token.clone(),
            worker_id: info.worker_id.clone(),
            fake_job: info.fake_job,
            received_at: info.received_at,
            accepted_at: info.accepted_at,
            assigned_at: info.assigned_at,
            // The dispatch timeline: a ready process is now handling the job.
            launched_at: lk_agents_net::clock::unix_now(),
            traceparent: cx.span().map(telemetry::SpanContext::traceparent),
        };
        // The runner's own guard refuses a second job and a runner that is not ready.
        self.spawner
            .run(async move { sup.start_job(fields).await })
            .await
            .map_err(|error| WorkerError::Failed(error.to_string()))?;
        self.state.lock().running_job = Some(info);
        Ok(())
    }

    async fn aclose(&self) {
        self.state.lock().closing = true;
        let Some(sup) = self.sup() else { return };
        let initialized = self.initialized();
        self.settle(false);
        self.spawner
            .run(async move {
                if initialized {
                    sup.shutdown().await;
                } else {
                    sup.kill().await;
                }
            })
            .await;
    }

    async fn kill(&self) {
        self.state.lock().closing = true;
        let Some(sup) = self.sup() else { return };
        self.settle(false);
        self.spawner.run(async move { sup.kill().await }).await;
    }

    async fn join(&self) -> bool {
        if !ipc_host::JobSupervisor::started(self) {
            return false;
        }
        let mut ran = self.ran.subscribe();
        ran.wait_for(Option::is_some).await.map(|done| done.unwrap_or(false)).unwrap_or(false)
    }
}

/// One launch's report channel (`_core.JobLaunch`): a process's receipt or a thread's return.
pub struct JobLaunch(Report);

pub(crate) enum Report {
    Process(Arc<ProcessReceipt>),
    Thread(Arc<watch::Sender<bool>>),
}

impl JobLaunch {
    pub(crate) fn report(report: Report) -> Arc<dyn LaunchReport> {
        Arc::new(Self(report))
    }
}

impl LaunchReport for JobLaunch {}

impl ipc_host::JobLaunch for JobLaunch {
    fn thread(&self) -> bool {
        matches!(self.0, Report::Thread(_))
    }

    fn process_started(
        &self,
        pid: u32,
        sentinel: i64,
        method: String,
        owned: Vec<i64>,
    ) -> Result<(), WorkerError> {
        match &self.0 {
            Report::Process(receipt) => receipt
                .report(pid, sentinel, &method, owned)
                .map_err(|error| WorkerError::Failed(error.to_string())),
            Report::Thread(_) => Err(WorkerError::Failed("a job thread has no process".into())),
        }
    }

    fn thread_exited(&self) -> Result<(), WorkerError> {
        match &self.0 {
            Report::Thread(exited) => {
                exited.send_replace(true);
                Ok(())
            }
            Report::Process(_) => Err(WorkerError::Failed("no job thread to report".into())),
        }
    }
}

/// The runner's diagnostics on the launcher's logger, naming its process and its job.
#[derive(Clone)]
pub(crate) struct JobLogger {
    logging: LogWriter,
    launcher: Arc<dyn JobLauncher>,
    pub(crate) state: Arc<Mutex<RunnerState>>,
}

impl JobLogger {
    #[cfg(test)]
    pub(crate) fn new(logging: LogWriter, launcher: Arc<dyn JobLauncher>) -> Self {
        Self { logging, launcher, state: Arc::default() }
    }

    /// One record, written by the bounded log writer.
    pub(crate) fn log(
        &self,
        level: Level,
        message: &str,
        mut extra: Extra,
        error: Option<HostError>,
    ) {
        {
            let state = self.state.lock();
            if let Some(pid) = state.pid {
                extra.push(("pid", ExtraValue::Int(i64::from(pid))));
            }
            if let Some(info) = &state.running_job {
                extra.push(("job_id", ExtraValue::Str(info.job.id.clone())));
                if let Some(room) = &info.job.room {
                    extra.push(("room", ExtraValue::Str(room.clone())));
                }
                if let Some(room_id) = &info.job.room_sid {
                    extra.push(("room_id", ExtraValue::Str(room_id.clone())));
                }
            }
        }
        let (message, launcher) = (message.to_owned(), self.launcher.clone());
        self.logging.submit_host(move |dropped| {
            if dropped > 0 {
                extra.push(("dropped_supervisor_logs", ExtraValue::Int(dropped as i64)));
            }
            if let Err(failure) = launcher.log(level as i32, message, extra_fields(extra), error) {
                log::warn!(target: crate::LOG, "process diagnostic failed: {failure}");
            }
        });
    }
}

impl SupervisorCallbacks for JobLogger {
    fn log(&self, level: Level, msg: &str, extra: Extra) {
        JobLogger::log(self, level, msg, extra, None);
    }
}

#[cfg(test)]
#[path = "owner_tests.rs"]
mod tests;
