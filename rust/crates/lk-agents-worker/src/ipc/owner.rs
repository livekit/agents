//! The worker's side of one job runner as its executor drives it (`_core.JobSupervisor`): the
//! launch through the process or thread executor, then initialization, supervision and the
//! runner's end, which continue whether or not the host still awaits them.

use lk_agents_abi::logging::LogWriter;
use lk_agents_net::task::{TaskSet, TokioSpawner};
use parking_lot::Mutex;
use std::sync::Arc;

use lk_agents_abi::errors::HostError;
use lk_agents_abi::invoke::CallContext;
use lk_agents_abi::worker::ipc_host::{
    self, JobExecutor as HostJobLauncher, SupervisedJob, WorkerError,
};
use lk_agents_abi::worker::job::StartJobFields;
use tokio::sync::watch;

use super::executor::ThreadExecutor;
use super::process::ProcessExecutor;
use super::{
    Extra, ExtraValue, HEALTH_CHECK_GRACE, Hello, InitError, Level, RunnerSupervisor,
    SupervisorCallbacks, SupervisorOptions, extra_fields, secs,
};
use lk_agents_net::env::ProcessEnv;

/// The executor the host reports its launch to, until the runner's supervision is over: it holds
/// the host's executor, so keeping it longer would be a cycle the host's collector cannot see.
enum Launcher {
    Process(Arc<ProcessExecutor>),
    Thread(Arc<ThreadExecutor>),
}

pub struct JobSupervisor {
    pub(crate) sup: RunnerSupervisor,
    launcher: Arc<Mutex<Option<Launcher>>>,
    spawner: TokioSpawner,
    /// The child's health timeout, from the supervision's ping settings.
    health_timeout_ms: Option<u64>,
    /// Whether the runner initialized, once that settled (answered, failed, timed out, killed).
    initialized: watch::Sender<Option<bool>>,
    /// Whether a runner ran, once its supervision is over.
    ran: watch::Sender<Option<bool>>,
    /// The completion report; it ends with the runner, and dropping this object does not cancel
    /// it.
    _tasks: Arc<TaskSet>,
}

impl SupervisedJob for JobSupervisor {}

impl JobSupervisor {
    fn settle(&self, initialized: bool) {
        self.initialized.send_if_modified(|v| v.is_none() && v.replace(initialized).is_none());
    }

    async fn wait_ran(&self) -> bool {
        let mut ran = self.ran.subscribe();
        ran.wait_for(Option::is_some).await.map(|done| done.unwrap_or(false)).unwrap_or(false)
    }
}

impl ipc_host::JobSupervisor for JobSupervisor {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        host: Arc<dyn HostJobLauncher>,
        thread: bool,
        initialize_timeout: f64,
        close_timeout: f64,
        memory_warn_mb: f64,
        memory_limit_mb: f64,
        ping_interval: f64,
        ping_timeout: f64,
        high_ping_threshold: f64,
        context: CallContext,
    ) -> Result<Self, WorkerError> {
        // Logging must be available before any launch.
        env.logging.initialize().map_err(WorkerError::Failed)?;
        let spawner = TokioSpawner::new((env.runtime)().map_err(WorkerError::Failed)?);
        let (launcher, executor): (_, Arc<dyn super::executor::RunnerExecutor>) = if thread {
            let executor = Arc::new(ThreadExecutor::new(host.clone(), context.clone()));
            (Launcher::Thread(executor.clone()), executor)
        } else {
            let executor =
                Arc::new(ProcessExecutor::new(host.clone(), context.clone(), env.logging));
            (Launcher::Process(executor.clone()), executor)
        };
        let opts = SupervisorOptions {
            runner: if thread { "job thread" } else { "process" },
            initialize_timeout,
            close_timeout,
            memory_warn_mb,
            memory_limit_mb,
            ping_interval,
            ping_timeout,
            high_ping_threshold,
        };
        // The child calls itself orphaned when the parent goes quiet past its pong deadline
        // (+10 s); without a pong deadline it has none.
        let health_timeout_ms = (ping_interval > 0.0 && ping_timeout > 0.0)
            .then(|| secs(ping_timeout + HEALTH_CHECK_GRACE).as_millis() as u64);
        let sup = {
            let _runtime = spawner.handle().enter();
            let logger =
                ExecutorLogger { logging: env.logging, host: Mutex::new(Some(host.clone())) };
            RunnerSupervisor::new(executor, logger, opts, lk_agents_net::blocking_io::blocking_io())
        };
        let launcher = Arc::new(Mutex::new(Some(launcher)));
        let ran = watch::channel(None).0;
        let tasks = Arc::new(TaskSet::default());
        let (supervised, reported, launched_by) = (sup.clone(), ran.clone(), launcher.clone());
        tasks.spawn(&spawner, "job runner completion", async move {
            let launched = supervised.wait_cleaned().await;
            if launched {
                let _ = host.completed(&context, supervised.completion()).await;
            }
            launched_by.lock().take();
            reported.send_replace(Some(launched));
        });
        Ok(Self {
            sup,
            launcher,
            spawner,
            health_timeout_ms,
            initialized: watch::channel(None).0,
            ran,
            _tasks: tasks,
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
            Ok(Some(elapsed)) => Ok(elapsed.as_secs_f64()),
            Ok(None) => Err(WorkerError::TimedOut("initialization timed out".into())),
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
        match &*self.launcher.lock() {
            Some(Launcher::Process(executor)) => executor
                .launched(pid, sentinel, &method, owned)
                .map_err(|error| WorkerError::Failed(error.to_string())),
            _ => Err(WorkerError::Failed("no process launch to report".into())),
        }
    }

    fn _core_thread_exited(&self) -> Result<(), WorkerError> {
        match &*self.launcher.lock() {
            Some(Launcher::Thread(executor)) => {
                executor.exited();
                Ok(())
            }
            _ => Err(WorkerError::Failed("no job thread to report".into())),
        }
    }

    async fn run(&self) -> bool {
        self.wait_ran().await
    }

    async fn _core_wait_exit(&self) -> bool {
        let sup = self.sup.clone();
        self.spawner.run(async move { sup.wait_exit().await }).await
    }

    async fn _core_join(&self) -> bool {
        self.wait_ran().await
    }

    async fn aclose(&self) {
        let initialized = self.initialized();
        self.settle(false);
        let sup = self.sup.clone();
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
        self.settle(false);
        let sup = self.sup.clone();
        self.spawner.run(async move { sup.kill().await }).await;
    }

    async fn start_job(&self, fields: StartJobFields) -> Result<(), WorkerError> {
        let sup = self.sup.clone();
        self.spawner
            .run(async move { sup.start_job(fields).await })
            .await
            .map_err(|error| WorkerError::Failed(error.to_string()))
    }

    fn killed(&self) -> bool {
        self.sup.killed()
    }
}

/// The supervisor's logger, until its supervision is over.
struct ExecutorLogger {
    logging: LogWriter,
    host: Mutex<Option<Arc<dyn HostJobLauncher>>>,
}

impl ExecutorLogger {
    fn emit(&self, level: Level, message: &str, extra: Extra, error: Option<HostError>) {
        let host = self.host.lock().clone();
        if let Some(host) = host {
            queue_log(self.logging, host, level, message, extra, error);
        }
    }
}

/// One record for the executor's logger, written by the bounded log writer.
pub(crate) fn queue_log(
    logging: LogWriter,
    host: Arc<dyn HostJobLauncher>,
    level: Level,
    message: &str,
    mut extra: Extra,
    error: Option<HostError>,
) {
    let message = message.to_owned();
    logging.submit_host(move |dropped| {
        if dropped > 0 {
            extra.push(("dropped_supervisor_logs", ExtraValue::Int(dropped as i64)));
        }
        if let Err(failure) = host.log(level as i32, message, extra_fields(extra), error) {
            log::warn!(target: crate::LOG, "process diagnostic failed: {failure}");
        }
    });
}

impl SupervisorCallbacks for ExecutorLogger {
    fn log(&self, level: Level, msg: &str, extra: Extra) {
        self.emit(level, msg, extra, None);
    }

    /// The executor is a host object: it drops after the guard.
    fn release(&self) {
        let host = self.host.lock().take();
        drop(host);
    }
}
