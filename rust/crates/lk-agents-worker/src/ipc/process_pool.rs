use crate::ipc::telemetry;
use std::sync::Arc;

use lk_agents_abi::errors::HostError;
use lk_agents_abi::invoke::CallContext;
use lk_agents_abi::worker::ipc::{JobRunnerOptions, JobStatus};
use lk_agents_abi::worker::ipc_host::{
    JobLaunchers, JobSupervisor as _, SupervisedJob, WorkerError,
};
use lk_agents_abi::worker::job::RunningJobInfo;

use super::owner::JobSupervisor;
use super::pool::{Executors, LaunchError, PoolOptions, ProcPool, ProcessEvent};
use super::supervisor::runner_log;
use super::{SupervisorCallbacks, secs};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::task::TokioSpawner;

/// One of the pool's processes, compared by identity.
pub struct Executor<P = JobSupervisor>(pub(super) Arc<P>);

impl<P> Clone for Executor<P> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<P> PartialEq for Executor<P> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::addr_eq(Arc::as_ptr(&self.0), Arc::as_ptr(&other.0))
    }
}

fn host_error(error: WorkerError) -> HostError {
    match error {
        WorkerError::Host(error) => error,
        error => HostError::new(error.to_string()),
    }
}

/// Told each job's status as its process takes it and ends.
pub type JobObserver = Box<dyn Fn(&RunningJobInfo, JobStatus) + Send + Sync>;

/// The host's launchers, the processes' settings, the context their host calls run in (the
/// pool's constructor's) and the job status observer.
struct HostExecutors {
    env: ProcessEnv,
    launchers: Arc<dyn JobLaunchers>,
    options: JobRunnerOptions,
    context: CallContext,
    observer: Option<JobObserver>,
}

#[async_trait::async_trait]
impl Executors for HostExecutors {
    type Process = JobSupervisor;
    async fn create(&self) -> Result<Executor, HostError> {
        let launcher = self.launchers.create(&telemetry::Cx::NONE).await?;
        let process =
            JobSupervisor::new(&self.env, launcher, self.options.clone(), self.context.clone())
                .map_err(host_error)?;
        Ok(Executor(Arc::new(process)))
    }

    async fn start(&self, proc: &Executor) -> Result<(), HostError> {
        proc.0.start().await.map_err(host_error)
    }

    async fn initialize(&self, proc: &Executor) -> Result<(), HostError> {
        proc.0.initialize().await.map(drop).map_err(host_error)
    }

    async fn launch(
        &self,
        proc: &Executor,
        job: &RunningJobInfo,
        cx: &telemetry::Cx,
    ) -> Result<(), HostError> {
        let process = &proc.0;
        process.launch_job(job.clone(), self.context.clone(), cx.clone()).await.map_err(host_error)
    }

    async fn close(&self, proc: &Executor) -> Result<(), HostError> {
        proc.0.aclose().await;
        Ok(())
    }

    fn failed(&self, proc: &Executor, message: &'static str, error: HostError) {
        runner_log!(proc.0.logger().runner(), log::Level::Error, exc_text:% = error.exc_text(); "{message}");
    }

    async fn wait_exit(&self, proc: &Executor) -> Result<(), HostError> {
        match proc.0.wait_exit().await? {
            true => Ok(()),
            false => Err(HostError::new("ready process finished without an OS exit observation")),
        }
    }

    async fn join(&self, proc: &Executor) -> Result<(), HostError> {
        proc.0.join().await;
        Ok(())
    }

    fn emit(&self, event: ProcessEvent, proc: &Executor) {
        let Some(observer) = &self.observer else { return };
        if matches!(event, ProcessEvent::Started | ProcessEvent::Closed | ProcessEvent::JobLaunched)
            && let (Some(job), Some(status)) = (proc.0.running_job(), proc.0.status())
        {
            observer(&job, status);
        }
    }
}

pub struct ProcessPool {
    pool: ProcPool,
    spawner: TokioSpawner,
}

impl ProcessPool {
    /// A pool reporting its jobs' status to `observer`.
    pub fn observed(
        env: &ProcessEnv,
        launchers: Arc<dyn JobLaunchers>,
        num_idle_processes: usize,
        options: JobRunnerOptions,
        context: CallContext,
        observer: Option<JobObserver>,
    ) -> Result<Self, WorkerError> {
        let opts = PoolOptions {
            num_idle_processes,
            initialize_timeout: secs(options.initialize_timeout),
            close_timeout: secs(options.close_timeout),
            instruments: &telemetry::process::INSTRUMENTS,
        };
        let failed =
            |error: lk_agents_net::env::ProcessError| WorkerError::Failed(error.to_string());
        let spawner = TokioSpawner::new((env.runtime)().map_err(failed)?);
        let rt = env.background().map_err(failed)?;
        let executors = HostExecutors { env: env.clone(), launchers, options, context, observer };
        Ok(Self { pool: ProcPool::new(Arc::new(executors), opts, rt), spawner })
    }

    /// The jobs whose process runs them now (`status` `RUNNING`).
    pub fn active_jobs(&self) -> Vec<RunningJobInfo> {
        self.pool
            .processes()
            .into_iter()
            .filter_map(|proc| {
                let process = &proc.0;
                (process.status() == Some(JobStatus::Running)).then(|| process.running_job())?
            })
            .collect()
    }

    /// The process that took the job `job_id`.
    pub fn job_process(&self, job_id: &str) -> Option<Arc<JobSupervisor>> {
        self.pool
            .processes()
            .into_iter()
            .find(|proc| proc.0.running_job().is_some_and(|info| info.job.id == job_id))
            .map(|proc| proc.0)
    }

    pub fn set_target_idle_processes(&self, n: usize) {
        // The pool's work for the new target runs on the injected runtime.
        let _runtime = self.spawner.handle().enter();
        self.pool.set_target_idle(n);
    }
}

impl lk_agents_abi::worker::ipc_host::ProcessPool for ProcessPool {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        launchers: Arc<dyn JobLaunchers>,
        num_idle_processes: usize,
        options: JobRunnerOptions,
        context: CallContext,
    ) -> Result<Self, WorkerError> {
        Self::observed(env, launchers, num_idle_processes, options, context, None)
    }

    fn processes(&self) -> Vec<Arc<dyn SupervisedJob>> {
        self.pool.processes().into_iter().map(|proc| proc.0 as Arc<dyn SupervisedJob>).collect()
    }

    async fn start(&self) -> Result<(), WorkerError> {
        let pool = self.pool.clone();
        self.spawner
            .run(async move { pool.start().await })
            .await
            .map_err(|error| WorkerError::Failed(error.to_string()))
    }

    async fn launch_job(
        &self,
        info: RunningJobInfo,
        _context: CallContext,
        cx: telemetry::Cx,
    ) -> Result<(), WorkerError> {
        let pool = self.pool.clone();
        self.spawner.run(async move { pool.launch_job(info, cx).await }).await.map_err(|error| {
            match error {
                LaunchError::Launch(error) => WorkerError::Host(error),
                error => WorkerError::Failed(error.to_string()),
            }
        })
    }

    async fn join_jobs(&self) {
        let pool = self.pool.clone();
        self.spawner.run(async move { pool.join_jobs().await }).await;
    }

    async fn aclose(&self) {
        let pool = self.pool.clone();
        self.spawner.run(async move { pool.aclose().await }).await;
    }
}

/// The server runtime's own pool, driven on the runtime it was made on.
#[async_trait::async_trait]
impl crate::server::host::JobPool for ProcessPool {
    async fn start(&self) -> Result<(), HostError> {
        let pool = self.pool.clone();
        self.spawner
            .run(async move { pool.start().await })
            .await
            .map_err(|error| HostError::new(error.to_string()))
    }

    async fn launch(&self, info: RunningJobInfo) -> Result<(), HostError> {
        let pool = self.pool.clone();
        let launched =
            self.spawner.run(async move { pool.launch_job(info, telemetry::Cx::NONE).await }).await;
        launched.map_err(|error| match error {
            LaunchError::Launch(error) => error,
            error => HostError::new(error.to_string()),
        })
    }

    async fn terminate(&self, job_id: &str) {
        if let Some(process) = self.job_process(job_id) {
            process.aclose().await;
        }
    }

    fn set_target_idle(&self, target: usize) {
        self.set_target_idle_processes(target);
    }

    fn active_jobs(&self) -> Vec<RunningJobInfo> {
        ProcessPool::active_jobs(self)
    }

    async fn join(&self) {
        let pool = self.pool.clone();
        self.spawner.run(async move { pool.join_jobs().await }).await;
    }

    async fn close(&self) {
        let pool = self.pool.clone();
        self.spawner.run(async move { pool.aclose().await }).await;
    }
}
