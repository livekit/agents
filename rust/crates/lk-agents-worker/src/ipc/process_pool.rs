//! The worker's warm job processes as the host drives them (`_core.ProcPool`): the pool keeps
//! them warm and hands each job the next ready one; each process is a [`JobSupervisor`] over a
//! launcher the host makes.

use crate::ipc::telemetry;
use lk_agents_abi::logging::LogWriter;
use std::any::Any;
use std::sync::Arc;

use lk_agents_abi::errors::HostError;
use lk_agents_abi::invoke::CallContext;
use lk_agents_abi::worker::ipc::{JobRunnerOptions, JobStatus};
use lk_agents_abi::worker::ipc_host::{
    JobSupervisor as _, PoolCallbacks, SupervisedJob, WorkerError,
};
use lk_agents_abi::worker::types::RunningJobInfo;

use super::owner::JobSupervisor;
use super::pool::{Executors, LaunchError, PoolJoinOutcome, PoolOptions, ProcPool, ProcessEvent};
use super::{Level, secs};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::task::TokioSpawner;

/// One of the pool's processes, compared by identity.
#[derive(Clone)]
pub struct Executor(pub(super) Arc<dyn SupervisedJob>);

impl PartialEq for Executor {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::addr_eq(Arc::as_ptr(&self.0), Arc::as_ptr(&other.0))
    }
}

impl Executor {
    fn supervisor(&self) -> Result<&JobSupervisor, HostError> {
        let process: &dyn Any = &*self.0;
        process.downcast_ref().ok_or_else(|| HostError::new("a foreign process in the pool"))
    }
}

fn host_error(error: WorkerError) -> HostError {
    match error {
        WorkerError::Host(error) => error,
        error => HostError::new(error.to_string()),
    }
}

/// The host's pool, the processes' settings and the context their host calls run in (the
/// pool's constructor's).
struct HostExecutors {
    env: ProcessEnv,
    host: Arc<dyn PoolCallbacks>,
    options: JobRunnerOptions,
    context: CallContext,
}

#[async_trait::async_trait]
impl Executors for HostExecutors {
    fn logger(&self) -> LogWriter {
        self.env.logging
    }

    async fn create(&self) -> Result<Executor, HostError> {
        let launcher = self.host.create(&telemetry::Cx::NONE).await?;
        let process =
            JobSupervisor::new(&self.env, launcher, self.options.clone(), self.context.clone())
                .map_err(host_error)?;
        Ok(Executor(Arc::new(process)))
    }

    async fn start(&self, proc: &Executor) -> Result<(), HostError> {
        proc.supervisor()?.start().await.map_err(host_error)
    }

    async fn initialize(&self, proc: &Executor) -> Result<(), HostError> {
        proc.supervisor()?.initialize().await.map(drop).map_err(host_error)
    }

    async fn launch(
        &self,
        proc: &Executor,
        job: &RunningJobInfo,
        cx: &telemetry::Cx,
    ) -> Result<(), HostError> {
        let process = proc.supervisor()?;
        process.launch_job(job.clone(), self.context.clone(), cx.clone()).await.map_err(host_error)
    }

    async fn close(&self, proc: &Executor) -> Result<(), HostError> {
        proc.supervisor()?.aclose().await;
        Ok(())
    }

    fn failed(&self, proc: &Executor, message: &'static str, error: HostError) {
        if let Ok(process) = proc.supervisor() {
            process.logger().log(Level::Error, message, vec![], Some(error));
        }
    }

    async fn wait_exit(&self, proc: &Executor) -> Result<(), HostError> {
        match proc.supervisor()?.wait_exit().await? {
            true => Ok(()),
            false => Err(HostError::new("ready process finished without an OS exit observation")),
        }
    }

    async fn join(&self, proc: &Executor) -> Result<PoolJoinOutcome, HostError> {
        Ok(match proc.supervisor()?.join().await {
            true => PoolJoinOutcome::Reaped,
            false => PoolJoinOutcome::NotLaunched,
        })
    }

    fn emit(&self, event: ProcessEvent, proc: &Executor) {
        self.host.emit(event.as_str().into(), proc.0.clone());
    }
}

pub struct ProcessPool {
    pool: ProcPool,
    spawner: TokioSpawner,
}

impl lk_agents_abi::worker::ipc_host::ProcessPool for ProcessPool {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        host: Arc<dyn PoolCallbacks>,
        num_idle_processes: usize,
        options: JobRunnerOptions,
        context: CallContext,
    ) -> Result<Self, WorkerError> {
        let opts = PoolOptions {
            num_idle_processes,
            initialize_timeout: secs(options.initialize_timeout),
            close_timeout: secs(options.close_timeout),
        };
        let spawner = TokioSpawner::new((env.runtime)().map_err(WorkerError::Failed)?);
        let executors = HostExecutors { env: env.clone(), host, options, context };
        Ok(Self { pool: ProcPool::new(Arc::new(executors), opts), spawner })
    }

    fn processes(&self) -> Vec<Arc<dyn SupervisedJob>> {
        self.pool.processes().into_iter().map(|proc| proc.0).collect()
    }

    fn active_jobs(&self) -> Vec<RunningJobInfo> {
        self.pool
            .processes()
            .into_iter()
            .filter_map(|proc| {
                let process = proc.supervisor().ok()?;
                (process.status() == Some(JobStatus::Running)).then(|| process.running_job())?
            })
            .collect()
    }

    fn get_by_job_id(&self, job_id: String) -> Option<Arc<dyn SupervisedJob>> {
        self.pool
            .processes()
            .into_iter()
            .find(|proc| {
                let job = proc.supervisor().ok().and_then(|process| process.running_job());
                job.is_some_and(|info| info.job.id == job_id)
            })
            .map(|proc| proc.0)
    }

    fn jobs_waiting(&self) -> usize {
        self.pool.jobs_waiting()
    }

    fn target_idle_processes(&self) -> usize {
        self.pool.target_idle()
    }

    fn set_target_idle_processes(&self, n: usize) {
        // The pool's work for the new target runs on the injected runtime.
        let _runtime = self.spawner.handle().enter();
        self.pool.set_target_idle(n);
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
                LaunchError::Admission(reason) => WorkerError::Failed(reason.to_string()),
                LaunchError::AcquireExhausted { attempts } => WorkerError::Failed(format!(
                    "no process became available after {attempts} attempts"
                )),
                LaunchError::NoProcess => {
                    WorkerError::Failed("no process became available after 3 attempts".into())
                }
                LaunchError::Launch(error) => WorkerError::Host(error),
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
