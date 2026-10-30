//! The worker's warm job processes as the host drives them (`_core.ProcPool`): the pool keeps
//! them warm and hands each job the next ready one; the host makes the executors.

use std::any::Any;
use std::sync::{Arc, OnceLock};

use lk_agents_abi::HostRef;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::invoke::CallContext;
use lk_agents_abi::worker::ipc_host::{JobExecutor, PoolCallbacks, WorkerError};

use super::owner::{ProcessOwner, queue_diagnostic};
use super::pool::{Executors, LaunchError, PoolEvent, PoolJoinOutcome, PoolOptions, ProcPool};
use super::{Level, ProcSupervisor, secs};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::task::TokioSpawner;

/// One executor and, once started, its native supervisor.
#[derive(Clone)]
pub struct Executor {
    pub(super) host: Arc<dyn JobExecutor>,
    supervisor: Arc<OnceLock<ProcSupervisor>>,
}

impl Executor {
    pub(super) fn new(host: Arc<dyn JobExecutor>) -> Self {
        Self { host, supervisor: Arc::new(OnceLock::new()) }
    }
}

/// Compared by identity.
impl PartialEq for Executor {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::addr_eq(Arc::as_ptr(&self.host), Arc::as_ptr(&other.host))
    }
}

/// The host's pool and the context its executors' steps run in (the pool's constructor's).
struct HostExecutors {
    host: Arc<dyn PoolCallbacks>,
    context: CallContext,
}

#[async_trait::async_trait]
impl Executors for HostExecutors {
    async fn create(&self) -> Result<Executor, HostError> {
        Ok(Executor::new(self.host.create().await?))
    }

    async fn start(&self, proc: &Executor) -> Result<(), HostError> {
        let started: Arc<dyn Any + Send + Sync> = proc.host.start(&self.context).await?;
        let owner = started
            .downcast::<ProcessOwner>()
            .map_err(|_| HostError::new("the executor started a foreign ProcSupervisor"))?;
        proc.supervisor
            .set(owner.sup.clone())
            .map_err(|_| HostError::new("process start already reported"))
    }

    async fn initialize(&self, proc: &Executor) -> Result<(), HostError> {
        proc.host.initialize(&self.context).await
    }

    /// The dispatch timeline: a warm process is now handling the job (`launched_at`).
    async fn launch(&self, proc: &Executor, job: &HostRef) -> Result<(), HostError> {
        proc.host.launch_job(&self.context, job.clone(), lk_agents_abi::unix_now()).await
    }

    async fn close(&self, proc: &Executor) -> Result<(), HostError> {
        proc.host.close(&self.context).await
    }

    fn failed(&self, proc: &Executor, message: &'static str, error: HostError) {
        queue_diagnostic(proc.host.clone(), Level::Error, message, vec![], Some(error));
    }

    async fn wait_exit(&self, proc: &Executor) -> Result<(), HostError> {
        let supervisor = proc
            .supervisor
            .get()
            .ok_or_else(|| HostError::new("ready process has no native supervisor"))?;
        match supervisor.wait_exit().await {
            true => Ok(()),
            false => Err(HostError::new("ready process finished without an OS exit observation")),
        }
    }

    async fn join(&self, proc: &Executor) -> Result<PoolJoinOutcome, HostError> {
        let reaped = match proc.supervisor.get() {
            Some(supervisor) => supervisor.wait_cleaned().await,
            None => proc.host.join(&self.context).await?,
        };
        Ok(if reaped { PoolJoinOutcome::Reaped } else { PoolJoinOutcome::NotLaunched })
    }

    fn emit(&self, event: PoolEvent, proc: &Executor) {
        self.host.emit(event.as_str().into(), proc.host.clone());
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
        initialize_timeout: f64,
        close_timeout: f64,
        context: CallContext,
    ) -> Result<Self, WorkerError> {
        let opts = PoolOptions {
            num_idle_processes,
            initialize_timeout: secs(initialize_timeout),
            close_timeout: secs(close_timeout),
        };
        let executors = HostExecutors { host, context };
        Ok(Self {
            pool: ProcPool::new(Arc::new(executors), opts),
            spawner: TokioSpawner::new((env.runtime)().map_err(WorkerError::Failed)?),
        })
    }

    fn processes(&self) -> Vec<Arc<dyn JobExecutor>> {
        self.pool.processes().into_iter().map(|proc| proc.host).collect()
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

    async fn launch_job(&self, job_id: String, info: HostRef) -> Result<(), WorkerError> {
        let pool = self.pool.clone();
        self.spawner.run(async move { pool.launch_job(&job_id, info).await }).await.map_err(
            |error| match error {
                LaunchError::Admission(reason) => WorkerError::Failed(reason.to_string()),
                LaunchError::AcquireExhausted { attempts } => WorkerError::Failed(format!(
                    "no process became available after {attempts} attempts"
                )),
                LaunchError::NoProcess => {
                    WorkerError::Failed("no process became available after 3 attempts".into())
                }
                LaunchError::Launch(error) => WorkerError::Host(error),
            },
        )
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
