//! The worker's warm job processes as the host drives them (`_core.ProcPool`): the pool keeps
//! them warm and hands each job the next ready one; the host makes the executors.

use std::any::Any;
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

use lk_agents_abi::HostRef;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::invoke::CallContext;
use lk_agents_abi::worker::ipc_host::{PoolHost, ProcessHost, WorkerError, WorkerProcPool};

use super::owner::{ProcessOwner, queue_diagnostic};
use super::pool::{
    LaunchError, PoolEvent, PoolHost as Allocation, PoolJoinOutcome, PoolOptions, ProcPool,
};
use super::{Level, ProcSupervisor, secs};
use crate::env::Env;
use crate::io::WorkerIo;

/// One executor and, once started, its native supervisor.
#[derive(Clone)]
pub struct Executor {
    host: Arc<dyn ProcessHost>,
    supervisor: Arc<OnceLock<ProcSupervisor>>,
}

/// Compared by identity.
impl PartialEq for Executor {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::addr_eq(Arc::as_ptr(&self.host), Arc::as_ptr(&other.host))
    }
}

/// The host's pool and the context its executors' steps run in (the pool's constructor's).
pub struct Executors<E> {
    host: Arc<dyn PoolHost>,
    context: CallContext,
    env: PhantomData<fn() -> E>,
}

#[async_trait::async_trait]
impl<E: Env> Allocation for Executors<E> {
    type Proc = Executor;
    type Job = HostRef;
    type Error = HostError;

    async fn create(&self) -> Result<Executor, HostError> {
        let host = self.host.create().await?;
        Ok(Executor { host, supervisor: Arc::new(OnceLock::new()) })
    }

    async fn start(&self, proc: &Executor) -> Result<(), HostError> {
        let started: Arc<dyn Any + Send + Sync> = proc.host.start(&self.context).await?;
        let owner = started
            .downcast::<ProcessOwner<E>>()
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

    fn close_failed(&self, proc: &Executor, error: HostError) {
        queue_diagnostic(
            proc.host.clone(),
            Level::Error,
            "error closing process",
            vec![],
            Some(error),
        );
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

    fn exit_failed(&self, proc: &Executor, error: HostError) {
        let message = "error observing process exit";
        queue_diagnostic(proc.host.clone(), Level::Error, message, vec![], Some(error));
    }

    async fn join(&self, proc: &Executor) -> Result<PoolJoinOutcome, HostError> {
        let reaped = match proc.supervisor.get() {
            Some(supervisor) => supervisor.wait_cleaned().await,
            None => proc.host.join(&self.context).await?,
        };
        Ok(if reaped { PoolJoinOutcome::Reaped } else { PoolJoinOutcome::NotLaunched })
    }

    fn join_failed(&self, proc: &Executor, error: HostError) {
        queue_diagnostic(
            proc.host.clone(),
            Level::Error,
            "error joining process",
            vec![],
            Some(error),
        );
    }

    fn emit(&self, event: PoolEvent, proc: &Executor) {
        self.host.emit(event.as_str().into(), proc.host.clone());
    }

    fn init_failed(&self, proc: &Executor, error: HostError) {
        let message = "error initializing process";
        queue_diagnostic(proc.host.clone(), Level::Error, message, vec![], Some(error));
    }
}

pub struct ProcessPool<E: Env> {
    pool: ProcPool<Executors<E>>,
    io: WorkerIo,
}

impl<E: Env> WorkerProcPool for ProcessPool<E> {
    fn new(
        host: Arc<dyn PoolHost>,
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
        let executors = Executors { host, context, env: PhantomData };
        Ok(Self {
            pool: ProcPool::new(executors, opts),
            io: E::worker_io().map_err(WorkerError::Failed)?,
        })
    }

    fn processes(&self) -> Vec<Arc<dyn ProcessHost>> {
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
        let _runtime = self.io.runtime_handle().enter();
        self.pool.set_target_idle(n);
    }

    async fn start(&self) -> Result<(), WorkerError> {
        let pool = self.pool.clone();
        self.io
            .run(async move { pool.start().await })
            .await
            .map_err(|error| WorkerError::Failed(error.to_string()))
    }

    async fn launch_job(&self, job_id: String, info: HostRef) -> Result<(), WorkerError> {
        let pool = self.pool.clone();
        self.io.run(async move { pool.launch_job(&job_id, info).await }).await.map_err(|error| {
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
        self.io.run(async move { pool.join_jobs().await }).await;
    }

    async fn aclose(&self) {
        let pool = self.pool.clone();
        self.io.run(async move { pool.aclose().await }).await;
    }
}
