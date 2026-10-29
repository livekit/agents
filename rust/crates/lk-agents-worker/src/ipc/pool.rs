//! The warm process pool: host execution and cleanup ownership around [`Allocation`].
//! Admission and process eligibility are changed under the same lock as owned resources;
//! host operations run after that lock is released.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::FutureExt;
use lk_agents_abi::{OwnedTask, SpawnerExt};
use lk_agents_net::task::TokioSpawner;
use tokio::sync::{Notify, watch};

use lk_agents_abi::HostRef;
use lk_agents_abi::errors::HostError;

use super::lock;
use super::process_pool::Executor;

mod allocation;
use allocation::{
    Allocation, PoolEffect, PoolFailure, PoolInput, PoolLimits, ProcessPhase, ReservationId,
    ReservationState,
};
pub use allocation::{
    MAX_ACQUIRE_ATTEMPTS, MAX_LAUNCH_ATTEMPTS, PoolAdmissionError, PoolCleanupObservation,
    PoolJoinOutcome, ProcessId, REFILL_INTERVAL,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolEvent {
    Created,
    Started,
    Ready,
    Closed,
    JobLaunched,
    CleanupIncomplete,
    JoinFailed,
    ExitFailed,
}

impl PoolEvent {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Created => "process_created",
            Self::Started => "process_started",
            Self::Ready => "process_ready",
            Self::Closed => "process_closed",
            Self::JobLaunched => "process_job_launched",
            Self::CleanupIncomplete => "process_cleanup_incomplete",
            Self::JoinFailed => "process_join_failed",
            Self::ExitFailed => "process_exit_failed",
        }
    }
}

/// The pool's steps on its executors: the host's in production, a fake in tests.
#[async_trait::async_trait]
pub trait Executors: Send + Sync + 'static {
    async fn create(&self) -> Result<Executor, HostError>;
    async fn start(&self, proc: &Executor) -> Result<(), HostError>;
    async fn initialize(&self, proc: &Executor) -> Result<(), HostError>;
    async fn launch(&self, proc: &Executor, job: &HostRef) -> Result<(), HostError>;
    async fn close(&self, proc: &Executor) -> Result<(), HostError>;
    /// A step that failed on `proc`, as the host's diagnostic `message`.
    fn failed(&self, proc: &Executor, message: &'static str, error: HostError);
    fn cleanup_incomplete(&self, observation: PoolCleanupObservation) {
        crate::diagnostics::emit(
            log::Level::Error,
            format!("process pool cleanup incomplete: {observation:?}"),
        );
    }
    async fn wait_exit(&self, proc: &Executor) -> Result<(), HostError>;
    async fn join(&self, proc: &Executor) -> Result<PoolJoinOutcome, HostError>;
    fn emit(&self, event: PoolEvent, proc: &Executor);
}

#[derive(Clone, Copy, Debug)]
pub struct PoolOptions {
    pub num_idle_processes: usize,
    pub initialize_timeout: Duration,
    pub close_timeout: Duration,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Readiness {
    Pending,
    Ready,
    Closed,
}

struct Inner {
    host: Arc<dyn Executors>,
    opts: PoolOptions,
    state: Mutex<Shared>,
    changed: Notify,
    idle_ready: watch::Sender<Readiness>,
    closed: watch::Sender<bool>,
    close_ready: watch::Sender<bool>,
}

#[derive(Debug)]
pub enum LaunchError {
    NoProcess,
    AcquireExhausted { attempts: u32 },
    Launch(HostError),
    Admission(PoolAdmissionError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolStartError {
    Closed,
    WarmupTimedOut { ready: usize, required: usize },
}

impl std::fmt::Display for PoolStartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Closed => f.write_str("process pool closed before becoming ready"),
            Self::WarmupTimedOut { ready, required } => {
                write!(f, "process pool warmup timed out: {ready} of {required} processes ready")
            }
        }
    }
}

pub struct ProcPool(Arc<Inner>);

struct Waiting {
    pool: ProcPool,
    request: ReservationId,
}

impl Drop for Waiting {
    fn drop(&mut self) {
        self.pool.handle(PoolInput::Cancel(self.request));
    }
}

impl Clone for ProcPool {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl ProcPool {
    pub fn new(host: Arc<dyn Executors>, opts: PoolOptions) -> Self {
        // CPU discovery is runtime configuration; constructing a pool performs no filesystem I/O.
        let limits = PoolLimits {
            idle_processes: opts.num_idle_processes,
            concurrent_initializations: 4,
            pending_jobs: 64,
            cleanup_timeout: opts.close_timeout * 3 + super::supervisor::KILL_GRACE,
        };
        Self(Arc::new(Inner {
            host,
            opts,
            state: Mutex::new(Shared {
                allocation: Allocation::new(limits),
                executors: BTreeMap::new(),
                tasks: Vec::new(),
                cleanup: None,
                next_request: 0,
                now: Duration::ZERO,
            }),
            changed: Notify::new(),
            idle_ready: watch::channel(Readiness::Pending).0,
            closed: watch::channel(false).0,
            close_ready: watch::channel(false).0,
        }))
    }

    pub fn processes(&self) -> Vec<Executor> {
        lock(&self.0.state).executors.values().cloned().collect()
    }

    pub fn jobs_waiting(&self) -> usize {
        lock(&self.0.state).allocation.waiting()
    }

    pub fn target_idle(&self) -> usize {
        lock(&self.0.state).allocation.target_idle()
    }

    pub fn set_target_idle(&self, n: usize) {
        self.handle(PoolInput::TargetIdle(n));
    }

    fn track(&self, task: OwnedTask) {
        let completed = {
            let mut state = lock(&self.0.state);
            let (completed, active) = std::mem::take(&mut state.tasks)
                .into_iter()
                .partition::<Vec<_>, _>(|task| task.is_finished());
            state.tasks = active;
            state.tasks.push(task);
            completed
        };
        for task in completed {
            if let Some(Err(error)) = task.now_or_never() {
                crate::diagnostics::emit(
                    log::Level::Error,
                    format!("process pool task failed: {error}"),
                );
            }
        }
    }

    fn handle(&self, input: PoolInput) {
        let effects = {
            let mut state = lock(&self.0.state);
            let now = state.now;
            let effects = state.allocation.handle(input, now);
            if state.allocation.is_closed() {
                self.0.closed.send_replace(true);
                self.0.idle_ready.send_replace(Readiness::Closed);
            } else if state.allocation.ready() >= self.0.opts.num_idle_processes {
                self.0.idle_ready.send_replace(Readiness::Ready);
            }
            effects
        };
        self.0.changed.notify_waiters();
        self.spawn(effects);
    }

    pub async fn start(&self) -> Result<(), PoolStartError> {
        let (start, effects) = {
            let mut state = lock(&self.0.state);
            if state.allocation.is_closed() {
                return Err(PoolStartError::Closed);
            }
            let start = !state.allocation.is_started();
            let now = state.now;
            (start, state.allocation.handle(PoolInput::Start, now))
        };
        self.spawn(effects);
        self.handle(PoolInput::Wake);
        if start {
            let pool = self.clone();
            self.track(TokioSpawner::current().spawn(async move { pool.refill().await }));
        }
        let mut ready = self.0.idle_ready.subscribe();
        let wait = self.0.opts.initialize_timeout + Duration::from_secs(2);
        match tokio::time::timeout(wait, ready.wait_for(|state| *state != Readiness::Pending)).await
        {
            Ok(Ok(readiness)) if *readiness == Readiness::Ready => Ok(()),
            Ok(_) => Err(PoolStartError::Closed),
            Err(_) => Err(PoolStartError::WarmupTimedOut {
                ready: lock(&self.0.state).allocation.ready(),
                required: self.0.opts.num_idle_processes,
            }),
        }
    }

    fn closed(&self) -> bool {
        *self.0.closed.borrow()
    }

    async fn refill(&self) {
        let origin = tokio::time::Instant::now();
        loop {
            if self.closed() && !lock(&self.0.state).allocation.has_processes() {
                break;
            }
            let delay = {
                let mut state = lock(&self.0.state);
                state.now = tokio::time::Instant::now().duration_since(origin);
                state
                    .allocation
                    .wake_at()
                    .map_or(REFILL_INTERVAL, |deadline| deadline.saturating_sub(state.now))
            };
            self.handle(PoolInput::Wake);
            tokio::time::sleep(delay).await;
        }
    }

    fn spawn(&self, effects: Vec<PoolEffect>) {
        for effect in effects {
            match effect {
                PoolEffect::Create(id) => {
                    let pool = self.clone();
                    self.track(
                        TokioSpawner::current().spawn(async move { pool.spawn_proc(id).await }),
                    );
                }
                PoolEffect::Close(id) => {
                    let process = lock(&self.0.state).executors.get(&id).cloned();
                    if let Some(process) = process {
                        let pool = self.clone();
                        self.track(TokioSpawner::current().spawn(async move {
                            if let Err(error) = pool.0.host.close(&process).await {
                                pool.0.host.emit(PoolEvent::CleanupIncomplete, &process);
                                pool.0.host.failed(&process, "error closing process", error);
                            }
                            pool.handle(PoolInput::Closed(id));
                        }));
                    }
                }
                PoolEffect::WatchExit(id) => {
                    let process = lock(&self.0.state).executors.get(&id).cloned();
                    if let Some(process) = process {
                        let pool = self.clone();
                        self.track(TokioSpawner::current().spawn(async move {
                            match pool.0.host.wait_exit(&process).await {
                                Ok(()) => pool.handle(PoolInput::Exited(id)),
                                Err(error) => {
                                    pool.0.host.emit(PoolEvent::ExitFailed, &process);
                                    pool.0.host.failed(
                                        &process,
                                        "error observing process exit",
                                        error,
                                    );
                                    pool.handle(PoolInput::ExitFailed(id));
                                }
                            }
                        }));
                    }
                }
                PoolEffect::Join(id) => {
                    let process = lock(&self.0.state).executors.get(&id).cloned();
                    if let Some(process) = process {
                        let pool = self.clone();
                        self.track(TokioSpawner::current().spawn(async move {
                            match pool.0.host.join(&process).await {
                                Ok(outcome) => {
                                    pool.handle(PoolInput::Joined { process: id, outcome })
                                }
                                Err(error) => {
                                    pool.0.host.emit(PoolEvent::JoinFailed, &process);
                                    pool.0.host.failed(&process, "error joining process", error);
                                    pool.handle(PoolInput::JoinFailed(id));
                                }
                            }
                        }));
                    }
                }
                PoolEffect::Released { process: id, outcome: _ } => {
                    let process = lock(&self.0.state).executors.remove(&id);
                    if let Some(process) = process {
                        self.0.host.emit(PoolEvent::Closed, &process);
                    }
                }
                PoolEffect::Refused { .. } => {}
                PoolEffect::CleanupIncomplete(observation) => {
                    self.0.host.cleanup_incomplete(observation);
                    self.0.close_ready.send_replace(true);
                }
                PoolEffect::CloseComplete => {
                    self.0.close_ready.send_replace(true);
                }
            }
        }
    }

    async fn spawn_proc(&self, id: ProcessId) {
        let process = match self.0.host.create().await {
            Ok(process) => process,
            Err(error) => {
                crate::diagnostics::emit(
                    log::Level::Error,
                    format!("error creating process: {error}"),
                );
                self.handle(PoolInput::SpawnFailed(id));
                self.handle(PoolInput::Cleaned(id));
                return;
            }
        };
        lock(&self.0.state).executors.insert(id, process.clone());
        self.handle(PoolInput::Created(id));
        if lock(&self.0.state).allocation.phase(id) != ProcessPhase::Initializing {
            return;
        }
        let mut closed = self.0.closed.subscribe();
        self.0.host.emit(PoolEvent::Created, &process);
        let steps = async {
            self.0.host.start(&process).await?;
            self.0.host.emit(PoolEvent::Started, &process);
            self.0.host.initialize(&process).await?;
            Ok::<_, HostError>(())
        };
        let initialized = tokio::select! {
            biased;
            _ = closed.wait_for(|closed| *closed) => false,
            result = steps => match result {
                Ok(()) => true,
                Err(error) => {
                    self.0.host.failed(&process, "error initializing process", error);
                    false
                }
            },
        };
        if !initialized {
            self.handle(PoolInput::SpawnFailed(id));
            return;
        }
        self.0.host.emit(PoolEvent::Ready, &process);
        self.handle(PoolInput::Ready(id));
    }

    async fn acquire(&self, request: ReservationId) -> Result<(ProcessId, Executor), PoolFailure> {
        loop {
            let changed = self.0.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            {
                let state = lock(&self.0.state);
                match state.allocation.reservation(request) {
                    Some(ReservationState::Reserved(id)) => {
                        if let Some(process) = state.executors.get(&id) {
                            return Ok((id, process.clone()));
                        }
                    }
                    Some(ReservationState::Failed(error)) => return Err(error),
                    None => return Err(PoolFailure::Closed),
                    _ => {}
                }
            }
            changed.await;
        }
    }

    pub async fn launch_job(&self, job_id: &str, job: HostRef) -> Result<(), LaunchError> {
        let (request, effects) = {
            let mut state = lock(&self.0.state);
            let request = ReservationId(state.next_request);
            state.next_request += 1;
            let now = state.now;
            let effects = state
                .allocation
                .handle(PoolInput::Admit { request, job_id: job_id.to_owned() }, now);
            (request, effects)
        };
        if let Some(reason) = effects.iter().find_map(|effect| match effect {
            PoolEffect::Refused { reason, .. } => Some(*reason),
            _ => None,
        }) {
            return Err(LaunchError::Admission(reason));
        }
        let _waiting = Waiting { pool: self.clone(), request };
        self.spawn(effects);
        self.0.changed.notify_waiters();
        let mut last_error = None;
        loop {
            let (id, process) = self.acquire(request).await.map_err(|error| match error {
                PoolFailure::AcquireExhausted { attempts } => {
                    LaunchError::AcquireExhausted { attempts }
                }
                PoolFailure::LaunchExhausted { .. } => {
                    last_error.take().map_or(LaunchError::NoProcess, LaunchError::Launch)
                }
                PoolFailure::Closed => LaunchError::Admission(PoolAdmissionError::Closed),
            })?;
            match self.0.host.launch(&process, &job).await {
                Ok(()) => {
                    self.handle(PoolInput::LaunchSucceeded { request, process: id });
                    if lock(&self.0.state).allocation.reservation(request)
                        == Some(ReservationState::Completed)
                    {
                        self.0.host.emit(PoolEvent::JobLaunched, &process);
                        self.handle(PoolInput::Forget(request));
                        return Ok(());
                    }
                }
                Err(error) => {
                    last_error = Some(error);
                    self.handle(PoolInput::LaunchFailed { request, process: id });
                }
            }
        }
    }

    pub async fn join_jobs(&self) {
        self.handle(PoolInput::Drain);
        loop {
            let changed = self.0.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if !lock(&self.0.state).allocation.has_jobs() {
                return;
            }
            changed.await;
        }
    }

    pub async fn aclose(&self) {
        let (first, effects, deadline) = {
            let mut state = lock(&self.0.state);
            let first = !state.allocation.is_closed();
            let now = state.now;
            let effects = state.allocation.handle(PoolInput::Close, now);
            (first, effects, state.allocation.close_deadline())
        };
        self.spawn(effects);
        self.handle(PoolInput::Wake);
        if first {
            let pool = self.clone();
            // A caller's cancellation cannot discard the join obligations: the pool owns them.
            let cleanup = TokioSpawner::current().spawn(async move {
                let cleanup = async {
                    loop {
                        let changed = pool.0.changed.notified();
                        tokio::pin!(changed);
                        changed.as_mut().enable();
                        let tasks = std::mem::take(&mut lock(&pool.0.state).tasks);
                        if tasks.is_empty() {
                            if !lock(&pool.0.state).allocation.has_processes() {
                                break;
                            }
                            changed.await;
                            continue;
                        }
                        for task in tasks {
                            if let Err(error) = task.await {
                                crate::diagnostics::emit(
                                    log::Level::Error,
                                    format!("process pool task failed: {error}"),
                                );
                            }
                        }
                    }
                };
                tokio::pin!(cleanup);
                // Refill can finish before other owned tasks; keep the cleanup deadline serviceable.
                let deadline = deadline.expect("first close owns a deadline");
                let timeout = deadline.saturating_sub(lock(&pool.0.state).now);
                if tokio::time::timeout(timeout, &mut cleanup).await.is_err() {
                    {
                        let mut state = lock(&pool.0.state);
                        state.now = state.now.max(deadline);
                    }
                    pool.handle(PoolInput::Wake);
                    cleanup.await;
                }
                pool.handle(PoolInput::CleanupComplete);
            });
            lock(&self.0.state).cleanup = Some(cleanup);
        }
        let _ = self.0.close_ready.subscribe().wait_for(|done| *done).await;
    }
}

/// The allocation and the resources the pool owns beside it, changed under one lock.
struct Shared {
    allocation: Allocation,
    executors: BTreeMap<ProcessId, Executor>,
    tasks: Vec<OwnedTask>,
    /// The close's join of `tasks`, started by the first close.
    cleanup: Option<OwnedTask>,
    next_request: u64,
    now: Duration,
}

#[cfg(test)]
mod tests;
