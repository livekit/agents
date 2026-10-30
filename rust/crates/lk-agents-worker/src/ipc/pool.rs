//! The warm process pool: admission, process eligibility, joins and the close deadline, with the
//! host's steps run as tasks the pool owns. Every change happens under one lock; a host step only
//! reports back into it.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use futures_util::FutureExt;
use lk_agents_abi::{OwnedTask, SpawnerExt};
use lk_agents_net::task::TokioSpawner;
use parking_lot::Mutex;
use tokio::sync::{Notify, watch};
use tokio::time::Instant;

use lk_agents_abi::HostRef;
use lk_agents_abi::errors::HostError;

use super::process_pool::Executor;

pub const MAX_ACQUIRE_ATTEMPTS: u32 = 3;
pub const MAX_LAUNCH_ATTEMPTS: u32 = 3;
/// How often the pool tops the warm processes up, and how long it waits before retrying a failed
/// spawn, join or exit observation.
pub const REFILL_INTERVAL: Duration = Duration::from_millis(100);
const CONCURRENT_INITIALIZATIONS: usize = 4;
const PENDING_JOBS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProcessId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ReservationId(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolJoinOutcome {
    Reaped,
    NotLaunched,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolAdmissionError {
    NotStarted,
    Draining,
    Closed,
    Full,
    DuplicateJob,
}

impl std::fmt::Display for PoolAdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NotStarted => "process pool has not started",
            Self::Draining => "process pool is draining",
            Self::Closed => "process pool is closed",
            Self::Full => "process pool pending job limit reached",
            Self::DuplicateJob => "job is already admitted to the process pool",
        })
    }
}

/// The processes still owned when the close deadline passed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolCleanupObservation {
    pub processes: Vec<ProcessId>,
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Creating,
    Initializing,
    Ready,
    Reserved,
    Running,
    Retiring,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Failure {
    AcquireExhausted { attempts: u32 },
    LaunchExhausted,
    Closed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReservationState {
    Waiting,
    Reserved(ProcessId),
    Failed(Failure),
}

struct Process {
    executor: Option<Executor>,
    phase: Phase,
    /// The reservation this process is created, initialized or reserved for.
    request: Option<ReservationId>,
    /// The job it was reserved for, released when it retires.
    job: Option<(String, ReservationId)>,
    /// The host's close returned.
    closed: bool,
    /// The reaping receipt, once a join returned one.
    receipt: Option<PoolJoinOutcome>,
    /// A join is running or scheduled.
    joining: bool,
    /// An exit observation is running or scheduled.
    watching: bool,
}

struct Reservation {
    job_id: String,
    state: ReservationState,
    acquire_attempts: u32,
    launch_attempts: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Readiness {
    Pending,
    Ready,
    Closed,
}

/// Everything the pool decides on and owns, changed under one lock.
struct State {
    target_idle: usize,
    started: bool,
    draining: bool,
    closed: bool,
    next_process: u64,
    next_request: u64,
    processes: BTreeMap<ProcessId, Process>,
    initializing: BTreeSet<ProcessId>,
    ready: VecDeque<ProcessId>,
    pending: VecDeque<ReservationId>,
    requests: BTreeMap<ReservationId, Reservation>,
    jobs: BTreeMap<String, ReservationId>,
    /// A spawn failed: no warm process is added before this.
    refill_paused_until: Option<Instant>,
    tasks: Vec<OwnedTask>,
    /// The close's join of `tasks`, started by the first close.
    cleanup: Option<OwnedTask>,
    /// The processes to report closed once the lock is released: a host call takes the host's
    /// own lock (Python's GIL), which a host thread reading the pool holds while it waits on
    /// this one.
    released: Vec<Executor>,
}

impl State {
    fn reservation(&self, id: ReservationId) -> Option<ReservationState> {
        self.requests.get(&id).map(|request| request.state)
    }

    fn executor(&self, id: ProcessId) -> Option<Executor> {
        self.processes.get(&id).and_then(|process| process.executor.clone())
    }

    fn has_jobs(&self) -> bool {
        !self.pending.is_empty()
            || self.processes.values().any(|p| matches!(p.phase, Phase::Reserved | Phase::Running))
    }

    fn release_job(&mut self, job: &str, request: ReservationId) {
        if self.jobs.get(job) == Some(&request) {
            self.jobs.remove(job);
        }
    }
}

struct Inner {
    host: Arc<dyn Executors>,
    opts: PoolOptions,
    cleanup_timeout: Duration,
    state: Mutex<State>,
    changed: Notify,
    idle_ready: watch::Sender<Readiness>,
    closed: watch::Sender<bool>,
    close_ready: watch::Sender<bool>,
}

pub struct ProcPool(Arc<Inner>);

impl Clone for ProcPool {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

/// A launch's reservation, cancelled when the launch is dropped before it completed.
struct Waiting {
    pool: ProcPool,
    request: ReservationId,
}

impl Drop for Waiting {
    fn drop(&mut self) {
        self.pool.update(|pool, st| pool.cancel(st, self.request));
    }
}

impl ProcPool {
    pub fn new(host: Arc<dyn Executors>, opts: PoolOptions) -> Self {
        // CPU discovery is runtime configuration; constructing a pool performs no filesystem I/O.
        Self(Arc::new(Inner {
            host,
            opts,
            cleanup_timeout: opts.close_timeout * 3,
            state: Mutex::new(State {
                target_idle: opts.num_idle_processes,
                started: false,
                draining: false,
                closed: false,
                next_process: 0,
                next_request: 0,
                processes: BTreeMap::new(),
                initializing: BTreeSet::new(),
                ready: VecDeque::new(),
                pending: VecDeque::new(),
                requests: BTreeMap::new(),
                jobs: BTreeMap::new(),
                refill_paused_until: None,
                tasks: Vec::new(),
                cleanup: None,
                released: Vec::new(),
            }),
            changed: Notify::new(),
            idle_ready: watch::channel(Readiness::Pending).0,
            closed: watch::channel(false).0,
            close_ready: watch::channel(false).0,
        }))
    }

    pub fn processes(&self) -> Vec<Executor> {
        self.0.state.lock().processes.values().filter_map(|p| p.executor.clone()).collect()
    }

    pub fn jobs_waiting(&self) -> usize {
        self.0.state.lock().pending.len()
    }

    pub fn target_idle(&self) -> usize {
        self.0.state.lock().target_idle
    }

    pub fn set_target_idle(&self, n: usize) {
        let idle = self.0.opts.num_idle_processes;
        self.update(|_, st| st.target_idle = n.min(idle));
    }

    /// Changes the state, then allocates, publishes readiness, tells the host what was released
    /// and wakes the waiters. No host call is made under the lock.
    fn update<R>(&self, f: impl FnOnce(&Self, &mut State) -> R) -> R {
        let (result, released) = {
            let mut st = self.0.state.lock();
            if st.refill_paused_until.is_some_and(|until| Instant::now() >= until) {
                st.refill_paused_until = None;
            }
            let result = f(self, &mut st);
            self.allocate(&mut st);
            if st.closed {
                self.0.closed.send_replace(true);
                self.0.idle_ready.send_replace(Readiness::Closed);
            } else if st.ready.len() >= self.0.opts.num_idle_processes {
                self.0.idle_ready.send_replace(Readiness::Ready);
            }
            (result, std::mem::take(&mut st.released))
        };
        for executor in released {
            self.0.host.emit(PoolEvent::Closed, &executor);
        }
        self.0.changed.notify_waiters();
        result
    }

    /// Runs `step` as a task the pool owns until its close joined it.
    fn own(&self, st: &mut State, step: impl Future<Output = ()> + Send + 'static) {
        let (finished, active) = std::mem::take(&mut st.tasks)
            .into_iter()
            .partition::<Vec<_>, _>(|task| task.is_finished());
        st.tasks = active;
        st.tasks.push(TokioSpawner::current().spawn(step));
        for task in finished {
            if let Some(Err(error)) = task.now_or_never() {
                // A queued log line, not a host call.
                crate::diagnostics::emit(
                    log::Level::Error,
                    format!("process pool task failed: {error}"),
                );
            }
        }
    }

    pub async fn start(&self) -> Result<(), PoolStartError> {
        let first = {
            let st = self.0.state.lock();
            if st.closed {
                return Err(PoolStartError::Closed);
            }
            !st.started
        };
        self.update(|_, st| st.started = true);
        if first {
            let pool = self.clone();
            self.update(|this, st| this.own(st, async move { pool.refill().await }));
        }
        let mut ready = self.0.idle_ready.subscribe();
        let wait = self.0.opts.initialize_timeout + Duration::from_secs(2);
        match tokio::time::timeout(wait, ready.wait_for(|state| *state != Readiness::Pending)).await
        {
            Ok(Ok(readiness)) if *readiness == Readiness::Ready => Ok(()),
            Ok(_) => Err(PoolStartError::Closed),
            Err(_) => Err(PoolStartError::WarmupTimedOut {
                ready: self.0.state.lock().ready.len(),
                required: self.0.opts.num_idle_processes,
            }),
        }
    }

    /// Tops the warm processes up until the pool closed and released every process.
    async fn refill(&self) {
        loop {
            {
                let st = self.0.state.lock();
                if st.closed && st.processes.is_empty() {
                    break;
                }
            }
            self.update(|_, _| {});
            tokio::time::sleep(REFILL_INTERVAL).await;
        }
    }

    /// Hands ready processes to waiting reservations in order, creates processes for those still
    /// waiting, then tops the warm ones up to the target.
    fn allocate(&self, st: &mut State) {
        if !st.started || st.closed {
            return;
        }
        let requests = &st.requests;
        st.pending.retain(|id| {
            requests.get(id).is_some_and(|request| request.state == ReservationState::Waiting)
        });
        while !st.ready.is_empty() && !st.pending.is_empty() {
            let first = *st.pending.front().unwrap();
            if st.processes.values().any(|p| p.request == Some(first) && p.phase == Phase::Retiring)
            {
                // The head reservation must finish cleanup before any replacement is assigned.
                // Spawning for later requests here would repeatedly replace their ready processes
                // while the head remains blocked, starving the runtime and growing the pool.
                return;
            }
            let process_id = st.ready.pop_front().unwrap();
            let request_id = st.pending.pop_front().unwrap();
            let request = st.requests.get_mut(&request_id).unwrap();
            request.state = ReservationState::Reserved(process_id);
            request.acquire_attempts = 0;
            let job = (request.job_id.clone(), request_id);
            for (other, process) in &mut st.processes {
                if *other != process_id && process.request == Some(request_id) {
                    process.request = None;
                }
            }
            let process = st.processes.get_mut(&process_id).unwrap();
            process.phase = Phase::Reserved;
            process.request = Some(request_id);
            process.job = Some(job);
        }
        for request_id in st.pending.iter().copied().collect::<Vec<_>>() {
            if st.processes.values().any(|p| p.request == Some(request_id)) {
                continue;
            }
            let initializing = st.processes.iter().find_map(|(id, p)| {
                (st.initializing.contains(id) && p.request.is_none() && p.phase != Phase::Retiring)
                    .then_some(*id)
            });
            let process_id = match initializing {
                Some(id) => id,
                None if st.initializing.len() < CONCURRENT_INITIALIZATIONS => self.create(st),
                None => break,
            };
            st.requests.get_mut(&request_id).unwrap().acquire_attempts += 1;
            st.processes.get_mut(&process_id).unwrap().request = Some(request_id);
        }
        if !st.draining && st.refill_paused_until.is_none() {
            let unassigned = st
                .processes
                .values()
                .filter(|p| {
                    p.request.is_none()
                        && matches!(p.phase, Phase::Creating | Phase::Initializing | Phase::Ready)
                })
                .count();
            for _ in unassigned..st.target_idle {
                if st.initializing.len() >= CONCURRENT_INITIALIZATIONS {
                    break;
                }
                self.create(st);
            }
        }
    }

    fn create(&self, st: &mut State) -> ProcessId {
        let id = ProcessId(st.next_process);
        st.next_process += 1;
        st.processes.insert(
            id,
            Process {
                executor: None,
                phase: Phase::Creating,
                request: None,
                job: None,
                closed: false,
                receipt: None,
                joining: false,
                watching: false,
            },
        );
        st.initializing.insert(id);
        let pool = self.clone();
        self.own(st, async move { pool.spawn_process(id).await });
        id
    }

    async fn spawn_process(&self, id: ProcessId) {
        let process = match self.0.host.create().await {
            Ok(process) => process,
            Err(error) => {
                crate::diagnostics::emit(
                    log::Level::Error,
                    format!("error creating process: {error}"),
                );
                self.update(|pool, st| {
                    pool.spawn_failed(st, id);
                    // Never created: nothing to close or join.
                    st.initializing.remove(&id);
                    st.processes.remove(&id);
                });
                return;
            }
        };
        let initializing = self.update(|pool, st| {
            let Some(entry) = st.processes.get_mut(&id) else { return false };
            entry.executor = Some(process.clone());
            match entry.phase {
                Phase::Creating => {
                    entry.phase = Phase::Initializing;
                    true
                }
                Phase::Retiring => {
                    pool.close_process(st, id);
                    false
                }
                _ => false,
            }
        });
        if !initializing {
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
            self.update(|pool, st| pool.spawn_failed(st, id));
            return;
        }
        self.0.host.emit(PoolEvent::Ready, &process);
        self.update(|pool, st| {
            let Some(entry) = st.processes.get_mut(&id) else { return };
            if entry.phase != Phase::Initializing {
                return;
            }
            entry.phase = Phase::Ready;
            entry.request = None;
            entry.joining = true;
            entry.watching = true;
            st.initializing.remove(&id);
            st.ready.push_back(id);
            pool.join(st, id, Duration::ZERO);
            pool.watch_exit(st, id, Duration::ZERO);
        });
    }

    fn spawn_failed(&self, st: &mut State, id: ProcessId) {
        st.refill_paused_until = Some(Instant::now() + REFILL_INTERVAL);
        self.retire(st, id);
    }

    /// Takes the process out of service and closes it; its reservation waits for another one
    /// unless its attempts are spent.
    fn retire(&self, st: &mut State, id: ProcessId) {
        let Some(process) = st.processes.get_mut(&id) else { return };
        if process.phase == Phase::Retiring {
            return;
        }
        let reserved = process.phase == Phase::Reserved;
        process.phase = Phase::Retiring;
        let request_id = process.request;
        st.ready.retain(|ready| *ready != id);
        let waiting = request_id.filter(|request_id| {
            st.reservation(*request_id).is_some_and(|state| {
                state == ReservationState::Waiting || state == ReservationState::Reserved(id)
            })
        });
        if let Some(request_id) = waiting {
            let request = st.requests.get_mut(&request_id).unwrap();
            if reserved {
                request.launch_attempts += 1;
            }
            let failure = if request.launch_attempts >= MAX_LAUNCH_ATTEMPTS {
                Some(Failure::LaunchExhausted)
            } else if request.acquire_attempts >= MAX_ACQUIRE_ATTEMPTS {
                Some(Failure::AcquireExhausted { attempts: request.acquire_attempts })
            } else {
                None
            };
            match failure {
                Some(failure) => {
                    request.state = ReservationState::Failed(failure);
                    let job_id = request.job_id.clone();
                    st.release_job(&job_id, request_id);
                }
                None => {
                    request.state = ReservationState::Waiting;
                    if !st.pending.contains(&request_id) {
                        st.pending.push_back(request_id);
                    }
                }
            }
        } else if let Some((job_id, request_id)) =
            st.processes.get_mut(&id).and_then(|process| process.job.take())
        {
            st.release_job(&job_id, request_id);
        }
        self.close_process(st, id);
    }

    /// The host's close of a created process; then its reaping receipt releases it.
    fn close_process(&self, st: &mut State, id: ProcessId) {
        let Some(executor) = st.executor(id) else { return };
        let pool = self.clone();
        self.own(st, async move {
            if let Err(error) = pool.0.host.close(&executor).await {
                pool.0.host.emit(PoolEvent::CleanupIncomplete, &executor);
                pool.0.host.failed(&executor, "error closing process", error);
            }
            pool.update(|pool, st| {
                let Some(process) = st.processes.get_mut(&id) else { return };
                process.closed = true;
                if process.receipt.is_some() {
                    pool.release(st, id);
                } else if !process.joining {
                    process.joining = true;
                    pool.join(st, id, Duration::ZERO);
                }
            });
        });
    }

    /// Waits for the process's reaping receipt, `after` a delay; a failed join is retried.
    fn join(&self, st: &mut State, id: ProcessId, after: Duration) {
        let Some(executor) = st.executor(id) else { return };
        let pool = self.clone();
        self.own(st, async move {
            tokio::time::sleep(after).await;
            match pool.0.host.join(&executor).await {
                Ok(outcome) => pool.update(|pool, st| {
                    let Some(process) = st.processes.get_mut(&id) else { return };
                    process.joining = false;
                    process.receipt = Some(outcome);
                    pool.retire(st, id);
                    if st.processes.get(&id).is_some_and(|process| process.closed) {
                        pool.release(st, id);
                    }
                }),
                Err(error) => {
                    pool.0.host.emit(PoolEvent::JoinFailed, &executor);
                    pool.0.host.failed(&executor, "error joining process", error);
                    pool.update(|pool, st| {
                        if st.processes.contains_key(&id) {
                            pool.retire(st, id);
                            pool.join(st, id, REFILL_INTERVAL);
                        }
                    });
                }
            }
        });
    }

    /// The process's OS exit, `after` a delay, takes it out of service; a failed observation is
    /// retried.
    fn watch_exit(&self, st: &mut State, id: ProcessId, after: Duration) {
        let Some(executor) = st.executor(id) else { return };
        let pool = self.clone();
        self.own(st, async move {
            tokio::time::sleep(after).await;
            match pool.0.host.wait_exit(&executor).await {
                Ok(()) => pool.update(|pool, st| {
                    if let Some(process) = st.processes.get_mut(&id) {
                        process.watching = false;
                        pool.retire(st, id);
                    }
                }),
                Err(error) => {
                    pool.0.host.emit(PoolEvent::ExitFailed, &executor);
                    pool.0.host.failed(&executor, "error observing process exit", error);
                    pool.update(|pool, st| {
                        if st.processes.contains_key(&id) {
                            pool.retire(st, id);
                            pool.watch_exit(st, id, REFILL_INTERVAL);
                        }
                    });
                }
            }
        });
    }

    fn release(&self, st: &mut State, id: ProcessId) {
        st.initializing.remove(&id);
        st.ready.retain(|ready| *ready != id);
        if let Some(process) = st.processes.remove(&id)
            && let Some(executor) = process.executor
        {
            st.released.push(executor);
        }
    }

    fn cancel(&self, st: &mut State, id: ReservationId) {
        let Some(request) = st.requests.remove(&id) else { return };
        st.pending.retain(|pending| *pending != id);
        if let ReservationState::Reserved(process) = request.state {
            self.retire(st, process);
        }
        st.release_job(&request.job_id, id);
        for process in st.processes.values_mut() {
            if process.request == Some(id) {
                process.request = None;
            }
        }
    }

    async fn acquire(&self, request: ReservationId) -> Result<(ProcessId, Executor), Failure> {
        loop {
            let changed = self.0.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            {
                let st = self.0.state.lock();
                match st.reservation(request) {
                    Some(ReservationState::Reserved(id)) => {
                        if let Some(executor) = st.executor(id) {
                            return Ok((id, executor));
                        }
                    }
                    Some(ReservationState::Failed(error)) => return Err(error),
                    None => return Err(Failure::Closed),
                    _ => {}
                }
            }
            changed.await;
        }
    }

    pub async fn launch_job(&self, job_id: &str, job: HostRef) -> Result<(), LaunchError> {
        let admitted = self.update(|_, st| {
            let request = ReservationId(st.next_request);
            st.next_request += 1;
            let refused = if st.closed {
                Some(PoolAdmissionError::Closed)
            } else if st.draining {
                Some(PoolAdmissionError::Draining)
            } else if !st.started {
                Some(PoolAdmissionError::NotStarted)
            } else if st.jobs.contains_key(job_id) {
                Some(PoolAdmissionError::DuplicateJob)
            } else if st.requests.len() >= PENDING_JOBS {
                Some(PoolAdmissionError::Full)
            } else {
                None
            };
            if let Some(reason) = refused {
                return Err(reason);
            }
            st.jobs.insert(job_id.to_owned(), request);
            st.requests.insert(
                request,
                Reservation {
                    job_id: job_id.to_owned(),
                    state: ReservationState::Waiting,
                    acquire_attempts: 0,
                    launch_attempts: 0,
                },
            );
            st.pending.push_back(request);
            Ok(request)
        });
        let request = admitted.map_err(LaunchError::Admission)?;
        let _waiting = Waiting { pool: self.clone(), request };
        let mut last_error = None;
        loop {
            let (id, process) = self.acquire(request).await.map_err(|error| match error {
                Failure::AcquireExhausted { attempts } => {
                    LaunchError::AcquireExhausted { attempts }
                }
                Failure::LaunchExhausted => {
                    last_error.take().map_or(LaunchError::NoProcess, LaunchError::Launch)
                }
                Failure::Closed => LaunchError::Admission(PoolAdmissionError::Closed),
            })?;
            match self.0.host.launch(&process, &job).await {
                Ok(()) => {
                    let completed = self.update(|_, st| {
                        if st.reservation(request) != Some(ReservationState::Reserved(id)) {
                            return false;
                        }
                        st.requests.remove(&request);
                        let entry = st.processes.get_mut(&id).unwrap();
                        entry.phase = Phase::Running;
                        entry.request = None;
                        true
                    });
                    if completed {
                        self.0.host.emit(PoolEvent::JobLaunched, &process);
                        return Ok(());
                    }
                }
                Err(error) => {
                    last_error = Some(error);
                    self.update(|pool, st| {
                        if st.reservation(request) == Some(ReservationState::Reserved(id)) {
                            pool.retire(st, id);
                        }
                    });
                }
            }
        }
    }

    pub async fn join_jobs(&self) {
        self.update(|_, st| {
            st.draining = true;
            st.refill_paused_until = None;
        });
        loop {
            let changed = self.0.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if !self.0.state.lock().has_jobs() {
                return;
            }
            changed.await;
        }
    }

    pub async fn aclose(&self) {
        let first = self.update(|pool, st| {
            let first = !st.closed;
            st.closed = true;
            st.refill_paused_until = None;
            st.pending.clear();
            let reservations: Vec<_> = st.requests.keys().copied().collect();
            for id in reservations {
                let request = st.requests.get_mut(&id).unwrap();
                request.state = ReservationState::Failed(Failure::Closed);
                let job_id = request.job_id.clone();
                st.release_job(&job_id, id);
            }
            let processes: Vec<_> = st.processes.keys().copied().collect();
            for id in processes {
                pool.retire(st, id);
            }
            first
        });
        if first {
            let pool = self.clone();
            // A caller's cancellation cannot discard the join obligations: the pool owns them.
            let cleanup = TokioSpawner::current().spawn(async move {
                let cleanup = async {
                    loop {
                        let changed = pool.0.changed.notified();
                        tokio::pin!(changed);
                        changed.as_mut().enable();
                        let tasks = std::mem::take(&mut pool.0.state.lock().tasks);
                        if tasks.is_empty() {
                            if pool.0.state.lock().processes.is_empty() {
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
                if tokio::time::timeout(pool.0.cleanup_timeout, &mut cleanup).await.is_err() {
                    let processes = pool.0.state.lock().processes.keys().copied().collect();
                    pool.0.host.cleanup_incomplete(PoolCleanupObservation { processes });
                    pool.0.close_ready.send_replace(true);
                    cleanup.await;
                }
                pool.0.close_ready.send_replace(true);
            });
            self.0.state.lock().cleanup = Some(cleanup);
        }
        let _ = self.0.close_ready.subscribe().wait_for(|done| *done).await;
    }
}

#[cfg(test)]
mod tests;
