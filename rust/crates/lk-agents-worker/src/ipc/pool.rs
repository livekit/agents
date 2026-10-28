//! Warm process allocation, host execution and cleanup ownership.
//! Admission and process eligibility are changed under the same lock as owned resources;
//! host operations run after that lock is released.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::FutureExt;
use tokio::sync::{Notify, watch};
use tokio::task::JoinHandle;

use super::lock;

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

#[async_trait::async_trait]
pub trait PoolHost: Send + Sync + 'static {
    type Proc: Clone + PartialEq + Send + Sync + 'static;
    type Job: Clone + Send + Sync + 'static;
    type Error: std::fmt::Display + Send + 'static;

    async fn create(&self) -> Result<Self::Proc, Self::Error>;
    async fn start(&self, proc: &Self::Proc) -> Result<(), Self::Error>;
    async fn initialize(&self, proc: &Self::Proc) -> Result<(), Self::Error>;
    async fn launch(&self, proc: &Self::Proc, job: &Self::Job) -> Result<(), Self::Error>;
    async fn close(&self, proc: &Self::Proc) -> Result<(), Self::Error>;
    fn close_failed(&self, proc: &Self::Proc, error: Self::Error);
    fn cleanup_incomplete(&self, observation: PoolCleanupObservation) {
        crate::diagnostics::emit(
            log::Level::Error,
            format!("process pool cleanup incomplete: {observation:?}"),
        );
    }
    async fn wait_exit(&self, proc: &Self::Proc) -> Result<(), Self::Error>;
    fn exit_failed(&self, proc: &Self::Proc, error: Self::Error);
    async fn join(&self, proc: &Self::Proc) -> Result<PoolJoinOutcome, Self::Error>;
    fn join_failed(&self, proc: &Self::Proc, error: Self::Error);
    fn emit(&self, event: PoolEvent, proc: &Self::Proc);
    fn init_failed(&self, proc: &Self::Proc, error: Self::Error);
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

struct Inner<H: PoolHost> {
    host: H,
    opts: PoolOptions,
    state: Mutex<State<H::Proc>>,
    changed: Notify,
    idle_ready: watch::Sender<Readiness>,
    closed: watch::Sender<bool>,
    close_ready: watch::Sender<bool>,
}

#[derive(Debug)]
pub enum LaunchError<E> {
    NoProcess,
    AcquireExhausted { attempts: u32 },
    Launch(E),
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

pub struct ProcPool<H: PoolHost>(Arc<Inner<H>>);

struct Waiting<H: PoolHost> {
    pool: ProcPool<H>,
    request: ReservationId,
}

impl<H: PoolHost> Drop for Waiting<H> {
    fn drop(&mut self) {
        self.pool.handle(PoolInput::Cancel(self.request));
    }
}

impl<H: PoolHost> Clone for ProcPool<H> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<H: PoolHost> ProcPool<H> {
    pub fn new(host: H, opts: PoolOptions) -> Self {
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
            state: Mutex::new(State::new(limits)),
            changed: Notify::new(),
            idle_ready: watch::channel(Readiness::Pending).0,
            closed: watch::channel(false).0,
            close_ready: watch::channel(false).0,
        }))
    }

    pub fn host(&self) -> &H {
        &self.0.host
    }

    pub fn processes(&self) -> Vec<H::Proc> {
        lock(&self.0.state).executors.values().cloned().collect()
    }

    pub fn jobs_waiting(&self) -> usize {
        lock(&self.0.state).waiting()
    }

    pub fn target_idle(&self) -> usize {
        lock(&self.0.state).target_idle()
    }

    pub fn set_target_idle(&self, n: usize) {
        self.handle(PoolInput::TargetIdle(n));
    }

    fn track(&self, task: JoinHandle<()>) {
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
            let effects = state.handle(input, now);
            if state.is_closed() {
                self.0.closed.send_replace(true);
                self.0.idle_ready.send_replace(Readiness::Closed);
            } else if state.ready() >= self.0.opts.num_idle_processes {
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
            if state.is_closed() {
                return Err(PoolStartError::Closed);
            }
            let start = !state.is_started();
            let now = state.now;
            (start, state.handle(PoolInput::Start, now))
        };
        self.spawn(effects);
        self.handle(PoolInput::Wake);
        if start {
            let pool = self.clone();
            self.track(tokio::spawn(async move { pool.refill().await }));
        }
        let mut ready = self.0.idle_ready.subscribe();
        let wait = self.0.opts.initialize_timeout + Duration::from_secs(2);
        match tokio::time::timeout(wait, ready.wait_for(|state| *state != Readiness::Pending)).await
        {
            Ok(Ok(readiness)) if *readiness == Readiness::Ready => Ok(()),
            Ok(_) => Err(PoolStartError::Closed),
            Err(_) => Err(PoolStartError::WarmupTimedOut {
                ready: lock(&self.0.state).ready(),
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
            if self.closed() && !lock(&self.0.state).has_processes() {
                break;
            }
            let delay = {
                let mut state = lock(&self.0.state);
                state.now = tokio::time::Instant::now().duration_since(origin);
                state
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
                    self.track(tokio::spawn(async move { pool.spawn_proc(id).await }));
                }
                PoolEffect::Close(id) => {
                    let process = lock(&self.0.state).executors.get(&id).cloned();
                    if let Some(process) = process {
                        let pool = self.clone();
                        self.track(tokio::spawn(async move {
                            if let Err(error) = pool.0.host.close(&process).await {
                                pool.0.host.emit(PoolEvent::CleanupIncomplete, &process);
                                pool.0.host.close_failed(&process, error);
                            }
                            pool.handle(PoolInput::Closed(id));
                        }));
                    }
                }
                PoolEffect::WatchExit(id) => {
                    let process = lock(&self.0.state).executors.get(&id).cloned();
                    if let Some(process) = process {
                        let pool = self.clone();
                        self.track(tokio::spawn(async move {
                            match pool.0.host.wait_exit(&process).await {
                                Ok(()) => pool.handle(PoolInput::Exited(id)),
                                Err(error) => {
                                    pool.0.host.emit(PoolEvent::ExitFailed, &process);
                                    pool.0.host.exit_failed(&process, error);
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
                        self.track(tokio::spawn(async move {
                            match pool.0.host.join(&process).await {
                                Ok(outcome) => {
                                    pool.handle(PoolInput::Joined { process: id, outcome })
                                }
                                Err(error) => {
                                    pool.0.host.emit(PoolEvent::JoinFailed, &process);
                                    pool.0.host.join_failed(&process, error);
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
        if lock(&self.0.state).phase(id) != ProcessPhase::Initializing {
            return;
        }
        let mut closed = self.0.closed.subscribe();
        self.0.host.emit(PoolEvent::Created, &process);
        let steps = async {
            self.0.host.start(&process).await?;
            self.0.host.emit(PoolEvent::Started, &process);
            self.0.host.initialize(&process).await?;
            Ok::<_, H::Error>(())
        };
        let initialized = tokio::select! {
            biased;
            _ = closed.wait_for(|closed| *closed) => false,
            result = steps => match result {
                Ok(()) => true,
                Err(error) => {
                    self.0.host.init_failed(&process, error);
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

    async fn acquire(&self, request: ReservationId) -> Result<(ProcessId, H::Proc), PoolFailure> {
        loop {
            let changed = self.0.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            {
                let state = lock(&self.0.state);
                match state.reservation(request) {
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

    pub async fn launch_job(&self, job_id: &str, job: H::Job) -> Result<(), LaunchError<H::Error>> {
        let (request, effects) = {
            let mut state = lock(&self.0.state);
            let request = ReservationId(state.next_request);
            state.next_request += 1;
            let now = state.now;
            let effects =
                state.handle(PoolInput::Admit { request, job_id: job_id.to_owned() }, now);
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
                    if lock(&self.0.state).reservation(request) == Some(ReservationState::Completed)
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
            if !lock(&self.0.state).has_jobs() {
                return;
            }
            changed.await;
        }
    }

    pub async fn aclose(&self) {
        let (first, effects, deadline) = {
            let mut state = lock(&self.0.state);
            let first = !state.is_closed();
            let now = state.now;
            let effects = state.handle(PoolInput::Close, now);
            (first, effects, state.close_deadline())
        };
        self.spawn(effects);
        self.handle(PoolInput::Wake);
        if first {
            let pool = self.clone();
            // A caller's cancellation cannot discard the join obligations.
            tokio::spawn(async move {
                let cleanup = async {
                    loop {
                        let changed = pool.0.changed.notified();
                        tokio::pin!(changed);
                        changed.as_mut().enable();
                        let tasks = std::mem::take(&mut lock(&pool.0.state).tasks);
                        if tasks.is_empty() {
                            if !lock(&pool.0.state).has_processes() {
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
        }
        let _ = self.0.close_ready.subscribe().wait_for(|done| *done).await;
    }
}

pub const MAX_ACQUIRE_ATTEMPTS: u32 = 3;
pub const MAX_LAUNCH_ATTEMPTS: u32 = 3;
pub const REFILL_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProcessId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ReservationId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolJoinOutcome {
    Reaped,
    NotLaunched,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProcessPhase {
    Creating,
    Initializing,
    Ready,
    Reserved,
    Running,
    Retiring,
    Cleaned,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PoolFailure {
    AcquireExhausted { attempts: u32 },
    LaunchExhausted { attempts: u32 },
    Closed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReservationState {
    Waiting,
    Reserved(ProcessId),
    Completed,
    Failed(PoolFailure),
}

#[derive(Clone, Copy, Debug)]
struct PoolLimits {
    pub idle_processes: usize,
    pub concurrent_initializations: usize,
    pub pending_jobs: usize,
    pub cleanup_timeout: Duration,
}

enum PoolInput {
    Start,
    Admit { request: ReservationId, job_id: String },
    Created(ProcessId),
    Ready(ProcessId),
    SpawnFailed(ProcessId),
    LaunchSucceeded { request: ReservationId, process: ProcessId },
    LaunchFailed { request: ReservationId, process: ProcessId },
    Exited(ProcessId),
    ExitFailed(ProcessId),
    Cleaned(ProcessId),
    Closed(ProcessId),
    Joined { process: ProcessId, outcome: PoolJoinOutcome },
    JoinFailed(ProcessId),
    Cancel(ReservationId),
    Forget(ReservationId),
    TargetIdle(usize),
    Drain,
    Close,
    CleanupComplete,
    Wake,
}

#[derive(Debug, PartialEq, Eq)]
enum PoolEffect {
    Create(ProcessId),
    Close(ProcessId),
    Join(ProcessId),
    WatchExit(ProcessId),
    Released { process: ProcessId, outcome: PoolJoinOutcome },
    Refused { request: ReservationId, reason: PoolAdmissionError },
    CleanupIncomplete(PoolCleanupObservation),
    CloseComplete,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolCleanupObservation {
    pub deadline: Duration,
    pub processes: Vec<ProcessId>,
}

struct Process {
    phase: ProcessPhase,
    request: Option<ReservationId>,
    job_id: Option<(String, ReservationId)>,
    close_settled: bool,
    receipt: Option<PoolJoinOutcome>,
}

struct Reservation {
    job_id: String,
    state: ReservationState,
    acquire_attempts: u32,
    launch_attempts: u32,
}

struct State<P> {
    executors: BTreeMap<ProcessId, P>,
    tasks: Vec<JoinHandle<()>>,
    next_request: u64,
    now: Duration,
    limits: PoolLimits,
    target_idle: usize,
    started: bool,
    draining: bool,
    closed: bool,
    next_process: u64,
    processes: BTreeMap<ProcessId, Process>,
    initializing: BTreeSet<ProcessId>,
    ready: VecDeque<ProcessId>,
    pending: VecDeque<ReservationId>,
    requests: BTreeMap<ReservationId, Reservation>,
    jobs: BTreeMap<String, ReservationId>,
    refill_at: Option<Duration>,
    close_deadline: Option<Duration>,
    joining: BTreeSet<ProcessId>,
    retry_join_at: BTreeMap<ProcessId, Duration>,
    watching_exit: BTreeSet<ProcessId>,
    retry_exit_at: BTreeMap<ProcessId, Duration>,
}

impl<P> State<P> {
    fn new(limits: PoolLimits) -> Self {
        assert!(limits.concurrent_initializations > 0);
        assert!(limits.pending_jobs > 0);
        Self {
            executors: BTreeMap::new(),
            tasks: Vec::new(),
            next_request: 0,
            now: Duration::ZERO,
            target_idle: limits.idle_processes,
            limits,
            started: false,
            draining: false,
            closed: false,
            next_process: 0,
            processes: BTreeMap::new(),
            initializing: BTreeSet::new(),
            ready: VecDeque::new(),
            pending: VecDeque::new(),
            requests: BTreeMap::new(),
            jobs: BTreeMap::new(),
            refill_at: None,
            close_deadline: None,
            joining: BTreeSet::new(),
            retry_join_at: BTreeMap::new(),
            watching_exit: BTreeSet::new(),
            retry_exit_at: BTreeMap::new(),
        }
    }

    fn phase(&self, id: ProcessId) -> ProcessPhase {
        self.processes.get(&id).map_or(ProcessPhase::Cleaned, |p| p.phase)
    }

    fn reservation(&self, id: ReservationId) -> Option<ReservationState> {
        self.requests.get(&id).map(|request| request.state)
    }

    fn waiting(&self) -> usize {
        self.pending.len()
    }

    fn target_idle(&self) -> usize {
        self.target_idle
    }

    fn ready(&self) -> usize {
        self.ready.len()
    }

    fn is_started(&self) -> bool {
        self.started
    }

    fn is_closed(&self) -> bool {
        self.closed
    }

    fn has_processes(&self) -> bool {
        !self.processes.is_empty()
    }

    fn has_jobs(&self) -> bool {
        !self.pending.is_empty()
            || self
                .processes
                .values()
                .any(|p| matches!(p.phase, ProcessPhase::Reserved | ProcessPhase::Running))
    }

    fn wake_at(&self) -> Option<Duration> {
        self.refill_at
            .into_iter()
            .chain(self.close_deadline)
            .chain(self.retry_join_at.values().copied())
            .chain(self.retry_exit_at.values().copied())
            .min()
    }

    fn close_deadline(&self) -> Option<Duration> {
        self.close_deadline
    }

    fn handle(&mut self, input: PoolInput, now: Duration) -> Vec<PoolEffect> {
        let mut effects = Vec::new();
        if self.close_deadline.is_some_and(|deadline| now >= deadline) {
            effects.push(PoolEffect::CleanupIncomplete(PoolCleanupObservation {
                deadline: self.close_deadline.take().unwrap(),
                processes: self.processes.keys().copied().collect(),
            }));
        }
        if self.refill_at.is_some_and(|deadline| now >= deadline) {
            self.refill_at = None;
        }
        let due_joins: Vec<_> =
            self.retry_join_at.iter().filter_map(|(id, at)| (*at <= now).then_some(*id)).collect();
        for id in due_joins {
            self.retry_join_at.remove(&id);
            if self.processes.contains_key(&id) && self.joining.insert(id) {
                effects.push(PoolEffect::Join(id));
            }
        }
        let due_exits: Vec<_> =
            self.retry_exit_at.iter().filter_map(|(id, at)| (*at <= now).then_some(*id)).collect();
        for id in due_exits {
            self.retry_exit_at.remove(&id);
            if self.processes.contains_key(&id) && self.watching_exit.insert(id) {
                effects.push(PoolEffect::WatchExit(id));
            }
        }
        match input {
            PoolInput::Start => self.started = true,
            PoolInput::Admit { request, job_id } => {
                let refused = if self.closed {
                    Some(PoolAdmissionError::Closed)
                } else if self.draining {
                    Some(PoolAdmissionError::Draining)
                } else if !self.started {
                    Some(PoolAdmissionError::NotStarted)
                } else if self.jobs.contains_key(&job_id) || self.requests.contains_key(&request) {
                    Some(PoolAdmissionError::DuplicateJob)
                } else if self.requests.len() >= self.limits.pending_jobs {
                    Some(PoolAdmissionError::Full)
                } else {
                    None
                };
                if let Some(reason) = refused {
                    effects.push(PoolEffect::Refused { request, reason });
                } else {
                    self.jobs.insert(job_id.clone(), request);
                    self.requests.insert(
                        request,
                        Reservation {
                            job_id,
                            state: ReservationState::Waiting,
                            acquire_attempts: 0,
                            launch_attempts: 0,
                        },
                    );
                    self.pending.push_back(request);
                }
            }
            PoolInput::Created(id) => {
                if let Some(process) = self.processes.get_mut(&id) {
                    if process.phase == ProcessPhase::Creating {
                        process.phase = ProcessPhase::Initializing;
                    } else if process.phase == ProcessPhase::Retiring {
                        effects.push(PoolEffect::Close(id));
                    }
                }
            }
            PoolInput::Ready(id) => {
                if let Some(process) = self.processes.get_mut(&id)
                    && process.phase == ProcessPhase::Initializing
                {
                    process.phase = ProcessPhase::Ready;
                    process.request = None;
                    self.initializing.remove(&id);
                    self.ready.push_back(id);
                    self.joining.insert(id);
                    effects.push(PoolEffect::Join(id));
                    self.watching_exit.insert(id);
                    effects.push(PoolEffect::WatchExit(id));
                }
            }
            PoolInput::SpawnFailed(id) => {
                self.refill_at = Some(now + REFILL_INTERVAL);
                self.retire(id, &mut effects);
            }
            PoolInput::LaunchSucceeded { request, process } => {
                if self.reservation(request) == Some(ReservationState::Reserved(process)) {
                    let pending = self.requests.get_mut(&request).unwrap();
                    pending.state = ReservationState::Completed;
                    let entry = self.processes.get_mut(&process).unwrap();
                    entry.phase = ProcessPhase::Running;
                    entry.request = None;
                }
            }
            PoolInput::LaunchFailed { request, process } => {
                if self.reservation(request) == Some(ReservationState::Reserved(process)) {
                    self.retire(process, &mut effects);
                }
            }
            PoolInput::Exited(id) => {
                self.watching_exit.remove(&id);
                self.retry_exit_at.remove(&id);
                self.retire(id, &mut effects);
            }
            PoolInput::ExitFailed(id) => {
                self.watching_exit.remove(&id);
                if self.processes.contains_key(&id) {
                    self.retire(id, &mut effects);
                    self.retry_exit_at.insert(id, now + REFILL_INTERVAL);
                }
            }
            PoolInput::Cleaned(id) => {
                self.retire(id, &mut effects);
                effects.retain(|effect| *effect != PoolEffect::Close(id));
                self.initializing.remove(&id);
                self.joining.remove(&id);
                self.retry_join_at.remove(&id);
                self.watching_exit.remove(&id);
                self.retry_exit_at.remove(&id);
                self.processes.remove(&id);
            }
            PoolInput::Closed(id) => {
                if let Some(process) = self.processes.get_mut(&id) {
                    process.close_settled = true;
                    if let Some(outcome) = process.receipt {
                        self.release(id, outcome, &mut effects);
                    } else if !self.joining.contains(&id) && !self.retry_join_at.contains_key(&id) {
                        self.joining.insert(id);
                        effects.push(PoolEffect::Join(id));
                    }
                }
            }
            PoolInput::Joined { process: id, outcome } => {
                self.joining.remove(&id);
                self.retry_join_at.remove(&id);
                if let Some(process) = self.processes.get_mut(&id) {
                    process.receipt = Some(outcome);
                    self.retire(id, &mut effects);
                    if self.processes.get(&id).is_some_and(|process| process.close_settled) {
                        self.release(id, outcome, &mut effects);
                    }
                }
            }
            PoolInput::JoinFailed(id) => {
                self.joining.remove(&id);
                if self.processes.contains_key(&id) {
                    self.retire(id, &mut effects);
                    self.retry_join_at.insert(id, now + REFILL_INTERVAL);
                }
            }
            PoolInput::Cancel(id) => {
                if let Some(request) = self.requests.remove(&id) {
                    self.pending.retain(|pending| *pending != id);
                    if let ReservationState::Reserved(process) = request.state {
                        self.retire(process, &mut effects);
                    }
                    if request.state != ReservationState::Completed {
                        release_job(&mut self.jobs, &request.job_id, id);
                    }
                    for process in self.processes.values_mut() {
                        if process.request == Some(id) {
                            process.request = None;
                        }
                    }
                }
            }
            PoolInput::Forget(id) => {
                if let Some(request) = self.requests.get(&id)
                    && matches!(
                        request.state,
                        ReservationState::Completed | ReservationState::Failed(_)
                    )
                {
                    self.requests.remove(&id);
                }
            }
            PoolInput::TargetIdle(n) => self.target_idle = n.min(self.limits.idle_processes),
            PoolInput::Drain => {
                self.draining = true;
                self.refill_at = None;
            }
            PoolInput::Close => {
                if !self.closed {
                    self.close_deadline = Some(now + self.limits.cleanup_timeout);
                }
                self.closed = true;
                self.refill_at = None;
                self.pending.clear();
                for (id, request) in &mut self.requests {
                    if request.state != ReservationState::Completed {
                        release_job(&mut self.jobs, &request.job_id, *id);
                        request.state = ReservationState::Failed(PoolFailure::Closed);
                    }
                }
                let ids: Vec<_> = self.processes.keys().copied().collect();
                for id in ids {
                    self.retire(id, &mut effects);
                }
            }
            PoolInput::CleanupComplete => {
                if self.closed && !self.has_processes() {
                    self.close_deadline = None;
                    effects.push(PoolEffect::CloseComplete);
                }
            }
            PoolInput::Wake => {}
        }
        self.allocate(&mut effects);
        effects
    }

    fn release(&mut self, id: ProcessId, outcome: PoolJoinOutcome, effects: &mut Vec<PoolEffect>) {
        self.initializing.remove(&id);
        self.joining.remove(&id);
        self.retry_join_at.remove(&id);
        self.watching_exit.remove(&id);
        self.retry_exit_at.remove(&id);
        self.ready.retain(|ready| *ready != id);
        self.processes.remove(&id);
        effects.push(PoolEffect::Released { process: id, outcome });
    }

    fn retire(&mut self, id: ProcessId, effects: &mut Vec<PoolEffect>) {
        let Some(process) = self.processes.get_mut(&id) else { return };
        if process.phase == ProcessPhase::Retiring {
            return;
        }
        let reserved = process.phase == ProcessPhase::Reserved;
        process.phase = ProcessPhase::Retiring;
        self.ready.retain(|ready| *ready != id);
        if let Some(request_id) = process.request
            && let Some(request) = self.requests.get_mut(&request_id)
            && (request.state == ReservationState::Waiting
                || request.state == ReservationState::Reserved(id))
        {
            if reserved {
                request.launch_attempts += 1;
            }
            if request.launch_attempts >= MAX_LAUNCH_ATTEMPTS {
                request.state = ReservationState::Failed(PoolFailure::LaunchExhausted {
                    attempts: request.launch_attempts,
                });
                release_job(&mut self.jobs, &request.job_id, request_id);
            } else if request.acquire_attempts >= MAX_ACQUIRE_ATTEMPTS {
                request.state = ReservationState::Failed(PoolFailure::AcquireExhausted {
                    attempts: request.acquire_attempts,
                });
                release_job(&mut self.jobs, &request.job_id, request_id);
            } else {
                request.state = ReservationState::Waiting;
                if !self.pending.contains(&request_id) {
                    self.pending.push_back(request_id);
                }
            }
        } else if let Some((job_id, request_id)) = process.job_id.take() {
            release_job(&mut self.jobs, &job_id, request_id);
        }
        effects.push(PoolEffect::Close(id));
    }

    fn allocate(&mut self, effects: &mut Vec<PoolEffect>) {
        if !self.started || self.closed {
            return;
        }
        self.pending.retain(|id| {
            self.requests.get(id).is_some_and(|request| request.state == ReservationState::Waiting)
        });
        while !self.ready.is_empty() && !self.pending.is_empty() {
            let first = self.pending.front().unwrap();
            if self.processes.values().any(|process| {
                process.request == Some(*first) && process.phase == ProcessPhase::Retiring
            }) {
                // The head reservation must finish cleanup before any replacement is assigned.
                // Spawning for later requests here would repeatedly replace their ready processes
                // while the head remains blocked, starving the runtime and growing the pool.
                return;
            }
            let process_id = self.ready.pop_front().unwrap();
            let request_id = self.pending.pop_front().unwrap();
            let request = self.requests.get_mut(&request_id).unwrap();
            request.state = ReservationState::Reserved(process_id);
            request.acquire_attempts = 0;
            for (other_id, process) in &mut self.processes {
                if *other_id != process_id && process.request == Some(request_id) {
                    process.request = None;
                }
            }
            let process = self.processes.get_mut(&process_id).unwrap();
            process.phase = ProcessPhase::Reserved;
            process.request = Some(request_id);
            process.job_id = Some((request.job_id.clone(), request_id));
        }
        for request_id in self.pending.iter().copied().collect::<Vec<_>>() {
            if self.processes.values().any(|p| p.request == Some(request_id)) {
                continue;
            }
            let process_id = self.processes.iter().find_map(|(id, process)| {
                (self.initializing.contains(id)
                    && process.request.is_none()
                    && process.phase != ProcessPhase::Retiring)
                    .then_some(*id)
            });
            let process_id = match process_id {
                Some(id) => id,
                None if self.initializing.len() < self.limits.concurrent_initializations => {
                    self.create(effects)
                }
                None => break,
            };
            self.requests.get_mut(&request_id).unwrap().acquire_attempts += 1;
            self.processes.get_mut(&process_id).unwrap().request = Some(request_id);
        }
        if !self.draining && self.refill_at.is_none() {
            let unassigned = self
                .processes
                .values()
                .filter(|process| {
                    process.request.is_none()
                        && matches!(
                            process.phase,
                            ProcessPhase::Creating
                                | ProcessPhase::Initializing
                                | ProcessPhase::Ready
                        )
                })
                .count();
            for _ in unassigned..self.target_idle {
                if self.initializing.len() >= self.limits.concurrent_initializations {
                    break;
                }
                self.create(effects);
            }
        }
    }

    fn create(&mut self, effects: &mut Vec<PoolEffect>) -> ProcessId {
        let id = ProcessId(self.next_process);
        self.next_process += 1;
        self.processes.insert(
            id,
            Process {
                phase: ProcessPhase::Creating,
                request: None,
                job_id: None,
                close_settled: false,
                receipt: None,
            },
        );
        self.initializing.insert(id);
        effects.push(PoolEffect::Create(id));
        id
    }
}

fn release_job(jobs: &mut BTreeMap<String, ReservationId>, job: &str, request: ReservationId) {
    if jobs.get(job) == Some(&request) {
        jobs.remove(job);
    }
}

#[cfg(test)]
#[path = "pool/allocation_tests.rs"]
mod allocation_tests;
#[cfg(test)]
mod tests;
