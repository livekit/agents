//! Async execution for the allocation policy in `pool_state`.
//!
//! The adapter owns resources, tasks and notifications. Admission, eligibility, retries and
//! drain decisions belong exclusively to `PoolState`; this module can move to runtime intact.

use std::collections::BTreeMap;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::FutureExt;
use tokio::sync::{Notify, watch};
use tokio::task::JoinHandle;

use super::lock;

#[path = "pool_state.rs"]
mod state;
pub use state::{
    MAX_ACQUIRE_ATTEMPTS, PoolAdmissionError, PoolCleanupObservation, PoolEffect, PoolFailure,
    PoolInput, PoolJoinOutcome, PoolLimits, PoolState, ProcessId, ProcessPhase, REFILL_INTERVAL,
    ReservationId, ReservationState,
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

struct State<P> {
    allocation: PoolState,
    executors: BTreeMap<ProcessId, P>,
    tasks: Vec<JoinHandle<()>>,
    next_request: u64,
    now: Duration,
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
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
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
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
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
            state: Mutex::new(State {
                allocation: PoolState::new(limits),
                executors: BTreeMap::new(),
                tasks: Vec::new(),
                next_request: 0,
                now: Duration::ZERO,
            }),
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
        lock(&self.0.state).allocation.waiting()
    }

    pub fn target_idle(&self) -> usize {
        lock(&self.0.state).allocation.target_idle()
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

    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
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
            self.track(tokio::spawn(async move { pool.refill().await }));
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

    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
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

    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
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

    #[allow(clippy::disallowed_macros, reason = "S0: migrate live execution/state to runtime")]
    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
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

    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    pub async fn launch_job(&self, job_id: &str, job: H::Job) -> Result<(), LaunchError<H::Error>> {
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

    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
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

    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
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
            // A caller's cancellation cannot discard the join obligations.
            tokio::spawn(async move {
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
                // Refill can finish before other owned tasks; keep the core deadline serviceable.
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

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Processes are numbers; `fail_init` makes every initialize fail, `fail_launch` the first n
    /// launches.
    #[derive(Default)]
    struct Host {
        #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
        next: AtomicUsize,
        fail_init: bool,
        #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
        fail_launch: AtomicUsize,
        /// A launch never completes (to cancel it mid-way).
        slow_launch: bool,
        slow_initialize: bool,
        close_gate: Option<watch::Sender<bool>>,
        join_receipt: Option<watch::Sender<bool>>,
        reap_gate: Option<watch::Sender<bool>>,
        fail_close: bool,
        cleanup_observed: Option<watch::Sender<Option<PoolCleanupObservation>>>,
        #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
        events: Mutex<Vec<(PoolEvent, usize)>>,
        #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
        closed: Mutex<Vec<usize>>,
        exits: Notify,
    }

    #[async_trait::async_trait]
    impl PoolHost for Arc<Host> {
        type Proc = usize;
        type Job = ();
        type Error = String;

        async fn create(&self) -> Result<usize, String> {
            Ok(self.next.fetch_add(1, Ordering::SeqCst))
        }
        async fn start(&self, _: &usize) -> Result<(), String> {
            Ok(())
        }
        async fn initialize(&self, _: &usize) -> Result<(), String> {
            if self.slow_initialize {
                std::future::pending::<()>().await;
            }
            if self.fail_init { Err("init failed".into()) } else { Ok(()) }
        }
        async fn launch(&self, _: &usize, _: &()) -> Result<(), String> {
            if self.slow_launch {
                std::future::pending::<()>().await;
            }
            let left = self.fail_launch.load(Ordering::SeqCst);
            if left > 0 {
                self.fail_launch.store(left - 1, Ordering::SeqCst);
                return Err("launch failed".into());
            }
            Ok(())
        }
        async fn close(&self, p: &usize) -> Result<(), String> {
            if let Some(gate) = &self.close_gate {
                let _ = gate.subscribe().wait_for(|released| *released).await;
            }
            self.closed.lock().unwrap().push(*p);
            self.exits.notify_waiters();
            if self.fail_close { Err("close failed".into()) } else { Ok(()) }
        }
        fn close_failed(&self, _: &usize, error: String) {
            assert!(self.fail_close);
            assert_eq!(error, "close failed", "cleanup must preserve its cause");
        }
        fn cleanup_incomplete(&self, observation: PoolCleanupObservation) {
            self.cleanup_observed
                .as_ref()
                .expect("unexpected cleanup deadline miss")
                .send_replace(Some(observation));
        }
        async fn wait_exit(&self, p: &usize) -> Result<(), String> {
            loop {
                let exited = self.exits.notified();
                tokio::pin!(exited);
                exited.as_mut().enable();
                if self.closed.lock().unwrap().contains(p) {
                    return Ok(());
                }
                exited.await;
            }
        }
        fn exit_failed(&self, _: &usize, error: String) {
            panic!("unexpected exit observation failure: {error}");
        }
        async fn join(&self, p: &usize) -> Result<PoolJoinOutcome, String> {
            loop {
                let exited = self.exits.notified();
                tokio::pin!(exited);
                exited.as_mut().enable();
                if self.closed.lock().unwrap().contains(p) {
                    if self.join_receipt.as_ref().is_some_and(|receipt| !*receipt.borrow()) {
                        return Err("join failed".into());
                    }
                    if let Some(gate) = &self.reap_gate {
                        let _ = gate.subscribe().wait_for(|reaped| *reaped).await;
                    }
                    return Ok(PoolJoinOutcome::Reaped);
                }
                exited.await;
            }
        }
        fn join_failed(&self, _: &usize, error: String) {
            assert_eq!(error, "join failed", "join must preserve its cause");
        }
        fn emit(&self, event: PoolEvent, p: &usize) {
            self.events.lock().unwrap().push((event, *p));
        }
        fn init_failed(&self, _: &usize, _: String) {}
    }

    fn pool(host: &Arc<Host>, idle: usize) -> ProcPool<Arc<Host>> {
        let opts = PoolOptions {
            num_idle_processes: idle,
            initialize_timeout: Duration::from_secs(1),
            close_timeout: Duration::from_secs(1),
        };
        ProcPool::new(host.clone(), opts)
    }

    fn count(host: &Host, event: PoolEvent) -> usize {
        host.events.lock().unwrap().iter().filter(|(e, _)| *e == event).count()
    }

    // Catches: start() returning before the idle processes are warm, a launch that does not
    // replace the process it took, or a close that leaves a process open.
    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    #[tokio::test]
    async fn keeps_the_idle_processes_warm_and_replaces_the_one_a_job_takes() {
        let host = Arc::new(Host::default());
        let pool = pool(&host, 2);
        pool.start().await.unwrap();
        assert_eq!(count(&host, PoolEvent::Ready), 2);
        pool.launch_job("j1", ()).await.unwrap();
        assert_eq!(count(&host, PoolEvent::JobLaunched), 1);
        tokio::time::sleep(REFILL_INTERVAL * 3).await;
        assert_eq!(count(&host, PoolEvent::Ready), 3, "the taken process is replaced");
        pool.aclose().await;
        let mut closed = host.closed.lock().unwrap().clone();
        closed.sort_unstable();
        closed.dedup();
        assert_eq!(closed, vec![0, 1, 2]);
        assert!(pool.processes().is_empty());
    }

    #[allow(clippy::disallowed_methods, reason = "S0: test harness runtime resource")]
    #[tokio::test]
    async fn exited_warm_process_is_never_assigned_to_a_job() {
        let host = Arc::new(Host::default());
        let pool = pool(&host, 1);
        pool.start().await.unwrap();
        host.close(&0).await.unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while pool.processes().contains(&0) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        pool.launch_job("after-exit", ()).await.unwrap();
        assert!(
            host.events
                .lock()
                .unwrap()
                .iter()
                .all(|(event, process)| *event != PoolEvent::JobLaunched || *process != 0),
            "an exited warmed process was assigned to a new job"
        );
        pool.aclose().await;
    }

    // Catches: #5868, a launch hanging on an empty warm queue when every spawn fails.
    #[tokio::test]
    async fn a_launch_raises_when_every_spawn_fails() {
        let host = Arc::new(Host { fail_init: true, ..Host::default() });
        let pool = pool(&host, 0);
        pool.start().await.unwrap();
        assert!(matches!(
            pool.launch_job("j1", ()).await,
            Err(LaunchError::AcquireExhausted { attempts: MAX_ACQUIRE_ATTEMPTS })
        ));
        assert_eq!(pool.jobs_waiting(), 0);
        assert!(pool.processes().is_empty());
        pool.aclose().await;
    }

    // Catches: a cancelled launch leaving `waiting` raised (the pool kept that many extra processes
    // warm) or the process it took open, neither warm nor running.
    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    #[tokio::test]
    async fn a_cancelled_launch_releases_its_wait_and_its_process() {
        let host = Arc::new(Host::default());
        let pool = pool(&host, 0);
        pool.start().await.unwrap();
        let waiting = tokio::spawn({
            let pool = pool.clone();
            async move { pool.launch_job("j1", ()).await }
        });
        tokio::time::sleep(Duration::from_millis(5)).await;
        waiting.abort();
        let _ = waiting.await;
        assert_eq!(pool.jobs_waiting(), 0, "the cancelled launch still counts as waiting");

        let slow = Arc::new(Host { slow_launch: true, ..Host::default() });
        let pool = self::pool(&slow, 1);
        pool.start().await.unwrap();
        let launching = tokio::spawn({
            let pool = pool.clone();
            async move { pool.launch_job("j1", ()).await }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        launching.abort();
        let _ = launching.await;
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(slow.closed.lock().unwrap().contains(&0), "the process taken is closed");
        pool.aclose().await;
    }

    // Catches: the drain returning while a job's process still runs, or waiting on the idle ones.
    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    #[tokio::test]
    async fn join_jobs_waits_for_the_processes_running_a_job() {
        let host = Arc::new(Host::default());
        let pool = pool(&host, 1);
        pool.start().await.unwrap();
        pool.launch_job("j1", ()).await.unwrap();
        let joining = tokio::spawn({
            let pool = pool.clone();
            async move { pool.join_jobs().await }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!joining.is_finished(), "the job's process still runs");
        host.close(&0).await.unwrap();
        tokio::time::timeout(Duration::from_secs(1), joining).await.unwrap().unwrap();
        pool.aclose().await;
    }

    // Catches: a failed launch not retried on a fresh process, or the failed one left open.
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    #[tokio::test]
    async fn a_failed_launch_is_retried_on_a_fresh_process() {
        let host = Arc::new(Host { fail_launch: AtomicUsize::new(1), ..Host::default() });
        let pool = pool(&host, 1);
        pool.start().await.unwrap();
        pool.launch_job("j1", ()).await.unwrap();
        assert_eq!(count(&host, PoolEvent::JobLaunched), 1);
        assert!(host.closed.lock().unwrap().contains(&0), "the process that failed is closed");
        pool.aclose().await;
    }

    #[tokio::test]
    async fn direct_launch_is_refused_after_drain_and_close() {
        let host = Arc::new(Host::default());
        let pool = pool(&host, 1);
        pool.start().await.unwrap();
        pool.join_jobs().await;
        assert!(matches!(
            pool.launch_job("after-drain", ()).await,
            Err(LaunchError::Admission(PoolAdmissionError::Draining))
        ));
        assert_eq!(host.next.load(Ordering::SeqCst), 1);
        pool.aclose().await;
        assert!(matches!(
            pool.launch_job("after-close", ()).await,
            Err(LaunchError::Admission(PoolAdmissionError::Closed))
        ));
        assert_eq!(host.next.load(Ordering::SeqCst), 1);
    }

    #[allow(clippy::disallowed_methods, reason = "S0: test harness runtime resource")]
    #[tokio::test]
    async fn close_callers_join_cleanup_even_when_first_caller_is_cancelled() {
        let gate = watch::channel(false).0;
        let host = Arc::new(Host { close_gate: Some(gate.clone()), ..Host::default() });
        let pool = pool(&host, 1);
        pool.start().await.unwrap();
        let first = tokio::spawn({
            let pool = pool.clone();
            async move { pool.aclose().await }
        });
        tokio::time::timeout(Duration::from_secs(1), async {
            while gate.receiver_count() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        first.abort();
        let _ = first.await;
        let second = tokio::spawn({
            let pool = pool.clone();
            async move { pool.aclose().await }
        });
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        assert!(!second.is_finished(), "a repeated close must join the owned cleanup");
        gate.send_replace(true);
        tokio::time::timeout(Duration::from_secs(1), second).await.unwrap().unwrap();
        assert!(pool.processes().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn warmup_timeout_never_reports_ready() {
        let host = Arc::new(Host { slow_initialize: true, ..Host::default() });
        let pool = pool(&host, 1);
        assert_eq!(
            pool.start().await,
            Err(PoolStartError::WarmupTimedOut { ready: 0, required: 1 })
        );
        assert_eq!(count(&host, PoolEvent::Ready), 0);
        pool.aclose().await;
    }

    #[tokio::test]
    async fn cleanup_failure_is_observed_without_failing_public_close() {
        let host = Arc::new(Host { fail_close: true, ..Host::default() });
        let pool = pool(&host, 1);
        pool.start().await.unwrap();
        pool.aclose().await;
        assert_eq!(count(&host, PoolEvent::CleanupIncomplete), 1);
        assert!(pool.processes().is_empty());
    }

    #[allow(clippy::disallowed_methods, reason = "S0: test harness runtime resource")]
    #[tokio::test(start_paused = true)]
    async fn close_deadline_returns_normally_and_retains_cleanup_ownership() {
        let gate = watch::channel(false).0;
        let observed = watch::channel(None).0;
        let host = Arc::new(Host {
            close_gate: Some(gate.clone()),
            cleanup_observed: Some(observed.clone()),
            ..Host::default()
        });
        let pool = pool(&host, 1);
        pool.start().await.unwrap();
        pool.aclose().await;
        assert_eq!(observed.borrow().as_ref().unwrap().processes, [ProcessId(0)]);
        assert_eq!(pool.processes(), [0], "the resource stays owned after public close returns");
        gate.send_replace(true);
        tokio::time::timeout(Duration::from_secs(1), async {
            while !pool.processes().is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(count(&host, PoolEvent::Closed), 1);
    }

    #[allow(clippy::disallowed_methods, reason = "S0: runtime migration")]
    #[tokio::test(start_paused = true)]
    async fn failed_join_cannot_report_cleaned_or_release_the_resource_owner() {
        let receipt = watch::channel(false).0;
        let observed = watch::channel(None).0;
        let host = Arc::new(Host {
            join_receipt: Some(receipt.clone()),
            cleanup_observed: Some(observed.clone()),
            ..Host::default()
        });
        let pool = pool(&host, 1);
        pool.start().await.unwrap();
        pool.aclose().await;
        assert_eq!(pool.processes(), [0], "failed join cannot release process ownership");
        assert_eq!(count(&host, PoolEvent::Closed), 0, "failed join is not cleaned");
        assert_eq!(observed.borrow().as_ref().unwrap().processes, [ProcessId(0)]);
        receipt.send_replace(true);
        tokio::time::timeout(REFILL_INTERVAL * 3, async {
            while !pool.processes().is_empty() {
                tokio::time::sleep(REFILL_INTERVAL).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(count(&host, PoolEvent::Closed), 1);
    }

    #[allow(clippy::disallowed_methods, reason = "S0: runtime migration")]
    #[tokio::test(start_paused = true)]
    async fn os_exit_removes_warm_eligibility_before_the_reaping_receipt() {
        let reaped = watch::channel(false).0;
        let host = Arc::new(Host { reap_gate: Some(reaped.clone()), ..Host::default() });
        let pool = pool(&host, 1);
        pool.start().await.unwrap();
        host.close(&0).await.unwrap();
        tokio::time::sleep(REFILL_INTERVAL).await;
        assert!(pool.processes().contains(&0), "ownership must remain until reaping");
        pool.launch_job("after-os-exit", ()).await.unwrap();
        let dead_process_was_launched = host
            .events
            .lock()
            .unwrap()
            .iter()
            .any(|(event, process)| *event == PoolEvent::JobLaunched && *process == 0);
        reaped.send_replace(true);
        pool.aclose().await;
        assert!(!dead_process_was_launched, "OS-exited warm process was assigned before reaping");
    }
}
