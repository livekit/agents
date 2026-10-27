//! The warm process pool (`ipc/proc_pool.py`'s `ProcPool`): how many processes to keep warm, which
//! one a job takes, and the retries when a spawn or a launch fails. Making an executor, and each
//! of its steps (start, initialize, launch, close, join), stay with the host.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::{Notify, Semaphore, watch};
use tokio::task::JoinHandle;

use super::lock;

/// At most this many processes initialize at once (`MAX_CONCURRENT_INITIALIZATIONS`): the CPU
/// count (the cgroup quota's, rounded up), at most 4.
pub fn max_concurrent_initializations() -> usize {
    (crate::CpuMonitor::new().cpu_count().ceil() as usize).clamp(1, 4)
}

const MAX_ACQUIRE_ATTEMPTS: u32 = 3;
const MAX_LAUNCH_ATTEMPTS: u32 = 3;
/// How often the pool tops the warm processes up.
const REFILL_INTERVAL: Duration = Duration::from_millis(100);

/// The pool's events, as `ProcPool` emits them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolEvent {
    Created,
    Started,
    Ready,
    Closed,
    JobLaunched,
}

impl PoolEvent {
    pub fn as_str(self) -> &'static str {
        match self {
            PoolEvent::Created => "process_created",
            PoolEvent::Started => "process_started",
            PoolEvent::Ready => "process_ready",
            PoolEvent::Closed => "process_closed",
            PoolEvent::JobLaunched => "process_job_launched",
        }
    }
}

/// What the pool asks of its host: the executors and their steps.
#[async_trait::async_trait]
pub trait PoolHost: Send + Sync + 'static {
    type Proc: Clone + PartialEq + Send + Sync + 'static;
    type Job: Clone + Send + Sync + 'static;
    type Error: std::fmt::Display + Send + 'static;

    async fn create(&self) -> Result<Self::Proc, Self::Error>;
    async fn start(&self, proc: &Self::Proc) -> Result<(), Self::Error>;
    async fn initialize(&self, proc: &Self::Proc) -> Result<(), Self::Error>;
    async fn launch(&self, proc: &Self::Proc, job: &Self::Job) -> Result<(), Self::Error>;
    async fn close(&self, proc: &Self::Proc);
    async fn join(&self, proc: &Self::Proc);
    fn emit(&self, event: PoolEvent, proc: &Self::Proc);
    /// A process failed to start or initialize ("error initializing process").
    fn init_failed(&self, proc: &Self::Proc, error: Self::Error);
}

#[derive(Clone, Copy, Debug)]
pub struct PoolOptions {
    pub num_idle_processes: usize,
    pub initialize_timeout: Duration,
}

struct State<P> {
    warmed: VecDeque<P>,
    executors: Vec<P>,
    /// The processes a job was launched on, until they exited.
    launched: Vec<P>,
    /// Spawn tasks still initializing their process.
    spawning: usize,
    waiting: usize,
    target_idle: usize,
    started: bool,
    tasks: Vec<JoinHandle<()>>,
}

struct Inner<H: PoolHost> {
    host: H,
    opts: PoolOptions,
    state: Mutex<State<H::Proc>>,
    /// A process was warmed or a spawn ended: an acquire looks again.
    changed: Notify,
    idle_ready: watch::Sender<bool>,
    closed: watch::Sender<bool>,
    init: Semaphore,
}

/// Why `launch_job` raised.
#[derive(Debug)]
pub enum LaunchError<E> {
    /// Every attempt's spawns failed to initialize.
    NoProcess,
    /// The last attempt's launch failed with the host's error.
    Launch(E),
}

pub struct ProcPool<H: PoolHost>(Arc<Inner<H>>);

/// A launch waiting for a process, until it got one or was dropped.
struct Waiting<'a, H: PoolHost>(&'a ProcPool<H>);

impl<'a, H: PoolHost> Waiting<'a, H> {
    fn new(pool: &'a ProcPool<H>) -> Self {
        lock(&pool.0.state).waiting += 1;
        Self(pool)
    }
}

impl<H: PoolHost> Drop for Waiting<'_, H> {
    fn drop(&mut self) {
        lock(&self.0.0.state).waiting -= 1;
    }
}

/// A process a launch took from the warm ones: closed when the launch is dropped before it ran.
struct Taken<H: PoolHost> {
    pool: ProcPool<H>,
    proc: Option<H::Proc>,
}

impl<H: PoolHost> Drop for Taken<H> {
    fn drop(&mut self) {
        let (Some(proc), Ok(rt)) = (self.proc.take(), tokio::runtime::Handle::try_current()) else {
            return;
        };
        let pool = self.pool.clone();
        let close = rt.spawn(async move { pool.0.host.close(&proc).await });
        self.pool.track(close);
    }
}

impl<H: PoolHost> Clone for ProcPool<H> {
    fn clone(&self) -> Self {
        ProcPool(self.0.clone())
    }
}

impl<H: PoolHost> ProcPool<H> {
    pub fn new(host: H, opts: PoolOptions) -> Self {
        ProcPool(Arc::new(Inner {
            host,
            opts,
            state: Mutex::new(State {
                warmed: VecDeque::new(),
                executors: Vec::new(),
                launched: Vec::new(),
                spawning: 0,
                waiting: 0,
                target_idle: opts.num_idle_processes,
                started: false,
                tasks: Vec::new(),
            }),
            changed: Notify::new(),
            idle_ready: watch::channel(false).0,
            closed: watch::channel(false).0,
            init: Semaphore::new(max_concurrent_initializations()),
        }))
    }

    pub fn host(&self) -> &H {
        &self.0.host
    }

    pub fn processes(&self) -> Vec<H::Proc> {
        lock(&self.0.state).executors.clone()
    }

    pub fn jobs_waiting(&self) -> usize {
        lock(&self.0.state).waiting
    }

    pub fn target_idle(&self) -> usize {
        lock(&self.0.state).target_idle
    }

    pub fn set_target_idle(&self, n: usize) {
        lock(&self.0.state).target_idle = n;
    }

    fn track(&self, task: JoinHandle<()>) {
        let mut st = lock(&self.0.state);
        st.tasks.retain(|t| !t.is_finished());
        st.tasks.push(task);
    }

    /// Starts topping up the warm processes and waits (bounded by the initialize timeout + 2 s)
    /// for the first ones.
    pub async fn start(&self) {
        {
            let mut st = lock(&self.0.state);
            if st.started {
                return;
            }
            st.started = true;
        }
        let pool = self.clone();
        self.track(tokio::spawn(async move { pool.refill().await }));
        if self.0.opts.num_idle_processes > 0 {
            let mut ready = self.0.idle_ready.subscribe();
            let wait = self.0.opts.initialize_timeout + Duration::from_secs(2);
            if tokio::time::timeout(wait, ready.wait_for(|r| *r)).await.is_err() {
                log::warn!(target: "livekit.agents", "timed out waiting for idle processes to initialize");
            }
        }
    }

    fn closed(&self) -> bool {
        *self.0.closed.borrow()
    }

    async fn refill(&self) {
        while !self.closed() {
            let to_spawn = {
                let st = lock(&self.0.state);
                let pending = st.warmed.len() + st.spawning;
                let target = st.target_idle.min(self.0.opts.num_idle_processes).max(st.waiting);
                target.saturating_sub(pending)
            };
            for _ in 0..to_spawn {
                self.spawn();
            }
            tokio::time::sleep(REFILL_INTERVAL).await;
        }
    }

    fn spawn(&self) {
        lock(&self.0.state).spawning += 1;
        let pool = self.clone();
        self.track(tokio::spawn(async move { pool.spawn_proc().await }));
    }

    async fn spawn_proc(&self) {
        let inner = &self.0;
        let proc = match inner.host.create().await {
            Ok(proc) => proc,
            Err(e) => {
                log::error!(target: "livekit.agents", "error creating process: {e}");
                lock(&inner.state).spawning -= 1;
                inner.changed.notify_waiters();
                return;
            }
        };
        lock(&inner.state).executors.push(proc.clone());
        let mut closed = inner.closed.subscribe();
        let warm = async {
            let _permit = inner.init.acquire().await.ok()?;
            if self.closed() {
                return Some(false);
            }
            inner.host.emit(PoolEvent::Created, &proc);
            let steps = async {
                inner.host.start(&proc).await?;
                inner.host.emit(PoolEvent::Started, &proc);
                inner.host.initialize(&proc).await?;
                inner.host.emit(PoolEvent::Ready, &proc);
                Ok::<_, H::Error>(())
            };
            match steps.await {
                Ok(()) => Some(true),
                Err(e) => {
                    inner.host.init_failed(&proc, e);
                    Some(false)
                }
            }
        };
        // Closing the pool cancels an initializing spawn; its process is still closed.
        let warmed = tokio::select! {
            w = warm => w.unwrap_or(false),
            _ = closed.wait_for(|c| *c) => false,
        };
        {
            let mut st = lock(&inner.state);
            st.spawning -= 1;
            if warmed {
                st.warmed.push_back(proc.clone());
                if st.warmed.len() >= inner.opts.num_idle_processes {
                    inner.idle_ready.send_replace(true);
                }
            } else {
                st.executors.retain(|p| p != &proc);
            }
        }
        inner.changed.notify_waiters();
        if !warmed {
            inner.host.close(&proc).await;
            inner.host.emit(PoolEvent::Closed, &proc);
            return;
        }
        let pool = self.clone();
        self.track(tokio::spawn(async move {
            pool.0.host.join(&proc).await;
            pool.0.host.emit(PoolEvent::Closed, &proc);
            let mut st = lock(&pool.0.state);
            st.executors.retain(|p| p != &proc);
            st.launched.retain(|p| p != &proc);
        }));
    }

    async fn acquire(&self, job_id: &str) -> Option<H::Proc> {
        for attempt in 1..=MAX_ACQUIRE_ATTEMPTS {
            let empty = {
                let st = lock(&self.0.state);
                (st.warmed.is_empty(), st.spawning < st.waiting)
            };
            if empty.0 && empty.1 {
                self.spawn();
            }
            if empty.0 {
                log::warn!(target: "livekit.agents", job_id = job_id;
                    "no warmed process available for job, waiting for one to be created");
            }
            loop {
                let changed = self.0.changed.notified();
                tokio::pin!(changed);
                changed.as_mut().enable();
                {
                    let mut st = lock(&self.0.state);
                    if let Some(proc) = st.warmed.pop_front() {
                        return Some(proc);
                    }
                    // every in-flight spawn ended without a process: try again
                    if st.spawning == 0 {
                        break;
                    }
                }
                changed.await;
            }
            log::warn!(target: "livekit.agents", job_id = job_id, attempt = attempt;
                "all in-flight spawns failed to initialize, retrying");
        }
        log::error!(target: "livekit.agents", job_id = job_id;
            "failed to acquire process for job after {MAX_ACQUIRE_ATTEMPTS} attempts");
        None
    }

    /// Runs `job` on a warm process, retrying on a fresh one when the launch fails. Cancel-safe:
    /// a launch dropped while it waits for a process stops counting as waiting, and the process
    /// it took and did not launch on is closed.
    pub async fn launch_job(&self, job_id: &str, job: H::Job) -> Result<(), LaunchError<H::Error>> {
        for attempt in 1..=MAX_LAUNCH_ATTEMPTS {
            let proc = {
                let _waiting = Waiting::new(self);
                self.acquire(job_id).await
            };
            let proc = proc.ok_or(LaunchError::NoProcess)?;
            let mut taken = Taken { pool: self.clone(), proc: Some(proc.clone()) };
            let launched = self.0.host.launch(&proc, &job).await;
            taken.proc = None;
            match launched {
                Ok(()) => {
                    lock(&self.0.state).launched.push(proc.clone());
                    self.0.host.emit(PoolEvent::JobLaunched, &proc);
                    return Ok(());
                }
                Err(e) => {
                    let pool = self.clone();
                    let failed = proc.clone();
                    self.track(tokio::spawn(async move { pool.0.host.close(&failed).await }));
                    if attempt == MAX_LAUNCH_ATTEMPTS {
                        log::error!(target: "livekit.agents", job_id = job_id;
                            "failed to launch job on process after {MAX_LAUNCH_ATTEMPTS} attempts");
                        return Err(LaunchError::Launch(e));
                    }
                    log::warn!(target: "livekit.agents", job_id = job_id, attempt = attempt;
                        "failed to launch job on process, retrying with a new process");
                }
            }
        }
        unreachable!("the last attempt returns")
    }

    /// Waits until every process a job was launched on exited, those launched meanwhile
    /// included (the drain).
    pub async fn join_jobs(&self) {
        loop {
            let running = lock(&self.0.state).launched.clone();
            if running.is_empty() {
                return;
            }
            let joins: Vec<_> = running
                .iter()
                .map(|proc| {
                    let (pool, proc) = (self.clone(), proc.clone());
                    tokio::spawn(async move { pool.0.host.join(&proc).await })
                })
                .collect();
            for join in joins {
                let _ = join.await;
            }
            lock(&self.0.state).launched.retain(|p| !running.contains(p));
        }
    }

    /// Stops the refill and the initializing spawns, closes every process and waits for the
    /// pool's tasks.
    pub async fn aclose(&self) {
        if !lock(&self.0.state).started {
            return;
        }
        self.0.closed.send_replace(true);
        // the initializing spawns close their own process first
        loop {
            let changed = self.0.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if lock(&self.0.state).spawning == 0 {
                break;
            }
            changed.await;
        }
        let closing: Vec<_> = self
            .processes()
            .into_iter()
            .map(|p| {
                let pool = self.clone();
                tokio::spawn(async move { pool.0.host.close(&p).await })
            })
            .collect();
        for task in closing {
            let _ = task.await;
        }
        loop {
            let tasks: Vec<_> = std::mem::take(&mut lock(&self.0.state).tasks);
            if tasks.is_empty() {
                break;
            }
            for task in tasks {
                let _ = task.await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Processes are numbers; `fail_init` makes every initialize fail, `fail_launch` the first n
    /// launches.
    #[derive(Default)]
    struct Host {
        next: AtomicUsize,
        fail_init: bool,
        fail_launch: AtomicUsize,
        /// A launch never completes (to cancel it mid-way).
        slow_launch: bool,
        events: Mutex<Vec<(PoolEvent, usize)>>,
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
        async fn close(&self, p: &usize) {
            self.closed.lock().unwrap().push(*p);
            self.exits.notify_waiters();
        }
        async fn join(&self, p: &usize) {
            loop {
                let exited = self.exits.notified();
                if self.closed.lock().unwrap().contains(p) {
                    return;
                }
                exited.await;
            }
        }
        fn emit(&self, event: PoolEvent, p: &usize) {
            self.events.lock().unwrap().push((event, *p));
        }
        fn init_failed(&self, _: &usize, _: String) {}
    }

    fn pool(host: &Arc<Host>, idle: usize) -> ProcPool<Arc<Host>> {
        let opts = PoolOptions { num_idle_processes: idle, initialize_timeout: Duration::from_secs(1) };
        ProcPool::new(host.clone(), opts)
    }

    fn count(host: &Host, event: PoolEvent) -> usize {
        host.events.lock().unwrap().iter().filter(|(e, _)| *e == event).count()
    }

    // Catches: start() returning before the idle processes are warm, a launch that does not
    // replace the process it took, or a close that leaves a process open.
    #[tokio::test]
    async fn keeps_the_idle_processes_warm_and_replaces_the_one_a_job_takes() {
        let host = Arc::new(Host::default());
        let pool = pool(&host, 2);
        pool.start().await;
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

    // Catches: #5868, a launch hanging on an empty warm queue when every spawn fails.
    #[tokio::test]
    async fn a_launch_raises_when_every_spawn_fails() {
        let host = Arc::new(Host { fail_init: true, ..Host::default() });
        let pool = pool(&host, 0);
        pool.start().await;
        assert!(matches!(pool.launch_job("j1", ()).await, Err(LaunchError::NoProcess)));
        assert_eq!(pool.jobs_waiting(), 0);
        assert!(pool.processes().is_empty());
        pool.aclose().await;
    }

    // Catches: a cancelled launch leaving `waiting` raised (the pool kept that many extra processes
    // warm) or the process it took open, neither warm nor running.
    #[tokio::test]
    async fn a_cancelled_launch_releases_its_wait_and_its_process() {
        let host = Arc::new(Host::default());
        let pool = pool(&host, 0);
        pool.start().await;
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
        pool.start().await;
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
    #[tokio::test]
    async fn join_jobs_waits_for_the_processes_running_a_job() {
        let host = Arc::new(Host::default());
        let pool = pool(&host, 1);
        pool.start().await;
        pool.join_jobs().await;
        pool.launch_job("j1", ()).await.unwrap();
        let joining = tokio::spawn({
            let pool = pool.clone();
            async move { pool.join_jobs().await }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!joining.is_finished(), "the job's process still runs");
        host.close(&0).await;
        tokio::time::timeout(Duration::from_secs(1), joining).await.unwrap().unwrap();
        pool.aclose().await;
    }

    // Catches: a failed launch not retried on a fresh process, or the failed one left open.
    #[tokio::test]
    async fn a_failed_launch_is_retried_on_a_fresh_process() {
        let host = Arc::new(Host { fail_launch: AtomicUsize::new(1), ..Host::default() });
        let pool = pool(&host, 1);
        pool.start().await;
        pool.launch_job("j1", ()).await.unwrap();
        assert_eq!(count(&host, PoolEvent::JobLaunched), 1);
        assert!(host.closed.lock().unwrap().contains(&0), "the process that failed is closed");
        pool.aclose().await;
    }
}
