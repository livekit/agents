use super::*;

use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};

use lk_agents_abi::invoke::CallContext;
use lk_agents_abi::worker::ipc::JobCompletion;
use lk_agents_abi::worker::ipc_host::{JobExecutor as HostJobLauncher, SupervisedJob};
use lk_agents_abi::{BoxFuture, RefRegistry};
use serde_json::{Map, Value};

/// Processes are numbered in creation order; `fail_init` makes every initialize fail,
/// `fail_launch` the first n launches.
#[derive(Default)]
struct Host {
    created: Mutex<Vec<Executor>>,
    fail_init: bool,

    fail_launch: AtomicUsize,
    /// A launch never completes (to cancel it mid-way).
    slow_launch: bool,
    slow_initialize: bool,
    close_gate: Option<watch::Sender<bool>>,
    join_receipt: Option<watch::Sender<bool>>,
    reap_gate: Option<watch::Sender<bool>>,
    fail_close: bool,
    cleanup_observed: Option<watch::Sender<Option<PoolCleanupObservation>>>,

    events: Mutex<Vec<(ProcessEvent, usize)>>,

    closed: Mutex<Vec<usize>>,
    exits: Notify,
    /// Each event reads the pool back, as the Python host's handlers can while the GIL is held.
    reads_pool: std::sync::OnceLock<ProcPool>,
}

impl Host {
    fn process(&self, n: usize) -> Executor {
        self.created.lock()[n].clone()
    }

    fn created(&self) -> usize {
        self.created.lock().len()
    }
}

/// The pool sees a process only by its identity: no step reaches the host's executor.
pub(in crate::ipc) struct Numbered(pub(in crate::ipc) usize);

impl HostJobLauncher for Numbered {
    fn launch<'a>(&'a self, _: &'a CallContext, _: i64) -> BoxFuture<'a, Result<(), HostError>> {
        unreachable!()
    }
    fn completed<'a>(
        &'a self,
        _: &'a CallContext,
        _: JobCompletion,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        unreachable!()
    }
    fn log(
        &self,
        _: i32,
        _: String,
        _: Map<String, Value>,
        _: Option<HostError>,
    ) -> Result<(), HostError> {
        unreachable!()
    }
    fn start<'a>(
        &'a self,
        _: &'a CallContext,
    ) -> BoxFuture<'a, Result<Arc<dyn SupervisedJob>, HostError>> {
        unreachable!()
    }
    fn initialize<'a>(&'a self, _: &'a CallContext) -> BoxFuture<'a, Result<(), HostError>> {
        unreachable!()
    }
    fn launch_job<'a>(
        &'a self,
        _: &'a CallContext,
        _: HostRef,
        _: f64,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        unreachable!()
    }
    fn close<'a>(&'a self, _: &'a CallContext) -> BoxFuture<'a, Result<(), HostError>> {
        unreachable!()
    }
    fn join<'a>(&'a self, _: &'a CallContext) -> BoxFuture<'a, Result<bool, HostError>> {
        unreachable!()
    }
}

fn number(process: &Executor) -> usize {
    let host: &dyn Any = &*process.host;
    host.downcast_ref::<Numbered>().expect("a numbered process").0
}

fn numbers(processes: &[Executor]) -> Vec<usize> {
    processes.iter().map(number).collect()
}

struct NoRefs;

impl RefRegistry for NoRefs {
    fn release(&self, _: u64) {}
}

/// The job the pool hands a process: it never reads it.
fn job() -> HostRef {
    let refs: Arc<dyn RefRegistry> = Arc::new(NoRefs);
    HostRef::new(0, &refs)
}

#[async_trait::async_trait]
impl Executors for Host {
    async fn create(&self) -> Result<Executor, HostError> {
        let mut created = self.created.lock();
        let process = Executor::new(Arc::new(Numbered(created.len())));
        created.push(process.clone());
        Ok(process)
    }
    async fn start(&self, _: &Executor) -> Result<(), HostError> {
        Ok(())
    }
    async fn initialize(&self, _: &Executor) -> Result<(), HostError> {
        if self.slow_initialize {
            std::future::pending::<()>().await;
        }
        if self.fail_init { Err(HostError::new("init failed")) } else { Ok(()) }
    }
    async fn launch(&self, _: &Executor, _: &HostRef) -> Result<(), HostError> {
        if self.slow_launch {
            std::future::pending::<()>().await;
        }
        let left = self.fail_launch.load(Ordering::SeqCst);
        if left > 0 {
            self.fail_launch.store(left - 1, Ordering::SeqCst);
            return Err(HostError::new("launch failed"));
        }
        Ok(())
    }
    async fn close(&self, p: &Executor) -> Result<(), HostError> {
        if let Some(gate) = &self.close_gate {
            let _ = gate.subscribe().wait_for(|released| *released).await;
        }
        self.closed.lock().push(number(p));
        self.exits.notify_waiters();
        if self.fail_close { Err(HostError::new("close failed")) } else { Ok(()) }
    }
    fn failed(&self, _: &Executor, message: &'static str, error: HostError) {
        match message {
            "error closing process" => {
                assert!(self.fail_close);
                assert_eq!(error.to_string(), "close failed", "cleanup must preserve its cause");
            }
            "error joining process" => {
                assert_eq!(error.to_string(), "join failed", "join must preserve its cause");
            }
            "error initializing process" => {}
            _ => panic!("unexpected {message}: {error}"),
        }
    }
    fn cleanup_incomplete(&self, observation: PoolCleanupObservation) {
        self.cleanup_observed
            .as_ref()
            .expect("unexpected cleanup deadline miss")
            .send_replace(Some(observation));
    }
    async fn wait_exit(&self, p: &Executor) -> Result<(), HostError> {
        loop {
            let exited = self.exits.notified();
            tokio::pin!(exited);
            exited.as_mut().enable();
            if self.closed.lock().contains(&number(p)) {
                return Ok(());
            }
            exited.await;
        }
    }
    async fn join(&self, p: &Executor) -> Result<PoolJoinOutcome, HostError> {
        loop {
            let exited = self.exits.notified();
            tokio::pin!(exited);
            exited.as_mut().enable();
            if self.closed.lock().contains(&number(p)) {
                if self.join_receipt.as_ref().is_some_and(|receipt| !*receipt.borrow()) {
                    return Err(HostError::new("join failed"));
                }
                if let Some(gate) = &self.reap_gate {
                    let _ = gate.subscribe().wait_for(|reaped| *reaped).await;
                }
                return Ok(PoolJoinOutcome::Reaped);
            }
            exited.await;
        }
    }
    fn emit(&self, event: ProcessEvent, p: &Executor) {
        if let Some(pool) = self.reads_pool.get() {
            pool.processes();
        }
        self.events.lock().push((event, number(p)));
    }
}

fn pool(host: &Arc<Host>, idle: usize) -> ProcPool {
    let opts = PoolOptions {
        num_idle_processes: idle,
        initialize_timeout: Duration::from_secs(1),
        close_timeout: Duration::from_secs(1),
    };
    ProcPool::new(host.clone(), opts)
}

fn count(host: &Host, event: ProcessEvent) -> usize {
    host.events.lock().iter().filter(|(e, _)| *e == event).count()
}

// Catches: start() returning before the idle processes are warm, a launch that does not
// replace the process it took, or a close that leaves a process open.

#[tokio::test]
async fn keeps_the_idle_processes_warm_and_replaces_the_one_a_job_takes() {
    let host = Arc::new(Host::default());
    let pool = pool(&host, 2);
    pool.start().await.unwrap();
    assert_eq!(count(&host, ProcessEvent::Ready), 2);
    pool.launch_job("j1", job()).await.unwrap();
    assert_eq!(count(&host, ProcessEvent::JobLaunched), 1);
    tokio::time::sleep(REFILL_INTERVAL * 3).await;
    assert_eq!(count(&host, ProcessEvent::Ready), 3, "the taken process is replaced");
    pool.aclose().await;
    let mut closed = host.closed.lock().clone();
    closed.sort_unstable();
    closed.dedup();
    assert_eq!(closed, vec![0, 1, 2]);
    assert!(pool.processes().is_empty());
}

#[tokio::test]
async fn exited_warm_process_is_never_assigned_to_a_job() {
    let host = Arc::new(Host::default());
    let pool = pool(&host, 1);
    pool.start().await.unwrap();
    host.close(&host.process(0)).await.unwrap();
    tokio::time::timeout(Duration::from_secs(1), async {
        while numbers(&pool.processes()).contains(&0) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    pool.launch_job("after-exit", job()).await.unwrap();
    assert!(
        host.events
            .lock()
            .iter()
            .all(|(event, process)| *event != ProcessEvent::JobLaunched || *process != 0),
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
        pool.launch_job("j1", job()).await,
        Err(LaunchError::AcquireExhausted { attempts: MAX_ACQUIRE_ATTEMPTS })
    ));
    assert_eq!(pool.jobs_waiting(), 0);
    // Every failed process is closed and reaped.
    tokio::time::timeout(Duration::from_secs(1), async {
        while !pool.processes().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    pool.aclose().await;
}

// Catches: a cancelled launch leaving `waiting` raised (the pool kept that many extra processes
// warm) or the process it took open, neither warm nor running.

#[tokio::test]
async fn a_cancelled_launch_releases_its_wait_and_its_process() {
    let host = Arc::new(Host::default());
    let pool = pool(&host, 0);
    pool.start().await.unwrap();
    let waiting = tokio::spawn({
        let pool = pool.clone();
        async move { pool.launch_job("j1", job()).await }
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
        async move { pool.launch_job("j1", job()).await }
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    launching.abort();
    let _ = launching.await;
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(slow.closed.lock().contains(&0), "the process taken is closed");
    pool.aclose().await;
}

// Catches: the drain returning while a job's process still runs, or waiting on the idle ones.

#[tokio::test]
async fn join_jobs_waits_for_the_processes_running_a_job() {
    let host = Arc::new(Host::default());
    let pool = pool(&host, 1);
    pool.start().await.unwrap();
    pool.launch_job("j1", job()).await.unwrap();
    let joining = tokio::spawn({
        let pool = pool.clone();
        async move { pool.join_jobs().await }
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(!joining.is_finished(), "the job's process still runs");
    host.close(&host.process(0)).await.unwrap();
    tokio::time::timeout(Duration::from_secs(1), joining).await.unwrap().unwrap();
    pool.aclose().await;
}

// Catches: a failed launch not retried on a fresh process, or the failed one left open.

#[tokio::test]
async fn a_failed_launch_is_retried_on_a_fresh_process() {
    let host = Arc::new(Host { fail_launch: AtomicUsize::new(1), ..Host::default() });
    let pool = pool(&host, 1);
    pool.start().await.unwrap();
    pool.launch_job("j1", job()).await.unwrap();
    assert_eq!(count(&host, ProcessEvent::JobLaunched), 1);
    assert!(host.closed.lock().contains(&0), "the process that failed is closed");
    pool.aclose().await;
}

#[tokio::test(start_paused = true)]
async fn failed_launch_waiting_for_reaping_does_not_spawn_unbounded_replacements() {
    let reaped = watch::channel(false).0;
    let host = Arc::new(Host {
        fail_launch: AtomicUsize::new(1),
        reap_gate: Some(reaped.clone()),
        ..Host::default()
    });
    let pool = pool(&host, 2);
    pool.start().await.unwrap();
    let requests: Vec<_> = (0..3)
        .map(|n| {
            let pool = pool.clone();
            tokio::spawn(async move { pool.launch_job(&format!("job-{n}"), job()).await })
        })
        .collect();

    // Timer progress also proves that ready-process churn cannot starve the event loop.
    tokio::time::sleep(Duration::from_millis(1)).await;
    let created = host.created();
    assert!(created <= 5, "two warmed processes plus three admitted requests bound creation");
    for _ in 0..3 {
        tokio::time::sleep(REFILL_INTERVAL).await;
        assert_eq!(host.created(), created, "cleanup must stop replacement churn");
    }
    reaped.send_replace(true);
    for request in requests {
        tokio::time::timeout(Duration::from_secs(4), request).await.unwrap().unwrap().unwrap();
    }
    pool.aclose().await;
}

#[tokio::test]
async fn direct_launch_is_refused_after_drain_and_close() {
    let host = Arc::new(Host::default());
    let pool = pool(&host, 1);
    pool.start().await.unwrap();
    pool.join_jobs().await;
    assert!(matches!(
        pool.launch_job("after-drain", job()).await,
        Err(LaunchError::Admission(PoolAdmissionError::Draining))
    ));
    assert_eq!(host.created(), 1);
    pool.aclose().await;
    assert!(matches!(
        pool.launch_job("after-close", job()).await,
        Err(LaunchError::Admission(PoolAdmissionError::Closed))
    ));
    assert_eq!(host.created(), 1);
}

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
    assert_eq!(pool.start().await, Err(PoolStartError::WarmupTimedOut { ready: 0, required: 1 }));
    assert_eq!(count(&host, ProcessEvent::Ready), 0);
    pool.aclose().await;
}

#[tokio::test]
async fn cleanup_failure_is_observed_without_failing_public_close() {
    let host = Arc::new(Host { fail_close: true, ..Host::default() });
    let pool = pool(&host, 1);
    pool.start().await.unwrap();
    pool.aclose().await;
    assert_eq!(count(&host, ProcessEvent::CleanupIncomplete), 1);
    assert!(pool.processes().is_empty());
}

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
    assert_eq!(
        numbers(&pool.processes()),
        [0],
        "the resource stays owned after public close returns"
    );
    gate.send_replace(true);
    tokio::time::timeout(Duration::from_secs(1), async {
        while !pool.processes().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(count(&host, ProcessEvent::Closed), 1);
}

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
    assert_eq!(numbers(&pool.processes()), [0], "failed join cannot release process ownership");
    assert_eq!(count(&host, ProcessEvent::Closed), 0, "failed join is not cleaned");
    assert_eq!(observed.borrow().as_ref().unwrap().processes, [ProcessId(0)]);
    receipt.send_replace(true);
    tokio::time::timeout(REFILL_INTERVAL * 3, async {
        while !pool.processes().is_empty() {
            tokio::time::sleep(REFILL_INTERVAL).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(count(&host, ProcessEvent::Closed), 1);
}

#[tokio::test(start_paused = true)]
async fn os_exit_removes_warm_eligibility_before_the_reaping_receipt() {
    let reaped = watch::channel(false).0;
    let host = Arc::new(Host { reap_gate: Some(reaped.clone()), ..Host::default() });
    let pool = pool(&host, 1);
    pool.start().await.unwrap();
    host.close(&host.process(0)).await.unwrap();
    tokio::time::sleep(REFILL_INTERVAL).await;
    assert!(numbers(&pool.processes()).contains(&0), "ownership must remain until reaping");
    pool.launch_job("after-os-exit", job()).await.unwrap();
    let dead_process_was_launched = host
        .events
        .lock()
        .iter()
        .any(|(event, process)| *event == ProcessEvent::JobLaunched && *process == 0);
    reaped.send_replace(true);
    pool.aclose().await;
    assert!(!dead_process_was_launched, "OS-exited warm process was assigned before reaping");
}

// Catches: a host call made under the pool's lock. The Python host's emit takes the GIL, which a
// loop thread reading `ProcPool.processes` holds while it waits on that lock: a reap during that
// read deadlocked the worker.
#[test]
fn an_event_handler_can_read_the_pool() {
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        runtime.block_on(async {
            let host = Arc::new(Host::default());
            let pool = pool(&host, 1);
            let _ = host.reads_pool.set(pool.clone());
            pool.start().await.unwrap();
            pool.launch_job("j1", job()).await.unwrap();
            pool.aclose().await;
            assert!(count(&host, ProcessEvent::Closed) >= 1);
        });
        let _ = done.send(());
    });
    finished
        .recv_timeout(Duration::from_secs(10))
        .expect("the pool deadlocked on a host call made under its lock");
}

mod generated_sequences {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn generated_exit_retry_cancel_and_drain_sequences_release_every_process() {
        for seed in 0..288_u64 {
            let phase = std::cell::Cell::new("warmup");
            tokio::time::timeout(Duration::from_secs(30), exercise(seed, &phase))
                .await
                .unwrap_or_else(|_| panic!("sequence stalled: seed {seed}, phase {}", phase.get()));
        }
    }

    async fn exercise(seed: u64, phase: &std::cell::Cell<&'static str>) {
        let pool_size = 1 + (seed % 3) as usize;
        let failure_count = ((seed / 3) % 3) as usize;
        let cancel_launch = (seed / 9) % 2 == 1;
        let yield_count = (seed / 18) % 4;
        let reverse_close_order = (seed / 72) % 2 == 1;
        let cancel_first_close = (seed / 144) % 2 == 1;
        let reaped = watch::channel(false).0;
        let host = Arc::new(Host {
            fail_launch: AtomicUsize::new(failure_count),
            reap_gate: Some(reaped.clone()),
            ..Host::default()
        });
        let pool = pool(&host, pool_size);
        pool.start().await.unwrap();
        phase.set("observe warm exit");

        // Exit is visible before the receipt. The exited process remains owned during these
        // launches, but it must already have lost its eligibility to receive a job.
        host.close(&host.process(0)).await.unwrap();
        tokio::time::sleep(REFILL_INTERVAL).await;
        assert!(numbers(&pool.processes()).contains(&0), "lost the reaping owner: seed {seed}");
        let mut requests = Vec::new();
        for n in 0..3 {
            let pool = pool.clone();
            requests.push(tokio::spawn(async move {
                pool.launch_job(&format!("seed-{seed}-job-{n}"), job()).await
            }));
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
        phase.set("duplicate admission");
        assert!(
            matches!(
                pool.launch_job(&format!("seed-{seed}-job-0"), job()).await,
                Err(LaunchError::Admission(PoolAdmissionError::DuplicateJob))
            ),
            "duplicate admitted: seed {seed}"
        );

        // Sometimes cancel an admitted request while an earlier failed launch is still waiting
        // for its process's receipt. The other requests must continue on fresh processes.
        let cancel = cancel_launch && !requests[1].is_finished();
        if cancel {
            requests[1].abort();
        }
        for _ in 0..yield_count {
            tokio::task::yield_now().await;
        }
        phase.set("settle launches");
        reaped.send_replace(true);
        for (job, request) in requests.into_iter().enumerate() {
            let result =
                tokio::time::timeout(Duration::from_secs(4), request).await.unwrap_or_else(|_| {
                    panic!("admitted request did not settle: seed {seed} job {job}")
                });
            if cancel && job == 1 {
                assert!(result.unwrap_err().is_cancelled());
            } else {
                result.unwrap().unwrap();
            }
        }
        assert_eq!(pool.jobs_waiting(), 0, "seed {seed}");

        phase.set("drain");
        let drain = tokio::spawn({
            let pool = pool.clone();
            async move { pool.join_jobs().await }
        });
        tokio::time::sleep(Duration::from_millis(1)).await;
        assert!(!drain.is_finished(), "drain finished with running jobs: seed {seed}");
        assert!(matches!(
            pool.launch_job("during-drain", job()).await,
            Err(LaunchError::Admission(PoolAdmissionError::Draining))
        ));
        let mut processes = pool.processes();
        if reverse_close_order {
            processes.reverse();
        }
        for process in processes {
            host.close(&process).await.unwrap();
            for _ in 0..yield_count {
                tokio::task::yield_now().await;
            }
        }
        tokio::time::timeout(Duration::from_secs(4), drain).await.unwrap().unwrap();

        phase.set("close");
        let first_close = tokio::spawn({
            let pool = pool.clone();
            async move { pool.aclose().await }
        });
        if cancel_first_close {
            tokio::task::yield_now().await;
            first_close.abort();
        }
        tokio::time::timeout(Duration::from_secs(4), pool.aclose()).await.unwrap();
        let _ = first_close.await;
        assert!(pool.processes().is_empty(), "resources retained after cleanup: seed {seed}");

        phase.set("verify events");
        let mut retired = std::collections::BTreeSet::new();
        let mut launched = std::collections::BTreeSet::new();
        let events = host.events.lock();
        for &(event, process) in events.iter() {
            match event {
                ProcessEvent::Closed => {
                    assert!(retired.insert(process), "process released twice: seed {seed}");
                }
                ProcessEvent::Ready | ProcessEvent::JobLaunched => {
                    assert!(!retired.contains(&process), "released process revived: seed {seed}");
                    if event == ProcessEvent::JobLaunched {
                        assert_ne!(process, 0, "exited warm process launched: seed {seed}");
                        assert!(launched.insert(process), "process served two jobs: seed {seed}");
                    }
                }
                _ => {}
            }
        }
        assert_eq!(launched.len(), 3 - usize::from(cancel), "seed {seed}");
        assert_eq!(retired.len(), host.created(), "process leaked: seed {seed}");
    }
}
