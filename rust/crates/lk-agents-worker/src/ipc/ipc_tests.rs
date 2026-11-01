use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt, DuplexStream};
use tokio::sync::watch;
use tokio::time::Instant;

use super::child::watch_parent;
use super::ipc_channel::{MAGIC, VERSION, memory};
use super::supervisor::{Extra, MEMORY_WARN_COOLDOWN, MEMORY_WARN_RESET_DELTA_MB, WarnGate};
use super::*;
use lk_agents_net::clock::unix_now_ms;

mod supervision {
    // Supervision contracts over the socket: each pins a past bug of the process lifecycle.

    use super::*;

    /// Lets the supervisor's actor work through what is queued, on paused time.
    async fn settle() {
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }
    }

    fn short(close_timeout: f64) -> SupervisorOptions {
        SupervisorOptions { initialize_timeout: 1.0, close_timeout, ..opts() }
    }

    /// Initialized and ready to take a job.
    async fn ready(sup: &RunnerSupervisor, child: &Child) {
        let init = waits(sup).await;
        assert_eq!(recv(child).await, ToChild::Hello(hello()));
        write(child, &ToParent::Ready { error: None }).await.unwrap();
        init.await.unwrap().unwrap();
    }

    fn errors(host: &FakeHost) -> Vec<String> {
        let logs = host.logs.lock();
        logs.iter()
            .filter(|(level, ..)| *level == Level::Error)
            .map(|(_, m, _)| m.clone())
            .collect()
    }

    // Catches: a stopped runner that never ends released, or its missed cleanup reported twice.
    #[tokio::test(start_paused = true)]
    async fn a_runner_that_does_not_end_after_its_stop_is_reported_once_and_kept() {
        let (sup, host, _child) = setup(short(1.0));
        let killing = sup.clone();
        tokio::spawn(async move { killing.kill().await });
        tokio::time::sleep(Duration::from_secs(4)).await;
        settle().await;
        let incomplete = |host: &FakeHost| {
            host.messages()
                .iter()
                .filter(|m| *m == "incomplete process cleanup; retaining ownership")
                .count()
        };
        assert_eq!(incomplete(&host), 1, "the deadline reports once");
        assert!(!host.released.load(Ordering::SeqCst));
        finish(&sup, &host, -9).await;
        assert!(sup.wait_cleaned().await);
        assert!(host.released.load(Ordering::SeqCst));
    }

    // Catches: a kill requested while the launch was still running leaving the late runner running.
    #[tokio::test(start_paused = true)]
    async fn a_kill_during_the_launch_stops_the_runner_once_it_exists() {
        let host = Arc::new(FakeHost::default());
        let (parent, _child) = memory();
        let gate = Arc::new(tokio::sync::Semaphore::new(0));
        let executor = FakeExecutor { gate: Some(gate.clone()), ..executor(parent, &host) };
        let sup = start(executor, &host, short(60.0));
        let killing = sup.clone();
        tokio::spawn(async move { killing.kill().await });
        settle().await;
        assert_eq!(host.kills.load(Ordering::SeqCst), 0, "nothing to stop yet");
        gate.add_permits(1);
        settle().await;
        assert_eq!(host.kills.load(Ordering::SeqCst), 1, "the late runner is stopped");
    }

    // Catches: a launched process the host never initializes living forever.
    #[tokio::test(start_paused = true)]
    async fn a_launched_process_cannot_wait_indefinitely_for_the_host_to_initialize_it() {
        let (_sup, host, _child) = setup(short(60.0));
        tokio::time::sleep(Duration::from_millis(1100)).await;
        settle().await;
        assert_eq!(host.kills.load(Ordering::SeqCst), 1);
        assert!(errors(&host).contains(
            &"process launch did not enter initialization before its deadline".to_owned()
        ));
    }

    // Catches: a launch refused before any runner existed reported as a runner that ended.
    #[tokio::test(start_paused = true)]
    async fn launch_rejection_without_a_process_is_not_reported_as_os_exit_or_reaping() {
        let host = Arc::new(FakeHost::default());
        let (parent, _child) = memory();
        let refusal = Some("spawn refused".into());
        let sup = start(FakeExecutor { refusal, ..executor(parent, &host) }, &host, short(60.0));
        let initialized = sup.initialize(hello()).await;
        assert!(matches!(initialized, Err(InitError::Failed(cause)) if cause == "spawn refused"));
        assert!(!sup.wait_exit().await, "no exit");
        assert!(!sup.wait_cleaned().await, "nothing ran");
        assert!(host.released.load(Ordering::SeqCst));
    }

    // Catches: a protocol failure read after the native exit lost, or its cause replaced.
    #[tokio::test(start_paused = true)]
    async fn ipc_protocol_failure_after_native_exit_keeps_the_original_cause() {
        let (sup, host, mut child) = setup_raw(short(60.0)).await;
        let init = waits(&sup).await;
        raw_hello(&mut child).await;
        finish(&sup, &host, 0).await;
        // the socket can still hold the failure that preceded the exit: a truncated Ready frame
        child.write_all(&100u32.to_be_bytes()).await.unwrap();
        child.write_all(&[1, 2]).await.unwrap();
        drop(child);
        assert!(matches!(init.await.unwrap(), Err(InitError::Protocol(_))));
        assert!(sup.wait_cleaned().await);
        assert!(sup.completion().failure.is_some(), "a zero exit keeps the IPC failure");
    }

    // Catches: a written start-job request taken as admitted, or a second job taken while one is
    // pending.
    #[tokio::test(start_paused = true)]
    async fn start_job_admission_ack_correlates_acceptance_and_rejects_duplicate_work() {
        let (sup, _host, child) = setup(short(60.0));
        ready(&sup, &child).await;
        let first = sup.clone();
        let started = tokio::spawn(async move { first.start_job(job()).await });
        let ToChild::StartJob { request_id, .. } = recv_skipping_pings(&child).await else {
            panic!("no start-job request")
        };
        assert!(!started.is_finished(), "writing a request does not establish admission");
        assert!(matches!(sup.start_job(job()).await, Err(StartJobError::AlreadyRunning)));
        write(&child, &ToParent::JobAccepted { request_id, error: None }).await.unwrap();
        assert!(started.await.unwrap().is_ok());
    }

    // Catches: a job that failed on a protocol error reported clean because the process exited 0.
    #[tokio::test(start_paused = true)]
    async fn pending_job_failure_is_not_replaced_by_a_zero_exit() {
        let (sup, host, mut child) = setup_raw(short(60.0)).await;
        let init = waits(&sup).await;
        raw_hello(&mut child).await;
        let ready = postcard::to_allocvec(&ToParent::Ready { error: None }).unwrap();
        child.write_u32(ready.len() as u32).await.unwrap();
        child.write_all(&ready).await.unwrap();
        init.await.unwrap().unwrap();
        let first = sup.clone();
        let started = tokio::spawn(async move { first.start_job(job()).await });
        // a frame whose message tag no version knows
        child.write_all(&1u32.to_be_bytes()).await.unwrap();
        child.write_all(&[0xff]).await.unwrap();
        assert!(matches!(started.await.unwrap(), Err(StartJobError::Protocol(_))));
        settle().await;
        assert_eq!(host.kills.load(Ordering::SeqCst), 1);
        finish(&sup, &host, 0).await;
        assert!(sup.completion().failure.is_some());
    }

    // Catches: a version refusal read after the exit losing its typed cause.
    #[tokio::test(start_paused = true)]
    async fn a_version_refusal_queued_before_exit_keeps_its_typed_cause() {
        let (parent, mut child) = tokio::io::duplex(1 << 16);
        let (sup, host) = supervise(Channel::new(parent), short(60.0));
        let init = waits(&sup).await;
        finish(&sup, &host, 0).await;
        let theirs = VERSION + 1;
        child.write_all(&MAGIC).await.unwrap();
        child.write_all(&theirs.to_be_bytes()).await.unwrap();
        assert!(matches!(init.await.unwrap(), Err(InitError::Version(v)) if v == theirs));
        assert!(sup.wait_cleaned().await);
        assert!(sup.completion().failure.is_some());
    }

    // Catches: a child's own shutdown still judged by the health pings, or left unbounded.
    #[tokio::test(start_paused = true)]
    async fn child_initiated_shutdown_uses_the_exit_deadline_instead_of_health_pings() {
        let opts = SupervisorOptions { ping_interval: 0.25, ping_timeout: 0.5, ..short(2.0) };
        let (sup, host, child) = setup(opts);
        ready(&sup, &child).await;
        sup.supervise().await;
        write(&child, &ToParent::ShuttingDown).await.unwrap();
        tokio::time::sleep(Duration::from_secs(1)).await;
        settle().await;
        assert_eq!(
            host.kills.load(Ordering::SeqCst),
            0,
            "governed by close_timeout, not the pings"
        );
        tokio::time::sleep(Duration::from_millis(1100)).await;
        settle().await;
        assert_eq!(host.kills.load(Ordering::SeqCst), 1, "the exit phase is still bounded");
    }

    // Catches: the memory warning missing the job's baseline and growth.
    #[tokio::test(start_paused = true)]
    async fn memory_warnings_include_the_job_baseline() {
        let opts = SupervisorOptions { memory_warn_mb: 500.0, ..short(60.0) };
        let (sup, host, child) = setup(opts);
        host.samples.lock().extend([Some(100.0), Some(600.04)]);
        ready(&sup, &child).await;
        let first = sup.clone();
        let started = tokio::spawn(async move { first.start_job(job()).await });
        let ToChild::StartJob { request_id, .. } = recv_skipping_pings(&child).await else {
            panic!("no start-job request")
        };
        write(&child, &ToParent::JobAccepted { request_id, error: None }).await.unwrap();
        started.await.unwrap().unwrap();
        sup.supervise().await;
        let warning = "process memory usage is above the warning threshold (advisory only, the process will not be stopped)";
        tokio::time::timeout(Duration::from_secs(30), async {
            while host.find(warning).is_none() {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        })
        .await
        .unwrap();
        let (level, extra) = host.find(warning).unwrap();
        assert_eq!(level, Level::Warning);
        let field = |name: &str| extra.iter().find(|(key, _)| *key == name).unwrap().1.clone();
        assert_eq!(field("memory_usage_mb"), ExtraValue::Float(600.0));
        assert_eq!(field("baseline_memory_mb"), ExtraValue::Float(100.0));
        assert_eq!(field("growth_memory_mb"), ExtraValue::Float(500.0));
        assert_eq!(field("has_running_job"), ExtraValue::Bool(true));
    }

    /// What the child reads within `within`, pings skipped; `None` when it reads nothing else.
    async fn next_within(child: &Child, within: Duration) -> Option<ToChild> {
        tokio::time::timeout(within, recv_skipping_pings(child)).await.ok()
    }

    // Catches: a job whose protobuf or timestamps are invalid reaching the child.
    #[tokio::test(start_paused = true)]
    async fn invalid_job_protobuf_and_timestamps_are_refused_at_the_launch_boundary() {
        use super::supervisor::validate_start_job as validate;
        let refused = |job: &StartJob, field: &str| matches!(validate(job), Err(StartJobError::InvalidFields(reason)) if reason.contains(field));
        let mut invalid = StartJob { job: vec![0xff], ..job() };
        assert!(refused(&invalid, "protobuf"));
        invalid.job.clear();
        invalid.launched_at = f64::NAN;
        assert!(refused(&invalid, "launched_at"));
        invalid.launched_at = -1.0;
        assert!(validate(&invalid).is_err());
        invalid.launched_at = 0.0;
        assert!(validate(&invalid).is_ok());

        let (sup, _host, child) = setup(short(60.0));
        ready(&sup, &child).await;
        let refused = sup.start_job(StartJob { job: vec![0xff], ..job() }).await;
        assert!(matches!(refused, Err(StartJobError::InvalidFields(_))));
        assert_eq!(next_within(&child, Duration::from_secs(1)).await, None);
    }

    // Catches: a job sent to a child that has not answered `Ready`.
    #[tokio::test(start_paused = true)]
    async fn work_is_refused_until_a_compatible_ready_reply() {
        let (sup, _host, child) = setup(short(60.0));
        let _init = waits(&sup).await;
        assert_eq!(recv(&child).await, ToChild::Hello(hello()));
        assert!(matches!(sup.start_job(job()).await, Err(StartJobError::NotReady)));
        assert_eq!(next_within(&child, Duration::from_millis(500)).await, None);
    }

    // Catches: a child that refused the protocol version still given a job.
    #[tokio::test(start_paused = true)]
    async fn no_job_is_accepted_after_a_version_refusal() {
        let (parent, mut child) = tokio::io::duplex(1 << 16);
        let (sup, host) = supervise(Channel::new(parent), short(60.0));
        let init = waits(&sup).await;
        child.write_all(&MAGIC).await.unwrap();
        child.write_all(&(VERSION + 1).to_be_bytes()).await.unwrap();
        assert!(matches!(init.await.unwrap(), Err(InitError::Version(_))));
        assert!(sup.start_job(job()).await.is_err());
        settle().await;
        assert_eq!(host.kills.load(Ordering::SeqCst), 1, "the refusal starts the cleanup");
    }

    // Catches: a runner with no readable footprint (a thread, a process gone) sampled every interval.
    #[tokio::test(start_paused = true)]
    async fn memory_sampling_stops_when_the_runner_has_no_footprint() {
        let opts = SupervisorOptions { memory_warn_mb: 500.0, ping_timeout: 120.0, ..short(60.0) };
        let (sup, host, child) = setup(opts);
        // A retry would read the second sample and warn.
        host.samples.lock().extend([None, Some(600.0)]);
        ready(&sup, &child).await;
        sup.supervise().await;
        let drain = tokio::spawn(async move { while child.recv().await.is_ok() {} });
        tokio::time::sleep(Duration::from_secs(30)).await;
        settle().await;
        assert!(
            !host.messages().iter().any(|m| m.contains("above the warning threshold")),
            "retried"
        );
        drop(drain);
    }
}

fn job() -> StartJob {
    StartJob {
        job: vec![0x0a, 0x06, b'A', b'J', b'_', b'1', b'2', b'3'],
        name: "agent".into(),
        identity: "ident é".into(),
        metadata: r#"{"k": 1}"#.into(),
        url: "wss://x.livekit.cloud".into(),
        token: "tok".into(),
        worker_id: "AW_1".into(),
        fake_job: true,
        received_at: 1761500000.25,
        accepted_at: 1761500000.5,
        assigned_at: 0.0,
        launched_at: 1761500001.0,
        traceparent: None,
    }
}

fn hello() -> Hello {
    Hello { asyncio_debug: false, http_proxy: None, health_timeout_ms: None }
}

// ------------------------------------------------------------------------------- supervision

/// The supervisor's diagnostics and the runner it launched, as the test drives them.
struct FakeHost {
    logs: Mutex<Vec<(Level, String, Extra)>>,
    logged: tokio::sync::Notify,
    /// `stop()` calls.
    kills: AtomicUsize,
    /// A process ends when stopped (its end of the channel drops); a thread does not.
    forced: AtomicBool,
    samples: Mutex<VecDeque<Option<f64>>>,
    /// The child's end, dropped by a forced stop as the process's death would close it.
    child: Mutex<Option<Child>>,
    released: AtomicBool,
    exit: watch::Sender<Option<Exit>>,
}

impl Default for FakeHost {
    fn default() -> Self {
        Self {
            logs: Mutex::default(),
            logged: tokio::sync::Notify::new(),
            kills: AtomicUsize::new(0),
            forced: AtomicBool::new(true),
            samples: Mutex::default(),
            child: Mutex::default(),
            released: AtomicBool::new(false),
            exit: watch::channel(None).0,
        }
    }
}

impl SupervisorCallbacks for Arc<FakeHost> {
    fn log(&self, level: Level, msg: &str, extra: Extra) {
        self.logs.lock().push((level, msg.to_owned(), extra));
        self.logged.notify_one();
    }

    fn release(&self) {
        self.released.store(true, Ordering::SeqCst);
    }
}

struct FakeRunner(Arc<FakeHost>);

#[async_trait::async_trait]
impl Runner for FakeRunner {
    async fn finished(&self) -> Exit {
        let exit = self.0.exit.subscribe().wait_for(Option::is_some).await.ok().and_then(|e| *e);
        exit.unwrap()
    }

    fn stop(&self) -> bool {
        self.0.kills.fetch_add(1, Ordering::SeqCst);
        let forced = self.0.forced.load(Ordering::SeqCst);
        if forced {
            self.0.child.lock().take();
        }
        forced
    }

    fn memory_mb(&self) -> Option<f64> {
        self.0.samples.lock().pop_front().unwrap_or(Some(100.0))
    }
}

/// Launches the fake runner on `channel` once `gate` opens, or refuses with `refusal`.
struct FakeExecutor {
    launch: Mutex<Option<(FakeRunner, Channel<ToChild, ToParent>)>>,
    gate: Option<Arc<tokio::sync::Semaphore>>,
    refusal: Option<String>,
}

#[async_trait::async_trait]
impl JobExecutor for FakeExecutor {
    async fn launch(&self) -> Result<(Box<dyn Runner>, Channel<ToChild, ToParent>), String> {
        if let Some(gate) = &self.gate {
            let _ = gate.acquire().await;
        }
        if let Some(refusal) = &self.refusal {
            return Err(refusal.clone());
        }
        let (runner, channel) = self.launch.lock().take().unwrap();
        Ok((Box::new(runner), channel))
    }
}

impl FakeHost {
    fn messages(&self) -> Vec<String> {
        self.logs.lock().iter().map(|(_, m, _)| m.clone()).collect()
    }

    fn find(&self, msg: &str) -> Option<(Level, Extra)> {
        let logs = self.logs.lock();
        logs.iter().find(|(_, m, _)| m == msg).map(|(l, _, e)| (*l, e.clone()))
    }

    /// The runner ends with `code`, killed when a forced stop came first.
    fn finish(&self, code: i64) {
        let killed = self.forced.load(Ordering::SeqCst) && self.kills.load(Ordering::SeqCst) > 0;
        self.exit.send_replace(Some(Exit { code, killed }));
    }
}

/// The runner ended and the supervisor saw it.
async fn finish(sup: &RunnerSupervisor, host: &FakeHost, code: i64) {
    host.finish(code);
    assert!(sup.wait_exit().await);
}

fn opts() -> SupervisorOptions {
    SupervisorOptions {
        runner: "process",
        initialize_timeout: 10.0,
        close_timeout: 60.0,
        memory_warn_mb: 0.0,
        memory_limit_mb: 0.0,
        ping_interval: 2.5,
        ping_timeout: 90.0,
        high_ping_threshold: 0.5,
    }
}

/// Never initialized by the test: the startup deadline stays out of the way.
fn uninitialized() -> SupervisorOptions {
    SupervisorOptions { initialize_timeout: 1000.0, ..opts() }
}

fn executor(parent: Channel<ToChild, ToParent>, host: &Arc<FakeHost>) -> FakeExecutor {
    let launch = Mutex::new(Some((FakeRunner(host.clone()), parent)));
    FakeExecutor { launch, gate: None, refusal: None }
}

fn start(
    executor: FakeExecutor,
    host: &Arc<FakeHost>,
    opts: SupervisorOptions,
) -> RunnerSupervisor {
    let blocking_io = blocking_io();
    RunnerSupervisor::new(Arc::new(executor), host.clone(), opts, blocking_io)
}

fn supervise(
    parent: Channel<ToChild, ToParent>,
    opts: SupervisorOptions,
) -> (RunnerSupervisor, Arc<FakeHost>) {
    let host = Arc::new(FakeHost::default());
    (start(executor(parent, &host), &host, opts), host)
}

/// The job process's end of the supervised channel.
type Child = Channel<ToParent, ToChild>;

fn setup(opts: SupervisorOptions) -> (RunnerSupervisor, Arc<FakeHost>, Child) {
    let (parent, child) = memory();
    let (sup, host) = supervise(parent, opts);
    (sup, host, child)
}

/// A child writing raw bytes: its preamble is written, the parent's is read.
async fn setup_raw(opts: SupervisorOptions) -> (RunnerSupervisor, Arc<FakeHost>, DuplexStream) {
    let (parent, mut child) = tokio::io::duplex(1 << 16);
    let (sup, host) = supervise(Channel::new(parent), opts);
    child.write_all(&MAGIC).await.unwrap();
    child.write_all(&VERSION.to_be_bytes()).await.unwrap();
    child.read_exact(&mut [0; 8]).await.unwrap();
    (sup, host, child)
}

/// Reads the parent's `Hello` off a raw child.
async fn raw_hello(child: &mut DuplexStream) {
    let len = child.read_u32().await.unwrap() as usize;
    child.read_exact(&mut vec![0; len]).await.unwrap();
}

async fn recv(child: &Child) -> ToChild {
    child.recv().await.unwrap()
}

async fn write(child: &Child, msg: &ToParent) -> Result<(), SendError> {
    child.send(msg.clone()).await
}

/// Reads until a message other than a ping.
async fn recv_skipping_pings(child: &Child) -> ToChild {
    loop {
        let msg = recv(child).await;
        if !matches!(msg, ToChild::Ping { .. }) {
            return msg;
        }
    }
}

async fn waits(
    sup: &RunnerSupervisor,
) -> tokio::task::JoinHandle<Result<Option<Duration>, InitError>> {
    let sup = sup.clone();
    let init = tokio::spawn(async move { sup.initialize(hello()).await });
    tokio::task::yield_now().await;
    init
}

// Catches: the hello not sent first, or the elapsed time not logged.

#[tokio::test(start_paused = true)]
async fn initialize_sends_hello_and_logs_the_elapsed_time() {
    let (sup, host, child) = setup(opts());
    let init = waits(&sup).await;
    assert_eq!(recv(&child).await, ToChild::Hello(hello()));
    tokio::time::sleep(Duration::from_millis(1234)).await;
    write(&child, &ToParent::Ready { error: None }).await.unwrap();
    let elapsed = init.await.unwrap().unwrap().unwrap();
    assert!(elapsed >= Duration::from_millis(1234));
    assert_eq!(host.messages(), ["initializing process", "process initialized"]);
    let (_, extra) = host.find("process initialized").unwrap();
    assert_eq!(extra, vec![("elapsed_time", ExtraValue::Float(1.23))]);
}

// Catches: a child that failed to initialize, speaks another version, or answers out of turn
// being treated as ready.
#[tokio::test(start_paused = true)]
async fn initialize_raises_the_child_error_and_a_version_mismatch() {
    let (sup, _host, child) = setup(opts());
    let init = waits(&sup).await;
    write(&child, &ToParent::Ready { error: Some("no model".into()) }).await.unwrap();
    assert_eq!(
        init.await.unwrap().unwrap_err().to_string(),
        "process initialization failed: no model"
    );

    let (parent, mut child) = tokio::io::duplex(1 << 16);
    let (sup, _host) = supervise(Channel::new(parent), opts());
    let init = waits(&sup).await;
    child.write_all(&MAGIC).await.unwrap();
    child.write_all(&(VERSION + 1).to_be_bytes()).await.unwrap();
    assert!(matches!(init.await.unwrap(), Err(InitError::Version(v)) if v == VERSION + 1));

    let (sup, _host, child) = setup(opts());
    let init = waits(&sup).await;
    write(&child, &ToParent::ShutdownAck).await.unwrap();
    assert!(matches!(init.await.unwrap(), Err(InitError::Unexpected)));

    let (sup, _host, child) = setup(opts());
    drop(child);
    let initialized = sup.initialize(hello()).await;
    assert!(matches!(initialized, Err(InitError::Closed)), "{initialized:?}");

    // a frame the parent cannot read closes the connection, with why
    let (sup, host, mut child) = setup_raw(opts()).await;
    let init = waits(&sup).await;
    child.write_all(&[0, 0, 0, 1, 99]).await.unwrap();
    assert!(matches!(
        init.await.unwrap(),
        Err(InitError::Protocol(failure))
            if matches!(*failure, ProtocolError::Channel(ChannelError::Malformed(_)))
    ));
    assert!(
        host.messages().iter().any(|m| m.starts_with("invalid IPC message: malformed message"))
    );
}

// Catches: a process stuck initializing left running past initialize_timeout.

#[tokio::test(start_paused = true)]
async fn initialize_timeout_kills_the_process() {
    let (sup, host, _child) = setup(opts());
    let start = Instant::now();
    assert!(sup.initialize(hello()).await.unwrap().is_none());
    assert_eq!(start.elapsed(), Duration::from_secs(10));
    assert_eq!(host.kills.load(Ordering::SeqCst), 1);
    assert_eq!(host.messages(), ["initializing process", "stopping process"]);
}

// Catches: a ping cadence or pong bookkeeping that differs from the reference: the first ping at
// once, then one per interval, a late pong warned about.

#[tokio::test(start_paused = true)]
async fn pings_on_the_interval_and_warns_on_a_late_pong() {
    let (sup, host, child) = setup(uninitialized());
    let start = Instant::now();
    sup.supervise().await;
    let ToChild::Ping { sent_ms } = recv(&child).await else { panic!() };
    assert_eq!(start.elapsed(), Duration::ZERO);
    assert!((unix_now_ms() - sent_ms as i64).abs() < 5_000);
    // the next ping waits for this one's pong
    tokio::time::sleep(Duration::from_secs(4)).await;
    write(&child, &ToParent::Pong { sent_ms: (unix_now_ms() - 2_000) as u64 }).await.unwrap();
    assert!(matches!(recv(&child).await, ToChild::Ping { .. }));
    assert_eq!(start.elapsed(), Duration::from_millis(5000));

    while host.find("process is unresponsive").is_none() {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let (level, extra) = host.find("process is unresponsive").unwrap();
    assert_eq!(level, Level::Warning);
    let ExtraValue::Int(delay) = extra[0].1 else { panic!() };
    assert!(extra[0].0 == "delay" && delay >= 2_000);

    write(&child, &ToParent::Pong { sent_ms: unix_now_ms() as u64 }).await.unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(host.messages().iter().filter(|m| *m == "process is unresponsive").count(), 1);
}

// Catches: pings queued to a child that stopped answering until the channel is full, which then
// fails as a transport error instead of the pong deadline's diagnosis.
#[tokio::test(start_paused = true)]
async fn one_ping_is_in_flight_at_a_time() {
    let (sup, _host, child) = setup(uninitialized());
    sup.supervise().await;
    tokio::time::sleep(Duration::from_secs(60)).await;
    assert!(matches!(recv(&child).await, ToChild::Ping { .. }));
    let next = tokio::time::timeout(Duration::from_millis(1), child.recv()).await;
    assert!(next.is_err(), "a second ping went out unanswered: {next:?}");
}

// Catches: a hung child (no pongs) never being killed, or pongs not pushing the deadline back.

#[tokio::test(start_paused = true)]
async fn missing_pongs_kill_the_process_after_ping_timeout() {
    let (sup, host, child) = setup(uninitialized());
    let start = Instant::now();
    sup.supervise().await;
    // one pong at 60 s moves the deadline to 150 s
    tokio::time::sleep(Duration::from_secs(60)).await;
    write(&child, &ToParent::Pong { sent_ms: unix_now_ms() as u64 }).await.unwrap();
    let drain = tokio::spawn(async move { while child.recv().await.is_ok() {} });
    while host.kills.load(Ordering::SeqCst) == 0 {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let killed_at = start.elapsed();
    assert!(
        killed_at >= Duration::from_secs(150) && killed_at < Duration::from_secs(151),
        "{killed_at:?}"
    );
    let (level, _) = host.find("process is unresponsive, stopping it").unwrap();
    assert_eq!(level, Level::Error);
    drop(drain);
}

// Catches: a graceful close that kills a child which acked, or logs a clean exit as a failure.

#[tokio::test(start_paused = true)]
async fn shutdown_waits_for_the_ack_and_the_exit() {
    let (sup, host, child) = setup(opts());
    sup.supervise().await;
    let close = tokio::spawn({
        let sup = sup.clone();
        async move { sup.shutdown().await }
    });
    assert_eq!(recv_skipping_pings(&child).await, ToChild::Shutdown { reason: String::new() });
    write(&child, &ToParent::ShutdownAck).await.unwrap();
    write(&child, &ToParent::Exiting { reason: "shutdown".into() }).await.unwrap();
    write(&child, &ToParent::ShuttingDown).await.unwrap();
    child.close().await;
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(!close.is_finished(), "returns only once the process exited");
    finish(&sup, &host, 0).await;
    close.await.unwrap();
    assert_eq!(host.kills.load(Ordering::SeqCst), 0);
    let (level, extra) = host.find("process exiting").unwrap();
    assert_eq!((level, extra), (Level::Info, vec![("reason", ExtraValue::Str("shutdown".into()))]));
    assert!(host.logs.lock().iter().all(|(l, _, _)| *l != Level::Error), "{:?}", host.messages());
    assert!(host.released.load(Ordering::SeqCst));
}

// Catches: a child that never acks keeping the close waiting forever.

#[tokio::test(start_paused = true)]
async fn shutdown_kills_a_process_that_does_not_ack() {
    let (sup, host, child) = setup(uninitialized());
    *host.child.lock() = Some(child);
    sup.supervise().await;
    let start = Instant::now();
    let close = tokio::spawn({
        let sup = sup.clone();
        async move { sup.shutdown().await }
    });
    while host.kills.load(Ordering::SeqCst) == 0 {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(start.elapsed() >= Duration::from_secs(60));
    finish(&sup, &host, -9).await;
    close.await.unwrap();
    assert!(sup.killed());
    let messages = host.messages();
    assert!(messages.contains(&"process did not ack shutdown in time, stopping it".into()));
    assert!(!messages.iter().any(|m| m.starts_with("process exited with non-zero")));
    assert!(!messages.contains(&"process did not exit in time, stopping it".into()));
}

// Catches: a runner that cannot be forced (a job thread) killed without being asked to shut down,
// or given up on when its ack and exit are late; it is asked, reported and waited for.

#[tokio::test(start_paused = true)]
async fn a_runner_that_cannot_be_forced_is_asked_and_waited_for() {
    let (sup, host, child) =
        setup(SupervisorOptions { runner: "job thread", ping_timeout: 0.0, ..opts() });
    host.forced.store(false, Ordering::SeqCst);
    sup.supervise().await;
    let close = tokio::spawn({
        let sup = sup.clone();
        async move { sup.kill().await }
    });
    assert_eq!(recv_skipping_pings(&child).await, ToChild::Shutdown { reason: String::new() });
    tokio::time::sleep(Duration::from_secs(61)).await;
    let messages = host.messages();
    assert!(messages.contains(&"job thread did not ack shutdown in time, waiting for it".into()));
    assert!(messages.contains(&"waiting for job thread to end".into()), "{messages:?}");
    assert!(!messages.contains(&"stopping job thread".into()), "{messages:?}");
    assert!(!close.is_finished(), "a thread that has not exited is waited for");
    finish(&sup, &host, 0).await;
    close.await.unwrap();
    assert!(!sup.killed());
}

// Catches: the reader's end leaving a close waiting for an ack that cannot arrive (#5929), or an
// abnormal exit not reported.

#[tokio::test(start_paused = true)]
async fn the_socket_closing_resolves_the_shutdown_waits() {
    let (sup, host, child) = setup(opts());
    sup.supervise().await;
    let close = tokio::spawn({
        let sup = sup.clone();
        async move { sup.shutdown().await }
    });
    assert_eq!(recv_skipping_pings(&child).await, ToChild::Shutdown { reason: String::new() });
    drop(child);
    tokio::time::sleep(Duration::from_secs(1)).await;
    finish(&sup, &host, 1).await;
    tokio::time::timeout(Duration::from_secs(1), close).await.unwrap().unwrap();
    assert!(!host.messages().iter().any(|m| m.contains("did not ack")));
    assert_eq!(host.kills.load(Ordering::SeqCst), 1, "unexpected EOF starts native cleanup");
    assert!(
        host.messages().iter().any(|message| message.starts_with("invalid IPC message:")),
        "the transport failure remains observable"
    );
}

#[test]
fn memory_warning_is_rate_limited() {
    let t0 = Duration::ZERO;
    let mut gate = WarnGate::default();
    assert!(gate.should_warn(520.0, t0));
    assert!(!gate.should_warn(521.0, t0 + Duration::from_secs(5)));
    assert!(!gate.should_warn(522.0, t0 + Duration::from_secs(10)));
    assert!(gate.should_warn(522.0, t0 + MEMORY_WARN_COOLDOWN + Duration::from_secs(1)));
    // growth past the reset delta re-emits before the cooldown
    let mut gate = WarnGate::default();
    assert!(gate.should_warn(520.0, t0));
    assert!(gate.should_warn(520.0 + MEMORY_WARN_RESET_DELTA_MB, t0 + Duration::from_secs(5)));
}

// The native sampler runs on the process runtime; observe its completion on a real clock.
#[tokio::test]
async fn a_native_memory_sample_enforces_the_process_limit() {
    let (sup, host, _child) = setup(SupervisorOptions { memory_limit_mb: 500.0, ..opts() });
    host.samples.lock().push_back(Some(600.0));
    sup.supervise().await;
    let message = "process exceeded memory limit, stopping it";
    let (level, _) = tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let notified = host.logged.notified();
            if let Some(log) = host.find(message) {
                break log;
            }
            notified.await;
        }
    })
    .await
    .expect("native sampler reports the memory limit");
    assert_eq!(level, Level::Error);
    assert_eq!(host.kills.load(Ordering::SeqCst), 1);
}

// ------------------------------------------------------------------------------------- child

type WatchedChannel = (
    tokio::sync::mpsc::Receiver<ToChild>,
    lk_agents_abi::OwnedTask<Result<(), ChannelError>>,
    Channel<ToChild, ToParent>,
    Arc<AtomicUsize>,
);

/// The child's watch over a channel whose parent end the test drives.
fn watched(health_timeout: Option<Duration>) -> WatchedChannel {
    let (parent, child) = memory();
    let orphaned = Arc::new(AtomicUsize::new(0));
    let count = orphaned.clone();
    let (inbox, watching) = watch_parent(Arc::new(child), health_timeout, move || {
        count.fetch_add(1, Ordering::SeqCst);
    });
    (inbox, watching, parent, orphaned)
}

// Catches: an orphaned child (the parent stopped talking) never noticing.
#[tokio::test(start_paused = true)]
async fn the_child_calls_itself_orphaned_when_the_parent_goes_quiet() {
    let (mut inbox, watching, parent, orphaned) = watched(Some(Duration::from_secs(100)));
    let start = Instant::now();
    tokio::time::sleep(Duration::from_secs(60)).await;
    parent.send(ToChild::Ping { sent_ms: 1 }).await.unwrap();
    assert_eq!(inbox.recv().await, Some(ToChild::Ping { sent_ms: 1 }));
    assert_eq!(inbox.recv().await, None);
    let quiet = start.elapsed();
    assert!(quiet >= Duration::from_secs(160) && quiet < Duration::from_secs(161), "{quiet:?}");
    assert_eq!(orphaned.load(Ordering::SeqCst), 1);
    assert!(matches!(watching.await.unwrap(), Err(ChannelError::Io { .. })));

    let (mut inbox, _watching, _parent, _) = watched(None);
    let pending = tokio::time::timeout(Duration::from_secs(10_000), inbox.recv()).await;
    assert!(pending.is_err(), "no health check without a ping deadline");
}

// Catches: a malformed frame from the parent read past, instead of ending the connection.
#[tokio::test]
async fn the_child_stops_at_a_malformed_frame() {
    let (parent, child) = tokio::io::duplex(1 << 16);
    let child = Arc::new(Channel::<ToParent, ToChild>::new(child));
    let (mut inbox, watching) = watch_parent(child, None, || {});
    let mut parent = parent;
    parent.write_all(&MAGIC).await.unwrap();
    parent.write_all(&VERSION.to_be_bytes()).await.unwrap();
    parent.write_all(&[0, 0, 0, 1, 99]).await.unwrap();
    assert_eq!(inbox.recv().await, None);
    assert!(matches!(watching.await.unwrap(), Err(ChannelError::Malformed(_))));
}

// Catches: a busy child loop (an undrained inbox) suspending the health check.
#[tokio::test(start_paused = true)]
async fn a_full_child_inbox_does_not_suspend_the_health_check() {
    let (_inbox, watching, parent, orphaned) = watched(Some(Duration::from_millis(100)));
    for _ in 0..65 {
        parent.send(ToChild::Ping { sent_ms: 0 }).await.unwrap();
    }
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(watching.is_finished());
    assert_eq!(orphaned.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn supervisor_blocked_writer() {
    let (sup, host, mut child) = setup_raw(SupervisorOptions { ping_timeout: 1.0, ..opts() }).await;
    let initialize = waits(&sup).await;
    raw_hello(&mut child).await;
    let ready = postcard::to_allocvec(&ToParent::Ready { error: None }).unwrap();
    child.write_u32(ready.len() as u32).await.unwrap();
    child.write_all(&ready).await.unwrap();
    initialize.await.unwrap().unwrap();
    let mut large = job();
    large.metadata = "x".repeat(3 << 20);
    let pending = tokio::spawn({
        let sup = sup.clone();
        async move { sup.start_job(large).await }
    });
    sup.supervise().await;
    let killed = tokio::time::timeout(Duration::from_secs(2), async {
        while host.kills.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    assert!(killed.is_ok(), "a stalled peer must not suspend the health deadline");
    finish(&sup, &host, -9).await;
    assert!(pending.await.unwrap().is_err(), "a blocked write cannot fabricate child acceptance");
}

#[tokio::test(start_paused = true)]
async fn shutdown_ack_phase_deadline() {
    let (sup, host, child) = setup(uninitialized());
    // Poll shutdown through command admission without waiting for the process to exit.
    let _ = tokio::time::timeout(Duration::ZERO, sup.shutdown()).await;
    assert!(matches!(recv_skipping_pings(&child).await, ToChild::Shutdown { .. }));
    write(&child, &ToParent::ShutdownAck).await.unwrap();
    tokio::time::sleep(Duration::from_secs(61)).await;
    assert_eq!(
        host.kills.load(Ordering::SeqCst),
        1,
        "acknowledging shutdown must not remove the next phase's deadline"
    );
}

#[tokio::test(start_paused = true)]
async fn repeated_initialize_callers_are_refused_without_replacing_the_first_waiter() {
    let (sup, _host, child) = setup(opts());
    let first = waits(&sup).await;
    assert!(matches!(recv(&child).await, ToChild::Hello(_)));
    assert!(matches!(sup.initialize(hello()).await, Err(InitError::AlreadyInitializing)));
    write(&child, &ToParent::Ready { error: None }).await.unwrap();
    assert!(first.await.unwrap().unwrap().is_some());
    assert!(matches!(sup.initialize(hello()).await, Err(InitError::AlreadyInitializing)));
}

// Each test owns its blocking service so saturation cannot affect other tests.
fn blocking_io() -> lk_agents_net::blocking_io::BlockingIo {
    lk_agents_net::blocking_io::BlockingIo::with_limits("lk-test-io", 4, &[4]).unwrap()
}
