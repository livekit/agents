use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use lk_agents_net::lock::Mutex;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt, DuplexStream};
use tokio::time::Instant;

use super::supervisor::{
    CleanupObservation, Extra, MEMORY_WARN_COOLDOWN, MEMORY_WARN_RESET_DELTA_MB,
    SignalFailure, WarnGate,
};
use super::channel::{MAGIC, VERSION, memory};
use super::child::watch_parent;
use super::*;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::unix_now_ms;

mod supervision;

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
    Hello {
        asyncio_debug: false,
        http_proxy: None,
        health_timeout_ms: None,
    }
}

// ------------------------------------------------------------------------------- supervision

type Sample = Result<(f64, String), MemoryError>;

#[derive(Default)]
struct FakeHost {
    logs: Mutex<Vec<(Level, String, Extra)>>,
    logged: tokio::sync::Notify,
    alive: AtomicBool,
    kills: AtomicUsize,
    samples: Mutex<VecDeque<Sample>>,

    /// The child's end, dropped by a kill as the process's death would close it.
    child: Mutex<Option<Child>>,
    released: AtomicBool,
    /// The kill fails with this cause.
    kill_error: Mutex<Option<String>>,
    incomplete: Mutex<Vec<CleanupObservation>>,
    signal_failures: Mutex<Vec<SignalFailure>>,
}

impl ProcessCallbacks for Arc<FakeHost> {
    fn log(&self, level: Level, msg: &str, extra: Extra) {
        self.logs.lock().push((level, msg.to_owned(), extra));
        self.logged.notify_one();
    }

    fn log_exception(&self, msg: &str, err: HostError) {
        self.log(Level::Error, &format!("{msg}: {err}"), Vec::new());
    }

    fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    fn kill(&self) -> Result<(), String> {
        self.kills.fetch_add(1, Ordering::SeqCst);
        if let Some(cause) = self.kill_error.lock().clone() {
            return Err(cause);
        }
        self.alive.store(false, Ordering::SeqCst);
        self.child.lock().take();
        Ok(())
    }

    fn sample_memory(&self) -> Sample {
        self.samples.lock().pop_front().unwrap_or(Ok((100.0, "uss".into())))
    }

    fn release(&self) {
        self.released.store(true, Ordering::SeqCst);
    }

    fn incomplete_cleanup(&self, observation: CleanupObservation) {
        self.incomplete.lock().push(observation);
    }

    fn signal_failed(&self, failure: SignalFailure) {
        self.signal_failures.lock().push(failure);
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
}

fn opts() -> SupervisorOptions {
    SupervisorOptions {
        // no signal: the fake host stands for the process
        pid: 0,
        initialize_timeout: 10.0,
        close_timeout: 60.0,
        memory_warn_mb: 0.0,
        memory_limit_mb: 0.0,
        ping_interval: 2.5,
        ping_timeout: 90.0,
        high_ping_threshold: 0.5,
        killable: true,
    }
}

/// The job process's end of the supervised channel.
type Child = Channel<ToParent, ToChild>;

fn supervise(parent: Channel<ToChild, ToParent>, opts: SupervisorOptions) -> (ProcSupervisor, Arc<FakeHost>) {
    let host = Arc::new(FakeHost::default());
    host.alive.store(true, Ordering::SeqCst);
    (ProcSupervisor::new(parent, host.clone(), opts, crate::test_support::blocking_io()), host)
}

fn setup(opts: SupervisorOptions) -> (ProcSupervisor, Arc<FakeHost>, Child) {
    let (parent, child) = memory();
    let (sup, host) = supervise(parent, opts);
    (sup, host, child)
}

/// A child writing raw bytes: its preamble is written, the parent's is read.
async fn setup_raw(opts: SupervisorOptions) -> (ProcSupervisor, Arc<FakeHost>, DuplexStream) {
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
    sup: &ProcSupervisor,
) -> tokio::task::JoinHandle<Result<Option<Duration>, InitError>> {
    let sup = sup.clone();
    let init = tokio::spawn(async move { sup.initialize(hello()).await });
    tokio::task::yield_now().await;
    init
}

// Catches: the hello not sent first, or the elapsed time not logged.

#[tokio::test(start_paused = true)]
async fn initialize_sends_hello_and_logs_the_elapsed_time() {
    let (sup, host, mut child) = setup(opts());
    let init = waits(&sup).await;
    assert_eq!(recv(&mut child).await, ToChild::Hello(hello()));
    tokio::time::sleep(Duration::from_millis(1234)).await;
    write(&mut child, &ToParent::Ready { error: None }).await.unwrap();
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
    let (sup, _host, mut child) = setup(opts());
    let init = waits(&sup).await;
    write(&mut child, &ToParent::Ready { error: Some("no model".into()) }).await.unwrap();
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

    let (sup, _host, mut child) = setup(opts());
    let init = waits(&sup).await;
    write(&mut child, &ToParent::ShutdownAck).await.unwrap();
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
    assert!(sup.killed());
    assert_eq!(host.messages(), ["initializing process", "killing process"]);
}

// Catches: a ping cadence or pong bookkeeping that differs from the reference: the first ping at
// once, then one per interval, a late pong warned about.

#[tokio::test(start_paused = true)]
async fn pings_on_the_interval_and_warns_on_a_late_pong() {
    let (sup, host, mut child) = setup(opts());
    let start = Instant::now();
    sup.supervise().await;
    let ToChild::Ping { sent_ms } = recv(&mut child).await else { panic!() };
    assert_eq!(start.elapsed(), Duration::ZERO);
    assert!((unix_now_ms() - sent_ms as i64).abs() < 5_000);
    // the next ping waits for this one's pong
    tokio::time::sleep(Duration::from_secs(4)).await;
    write(&mut child, &ToParent::Pong { sent_ms: (unix_now_ms() - 2_000) as u64 }).await.unwrap();
    assert!(matches!(recv(&mut child).await, ToChild::Ping { .. }));
    assert_eq!(start.elapsed(), Duration::from_millis(5000));

    while host.find("process is unresponsive").is_none() {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let (level, extra) = host.find("process is unresponsive").unwrap();
    assert_eq!(level, Level::Warning);
    let ExtraValue::Int(delay) = extra[0].1 else { panic!() };
    assert!(extra[0].0 == "delay" && delay >= 2_000);

    write(&mut child, &ToParent::Pong { sent_ms: unix_now_ms() as u64 }).await.unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(host.messages().iter().filter(|m| *m == "process is unresponsive").count(), 1);
}

// Catches: pings queued to a child that stopped answering until the channel is full, which then
// fails as a transport error instead of the pong deadline's diagnosis.
#[tokio::test(start_paused = true)]
async fn one_ping_is_in_flight_at_a_time() {
    let (sup, _host, child) = setup(opts());
    sup.supervise().await;
    tokio::time::sleep(Duration::from_secs(60)).await;
    assert!(matches!(recv(&child).await, ToChild::Ping { .. }));
    let next = tokio::time::timeout(Duration::from_millis(1), child.recv()).await;
    assert!(next.is_err(), "a second ping went out unanswered: {next:?}");
}

// Catches: a hung child (no pongs) never being killed, or pongs not pushing the deadline back.

#[tokio::test(start_paused = true)]
async fn missing_pongs_kill_the_process_after_ping_timeout() {
    let (sup, host, mut child) = setup(opts());
    let start = Instant::now();
    sup.supervise().await;
    // one pong at 60 s moves the deadline to 150 s
    tokio::time::sleep(Duration::from_secs(60)).await;
    write(&mut child, &ToParent::Pong { sent_ms: unix_now_ms() as u64 }).await.unwrap();
    let drain = tokio::spawn(async move { while child.recv().await.is_ok() {} });
    while host.kills.load(Ordering::SeqCst) == 0 {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let killed_at = start.elapsed();
    assert!(
        killed_at >= Duration::from_secs(150) && killed_at < Duration::from_secs(151),
        "{killed_at:?}"
    );
    let (level, _) = host.find("process is unresponsive, killing process").unwrap();
    assert_eq!(level, Level::Error);
    drop(drain);
}

// Catches: a graceful close that kills a child which acked, or logs a clean exit as a failure.

#[tokio::test(start_paused = true)]
async fn shutdown_waits_for_the_ack_and_the_exit() {
    let (sup, host, mut child) = setup(opts());
    sup.supervise().await;
    let close = tokio::spawn({
        let sup = sup.clone();
        async move { sup.shutdown().await }
    });
    assert_eq!(recv_skipping_pings(&mut child).await, ToChild::Shutdown { reason: String::new() });
    write(&mut child, &ToParent::ShutdownAck).await.unwrap();
    write(&mut child, &ToParent::Exiting { reason: "shutdown".into() }).await.unwrap();
    write(&mut child, &ToParent::ShuttingDown).await.unwrap();
    child.close().await;
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(!close.is_finished(), "returns only once the process exited");
    sup.exited(Some(0)).await;
    sup.reaped(0).await;
    close.await.unwrap();
    assert_eq!(host.kills.load(Ordering::SeqCst), 0);
    let (level, extra) = host.find("process exiting").unwrap();
    assert_eq!((level, extra), (Level::Info, vec![("reason", ExtraValue::Str("shutdown".into()))]));
    assert!(
        host.logs.lock().iter().all(|(l, _, _)| *l != Level::Error),
        "{:?}",
        host.messages()
    );
    assert!(host.released.load(Ordering::SeqCst));
}

// Catches: a child that never acks keeping the close waiting forever.

#[tokio::test(start_paused = true)]
async fn shutdown_kills_a_process_that_does_not_ack() {
    let (sup, host, child) = setup(opts());
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
    sup.exited(Some(-9)).await;
    sup.reaped(-9).await;
    close.await.unwrap();
    let messages = host.messages();
    assert!(messages.contains(&"process did not ack shutdown in time, killing process".into()));
    assert!(!messages.iter().any(|m| m.starts_with("process exited with non-zero")));
    assert!(!messages.contains(&"process did not exit in time, killing process".into()));
}

// Catches: a job thread killed (it cannot be) or given up on when its ack and exit are late; it is
// reported and waited for.

#[tokio::test(start_paused = true)]
async fn a_job_thread_that_does_not_ack_is_waited_for_not_killed() {
    let (sup, host, _child) =
        setup(SupervisorOptions { killable: false, ping_timeout: 0.0, ..opts() });
    sup.supervise().await;
    let close = tokio::spawn({
        let sup = sup.clone();
        async move { sup.kill().await }
    });
    tokio::time::sleep(Duration::from_secs(61)).await;
    assert!(host.messages().contains(&"job did not ack shutdown in time".into()));
    assert!(!close.is_finished(), "a thread that has not exited is waited for");
    sup.exited(Some(0)).await;
    sup.reaped(0).await;
    close.await.unwrap();
    assert_eq!(host.kills.load(Ordering::SeqCst), 0);
    assert!(!host.messages().iter().any(|m| m.contains("unresponsive, killing")));
}

// Catches: the reader's end leaving a close waiting for an ack that cannot arrive (#5929), or an
// abnormal exit not reported.

#[tokio::test(start_paused = true)]
async fn the_socket_closing_resolves_the_shutdown_waits() {
    let (sup, host, mut child) = setup(opts());
    sup.supervise().await;
    let close = tokio::spawn({
        let sup = sup.clone();
        async move { sup.shutdown().await }
    });
    assert_eq!(recv_skipping_pings(&mut child).await, ToChild::Shutdown { reason: String::new() });
    drop(child);
    tokio::time::sleep(Duration::from_secs(1)).await;
    sup.exited(Some(1)).await;
    sup.reaped(1).await;
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
    host.samples.lock().push_back(Ok((600.0, "pss".into())));
    sup.supervise().await;
    let message = "job process exceeded memory limit, killing it";
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

/// The child's watch over a channel whose parent end the test drives.
fn watched(
    health_timeout: Option<Duration>,
) -> (
    tokio::sync::mpsc::Receiver<ToChild>,
    lk_agents_abi::OwnedTask<Result<(), ChannelError>>,
    Channel<ToChild, ToParent>,
    Arc<AtomicUsize>,
) {
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
    assert!(!pending.is_finished(), "a blocked write cannot fabricate child acceptance");
    sup.exited(Some(-9)).await;
    sup.reaped(-9).await;
    assert!(pending.await.unwrap().is_err());
}

#[tokio::test(start_paused = true)]
async fn shutdown_ack_phase_deadline() {
    let (sup, host, mut child) = setup(opts());
    // Poll shutdown through command admission without waiting for the process to exit.
    let _ = tokio::time::timeout(Duration::ZERO, sup.shutdown()).await;
    assert!(matches!(recv_skipping_pings(&mut child).await, ToChild::Shutdown { .. }));
    write(&mut child, &ToParent::ShutdownAck).await.unwrap();
    tokio::time::sleep(Duration::from_secs(61)).await;
    assert_eq!(
        host.kills.load(Ordering::SeqCst),
        1,
        "acknowledging shutdown must not remove the next phase's deadline"
    );
}

#[tokio::test(start_paused = true)]
async fn repeated_initialize_callers_are_refused_without_replacing_the_first_waiter() {
    let (sup, _host, mut child) = setup(opts());
    let first = waits(&sup).await;
    assert!(matches!(recv(&mut child).await, ToChild::Hello(_)));
    assert!(matches!(sup.initialize(hello()).await, Err(InitError::AlreadyInitializing)));
    write(&mut child, &ToParent::Ready { error: None }).await.unwrap();
    assert!(first.await.unwrap().unwrap().is_some());
    assert!(matches!(sup.initialize(hello()).await, Err(InitError::AlreadyInitializing)));
    sup.exited(Some(0)).await;
    sup.reaped(0).await;
}

#[tokio::test(start_paused = true)]
async fn native_exit_wakes_eligibility_before_the_reaping_receipt_releases_the_owner() {
    let (sup, host, _child) = setup(opts());
    sup.exited(Some(0)).await;
    assert!(sup.wait_exit().await);
    assert!(!host.released.load(Ordering::SeqCst));
    assert!(tokio::time::timeout(Duration::ZERO, sup.wait_cleaned()).await.is_err());
    sup.reaped(0).await;
    assert!(sup.wait_cleaned().await);
    assert!(host.released.load(Ordering::SeqCst));
}
