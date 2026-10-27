use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncWriteExt, DuplexStream};
use tokio::net::UnixStream;
use tokio::time::Instant;

use super::supervisor::{
    Extra, KILL_GRACE, MEMORY_WARN_COOLDOWN, MEMORY_WARN_RESET_DELTA_MB, WarnGate, time_ms,
};
use super::wire::{MAX_FRAME, decode, encode, read, write};
use super::*;

// ---------------------------------------------------------------------------------------- wire

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
    }
}

fn hello() -> Hello {
    Hello {
        version: PROTOCOL_VERSION,
        asyncio_debug: false,
        http_proxy: None,
        health_timeout_ms: None,
    }
}

// Catches: a message that does not survive its own encoding (a field lost, reordered or widened).
#[test]
fn every_message_round_trips() {
    let to_child = [
        ToChild::Hello(Hello {
            version: PROTOCOL_VERSION,
            asyncio_debug: true,
            http_proxy: Some("http://proxy:8080".into()),
            health_timeout_ms: Some(70_000),
        }),
        ToChild::Hello(hello()),
        ToChild::Ping { sent_ms: 1761500000123 },
        ToChild::StartJob(Box::new(job())),
        ToChild::StartJob(Box::default()),
        ToChild::Shutdown { reason: String::new() },
        ToChild::Shutdown { reason: "room disconnected".into() },
        ToChild::DumpStacks,
    ];
    for msg in to_child {
        assert_eq!(decode::<ToChild>(&encode(&msg)).unwrap(), msg);
    }
    let to_parent = [
        ToParent::Ready { version: PROTOCOL_VERSION, error: None },
        ToParent::Ready { version: 7, error: Some("boom ✗".into()) },
        ToParent::Pong { sent_ms: u64::MAX },
        ToParent::ShutdownAck,
        ToParent::ShuttingDown,
        ToParent::Exiting { reason: "shutdown".into() },
    ];
    for msg in to_parent {
        assert_eq!(decode::<ToParent>(&encode(&msg)).unwrap(), msg);
    }
}

// Catches: a frame with trailing bytes, a truncated one or an unknown variant read as a message.
#[test]
fn malformed_messages_are_errors() {
    let mut extra = encode(&ToParent::ShutdownAck);
    extra.push(0);
    assert!(matches!(decode::<ToParent>(&extra), Err(WireError::Malformed(_))));
    let pong = encode(&ToParent::Pong { sent_ms: 1 << 40 });
    assert!(matches!(decode::<ToParent>(&pong[..pong.len() - 1]), Err(WireError::Malformed(_))));
    assert!(matches!(decode::<ToParent>(&[99]), Err(WireError::Malformed(_))));
    assert!(matches!(decode::<ToChild>(&[]), Err(WireError::Malformed(_))));
}

// Catches: a length prefix trusted past the limit (an allocation the size of a bad header), or
// the framing out of step with the reader.
#[tokio::test]
async fn frames_are_length_prefixed_and_bounded() {
    let (mut a, mut b) = tokio::io::duplex(1 << 16);
    write(&mut a, &ToChild::Ping { sent_ms: 5 }).await.unwrap();
    write(&mut a, &ToChild::DumpStacks).await.unwrap();
    assert_eq!(read::<ToChild, _>(&mut b).await.unwrap(), ToChild::Ping { sent_ms: 5 });
    assert_eq!(read::<ToChild, _>(&mut b).await.unwrap(), ToChild::DumpStacks);

    a.write_all(&((MAX_FRAME + 1) as u32).to_be_bytes()).await.unwrap();
    assert!(
        matches!(read::<ToChild, _>(&mut b).await, Err(WireError::TooLarge(n)) if n == MAX_FRAME + 1)
    );

    let big = StartJob { job: vec![0; MAX_FRAME], ..job() };
    assert!(
        write(&mut a, &ToChild::StartJob(Box::new(big))).await.is_err(),
        "an oversized message is not sent"
    );
    drop(a);
    assert!(matches!(read::<ToChild, _>(&mut b).await, Err(WireError::Io(_))));
}

// ------------------------------------------------------------------------------- supervision

type Sample = Result<(f64, String), MemoryError<String>>;

#[derive(Default)]
struct FakeHost {
    logs: Mutex<Vec<(Level, String, Extra)>>,
    alive: AtomicBool,
    kills: AtomicUsize,
    samples: Mutex<VecDeque<Sample>>,
    /// The child's end, dropped by a kill as the process's death would close it.
    child: Mutex<Option<DuplexStream>>,
    released: AtomicBool,
}

impl ProcHost for Arc<FakeHost> {
    type Error = String;

    fn log(&self, level: Level, msg: &str, extra: Extra) {
        self.logs.lock().unwrap().push((level, msg.to_owned(), extra));
    }

    fn log_exception(&self, msg: &str, err: String) {
        self.log(Level::Error, &format!("{msg}: {err}"), Vec::new());
    }

    fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    fn kill(&self) {
        self.kills.fetch_add(1, Ordering::SeqCst);
        self.alive.store(false, Ordering::SeqCst);
        self.child.lock().unwrap().take();
    }

    fn sample_memory(&self) -> Sample {
        self.samples.lock().unwrap().pop_front().unwrap_or(Ok((100.0, "uss".into())))
    }

    fn release(&self) {
        self.released.store(true, Ordering::SeqCst);
    }
}

impl FakeHost {
    fn messages(&self) -> Vec<String> {
        self.logs.lock().unwrap().iter().map(|(_, m, _)| m.clone()).collect()
    }

    fn find(&self, msg: &str) -> Option<(Level, Extra)> {
        let logs = self.logs.lock().unwrap();
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

fn setup(opts: SupervisorOptions) -> (ProcSupervisor, Arc<FakeHost>, DuplexStream) {
    let (parent, child) = tokio::io::duplex(1 << 16);
    let host = Arc::new(FakeHost::default());
    host.alive.store(true, Ordering::SeqCst);
    (ProcSupervisor::new(parent, host.clone(), opts), host, child)
}

async fn recv(child: &mut DuplexStream) -> ToChild {
    read(child).await.unwrap()
}

/// Reads until a message other than a ping.
async fn recv_skipping_pings(child: &mut DuplexStream) -> ToChild {
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
    write(&mut child, &ToParent::Ready { version: PROTOCOL_VERSION, error: None }).await.unwrap();
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
    write(
        &mut child,
        &ToParent::Ready { version: PROTOCOL_VERSION, error: Some("no model".into()) },
    )
    .await
    .unwrap();
    assert_eq!(
        init.await.unwrap().unwrap_err().to_string(),
        "process initialization failed: no model"
    );

    let (sup, _host, mut child) = setup(opts());
    let init = waits(&sup).await;
    write(&mut child, &ToParent::Ready { version: PROTOCOL_VERSION + 1, error: None })
        .await
        .unwrap();
    assert!(matches!(init.await.unwrap(), Err(InitError::Version(v)) if v == PROTOCOL_VERSION + 1));

    let (sup, _host, mut child) = setup(opts());
    let init = waits(&sup).await;
    write(&mut child, &ToParent::ShutdownAck).await.unwrap();
    assert!(matches!(init.await.unwrap(), Err(InitError::Unexpected)));

    let (sup, _host, child) = setup(opts());
    drop(child);
    assert!(matches!(sup.initialize(hello()).await, Err(InitError::Closed)));

    // a frame the parent cannot read closes the connection, with why
    let (sup, host, mut child) = setup(opts());
    let init = waits(&sup).await;
    child.write_all(&[0, 0, 0, 1, 99]).await.unwrap();
    assert!(matches!(init.await.unwrap(), Err(InitError::Closed)));
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
    assert!((time_ms() - sent_ms as i64).abs() < 5_000);
    assert!(matches!(recv(&mut child).await, ToChild::Ping { .. }));
    assert_eq!(start.elapsed(), Duration::from_millis(2500));

    write(&mut child, &ToParent::Pong { sent_ms: (time_ms() - 2_000) as u64 }).await.unwrap();
    while host.find("process is unresponsive").is_none() {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let (level, extra) = host.find("process is unresponsive").unwrap();
    assert_eq!(level, Level::Warning);
    let ExtraValue::Int(delay) = extra[0].1 else { panic!() };
    assert!(extra[0].0 == "delay" && delay >= 2_000);

    write(&mut child, &ToParent::Pong { sent_ms: time_ms() as u64 }).await.unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(host.messages().iter().filter(|m| *m == "process is unresponsive").count(), 1);
}

// Catches: a hung child (no pongs) never being killed, or pongs not pushing the deadline back.
#[tokio::test(start_paused = true)]
async fn missing_pongs_kill_the_process_after_ping_timeout() {
    let (sup, host, mut child) = setup(opts());
    let start = Instant::now();
    sup.supervise().await;
    // one pong at 60 s moves the deadline to 150 s
    tokio::time::sleep(Duration::from_secs(60)).await;
    write(&mut child, &ToParent::Pong { sent_ms: time_ms() as u64 }).await.unwrap();
    let drain = tokio::spawn(async move { while read::<ToChild, _>(&mut child).await.is_ok() {} });
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
    drop(child);
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(!close.is_finished(), "returns only once the process exited");
    sup.exited(Some(0)).await;
    close.await.unwrap();
    assert_eq!(host.kills.load(Ordering::SeqCst), 0);
    let (level, extra) = host.find("process exiting").unwrap();
    assert_eq!((level, extra), (Level::Info, vec![("reason", ExtraValue::Str("shutdown".into()))]));
    assert!(host.logs.lock().unwrap().iter().all(|(l, _, _)| *l != Level::Error));
    assert!(host.released.load(Ordering::SeqCst));
}

// Catches: a child that never acks keeping the close waiting forever.
#[tokio::test(start_paused = true)]
async fn shutdown_kills_a_process_that_does_not_ack() {
    let (sup, host, child) = setup(opts());
    *host.child.lock().unwrap() = Some(child);
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
    tokio::time::timeout(Duration::from_secs(1), close).await.unwrap().unwrap();
    assert!(!host.messages().iter().any(|m| m.contains("did not ack")));
    let (level, _) = host.find("process exited with non-zero exit code 1").unwrap();
    assert_eq!(level, Level::Error);
}

#[test]
fn memory_warning_is_rate_limited() {
    let t0 = Instant::now();
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

// Catches: warnings on every sample, a missing advisory note, or wrong diagnostic fields.
#[tokio::test(start_paused = true)]
async fn memory_monitor_warns_with_the_baseline_and_growth() {
    let (sup, host, mut child) = setup(SupervisorOptions { memory_warn_mb: 500.0, ..opts() });
    host.samples.lock().unwrap().extend([
        Ok((100.0, "uss".into())),
        Ok((600.04, "uss".into())),
        Ok((610.0, "uss".into())),
        Ok((700.0, "uss".into())),
    ]);
    sup.start_job(job()).await;
    sup.supervise().await;
    let drain = tokio::spawn(async move { while read::<ToChild, _>(&mut child).await.is_ok() {} });
    tokio::time::sleep(Duration::from_secs(16)).await;
    let warning = "job process memory usage is above the warning threshold \
                   (advisory only, the process will not be terminated)";
    let warnings: Vec<_> =
        host.logs.lock().unwrap().iter().filter(|(_, m, _)| m == warning).cloned().collect();
    assert_eq!(warnings.len(), 2, "{:?}", host.messages());
    let (level, _, extra) = &warnings[0];
    assert_eq!(*level, Level::Warning);
    assert_eq!(
        extra,
        &vec![
            ("memory_usage_mb", ExtraValue::Float(600.0)),
            ("memory_metric", ExtraValue::Str("uss".into())),
            ("memory_warn_mb", ExtraValue::Float(500.0)),
            ("memory_limit_mb", ExtraValue::Float(0.0)),
            ("uptime", ExtraValue::Float(5.0)),
            ("has_running_job", ExtraValue::Bool(true)),
            ("baseline_memory_mb", ExtraValue::Float(100.0)),
            ("growth_memory_mb", ExtraValue::Float(500.0)),
        ]
    );
    assert_eq!(host.kills.load(Ordering::SeqCst), 0);
    drop(drain);
}

#[tokio::test(start_paused = true)]
async fn memory_limit_kills_and_unavailable_stops_monitoring() {
    let (sup, host, _child) = setup(SupervisorOptions { memory_limit_mb: 500.0, ..opts() });
    host.samples.lock().unwrap().push_back(Ok((600.0, "pss".into())));
    sup.supervise().await;
    tokio::time::sleep(Duration::from_secs(1)).await;
    let (level, _) = host.find("job process exceeded memory limit, killing it").unwrap();
    assert_eq!(level, Level::Error);
    assert_eq!(host.kills.load(Ordering::SeqCst), 1);

    let (sup, host, _child) = setup(SupervisorOptions { memory_warn_mb: 500.0, ..opts() });
    host.samples.lock().unwrap().extend([
        Err(MemoryError::Failed("oops".into())),
        Err(MemoryError::Unavailable("process no longer exists".into())),
    ]);
    sup.supervise().await;
    tokio::time::sleep(Duration::from_secs(30)).await;
    let messages = host.messages();
    assert!(messages.contains(&"Error in memory monitoring task: oops".into()));
    assert!(
        messages
            .contains(&"failed to get memory info for process: process no longer exists".into())
    );
    assert_eq!(host.samples.lock().unwrap().len(), 0);

    // an interruption (#4664's SIGTERM) stops the monitor without being logged as an error
    let (sup, host, _child) = setup(SupervisorOptions { memory_warn_mb: 500.0, ..opts() });
    host.samples.lock().unwrap().extend([Err(MemoryError::Interrupted), Ok((600.0, "uss".into()))]);
    sup.supervise().await;
    tokio::time::sleep(Duration::from_secs(30)).await;
    assert!(host.messages().is_empty(), "{:?}", host.messages());
    assert_eq!(host.samples.lock().unwrap().len(), 1);
}

// Catches: the kill skipping SIGUSR1 (the child's stack dump) or killing an already dead child.
#[cfg(unix)]
#[tokio::test]
async fn kill_sends_sigusr1_then_kills() {
    struct ChildHost(Mutex<std::process::Child>, AtomicUsize);
    impl ProcHost for Arc<ChildHost> {
        type Error = String;
        fn log(&self, _: Level, _: &str, _: Extra) {}
        fn log_exception(&self, _: &str, _: String) {}
        fn is_alive(&self) -> bool {
            matches!(self.0.lock().unwrap().try_wait(), Ok(None))
        }
        fn kill(&self) {
            self.1.fetch_add(1, Ordering::SeqCst);
            let _ = self.0.lock().unwrap().kill();
        }
        fn sample_memory(&self) -> Result<(f64, String), MemoryError<String>> {
            Err(MemoryError::Unavailable(String::new()))
        }
    }
    let child = std::process::Command::new("sleep").arg("30").spawn().unwrap();
    let pid = child.id();
    let host = Arc::new(ChildHost(Mutex::new(child), AtomicUsize::new(0)));
    let (parent, _peer) = UnixStream::pair().unwrap();
    let sup = ProcSupervisor::new(parent, host.clone(), SupervisorOptions { pid, ..opts() });
    let start = std::time::Instant::now();
    sup.kill().await;
    assert!(start.elapsed() >= KILL_GRACE, "the grace before the kill");
    assert!(sup.killed());
    // SIGUSR1's default action ends `sleep` within the grace: no SIGKILL needed
    assert_eq!(host.1.load(Ordering::SeqCst), 0);
    assert!(!host.is_alive());
    sup.kill().await;
    assert_eq!(host.1.load(Ordering::SeqCst), 0);
}

// ------------------------------------------------------------------------------------- client

async fn connected(health_timeout_ms: Option<u64>) -> (IpcClient, DuplexStream) {
    let (mut parent, child) = tokio::io::duplex(1 << 16);
    write(&mut parent, &ToChild::Hello(Hello { health_timeout_ms, ..hello() })).await.unwrap();
    let (client, got) = IpcClient::connect(child).await.unwrap();
    assert_eq!(got.health_timeout_ms, health_timeout_ms);
    (client, parent)
}

// Catches: the client answering a parent of another version as if it spoke it, or reading
// anything but `Hello` first.
#[tokio::test]
async fn client_fails_on_a_version_mismatch_and_says_why() {
    let (mut parent, child) = tokio::io::duplex(1 << 16);
    write(&mut parent, &ToChild::Hello(Hello { version: PROTOCOL_VERSION + 1, ..hello() }))
        .await
        .unwrap();
    let err = IpcClient::connect(child).await.err().unwrap();
    assert!(matches!(err, WireError::Version { theirs } if theirs == PROTOCOL_VERSION + 1));
    let ToParent::Ready { version, error } = read(&mut parent).await.unwrap() else { panic!() };
    assert_eq!(version, PROTOCOL_VERSION);
    assert!(error.unwrap().contains("protocol version"));

    let (mut parent, child) = tokio::io::duplex(1 << 16);
    write(&mut parent, &ToChild::DumpStacks).await.unwrap();
    assert!(matches!(IpcClient::connect(child).await, Err(WireError::Unexpected(_))));
}

// Catches: what the child sends reordered, or lost on close.
#[tokio::test]
async fn client_sends_in_order_and_flushes_on_close() {
    let (client, mut parent) = connected(None).await;
    write(&mut parent, &ToChild::Ping { sent_ms: 1 }).await.unwrap();
    let reading = tokio::spawn(async move {
        let mut got = Vec::new();
        while let Ok(msg) = read::<ToParent, _>(&mut parent).await {
            got.push(msg);
        }
        got
    });
    for i in 0..100 {
        client.send(ToParent::Pong { sent_ms: i }).await.unwrap();
    }
    assert_eq!(client.recv().await, Some(ToChild::Ping { sent_ms: 1 }));
    client.close().await;
    let got = reading.await.unwrap();
    assert_eq!(got, (0..100).map(|i| ToParent::Pong { sent_ms: i }).collect::<Vec<_>>());
    assert!(client.send(ToParent::ShutdownAck).await.is_err());
}

// Catches: an orphaned child (the parent stopped talking) never noticing.
#[tokio::test(start_paused = true)]
async fn client_health_check_closes_when_the_parent_goes_quiet() {
    let (client, mut parent) = connected(Some(100_000)).await;
    let start = Instant::now();
    tokio::time::sleep(Duration::from_secs(60)).await;
    write(&mut parent, &ToChild::Ping { sent_ms: 1 }).await.unwrap();
    assert_eq!(client.recv().await, Some(ToChild::Ping { sent_ms: 1 }));
    assert_eq!(client.recv().await, None);
    let quiet = start.elapsed();
    assert!(quiet >= Duration::from_secs(160) && quiet < Duration::from_secs(161), "{quiet:?}");

    let (client, _parent) = connected(None).await;
    let pending = tokio::time::timeout(Duration::from_secs(10_000), client.recv()).await;
    assert!(pending.is_err(), "no health check without a ping deadline");
}

// Catches: a malformed frame from the parent read past, instead of ending the connection.
#[tokio::test]
async fn client_stops_at_a_malformed_frame() {
    let (client, mut parent) = connected(None).await;
    parent.write_all(&[0, 0, 0, 1, 99]).await.unwrap();
    write(&mut parent, &ToChild::DumpStacks).await.unwrap();
    assert_eq!(client.recv().await, None);
}
