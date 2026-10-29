use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncWriteExt, DuplexStream};
use tokio::net::UnixStream;
use tokio::time::Instant;

use super::supervisor::time_ms;
use super::supervisor::{
    Extra, KILL_GRACE, MEMORY_WARN_COOLDOWN, MEMORY_WARN_RESET_DELTA_MB, WarnGate,
};
use super::wire::{MAX_FRAME, decode, encode, read, write};
use super::*;
use lk_agents_abi::errors::HostError;

mod writer;

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
        traceparent: None,
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
        ToChild::StartJob { request_id: 1, job: Box::new(job()) },
        ToChild::StartJob { request_id: 2, job: Box::default() },
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
        ToParent::JobAccepted { request_id: 1, error: None },
        ToParent::JobAccepted { request_id: 1, error: Some("refused".into()) },
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
        write(&mut a, &ToChild::StartJob { request_id: 1, job: Box::new(big) }).await.is_err(),
        "an oversized message is not sent"
    );
    drop(a);
    assert!(matches!(read::<ToChild, _>(&mut b).await, Err(WireError::Eof)));
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
    child: Mutex<Option<DuplexStream>>,
    released: AtomicBool,
}

impl ProcHost for Arc<FakeHost> {
    fn log(&self, level: Level, msg: &str, extra: Extra) {
        self.logs.lock().unwrap().push((level, msg.to_owned(), extra));
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
        self.alive.store(false, Ordering::SeqCst);
        self.child.lock().unwrap().take();
        Ok(())
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
    (
        ProcSupervisor::new(parent, host.clone(), opts, crate::test_support::blocking_io()),
        host,
        child,
    )
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
    let initialized = sup.initialize(hello()).await;
    assert!(matches!(initialized, Err(InitError::Closed)), "{initialized:?}");

    // a frame the parent cannot read closes the connection, with why
    let (sup, host, mut child) = setup(opts());
    let init = waits(&sup).await;
    child.write_all(&[0, 0, 0, 1, 99]).await.unwrap();
    assert!(
        matches!(init.await.unwrap(), Err(InitError::Protocol(failure)) if matches!(*failure, WireError::Malformed(_)))
    );
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
    sup.reaped(0).await;
    close.await.unwrap();
    assert_eq!(host.kills.load(Ordering::SeqCst), 0);
    let (level, extra) = host.find("process exiting").unwrap();
    assert_eq!((level, extra), (Level::Info, vec![("reason", ExtraValue::Str("shutdown".into()))]));
    assert!(
        host.logs.lock().unwrap().iter().all(|(l, _, _)| *l != Level::Error),
        "{:?}",
        host.messages()
    );
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
    host.samples.lock().unwrap().push_back(Ok((600.0, "pss".into())));
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

// Catches: the kill skipping SIGUSR1 (the child's stack dump) or killing an already dead child.

#[cfg(unix)]
#[tokio::test]
async fn kill_sends_sigusr1_then_kills() {
    struct ChildHost(Mutex<std::process::Child>, AtomicUsize);
    impl ProcHost for Arc<ChildHost> {
        fn log(&self, _: Level, _: &str, _: Extra) {}
        fn log_exception(&self, _: &str, _: HostError) {}
        fn is_alive(&self) -> bool {
            matches!(self.0.lock().unwrap().try_wait(), Ok(None))
        }
        fn kill(&self) -> Result<(), String> {
            self.1.fetch_add(1, Ordering::SeqCst);
            self.0.lock().unwrap().kill().map_err(|e| e.to_string())
        }
        fn request_stack_dump(&self) -> Result<bool, String> {
            let pid = self.0.lock().unwrap().id();
            // SAFETY: the Child remains owned and unreaped while this signal is submitted.
            let sent = unsafe { libc::kill(pid as libc::pid_t, libc::SIGUSR1) };
            if sent == 0 { Ok(true) } else { Err(std::io::Error::last_os_error().to_string()) }
        }
        fn sample_memory(&self) -> Result<(f64, String), MemoryError> {
            Err(MemoryError::Unavailable(String::new()))
        }
    }
    let child = std::process::Command::new("sleep").arg("30").spawn().unwrap();
    let pid = child.id();
    let host = Arc::new(ChildHost(Mutex::new(child), AtomicUsize::new(0)));
    let (parent, _peer) = UnixStream::pair().unwrap();
    let sup = ProcSupervisor::new(
        parent,
        host.clone(),
        SupervisorOptions { pid, ..opts() },
        crate::test_support::blocking_io(),
    );
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
    assert!(matches!(client.wait_reader().await, Err(WireError::Malformed(_))));
}

#[tokio::test(start_paused = true)]
async fn supervisor_blocked_writer() {
    let (parent, mut child) = UnixStream::pair().unwrap();
    let host = Arc::new(FakeHost::default());
    host.alive.store(true, Ordering::SeqCst);
    let sup = ProcSupervisor::new(
        parent,
        host.clone(),
        SupervisorOptions { ping_timeout: 1.0, ..opts() },
        crate::test_support::blocking_io(),
    );
    let initialize = waits(&sup).await;
    assert!(matches!(read::<ToChild, _>(&mut child).await.unwrap(), ToChild::Hello(_)));
    write(&mut child, &ToParent::Ready { version: PROTOCOL_VERSION, error: None }).await.unwrap();
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
async fn child_inbox_saturation_health() {
    let (client, mut parent) = connected(Some(100)).await;
    for _ in 0..65 {
        write(&mut parent, &ToChild::DumpStacks).await.unwrap();
    }
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(
        lk_agents_abi::lock(&client.reader).as_ref().unwrap().is_finished(),
        "an undrained application inbox must not suspend the health loop"
    );
    client.close().await;
}

#[tokio::test(start_paused = true)]
async fn repeated_initialize_callers_are_refused_without_replacing_the_first_waiter() {
    let (sup, _host, mut child) = setup(opts());
    let first = waits(&sup).await;
    assert!(matches!(recv(&mut child).await, ToChild::Hello(_)));
    assert!(matches!(sup.initialize(hello()).await, Err(InitError::AlreadyInitializing)));
    write(&mut child, &ToParent::Ready { version: PROTOCOL_VERSION, error: None }).await.unwrap();
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
