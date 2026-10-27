use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::DuplexStream;
use tokio::net::UnixStream;
use tokio::time::Instant;

use super::*;

// ------------------------------------------------------------------------------------ codec

fn hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn s(v: &str) -> Value {
    Value::Str(v.to_owned())
}

/// The reference encoder's bytes (`livekit-agents` `channel._write_message`) for every message,
/// with the fields they hold.
fn vectors() -> Vec<(&'static str, i32, Vec<Value>)> {
    let job = hex("0a06414a5f3132331a0f0a04524d5f311207726f6f6d2dc3a92a026e73");
    vec![
        (
            "0000000001000020400000b4420000003f00000011687474703a2f2f70726f78793a38303830",
            INITIALIZE_REQUEST,
            vec![
                Value::Bool(true),
                Value::Float(2.5),
                Value::Float(90.0),
                Value::Float(0.5),
                s("http://proxy:8080"),
            ],
        ),
        (
            "000000000000000000000000000000000000000000",
            INITIALIZE_REQUEST,
            vec![
                Value::Bool(false),
                Value::Float(0.0),
                Value::Float(0.0),
                Value::Float(0.0),
                s(""),
            ],
        ),
        ("0000000100000000", INITIALIZE_RESPONSE, vec![s("")]),
        ("0000000100000008626f6f6d20e29c97", INITIALIZE_RESPONSE, vec![s("boom ✗")]),
        ("000000020000019a2194ef7b", PING_REQUEST, vec![Value::Long(1761500000123)]),
        (
            "000000030000019a2194ef7b0000019a2194f0c8",
            PONG_RESPONSE,
            vec![Value::Long(1761500000123), Value::Long(1761500000456)],
        ),
        (
            "000000040000001d0a06414a5f3132331a0f0a04524d5f311207726f6f6d2dc3a92a026e7300000005\
             6167656e74000000056964656e74000000087b226b223a20317d000000157773733a2f2f782e6c69\
             76656b69742e636c6f756400000003746f6b0000000441575f3101b4e607d8963fda41000020d896\
             3fda41000050d8963fda41000080d8963fda41",
            START_JOB_REQUEST,
            vec![
                Value::Bytes(job),
                s("agent"),
                s("ident"),
                s("{\"k\": 1}"),
                s("wss://x.livekit.cloud"),
                s("tok"),
                s("AW_1"),
                Value::Bool(true),
                Value::Double(1761500000.123456),
                Value::Double(1761500000.5),
                Value::Double(1761500001.25),
                Value::Double(1761500002.0),
            ],
        ),
        ("0000000500000003627965", SHUTDOWN_REQUEST, vec![s("bye")]),
        ("0000000600000004646f6e65", EXITING, vec![s("done")]),
        (
            "00000007000000136c6b5f656e645f6f665f7574746572616e6365000000027231000000030001ff",
            INFERENCE_REQUEST,
            vec![s("lk_end_of_utterance"), s("r1"), Value::Bytes(vec![0, 1, 0xff])],
        ),
        (
            "00000008000000027231010000000361626300000000",
            INFERENCE_RESPONSE,
            vec![s("r1"), Value::Bytes(b"abc".to_vec()), s("")],
        ),
        (
            "00000008000000027232010000000000000000",
            INFERENCE_RESPONSE,
            vec![s("r2"), Value::Bytes(Vec::new()), s("")],
        ),
        (
            "0000000800000002723300000000156e6f20696e666572656e6365206578656375746f72",
            INFERENCE_RESPONSE,
            vec![s("r3"), Value::Absent, s("no inference executor")],
        ),
        ("00000009", DUMP_STACK_TRACE_REQUEST, vec![]),
        ("0000000a", SHUTDOWN_REQUEST_ACK, vec![]),
        ("0000000b", SHUTTING_DOWN, vec![]),
    ]
}

// Catches: a field order, width, endianness or optional-bytes layout that differs from the
// reference encoder, which would break a parent and a child built from different stacks.
#[test]
fn every_message_round_trips_against_the_reference_bytes() {
    let mut seen = std::collections::BTreeSet::new();
    for (wire, id, fields) in vectors() {
        let bytes = hex(wire);
        let decoded = IpcMessage::decode(&bytes).unwrap();
        let expected = IpcMessage::new(id, fields).unwrap();
        assert_eq!(decoded, expected, "{}", expected.name());
        assert_eq!(expected.encode(), bytes, "{}", expected.name());
        seen.insert(id);
    }
    assert_eq!(seen.len(), MESSAGES.len(), "a message has no reference vector");
}

// Catches: accepting a frame the reference cannot have written as if it were a message.
#[test]
fn malformed_messages_are_errors() {
    assert_eq!(IpcMessage::decode(&hex("0000000c")), Err(CodecError::UnknownId(12)));
    assert_eq!(IpcMessage::decode(&hex("ffffffff")), Err(CodecError::UnknownId(-1)));
    assert_eq!(IpcMessage::decode(&hex("000000020000")), Err(CodecError::Truncated));
    assert_eq!(IpcMessage::decode(&hex("0000000100000002c328")), Err(CodecError::Utf8("error")));
    assert_eq!(IpcMessage::new(PING_REQUEST, vec![]), Err(CodecError::Arity("PingRequest", 1)));
    assert_eq!(IpcMessage::new(PING_REQUEST, vec![s("x")]), Err(CodecError::Type("timestamp")));
    // a nonzero byte is true, as `bool.from_bytes` reads it
    let msg = IpcMessage::decode(&hex("0000000002000000000000000000000000000000000000")).unwrap();
    assert_eq!(msg.get("asyncio_debug"), Some(&Value::Bool(true)));
}

#[tokio::test]
async fn frames_are_length_prefixed() {
    let (mut a, mut b) = UnixStream::pair().unwrap();
    write_frame(&mut a, b"hello").await.unwrap();
    let mut raw = [0u8; 9];
    b.read_exact(&mut raw).await.unwrap();
    assert_eq!(&raw, b"\x00\x00\x00\x05hello");
    write_frame(&mut a, b"").await.unwrap();
    assert_eq!(read_frame(&mut b).await.unwrap(), b"");
    drop(a);
    assert!(read_frame(&mut b).await.is_err());
}

// ------------------------------------------------------------------------------- supervisor

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

    fn has_running_job(&self) -> bool {
        true
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
        process_kind: "job".into(),
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

fn setup(opts: SupervisorOptions) -> (ProcSupervisor<Arc<FakeHost>>, Arc<FakeHost>, DuplexStream) {
    let (parent, child) = tokio::io::duplex(1 << 16);
    let host = Arc::new(FakeHost::default());
    host.alive.store(true, Ordering::SeqCst);
    (ProcSupervisor::new(parent, host.clone(), opts), host, child)
}

async fn send(child: &mut DuplexStream, id: i32, fields: Vec<Value>) {
    write_frame(child, &IpcMessage::new(id, fields).unwrap().encode()).await.unwrap();
}

async fn recv(child: &mut DuplexStream) -> IpcMessage {
    IpcMessage::decode(&read_frame(child).await.unwrap()).unwrap()
}

/// Reads until a message other than a ping.
async fn recv_skipping_pings(child: &mut DuplexStream) -> IpcMessage {
    loop {
        let msg = recv(child).await;
        if msg.id() != PING_REQUEST {
            return msg;
        }
    }
}

#[tokio::test(start_paused = true)]
async fn initialize_reads_the_response_and_logs_the_elapsed_time() {
    let (sup, host, mut child) = setup(opts());
    let init = tokio::spawn({
        let sup = sup.clone();
        async move { sup.initialize().await }
    });
    tokio::time::sleep(Duration::from_millis(1234)).await;
    send(&mut child, INITIALIZE_RESPONSE, vec![s("")]).await;
    let elapsed = init.await.unwrap().unwrap().unwrap();
    assert!(elapsed >= Duration::from_millis(1234));
    assert_eq!(host.messages(), ["initializing process", "process initialized"]);
    let (_, extra) = host.find("process initialized").unwrap();
    assert_eq!(extra, vec![("elapsed_time", ExtraValue::Float(1.23))]);
}

// Catches: a child whose initialize_process_fnc raised being treated as ready.
#[tokio::test(start_paused = true)]
async fn initialize_raises_the_child_error() {
    let (sup, _host, mut child) = setup(opts());
    send(&mut child, INITIALIZE_RESPONSE, vec![s("no model")]).await;
    let err = sup.initialize().await.unwrap_err();
    assert_eq!(err.to_string(), "process initialization failed: no model");

    let (sup, _host, mut child) = setup(opts());
    send(&mut child, PING_REQUEST, vec![Value::Long(1)]).await;
    let err = sup.initialize().await.unwrap_err();
    assert_eq!(err.to_string(), "first message must be InitializeResponse");

    let (sup, _host, child) = setup(opts());
    drop(child);
    assert!(matches!(sup.initialize().await, Err(InitError::Closed)));
}

// Catches: a process stuck in initialize_process_fnc left running past initialize_timeout.
#[tokio::test(start_paused = true)]
async fn initialize_timeout_kills_the_process() {
    let (sup, host, _child) = setup(opts());
    let start = Instant::now();
    assert!(sup.initialize().await.unwrap().is_none());
    assert_eq!(start.elapsed(), Duration::from_secs(10));
    assert_eq!(host.kills.load(Ordering::SeqCst), 1);
    assert!(sup.killed());
    assert_eq!(host.messages(), ["initializing process", "killing process"]);
}

// Catches: a ping cadence or pong bookkeeping that differs from the reference: the first ping at
// once, then one per interval, every message forwarded, a late pong warned about.
#[tokio::test(start_paused = true)]
async fn pings_on_the_interval_and_warns_on_a_late_pong() {
    let (sup, host, mut child) = setup(opts());
    let start = Instant::now();
    sup.supervise();
    let first = recv(&mut child).await;
    assert_eq!(first.id(), PING_REQUEST);
    assert_eq!(start.elapsed(), Duration::ZERO);
    let Some(Value::Long(ts)) = first.get("timestamp").cloned() else { panic!() };
    assert!((time_ms() - ts as i64).abs() < 5_000);
    assert_eq!(recv(&mut child).await.id(), PING_REQUEST);
    assert_eq!(start.elapsed(), Duration::from_millis(2500));

    let late = (time_ms() - 2_000) as u64;
    send(&mut child, PONG_RESPONSE, vec![Value::Long(ts), Value::Long(late)]).await;
    assert_eq!(IpcMessage::decode(&sup.recv().await.unwrap()).unwrap().id(), PONG_RESPONSE);
    let (level, extra) = host.find("process is unresponsive").unwrap();
    assert_eq!(level, Level::Warning);
    let ExtraValue::Int(delay) = extra[0].1 else { panic!() };
    assert!(extra[0].0 == "delay" && delay >= 2_000);

    let fresh = time_ms() as u64;
    send(&mut child, PONG_RESPONSE, vec![Value::Long(ts), Value::Long(fresh)]).await;
    sup.recv().await.unwrap();
    assert_eq!(host.messages().iter().filter(|m| *m == "process is unresponsive").count(), 1);
}

// Catches: a hung child (no pongs) never being killed, or pongs not pushing the deadline back.
#[tokio::test(start_paused = true)]
async fn missing_pongs_kill_the_process_after_ping_timeout() {
    let (sup, host, mut child) = setup(opts());
    let start = Instant::now();
    sup.supervise();
    // one pong at 60 s moves the deadline to 150 s
    tokio::time::sleep(Duration::from_secs(60)).await;
    send(&mut child, PONG_RESPONSE, vec![Value::Long(0), Value::Long(time_ms() as u64)]).await;
    let drain = tokio::spawn(async move { while read_frame(&mut child).await.is_ok() {} });
    while host.kills.load(Ordering::SeqCst) == 0 {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let killed_at = start.elapsed();
    assert!(killed_at >= Duration::from_secs(150) && killed_at < Duration::from_secs(151));
    let (level, _) = host.find("process is unresponsive, killing process").unwrap();
    assert_eq!(level, Level::Error);
    drop(drain);
}

// Catches: a graceful close that kills a child which acked, or logs a clean exit as a failure.
#[tokio::test(start_paused = true)]
async fn shutdown_waits_for_the_ack_and_the_exit() {
    let (sup, host, mut child) = setup(opts());
    sup.supervise();
    let close = tokio::spawn({
        let sup = sup.clone();
        async move { sup.shutdown().await }
    });
    let req = recv_skipping_pings(&mut child).await;
    assert_eq!(req, IpcMessage::shutdown_request());
    send(&mut child, SHUTDOWN_REQUEST_ACK, vec![]).await;
    send(&mut child, EXITING, vec![s("shutdown")]).await;
    send(&mut child, SHUTTING_DOWN, vec![]).await;
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
    // every message reached the host before the stream ended
    let mut ids = Vec::new();
    while let Ok(frame) = sup.recv().await {
        ids.push(IpcMessage::decode(&frame).unwrap().id());
    }
    assert_eq!(ids, [SHUTDOWN_REQUEST_ACK, EXITING, SHUTTING_DOWN]);
}

// Catches: a child that never acks keeping the close waiting forever.
#[tokio::test(start_paused = true)]
async fn shutdown_kills_a_process_that_does_not_ack() {
    let (sup, host, child) = setup(opts());
    *host.child.lock().unwrap() = Some(child);
    sup.supervise();
    let start = Instant::now();
    let close = tokio::spawn({
        let sup = sup.clone();
        async move { sup.shutdown().await }
    });
    while host.kills.load(Ordering::SeqCst) == 0 {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(start.elapsed() >= Duration::from_secs(60));
    // the kill closed the socket: the reader ends and resolves ShuttingDown
    sup.exited(Some(-9)).await;
    close.await.unwrap();
    let messages = host.messages();
    assert!(messages.contains(&"process did not ack shutdown in time, killing process".into()));
    assert!(!messages.iter().any(|m| m.starts_with("process exited with non-zero")));
    assert!(!messages.contains(&"process did not exit in time, killing process".into()));
}

// Catches: a job thread killed (it cannot be) or given up on when its ack and exit are late; it is
// reported and waited for, as `ThreadJobExecutor.aclose` does.
#[tokio::test(start_paused = true)]
async fn a_job_thread_that_does_not_ack_is_waited_for_not_killed() {
    let (sup, host, _child) = setup(SupervisorOptions { killable: false, ping_timeout: 0.0, ..opts() });
    sup.supervise();
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

// Catches: the read loop's end leaving a close waiting for an ack that cannot arrive (#5929).
#[tokio::test(start_paused = true)]
async fn exited_resolves_the_shutdown_waits() {
    let (sup, host, _child) = setup(opts());
    sup.supervise();
    tokio::time::sleep(Duration::from_secs(1)).await;
    sup.exited(Some(1)).await;
    tokio::time::timeout(Duration::from_secs(1), sup.inner.shutdown_ack.wait()).await.unwrap();
    tokio::time::timeout(Duration::from_secs(1), sup.inner.shutting_down.wait()).await.unwrap();
    assert!(sup.recv().await.is_err());
    assert!(sup.send(b"x").await.is_err());
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
    let (sup, host, _child) = setup(SupervisorOptions { memory_warn_mb: 500.0, ..opts() });
    host.samples.lock().unwrap().extend([
        Ok((100.0, "uss".into())),
        Ok((600.04, "uss".into())),
        Ok((610.0, "uss".into())),
        Ok((700.0, "uss".into())),
    ]);
    sup.supervise();
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
}

#[tokio::test(start_paused = true)]
async fn memory_limit_kills_and_unavailable_stops_monitoring() {
    let (sup, host, _child) = setup(SupervisorOptions { memory_limit_mb: 500.0, ..opts() });
    host.samples.lock().unwrap().push_back(Ok((600.0, "pss".into())));
    sup.supervise();
    tokio::time::sleep(Duration::from_secs(1)).await;
    let (level, _) = host.find("job process exceeded memory limit, killing it").unwrap();
    assert_eq!(level, Level::Error);
    assert_eq!(host.kills.load(Ordering::SeqCst), 1);

    let (sup, host, _child) = setup(SupervisorOptions { memory_warn_mb: 500.0, ..opts() });
    host.samples.lock().unwrap().extend([
        Err(MemoryError::Failed("oops".into())),
        Err(MemoryError::Unavailable("process no longer exists".into())),
    ]);
    sup.supervise();
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
    sup.supervise();
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
        fn has_running_job(&self) -> bool {
            false
        }
    }
    let child = std::process::Command::new("sleep").arg("30").spawn().unwrap();
    let pid = child.id();
    let host = Arc::new(ChildHost(Mutex::new(child), AtomicUsize::new(0)));
    let (parent, _peer) = UnixStream::pair().unwrap();
    let sup = ProcSupervisor::new(parent, host.clone(), SupervisorOptions { pid, ..opts() });
    sup.kill().await;
    assert!(sup.killed());
    // SIGUSR1's default action ends `sleep` within the grace: no SIGKILL needed
    assert_eq!(host.1.load(Ordering::SeqCst), 0);
    assert!(!host.is_alive());
    sup.kill().await;
    assert_eq!(host.1.load(Ordering::SeqCst), 0);
}

// ----------------------------------------------------------------------------------- client

fn served<S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Send + 'static>(
    stream: S,
    health_timeout: Option<Duration>,
) -> IpcClient {
    let client = IpcClient::new(stream);
    client.serve(health_timeout);
    client
}

// Catches: the handshake's request read by the reader instead, or the answers to it and what
// follows sent out of order, or lost on close.
#[tokio::test]
async fn client_handshakes_then_sends_in_order_and_flushes_on_close() {
    let (mut parent, child) = UnixStream::pair().unwrap();
    let client = IpcClient::new(child);
    write_frame(&mut parent, b"init").await.unwrap();
    write_frame(&mut parent, b"next").await.unwrap();
    assert_eq!(client.handshake().await.unwrap(), b"init");
    for i in 0..100u8 {
        client.queue(vec![i]).unwrap();
    }
    client.serve(None);
    assert_eq!(client.recv().await.unwrap(), b"next");
    client.close().await;
    for i in 0..100u8 {
        assert_eq!(read_frame(&mut parent).await.unwrap(), vec![i]);
    }
    assert!(read_frame(&mut parent).await.is_err(), "the socket closed after the queue");
    assert!(client.queue(vec![0]).is_err());
}

// Catches: an orphaned child (the parent stopped talking) never noticing.
#[tokio::test(start_paused = true)]
async fn client_health_check_closes_when_the_parent_goes_quiet() {
    let (mut parent, child) = tokio::io::duplex(1 << 16);
    let client = served(child, Some(Duration::from_secs(100)));
    let start = Instant::now();
    tokio::time::sleep(Duration::from_secs(60)).await;
    write_frame(&mut parent, &IpcMessage::ping(1).encode()).await.unwrap();
    assert_eq!(IpcMessage::decode(&client.recv().await.unwrap()).unwrap(), IpcMessage::ping(1));
    assert!(client.recv().await.is_err());
    let quiet = start.elapsed();
    assert!(quiet >= Duration::from_secs(160) && quiet < Duration::from_secs(161), "{quiet:?}");

    let (_parent, child) = tokio::io::duplex(1 << 16);
    let client = served(child, None);
    let pending = tokio::time::timeout(Duration::from_secs(10_000), client.recv()).await;
    assert!(pending.is_err(), "no health check without a ping interval");
}

#[tokio::test]
async fn client_sends_and_closes() {
    let (mut parent, child) = UnixStream::pair().unwrap();
    let client = served(child, None);
    client.send(b"abc").await.unwrap();
    assert_eq!(read_frame(&mut parent).await.unwrap(), b"abc");
    drop(parent);
    assert!(client.recv().await.is_err());
    client.close().await;
    assert!(client.send(b"x").await.is_err());
}
