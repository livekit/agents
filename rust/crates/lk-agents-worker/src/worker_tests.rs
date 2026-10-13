//! Connection lifecycle and scheduling tests using scripted peers and gated host operations.

use crate::transport::{
    Connected, TransportError, TransportEvent, WorkerSink, WorkerStream, WorkerTransport,
};

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;

use crate::EnvOption;
use crate::JobDecision;
use crate::protocol::{self, RegisterWorkerResponse, ServerMessage, server_message};
use crate::worker::*;
use crate::worker::{HostError, JobId, WorkerHost};
use crate::{WorkerConfig, WorkerState};
use lk_agents_abi::{Clock, Ts};

// The two pure decisions

// Catches: an exponential backoff replacing `min(n * 2, 10)`, losing the immediate first retry.
#[test]
fn the_first_reconnect_is_immediate_and_the_ladder_caps_at_ten() {
    let ladder: Vec<u64> = (0..8).map(|n| retry_delay(n).as_secs()).collect();
    assert_eq!(ladder, vec![0, 2, 4, 6, 8, 10, 10, 10]);
}

// A scripted transport

enum Step {
    /// The connect attempt fails.
    ConnectFails,
    /// The connect attempt succeeds and the socket yields these messages, then blocks forever.
    Serves(Vec<Result<ServerMessage, TransportError>>),
}

#[derive(Default)]
struct Wire {
    sent: Vec<protocol::WorkerMessage>,
    attempts: u32,
}

struct ScriptedTransport {
    script: Mutex<std::collections::VecDeque<Step>>,
    wire: Arc<Mutex<Wire>>,
}

struct Sink(Arc<Mutex<Wire>>);

#[async_trait]
impl WorkerSink for Sink {
    async fn send(&self, message: protocol::WorkerMessage) -> Result<(), TransportError> {
        self.0.lock().unwrap().sent.push(message);
        Ok(())
    }
    async fn close(&self) {}
}

struct Stream(std::collections::VecDeque<Result<ServerMessage, TransportError>>);

#[async_trait]
impl WorkerStream for Stream {
    async fn recv(&mut self) -> Result<TransportEvent, TransportError> {
        match self.0.pop_front() {
            Some(message) => message.map(TransportEvent::ServerMessage),
            // An exhausted script means the peer went quiet, like a real idle socket; `Closed`
            // would make every test look like a drop.
            None => std::future::pending().await,
        }
    }
}

#[async_trait]
impl WorkerTransport for ScriptedTransport {
    async fn connect(
        &self,
        _url: &str,
        _token: Option<&str>,
        _headers: Vec<(String, String)>,
    ) -> Result<Connected, TransportError> {
        self.wire.lock().unwrap().attempts += 1;
        match self.script.lock().unwrap().pop_front() {
            Some(Step::ConnectFails) | None => Err(TransportError::Io("refused".into())),
            Some(Step::Serves(frames)) => Ok(Connected {
                sink: Arc::new(Sink(self.wire.clone())),
                stream: Box::new(Stream(frames.into())),
            }),
        }
    }
}

/// Rejects every job, fails every launch, and answers `load_fnc` with `user` (`None`: none given).
#[derive(Default)]
struct LoadHost {
    user: Option<f32>,
    /// How many times `load_fnc` was asked.
    user_calls: Arc<Mutex<u32>>,
}

#[async_trait]
impl WorkerHost for LoadHost {
    async fn on_job_request(&self, _r: crate::JobRequest) -> JobDecision {
        JobDecision::Reject { terminate: false }
    }
    async fn launch_job(&self, _i: crate::RunningJobInfo) -> Result<JobId, HostError> {
        Err(HostError::new("no pool"))
    }
    async fn terminate_job(&self, _id: JobId) -> Result<(), HostError> {
        Ok(())
    }
    async fn user_load(&self) -> Option<f32> {
        *self.user_calls.lock().unwrap() += 1;
        self.user
    }
}

fn runtime_cfg(max_retry: u32) -> RuntimeConfig {
    RuntimeConfig {
        server_url: "wss://gw".into(),
        worker_token: None,
        headers: vec![],
        api_key: String::new(),
        api_secret: String::new(),
        max_retry,
        connection_timeout: Duration::from_secs(10),
    }
}

struct FrozenClock;
#[async_trait::async_trait]
impl Clock for FrozenClock {
    fn now(&self) -> Ts {
        Ts::ZERO
    }
    fn wall(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH
    }
    /// Frozen: a wait on it never ends. No test here arms one.
    async fn wait_until(&self, _at: Ts) {
        std::future::pending::<()>().await
    }
}

fn register_message(worker_id: &str) -> Result<ServerMessage, TransportError> {
    let msg = ServerMessage {
        message: Some(server_message::Message::Register(RegisterWorkerResponse {
            worker_id: worker_id.to_owned(),
            ..Default::default()
        })),
    };
    Ok(msg)
}

type Scripted = Worker<ScriptedTransport, LoadHost, FrozenClock>;

fn conn(steps: Vec<Step>, max_retry: u32) -> (Scripted, Arc<Mutex<Wire>>, WorkerHandle) {
    conn_with(steps, max_retry, WorkerConfig { ws_url: "wss://gw".into(), ..Default::default() }, LoadHost::default())
}

fn conn_with(
    steps: Vec<Step>,
    max_retry: u32,
    cfg_worker: WorkerConfig,
    host: LoadHost,
) -> (Scripted, Arc<Mutex<Wire>>, WorkerHandle) {
    let wire = Arc::new(Mutex::new(Wire::default()));
    let t = ScriptedTransport { script: Mutex::new(steps.into()), wire: wire.clone() };
    let m = WorkerState::new(cfg_worker);
    let (c, h) = Worker::with_components(t, runtime_cfg(max_retry), m, host, FrozenClock);
    (c, wire, h)
}

// The connection itself

// Catches: the register message not being the first thing written to the socket.
#[tokio::test(start_paused = true)]
async fn register_is_the_first_message_written_and_the_worker_id_lands() {
    let (mut c, wire, _h) = conn(vec![Step::Serves(vec![register_message("W-7")])], 0);
    // The stream blocks after the register response, so drive one connection with a deadline.
    let _ = tokio::time::timeout(Duration::from_millis(50), c.run()).await;
    let sent = &wire.lock().unwrap().sent;
    assert!(!sent.is_empty(), "nothing was written to the socket");
    let first = &sent[0];
    assert!(
        matches!(first.message, Some(livekit_protocol::worker_message::Message::Register(_))),
        "first message was {:?}",
        first.message
    );
    assert_eq!(c.state().worker_id(), "W-7");
}

// Catches: an off-by-one in the retry ladder; `max_retry = 2` means three attempts, not two.
#[tokio::test(start_paused = true)]
async fn max_retry_two_means_three_connect_attempts() {
    let (mut c, wire, _h) =
        conn(vec![Step::ConnectFails, Step::ConnectFails, Step::ConnectFails], 2);
    let err = c.run().await.unwrap_err();
    assert!(matches!(err, RuntimeError::RetriesExhausted { retries: 2, .. }), "got {err:?}");
    assert_eq!(wire.lock().unwrap().attempts, 3);
}

// Catches: treating a mid-session close as fatal instead of reconnecting.
#[tokio::test(start_paused = true)]
async fn a_closed_connection_reconnects_and_the_machine_is_told() {
    let (mut c, wire, _h) = conn(
        vec![
            Step::Serves(vec![
                register_message("W-1"),
                Err(TransportError::Closed("peer disconnected".into())),
            ]),
            Step::Serves(vec![register_message("W-2")]),
        ],
        5,
    );
    let _ = tokio::time::timeout(Duration::from_millis(50), c.run()).await;
    assert_eq!(wire.lock().unwrap().attempts, 2, "did not reconnect after the close");
    assert_eq!(c.state().worker_id(), "W-2", "did not re-register on the new socket");
}

// Catches: `Drain` not reaching the machine while `run()` is in flight.
#[tokio::test(start_paused = true)]
async fn drain_reaches_a_running_connection_from_another_task() {
    let (mut c, _wire, h) = conn(vec![Step::Serves(vec![register_message("W-1")])], 0);
    let h2 = h.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(5)).await;
        h2.send(WorkerCommand::Drain { timeout: None }).unwrap();
    });
    let _ = tokio::time::timeout(Duration::from_millis(200), c.run()).await;
    assert!(c.state().draining(), "drain never reached the machine");
    assert!(h.snapshot().draining, "the snapshot did not follow the machine");
}

// Catches: `Close` being ignored during the reconnect backoff.// Catches: `Close` being ignored during the reconnect backoff.
#[tokio::test(start_paused = true)]
async fn close_during_the_reconnect_backoff_stops_the_ladder() {
    // enough failures to exhaust patience if Close were ignored
    let steps = (0..20).map(|_| Step::ConnectFails).collect();
    let (mut c, wire, h) = conn(steps, 50);
    let h2 = h.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(3)).await;
        h2.send(WorkerCommand::Close).unwrap();
    });
    c.run().await.expect("close should be a clean return, not GaveUp");
    // 0s + 2s of backoff elapsed, so only a handful of attempts -- nowhere near 50
    assert!(wire.lock().unwrap().attempts < 5, "kept reconnecting after Close");
    assert_eq!(h.snapshot().finished.as_deref(), Some("closed"));
}

// The idle-process target

// Catches: the idle-process target being computed and then dropped before it reaches the handle,
// a snapshot read stealing it from the next reader, or devmode's infinite threshold ("leave the
// pool alone") turned into a target of zero.
#[tokio::test(start_paused = true)]
async fn registering_leaves_the_idle_process_target_on_the_handle() {
    let prod = WorkerConfig { ws_url: "wss://gw".into(), ..Default::default() };
    let dev = WorkerConfig {
        devmode: true,
        load_threshold: EnvOption::Split { dev: f32::INFINITY, prod: 0.75 },
        ..prod.clone()
    };
    let cap = Some(crate::hw::cpu::idle_process_target(f32::INFINITY));
    for (cfg, want) in [(prod, cap), (dev, None)] {
        let steps = vec![Step::Serves(vec![register_message("W-1")])];
        let (mut c, _wire, h) = conn_with(steps, 0, cfg, LoadHost::default());
        let _ = tokio::time::timeout(Duration::from_millis(50), c.run()).await;
        assert_eq!(h.snapshot().target_idle_processes, want);
        assert_eq!(h.snapshot().target_idle_processes, want, "a read must not consume it");
    }
}

// The default load function: which side samples, and what "no answer" means

fn load_conn(host: LoadHost) -> (Scripted, WorkerHandle) {
    // Finite, so `worker_load` is actually compared against something (devmode's is `inf`).
    let cfg = WorkerConfig { ws_url: "wss://gw".into(), load_threshold: EnvOption::Fixed(0.7), ..Default::default() };
    let (c, _wire, h) = conn_with(vec![], 0, cfg, host);
    (c, h)
}

// Catches: the host inventing a load when there is no `load_fnc`; `None` must make the core sample.
#[tokio::test(start_paused = true)]
async fn no_user_load_function_means_the_core_samples_for_itself() {
    let (mut c, _h) = load_conn(LoadHost::default());
    c.sample_load().await.expect("sampled");
    c.sample_load().await.expect("sampled");
    // With no user override, use the core CPU monitor.
    let load = c.state().worker_load();
    assert!(load.is_finite() && (0.0..=1.0).contains(&load), "load out of range: {load}");
}

// Catches: the user's `load_fnc` being ignored for the CPU reading, or asked twice per sample.
#[tokio::test(start_paused = true)]
async fn a_user_load_function_wins_and_is_asked_exactly_once() {
    let user_calls = Arc::new(Mutex::new(0));
    let (mut c, h) = load_conn(LoadHost { user: Some(0.42), user_calls: user_calls.clone() });
    c.sample_load().await.expect("sampled");
    assert!((c.state().worker_load() - 0.42).abs() < 1e-6, "{}", c.state().worker_load());
    assert_eq!(*user_calls.lock().unwrap(), 1);
    c.publish();
    assert_eq!(h.snapshot().worker_load, c.state().worker_load());
}

// Runtime scheduling tests use channels for the peer and gates for slow host operations.
struct TestClock(tokio::time::Instant);
#[async_trait]
impl Clock for TestClock {
    fn now(&self) -> Ts {
        Ts::from_nanos(self.0.elapsed().as_nanos() as u64)
    }
    fn wall(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH
    }
    async fn wait_until(&self, at: Ts) {
        tokio::time::sleep(at.saturating_duration_since(self.now())).await;
    }
}

struct ChannelTransport {
    stall_writes: Arc<std::sync::atomic::AtomicBool>,
    incoming:
        Mutex<Option<tokio::sync::mpsc::UnboundedReceiver<Result<ServerMessage, TransportError>>>>,
    sent: tokio::sync::mpsc::UnboundedSender<protocol::WorkerMessage>,
    closed: Arc<std::sync::atomic::AtomicUsize>,
}
struct ChannelSink {
    stall_writes: Arc<std::sync::atomic::AtomicBool>,
    sent: tokio::sync::mpsc::UnboundedSender<protocol::WorkerMessage>,
    closed: Arc<std::sync::atomic::AtomicUsize>,
}
#[async_trait]
impl WorkerSink for ChannelSink {
    async fn send(&self, message: protocol::WorkerMessage) -> Result<(), TransportError> {
        if self.stall_writes.load(std::sync::atomic::Ordering::SeqCst) {
            return std::future::pending().await;
        }
        self.sent.send(message).map_err(|e| TransportError::Io(e.to_string()))
    }
    async fn close(&self) {
        self.closed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}
struct ChannelStream(tokio::sync::mpsc::UnboundedReceiver<Result<ServerMessage, TransportError>>);
#[async_trait]
impl WorkerStream for ChannelStream {
    async fn recv(&mut self) -> Result<TransportEvent, TransportError> {
        self.0
            .recv()
            .await
            .unwrap_or_else(|| Err(TransportError::Closed("peer ended".into())))
            .map(TransportEvent::ServerMessage)
    }
}
#[async_trait]
impl WorkerTransport for ChannelTransport {
    async fn connect(
        &self,
        _: &str,
        _: Option<&str>,
        _: Vec<(String, String)>,
    ) -> Result<Connected, TransportError> {
        let incoming = self
            .incoming
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| TransportError::Io("no more connections".into()))?;
        Ok(Connected {
            sink: Arc::new(ChannelSink {
                sent: self.sent.clone(),
                closed: self.closed.clone(),
                stall_writes: self.stall_writes.clone(),
            }),
            stream: Box::new(ChannelStream(incoming)),
        })
    }
}

/// Holds decisions and launches at explicit gates so tests control completion order.
struct GatedHost {
    started: tokio::sync::mpsc::UnboundedSender<&'static str>,
    decisions: Arc<tokio::sync::Semaphore>,
    launches: Arc<tokio::sync::Semaphore>,
    dropped: Arc<std::sync::atomic::AtomicUsize>,
}
struct OperationGuard(Arc<std::sync::atomic::AtomicUsize>);
impl Drop for OperationGuard {
    fn drop(&mut self) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}
#[async_trait]
impl WorkerHost for GatedHost {
    async fn on_job_request(&self, _: crate::JobRequest) -> JobDecision {
        let _guard = OperationGuard(self.dropped.clone());
        self.started.send("decision").unwrap();
        self.decisions.acquire().await.unwrap().forget();
        JobDecision::Accept(Default::default())
    }
    async fn launch_job(&self, info: crate::RunningJobInfo) -> Result<JobId, HostError> {
        let _guard = OperationGuard(self.dropped.clone());
        self.started.send("launch").unwrap();
        self.launches.acquire().await.unwrap().forget();
        Ok(info.job.id)
    }
    async fn terminate_job(&self, _: JobId) -> HostResult {
        Ok(())
    }
    async fn user_load(&self) -> Option<f32> {
        Some(0.0)
    }
}

struct RuntimeRig {
    stall_writes: Arc<std::sync::atomic::AtomicBool>,
    runtime: Worker<ChannelTransport, GatedHost, TestClock>,
    handle: WorkerHandle,
    peer: tokio::sync::mpsc::UnboundedSender<Result<ServerMessage, TransportError>>,
    sent: tokio::sync::mpsc::UnboundedReceiver<protocol::WorkerMessage>,
    started: tokio::sync::mpsc::UnboundedReceiver<&'static str>,
    decisions: Arc<tokio::sync::Semaphore>,
    launches: Arc<tokio::sync::Semaphore>,
    dropped: Arc<std::sync::atomic::AtomicUsize>,
    closed: Arc<std::sync::atomic::AtomicUsize>,
}
fn runtime_rig() -> RuntimeRig {
    runtime_rig_with_retry(0)
}

fn runtime_rig_with_retry(max_retry: u32) -> RuntimeRig {
    let stall_writes = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (peer, incoming) = tokio::sync::mpsc::unbounded_channel();
    let (send, sent) = tokio::sync::mpsc::unbounded_channel();
    let (start, started) = tokio::sync::mpsc::unbounded_channel();
    let decisions = Arc::new(tokio::sync::Semaphore::new(0));
    let launches = Arc::new(tokio::sync::Semaphore::new(0));
    let dropped = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let closed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let (runtime, handle) = Worker::with_components(
        ChannelTransport {
            stall_writes: stall_writes.clone(),
            incoming: Mutex::new(Some(incoming)),
            sent: send,
            closed: closed.clone(),
        },
        runtime_cfg(max_retry),
        WorkerState::new(WorkerConfig {
            load_threshold: EnvOption::Fixed(0.7),
            ..Default::default()
        }),
        GatedHost {
            started: start,
            decisions: decisions.clone(),
            launches: launches.clone(),
            dropped: dropped.clone(),
        },
        TestClock(tokio::time::Instant::now()),
    );
    RuntimeRig {
        runtime,
        handle,
        peer,
        sent,
        started,
        decisions,
        launches,
        dropped,
        closed,
        stall_writes,
    }
}
fn offer(id: &str) -> ServerMessage {
    ServerMessage {
        message: Some(server_message::Message::Availability(protocol::AvailabilityRequest {
            job: Some(protocol::Job { id: id.into(), ..Default::default() }),
            ..Default::default()
        })),
    }
}
fn assignment_message(id: &str) -> ServerMessage {
    ServerMessage {
        message: Some(server_message::Message::Assignment(protocol::JobAssignment {
            job: Some(protocol::Job { id: id.into(), ..Default::default() }),
            token: "token".into(),
            url: Some("wss://assigned-sfu".into()),
        })),
    }
}
async fn until_sent(
    sent: &mut tokio::sync::mpsc::UnboundedReceiver<protocol::WorkerMessage>,
    predicate: impl Fn(&protocol::WorkerMessage) -> bool,
) {
    tokio::time::timeout(Duration::from_secs(4), async {
        loop {
            if predicate(&sent.recv().await.expect("runtime stopped")) {
                break;
            }
        }
    })
    .await
    .expect("expected worker message");
}

#[tokio::test(start_paused = true)]
async fn slow_decisions_do_not_block_other_requests_status_timers_or_close() {
    let RuntimeRig { mut runtime, handle, peer, mut sent, mut started, dropped, .. } =
        runtime_rig();
    peer.send(register_message("worker")).unwrap();
    let task = tokio::spawn(async move { runtime.run().await });
    peer.send(Ok(offer("a"))).unwrap();
    peer.send(Ok(offer("b"))).unwrap();
    for _ in 0..2 {
        assert_eq!(started.recv().await, Some("decision"));
    }
    until_sent(&mut sent, |m| {
        matches!(m.message, Some(livekit_protocol::worker_message::Message::UpdateWorker(_)))
    })
    .await;
    assert_eq!(handle.snapshot().reserved_slots, 2);
    handle.send(WorkerCommand::Close).unwrap();
    tokio::time::timeout(Duration::from_secs(1), task).await.unwrap().unwrap().unwrap();
    assert_eq!(
        dropped.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "close must cancel outstanding host work"
    );
}

#[tokio::test(start_paused = true)]
async fn drain_timeout_is_reported_while_a_host_decision_is_pending() {
    let RuntimeRig { mut runtime, handle, peer, mut started, sent: _sent, .. } = runtime_rig();
    peer.send(register_message("worker")).unwrap();
    peer.send(Ok(offer("a"))).unwrap();
    let mut notifications = handle.subscribe();
    let task = tokio::spawn(async move { runtime.run().await });
    assert_eq!(started.recv().await, Some("decision"));
    assert_eq!(handle.drain(Some(Duration::from_secs(1))).await, Err("drain timed out".into()));
    assert_eq!(handle.snapshot().drain_outcome.as_deref(), Some("timed_out"));
    let mut observed = false;
    while let Ok(event) = notifications.try_recv() {
        observed |= matches!(event, WorkerNotification::DrainTimedOut(_));
    }
    assert!(observed);
    handle.send(WorkerCommand::Close).unwrap();
    task.await.unwrap().unwrap();
}

#[tokio::test(start_paused = true)]
async fn slow_launch_allows_status_updates_and_drain_waits_for_early_job_completion() {
    let RuntimeRig {
        mut runtime, handle, peer, mut sent, mut started, decisions, launches, ..
    } = runtime_rig();
    decisions.add_permits(1);
    peer.send(register_message("worker")).unwrap();
    peer.send(Ok(offer("a"))).unwrap();
    let task = tokio::spawn(async move { runtime.run().await });
    assert_eq!(started.recv().await, Some("decision"));
    until_sent(&mut sent, |m| {
        matches!(m.message, Some(livekit_protocol::worker_message::Message::Availability(_)))
    })
    .await;
    peer.send(Ok(assignment_message("a"))).unwrap();
    assert_eq!(started.recv().await, Some("launch"));
    until_sent(&mut sent, |m| {
        matches!(m.message, Some(livekit_protocol::worker_message::Message::UpdateWorker(_)))
    })
    .await;
    handle
        .send(WorkerCommand::JobStatus {
            job_id: "a".into(),
            status: protocol::JobStatus::JsSuccess,
        })
        .unwrap();
    let h = handle.clone();
    let drain = tokio::spawn(async move { h.drain(Some(Duration::from_secs(5))).await });
    tokio::task::yield_now().await;
    assert!(!drain.is_finished(), "drain must wait for launch completion");
    launches.add_permits(1);
    drain.await.unwrap().unwrap();
    assert_eq!(handle.snapshot().active_jobs, 0);
    handle.send(WorkerCommand::Close).unwrap();
    task.await.unwrap().unwrap();
}

#[tokio::test(start_paused = true)]
async fn registration_failure_closes_the_transport_and_emits_stopped() {
    let RuntimeRig { mut runtime, handle, peer, closed, sent: _sent, .. } = runtime_rig();
    peer.send(Ok(offer("not-a-registration"))).unwrap();
    let mut notifications = handle.subscribe();
    assert!(matches!(runtime.run().await, Err(RuntimeError::RetriesExhausted { retries: 0, .. })));
    assert_eq!(closed.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(matches!(notifications.recv().await.unwrap(), WorkerNotification::Stopped(_)));
    assert!(handle.send(WorkerCommand::Close).is_err());
}

#[tokio::test(start_paused = true)]
async fn close_during_registration_does_not_wait_for_the_peer() {
    let RuntimeRig { mut runtime, handle, peer: _peer, mut sent, closed, .. } = runtime_rig();
    let task = tokio::spawn(async move { runtime.run().await });
    sent.recv().await.expect("register must be sent");
    handle.send(WorkerCommand::Close).unwrap();
    tokio::time::timeout(Duration::from_secs(1), task).await.unwrap().unwrap().unwrap();
    assert_eq!(closed.load(std::sync::atomic::Ordering::SeqCst), 1);
}

struct HangingTransport(tokio::sync::Notify);
#[async_trait]
impl WorkerTransport for HangingTransport {
    async fn connect(
        &self,
        _: &str,
        _: Option<&str>,
        _: Vec<(String, String)>,
    ) -> Result<Connected, TransportError> {
        self.0.notify_one();
        std::future::pending().await
    }
}
#[tokio::test(start_paused = true)]
async fn closing_cancels_an_unfinished_dial() {
    let (mut runtime, handle) = Worker::with_components(
        HangingTransport(tokio::sync::Notify::new()),
        runtime_cfg(0),
        WorkerState::new(WorkerConfig::default()),
        LoadHost::default(),
        TestClock(tokio::time::Instant::now()),
    );
    let h = handle.clone();
    let driver = async { runtime.run().await };
    let closer = async {
        tokio::time::sleep(Duration::from_millis(1)).await;
        h.send(WorkerCommand::Close).unwrap();
    };
    let (result, ()) =
        tokio::time::timeout(Duration::from_secs(1), async { tokio::join!(driver, closer) })
            .await
            .unwrap();
    result.unwrap();
    assert_eq!(handle.snapshot().finished.as_deref(), Some("closed"));
}

#[tokio::test(start_paused = true)]
async fn dropping_command_handles_does_not_starve_registration_deadlines() {
    let RuntimeRig { mut runtime, handle, peer: _peer, sent: _sent, closed, .. } = runtime_rig();
    drop(handle);
    let result = tokio::time::timeout(Duration::from_secs(11), runtime.run()).await.unwrap();
    assert!(matches!(result, Err(RuntimeError::RetriesExhausted { retries: 0, .. })));
    assert_eq!(closed.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn write_failure_closes_connection_and_exposes_cause() {
    let RuntimeRig { mut runtime, handle, peer: _peer, sent, closed, .. } = runtime_rig();
    drop(sent);
    let error = runtime.run().await.unwrap_err().to_string();
    assert!(error.contains("channel closed"), "{error}");
    assert_eq!(handle.snapshot().finished.as_deref(), Some(error.as_str()));
    assert_eq!(closed.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(handle.send(WorkerCommand::Close).is_err());
}

#[tokio::test(start_paused = true)]
async fn dropping_runtime_notifies_waiters_and_closes_command_queue() {
    let RuntimeRig { runtime, handle, .. } = runtime_rig();
    let mut notifications = handle.subscribe();
    drop(runtime);
    assert_eq!(
        notifications.recv().await.unwrap(),
        WorkerNotification::Stopped(WorkerStopped { reason: "worker runtime cancelled".into() })
    );
    assert!(handle.snapshot().finished.is_some());
    assert!(handle.send(WorkerCommand::Close).is_err());
}

#[tokio::test(start_paused = true)]
async fn a_stalled_write_after_registration_expires_and_closes_the_connection() {
    let RuntimeRig { mut runtime, handle, peer, sent: _sent, closed, stall_writes, .. } =
        runtime_rig();
    let mut notifications = handle.subscribe();
    peer.send(register_message("worker")).unwrap();
    let task = tokio::spawn(async move { runtime.run().await });
    assert!(matches!(notifications.recv().await.unwrap(), WorkerNotification::Registered(_)));
    stall_writes.store(true, std::sync::atomic::Ordering::SeqCst);
    // A periodic status write must expire even while load/status timers continue firing.
    let error = tokio::time::timeout(Duration::from_secs(15), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err()
        .to_string();
    assert!(error.contains("control write deadline exceeded"), "{error}");
    assert_eq!(closed.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(handle.snapshot().finished.as_deref(), Some(error.as_str()));
}

#[tokio::test(start_paused = true)]
async fn disconnect_cancels_pending_host_decisions_before_reconnect_finishes() {
    let RuntimeRig { mut runtime, handle, peer, mut started, sent: _sent, dropped, .. } =
        runtime_rig_with_retry(5);
    peer.send(register_message("worker")).unwrap();
    peer.send(Ok(offer("pending"))).unwrap();
    let task = tokio::spawn(async move { runtime.run().await });
    assert_eq!(started.recv().await, Some("decision"));
    peer.send(Err(TransportError::Closed("peer disconnected".into()))).unwrap();
    tokio::time::timeout(Duration::from_secs(1), async {
        while dropped.load(std::sync::atomic::Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("disconnect must cancel the outstanding decision");
    assert!(!task.is_finished(), "worker should still be reconnecting");
    assert_eq!(dropped.load(std::sync::atomic::Ordering::SeqCst), 1);
    handle.send(WorkerCommand::Close).unwrap();
    task.await.unwrap().unwrap();
}

#[tokio::test(start_paused = true)]
async fn a_full_command_queue_returns_the_unsent_command() {
    let RuntimeRig { runtime: _runtime, handle, .. } = runtime_rig();
    for _ in 0..64 {
        handle.send(WorkerCommand::Close).unwrap();
    }
    let command = WorkerCommand::Drain { timeout: Some(Duration::from_secs(3)) };
    assert_eq!(handle.send(command.clone()), Err(command));
}

