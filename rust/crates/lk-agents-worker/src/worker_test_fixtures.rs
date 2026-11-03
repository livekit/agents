//! Scripted transports, gated hosts and worker rigs used by the worker tests.

use crate::transport::{
    Connected, TransportError, TransportEvent, WorkerSink, WorkerStream, WorkerTransport,
};

use parking_lot::Mutex;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;

use crate::EnvOption;
use crate::JobDecision;
use crate::WorkerConfig;
use crate::*;
use crate::{HostError, JobId, WorkerHost};
use livekit_protocol::{self as protocol, RegisterWorkerResponse, ServerMessage, server_message};
use lk_agents_abi::{Clock, Ts};

// A scripted transport

pub(super) enum Step {
    /// The connect attempt fails.
    ConnectFails,
    /// The connect attempt succeeds and the socket yields these messages, then blocks forever.
    Serves(Vec<Result<ServerMessage, TransportError>>),
}

#[derive(Default)]
pub(super) struct Wire {
    pub(super) sent: Vec<protocol::WorkerMessage>,
    pub(super) attempts: u32,
}

pub(super) struct ScriptedTransport {
    pub(super) script: Mutex<std::collections::VecDeque<Step>>,
    pub(super) wire: Arc<Mutex<Wire>>,
}

pub(super) struct Sink(pub(super) Arc<Mutex<Wire>>);

#[async_trait]
impl WorkerSink for Sink {
    async fn send(&self, message: protocol::WorkerMessage) -> Result<(), TransportError> {
        self.0.lock().sent.push(message);
        Ok(())
    }
    async fn close(&self) {}
}

pub(super) struct Stream(
    pub(super) std::collections::VecDeque<Result<ServerMessage, TransportError>>,
);

#[async_trait]
impl WorkerStream for Stream {
    async fn recv(&mut self) -> Result<TransportEvent, TransportError> {
        match self.0.pop_front() {
            Some(message) => message.map(Box::new).map(TransportEvent::ServerMessage),
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
        self.wire.lock().attempts += 1;
        match self.script.lock().pop_front() {
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
pub(super) struct LoadHost {
    pub(super) user: Option<f32>,

    /// How many times `load_fnc` was asked.
    pub(super) user_calls: Arc<Mutex<u32>>,
}

#[async_trait]
impl WorkerHost for LoadHost {
    async fn on_job_request(&self, _r: crate::JobRequest) -> JobDecision {
        JobDecision::Reject
    }
    async fn launch_job(&self, _i: crate::RunningJobInfo) -> Result<JobId, HostError> {
        Err(HostError::new("no pool"))
    }
    async fn terminate_job(&self, _id: JobId) -> Result<(), HostError> {
        Ok(())
    }
    async fn user_load(&self) -> Option<f32> {
        *self.user_calls.lock() += 1;
        self.user
    }
}

pub(super) fn runtime_cfg(max_retry: u32) -> RuntimeConfig {
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

pub(super) struct FrozenClock;
impl Clock for FrozenClock {
    fn now(&self) -> Ts {
        Ts::ZERO
    }
    fn wall(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH
    }
    /// Frozen: a wait on it never ends. No test here arms one.
    fn wait_until(&self, _at: Ts) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(async move { std::future::pending::<()>().await })
    }
}

pub(super) fn register_message(worker_id: &str) -> Result<ServerMessage, TransportError> {
    let msg = ServerMessage {
        message: Some(server_message::Message::Register(RegisterWorkerResponse {
            worker_id: worker_id.to_owned(),
            ..Default::default()
        })),
    };
    Ok(msg)
}

pub(super) fn conn(steps: Vec<Step>, max_retry: u32) -> (Worker, Arc<Mutex<Wire>>, WorkerClient) {
    conn_with(
        steps,
        max_retry,
        WorkerConfig { ws_url: "wss://gw".into(), ..Default::default() },
        LoadHost::default(),
    )
}

pub(super) fn conn_with(
    steps: Vec<Step>,
    max_retry: u32,
    cfg_worker: WorkerConfig,
    host: LoadHost,
) -> (Worker, Arc<Mutex<Wire>>, WorkerClient) {
    let wire = Arc::new(Mutex::new(Wire::default()));
    let t = ScriptedTransport { script: Mutex::new(steps.into()), wire: wire.clone() };
    let (c, h) = Worker::with_components(
        Arc::new(t),
        runtime_cfg(max_retry),
        cfg_worker,
        Arc::new(host),
        Arc::new(FrozenClock),
        blocking_io(),
    );
    (c, wire, h)
}

pub(super) fn load_conn(host: LoadHost) -> (Worker, WorkerClient) {
    // Finite, so `worker_load` is actually compared against something (devmode's is `inf`).
    let cfg = WorkerConfig {
        ws_url: "wss://gw".into(),
        load_threshold: EnvOption::Fixed(0.7),
        ..Default::default()
    };
    let (c, _wire, h) = conn_with(vec![], 0, cfg, host);
    (c, h)
}

pub(super) struct TestClock(pub(super) tokio::time::Instant);
impl Clock for TestClock {
    fn now(&self) -> Ts {
        Ts::from_nanos(self.0.elapsed().as_nanos() as u64)
    }
    fn wall(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH
    }

    fn wait_until(&self, at: Ts) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(async move {
            tokio::time::sleep(at.saturating_duration_since(self.now())).await;
        })
    }
}

pub(super) struct ChannelTransport {
    pub(super) stall_writes: Arc<std::sync::atomic::AtomicBool>,

    /// One peer per connection, in dial order.
    pub(super) incoming: Connections,

    pub(super) sent: tokio::sync::mpsc::UnboundedSender<protocol::WorkerMessage>,
    pub(super) closed: Arc<std::sync::atomic::AtomicUsize>,
}
pub(super) struct ChannelSink {
    pub(super) stall_writes: Arc<std::sync::atomic::AtomicBool>,
    pub(super) sent: tokio::sync::mpsc::UnboundedSender<protocol::WorkerMessage>,
    pub(super) closed: Arc<std::sync::atomic::AtomicUsize>,
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

pub(super) struct ChannelStream(
    pub(super) tokio::sync::mpsc::UnboundedReceiver<Result<ServerMessage, TransportError>>,
);
#[async_trait]
impl WorkerStream for ChannelStream {
    async fn recv(&mut self) -> Result<TransportEvent, TransportError> {
        self.0
            .recv()
            .await
            .unwrap_or_else(|| Err(TransportError::Closed("peer ended".into())))
            .map(Box::new)
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
            .pop_front()
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

pub(super) type Peer = tokio::sync::mpsc::UnboundedSender<Result<ServerMessage, TransportError>>;
pub(super) type Connections = Arc<
    Mutex<
        std::collections::VecDeque<
            tokio::sync::mpsc::UnboundedReceiver<Result<ServerMessage, TransportError>>,
        >,
    >,
>;

/// The load the host reports, its sample count, and a gate it waits on when closed.
pub(super) struct HostLoad {
    pub(super) value: Mutex<f32>,
    pub(super) samples: std::sync::atomic::AtomicUsize,
    pub(super) gated: std::sync::atomic::AtomicBool,
    pub(super) gate: tokio::sync::Semaphore,
}

impl Default for HostLoad {
    fn default() -> Self {
        Self {
            value: Mutex::new(0.0),
            samples: Default::default(),
            gated: Default::default(),
            gate: tokio::sync::Semaphore::new(0),
        }
    }
}

/// Holds decisions and launches at explicit gates so tests control completion order.
pub(super) struct GatedHost {
    pub(super) load: Arc<HostLoad>,
    pub(super) launched: Arc<Mutex<Vec<crate::RunningJobInfo>>>,
    pub(super) started: tokio::sync::mpsc::UnboundedSender<&'static str>,
    pub(super) decisions: Arc<tokio::sync::Semaphore>,
    pub(super) launches: Arc<tokio::sync::Semaphore>,
    pub(super) dropped: Arc<std::sync::atomic::AtomicUsize>,
    /// Launches fail with a host error.
    pub(super) failing_launches: Arc<std::sync::atomic::AtomicBool>,
}

pub(super) struct OperationGuard(pub(super) Arc<std::sync::atomic::AtomicUsize>);
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
        JobDecision::Accept(crate::AcceptArguments {
            identity: "agent-identity".into(),
            name: "Agent Name".into(),
            metadata: "participant-metadata".into(),
            attributes: vec![(lk_agents_abi::ATTRIBUTE_AGENT_NAME.into(), "per-job-agent".into())],
        })
    }
    async fn launch_job(&self, info: crate::RunningJobInfo) -> Result<JobId, HostError> {
        let _guard = OperationGuard(self.dropped.clone());
        self.launched.lock().push(info.clone());
        self.started.send("launch").unwrap();
        self.launches.acquire().await.unwrap().forget();
        if self.failing_launches.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(HostError::new("launch failed"));
        }
        Ok(info.job.id)
    }
    async fn terminate_job(&self, _: JobId) -> HostResult {
        Ok(())
    }
    async fn user_load(&self) -> Option<f32> {
        if self.load.gated.load(std::sync::atomic::Ordering::SeqCst) {
            self.load.gate.acquire().await.unwrap().forget();
        }
        self.load.samples.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Some(*self.load.value.lock())
    }
}

pub(super) struct RuntimeRig {
    pub(super) connections: Connections,
    pub(super) load: Arc<HostLoad>,
    pub(super) launched: Arc<Mutex<Vec<crate::RunningJobInfo>>>,
    pub(super) stall_writes: Arc<std::sync::atomic::AtomicBool>,
    pub(super) runtime: Worker,
    pub(super) handle: WorkerClient,
    pub(super) peer: tokio::sync::mpsc::UnboundedSender<Result<ServerMessage, TransportError>>,
    pub(super) sent: tokio::sync::mpsc::UnboundedReceiver<protocol::WorkerMessage>,
    pub(super) started: tokio::sync::mpsc::UnboundedReceiver<&'static str>,
    pub(super) decisions: Arc<tokio::sync::Semaphore>,
    pub(super) launches: Arc<tokio::sync::Semaphore>,
    pub(super) dropped: Arc<std::sync::atomic::AtomicUsize>,
    pub(super) closed: Arc<std::sync::atomic::AtomicUsize>,
    pub(super) failing_launches: Arc<std::sync::atomic::AtomicBool>,
}
pub(super) fn runtime_rig() -> RuntimeRig {
    runtime_rig_with_retry(0)
}

pub(super) fn runtime_rig_with_retry(max_retry: u32) -> RuntimeRig {
    let cfg = WorkerConfig { load_threshold: EnvOption::Fixed(0.7), ..Default::default() };
    runtime_rig_with(max_retry, cfg)
}

pub(super) fn runtime_rig_with(max_retry: u32, cfg: WorkerConfig) -> RuntimeRig {
    let stall_writes = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (peer, incoming) = tokio::sync::mpsc::unbounded_channel();
    let (send, sent) = tokio::sync::mpsc::unbounded_channel();
    let (start, started) = tokio::sync::mpsc::unbounded_channel();
    let decisions = Arc::new(tokio::sync::Semaphore::new(0));
    let launches = Arc::new(tokio::sync::Semaphore::new(0));
    let dropped = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let closed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let failing_launches = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let connections: Connections = Arc::new(Mutex::new([incoming].into()));
    let load = Arc::new(HostLoad::default());
    let launched = Arc::new(Mutex::new(Vec::new()));
    let (runtime, handle) = Worker::with_components(
        Arc::new(ChannelTransport {
            stall_writes: stall_writes.clone(),
            incoming: connections.clone(),
            sent: send,
            closed: closed.clone(),
        }),
        runtime_cfg(max_retry),
        cfg,
        Arc::new(GatedHost {
            load: load.clone(),
            launched: launched.clone(),
            started: start,
            decisions: decisions.clone(),
            launches: launches.clone(),
            dropped: dropped.clone(),
            failing_launches: failing_launches.clone(),
        }),
        Arc::new(TestClock(tokio::time::Instant::now())),
        blocking_io(),
    );
    RuntimeRig {
        connections,
        load,
        launched,
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
        failing_launches,
    }
}
pub(super) fn offer(id: &str) -> ServerMessage {
    ServerMessage {
        message: Some(server_message::Message::Availability(protocol::AvailabilityRequest {
            job: Some(protocol::Job { id: id.into(), ..Default::default() }),
            ..Default::default()
        })),
    }
}
pub(super) fn assignment_message(id: &str) -> ServerMessage {
    ServerMessage {
        message: Some(server_message::Message::Assignment(protocol::JobAssignment {
            job: Some(protocol::Job { id: id.into(), ..Default::default() }),
            token: "token".into(),
            url: Some("wss://assigned-sfu".into()),
        })),
    }
}

pub(super) async fn until_sent(
    sent: &mut tokio::sync::mpsc::UnboundedReceiver<protocol::WorkerMessage>,
    predicate: impl Fn(&protocol::WorkerMessage) -> bool,
) -> protocol::WorkerMessage {
    tokio::time::timeout(Duration::from_secs(4), async {
        loop {
            let message = sent.recv().await.expect("runtime stopped");
            if predicate(&message) {
                break message;
            }
        }
    })
    .await
    .expect("expected worker message")
}

/// A server that never answers the dial.
pub(super) struct HangingTransport(tokio::sync::Notify, crate::transport::TransportKind);
impl HangingTransport {
    pub(super) fn new(kind: crate::transport::TransportKind) -> Self {
        Self(tokio::sync::Notify::new(), kind)
    }
}
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

    fn kind(&self) -> crate::transport::TransportKind {
        self.1
    }
}

// Each test owns its blocking service so saturation cannot affect other tests.
pub(super) fn blocking_io() -> lk_agents_net::blocking_io::BlockingIo {
    lk_agents_net::blocking_io::BlockingIo::with_limits("lk-test-io", 4, &[4]).unwrap()
}
