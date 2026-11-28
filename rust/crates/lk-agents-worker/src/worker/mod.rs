use crate::http::IncomingRequest;
use crate::hw::cpu::CpuMonitor;
use crate::transport::{
    Connected, EndpointStream, TransportError, TransportEvent, TransportKind, WorkerSink,
    WorkerTransport,
};
use crate::{DefaultLoadCalc, WorkerConfig};
use crate::{JobDecision, JobRequest, RunningJobInfo};
use async_trait::async_trait;
use futures_util::{FutureExt, StreamExt, future::BoxFuture, stream::FuturesUnordered};
use livekit_protocol::{
    self as proto, JobStatus, RegisterWorkerResponse, ServerMessage, WorkerMessage,
};
use lk_agents_abi::errors::{ApiError, ErrorKind};
use lk_agents_abi::{Clock, Spawner, TimerSet, Ts, WallAnchor};
use lk_agents_abi::{OwnedTask, SpawnerExt};
use lk_agents_net::job_token::JobToken;
use lk_agents_net::owned::Background;
use lk_agents_net::signal;
use parking_lot::Mutex;
use tokio::sync::oneshot;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

// Host futures are cancelled when the worker stops; writes are ordered, bounded and
// per-connection, and a reconnect regenerates state rather than replaying.

/// Default ping interval.
pub const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);

const WORKER_PING_INTERVAL: Duration = Duration::from_secs(1);
const WORKER_PONG_TIMEOUT: Duration = Duration::from_secs(2);

const MAX_PENDING_WRITES: usize = 64;

/// How long the loop sleeps when no timer is due; any input wakes it sooner.
const IDLE_WAKE: Duration = Duration::from_secs(3600);

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("{0}")]
    Configuration(String),
    #[error(transparent)]
    Connection(#[from] TransportError),
    #[error("expected register response as first message")]
    ExpectedRegister,
    #[error("worker connection failed after {retries} retries: {source}")]
    RetriesExhausted { retries: u32, source: Box<RuntimeError> },
    #[error(
        "could not reach {url} over WebTransport (QUIC over UDP): no answer within {after:?}. A \
         worker with agent endpoints or HTTP routes registers over WebTransport; the server does \
         not accept it (LiveKit Cloud production does not yet) or the network drops UDP"
    )]
    WebTransportUnreachable { url: String, after: Duration },
}

mod jobs;
use jobs::{ConnState, Pending, WorkerTimer};
mod decision;
mod load_driver;
use jobs::JOB_TOKEN_TIMEOUT;
pub use jobs::{Slot, retry_delay};
#[cfg(test)]
mod host_failure_tests;
#[cfg(test)]
#[path = "worker_tests.rs"]
mod tests;

/// What a client asks of a running worker.
#[derive(Debug)]
pub enum WorkerCommand {
    Drain {
        timeout: Option<Duration>,
    },
    Close,
    JobStatus {
        job_id: JobId,
        status: JobStatus,
    },
    /// Token mode: asks the server for a token with `claims` for the job `job_id`.
    CreateJobToken {
        job_id: JobId,
        claims: String,
        reply: oneshot::Sender<Result<JobToken, ApiError>>,
    },
    /// Token mode, on a reload: tells the server that these kept jobs now run on this worker, so
    /// that it issues their tokens.
    MigrateJobs(Vec<RunningJobInfo>),
}

/// What the worker tells its server as it happens: its registrations, a lost connection, a
/// drain's outcome and its end.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerNotification {
    Registered(RegisteredInfo),
    /// The registered connection was lost; the worker registers again.
    Disconnected,
    Drained(DrainOutcome),
    Stopped(WorkerStopped),
}

pub use lk_agents_abi::worker::types::{DrainOutcome, RegisteredInfo, Snapshot, WorkerStopped};

/// A drain's semantic failure. Error messages are display text, never discriminants.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DrainError {
    #[error("drain timed out")]
    TimedOut,
    #[error("server closed during drain")]
    Closed,
    #[error("worker stopped before drain completed: {0}")]
    WorkerStopped(String),
    #[error("worker stopped or command queue full")]
    WorkerUnavailable,
    #[error("drain outcome channel closed")]
    OutcomeUnavailable,
    #[error("server cleanup failed during drain: {0}")]
    Cleanup(crate::server::ServerError),
    #[error(transparent)]
    Host(#[from] HostError),
}

impl From<DrainError> for lk_agents_abi::worker::ipc_host::WorkerError {
    fn from(error: DrainError) -> Self {
        match error {
            DrainError::TimedOut => Self::TimedOut(error.to_string()),
            DrainError::Host(error) => Self::Host(error),
            error => Self::Failed(error.to_string()),
        }
    }
}

#[derive(Clone)]
pub struct WorkerClient {
    tx: tokio::sync::mpsc::Sender<WorkerCommand>,
    control: Arc<signal::Shared<WorkerControl>>,
    snapshot: Arc<parking_lot::Mutex<Snapshot>>,
    notifications: tokio::sync::broadcast::Sender<WorkerNotification>,
    /// Times out a job token request.
    clock: Arc<dyn Clock>,
}

impl WorkerClient {
    /// Fails when the worker has stopped or its queue is full; the command is handed back.
    pub fn send(&self, cmd: WorkerCommand) -> Result<(), WorkerCommand> {
        if self.tx.is_closed() {
            return Err(cmd);
        }
        match cmd {
            WorkerCommand::Close => {
                self.control.modify(|control| control.close = true);
                Ok(())
            }
            WorkerCommand::Drain { timeout } => {
                self.control.modify(|control| {
                    control.drain.get_or_insert(timeout);
                });
                Ok(())
            }
            command => self.tx.try_send(command).map_err(|e| e.into_inner()),
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        self.snapshot.lock().clone()
    }

    /// Token mode: asks the server for a token with `claims` (JSON, the payload of an access
    /// token) for the job `job_id`, which runs on this worker.
    pub async fn create_job_token(&self, job_id: &str, claims: &str) -> Result<JobToken, ApiError> {
        let (reply, answer) = oneshot::channel();
        let (job_id, claims) = (job_id.to_owned(), claims.to_owned());
        if self.send(WorkerCommand::CreateJobToken { job_id, claims, reply }).is_err() {
            return Err(ApiError::new(ErrorKind::Connection, "the worker stopped"));
        }
        match lk_agents_abi::timeout(&*self.clock, JOB_TOKEN_TIMEOUT, answer).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(ApiError::new(
                ErrorKind::Connection,
                "the connection to the LiveKit server closed before it answered the job token \
                 request",
            )),
            Err(_) => Err(ApiError::new(
                ErrorKind::Timeout,
                format!(
                    "the LiveKit server did not answer the job token request within {}s",
                    JOB_TOKEN_TIMEOUT.as_secs()
                ),
            )),
        }
    }

    /// Holds 128 notifications; a lagging receiver gets `RecvError::Lagged`.
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<WorkerNotification> {
        self.notifications.subscribe()
    }

    /// Waits for the state's drain result, including outstanding decisions and launches.
    pub async fn drain(&self, timeout: Option<Duration>) -> Result<(), DrainError> {
        let mut notifications = self.subscribe();
        self.send(WorkerCommand::Drain { timeout }).map_err(|_| DrainError::WorkerUnavailable)?;
        loop {
            let state = self.snapshot();
            if let Some(outcome) = state.drain_outcome {
                return match outcome {
                    DrainOutcome::Finished => Ok(()),
                    DrainOutcome::TimedOut => Err(DrainError::TimedOut),
                };
            }
            if let Some(reason) = state.finished {
                return Err(DrainError::WorkerStopped(reason));
            }
            match notifications.recv().await {
                Ok(_) | Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    return Err(DrainError::WorkerUnavailable);
                }
            }
        }
    }
}

#[derive(Clone, Default, PartialEq)]
struct WorkerControl {
    close: bool,
    drain: Option<Option<Duration>>,
}

pub struct RuntimeConfig {
    pub server_url: String,
    pub worker_token: Option<String>,
    /// Authorization is minted separately for each connection attempt.
    pub headers: Vec<(String, String)>,
    /// When both are empty, the worker uses `token` if it is set. Otherwise it sends only
    /// `headers`.
    pub api_key: String,
    pub api_secret: String,
    /// Token mode: the worker token (`LIVEKIT_TOKEN`), sent as is when no key and no secret are
    /// set. Read at each connection attempt, because the CLI can replace it.
    pub token: Arc<Mutex<String>>,
    pub max_retry: u32,
    /// Bounds dialing, registration, writes and graceful transport shutdown.
    pub connection_timeout: Duration,
}

pub struct Worker {
    transport: Arc<dyn WorkerTransport>,
    runtime_cfg: RuntimeConfig,
    cfg: WorkerConfig,
    host: Arc<dyn WorkerHost>,
    pub(super) blocking_io: lk_agents_net::blocking_io::BlockingIo,
    clock: Arc<dyn Clock>,
    /// Where the host's decisions and the endpoint exchanges run.
    spawner: Arc<dyn Spawner>,
    default_load: DefaultLoadCalc,
    load_health: crate::load::LoadHealth,
    load_available: bool,
    load_requested: lk_agents_abi::Ts,
    /// How `load_fnc` fails now, if it does: logged once per change.
    load_failure: Option<std::mem::Discriminant<crate::load::LoadError>>,
    starting_jobs: BTreeMap<JobId, lk_agents_abi::Ts>,
    cpu: Option<Box<CpuMonitor>>,
    rx: tokio::sync::mpsc::Receiver<WorkerCommand>,
    control: Arc<signal::Shared<WorkerControl>>,
    snapshot: Arc<parking_lot::Mutex<Snapshot>>,
    notifications: tokio::sync::broadcast::Sender<WorkerNotification>,
    operations: FuturesUnordered<BoxFuture<'static, Completion>>,
    load_pending: bool,
    send_failure: Option<TransportError>,
    /// Joins decisions after their owning pending states cancel them.
    decision_tasks: Arc<lk_agents_net::task::TaskSet>,
    /// Endpoint exchanges in flight, each cancelled when removed.
    endpoints: HashMap<u64, OwnedTask>,
    next_endpoint: u64,
    closing: bool,
    /// Bumped on each connection: a decision or launch of an older one is stale.
    generation: u64,
    /// The connection is to be replaced once its writes settled.
    rollover_requested: bool,
    timers: TimerSet<WorkerTimer>,
    conn: ConnState,
    /// A registration succeeded once: later connection failures are reconnects.
    registered_once: bool,
    worker_id: String,
    worker_load: f32,
    pending: Vec<Pending>,
    /// Each settled job's last status, answered again to a repeated assignment.
    dispositions: BTreeMap<JobId, JobStatus>,
    assignment_capacity: usize,
    /// A full assignment history is rolling over: new offers are refused until it reconnects.
    rolling_over: bool,
    /// Requests parked until the next load sample, in arrival order.
    awaiting_load: VecDeque<Slot>,
    active: Vec<RunningJobInfo>,
    next_slot: u32,
    draining: bool,
    drain_reported: bool,
    previous_status: Option<crate::WorkerStatus>,
    /// Places the dispatch timeline in unix time.
    wall: WallAnchor,
    /// Token mode: the job token requests the server has not answered yet, by request id.
    job_token_requests: HashMap<String, oneshot::Sender<Result<JobToken, ApiError>>>,
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.operations.clear();
        let mut snapshot = self.snapshot.lock();
        if snapshot.finished.is_none() {
            let reason = "worker runtime cancelled".to_string();
            snapshot.finished = Some(reason.clone());
            let _ = self.notifications.send(WorkerNotification::Stopped(WorkerStopped { reason }));
        }
    }
}

struct PendingWrite {
    message: WorkerMessage,
    accepted: Option<JobId>,
    deadline: Ts,
}

enum WriteCompletion {
    Message,
    Availability(String),
    Ping,
}

/// Encode the core's ordered acceptance without making another admission decision.
fn availability_message(job_id: &str, arguments: crate::AcceptArguments) -> WorkerMessage {
    WorkerMessage {
        message: Some(livekit_protocol::worker_message::Message::Availability(
            proto::AvailabilityResponse {
                job_id: job_id.into(),
                available: true,
                participant_identity: arguments.identity,
                participant_name: arguments.name,
                participant_metadata: arguments.metadata,
                participant_attributes: arguments.attributes.into_iter().collect(),
                ..Default::default()
            },
        )),
    }
}

/// Short-lived write contention only: not a replay log or a delivery guarantee.
struct Outbox {
    pending: VecDeque<PendingWrite>,
    timeout: Duration,
    clock: Arc<dyn Clock>,
}
impl Outbox {
    fn new(timeout: Duration, clock: Arc<dyn Clock>) -> Self {
        Self { pending: VecDeque::new(), timeout, clock }
    }

    fn push(
        &mut self,
        message: WorkerMessage,
        accepted: Option<JobId>,
    ) -> Result<(), TransportError> {
        use livekit_protocol::worker_message::Message as M;
        use prost::Message;
        if message.encoded_len() > crate::transport::MAX_CONTROL_MESSAGE {
            return Err(TransportError::Failed("control message exceeds 1 MiB".into()));
        }
        // A status sample supersedes older unsent samples, including when entering drain.
        match &message.message {
            Some(M::UpdateWorker(_)) => {
                self.pending.retain(|p| !matches!(p.message.message, Some(M::UpdateWorker(_))))
            }
            Some(M::Ping(_)) => {
                self.pending.retain(|p| !matches!(p.message.message, Some(M::Ping(_))))
            }
            _ => {}
        }
        if self.pending.len() >= MAX_PENDING_WRITES {
            return Err(TransportError::Failed("control write queue is full".into()));
        }
        self.pending.push_back(PendingWrite {
            message,
            accepted,
            deadline: self.clock.now().saturating_add(self.timeout),
        });
        Ok(())
    }
    fn pop_front(&mut self) -> Option<PendingWrite> {
        self.pending.pop_front()
    }
    fn clear(&mut self) {
        self.pending.clear();
    }
}

// Stored between polls so unrelated runtime events cannot cancel a dial or lose a partial write.
enum ConnectionEvent {
    Connected,
    Registered(RegisterWorkerResponse),
    ServerMessage(Box<ServerMessage>),
    AvailabilitySent(String),
    EndpointRequest(crate::EndpointStream),
    /// A write, retry timer or cleanup completed.
    Progress,
}

struct ConnectionIo {
    state: ConnectionState,
    outbox: Outbox,
    cleanups: FuturesUnordered<BoxFuture<'static, ()>>,
    timeout: Duration,
    clock: Arc<dyn Clock>,
}

enum ConnectionState {
    Waiting(Ts),
    Dialing(BoxFuture<'static, Result<Connected, TransportError>>),
    Active(ActiveConnection),
}

struct ActiveConnection {
    transport: Connected,
    /// `None` after the server's registration response has been received.
    registration_deadline: Option<Ts>,
    heartbeat: Option<Heartbeat>,
    writing: Option<BoxFuture<'static, Result<WriteCompletion, TransportError>>>,
}

#[derive(Clone, Copy)]
enum Heartbeat {
    SendPingAt(Ts),
    SendingPing { timestamp: i64, pong_received: bool },
    AwaitingPong { timestamp: i64, sent_at: Ts },
}

impl Heartbeat {
    fn deadline(self) -> Option<Ts> {
        match self {
            Self::SendPingAt(at) => Some(at),
            Self::SendingPing { .. } => None,
            Self::AwaitingPong { sent_at, .. } => Some(sent_at.saturating_add(WORKER_PONG_TIMEOUT)),
        }
    }
}

impl ConnectionIo {
    fn new(timeout: Duration, clock: Arc<dyn Clock>) -> Self {
        Self {
            state: ConnectionState::Waiting(clock.now()),
            outbox: Outbox::new(timeout, clock.clone()),
            cleanups: FuturesUnordered::new(),
            timeout,
            clock,
        }
    }

    fn can_rollover(&self) -> bool {
        self.outbox.pending.is_empty()
            && matches!(&self.state, ConnectionState::Active(active) if active.writing.is_none())
    }

    fn is_connected(&self) -> bool {
        matches!(self.state, ConnectionState::Active(_))
    }

    async fn next_event(
        &mut self,
        transport: Arc<dyn WorkerTransport>,
        config: &RuntimeConfig,
    ) -> Result<ConnectionEvent, RuntimeError> {
        tokio::select! {
            event = self.state.next_event(&mut self.outbox, transport, config, &self.clock) => event,
            _ = self.cleanups.next(), if !self.cleanups.is_empty() => Ok(ConnectionEvent::Progress),
        }
    }

    fn disconnect(&mut self) {
        let waiting = ConnectionState::Waiting(self.clock.now());
        let previous = std::mem::replace(&mut self.state, waiting);
        self.outbox.clear();
        if let ConnectionState::Active(active) = previous {
            let (timeout, clock) = (self.timeout, self.clock.clone());
            // Cancel the in-flight write before attempting to acquire the sink for shutdown.
            drop(active.writing);
            self.cleanups.push(
                async move {
                    let closing = active.transport.sink.close();
                    if lk_agents_abi::timeout(&*clock, timeout, closing).await.is_err() {
                        log::warn!(target: lk_agents_abi::LOG_TARGET, timeout_ms = timeout.as_millis() as u64; "worker transport close timed out");
                    }
                }
                .boxed(),
            );
        }
    }

    fn retry_after(&mut self, delay: Duration) {
        debug_assert!(matches!(self.state, ConnectionState::Waiting(_)));
        self.state = ConnectionState::Waiting(self.clock.now().saturating_add(delay));
    }

    async fn close(&mut self) {
        self.disconnect();
        while self.cleanups.next().await.is_some() {}
    }
}

impl ConnectionState {
    async fn next_event(
        &mut self,
        outbox: &mut Outbox,
        transport: Arc<dyn WorkerTransport>,
        config: &RuntimeConfig,
        clock: &Arc<dyn Clock>,
    ) -> Result<ConnectionEvent, RuntimeError> {
        match self {
            Self::Waiting(retry_at) => {
                clock.wait_until(*retry_at).await;
                *self = Self::Dialing(dial(transport, config, clock.clone())?);
                Ok(ConnectionEvent::Progress)
            }
            Self::Dialing(connecting) => {
                let connected = connecting.await?;
                let registration_deadline =
                    Some(clock.now().saturating_add(config.connection_timeout));
                *self = Self::Active(ActiveConnection {
                    transport: connected,
                    registration_deadline,
                    heartbeat: (transport.kind() == TransportKind::WebSocket)
                        .then(|| Heartbeat::SendPingAt(clock.now())),
                    writing: None,
                });
                Ok(ConnectionEvent::Connected)
            }
            Self::Active(connection) => connection.next_event(outbox, clock).await,
        }
    }
}

fn dial(
    transport: Arc<dyn WorkerTransport>,
    config: &RuntimeConfig,
    clock: Arc<dyn Clock>,
) -> Result<BoxFuture<'static, Result<Connected, TransportError>>, RuntimeError> {
    let url = config.server_url.clone();
    let token = config.worker_token.clone();
    let timeout = config.connection_timeout;
    let mut headers = config.headers.clone();
    if !headers.iter().any(|(name, _)| name.eq_ignore_ascii_case("User-Agent")) {
        headers.push(("User-Agent".to_owned(), lk_agents_net::job_identity::user_agent()));
    }
    if !config.api_key.is_empty() && !config.api_secret.is_empty() {
        let jwt = lk_agents_net::token::worker_token(&config.api_key, &config.api_secret)
            .map_err(|e| RuntimeError::Configuration(e.to_string()))?;
        headers.push(("Authorization".to_owned(), format!("Bearer {jwt}")));
    } else if config.api_key.is_empty() && config.api_secret.is_empty() {
        // Token mode: send the worker token unchanged.
        let token = config.token.lock().clone();
        if !token.is_empty() {
            headers.push(("Authorization".to_owned(), format!("Bearer {token}")));
        }
    }
    Ok(async move {
        let connecting = transport.connect(&url, token.as_deref(), headers);
        lk_agents_abi::timeout(&*clock, timeout, connecting)
            .await
            .map_err(|_| TransportError::TimedOut(timeout))?
    }
    .boxed())
}

impl ActiveConnection {
    async fn next_event(
        &mut self,
        outbox: &mut Outbox,
        clock: &Arc<dyn Clock>,
    ) -> Result<ConnectionEvent, RuntimeError> {
        if self.writing.is_none()
            && let Some(pending) = outbox.pop_front()
        {
            self.writing = Some(pending.send(self.transport.sink.clone(), clock.clone()).boxed());
        }
        let heartbeat = self.heartbeat.filter(|_| self.registration_deadline.is_none());
        tokio::select! {
            biased;
            () = wait_for_deadline(&**clock, heartbeat.and_then(Heartbeat::deadline)) => {
                if matches!(heartbeat, Some(Heartbeat::SendPingAt(_))) {
                    let timestamp = (lk_agents_abi::unix_secs(clock.wall()) * 1000.0) as i64;
                    outbox.push(WorkerMessage {
                        message: Some(proto::worker_message::Message::Ping(proto::WorkerPing {
                            timestamp,
                        })),
                    }, None)?;
                    self.heartbeat = Some(Heartbeat::SendingPing { timestamp, pong_received: false });
                    return Ok(ConnectionEvent::Progress);
                }
                // Check a ready reply once. Other traffic cannot extend the deadline.
                if let Some(received) = self.transport.stream.recv().now_or_never() {
                    let event = self.receive(received?)?;
                    if matches!(self.heartbeat, Some(Heartbeat::SendPingAt(_))) {
                        return Ok(event);
                    }
                }
                Err(TransportError::Failed("worker ping timed out waiting for LiveKit pong".into()).into())
            },
            sent = await_write(&mut self.writing) => {
                self.writing = None;
                Ok(match sent? {
                    WriteCompletion::Availability(job_id) => ConnectionEvent::AvailabilitySent(job_id),
                    WriteCompletion::Message => ConnectionEvent::Progress,
                    WriteCompletion::Ping => {
                        if let Some(Heartbeat::SendingPing { timestamp, pong_received }) = self.heartbeat {
                            let sent_at = clock.now();
                            self.heartbeat = Some(if pong_received {
                                Heartbeat::SendPingAt(sent_at.saturating_add(WORKER_PING_INTERVAL))
                            } else {
                                Heartbeat::AwaitingPong { timestamp, sent_at }
                            });
                        }
                        ConnectionEvent::Progress
                    }
                })
            }
            received = self.transport.stream.recv() => self.receive(received?),
            _ = wait_for_deadline(&**clock, self.registration_deadline) => {
                Err(TransportError::Failed("registration timed out".into()).into())
            }
        }
    }

    fn receive(&mut self, event: TransportEvent) -> Result<ConnectionEvent, RuntimeError> {
        match event {
            TransportEvent::ServerMessage(message) if self.registration_deadline.is_some() => {
                match message.message {
                    Some(proto::server_message::Message::Register(response)) => {
                        self.registration_deadline = None;
                        Ok(ConnectionEvent::Registered(response))
                    }
                    // Prost drops future oneof members. An extension must neither reject
                    // this connection nor satisfy or extend its registration deadline.
                    None => Ok(ConnectionEvent::Progress),
                    _ => Err(RuntimeError::ExpectedRegister),
                }
            }
            TransportEvent::ServerMessage(message) => {
                if let Some(proto::server_message::Message::Pong(pong)) = &message.message {
                    match &mut self.heartbeat {
                        Some(Heartbeat::SendingPing { timestamp, pong_received })
                            if *timestamp == pong.last_timestamp =>
                        {
                            *pong_received = true;
                        }
                        Some(Heartbeat::AwaitingPong { timestamp, sent_at })
                            if *timestamp == pong.last_timestamp =>
                        {
                            self.heartbeat = Some(Heartbeat::SendPingAt(
                                sent_at.saturating_add(WORKER_PING_INTERVAL),
                            ));
                        }
                        _ => {}
                    }
                }
                Ok(ConnectionEvent::ServerMessage(message))
            }
            TransportEvent::EndpointRequest(stream) if self.registration_deadline.is_some() => {
                stream.refuse();
                Ok(ConnectionEvent::Progress)
            }
            TransportEvent::EndpointRequest(stream) => Ok(ConnectionEvent::EndpointRequest(stream)),
        }
    }
}

async fn await_write(
    writing: &mut Option<BoxFuture<'static, Result<WriteCompletion, TransportError>>>,
) -> Result<WriteCompletion, TransportError> {
    match writing {
        Some(writing) => writing.await,
        None => std::future::pending().await,
    }
}

async fn wait_for_deadline(clock: &dyn Clock, deadline: Option<Ts>) {
    match deadline {
        Some(deadline) => clock.wait_until(deadline).await,
        None => std::future::pending().await,
    }
}

impl PendingWrite {
    async fn send(
        self,
        sink: Arc<dyn WorkerSink>,
        clock: Arc<dyn Clock>,
    ) -> Result<WriteCompletion, TransportError> {
        let expired = || TransportError::Failed("control write deadline exceeded".into());
        if clock.now() >= self.deadline {
            return Err(expired());
        }
        let is_ping = matches!(self.message.message, Some(proto::worker_message::Message::Ping(_)));
        lk_agents_abi::before(&*clock, Some(self.deadline), sink.send(self.message))
            .await
            .map_err(|_| expired())??;
        Ok(match self.accepted {
            Some(job_id) => WriteCompletion::Availability(job_id),
            None if is_ping => WriteCompletion::Ping,
            None => WriteCompletion::Message,
        })
    }
}

enum Completion {
    Decision { generation: u64, slot: Slot, decision: JobDecision },
    Launched { generation: u64, slot: Slot, result: Result<JobId, HostError> },
    Load(Option<Result<f32, crate::load::LoadError>>),
    CpuSample(load_driver::CpuSampleResult),
    Done,
    EndpointDone(u64),
}

/// A host callback may unwind without taking the worker and its other jobs with it. Keep
/// constructing the callback's future inside `call`, so synchronous failures are caught too.
async fn host_outcome<T>(
    operation: &'static str,
    call: impl Future<Output = T> + Send,
) -> Result<T, HostError> {
    std::panic::AssertUnwindSafe(call)
        .catch_unwind()
        .await
        .map_err(|_| HostError::new(format!("{operation} callback panicked")))
}

impl Worker {
    /// Starts nothing. Subscribe through the handle before awaiting [`Self::run`] to see
    /// registration.
    pub fn new(
        config: crate::WorkerConfig,
        connection: RuntimeConfig,
        transport: crate::TransportConfig,
        host: Arc<dyn WorkerHost>,
        blocking_io: lk_agents_net::blocking_io::BlockingIo,
        rt: Background,
    ) -> (Self, WorkerClient) {
        let transport = transport.build(rt.clone()).into();
        Self::with_components(transport, connection, config, host, rt, blocking_io)
    }

    /// Explicit runtime components, for custom drivers and tests.
    pub fn with_components(
        transport: Arc<dyn WorkerTransport>,
        runtime_cfg: RuntimeConfig,
        cfg: WorkerConfig,
        host: Arc<dyn WorkerHost>,
        rt: Background,
        blocking_io: lk_agents_net::blocking_io::BlockingIo,
    ) -> (Self, WorkerClient) {
        let Background { clock, spawner, .. } = rt;
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let control = Arc::new(signal::Shared::new(WorkerControl::default()));
        let (notifications, _) = tokio::sync::broadcast::channel(128);
        let snapshot = Arc::new(parking_lot::Mutex::new(Snapshot::default()));
        let handle = WorkerClient {
            tx,
            control: control.clone(),
            snapshot: snapshot.clone(),
            notifications: notifications.clone(),
            clock: clock.clone(),
        };
        (
            Self {
                transport,
                runtime_cfg,
                cfg,
                wall: WallAnchor::of(&*clock),
                host,
                blocking_io,
                load_health: crate::load::LoadHealth::starting(clock.now()),
                clock,
                spawner,
                default_load: DefaultLoadCalc::new(),
                load_available: true,
                load_requested: lk_agents_abi::Ts::ZERO,
                load_failure: None,
                starting_jobs: BTreeMap::new(),
                cpu: None,
                rx,
                control,
                snapshot,
                notifications,
                operations: FuturesUnordered::new(),
                load_pending: false,
                send_failure: None,
                decision_tasks: Arc::default(),
                endpoints: HashMap::new(),
                next_endpoint: 0,
                closing: false,
                generation: 0,
                rollover_requested: false,
                timers: TimerSet::new(),
                conn: ConnState::Disconnected,
                registered_once: false,
                worker_id: String::new(),
                worker_load: 0.0,
                pending: Vec::new(),
                dispositions: BTreeMap::new(),
                assignment_capacity: jobs::DEFAULT_ASSIGNMENT_CAPACITY,
                rolling_over: false,
                awaiting_load: VecDeque::new(),
                active: Vec::new(),
                next_slot: 0,
                draining: false,
                drain_reported: false,
                previous_status: None,
                job_token_requests: HashMap::new(),
            },
            handle,
        )
    }

    fn shared(&self) -> parking_lot::MutexGuard<'_, Snapshot> {
        self.snapshot.lock()
    }

    fn publish(&self) {
        let reserved_slots = self.reserved_slots();
        let mut s = self.shared();
        s.worker_id = self.worker_id.clone();
        s.draining = self.draining;
        s.active_jobs = self.active.len();
        s.reserved_slots = reserved_slots;
        s.worker_load = self.worker_load;
        s.registered = self.conn == ConnState::Registered;
    }

    fn notify(&self, event: WorkerNotification) {
        let _ = self.notifications.send(event);
    }

    pub async fn run(&mut self) -> Result<(), RuntimeError> {
        let mut connection =
            ConnectionIo::new(self.runtime_cfg.connection_timeout, self.clock.clone());
        // The 0.5 s load timer lives outside any connection.
        self.timers.set_deadline(
            WorkerTimer::LoadSample,
            self.clock.now().saturating_add(crate::UPDATE_LOAD_INTERVAL),
        );
        self.sample_load();
        self.publish();
        let result = self.run_loop(&mut connection).await;

        self.decision_tasks.close();
        self.pending.clear();
        self.operations.clear();
        let deadline = self.clock.now().saturating_add(self.runtime_cfg.connection_timeout);
        if !self.decision_tasks.wait_empty(&*self.clock, Some(deadline)).await {
            log::warn!(target: lk_agents_abi::LOG_TARGET, "job decisions did not settle before worker shutdown");
        }
        if let Err(error) =
            host_outcome("endpoint shutdown", async { self.host.stop_endpoints().await }).await
        {
            log::warn!(target: lk_agents_abi::LOG_TARGET, "endpoint shutdown failed; closing the worker transport: {error}");
        }
        self.endpoints.clear();
        self.load_pending = false;
        self.rx.close();
        connection.close().await;
        self.conn = ConnState::Disconnected;
        self.publish();
        let reason =
            result.as_ref().err().map(ToString::to_string).unwrap_or_else(|| "closed".into());
        self.shared().finished = Some(reason.clone());
        self.notify(WorkerNotification::Stopped(WorkerStopped { reason }));
        result
    }

    async fn run_loop(&mut self, connection: &mut ConnectionIo) -> Result<(), RuntimeError> {
        let mut commands_open = true;
        let mut retries: u32 = 0;
        let control = self.control.clone();
        let mut seen = WorkerControl::default();
        while !self.closing {
            self.tick(&mut connection.outbox);
            self.publish();
            self.maybe_rollover(connection);
            let current = control.with(|control| control.clone());
            if current != seen {
                seen = current.clone();
                self.handle_control(current, &mut connection.outbox);
                if self.closing {
                    break;
                }
            }
            let clock = self.clock.clone();
            let wake = self
                .timers
                .next_deadline()
                .unwrap_or_else(|| clock.now().saturating_add(IDLE_WAKE));
            let result = tokio::select! {
                request = control.wait_until(|control| (*control != seen).then(|| control.clone())) => {
                    seen = request.clone();
                    self.handle_control(request, &mut connection.outbox);
                    Ok(())
                }
                command = self.rx.recv(), if commands_open => {
                    match command {
                        Some(command) => self.handle_command(command, &mut connection.outbox),
                        None => commands_open = false,
                    }
                    Ok(())
                }
                completion = self.operations.next(), if !self.operations.is_empty() => {
                    if let Some(completion) = completion {
                        self.complete(completion, &mut connection.outbox);
                    }
                    Ok(())
                }
                completion = jobs::next(&mut self.pending) => {
                    self.complete(completion, &mut connection.outbox);
                    Ok(())
                }
                event = connection.next_event(self.transport.clone(), &self.runtime_cfg) => event.and_then(|event| {
                    // A registration ends the reconnect attempts.
                    if let ConnectionEvent::Registered(response) = &event {
                        self.check_token_mode_server(response)?;
                        retries = 0;
                    }
                    self.handle_connection_event(event, &mut connection.outbox);
                    Ok(())
                }),
                () = clock.wait_until(wake) => Ok(()),
            };
            self.settle();
            self.publish();
            // Commands may update local state while offline, but their writes must not be replayed.
            if !connection.is_connected() {
                connection.outbox.clear();
                self.send_failure = None;
            }
            let result =
                result.and_then(|()| self.send_failure.take().map_or(Ok(()), |e| Err(e.into())));
            if let Err(error) = result {
                self.reconnect(connection, error, &mut retries)?;
            }
            self.maybe_rollover(connection);
        }
        Ok(())
    }

    fn handle_control(&mut self, control: WorkerControl, outbox: &mut Outbox) {
        self.tick(outbox);
        if let Some(timeout) = control.drain {
            self.drain(timeout, outbox);
        }
        self.closing |= control.close;
    }

    fn handle_command(&mut self, command: WorkerCommand, outbox: &mut Outbox) {
        self.tick(outbox);
        match command {
            WorkerCommand::Close => self.closing = true,
            WorkerCommand::Drain { timeout } => self.drain(timeout, outbox),
            WorkerCommand::JobStatus { job_id, status } => {
                self.on_job_status(job_id, status, outbox)
            }
            WorkerCommand::CreateJobToken { job_id, claims, reply } => {
                self.request_job_token(job_id, claims, reply, outbox)
            }
            WorkerCommand::MigrateJobs(jobs) => {
                if self.conn == ConnState::Registered {
                    self.queue(jobs::migrate_jobs(&jobs), outbox);
                }
            }
        }
    }

    fn handle_connection_event(&mut self, event: ConnectionEvent, outbox: &mut Outbox) {
        self.tick(outbox);
        match event {
            ConnectionEvent::Connected => self.on_connecting(outbox),
            ConnectionEvent::Registered(response) => self.on_registered(response, outbox),
            ConnectionEvent::ServerMessage(message) => self.on_server(*message, outbox),
            ConnectionEvent::AvailabilitySent(job_id) => self.on_availability_sent(job_id),
            ConnectionEvent::EndpointRequest(stream) => self.serve_endpoint(stream),
            ConnectionEvent::Progress => {}
        }
    }

    /// Schedule one incoming stream; `http` reads its preamble and hands the bytes to the host. A stream
    /// aborted by a reconnect or shutdown is dropped, which quinn resets with code 0 (`HSR_ABORT`).
    fn serve_endpoint(&mut self, stream: EndpointStream) {
        let host = self.host.clone();
        let id = self.next_endpoint;
        self.next_endpoint = self.next_endpoint.wrapping_add(1);
        let (served, done) = signal::once::<()>();
        let clock = self.clock.clone();
        let task = self.spawner.spawn(async move {
            let _served = served;
            crate::http::accept(host, stream, clock).await;
        });
        self.endpoints.insert(id, task);
        self.operations.push(
            async move {
                done.until(Option::is_some).await;
                Completion::EndpointDone(id)
            }
            .boxed(),
        );
    }

    /// Reconnects after a lost connection: at once first, then on a linear ladder capped at ten
    /// seconds, until `max_retry` attempts are spent.
    fn reconnect(
        &mut self,
        connection: &mut ConnectionIo,
        error: RuntimeError,
        retries: &mut u32,
    ) -> Result<(), RuntimeError> {
        if matches!(error, RuntimeError::Configuration(_)) {
            return Err(error);
        }
        // Never registered and no QUIC answer at all: retrying cannot change the server's or the
        // network's answer, so startup fails with what to fix.
        if let RuntimeError::Connection(TransportError::TimedOut(after)) = error
            && !self.registered_once
            && self.transport.kind() == crate::transport::TransportKind::WebTransport
        {
            return Err(RuntimeError::WebTransportUnreachable {
                url: self.runtime_cfg.server_url.clone(),
                after,
            });
        }
        self.disconnect(connection);
        if *retries >= self.runtime_cfg.max_retry {
            return Err(RuntimeError::RetriesExhausted {
                retries: *retries,
                source: Box::new(error),
            });
        }
        let delay = retry_delay(*retries);
        log::warn!(target: lk_agents_abi::LOG_TARGET, url = self.runtime_cfg.server_url.as_str(), transport:? = self.transport.kind(), retry = *retries + 1, max_retry = self.runtime_cfg.max_retry, delay:? = delay; "worker connection failed: {error}");
        connection.retry_after(delay);
        *retries += 1;
        Ok(())
    }

    fn disconnect(&mut self, connection: &mut ConnectionIo) {
        self.rollover_requested = false;
        connection.disconnect();
        self.endpoints.clear();
        self.on_disconnected();
        self.settle();
        connection.outbox.clear();
        self.send_failure = None;
    }

    fn maybe_rollover(&mut self, connection: &mut ConnectionIo) {
        if self.rollover_requested && connection.can_rollover() {
            self.rollover_requested = false;
            self.disconnect(connection);
            connection.retry_after(Duration::ZERO);
        }
    }

    fn complete(&mut self, completion: Completion, outbox: &mut Outbox) {
        self.tick(outbox);
        match completion {
            Completion::Decision { generation, slot, decision } => {
                self.on_decision(generation, slot, decision, outbox)
            }
            Completion::Launched { generation, slot, result } => {
                self.on_launched(generation, slot, result, outbox)
            }
            Completion::Load(load) => self.complete_user_load(load, outbox),
            Completion::CpuSample(result) => self.complete_cpu_sample(result, outbox),
            Completion::Done => {}
            Completion::EndpointDone(id) => {
                self.endpoints.remove(&id);
            }
        }
    }
}

// The host owns job execution and process supervision.

/// As the server issues it (`Job.id`).
pub type JobId = String;

pub use lk_agents_abi::errors::{HostError, HostResult};

#[async_trait]
pub trait WorkerHost: Send + Sync {
    /// Whether this worker takes the job.
    async fn on_job_request(&self, req: JobRequest) -> JobDecision;

    /// Returns the LiveKit job ID.
    async fn launch_job(&self, info: RunningJobInfo) -> HostResult<JobId>;

    /// Unknown jobs are ignored by the state machine.
    async fn terminate_job(&self, id: JobId) -> HostResult;

    /// The user's `load_fnc`. `None` selects the CPU-based calculator; `0.0` would report the
    /// worker permanently idle.
    async fn user_load(&self) -> Option<HostResult<f32>>;

    /// One endpoint request, preamble already parsed, served to completion: the host writes the
    /// HTTP/1.1 response and finishes or resets the stream (and after a `101`, pipes bytes both
    /// ways until either side ends). Polled independently of control traffic; the default refuses,
    /// which is sound because nothing has looked at the request.
    async fn serve_endpoint(&self, request: IncomingRequest) {
        request.stream.refuse();
    }

    /// The worker is stopping while its connection still carries the endpoint streams: end the
    /// exchanges still running in a way their peer can read. Bounded by the host.
    async fn stop_endpoints(&self) {}
}

#[cfg(test)]
mod write_tests {
    use super::*;
    use lk_agents_net::clock::RealClock;

    #[tokio::test(start_paused = true)]
    async fn pending_statuses_coalesce_but_availability_replies_keep_their_order() {
        let mut queue = Outbox::new(Duration::from_secs(10), Arc::new(RealClock::new()));
        queue.push(jobs::update_worker(crate::WorkerStatus::Available, 0.1, 0), None).unwrap();
        queue.push(jobs::availability_reject("first"), None).unwrap();
        queue.push(jobs::availability_reject("second"), None).unwrap();
        queue.push(jobs::update_worker_draining(0), None).unwrap();
        use livekit_protocol::worker_message::Message as M;
        assert!(
            matches!(queue.pop_front().unwrap().message.message, Some(M::Availability(a)) if a.job_id == "first")
        );
        assert!(
            matches!(queue.pop_front().unwrap().message.message, Some(M::Availability(a)) if a.job_id == "second")
        );
        assert!(
            matches!(queue.pop_front().unwrap().message.message, Some(M::UpdateWorker(s)) if s.status == Some(livekit_protocol::WorkerStatus::WsFull as i32))
        );
        assert!(queue.pop_front().is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn queue_capacity_and_write_age_are_bounded() {
        let clock: Arc<dyn Clock> = Arc::new(RealClock::new());
        let mut queue = Outbox::new(Duration::from_secs(1), clock.clone());
        for i in 0..MAX_PENDING_WRITES {
            queue.push(jobs::availability_reject(&i.to_string()), None).unwrap();
        }
        assert!(queue.push(jobs::availability_reject("overflow"), None).is_err());
        tokio::time::advance(Duration::from_secs(2)).await;
        assert!(queue.pop_front().unwrap().deadline < clock.now());
        queue.clear();
        assert!(queue.pop_front().is_none(), "disconnected writes must not be replayed");
    }

    impl Worker {
        /// The snapshot as the worker would publish it now.
        pub(crate) fn snapshot_now(&self) -> Snapshot {
            self.publish();
            self.shared().clone()
        }

        pub(crate) fn set_assignment_capacity(&mut self, capacity: usize) {
            self.assignment_capacity = capacity;
        }

        pub(crate) async fn sample_load_now(&mut self) -> Result<(), RuntimeError> {
            let mut outbox = Outbox::new(self.runtime_cfg.connection_timeout, self.clock.clone());
            self.sample_load();
            while let Some(completed) = self.operations.next().await {
                self.complete(completed, &mut outbox);
            }
            Ok(())
        }
    }
}
