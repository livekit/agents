//! Worker runtime: drives the state, the control connection and host operations. Host futures
//! are cancelled when it stops; writes are ordered, bounded and per-connection, and a reconnect
//! regenerates state rather than replaying.

use crate::hw::cpu::CpuMonitor;
use crate::protocol::{self, JobStatus, RegisterWorkerResponse, ServerMessage, WorkerMessage};
use crate::transport::{
    Connected, EndpointStream, HttpStreamResetCode, StreamPreamble, TransportError, TransportEvent,
    WorkerSink, WorkerTransport,
};
use crate::{DefaultLoadCalc, WorkerConfig};
use crate::{JobDecision, JobRequest, RunningJobInfo};
use async_trait::async_trait;
use futures_util::{FutureExt, StreamExt, future::BoxFuture, stream::FuturesUnordered};
use lk_agents_abi::{Clock, TimerSet, WallAnchor};
use lk_agents_abi::{OwnedTask, SpawnerExt};
use lk_agents_net::clock::RealClock;
use lk_agents_net::task::TokioSpawner;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::Instant;

/// Default ping interval.
pub const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);

const MAX_PENDING_WRITES: usize = 64;

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
}

mod jobs;
use jobs::{ConnState, Pending, WorkerTimer};
pub use jobs::{Slot, retry_delay};

/// What a client asks of a running worker.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerCommand {
    Drain { timeout: Option<Duration> },
    Close,
    JobStatus { job_id: JobId, status: JobStatus },
}

/// Each variant carries a named `lk_agents_abi::worker::worker` record: the binding projects it into
/// the dict the host matches on.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerNotification {
    Registered(WorkerRegistered),
    DrainFinished(WorkerDrainFinished),
    DrainTimedOut(WorkerDrainTimedOut),
    Stopped(WorkerStopped),
}

pub use lk_agents_abi::worker::worker::{
    Snapshot, WorkerDrainFinished, WorkerDrainTimedOut, WorkerRegistered, WorkerStopped,
};

#[derive(Clone)]
pub struct WorkerClient {
    tx: tokio::sync::mpsc::Sender<WorkerCommand>,
    control: tokio::sync::watch::Sender<WorkerControl>,
    snapshot: Arc<std::sync::Mutex<Snapshot>>,
    notifications: tokio::sync::broadcast::Sender<WorkerNotification>,
}

impl WorkerClient {
    /// Fails when the worker has stopped or its queue is full; the command is handed back.
    pub fn send(&self, cmd: WorkerCommand) -> Result<(), WorkerCommand> {
        if self.tx.is_closed() || self.control.is_closed() {
            return Err(cmd);
        }
        match cmd {
            WorkerCommand::Close => {
                self.control.send_if_modified(|control| {
                    if control.close {
                        return false;
                    }
                    control.close = true;
                    true
                });
                Ok(())
            }
            WorkerCommand::Drain { timeout } => {
                self.control.send_if_modified(|control| {
                    if control.drain.is_some() {
                        return false;
                    }
                    control.drain = Some(timeout);
                    true
                });
                Ok(())
            }
            command => self.tx.try_send(command).map_err(|e| e.into_inner()),
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        self.snapshot.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Holds 128 notifications; a lagging receiver gets `RecvError::Lagged`.
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<WorkerNotification> {
        self.notifications.subscribe()
    }

    /// Waits for the state's drain result, including outstanding decisions and launches.
    pub async fn drain(&self, timeout: Option<Duration>) -> Result<(), String> {
        let mut notifications = self.subscribe();
        self.send(WorkerCommand::Drain { timeout })
            .map_err(|_| "worker stopped or command queue full".to_string())?;
        loop {
            let state = self.snapshot();
            if let Some(outcome) = state.drain_outcome {
                return if outcome == "finished" { Ok(()) } else { Err("drain timed out".into()) };
            }
            if let Some(reason) = state.finished {
                return Err(reason);
            }
            notifications.recv().await.map_err(|e| e.to_string())?;
        }
    }
}

#[derive(Clone, Default)]
struct WorkerControl {
    close: bool,
    drain: Option<Option<Duration>>,
}

pub struct RuntimeConfig {
    pub server_url: String,
    pub worker_token: Option<String>,
    /// Authorization is minted separately for each connection attempt.
    pub headers: Vec<(String, String)>,
    /// Both empty means "send `headers` alone".
    pub api_key: String,
    pub api_secret: String,
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
    default_load: DefaultLoadCalc,
    cpu: Option<CpuMonitor>,
    rx: tokio::sync::mpsc::Receiver<WorkerCommand>,
    control: tokio::sync::watch::Receiver<WorkerControl>,
    snapshot: Arc<std::sync::Mutex<Snapshot>>,
    notifications: tokio::sync::broadcast::Sender<WorkerNotification>,
    operations: FuturesUnordered<BoxFuture<'static, Completion>>,
    load_pending: bool,
    send_failure: Option<TransportError>,
    /// Host decisions and endpoint exchanges in flight, each its own task; removing one
    /// cancels it, and its completion still reaches `operations`.
    requests: HashMap<Slot, OwnedTask>,
    endpoints: HashMap<u64, OwnedTask>,
    next_endpoint: u64,
    closing: bool,
    /// Bumped on each connection: a decision or launch of an older one is stale.
    generation: u64,
    /// The connection is to be replaced once its writes settled.
    rollover_requested: bool,
    timers: TimerSet<WorkerTimer>,
    conn: ConnState,
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
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.operations.clear();
        let mut snapshot = self.snapshot.lock().unwrap_or_else(|e| e.into_inner());
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
    deadline: tokio::time::Instant,
}

/// Encode the core's ordered acceptance without making another admission decision.
fn availability_message(job_id: &str, arguments: crate::AcceptArguments) -> WorkerMessage {
    WorkerMessage {
        message: Some(livekit_protocol::worker_message::Message::Availability(
            protocol::AvailabilityResponse {
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
}
impl Outbox {
    fn new(timeout: Duration) -> Self {
        Self { pending: VecDeque::new(), timeout }
    }

    fn push(
        &mut self,
        message: WorkerMessage,
        accepted: Option<JobId>,
    ) -> Result<(), TransportError> {
        use livekit_protocol::worker_message::Message as M;
        use prost::Message;
        if message.encoded_len() > crate::transport::MAX_CONTROL_MESSAGE {
            return Err(TransportError::Protocol("control message exceeds 1 MiB".into()));
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
            return Err(TransportError::Io("control write queue is full".into()));
        }
        self.pending.push_back(PendingWrite {
            message,
            accepted,
            deadline: tokio::time::Instant::now() + self.timeout,
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
    ServerMessage(ServerMessage),
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
}

enum ConnectionState {
    Waiting(Instant),
    Dialing(BoxFuture<'static, Result<Connected, TransportError>>),
    Active(ActiveConnection),
}

struct ActiveConnection {
    transport: Connected,
    /// `None` after the server's registration response has been received.
    registration_deadline: Option<Instant>,
    writing: Option<BoxFuture<'static, Result<Option<String>, TransportError>>>,
}

impl ConnectionIo {
    fn new(timeout: Duration) -> Self {
        Self {
            state: ConnectionState::Waiting(Instant::now()),
            outbox: Outbox::new(timeout),
            cleanups: FuturesUnordered::new(),
            timeout,
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
            event = self.state.next_event(&mut self.outbox, transport, config) => event,
            _ = self.cleanups.next(), if !self.cleanups.is_empty() => Ok(ConnectionEvent::Progress),
        }
    }

    fn disconnect(&mut self) {
        let previous = std::mem::replace(&mut self.state, ConnectionState::Waiting(Instant::now()));
        self.outbox.clear();
        if let ConnectionState::Active(active) = previous {
            let timeout = self.timeout;
            // Cancel the in-flight write before attempting to acquire the sink for shutdown.
            drop(active.writing);
            self.cleanups.push(
                async move {
                    let _ = tokio::time::timeout(timeout, active.transport.sink.close()).await;
                }
                .boxed(),
            );
        }
    }

    fn retry_after(&mut self, delay: Duration) {
        debug_assert!(matches!(self.state, ConnectionState::Waiting(_)));
        self.state = ConnectionState::Waiting(Instant::now() + delay);
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
    ) -> Result<ConnectionEvent, RuntimeError> {
        match self {
            Self::Waiting(retry_at) => {
                tokio::time::sleep_until(*retry_at).await;
                *self = Self::Dialing(dial(transport, config)?);
                Ok(ConnectionEvent::Progress)
            }
            Self::Dialing(connecting) => {
                let connected = connecting.await?;
                *self = Self::Active(ActiveConnection {
                    transport: connected,
                    registration_deadline: Some(Instant::now() + config.connection_timeout),
                    writing: None,
                });
                Ok(ConnectionEvent::Connected)
            }
            Self::Active(connection) => connection.next_event(outbox).await,
        }
    }
}

fn dial(
    transport: Arc<dyn WorkerTransport>,
    config: &RuntimeConfig,
) -> Result<BoxFuture<'static, Result<Connected, TransportError>>, RuntimeError> {
    let url = config.server_url.clone();
    let token = config.worker_token.clone();
    let timeout = config.connection_timeout;
    let mut headers = config.headers.clone();
    if !config.api_key.is_empty() && !config.api_secret.is_empty() {
        let jwt = crate::auth::worker_token(&config.api_key, &config.api_secret)
            .map_err(|e| RuntimeError::Configuration(e.to_string()))?;
        headers.push(crate::auth::authorization_header(&jwt));
    }
    Ok(async move {
        tokio::time::timeout(timeout, transport.connect(&url, token.as_deref(), headers))
            .await
            .map_err(|_| TransportError::Io("connection attempt timed out".into()))?
    }
    .boxed())
}

impl ActiveConnection {
    async fn next_event(&mut self, outbox: &mut Outbox) -> Result<ConnectionEvent, RuntimeError> {
        if self.writing.is_none()
            && let Some(pending) = outbox.pop_front()
        {
            self.writing = Some(pending.send(self.transport.sink.clone()).boxed());
        }
        tokio::select! {
            sent = await_write(&mut self.writing) => {
                self.writing = None;
                Ok(match sent? {
                    Some(job_id) => ConnectionEvent::AvailabilitySent(job_id),
                    None => ConnectionEvent::Progress,
                })
            }
            received = self.transport.stream.recv() => self.receive(received?),
            _ = wait_for_deadline(self.registration_deadline) => {
                Err(TransportError::Io("registration timed out".into()).into())
            }
        }
    }

    fn receive(&mut self, event: TransportEvent) -> Result<ConnectionEvent, RuntimeError> {
        match event {
            TransportEvent::ServerMessage(message) if self.registration_deadline.is_some() => {
                match message.message {
                    Some(protocol::server_message::Message::Register(response)) => {
                        self.registration_deadline = None;
                        Ok(ConnectionEvent::Registered(response))
                    }
                    _ => Err(RuntimeError::ExpectedRegister),
                }
            }
            TransportEvent::ServerMessage(message) => Ok(ConnectionEvent::ServerMessage(message)),
            TransportEvent::EndpointRequest(stream) if self.registration_deadline.is_some() => {
                stream.refuse();
                Ok(ConnectionEvent::Progress)
            }
            TransportEvent::EndpointRequest(stream) => Ok(ConnectionEvent::EndpointRequest(stream)),
        }
    }
}

async fn await_write(
    writing: &mut Option<BoxFuture<'static, Result<Option<String>, TransportError>>>,
) -> Result<Option<String>, TransportError> {
    match writing {
        Some(writing) => writing.await,
        None => std::future::pending().await,
    }
}

async fn wait_for_deadline(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => std::future::pending().await,
    }
}

impl PendingWrite {
    async fn send(self, sink: Arc<dyn WorkerSink>) -> Result<Option<String>, TransportError> {
        let expired = || TransportError::Io("control write deadline exceeded".into());
        if Instant::now() >= self.deadline {
            return Err(expired());
        }
        tokio::time::timeout_at(self.deadline, sink.send(self.message))
            .await
            .map_err(|_| expired())??;
        Ok(self.accepted)
    }
}

enum Completion {
    Decision { generation: u64, slot: Slot, decision: JobDecision },
    Launched { generation: u64, slot: Slot, result: Result<JobId, HostError> },
    Load(Option<f32>),
    CpuSample(Result<(CpuMonitor, Option<f32>), tokio::task::JoinError>),
    Done,
    EndpointDone(u64),
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
    ) -> (Self, WorkerClient) {
        Self::with_components(
            transport.build().into(),
            connection,
            config,
            host,
            Arc::new(RealClock::new()),
            blocking_io,
        )
    }

    /// Explicit runtime components, for custom drivers and tests.
    pub fn with_components(
        transport: Arc<dyn WorkerTransport>,
        runtime_cfg: RuntimeConfig,
        cfg: WorkerConfig,
        host: Arc<dyn WorkerHost>,
        clock: Arc<dyn Clock>,
        blocking_io: lk_agents_net::blocking_io::BlockingIo,
    ) -> (Self, WorkerClient) {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let (control_tx, control) = tokio::sync::watch::channel(WorkerControl::default());
        let (notifications, _) = tokio::sync::broadcast::channel(128);
        let snapshot = Arc::new(std::sync::Mutex::new(Snapshot::default()));
        let handle = WorkerClient {
            tx,
            control: control_tx,
            snapshot: snapshot.clone(),
            notifications: notifications.clone(),
        };
        (
            Self {
                transport,
                runtime_cfg,
                cfg,
                wall: WallAnchor::of(&*clock),
                host,
                blocking_io,
                clock,
                default_load: DefaultLoadCalc::new(),
                cpu: None,
                rx,
                control,
                snapshot,
                notifications,
                operations: FuturesUnordered::new(),
                load_pending: false,
                send_failure: None,
                requests: HashMap::new(),
                endpoints: HashMap::new(),
                next_endpoint: 0,
                closing: false,
                generation: 0,
                rollover_requested: false,
                timers: TimerSet::new(),
                conn: ConnState::Disconnected,
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
            },
            handle,
        )
    }

    fn shared(&self) -> std::sync::MutexGuard<'_, Snapshot> {
        self.snapshot.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn publish(&self) {
        let reserved_slots = self.reserved_slots();
        let mut s = self.shared();
        s.worker_id = self.worker_id.clone();
        s.draining = self.draining;
        s.active_jobs = self.active.len();
        s.reserved_slots = reserved_slots;
        s.worker_load = self.worker_load;
    }

    /// The server has answered the registration on the current connection.
    pub fn registered(&self) -> bool {
        self.conn == ConnState::Registered
    }

    fn notify(&self, event: WorkerNotification) {
        let _ = self.notifications.send(event);
    }

    pub async fn run(&mut self) -> Result<(), RuntimeError> {
        let mut connection = ConnectionIo::new(self.runtime_cfg.connection_timeout);
        // The 0.5 s load timer lives outside any connection.
        self.timers.arm(
            WorkerTimer::LoadSample,
            self.clock.now().saturating_add(crate::UPDATE_LOAD_INTERVAL),
        );
        self.sample_load();
        self.publish();
        let result = self.run_loop(&mut connection).await;

        self.operations.clear();
        self.requests.clear();
        self.endpoints.clear();
        self.load_pending = false;
        self.rx.close();
        connection.close().await;
        let reason =
            result.as_ref().err().map(ToString::to_string).unwrap_or_else(|| "closed".into());
        self.shared().finished = Some(reason.clone());
        self.notify(WorkerNotification::Stopped(WorkerStopped { reason }));
        result
    }

    async fn run_loop(&mut self, connection: &mut ConnectionIo) -> Result<(), RuntimeError> {
        let mut commands_open = true;
        let mut retries: u32 = 0;
        let mut control_open = true;
        while !self.closing {
            self.tick(&mut connection.outbox);
            self.publish();
            self.maybe_rollover(connection);
            if self.control.has_changed().unwrap_or(false) {
                self.handle_control(&mut connection.outbox);
                if self.closing {
                    break;
                }
            }
            let timer = self
                .timers
                .next_deadline()
                .map(|at| at.saturating_duration_since(self.clock.now()))
                .unwrap_or(Duration::from_secs(3600));
            let result = tokio::select! {
                changed = self.control.changed(), if control_open => {
                    if changed.is_ok() {
                        self.handle_control(&mut connection.outbox);
                    } else {
                        control_open = false;
                    }
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
                event = connection.next_event(self.transport.clone(), &self.runtime_cfg) => {
                    event.map(|event| {
                        if matches!(event, ConnectionEvent::Registered(_)) {
                            retries = 0;
                        }
                        self.handle_connection_event(event, &mut connection.outbox)
                    })
                }
                _ = tokio::time::sleep(timer) => Ok(()),
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

    fn handle_control(&mut self, outbox: &mut Outbox) {
        self.tick(outbox);
        let control = self.control.borrow_and_update().clone();
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
        }
    }

    fn handle_connection_event(&mut self, event: ConnectionEvent, outbox: &mut Outbox) {
        self.tick(outbox);
        match event {
            ConnectionEvent::Connected => self.on_connecting(outbox),
            ConnectionEvent::Registered(response) => self.on_registered(response, outbox),
            ConnectionEvent::ServerMessage(message) => self.on_server(message, outbox),
            ConnectionEvent::AvailabilitySent(job_id) => self.on_availability_sent(job_id),
            ConnectionEvent::EndpointRequest(stream) => self.serve_endpoint(stream),
            ConnectionEvent::Progress => {}
        }
    }

    /// One exchange stream: the preamble is read here, then the host serves the bytes. A stream
    /// aborted by a reconnect or shutdown is dropped, which quinn resets with code 0 (`HSR_ABORT`).
    fn serve_endpoint(&mut self, stream: EndpointStream) {
        let host = self.host.clone();
        let id = self.next_endpoint;
        self.next_endpoint = self.next_endpoint.wrapping_add(1);
        let (served, done) = tokio::sync::oneshot::channel::<()>();
        let task = TokioSpawner::current().spawn(async move {
            let _served = served;
            serve_endpoint(host, stream).await;
        });
        self.endpoints.insert(id, task);
        self.operations.push(
            async move {
                let _ = done.await;
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
        self.disconnect(connection);
        if *retries >= self.runtime_cfg.max_retry {
            return Err(RuntimeError::RetriesExhausted {
                retries: *retries,
                source: Box::new(error),
            });
        }
        connection.retry_after(retry_delay(*retries));
        *retries += 1;
        Ok(())
    }

    fn disconnect(&mut self, connection: &mut ConnectionIo) {
        self.rollover_requested = false;
        connection.disconnect();
        self.requests.clear();
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

    /// Samples the load (the user's `load_fnc`, else the CPU), one sample at a time.
    fn sample_load(&mut self) {
        if self.load_pending {
            return;
        }
        self.load_pending = true;
        let host = self.host.clone();
        self.operations.push(async move { Completion::Load(host.user_load().await) }.boxed());
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
            Completion::Load(Some(load)) => {
                self.load_pending = false;
                self.on_load(load, outbox);
            }
            Completion::Load(None) => {
                let blocking_io = self.blocking_io.clone();
                let permit = match blocking_io.try_admit() {
                    Ok(permit) => permit,
                    Err(error) => {
                        self.load_pending = false;
                        crate::diagnostics::emit(
                            log::Level::Debug,
                            format!("worker CPU sample deferred error={error}"),
                        );
                        self.on_load(self.default_load.get(), outbox);
                        return;
                    }
                };
                let cpu = self.cpu.take();
                let sample = blocking_io.spawn_admitted(permit, move || {
                    let mut cpu = cpu.unwrap_or_default();
                    let sample = cpu.cpu_percent();
                    (cpu, sample)
                });
                self.operations.push(async move { Completion::CpuSample(sample.await) }.boxed());
            }
            Completion::CpuSample(result) => {
                self.load_pending = false;
                match result {
                    Ok((cpu, sample)) => {
                        self.cpu = Some(cpu);
                        if let Some(sample) = sample {
                            self.default_load.add_sample(sample);
                        }
                    }
                    Err(error) => crate::diagnostics::emit(
                        log::Level::Error,
                        format!("worker CPU sampler failed error={error}"),
                    ),
                }
                self.on_load(self.default_load.get(), outbox);
            }
            Completion::Done => {}
            Completion::EndpointDone(id) => {
                self.endpoints.remove(&id);
            }
        }
    }
}

/// `wire.MaxPreambleSize` is small, so a peer that opens a stream and sends nothing is the only
/// way to wait here; the node writes the preamble in the same call as the request head.
const PREAMBLE_TIMEOUT: Duration = Duration::from_secs(30);

async fn serve_endpoint<P: WorkerHost + ?Sized>(host: Arc<P>, mut stream: EndpointStream) {
    let preamble = match tokio::time::timeout(PREAMBLE_TIMEOUT, stream.read_preamble()).await {
        Ok(Ok(preamble)) => preamble,
        Ok(Err(error)) => {
            crate::diagnostics::emit(
                log::Level::Warn,
                format!("endpoint stream without a valid preamble error={error}"),
            );
            stream.reset(HttpStreamResetCode::HsrProtocol);
            return;
        }
        Err(_) => {
            crate::diagnostics::emit(
                log::Level::Warn,
                "endpoint stream sent no preamble in time".into(),
            );
            stream.reset(HttpStreamResetCode::HsrProtocol);
            return;
        }
    };
    host.serve_endpoint(EndpointRequest { preamble, stream }).await;
}

/// One request on the endpoint data plane, as the host receives it: the node's preamble and the
/// stream positioned at the first HTTP/1.1 byte. `preamble.thread_id` is empty when the node
/// resolved no thread; `preamble.thread_grant` is the verified token's agentThread claim,
/// `None` when it carried none; `kind` is `AEK_TEXT` for a route the manifest declared so.
#[derive(Debug)]
pub struct EndpointRequest {
    pub preamble: StreamPreamble,
    pub stream: EndpointStream,
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
    async fn user_load(&self) -> Option<f32>;

    /// One endpoint request, preamble already parsed, served to completion: the host writes the
    /// HTTP/1.1 response and finishes or resets the stream (and after a `101`, pipes bytes both
    /// ways until either side ends). Polled independently of control traffic; the default refuses,
    /// which is sound because nothing has looked at the request.
    async fn serve_endpoint(&self, request: EndpointRequest) {
        request.stream.refuse();
    }
}

#[cfg(test)]
mod write_tests {
    use super::*;
    use crate::protocol;

    #[tokio::test(start_paused = true)]
    async fn pending_statuses_coalesce_but_availability_replies_keep_their_order() {
        let mut queue = Outbox::new(Duration::from_secs(10));
        queue.push(protocol::update_worker(crate::WorkerStatus::Available, 0.1, 0), None).unwrap();
        queue.push(protocol::availability_reject("first"), None).unwrap();
        queue.push(protocol::availability_reject("second"), None).unwrap();
        queue.push(protocol::update_worker_draining(0), None).unwrap();
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
        let mut queue = Outbox::new(Duration::from_secs(1));
        for i in 0..MAX_PENDING_WRITES {
            queue.push(protocol::availability_reject(&i.to_string()), None).unwrap();
        }
        assert!(queue.push(protocol::availability_reject("overflow"), None).is_err());
        tokio::time::advance(Duration::from_secs(2)).await;
        assert!(queue.pop_front().unwrap().deadline < tokio::time::Instant::now());
        queue.clear();
        assert!(queue.pop_front().is_none(), "disconnected writes must not be replayed");
    }

    impl Worker {
        /// The snapshot as the worker would publish it now.
        pub(crate) fn snapshot_now(&self) -> Snapshot {
            self.publish();
            self.shared().clone()
        }

        pub(crate) async fn sample_load_now(&mut self) -> Result<(), RuntimeError> {
            let mut outbox = Outbox::new(self.runtime_cfg.connection_timeout);
            self.sample_load();
            while let Some(completed) = self.operations.next().await {
                self.complete(completed, &mut outbox);
            }
            Ok(())
        }
    }
}
