//! Worker runtime: drives the state, the control connection and host operations. Host futures
//! are cancelled when it stops; writes are ordered, bounded and per-connection, and a reconnect
//! regenerates state rather than replaying.

use crate::LOG;
use crate::Slot;
use crate::hw::cpu::CpuMonitor;
use crate::protocol::{self, JobStatus, RegisterWorkerResponse, ServerMessage, WorkerMessage};
use crate::transport::{
    Connected, EndpointStream, HttpStreamResetCode, StreamPreamble, TransportError,
    TransportEvent, WorkerSink, WorkerTransport,
};
use crate::{DefaultLoadCalc, WorkerState, WorkerStateEffect, WorkerStateInput};
use crate::{JobDecision, JobRequest, RunningJobInfo};
use async_trait::async_trait;
use futures_util::future::{AbortHandle, Abortable};
use futures_util::{FutureExt, StreamExt, future::BoxFuture, stream::FuturesUnordered};
use lk_agents_abi::{Clock, RealClock};
use std::collections::{HashMap, VecDeque};
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

/// `min(retry_count * 2, 10)` s before the increment: 0, 2, 4, ... The leading zero is deliberate:
/// a single drop reconnects without a visible gap.
pub fn retry_delay(retry_count: u32) -> Duration {
    Duration::from_secs((retry_count as u64 * 2).min(10))
}

/// Connection events and host-operation completions are [`WorkerStateInput`]s instead.
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
pub struct WorkerHandle {
    tx: tokio::sync::mpsc::Sender<WorkerCommand>,
    snapshot: Arc<std::sync::Mutex<Snapshot>>,
    notifications: tokio::sync::broadcast::Sender<WorkerNotification>,
}

impl WorkerHandle {
    /// Fails when the worker has stopped or its queue is full; the command is handed back.
    pub fn send(&self, cmd: WorkerCommand) -> Result<(), WorkerCommand> {
        self.tx.try_send(cmd).map_err(|e| e.into_inner())
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

pub struct Worker<T: WorkerTransport, P: WorkerHost, C: Clock> {
    transport: Arc<T>,
    cfg: RuntimeConfig,
    state: WorkerState,
    host: Arc<P>,
    clock: C,
    default_load: DefaultLoadCalc,
    cpu: CpuMonitor,
    rx: tokio::sync::mpsc::Receiver<WorkerCommand>,
    snapshot: Arc<std::sync::Mutex<Snapshot>>,
    notifications: tokio::sync::broadcast::Sender<WorkerNotification>,
    operations: FuturesUnordered<BoxFuture<'static, Completion>>,
    load_pending: bool,
    send_failure: Option<TransportError>,
    requests: HashMap<Slot, AbortHandle>,
    endpoints: HashMap<u64, AbortHandle>,
    next_endpoint: u64,
    closing: bool,
}

impl<T: WorkerTransport, P: WorkerHost, C: Clock> Drop for Worker<T, P, C> {
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
    deadline: tokio::time::Instant,
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
    fn push(&mut self, message: WorkerMessage) -> Result<(), TransportError> {
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

    fn is_connected(&self) -> bool {
        matches!(self.state, ConnectionState::Active(_))
    }

    async fn next_event<T: WorkerTransport + 'static>(
        &mut self,
        transport: Arc<T>,
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
    async fn next_event<T: WorkerTransport + 'static>(
        &mut self,
        outbox: &mut Outbox,
        transport: Arc<T>,
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

fn dial<T: WorkerTransport + 'static>(
    transport: Arc<T>,
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
        let accepted = match &self.message.message {
            Some(livekit_protocol::worker_message::Message::Availability(a)) if a.available => {
                Some(a.job_id.clone())
            }
            _ => None,
        };
        let expired = || TransportError::Io("control write deadline exceeded".into());
        if Instant::now() >= self.deadline {
            return Err(expired());
        }
        tokio::time::timeout_at(self.deadline, sink.send(self.message))
            .await
            .map_err(|_| expired())??;
        Ok(accepted)
    }
}

enum Completion {
    Input(WorkerStateInput),
    Load(Option<f32>),
    Done,
    EndpointDone(u64),
}

impl<P: WorkerHost + 'static> Worker<Box<dyn WorkerTransport>, P, RealClock> {
    /// Resolves omitted options from the environment; starts nothing. Subscribe through the handle
    /// before awaiting [`Self::run`] to see registration.
    pub fn new(options: crate::WorkerOptions, host: P) -> (Self, WorkerHandle) {
        let (config, connection, transport) = options.resolve();
        Self::with_components(
            transport.build(),
            connection,
            WorkerState::new(config),
            host,
            RealClock::new(),
        )
    }
}

impl<T: WorkerTransport + 'static, P: WorkerHost + 'static, C: Clock> Worker<T, P, C> {
    /// Explicit runtime components, for custom drivers and tests.
    pub fn with_components(
        transport: T,
        cfg: RuntimeConfig,
        state: WorkerState,
        host: P,
        clock: C,
    ) -> (Self, WorkerHandle) {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let (notifications, _) = tokio::sync::broadcast::channel(128);
        let snapshot = Arc::new(std::sync::Mutex::new(Snapshot::default()));
        let handle =
            WorkerHandle { tx, snapshot: snapshot.clone(), notifications: notifications.clone() };
        (
            Self {
                transport: Arc::new(transport),
                cfg,
                state,
                host: Arc::new(host),
                clock,
                default_load: DefaultLoadCalc::new(),
                cpu: CpuMonitor::new(),
                rx,
                snapshot,
                notifications,
                operations: FuturesUnordered::new(),
                load_pending: false,
                send_failure: None,
                requests: HashMap::new(),
                endpoints: HashMap::new(),
                next_endpoint: 0,
                closing: false,
            },
            handle,
        )
    }

    pub(crate) fn publish(&self) {
        let mut s = self.snapshot.lock().unwrap_or_else(|e| e.into_inner());
        s.worker_id = self.state.worker_id().to_owned();
        s.draining = self.state.draining();
        s.active_jobs = self.state.active_jobs().len();
        s.reserved_slots = self.state.reserved_slots();
        s.worker_load = self.state.worker_load();
    }

    pub fn state(&self) -> &WorkerState {
        &self.state
    }
    pub fn state_mut(&mut self) -> &mut WorkerState {
        &mut self.state
    }

    fn notify(&self, event: WorkerNotification) {
        let _ = self.notifications.send(event);
    }

    pub async fn run(&mut self) -> Result<(), RuntimeError> {
        let mut connection = ConnectionIo::new(self.cfg.connection_timeout);
        let effects = self.state.start(self.clock.now());
        self.execute(effects, &mut connection.outbox);
        let result = self.run_loop(&mut connection).await;

        self.operations.clear();
        self.requests.clear();
        self.endpoints.clear();
        self.load_pending = false;
        self.rx.close();
        connection.close().await;
        let reason =
            result.as_ref().err().map(ToString::to_string).unwrap_or_else(|| "closed".into());
        self.snapshot.lock().unwrap_or_else(|e| e.into_inner()).finished = Some(reason.clone());
        self.notify(WorkerNotification::Stopped(WorkerStopped { reason }));
        result
    }

    async fn run_loop(&mut self, connection: &mut ConnectionIo) -> Result<(), RuntimeError> {
        let mut commands_open = true;
        let mut retries = 0;
        while !self.closing {
            let timer = self
                .state
                .next_deadline()
                .map(|at| at.saturating_duration_since(self.clock.now()))
                .unwrap_or(Duration::from_secs(3600));
            let result = tokio::select! {
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
                event = connection.next_event(self.transport.clone(), &self.cfg) => {
                    event.map(|event| self.handle_connection_event(event, &mut connection.outbox))
                }
                _ = tokio::time::sleep(timer) => {
                    let effects = self.state.tick(self.clock.now());
                    self.execute(effects, &mut connection.outbox);
                    Ok(())
                }
            };
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
        }
        Ok(())
    }

    fn handle_command(&mut self, command: WorkerCommand, outbox: &mut Outbox) {
        match command {
            WorkerCommand::Close => self.closing = true,
            WorkerCommand::Drain { timeout } => {
                let effects = self.state.drain(self.clock.now(), timeout);
                self.execute(effects, outbox);
            }
            WorkerCommand::JobStatus { job_id, status } => {
                self.feed(WorkerStateInput::JobStatus { job_id, status }, outbox);
            }
        }
    }

    fn handle_connection_event(&mut self, event: ConnectionEvent, outbox: &mut Outbox) {
        match event {
            ConnectionEvent::Connected => self.feed(WorkerStateInput::Connecting, outbox),
            ConnectionEvent::Registered(response) => {
                self.feed(WorkerStateInput::Registered(response), outbox);
            }
            ConnectionEvent::ServerMessage(message) => {
                self.feed(WorkerStateInput::Server(message), outbox)
            }
            ConnectionEvent::AvailabilitySent(job_id) => {
                self.feed(WorkerStateInput::AvailabilitySent { job_id }, outbox)
            }
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
        let (abort, registration) = AbortHandle::new_pair();
        self.endpoints.insert(id, abort);
        self.operations.push(
            async move {
                let _ = Abortable::new(serve_endpoint(host, stream), registration).await;
                Completion::EndpointDone(id)
            }
            .boxed(),
        );
    }

    fn reconnect(
        &mut self,
        connection: &mut ConnectionIo,
        error: RuntimeError,
        retries: &mut u32,
    ) -> Result<(), RuntimeError> {
        if matches!(error, RuntimeError::Configuration(_)) {
            return Err(error);
        }
        connection.disconnect();
        for (_, request) in self.requests.drain() {
            request.abort();
        }
        for (_, endpoint) in self.endpoints.drain() {
            endpoint.abort();
        }
        self.feed(WorkerStateInput::Disconnected, &mut connection.outbox);
        connection.outbox.clear();
        self.send_failure = None;
        if *retries >= self.cfg.max_retry {
            log::error!(target: LOG, error = error.to_string().as_str(); "failed to connect to livekit after {retries} retries");
            return Err(RuntimeError::RetriesExhausted {
                retries: *retries,
                source: Box::new(error),
            });
        }
        let delay = retry_delay(*retries);
        *retries += 1;
        log::warn!(target: LOG, retry_count = *retries, max_retry = self.cfg.max_retry, error = error.to_string().as_str();
            "failed to connect to livekit, retrying in {}s", delay.as_secs_f64());
        connection.retry_after(delay);
        Ok(())
    }

    fn feed(&mut self, input: WorkerStateInput, outbox: &mut Outbox) {
        let effects = self.state.handle(self.clock.now(), input);
        self.execute(effects, outbox);
    }

    fn complete(&mut self, completion: Completion, outbox: &mut Outbox) {
        match completion {
            Completion::Input(input) => {
                if let WorkerStateInput::RequestDecision { slot, .. } = &input {
                    self.requests.remove(slot);
                }
                self.feed(input, outbox);
            }
            Completion::Load(load) => {
                self.load_pending = false;
                let load = load.unwrap_or_else(|| {
                    if let Some(sample) = self.cpu.cpu_percent() {
                        self.default_load.add_sample(sample);
                    }
                    self.default_load.get()
                });
                self.feed(WorkerStateInput::LoadSampled(load), outbox);
            }
            Completion::Done => {}
            Completion::EndpointDone(id) => {
                self.endpoints.remove(&id);
            }
        }
    }

    fn execute(&mut self, effects: Vec<WorkerStateEffect>, outbox: &mut Outbox) {
        for effect in effects {
            let host = self.host.clone();
            match effect {
                WorkerStateEffect::Send(message) => {
                    if let Err(error) = outbox.push(message) {
                        self.send_failure = Some(error);
                    }
                }
                WorkerStateEffect::AskUser { slot, request } => {
                    let (abort, registration) = AbortHandle::new_pair();
                    self.requests.insert(slot, abort);
                    self.operations.push(
                        async move {
                            match Abortable::new(host.on_job_request(request), registration).await {
                                Ok(decision) => Completion::Input(WorkerStateInput::RequestDecision {
                                    slot,
                                    decision,
                                }),
                                Err(_) => Completion::Done,
                            }
                        }
                        .boxed(),
                    );
                }
                WorkerStateEffect::LaunchJob { slot, info } => self.operations.push(
                    async move {
                        Completion::Input(WorkerStateInput::JobLaunched {
                            slot,
                            result: host.launch_job(*info).await,
                        })
                    }
                    .boxed(),
                ),
                WorkerStateEffect::TerminateJob(id) => self.operations.push(
                    async move {
                        if let Err(error) = host.terminate_job(id.clone()).await {
                            log::error!(target: LOG, job_id = id.as_str(), error = error.to_string().as_str(); "failed to terminate job");
                        }
                        Completion::Done
                    }
                    .boxed(),
                ),
                WorkerStateEffect::SampleLoad if !self.load_pending => {
                    self.load_pending = true;
                    self.operations
                        .push(async move { Completion::Load(host.user_load().await) }.boxed());
                }
                WorkerStateEffect::SampleLoad => {}
                WorkerStateEffect::Registered { worker_id, region, protocol, deployment, agent_name } => {
                    self.notify(WorkerNotification::Registered(WorkerRegistered {
                        worker_id,
                        region,
                        protocol,
                        deployment,
                        agent_name,
                    }));
                }
                WorkerStateEffect::SetTargetIdleProcesses(n) => {
                    self.snapshot.lock().unwrap_or_else(|e| e.into_inner()).target_idle_processes =
                        Some(n);
                }
                WorkerStateEffect::DrainFinished => {
                    self.snapshot.lock().unwrap_or_else(|e| e.into_inner()).drain_outcome =
                        Some("finished".into());
                    self.notify(WorkerNotification::DrainFinished(WorkerDrainFinished {}));
                }
                WorkerStateEffect::DrainTimedOut => {
                    self.snapshot.lock().unwrap_or_else(|e| e.into_inner()).drain_outcome =
                        Some("timed_out".into());
                    self.notify(WorkerNotification::DrainTimedOut(WorkerDrainTimedOut {}));
                }
            }
        }
        self.publish();
    }
}


/// `wire.MaxPreambleSize` is small, so a peer that opens a stream and sends nothing is the only
/// way to wait here; the node writes the preamble in the same call as the request head.
const PREAMBLE_TIMEOUT: Duration = Duration::from_secs(30);

async fn serve_endpoint<P: WorkerHost + ?Sized>(host: Arc<P>, mut stream: EndpointStream) {
    let preamble = match tokio::time::timeout(PREAMBLE_TIMEOUT, stream.read_preamble()).await {
        Ok(Ok(preamble)) => preamble,
        Ok(Err(error)) => {
            log::warn!(target: LOG, error = error.to_string().as_str(); "endpoint stream without a valid preamble");
            stream.reset(HttpStreamResetCode::HsrProtocol);
            return;
        }
        Err(_) => {
            log::warn!(target: LOG, "endpoint stream sent no preamble in time");
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
    /// The user's `request_fnc`; Rust auto-rejects if it returns without answering. No deadline:
    /// an implementation's own timeout must report [`JobDecision::Failed`], not hang.
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
        queue.push(protocol::update_worker(crate::WorkerStatus::Available, 0.1, 0)).unwrap();
        queue.push(protocol::availability_reject("first", false)).unwrap();
        queue.push(protocol::availability_reject("second", false)).unwrap();
        queue.push(protocol::update_worker_draining(0)).unwrap();
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
            queue.push(protocol::availability_reject(&i.to_string(), false)).unwrap();
        }
        assert!(queue.push(protocol::availability_reject("overflow", false)).is_err());
        tokio::time::advance(Duration::from_secs(2)).await;
        assert!(queue.pop_front().unwrap().deadline < tokio::time::Instant::now());
        queue.clear();
        assert!(queue.pop_front().is_none(), "disconnected writes must not be replayed");
    }

    impl<T: WorkerTransport + 'static, P: WorkerHost + 'static, C: Clock> Worker<T, P, C> {
        pub(crate) async fn sample_load(&mut self) -> Result<(), RuntimeError> {
            let mut outbox = Outbox::new(self.cfg.connection_timeout);
            self.execute(vec![WorkerStateEffect::SampleLoad], &mut outbox);
            while let Some(completed) = self.operations.next().await {
                self.complete(completed, &mut outbox);
            }
            Ok(())
        }
    }
}
