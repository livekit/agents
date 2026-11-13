pub mod authority;
pub mod host;
pub mod remote;

use std::collections::BTreeMap;
use std::future::Future;
use std::ops::ControlFlow;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::transport::HttpRequest;
use a2a::{AgentCard, JsonRpcId, Message};
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::storage::thread::TaskReason;
use lk_agents_abi::{ApiConnectOptions, Clock, Closed, StreamOutput, Ts};
use lk_agents_core::a2a::client::{
    Client, ClientError, HttpMethod, ReplyState, Request, TaskUpdate,
};
use lk_agents_core::a2a::rpc::TaskRef;
use tokio::sync::{Mutex as AsyncMutex, mpsc};
use tokio_util::sync::CancellationToken;

use crate::env::{ProcessEnv, ProcessError};
use crate::http::exchange::{Exchange, HttpEvent};
use crate::http::{BodyReader, GatewayHttp};
use crate::signal;
use crate::source::{Pulled, SharedStream};
use crate::sse::{SseDecoder, SseEvent};
use lk_agents_abi::{OwnedTask, SpawnerExt, TaskOutcome};
use parking_lot::Mutex;

const RESPONSE_LIMIT: usize = 8 * 1024 * 1024;
const UPDATE_QUEUE: usize = 16;
/// The media type of a streamed (SSE) reply.
const EVENT_STREAM: &str = "text/event-stream";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CleanupObservation {
    /// The remote may have accepted a request whose task identity was not received locally,
    /// and sending the same message again could not learn it either.
    UnacknowledgedRequest {
        request: u64,
    },
    CancelUncertain {
        task: TaskRef,
        error: String,
    },
}

/// What a call's delivery learned of its remote task.
#[derive(Clone, Default)]
struct Progress {
    acknowledgement: Option<Result<TaskRef, ClientError>>,
    identity: Option<TaskRef>,
    terminal: bool,
}

/// Resolves, or ends, once the call's task and its cleanup are gone.
type Settled = signal::Cell<Option<()>>;

struct Operation {
    _task: OwnedTask,
    stop: CancellationToken,
    settled: Settled,
}

struct Inner {
    core: Mutex<Client>,
    http: Arc<GatewayHttp>,
    spawner: Arc<dyn lk_agents_abi::Spawner>,
    clock: Arc<dyn Clock>,
    connect: AsyncMutex<()>,
    closing: CancellationToken,
    requests: AtomicU64,
    operations: Mutex<BTreeMap<u64, Operation>>,
    observations: Mutex<Vec<CleanupObservation>>,
    endpoint: url::Url,
    options: ApiConnectOptions,
}

/// A connection owns its outstanding exchanges; it holds no remote-tool conversation state.
pub struct AgentClient {
    inner: Arc<Inner>,
}

impl AgentClient {
    pub fn new(
        env: &ProcessEnv,
        endpoint: String,
        token: Option<String>,
        options: ApiConnectOptions,
    ) -> Result<Self, ClientError> {
        options.validate().map_err(|error| ClientError::InvalidInput(error.to_string()))?;
        let endpoint_url = checked_url(&endpoint)?;
        let core = Client::new(endpoint, token)?;
        let unavailable = |error: ProcessError| ClientError::Transport(error.into());
        Ok(Self {
            inner: Arc::new(Inner {
                core: Mutex::new(core),
                http: (env.http)().map_err(unavailable)?,
                spawner: (env.spawner)().map_err(unavailable)?,
                clock: env.clock.clone(),
                endpoint: endpoint_url,
                options,
                connect: AsyncMutex::new(()),
                closing: CancellationToken::new(),
                requests: AtomicU64::new(1),
                operations: Mutex::new(BTreeMap::new()),
                observations: Mutex::new(Vec::new()),
            }),
        })
    }

    /// Automatic and idempotent: concurrent first operations fetch exactly one card.
    pub async fn connect(&self) -> Result<AgentCard, ClientError> {
        let _gate = self.inner.connect.lock().await;
        self.inner.ensure_open()?;
        if let Some(card) = self.inner.core.lock().card() {
            return Ok(card.clone());
        }
        let request = self.inner.core.lock().card_request()?;
        let discovery = self.inner.read_response(&request);
        let body = tokio::select! {
            biased;
            () = self.inner.closing.cancelled() => return Err(ClientError::closed()),
            body = discovery => body?,
        };
        self.inner.ensure_open()?;
        // Validate routing before any bearer is sent to a peer-provided interface URL.
        let card: AgentCard =
            serde_json::from_slice(&body).map_err(ClientError::invalid_response)?;
        for interface in &card.supported_interfaces {
            if interface.protocol_binding == a2a::TRANSPORT_PROTOCOL_JSONRPC {
                let rpc = checked_url(&interface.url)?;
                if rpc.origin() != self.inner.endpoint.origin() {
                    return Err(ClientError::invalid_response(
                        "the card's JSONRPC interface must have the endpoint's origin",
                    ));
                }
            }
        }
        Ok(self.inner.core.lock().accept_card(&body)?.clone())
    }

    pub fn offers_extension(&self) -> bool {
        self.inner.core.lock().offers_extension()
    }

    /// An explicitly supplied, validated conversation is sent only to a capable peer and
    /// omitted when an explicit caller thread lets the endpoint load its permitted history.
    pub async fn send(
        &self,
        mut message: Message,
        thread_id: Option<&str>,
        task_id: Option<&str>,
        history: Option<&lk_agents_core::a2a::caller_history::CallerHistory>,
        cx: &lk_agents_telemetry::Cx,
    ) -> Result<Reply, ClientError> {
        self.inner.core.lock().validate_message(&message, thread_id, task_id)?;
        self.connect().await?;
        if thread_id.is_none()
            && message.context_id.is_none()
            && self.offers_extension()
            && let Some(history) = history
        {
            message
                .parts
                .push(history.to_part().map_err(|e| ClientError::InvalidInput(e.to_string()))?);
        }
        let id = self.inner.next_id();
        let request = self.inner.core.lock().send(
            message,
            thread_id,
            task_id,
            JsonRpcId::Number(id as i64),
        )?;
        self.start(id, request.traced(cx.span())).await
    }

    pub async fn subscribe(
        &self,
        task: TaskRef,
        offset: Option<&str>,
        cx: &lk_agents_telemetry::Cx,
    ) -> Result<Reply, ClientError> {
        self.connect().await?;
        let id = self.inner.next_id();
        let request =
            self.inner.core.lock().subscribe(&task, offset, JsonRpcId::Number(id as i64))?;
        self.start(id, request.traced(cx.span())).await
    }

    async fn start(&self, id: u64, request: Request) -> Result<Reply, ClientError> {
        let (task, updates) = self.open(id, request).await?;
        Ok(Reply {
            task_id: task.id,
            thread_id: task.context_id,
            source: SharedStream::new(Box::new(updates), self.inner.clock.clone()),
            clock: self.inner.clock.clone(),
        })
    }

    /// Runs the request as an owned call; its task once the peer acknowledged it.
    async fn open(
        &self,
        id: u64,
        request: Request,
    ) -> Result<(TaskRef, ReplyUpdates), ClientError> {
        self.inner.ensure_open()?;
        let (updates, rx) = mpsc::channel(UPDATE_QUEUE);
        // Each producer lives with its writer: the delivery in the call's task, admission here,
        // consumption in the reply, settlement in the task's settle hook.
        let (progress, acknowledged) =
            signal::signal(Progress { identity: request.task.clone(), ..Progress::default() });
        let (admit, admitted) = signal::once::<()>();
        let (consume, consumed) = signal::signal(false);
        let (settle, settled) = signal::once::<()>();
        let stop = CancellationToken::new();
        let operation = OperationGuard { stop: stop.clone(), settled: settled.clone() };
        let message = OutboundMessage::of(&request);
        let call = RemoteCall {
            delivery: Delivery { inner: self.inner.clone(), updates, progress },
            task: RemoteTask {
                inner: self.inner.clone(),
                id,
                task_response: request.task_response,
                message: message.clone(),
                progress: acknowledged.clone(),
            },
            admitted: admitted.clone(),
            consumed,
            stop: stop.clone(),
        };
        let inner = self.inner.clone();
        let remote = acknowledged.clone();
        let task = self.inner.spawner.spawn_settled(call.run(request), move |outcome| async move {
            let admitted = admitted.get().is_some();
            inner.cancel_unfinished_call(id, message.as_ref(), outcome, admitted, &remote).await;
            inner.forget_call(id);
            // The client is released before the call is announced settled.
            drop(inner);
            settle.resolve(());
        });
        {
            let mut operations = self.inner.operations.lock();
            self.inner.ensure_open()?;
            operations.insert(id, Operation { _task: task, stop, settled });
        }
        let updates = ReplyUpdates { rx, consume, operation, clock: self.inner.clock.clone() };
        admit.resolve(());
        let task = acknowledged
            .wait_map(|progress| progress.acknowledgement.clone())
            .await
            .map_err(|_| ClientError::incomplete())??;
        Ok((task, updates))
    }

    /// Always sends contextId and the full-URI reason, independent of card capability hints.
    pub async fn cancel(
        &self,
        task: TaskRef,
        reason: Option<&str>,
    ) -> Result<TaskUpdate, ClientError> {
        self.connect().await?;
        let id = self.inner.next_id();
        let request = self.inner.core.lock().cancel(&task, reason, JsonRpcId::Number(id as i64))?;
        let (_, mut updates) = self.open(id, request).await?;
        updates.recv().await?.ok_or_else(ClientError::incomplete)
    }

    /// Stops admission and settles owned exchanges. Missed cleanup remains observable and owned.
    pub async fn aclose(&self) {
        self.inner.closing.cancel();
        let waiting = self.inner.cancel_all();
        for settled in waiting {
            let _ = settled.wait().await;
        }
        let operations = self.inner.operations.lock();
        if operations.is_empty() {
            self.inner.core.lock().close();
        }
    }

    pub fn cleanup_observations(&self) -> Vec<CleanupObservation> {
        self.inner.observations.lock().clone()
    }
}

impl Drop for AgentClient {
    fn drop(&mut self) {
        self.inner.closing.cancel();
        self.inner.cancel_all();
    }
}

impl Inner {
    fn ensure_open(&self) -> Result<(), ClientError> {
        if self.closing.is_cancelled() { Err(ClientError::closed()) } else { Ok(()) }
    }

    fn next_id(&self) -> u64 {
        self.requests.fetch_add(1, Ordering::Relaxed)
    }

    fn observe(&self, observation: CleanupObservation) {
        let mut observations = self.observations.lock();
        let first = !observations.contains(&observation);
        if first {
            observations.push(observation.clone());
        }
        drop(observations);
        if first {
            log::error!(target: lk_agents_abi::LOG_TARGET, "A2A cleanup incomplete: {observation:?}");
        }
    }

    fn cancel_all(&self) -> Vec<Settled> {
        self.operations
            .lock()
            .values_mut()
            .map(|op| {
                op.stop.cancel();
                op.settled.clone()
            })
            .collect()
    }

    fn exchange(&self, request: &Request) -> Exchange {
        Exchange::new(
            self.http.clone(),
            match request.method {
                HttpMethod::Get => reqwest::Method::GET,
                HttpMethod::Post => reqwest::Method::POST,
            },
            HttpRequest {
                url: request.url.clone(),
                headers: request.headers.clone().into_iter().collect(),
                body: request.body.clone(),
            },
            &self.options,
            self.clock.clone(),
        )
    }

    async fn read_response(&self, request: &Request) -> Result<Vec<u8>, ClientError> {
        let mut exchange = self.exchange(request);
        let mut body = BodyReader::new(RESPONSE_LIMIT);
        loop {
            match exchange.next().await.map_err(ClientError::Transport)? {
                HttpEvent::Started => body.clear(),
                HttpEvent::Headers(_) => {}
                HttpEvent::Bytes(bytes) => {
                    body.append(&bytes).map_err(ClientError::invalid_response)?
                }
                HttpEvent::End => return Ok(body.finish()),
            }
        }
    }

    async fn cancel_remote(
        &self,
        task: &TaskRef,
        reason: Option<&str>,
    ) -> Result<TaskUpdate, ClientError> {
        let id = JsonRpcId::Number(self.next_id() as i64);
        let request = self.core.lock().cancel(task, reason, id.clone())?;
        let body = self.read_response(&request).await?;
        ReplyState::new(Some(task.clone())).decode_task(&body, &id)
    }

    async fn cancel_owned(&self, task: &TaskRef) {
        if let Err(error) =
            self.cancel_remote(task, Some(TaskReason::CallerCanceled.as_str())).await
        {
            self.observe(CleanupObservation::CancelUncertain {
                task: task.clone(),
                error: error.to_string(),
            });
        }
    }

    /// A call whose task did not complete still owes its nonterminal remote task a cancel.
    async fn cancel_unfinished_call(
        &self,
        id: u64,
        message: Option<&OutboundMessage>,
        outcome: TaskOutcome,
        admitted: bool,
        remote: &signal::Cell<Progress>,
    ) {
        let state = remote.get();
        if outcome == TaskOutcome::Completed || state.terminal || !admitted {
            return;
        }
        let task = state.identity;
        match task {
            Some(task) => self.cancel_owned(&task).await,
            None => self.cancel_unacknowledged(id, message).await,
        }
    }

    /// A send whose acknowledgment never came is resolved by sending the same message again:
    /// the peer dedupes its messageId and answers with the task the first send created, which is
    /// then cancelled. Only a send that cannot be resolved so is observed.
    async fn cancel_unacknowledged(&self, request: u64, message: Option<&OutboundMessage>) {
        let identified = match message {
            Some(message) => self.identify(message).await,
            None => Err(ClientError::incomplete()),
        };
        match identified {
            Ok(Some(task)) => self.cancel_owned(&task).await,
            Ok(None) => {}
            Err(_) => self.observe(CleanupObservation::UnacknowledgedRequest { request }),
        }
    }

    /// The task `message` created, from the first update of its resend; `None` when the peer
    /// answered with a message or the task already ended, leaving nothing owed.
    async fn identify(&self, message: &OutboundMessage) -> Result<Option<TaskRef>, ClientError> {
        let mut first = None;
        let read = exchange(self, &message.0, &mut Updates::First(&mut first)).await;
        let Some(update) = first else {
            return read.and(Err(ClientError::incomplete()));
        };
        let task = TaskRef { id: update.task_id, context_id: update.context_id };
        Ok((!update.state.is_terminal()).then_some(task))
    }

    fn forget_call(&self, id: u64) {
        let mut operations = self.operations.lock();
        operations.remove(&id);
        // One scope: no exchange settles between the check and the close.
        if operations.is_empty() && self.closing.is_cancelled() {
            self.core.lock().close();
        }
    }
}

/// One request's exchange, run as an owned task. It delivers the reply's updates and owes the
/// remote task cleanup when the caller leaves or the exchange fails.
struct RemoteCall {
    delivery: Delivery,
    task: RemoteTask,
    /// The caller registered the call and holds its reply.
    admitted: signal::Cell<Option<()>>,
    /// The caller read the settling update.
    consumed: signal::Cell<bool>,
    stop: CancellationToken,
}

impl RemoteCall {
    async fn run(mut self, request: Request) {
        tokio::select! {
            biased;
            () = self.stop.cancelled() => return,
            admitted = self.admitted.wait() => {
                if admitted.is_err() {
                    return;
                }
            }
        }
        let result = {
            let inner = self.delivery.inner.clone();
            let mut updates = Updates::Call(&mut self.delivery);
            let exchange = exchange(&inner, &request, &mut updates);
            tokio::pin!(exchange);
            tokio::select! {
                biased;
                () = self.stop.cancelled() => {
                    self.task.cancel_abandoned(exchange.as_mut()).await;
                    None
                },
                result = &mut exchange => Some(result),
            }
        };
        match result {
            Some(Ok(())) => self.cancel_unread_question().await,
            Some(Err(error)) => self.report_failure(error).await,
            None => self.delivery.fail_acknowledgement(ClientError::closed()),
        }
    }

    /// Until the recipient consumes a buffered question, abandoning its reply still owes
    /// cancellation of the nonterminal remote task.
    async fn cancel_unread_question(&mut self) {
        if self.task.progress.with(|p| p.terminal) || self.consumed.get() {
            return;
        }
        tokio::select! {
            biased;
            _ = self.consumed.wait_for(|consumed| *consumed) => {},
            () = self.stop.cancelled() => {},
        }
        if self.consumed.get() {
            return;
        }
        let task = self.task.progress.with(|p| p.identity.clone());
        if let Some(task) = task {
            self.task.inner.cancel_owned(&task).await;
        }
    }

    async fn report_failure(&mut self, error: ClientError) {
        self.delivery.fail_acknowledgement(error.clone());
        self.task.cancel_after_failure(&error).await;
        tokio::select! {
            biased;
            () = self.stop.cancelled() => {},
            _ = self.delivery.updates.send(Err(error)) => {},
        }
    }
}

/// A message send as first prepared, byte for byte, messageId included: every resend is this
/// one, so the peer, which dedupes on messageId, applies it once and answers a resend with the
/// task the first send created.
#[derive(Clone)]
struct OutboundMessage(Request);

impl OutboundMessage {
    /// A send that may create a task this client does not know yet; `None` for a request
    /// that names its task (an answer, a subscribe, a cancel).
    fn of(request: &Request) -> Option<Self> {
        (request.task.is_none() && !request.task_response).then(|| Self(request.clone()))
    }
}

/// The remote task a call created, as far as it is known here, and the cleanup it is owed.
struct RemoteTask {
    inner: Arc<Inner>,
    id: u64,
    task_response: bool,
    /// How an unacknowledged send learns its task.
    message: Option<OutboundMessage>,
    progress: signal::Cell<Progress>,
}

impl RemoteTask {
    async fn cancel_abandoned(
        &self,
        exchange: Pin<&mut impl Future<Output = Result<(), ClientError>>>,
    ) {
        if self.task_response {
            if let Err(error) = exchange.await
                && let Some(task) = self.progress.with(|p| p.identity.clone())
            {
                self.inner.observe(CleanupObservation::CancelUncertain {
                    task,
                    error: error.to_string(),
                });
            }
            return;
        }
        match self.await_identity(exchange).await {
            Some(task) if !self.progress.with(|p| p.terminal) => {
                self.inner.cancel_owned(&task).await;
            }
            Some(_) => {}
            None => self.inner.cancel_unacknowledged(self.id, self.message.as_ref()).await,
        }
    }

    /// Keeps the original send alive until its ack, which names the task without a resend.
    async fn await_identity(
        &self,
        exchange: Pin<&mut impl Future<Output = Result<(), ClientError>>>,
    ) -> Option<TaskRef> {
        tokio::select! {
            biased;
            task = self.progress.wait_map(|p| p.identity.clone()) => task.ok(),
            _ = exchange => self.progress.with(|p| p.identity.clone()),
        }
    }

    /// A failed exchange cancels its nonterminal remote task, or records why it cannot.
    async fn cancel_after_failure(&self, error: &ClientError) {
        if self.progress.with(|p| p.terminal) {
            return;
        }
        let task = self.progress.with(|p| p.identity.clone());
        let Some(task) = task else {
            // a refused RPC created nothing
            if !matches!(error, ClientError::Rpc { .. }) {
                self.inner.cancel_unacknowledged(self.id, self.message.as_ref()).await;
            }
            return;
        };
        if self.task_response {
            self.inner
                .observe(CleanupObservation::CancelUncertain { task, error: error.to_string() });
        } else {
            self.inner.cancel_owned(&task).await;
        }
    }
}

/// Hands a call's decoded updates to its caller; the first one acknowledges the call.
struct Delivery {
    inner: Arc<Inner>,
    updates: mpsc::Sender<Result<TaskUpdate, ClientError>>,
    progress: signal::Producer<Progress>,
}

impl Delivery {
    fn fail_acknowledgement(&self, error: ClientError) {
        self.progress.modify(|p| {
            p.acknowledgement.get_or_insert(Err(error));
        });
    }

    async fn deliver(&mut self, update: TaskUpdate) -> Result<(), ClientError> {
        let task = TaskRef { id: update.task_id.clone(), context_id: update.context_id.clone() };
        let first = self.progress.with(|p| p.acknowledgement.is_none());
        self.progress.modify(|p| {
            if !task.id.is_empty() {
                p.identity = Some(task.clone());
            }
            p.terminal = update.state.is_terminal();
        });
        if first {
            self.inner.core.lock().acknowledge(&update)?;
            self.progress.modify(|p| p.acknowledgement = Some(Ok(task)));
        }
        self.updates.send(Ok(update)).await.map_err(|_| ClientError::closed())
    }
}

/// IDs are available when `send()` returns, while updates keep their one-consumer source.
pub struct Reply {
    pub task_id: String,
    pub thread_id: String,
    source: SharedStream<TaskUpdate>,
    clock: Arc<dyn Clock>,
}

impl Reply {
    /// `None` while another pull of this reply is in flight.
    pub fn try_pull(
        &self,
    ) -> Option<impl Future<Output = Result<Pulled<TaskUpdate>, ProviderError>> + '_> {
        self.source.try_pull()
    }
    pub async fn aclose(&self, deadline: Ts) -> Closed {
        self.source.close(deadline).await
    }
}

struct OperationGuard {
    stop: CancellationToken,
    settled: Settled,
}

impl OperationGuard {
    fn cancel(&self) {
        self.stop.cancel();
    }
}

impl Drop for OperationGuard {
    fn drop(&mut self) {
        self.cancel();
    }
}

struct ReplyUpdates {
    rx: mpsc::Receiver<Result<TaskUpdate, ClientError>>,
    operation: OperationGuard,
    clock: Arc<dyn Clock>,
    consume: signal::Producer<bool>,
}

impl ReplyUpdates {
    async fn recv(&mut self) -> Result<Option<TaskUpdate>, ClientError> {
        let update = self.rx.recv().await.transpose()?;
        if update.as_ref().is_some_and(TaskUpdate::is_settled) {
            self.consume.set(true);
        }
        Ok(update)
    }
}

impl StreamOutput<TaskUpdate> for ReplyUpdates {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<TaskUpdate>, ProviderError>> {
        Box::pin(async move {
            self.recv().await.map_err(|error| match error {
                ClientError::Transport(error) => ProviderError::Local(error),
                error => ApiError::new(error.to_string()).with_retryable(false).into(),
            })
        })
    }
    fn cancel_safe(&self) -> bool {
        true
    }
    fn close(self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move {
            self.operation.cancel();
            let clock = self.clock.clone();
            Closed::bounded(&*clock, deadline, async move {
                let _ = self.operation.settled.wait().await;
            })
            .await
        })
    }
}

/// Where a reply's updates go.
enum Updates<'a> {
    /// Every one to the call's caller.
    Call(&'a mut Delivery),
    /// Only the first, which ends the read: how a resend learns the task its send created.
    First(&'a mut Option<TaskUpdate>),
}

impl Updates<'_> {
    /// `Break` once no more updates are wanted.
    async fn deliver(
        &mut self,
        exchange: &mut Exchange,
        update: TaskUpdate,
    ) -> Result<ControlFlow<()>, ClientError> {
        match self {
            Updates::Call(delivery) => {
                exchange.delivered();
                exchange
                    .wait_for_consumer(delivery.deliver(update))
                    .await
                    .map_err(ClientError::Transport)??;
                Ok(ControlFlow::Continue(()))
            }
            Updates::First(first) => {
                **first = Some(update);
                Ok(ControlFlow::Break(()))
            }
        }
    }
}

async fn exchange(
    inner: &Inner,
    request: &Request,
    updates: &mut Updates<'_>,
) -> Result<(), ClientError> {
    let mut reader = ReplyReader::new(request)?;
    let mut exchange = inner.exchange(request);
    loop {
        match exchange.next().await.map_err(ClientError::Transport)? {
            HttpEvent::Started => reader.restart(),
            HttpEvent::Headers(headers) => reader.framing = ReplyFraming::of(&headers),
            HttpEvent::Bytes(bytes) if reader.framing == ReplyFraming::EventStream => {
                if reader.read_events(&bytes, &mut exchange, updates).await?.is_break() {
                    return reader.state.finish();
                }
            }
            HttpEvent::Bytes(bytes) => {
                reader.body.append(&bytes).map_err(ClientError::invalid_response)?
            }
            HttpEvent::End => return reader.finish(&mut exchange, updates).await,
        }
    }
}

/// How a reply body is framed, read from its content type.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ReplyFraming {
    Json,
    EventStream,
}

impl ReplyFraming {
    fn of(headers: &reqwest::header::HeaderMap) -> Self {
        let media_type = headers
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split(';').next());
        if media_type.is_some_and(|value| value.trim() == EVENT_STREAM) {
            Self::EventStream
        } else {
            Self::Json
        }
    }
}

/// Decodes one call's reply; each retried attempt starts it over.
struct ReplyReader<'a> {
    request: &'a Request,
    id: JsonRpcId,
    state: ReplyState,
    framing: ReplyFraming,
    decoder: SseDecoder,
    events: Vec<SseEvent>,
    body: BodyReader,
}

impl<'a> ReplyReader<'a> {
    fn new(request: &'a Request) -> Result<Self, ClientError> {
        let id = request
            .id
            .clone()
            .ok_or_else(|| ClientError::InvalidInput("an RPC needs an id".into()))?;
        Ok(Self {
            request,
            id,
            state: ReplyState::for_request(request),
            framing: ReplyFraming::Json,
            decoder: SseDecoder::new(),
            events: Vec::new(),
            body: BodyReader::new(RESPONSE_LIMIT),
        })
    }

    fn restart(&mut self) {
        self.state = ReplyState::for_request(self.request);
        self.decoder = SseDecoder::new();
        self.body.clear();
        self.events.clear();
    }

    /// Delivers each event's update in order; breaks as soon as the reply has settled or no more
    /// updates are wanted.
    async fn read_events(
        &mut self,
        bytes: &[u8],
        exchange: &mut Exchange,
        updates: &mut Updates<'_>,
    ) -> Result<ControlFlow<()>, ClientError> {
        self.decoder.feed(bytes, &mut self.events).map_err(ClientError::invalid_response)?;
        for event in std::mem::take(&mut self.events) {
            if let Some(update) = self.state.decode(event.data.as_bytes(), &self.id)?
                && updates.deliver(exchange, update).await?.is_break()
            {
                return Ok(ControlFlow::Break(()));
            }
            if self.state.settled() {
                return Ok(ControlFlow::Break(()));
            }
        }
        Ok(ControlFlow::Continue(()))
    }

    /// An event stream must end cleanly; a JSON body carries the reply's one update.
    async fn finish(
        mut self,
        exchange: &mut Exchange,
        updates: &mut Updates<'_>,
    ) -> Result<(), ClientError> {
        if self.framing == ReplyFraming::EventStream {
            self.decoder.finish().map_err(ClientError::invalid_response)?;
            return self.state.finish();
        }
        let update = if self.request.task_response {
            Some(self.state.decode_task(self.body.as_bytes(), &self.id)?)
        } else {
            self.state.decode(self.body.as_bytes(), &self.id)?
        };
        if let Some(update) = update {
            // the body's one update: nothing follows it to stop
            let _ = updates.deliver(exchange, update).await?;
        }
        self.state.finish()
    }
}

fn checked_url(url: &str) -> Result<url::Url, ClientError> {
    let url = url::Url::parse(url).map_err(|e| ClientError::InvalidInput(e.to_string()))?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        return Err(ClientError::InvalidInput(
            "an endpoint must be an HTTP(S) URL without credentials or a fragment".into(),
        ));
    }
    Ok(url)
}

#[cfg(test)]
#[path = "a2a_client_tests.rs"]
mod tests;
