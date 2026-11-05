//! Native A2A client execution. Protocol decisions stay in `lk-agents-a2a::client`; discovery,
//! HTTP/SSE and owned cancellation run here through the shared HTTP pool and task/source owners.

pub mod authority;
pub mod host;
pub mod remote;

use std::collections::BTreeMap;
use std::future::Future;
use std::ops::ControlFlow;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use crate::transport::HttpRequest;
use a2a::{AgentCard, JsonRpcId, Message};
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::{ApiConnectOptions, Clock, Closed, HOST_TEARDOWN_TIMEOUT, StreamOutput, Ts};
use lk_agents_core::a2a::client::{
    Client, ClientError, HttpMethod, ReplyState, Request, TaskUpdate,
};
use lk_agents_core::a2a::rpc::TaskRef;
use tokio::sync::{Mutex as AsyncMutex, Notify, mpsc, oneshot, watch};

use crate::env::ProcessEnv;
use crate::http::GatewayHttp;
use crate::http::exchange::{Exchange, HttpEvent};
use crate::source::{PullError, Pulled, SharedStream};
use crate::sse::{SseDecoder, SseEvent};
use crate::task::TokioSpawner;
use lk_agents_abi::{OwnedTask, SpawnerExt, TaskOutcome};
use parking_lot::Mutex;

const RESPONSE_LIMIT: usize = 8 * 1024 * 1024;
const UPDATE_QUEUE: usize = 16;
/// The reason sent when the caller's exit or a failed exchange cancels the remote task.
const CALLER_CANCELED: &str = "caller-canceled";
/// The media type of a streamed (SSE) reply.
const EVENT_STREAM: &str = "text/event-stream";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CleanupObservation {
    /// The remote may have accepted a request whose task identity was not received locally.
    UnacknowledgedRequest {
        request: u64,
    },
    CancelUncertain {
        task: TaskRef,
        error: String,
    },
    CancellationIncomplete {
        task: TaskRef,
    },
    Incomplete {
        requests: Vec<u64>,
    },
}

struct Operation {
    _task: OwnedTask,
    cancel: watch::Sender<bool>,
    settled: watch::Receiver<bool>,
}

struct Inner {
    core: Mutex<Client>,
    http: Arc<GatewayHttp>,
    runtime: tokio::runtime::Handle,
    clock: Arc<dyn Clock>,
    connect: AsyncMutex<()>,
    closed: AtomicBool,
    closing: Notify,
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
        // A process that cannot lend a runtime will not gain one on retry.
        let unavailable =
            |error| ClientError::Transport(ApiError::new(error).with_retryable(false));
        Ok(Self {
            inner: Arc::new(Inner {
                core: Mutex::new(core),
                http: (env.http)().map_err(unavailable)?,
                runtime: (env.runtime)().map_err(unavailable)?,
                clock: env.clock.clone(),
                endpoint: endpoint_url,
                options,
                connect: AsyncMutex::new(()),
                closed: AtomicBool::new(false),
                closing: Notify::new(),
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
        let closing = self.inner.closing.notified();
        self.inner.ensure_open()?;
        let discovery = self.inner.read_response(&request);
        let body = tokio::select! {
            biased;
            _ = closing => return Err(ClientError::Closed),
            body = discovery => body?,
        };
        self.inner.ensure_open()?;
        // Validate routing before any bearer is sent to a peer-provided interface URL.
        let card: AgentCard = serde_json::from_slice(&body)
            .map_err(|e| ClientError::InvalidResponse(e.to_string()))?;
        for interface in &card.supported_interfaces {
            if interface.protocol_binding == a2a::TRANSPORT_PROTOCOL_JSONRPC {
                let rpc = checked_url(&interface.url)?;
                if rpc.origin() != self.inner.endpoint.origin() {
                    return Err(ClientError::InvalidResponse(
                        "the card's JSONRPC interface must have the endpoint's origin".into(),
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
        cx: &crate::telemetry::Cx,
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
        cx: &crate::telemetry::Cx,
    ) -> Result<Reply, ClientError> {
        self.connect().await?;
        let id = self.inner.next_id();
        let request =
            self.inner.core.lock().subscribe(&task, offset, JsonRpcId::Number(id as i64))?;
        self.start(id, request.traced(cx.span())).await
    }

    async fn start(&self, id: u64, request: Request) -> Result<Reply, ClientError> {
        self.inner.ensure_open()?;
        let (updates, rx) = mpsc::channel(UPDATE_QUEUE);
        let (ack, acknowledged) = oneshot::channel();
        let (admit, admitted) = oneshot::channel();
        let (done, settled) = watch::channel(false);
        let (cancel, cancelled) = watch::channel(false);
        let (delivered, consumed) = watch::channel(false);
        let (known, identity) = watch::channel(request.task.clone());
        let remote = identity.clone();
        let terminal = Arc::new(AtomicBool::new(false));
        let call = RemoteCall {
            delivery: Delivery {
                inner: self.inner.clone(),
                ack: Some(ack),
                updates,
                identity: known,
                terminal: terminal.clone(),
            },
            task: RemoteTask {
                inner: self.inner.clone(),
                id,
                task_response: request.task_response,
                identity,
                terminal: terminal.clone(),
            },
            cancelled,
            consumed,
        };
        let inner = self.inner.clone();
        let task = TokioSpawner::new(self.inner.runtime.clone()).spawn_settled(
            call.run(request, admitted),
            move |outcome| async move {
                inner.cancel_unfinished_call(id, outcome, &terminal, &remote).await;
                inner.forget_call(id);
                // The client is released before the call is announced settled.
                drop(inner);
                done.send_replace(true);
            },
        );
        {
            let mut operations = self.inner.operations.lock();
            self.inner.ensure_open()?;
            operations.insert(id, Operation { _task: task, cancel, settled: settled.clone() });
        }
        let source = SharedStream::new(
            Box::new(ReplyUpdates {
                rx,
                delivered,
                operation: OperationGuard { inner: Arc::downgrade(&self.inner), id, settled },
                clock: self.inner.clock.clone(),
            }),
            self.inner.clock.clone(),
        );
        let _ = admit.send(());
        let task = acknowledged.await.map_err(|_| ClientError::Incomplete)??;
        Ok(Reply {
            task_id: task.id,
            thread_id: task.context_id,
            source,
            clock: self.inner.clock.clone(),
        })
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
        let reply = self.start(id, request).await?;
        match reply.pull().await {
            Ok(Pulled::Item(update)) => Ok(update),
            Ok(_) => Err(ClientError::Incomplete),
            Err(PullError::Failed(ProviderError::Local(error))) => {
                Err(ClientError::Transport(error))
            }
            Err(error) => Err(transport_error(error.to_string())),
        }
    }

    /// Stops admission and settles owned exchanges. Missed cleanup remains observable and owned.
    pub async fn aclose(&self) {
        self.inner.closed.store(true, Ordering::Release);
        self.inner.closing.notify_waiters();
        let waiting = self.inner.cancel_all();
        let wait = async {
            for mut settled in waiting {
                while !*settled.borrow_and_update() {
                    if settled.changed().await.is_err() {
                        break;
                    }
                }
            }
        };
        if tokio::time::timeout(HOST_TEARDOWN_TIMEOUT, wait).await.is_err() {
            let requests = self.inner.operations.lock().keys().copied().collect();
            self.inner.observe(CleanupObservation::Incomplete { requests });
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
        self.inner.closed.store(true, Ordering::Release);
        self.inner.closing.notify_waiters();
        self.inner.cancel_all();
    }
}

impl Inner {
    fn ensure_open(&self) -> Result<(), ClientError> {
        if self.closed.load(Ordering::Acquire) { Err(ClientError::Closed) } else { Ok(()) }
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
            log::error!(target: "livekit.agents", "A2A cleanup incomplete: {observation:?}");
        }
    }

    fn cancel_all(&self) -> Vec<watch::Receiver<bool>> {
        self.operations
            .lock()
            .values_mut()
            .map(|op| {
                op.cancel.send_replace(true);
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
                headers: request.headers.clone(),
                body: request.body.clone(),
            },
            &self.options,
            self.clock.clone(),
        )
    }

    async fn read_response(&self, request: &Request) -> Result<Vec<u8>, ClientError> {
        let mut exchange = self.exchange(request);
        let mut body = Vec::new();
        loop {
            match exchange.next().await.map_err(ClientError::Transport)? {
                HttpEvent::Started => body.clear(),
                HttpEvent::Headers(_) => {}
                HttpEvent::Bytes(bytes) => append_body(&mut body, &bytes)?,
                HttpEvent::End => return Ok(body),
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
        let cancellation = self.cancel_remote(task, Some(CALLER_CANCELED));
        tokio::pin!(cancellation);
        let result = match tokio::time::timeout(HOST_TEARDOWN_TIMEOUT, &mut cancellation).await {
            Ok(result) => result,
            Err(_) => {
                self.observe(CleanupObservation::CancellationIncomplete { task: task.clone() });
                cancellation.await
            }
        };
        if let Err(error) = result {
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
        outcome: TaskOutcome,
        terminal: &AtomicBool,
        remote: &watch::Receiver<Option<TaskRef>>,
    ) {
        if outcome == TaskOutcome::Completed || terminal.load(Ordering::Acquire) {
            return;
        }
        let task = remote.borrow().clone();
        match task {
            Some(task) => self.cancel_owned(&task).await,
            None => self.observe(CleanupObservation::UnacknowledgedRequest { request: id }),
        }
    }

    fn forget_call(&self, id: u64) {
        let mut operations = self.operations.lock();
        operations.remove(&id);
        // One scope: no exchange settles between the check and the close.
        if operations.is_empty() && self.closed.load(Ordering::Acquire) {
            self.core.lock().close();
        }
    }
}

/// One request's exchange, run as an owned task. It delivers the reply's updates and owes the
/// remote task cleanup when the caller leaves or the exchange fails.
struct RemoteCall {
    delivery: Delivery,
    task: RemoteTask,
    cancelled: watch::Receiver<bool>,
    consumed: watch::Receiver<bool>,
}

impl RemoteCall {
    async fn run(mut self, request: Request, admitted: oneshot::Receiver<()>) {
        if admitted.await.is_err() {
            return;
        }
        match self.exchange_unless_cancelled(request).await {
            Ok(()) => self.cancel_unread_question().await,
            Err(error) => self.report_failure(error).await,
        }
    }

    async fn exchange_unless_cancelled(&mut self, request: Request) -> Result<(), ClientError> {
        let Self { delivery, task, cancelled, .. } = self;
        let exchange = exchange(request, delivery);
        tokio::pin!(exchange);
        tokio::select! {
            biased;
            _ = cancelled.changed() => task.cancel_on_caller_exit(exchange.as_mut()).await,
            result = &mut exchange => result,
        }
    }

    /// Until the recipient consumes a buffered question, abandoning its reply still owes
    /// cancellation of the nonterminal remote task.
    async fn cancel_unread_question(&mut self) {
        if self.task.terminal.load(Ordering::Acquire) || *self.consumed.borrow() {
            return;
        }
        tokio::select! {
            _ = self.consumed.changed() => {},
            _ = self.cancelled.changed() => {},
        }
        if *self.consumed.borrow() {
            return;
        }
        let task = self.task.identity.borrow().clone();
        if let Some(task) = task {
            self.task.inner.cancel_owned(&task).await;
        }
    }

    async fn report_failure(&mut self, error: ClientError) {
        if let Some(ack) = self.delivery.ack.take() {
            let _ = ack.send(Err(error.clone()));
        }
        if *self.cancelled.borrow() {
            return;
        }
        self.task.cancel_after_failure(&error).await;
        tokio::select! {
            _ = self.delivery.updates.send(Err(error)) => {},
            _ = self.cancelled.changed() => {},
        }
    }
}

/// The remote task a call created, as far as it is known here, and the cleanup it is owed.
struct RemoteTask {
    inner: Arc<Inner>,
    id: u64,
    task_response: bool,
    identity: watch::Receiver<Option<TaskRef>>,
    terminal: Arc<AtomicBool>,
}

impl RemoteTask {
    /// Cleanup runs within the teardown bound; past it, the overrun is recorded and cleanup
    /// is still awaited.
    async fn cancel_on_caller_exit(
        &self,
        exchange: Pin<&mut impl Future<Output = Result<(), ClientError>>>,
    ) -> Result<(), ClientError> {
        let cleanup = self.cancel_abandoned(exchange);
        tokio::pin!(cleanup);
        if tokio::time::timeout(HOST_TEARDOWN_TIMEOUT, &mut cleanup).await.is_err() {
            self.inner.observe(CleanupObservation::Incomplete { requests: vec![self.id] });
            cleanup.await;
        }
        Err(ClientError::Closed)
    }

    async fn cancel_abandoned(
        &self,
        exchange: Pin<&mut impl Future<Output = Result<(), ClientError>>>,
    ) {
        if self.task_response {
            if let Err(error) = exchange.await
                && let Some(task) = self.identity.borrow().clone()
            {
                self.inner.observe(CleanupObservation::CancelUncertain {
                    task,
                    error: error.to_string(),
                });
            }
            return;
        }
        match self.await_identity(exchange).await {
            Some(task) if !self.terminal.load(Ordering::Acquire) => {
                if let Err(error) = self.inner.cancel_remote(&task, Some(CALLER_CANCELED)).await {
                    self.inner.observe(CleanupObservation::CancelUncertain {
                        task,
                        error: error.to_string(),
                    });
                }
            }
            Some(_) => {}
            None => {
                self.inner.observe(CleanupObservation::UnacknowledgedRequest { request: self.id })
            }
        }
    }

    /// Keeps the original send alive until its ack; resending a business request just to
    /// discover its task could duplicate a remote effect.
    async fn await_identity(
        &self,
        mut exchange: Pin<&mut impl Future<Output = Result<(), ClientError>>>,
    ) -> Option<TaskRef> {
        let mut identity = self.identity.clone();
        loop {
            if let Some(task) = identity.borrow_and_update().clone() {
                return Some(task);
            }
            tokio::select! {
                _ = identity.changed() => {},
                _ = &mut exchange => return identity.borrow().clone(),
            }
        }
    }

    /// A failed exchange cancels its nonterminal remote task, or records why it cannot.
    async fn cancel_after_failure(&self, error: &ClientError) {
        if self.terminal.load(Ordering::Acquire) {
            return;
        }
        let task = self.identity.borrow().clone();
        let Some(task) = task else {
            if !matches!(error, ClientError::Rpc { .. }) {
                self.inner.observe(CleanupObservation::UnacknowledgedRequest { request: self.id });
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
    ack: Option<oneshot::Sender<Result<TaskRef, ClientError>>>,
    updates: mpsc::Sender<Result<TaskUpdate, ClientError>>,
    identity: watch::Sender<Option<TaskRef>>,
    terminal: Arc<AtomicBool>,
}

impl Delivery {
    async fn deliver(&mut self, update: TaskUpdate) -> Result<(), ClientError> {
        let task = TaskRef { id: update.task_id.clone(), context_id: update.context_id.clone() };
        if !task.id.is_empty() {
            self.identity.send_replace(Some(task.clone()));
        }
        self.terminal.store(update.state.is_terminal(), Ordering::Release);
        if let Some(ack) = self.ack.take() {
            self.inner.core.lock().acknowledge(&update)?;
            let _ = ack.send(Ok(task));
        }
        self.updates.send(Ok(update)).await.map_err(|_| ClientError::Closed)
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
    pub async fn pull(&self) -> Result<Pulled<TaskUpdate>, PullError> {
        self.source.pull().await
    }
    pub async fn aclose(&self, deadline: Ts) -> Closed {
        self.source.close(deadline).await
    }
}

struct OperationGuard {
    inner: Weak<Inner>,
    id: u64,
    settled: watch::Receiver<bool>,
}

impl OperationGuard {
    fn cancel(&self) {
        if let Some(inner) = self.inner.upgrade()
            && let Some(operation) = inner.operations.lock().get_mut(&self.id)
        {
            operation.cancel.send_replace(true);
        }
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
    delivered: watch::Sender<bool>,
}

impl StreamOutput<TaskUpdate> for ReplyUpdates {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<TaskUpdate>, ProviderError>> {
        Box::pin(async move {
            let update = self.rx.recv().await.transpose().map_err(|e| match e {
                ClientError::Transport(error) => ProviderError::Local(error),
                error => {
                    ProviderError::from(ApiError::new(error.to_string()).with_retryable(false))
                }
            })?;
            if update.as_ref().is_some_and(TaskUpdate::is_settled) {
                self.delivered.send_replace(true);
            }
            Ok(update)
        })
    }
    fn cancel_safe(&self) -> bool {
        true
    }
    fn close(mut self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move {
            self.operation.cancel();
            let clock = self.clock.clone();
            Closed::bounded(&*clock, deadline, async move {
                while !*self.operation.settled.borrow_and_update() {
                    if self.operation.settled.changed().await.is_err() {
                        break;
                    }
                }
            })
            .await
        })
    }
}

async fn exchange(request: Request, delivery: &mut Delivery) -> Result<(), ClientError> {
    let mut reader = ReplyReader::new(&request)?;
    let mut exchange = delivery.inner.exchange(&request);
    loop {
        match exchange.next().await.map_err(ClientError::Transport)? {
            HttpEvent::Started => reader.restart(),
            HttpEvent::Headers(headers) => reader.framing = ReplyFraming::of(&headers),
            HttpEvent::Bytes(bytes) if reader.framing == ReplyFraming::EventStream => {
                if reader.read_events(&bytes, &mut exchange, delivery).await?.is_break() {
                    return reader.state.finish();
                }
            }
            HttpEvent::Bytes(bytes) => append_body(&mut reader.body, &bytes)?,
            HttpEvent::End => return reader.finish(&mut exchange, delivery).await,
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
    body: Vec<u8>,
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
            body: Vec::new(),
        })
    }

    fn restart(&mut self) {
        self.state = ReplyState::for_request(self.request);
        self.decoder = SseDecoder::new();
        self.body.clear();
        self.events.clear();
    }

    /// Delivers each event's update in order; breaks as soon as the reply has settled.
    async fn read_events(
        &mut self,
        bytes: &[u8],
        exchange: &mut Exchange,
        delivery: &mut Delivery,
    ) -> Result<ControlFlow<()>, ClientError> {
        self.decoder
            .feed(bytes, &mut self.events)
            .map_err(|error| ClientError::InvalidResponse(error.to_string()))?;
        for event in self.events.drain(..) {
            if let Some(update) = self.state.decode(event.data.as_bytes(), &self.id)? {
                exchange.delivered();
                exchange
                    .wait_for_consumer(delivery.deliver(update))
                    .await
                    .map_err(ClientError::Transport)??;
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
        delivery: &mut Delivery,
    ) -> Result<(), ClientError> {
        if self.framing == ReplyFraming::EventStream {
            self.decoder
                .finish()
                .map_err(|error| ClientError::InvalidResponse(error.to_string()))?;
            return self.state.finish();
        }
        let update = if self.request.task_response {
            Some(self.state.decode_task(&self.body, &self.id)?)
        } else {
            self.state.decode(&self.body, &self.id)?
        };
        if let Some(update) = update {
            exchange
                .wait_for_consumer(delivery.deliver(update))
                .await
                .map_err(ClientError::Transport)??;
        }
        self.state.finish()
    }
}

fn append_body(body: &mut Vec<u8>, bytes: &[u8]) -> Result<(), ClientError> {
    if body.len() + bytes.len() > RESPONSE_LIMIT {
        return Err(ClientError::InvalidResponse(format!(
            "the answer exceeds {RESPONSE_LIMIT} bytes"
        )));
    }
    body.extend_from_slice(bytes);
    Ok(())
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

fn transport_error(error: String) -> ClientError {
    ClientError::Transport(ApiError::connection(Some(error)))
}
