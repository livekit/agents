pub mod authority;
pub mod host;
pub mod observer;

use std::collections::BTreeMap;
use std::future::Future;
use std::ops::ControlFlow;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::transport::HttpRequest;
use a2a::{AgentCard, JsonRpcId, Message};
use lk_agents_abi::errors::{ApiError, ErrorKind, ProviderError};
use lk_agents_abi::{ApiConnectOptions, Clock, Closed, StreamOutput, Ts};
use lk_agents_core::a2a::client::{
    Client, ClientError, HttpMethod, ReplyEvent, ReplyState, Request, TaskUpdate,
};
use lk_agents_core::a2a::rpc::TaskRef;
use tokio::sync::{Mutex as AsyncMutex, mpsc};
use tokio_util::sync::CancellationToken;

use crate::env::{ProcessEnv, ProcessError};
use crate::http::BodyReader;
use crate::http::exchange::{Exchange, HttpEvent};
use crate::http::port::HttpExchangePort;
use crate::signal;
use crate::source::{Pulled, SharedStream};
use crate::sse::{SseDecoder, SseEvent};
use lk_agents_abi::{OwnedTask, SpawnerExt};
use parking_lot::Mutex;

const RESPONSE_LIMIT: usize = 8 * 1024 * 1024;
const UPDATE_QUEUE: usize = 16;
/// The media type of a streamed (SSE) reply.
const EVENT_STREAM: &str = "text/event-stream";

/// Admission is signaled by the first valid update. Transport teardown does not change the
/// remote task's lifecycle; only an explicit CancelTask request may do that.
#[derive(Clone, Default)]
struct Progress {
    acknowledgement: Option<Result<TaskRef, ClientError>>,
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
    http: Arc<dyn HttpExchangePort>,
    spawner: Arc<dyn lk_agents_abi::Spawner>,
    clock: Arc<dyn Clock>,
    connect: AsyncMutex<()>,
    closing: CancellationToken,
    requests: AtomicU64,
    operations: Mutex<BTreeMap<u64, Operation>>,
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
        Self::with_ports(
            endpoint_url,
            core,
            options,
            (env.http)().map_err(unavailable)?,
            (env.spawner)().map_err(unavailable)?,
            env.clock.clone(),
        )
    }

    /// Inject only the socket boundary; production request policy, framing and ownership run
    /// unchanged on virtual time in deterministic simulations.
    pub fn with_transport(
        endpoint: String,
        token: Option<String>,
        options: ApiConnectOptions,
        http: Arc<dyn HttpExchangePort>,
        spawner: Arc<dyn lk_agents_abi::Spawner>,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, ClientError> {
        options.validate().map_err(|error| ClientError::InvalidInput(error.to_string()))?;
        let endpoint_url = checked_url(&endpoint)?;
        let core = Client::new(endpoint, token)?;
        Self::with_ports(endpoint_url, core, options, http, spawner, clock)
    }

    fn with_ports(
        endpoint_url: url::Url,
        core: Client,
        options: ApiConnectOptions,
        http: Arc<dyn HttpExchangePort>,
        spawner: Arc<dyn lk_agents_abi::Spawner>,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, ClientError> {
        Ok(Self {
            inner: Arc::new(Inner {
                core: Mutex::new(core),
                http,
                spawner,
                clock,
                endpoint: endpoint_url,
                options,
                connect: AsyncMutex::new(()),
                closing: CancellationToken::new(),
                requests: AtomicU64::new(1),
                operations: Mutex::new(BTreeMap::new()),
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
        message: Message,
        thread_id: Option<&str>,
        task_id: Option<&str>,
        history: Option<&lk_agents_core::a2a::caller_history::CallerHistory>,
        cx: &lk_agents_telemetry::Cx,
    ) -> Result<Reply, ClientError> {
        self.send_observed(message, thread_id, task_id, history, cx, None).await
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "same operation with an optional native observation port"
    )]
    pub async fn send_observed(
        &self,
        mut message: Message,
        thread_id: Option<&str>,
        task_id: Option<&str>,
        history: Option<&lk_agents_core::a2a::caller_history::CallerHistory>,
        cx: &lk_agents_telemetry::Cx,
        observer: Option<Arc<dyn observer::RequestObserver>>,
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
        self.start(id, request.traced(cx.span()), observer).await
    }

    pub async fn subscribe(
        &self,
        task: TaskRef,
        offset: Option<&str>,
        cx: &lk_agents_telemetry::Cx,
    ) -> Result<Reply, ClientError> {
        self.subscribe_observed(task, offset, cx, None).await
    }

    pub async fn subscribe_observed(
        &self,
        task: TaskRef,
        offset: Option<&str>,
        cx: &lk_agents_telemetry::Cx,
        observer: Option<Arc<dyn observer::RequestObserver>>,
    ) -> Result<Reply, ClientError> {
        self.connect().await?;
        let id = self.inner.next_id();
        let request =
            self.inner.core.lock().subscribe(&task, offset, JsonRpcId::Number(id as i64))?;
        self.start(id, request.traced(cx.span()), observer).await
    }

    async fn start(
        &self,
        id: u64,
        request: Request,
        observer: Option<Arc<dyn observer::RequestObserver>>,
    ) -> Result<Reply, ClientError> {
        let (task, updates) = self.open(id, request, observer).await?;
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
        observer: Option<Arc<dyn observer::RequestObserver>>,
    ) -> Result<(TaskRef, ReplyUpdates), ClientError> {
        self.inner.ensure_open()?;
        if let Some(observer) = &observer {
            observer.prepared(&request).await?;
            observer.dispatch_started().await?;
        }
        let (updates, rx) = mpsc::channel(UPDATE_QUEUE);
        // Each producer lives with its writer: the delivery in the call's task, admission here,
        // consumption in the reply, settlement in the task's settle hook.
        let (progress, acknowledged) = signal::signal(Progress::default());
        let (admit, admitted) = signal::once::<()>();
        let (settle, settled) = signal::once::<()>();
        let stop = CancellationToken::new();
        let operation = OperationGuard { stop: stop.clone(), settled: settled.clone() };
        let call = RemoteCall {
            delivery: Delivery { inner: self.inner.clone(), updates, progress, observer },
            task_response: request.task_response,
            admitted: admitted.clone(),
            stop: stop.clone(),
        };
        let inner = self.inner.clone();
        let task = self.inner.spawner.spawn_settled(call.run(request), move |_| async move {
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
        let updates = ReplyUpdates { rx, operation, clock: self.inner.clock.clone() };
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
        cx: &lk_agents_telemetry::Cx,
        observer: Option<Arc<dyn observer::RequestObserver>>,
    ) -> Result<TaskUpdate, ClientError> {
        self.connect().await?;
        let id = self.inner.next_id();
        let request = self.inner.core.lock().cancel(&task, reason, JsonRpcId::Number(id as i64))?;
        let (_, mut updates) = self.open(id, request.traced(cx.span()), observer).await?;
        updates.recv().await?.and_then(|event| event.update).ok_or_else(ClientError::incomplete)
    }

    /// Reconciliation reads an existing task. It never repeats the message that created it.
    pub async fn get_task(
        &self,
        task: TaskRef,
        cx: &lk_agents_telemetry::Cx,
        observer: Option<Arc<dyn observer::RequestObserver>>,
    ) -> Result<a2a::Task, ClientError> {
        self.connect().await?;
        let id = self.inner.next_id();
        let request = self.inner.core.lock().get_task(&task, JsonRpcId::Number(id as i64))?;
        let (_, mut updates) = self.open(id, request.traced(cx.span()), observer).await?;
        match updates.recv().await?.map(|event| event.response) {
            Some(a2a::StreamResponse::Task(task)) => Ok(task),
            _ => Err(ClientError::invalid_response("GetTask did not return its task")),
        }
    }

    /// Stops admission and settles owned exchanges. Missed cleanup remains observable and owned.
    pub async fn aclose(&self) {
        self.inner.closing.cancel();
        let waiting = self.inner.cancel_all();
        for settled in waiting {
            settled.until(Option::is_some).await;
        }
        let operations = self.inner.operations.lock();
        if operations.is_empty() {
            self.inner.core.lock().close();
        }
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
        let mut options = self.options;
        // A message ID does not guarantee business idempotency on an arbitrary A2A peer.
        // Recovery reconciles uncertain sends; the transport never resends them.
        if !request.replay_safe {
            options.max_retry = 0;
        }
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
            &options,
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

    fn forget_call(&self, id: u64) {
        let mut operations = self.operations.lock();
        operations.remove(&id);
        // One scope: no exchange settles between the check and the close.
        if operations.is_empty() && self.closing.is_cancelled() {
            self.core.lock().close();
        }
    }
}

/// Owns the local exchange, not the remote business task. Closing observation releases the
/// socket. An explicit control request already in flight is drained to learn its outcome.
struct RemoteCall {
    delivery: Delivery,
    task_response: bool,
    admitted: signal::Cell<Option<()>>,
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
            let exchange = exchange(&inner, &request, &mut self.delivery);
            tokio::pin!(exchange);
            tokio::select! {
                biased;
                () = self.stop.cancelled(), if !self.task_response => None,
                result = &mut exchange => Some(result),
            }
        };
        let result = if let Some(observer) = &self.delivery.observer {
            let error = result.as_ref().and_then(|result| result.as_ref().err()).cloned();
            match observer.ended(error).await {
                Ok(()) => result,
                Err(error) => Some(Err(error)),
            }
        } else {
            result
        };
        match result {
            Some(Ok(())) => {}
            Some(Err(error)) => {
                self.delivery.fail_acknowledgement(error.clone());
                tokio::select! {
                    biased;
                    () = self.stop.cancelled() => {},
                    _ = self.delivery.updates.send(Err(error)) => {},
                }
            }
            None => self.delivery.fail_acknowledgement(ClientError::closed()),
        }
    }
}

/// Hands a call's decoded updates to its caller; the first one acknowledges the call.
struct Delivery {
    inner: Arc<Inner>,
    updates: mpsc::Sender<Result<ReplyEvent, ClientError>>,
    progress: signal::Producer<Progress>,
    observer: Option<Arc<dyn observer::RequestObserver>>,
}

impl Delivery {
    async fn forward(
        &mut self,
        exchange: &mut Exchange,
        update: ReplyEvent,
    ) -> Result<(), ClientError> {
        exchange.delivered();
        exchange.wait_for_consumer(self.deliver(update)).await.map_err(ClientError::Transport)?
    }

    fn fail_acknowledgement(&self, error: ClientError) {
        self.progress.modify(|p| {
            p.acknowledgement.get_or_insert(Err(error));
        });
    }

    async fn deliver(&mut self, update: ReplyEvent) -> Result<(), ClientError> {
        let task = update.route();
        let first = self.progress.with(|p| p.acknowledgement.is_none());
        if first {
            self.inner.core.lock().acknowledge_context(&task.context_id)?;
            self.progress.modify(|p| p.acknowledgement = Some(Ok(task)));
        }
        self.updates.send(Ok(update)).await.map_err(|_| ClientError::closed())
    }
}

/// IDs are available when `send()` returns, while updates keep their one-consumer source.
pub struct Reply {
    pub task_id: String,
    pub thread_id: String,
    source: SharedStream<ReplyEvent>,
    clock: Arc<dyn Clock>,
}

impl Reply {
    /// `None` while another pull of this reply is in flight.
    pub fn try_pull(
        &self,
    ) -> Option<impl Future<Output = Result<Pulled<TaskUpdate>, ProviderError>> + '_> {
        let first = self.source.try_pull()?;
        Some(async move {
            let mut pulled = first.await?;
            loop {
                match pulled {
                    Pulled::Item(ReplyEvent { update: Some(update), .. }) => {
                        return Ok(Pulled::Item(update));
                    }
                    Pulled::Item(_) => {}
                    Pulled::End => return Ok(Pulled::End),
                    Pulled::Cancelled => return Ok(Pulled::Cancelled),
                }
                // Reclaim in this poll before yielding again; an artifact-only event does not
                // hand a legacy pull's consumer position to a second caller.
                let next = self.source.try_pull().expect("the completed pull released its claim");
                pulled = next.await?;
            }
        })
    }

    /// The native owner's canonical observation path, before presentation normalization.
    pub fn try_pull_event(
        &self,
    ) -> Option<impl Future<Output = Result<Pulled<ReplyEvent>, ProviderError>> + '_> {
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
    rx: mpsc::Receiver<Result<ReplyEvent, ClientError>>,
    operation: OperationGuard,
    clock: Arc<dyn Clock>,
}

impl ReplyUpdates {
    async fn recv(&mut self) -> Result<Option<ReplyEvent>, ClientError> {
        self.rx.recv().await.transpose()
    }
}

impl StreamOutput<ReplyEvent> for ReplyUpdates {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<ReplyEvent>, ProviderError>> {
        Box::pin(async move {
            self.recv().await.map_err(|error| match error {
                ClientError::Transport(error) => ProviderError::Local(error),
                error @ ClientError::Protocol(_) => {
                    ApiError::new(ErrorKind::Protocol, error.to_string()).into()
                }
                error => ApiError::new(ErrorKind::InvalidRequest, error.to_string()).into(),
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
                self.operation.settled.until(Option::is_some).await;
            })
            .await
        })
    }
}

async fn exchange(
    inner: &Inner,
    request: &Request,
    updates: &mut Delivery,
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
        updates: &mut Delivery,
    ) -> Result<ControlFlow<()>, ClientError> {
        self.decoder.feed(bytes, &mut self.events).map_err(ClientError::invalid_response)?;
        for event in std::mem::take(&mut self.events) {
            let update = self.decode(event.data.as_bytes(), false, exchange, updates).await?;
            updates.forward(exchange, update).await?;
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
        updates: &mut Delivery,
    ) -> Result<(), ClientError> {
        if self.framing == ReplyFraming::EventStream {
            self.decoder.finish().map_err(ClientError::invalid_response)?;
            return self.state.finish();
        }
        let body = self.body.as_bytes().to_vec();
        let update = self.decode(&body, self.request.task_response, exchange, updates).await?;
        updates.forward(exchange, update).await?;
        if self.request.task_response { Ok(()) } else { self.state.finish() }
    }

    async fn decode(
        &mut self,
        body: &[u8],
        task_response: bool,
        exchange: &mut Exchange,
        updates: &Delivery,
    ) -> Result<ReplyEvent, ClientError> {
        let response = lk_agents_core::a2a::client::decode_response(body, &self.id, task_response)?;
        // The peer has sent content: a checkpoint or normalization failure cannot make this
        // business request safe to repeat. Record the raw SDK value before normalization.
        exchange.delivered();
        if let Some(observer) = &updates.observer {
            observer.response(self.request, response.clone()).await?;
        }
        let update = if task_response {
            // A unary task lookup is a protocol snapshot, including unfamiliar-to-legacy
            // statuses. Its raw consumer does not need the legacy TaskUpdate projection.
            match &response {
                a2a::StreamResponse::Task(task)
                    if matches!(
                        task.status.state,
                        a2a::TaskState::AuthRequired | a2a::TaskState::Unspecified
                    ) =>
                {
                    None
                }
                _ => self.state.accept(response.clone())?,
            }
        } else {
            self.state.accept(response.clone())?
        };
        Ok(ReplyEvent { response, update })
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
