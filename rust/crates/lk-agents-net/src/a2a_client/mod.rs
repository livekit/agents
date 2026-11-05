//! Native A2A client execution. Protocol decisions stay in `lk-agents-a2a::client`; discovery,
//! HTTP/SSE and owned cancellation run here through the shared HTTP pool and task/source owners.

pub mod authority;
pub mod host;
pub mod remote;

use std::collections::BTreeMap;
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
use crate::sse::SseDecoder;
use crate::task::TokioSpawner;
use lk_agents_abi::{OwnedTask, SpawnerExt, TaskOutcome};
use parking_lot::Mutex;

const RESPONSE_LIMIT: usize = 8 * 1024 * 1024;
const UPDATE_QUEUE: usize = 16;

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
        let expected = request.task.clone();
        let (tx, rx) = mpsc::channel(UPDATE_QUEUE);
        let (ack, acknowledged) = oneshot::channel();
        let (admit, admitted) = oneshot::channel();
        let (done, settled) = watch::channel(false);
        let (cancel, mut cancelled) = watch::channel(false);
        let (delivered, mut delivery) = watch::channel(false);
        let (known, mut known_task) = watch::channel(expected.clone());
        let remote = known_task.clone();
        let terminal = Arc::new(AtomicBool::new(false));
        let work_inner = self.inner.clone();
        let work_terminal = terminal.clone();
        let cleanup_inner = self.inner.clone();
        let task_response = request.task_response;
        let work = async move {
            if admitted.await.is_err() {
                return;
            }
            let mut ack = Some(ack);
            let result = {
                let exchange =
                    exchange(&work_inner, request, &mut ack, &tx, &work_terminal, &known);
                tokio::pin!(exchange);
                tokio::select! {
                    biased;
                    _ = cancelled.changed() => {
                        let cleanup = async {
                            if task_response {
                                if let Err(error) = exchange.as_mut().await
                                    && let Some(task) = known_task.borrow().clone()
                                {
                                    work_inner.observe(CleanupObservation::CancelUncertain {
                                        task,
                                        error: error.to_string(),
                                    });
                                }
                                return;
                            }
                            // Keep the original send alive until its ack; resending a business
                            // request just to discover its task could duplicate a remote effect.
                            let task = loop {
                                if let Some(task) = known_task.borrow_and_update().clone() {
                                    break Some(task);
                                }
                                tokio::select! {
                                    _ = known_task.changed() => {},
                                    _ = &mut exchange => break known_task.borrow().clone(),
                                }
                            };
                            match task {
                                Some(task) if !work_terminal.load(Ordering::Acquire) => {
                                    let reason = Some("caller-canceled");
                                    let cancel = work_inner.cancel_remote(&task, reason);
                                    if let Err(error) = cancel.await {
                                        work_inner.observe(CleanupObservation::CancelUncertain {
                                            task,
                                            error: error.to_string(),
                                        });
                                    }
                                }
                                Some(_) => {},
                                None => work_inner.observe(
                                    CleanupObservation::UnacknowledgedRequest { request: id },
                                ),
                            }
                        };
                        tokio::pin!(cleanup);
                        let teardown = tokio::time::timeout(HOST_TEARDOWN_TIMEOUT, &mut cleanup);
                        if teardown.await.is_err() {
                            let requests = vec![id];
                            work_inner.observe(CleanupObservation::Incomplete { requests });
                            cleanup.await;
                        }
                        Err(ClientError::Closed)
                    },
                    result = &mut exchange => result,
                }
            };
            if result.is_ok() && !work_terminal.load(Ordering::Acquire) && !*delivery.borrow() {
                // Until the recipient consumes a buffered question, abandoning its reply
                // still owes cancellation of the nonterminal remote task.
                tokio::select! {
                    _ = delivery.changed() => {},
                    _ = cancelled.changed() => {},
                }
                if !*delivery.borrow() {
                    let task = known_task.borrow().clone();
                    if let Some(task) = task {
                        work_inner.cancel_owned(&task).await;
                    }
                }
            }
            if let Err(error) = result {
                if let Some(ack) = ack.take() {
                    let _ = ack.send(Err(error.clone()));
                }
                if !*cancelled.borrow() {
                    if !work_terminal.load(Ordering::Acquire) {
                        let task = known_task.borrow().clone();
                        if let Some(task) = task {
                            if task_response {
                                work_inner.observe(CleanupObservation::CancelUncertain {
                                    task,
                                    error: error.to_string(),
                                });
                            } else {
                                work_inner.cancel_owned(&task).await;
                            }
                        } else if !matches!(error, ClientError::Rpc { .. }) {
                            work_inner
                                .observe(CleanupObservation::UnacknowledgedRequest { request: id });
                        }
                    }
                    tokio::select! {
                        _ = tx.send(Err(error)) => {},
                        _ = cancelled.changed() => {},
                    }
                }
            }
        };
        let task = TokioSpawner::new(self.inner.runtime.clone()).spawn_settled(
            work,
            move |outcome| async move {
                {
                    let inner = cleanup_inner;
                    if outcome != TaskOutcome::Completed && !terminal.load(Ordering::Acquire) {
                        let task = remote.borrow().clone();
                        match task {
                            Some(task) => {
                                inner.cancel_owned(&task).await;
                            }
                            None => inner
                                .observe(CleanupObservation::UnacknowledgedRequest { request: id }),
                        }
                    }
                    let mut operations = inner.operations.lock();
                    operations.remove(&id);
                    // One scope: no exchange settles between the check and the close.
                    if operations.is_empty() && inner.closed.load(Ordering::Acquire) {
                        inner.core.lock().close();
                    }
                }
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
        let cancellation = self.cancel_remote(task, Some("caller-canceled"));
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

async fn exchange(
    inner: &Inner,
    request: Request,
    ack: &mut Option<oneshot::Sender<Result<TaskRef, ClientError>>>,
    tx: &mpsc::Sender<Result<TaskUpdate, ClientError>>,
    terminal: &AtomicBool,
    known: &watch::Sender<Option<TaskRef>>,
) -> Result<(), ClientError> {
    let id =
        request.id.clone().ok_or_else(|| ClientError::InvalidInput("an RPC needs an id".into()))?;
    let mut exchange = inner.exchange(&request);
    let mut state = ReplyState::for_request(&request);
    let mut decoder = SseDecoder::new();
    let mut body = Vec::new();
    let mut sse = false;
    let mut events = Vec::new();
    loop {
        match exchange.next().await.map_err(ClientError::Transport)? {
            HttpEvent::Started => {
                state = ReplyState::for_request(&request);
                decoder = SseDecoder::new();
                body.clear();
                events.clear();
            }
            HttpEvent::Headers(headers) => {
                sse = headers.get("content-type").and_then(|h| h.to_str().ok()).is_some_and(|v| {
                    v.split(';').next().is_some_and(|v| v.trim() == "text/event-stream")
                });
            }
            HttpEvent::Bytes(bytes) if sse => {
                decoder
                    .feed(&bytes, &mut events)
                    .map_err(|error| ClientError::InvalidResponse(error.to_string()))?;
                for event in events.drain(..) {
                    if let Some(update) = state.decode(event.data.as_bytes(), &id)? {
                        exchange.delivered();
                        exchange
                            .wait_for_consumer(deliver(inner, update, ack, tx, terminal, known))
                            .await
                            .map_err(ClientError::Transport)??;
                    }
                    if state.settled() {
                        return state.finish();
                    }
                }
            }
            HttpEvent::Bytes(bytes) => append_body(&mut body, &bytes)?,
            HttpEvent::End => {
                if sse {
                    decoder
                        .finish()
                        .map_err(|error| ClientError::InvalidResponse(error.to_string()))?;
                } else {
                    let update = if request.task_response {
                        Some(state.decode_task(&body, &id)?)
                    } else {
                        state.decode(&body, &id)?
                    };
                    if let Some(update) = update {
                        exchange
                            .wait_for_consumer(deliver(inner, update, ack, tx, terminal, known))
                            .await
                            .map_err(ClientError::Transport)??;
                    }
                }
                return state.finish();
            }
        }
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

async fn deliver(
    inner: &Inner,
    update: TaskUpdate,
    ack: &mut Option<oneshot::Sender<Result<TaskRef, ClientError>>>,
    tx: &mpsc::Sender<Result<TaskUpdate, ClientError>>,
    terminal: &AtomicBool,
    known: &watch::Sender<Option<TaskRef>>,
) -> Result<(), ClientError> {
    let task = TaskRef { id: update.task_id.clone(), context_id: update.context_id.clone() };
    if !task.id.is_empty() {
        known.send_replace(Some(task.clone()));
    }
    terminal.store(update.state.is_terminal(), Ordering::Release);
    if let Some(ack) = ack.take() {
        inner.core.lock().acknowledge(&update)?;
        let _ = ack.send(Ok(task));
    }
    tx.send(Ok(update)).await.map_err(|_| ClientError::Closed)
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
