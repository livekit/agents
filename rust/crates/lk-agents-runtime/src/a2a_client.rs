//! Native A2A client execution. Protocol decisions stay in `lk-agents-a2a::client`; discovery,
//! HTTP/SSE and owned cancellation run here through the shared HTTP pool and task/source owners.

pub mod authority;

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use a2a::{AgentCard, JsonRpcId, Message};
use futures_util::StreamExt;
use lk_agents_a2a::client::{Client, ClientError, ReplyState, Request, TaskUpdate};
use lk_agents_a2a::rpc::TaskRef;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::{Clock, Closed, HOST_TEARDOWN_TIMEOUT, StreamSource, Ts};
use lk_agents_inference::transport::HttpRequest;
use tokio::sync::{Mutex as AsyncMutex, Notify, mpsc, oneshot, watch};

use crate::http::{GatewayHttp, Reporter, read_body};
use crate::source::{NativeSource, PullError, Pulled};
use crate::sse::SseDecoder;
use crate::task::{OwnedTask, TaskOutcome};

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
}

/// A connection owns its outstanding exchanges; it holds no remote-tool conversation state.
pub struct AgentClient {
    inner: Arc<Inner>,
}

impl AgentClient {
    pub fn new(
        endpoint: String,
        token: Option<String>,
        runtime: tokio::runtime::Handle,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, ClientError> {
        let endpoint_url = checked_url(&endpoint)?;
        let core = Client::new(endpoint, token)?;
        Ok(Self {
            inner: Arc::new(Inner {
                core: Mutex::new(core),
                http: GatewayHttp::shared().map_err(transport_error)?,
                runtime,
                clock,
                endpoint: endpoint_url,
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
        if let Some(card) = self.inner.core.lock().map_err(poisoned)?.card() {
            return Ok(card.clone());
        }
        let request = self.inner.core.lock().map_err(poisoned)?.card_request()?;
        let closing = self.inner.closing.notified();
        self.inner.ensure_open()?;
        let discovery = async {
            let response = self
                .inner
                .http
                .get(request.url, request.headers)
                .await
                .map_err(|e| transport_error(format!("{e:?}")))?;
            read_body(response, RESPONSE_LIMIT).await.map_err(|e| transport_error(format!("{e:?}")))
        };
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
        Ok(self.inner.core.lock().map_err(poisoned)?.accept_card(&body)?.clone())
    }

    pub fn offers_extension(&self) -> Result<bool, ClientError> {
        Ok(self.inner.core.lock().map_err(poisoned)?.offers_extension())
    }

    pub async fn send(
        &self,
        message: Message,
        thread_id: Option<&str>,
        task_id: Option<&str>,
    ) -> Result<Reply, ClientError> {
        self.send_with_history(message, thread_id, task_id, None).await
    }

    /// Text is converted once, before execution, so retries retain its generated messageId.
    pub async fn send_text(
        &self,
        text: String,
        thread_id: Option<&str>,
        task_id: Option<&str>,
    ) -> Result<Reply, ClientError> {
        self.send(
            Message {
                message_id: crate::storage::shortuuid("msg_"),
                role: a2a::Role::User,
                parts: vec![a2a::Part::text(text)],
                context_id: None,
                task_id: None,
                metadata: None,
                extensions: None,
                reference_task_ids: None,
            },
            thread_id,
            task_id,
        )
        .await
    }

    /// An explicitly supplied, validated conversation is sent only to a capable peer and
    /// omitted when an explicit caller thread lets the endpoint load its permitted history.
    pub async fn send_with_history(
        &self,
        mut message: Message,
        thread_id: Option<&str>,
        task_id: Option<&str>,
        history: Option<&lk_agents_a2a::caller_history::CallerHistory>,
    ) -> Result<Reply, ClientError> {
        self.inner.core.lock().map_err(poisoned)?.validate_message(&message, thread_id, task_id)?;
        self.connect().await?;
        if thread_id.is_none()
            && message.context_id.is_none()
            && self.offers_extension()?
            && let Some(history) = history
        {
            message
                .parts
                .push(history.to_part().map_err(|e| ClientError::InvalidInput(e.to_string()))?);
        }
        let id = self.inner.next_id();
        let request = self.inner.core.lock().map_err(poisoned)?.send(
            message,
            thread_id,
            task_id,
            JsonRpcId::Number(id as i64),
        )?;
        self.start(id, request).await
    }

    pub async fn subscribe(
        &self,
        task: TaskRef,
        offset: Option<&str>,
    ) -> Result<Reply, ClientError> {
        self.connect().await?;
        let id = self.inner.next_id();
        let request = self.inner.core.lock().map_err(poisoned)?.subscribe(
            &task,
            offset,
            JsonRpcId::Number(id as i64),
        )?;
        self.start(id, request).await
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
        let remote = Arc::new(Mutex::new(expected.clone()));
        let terminal = Arc::new(AtomicBool::new(false));
        let work_inner = self.inner.clone();
        let work_remote = remote.clone();
        let work_terminal = terminal.clone();
        let cleanup_inner = self.inner.clone();
        let task_response = request.task_response;
        let work = async move {
            if admitted.await.is_err() {
                return;
            }
            let mut ack = Some(ack);
            let result = {
                let exchange = exchange(
                    &work_inner,
                    request,
                    &mut ack,
                    &tx,
                    &work_remote,
                    &work_terminal,
                    &known,
                );
                tokio::pin!(exchange);
                tokio::select! {
                    biased;
                    _ = cancelled.changed() => {
                        let cleanup = async {
                            if task_response {
                                if let Err(error) = exchange.as_mut().await
                                    && let Some(task) = known_task.borrow().clone() {
                                        work_inner.observe(CleanupObservation::CancelUncertain {
                                            task, error: error.to_string(),
                                        });
                                }
                                return;
                            }
                            // Keep the original send alive until its ack; resending a business
                            // request just to discover its task could duplicate a remote effect.
                            let task = loop {
                                if let Some(task) = known_task.borrow_and_update().clone() { break Some(task); }
                                tokio::select! {
                                    _ = known_task.changed() => {},
                                    _ = &mut exchange => break known_task.borrow().clone(),
                                }
                            };
                            match task {
                                Some(task) if !work_terminal.load(Ordering::Acquire) => {
                                    if let Err(error) = work_inner.cancel_remote(&task, Some("caller-canceled")).await {
                                        work_inner.observe(CleanupObservation::CancelUncertain {
                                            task, error: error.to_string(),
                                        });
                                    }
                                }
                                Some(_) => {},
                                None => work_inner.observe(CleanupObservation::UnacknowledgedRequest { request: id }),
                            }
                        };
                        tokio::pin!(cleanup);
                        if tokio::time::timeout(HOST_TEARDOWN_TIMEOUT, &mut cleanup).await.is_err() {
                            work_inner.observe(CleanupObservation::Incomplete { requests: vec![id] });
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
                    let task = work_remote.lock().ok().and_then(|task| task.clone());
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
                        let task = work_remote.lock().ok().and_then(|task| task.clone());
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
        let task = OwnedTask::spawn(&self.inner.runtime, work, move |outcome| async move {
            {
                let inner = cleanup_inner;
                if outcome != TaskOutcome::Completed && !terminal.load(Ordering::Acquire) {
                    let task = remote.lock().ok().and_then(|task| task.clone());
                    match task {
                        Some(task) => {
                            inner.cancel_owned(&task).await;
                        }
                        None => {
                            inner.observe(CleanupObservation::UnacknowledgedRequest { request: id })
                        }
                    }
                }
                let empty = if let Ok(mut operations) = inner.operations.lock() {
                    operations.remove(&id);
                    operations.is_empty()
                } else {
                    false
                };
                if empty
                    && inner.closed.load(Ordering::Acquire)
                    && let Ok(mut core) = inner.core.lock()
                {
                    core.close();
                }
            }
            done.send_replace(true);
        });
        {
            let mut operations = self.inner.operations.lock().map_err(poisoned)?;
            self.inner.ensure_open()?;
            operations.insert(id, Operation { _task: task, cancel, settled: settled.clone() });
        }
        let source = NativeSource::new(Box::new(ReplyUpdates {
            rx,
            delivered,
            operation: OperationGuard { inner: Arc::downgrade(&self.inner), id, settled },
            clock: self.inner.clock.clone(),
        }));
        let _ = admit.send(());
        let task = acknowledged.await.map_err(|_| ClientError::Incomplete)??;
        Ok(Reply { task_id: task.id, thread_id: task.context_id, source })
    }

    /// Always sends contextId and the full-URI reason, independent of card capability hints.
    pub async fn cancel(
        &self,
        task: TaskRef,
        reason: Option<&str>,
    ) -> Result<TaskUpdate, ClientError> {
        self.connect().await?;
        let id = self.inner.next_id();
        let request = self.inner.core.lock().map_err(poisoned)?.cancel(
            &task,
            reason,
            JsonRpcId::Number(id as i64),
        )?;
        let reply = self.start(id, request).await?;
        match reply.pull().await {
            Ok(Pulled::Item(update)) => Ok(update),
            Ok(_) => Err(ClientError::Incomplete),
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
            let requests = self
                .inner
                .operations
                .lock()
                .map(|ops| ops.keys().copied().collect())
                .unwrap_or_default();
            self.inner.observe(CleanupObservation::Incomplete { requests });
        }
        if self.inner.operations.lock().is_ok_and(|ops| ops.is_empty())
            && let Ok(mut core) = self.inner.core.lock()
        {
            core.close();
        }
    }

    pub fn cleanup_observations(&self) -> Vec<CleanupObservation> {
        self.inner.observations.lock().map(|o| o.clone()).unwrap_or_default()
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
        if let Ok(mut observations) = self.observations.lock()
            && !observations.contains(&observation)
        {
            log::error!(target: "livekit.agents", "A2A cleanup incomplete: {observation:?}");
            observations.push(observation);
        }
    }

    fn cancel_all(&self) -> Vec<watch::Receiver<bool>> {
        self.operations
            .lock()
            .map(|mut operations| {
                operations
                    .values_mut()
                    .map(|op| {
                        op.cancel.send_replace(true);
                        op.settled.clone()
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    async fn post(&self, request: &Request) -> Result<reqwest::Response, ClientError> {
        let (tx, _rx) = mpsc::channel(2);
        let reporter = Reporter { tx, clock: self.clock.clone() };
        self.http
            .send(
                HttpRequest {
                    url: request.url.clone(),
                    headers: request.headers.clone(),
                    body: request.body.clone(),
                },
                &reporter,
                |_| (),
            )
            .await
            .map_err(|e| transport_error(format!("{e:?}")))
    }

    async fn cancel_remote(
        &self,
        task: &TaskRef,
        reason: Option<&str>,
    ) -> Result<TaskUpdate, ClientError> {
        let id = JsonRpcId::Number(self.next_id() as i64);
        let request = self.core.lock().map_err(poisoned)?.cancel(task, reason, id.clone())?;
        let response = self.post(&request).await?;
        let body = read_body(response, RESPONSE_LIMIT)
            .await
            .map_err(|e| transport_error(format!("{e:?}")))?;
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
    source: NativeSource<TaskUpdate>,
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
            && let Ok(mut operations) = inner.operations.lock()
            && let Some(operation) = operations.get_mut(&self.id)
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

#[async_trait::async_trait]
impl StreamSource<TaskUpdate> for ReplyUpdates {
    async fn next(&mut self) -> Result<Option<TaskUpdate>, ProviderError> {
        let update =
            self.rx.recv().await.transpose().map_err(|e| {
                ProviderError::from(ApiError::new(e.to_string()).with_retryable(false))
            })?;
        if update.as_ref().is_some_and(TaskUpdate::is_settled) {
            self.delivered.send_replace(true);
        }
        Ok(update)
    }
    fn cancel_safe(&self) -> bool {
        true
    }
    async fn close(mut self: Box<Self>, deadline: Ts) -> Closed {
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
    }
}

async fn exchange(
    inner: &Inner,
    request: Request,
    ack: &mut Option<oneshot::Sender<Result<TaskRef, ClientError>>>,
    tx: &mpsc::Sender<Result<TaskUpdate, ClientError>>,
    remote: &Mutex<Option<TaskRef>>,
    terminal: &AtomicBool,
    known: &watch::Sender<Option<TaskRef>>,
) -> Result<(), ClientError> {
    let id =
        request.id.clone().ok_or_else(|| ClientError::InvalidInput("an RPC needs an id".into()))?;
    let response = inner.post(&request).await?;
    let sse = response
        .headers()
        .get("content-type")
        .and_then(|h| h.to_str().ok())
        .is_some_and(|v| v.split(';').next().is_some_and(|v| v.trim() == "text/event-stream"));
    let mut state = ReplyState::for_request(&request);
    if !sse {
        let body = read_body(response, RESPONSE_LIMIT)
            .await
            .map_err(|e| transport_error(format!("{e:?}")))?;
        let update = if request.task_response {
            Some(state.decode_task(&body, &id)?)
        } else {
            state.decode(&body, &id)?
        };
        if let Some(update) = update {
            deliver(inner, update, ack, tx, remote, terminal, known).await?;
        }
        return state.finish();
    }
    let mut decoder = SseDecoder::new();
    let mut chunks = response.bytes_stream();
    let mut events = Vec::new();
    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.map_err(|e| transport_error(e.to_string()))?;
        decoder.feed(&chunk, &mut events).map_err(|e| transport_error(e.to_string()))?;
        for event in events.drain(..) {
            if let Some(update) = state.decode(event.data.as_bytes(), &id)? {
                deliver(inner, update, ack, tx, remote, terminal, known).await?;
            }
            if state.settled() {
                return state.finish();
            }
        }
    }
    decoder.finish().map_err(|e| transport_error(e.to_string()))?;
    state.finish()
}

async fn deliver(
    inner: &Inner,
    update: TaskUpdate,
    ack: &mut Option<oneshot::Sender<Result<TaskRef, ClientError>>>,
    tx: &mpsc::Sender<Result<TaskUpdate, ClientError>>,
    remote: &Mutex<Option<TaskRef>>,
    terminal: &AtomicBool,
    known: &watch::Sender<Option<TaskRef>>,
) -> Result<(), ClientError> {
    let task = TaskRef { id: update.task_id.clone(), context_id: update.context_id.clone() };
    if !task.id.is_empty() {
        *remote.lock().map_err(poisoned)? = Some(task.clone());
        known.send_replace(Some(task.clone()));
    }
    terminal.store(update.state.is_terminal(), Ordering::Release);
    if let Some(ack) = ack.take() {
        inner.core.lock().map_err(poisoned)?.acknowledge(&update)?;
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
    ClientError::InvalidResponse(error)
}
fn poisoned<T>(_: std::sync::PoisonError<T>) -> ClientError {
    ClientError::InvalidResponse("the A2A client state is poisoned".into())
}
