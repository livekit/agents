pub mod serve;
pub(crate) mod stream;

use crate::a2a::Turn;
use crate::telemetry;
use a2a::JsonRpcId;
use lk_agents_abi::Spawner;
use lk_agents_abi::a2a::auth::{AuthOperation, AuthPrincipal};
use lk_agents_abi::a2a::card::EndpointDescriptor;
use lk_agents_abi::a2a::door::{
    A2ADoor, DoorConfig, DoorError, DoorFrame, TextEndpoint, TextServing,
};
use lk_agents_abi::a2a::turn::TurnHost;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::storage::{Storage as _, StorageError, StorageNamespace};
use lk_agents_abi::worker::http::EndpointExchange;
use lk_agents_core::a2a::auth::{authorize, first_message_scope};
use lk_agents_core::a2a::rpc::{self, Call, Request, TaskRef};
use lk_agents_core::a2a::wire;
use lk_agents_net::clock::unix_now_ms;
use lk_agents_net::env::{ProcessEnv, ProcessError};
use lk_agents_net::task::TaskSet;
use lk_agents_storage::storage::Namespace;
use lk_agents_storage::thread::db::TaskQuery;
use lk_agents_storage::thread::schema::TaskRow;
use lk_agents_storage::{Thread, shortuuid};
use parking_lot::Mutex as BudgetMutex;
use std::any::Any;
use std::sync::Arc;
use stream::Sender;
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;

// Dropping a response's observer leaves accepted work owned by the door.

const RESPONSE_CAPACITY: usize = 64;

/// What a sent message's turn runs its handler in.
struct TurnCall {
    host_context: Option<lk_agents_abi::HostRef>,
    span: Option<telemetry::SpanContext>,
}
/// The transport's database frame ceiling also bounds one decoded door payload.
pub const MAX_PAYLOAD_BYTES: usize = 16 << 20;

pub struct Door {
    inner: Arc<Inner>,
    /// What the card publishes; its URL is the request's.
    endpoint: EndpointDescriptor,
    /// Accepted operations, which `aclose` waits out.
    tasks: Arc<TaskSet>,
    spawner: Arc<dyn Spawner>,
    admission: Admission,
}

struct Inner {
    server: Option<std::sync::Weak<dyn lk_agents_abi::worker::server_host::ServerConfiguration>>,
    /// The host's namespace, whose backend opens on first use.
    namespace: Arc<Namespace>,
    endpoint_name: String,
    partition_key: String,
    holder: String,
    host: Arc<dyn TurnHost>,
    options: lk_agents_core::a2a::TurnOptions,
    /// The namespace's A2A work, which counts this door's operations too.
    owner: Arc<TaskSet>,
    clock: Arc<dyn lk_agents_abi::Clock>,
    spawner: fn() -> Result<Arc<dyn Spawner>, ProcessError>,
    tracer: telemetry::Tracer,
}

impl A2ADoor for Door {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        storage: Arc<dyn StorageNamespace>,
        config: DoorConfig,
        host: Arc<dyn TurnHost>,
        server: Option<Arc<dyn lk_agents_abi::worker::server_host::ServerConfiguration>>,
    ) -> Result<Self, StorageError> {
        let storage: Arc<dyn Any + Send + Sync> = storage;
        let namespace = storage.downcast::<Namespace>().map_err(|_| {
            StorageError::Invalid("an A2A door needs this process's Storage".into())
        })?;
        let spawner = (env.spawner)()
            .map_err(|error| StorageError::Unavailable(format!("A2A runtime: {error}")))?;
        lk_agents_core::a2a::card::agent_card(config.endpoint.clone())
            .map_err(|e| StorageError::Invalid(e.to_string()))?;
        if config.partition_key.is_empty() {
            return Err(StorageError::Invalid("an A2A door requires a partition_key".into()));
        }
        Ok(Self {
            inner: Arc::new(Inner {
                server: server.as_ref().map(Arc::downgrade),
                owner: namespace.a2a_tasks(),
                namespace,
                endpoint_name: config.endpoint.name.clone(),
                partition_key: config.partition_key,
                holder: config.holder,
                host,
                options: config.options,
                clock: env.clock.clone(),
                spawner: env.spawner,
                tracer: env.tracer.clone(),
            }),
            endpoint: config.endpoint,
            tasks: Arc::default(),
            spawner,
            admission: Admission::default(),
        })
    }

    fn max_payload_bytes(&self) -> usize {
        MAX_PAYLOAD_BYTES
    }

    async fn aclose(&self) {
        let clock = &*self.inner.clock;
        let deadline = clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        if !self.tasks.wait_empty(clock, Some(deadline)).await {
            log::error!(target: lk_agents_abi::LOG_TARGET, "A2A door cleanup missed its deadline; runtime retains accepted operations");
        }
    }
}

impl TextEndpoint for Door {
    fn name(&self) -> &str {
        &self.endpoint.name
    }
    fn recovery_handler(
        &self,
        location: &str,
        project: &str,
        partition_key: &str,
    ) -> Option<Arc<dyn TurnHost>> {
        (self.inner.namespace.location() == location
            && self.inner.namespace.project_id() == project
            && self.inner.partition_key == partition_key)
            .then(|| self.inner.host.clone())
    }

    fn serve(
        &self,
        cx: &telemetry::Cx,
        exchange: Arc<dyn EndpointExchange>,
        serving: TextServing,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
        let cx = cx.clone();
        Box::pin(async move {
            let exchange: Arc<dyn Any + Send + Sync> = exchange;
            let exchange = exchange
                .downcast::<lk_agents_worker::http::Exchange>()
                .map_err(|_| HostError::new("expected the worker's HTTP exchange"))?;
            serve::serve(self, &*exchange, serving, cx).await
        })
    }
}

impl Door {
    /// One JSON-RPC request; `body_size` is what the request carried, past a body cut at the
    /// payload budget. Its `a2a_request` opens under `parent`, the exchange's `http_request`, or
    /// at the door's root.
    #[allow(clippy::too_many_arguments, reason = "one request's exchange")]
    pub async fn dispatch(
        &self,
        body: Vec<u8>,
        principal: AuthPrincipal,
        version: Option<String>,
        _offset: Option<String>,
        body_size: Option<usize>,
        host_context: Option<lk_agents_abi::HostRef>,
        parent: Option<telemetry::Cx>,
    ) -> Response {
        let body_size = body_size.unwrap_or(body.len()).max(body.len());
        let admission = self.admission.try_exchange(body_size);
        let decoded = rpc::admit(
            &body,
            body_size,
            MAX_PAYLOAD_BYTES,
            admission.is_some(),
            version.as_deref(),
        );
        let streaming = decoded.as_ref().is_ok_and(Request::is_streaming);
        let (sender, receiver) = lk_agents_net::outbox::outbox(RESPONSE_CAPACITY);
        let cancel = CancellationToken::new();
        let request = match decoded {
            Ok(request) => request,
            Err(refusal) => {
                let sent =
                    sender.try_send(Ok(wire::frame(rpc::failure(refusal.id, refusal.error), None)));
                debug_assert!(sent, "a new response queue has room");
                return Response { streaming, receiver: Mutex::new(receiver), cancel };
            }
        };
        let inner = self.inner.clone();
        let ownership = self.inner.owner.track("a2a operation");
        self.tasks.spawn(&*self.spawner, "a2a operation", async move {
            let _ownership = ownership;
            let _admission = admission;
            let id = request.id.clone();
            if let Err(error) =
                inner.execute(request, principal, sender.clone(), host_context, parent).await
                && !sender.try_send(match error {
                    DoorError::Storage(error) => Ok(rpc::storage_failure(id, &error)),
                    auth => Err(auth),
                })
            {
                log::warn!(target: lk_agents_abi::LOG_TARGET, capacity = RESPONSE_CAPACITY; "A2A response queue full while reporting a request failure");
            }
        });
        Response { streaming, receiver: Mutex::new(receiver), cancel }
    }
}

pub struct Response {
    streaming: bool,
    receiver: Mutex<mpsc::Receiver<Result<DoorFrame, DoorError>>>,
    cancel: CancellationToken,
}

impl Response {
    pub fn streaming(&self) -> bool {
        self.streaming
    }
    pub async fn next(&self) -> Result<Option<DoorFrame>, DoorError> {
        let mut receiver = self.receiver.lock().await;
        tokio::select! {
            _ = self.cancel.cancelled() => Ok(None),
            item = receiver.recv() => item.transpose(),
        }
    }
    pub async fn aclose(&self) {
        self.cancel.cancel();
        self.receiver.lock().await.close();
    }
}

impl Drop for Response {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

impl Inner {
    async fn thread(
        &self,
        principal: &AuthPrincipal,
        id: &str,
        operation: AuthOperation,
    ) -> Result<Thread, DoorError> {
        let storage = self.namespace.storage().await?;
        let info = storage.get_thread(id).await?;
        authorize(principal, operation, id, Some(&info.scope))?;
        Ok(storage.open(id).await?)
    }

    async fn execute(
        &self,
        request: Request,
        principal: AuthPrincipal,
        sender: Sender,
        host_context: Option<lk_agents_abi::HostRef>,
        parent: Option<telemetry::Cx>,
    ) -> Result<(), DoorError> {
        let id = request.id;
        let frame = match request.call {
            Call::Send { message, history_length, streaming } => {
                let parent = parent.unwrap_or_else(|| self.tracer.root());
                let span = telemetry::start(
                    &parent,
                    telemetry::A2aRequest {
                        operation: telemetry::GenAiOperation::InvokeAgent,
                        agent_name: self.endpoint_name.clone(),
                        task_id: message.task_id.clone().unwrap_or_default(),
                    },
                );
                let call = TurnCall { host_context, span: span.cx().span() };
                let sent =
                    self.send(id, &principal, message, history_length, streaming, &sender, call);
                let sent = sent.await;
                telemetry::end(span, sent.as_ref().map(|_| ()));
                let frames = sent?;
                for frame in frames {
                    sender.send(Ok(frame)).await;
                }
                return Ok(());
            }
            Call::Subscribe { .. } => rpc::subscription_unsupported(id),
            Call::Get { task, history_length } => {
                let thread = self.thread(&principal, &task.context_id, AuthOperation::Read).await?;
                let row = thread.task(&task.id).await;
                thread.close().await;
                rpc::got(id, &task, row?.as_ref(), history_length)
            }
            Call::List(query) => {
                let thread =
                    self.thread(&principal, &query.context_id, AuthOperation::Read).await?;
                let page = thread
                    .list_tasks(&TaskQuery {
                        status: query.status,
                        updated_after: query.updated_after,
                        page_size: query.page_size,
                        page_token: query.page_token.clone(),
                        ..TaskQuery::default()
                    })
                    .await;
                thread.close().await;
                let page = page?;
                let (token, size, total) = (page.next_page_token, page.page_size, page.total_size);
                rpc::listed(id, &query, &page.tasks, token, size, total)
            }
            Call::Cancel { task, reason } => {
                let thread =
                    self.thread(&principal, &task.context_id, AuthOperation::Write).await?;
                let after = self.cancel(&thread, &task, reason).await;
                thread.close().await;
                rpc::canceled(id, &task, after?.as_ref())
            }
        };
        sender.send(Ok(frame)).await;
        Ok(())
    }

    /// The row after a client's cancel; a terminal row stays as it is.
    async fn cancel(
        &self,
        thread: &Thread,
        task: &TaskRef,
        reason: Option<String>,
    ) -> Result<Option<TaskRow>, StorageError> {
        let Some(row) = thread.task(&task.id).await? else { return Ok(None) };
        match rpc::cancellation(&row, reason) {
            Some(metadata) => {
                thread.partition(&row.partition_key).cancel_task(&task.id, metadata).await
            }
            None => Ok(Some(row)),
        }
    }

    /// The thread a message is for: the one it names, else a new one in the grant's scope;
    /// `None` when the named thread does not exist.
    async fn context(
        &self,
        principal: &AuthPrincipal,
        context: Option<&str>,
    ) -> Result<Option<String>, DoorError> {
        let (id, scope) = match context {
            None => {
                let request = livekit_protocol::agent::agent_thread::CreateRequest {
                    scope: first_message_scope(principal)?.to_owned(),
                    ..Default::default()
                };
                let info = self.namespace.storage().await?.create_thread(request).await?;
                (info.thread_id, Some(info.scope))
            }
            Some(context) => match self.namespace.storage().await?.get_thread(context).await {
                Ok(info) => (context.to_owned(), Some(info.scope)),
                Err(StorageError::NotFound(_)) => (context.to_owned(), None),
                Err(error) => return Err(error.into()),
            },
        };
        authorize(principal, AuthOperation::Write, &id, scope.as_deref())?;
        Ok(scope.is_some().then_some(id))
    }

    #[allow(clippy::too_many_arguments, reason = "one request's exchange")]
    async fn send(
        &self,
        id: JsonRpcId,
        principal: &AuthPrincipal,
        mut message: a2a::Message,
        history: Option<usize>,
        streaming: bool,
        sender: &Sender,
        call: TurnCall,
    ) -> Result<Vec<DoorFrame>, DoorError> {
        let context = match self.context(principal, message.context_id.as_deref()).await? {
            Some(context) => context,
            None => {
                let unknown = message.context_id.unwrap_or_default();
                let row = rpc::unknown_thread_task(
                    &shortuuid("AST_"),
                    &self.partition_key,
                    unix_now_ms(),
                    shortuuid("msg_"),
                );
                return Ok(rpc::unpersisted(id, &row, &unknown, streaming, history));
            }
        };
        let target = lk_agents_abi::a2a::remote::DelegationPartition {
            storage_location: lk_agents_abi::storage::Storage::location(&*self.namespace),
            storage_project: lk_agents_abi::storage::Storage::project_id(&*self.namespace),
            thread_id: context.clone(),
            partition_key: self.partition_key.clone(),
        };
        let metadata = message
            .metadata
            .as_ref()
            .map(|metadata| {
                metadata.iter().map(|(key, value)| (key.clone(), value.clone())).collect()
            })
            .unwrap_or_default();
        lk_agents_core::a2a::delegation::validate_partition_delegation(&metadata, &target)?;
        message.context_id = Some(context.clone());
        let thread = self.namespace.storage().await?.open(&context).await?;
        let continuing = match message.task_id.clone() {
            None => None,
            Some(task_id) => {
                let task = TaskRef { id: task_id, context_id: context.clone() };
                let row = thread.task(&task.id).await;
                match rpc::follow_up(id.clone(), &task, row?.as_ref(), &self.partition_key) {
                    Ok(continuing) => continuing,
                    Err(refused) => {
                        thread.close().await;
                        return Ok(vec![refused]);
                    }
                }
            }
        };
        let publisher = Arc::new(stream::Publisher::new(
            id.clone(),
            context.clone(),
            continuing,
            streaming.then_some(sender.clone()),
        ));
        let turn = Turn {
            server: self.server.clone(),
            namespace: Some(self.namespace.clone()),
            endpoint: Some(self.endpoint_name.clone()),
            publisher: Some(publisher.clone()),
            host_context: call.host_context,
            span: call.span,
            resolvers: self.owner.clone(),
            ..Turn::new(
                thread.clone(),
                &self.holder,
                &self.partition_key,
                self.host.clone(),
                self.clock.clone(),
                self.spawner,
                self.options.clone(),
            )
        };
        let result = turn.run_message(message, None).await;
        thread.close().await;
        match result {
            Ok(result) => {
                publisher.finish(&result).await;
                let answer = rpc::sent(id, &result.task, &context, history);
                Ok(if streaming { Vec::new() } else { vec![answer] })
            }
            Err(error) => Err(error.into()),
        }
    }
}

#[derive(Default)]
struct Budget {
    handlers: usize,
    bytes: usize,
}

#[derive(Clone)]
struct Admission {
    budget: Arc<BudgetMutex<Budget>>,
    max_handlers: usize,
    max_bytes: usize,
}

impl Default for Admission {
    fn default() -> Self {
        Self { budget: Arc::default(), max_handlers: 256, max_bytes: 512 << 20 }
    }
}

impl Admission {
    fn try_exchange(&self, bytes: usize) -> Option<Reservation> {
        let mut budget = self.budget.lock();
        let total = budget.bytes.checked_add(bytes)?;
        if budget.handlers >= self.max_handlers || total > self.max_bytes {
            return None;
        }
        budget.handlers += 1;
        budget.bytes = total;
        Some(Reservation { budget: self.budget.clone(), bytes })
    }
}

struct Reservation {
    budget: Arc<BudgetMutex<Budget>>,
    bytes: usize,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let mut budget = self.budget.lock();
        budget.bytes -= self.bytes;
        budget.handlers -= 1;
    }
}

#[cfg(test)]
#[path = "door_tests.rs"]
mod tests;
