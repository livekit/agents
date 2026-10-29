//! Executes decoded A2A RPCs against storage. A response observes one runtime-owned operation;
//! dropping that observer leaves accepted work and its durable stream owned by the door.

pub(crate) mod stream;

use lk_agents_abi::worker::endpoint::EndpointLimits;
use lk_agents_worker::endpoint::budget::EndpointBudget;
use std::any::Any;
use std::sync::{Arc, Mutex as StdMutex};

use a2a::JsonRpcId;
use lk_agents_abi::a2a::auth::{AuthOperation, AuthPrincipal};
use lk_agents_abi::a2a::door::{A2ADoor, A2AResponse, DoorConfig, DoorError, DoorFrame};
use lk_agents_abi::a2a::turn::TurnHost;
use lk_agents_abi::storage::{StorageError, StorageNamespace};
use lk_agents_core::a2a::auth::{authorize, first_message_scope};
use lk_agents_core::a2a::rpc::{self, Call, Request, TaskRef};
use lk_agents_core::a2a::wire;
use lk_agents_storage::db::TaskQuery;
use lk_agents_storage::host::HostStorage;
use lk_agents_storage::schema::TaskRow;
use lk_agents_storage::{Storage, Thread, now_ms, shortuuid};
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use crate::a2a::Turn;
use crate::process::Process;
use stream::Sender;

const RESPONSE_CAPACITY: usize = 64;
/// The transport's database frame ceiling also bounds one decoded door payload.
pub const MAX_PAYLOAD_BYTES: usize = 16 << 20;

pub struct Door {
    inner: Arc<Inner>,
    tasks: TaskTracker,
    runtime: tokio::runtime::Handle,
    budget: Arc<StdMutex<EndpointBudget>>,
    owner: TaskTracker,
}

/// Where a door's storage comes from: an open backend, or a host namespace opening its own on
/// first use.
enum Store {
    Open(Arc<dyn Storage>),
    Namespace(Arc<HostStorage<Process>>),
}

struct Inner {
    storage: Store,
    lane: String,
    holder: String,
    host: Arc<dyn TurnHost>,
    options: lk_agents_core::a2a::TurnOptions,
    owner: TaskTracker,
}

impl Door {
    /// A door over an open backend, on the caller's runtime.
    pub fn new(
        storage: Arc<dyn Storage>,
        config: DoorConfig,
        host: Arc<dyn TurnHost>,
        owner: TaskTracker,
    ) -> Result<Self, StorageError> {
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|error| StorageError::Unavailable(format!("A2A runtime: {error}")))?;
        Self::over(Store::Open(storage), config, host, owner, runtime)
    }

    fn over(
        storage: Store,
        config: DoorConfig,
        host: Arc<dyn TurnHost>,
        owner: TaskTracker,
        runtime: tokio::runtime::Handle,
    ) -> Result<Self, StorageError> {
        lk_agents_core::a2a::card::agent_card(config.endpoint)
            .map_err(|e| StorageError::Invalid(e.to_string()))?;
        if config.lane.is_empty() {
            return Err(StorageError::Invalid("an A2A door requires a lane".into()));
        }
        Ok(Self {
            inner: Arc::new(Inner {
                storage,
                lane: config.lane,
                holder: config.holder,
                host,
                options: config.options,
                owner: owner.clone(),
            }),
            tasks: TaskTracker::new(),
            runtime,
            budget: Arc::new(StdMutex::new(EndpointBudget::new(EndpointLimits::default()))),
            owner,
        })
    }
}

impl A2ADoor for Door {
    type Response = Response;

    fn new(
        storage: Arc<dyn StorageNamespace>,
        config: DoorConfig,
        host: Arc<dyn TurnHost>,
    ) -> Result<Self, StorageError> {
        let storage: Arc<dyn Any + Send + Sync> = storage;
        let storage = storage.downcast::<HostStorage<Process>>().map_err(|_| {
            StorageError::Invalid("an A2A door needs this process's Storage".into())
        })?;
        let runtime = crate::process::caller_handle()
            .map_err(|error| StorageError::Unavailable(format!("A2A runtime: {error}")))?;
        let owner = storage.a2a_tasks();
        Self::over(Store::Namespace(storage), config, host, owner, runtime)
    }

    fn max_payload_bytes(&self) -> usize {
        MAX_PAYLOAD_BYTES
    }

    async fn dispatch(
        &self,
        body: Vec<u8>,
        principal: AuthPrincipal,
        version: Option<String>,
        offset: Option<String>,
        body_size: Option<usize>,
        host_context: Option<lk_agents_abi::HostRef>,
    ) -> Response {
        let body_size = body_size.unwrap_or(body.len()).max(body.len());
        let admission = {
            let mut budget = lk_agents_abi::lock(&self.budget);
            if !budget.admit_handler() {
                None
            } else if !budget.admit_payload(body_size) {
                budget.release_handler();
                None
            } else {
                Some(Admission { budget: self.budget.clone(), bytes: body_size })
            }
        };
        let decoded = rpc::admit(
            &body,
            body_size,
            MAX_PAYLOAD_BYTES,
            admission.is_some(),
            version.as_deref(),
        );
        let streaming = decoded.as_ref().is_ok_and(Request::is_streaming);
        let (sender, receiver) = mpsc::channel(RESPONSE_CAPACITY);
        let cancel = CancellationToken::new();
        let request = match decoded {
            Ok(request) => request,
            Err(refusal) => {
                let _ =
                    sender.try_send(Ok(wire::frame(rpc::failure(refusal.id, refusal.error), None)));
                return Response { streaming, receiver: Mutex::new(receiver), cancel };
            }
        };
        let inner = self.inner.clone();
        let observer = cancel.clone();
        let ownership = self.owner.token();
        self.tasks.spawn_on(
            async move {
                let _ownership = ownership;
                let _admission = admission;
                let id = request.id.clone();
                if let Err(error) = inner
                    .execute(request, principal, offset, sender.clone(), observer, host_context)
                    .await
                {
                    let _ = sender.try_send(match error {
                        DoorError::Storage(error) => Ok(rpc::storage_failure(id, &error)),
                        auth => Err(auth),
                    });
                }
            },
            &self.runtime,
        );
        Response { streaming, receiver: Mutex::new(receiver), cancel }
    }

    async fn aclose(&self) {
        self.tasks.close();
        if tokio::time::timeout(lk_agents_abi::HOST_TEARDOWN_TIMEOUT, self.tasks.wait())
            .await
            .is_err()
        {
            log::error!(target: "livekit.agents", "A2A door cleanup missed its deadline; runtime retains accepted operations");
        }
    }
}

struct Admission {
    budget: Arc<StdMutex<EndpointBudget>>,
    bytes: usize,
}

impl Drop for Admission {
    fn drop(&mut self) {
        let mut budget = lk_agents_abi::lock(&self.budget);
        budget.release_payload(self.bytes);
        budget.release_handler();
    }
}

pub struct Response {
    streaming: bool,
    receiver: Mutex<mpsc::Receiver<Result<DoorFrame, DoorError>>>,
    cancel: CancellationToken,
}

impl A2AResponse for Response {
    fn streaming(&self) -> bool {
        self.streaming
    }
    async fn next(&self) -> Result<Option<DoorFrame>, DoorError> {
        let mut receiver = self.receiver.lock().await;
        tokio::select! {
            _ = self.cancel.cancelled() => Ok(None),
            item = receiver.recv() => item.transpose(),
        }
    }
    async fn aclose(&self) {
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
    async fn storage(&self) -> Result<Arc<dyn Storage>, StorageError> {
        match &self.storage {
            Store::Open(storage) => Ok(storage.clone()),
            Store::Namespace(namespace) => namespace.shared_storage().await,
        }
    }

    async fn thread(
        &self,
        principal: &AuthPrincipal,
        id: &str,
        operation: AuthOperation,
    ) -> Result<Thread, DoorError> {
        let info = self.storage().await?.get_thread(id).await?;
        authorize(principal, operation, id, Some(&info.scope))?;
        Ok(self.storage().await?.open(id, &self.holder).await?)
    }

    async fn execute(
        &self,
        request: Request,
        principal: AuthPrincipal,
        offset: Option<String>,
        sender: Sender,
        observer: CancellationToken,
        host_context: Option<lk_agents_abi::HostRef>,
    ) -> Result<(), DoorError> {
        let id = request.id;
        let frame = match request.call {
            Call::Send { message, history_length, streaming } => {
                let frames = self
                    .send(id, &principal, message, history_length, streaming, &sender, host_context)
                    .await?;
                for frame in frames {
                    let _ = sender.send(Ok(frame)).await;
                }
                return Ok(());
            }
            Call::Subscribe { task, offset: requested } => {
                return self
                    .subscribe(id, &principal, task, requested.or(offset), sender, observer)
                    .await;
            }
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
        let _ = sender.send(Ok(frame)).await;
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
            Some(metadata) => thread.lane(&row.lane).cancel_task(&task.id, metadata).await,
            None => Ok(Some(row)),
        }
    }

    /// The thread a message is for: the one it names, else a new one in the grant's scope;
    /// `Err` names a thread that does not exist.
    async fn context(
        &self,
        principal: &AuthPrincipal,
        context: Option<&str>,
    ) -> Result<Result<String, String>, DoorError> {
        let (id, scope) = match context {
            None => {
                let request = livekit_protocol::agent::agent_thread::CreateRequest {
                    scope: first_message_scope(principal)?.to_owned(),
                    ..Default::default()
                };
                let info = self.storage().await?.create_thread(request).await?;
                (info.thread_id, Some(info.scope))
            }
            Some(context) => match self.storage().await?.get_thread(context).await {
                Ok(info) => (context.to_owned(), Some(info.scope)),
                Err(StorageError::NotFound(_)) => (context.to_owned(), None),
                Err(error) => return Err(error.into()),
            },
        };
        authorize(principal, AuthOperation::Write, &id, scope.as_deref())?;
        Ok(if scope.is_some() { Ok(id) } else { Err(id) })
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
        host_context: Option<lk_agents_abi::HostRef>,
    ) -> Result<Vec<DoorFrame>, DoorError> {
        let context = match self.context(principal, message.context_id.as_deref()).await? {
            Ok(context) => context,
            Err(unknown) => {
                let row = rpc::unknown_thread_task(
                    &shortuuid("AST_"),
                    &self.lane,
                    now_ms(),
                    shortuuid("msg_"),
                );
                return Ok(rpc::unpersisted(id, &row, &unknown, streaming, history));
            }
        };
        message.context_id = Some(context.clone());
        let thread = self.storage().await?.open(&context, &self.holder).await?;
        let continuing = match message.task_id.clone() {
            None => None,
            Some(task_id) => {
                let task = TaskRef { id: task_id, context_id: context.clone() };
                let row = thread.task(&task.id).await;
                match rpc::follow_up(id.clone(), &task, row?.as_ref(), &self.lane) {
                    Ok(continuing) => continuing,
                    Err(refused) => {
                        thread.close().await;
                        return Ok(vec![refused]);
                    }
                }
            }
        };
        let streams = self.storage().await?.response_streams();
        let created = streams.create(None).await?;
        let publisher = Arc::new(stream::Publisher::new(
            streams.clone(),
            created.stream_id.clone(),
            id.clone(),
            context.clone(),
            continuing,
            streaming.then_some(sender.clone()),
        ));
        let turn = Turn::new(
            thread.clone(),
            &self.lane,
            self.host.clone(),
            crate::process::host_clock(),
            self.options.clone(),
        )
        .with_stream(created.clone())
        .with_owner(self.owner.clone())
        .with_publisher(publisher.clone());
        let turn = match host_context {
            Some(context) => turn.with_context(context),
            None => turn,
        };
        let result = turn.run_message(message, None).await;
        thread.close().await;
        match result {
            Ok(result) => {
                publisher.finish(&result).await?;
                let answer = rpc::sent(id, &result.task, &context, history);
                Ok(if streaming { Vec::new() } else { vec![answer] })
            }
            Err(error) => {
                streams.close(&created.stream_id).await?;
                Err(error.into())
            }
        }
    }

    /// The current task, then its response streams from the cursor, followed live while the
    /// task runs.
    async fn subscribe(
        &self,
        id: JsonRpcId,
        principal: &AuthPrincipal,
        task: TaskRef,
        offset: Option<String>,
        sender: Sender,
        observer: CancellationToken,
    ) -> Result<(), DoorError> {
        let thread = self.thread(principal, &task.context_id, AuthOperation::Read).await?;
        let row = thread.task(&task.id).await;
        thread.close().await;
        let Some(row) = row? else {
            let _ = sender.send(Ok(rpc::task_not_found(id, &task))).await;
            return Ok(());
        };
        // the authoritative Task comes first, even when replay has expired
        let current = a2a::StreamResponse::Task(wire::task(&row, &task.context_id, None, true));
        if sender.send(Ok(wire::success(id.clone(), current, None))).await.is_err() {
            return Ok(());
        }
        let streams = self.storage().await?.response_streams();
        for rpc::Replay { stream, after, follow } in rpc::resume(&task, &row, offset.as_deref())? {
            let mut reader = streams.read(&stream.stream_id, after.as_deref(), follow);
            loop {
                let next = tokio::select! {
                    _ = observer.cancelled() => return Ok(()),
                    next = reader.next() => next?,
                };
                let Some(bytes) = next else { break };
                let offset = reader.next_offset().into();
                let frame =
                    wire::replayed(id.clone(), &task.id, &stream.stream_id, offset, &bytes)?;
                if sender.send(Ok(frame)).await.is_err() {
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}
