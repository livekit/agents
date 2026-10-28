//! Executes decoded A2A RPCs against storage. A response observes one runtime-owned operation;
//! dropping that observer leaves accepted work and its durable stream owned by the door.

mod stream;
mod wire;

use lk_agents_abi::worker::endpoint::EndpointLimits;
use lk_agents_worker::endpoint_budget::EndpointBudget;
use std::sync::{Arc, Mutex as StdMutex};

use a2a::{JsonRpcId, error_code};
use lk_agents_abi::a2a::auth::{AuthError, AuthOperation, AuthPrincipal};
use lk_agents_abi::a2a::door::{A2ADoor, A2AResponse, DoorConfig, DoorError, DoorFrame};
use lk_agents_abi::a2a::turn::TurnHost;
use lk_agents_abi::storage::StorageError;
use lk_agents_core::a2a::auth::{allows, allows_first_message};
use lk_agents_core::a2a::rpc::{self, Call, Request, TaskRef};
use lk_agents_storage::db::TaskQuery;

use lk_agents_storage::run::{REASON_KEY, REASON_THREAD_NOT_FOUND, TaskState};
use serde_json::Value;
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use crate::a2a::Turn;
use crate::storage::{Storage, Thread, now_ms, shortuuid};

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

struct Inner {
    storage: Arc<dyn Storage>,
    lane: String,
    holder: String,
    host: Arc<dyn TurnHost>,
    options: lk_agents_core::a2a::TurnOptions,
    owner: TaskTracker,
}

impl Door {
    pub fn new(
        storage: Arc<dyn Storage>,
        config: DoorConfig,
        host: Arc<dyn TurnHost>,
        owner: TaskTracker,
    ) -> Result<Self, StorageError> {
        crate::a2a_card::agent_card(config.endpoint)
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
            runtime: tokio::runtime::Handle::try_current()
                .map_err(|error| StorageError::Unavailable(format!("A2A runtime: {error}")))?,
            budget: Arc::new(StdMutex::new(EndpointBudget::new(EndpointLimits::default()))),
            owner,
        })
    }
}

impl A2ADoor for Door {
    type Response = Response;

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
        let decoded = if body_size > MAX_PAYLOAD_BYTES {
            Err(Box::new(rpc::Refusal {
                id: JsonRpcId::Null,
                error: rpc::rpc_error(
                    error_code::INVALID_PARAMS,
                    "A2A payload exceeds the transport budget",
                ),
            }))
        } else if admission.is_none() {
            Err(Box::new(rpc::Refusal {
                id: JsonRpcId::Null,
                error: rpc::rpc_error(
                    error_code::INTERNAL_ERROR,
                    "endpoint request budget exhausted",
                ),
            }))
        } else {
            rpc::decode(&body)
        };
        let decoded = decoded.and_then(|request| {
            rpc::check_version(version.as_deref())
                .map(|()| request.clone())
                .map_err(|error| Box::new(rpc::Refusal { id: request.id, error }))
        });
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
                    match error {
                        DoorError::Auth(error) => {
                            let _ = sender.try_send(Err(DoorError::Auth(error)));
                        }
                        DoorError::Storage(error) => {
                            let code = match error {
                                StorageError::Invalid(_) => error_code::INVALID_PARAMS,
                                _ => error_code::INTERNAL_ERROR,
                            };
                            let _ = sender.try_send(Ok(wire::frame(
                                rpc::failure(id, rpc::rpc_error(code, error.to_string())),
                                None,
                            )));
                        }
                    }
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

struct Exchange {
    id: JsonRpcId,
    principal: AuthPrincipal,
    sender: mpsc::Sender<Result<DoorFrame, DoorError>>,
}

impl Inner {
    fn authorize(
        principal: &AuthPrincipal,
        operation: AuthOperation,
        id: &str,
        scope: Option<&str>,
    ) -> Result<(), AuthError> {
        let grant = principal.grant.as_ref().ok_or_else(|| {
            AuthError::Forbidden("endpoint call permission does not grant thread access".into())
        })?;
        if allows(grant, operation, id, scope) {
            Ok(())
        } else {
            Err(AuthError::Forbidden(format!("the thread grant does not allow {operation:?}")))
        }
    }

    async fn thread(
        &self,
        principal: &AuthPrincipal,
        id: &str,
        operation: AuthOperation,
    ) -> Result<Thread, DoorError> {
        let info = self.storage.get_thread(id).await?;
        Self::authorize(principal, operation, id, Some(&info.scope))?;
        Ok(self.storage.open(id, &self.holder).await?)
    }

    async fn execute(
        &self,
        request: Request,
        principal: AuthPrincipal,
        offset: Option<String>,
        sender: mpsc::Sender<Result<DoorFrame, DoorError>>,
        observer: CancellationToken,
        host_context: Option<lk_agents_abi::HostRef>,
    ) -> Result<(), DoorError> {
        match request.call {
            Call::Send { message, history_length, streaming } => {
                self.send(
                    Exchange { id: request.id, principal, sender },
                    message,
                    history_length,
                    streaming,
                    host_context,
                )
                .await
            }
            Call::Get { task, history_length } => {
                let thread = self.thread(&principal, &task.context_id, AuthOperation::Read).await?;
                let row = thread.task(&task.id).await?;
                thread.close().await;
                match row {
                    Some(row) => {
                        let _ = sender
                            .send(Ok(wire::success(
                                request.id,
                                wire::task(&row, &task.context_id, history_length, true),
                                None,
                            )))
                            .await;
                    }
                    None => Self::not_found(request.id, &task, &sender).await,
                }
                Ok(())
            }
            Call::List(query) => {
                let thread =
                    self.thread(&principal, &query.context_id, AuthOperation::Read).await?;
                let page = thread
                    .list_tasks(&TaskQuery {
                        status: query.status,
                        updated_after: query.updated_after,
                        page_size: query.page_size.unwrap_or(50),
                        page_token: query.page_token,
                    })
                    .await?;
                thread.close().await;
                let result = a2a::ListTasksResponse {
                    tasks: page
                        .tasks
                        .iter()
                        .map(|row| {
                            wire::task(
                                row,
                                &query.context_id,
                                Some(query.history_length.unwrap_or(0)),
                                query.include_artifacts,
                            )
                        })
                        .collect(),
                    next_page_token: page.next_page_token,
                    page_size: page.page_size as i32,
                    total_size: i32::try_from(page.total_size).map_err(|_| {
                        StorageError::TooLarge {
                            what: "task list".into(),
                            limit: i32::MAX as usize,
                        }
                    })?,
                };
                let _ = sender.send(Ok(wire::success(request.id, result, None))).await;
                Ok(())
            }
            Call::Cancel { task, reason } => {
                let thread =
                    self.thread(&principal, &task.context_id, AuthOperation::Write).await?;
                let row = thread.task(&task.id).await?;
                let Some(row) = row else {
                    thread.close().await;
                    Self::not_found(request.id, &task, &sender).await;
                    return Ok(());
                };
                let mut metadata = row.data.metadata.clone();
                if let Some(reason) = reason {
                    metadata.insert(REASON_KEY.into(), Value::String(reason));
                }
                let result = if row.status.is_terminal() {
                    None
                } else {
                    thread.lane(&row.lane).cancel_task(&task.id, metadata).await?
                };
                thread.close().await;
                if let Some(row) = result.filter(|r| r.status == TaskState::Canceled) {
                    let _ = sender
                        .send(Ok(wire::success(
                            request.id,
                            wire::task(&row, &task.context_id, None, true),
                            None,
                        )))
                        .await;
                } else {
                    let _ = sender
                        .send(Ok(wire::frame(
                            rpc::failure(
                                request.id,
                                rpc::rpc_error(
                                    error_code::TASK_NOT_CANCELABLE,
                                    "the task is already terminal",
                                ),
                            ),
                            None,
                        )))
                        .await;
                }
                Ok(())
            }
            Call::Subscribe { task, offset: requested } => {
                self.subscribe(
                    Exchange { id: request.id, principal, sender },
                    task,
                    requested.or(offset),
                    observer,
                )
                .await
            }
        }
    }

    async fn not_found(
        id: JsonRpcId,
        task: &TaskRef,
        sender: &mpsc::Sender<Result<DoorFrame, DoorError>>,
    ) {
        let _ = sender
            .send(Ok(wire::frame(
                rpc::failure(
                    id,
                    rpc::rpc_error(
                        error_code::TASK_NOT_FOUND,
                        format!("task {} is not in thread {}", task.id, task.context_id),
                    ),
                ),
                None,
            )))
            .await;
    }

    async fn send(
        &self,
        exchange: Exchange,
        mut message: a2a::Message,
        history: Option<usize>,
        streaming: bool,
        host_context: Option<lk_agents_abi::HostRef>,
    ) -> Result<(), DoorError> {
        let Exchange { id, principal, sender } = exchange;
        let context = if let Some(context) =
            message.context_id.as_deref().filter(|id| !id.is_empty())
        {
            match self.storage.get_thread(context).await {
                Ok(info) => {
                    Self::authorize(&principal, AuthOperation::Write, context, Some(&info.scope))?;
                    context.to_owned()
                }
                Err(StorageError::NotFound(_)) => {
                    Self::authorize(&principal, AuthOperation::Write, context, None)?;
                    let row = lk_agents_core::a2a::unpersisted_failed(
                        &shortuuid("AST_"),
                        &self.lane,
                        None,
                        REASON_THREAD_NOT_FOUND,
                        now_ms(),
                        shortuuid("msg_"),
                    );
                    if streaming {
                        let _ = sender
                            .send(Ok(wire::success(
                                id.clone(),
                                wire::submitted(&row.task_id, context, now_ms()),
                                None,
                            )))
                            .await;
                        let _ = sender
                            .send(Ok(wire::success(id, wire::final_status(&row, context), None)))
                            .await;
                    } else {
                        let _ = sender
                            .send(Ok(wire::success(
                                id,
                                a2a::SendMessageResponse::Task(wire::task(
                                    &row, context, history, true,
                                )),
                                None,
                            )))
                            .await;
                    }
                    return Ok(());
                }
                Err(error) => return Err(error.into()),
            }
        } else {
            let grant =
                principal.grant.as_ref().filter(|grant| allows_first_message(grant)).ok_or_else(
                    || {
                        AuthError::Forbidden(
                            "the first message requires scoped create and write authority".into(),
                        )
                    },
                )?;
            let info = self
                .storage
                .create_thread(livekit_protocol::agent::agent_thread::CreateRequest {
                    scope: grant.scope.clone(),
                    ..Default::default()
                })
                .await?;
            Self::authorize(&principal, AuthOperation::Write, &info.thread_id, Some(&info.scope))?;
            info.thread_id
        };
        message.context_id = Some(context.clone());
        let thread = self.storage.open(&context, &self.holder).await?;
        let mut continuing = None;
        if let Some(task) = message.task_id.as_deref() {
            let row = thread.task(task).await?;
            match row {
                None => {
                    thread.close().await;
                    Self::not_found(id, &TaskRef { id: task.into(), context_id: context }, &sender)
                        .await;
                    return Ok(());
                }
                Some(row) if row.status.is_terminal() => {
                    thread.close().await;
                    let _ = sender
                        .send(Ok(wire::frame(
                            rpc::failure(
                                id,
                                rpc::rpc_error(
                                    error_code::UNSUPPORTED_OPERATION,
                                    "a follow-up to a terminal task must start a new task",
                                ),
                            ),
                            None,
                        )))
                        .await;
                    return Ok(());
                }
                Some(row)
                    if row.lane == self.lane
                        && row.status == TaskState::InputRequired
                        && row.internal.as_ref().is_some_and(|internal| internal.ask.is_some()) =>
                {
                    continuing = Some(task.to_owned());
                }
                _ => {}
            }
        }
        let streams = self.storage.response_streams();
        let created = streams.create(None).await?;
        let task_id = shortuuid("AST_");
        let sink = Arc::new(stream::Publisher::new(
            streams.clone(),
            created.stream_id.clone(),
            id.clone(),
            streaming.then_some(sender.clone()),
        ));
        let host =
            Arc::new(stream::WireHost::new(self.host.clone(), sink.clone(), context.clone()));
        let turn = Turn::new(
            thread.clone(),
            &self.lane,
            host,
            crate::process::host_clock(),
            self.options.clone(),
        )
        .with_stream(created.clone())
        .with_owner(self.owner.clone())
        .with_acceptance(Arc::new(stream::Accepted {
            task_id: continuing,
            publisher: sink.clone(),
            context: context.clone(),
        }));
        let turn = match host_context {
            Some(context) => turn.with_context(context),
            None => turn,
        };
        let result = turn.run_message(message, Some(task_id)).await;
        thread.close().await;
        match result {
            Ok(result) => {
                sink.finish(&result, &context).await?;
                if !streaming {
                    let _ = sender
                        .send(Ok(wire::success(
                            id,
                            a2a::SendMessageResponse::Task(wire::task(
                                &result.task,
                                &context,
                                history,
                                true,
                            )),
                            None,
                        )))
                        .await;
                }
                Ok(())
            }
            Err(error) => {
                streams.close(&created.stream_id).await?;
                Err(error.into())
            }
        }
    }

    async fn subscribe(
        &self,
        exchange: Exchange,
        task: TaskRef,
        offset: Option<String>,
        observer: CancellationToken,
    ) -> Result<(), DoorError> {
        let Exchange { id, principal, sender } = exchange;
        let thread = self.thread(&principal, &task.context_id, AuthOperation::Read).await?;
        let row = thread.task(&task.id).await?;
        thread.close().await;
        let Some(row) = row else {
            Self::not_found(id, &task, &sender).await;
            return Ok(());
        };
        // Subscription starts with the authoritative Task even when replay has expired.
        if sender
            .send(Ok(wire::success(
                id.clone(),
                a2a::StreamResponse::Task(wire::task(&row, &task.context_id, None, true)),
                None,
            )))
            .await
            .is_err()
        {
            return Ok(());
        }
        let refs =
            row.internal.as_ref().map(|internal| internal.streams.as_slice()).unwrap_or_default();
        let streams = self.storage.response_streams();
        let cursor = offset.as_deref().map(stream::Cursor::decode).transpose()?;
        if let Some(cursor) = &cursor {
            if cursor.task_id != task.id || !refs.iter().any(|s| s.stream_id == cursor.stream_id) {
                return Err(StorageError::Invalid(
                    "response cursor belongs to a different task".into(),
                )
                .into());
            }
        }
        let start = cursor
            .as_ref()
            .and_then(|c| refs.iter().position(|s| s.stream_id == c.stream_id))
            .unwrap_or(0);
        for (index, stream) in refs.iter().enumerate().skip(start) {
            let after = cursor
                .as_ref()
                .filter(|c| c.stream_id == stream.stream_id)
                .map(|c| c.offset.as_str());
            let mut reader = streams.read(
                &stream.stream_id,
                after,
                !row.status.is_terminal() && index + 1 == refs.len(),
            );
            loop {
                let next = tokio::select! { _ = observer.cancelled() => return Ok(()), next = reader.next() => next? };
                let Some(bytes) = next else { break };
                let event: a2a::StreamResponse = serde_json::from_slice(&bytes)
                    .map_err(|e| StorageError::Unavailable(format!("stored A2A event: {e}")))?;
                let cursor = stream::Cursor {
                    task_id: task.id.clone(),
                    stream_id: stream.stream_id.clone(),
                    offset: reader.next_offset().into(),
                }
                .encode();
                if sender.send(Ok(wire::success(id.clone(), event, Some(&cursor)))).await.is_err() {
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}
