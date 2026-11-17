//! A readable restart story over the real session owner, A2A client and SQLite binding.

use super::*;
use lk_agents_abi::a2a_protocol as a2a;
use lk_agents_abi::delegation::remote::RemoteRecovery;
use lk_agents_abi::delegation::remote::{RemoteCommand, RemoteInput, RemoteTaskRef};
use lk_agents_abi::errors::HostError;
use lk_agents_abi::session::SessionCommands;
use lk_agents_abi::storage::context::ContextActivation as _;
use lk_agents_abi::storage::session::{PersistedSessionState as _, SessionBinding as _};
use lk_agents_abi::{BoxFuture, Closed};
use lk_agents_core::a2a::client::ClientError;
use lk_agents_core::delegation::remote::journal::recovery;
use lk_agents_core::request::AttemptFailure;
use lk_agents_runtime::delegation::remote::journal::Journal;
use lk_agents_runtime::net::a2a_client::{AgentClient, observer::RequestObserver};
use lk_agents_runtime::net::http::exchange::ResponsePolicy;
use lk_agents_runtime::net::http::port::{HttpExchangePort, HttpResponse};
use lk_agents_runtime::net::http::{HeaderMap, Method, Reporter, Sent, StatusCode};
use lk_agents_runtime::net::transport::HttpRequest;
use lk_agents_runtime::storage::context::store::ContextStore;
use lk_agents_runtime::storage::session_binding::SessionBinding;
use lk_agents_runtime::storage::session_state::SessionState;
use serde_json::Value;

const ENDPOINT: &str = "https://booking.test";

#[derive(Clone, Copy, Default)]
enum CancelReply {
    #[default]
    Confirmed,
    Refused,
    Lost,
}

#[derive(Default)]
struct Service {
    messages: Mutex<Vec<a2a::Message>>,
    bookings: Mutex<Vec<String>>,
    lose_initial: bool,
    direct_reply: bool,
    invalid_unary_delta: bool,
    cancel_reply: CancelReply,
    cancellations: Mutex<u32>,
    canceled: Mutex<bool>,
}

impl HttpExchangePort for Service {
    fn send<'a>(
        &'a self,
        method: Method,
        request: HttpRequest,
        _policy: ResponsePolicy,
        reporter: Reporter<(u32, Sent)>,
        attempt: u32,
    ) -> BoxFuture<'a, Result<HttpResponse, AttemptFailure>> {
        Box::pin(async move {
            reporter.report((attempt, Sent::Connected)).await;
            reporter.report((attempt, Sent::RequestWritten)).await;
            let payload = if method == Method::GET {
                json!({"name":"booking","description":"Book a seat","version":"1",
                    "supportedInterfaces":[{"url":format!("{ENDPOINT}/rpc"),"protocolBinding":"JSONRPC","protocolVersion":"1.0"}],
                    "capabilities":{"streaming":!self.invalid_unary_delta},"defaultInputModes":["text/plain"],"defaultOutputModes":["text/plain"],"skills":[]})
            } else {
                let rpc: a2a::JsonRpcRequest = serde_json::from_slice(&request.body).unwrap();
                match rpc.method.as_str() {
                    a2a::methods::SEND_MESSAGE | a2a::methods::SEND_STREAMING_MESSAGE => {
                        let send: a2a::SendMessageRequest =
                            serde_json::from_value(rpc.params.unwrap()).unwrap();
                        let responding = send.message.task_id.is_some();
                        if responding {
                            let date = send
                                .message
                                .parts
                                .iter()
                                .filter_map(|part| match &part.content {
                                    a2a::PartContent::Text(text) => Some(text.as_str()),
                                    _ => None,
                                })
                                .collect();
                            self.bookings.lock().push(date);
                        }
                        self.messages.lock().push(send.message);
                        if self.invalid_unary_delta {
                            json!({"jsonrpc":"2.0","id":rpc.id,"result":{"statusUpdate":{
                                "taskId":"booking","contextId":"trip","status":{"state":"TASK_STATE_COMPLETED"}
                            }}})
                        } else if self.direct_reply {
                            json!({"jsonrpc":"2.0","id":rpc.id,"result":{"message":{
                                "messageId":"receipt","role":"ROLE_AGENT","parts":[{"text":"Booked"}]
                            }}})
                        } else {
                            if responding || self.lose_initial {
                                // The service accepted the answer and performed the booking. The
                                // socket died before the caller received any acknowledgement.
                                return Err(AttemptFailure::Connection(
                                    "response lost after booking".into(),
                                ));
                            }
                            json!({"jsonrpc":"2.0","id":rpc.id,"result":{"task":{
                                "id":"booking","contextId":"trip","status":{"state":"TASK_STATE_INPUT_REQUIRED",
                                "message":{"messageId":"date","role":"ROLE_AGENT","parts":[{"text":"Which day?"}]}}
                            }}})
                        }
                    }
                    a2a::methods::GET_TASK => {
                        let booked = !self.bookings.lock().is_empty();
                        let state = if *self.canceled.lock() {
                            "TASK_STATE_CANCELED"
                        } else if booked {
                            "TASK_STATE_COMPLETED"
                        } else {
                            "TASK_STATE_INPUT_REQUIRED"
                        };
                        json!({"jsonrpc":"2.0","id":rpc.id,"result":{
                            "id":"booking","contextId":"trip","status":{"state":state},
                            "history":self.messages.lock().clone(),"artifacts":booked.then(|| json!([{"artifactId":"receipt","parts":[{"text":"Booked"}]}]))
                        }})
                    }
                    a2a::methods::CANCEL_TASK => {
                        let params = rpc.params.unwrap();
                        assert_eq!(params["id"], "booking");
                        assert_eq!(params["contextId"], "trip");
                        *self.cancellations.lock() += 1;
                        match self.cancel_reply {
                            CancelReply::Refused => json!({"jsonrpc":"2.0","id":rpc.id,
                                "error":{"code":a2a::error_code::TASK_NOT_CANCELABLE,
                                    "message":"Booking is protected","data":{"stage":"booking"}}}),
                            CancelReply::Confirmed | CancelReply::Lost => {
                                *self.canceled.lock() = true;
                                if matches!(self.cancel_reply, CancelReply::Lost) {
                                    return Err(AttemptFailure::Connection(
                                        "the cancellation was applied before its reply was lost"
                                            .into(),
                                    ));
                                }
                                json!({"jsonrpc":"2.0","id":rpc.id,"result":{
                                    "id":"booking","contextId":"trip",
                                    "status":{"state":"TASK_STATE_CANCELED"}
                                }})
                            }
                        }
                    }
                    method => panic!("unexpected operation {method}"),
                }
            };
            let mut headers = HeaderMap::new();
            headers.insert("content-type", "application/json".parse().unwrap());
            let bytes = bytes::Bytes::from(serde_json::to_vec(&payload).unwrap());
            Ok(HttpResponse {
                status: StatusCode::OK,
                headers,
                body: Box::pin(futures_util::stream::iter([Ok(bytes)])),
            })
        })
    }
}

#[tokio::test(start_paused = true)]
async fn accepted_initial_send_without_a_task_reference_remains_unknown_after_restart() {
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
    let env = environment(clock.clone());
    let database = Store::open(31, false, clock.clone()).await;
    let service = Arc::new(Service { lose_initial: true, ..Default::default() });
    let store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    let (first, updates, binding, _) = owner(&env, store.clone()).await;
    let client = AgentClient::with_transport(
        ENDPOINT.into(),
        None,
        Default::default(),
        service.clone(),
        (env.process.spawner)().unwrap(),
        clock.clone(),
    )
    .unwrap();
    assert!(
        client
            .send_observed(
                message("start", "Book a seat", false),
                None,
                None,
                None,
                &env.tracer.root(),
                Some(journal(first.clone(), "start"))
            )
            .await
            .is_err()
    );
    client.aclose().await;
    assert_eq!(
        Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
        Closed::Clean
    );
    binding.close();
    store.finish_session(false).await.unwrap();
    drop(first);

    let fresh_store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    let (second, updates, binding, view) = owner(&env, fresh_store.clone()).await;
    let snapshot = view._snapshot_data().unwrap().unwrap();
    let remote = &snapshot.remote_agents[ENDPOINT];
    assert_eq!(recovery(remote, "start").unwrap(), RemoteRecovery::Uncertain);
    assert!(remote.tasks.is_empty(), "unknown task identity is not fabricated");
    assert_eq!(remote.events.len(), 2, "only the request and observed failure are journaled");
    assert!(
        resume(&env, service.clone(), second.clone()).await.is_err(),
        "an uncertain original send cannot become new work after restart"
    );
    assert_eq!(service.messages.lock().len(), 1, "the accepted request was never resent");
    assert_eq!(
        Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
        Closed::Clean
    );
    binding.close();
    fresh_store.finish_session(false).await.unwrap();
    drop(second);
    assert!(!database.held(PARTITION_KEY).await);
}

#[tokio::test(start_paused = true)]
async fn a_fresh_owner_returns_an_observed_message_reply_without_another_business_send() {
    use lk_agents_abi::a2a::remote::RemoteEvent;
    use lk_agents_abi::llm::tool_executor::ToolOutcome;

    let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
    let env = environment(clock.clone());
    let database = Store::open(73, false, clock.clone()).await;
    let service = Arc::new(Service { direct_reply: true, ..Default::default() });
    let store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    let (first, updates, binding, _) = owner(&env, store.clone()).await;
    assert!(matches!(resume(&env, service.clone(), first.clone()).await.unwrap(),
        RemoteEvent::Outcome { outcome: ToolOutcome::Value { text, .. } } if text == "Booked"));
    assert_eq!(
        Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
        Closed::Clean
    );
    binding.close();
    store.finish_session(false).await.unwrap();
    drop(first);

    let store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    let (second, updates, binding, view) = owner(&env, store.clone()).await;
    let before = view._snapshot_data().unwrap().unwrap().remote_agents;
    let mut another_call = origin();
    another_call.call_id = "another-local-call".into();
    assert_eq!(
        second.recover(ENDPOINT, ENDPOINT, &another_call).await.unwrap(),
        None,
        "new work does not consume a different local call's receipt"
    );
    assert!(second.recover(ENDPOINT, "https://another.test", &origin()).await.is_err());
    assert!(matches!(resume(&env, service.clone(), second.clone()).await.unwrap(),
        RemoteEvent::Outcome { outcome: ToolOutcome::Value { text, .. } } if text == "Booked"));
    assert_eq!(service.messages.lock().len(), 1, "the original business message was not resent");
    assert_eq!(
        view._snapshot_data().unwrap().unwrap().remote_agents,
        before,
        "reading an accepted reply does not journal a second exchange"
    );
    assert_eq!(
        Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
        Closed::Clean
    );
    binding.close();
    store.finish_session(false).await.unwrap();
    assert!(!database.held(PARTITION_KEY).await);
}

#[tokio::test(start_paused = true)]
async fn a_unary_send_cannot_accept_a_stream_delta_without_a_conversation_observer() {
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
    let env = environment(clock.clone());
    let service = Arc::new(Service { invalid_unary_delta: true, ..Default::default() });
    let client = AgentClient::with_transport(
        ENDPOINT.into(),
        None,
        Default::default(),
        service.clone(),
        (env.process.spawner)().unwrap(),
        clock.clone(),
    )
    .unwrap();
    let sent = client
        .send(message("start", "Book a seat", false), None, None, None, &env.tracer.root())
        .await;
    assert!(
        sent.is_err(),
        "SendMessage accepts an SDK Task or Message, not a streaming status event"
    );
    client.aclose().await;
    assert_eq!(service.messages.lock().len(), 1, "a protocol error cannot repeat business work");
}

#[tokio::test(start_paused = true)]
async fn a_wrong_operation_response_remains_in_the_journal_without_becoming_task_state() {
    use lk_agents_abi::delegation::remote::{A2AJournalPayload, A2AResponse};
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
    let env = environment(clock.clone());
    let database = Store::open(79, false, clock.clone()).await;
    let service = Arc::new(Service { invalid_unary_delta: true, ..Default::default() });
    let store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    let (runner, updates, binding, view) = owner(&env, store.clone()).await;
    assert!(resume(&env, service.clone(), runner.clone()).await.is_err());
    let snapshot = view._snapshot_data().unwrap().unwrap();
    let remote = &snapshot.remote_agents[ENDPOINT];
    assert!(remote.tasks.is_empty(), "a rejected response cannot fabricate a completed booking");
    assert!(
        remote.events.iter().any(|entry| matches!(&entry.payload,
        A2AJournalPayload::Response(response) if matches!(response.as_ref(),
            A2AResponse::Stream(a2a::StreamResponse::StatusUpdate(_))))),
        "the original typed protocol content remains available for diagnosis"
    );
    assert_eq!(remote.events.len(), 3, "one request, its refused response and its observed error");
    assert_eq!(service.messages.lock().len(), 1);
    assert_eq!(
        Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
        Closed::Clean
    );
    binding.close();
    store.finish_session(false).await.unwrap();
    assert!(!database.held(PARTITION_KEY).await);
}

fn message(id: &str, text: &str, task: bool) -> a2a::Message {
    serde_json::from_value(json!({"messageId":id,"role":"ROLE_USER","parts":[{"text":text}],
        "contextId":task.then_some("trip"),"taskId":task.then_some("booking")
    }))
    .unwrap()
}

fn environment(clock: Arc<dyn Clock>) -> Arc<Env> {
    super::environment(clock, vec![(Callback::LlmOpen, Fault::Hang)])
}

async fn owner(
    env: &Arc<Env>,
    store: ContextStore,
) -> (
    Arc<lk_agents_runtime::text_session::Runner>,
    lk_agents_runtime::text_session::Updates,
    Arc<SessionBinding>,
    SessionState,
) {
    store.begin_session().await.unwrap();
    let view = SessionState::new(Some(store.load_session_snapshot().await.unwrap())).unwrap();
    let id = env.add_run(RunKind::Session);
    let fakes = RunFakes::new(env.clone(), id, None);
    let agent = TextSessionAgent {
        definition: Env::agent(),
        llm: Arc::new(FakeLlm(fakes.clone())),
        host: Arc::new(FakeHost(fakes)),
        output_type_explicit: false,
        agent_tools: BTreeMap::new(),
        host_context: HostRef::new(0, &env.refs),
        tools_host: None,
    };
    let binding = Arc::new(SessionBinding::new(store, view.clone()).unwrap());
    let runner = Arc::new(
        lk_agents_runtime::text_session::Runner::new(
            &env.process,
            "desk".into(),
            vec![agent],
            4,
            None,
        )
        .unwrap(),
    );
    let updates = runner
        .run(TextSessionRequest {
            session_sink: Some(binding.clone()),
            session_id: "conversation".into(),
            id_seed: Some(23),
            execution_id: format!("exec_{id}"),
            task_id: None,
            message: ChatItem::Message(env.message(&format!("message_{id}"))),
            delegated: false,
            conn_options: Default::default(),
        })
        .unwrap();
    (runner, updates, binding, view)
}

fn origin() -> lk_agents_abi::session::ToolCallRef {
    lk_agents_abi::session::ToolCallRef { agent_id: "root".into(), call_id: "book-seat".into() }
}

#[derive(Default)]
struct Presenter {
    answer: Option<String>,
    lose_write_ack: Option<Arc<Store>>,
}

impl lk_agents_abi::a2a::remote::DelegationHost for Presenter {
    fn progress<'a>(
        &'a self,
        _: &lk_agents_abi::telemetry::Cx,
        _: &'a lk_agents_abi::invoke::CallContext,
        _: Option<String>,
        _: serde_json::Map<String, Value>,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn delegated<'a>(
        &'a self,
        _: &lk_agents_abi::telemetry::Cx,
        _: &'a lk_agents_abi::invoke::CallContext,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn ask<'a>(
        &'a self,
        _: &lk_agents_abi::telemetry::Cx,
        _: &'a lk_agents_abi::invoke::CallContext,
        question: String,
        _: bool,
    ) -> BoxFuture<'a, Result<String, HostError>> {
        Box::pin(async move {
            assert_eq!(question, "Which day?");
            if let Some(database) = &self.lose_write_ack {
                database.lose_next_write_acknowledgement();
            }
            Ok(self.answer.clone().expect("a text invocation returns its input request"))
        })
    }
}

async fn resume(
    env: &Arc<Env>,
    service: Arc<Service>,
    owner: Arc<dyn SessionCommands>,
) -> Result<lk_agents_abi::a2a::remote::RemoteEvent, lk_agents_abi::a2a::client::AgentClientError> {
    interact(env, service, owner, Presenter::default()).await
}

async fn interact(
    env: &Arc<Env>,
    service: Arc<Service>,
    owner: Arc<dyn SessionCommands>,
    presenter: Presenter,
) -> Result<lk_agents_abi::a2a::remote::RemoteEvent, lk_agents_abi::a2a::client::AgentClientError> {
    use lk_agents_abi::a2a::remote::RemoteInteraction as _;
    use lk_agents_runtime::delegation::remote::{RemoteInteraction, RemoteInvocation};
    let interaction = RemoteInteraction::new(
        ENDPOINT.into(),
        "booking".into(),
        ChatItem::FunctionCall(lk_agents_abi::FunctionCall {
            id: "local-call".into(),
            call_id: origin().call_id,
            name: "booking".into(),
            arguments: "{}".into(),
            created_at: std::time::UNIX_EPOCH,
            extra: Default::default(),
            group_id: None,
        }),
        String::new(),
        String::new(),
        String::new(),
    )
    .unwrap();
    let client = AgentClient::with_transport(
        ENDPOINT.into(),
        None,
        Default::default(),
        service,
        (env.process.spawner)().unwrap(),
        env.clock.clone(),
    )
    .unwrap();
    let result = interaction
        .run_with_client(
            &env.process,
            client,
            RemoteInvocation {
                arguments: json!({"request":"Book a seat"}),
                structured: false,
                history: Vec::new(),
                shared: None,
                live: presenter.answer.is_some(),
                host: Arc::new(presenter),
                context: Default::default(),
                cx: env.tracer.root(),
                owner: Some(owner),
                origin: Some(origin()),
            },
        )
        .await;
    interaction.settled().await;
    result
}

fn journal(owner: Arc<dyn SessionCommands>, id: &str) -> Arc<dyn RequestObserver> {
    Arc::new(Journal::new(owner, ENDPOINT.into(), ENDPOINT.into(), id.into(), Some(origin()), None))
}

#[tokio::test(start_paused = true)]
async fn a_fresh_owner_dispatches_an_unsent_intent_once_with_its_original_message() {
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
    let env = environment(clock.clone());
    let database = Store::open(47, false, clock.clone()).await;
    let service = Arc::new(Service::default());
    let store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    let (first, updates, binding, _) = owner(&env, store.clone()).await;
    first
        .remote(RemoteCommand {
            remote_id: ENDPOINT.into(),
            endpoint: ENDPOINT.into(),
            input: RemoteInput::Prepare {
                request_id: "start".into(),
                request: lk_agents_abi::delegation::remote::A2ARequest::Send(Box::new(
                    a2a::SendMessageRequest {
                        message: message("start", "Book a seat", false),
                        configuration: None,
                        metadata: None,
                        tenant: None,
                    },
                )),
                streaming: true,
                origin: Some(origin()),
                input_request: None,
            },
        })
        .await
        .unwrap();
    // The intent is durable, but the observer ended before its dispatch checkpoint.
    journal(first.clone(), "start")
        .ended(Some(ClientError::Protocol("the local observer ended before dispatch".into())))
        .await
        .unwrap();
    assert!(service.messages.lock().is_empty());
    assert_eq!(
        Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
        Closed::Clean
    );
    binding.close();
    store.finish_session(false).await.unwrap();
    drop(first);

    let fresh_store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    let (second, updates, binding, view) = owner(&env, fresh_store.clone()).await;
    let restored = view._snapshot_data().unwrap().unwrap();
    assert!(matches!(
        recovery(&restored.remote_agents[ENDPOINT], "start").unwrap(),
        RemoteRecovery::Dispatch { .. }
    ));
    let event = resume(&env, service.clone(), second.clone()).await.unwrap();
    assert!(
        matches!(event, lk_agents_abi::a2a::remote::RemoteEvent::InputRequired { question, .. } if question == "Which day?")
    );
    let snapshot = view._snapshot_data().unwrap().unwrap();
    let remote = &snapshot.remote_agents[ENDPOINT];
    assert_eq!(service.messages.lock().as_slice(), [message("start", "Book a seat", false)]);
    assert!(remote.requests["start"].dispatch_started);
    assert_eq!(remote.events.len(), 3, "one request, the earlier error and the observed task");
    assert_eq!(remote.tasks.len(), 1);
    assert_eq!(
        Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
        Closed::Clean
    );
    binding.close();
    fresh_store.finish_session(false).await.unwrap();
    assert!(!database.held(PARTITION_KEY).await);
}

#[tokio::test(start_paused = true)]
async fn accepted_response_lost_then_fresh_owner_reads_the_same_task_without_booking_twice() {
    reconcile_a_booked_task(false).await;
}

#[tokio::test(start_paused = true)]
async fn lost_intent_write_ack_and_remote_response_do_not_book_twice_after_restart() {
    reconcile_a_booked_task(true).await;
}

#[tokio::test(start_paused = true)]
async fn an_observed_cancellation_is_checkpointed_before_its_receipt() {
    reconcile_cancellation(CancelReply::Confirmed).await;
}

#[tokio::test(start_paused = true)]
async fn a_lost_cancellation_reply_is_reconciled_after_restart_without_canceling_twice() {
    reconcile_cancellation(CancelReply::Lost).await;
}

#[tokio::test(start_paused = true)]
async fn a_refused_cancellation_retains_its_protocol_error_and_does_not_invent_success() {
    reconcile_cancellation(CancelReply::Refused).await;
}

async fn reconcile_cancellation(cancel_reply: CancelReply) {
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
    let env = environment(clock.clone());
    let database = Store::open(59, false, clock.clone()).await;
    let service = Arc::new(Service { cancel_reply, ..Default::default() });
    let store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    let (first, updates, binding, view) = owner(&env, store.clone()).await;
    let client = AgentClient::with_transport(
        ENDPOINT.into(),
        None,
        Default::default(),
        service.clone(),
        (env.process.spawner)().unwrap(),
        clock.clone(),
    )
    .unwrap();
    let reply = client
        .send_observed(
            message("start", "Book a seat", false),
            None,
            None,
            None,
            &env.tracer.root(),
            Some(journal(first.clone(), "start")),
        )
        .await
        .unwrap();
    reply.aclose(clock.now().saturating_add(Duration::from_secs(1))).await;
    assert_eq!(*service.cancellations.lock(), 0, "closing observation did not cancel work");
    let task =
        lk_agents_core::a2a::rpc::TaskRef { id: "booking".into(), context_id: "trip".into() };
    let result = client
        .cancel(task.clone(), None, &env.tracer.root(), Some(journal(first.clone(), "cancel")))
        .await;
    let snapshot = view._snapshot_data().unwrap().unwrap();
    let remote = &snapshot.remote_agents[ENDPOINT];
    assert!(remote.tasks["booking"].cancel_requested);
    match cancel_reply {
        CancelReply::Confirmed => {
            assert_eq!(result.unwrap().state, a2a::TaskState::Canceled);
            assert_eq!(remote.tasks["booking"].task.status.state, a2a::TaskState::Canceled);
        }
        CancelReply::Refused => {
            assert!(matches!(result, Err(ClientError::Rpc { code, .. })
                if code == a2a::error_code::TASK_NOT_CANCELABLE));
            let error = remote
                .events
                .iter()
                .find_map(|event| match &event.payload {
                    lk_agents_abi::delegation::remote::A2AJournalPayload::Error(
                        lk_agents_abi::delegation::remote::RemoteFailure::Rpc { error },
                    ) => Some(error),
                    _ => None,
                })
                .unwrap();
            assert_eq!(error.data, Some(json!({"stage":"booking"})));
            assert_eq!(remote.tasks["booking"].task.status.state, a2a::TaskState::InputRequired);
        }
        CancelReply::Lost => {
            assert!(result.is_err());
            assert_eq!(remote.tasks["booking"].task.status.state, a2a::TaskState::InputRequired);
        }
    }
    client.aclose().await;
    assert_eq!(
        Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
        Closed::Clean
    );
    binding.close();
    store.finish_session(false).await.unwrap();
    drop(first);

    // A fresh owner reconciles the recorded task; it does not repeat the control request.
    let store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    let (second, updates, binding, view) = owner(&env, store.clone()).await;
    let restored = view._snapshot_data().unwrap().unwrap();
    assert_eq!(
        recovery(&restored.remote_agents[ENDPOINT], "cancel").unwrap(),
        RemoteRecovery::GetTask(RemoteTaskRef {
            task_id: task.id.clone(),
            context_id: task.context_id.clone()
        })
    );
    let client = AgentClient::with_transport(
        ENDPOINT.into(),
        None,
        Default::default(),
        service.clone(),
        (env.process.spawner)().unwrap(),
        clock.clone(),
    )
    .unwrap();
    let observed = client
        .get_task(task, &env.tracer.root(), Some(journal(second.clone(), "reconcile")))
        .await
        .unwrap();
    let expected = match cancel_reply {
        CancelReply::Confirmed | CancelReply::Lost => a2a::TaskState::Canceled,
        CancelReply::Refused => a2a::TaskState::InputRequired,
    };
    assert_eq!(observed.status.state, expected);
    let snapshot = view._snapshot_data().unwrap().unwrap();
    assert_eq!(snapshot.remote_agents[ENDPOINT].tasks["booking"].task.status.state, expected);
    assert_eq!(*service.cancellations.lock(), 1, "recovery did not repeat cancellation");
    assert_eq!(service.messages.lock().len(), 1, "recovery did not repeat the business request");
    assert!(service.bookings.lock().is_empty());
    client.aclose().await;
    assert_eq!(
        Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
        Closed::Clean
    );
    binding.close();
    store.finish_session(false).await.unwrap();
    assert!(!database.held(PARTITION_KEY).await);
}

async fn reconcile_a_booked_task(lose_intent_write_ack: bool) {
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
    let env = environment(clock.clone());
    let database = Arc::new(Store::open(23, false, clock.clone()).await);
    let service = Arc::new(Service::default());
    let store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    let (first, updates, binding, view) = owner(&env, store.clone()).await;
    // The real executor asks and sends the person's response. No test command reserves the
    // input occurrence: its committed observation must carry that correlation through.
    assert!(
        interact(
            &env,
            service.clone(),
            first.clone(),
            Presenter {
                answer: Some("Friday".into()),
                lose_write_ack: lose_intent_write_ack.then(|| database.clone()),
            }
        )
        .await
        .is_err()
    );
    let snapshot = view._snapshot_data().unwrap().unwrap();
    let remote = &snapshot.remote_agents[ENDPOINT];
    let (answer_request, response) = remote
        .requests
        .iter()
        .find(|(_, request)| request.input_request.is_some())
        .expect("the actual send reserved the input occurrence before dispatch");
    let reference = response.input_request.as_ref().unwrap();
    let (remote_id, task, occurrence) = reference.remote().unwrap();
    assert_eq!(remote_id, ENDPOINT);
    assert_eq!(*task, RemoteTaskRef { context_id: "trip".into(), task_id: "booking".into() });
    let answer_request = answer_request.clone();
    let occurrence = occurrence.to_owned();
    let input = &remote.tasks["booking"].input_requests[&occurrence];
    assert_eq!(input.state, lk_agents_abi::session::InputRequestState::Responding);
    let answer_message =
        input.response_message_id.clone().expect("the outgoing response is retained");
    assert_eq!(service.messages.lock()[1].message_id, answer_message);
    if lose_intent_write_ack {
        assert!(
            database.faulted(),
            "SQLite committed the intent before losing its acknowledgement"
        );
    }
    assert_eq!(
        Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
        Closed::Clean
    );
    binding.close();
    store.finish_session(false).await.unwrap();
    drop(first);
    assert!(!database.held(PARTITION_KEY).await);

    let fresh_store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    let (second, updates, binding, view) = owner(&env, fresh_store.clone()).await;
    let restored = view._snapshot_data().unwrap().unwrap();
    let remote = &restored.remote_agents[ENDPOINT];
    assert!(matches!(recovery(remote, &answer_request).unwrap(), RemoteRecovery::GetTask(_)));
    assert_eq!(second.default_context(ENDPOINT, ENDPOINT).await.unwrap().as_deref(), Some("trip"));
    assert!(second.default_context(ENDPOINT, "https://another.test").await.is_err());
    assert_eq!(
        second.default_context("another-remote", "https://another.test").await.unwrap(),
        None
    );
    assert_eq!(
        view._snapshot_data().unwrap().unwrap().remote_agents,
        restored.remote_agents,
        "reading the saved context does not append communication or create remote state"
    );
    assert_eq!(
        remote.tasks["booking"].input_requests[&occurrence].response_message_id.as_deref(),
        Some(answer_message.as_str())
    );
    let event = resume(&env, service.clone(), second.clone()).await.unwrap();
    assert!(matches!(event, lk_agents_abi::a2a::remote::RemoteEvent::Outcome {
        outcome: lk_agents_abi::llm::tool_executor::ToolOutcome::Value { text, .. }
    } if text == "Booked"));
    let final_state = view._snapshot_data().unwrap().unwrap();
    let remote = &final_state.remote_agents[ENDPOINT];
    assert_eq!(
        remote.tasks["booking"].input_requests[&occurrence].state,
        lk_agents_abi::session::InputRequestState::Responded
    );
    assert_eq!(
        service.messages.lock().len(),
        2,
        "one original request and one answer, never a replay"
    );
    assert_eq!(service.bookings.lock().as_slice(), ["Friday"], "one committed business booking");
    assert_eq!(remote.events.len(), 6, "a retried checkpoint cannot append communication twice");
    assert_eq!(remote.tasks["booking"].task.artifacts.as_ref().unwrap().len(), 1);
    assert_eq!(
        Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
        Closed::Clean
    );
    binding.close();
    fresh_store.finish_session(false).await.unwrap();
    assert!(!database.held(PARTITION_KEY).await);
}
