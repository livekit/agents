//! A readable restart story over the real session owner, A2A client and SQLite binding.

use super::*;
use lk_agents_abi::a2a_protocol as a2a;
use lk_agents_abi::delegation::remote::{
    InputRequestRef, RemoteCommand, RemoteCommands, RemoteInput, RemoteTaskRef,
};
use lk_agents_abi::storage::context::ContextActivation as _;
use lk_agents_abi::storage::session::{PersistedSessionState as _, SessionBinding as _};
use lk_agents_abi::{BoxFuture, Closed};
use lk_agents_core::a2a::client::ClientError;
use lk_agents_core::delegation::remote::journal::{Recovery, recovery};
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
                    "capabilities":{"streaming":true},"defaultInputModes":["text/plain"],"defaultOutputModes":["text/plain"],"skills":[]})
            } else {
                let rpc: a2a::JsonRpcRequest = serde_json::from_slice(&request.body).unwrap();
                match rpc.method.as_str() {
                    a2a::methods::SEND_STREAMING_MESSAGE => {
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
    assert_eq!(recovery(remote, "start").unwrap(), Recovery::Uncertain);
    assert!(remote.tasks.is_empty(), "unknown task identity is not fabricated");
    assert_eq!(remote.events.len(), 2, "only the request and observed failure are journaled");
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

fn message(id: &str, text: &str, task: bool) -> a2a::Message {
    serde_json::from_value(json!({"messageId":id,"role":"ROLE_USER","parts":[{"text":text}],
        "contextId":task.then_some("trip"),"taskId":task.then_some("booking")
    }))
    .unwrap()
}

fn environment(clock: Arc<dyn Clock>) -> Arc<Env> {
    let scenario =
        Scenario { forced: vec![(Callback::LlmOpen, Fault::Hang)], ..Scenario::direct(23) };
    let tracer = lk_agents_telemetry::Tracer::disabled(clock.clone());
    let process = ProcessEnv {
        runtime: || Ok(tokio::runtime::Handle::current()),
        spawner: || Ok(Arc::new(lk_agents_runtime::net::task::TokioSpawner::current())),
        clock: clock.clone(),
        http: || Err(ProcessError::Start("the DST has no sockets".into())),
        tracer: tracer.clone(),
        tasks: Arc::default(),
    };
    Arc::new(Env {
        clock,
        refs: Arc::new(NoRefs),
        tracer,
        process,
        scenario,
        calm: Default::default(),
        recording: Mutex::default(),
    })
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
        lk_agents_runtime::text_session::Runner::new(&env.process, "desk".into(), vec![agent], 4)
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

fn journal(owner: Arc<dyn RemoteCommands>, id: &str) -> Arc<dyn RequestObserver> {
    Arc::new(Journal {
        owner,
        endpoint: ENDPOINT.into(),
        remote_id: ENDPOINT.into(),
        request_id: id.into(),
        origin: None,
    })
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
                origin: None,
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
    assert_eq!(recovery(&restored.remote_agents[ENDPOINT], "start").unwrap(), Recovery::Dispatch);
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
            Some(journal(second.clone(), "start")),
        )
        .await
        .unwrap();
    reply.aclose(clock.now().saturating_add(Duration::from_secs(1))).await;
    let snapshot = view._snapshot_data().unwrap().unwrap();
    let remote = &snapshot.remote_agents[ENDPOINT];
    assert_eq!(service.messages.lock().as_slice(), [message("start", "Book a seat", false)]);
    assert!(remote.requests["start"].dispatch_started);
    assert_eq!(remote.events.len(), 3, "one request, the earlier error and the observed task");
    assert_eq!(remote.tasks.len(), 1);
    client.aclose().await;
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
            assert_eq!(result.unwrap().state, lk_agents_abi::storage::TaskState::Canceled);
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
        Recovery::GetTask(RemoteTaskRef {
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
    let database = Store::open(23, false, clock.clone()).await;
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
    let snapshot = view._snapshot_data().unwrap().unwrap();
    let occurrence = snapshot.remote_agents[ENDPOINT].tasks["booking"]
        .input_requests
        .keys()
        .next()
        .unwrap()
        .clone();
    // Reserve the exact response before transmission. The production owner checkpoints this
    // command; the client then records the same immutable body and its dispatch marker.
    let input = InputRequestRef {
        task: RemoteTaskRef { context_id: "trip".into(), task_id: "booking".into() },
        occurrence_id: occurrence.clone(),
    };
    if lose_intent_write_ack {
        database.lose_next_write_acknowledgement();
    }
    first
        .remote(RemoteCommand {
            remote_id: ENDPOINT.into(),
            endpoint: ENDPOINT.into(),
            input: RemoteInput::Prepare {
                request_id: "answer".into(),
                request: lk_agents_abi::delegation::remote::A2ARequest::Send(Box::new(
                    a2a::SendMessageRequest {
                        message: message("answer", "Friday", true),
                        configuration: None,
                        metadata: None,
                        tenant: None,
                    },
                )),
                streaming: true,
                origin: None,
                input_request: Some(input),
            },
        })
        .await
        .unwrap();
    if lose_intent_write_ack {
        assert!(
            database.faulted(),
            "SQLite committed the intent before losing its acknowledgement"
        );
    }
    assert!(service.bookings.lock().is_empty(), "the durable intent alone did not book anything");
    // Use the journal's reservation on the same operation, never a second prepared request.
    let observer = journal(first.clone(), "answer");
    let answer = client
        .send_observed(
            message("answer", "Friday", true),
            None,
            None,
            None,
            &env.tracer.root(),
            Some(observer),
        )
        .await;
    assert!(answer.is_err());
    client.aclose().await;
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
    assert!(matches!(recovery(remote, "answer").unwrap(), Recovery::GetTask(_)));
    assert_eq!(
        remote.tasks["booking"].input_requests[&occurrence].response_message_id.as_deref(),
        Some("answer")
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
    let task = client
        .get_task(
            lk_agents_core::a2a::rpc::TaskRef { id: "booking".into(), context_id: "trip".into() },
            &env.tracer.root(),
            Some(journal(second.clone(), "reconcile")),
        )
        .await
        .unwrap();
    assert_eq!(task.status.state, a2a::TaskState::Completed);
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
    client.aclose().await;
    assert_eq!(
        Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
        Closed::Clean
    );
    binding.close();
    fresh_store.finish_session(false).await.unwrap();
    assert!(!database.held(PARTITION_KEY).await);
}
