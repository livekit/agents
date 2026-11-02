//! The native JSON-RPC door against the shared backend contract.

use std::sync::Arc;
use std::time::Duration;

use a2a::{JsonRpcResponse, StreamResponse};
use lk_agents_abi::a2a::auth::{AuthGrant, AuthPrincipal, PrincipalKind};
use lk_agents_abi::a2a::card::EndpointDescriptor;
use lk_agents_abi::a2a::door::{A2ADoor as _, A2AResponse as _, DoorConfig, DoorError};
use lk_agents_abi::a2a::options::{DebounceSpec, Policy, PolicySpec, TurnOptions};
use lk_agents_abi::a2a::turn::{HandlerOutput, TurnContext, TurnEvent, TurnHost, Verdict};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{Closed, StreamInput, StreamOutput, Ts};
use lk_agents_runtime::a2a::door::Door;
use lk_agents_storage::testing::{Backend, Store};
use serde_json::{Value, json};

struct Host;
struct Sink;
struct Events(std::collections::VecDeque<TurnEvent>);

impl StreamInput<Verdict> for Sink {
    fn push(&self, _: Verdict) -> Result<(), ProviderError> {
        Ok(())
    }
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn end_input(&self) {}
}

impl StreamOutput<TurnEvent> for Events {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<TurnEvent>, ProviderError>> {
        Box::pin(async move { Ok(self.0.pop_front()) })
    }
    fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<TurnEvent> for Events {}

impl TurnHost for Host {
    fn handle<'a>(
        &'a self,
        ctx: &'a TurnContext,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            lk_agents_abi::DuplexStream<
                Box<dyn StreamInput<Verdict>>,
                Box<dyn StreamOutput<TurnEvent>>,
            >,
            ProviderError,
        >,
    > {
        Box::pin(async move {
            let events = [
                TurnEvent::Progress { task_id: ctx.task_id.clone(), text: "working".into() },
                TurnEvent::Token { task_id: ctx.task_id.clone(), delta: "reply".into() },
                TurnEvent::Output {
                    output: Box::new(HandlerOutput {
                        output: Some(json!("reply")),
                        ..Default::default()
                    }),
                },
            ];
            Ok(lk_agents_abi::DuplexStream::new(
                Box::new(Sink) as Box<dyn StreamInput<_>>,
                Box::new(Events(events.into())) as Box<dyn StreamOutput<_>>,
            ))
        })
    }
}

struct AskingHost;

impl TurnHost for AskingHost {
    fn handle<'a>(
        &'a self,
        ctx: &'a TurnContext,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            lk_agents_abi::DuplexStream<
                Box<dyn StreamInput<Verdict>>,
                Box<dyn StreamOutput<TurnEvent>>,
            >,
            ProviderError,
        >,
    > {
        Box::pin(async move {
            let text = ctx
                .batch
                .last()
                .and_then(|message| message.parts.first())
                .and_then(|part| match part {
                    lk_agents_abi::a2a::client::MessagePart::TextPart { text, .. } => {
                        Some(text.as_str())
                    }
                    _ => None,
                })
                .unwrap_or_default();
            let output = if text == "start" {
                HandlerOutput { input_required: Some("Your name?".into()), ..Default::default() }
            } else {
                HandlerOutput { output: Some(json!(format!("Hello {text}"))), ..Default::default() }
            };
            Ok(lk_agents_abi::DuplexStream::new(
                Box::new(Sink) as Box<dyn StreamInput<_>>,
                Box::new(Events(
                    [
                        TurnEvent::Progress {
                            task_id: ctx.task_id.clone(),
                            text: "answering".into(),
                        },
                        TurnEvent::Output { output: Box::new(output) },
                    ]
                    .into(),
                )) as Box<dyn StreamOutput<_>>,
            ))
        })
    }
}

struct BurstHost;

impl TurnHost for BurstHost {
    fn handle<'a>(
        &'a self,
        ctx: &'a TurnContext,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            lk_agents_abi::DuplexStream<
                Box<dyn StreamInput<Verdict>>,
                Box<dyn StreamOutput<TurnEvent>>,
            >,
            ProviderError,
        >,
    > {
        Box::pin(async move {
            let mut events: std::collections::VecDeque<_> = (0..100)
                .map(|n| TurnEvent::Progress {
                    task_id: ctx.task_id.clone(),
                    text: format!("step {n}"),
                })
                .collect();
            events.push_back(TurnEvent::Output {
                output: Box::new(HandlerOutput {
                    output: Some(json!("done")),
                    ..Default::default()
                }),
            });
            Ok(lk_agents_abi::DuplexStream::new(
                Box::new(Sink) as Box<dyn StreamInput<_>>,
                Box::new(Events(events)) as Box<dyn StreamOutput<_>>,
            ))
        })
    }
}

fn principal(write: bool) -> AuthPrincipal {
    AuthPrincipal {
        kind: PrincipalKind::Grant,
        grant: Some(AuthGrant {
            scope: "customer:42".into(),
            create: true,
            write,
            read: true,
            ..Default::default()
        }),
        identity: "caller".into(),
        claims: Default::default(),
        endpoint_call: false,
    }
}

fn door(store: &Store, host: Arc<dyn TurnHost>) -> Door {
    let env = lk_agents_net::env::ProcessEnv::current_runtime();
    Door::new(&env, store.namespace.clone(), config(), host).unwrap()
}

fn config() -> DoorConfig {
    DoorConfig {
        lane: "desk".into(),
        holder: "door-worker".into(),
        endpoint: EndpointDescriptor {
            name: "desk".into(),
            description: "answers".into(),
            version: "v1".into(),
            url: "http://agent/desk/v1".into(),
        },
        options: TurnOptions {
            policy: PolicySpec::All(Policy::Queue),
            debounce: DebounceSpec::All(Duration::ZERO),
            ..Default::default()
        },
    }
}

fn body(method: &str, params: Value) -> Vec<u8> {
    serde_json::to_vec(&json!({"jsonrpc": "2.0", "id": method, "method": method, "params": params}))
        .unwrap()
}

async fn create_requires_write_before_allocating_a_thread(backend: Backend) {
    let store = backend.store(Default::default()).await;
    let door = door(&store, Arc::new(Host));
    let response = door.dispatch(body("SendMessage", json!({"message": {"messageId": "m1", "role": "ROLE_USER", "parts": [{"text": "hello"}]}})), principal(false), None, None, None, None).await;
    assert!(matches!(response.next().await, Err(DoorError::Auth(_))));
    let listed = store
        .storage
        .list_threads(livekit_protocol::agent::agent_thread::ListRequest {
            scope: "customer:42".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(listed.threads.is_empty());
    door.aclose().await;
    store.close().await;
}

async fn live_streams_have_no_replay_cursor_and_subscriptions_are_unsupported(backend: Backend) {
    let store = backend.store(Default::default()).await;
    let door = door(&store, Arc::new(Host));
    let response = door.dispatch(body("SendStreamingMessage", json!({"message": {"messageId": "m1", "role": "ROLE_USER", "parts": [{"text": "hello"}]}})), principal(true), Some("1.0".into()), None, None, None).await;
    let mut events = Vec::new();
    while let Some(frame) = response.next().await.unwrap() {
        assert!(!frame.sse.lines().any(|line| line.starts_with("id: ")));
        let envelope: JsonRpcResponse = serde_json::from_str(&frame.json).unwrap();
        events.push(serde_json::from_value::<StreamResponse>(envelope.result.unwrap()).unwrap());
    }
    let StreamResponse::Task(task) = &events[0] else { panic!("task first") };
    let StreamResponse::StatusUpdate(status) = events.last().unwrap() else {
        panic!("status last")
    };
    assert_eq!(status.status.state, a2a::TaskState::Completed);
    for method in ["SubscribeToTask", "tasks/resubscribe"] {
        let response = door
            .dispatch(
                body(
                    method,
                    json!({"id": task.id, "contextId": task.context_id, "offset": "old-cursor"}),
                ),
                principal(true),
                None,
                None,
                None,
                None,
            )
            .await;
        let envelope: JsonRpcResponse =
            serde_json::from_str(&response.next().await.unwrap().unwrap().json).unwrap();
        assert_eq!(envelope.error.unwrap().code, a2a::error_code::UNSUPPORTED_OPERATION);
        assert!(response.next().await.unwrap().is_none());
    }
    let response = door
        .dispatch(
            body("GetTask", json!({"id": task.id, "contextId": task.context_id})),
            principal(true),
            None,
            None,
            None,
            None,
        )
        .await;
    let envelope: JsonRpcResponse =
        serde_json::from_str(&response.next().await.unwrap().unwrap().json).unwrap();
    let current: a2a::Task = serde_json::from_value(envelope.result.unwrap()).unwrap();
    assert_eq!(current.status.state, a2a::TaskState::Completed);
    door.aclose().await;
    store.close().await;
}

async fn streaming_acceptance_precedes_the_lane_wait(backend: Backend) {
    let store = backend.store(Default::default()).await;
    let created = store
        .storage
        .create_thread(livekit_protocol::agent::agent_thread::CreateRequest {
            scope: "customer:42".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    let thread = store.storage.open(&created.thread_id, "holding-worker").await.unwrap();
    let mut held = thread
        .lane("desk")
        .open(lk_agents_storage::RunOptions::new("hold", "held-task"))
        .await
        .unwrap();
    let door = door(&store, Arc::new(Host));
    let response = Arc::new(door.dispatch(body("SendStreamingMessage", json!({"message": {"contextId": created.thread_id, "messageId": "queued", "role": "ROLE_USER", "parts": [{"text": "hello"}]}})), principal(true), None, None, None, None).await);
    let first = tokio::time::timeout(std::time::Duration::from_secs(1), response.next())
        .await
        .expect("durable acceptance must not wait for the active run")
        .unwrap()
        .unwrap();
    let ack: JsonRpcResponse = serde_json::from_str(&first.json).unwrap();
    let StreamResponse::Task(task) = serde_json::from_value(ack.result.unwrap()).unwrap() else {
        panic!("first frame must be a Task")
    };
    assert_eq!(task.status.state, a2a::TaskState::Submitted);
    assert!(thread.task(task.id.as_str()).await.unwrap().is_some());
    let waiting = response.clone();
    let waiting = tokio::spawn(async move { waiting.next().await });
    tokio::task::yield_now().await;
    tokio::time::timeout(std::time::Duration::from_secs(1), response.aclose())
        .await
        .expect("observer close must settle a pending read");
    assert!(waiting.await.unwrap().unwrap().is_none());
    held.abandon(lk_agents_storage::thread::run::RunOutcome::Canceled, "test-complete")
        .await
        .unwrap();
    door.aclose().await;
    assert_eq!(
        thread.task(task.id.as_str()).await.unwrap().unwrap().status,
        lk_agents_storage::thread::run::TaskState::Completed
    );
    thread.close().await;
    door.aclose().await;
    store.close().await;
}

async fn a_continuation_keeps_one_wire_identity(backend: Backend) {
    let store = backend.store(Default::default()).await;
    let door = door(&store, Arc::new(AskingHost));
    let first = door.dispatch(body("SendMessage", json!({"message": {"messageId": "start", "role": "ROLE_USER", "parts": [{"text": "start"}]}})), principal(true), None, None, None, None).await;
    let envelope: JsonRpcResponse =
        serde_json::from_str(&first.next().await.unwrap().unwrap().json).unwrap();
    let a2a::SendMessageResponse::Task(task) =
        serde_json::from_value(envelope.result.unwrap()).unwrap()
    else {
        panic!("task")
    };
    assert_eq!(task.status.state, a2a::TaskState::InputRequired);
    let second = door.dispatch(body("SendStreamingMessage", json!({"message": {"messageId": "answer", "contextId": task.context_id, "taskId": task.id, "role": "ROLE_USER", "parts": [{"text": "Ada"}]}})), principal(true), None, None, None, None).await;
    let mut events = Vec::new();
    while let Some(frame) = second.next().await.unwrap() {
        let envelope: JsonRpcResponse = serde_json::from_str(&frame.json).unwrap();
        events.push(serde_json::from_value::<StreamResponse>(envelope.result.unwrap()).unwrap());
    }
    let StreamResponse::Task(accepted) = &events[0] else { panic!("submitted task first") };
    assert_eq!(accepted.id, task.id);
    let statuses: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            StreamResponse::StatusUpdate(update) => Some(update),
            _ => None,
        })
        .collect();
    assert_eq!(statuses.len(), 2, "progress and final status");
    assert!(statuses.iter().all(|update| update.task_id == task.id));
    assert_eq!(statuses[1].status.state, a2a::TaskState::Completed);
    let thread = store.storage.open(&task.context_id, "reader").await.unwrap();
    let saved = thread.task(&task.id).await.unwrap().unwrap();
    assert_eq!(saved.data.history.len(), 2);
    assert!(saved.internal.is_none());
    let mut questions = thread
        .lane("desk")
        .open(lk_agents_storage::RunOptions::new("questions", "question-batch"))
        .await
        .unwrap();
    for id in ["AST_older", "AST_newer"] {
        questions.tasks.wait(id, "Which name?", Default::default()).await.unwrap();
    }
    questions.commit(Default::default()).await.unwrap();
    let older = door.dispatch(body("SendMessage", json!({"message": {"messageId": "older-answer", "contextId": task.context_id, "taskId": "AST_older", "role": "ROLE_USER", "parts": [{"text": "Grace"}]}})), principal(true), None, None, None, None).await;
    let envelope: JsonRpcResponse =
        serde_json::from_str(&older.next().await.unwrap().unwrap().json).unwrap();
    let a2a::SendMessageResponse::Task(answered) =
        serde_json::from_value(envelope.result.unwrap()).unwrap()
    else {
        panic!("task")
    };
    assert_eq!(answered.id, "AST_older");
    assert_eq!(answered.status.state, a2a::TaskState::Completed);
    assert_eq!(
        thread.task("AST_newer").await.unwrap().unwrap().status,
        lk_agents_storage::thread::run::TaskState::InputRequired
    );
    thread.close().await;
    door.aclose().await;
    store.close().await;
}

async fn a_slow_observer_detaches_without_canceling_the_task(backend: Backend) {
    let store = backend.store(Default::default()).await;
    let door = door(&store, Arc::new(BurstHost));
    let response = door.dispatch(body("SendStreamingMessage", json!({"message": {"messageId": "burst", "role": "ROLE_USER", "parts": [{"text": "hello"}]}})), principal(true), None, None, None, None).await;
    door.aclose().await;
    let mut task = None;
    let mut detached = false;
    while let Some(frame) = response.next().await.unwrap() {
        assert!(!frame.sse.lines().any(|line| line.starts_with("id: ")));
        let envelope: JsonRpcResponse = serde_json::from_str(&frame.json).unwrap();
        if let Some(error) = envelope.error {
            detached = true;
            assert!(error.message.contains("GetTask"));
        } else if let StreamResponse::Task(value) =
            serde_json::from_value(envelope.result.unwrap()).unwrap()
        {
            task = Some(value);
        }
    }
    assert!(detached);
    let task = task.unwrap();
    let thread = store.storage.open(&task.context_id, "reader").await.unwrap();
    let saved = thread.task(&task.id).await.unwrap().unwrap();
    assert_eq!(saved.status, lk_agents_storage::TaskState::Completed);
    thread.close().await;
    store.close().await;
}

lk_agents_storage::contract!(
    create_requires_write_before_allocating_a_thread,
    live_streams_have_no_replay_cursor_and_subscriptions_are_unsupported,
    streaming_acceptance_precedes_the_lane_wait,
    a_continuation_keeps_one_wire_identity,
    a_slow_observer_detaches_without_canceling_the_task
);
