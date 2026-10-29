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
use lk_agents_abi::{Closed, StreamSink, StreamSource, Ts};
use lk_agents_runtime::a2a::door::Door;
use lk_agents_storage::testing::Backend;
use serde_json::{Value, json};

struct Host;
struct Sink;
struct Events(std::collections::VecDeque<TurnEvent>);

impl StreamSink<Verdict> for Sink {
    fn push(&self, _: Verdict) -> Result<(), ProviderError> {
        Ok(())
    }
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn end_input(&self) {}
}

impl StreamSource<TurnEvent> for Events {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<TurnEvent>, ProviderError>> {
        Box::pin(async move { Ok(self.0.pop_front()) })
    }
    fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}

impl TurnHost for Host {
    fn handle<'a>(
        &'a self,
        ctx: &'a TurnContext,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<(Box<dyn StreamSink<Verdict>>, Box<dyn StreamSource<TurnEvent>>), ProviderError>,
    > {
        Box::pin(async move {
            let events = [
                TurnEvent::Progress { task_id: ctx.task_id.clone(), text: "working".into() },
                TurnEvent::Token { task_id: ctx.task_id.clone(), delta: "reply".into() },
                TurnEvent::Output {
                    output: HandlerOutput { output: Some(json!("reply")), ..Default::default() },
                },
            ];
            Ok((
                Box::new(Sink) as Box<dyn StreamSink<_>>,
                Box::new(Events(events.into())) as Box<dyn StreamSource<_>>,
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
        Result<(Box<dyn StreamSink<Verdict>>, Box<dyn StreamSource<TurnEvent>>), ProviderError>,
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
            Ok((
                Box::new(Sink) as Box<dyn StreamSink<_>>,
                Box::new(Events(
                    [
                        TurnEvent::Progress {
                            task_id: ctx.task_id.clone(),
                            text: "answering".into(),
                        },
                        TurnEvent::Output { output },
                    ]
                    .into(),
                )) as Box<dyn StreamSource<_>>,
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
        Result<(Box<dyn StreamSink<Verdict>>, Box<dyn StreamSource<TurnEvent>>), ProviderError>,
    > {
        Box::pin(async move {
            let mut events: std::collections::VecDeque<_> = (0..100)
                .map(|n| TurnEvent::Progress {
                    task_id: ctx.task_id.clone(),
                    text: format!("step {n}"),
                })
                .collect();
            events.push_back(TurnEvent::Output {
                output: HandlerOutput { output: Some(json!("done")), ..Default::default() },
            });
            Ok((
                Box::new(Sink) as Box<dyn StreamSink<_>>,
                Box::new(Events(events)) as Box<dyn StreamSource<_>>,
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

fn door(storage: Arc<dyn lk_agents_storage::Storage>, host: Arc<dyn TurnHost>) -> Door {
    let clock = Arc::new(lk_agents_abi::RealClock::new());
    Door::new(storage, config(), host, Arc::default(), clock).unwrap()
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
    let door = door(store.storage.clone(), Arc::new(Host));
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

async fn subscriptions_use_task_records_and_resume_each_durable_event(backend: Backend) {
    let store = backend.store(Default::default()).await;
    let door = door(store.storage.clone(), Arc::new(Host));
    let response = door.dispatch(body("SendStreamingMessage", json!({"message": {"messageId": "m1", "role": "ROLE_USER", "parts": [{"text": "hello"}]}})), principal(true), Some("1.0".into()), None, None, None).await;
    let mut frames = Vec::new();
    while let Some(frame) = response.next().await.unwrap() {
        frames.push(frame);
    }
    let ack: JsonRpcResponse = serde_json::from_str(&frames[0].json).unwrap();
    let StreamResponse::Task(task) = serde_json::from_value(ack.result.unwrap()).unwrap() else {
        panic!("first frame must be a Task")
    };
    let context = task.context_id;
    let task_id = task.id.to_string();
    let thread = store.storage.open(&context, "reader").await.unwrap();
    let record = thread.task(&task_id).await.unwrap().unwrap();
    let reference = &record.internal.as_ref().unwrap().streams[0];
    assert_ne!(reference.stream_id, task_id);
    let cursor = frames[1].sse.lines().find_map(|line| line.strip_prefix("id: ")).unwrap();
    let resumed = door
        .dispatch(
            body("SubscribeToTask", json!({"id": task_id, "contextId": context, "offset": cursor})),
            principal(true),
            None,
            None,
            None,
            None,
        )
        .await;
    let current: JsonRpcResponse =
        serde_json::from_str(&resumed.next().await.unwrap().unwrap().json).unwrap();
    let StreamResponse::Task(current) = serde_json::from_value(current.result.unwrap()).unwrap()
    else {
        panic!("subscription starts with current Task")
    };
    assert_eq!(current.status.state, a2a::TaskState::Completed);
    let mut replay = Vec::new();
    while let Some(frame) = resumed.next().await.unwrap() {
        let envelope: JsonRpcResponse = serde_json::from_str(&frame.json).unwrap();
        assert_eq!(envelope.id, "SubscribeToTask".into());
        replay.push(envelope.result.unwrap());
    }
    let expected: Vec<_> = frames[2..]
        .iter()
        .map(|f| serde_json::from_str::<JsonRpcResponse>(&f.json).unwrap().result.unwrap())
        .collect();
    assert_eq!(replay, expected);
    thread.close().await;
    store.storage.delete_thread(&context).await.unwrap();
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
    let door = door(store.storage.clone(), Arc::new(Host));
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
    held.abandon(lk_agents_storage::run::RunOutcome::Canceled, "test-complete").await.unwrap();
    door.aclose().await;
    assert_eq!(
        thread.task(task.id.as_str()).await.unwrap().unwrap().status,
        lk_agents_storage::run::TaskState::Completed
    );
    thread.close().await;
    door.aclose().await;
    store.close().await;
}

async fn a_continuation_keeps_one_wire_identity_and_all_streams(backend: Backend) {
    let store = backend.store(Default::default()).await;
    let door = door(store.storage.clone(), Arc::new(AskingHost));
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
    assert_eq!(saved.internal.unwrap().streams.len(), 2);
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
        lk_agents_storage::run::TaskState::InputRequired
    );
    thread.close().await;
    door.aclose().await;
    store.close().await;
}

async fn a_slow_observer_can_resume_every_undelivered_event(backend: Backend) {
    let store = backend.store(Default::default()).await;
    let door = door(store.storage.clone(), Arc::new(BurstHost));
    let response = door.dispatch(body("SendStreamingMessage", json!({"message": {"messageId": "burst", "role": "ROLE_USER", "parts": [{"text": "hello"}]}})), principal(true), None, None, None, None).await;
    // Leave the observer idle until its bounded queue has detached from durable work.
    door.aclose().await;
    let mut task = None;
    let mut cursor = None;
    let mut delivered = 0;
    let mut detached = false;
    while let Some(frame) = response.next().await.unwrap() {
        let envelope: JsonRpcResponse = serde_json::from_str(&frame.json).unwrap();
        if envelope.error.is_some() {
            detached = true;
            assert!(
                !frame.sse.lines().any(|line| line.starts_with("id: ")),
                "detachment must not acknowledge an undelivered durable event"
            );
        } else {
            if let StreamResponse::Task(value) =
                serde_json::from_value(envelope.result.unwrap()).unwrap()
            {
                task = Some(value);
            }
            cursor = frame
                .sse
                .lines()
                .find_map(|line| line.strip_prefix("id: "))
                .map(str::to_owned)
                .or(cursor);
            delivered += 1;
        }
    }
    assert!(detached);
    let task = task.unwrap();
    let resumed = door
        .dispatch(
            body(
                "SubscribeToTask",
                json!({"id": task.id, "contextId": task.context_id, "offset": cursor}),
            ),
            principal(true),
            None,
            None,
            None,
            None,
        )
        .await;
    let current: JsonRpcResponse =
        serde_json::from_str(&resumed.next().await.unwrap().unwrap().json).unwrap();
    let StreamResponse::Task(current) = serde_json::from_value(current.result.unwrap()).unwrap()
    else {
        panic!("current task")
    };
    assert_eq!(current.status.state, a2a::TaskState::Completed);
    let mut replayed = 0;
    while let Some(frame) = resumed.next().await.unwrap() {
        let envelope: JsonRpcResponse = serde_json::from_str(&frame.json).unwrap();
        assert!(envelope.error.is_none());
        replayed += 1;
    }
    assert_eq!(
        delivered + replayed,
        103,
        "submitted, 100 progress updates, artifact, final status"
    );
    store.close().await;
}

lk_agents_storage::contract!(
    create_requires_write_before_allocating_a_thread,
    subscriptions_use_task_records_and_resume_each_durable_event,
    streaming_acceptance_precedes_the_lane_wait,
    a_continuation_keeps_one_wire_identity_and_all_streams,
    a_slow_observer_can_resume_every_undelivered_event
);
