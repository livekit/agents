//! The native JSON-RPC door against the shared backend contract.

use std::sync::Arc;
use std::time::Duration;

use a2a::{JsonRpcResponse, StreamResponse};
use lk_agents_abi::a2a::auth::Credentials;
use lk_agents_abi::a2a::auth::{AuthGrant, AuthPrincipal, PrincipalKind};
use lk_agents_abi::a2a::card::EndpointDescriptor;
use lk_agents_abi::a2a::door::{A2ADoor as _, DoorConfig, DoorError, TextServing};
use lk_agents_abi::a2a::metadata::EXTENSION_URI;
use lk_agents_abi::a2a::options::{DebounceSpec, Policy, PolicySpec, TurnOptions};
use lk_agents_abi::a2a::turn::{HandlerOutput, HandlerPort, TurnContext, TurnHost};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::worker::http::{
    EndpointRequest, ExchangeError, ExchangeEvent, ExchangeMessage, HttpExchange, HttpScheme,
    RequestHead,
};
use lk_agents_runtime::a2a::door::serve::serve;
use lk_agents_runtime::a2a::door::{Door, MAX_PAYLOAD_BYTES};
use lk_agents_storage::testing::{Backend, Store};
use serde_json::{Value, json};

/// A scripted handler: `run` reports through the turn and returns the output.
macro_rules! host {
    ($name:ident, |$ctx:ident, $turn:ident| $body:expr) => {
        struct $name;

        impl TurnHost for $name {
            fn handle<'a>(
                &'a self,
                _cx: &lk_agents_abi::telemetry::Cx,
                $ctx: &'a TurnContext,
                $turn: Arc<dyn HandlerPort>,
                _cancel: Arc<dyn lk_agents_abi::invoke::Cancellation>,
            ) -> lk_agents_abi::BoxFuture<'a, Result<HandlerOutput, ProviderError>> {
                Box::pin(async move { Ok($body) })
            }
        }
    };
}

host!(Host, |_ctx, turn| {
    turn.progress("working".into()).await;
    turn.token("reply".into()).await;
    HandlerOutput { output: Some(json!("reply")), ..Default::default() }
});

host!(AskingHost, |ctx, turn| {
    let text = ctx
        .batch
        .last()
        .and_then(|message| message.parts.first())
        .and_then(|part| match part {
            lk_agents_abi::a2a::client::MessagePart::TextPart { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .unwrap_or_default();
    turn.progress("answering".into()).await;
    if text == "start" {
        HandlerOutput { input_required: Some("Your name?".into()), ..Default::default() }
    } else {
        HandlerOutput { output: Some(json!(format!("Hello {text}"))), ..Default::default() }
    }
});

host!(BurstHost, |_ctx, turn| {
    for n in 0..100 {
        turn.progress(format!("step {n}")).await;
    }
    HandlerOutput { output: Some(json!("done")), ..Default::default() }
});

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
    let response = door.dispatch(body("SendMessage", json!({"message": {"messageId": "m1", "role": "ROLE_USER", "parts": [{"text": "hello"}]}})), principal(false), None, None, None, None, None).await;
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
    let response = door.dispatch(body("SendStreamingMessage", json!({"message": {"messageId": "m1", "role": "ROLE_USER", "parts": [{"text": "hello"}]}})), principal(true), Some("1.0".into()), None, None, None, None).await;
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
    let response = Arc::new(door.dispatch(body("SendStreamingMessage", json!({"message": {"contextId": created.thread_id, "messageId": "queued", "role": "ROLE_USER", "parts": [{"text": "hello"}]}})), principal(true), None, None, None, None, None).await);
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
    let first = door.dispatch(body("SendMessage", json!({"message": {"messageId": "start", "role": "ROLE_USER", "parts": [{"text": "start"}]}})), principal(true), None, None, None, None, None).await;
    let envelope: JsonRpcResponse =
        serde_json::from_str(&first.next().await.unwrap().unwrap().json).unwrap();
    let a2a::SendMessageResponse::Task(task) =
        serde_json::from_value(envelope.result.unwrap()).unwrap()
    else {
        panic!("task")
    };
    assert_eq!(task.status.state, a2a::TaskState::InputRequired);
    let second = door.dispatch(body("SendStreamingMessage", json!({"message": {"messageId": "answer", "contextId": task.context_id, "taskId": task.id, "role": "ROLE_USER", "parts": [{"text": "Ada"}]}})), principal(true), None, None, None, None, None).await;
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
    let older = door.dispatch(body("SendMessage", json!({"message": {"messageId": "older-answer", "contextId": task.context_id, "taskId": "AST_older", "role": "ROLE_USER", "parts": [{"text": "Grace"}]}})), principal(true), None, None, None, None, None).await;
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
    let response = door.dispatch(body("SendStreamingMessage", json!({"message": {"messageId": "burst", "role": "ROLE_USER", "parts": [{"text": "hello"}]}})), principal(true), None, None, None, None, None).await;
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

// The endpoint over HTTP: the worker's exchange, faked at its binding contract.

struct Exchange {
    head: RequestHead,
    body: parking_lot::Mutex<std::collections::VecDeque<ExchangeEvent>>,
    sent: parking_lot::Mutex<Vec<ExchangeMessage>>,
}

impl HttpExchange for Exchange {
    fn head(&self) -> RequestHead {
        self.head.clone()
    }
    async fn receive(&self) -> Result<ExchangeEvent, ExchangeError> {
        Ok(self.body.lock().pop_front().unwrap_or(ExchangeEvent::Disconnected))
    }
    async fn send(&self, message: ExchangeMessage) -> Result<(), ExchangeError> {
        self.sent.lock().push(message);
        Ok(())
    }
}

/// What one exchange answered: its status, headers and whole body.
struct Answer {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    complete: bool,
}

impl Answer {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(key, _)| key == name).map(|(_, value)| value.as_str())
    }
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap()
    }
}

const KEY: &str = "APIserve";
const SECRET: &str = "serve-secret-serve-secret-serve-secret";

fn credentials() -> lk_agents_net::a2a_auth::Credentials {
    <lk_agents_net::a2a_auth::Credentials as Credentials>::new(KEY.into(), SECRET.into())
}

/// A worker holding the project's credentials, reached directly.
fn serving() -> TextServing {
    TextServing {
        request: EndpointRequest { route: "/desk/{path:path}".into(), ..Default::default() },
        agent_name: "front".into(),
        credentials: Some((KEY.into(), SECRET.into())),
        ..Default::default()
    }
}

async fn exchange(
    door: &Door,
    serving: TextServing,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Vec<Vec<u8>>,
) -> Answer {
    let mut events: std::collections::VecDeque<_> =
        body.iter().map(|data| ExchangeEvent::Body { data: data.clone(), more: true }).collect();
    events.push_back(ExchangeEvent::Body { data: Vec::new(), more: false });
    let mut fields = vec![(b"host".to_vec(), b"agents.test".to_vec())];
    fields.extend(headers.iter().map(|(k, v)| (k.as_bytes().to_vec(), v.as_bytes().to_vec())));
    let exchange = Exchange {
        head: RequestHead {
            method: method.into(),
            path: path.into(),
            raw_path: path.as_bytes().to_vec(),
            headers: fields,
            ..Default::default()
        },
        body: parking_lot::Mutex::new(events),
        sent: Default::default(),
    };
    serve(door, &exchange, serving).await.unwrap();
    let mut answer = Answer { status: 0, headers: Vec::new(), body: Vec::new(), complete: false };
    for message in exchange.sent.into_inner() {
        match message {
            ExchangeMessage::Start { status, headers } => {
                answer.status = status;
                answer.headers = headers
                    .into_iter()
                    .map(|(k, v)| (String::from_utf8(k).unwrap(), String::from_utf8(v).unwrap()))
                    .collect();
            }
            ExchangeMessage::Body { data, more } => {
                assert!(!answer.complete, "nothing follows the last body piece");
                answer.body.extend(data);
                answer.complete = !more;
            }
            other => panic!("not an HTTP message: {other:?}"),
        }
    }
    assert!(answer.complete);
    answer
}

const CARD: &str = "/desk/.well-known/agent-card.json";

// Catches: a caller admitted out of order: the console vouches for everyone, a worker holding
// credentials verifies the bearer (the node's verdict standing for a stripped project
// credential), and without credentials only the node's verdict admits.
async fn the_caller_is_authenticated_in_order(backend: Backend) {
    let store = backend.store(Default::default()).await;
    let door = door(&store, Arc::new(Host));
    let get = |serving, headers: Vec<(&'static str, String)>| {
        let door = &door;
        async move {
            let headers: Vec<_> = headers.iter().map(|(k, v)| (*k, v.as_str())).collect();
            exchange(door, serving, "GET", CARD, &headers, Vec::new()).await
        }
    };
    let bearer = |token: String| vec![("Authorization", format!("Bearer {token}"))];
    let project = credentials().project_token("caller".into(), 60).unwrap();
    let other = <lk_agents_net::a2a_auth::Credentials as Credentials>::new(
        "APIother".into(),
        SECRET.into(),
    );

    let refused = get(serving(), Vec::new()).await;
    assert_eq!(refused.status, 401);
    assert_eq!(refused.json(), json!({"error": "a LiveKit bearer token is required"}));
    assert_eq!(refused.header("content-type"), Some("application/json"));
    assert_eq!(get(serving(), bearer(project.clone())).await.status, 200);
    // header names match case-insensitively; the grant header is a carrier of its own
    let mixed = vec![("AUTHORIZATION", format!("Bearer {project}"))];
    assert_eq!(get(serving(), mixed).await.status, 200);
    let grant = credentials()
        .thread_grant_token(
            AuthGrant { all: true, read: true, ..Default::default() },
            "g".into(),
            60,
        )
        .unwrap();
    assert_eq!(get(serving(), vec![("X-LiveKit-Thread-Grant", grant)]).await.status, 200);
    let forged = other.project_token("caller".into(), 60).unwrap();
    assert_eq!(get(serving(), bearer(forged)).await.status, 401);
    let neither =
        credentials().thread_grant_token(AuthGrant::default(), "nobody".into(), 60).unwrap();
    let forbidden = get(serving(), bearer(neither)).await;
    assert_eq!(forbidden.status, 403);
    assert!(forbidden.json()["error"].as_str().unwrap().contains("neither"));

    let node = |authorized| {
        let mut serving = serving();
        serving.request.authorized = authorized;
        serving
    };
    assert_eq!(get(node(true), Vec::new()).await.status, 200, "the node's verdict");
    let bare = |authorized| TextServing { credentials: None, ..node(authorized) };
    assert_eq!(get(bare(false), Vec::new()).await.status, 401);
    assert_eq!(get(bare(true), Vec::new()).await.status, 200);
    let console = |mut serving: TextServing| {
        serving.request.unauthenticated = true;
        serving
    };
    assert_eq!(get(console(bare(false)), Vec::new()).await.status, 200, "the console's listener");
    door.aclose().await;
    store.close().await;
}

async fn the_card_and_the_methods_are_the_only_routes(backend: Backend) {
    let store = backend.store(Default::default()).await;
    let door = door(&store, Arc::new(Host));
    let open = || TextServing { credentials: None, ..node_authorized() };
    for path in ["/desk", "/desk/content", "/desk/tasks", "/other/v1", "/deskv1"] {
        let answer = exchange(&door, open(), "GET", path, &[], Vec::new()).await;
        assert_eq!((answer.status, answer.json()), (404, json!({"error": "Not Found"})), "{path}");
    }
    let answer = exchange(&door, open(), "GET", "/desk/v1", &[], Vec::new()).await;
    assert_eq!((answer.status, answer.json()), (405, json!({"error": "Method Not Allowed"})));
    let answer = exchange(&door, open(), "POST", CARD, &[], Vec::new()).await;
    assert_eq!(answer.status, 405);
    door.aclose().await;
    store.close().await;
}

fn node_authorized() -> TextServing {
    let mut serving = serving();
    serving.request.authorized = true;
    serving
}

// Catches: a card pointing callers somewhere the request did not reach.
async fn the_card_publishes_where_the_request_reached_the_endpoint(backend: Backend) {
    let store = backend.store(Default::default()).await;
    let door = door(&store, Arc::new(Host));
    let url = |endpoint_url: Option<&str>, scheme| {
        let mut serving = node_authorized();
        serving.endpoint_url = endpoint_url.map(str::to_owned);
        serving.request.scheme = scheme;
        let door = &door;
        async move {
            let card = exchange(door, serving, "GET", CARD, &[], Vec::new()).await.json();
            card["supportedInterfaces"][0]["url"].as_str().unwrap().to_owned()
        }
    };
    assert_eq!(url(None, HttpScheme::Http).await, "http://agents.test/desk/v1");
    assert_eq!(
        url(Some("/agents/front/default/desk"), HttpScheme::Https).await,
        "https://agents.test/agents/front/default/desk/v1"
    );
    assert_eq!(
        url(Some("https://acme.livekit.cloud/agents/front/prod/desk"), HttpScheme::Http).await,
        "https://acme.livekit.cloud/agents/front/prod/desk/v1"
    );
    let card = exchange(&door, node_authorized(), "GET", CARD, &[], Vec::new()).await.json();
    assert_eq!((card["name"].as_str(), card["version"].as_str()), (Some("desk"), Some("v1")));
    assert!(card["securityRequirements"].as_array().is_some_and(|r| !r.is_empty()));
    door.aclose().await;
    store.close().await;
}

// Catches: a payload over the budget buffered whole or reaching the turn.
async fn a_body_over_the_budget_is_refused_before_any_turn(backend: Backend) {
    let store = backend.store(Default::default()).await;
    let door = door(&store, Arc::new(Host));
    let chunk = vec![b' '; MAX_PAYLOAD_BYTES / 4];
    let body = vec![chunk.clone(), chunk.clone(), chunk.clone(), chunk, vec![b' ']];
    let answer = exchange(&door, node_authorized(), "POST", "/desk/v1", &[], body).await;
    assert_eq!(answer.status, 200);
    assert_eq!(answer.json()["error"]["code"], -32602);
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

// Catches: a stream answered as one JSON body, or SSE without the headers a client relies on.
async fn a_streaming_method_answers_sse_and_echoes_the_extension(backend: Backend) {
    let store = backend.store(Default::default()).await;
    let door = door(&store, Arc::new(Host));
    let grant = credentials()
        .thread_grant_token(
            AuthGrant {
                scope: "customer:42".into(),
                create: true,
                write: true,
                read: true,
                ..Default::default()
            },
            "caller".into(),
            60,
        )
        .unwrap();
    let authorization = format!("Bearer {grant}");
    let message = |method: &str, id: &str| {
        vec![body(
            method,
            json!({"message": {"messageId": id, "role": "ROLE_USER", "parts": [{"text": "hi"}]}}),
        )]
    };
    let headers = [("authorization", authorization.as_str()), ("a2a-extensions", EXTENSION_URI)];
    let sse = exchange(
        &door,
        serving(),
        "POST",
        "/desk/v1",
        &headers,
        message("SendStreamingMessage", "m1"),
    )
    .await;
    assert_eq!(sse.status, 200);
    assert_eq!(sse.header("content-type"), Some("text/event-stream"));
    assert_eq!(sse.header("cache-control"), Some("no-cache"));
    assert_eq!(sse.header("a2a-extensions"), Some(EXTENSION_URI));
    let text = String::from_utf8(sse.body).unwrap();
    let events: Vec<Value> = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    assert!(events.len() > 2, "{text}");
    assert!(events[0]["result"]["task"].is_object());
    assert_eq!(
        events.last().unwrap()["result"]["statusUpdate"]["status"]["state"],
        "TASK_STATE_COMPLETED"
    );

    let plain = [("authorization", authorization.as_str())];
    let json =
        exchange(&door, serving(), "POST", "/desk/v1", &plain, message("SendMessage", "m2")).await;
    assert_eq!(json.header("content-type"), Some("application/json"));
    assert_eq!(json.header("a2a-extensions"), None);
    assert_eq!(json.json()["result"]["task"]["status"]["state"], "TASK_STATE_COMPLETED");

    // the door's own refusal: a grant that may not write gets 403, not a JSON-RPC error
    let read_only = credentials()
        .thread_grant_token(
            AuthGrant { scope: "customer:42".into(), read: true, ..Default::default() },
            "reader".into(),
            60,
        )
        .unwrap();
    let read_only = format!("Bearer {read_only}");
    let refused = exchange(
        &door,
        serving(),
        "POST",
        "/desk/v1",
        &[("authorization", read_only.as_str())],
        message("SendMessage", "m3"),
    )
    .await;
    assert_eq!(refused.status, 403);
    assert!(refused.json()["error"].is_string());
    door.aclose().await;
    store.close().await;
}

// Catches: a bearer authorizing past its grant: a scope covers its own threads only, explicit
// ids cover theirs but create none, and a project credential alone reaches no thread.
async fn thread_authority_is_checked_on_every_request(backend: Backend) {
    let store = backend.store(Default::default()).await;
    let door = door(&store, Arc::new(Host));
    let mint = |grant: AuthGrant| {
        format!("Bearer {}", credentials().thread_grant_token(grant, "caller".into(), 60).unwrap())
    };
    let scoped = |scope: &str| {
        mint(AuthGrant {
            scope: scope.into(),
            create: true,
            write: true,
            read: true,
            list: true,
            ..Default::default()
        })
    };
    let call = |authorization: String, method: &'static str, params: Value| {
        let door = &door;
        async move {
            let headers = [("authorization", authorization.as_str())];
            exchange(door, serving(), "POST", "/desk/v1", &headers, vec![body(method, params)])
                .await
        }
    };
    let message = |id: &str, context: Option<&str>| {
        let mut message = json!({"messageId": id, "role": "ROLE_USER", "parts": [{"text": "hi"}]});
        if let Some(context) = context {
            message["contextId"] = json!(context);
        }
        json!({ "message": message })
    };
    let alice = scoped("user:alice");
    let created = call(alice.clone(), "SendMessage", message("m1", None)).await;
    let context = created.json()["result"]["task"]["contextId"].as_str().unwrap().to_owned();
    let info = store.storage.get_thread(&context).await.unwrap();
    assert_eq!(info.scope, "user:alice");

    let bob = call(scoped("user:bob"), "SendMessage", message("m2", Some(&context))).await;
    assert_eq!(bob.status, 403);
    let listed = call(alice, "ListTasks", json!({"contextId": context})).await;
    assert_eq!(listed.status, 200);
    let by_id = mint(AuthGrant {
        thread_ids: vec![context.clone()],
        create: true,
        write: true,
        read: true,
        ..Default::default()
    });
    let listed = call(by_id.clone(), "ListTasks", json!({"contextId": context})).await;
    assert_eq!(listed.json()["result"]["tasks"].as_array().unwrap().len(), 1);
    assert_eq!(call(by_id, "SendMessage", message("m3", None)).await.status, 403);
    let project = format!("Bearer {}", credentials().project_token("front".into(), 60).unwrap());
    assert_eq!(call(project, "SendMessage", message("m4", Some(&context))).await.status, 403);
    door.aclose().await;
    store.close().await;
}

lk_agents_storage::contract!(
    create_requires_write_before_allocating_a_thread,
    live_streams_have_no_replay_cursor_and_subscriptions_are_unsupported,
    streaming_acceptance_precedes_the_lane_wait,
    a_continuation_keeps_one_wire_identity,
    a_slow_observer_detaches_without_canceling_the_task,
    the_caller_is_authenticated_in_order,
    the_card_and_the_methods_are_the_only_routes,
    the_card_publishes_where_the_request_reached_the_endpoint,
    a_body_over_the_budget_is_refused_before_any_turn,
    a_streaming_method_answers_sse_and_echoes_the_extension,
    thread_authority_is_checked_on_every_request
);
