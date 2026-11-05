//! The native A2A client against a local HTTP/SSE peer, including chunked request bodies and
//! fragmented response bytes. The peer owns every accepted socket and settles them on shutdown.

use lk_agents_net::telemetry::Cx;
use std::collections::BTreeMap;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use a2a::{JsonRpcRequest, JsonRpcResponse, Message, Part, PartContent, Role};
use lk_agents_abi::a2a::metadata::REASON_KEY;
use lk_agents_abi::errors::{ApiErrorKind, ProviderError, TimeoutPhase};
use lk_agents_abi::storage::TaskState;
use lk_agents_abi::{ApiConnectOptions, Clock, Closed};
use lk_agents_core::a2a::client::{ClientError, TaskUpdate};
use lk_agents_core::a2a::rpc::TaskRef;
use lk_agents_net::a2a_client::{AgentClient, CleanupObservation, Reply};
use lk_agents_net::clock::RealClock;
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::source::{PullError, Pulled};
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Notify, mpsc, oneshot};
use tokio::task::{JoinHandle, JoinSet};

const LIMIT: Duration = Duration::from_secs(10);

#[derive(Clone, Copy)]
enum Scenario {
    Question,
    Waiting,
    Truncated,
    DelayedAck,
    ResetBeforeAck,
    FastQuestion,
    DelayedCancel,
    StalledCardHeaders,
    StalledCardBody,
    StalledRpcHeaders,
    StalledCancelBody,
    RetryOnce,
}

#[derive(Default)]
struct AckGate {
    waiting: Notify,
    release: Notify,
    finished: Notify,
    retries: AtomicUsize,
}

#[derive(Debug)]
struct Observed {
    method: String,
    path: String,
    headers: BTreeMap<String, String>,
    rpc: Option<JsonRpcRequest>,
}

struct Peer {
    url: String,
    observed: mpsc::Receiver<Observed>,
    shutdown: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<io::Result<()>>>,
    ack_gate: Arc<AckGate>,
}

impl Peer {
    async fn start(scenario: Scenario) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/expert", listener.local_addr().unwrap());
        let interface = format!("{url}/v1");
        let (requests, observed) = mpsc::channel(32);
        let (shutdown, mut stopping) = oneshot::channel();
        let ack_gate = Arc::new(AckGate::default());
        let peer_gate = ack_gate.clone();
        let task = tokio::spawn(async move {
            let mut sockets = JoinSet::new();
            loop {
                tokio::select! {
                    _ = &mut stopping => break,
                    socket = listener.accept() => {
                        let (socket, _) = socket?;
                        let interface = interface.clone();
                        let requests = requests.clone();
                        let gate = peer_gate.clone();
                        sockets.spawn(async move {
                            match serve(socket, &interface, scenario, requests, gate).await {
                                Err(error) if matches!(scenario, Scenario::Waiting | Scenario::DelayedAck)
                                    && matches!(error.kind(), io::ErrorKind::BrokenPipe | io::ErrorKind::ConnectionReset) => Ok(()),
                                result => result,
                            }
                        });
                    }
                }
            }
            drop(requests);
            while let Some(result) = sockets.join_next().await {
                result.map_err(io::Error::other)??;
            }
            Ok(())
        });
        Self { url, observed, shutdown: Some(shutdown), task: Some(task), ack_gate }
    }

    async fn finish(mut self) -> Vec<Observed> {
        let _ = self.shutdown.take().unwrap().send(());
        let mut task = self.task.take().unwrap();
        let result = match tokio::time::timeout(LIMIT, &mut task).await {
            Ok(result) => result,
            Err(_) => {
                task.abort();
                let _ = task.await;
                panic!("the controlled peer did not release its sockets");
            }
        };
        result.unwrap().unwrap();
        let mut observed = Vec::new();
        while let Some(request) = self.observed.recv().await {
            observed.push(request);
        }
        observed
    }
}

impl Drop for Peer {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

async fn read_request(socket: TcpStream) -> io::Result<(BufReader<TcpStream>, Observed)> {
    let mut socket = BufReader::new(socket);
    let mut start = String::new();
    socket.read_line(&mut start).await?;
    let mut start = start.split_whitespace();
    let method = start.next().ok_or_else(|| io::Error::other("missing method"))?.to_owned();
    let path = start.next().ok_or_else(|| io::Error::other("missing path"))?.to_owned();
    let mut headers = BTreeMap::new();
    loop {
        let mut line = String::new();
        socket.read_line(&mut line).await?;
        if line == "\r\n" {
            break;
        }
        let (name, value) =
            line.split_once(':').ok_or_else(|| io::Error::other("malformed header"))?;
        headers.insert(name.to_ascii_lowercase(), value.trim().to_owned());
    }
    let mut body = Vec::new();
    if headers.get("transfer-encoding").is_some_and(|value| value == "chunked") {
        loop {
            let mut line = String::new();
            socket.read_line(&mut line).await?;
            let length = usize::from_str_radix(line.trim().split(';').next().unwrap(), 16)
                .map_err(io::Error::other)?;
            if length == 0 {
                loop {
                    line.clear();
                    socket.read_line(&mut line).await?;
                    if line == "\r\n" {
                        break;
                    }
                }
                break;
            }
            let at = body.len();
            body.resize(at + length, 0);
            socket.read_exact(&mut body[at..]).await?;
            let mut delimiter = [0; 2];
            socket.read_exact(&mut delimiter).await?;
            if delimiter != *b"\r\n" {
                return Err(io::Error::other("malformed chunk"));
            }
        }
    } else if let Some(length) = headers.get("content-length") {
        body.resize(length.parse::<usize>().map_err(io::Error::other)?, 0);
        socket.read_exact(&mut body).await?;
    }
    let rpc = if body.is_empty() {
        None
    } else {
        Some(serde_json::from_slice(&body).map_err(io::Error::other)?)
    };
    Ok((socket, Observed { method, path, headers, rpc }))
}

async fn serve(
    socket: TcpStream,
    interface: &str,
    scenario: Scenario,
    requests: mpsc::Sender<Observed>,
    ack_gate: Arc<AckGate>,
) -> io::Result<()> {
    let (mut socket, request) = read_request(socket).await?;
    let rpc = request.rpc.clone();
    let is_get = request.method == "GET";
    requests.send(request).await.map_err(io::Error::other)?;
    if (is_get && matches!(scenario, Scenario::StalledCardHeaders | Scenario::StalledCardBody))
        || (!is_get && matches!(scenario, Scenario::StalledRpcHeaders))
    {
        if matches!(scenario, Scenario::StalledCardBody) {
            socket.get_mut().write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1000\r\nConnection: close\r\n\r\n{").await?;
        }
        let mut discarded = Vec::new();
        socket.read_to_end(&mut discarded).await?;
        return Ok(());
    }
    if is_get {
        let body = serde_json::to_vec(&json!({"name": "expert", "version": "1",
            "supportedInterfaces": [{"url": interface, "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
            "capabilities": {"streaming": true, "extensions": if matches!(scenario, Scenario::Question) {
                vec![json!({"uri": lk_agents_abi::a2a::metadata::EXTENSION_URI})]
            } else { vec![] }}})).unwrap();
        return respond(socket.get_mut(), "application/json", &body).await;
    }
    let rpc = rpc.ok_or_else(|| io::Error::other("POST without JSON-RPC"))?;
    let params = rpc.params.clone().ok_or_else(|| io::Error::other("RPC without params"))?;
    if matches!(scenario, Scenario::RetryOnce)
        && rpc.method != "CancelTask"
        && ack_gate.retries.fetch_add(1, Ordering::SeqCst) == 0
    {
        return socket.get_mut().shutdown().await;
    }
    if rpc.method == "CancelTask" {
        if matches!(scenario, Scenario::StalledCancelBody) {
            socket.get_mut().write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1000\r\nConnection: close\r\n\r\n{").await?;
            let mut discarded = Vec::new();
            socket.read_to_end(&mut discarded).await?;
            return Ok(());
        }
        if matches!(scenario, Scenario::DelayedCancel) {
            ack_gate.waiting.notify_one();
            ack_gate.release.notified().await;
        }
        let result = json!({"id": params["id"], "contextId": params["contextId"],
            "status": {"state": "TASK_STATE_CANCELED"}});
        let body = serde_json::to_vec(&JsonRpcResponse::success(rpc.id, result)).unwrap();
        return respond(socket.get_mut(), "application/json", &body).await;
    }
    if matches!(scenario, Scenario::ResetBeforeAck) {
        return socket.get_mut().shutdown().await;
    }
    if matches!(scenario, Scenario::DelayedAck) {
        ack_gate.waiting.notify_one();
        ack_gate.release.notified().await;
    }
    let (task, context, answer) = if rpc.method == "SubscribeToTask" {
        (
            params["id"].as_str().unwrap().to_owned(),
            params["contextId"].as_str().unwrap().to_owned(),
            true,
        )
    } else {
        let message: a2a::SendMessageRequest =
            serde_json::from_value(params).map_err(io::Error::other)?;
        let answer = message.message.task_id.is_some();
        (
            message.message.task_id.unwrap_or_else(|| "remote-task".into()),
            message.message.context_id.unwrap_or_else(|| "remote-context".into()),
            answer,
        )
    };
    let mut events = vec![
        json!({"task": {"id": task, "contextId": context, "status": {"state": "TASK_STATE_SUBMITTED"}}}),
    ];
    events.push(json!({"statusUpdate": {"taskId": task, "contextId": context,
        "status": {"state": "TASK_STATE_WORKING", "message": {"messageId": "progress", "role": "ROLE_AGENT", "parts": [{"text": "Checking café"}]}}}}));
    match scenario {
        Scenario::Question | Scenario::FastQuestion | Scenario::RetryOnce if !answer => events.push(json!({"statusUpdate": {"taskId": task, "contextId": context,
            "status": {"state": "TASK_STATE_INPUT_REQUIRED", "message": {"messageId": "question", "role": "ROLE_AGENT", "parts": [{"text": "Which account?"}]}}}})),
        Scenario::Question | Scenario::FastQuestion | Scenario::RetryOnce => {
            events.push(json!({"artifactUpdate": {"taskId": task, "contextId": context,
                "artifact": {"artifactId": "answer", "name": "answer", "parts": [{"text": "Confirmed "}]}}}));
            events.push(json!({"artifactUpdate": {"taskId": task, "contextId": context,
                "artifact": {"artifactId": "answer", "parts": [{"data": {"receipt": 42}}]}, "append": true, "lastChunk": true}}));
            events.push(json!({"statusUpdate": {"taskId": task, "contextId": context,
                "status": {"state": "TASK_STATE_COMPLETED"}, "metadata": {"https://livekit.io/a2a/ext/agents/v1/verbatim": "Confirmed "}}}));
        }
        Scenario::Waiting | Scenario::Truncated | Scenario::DelayedAck | Scenario::StalledCancelBody => {
            events.push(json!({"artifactUpdate": {"taskId": task, "contextId": context,
                "artifact": {"artifactId": "partial", "parts": [{"text": "unaccepted partial"}]}}}));
        }
        Scenario::ResetBeforeAck | Scenario::DelayedCancel | Scenario::StalledCardHeaders | Scenario::StalledCardBody | Scenario::StalledRpcHeaders => unreachable!("unary response already served"),
    }
    let mut body = Vec::new();
    for (index, event) in events.into_iter().enumerate() {
        let response =
            serde_json::to_string(&JsonRpcResponse::success(rpc.id.clone(), event)).unwrap();
        body.extend_from_slice(format!("id: event-{index}\r\ndata: {response}\r\n\r\n").as_bytes());
    }
    if matches!(scenario, Scenario::Truncated) {
        body.extend_from_slice(b"data: {\"jsonrpc\":");
    }
    if matches!(scenario, Scenario::Waiting | Scenario::DelayedAck) {
        socket.get_mut().write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n").await?;
        for fragment in body.chunks(7) {
            socket.get_mut().write_all(format!("{:x}\r\n", fragment.len()).as_bytes()).await?;
            socket.get_mut().write_all(fragment).await?;
            socket.get_mut().write_all(b"\r\n").await?;
        }
        socket.get_mut().flush().await?;
        let mut discarded = Vec::new();
        socket.read_to_end(&mut discarded).await?;
        Ok(())
    } else {
        let response = respond(socket.get_mut(), "text/event-stream", &body).await;
        if matches!(scenario, Scenario::FastQuestion) {
            ack_gate.finished.notify_one();
        }
        response
    }
}

async fn respond(socket: &mut TcpStream, content_type: &str, body: &[u8]) -> io::Result<()> {
    socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).as_bytes()).await?;
    for fragment in body.chunks(7) {
        socket.write_all(fragment).await?;
    }
    socket.shutdown().await
}

fn message(id: &str, text: &str) -> Message {
    Message {
        message_id: id.into(),
        context_id: None,
        task_id: None,
        role: Role::User,
        parts: vec![Part::text(text)],
        metadata: None,
        extensions: None,
        reference_task_ids: None,
    }
}

/// The test's runtime on `clock`.
fn env(clock: Arc<dyn Clock>) -> ProcessEnv {
    ProcessEnv { clock, ..ProcessEnv::current_runtime() }
}

fn client(peer: &Peer, clock: &Arc<RealClock>) -> AgentClient {
    let options = ApiConnectOptions::default();
    AgentClient::new(&env(clock.clone()), peer.url.clone(), Some("test-token".into()), options)
        .unwrap()
}

async fn updates(reply: &Reply) -> Vec<TaskUpdate> {
    tokio::time::timeout(LIMIT, async {
        let mut updates = Vec::new();
        loop {
            match reply.pull().await.unwrap() {
                Pulled::Item(update) => updates.push(update),
                Pulled::End => return updates,
                Pulled::Cancelled => panic!("a live reply was canceled"),
            }
        }
    })
    .await
    .expect("reply stream deadline")
}

#[tokio::test]
async fn agent_client_peer_matrix_preserves_task_continuation_artifacts_and_version_headers() {
    let peer = Peer::start(Scenario::Question).await;
    let clock = Arc::new(RealClock::new());
    let client = client(&peer, &clock);
    let mut outgoing = message("stable-first-message", "Please check");
    outgoing
        .parts
        .extend([Part::data(json!({"account": 42})), Part::url("https://example.test/input.pdf")]);
    let history = lk_agents_core::a2a::caller_history::CallerHistory::decode(json!({"items": [{
        "id": "caller-message", "type": "message", "created_at": 1.0,
        "role": "user", "content": ["Public caller context"]
    }]}))
    .unwrap();
    let question = client.send(outgoing, None, None, Some(&history), &Cx::NONE).await.unwrap();
    assert_eq!((&*question.task_id, &*question.thread_id), ("remote-task", "remote-context"));
    let first = updates(&question).await;
    assert_eq!(first.last().unwrap().state, TaskState::InputRequired);
    assert_eq!(first.last().unwrap().text, "Which account?");
    assert!(first.iter().any(|u| u.state == TaskState::Working && u.text == "Checking café"));
    let answer = client
        .send(
            message("answer", "Account 42"),
            Some(&question.thread_id),
            Some(&question.task_id),
            None,
            &Cx::NONE,
        )
        .await
        .unwrap();
    let second = updates(&answer).await;
    let done = second.last().unwrap();
    assert_eq!(done.state, TaskState::Completed);
    assert_eq!(done.text, "Confirmed {\"receipt\":42}");
    assert_eq!(done.metadata["livekit/verbatim"], "Confirmed ");
    assert!(matches!(done.artifacts[0].parts[1].content, PartContent::Data(_)));
    let task = TaskRef { id: answer.task_id.clone(), context_id: answer.thread_id.clone() };
    let subscribed = client.subscribe(task, Some("opaque-cursor"), &Cx::NONE).await.unwrap();
    assert_eq!(updates(&subscribed).await.last().unwrap().text, done.text);
    client.connect().await.unwrap();
    client.connect().await.unwrap();
    client.aclose().await;
    assert!(client.cleanup_observations().is_empty());

    let cancel_only = self::client(&peer, &clock);
    for id in ["running-one", "running-two"] {
        let canceled = cancel_only
            .cancel(
                TaskRef { id: id.into(), context_id: "cancel-context".into() },
                Some("user-canceled"),
            )
            .await
            .unwrap();
        assert_eq!(canceled.state, TaskState::Canceled);
    }
    cancel_only.aclose().await;
    let requests = peer.finish().await;
    assert_eq!(
        requests.iter().filter(|r| r.method == "GET").count(),
        2,
        "each client connection fetches its card only once"
    );
    for request in &requests {
        assert_eq!(request.headers.get("a2a-version").map(String::as_str), Some("1.0"));
        assert_eq!(
            request.headers.get("authorization").map(String::as_str),
            Some("Bearer test-token")
        );
        assert!(
            request.path == "/expert/v1" || request.path == "/expert/.well-known/agent-card.json"
        );
    }
    let sends: Vec<_> = requests
        .iter()
        .filter_map(|r| r.rpc.as_ref())
        .filter(|r| r.method == "SendStreamingMessage")
        .collect();
    let first: a2a::SendMessageRequest =
        serde_json::from_value(sends[0].params.clone().unwrap()).unwrap();
    assert_eq!(first.message.message_id, "stable-first-message");
    assert_eq!(first.message.parts.len(), 4);
    assert_eq!(
        lk_agents_core::a2a::caller_history::CallerHistory::from_message(&first.message).unwrap(),
        Some(history)
    );
    let resumed: a2a::SendMessageRequest =
        serde_json::from_value(sends[1].params.clone().unwrap()).unwrap();
    assert_eq!(resumed.message.task_id.as_deref(), Some("remote-task"));
    assert_eq!(resumed.message.context_id.as_deref(), Some("remote-context"));
    assert!(!resumed.message.message_id.is_empty());
    assert_eq!(resumed.message.parts, vec![Part::text("Account 42")]);
    let subscription = requests
        .iter()
        .filter_map(|r| r.rpc.as_ref())
        .find(|r| r.method == "SubscribeToTask")
        .unwrap();
    assert_eq!(subscription.params.as_ref().unwrap()["offset"], "opaque-cursor");
}

#[tokio::test]
async fn closing_or_dropping_an_acknowledged_reply_cancels_the_exact_task_without_manufacturing_success()
 {
    let peer = Peer::start(Scenario::Waiting).await;
    let clock = Arc::new(RealClock::new());
    for explicit_close in [true, false] {
        let client = client(&peer, &clock);
        let reply =
            client.send(message("cancel-me", "Work"), None, None, None, &Cx::NONE).await.unwrap();
        let first = reply.pull().await.unwrap();
        assert!(matches!(first, Pulled::Item(TaskUpdate { state: TaskState::Submitted, .. })));
        if explicit_close {
            assert_eq!(reply.aclose(clock.now().saturating_add(LIMIT)).await, Closed::Clean);
            assert!(matches!(reply.pull().await.unwrap(), Pulled::End));
        }
        drop(reply);
        client.aclose().await;
        assert!(client.cleanup_observations().is_empty());
    }
    let requests = peer.finish().await;
    let cancellations: Vec<_> = requests
        .iter()
        .filter_map(|r| r.rpc.as_ref())
        .filter(|r| r.method == "CancelTask")
        .collect();
    assert_eq!(cancellations.len(), 2, "one cancellation for each reply");
    for cancellation in cancellations {
        let params = cancellation.params.as_ref().unwrap();
        assert_eq!(params["id"], "remote-task");
        assert_eq!(params["contextId"], "remote-context");
        assert_eq!(params["metadata"][REASON_KEY], "caller-canceled");
    }
    for request in requests {
        assert_eq!(request.headers.get("a2a-version").map(String::as_str), Some("1.0"));
    }
}

#[tokio::test]
async fn a_truncated_remote_stream_reports_failure_after_progress_and_never_a_final_answer() {
    let peer = Peer::start(Scenario::Truncated).await;
    let clock = Arc::new(RealClock::new());
    let client = client(&peer, &clock);
    let reply =
        client.send(message("truncated", "Work"), None, None, None, &Cx::NONE).await.unwrap();
    let mut progress = false;
    let failure = tokio::time::timeout(LIMIT, async {
        loop {
            match reply.pull().await {
                Ok(Pulled::Item(update)) => {
                    assert_ne!(update.state, TaskState::Completed);
                    progress |= update.state == TaskState::Working;
                }
                Err(error) => return error,
                ended => panic!("an incomplete task became a clean end: {ended:?}"),
            }
        }
    })
    .await
    .unwrap();
    assert!(matches!(failure, PullError::Failed(_)));
    assert!(progress);
    client.aclose().await;
    assert!(client.cleanup_observations().is_empty());
    let requests = peer.finish().await;
    let cancellations: Vec<_> = requests
        .iter()
        .filter_map(|r| r.rpc.as_ref())
        .filter(|r| r.method == "CancelTask")
        .collect();
    assert_eq!(
        cancellations.len(),
        1,
        "transport failure still owes cleanup of the known remote task"
    );
    assert_eq!(cancellations[0].params.as_ref().unwrap()["contextId"], "remote-context");
}

#[tokio::test]
async fn malformed_messages_fail_without_discovery_or_remote_side_effects() {
    let peer = Peer::start(Scenario::Question).await;
    let clock = Arc::new(RealClock::new());
    let client = client(&peer, &clock);
    let mut malformed = message("", "missing id");
    assert!(matches!(
        client.send(malformed.clone(), None, None, None, &Cx::NONE).await,
        Err(ClientError::InvalidInput(_))
    ));
    malformed.message_id = "missing-context".into();
    assert!(matches!(
        client.send(malformed, None, Some("pending"), None, &Cx::NONE).await,
        Err(ClientError::InvalidInput(_))
    ));
    client.aclose().await;
    assert!(peer.finish().await.is_empty());
}

#[test]
fn an_unusable_process_runtime_is_a_final_error_not_a_retryable_connection_failure() {
    let env = ProcessEnv {
        runtime: || Err("the process runtime belongs to the parent of this fork".into()),
        ..ProcessEnv::current_runtime()
    };
    let options = ApiConnectOptions::default();
    let Err(ClientError::Transport(error)) =
        AgentClient::new(&env, "http://127.0.0.1:9/agent".into(), None, options)
    else {
        panic!("a client was built without a usable runtime");
    };
    assert!(!error.retryable && error.kind != ApiErrorKind::Connection, "{error:?}");
}

#[tokio::test]
async fn cancel_before_ack_drains_the_original_request_then_cancels_without_resending() {
    let peer = Peer::start(Scenario::DelayedAck).await;
    let clock = Arc::new(RealClock::new());
    let client = Arc::new(client(&peer, &clock));
    let sending = tokio::spawn({
        let client = client.clone();
        async move {
            client
                .send(message("accepted-before-cancel", "Work once"), None, None, None, &Cx::NONE)
                .await
        }
    });
    tokio::time::timeout(LIMIT, peer.ack_gate.waiting.notified()).await.unwrap();
    sending.abort();
    assert!(matches!(sending.await, Err(error) if error.is_cancelled()));
    peer.ack_gate.release.notify_one();
    client.aclose().await;
    assert!(client.cleanup_observations().is_empty());
    let requests = peer.finish().await;
    let rpcs: Vec<_> = requests.iter().filter_map(|r| r.rpc.as_ref()).collect();
    assert_eq!(
        rpcs.iter().filter(|r| r.method == "SendStreamingMessage").count(),
        1,
        "cancellation must never resend the business request to discover its task"
    );
    let cancellations: Vec<_> = rpcs.into_iter().filter(|r| r.method == "CancelTask").collect();
    assert_eq!(cancellations.len(), 1);
    let params = cancellations[0].params.as_ref().unwrap();
    assert_eq!(params["id"], "remote-task");
    assert_eq!(params["contextId"], "remote-context");
    assert_eq!(params["metadata"][REASON_KEY], "caller-canceled");
}

#[tokio::test]
async fn a_request_lost_before_ack_is_reported_as_uncertain_and_is_never_resent() {
    let peer = Peer::start(Scenario::ResetBeforeAck).await;
    let clock = Arc::new(RealClock::new());
    let client = AgentClient::new(
        &env(clock.clone()),
        peer.url.clone(),
        Some("test-token".into()),
        ApiConnectOptions { max_retry: 0, ..Default::default() },
    )
    .unwrap();
    let result = tokio::time::timeout(
        LIMIT,
        client.send(message("accepted-with-lost-ack", "Work once"), None, None, None, &Cx::NONE),
    )
    .await
    .unwrap();
    assert!(result.is_err());
    client.aclose().await;
    assert!(matches!(
        client.cleanup_observations().as_slice(),
        [CleanupObservation::UnacknowledgedRequest { .. }]
    ));
    let requests = peer.finish().await;
    let methods: Vec<_> =
        requests.iter().filter_map(|r| r.rpc.as_ref()).map(|r| r.method.as_str()).collect();
    assert_eq!(methods, ["SendStreamingMessage"]);
}

#[tokio::test]
async fn an_unconsumed_input_question_is_canceled_even_after_the_remote_stream_ends() {
    let peer = Peer::start(Scenario::FastQuestion).await;
    let clock = Arc::new(RealClock::new());
    let client = client(&peer, &clock);
    let reply = client
        .send(message("abandoned-question", "Work"), None, None, None, &Cx::NONE)
        .await
        .unwrap();
    assert!(matches!(
        reply.pull().await.unwrap(),
        Pulled::Item(TaskUpdate { state: TaskState::Submitted, .. })
    ));
    assert!(matches!(
        reply.pull().await.unwrap(),
        Pulled::Item(TaskUpdate { state: TaskState::Working, .. })
    ));
    tokio::time::timeout(LIMIT, peer.ack_gate.finished.notified()).await.unwrap();
    drop(reply);
    client.aclose().await;
    assert!(client.cleanup_observations().is_empty());
    let requests = peer.finish().await;
    let cancellations: Vec<_> = requests
        .iter()
        .filter_map(|r| r.rpc.as_ref())
        .filter(|r| r.method == "CancelTask")
        .collect();
    assert_eq!(cancellations.len(), 1);
    let params = cancellations[0].params.as_ref().unwrap();
    assert_eq!(params["id"], "remote-task");
    assert_eq!(params["contextId"], "remote-context");
    assert_eq!(params["metadata"][REASON_KEY], "caller-canceled");
}

#[tokio::test]
async fn client_close_owns_a_stalled_cancel_request_until_its_original_ack_is_drained() {
    let peer = Peer::start(Scenario::DelayedCancel).await;
    let clock = Arc::new(RealClock::new());
    let client = Arc::new(client(&peer, &clock));
    let cancelling = tokio::spawn({
        let client = client.clone();
        async move {
            client
                .cancel(
                    TaskRef { id: "cancel-task".into(), context_id: "cancel-context".into() },
                    Some("user-canceled"),
                )
                .await
        }
    });
    tokio::time::timeout(LIMIT, peer.ack_gate.waiting.notified()).await.unwrap();
    let close_started = Arc::new(Notify::new());
    let closing = tokio::spawn({
        let client = client.clone();
        let started = close_started.clone();
        async move {
            started.notify_one();
            client.aclose().await
        }
    });
    close_started.notified().await;
    peer.ack_gate.release.notify_one();
    tokio::time::timeout(LIMIT, closing).await.unwrap().unwrap();
    let outcome = tokio::time::timeout(LIMIT, cancelling).await.unwrap().unwrap();
    match outcome {
        Ok(update) => assert_eq!(update.state, TaskState::Canceled),
        Err(error) => assert!(matches!(error, ClientError::Closed | ClientError::Incomplete)),
    }
    assert!(client.cleanup_observations().is_empty());
    let requests = peer.finish().await;
    let rpcs: Vec<_> = requests.iter().filter_map(|r| r.rpc.as_ref()).collect();
    assert_eq!(
        rpcs.len(),
        1,
        "close drains the original cancel without issuing another control request"
    );
    assert_eq!(rpcs[0].method, "CancelTask");
    let params = rpcs[0].params.as_ref().unwrap();
    assert_eq!(params["id"], "cancel-task");
    assert_eq!(params["contextId"], "cancel-context");
    assert_eq!(params["metadata"][REASON_KEY], "user-canceled");
}

#[tokio::test]
async fn a_cleanup_deadline_reports_incomplete_but_retains_the_original_cancel_future() {
    let peer = Peer::start(Scenario::DelayedCancel).await;
    let clock = Arc::new(RealClock::new());
    let client = Arc::new(client(&peer, &clock));
    let cancelling = tokio::spawn({
        let client = client.clone();
        async move {
            client
                .cancel(
                    TaskRef { id: "slow-cancel".into(), context_id: "slow-context".into() },
                    Some("user-canceled"),
                )
                .await
        }
    });
    tokio::time::timeout(LIMIT, peer.ack_gate.waiting.notified()).await.unwrap();
    tokio::time::pause();
    let close_started = Arc::new(Notify::new());
    let closing = tokio::spawn({
        let client = client.clone();
        let started = close_started.clone();
        async move {
            started.notify_one();
            client.aclose().await
        }
    });
    close_started.notified().await;
    tokio::time::advance(lk_agents_abi::HOST_TEARDOWN_TIMEOUT + Duration::from_secs(1)).await;
    closing.await.unwrap();
    let observations = client.cleanup_observations();
    assert!(
        observations.iter().any(
            |o| matches!(o, CleanupObservation::Incomplete { requests } if !requests.is_empty())
        )
    );
    assert!(!cancelling.is_finished(), "expiry must keep ownership of the original request");
    tokio::time::resume();
    peer.ack_gate.release.notify_one();
    let result = tokio::time::timeout(LIMIT, cancelling).await.unwrap().unwrap();
    match result {
        Ok(update) => assert_eq!(update.state, TaskState::Canceled),
        Err(error) => assert!(matches!(error, ClientError::Closed | ClientError::Incomplete)),
    }
    client.aclose().await;
    assert!(!client.cleanup_observations().iter().any(|o| matches!(
        o,
        CleanupObservation::CancelUncertain { .. }
            | CleanupObservation::UnacknowledgedRequest { .. }
    )));
    let requests = peer.finish().await;
    let methods: Vec<_> =
        requests.iter().filter_map(|r| r.rpc.as_ref()).map(|r| r.method.as_str()).collect();
    assert_eq!(methods, ["CancelTask"], "the original control request is retained, never replaced");
}

#[tokio::test]
async fn discovery_rpc_and_cancel_stalls_keep_their_timeout_phase() {
    for (scenario, total, cancel) in [
        (Scenario::StalledCardHeaders, false, false),
        (Scenario::StalledCardBody, false, false),
        (Scenario::StalledRpcHeaders, false, false),
        (Scenario::StalledCancelBody, false, true),
        (Scenario::StalledCardBody, true, false),
    ] {
        let peer = Peer::start(scenario).await;
        let clock = Arc::new(RealClock::new());
        let options = ApiConnectOptions {
            max_retry: 0,
            read_timeout: Some(if total { 10.0 } else { 0.05 }),
            total_timeout: total.then_some(0.05),
            ..Default::default()
        };
        let client =
            AgentClient::new(&env(clock), peer.url.clone(), Some("test-token".into()), options)
                .unwrap();
        let result = tokio::time::timeout(LIMIT, async {
            if cancel {
                client
                    .cancel(
                        TaskRef { id: "stalled-task".into(), context_id: "stalled-context".into() },
                        None,
                    )
                    .await
                    .map(|_| ())
            } else {
                client.send(message("work", "Work"), None, None, None, &Cx::NONE).await.map(|_| ())
            }
        })
        .await
        .expect("phase deadline must settle before the peer test deadline");
        let ClientError::Transport(error) = result.unwrap_err() else {
            panic!("transport error lost its API error type");
        };
        assert_eq!(error.kind, ApiErrorKind::Timeout);
        assert_eq!(error.phase, Some(if total { TimeoutPhase::Total } else { TimeoutPhase::Read }));
        client.aclose().await;
        let requests = peer.finish().await;
        assert!(
            requests
                .iter()
                .all(|r| r.headers.get("a2a-version").map(String::as_str) == Some("1.0"))
        );
    }
}

#[tokio::test]
async fn stalled_acknowledged_stream_retains_read_timeout_through_the_reply_source() {
    let peer = Peer::start(Scenario::Waiting).await;
    let clock = Arc::new(RealClock::new());
    let client = AgentClient::new(
        &env(clock),
        peer.url.clone(),
        Some("test-token".into()),
        ApiConnectOptions { max_retry: 3, read_timeout: Some(0.05), ..Default::default() },
    )
    .unwrap();
    let reply = client.send(message("work", "Work"), None, None, None, &Cx::NONE).await.unwrap();
    loop {
        match reply.pull().await {
            Ok(Pulled::Item(update)) => assert!(!update.state.is_terminal()),
            Err(PullError::Failed(ProviderError::Local(error))) => {
                assert_eq!(error.kind, ApiErrorKind::Timeout);
                assert_eq!(error.phase, Some(TimeoutPhase::Read));
                break;
            }
            other => panic!("stalled stream lost the typed timeout: {other:?}"),
        }
    }
    client.aclose().await;
    let requests = peer.finish().await;
    assert_eq!(
        requests
            .iter()
            .filter_map(|r| r.rpc.as_ref())
            .filter(|r| r.method == "SendStreamingMessage")
            .count(),
        1,
        "progress already delivered cannot be replayed"
    );
}

#[tokio::test]
async fn retry_before_ack_preserves_the_business_message_and_rpc_identity() {
    let peer = Peer::start(Scenario::RetryOnce).await;
    let clock = Arc::new(RealClock::new());
    let client = client(&peer, &clock);
    let reply = client
        .send(message("stable-across-retry", "Work once"), None, None, None, &Cx::NONE)
        .await
        .unwrap();
    assert_eq!(updates(&reply).await.last().unwrap().state, TaskState::InputRequired);
    client.aclose().await;
    let requests = peer.finish().await;
    let sends: Vec<_> = requests
        .iter()
        .filter_map(|r| r.rpc.as_ref())
        .filter(|r| r.method == "SendStreamingMessage")
        .collect();
    assert_eq!(sends.len(), 2);
    assert_eq!(sends[0], sends[1], "retry preserves the prepared request, including messageId");
}

#[tokio::test]
async fn remote_redirect_cannot_leave_the_configured_origin() {
    let untrusted = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let destination = format!("http://{}/stolen", untrusted.local_addr().unwrap());
    let contacted = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let observed = contacted.clone();
    let (stop, mut stopping) = oneshot::channel();
    let target = tokio::spawn(async move {
        tokio::select! {
            biased;
            accepted = untrusted.accept() => {
                let _ = accepted.unwrap();
                observed.store(true, Ordering::SeqCst);
            }
            _ = &mut stopping => {},
        }
    });
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/expert", listener.local_addr().unwrap());
    let interface = format!("{endpoint}/v1");
    let redirector = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let (mut socket, request) = read_request(socket).await.unwrap();
        assert_eq!(request.method, "GET");
        let card = serde_json::to_vec(&json!({"name": "expert", "version": "1",
            "supportedInterfaces": [{"url": interface, "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
            "capabilities": {"streaming": false}})).unwrap();
        respond(socket.get_mut(), "application/json", &card).await.unwrap();
        let (socket, _) = listener.accept().await.unwrap();
        let (mut socket, request) = read_request(socket).await.unwrap();
        assert_eq!(request.method, "POST");
        socket.get_mut().write_all(format!("HTTP/1.1 303 See Other\r\nLocation: {destination}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").as_bytes()).await.unwrap();
        socket.get_mut().shutdown().await.unwrap();
    });
    let client = AgentClient::new(
        &env(Arc::new(RealClock::new())),
        endpoint,
        Some("private-token".into()),
        ApiConnectOptions { max_retry: 0, ..Default::default() },
    )
    .unwrap();
    let error = client
        .send(message("private", "Private caller history"), None, None, None, &Cx::NONE)
        .await
        .err()
        .unwrap();
    assert!(
        matches!(error, ClientError::Transport(ref error) if error.kind == ApiErrorKind::Connection)
    );
    client.aclose().await;
    redirector.await.unwrap();
    let _ = stop.send(());
    target.await.unwrap();
    assert!(!contacted.load(Ordering::SeqCst), "redirect crossed the authority boundary");
}
