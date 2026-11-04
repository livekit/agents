use super::*;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::llm::mcp::McpInvocation;
use lk_agents_abi::llm::mcp::McpSession as _;
use lk_agents_abi::utils::stream::HOST_TEARDOWN_TIMEOUT;
use rmcp::model::*;
use rmcp::service::RoleClient;
use rmcp::transport::Transport;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::{Mutex, Notify, mpsc};

/// One in-process server connection: answers the handshake, pages tools, and runs the named
/// test tools.
pub(super) struct Peer {
    replies: Mutex<mpsc::Receiver<ServerJsonRpcMessage>>,
    send: mpsc::Sender<ServerJsonRpcMessage>,
    called: Notify,
    receiving: Notify,
    dropped: Notify,
    block_close: AtomicBool,
    release_close: Notify,
    lost: Notify,
    cancelled: Notify,
    silent: bool,
}

/// The server: each connection attempt is a new peer.
#[derive(Default)]
struct Server {
    peers: parking_lot::Mutex<Vec<Arc<Peer>>>,
    /// Handshakes never answered while set.
    silent: AtomicBool,
    /// Connections refused while set.
    refuse: AtomicBool,
    attempts: AtomicUsize,
}

impl Server {
    fn peer(&self) -> Arc<Peer> {
        self.peers.lock().last().unwrap().clone()
    }
    fn connect(self: &Arc<Self>) -> FakeTransport {
        self.attempts.fetch_add(1, Ordering::SeqCst);
        let (send, replies) = mpsc::channel(16);
        let peer = Arc::new(Peer {
            replies: Mutex::new(replies),
            send,
            called: Notify::new(),
            receiving: Notify::new(),
            dropped: Notify::new(),
            block_close: AtomicBool::new(false),
            release_close: Notify::new(),
            lost: Notify::new(),
            cancelled: Notify::new(),
            silent: self.silent.load(Ordering::SeqCst),
        });
        self.peers.lock().push(peer.clone());
        FakeTransport { peer, refuse: self.refuse.load(Ordering::SeqCst) }
    }
}

pub(super) struct FakeTransport {
    peer: Arc<Peer>,
    refuse: bool,
}

struct Reading(Arc<Peer>);
impl Drop for Reading {
    fn drop(&mut self) {
        self.0.dropped.notify_one();
    }
}

fn reply(id: RequestId, result: ServerResult) -> ServerJsonRpcMessage {
    ServerJsonRpcMessage::Response(JsonRpcResponse { jsonrpc: JsonRpcVersion2_0, id, result })
}

fn progress(token: ProgressToken) -> ServerJsonRpcMessage {
    let mut params = ProgressNotificationParam::new(token, 1.0);
    params.message = Some("working".into());
    ServerJsonRpcMessage::Notification(JsonRpcNotification {
        jsonrpc: JsonRpcVersion2_0,
        notification: ServerNotification::ProgressNotification(ProgressNotification::new(params)),
    })
}

fn text(value: &str) -> ServerResult {
    ServerResult::CallToolResult(CallToolResult::success(vec![ContentBlock::text(value)]))
}

impl Peer {
    async fn handle(self: Arc<Self>, message: ClientJsonRpcMessage) {
        let request = match message {
            ClientJsonRpcMessage::Request(request) => request,
            ClientJsonRpcMessage::Notification(notification) => {
                if let ClientNotification::CancelledNotification(_) = notification.notification {
                    self.cancelled.notify_one();
                }
                return;
            }
            _ => return,
        };
        let progress_token = request.request.get_meta().get_progress_token();
        let result = match request.request {
            ClientRequest::InitializeRequest(_) if self.silent => return,
            ClientRequest::InitializeRequest(request) => {
                let mut capabilities = ServerCapabilities::default();
                capabilities.tools = Some(Default::default());
                let mut result = InitializeResult::new(capabilities);
                result.protocol_version = request.params.protocol_version;
                result.server_info = Implementation::new("regression", "1");
                ServerResult::InitializeResult(result)
            }
            ClientRequest::ListToolsRequest(request) => {
                let second = request.params.and_then(|p| p.cursor).as_deref() == Some("second");
                ServerResult::ListToolsResult(ListToolsResult {
                    tools: vec![Tool::new(
                        if second { "second" } else { "first" },
                        "test tool",
                        serde_json::Map::new(),
                    )],
                    next_cursor: (!second).then(|| "second".into()),
                    ..Default::default()
                })
            }
            ClientRequest::CallToolRequest(call) => {
                self.called.notify_one();
                match call.params.name.as_ref() {
                    "echo" => text("hello"),
                    "lose" => {
                        self.lost.notify_one();
                        return;
                    }
                    "progress" => {
                        let _ = self.send.send(progress(progress_token.unwrap())).await;
                        return;
                    }
                    "slow" => {
                        // Replies after 1.5 request timeouts, with progress at each half.
                        let (send, id, token) = (self.send.clone(), request.id, progress_token);
                        tokio::spawn(async move {
                            for _ in 0..3 {
                                tokio::time::sleep(Duration::from_millis(500)).await;
                                let _ = send.send(progress(token.clone().unwrap())).await;
                            }
                            let _ = send.send(reply(id, text("done"))).await;
                        });
                        return;
                    }
                    _ => return,
                }
            }
            _ => return,
        };
        let _ = self.send.send(reply(request.id, result)).await;
    }
}

impl Transport<RoleClient> for FakeTransport {
    type Error = std::io::Error;
    fn send(
        &mut self,
        item: ClientJsonRpcMessage,
    ) -> impl Future<Output = Result<(), std::io::Error>> + Send + 'static {
        let (peer, refuse) = (self.peer.clone(), self.refuse);
        async move {
            if refuse {
                return Err(std::io::Error::other("connection refused"));
            }
            peer.handle(item).await;
            Ok(())
        }
    }
    async fn receive(&mut self) -> Option<ServerJsonRpcMessage> {
        let _reading = Reading(self.peer.clone());
        self.peer.receiving.notify_one();
        let mut replies = self.peer.replies.lock().await;
        tokio::select! {
            _ = self.peer.lost.notified() => None,
            reply = replies.recv() => reply,
        }
    }
    async fn close(&mut self) -> Result<(), std::io::Error> {
        if self.peer.block_close.load(Ordering::SeqCst) {
            self.peer.release_close.notified().await;
        }
        Ok(())
    }
}

async fn connected() -> (Arc<McpSession>, Arc<Server>) {
    connected_with_timeout(Some(30.0)).await
}

async fn connected_with_timeout(request: Option<f64>) -> (Arc<McpSession>, Arc<Server>) {
    let server = Arc::new(Server::default());
    let session = Arc::new(McpSession::new(5.0, request, None).unwrap());
    let factory = server.clone();
    session.connect_fake(move || factory.connect()).await.unwrap();
    (session, server)
}

async fn call(session: &McpSession, name: &str) -> Result<serde_json::Value, McpError> {
    session.call_tool(name.into(), Default::default(), None, None, None).await
}

#[tokio::test]
async fn discovery_follows_the_next_cursor() {
    let (session, _) = connected().await;
    let tools = session.list_tools().await.unwrap();
    session.close().await;
    assert_eq!(tools.as_array().unwrap().len(), 2, "discovery must include the second page");
}

#[tokio::test]
async fn close_settles_a_borrowed_in_flight_call() {
    let (session, server) = connected().await;
    let caller = session.clone();
    let mut call = tokio::spawn(async move { super::tests::call(&caller, "wait").await });
    server.peer().called.notified().await;
    session.close().await;
    let result = tokio::time::timeout(Duration::from_millis(100), &mut call).await;
    if result.is_err() {
        call.abort();
        let _ = call.await;
    }
    assert!(result.is_ok(), "close returned with the borrowed call still pending");
}

#[tokio::test]
async fn close_joins_the_blocked_reader() {
    let (session, server) = connected().await;
    let peer = server.peer();
    peer.receiving.notified().await;
    // Consume receive completions from the handshake before waiting for the blocked read.
    while tokio::time::timeout(Duration::from_millis(1), peer.dropped.notified()).await.is_ok() {}
    session.close().await;
    assert!(
        tokio::time::timeout(Duration::from_millis(100), peer.dropped.notified()).await.is_ok(),
        "close left the receive task alive"
    );
}

#[tokio::test(start_paused = true)]
async fn close_returns_at_the_teardown_deadline_when_the_transport_close_blocks() {
    let (session, server) = connected().await;
    let peer = server.peer();
    peer.block_close.store(true, Ordering::SeqCst);
    let closed = tokio::time::timeout(HOST_TEARDOWN_TIMEOUT * 2, session.close()).await;
    peer.release_close.notify_one();
    assert!(closed.is_ok(), "a blocked transport close held the session's close past its deadline");
}

// Catches: a lost connection stays in place and every later call fails until the toolset closes,
// which on a call is the rest of the call.
#[tokio::test]
async fn a_lost_connection_reconnects_on_the_next_call() {
    let (session, server) = connected().await;
    assert!(call(&session, "lose").await.is_err());
    let result = call(&session, "echo").await.unwrap();
    assert_eq!(result["content"][0]["text"], "hello");
    assert_eq!(server.attempts.load(Ordering::SeqCst), 2);
    session.close().await;
}

// Catches: each call during an outage waiting out a full connect, stalling every turn.
#[tokio::test(start_paused = true)]
async fn a_failed_reconnect_fails_calls_fast_until_its_backoff_ends() {
    let (session, server) = connected().await;
    server.refuse.store(true, Ordering::SeqCst);
    assert!(call(&session, "lose").await.is_err());
    assert!(call(&session, "echo").await.is_err(), "the refused reconnect");
    let attempts = server.attempts.load(Ordering::SeqCst);
    let fast = tokio::time::timeout(Duration::from_millis(1), call(&session, "echo")).await;
    assert!(matches!(fast, Ok(Err(McpError::Unavailable))), "{fast:?}");
    assert_eq!(server.attempts.load(Ordering::SeqCst), attempts, "no attempt inside the backoff");
    assert!(session.retry_in() > 0.0);
    server.refuse.store(false, Ordering::SeqCst);
    tokio::time::sleep(Duration::from_secs_f64(session.retry_in())).await;
    assert_eq!(call(&session, "echo").await.unwrap()["content"][0]["text"], "hello");
    session.close().await;
}

// Catches: a server that accepts the connection and never answers holding an activation forever.
#[tokio::test(start_paused = true)]
async fn init_timeout_bounds_a_silent_handshake() {
    let server = Arc::new(Server::default());
    server.silent.store(true, Ordering::SeqCst);
    let session = McpSession::new(2.0, None, None).unwrap();
    let factory = server.clone();
    let result = tokio::time::timeout(
        Duration::from_secs(3),
        session.connect_fake(move || factory.connect()),
    )
    .await;
    assert!(matches!(result, Ok(Err(McpError::Timeout { .. }))), "{result:?}");
    session.close().await;
}

// Catches: a hard total timeout killing a long tool that keeps reporting progress.
#[tokio::test(start_paused = true)]
async fn progress_resets_the_request_timeout() {
    let (session, _) = connected_with_timeout(Some(1.0)).await;
    let result = call(&session, "slow").await.unwrap();
    assert_eq!(result["content"][0]["text"], "done");
    session.close().await;
}

#[tokio::test(start_paused = true)]
async fn a_per_call_timeout_replaces_the_session_limit() {
    let (session, _) = connected_with_timeout(None).await;
    let result = session.call_tool("wait".into(), Default::default(), Some(1.0), None, None).await;
    assert!(matches!(result, Err(McpError::Timeout { .. })), "{result:?}");
    session.close().await;
}

// Catches: an interrupted tool call leaving the server working on a request nobody awaits.
#[tokio::test]
async fn a_dropped_call_is_cancelled_on_the_server() {
    let (session, server) = connected().await;
    let caller = session.clone();
    let call = tokio::spawn(async move { super::tests::call(&caller, "wait").await });
    let peer = server.peer();
    peer.called.notified().await;
    call.abort();
    let _ = call.await;
    assert!(
        tokio::time::timeout(Duration::from_secs(1), peer.cancelled.notified()).await.is_ok(),
        "the server was not told the request was cancelled"
    );
    session.close().await;
}

// Catches: a server outage emptying the agent's tools though they were listed before.
#[tokio::test(start_paused = true)]
async fn listing_keeps_the_last_tools_while_the_server_is_unreachable() {
    let (session, server) = connected().await;
    assert_eq!(session.list_tools().await.unwrap().as_array().unwrap().len(), 2);
    server.refuse.store(true, Ordering::SeqCst);
    assert!(call(&session, "lose").await.is_err());
    let tools = session.list_tools().await.unwrap();
    assert_eq!(tools.as_array().unwrap().len(), 2);
    session.close().await;
}

struct BlockedProgress(Notify);
impl McpProgress for BlockedProgress {
    fn update<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a McpInvocation,
        _: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async move {
            self.0.notify_one();
            std::future::pending().await
        })
    }
}

#[tokio::test]
async fn close_cancels_a_call_waiting_on_its_host_progress_callback() {
    let (session, _) = connected().await;
    let sink = Arc::new(BlockedProgress(Notify::new()));
    let caller = session.clone();
    let progress = sink.clone();
    let call = tokio::spawn(async move {
        caller.call_tool("progress".into(), Default::default(), None, Some(progress), None).await
    });
    tokio::time::timeout(Duration::from_secs(1), sink.0.notified()).await.unwrap();
    session.close().await;
    assert!(matches!(call.await.unwrap(), Err(McpError::Cancelled(_))));
}

#[tokio::test(start_paused = true)]
async fn a_zero_request_timeout_expires_instead_of_disabling_the_limit() {
    let (session, _) = connected_with_timeout(Some(0.0)).await;
    let result = tokio::time::timeout(Duration::from_secs(1), call(&session, "wait")).await;
    session.close().await;
    assert!(
        matches!(result, Ok(Err(McpError::Timeout { timeout, .. })) if timeout.is_zero()),
        "zero timeout left the request pending: {result:?}"
    );
}
