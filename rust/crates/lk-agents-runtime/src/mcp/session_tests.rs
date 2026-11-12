#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "tests drive tokio's sync primitives directly"
)]
use super::*;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::invoke::CallContext;
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
    /// Tool list pages served.
    lists: AtomicUsize,
}

/// The server: each connection attempt is a new peer.
#[derive(Default)]
pub(crate) struct Server {
    peers: parking_lot::Mutex<Vec<Arc<Peer>>>,
    /// Handshakes never answered while set.
    silent: AtomicBool,
    /// Connections refused while set.
    pub(crate) refuse: AtomicBool,
    attempts: AtomicUsize,
}

impl Server {
    fn peer(&self) -> Arc<Peer> {
        self.peers.lock().last().unwrap().clone()
    }
    pub(crate) fn connect(self: &Arc<Self>) -> FakeTransport {
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
            lists: AtomicUsize::new(0),
        });
        self.peers.lock().push(peer.clone());
        FakeTransport { peer, refuse: self.refuse.load(Ordering::SeqCst) }
    }
}

pub(crate) struct FakeTransport {
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
                self.lists.fetch_add(1, Ordering::SeqCst);
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
                    "change_tools" => {
                        let changed = ServerNotification::ToolListChangedNotification(
                            ToolListChangedNotification::default(),
                        );
                        let notification = ServerJsonRpcMessage::Notification(JsonRpcNotification {
                            jsonrpc: JsonRpcVersion2_0,
                            notification: changed,
                        });
                        let _ = self.send.send(notification).await;
                        text("changed")
                    }
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

// Catches: a server's `tools/list_changed` ignored, so a tool it enables stays invisible until
// the idle close; or every listing fetching again once it was seen.
#[tokio::test]
async fn a_tools_changed_notification_makes_the_next_listing_fetch_again() {
    let (session, server) = connected().await;
    session.list_tools().await.unwrap();
    let peer = server.peer();
    assert_eq!(peer.lists.load(Ordering::SeqCst), 2);
    session.list_tools().await.unwrap();
    assert_eq!(peer.lists.load(Ordering::SeqCst), 2, "an unchanged listing is cached");
    assert!(!session.tools_changed());

    call(&session, "change_tools").await.unwrap();
    tokio::time::timeout(Duration::from_secs(1), async {
        while !session.tools_changed() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the notification is seen");
    session.list_tools().await.unwrap();
    assert_eq!(peer.lists.load(Ordering::SeqCst), 4, "both pages fetched again");
    assert!(!session.tools_changed());
    session.list_tools().await.unwrap();
    assert_eq!(peer.lists.load(Ordering::SeqCst), 4, "and cached again");
    session.close().await;
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
        _: &'a CallContext,
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

// Catches: a call without a limit to a server that never replies outliving the session that
// made it.
#[tokio::test]
async fn an_unlimited_call_ends_with_its_session() {
    let (session, server) = connected_with_timeout(None).await;
    let caller = session.clone();
    let call = tokio::spawn(async move { super::tests::call(&caller, "wait").await });
    server.peer().called.notified().await;
    session.close().await;
    let ended = tokio::time::timeout(Duration::from_secs(1), call).await;
    assert!(matches!(ended, Ok(Ok(Err(McpError::Cancelled(_))))), "{ended:?}");
}

mod lifecycle {
    // Runs tests/host/mcp_peer.py with MCP_TEST_PYTHON (default python3), which needs
    // the MCP SDK, FastMCP and uvicorn: the repository's dev environment.
    use std::error::Error;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::Duration;

    use crate::mcp::McpSession;
    use lk_agents_abi::llm::mcp::{McpError, McpSession as _};
    use lk_agents_net::env::ProcessEnv;
    use serde_json::Map;

    type Result<T = ()> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

    struct Peer {
        directory: PathBuf,
        python: String,
        script: String,
    }
    impl Peer {
        fn new(mode: &str) -> Self {
            let directory =
                std::env::temp_dir().join(format!("lk-mcp-{}-{mode}", std::process::id()));
            std::fs::create_dir_all(&directory).unwrap();
            Self {
                directory,
                python: std::env::var("MCP_TEST_PYTHON").unwrap_or_else(|_| "python3".into()),
                script: Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../../tests/host/mcp_peer.py")
                    .display()
                    .to_string(),
            }
        }
        fn args(&self, mode: &str) -> Vec<String> {
            vec![self.script.clone(), mode.into(), self.directory.display().to_string()]
        }
        /// The file `name` once it exists and `ready` accepts its text.
        async fn receipt_when(&self, name: &str, ready: impl Fn(&str) -> bool) -> Result<String> {
            Ok(tokio::time::timeout(Duration::from_secs(10), async {
                loop {
                    match std::fs::read_to_string(self.directory.join(name)) {
                        Ok(text) if ready(&text) => return Ok::<_, std::io::Error>(text),
                        Ok(_) => tokio::time::sleep(Duration::from_millis(10)).await,
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                            tokio::time::sleep(Duration::from_millis(10)).await
                        }
                        Err(error) => return Err(error),
                    }
                }
            })
            .await??)
        }
    }
    impl Drop for Peer {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.directory);
        }
    }

    async fn exercise(session: Arc<McpSession>, peer: &Peer) -> Result {
        let tools = session.list_tools().await?;
        let names: Vec<_> = tools.as_array().unwrap().iter().map(|tool| &tool["name"]).collect();
        assert!(names.contains(&&"echo".into()), "first page: {names:?}");
        assert!(names.contains(&&"exit".into()), "second page: {names:?}");
        let error =
            session.call_tool("reject".into(), Map::new(), None, None, None).await.unwrap_err();
        match &error {
            McpError::Protocol { code, data, cause, .. } => {
                assert_eq!(*code, -32602);
                assert_eq!(data.as_ref().unwrap()["field"], "seat");
                assert!(cause.downcast_ref::<rmcp::ServiceError>().is_some());
            }
            other => panic!("lost MCP error taxonomy: {other:?}"),
        }
        let caller = session.clone();
        let call = tokio::spawn(async move {
            caller.call_tool("wait".into(), Map::new(), None, None, None).await
        });
        peer.receipt_when("journal.jsonl", |text| text.contains(r#""event": "entered""#)).await?;
        session.close().await;
        assert!(
            matches!(call.await?, Err(McpError::Cancelled(_))),
            "borrowed request must settle on close"
        );
        assert!(!session.connected());
        Ok(())
    }

    #[tokio::test]
    async fn mcp_real_server_lifecycle_stdio() -> Result {
        let peer = Peer::new("stdio");
        let session = Arc::new(McpSession::new(5.0, Some(30.0), Some(30.0))?);
        let result = async {
            session.connect_stdio(peer.python.clone(), peer.args("stdio"), None, None).await?;
            exercise(session.clone(), &peer).await?;
            let pid: i32 = peer.receipt_when("pid", |_| true).await?.parse()?;
            #[cfg(unix)]
            assert!(
                !tokio::process::Command::new("kill")
                    .args(["-0", &pid.to_string()])
                    .output()
                    .await?
                    .status
                    .success()
            );
            Ok(())
        }
        .await;
        session.close().await;
        result
    }

    #[tokio::test]
    async fn mcp_real_server_lifecycle_http() -> Result {
        let peer = Peer::new("http");
        let mut child = tokio::process::Command::new(&peer.python)
            .args(peer.args("http"))
            .kill_on_drop(true)
            .spawn()?;
        let session = Arc::new(McpSession::new(5.0, Some(30.0), Some(30.0))?);
        let result = async {
            let url = peer.receipt_when("url", |_| true).await?;
            session
                .connect_http(&ProcessEnv::current_runtime(), url, Default::default(), None)
                .await?;
            exercise(session.clone(), &peer).await
        }
        .await;
        session.close().await;
        child.kill().await?;
        child.wait().await?;
        result
    }
}
