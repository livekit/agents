use super::*;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::llm::mcp::McpInvocation;
use lk_agents_abi::llm::mcp::McpSession as _;
use rmcp::model::*;
use serde_json::Value;
use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::{Mutex, Notify, mpsc};

struct Peer {
    replies: Mutex<mpsc::Receiver<Value>>,
    send: mpsc::Sender<Value>,
    called: Notify,
    receiving: Arc<Notify>,
    dropped: Arc<Notify>,
    block_close: AtomicBool,
    release_close: Notify,
    fail_receive: Notify,
}

struct Reading(Arc<Notify>);
impl Drop for Reading {
    fn drop(&mut self) {
        self.0.notify_one();
    }
}

#[async_trait::async_trait]
impl McpTransport for Peer {
    async fn start(&self, _: &McpInvocation) -> Result<(), HostError> {
        Ok(())
    }
    async fn send(&self, _: &McpInvocation, m: &Value) -> Result<(), HostError> {
        let message: ClientJsonRpcMessage = serde_json::from_value(m.clone()).unwrap();
        let ClientJsonRpcMessage::Request(request) = message else { return Ok(()) };
        let progress_token = request.request.get_meta().get_progress_token();
        let result = match request.request {
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
                let mut page = ListToolsResult::default();
                page.tools = vec![Tool::new(
                    if second { "second" } else { "first" },
                    "test tool",
                    serde_json::Map::new(),
                )];
                page.next_cursor = (!second).then(|| "second".into());
                ServerResult::ListToolsResult(page)
            }
            ClientRequest::CallToolRequest(request) => {
                self.called.notify_one();
                if request.params.name == "fail" {
                    self.fail_receive.notify_one();
                }
                if request.params.name == "progress" {
                    let token = progress_token.unwrap();
                    let mut params = ProgressNotificationParam::new(token, 1.0);
                    params.message = Some("working".into());
                    let message = ServerJsonRpcMessage::Notification(JsonRpcNotification {
                        jsonrpc: JsonRpcVersion2_0,
                        notification: ServerNotification::ProgressNotification(
                            ProgressNotification::new(params),
                        ),
                    });
                    self.send.send(serde_json::to_value(message).unwrap()).await.unwrap();
                }
                return Ok(());
            }
            _ => return Ok(()),
        };
        let reply = ServerJsonRpcMessage::Response(JsonRpcResponse {
            jsonrpc: JsonRpcVersion2_0,
            id: request.id,
            result,
        });
        self.send.send(serde_json::to_value(reply).unwrap()).await.unwrap();
        Ok(())
    }
    async fn receive(&self, _: &McpInvocation) -> Result<Value, HostError> {
        let _reading = Reading(self.dropped.clone());
        self.receiving.notify_one();
        tokio::select! {
            _ = self.fail_receive.notified() => Err(HostError::new("reader connection reset")),
            reply = async { self.replies.lock().await.recv().await } => Ok(reply.unwrap_or(Value::Null)),
        }
    }
    async fn close(&self, _: &McpInvocation) -> Result<(), HostError> {
        if self.block_close.load(Ordering::SeqCst) {
            self.release_close.notified().await;
        }
        Ok(())
    }
}

async fn connected() -> (Arc<McpSession>, Arc<Peer>) {
    let (send, replies) = mpsc::channel(16);
    let peer = Arc::new(Peer {
        replies: Mutex::new(replies),
        send,
        called: Notify::new(),
        receiving: Arc::default(),
        dropped: Arc::default(),
        block_close: AtomicBool::new(false),
        release_close: Notify::new(),
        fail_receive: Notify::new(),
    });
    let session = Arc::new(McpSession::new(30.0));
    session.connect_host(peer.clone(), None).await.unwrap();
    (session, peer)
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
    let (session, peer) = connected().await;
    let caller = session.clone();
    let mut call = tokio::spawn(async move {
        caller.call_tool("wait".into(), Default::default(), None, None).await
    });
    peer.called.notified().await;
    session.close().await;
    let result = tokio::time::timeout(Duration::from_millis(100), &mut call).await;
    if result.is_err() {
        call.abort();
        let _ = call.await;
    }
    assert!(result.is_ok(), "close returned with the borrowed call still pending");
}

#[tokio::test]
async fn close_joins_the_blocked_host_reader() {
    let (session, peer) = connected().await;
    peer.receiving.notified().await;
    // Consume receive completions from the handshake before waiting for the blocked read.
    while tokio::time::timeout(Duration::from_millis(1), peer.dropped.notified()).await.is_ok() {}
    session.close().await;
    assert!(
        tokio::time::timeout(Duration::from_millis(100), peer.dropped.notified()).await.is_ok(),
        "close left the host receive task alive"
    );
}

#[test]
fn result_renderer_uses_upstream_content_records() {
    let result =
        rmcp::model::CallToolResult::success(vec![rmcp::model::ContentBlock::text("hello")]);
    let expected = serde_json::to_string(&result.content[0]).unwrap();
    assert_eq!(
        super::render::render_result("echo", &serde_json::to_value(result).unwrap()).unwrap(),
        expected
    );
}

#[tokio::test(start_paused = true)]
async fn missed_cleanup_deadline_keeps_ownership_and_never_reports_clean() {
    let (session, peer) = connected().await;
    peer.block_close.store(true, Ordering::SeqCst);
    session.close().await;
    let cleanup = session.cleanup();
    assert_eq!(cleanup.status, McpCleanupStatus::Incomplete);
    assert!(!cleanup.service_settled);
    peer.release_close.notify_one();
    session.close().await;
    let cleanup = session.cleanup();
    assert!(cleanup.service_settled && cleanup.reader_settled && cleanup.calls_settled);
    assert_eq!(
        cleanup.status,
        McpCleanupStatus::Incomplete,
        "settlement must not erase a missed deadline"
    );
}

#[tokio::test]
async fn reader_failure_reaches_the_call_with_its_host_cause() {
    let (session, _) = connected().await;
    let error = session.call_tool("fail".into(), Default::default(), None, None).await.unwrap_err();
    assert!(error.source().is_some());
    let McpError::Transport(cause) = &error else { panic!("lost transport taxonomy: {error:?}") };
    let Some(McpError::Host(cause)) = cause.downcast_ref::<McpError>() else {
        panic!("lost host cause: {cause:?}")
    };
    assert_eq!(cause.message, "reader connection reset");
    session.close().await;
    assert_eq!(session.cleanup().status, McpCleanupStatus::Clean);
}

struct BlockedProgress(Notify);
#[async_trait::async_trait]
impl McpProgress for BlockedProgress {
    async fn update(&self, _: &McpInvocation, _: &str) -> Result<(), HostError> {
        self.0.notify_one();
        std::future::pending().await
    }
}

#[tokio::test]
async fn cancelling_a_host_progress_callback_without_a_receipt_is_incomplete() {
    let (session, _) = connected().await;
    let sink = Arc::new(BlockedProgress(Notify::new()));
    let caller = session.clone();
    let progress = sink.clone();
    let call = tokio::spawn(async move {
        caller.call_tool("progress".into(), Default::default(), Some(progress), None).await
    });
    tokio::time::timeout(Duration::from_secs(1), sink.0.notified()).await.unwrap();
    session.close().await;
    assert!(matches!(call.await.unwrap(), Err(McpError::Cancelled(_))));
    assert_eq!(session.cleanup().status, McpCleanupStatus::Incomplete);
}
