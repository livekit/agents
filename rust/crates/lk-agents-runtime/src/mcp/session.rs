//! The SDK service has one owner; requests borrow its peer, never its join handle.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::StreamExt;
use lk_agents_abi::llm::mcp::{McpInvocation, McpSession as McpSessionPort, timeout_duration};
use lk_agents_abi::utils::host_ref::HostRef;
use lk_agents_abi::utils::stream::HOST_TEARDOWN_TIMEOUT;
use rmcp::handler::client::progress::ProgressDispatcher;
use rmcp::model::{
    CallToolRequest, CallToolRequestParams, ClientRequest, PaginatedRequestParams,
    ProgressNotificationParam, ServerResult, Tool,
};
use rmcp::service::serve_client_with_ct;
use rmcp::service::{
    NotificationContext, PeerRequestOptions, RoleClient, RunningService, ServiceError,
};
use rmcp::transport::Transport;
use rmcp::{ClientHandler, Peer};
use serde_json::{Map, Value};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use super::transport::{HostStreams, Observed, StdioTransport, TransportReceipt, connect_http};
use super::{McpCleanup, McpError, McpProgress, McpTransport};

#[derive(Clone, Default)]
struct Progress(ProgressDispatcher);
impl ClientHandler for Progress {
    async fn on_progress(
        &self,
        params: ProgressNotificationParam,
        _: NotificationContext<RoleClient>,
    ) {
        self.0.handle_notification(params).await;
    }
}

struct Connection {
    peer: Peer<RoleClient>,
    progress: Progress,
    cancel: CancellationToken,
    service: Mutex<Option<RunningService<RoleClient, Progress>>>,
    state: Mutex<ConnectionState>,
    changed: Notify,
    receipt: Arc<TransportReceipt>,
}

#[derive(Default)]
struct ConnectionState {
    closing: bool,
    calls: usize,
    settled: bool,
    incomplete: bool,
    service_settled: bool,
}

struct Call {
    connection: Arc<Connection>,
    host_callback_pending: bool,
}
impl Drop for Call {
    fn drop(&mut self) {
        let mut state = self.connection.state.lock().unwrap();
        state.calls -= 1;
        state.incomplete |= self.host_callback_pending;
        drop(state);
        if self.host_callback_pending {
            log::error!("MCP host callback cancellation lacks a settlement receipt");
        }
        self.connection.changed.notify_waiters();
    }
}

impl Connection {
    fn borrow(self: &Arc<Self>) -> Result<Call, McpError> {
        let mut state = self.state.lock().unwrap();
        if state.closing {
            return Err(McpError::Unavailable);
        }
        state.calls += 1;
        Ok(Call { connection: self.clone(), host_callback_pending: false })
    }

    fn close(self: &Arc<Self>) {
        self.state.lock().unwrap().closing = true;
        self.cancel.cancel();
        let Some(service) = self.service.lock().unwrap().take() else { return };
        let owner = self.clone();
        // The owner outlives a cancelled close waiter or a missed cleanup deadline.
        tokio::spawn(async move {
            let joined = service.waiting().await;
            owner.receipt.closed().await;
            {
                let mut state = owner.state.lock().unwrap();
                state.service_settled = matches!(
                    joined,
                    Ok(rmcp::service::QuitReason::Cancelled | rmcp::service::QuitReason::Closed)
                );
                state.incomplete |= !state.service_settled || owner.receipt.failed();
            }
            loop {
                let changed = owner.changed.notified();
                if owner.state.lock().unwrap().calls == 0 {
                    break;
                }
                changed.await;
            }
            owner.state.lock().unwrap().settled = true;
            owner.changed.notify_waiters();
        });
    }

    async fn settled(&self) {
        loop {
            let changed = self.changed.notified();
            if self.state.lock().unwrap().settled {
                return;
            }
            changed.await;
        }
    }

    fn error(&self, error: ServiceError) -> McpError {
        self.receipt
            .failure()
            .map_or_else(|| service_error(error), |cause| McpError::Transport(cause))
    }

    fn cleanup(&self) -> McpCleanup {
        let state = self.state.lock().unwrap();
        let mut receipt = self.receipt.cleanup();
        receipt.service_settled = state.service_settled;
        receipt.calls_settled = state.calls == 0;
        receipt.status = lk_agents_core::mcp::cleanup_status(
            state.incomplete || self.receipt.failed(),
            state.settled,
            receipt.reader_settled,
        );
        receipt
    }
}

pub struct McpSession {
    timeout: Duration,
    connection: Mutex<Option<Arc<Connection>>>,
    previous: Mutex<Option<Arc<Connection>>>,
    initializing: Mutex<Option<Arc<TransportReceipt>>>,
    connecting: tokio::sync::Mutex<()>,
    cancel: Mutex<CancellationToken>,
    cache: Mutex<Option<Vec<Tool>>>,
    allowed: Mutex<Vec<String>>,
}

impl McpSessionPort for McpSession {
    fn new(client_session_timeout_seconds: f64) -> Result<Self, McpError> {
        Ok(Self {
            timeout: timeout_duration(
                "client_session_timeout_seconds",
                client_session_timeout_seconds,
            )?,
            connection: Mutex::default(),
            previous: Mutex::default(),
            initializing: Mutex::default(),
            connecting: tokio::sync::Mutex::default(),
            cancel: Mutex::new(CancellationToken::new()),
            cache: Mutex::default(),
            allowed: Mutex::default(),
        })
    }
    fn connected(&self) -> bool {
        self.connection
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|c| !c.cancel.is_cancelled() && !c.peer.is_transport_closed())
    }
    fn set_allowed_tools(&self, allowed: Vec<String>) {
        *self.allowed.lock().unwrap() = allowed;
    }
    fn invalidate_cache(&self) {
        *self.cache.lock().unwrap() = None;
    }

    async fn connect_http(
        &self,
        url: String,
        headers: Vec<(String, String)>,
        transport: String,
        timeout: f64,
        sse_read_timeout: f64,
    ) -> Result<(), McpError> {
        let timeout = timeout_duration("timeout", timeout)?;
        let sse_read_timeout = timeout_duration("sse_read_timeout", sse_read_timeout)?;
        let _connecting = self.connecting.lock().await;
        if self.connected() {
            return Ok(());
        }
        let cancel = self.begin_connect();
        let receipt = Arc::new(TransportReceipt::default());
        let transport = tokio::select! {
            _ = cancel.cancelled() => return Err(cancelled()),
            transport = connect_http(url, headers, &transport, timeout, sse_read_timeout, receipt.clone()) => transport?,
        };
        self.serve(transport, receipt, cancel).await
    }
    async fn connect_stdio(
        &self,
        command: String,
        args: Vec<String>,
        env: Vec<(String, String)>,
        cwd: Option<String>,
    ) -> Result<(), McpError> {
        let _connecting = self.connecting.lock().await;
        if self.connected() {
            return Ok(());
        }
        let cancel = self.begin_connect();
        let receipt = Arc::new(TransportReceipt::default());
        let transport = StdioTransport::spawn(command, args, env, cwd, receipt.clone())?;
        self.serve(transport, receipt, cancel).await
    }
    async fn connect_host(
        &self,
        host: Arc<dyn McpTransport>,
        host_context: Option<HostRef>,
    ) -> Result<(), McpError> {
        let _connecting = self.connecting.lock().await;
        if self.connected() {
            return Ok(());
        }
        let cancel = self.begin_connect();
        let invocation = McpInvocation { host_context };
        host.start(&invocation).await?;
        let receipt = Arc::new(TransportReceipt::default());
        let transport = HostStreams::new(host, invocation, receipt.clone());
        self.serve(transport, receipt, cancel).await
    }
    async fn list_tools(&self) -> Result<Value, McpError> {
        let connection = self.live(McpError::NotInitialized)?;
        let _call = connection.borrow()?;
        let cached = self.cache.lock().unwrap().clone();
        let tools = match cached {
            Some(tools) => tools,
            None => {
                let mut tools = Vec::new();
                let mut cursor = None;
                loop {
                    let params = PaginatedRequestParams::default().with_cursor(cursor);
                    let request = ClientRequest::ListToolsRequest(
                        rmcp::model::ListToolsRequest::with_param(params),
                    );
                    let result = self.request(&connection, request).await?;
                    let ServerResult::ListToolsResult(page) = result else {
                        return Err(McpError::UnexpectedResponse);
                    };
                    tools.extend(page.tools);
                    cursor = page.next_cursor;
                    if cursor.is_none() {
                        break;
                    }
                }
                // A closing connection must not repopulate the next run's cache.
                if !connection.cancel.is_cancelled() {
                    *self.cache.lock().unwrap() = Some(tools.clone());
                }
                tools
            }
        };
        let allowed = self.allowed.lock().unwrap().clone();
        let tools: Vec<Tool> = tools
            .into_iter()
            .filter(|t| allowed.is_empty() || allowed.iter().any(|a| a == &t.name))
            .collect();
        serde_json::to_value(tools).map_err(|e| McpError::Decode(Arc::new(e)))
    }
    async fn call_tool(
        &self,
        name: String,
        arguments: Map<String, Value>,
        progress: Option<Arc<dyn McpProgress>>,
        host_context: Option<HostRef>,
    ) -> Result<Value, McpError> {
        let connection = self.live(McpError::Unavailable)?;
        let mut call = connection.borrow()?;
        let mut params = CallToolRequestParams::new(name);
        params.arguments = Some(arguments);
        let request = ClientRequest::CallToolRequest(CallToolRequest::new(params));
        let handle = tokio::select! {
            _ = connection.cancel.cancelled() => return Err(cancelled()),
            result = connection.peer.send_cancellable_request(request, self.options()) => result.map_err(|e| connection.error(e))?,
        };
        let mut updates = connection.progress.0.subscribe(handle.progress_token.clone()).await;
        let response = handle.await_response();
        tokio::pin!(response);
        let result = loop {
            tokio::select! {
                biased;
                _ = connection.cancel.cancelled() => return Err(cancelled()),
                result = &mut response => break result.map_err(|e| connection.error(e))?,
                Some(update) = updates.next() => {
                    if let (Some(sink), Some(message)) = (&progress, update.message.as_deref().filter(|m| !m.is_empty())) {
                        let invocation = McpInvocation { host_context: host_context.clone() };
                        call.host_callback_pending = true;
                        tokio::select! {
                            _ = connection.cancel.cancelled() => continue,
                            result = sink.update(&invocation, message) => {
                                call.host_callback_pending = false;
                                result?;
                            },
                        }
                    }
                }
            }
        };
        match result {
            ServerResult::CallToolResult(result) => {
                serde_json::to_value(result).map_err(|e| McpError::Decode(Arc::new(e)))
            }
            _ => Err(McpError::UnexpectedResponse),
        }
    }
    async fn close(&self) {
        self.cancel.lock().unwrap().cancel();
        let _connecting = self.connecting.lock().await;
        let connection = self.connection.lock().unwrap().take();
        if let Some(ref connection) = connection {
            connection.close();
            *self.previous.lock().unwrap() = Some(connection.clone());
        }
        self.invalidate_cache();
        let previous = self.previous.lock().unwrap().clone();
        if let Some(connection) = previous
            && tokio::time::timeout(HOST_TEARDOWN_TIMEOUT, connection.settled()).await.is_err()
        {
            connection.state.lock().unwrap().incomplete = true;
            log::error!("MCP incomplete cleanup; runtime retains ownership");
        }
        let initializing = self.initializing.lock().unwrap().clone();
        if let Some(receipt) = initializing
            && tokio::time::timeout(HOST_TEARDOWN_TIMEOUT, receipt.closed()).await.is_err()
        {
            receipt.missed_deadline();
            log::error!("MCP incomplete initialization cleanup; runtime retains ownership");
        }
    }
    fn cleanup(&self) -> McpCleanup {
        if let Some(connection) = self.previous.lock().unwrap().as_ref() {
            return connection.cleanup();
        }
        self.initializing
            .lock()
            .unwrap()
            .as_ref()
            .map_or_else(McpCleanup::default, |receipt| receipt.cleanup())
    }
}

impl McpSession {
    fn begin_connect(&self) -> CancellationToken {
        let cancel = CancellationToken::new();
        *self.cancel.lock().unwrap() = cancel.clone();
        cancel
    }
    async fn serve<T: Transport<RoleClient> + 'static>(
        &self,
        transport: T,
        receipt: Arc<TransportReceipt>,
        cancel: CancellationToken,
    ) -> Result<(), McpError> {
        let progress = Progress::default();
        *self.initializing.lock().unwrap() = Some(receipt.clone());
        let service = match serve_client_with_ct(
            progress.clone(),
            Observed::new(transport, receipt.clone()),
            cancel.clone(),
        )
        .await
        {
            Ok(service) => service,
            Err(error) => {
                // Observed owns the transport even if the SDK handshake is cancelled.
                if tokio::time::timeout(HOST_TEARDOWN_TIMEOUT, receipt.closed()).await.is_err() {
                    receipt.missed_deadline();
                    log::error!("MCP incomplete initialization cleanup; runtime retains ownership");
                }
                return Err(McpError::Initialization(Arc::new(error)));
            }
        };
        *self.initializing.lock().unwrap() = None;
        let peer = service.peer().clone();
        *self.connection.lock().unwrap() = Some(Arc::new(Connection {
            peer,
            progress,
            cancel,
            service: Mutex::new(Some(service)),
            state: Mutex::default(),
            changed: Notify::new(),
            receipt,
        }));
        Ok(())
    }
    fn live(&self, error: McpError) -> Result<Arc<Connection>, McpError> {
        self.connection.lock().unwrap().clone().ok_or(error)
    }
    fn options(&self) -> PeerRequestOptions {
        let mut options = PeerRequestOptions::no_options();
        options.timeout = Some(self.timeout);
        options
    }
    async fn request(
        &self,
        connection: &Connection,
        request: ClientRequest,
    ) -> Result<ServerResult, McpError> {
        tokio::select! {
            _ = connection.cancel.cancelled() => Err(cancelled()),
            result = async { connection.peer.send_cancellable_request(request, self.options()).await?.await_response().await } => result.map_err(|e| connection.error(e)),
        }
    }
}

impl Drop for McpSession {
    fn drop(&mut self) {
        self.cancel.get_mut().unwrap().cancel();
        if let Some(connection) = self.connection.get_mut().unwrap().take() {
            connection.close();
        }
    }
}

pub(super) fn service_error(error: ServiceError) -> McpError {
    match error {
        ServiceError::McpError(ref rpc) => McpError::Protocol {
            code: rpc.code.0,
            message: rpc.message.to_string(),
            data: rpc.data.clone(),
            cause: Arc::new(error),
        },
        ServiceError::Timeout { timeout } => McpError::Timeout { timeout, cause: Arc::new(error) },
        ServiceError::Cancelled { .. } => McpError::Cancelled(Arc::new(error)),
        _ => McpError::Transport(Arc::new(error)),
    }
}

fn cancelled() -> McpError {
    McpError::Cancelled(Arc::new(ServiceError::Cancelled { reason: Some("session closed".into()) }))
}
