use crate::telemetry;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::{Clock, SpawnerExt, timeout};
use lk_agents_net::owned::Background;
use lk_agents_net::signal::Shared;
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures_util::StreamExt;
use lk_agents_abi::invoke::CallContext;
use lk_agents_abi::llm::mcp::{McpSession as McpSessionPort, McpTimeout};
use lk_agents_abi::utils::host_ref::HostRef;
use lk_agents_abi::utils::stream::HOST_TEARDOWN_TIMEOUT;
use lk_agents_core::mcp::timeout_duration;
use lk_agents_core::mcp::{
    ConnectionSettlement, ListStep, ReconnectGate, ToolListing, TransportSettlement,
    progress_message,
};
use lk_agents_net::env::ProcessEnv;
use rmcp::handler::client::progress::ProgressDispatcher;
use rmcp::model::{
    CallToolRequest, CallToolRequestParams, CancelledNotification, CancelledNotificationParam,
    ClientRequest, PaginatedRequestParams, ProgressNotificationParam, RequestId, ServerResult,
};
use rmcp::service::serve_client_with_ct;
use rmcp::service::{
    NotificationContext, PeerRequestOptions, QuitReason, RoleClient, RunningService, ServiceError,
};
use rmcp::transport::Transport;
use rmcp::{ClientHandler, Peer};
use serde_json::{Map, Value};
use tokio_util::sync::CancellationToken;

use super::transport::{Observed, StdioTransport, TransportReceipt, connect_http};
use super::{McpError, McpProgress};

// Requests borrow the SDK service's peer, never its join handle, which has one owner.

/// The `notifications/cancelled` reason for a request its caller stopped awaiting.
const CALLER_CANCELLED: &str = "cancelled by the caller";
/// The cancellation reason of a request the session's close ended.
const SESSION_CLOSED: &str = "session closed";

/// Hands a tool call's progress message to the host's sink. The connection's cancel abandons
/// the update; the call's next turn then reports the cancel.
async fn forward_progress(
    connection: &Connection,
    sink: Option<&Arc<dyn McpProgress>>,
    host_context: &Option<HostRef>,
    update: ProgressNotificationParam,
) -> Result<(), McpError> {
    let (Some(sink), Some(message)) = (sink, progress_message(update.message.as_deref())) else {
        return Ok(());
    };
    let invocation = CallContext { host_context: host_context.clone(), ..CallContext::default() };
    tokio::select! {
        _ = connection.cancel.cancelled() => {}
        result = sink.update(&telemetry::Cx::NONE, &invocation, message) => result?,
    }
    Ok(())
}

/// What one connection's server sends unasked: progress for its calls, and word that its tools
/// changed.
#[derive(Clone, Default)]
struct Client {
    progress: ProgressDispatcher,
    /// Set by `notifications/tools/list_changed`; the next listing fetches again.
    tools_changed: Arc<AtomicBool>,
}
impl ClientHandler for Client {
    async fn on_progress(
        &self,
        params: ProgressNotificationParam,
        _: NotificationContext<RoleClient>,
    ) {
        self.progress.handle_notification(params).await;
    }
    async fn on_tool_list_changed(&self, _: NotificationContext<RoleClient>) {
        self.tools_changed.store(true, Ordering::SeqCst);
    }
}

/// How the session connects, kept so that it can connect again.
#[derive(Clone)]
enum Target {
    Http {
        env: ProcessEnv,
        url: String,
        headers: Vec<(String, String)>,
        auth: Option<String>,
    },
    Stdio {
        command: String,
        args: Vec<String>,
        env: Vec<(String, String)>,
        cwd: Option<String>,
    },
    #[cfg(test)]
    Fake(Arc<dyn Fn() -> tests::FakeTransport + Send + Sync>),
}

impl Target {
    /// The same server with the same credentials.
    fn same_as(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Http { url, headers, auth, .. },
                Self::Http { url: other_url, headers: other_headers, auth: other_auth, .. },
            ) => url == other_url && headers == other_headers && auth == other_auth,
            (
                Self::Stdio { command, args, env, cwd },
                Self::Stdio { command: c, args: a, env: e, cwd: d },
            ) => command == c && args == a && env == e && cwd == d,
            _ => false,
        }
    }
}

struct Connection {
    peer: Peer<RoleClient>,
    /// The session's: runs the close's settlement and the cancel notifications.
    background: Background,
    client: Client,
    cancel: CancellationToken,
    service: Mutex<Option<RunningService<RoleClient, Client>>>,
    state: Shared<ConnectionSettlement>,
    receipt: Arc<TransportReceipt>,
    /// The close's settlement, which holds the connection until it ends.
    closing: std::sync::OnceLock<lk_agents_abi::OwnedTask>,
}

struct Call(Arc<Connection>);
impl Drop for Call {
    fn drop(&mut self) {
        self.0.state.modify(ConnectionSettlement::release);
    }
}

/// A request the caller stopped awaiting, an interruption among them, is cancelled on the server
/// by a notification the session's background sends.
struct CancelOnDrop<'a> {
    connection: &'a Connection,
    id: Option<RequestId>,
}
impl Drop for CancelOnDrop<'_> {
    fn drop(&mut self) {
        let Some(id) = self.id.take() else { return };
        let peer = self.connection.peer.clone();
        self.connection.background.spawn("MCP cancel notification", async move {
            let reason = Some(CALLER_CANCELLED.into());
            let notification =
                CancelledNotification::new(CancelledNotificationParam::new(Some(id), reason));
            if let Err(error) = peer.send_notification(notification.into()).await {
                log::warn!(target: lk_agents_abi::LOG_TARGET, error:% = error; "MCP notification could not be sent");
            }
        });
    }
}

impl Connection {
    fn alive(&self) -> bool {
        !self.cancel.is_cancelled() && !self.peer.is_transport_closed()
    }

    fn borrow(self: &Arc<Self>) -> Result<Call, McpError> {
        self.state.with(ConnectionSettlement::borrow)?;
        Ok(Call(self.clone()))
    }

    fn close(self: &Arc<Self>) {
        self.state.modify(ConnectionSettlement::close);
        self.cancel.cancel();
        let Some(service) = self.service.lock().take() else { return };
        let owner = self.clone();
        // The owner outlives a cancelled close waiter or a missed cleanup deadline.
        let closing = self.background.spawner.spawn(async move {
            match service.waiting().await {
                Ok(QuitReason::JoinError(error)) | Err(error) => {
                    log::warn!(target: lk_agents_abi::LOG_TARGET, "MCP service ended with a failure: {error}");
                }
                Ok(_) => {}
            }
            owner.receipt.closed().await;
            owner.state.wait_until(|state| state.settle_when_idle().then_some(())).await;
        });
        let _ = self.closing.set(closing);
    }

    async fn settled(&self) {
        self.state.wait_until(|state| state.settled().then_some(())).await;
    }

    fn error(&self, error: ServiceError) -> McpError {
        self.receipt
            .read(TransportSettlement::failure)
            .map_or_else(|| service_error(error), |cause| McpError::Transport(cause))
    }
}

pub struct McpSession {
    init_timeout: Duration,
    request_timeout: Option<Duration>,
    read_timeout: Option<Duration>,
    state: Mutex<State>,
    /// The host's: the reconnect gate's time, the session's waits and its tasks.
    background: Background,
    /// Serializes connection attempts and teardown, including their awaits.
    connecting: tokio::sync::Mutex<()>,
}

#[derive(Default)]
struct State {
    target: Option<Target>,
    /// Set by `close`, cleared by a `connect_*`: a closed session never reconnects by itself.
    closed: bool,
    gate: ReconnectGate,
    connection: Option<Arc<Connection>>,
    previous: Option<Arc<Connection>>,
    initializing: Option<Arc<TransportReceipt>>,
    cancel: CancellationToken,
    listing: ToolListing,
}

impl McpSessionPort for McpSession {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        init_timeout: f64,
        request_timeout: Option<f64>,
        read_timeout: Option<f64>,
    ) -> Result<Self, McpError> {
        let limit = |field, seconds: Option<f64>| {
            seconds.map(|seconds| timeout_duration(field, seconds)).transpose()
        };
        Ok(Self {
            init_timeout: timeout_duration(McpTimeout::Init, init_timeout)?,
            request_timeout: limit(McpTimeout::Request, request_timeout)?,
            read_timeout: limit(McpTimeout::Read, read_timeout)?,
            state: Mutex::default(),
            background: env.background().map_err(|e| HostError::new(e.to_string()))?,
            connecting: tokio::sync::Mutex::default(),
        })
    }
    fn connected(&self) -> bool {
        self.live().is_some()
    }
    fn retry_in(&self) -> f64 {
        self.retry_delay().as_secs_f64()
    }

    async fn connect_http(
        &self,
        env: &ProcessEnv,
        url: String,
        headers: BTreeMap<String, String>,
        auth: Option<String>,
    ) -> Result<(), McpError> {
        let headers = headers.into_iter().collect();
        self.connect_to(Target::Http { env: env.clone(), url, headers, auth }).await
    }
    async fn connect_stdio(
        &self,
        command: String,
        args: Vec<String>,
        env: Option<BTreeMap<String, String>>,
        cwd: Option<String>,
    ) -> Result<(), McpError> {
        let env = env.unwrap_or_default().into_iter().collect();
        self.connect_to(Target::Stdio { command, args, env, cwd }).await
    }
    async fn list_tools(&self) -> Result<Value, McpError> {
        // A connection that answered but could not list counts against the gate, as a failed
        // connect already does, so that retries back off.
        let mut reached = false;
        let fetch = async {
            let connection = self.ensure(McpError::NotInitialized).await?;
            reached = true;
            let _call = connection.borrow()?;
            let mut step = {
                let mut state = self.state.lock();
                if connection.client.tools_changed.swap(false, Ordering::SeqCst) {
                    state.listing.refresh();
                }
                state.listing.list()
            };
            loop {
                let (pages, cursor) = match step {
                    ListStep::Done(tools) => return Ok(tools),
                    ListStep::Fetch { pages, cursor } => (pages, cursor),
                };
                let params = PaginatedRequestParams::default().with_cursor(cursor);
                let request = ClientRequest::ListToolsRequest(
                    rmcp::model::ListToolsRequest::with_param(params),
                );
                let ServerResult::ListToolsResult(page) =
                    self.request(&connection, request).await?
                else {
                    return Err(McpError::UnexpectedResponse);
                };
                let tools = page
                    .tools
                    .into_iter()
                    .map(serde_json::to_value)
                    .collect::<Result<_, _>>()
                    .map_err(|e| McpError::Decode(Arc::new(e)))?;
                let closing = connection.cancel.is_cancelled();
                step = self.state.lock().listing.page(pages, tools, page.next_cursor, closing);
            }
        };
        let result = match timeout(self.clock(), self.init_timeout, fetch).await {
            Ok(result) => result,
            Err(_) => Err(timed_out(self.init_timeout)),
        };
        if result.is_err() && reached {
            self.state.lock().gate.failed(self.clock().now());
        }
        match result {
            Ok(tools) => Ok(tools.into()),
            Err(error) => match self.state.lock().listing.last_known() {
                Some(tools) if !matches!(error, McpError::NotInitialized) => {
                    log::warn!(target: lk_agents_abi::LOG_TARGET, "MCP tool listing failed, using the last listing: {error}");
                    Ok(tools.into())
                }
                _ => Err(error),
            },
        }
    }
    async fn call_tool(
        &self,
        name: String,
        arguments: Map<String, Value>,
        timeout: Option<f64>,
        progress: Option<Arc<dyn McpProgress>>,
        host_context: Option<HostRef>,
    ) -> Result<Value, McpError> {
        let timeout = match timeout {
            Some(seconds) => Some(timeout_duration(McpTimeout::Tool, seconds)?),
            None => self.request_timeout,
        };
        let connection = self.ensure(McpError::Unavailable).await?;
        let _call = connection.borrow()?;
        let mut params = CallToolRequestParams::new(name);
        params.arguments = Some(arguments);
        let request = ClientRequest::CallToolRequest(CallToolRequest::new(params));
        let handle = tokio::select! {
            _ = connection.cancel.cancelled() => return Err(cancelled()),
            result = connection.peer.send_cancellable_request(request, options(timeout)) => result.map_err(|e| connection.error(e))?,
        };
        let mut guard = CancelOnDrop { connection: &connection, id: Some(handle.id.clone()) };
        let mut updates = connection.client.progress.subscribe(handle.progress_token.clone()).await;
        let response = handle.await_response();
        tokio::pin!(response);
        let result = loop {
            tokio::select! {
                biased;
                _ = connection.cancel.cancelled() => return Err(cancelled()),
                result = &mut response => {
                    // A reply, or a timeout the SDK already cancelled on the server.
                    guard.id = None;
                    break result.map_err(|e| connection.error(e))?;
                }
                Some(update) = updates.next() => {
                    forward_progress(&connection, progress.as_ref(), &host_context, update).await?
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
        let cancel = {
            let mut state = self.state.lock();
            state.closed = true;
            state.cancel.clone()
        };
        cancel.cancel();
        let _connecting = self.connecting.lock().await;
        self.retire();
        let previous = self.state.lock().previous.clone();
        if let Some(connection) = previous
            && timeout(self.clock(), HOST_TEARDOWN_TIMEOUT, connection.settled()).await.is_err()
        {
            log::error!(target: lk_agents_abi::LOG_TARGET, "MCP incomplete cleanup; runtime retains ownership");
        }
        let initializing = self.state.lock().initializing.clone();
        if let Some(receipt) = initializing
            && timeout(self.clock(), HOST_TEARDOWN_TIMEOUT, receipt.closed()).await.is_err()
        {
            log::error!(target: lk_agents_abi::LOG_TARGET, "MCP incomplete initialization cleanup; runtime retains ownership");
        }
    }
}

impl McpSession {
    /// Whether the live connection's server said its tools changed since they were listed.
    pub(super) fn tools_changed(&self) -> bool {
        self.live().is_some_and(|c| c.client.tools_changed.load(Ordering::SeqCst))
    }

    pub(super) fn retry_delay(&self) -> Duration {
        self.state.lock().gate.wait(self.clock().now())
    }

    fn clock(&self) -> &dyn Clock {
        &*self.background.clock
    }

    async fn connect_to(&self, target: Target) -> Result<(), McpError> {
        let _connecting = self.connecting.lock().await;
        // New credentials replace the connection made with the old ones.
        let (same, previous) = {
            let mut state = self.state.lock();
            let same = state.target.as_ref().is_some_and(|current| current.same_as(&target));
            let previous = state.target.replace(target);
            state.closed = false;
            (same, previous)
        };
        drop(previous);
        if same && self.connected() {
            return Ok(());
        }
        self.connect().await
    }

    #[cfg(test)]
    pub(super) async fn connect_fake(
        &self,
        transport: impl Fn() -> tests::FakeTransport + Send + Sync + 'static,
    ) -> Result<(), McpError> {
        self.connect_to(Target::Fake(Arc::new(transport))).await
    }

    /// The live connection, reconnecting a lost one when the gate allows; `missing` when the
    /// session has no target or was closed.
    async fn ensure(&self, missing: McpError) -> Result<Arc<Connection>, McpError> {
        if let Some(connection) = self.live() {
            return Ok(connection);
        }
        let _connecting = self.connecting.lock().await;
        if let Some(connection) = self.live() {
            return Ok(connection);
        }
        {
            let state = self.state.lock();
            if state.closed || state.target.is_none() {
                return Err(missing);
            }
            if !state.gate.wait(self.clock().now()).is_zero() {
                return Err(McpError::Unavailable);
            }
        }
        self.connect().await?;
        self.live().ok_or(McpError::Unavailable)
    }

    /// One bounded attempt on the target; the caller holds `connecting`.
    async fn connect(&self) -> Result<(), McpError> {
        self.retire();
        let cancel = CancellationToken::new();
        let target = {
            let mut state = self.state.lock();
            let Some(target) = state.target.clone() else { return Err(McpError::NotInitialized) };
            state.cancel = cancel.clone();
            target
        };
        let receipt = Arc::new(TransportReceipt::default());
        let attempt = async {
            match target {
                Target::Http { env, url, headers, auth } => {
                    // Connecting and each write are bounded like the handshake; a reply stream
                    // by `read_timeout`.
                    let options = lk_agents_abi::ApiConnectOptions {
                        max_retry: 0,
                        connect_timeout: Some(self.init_timeout.as_secs_f64()),
                        write_timeout: Some(self.init_timeout.as_secs_f64()),
                        read_timeout: self.read_timeout.map(|limit| limit.as_secs_f64()),
                        ..Default::default()
                    };
                    let transport = connect_http(
                        &env,
                        url,
                        headers,
                        auth,
                        options,
                        receipt.clone(),
                        cancel.clone(),
                    )?;
                    self.serve(transport, receipt.clone(), cancel.clone()).await
                }
                Target::Stdio { command, args, env, cwd } => {
                    let clock = self.background.clock.clone();
                    let transport = StdioTransport::spawn(command, args, env, cwd, clock)?;
                    self.serve(transport, receipt.clone(), cancel.clone()).await
                }
                #[cfg(test)]
                Target::Fake(make) => self.serve(make(), receipt.clone(), cancel.clone()).await,
            }
        };
        tokio::pin!(attempt);
        let deadline = self.clock().now().saturating_add(self.init_timeout);
        let result = tokio::select! {
            biased;
            result = &mut attempt => result,
            () = self.clock().wait_until(deadline) => {
                // Cancelling lets the attempt release its transport; how it then ends is the
                // deadline's doing.
                cancel.cancel();
                match attempt.await {
                    Ok(()) | Err(_) => Err(timed_out(self.init_timeout)),
                }
            }
        };
        let now = self.clock().now();
        match &result {
            Ok(()) => {
                let mut state = self.state.lock();
                state.gate.succeeded();
                state.listing.refresh();
            }
            Err(error) => {
                log::warn!(target: lk_agents_abi::LOG_TARGET, "MCP connection failed: {error}");
                self.state.lock().gate.failed(now);
            }
        }
        result
    }

    /// Close the current connection, alive or lost; `close` waits for the one retired last.
    fn retire(&self) {
        let connection = self.state.lock().connection.take();
        if let Some(connection) = connection {
            connection.close();
            let previous = self.state.lock().previous.replace(connection);
            drop(previous);
        }
    }

    async fn serve<T: Transport<RoleClient> + 'static>(
        &self,
        transport: T,
        receipt: Arc<TransportReceipt>,
        cancel: CancellationToken,
    ) -> Result<(), McpError> {
        let client = Client::default();
        let previous = self.state.lock().initializing.replace(receipt.clone());
        drop(previous);
        let service = serve_client_with_ct(
            client.clone(),
            Observed::new(transport, receipt.clone(), self.background.clone()),
            cancel.clone(),
        )
        .await
        .map_err(|error| McpError::Initialization(Arc::new(error)));
        // Observed owns the transport even if the SDK handshake is cancelled.
        if service.is_err()
            && timeout(self.clock(), HOST_TEARDOWN_TIMEOUT, receipt.closed()).await.is_err()
        {
            log::error!(target: lk_agents_abi::LOG_TARGET, "MCP incomplete initialization cleanup; runtime retains ownership");
        }
        let service = service?;
        let peer = service.peer().clone();
        let connection = Arc::new(Connection {
            peer,
            background: self.background.clone(),
            client,
            cancel,
            service: Mutex::new(Some(service)),
            state: Shared::new(ConnectionSettlement::default()),
            receipt,
            closing: std::sync::OnceLock::new(),
        });
        let (initializing, previous) = {
            let mut state = self.state.lock();
            (state.initializing.take(), state.connection.replace(connection))
        };
        drop((initializing, previous));
        Ok(())
    }
    fn live(&self) -> Option<Arc<Connection>> {
        let connection = self.state.lock().connection.clone();
        connection.filter(|c| c.alive())
    }
    async fn request(
        &self,
        connection: &Connection,
        request: ClientRequest,
    ) -> Result<ServerResult, McpError> {
        tokio::select! {
            _ = connection.cancel.cancelled() => Err(cancelled()),
            result = async { connection.peer.send_cancellable_request(request, options(self.request_timeout)).await?.await_response().await } => result.map_err(|e| connection.error(e)),
        }
    }
}

/// A request's limit, reset by each of its progress notifications.
fn options(timeout: Option<Duration>) -> PeerRequestOptions {
    let mut options = PeerRequestOptions::no_options();
    options.timeout = timeout;
    options.reset_timeout_on_progress = true;
    options
}

impl Drop for McpSession {
    fn drop(&mut self) {
        let state = self.state.get_mut();
        state.cancel.cancel();
        if let Some(connection) = state.connection.take() {
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

fn timed_out(timeout: Duration) -> McpError {
    let cause = std::io::Error::from(std::io::ErrorKind::TimedOut);
    McpError::Timeout { timeout, cause: Arc::new(cause) }
}

fn cancelled() -> McpError {
    McpError::Cancelled(Arc::new(ServiceError::Cancelled { reason: Some(SESSION_CLOSED.into()) }))
}

#[cfg(test)]
#[path = "session_tests.rs"]
pub(super) mod tests;

#[cfg(test)]
mod target_tests {
    use super::*;
    // Catches: a refreshed token kept on the old connection, or an unchanged one reconnecting
    // before every request.
    #[tokio::test]
    async fn only_changed_credentials_are_a_new_connection() {
        let http = |token: &str| Target::Http {
            env: ProcessEnv::current_runtime(),
            url: "https://mcp.example.com/mcp".into(),
            headers: vec![("x-tenant".into(), "acme".into())],
            auth: Some(token.into()),
        };
        assert!(http("first").same_as(&http("first")));
        assert!(!http("first").same_as(&http("refreshed")));
    }
}
