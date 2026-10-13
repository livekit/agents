//! `MCPServer` over an `rmcp` session: stdio and streamable HTTP are rmcp's transports, SSE is
//! [`connect_sse`], a subclass's `client_streams()` is [`McpTransport`]. An injected `_client` is
//! [`McpClient`], which answers `list_tools`/`call_tool` itself.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::StreamExt;
use lk_agents_abi::errors::HostError;
use rmcp::handler::client::progress::ProgressDispatcher;
use rmcp::model::{
    CallToolRequest, CallToolRequestParams, ClientJsonRpcMessage, ClientRequest,
    ProgressNotificationParam, ServerJsonRpcMessage, ServerResult,
};
use rmcp::service::{
    NotificationContext, PeerRequestOptions, RoleClient, RunningService, ServiceError,
};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{IntoTransport, StreamableHttpClientTransport, TokioChildProcess, Transport};
use rmcp::{ClientHandler, serve_client};
use serde_json::{Map, Value};

use super::{HttpTransport, McpError, conn};

const LOG: &str = "livekit.agents";

/// A subclass's `client_streams()`, as the host adapter's four calls (`llm/mcp.py`'s `_Streams`).
#[lk_agents_abi::host(proxy(host = "livekit.agents.llm.mcp._Streams"))]
#[async_trait::async_trait]
pub trait McpTransport: Send + Sync {
    async fn start(&self) -> Result<(), HostError>;
    async fn send(&self, message: &Value) -> Result<(), HostError>;
    /// The next JSON-RPC message; `None` once the streams close.
    async fn receive(&self) -> Result<Option<Value>, HostError>;
    async fn close(&self);
}

/// An injected `_client`: a `ClientSession` stand-in answering the two calls itself (`_Client`).
#[lk_agents_abi::host(proxy(host = "livekit.agents.llm.mcp._Client"))]
#[async_trait::async_trait]
pub trait McpClient: Send + Sync {
    async fn list_tools(&self) -> Result<Value, HostError>;
    async fn call_tool(&self, name: &str, arguments: &Value) -> Result<Value, HostError>;
}

/// Where a `report_progress` tool's messages go: the tool's `RunContext` (`_Progress`).
#[lk_agents_abi::host(proxy(host = "livekit.agents.llm.mcp._Progress"))]
#[async_trait::async_trait]
pub trait McpProgress: Send + Sync {
    async fn update(&self, message: &str) -> Result<(), HostError>;
}

#[derive(Debug, Clone, Default)]
pub struct Progress(ProgressDispatcher);

impl ClientHandler for Progress {
    async fn on_progress(
        &self,
        params: ProgressNotificationParam,
        _: NotificationContext<RoleClient>,
    ) {
        self.0.handle_notification(params).await;
    }
}

enum Backend {
    Session(RunningService<RoleClient, Progress>),
    Client(Arc<dyn McpClient>),
}

pub struct McpSession {
    /// `read_timeout_seconds`: `None` for a falsy `client_session_timeout_seconds`.
    timeout: Option<Duration>,
    backend: Mutex<Option<Arc<Backend>>>,
    /// `_raw_tools`; `None` is `_cache_dirty`.
    cache: Mutex<Option<Vec<Value>>>,
    /// `MCPServerHTTP`'s `allowed_tools`; empty is no filter, as `set(x) if x else None` is.
    allowed: Mutex<Vec<String>>,
    connecting: tokio::sync::Mutex<()>,
}

#[lk_agents_abi::host(object(cell = "arc"))]
impl McpSession {
    #[host(new)]
    pub fn new(client_session_timeout_seconds: f64) -> Self {
        McpSession {
            timeout: secs(client_session_timeout_seconds),
            backend: Mutex::new(None),
            cache: Mutex::new(None),
            allowed: Mutex::new(Vec::new()),
            connecting: tokio::sync::Mutex::new(()),
        }
    }

    /// `MCPServer.initialized`. A session whose transport closed is not connected, as the
    /// reference's `_run_client` clears `_client` when it ends.
    #[host(getter)]
    pub fn connected(&self) -> bool {
        match self.backend.lock().expect("mcp backend").as_deref() {
            Some(Backend::Session(s)) => !s.is_closed(),
            Some(Backend::Client(_)) => true,
            None => false,
        }
    }

    #[host(setter)]
    pub fn set_allowed_tools(&self, allowed: Vec<String>) {
        *self.allowed.lock().expect("mcp allowed") = allowed;
    }

    #[host(method)]
    pub fn invalidate_cache(&self) {
        *self.cache.lock().expect("mcp cache") = None;
    }

    /// `_client = client`: answer from a host `ClientSession` stand-in.
    #[host(method)]
    pub fn attach_client(
        &self,
        client: Arc<dyn McpClient>,
    ) {
        self.install(Backend::Client(client));
    }

    /// `_client = None`.
    #[host(method)]
    pub fn detach(&self) {
        drop(self.backend.lock().expect("mcp backend").take());
        self.invalidate_cache();
    }

    /// `MCPServerHTTP.initialize()`: `httpx`'s client as `_create_http_client` builds it -- the
    /// headers on every request, redirects followed, `timeout` to connect and `sse_read_timeout`
    /// per read -- under streamable HTTP or SSE.
    #[host(method)]
    pub async fn connect_http(
        &self,
        url: String,
        headers: HashMap<String, String>,
        transport: HttpTransport,
        timeout: f64,
        sse_read_timeout: f64,
    ) -> Result<(), McpError> {
        let Some(_connecting) = self.connecting().await else { return Ok(()) };
        crate::transport::install_tls_provider();
        let mut client = reqwest::Client::builder()
            .default_headers(reqwest::header::HeaderMap::try_from(&headers).map_err(conn)?)
            .redirect(reqwest::redirect::Policy::limited(20));
        if let Some(t) = secs(timeout) {
            client = client.connect_timeout(t);
        }
        if let Some(t) = secs(sse_read_timeout) {
            client = client.read_timeout(t);
        }
        let client = client.build().map_err(conn)?;
        match transport {
            HttpTransport::StreamableHttp => {
                let config = StreamableHttpClientTransportConfig::with_uri(url);
                self.serve(StreamableHttpClientTransport::with_client(client, config)).await
            }
            HttpTransport::Sse => self.serve(connect_sse(client, &url).await?).await,
        }
    }

    /// `MCPServerStdio.initialize()`. The child's environment is the SDK's: the few variables
    /// `get_default_environment()` inherits, then `env` -- not the whole parent environment.
    #[host(method)]
    pub async fn connect_stdio(
        &self,
        command: String,
        args: Vec<String>,
        env: HashMap<String, String>,
        #[host(required)] cwd: Option<String>,
    ) -> Result<(), McpError> {
        let inherited = if cfg!(windows) {
            "APPDATA HOMEDRIVE HOMEPATH LOCALAPPDATA PATH PATHEXT PROCESSOR_ARCHITECTURE SYSTEMDRIVE SYSTEMROOT TEMP USERNAME USERPROFILE"
        } else {
            "HOME LOGNAME PATH SHELL TERM USER"
        };
        let Some(_connecting) = self.connecting().await else { return Ok(()) };
        let mut cmd = tokio::process::Command::new(&command);
        let defaults = std::env::vars()
            .filter(|(k, v)| inherited.split(' ').any(|i| i == k) && !v.starts_with("()"));
        cmd.args(&args).env_clear().envs(defaults).envs(&env);
        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        self.serve(TokioChildProcess::new(cmd).map_err(conn)?).await
    }

    /// `initialize()` for a subclass that implements `client_streams()` itself.
    #[host(method)]
    pub async fn connect_host(
        &self,
        transport: Arc<dyn McpTransport>,
    ) -> Result<(), McpError> {
        let Some(_connecting) = self.connecting().await else { return Ok(()) };
        transport.start().await.map_err(conn)?;
        let served = self.serve(HostStreams::new(Arc::clone(&transport))).await;
        if served.is_err() {
            transport.close().await;
        }
        served
    }

    /// `_list_raw_tools()` (one page, as the reference) through the cache, then `allowed_tools`.
    #[host(method)]
    pub async fn list_tools(&self) -> Result<Vec<Value>, McpError> {
        let cached = self.cache.lock().expect("mcp cache").clone();
        let tools = match cached {
            Some(tools) => tools,
            None => {
                let tools = match &*self.live(McpError::NotInitialized)? {
                    Backend::Session(s) => match self
                        .request(s, ClientRequest::ListToolsRequest(Default::default()))
                        .await
                        .map_err(conn)?
                    {
                        ServerResult::ListToolsResult(r) => {
                            serde_json::to_value(r.tools).map_err(conn)?
                        }
                        other => {
                            return Err(conn(format!(
                                "unexpected response to tools/list: {other:?}"
                            )));
                        }
                    },
                    Backend::Client(c) => c.list_tools().await.map_err(conn)?,
                };
                let Value::Array(tools) = tools else {
                    return Err(conn("tools/list answered no list"));
                };
                *self.cache.lock().expect("mcp cache") = Some(tools.clone());
                tools
            }
        };
        let allowed = self.allowed.lock().expect("mcp allowed").clone();
        Ok(tools
            .into_iter()
            .filter(|t| allowed.is_empty() || allowed.iter().any(|a| t["name"] == a.as_str()))
            .collect())
    }

    /// `_tool_called` / `_tool_called_nonblocking`: the raw `CallToolResult`, as wire JSON. With
    /// `progress`, every non-empty progress message is awaited through `ctx.update` in order.
    #[host(method)]
    pub async fn call_tool(
        &self,
        name: String,
        arguments: Map<String, Value>,
        progress: Option<Arc<dyn McpProgress>>,
    ) -> Result<Value, McpError> {
        let backend = self.live(McpError::Unavailable)?;
        let s = match &*backend {
            Backend::Client(c) => {
                return c.call_tool(&name, &Value::Object(arguments)).await.map_err(conn);
            }
            Backend::Session(s) => s,
        };
        let mut params = CallToolRequestParams::new(name.clone());
        params.arguments = Some(arguments);
        let request = ClientRequest::CallToolRequest(CallToolRequest::new(params));
        let handle =
            s.peer().send_cancellable_request(request, self.options()).await.map_err(call_error)?;
        let mut updates = s.service().0.subscribe(handle.progress_token.clone()).await;
        let response = handle.await_response();
        tokio::pin!(response);
        let result = loop {
            tokio::select! {
                biased;
                Some(p) = updates.next() => {
                    let (Some(sink), Some(message)) = (&progress, p.message.as_deref().filter(|m| !m.is_empty())) else { continue };
                    log::debug!(target: LOG, tool_name = name.as_str(), progress = p.progress, total:? = p.total,
                        "lk.pii.progress_message" = message; "MCPTool progress");
                    let _ = sink.update(message).await;
                }
                r = &mut response => break r.map_err(call_error)?,
            }
        };
        match result {
            ServerResult::CallToolResult(r) => serde_json::to_value(r).map_err(conn),
            other => Err(conn(format!("unexpected response to tools/call: {other:?}"))),
        }
    }

    /// `aclose()`: cancel the session and wait for the transport to close.
    #[host(method)]
    pub async fn close(&self) {
        let backend = self.backend.lock().expect("mcp backend").take();
        self.invalidate_cache();
        if let Some(Ok(Backend::Session(s))) = backend.map(Arc::try_unwrap) {
            let _ = s.cancel().await;
        }
    }
}

impl McpSession {
    /// `None` (after the reference's warning) when already connected.
    async fn connecting(&self) -> Option<tokio::sync::MutexGuard<'_, ()>> {
        let guard = self.connecting.lock().await;
        if self.connected() {
            log::warn!(target: LOG, "MCPServer is already initializing");
            return None;
        }
        Some(guard)
    }

    async fn serve<T, E, A>(&self, transport: T) -> Result<(), McpError>
    where
        T: IntoTransport<RoleClient, E, A>,
        E: std::error::Error + Send + Sync + 'static,
    {
        self.install(Backend::Session(
            serve_client(Progress::default(), transport).await.map_err(conn)?,
        ));
        Ok(())
    }

    fn install(&self, backend: Backend) {
        *self.backend.lock().expect("mcp backend") = Some(Arc::new(backend));
        self.invalidate_cache();
    }

    fn live(&self, none: McpError) -> Result<Arc<Backend>, McpError> {
        self.backend.lock().expect("mcp backend").clone().ok_or(none)
    }

    fn options(&self) -> PeerRequestOptions {
        let mut options = PeerRequestOptions::no_options();
        options.timeout = self.timeout;
        options
    }

    async fn request(
        &self,
        s: &RunningService<RoleClient, Progress>,
        r: ClientRequest,
    ) -> Result<ServerResult, ServiceError> {
        s.peer().send_cancellable_request(r, self.options()).await?.await_response().await
    }
}

/// `timedelta(seconds=x) if x else None`: zero (or anything not positive) is no timeout at all.
fn secs(x: f64) -> Option<Duration> {
    if x > 0.0 { Duration::try_from_secs_f64(x).ok() } else { None }
}

/// A call after `aclose` is the reference's "unavailable" sentence; anything else (a timeout, a
/// JSON-RPC error) is a failed call, which the reference raises as the SDK's `McpError`.
fn call_error(e: ServiceError) -> McpError {
    match e {
        ServiceError::TransportClosed => McpError::Unavailable,
        other => conn(other),
    }
}

/// `sse_client`: GET the endpoint, wait for its `endpoint` event, then POST each message to that
/// URL and read the replies off the event stream's `message` events.
async fn connect_sse(
    client: reqwest::Client,
    url: &str,
) -> Result<
    (
        impl futures_util::Sink<ClientJsonRpcMessage, Error = McpError> + Send + Unpin + 'static,
        impl futures_util::Stream<Item = ServerJsonRpcMessage> + Send + Unpin + 'static,
    ),
    McpError,
> {
    let base = url::Url::parse(url).map_err(conn)?;
    let response = client
        .get(base.clone())
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .header(reqwest::header::CACHE_CONTROL, "no-store")
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(conn)?;
    let mut events = Box::pin(sse_stream::SseStream::from_bytes_stream(response.bytes_stream()));
    let endpoint = loop {
        match events.next().await {
            Some(Ok(e)) if e.event.as_deref() == Some("endpoint") => {
                break base.join(e.data.as_deref().unwrap_or_default()).map_err(conn)?;
            }
            Some(Ok(_)) => continue,
            Some(Err(e)) => return Err(conn(e)),
            None => return Err(conn("the SSE stream closed before its endpoint event")),
        }
    };
    if endpoint.origin() != base.origin() {
        return Err(conn(format!("Endpoint origin does not match connection origin: {endpoint}")));
    }
    let messages = events.filter_map(|e| {
        std::future::ready(match e {
            Ok(e) if e.event.as_deref().unwrap_or("message") == "message" => e
                .data
                .filter(|d| !d.is_empty())
                .and_then(|d| serde_json::from_str::<ServerJsonRpcMessage>(&d).ok()),
            _ => None,
        })
    });
    let post = futures_util::sink::unfold(
        (client, endpoint),
        |(client, endpoint), m: ClientJsonRpcMessage| async move {
            let body = serde_json::to_vec(&m).map_err(conn)?;
            let post = client
                .post(endpoint.clone())
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(body);
            post.send().await.and_then(reqwest::Response::error_for_status).map_err(conn)?;
            Ok::<_, McpError>((client, endpoint))
        },
    );
    Ok((Box::pin(post), Box::pin(messages)))
}

/// [`McpTransport`] as `rmcp`'s transport. rmcp drops and re-polls `receive()`, and a host call
/// that already took a message off the host's stream cannot be undone, so one task reads the host
/// and `receive()` reads its channel.
struct HostStreams(Arc<dyn McpTransport>, tokio::sync::mpsc::Receiver<ServerJsonRpcMessage>);

impl HostStreams {
    fn new(host: Arc<dyn McpTransport>) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let reader = Arc::clone(&host);
        tokio::spawn(async move {
            while let Ok(Some(v)) = reader.receive().await {
                // An unreadable message is skipped, as the SDK's session logs and skips one.
                let Ok(message) = serde_json::from_value(v) else { continue };
                if tx.send(message).await.is_err() {
                    break;
                }
            }
        });
        HostStreams(host, rx)
    }
}

impl Transport<RoleClient> for HostStreams {
    type Error = HostError;

    fn send(
        &mut self,
        item: ClientJsonRpcMessage,
    ) -> impl Future<Output = Result<(), HostError>> + Send + 'static {
        let host = Arc::clone(&self.0);
        async move {
            host.send(&serde_json::to_value(item).map_err(|e| HostError::new(e.to_string()))?).await
        }
    }

    fn receive(&mut self) -> impl Future<Output = Option<ServerJsonRpcMessage>> + Send {
        self.1.recv()
    }

    async fn close(&mut self) -> Result<(), HostError> {
        self.0.close().await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! Both HTTP transports against one loopback server: `/sse` + `/messages` (legacy SSE) and
    //! `/mcp` (streamable HTTP, JSON bodies). Every request's head goes to `heads`.
    use super::*;
    use serde_json::json;
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};

    fn answer(m: &Value) -> Option<String> {
        let result = match m["method"].as_str()? {
            "initialize" => {
                json!({"protocolVersion": m["params"]["protocolVersion"], "capabilities": {"tools": {}}, "serverInfo": {"name": "t", "version": "1"}})
            }
            "tools/list" => {
                json!({"tools": [{"name": "echo", "inputSchema": {}}, {"name": "hidden", "inputSchema": {}}]})
            }
            _ => json!({"content": [{"type": "text", "text": m["params"]["arguments"]["text"]}]}),
        };
        Some(json!({"jsonrpc": "2.0", "id": m.get("id")?, "result": result}).to_string())
    }

    async fn serve(
        socket: tokio::net::TcpStream,
        heads: UnboundedSender<String>,
        sse: Arc<Mutex<Option<UnboundedSender<String>>>>,
    ) {
        let (r, mut w) = socket.into_split();
        let mut r = BufReader::new(r);
        loop {
            let (mut head, mut line) = (String::new(), String::new());
            while r.read_line(&mut line).await.unwrap_or(0) > 2 {
                head.push_str(&std::mem::take(&mut line).to_lowercase());
            }
            let len = head
                .lines()
                .find_map(|l| l.strip_prefix("content-length: "))
                .map_or(0, |n| n.trim().parse().unwrap());
            let mut body = vec![0; len];
            if head.is_empty() || r.read_exact(&mut body).await.is_err() {
                return;
            }
            let _ = heads.send(head.clone());
            let reply = serde_json::from_slice(&body).ok().and_then(|m: Value| answer(&m));
            let (status, body) = if head.starts_with("get /sse") {
                let (tx, mut rx) = unbounded_channel();
                *sse.lock().unwrap() = Some(tx);
                w.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\nevent: endpoint\ndata: /messages?s=1\n\n").await.unwrap();
                while let Some(m) = rx.recv().await {
                    w.write_all(format!("event: message\ndata: {m}\n\n").as_bytes()).await.unwrap();
                }
                return;
            } else if head.starts_with("post /messages") {
                reply.map(|m| sse.lock().unwrap().as_ref().unwrap().send(m));
                ("202 Accepted", String::new())
            } else if let (true, Some(m)) = (head.starts_with("post /mcp"), reply) {
                ("200 OK", m)
            } else {
                ("202 Accepted", String::new())
            };
            let out = format!(
                "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
                body.len()
            );
            w.write_all(out.as_bytes()).await.unwrap();
        }
    }

    // Catches: SSE refused or upgraded, the POST aimed anywhere but the `endpoint` event's URL, a
    // reply read from the POST body instead of the stream; streamable HTTP not driven; headers
    // dropped; `allowed_tools` ignored.
    async fn round_trip(path: &str, transport: HttpTransport) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}{path}", listener.local_addr().unwrap());
        let (heads, mut seen) = unbounded_channel();
        tokio::spawn(async move {
            let sse = Arc::default();
            while let Ok((socket, _)) = listener.accept().await {
                tokio::spawn(serve(socket, heads.clone(), Arc::clone(&sse)));
            }
        });
        let session = McpSession::new(5.0);
        session.set_allowed_tools(vec!["echo".into()]);
        let headers = HashMap::from([("x-test".to_owned(), "1".to_owned())]);
        session.connect_http(url, headers, transport, 5.0, 300.0).await.unwrap();
        let tools = session.list_tools().await.unwrap();
        assert_eq!(tools.iter().map(|t| &t["name"]).collect::<Vec<_>>(), ["echo"]);
        let result = session
            .call_tool("echo".into(), json!({"text": "hi"}).as_object().unwrap().clone(), None)
            .await
            .unwrap();
        assert_eq!(result["content"][0]["text"], "hi");
        session.close().await;
        assert!(!session.connected());
        while let Ok(head) = seen.try_recv() {
            assert!(head.contains("x-test: 1"), "{head}");
        }
    }

    #[tokio::test]
    async fn the_legacy_sse_transport_round_trips() {
        tokio::time::timeout(Duration::from_secs(10), round_trip("/sse", HttpTransport::Sse))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn streamable_http_round_trips() {
        tokio::time::timeout(
            Duration::from_secs(10),
            round_trip("/mcp", HttpTransport::StreamableHttp),
        )
        .await
        .unwrap();
    }

    // Catches: a call on a closed session answered with a transport error instead of the
    // reference's sentence, and `list_tools` before `initialize()` not being its RuntimeError.
    #[tokio::test]
    async fn a_session_that_is_not_connected_says_so_in_the_references_words() {
        let session = McpSession::new(0.0);
        assert_eq!(session.list_tools().await, Err(McpError::NotInitialized));
        assert_eq!(
            session.call_tool("t".into(), Map::new(), None).await,
            Err(McpError::Unavailable)
        );
    }
}
