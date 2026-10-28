//! SDK transports with runtime-owned reader and process settlement receipts.

use std::pin::Pin;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::{Sink, Stream, StreamExt};
use rmcp::model::{ClientJsonRpcMessage, ServerJsonRpcMessage};
use rmcp::service::RoleClient;
use rmcp::transport::async_rw::AsyncRwTransport;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{StreamableHttpClientTransport, Transport};
use tokio::sync::Notify;

use super::http_client::CleanupClient;
use super::{McpCleanup, McpCleanupStatus, McpError, McpTransport, conn};
use lk_agents_abi::llm::mcp::McpInvocation;

#[derive(Default)]
pub(super) struct TransportReceipt {
    state: Mutex<ReceiptState>,
    changed: Notify,
}
#[derive(Default)]
struct ReceiptState {
    closed: bool,
    readers: usize,
    child_pending: bool,
    exit: Option<std::process::ExitStatus>,
    failure: Option<Arc<McpError>>,
    cleanup_failed: bool,
    deadline_missed: bool,
    // The SDK can stop trying a failed DELETE; the runtime retains the remote resource identity.
    http_session: Option<HttpSession>,
}
type HttpSession = (
    reqwest::Client,
    Arc<str>,
    Arc<str>,
    Option<String>,
    std::collections::HashMap<reqwest::header::HeaderName, reqwest::header::HeaderValue>,
);

struct Reader(Arc<TransportReceipt>);
impl Drop for Reader {
    fn drop(&mut self) {
        self.0.state.lock().unwrap().readers -= 1;
        self.0.changed.notify_waiters();
    }
}

impl TransportReceipt {
    fn reader(self: &Arc<Self>) -> Reader {
        self.state.lock().unwrap().readers += 1;
        Reader(self.clone())
    }
    pub fn track_reader(
        self: &Arc<Self>,
        stream: futures_util::stream::BoxStream<
            'static,
            Result<sse_stream::Sse, sse_stream::Error>,
        >,
    ) -> futures_util::stream::BoxStream<'static, Result<sse_stream::Sse, sse_stream::Error>> {
        let reader = self.reader();
        stream
            .map(move |event| {
                let _reader = &reader;
                event
            })
            .boxed()
    }
    pub fn deleting_http_session(
        &self,
        client: reqwest::Client,
        uri: Arc<str>,
        session: Arc<str>,
        auth: Option<String>,
        headers: std::collections::HashMap<
            reqwest::header::HeaderName,
            reqwest::header::HeaderValue,
        >,
    ) {
        self.state.lock().unwrap().http_session = Some((client, uri, session, auth, headers));
    }
    pub fn deleted_http_session(&self) {
        self.state.lock().unwrap().http_session = None;
    }
    pub fn transport_failure(&self, error: McpError) {
        self.fail(error);
    }

    pub fn failed(&self) -> bool {
        let state = self.state.lock().unwrap();
        state.cleanup_failed
            || state.child_pending
            || state.deadline_missed
            || state.http_session.is_some()
    }
    pub fn missed_deadline(&self) {
        self.state.lock().unwrap().deadline_missed = true;
    }
    pub fn failure(&self) -> Option<Arc<McpError>> {
        self.state.lock().unwrap().failure.clone()
    }
    fn fail(&self, error: McpError) {
        self.state.lock().unwrap().failure.get_or_insert_with(|| Arc::new(error));
    }
    pub fn cleanup(&self) -> McpCleanup {
        let state = self.state.lock().unwrap();
        McpCleanup {
            status: if state.deadline_missed || state.cleanup_failed || state.http_session.is_some()
            {
                McpCleanupStatus::Incomplete
            } else if state.closed && !state.child_pending && state.readers == 0 {
                McpCleanupStatus::Clean
            } else {
                McpCleanupStatus::Pending
            },
            service_settled: state.closed,
            calls_settled: true,
            reader_settled: state.closed && state.readers == 0,
            child_reaped: state.exit.is_some(),
            child_exit_code: state.exit.and_then(|s| s.code()),
            #[cfg(unix)]
            child_exit_signal: state.exit.and_then(|s| {
                use std::os::unix::process::ExitStatusExt;
                s.signal()
            }),
            #[cfg(not(unix))]
            child_exit_signal: None,
        }
    }
    pub async fn closed(&self) {
        loop {
            let changed = self.changed.notified();
            let ready = {
                let state = self.state.lock().unwrap();
                state.closed && state.readers == 0
            };
            if ready {
                return;
            }
            changed.await;
        }
    }
}

/// The same owner closes a failed/cancelled handshake and an established SDK service.
pub(super) struct Observed<T: Transport<RoleClient> + 'static> {
    inner: Option<T>,
    receipt: Arc<TransportReceipt>,
}
impl<T: Transport<RoleClient> + 'static> Observed<T> {
    pub fn new(inner: T, receipt: Arc<TransportReceipt>) -> Self {
        Self { inner: Some(inner), receipt }
    }
}
impl<T: Transport<RoleClient> + 'static> Transport<RoleClient> for Observed<T> {
    type Error = McpError;
    fn send(
        &mut self,
        item: ClientJsonRpcMessage,
    ) -> impl Future<Output = Result<(), McpError>> + Send + 'static {
        let send = self.inner.as_mut().unwrap().send(item);
        let receipt = self.receipt.clone();
        async move {
            send.await.map_err(|e| {
                let cause: lk_agents_abi::llm::mcp::McpCause = Arc::new(e);
                receipt.fail(McpError::Transport(cause.clone()));
                McpError::Transport(cause)
            })
        }
    }
    async fn receive(&mut self) -> Option<ServerJsonRpcMessage> {
        self.inner.as_mut().unwrap().receive().await
    }
    async fn close(&mut self) -> Result<(), McpError> {
        if let Some(mut inner) = self.inner.take() {
            finish(&mut inner, &self.receipt).await;
        }
        Ok(())
    }
}
async fn finish<T: Transport<RoleClient>>(inner: &mut T, receipt: &TransportReceipt) {
    if let Err(error) = inner.close().await {
        receipt.fail(conn(error));
        receipt.state.lock().unwrap().cleanup_failed = true;
    }
    receipt.state.lock().unwrap().closed = true;
    receipt.changed.notify_waiters();
}
impl<T: Transport<RoleClient> + 'static> Drop for Observed<T> {
    fn drop(&mut self) {
        if let Some(mut inner) = self.inner.take() {
            let receipt = self.receipt.clone();
            tokio::spawn(async move {
                finish(&mut inner, &receipt).await;
            });
        }
    }
}

pub(super) struct HostStreams {
    host: Arc<dyn McpTransport>,
    invocation: McpInvocation,
    receiver: tokio::sync::mpsc::Receiver<ServerJsonRpcMessage>,
    reader: tokio::task::JoinHandle<()>,
    receipt: Arc<TransportReceipt>,
}
impl HostStreams {
    pub fn new(
        host: Arc<dyn McpTransport>,
        invocation: McpInvocation,
        receipt: Arc<TransportReceipt>,
    ) -> Self {
        let (sender, receiver) = tokio::sync::mpsc::channel(16);
        let reader_host = host.clone();
        let reader_receipt = receipt.clone();
        let reader_invocation = invocation.clone();
        let reader_guard = receipt.reader();
        let reader = tokio::spawn(async move {
            let _reader_guard = reader_guard;
            loop {
                let value = match reader_host.receive(&reader_invocation).await {
                    Ok(serde_json::Value::Null) => break,
                    Ok(value) => value,
                    Err(error) => {
                        reader_receipt.fail(McpError::Host(error));
                        break;
                    }
                };
                match serde_json::from_value(value) {
                    Ok(message) => {
                        if sender.send(message).await.is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        reader_receipt.fail(McpError::Decode(Arc::new(error)));
                        break;
                    }
                }
            }
        });
        Self { host, invocation, receiver, reader, receipt }
    }
}
impl Transport<RoleClient> for HostStreams {
    type Error = McpError;
    fn send(
        &mut self,
        item: ClientJsonRpcMessage,
    ) -> impl Future<Output = Result<(), McpError>> + Send + 'static {
        let host = self.host.clone();
        let invocation = self.invocation.clone();
        async move {
            let value = serde_json::to_value(item).map_err(|e| McpError::Decode(Arc::new(e)))?;
            host.send(&invocation, &value).await.map_err(McpError::Host)
        }
    }
    async fn receive(&mut self) -> Option<ServerJsonRpcMessage> {
        self.receiver.recv().await
    }
    async fn close(&mut self) -> Result<(), McpError> {
        self.reader.abort();
        match (&mut self.reader).await {
            Ok(()) => {}
            Err(error) if error.is_cancelled() => {}
            Err(error) => {
                self.receipt.fail(conn(error));
                self.receipt.state.lock().unwrap().cleanup_failed = true;
            }
        }
        self.host.close(&self.invocation).await.map_err(McpError::Host)
    }
}

pub(super) struct StdioTransport {
    io: AsyncRwTransport<RoleClient, tokio::process::ChildStdout, tokio::process::ChildStdin>,
    child: tokio::process::Child,
    receipt: Arc<TransportReceipt>,
}
impl StdioTransport {
    pub fn spawn(
        command: String,
        args: Vec<String>,
        env: Vec<(String, String)>,
        cwd: Option<String>,
        receipt: Arc<TransportReceipt>,
    ) -> Result<Self, McpError> {
        let inherited = if cfg!(windows) {
            "APPDATA HOMEDRIVE HOMEPATH LOCALAPPDATA PATH PATHEXT PROCESSOR_ARCHITECTURE SYSTEMDRIVE SYSTEMROOT TEMP USERNAME USERPROFILE"
        } else {
            "HOME LOGNAME PATH SHELL TERM USER"
        };
        let defaults = std::env::vars()
            .filter(|(k, v)| inherited.split(' ').any(|i| i == k) && !v.starts_with("()"));
        let mut command = tokio::process::Command::new(command);
        command
            .args(args)
            .env_clear()
            .envs(defaults)
            .envs(env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        if let Some(cwd) = cwd {
            command.current_dir(cwd);
        }
        let mut child = command.spawn().map_err(conn)?;
        let io = AsyncRwTransport::new(child.stdout.take().unwrap(), child.stdin.take().unwrap());
        receipt.state.lock().unwrap().child_pending = true;
        Ok(Self { io, child, receipt })
    }
}
impl Transport<RoleClient> for StdioTransport {
    type Error = McpError;
    fn send(
        &mut self,
        item: ClientJsonRpcMessage,
    ) -> impl Future<Output = Result<(), McpError>> + Send + 'static {
        let send = self.io.send(item);
        async move { send.await.map_err(conn) }
    }
    async fn receive(&mut self) -> Option<ServerJsonRpcMessage> {
        self.io.receive().await
    }
    async fn close(&mut self) -> Result<(), McpError> {
        self.io.close().await.map_err(conn)?;
        let status = match tokio::time::timeout(Duration::from_secs(3), self.child.wait()).await {
            Ok(status) => status.map_err(conn)?,
            Err(_) => {
                self.child.start_kill().map_err(conn)?;
                self.child.wait().await.map_err(conn)?
            }
        };
        let mut state = self.receipt.state.lock().unwrap();
        state.exit = Some(status);
        state.child_pending = false;
        Ok(())
    }
}

type Messages = Pin<Box<dyn Stream<Item = ServerJsonRpcMessage> + Send>>;
type Posts = Pin<Box<dyn Sink<ClientJsonRpcMessage, Error = McpError> + Send>>;
pub(super) enum HttpConnection {
    Streamable(StreamableHttpClientTransport<CleanupClient>),
    Sse(rmcp::transport::sink_stream::SinkStreamTransport<Posts, Messages>),
}
impl Transport<RoleClient> for HttpConnection {
    type Error = McpError;
    fn send(
        &mut self,
        item: ClientJsonRpcMessage,
    ) -> impl Future<Output = Result<(), McpError>> + Send + 'static {
        // rmcp's tuple adapter owns the sink; streamable HTTP uses its upstream transport.
        match self {
            Self::Streamable(transport) => {
                let send = transport.send(item);
                Box::pin(async move { send.await.map_err(conn) })
                    as Pin<Box<dyn Future<Output = Result<(), McpError>> + Send>>
            }
            Self::Sse(transport) => {
                let send = <_ as Transport<RoleClient>>::send(transport, item);
                Box::pin(send)
            }
        }
    }
    async fn receive(&mut self) -> Option<ServerJsonRpcMessage> {
        match self {
            Self::Streamable(t) => t.receive().await,
            Self::Sse(t) => <_ as Transport<RoleClient>>::receive(t).await,
        }
    }
    async fn close(&mut self) -> Result<(), McpError> {
        match self {
            Self::Streamable(t) => t.close().await.map_err(conn),
            Self::Sse(t) => <_ as Transport<RoleClient>>::close(t).await,
        }
    }
}

pub(super) async fn connect_http(
    url: String,
    headers: Vec<(String, String)>,
    transport: &str,
    timeout: Duration,
    read_timeout: Duration,
    receipt: Arc<TransportReceipt>,
) -> Result<HttpConnection, McpError> {
    lk_agents_inference::transport::install_tls_provider();
    let headers = headers
        .into_iter()
        .map(|(key, value)| {
            Ok((
                reqwest::header::HeaderName::try_from(key).map_err(conn)?,
                reqwest::header::HeaderValue::try_from(value).map_err(conn)?,
            ))
        })
        .collect::<Result<reqwest::header::HeaderMap, McpError>>()?;
    let client = reqwest::Client::builder()
        .retry(reqwest::retry::never())
        .default_headers(headers)
        .redirect(reqwest::redirect::Policy::limited(20))
        .connect_timeout(timeout)
        .read_timeout(read_timeout)
        .build()
        .map_err(conn)?;
    match super::http_transport(&url, Some(transport))? {
        super::HttpTransport::StreamableHttp => {
            Ok(HttpConnection::Streamable(StreamableHttpClientTransport::with_client(
                CleanupClient { client, receipt: receipt.clone() },
                StreamableHttpClientTransportConfig::with_uri(url),
            )))
        }
        super::HttpTransport::Sse => {
            let (posts, messages) = connect_sse(client, &url, receipt).await?;
            Ok(HttpConnection::Sse(rmcp::transport::sink_stream::SinkStreamTransport::new(
                posts, messages,
            )))
        }
    }
}

async fn connect_sse(
    client: reqwest::Client,
    url: &str,
    receipt: Arc<TransportReceipt>,
) -> Result<(Posts, Messages), McpError> {
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
            Some(Ok(event)) if event.event.as_deref() == Some("endpoint") => {
                break base.join(event.data.as_deref().unwrap_or_default()).map_err(conn)?;
            }
            Some(Ok(_)) => continue,
            Some(Err(error)) => return Err(conn(error)),
            None => {
                return Err(conn(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "SSE closed before endpoint",
                )));
            }
        }
    };
    if endpoint.origin() != base.origin() {
        return Err(conn(std::io::Error::other("SSE endpoint origin differs from connection")));
    }
    let messages = events
        .scan((), move |_, event| {
            std::future::ready(match event {
                Ok(event) if event.event.as_deref().unwrap_or("message") == "message" => {
                    match event.data.filter(|data| !data.is_empty()) {
                        Some(data) => match serde_json::from_str(&data) {
                            Ok(message) => Some(Some(message)),
                            Err(error) => {
                                receipt.fail(McpError::Decode(Arc::new(error)));
                                None
                            }
                        },
                        None => Some(None),
                    }
                }
                Ok(_) => Some(None),
                Err(error) => {
                    receipt.fail(conn(error));
                    None
                }
            })
        })
        .filter_map(std::future::ready);
    let posts = futures_util::sink::unfold(
        (client, endpoint),
        |(client, endpoint), message: ClientJsonRpcMessage| async move {
            client
                .post(endpoint.clone())
                .json(&message)
                .send()
                .await
                .and_then(reqwest::Response::error_for_status)
                .map_err(conn)?;
            Ok::<_, McpError>((client, endpoint))
        },
    );
    Ok((Box::pin(posts), Box::pin(messages)))
}
