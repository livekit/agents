//! SDK transports with runtime-owned reader and process settlement receipts.

use lk_agents_abi::{OwnedTask, SpawnerExt, TaskError};
use lk_agents_net::task::TokioSpawner;
use std::pin::Pin;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::{Sink, Stream, StreamExt};
use lk_agents_core::mcp::TransportSettlement;
use lk_agents_core::mcp::http::{LegacySse, sse_message};
use rmcp::model::{ClientJsonRpcMessage, ServerJsonRpcMessage};
use rmcp::service::RoleClient;
use rmcp::transport::async_rw::AsyncRwTransport;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{StreamableHttpClientTransport, Transport};
use tokio::sync::Notify;

use super::http::CleanupClient;
use super::{McpError, McpTransport, conn};
use lk_agents_abi::llm::mcp::McpInvocation;
use lk_agents_net::env::ProcessEnv;

/// The transport's settlement (`lk_agents_core::mcp::TransportSettlement`) and its waiters.
#[derive(Default)]
pub(super) struct TransportReceipt {
    state: Mutex<TransportSettlement>,
    changed: Notify,
}

struct Reader(Arc<TransportReceipt>);
impl Drop for Reader {
    fn drop(&mut self) {
        self.0.update(TransportSettlement::reader_closed);
    }
}

impl TransportReceipt {
    pub fn update<T>(&self, change: impl FnOnce(&mut TransportSettlement) -> T) -> T {
        let out = change(&mut self.state.lock().unwrap());
        self.changed.notify_waiters();
        out
    }
    pub fn read<T>(&self, read: impl FnOnce(&TransportSettlement) -> T) -> T {
        read(&self.state.lock().unwrap())
    }
    fn reader(self: &Arc<Self>) -> Reader {
        self.update(TransportSettlement::reader_opened);
        Reader(self.clone())
    }
    pub fn track_reader<T: Send + 'static>(
        self: &Arc<Self>,
        stream: futures_util::stream::BoxStream<'static, T>,
    ) -> futures_util::stream::BoxStream<'static, T> {
        let reader = self.reader();
        stream
            .map(move |event| {
                let _reader = &reader;
                event
            })
            .boxed()
    }
    fn fail(&self, error: McpError) {
        self.update(|state| state.fail(error));
    }
    pub async fn closed(&self) {
        loop {
            let changed = self.changed.notified();
            if self.read(TransportSettlement::drained) {
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
    let failed = inner.close().await.err();
    receipt.update(|state| {
        if let Some(error) = failed {
            state.cleanup_failed(conn(error));
        }
        state.close();
    });
}
impl<T: Transport<RoleClient> + 'static> Drop for Observed<T> {
    fn drop(&mut self) {
        if let Some(mut inner) = self.inner.take() {
            let receipt = self.receipt.clone();
            // Detached: the transport is being dropped; finishing settles its receipt.
            TokioSpawner::current().spawn_detached(async move {
                finish(&mut inner, &receipt).await;
            });
        }
    }
}

pub(super) struct HostStreams {
    host: Arc<dyn McpTransport>,
    invocation: McpInvocation,
    receiver: tokio::sync::mpsc::Receiver<ServerJsonRpcMessage>,
    /// Cancelled with the streams.
    reader: OwnedTask,
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
        let reader = TokioSpawner::current().spawn(async move {
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
        self.reader.cancel();
        match (&mut self.reader).await {
            Ok(()) => {}
            Err(TaskError::Cancelled) => {}
            Err(error) => self.receipt.update(|state| state.cleanup_failed(conn(error))),
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
        receipt.update(TransportSettlement::child_spawned);
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
        #[cfg(unix)]
        let signal = std::os::unix::process::ExitStatusExt::signal(&status);
        #[cfg(not(unix))]
        let signal = None;
        self.receipt.update(|state| state.child_exited(status.code(), signal));
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
    env: &ProcessEnv,
    url: String,
    headers: Vec<(String, String)>,
    transport: &str,
    timeout: Duration,
    read_timeout: Duration,
    receipt: Arc<TransportReceipt>,
) -> Result<HttpConnection, McpError> {
    let headers = headers
        .into_iter()
        .map(|(name, value)| {
            Ok((
                reqwest::header::HeaderName::try_from(name).map_err(conn)?,
                reqwest::header::HeaderValue::try_from(value).map_err(conn)?,
            ))
        })
        .collect::<Result<reqwest::header::HeaderMap, McpError>>()?;
    let client = CleanupClient {
        http: (env.http)().map_err(|error| conn(std::io::Error::other(error)))?,
        headers,
        options: lk_agents_abi::ApiConnectOptions {
            max_retry: 0,
            connect_timeout: Some(timeout.as_secs_f64()),
            write_timeout: Some(timeout.as_secs_f64()),
            read_timeout: Some(read_timeout.as_secs_f64()),
            ..Default::default()
        },
        clock: env.clock.clone(),
        receipt: receipt.clone(),
    };
    match lk_agents_core::mcp::http_transport(Some(transport))? {
        lk_agents_core::mcp::HttpTransport::StreamableHttp => {
            Ok(HttpConnection::Streamable(StreamableHttpClientTransport::with_client(
                client,
                StreamableHttpClientTransportConfig::with_uri(url),
            )))
        }
        lk_agents_core::mcp::HttpTransport::Sse => {
            let (posts, messages) = connect_sse(client, &url, receipt).await?;
            Ok(HttpConnection::Sse(rmcp::transport::sink_stream::SinkStreamTransport::new(
                posts, messages,
            )))
        }
    }
}

async fn connect_sse(
    client: CleanupClient,
    url: &str,
    receipt: Arc<TransportReceipt>,
) -> Result<(Posts, Messages), McpError> {
    let legacy = LegacySse::new(url).map_err(conn)?;
    let mut events = client.legacy_get(url).await.map_err(conn)?;
    let endpoint = loop {
        match events.next().await {
            Ok(Some(event)) => match legacy.endpoint(&event.event, &event.data) {
                Some(endpoint) => break endpoint.map_err(conn)?,
                None => continue,
            },
            Err(error) => return Err(conn(error)),
            Ok(None) => {
                return Err(conn(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "SSE closed before endpoint",
                )));
            }
        }
    };
    events.expect_response(false);
    let events = receipt.track_reader(events.into_stream());
    let messages = events
        .scan((), move |_, event| {
            std::future::ready(match event {
                Ok(event) => match sse_message(&event.event, &event.data) {
                    Some(data) => match serde_json::from_str(data) {
                        Ok(message) => Some(Some(message)),
                        Err(error) => {
                            receipt.fail(McpError::Decode(Arc::new(error)));
                            None
                        }
                    },
                    None => Some(None),
                },
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
            client.legacy_post(endpoint.as_str(), message).await.map_err(conn)?;
            Ok::<_, McpError>((client, endpoint))
        },
    );
    Ok((Box::pin(posts), Box::pin(messages)))
}
