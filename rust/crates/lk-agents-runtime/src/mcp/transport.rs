//! SDK transports whose readers and close the runtime tracks until they drain.

use lk_agents_abi::SpawnerExt;
use lk_agents_net::task::TokioSpawner;
use parking_lot::Mutex;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use lk_agents_core::mcp::TransportSettlement;
use rmcp::model::{ClientJsonRpcMessage, ServerJsonRpcMessage};
use rmcp::service::RoleClient;
use rmcp::transport::async_rw::AsyncRwTransport;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{StreamableHttpClientTransport, Transport};
use tokio::sync::Notify;

use super::http::CleanupClient;
use super::{McpError, conn};
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
        let out = change(&mut self.state.lock());
        self.changed.notify_waiters();
        out
    }
    pub fn read<T>(&self, read: impl FnOnce(&TransportSettlement) -> T) -> T {
        read(&self.state.lock())
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
    pub fn fail(&self, error: McpError) {
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
            state.fail(conn(error));
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

pub(super) struct StdioTransport {
    io: AsyncRwTransport<RoleClient, tokio::process::ChildStdout, tokio::process::ChildStdin>,
    child: tokio::process::Child,
}
impl StdioTransport {
    pub fn spawn(
        command: String,
        args: Vec<String>,
        env: Vec<(String, String)>,
        cwd: Option<String>,
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
        Ok(Self { io, child })
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
        if let Ok(status) = tokio::time::timeout(Duration::from_secs(3), self.child.wait()).await {
            return status.map(drop).map_err(conn);
        }
        self.child.start_kill().map_err(conn)?;
        self.child.wait().await.map(drop).map_err(conn)
    }
}

/// The Streamable HTTP transport over the shared exchange; the SDK opens the session on its
/// first message.
pub(super) fn connect_http(
    env: &ProcessEnv,
    url: String,
    headers: Vec<(String, String)>,
    timeout: Duration,
    read_timeout: Option<Duration>,
    receipt: Arc<TransportReceipt>,
) -> Result<StreamableHttpClientTransport<CleanupClient>, McpError> {
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
            read_timeout: read_timeout.map(|limit| limit.as_secs_f64()),
            ..Default::default()
        },
        clock: env.clock.clone(),
        receipt: receipt.clone(),
    };
    Ok(StreamableHttpClientTransport::with_client(
        client,
        StreamableHttpClientTransportConfig::with_uri(url),
    ))
}
