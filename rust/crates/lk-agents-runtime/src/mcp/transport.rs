use lk_agents_abi::{Clock, timeout};
use lk_agents_net::owned::Background;
use lk_agents_net::signal::Shared;
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

use super::http::CleanupClient;
use super::{McpError, conn};
use lk_agents_net::env::ProcessEnv;

/// The transport's settlement (`lk_agents_core::mcp::TransportSettlement`) and its waiters.
pub(super) struct TransportReceipt(Shared<TransportSettlement>);

impl Default for TransportReceipt {
    fn default() -> Self {
        Self(Shared::new(TransportSettlement::default()))
    }
}

struct Reader(Arc<TransportReceipt>);
impl Drop for Reader {
    fn drop(&mut self) {
        self.0.update(TransportSettlement::reader_closed);
    }
}

impl TransportReceipt {
    pub fn update<T>(&self, change: impl FnOnce(&mut TransportSettlement) -> T) -> T {
        self.0.modify(change)
    }
    pub fn read<T>(&self, read: impl FnOnce(&TransportSettlement) -> T) -> T {
        self.0.with(|state| read(state))
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
        self.0.wait_until(|state| state.drained().then_some(())).await;
    }
}

/// The same owner closes a failed/cancelled handshake and an established SDK service; one
/// dropped unclosed finishes as a cleanup of the session's background.
pub(super) struct Observed<T: Transport<RoleClient> + 'static> {
    inner: Option<T>,
    receipt: Arc<TransportReceipt>,
    background: Background,
}
impl<T: Transport<RoleClient> + 'static> Observed<T> {
    pub fn new(inner: T, receipt: Arc<TransportReceipt>, background: Background) -> Self {
        Self { inner: Some(inner), receipt, background }
    }
}
impl<T: Transport<RoleClient> + 'static> Transport<RoleClient> for Observed<T> {
    type Error = McpError;
    fn send(
        &mut self,
        item: ClientJsonRpcMessage,
    ) -> impl Future<Output = Result<(), McpError>> + Send + 'static {
        let send = self.inner.as_mut().expect("rmcp never sends after close").send(item);
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
        match &mut self.inner {
            Some(inner) => inner.receive().await,
            None => None,
        }
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
            // Bounded by the transport's own close; finishing settles its receipt.
            self.background.cleanup("MCP transport close", async move {
                finish(&mut inner, &receipt).await;
            });
        }
    }
}

/// How long a stdio server whose input closed may take to exit before it is killed.
const STDIO_EXIT_GRACE: Duration = Duration::from_secs(3);

pub(super) struct StdioTransport {
    io: AsyncRwTransport<RoleClient, tokio::process::ChildStdout, tokio::process::ChildStdin>,
    child: tokio::process::Child,
    clock: Arc<dyn Clock>,
}
impl StdioTransport {
    pub fn spawn(
        command: String,
        args: Vec<String>,
        env: Vec<(String, String)>,
        cwd: Option<String>,
        clock: Arc<dyn Clock>,
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
        let stdout = child.stdout.take().expect("stdout is piped");
        let stdin = child.stdin.take().expect("stdin is piped");
        let io = AsyncRwTransport::new(stdout, stdin);
        Ok(Self { io, child, clock })
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
        let exited = timeout(&*self.clock, STDIO_EXIT_GRACE, self.child.wait()).await;
        if let Ok(status) = exited {
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
    auth: Option<String>,
    options: lk_agents_abi::ApiConnectOptions,
    receipt: Arc<TransportReceipt>,
    cancel: tokio_util::sync::CancellationToken,
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
        options,
        clock: env.clock.clone(),
        receipt: receipt.clone(),
        cancel,
    };
    let mut config = StreamableHttpClientTransportConfig::with_uri(url);
    if let Some(token) = auth {
        config = config.auth_header(token);
    }
    Ok(StreamableHttpClientTransport::with_client(client, config))
}
