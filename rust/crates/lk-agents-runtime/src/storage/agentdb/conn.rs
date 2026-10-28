//! One authenticated WebSocket to one database (`wsdb`): one protobuf per frame, correlated by
//! `request_id`. The Go client in agents-private/agent-db (`pkg/agentdb/client`) is the reference.
//!
//! * `Hello` first, with a token minted for this dial.
//! * `Exec` answers one `ExecResult`; `Query` streams `Columns`, `ColumnBatch`* and `Done`, paced
//!   by `Credit`; `Batch` applies its statements atomically, one result per statement, then `Done`.
//! * A dropped socket is redialed within a budget; a request issued meanwhile waits, and requests
//!   in flight fail [`WsdbError::Disconnected`] (outcome unknown).
//! * A request answered `retry` applied nothing and is sent again, twice at most.

use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::{SinkExt as _, StreamExt as _};
use livekit_api::access_token::{AccessToken, AgentGrants};
pub use livekit_protocol::agent_db::wire;
use prost::Message as _;
use tokio::sync::{mpsc, watch};
use tokio_tungstenite::tungstenite::client::IntoClientRequest as _;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::tungstenite::{self, Message};
use wire as pb;

use lk_agents_storage::StorageError;
use lk_agents_storage::sql::{BatchResult, Row, SqlValue, Statement, StatementResult};

/// A bound value as the wire carries it.
fn to_wire(v: &SqlValue) -> wire::Value {
    use wire::value::Value as V;
    let value = match v {
        SqlValue::Null => V::NullValue(true),
        SqlValue::Int(i) => V::IntValue(*i),
        SqlValue::Double(d) => V::DoubleValue(*d),
        SqlValue::Text(s) => V::TextValue(s.clone()),
        SqlValue::Blob(b) => V::BlobValue(b.clone()),
    };
    wire::Value { value: Some(value) }
}

/// A statement as the wire carries it.
pub fn statement(s: &Statement) -> wire::Statement {
    wire::Statement {
        sql: s.sql.to_string(),
        params: s.params.iter().map(to_wire).collect(),
        lang: 0,
    }
}

/// A batch's rows. Values are typed per row: walk `types` and take the next entry from the
/// matching array; text and blobs are concatenated with one exclusive end offset each. A
/// truncated column reads as `NULL`, as the Go client does, rather than failing the batch.
pub fn decode_batch(b: &wire::ColumnBatch) -> Vec<Row> {
    use wire::ValueType;
    let rows = b.rows as usize;
    let mut out: Vec<Row> = (0..rows).map(|_| Vec::with_capacity(b.columns.len())).collect();
    for col in &b.columns {
        let (mut ints, mut doubles, mut texts, mut blobs) = (0usize, 0usize, 0usize, 0usize);
        let (mut text_at, mut blob_at) = (0usize, 0usize);
        for (r, row) in out.iter_mut().enumerate() {
            let tag =
                col.types.get(r).copied().map(i32::from).and_then(|t| ValueType::try_from(t).ok());
            let cell = match tag {
                Some(ValueType::Int) if ints < col.ints.len() => {
                    ints += 1;
                    SqlValue::Int(col.ints[ints - 1])
                }
                Some(ValueType::Double) if doubles < col.doubles.len() => {
                    doubles += 1;
                    SqlValue::Double(col.doubles[doubles - 1])
                }
                Some(ValueType::Text) if texts < col.text_ends.len() => {
                    let end = col.text_ends[texts] as usize;
                    texts += 1;
                    let cell = if end >= text_at && end <= col.text_data.len() {
                        SqlValue::Text(
                            String::from_utf8_lossy(&col.text_data[text_at..end]).into_owned(),
                        )
                    } else {
                        SqlValue::Null
                    };
                    text_at = end;
                    cell
                }
                Some(ValueType::Blob) if blobs < col.blob_ends.len() => {
                    let end = col.blob_ends[blobs] as usize;
                    blobs += 1;
                    let cell = if end >= blob_at && end <= col.blob_data.len() {
                        SqlValue::Blob(col.blob_data[blob_at..end].to_vec())
                    } else {
                        SqlValue::Null
                    };
                    blob_at = end;
                    cell
                }
                _ => SqlValue::Null,
            };
            row.push(cell);
        }
    }
    out
}

/// The inverse of [`decode_batch`], for tests and fakes: rows into one column-oriented batch.
pub fn encode_batch(rows: &[Row], width: usize) -> wire::ColumnBatch {
    use wire::ValueType;
    let mut columns: Vec<wire::Column> = (0..width).map(|_| wire::Column::default()).collect();
    for row in rows {
        for (i, col) in columns.iter_mut().enumerate() {
            match row.get(i).unwrap_or(&SqlValue::Null) {
                SqlValue::Null => col.types.push(ValueType::Null as u8),
                SqlValue::Int(v) => {
                    col.types.push(ValueType::Int as u8);
                    col.ints.push(*v);
                }
                SqlValue::Double(v) => {
                    col.types.push(ValueType::Double as u8);
                    col.doubles.push(*v);
                }
                SqlValue::Text(s) => {
                    col.types.push(ValueType::Text as u8);
                    col.text_data.extend_from_slice(s.as_bytes());
                    col.text_ends.push(col.text_data.len() as u32);
                }
                SqlValue::Blob(b) => {
                    col.types.push(ValueType::Blob as u8);
                    col.blob_data.extend_from_slice(b);
                    col.blob_ends.push(col.blob_data.len() as u32);
                }
            }
        }
    }
    wire::ColumnBatch { columns, rows: rows.len() as u32, statement: 0 }
}

type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;
type WsRead = futures_util::stream::SplitStream<WsStream>;

#[derive(Debug, Clone, thiserror::Error)]
pub enum WsdbError {
    /// `close` was called, or reconnection gave up.
    #[error("agentdb client: connection closed")]
    Closed,
    /// The connection dropped with this request in flight; a write's outcome is unknown.
    #[error("agentdb client: connection lost, request outcome unknown")]
    Disconnected,
    /// A server-reported error with its stable code (`not_found`, `unauthenticated`, `unavailable`, ..).
    #[error("agentdb {code}: {message}")]
    Server { code: String, message: String },
    #[error("agentdb client: protocol: {0}")]
    Protocol(String),
    #[error("agentdb client: {0}")]
    Io(String),
    #[error("agentdb client: timed out")]
    Timeout,
    /// A request frame over the server's limit, refused before it was sent.
    #[error("agentdb client: a {0} byte frame is over the {MAX_FRAME} byte limit")]
    TooLarge(usize),
}

impl WsdbError {
    /// A `Hello` refusal no redial can change: the database is gone or the key is refused.
    fn is_final(&self) -> bool {
        matches!(self, WsdbError::Server { code, .. }
            if matches!(code.as_str(), "not_found" | "unauthenticated" | "permission_denied"))
    }
}

/// The largest frame the server reads; a larger one closes the socket.
pub const MAX_FRAME: usize = 16 << 20;
/// How long a frame may take to send, and the read deadline past a missed pong, when the server
/// names no ping timeout.
const DEFAULT_PING_TIMEOUT: Duration = Duration::from_secs(10);

/// Reconnection, on by default: a node draining or an idle database evicted severs the socket.
#[derive(Debug, Clone)]
pub struct Reconnect {
    pub disable: bool,
    pub min_backoff: Duration,
    pub max_backoff: Duration,
    /// How long to keep trying before giving up.
    pub budget: Duration,
}

impl Default for Reconnect {
    fn default() -> Self {
        Self {
            disable: false,
            min_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(5),
            budget: Duration::from_secs(60),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Options {
    /// The data-plane endpoint, `ws://host:port/db`.
    pub url: String,
    pub database_id: String,
    /// The receive buffer, in batches, per streaming query.
    pub credits: u32,
    /// Bounds the WebSocket dial plus `Hello`.
    pub dial_timeout: Duration,
    pub reconnect: Reconnect,
}

impl Options {
    pub fn new(url: impl Into<String>, database_id: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            database_id: database_id.into(),
            credits: 8,
            dial_timeout: Duration::from_secs(10),
            reconnect: Reconnect::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExecResult {
    pub rows_affected: i64,
    pub last_insert_id: i64,
    pub tip: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Connecting,
    Ready,
    Closed,
}

struct State {
    pending: HashMap<u32, mpsc::UnboundedSender<pb::ServerMessage>>,
    /// The live socket's writer; `None` between a drop and the redial.
    tx: Option<mpsc::UnboundedSender<Vec<u8>>>,
    closed: bool,
    /// Why the connection gave up for good; later calls fail with it.
    fatal: Option<WsdbError>,
    tip: i64,
}

struct Inner {
    opts: Options,
    minter: TokenMinter,
    next_id: AtomicU32,
    state: Mutex<State>,
    phase: watch::Sender<Phase>,
    tasks: Mutex<Vec<tokio::task::JoinHandle<()>>>,
}

/// One authenticated connection to one database; cheap to clone, closed by [`Conn::close`] or
/// when the last clone drops. The connection's own tasks hold it without the drop guard, so they
/// do not keep it open.
#[derive(Clone)]
pub struct Conn(Arc<Inner>, #[allow(dead_code)] Option<Arc<CloseOnDrop>>);

struct CloseOnDrop(std::sync::Weak<Inner>);

impl Drop for CloseOnDrop {
    fn drop(&mut self) {
        if let Some(inner) = self.0.upgrade() {
            inner.close();
        }
    }
}

impl Inner {
    fn close(&self) {
        {
            let mut st = self.state.lock().expect("state");
            if st.closed {
                return;
            }
            st.closed = true;
            st.pending.clear();
            st.tx = None;
        }
        self.phase.send_replace(Phase::Closed);
        for t in self.tasks.lock().map(|mut t| std::mem::take(&mut *t)).unwrap_or_default() {
            t.abort();
        }
    }
}

/// A live socket's read half, with what ends it besides the server.
struct Socket {
    read: WsRead,
    /// Resolves when the writer stopped: a send failed or timed out.
    writer_gone: tokio::sync::oneshot::Receiver<()>,
    /// No frame for this long (pongs included) is a dead socket.
    read_deadline: Option<Duration>,
    ping_interval: Duration,
}

impl Conn {
    /// Dials, says `Hello`, and starts the supervisor.
    pub async fn dial(opts: Options, minter: TokenMinter) -> Result<Conn, WsdbError> {
        super::install_tls_provider();
        let (phase, _) = watch::channel(Phase::Connecting);
        let inner = Arc::new(Inner {
            opts,
            minter,
            next_id: AtomicU32::new(0),
            state: Mutex::new(State {
                pending: HashMap::new(),
                tx: None,
                closed: false,
                fatal: None,
                tip: 0,
            }),
            phase,
            tasks: Mutex::new(Vec::new()),
        });
        let guard = Arc::new(CloseOnDrop(Arc::downgrade(&inner)));
        let conn = Conn(inner, Some(guard));
        let socket = conn.connect().await?;
        let ping_interval = socket.ping_interval;
        let supervisor = tokio::spawn(conn.unguarded().supervise(socket));
        let mut tasks = vec![supervisor];
        if !ping_interval.is_zero() {
            tasks.push(tokio::spawn(conn.unguarded().ping_loop(ping_interval)));
        }
        conn.0.tasks.lock().expect("tasks").extend(tasks);
        Ok(conn)
    }

    /// A handle for the connection's own tasks: it does not keep the connection open.
    fn unguarded(&self) -> Conn {
        Conn(self.0.clone(), None)
    }

    /// The durable commit sequence last observed.
    pub fn tip(&self) -> i64 {
        self.0.state.lock().expect("state").tip
    }

    pub fn database_id(&self) -> &str {
        &self.0.opts.database_id
    }

    /// Dial, `Hello`, publish the writer.
    async fn connect(&self) -> Result<Socket, WsdbError> {
        let token = self.0.minter.mint().map_err(|e| WsdbError::Io(e.to_string()))?;
        // the token also rides the upgrade, so a proxy in front of agent-db can admit the socket
        let mut url = url::Url::parse(&self.0.opts.url)
            .map_err(|e| WsdbError::Io(format!("dial {}: {e}", self.0.opts.url)))?;
        url.query_pairs_mut().append_pair("access_token", &token);
        let request = url
            .as_str()
            .into_client_request()
            .map_err(|e| WsdbError::Io(format!("dial {}: {e}", self.0.opts.url)))?;
        let config = WebSocketConfig::default()
            .max_frame_size(Some(MAX_FRAME))
            .max_message_size(Some(MAX_FRAME));
        // no Nagle: a request that follows a credit frame must not wait for the delayed ack
        let dial = tokio_tungstenite::connect_async_with_config(request, Some(config), true);
        let (ws, _) = tokio::time::timeout(self.0.opts.dial_timeout, dial)
            .await
            .map_err(|_| WsdbError::Timeout)?
            .map_err(|e| match e {
                // the upgrade's own refusal is final, as the same refusal at `Hello` is
                tungstenite::Error::Http(r) if matches!(r.status().as_u16(), 401 | 403) => {
                    WsdbError::Server {
                        code: "unauthenticated".into(),
                        message: r.status().to_string(),
                    }
                }
                tungstenite::Error::Http(r) if r.status().as_u16() == 404 => {
                    WsdbError::Server { code: "not_found".into(), message: r.status().to_string() }
                }
                e => WsdbError::Io(format!("dial {}: {e}", self.0.opts.url)),
            })?;
        let (mut sink, mut read) = ws.split();
        let hello = pb::ClientMessage {
            request_id: self.next_id(),
            message: Some(pb::client_message::Message::Hello(pb::Hello {
                token,
                database_id: self.0.opts.database_id.clone(),
            })),
        };
        sink.send(Message::binary(hello.encode_to_vec()))
            .await
            .map_err(|e| WsdbError::Io(e.to_string()))?;
        let first = async {
            loop {
                match read.next().await {
                    Some(Ok(Message::Binary(bytes))) => {
                        return pb::ServerMessage::decode(bytes.as_ref())
                            .map_err(|e| WsdbError::Protocol(e.to_string()));
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        return Err(WsdbError::Io("socket closed during hello".into()));
                    }
                    Some(Ok(_)) => {}
                    Some(Err(e)) => return Err(WsdbError::Io(e.to_string())),
                }
            }
        };
        let resp = tokio::time::timeout(self.0.opts.dial_timeout, first)
            .await
            .map_err(|_| WsdbError::Timeout)??;
        let ok = match resp.message {
            Some(pb::server_message::Message::HelloOk(ok)) => ok,
            Some(pb::server_message::Message::Error(e)) => {
                return Err(WsdbError::Server { code: e.code, message: e.message });
            }
            other => {
                return Err(WsdbError::Protocol(format!("unexpected hello response {other:?}")));
            }
        };
        let ping_interval = Duration::from_millis(u64::from(ok.ping_interval_ms));
        let ping_timeout = match ok.ping_timeout_ms {
            0 => DEFAULT_PING_TIMEOUT,
            ms => Duration::from_millis(u64::from(ms)),
        };
        // the writer owns the sink; a send that does not finish within the ping timeout is a dead
        // socket
        let (tx, mut rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let (gone, writer_gone) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            while let Some(bytes) = rx.recv().await {
                let sent = tokio::time::timeout(ping_timeout, sink.send(Message::binary(bytes)));
                if !matches!(sent.await, Ok(Ok(()))) {
                    break;
                }
            }
            drop(gone);
            let _ = tokio::time::timeout(ping_timeout, sink.close()).await;
        });
        {
            let mut st = self.0.state.lock().expect("state");
            st.tip = st.tip.max(ok.tip); // a reconnect can only observe a later tip
            st.tx = Some(tx);
        }
        self.0.phase.send_replace(Phase::Ready);
        let read_deadline = (!ping_interval.is_zero())
            .then(|| ping_interval + ping_timeout + Duration::from_secs(1));
        Ok(Socket { read, writer_gone, read_deadline, ping_interval })
    }

    /// Reads the live socket and reconnects when it dies: one task owns the socket's lifetime.
    async fn supervise(self, mut socket: Socket) {
        loop {
            let cause = self.read_loop(&mut socket).await;
            if self.is_closed() {
                return;
            }
            self.dropped(&cause);
            if self.0.opts.reconnect.disable {
                self.close();
                return;
            }
            match self.redial().await {
                Some(next) => socket = next,
                None => {
                    self.close();
                    return;
                }
            }
        }
    }

    async fn read_loop(&self, socket: &mut Socket) -> WsdbError {
        loop {
            let next = async {
                match socket.read_deadline {
                    Some(d) => tokio::time::timeout(d, socket.read.next()).await.ok(),
                    None => Some(socket.read.next().await),
                }
            };
            let next = tokio::select! {
                next = next => next,
                _ = &mut socket.writer_gone => return WsdbError::Io("the socket stopped sending".into()),
            };
            let Some(next) = next else {
                return WsdbError::Timeout; // not even a pong within the deadline
            };
            match next {
                Some(Ok(Message::Binary(bytes))) => match pb::ServerMessage::decode(bytes.as_ref())
                {
                    Ok(resp)
                        if matches!(resp.message, Some(pb::server_message::Message::Pong(_))) => {}
                    Ok(resp) => {
                        let st = self.0.state.lock().expect("state");
                        if let Some(ch) = st.pending.get(&resp.request_id) {
                            let _ = ch.send(resp);
                        }
                    }
                    Err(e) => return WsdbError::Protocol(e.to_string()),
                },
                Some(Ok(Message::Close(_))) | None => return WsdbError::Io("socket closed".into()),
                Some(Ok(_)) => {}
                Some(Err(e)) => return WsdbError::Io(e.to_string()),
            }
        }
    }

    /// Fails everything in flight and closes the ready gate. Only after the read loop returned, so
    /// no response can still be delivered.
    fn dropped(&self, cause: &WsdbError) {
        let mut st = self.0.state.lock().expect("state");
        st.pending.clear();
        st.tx = None;
        drop(st);
        self.0.phase.send_replace(Phase::Connecting);
        log::warn!(target: "livekit.agents", databaseId = self.0.opts.database_id.as_str(); "agentdb connection lost: {cause}");
    }

    /// Exponential backoff until the budget runs out.
    async fn redial(&self) -> Option<Socket> {
        let r = &self.0.opts.reconnect;
        let deadline = tokio::time::Instant::now() + r.budget;
        let mut backoff = r.min_backoff;
        loop {
            match self.connect().await {
                Ok(socket) => {
                    log::info!(target: "livekit.agents", databaseId = self.0.opts.database_id.as_str(); "agentdb reconnected");
                    return Some(socket);
                }
                Err(e) if e.is_final() => {
                    log::warn!(target: "livekit.agents", databaseId = self.0.opts.database_id.as_str(); "agentdb redial refused for good: {e}");
                    self.0.state.lock().expect("state").fatal = Some(e);
                    return None;
                }
                Err(e) => {
                    log::warn!(target: "livekit.agents", databaseId = self.0.opts.database_id.as_str(); "agentdb redial failed: {e}")
                }
            }
            if tokio::time::Instant::now() >= deadline || self.is_closed() {
                return None;
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(r.max_backoff);
        }
    }

    async fn ping_loop(self, interval: Duration) {
        let mut tick = tokio::time::interval(interval);
        tick.tick().await;
        loop {
            tick.tick().await;
            if *self.0.phase.borrow() != Phase::Ready {
                continue; // mid-reconnect: a ping before Hello is a protocol error
            }
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0);
            let _ = self.write(
                self.next_id(),
                pb::client_message::Message::Ping(pb::Ping { timestamp_ms: now }),
            );
        }
    }

    fn is_closed(&self) -> bool {
        self.0.state.lock().expect("state").closed
    }

    /// Open, nothing in flight: a connection another opener can take.
    pub fn is_idle(&self) -> bool {
        let st = self.0.state.lock().expect("state");
        !st.closed && st.pending.is_empty()
    }

    /// Ends the connection; every waiter and every later call fails [`WsdbError::Closed`].
    pub fn close(&self) {
        self.0.close();
    }

    /// Why a closed connection refuses a call: what made it give up, else `Closed`.
    fn closed_error(&self) -> WsdbError {
        self.0.state.lock().expect("state").fatal.clone().unwrap_or(WsdbError::Closed)
    }

    fn next_id(&self) -> u32 {
        self.0.next_id.fetch_add(1, Ordering::Relaxed) + 1
    }

    async fn register(&self) -> Result<Request<'_>, WsdbError> {
        // a request issued mid-reconnect is delayed rather than failed
        let mut phase = self.0.phase.subscribe();
        if self.is_closed()
            || !matches!(phase.wait_for(|p| *p != Phase::Connecting).await, Ok(p) if *p == Phase::Ready)
        {
            return Err(self.closed_error());
        }
        let id = self.next_id();
        let (tx, rx) = mpsc::unbounded_channel();
        let mut st = self.0.state.lock().expect("state");
        if st.closed {
            drop(st);
            return Err(self.closed_error());
        }
        st.pending.insert(id, tx);
        Ok(Request { conn: self, id, rx, answered: false })
    }

    fn write(
        &self,
        request_id: u32,
        message: pb::client_message::Message,
    ) -> Result<(), WsdbError> {
        let bytes = pb::ClientMessage { request_id, message: Some(message) }.encode_to_vec();
        if bytes.len() > MAX_FRAME {
            return Err(WsdbError::TooLarge(bytes.len()));
        }
        let st = self.0.state.lock().expect("state");
        if st.closed {
            drop(st);
            return Err(self.closed_error());
        }
        match &st.tx {
            Some(tx) => tx.send(bytes).map_err(|_| WsdbError::Disconnected),
            None => Err(WsdbError::Disconnected),
        }
    }

    async fn wait(
        &self,
        rx: &mut mpsc::UnboundedReceiver<pb::ServerMessage>,
    ) -> Result<pb::ServerMessage, WsdbError> {
        match rx.recv().await {
            None if self.is_closed() => Err(self.closed_error()),
            None => Err(WsdbError::Disconnected),
            Some(pb::ServerMessage {
                message: Some(pb::server_message::Message::Error(e)),
                ..
            }) => Err(WsdbError::Server { code: e.code, message: e.message }),
            Some(resp) => Ok(resp),
        }
    }

    fn observe_tip(&self, tip: i64) {
        let mut st = self.0.state.lock().expect("state");
        st.tip = st.tip.max(tip);
    }

    /// One write statement; returns once durable.
    pub async fn exec(&self, s: &Statement) -> Result<ExecResult, WsdbError> {
        let stmt = statement(s);
        resend(|| self.exec_once(stmt.clone())).await
    }

    async fn exec_once(&self, stmt: pb::Statement) -> Result<ExecResult, WsdbError> {
        let mut req = self.register().await?;
        self.write(req.id, pb::client_message::Message::Exec(stmt))?;
        let r = match req.next().await?.message {
            Some(pb::server_message::Message::ExecResult(r)) => r,
            other => {
                return Err(WsdbError::Protocol(format!("unexpected exec response {other:?}")));
            }
        };
        req.answered = true;
        self.observe_tip(r.tip);
        Ok(ExecResult {
            rows_affected: r.rows_affected,
            last_insert_id: r.last_insert_id,
            tip: r.tip,
        })
    }

    /// A query's rows, all of them: one credit back per batch consumed.
    pub async fn query_all(&self, s: &Statement) -> Result<Vec<Row>, WsdbError> {
        let stmt = statement(s);
        resend(|| self.query_once(stmt.clone())).await
    }

    async fn query_once(&self, stmt: pb::Statement) -> Result<Vec<Row>, WsdbError> {
        let mut req = self.register().await?;
        let id = req.id;
        self.write(id, pb::client_message::Message::Query(stmt))?;
        let mut rows = Vec::new();
        loop {
            match req.next().await?.message {
                Some(pb::server_message::Message::Columns(_)) => {}
                Some(pb::server_message::Message::ColumnBatch(b)) => {
                    rows.extend(decode_batch(&b));
                    let _ = self
                        .write(id, pb::client_message::Message::Credit(pb::Credit { batches: 1 }));
                }
                Some(pb::server_message::Message::Done(d)) => {
                    req.answered = true;
                    self.observe_tip(d.tip);
                    return Ok(rows);
                }
                other => {
                    return Err(WsdbError::Protocol(format!(
                        "unexpected query response {other:?}"
                    )));
                }
            }
        }
    }

    /// Statements applied atomically, in order, each answered in order.
    pub async fn batch(&self, statements: &[Statement]) -> Result<BatchResult, WsdbError> {
        let statements: Vec<pb::Statement> = statements.iter().map(statement).collect();
        resend(|| self.batch_once(statements.clone())).await
    }

    async fn batch_once(&self, statements: Vec<pb::Statement>) -> Result<BatchResult, WsdbError> {
        let n = statements.len();
        let mut req = self.register().await?;
        let id = req.id;
        let r = async {
            self.write(id, pb::client_message::Message::Batch(pb::Batch { statements }))?;
            // the receive window up front: a result larger than the server's initial window
            // does not wait a round trip for the first replenishment
            self.write(
                id,
                pb::client_message::Message::Credit(pb::Credit { batches: self.0.opts.credits }),
            )?;
            let mut results: Vec<Option<StatementResult>> = (0..n).map(|_| None).collect();
            let slot =
                |results: &mut Vec<Option<StatementResult>>, i: u32| -> Result<usize, WsdbError> {
                    let i = i as usize;
                    if i >= results.len() {
                        return Err(WsdbError::Protocol(format!(
                            "batch result for statement {i} of {n}"
                        )));
                    }
                    Ok(i)
                };
            loop {
                match req.next().await?.message {
                    Some(pb::server_message::Message::Columns(c)) => {
                        let i = slot(&mut results, c.statement)?;
                        results[i] = Some(StatementResult::Rows(Vec::new()));
                    }
                    Some(pb::server_message::Message::ColumnBatch(b)) => {
                        let i = slot(&mut results, b.statement)?;
                        let rows = decode_batch(&b);
                        match &mut results[i] {
                            Some(StatementResult::Rows(set)) => set.extend(rows),
                            _ => results[i] = Some(StatementResult::Rows(rows)),
                        }
                        let _ = self.write(
                            id,
                            pb::client_message::Message::Credit(pb::Credit { batches: 1 }),
                        );
                    }
                    Some(pb::server_message::Message::ExecResult(r)) => {
                        let i = slot(&mut results, r.statement)?;
                        results[i] = Some(StatementResult::Exec {
                            rows_affected: r.rows_affected,
                            last_insert_id: r.last_insert_id,
                        });
                    }
                    Some(pb::server_message::Message::Done(d)) => {
                        let results = results
                            .into_iter()
                            .map(|r| {
                                r.unwrap_or(StatementResult::Exec {
                                    rows_affected: 0,
                                    last_insert_id: 0,
                                })
                            })
                            .collect();
                        req.answered = true;
                        self.observe_tip(d.tip);
                        return Ok(BatchResult { results });
                    }
                    other => {
                        return Err(WsdbError::Protocol(format!(
                            "unexpected batch response {other:?}"
                        )));
                    }
                }
            }
        }
        .await?;
        Ok(r)
    }
}

/// One request in flight. Dropped before its answer (an error, or the caller's future dropped),
/// it leaves the pending table and tells the server to stop.
struct Request<'a> {
    conn: &'a Conn,
    id: u32,
    rx: mpsc::UnboundedReceiver<pb::ServerMessage>,
    answered: bool,
}

impl Request<'_> {
    async fn next(&mut self) -> Result<pb::ServerMessage, WsdbError> {
        self.conn.wait(&mut self.rx).await
    }
}

impl Drop for Request<'_> {
    fn drop(&mut self) {
        self.conn.0.state.lock().expect("state").pending.remove(&self.id);
        if !self.answered {
            let _ = self.conn.write(self.id, pb::client_message::Message::Cancel(pb::Cancel {}));
        }
    }
}

pub const TOKEN_TTL: Duration = Duration::from_secs(3600);

/// The API key and secret, held for the life of the client; a token is minted on every dial.
#[derive(Clone)]
pub struct TokenMinter {
    api_key: String,
    api_secret: String,
}

impl std::fmt::Debug for TokenMinter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenMinter").field("api_key", &self.api_key).finish_non_exhaustive()
    }
}

impl TokenMinter {
    pub fn new(
        api_key: impl Into<String>,
        api_secret: impl Into<String>,
    ) -> Result<Self, StorageError> {
        let (api_key, api_secret) = (api_key.into(), api_secret.into());
        if api_key.is_empty() || api_secret.is_empty() {
            return Err(StorageError::Invalid(
                "LIVEKIT_API_KEY and LIVEKIT_API_SECRET are required".into(),
            ));
        }
        Ok(Self { api_key, api_secret })
    }

    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    pub fn api_secret(&self) -> &str {
        &self.api_secret
    }

    /// A fresh token: [`TOKEN_TTL`] long, with the `agent.databaseAdmin` grant.
    pub fn mint(&self) -> Result<String, StorageError> {
        AccessToken::with_api_key(&self.api_key, &self.api_secret)
            .with_ttl(TOKEN_TTL)
            .with_agent_grants(AgentGrants { database_admin: true, ..Default::default() })
            .to_jwt()
            .map_err(|e| StorageError::Invalid(format!("token: {e}")))
    }
}

/// The pauses before each resend of a request the server answered `retry`.
const RETRY_PAUSES: [Duration; 2] = [Duration::from_millis(50), Duration::from_millis(200)];

/// Sends again a request answered `retry`: the server applied nothing, so the same request is safe.
async fn resend<T, F>(send: impl Fn() -> F) -> Result<T, WsdbError>
where
    F: Future<Output = Result<T, WsdbError>>,
{
    for pause in RETRY_PAUSES {
        match send().await {
            Err(WsdbError::Server { code, .. }) if code == "retry" => {
                tokio::time::sleep(pause).await
            }
            out => return out,
        }
    }
    send().await
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use tokio::net::TcpListener;

    #[test]
    fn a_mixed_column_round_trips() {
        let rows: Vec<Row> = vec![
            vec![SqlValue::Int(1), SqlValue::Text("a".into()), SqlValue::Null],
            vec![
                SqlValue::Text("mixed".into()),
                SqlValue::Text("".into()),
                SqlValue::Blob(vec![1, 2]),
            ],
            vec![SqlValue::Double(2.5), SqlValue::Text("bb".into()), SqlValue::Int(-7)],
        ];
        let batch = encode_batch(&rows, 3);
        assert_eq!(decode_batch(&batch), rows);
    }

    #[test]
    fn a_truncated_column_reads_as_null() {
        let mut batch = encode_batch(&[vec![SqlValue::Int(1)], vec![SqlValue::Int(2)]], 1);
        batch.columns[0].ints.pop();
        assert_eq!(decode_batch(&batch), vec![vec![SqlValue::Int(1)], vec![SqlValue::Null]]);
    }

    // A fake `wsdb` server on a loopback listener: enough of the protocol to exercise the
    // client's framing, the per-statement `Batch` shape, the credit flow and the reconnect.

    pub(crate) type Served = Arc<Mutex<Vec<pb::ClientMessage>>>;

    /// Accepts connections forever; `script` answers one request. Every client frame is kept in
    /// the returned log. An exec of `drop` closes the socket when `drop_once` is set, once.
    pub(crate) async fn serve(
        script: fn(&pb::ClientMessage) -> Vec<pb::ServerMessage>,
        drop_once: bool,
    ) -> (String, Served) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}/db", listener.local_addr().unwrap());
        let log: Served = Arc::new(Mutex::new(Vec::new()));
        let seen = log.clone();
        tokio::spawn(async move {
            let dropped_once = Arc::new(AtomicBool::new(false));
            loop {
                let (tcp, _) = listener.accept().await.unwrap();
                let (seen, dropped_once) = (seen.clone(), dropped_once.clone());
                tokio::spawn(async move {
                    let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
                    while let Some(Ok(Message::Binary(b))) = ws.next().await {
                        let msg = pb::ClientMessage::decode(b.as_ref()).unwrap();
                        seen.lock().unwrap().push(msg.clone());
                        if drop_once
                            && matches!(&msg.message, Some(pb::client_message::Message::Exec(s)) if s.sql == "drop")
                            && !dropped_once.swap(true, Ordering::SeqCst)
                        {
                            break; // the socket closes with the request unanswered
                        }
                        for mut resp in script(&msg) {
                            resp.request_id = msg.request_id;
                            if ws.send(Message::binary(resp.encode_to_vec())).await.is_err() {
                                break;
                            }
                        }
                    }
                });
            }
        });
        (url, log)
    }

    pub(crate) fn reply(m: pb::server_message::Message) -> pb::ServerMessage {
        pb::ServerMessage { request_id: 0, message: Some(m) }
    }

    pub(crate) fn hello_ok() -> pb::ServerMessage {
        reply(pb::server_message::Message::HelloOk(pb::HelloOk {
            tip: 7,
            ping_interval_ms: 0,
            ping_timeout_ms: 0,
        }))
    }

    pub(crate) fn script(m: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
        use pb::client_message::Message as C;
        use pb::server_message::Message as S;
        match &m.message {
            Some(C::Hello(h)) if h.token.split('.').count() == 3 && h.database_id == "DB_1" => {
                vec![hello_ok()]
            }
            Some(C::Hello(_)) => vec![reply(S::Error(pb::Error {
                code: "not_found".into(),
                message: "no such database".into(),
            }))],
            Some(C::Exec(s)) if s.sql == "boom" => vec![reply(S::Error(pb::Error {
                code: "exec".into(),
                message: "constraint".into(),
            }))],
            Some(C::Exec(s)) if s.sql == "busy" => {
                vec![reply(S::Error(pb::Error { code: "retry".into(), message: "busy".into() }))]
            }
            Some(C::Exec(_)) => vec![reply(S::ExecResult(pb::ExecResult {
                rows_affected: 1,
                last_insert_id: 9,
                tip: 8,
                statement: 0,
            }))],
            Some(C::Query(_)) => {
                // Two batches, so the client's per-batch credit is observable.
                let mut b1 = encode_batch(&[vec![SqlValue::Int(1), "a".into()]], 2);
                let mut b2 = encode_batch(&[vec![SqlValue::Int(2), SqlValue::Null]], 2);
                b1.statement = 0;
                b2.statement = 0;
                vec![
                    reply(S::Columns(pb::Columns {
                        names: vec!["n".into(), "s".into()],
                        statement: 0,
                    })),
                    reply(S::ColumnBatch(b1)),
                    reply(S::ColumnBatch(b2)),
                    reply(S::Done(pb::Done { tip: 8, total_rows: 2 })),
                ]
            }
            Some(C::Batch(b)) => {
                // Statement 0: an exec. Statement 1: rows. Statement 2: a RETURNING with no rows,
                // which still sends its Columns. Statement 3: rows again.
                assert_eq!(b.statements.len(), 4);
                let mut rows1 = encode_batch(&[vec![SqlValue::Int(3)]], 1);
                rows1.statement = 1;
                let mut rows3 = encode_batch(&[vec!["x".into()], vec!["y".into()]], 1);
                rows3.statement = 3;
                vec![
                    reply(S::ExecResult(pb::ExecResult {
                        rows_affected: 2,
                        last_insert_id: 0,
                        tip: 0,
                        statement: 0,
                    })),
                    reply(S::Columns(pb::Columns { names: vec!["version".into()], statement: 1 })),
                    reply(S::ColumnBatch(rows1)),
                    reply(S::Columns(pb::Columns { names: vec!["version".into()], statement: 2 })),
                    reply(S::Columns(pb::Columns { names: vec!["id".into()], statement: 3 })),
                    reply(S::ColumnBatch(rows3)),
                    reply(S::Done(pb::Done { tip: 9, total_rows: 3 })),
                ]
            }
            _ => vec![],
        }
    }

    fn st(sql: &'static str, params: Vec<SqlValue>) -> Statement {
        Statement { sql: sql.into(), params }
    }

    fn minter() -> TokenMinter {
        TokenMinter::new("devkey", "secret").unwrap()
    }

    pub(crate) async fn dial(url: &str, db: &str) -> Result<Conn, WsdbError> {
        let mut opts = Options::new(url, db);
        opts.reconnect.min_backoff = Duration::from_millis(10);
        opts.reconnect.budget = Duration::from_secs(5);
        Conn::dial(opts, minter()).await
    }

    #[tokio::test]
    async fn hello_carries_the_token_and_a_refused_database_is_a_server_error() {
        let (url, _) = serve(script, false).await;
        let conn = dial(&url, "DB_1").await.unwrap();
        assert_eq!(conn.tip(), 7);
        let err = dial(&url, "DB_nope").await.err().unwrap();
        assert!(matches!(&err, WsdbError::Server { code, .. } if code == "not_found"), "{err}");
        conn.close();
    }

    #[tokio::test]
    async fn exec_query_and_credits() {
        let (url, log) = serve(script, false).await;
        let conn = dial(&url, "DB_1").await.unwrap();
        let r = conn.exec(&st("INSERT", vec!["v".into()])).await.unwrap();
        assert_eq!((r.rows_affected, r.last_insert_id, r.tip), (1, 9, 8));
        let err = conn.exec(&st("boom", vec![])).await.err().unwrap();
        assert!(matches!(err, WsdbError::Server { code, .. } if code == "exec"));
        // `retry` applied nothing: sent three times, then the error
        let err = conn.exec(&st("busy", vec![])).await.err().unwrap();
        assert!(matches!(err, WsdbError::Server { code, .. } if code == "retry"));
        let busy = |log: &Served| {
            log.lock()
                .unwrap()
                .iter()
                .filter(|m| matches!(&m.message, Some(pb::client_message::Message::Exec(s)) if s.sql == "busy"))
                .count()
        };
        assert_eq!(busy(&log), 3);

        let rows = conn.query_all(&st("SELECT", vec![])).await.unwrap();
        assert_eq!(
            rows,
            vec![vec![SqlValue::Int(1), "a".into()], vec![SqlValue::Int(2), SqlValue::Null]]
        );
        // One credit per batch consumed, on the query's request id.
        tokio::time::sleep(Duration::from_millis(50)).await;
        let log = log.lock().unwrap();
        let query_id = log
            .iter()
            .find(|m| matches!(m.message, Some(pb::client_message::Message::Query(_))))
            .unwrap()
            .request_id;
        let credits = log
            .iter()
            .filter(|m| {
                m.request_id == query_id
                    && matches!(m.message, Some(pb::client_message::Message::Credit(_)))
            })
            .count();
        assert_eq!(credits, 2);
    }

    #[tokio::test]
    async fn a_batch_answers_per_statement_and_an_empty_returning_is_rows() {
        let (url, _) = serve(script, false).await;
        let conn = dial(&url, "DB_1").await.unwrap();
        let stmts: Vec<Statement> = (0..4).map(|_| st("S", vec![])).collect();
        let out = conn.batch(&stmts).await.unwrap();
        assert_eq!(conn.tip(), 9);
        assert_eq!(out.results[0], StatementResult::Exec { rows_affected: 2, last_insert_id: 0 });
        assert_eq!(out.results[1].rows(), &[vec![SqlValue::Int(3)]]);
        // the acquire that lost: Columns, no ColumnBatch
        assert_eq!(out.results[2], StatementResult::Rows(vec![]));
        assert_eq!(out.results[3].rows().len(), 2);
    }

    #[tokio::test]
    async fn a_dropped_socket_fails_the_request_in_flight_and_redials() {
        let (url, log) = serve(script, true).await;
        let conn = dial(&url, "DB_1").await.unwrap();
        // The fake drops the socket on this exec: the request fails and the client redials.
        let err = conn.exec(&st("drop", vec![])).await.err().unwrap();
        assert!(matches!(err, WsdbError::Disconnected), "{err}");
        let r = conn.exec(&st("INSERT", vec![])).await.unwrap();
        assert_eq!(r.rows_affected, 1);
        let hellos = log
            .lock()
            .unwrap()
            .iter()
            .filter(|m| matches!(m.message, Some(pb::client_message::Message::Hello(_))))
            .count();
        assert_eq!(hellos, 2, "one Hello per dial: the first and the redial");
        conn.close();
        assert!(matches!(conn.exec(&st("INSERT", vec![])).await, Err(WsdbError::Closed)));
    }

    fn hellos(log: &Served) -> usize {
        let log = log.lock().unwrap();
        log.iter()
            .filter(|m| matches!(m.message, Some(pb::client_message::Message::Hello(_))))
            .count()
    }

    #[tokio::test]
    async fn a_socket_that_stops_answering_pings_is_redialed() {
        fn silent(m: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
            use pb::client_message::Message as C;
            match &m.message {
                Some(C::Hello(_)) => {
                    vec![reply(pb::server_message::Message::HelloOk(pb::HelloOk {
                        tip: 7,
                        ping_interval_ms: 20,
                        ping_timeout_ms: 20,
                    }))]
                }
                Some(C::Ping(_)) => vec![], // the node is gone: no pong
                other => {
                    script(&pb::ClientMessage { request_id: m.request_id, message: other.clone() })
                }
            }
        }
        let (url, log) = serve(silent, false).await;
        let conn = dial(&url, "DB_1").await.unwrap();
        tokio::time::sleep(Duration::from_millis(1500)).await;
        assert!(hellos(&log) >= 2, "redialed past the read deadline");
        assert_eq!(conn.exec(&st("INSERT", vec![])).await.unwrap().rows_affected, 1);
        conn.close();
    }

    #[tokio::test]
    async fn a_redial_the_server_refuses_for_good_fails_later_calls_with_its_error() {
        fn refuses_again(m: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
            static HELLOS: AtomicU32 = AtomicU32::new(0);
            match &m.message {
                Some(pb::client_message::Message::Hello(_))
                    if HELLOS.fetch_add(1, Ordering::SeqCst) > 0 =>
                {
                    vec![reply(pb::server_message::Message::Error(pb::Error {
                        code: "not_found".into(),
                        message: "the database was deleted".into(),
                    }))]
                }
                _ => script(m),
            }
        }
        let (url, log) = serve(refuses_again, true).await;
        let conn = dial(&url, "DB_1").await.unwrap();
        assert!(matches!(conn.exec(&st("drop", vec![])).await, Err(WsdbError::Disconnected)));
        let err = conn.exec(&st("INSERT", vec![])).await.err().unwrap();
        assert!(matches!(&err, WsdbError::Server { code, .. } if code == "not_found"), "{err}");
        assert_eq!(hellos(&log), 2, "no redial after a final refusal");
    }

    #[tokio::test]
    async fn a_frame_over_the_limit_is_refused_before_it_is_sent() {
        let (url, _) = serve(script, false).await;
        let conn = dial(&url, "DB_1").await.unwrap();
        let big = SqlValue::Text("x".repeat(MAX_FRAME + 1));
        assert!(matches!(conn.exec(&st("INSERT", vec![big])).await, Err(WsdbError::TooLarge(_))));
        assert_eq!(conn.exec(&st("INSERT", vec![])).await.unwrap().rows_affected, 1, "still open");
        conn.close();
    }

    #[tokio::test]
    async fn dropping_the_last_handle_closes_the_connection() {
        let (url, _) = serve(script, false).await;
        let conn = dial(&url, "DB_1").await.unwrap();
        let inner = Arc::downgrade(&conn.0);
        let other = conn.clone();
        drop(conn);
        assert!(!other.is_closed(), "a clone keeps it open");
        drop(other);
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(inner.upgrade().is_none(), "its tasks no longer hold it");
    }

    #[test]
    fn the_token_carries_the_database_admin_grant_and_verifies() {
        use livekit_api::access_token::TokenVerifier;
        let token = TokenMinter::new("devkey", "secret").unwrap().mint().unwrap();
        let claims = TokenVerifier::with_api_key("devkey", "secret").verify(&token).unwrap();
        assert!(claims.agent.database_admin);
        assert_eq!(claims.exp - claims.nbf, TOKEN_TTL.as_secs() as usize);
    }
}
