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
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use futures_util::{SinkExt as _, StreamExt as _};
pub use livekit_protocol::agent_db::wire;
use lk_agents_net::lock::Mutex;
use lk_agents_net::task::{TaskSet, TokioSpawner};
use lk_agents_net::transport::TransportError;
use lk_agents_net::websocket::{Message, RawWebSocket, connect_with_limit};
use prost::Message as _;
use tokio::runtime::Handle;
use tokio::sync::{mpsc, watch};
use wire as pb;

mod results;
use results::Results;

use crate::StorageError;
use crate::sql::{BatchResult, Row, SqlValue, Statement, StatementResult};

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
/// matching array; text and blobs are concatenated with one exclusive end offset each. A column
/// that does not hold what its types announce, or text that is not UTF-8, is a malformed answer:
/// refused, never read as `NULL`.
pub fn decode_batch(b: &wire::ColumnBatch) -> Result<Vec<Row>, WsdbError> {
    use wire::ValueType;
    let malformed = |what: &str| WsdbError::Protocol(format!("a malformed column batch: {what}"));
    let rows = b.rows as usize;
    let row_bytes = b
        .columns
        .len()
        .checked_mul(std::mem::size_of::<SqlValue>())
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Row>()))
        .and_then(|bytes| bytes.checked_mul(rows));
    if row_bytes.is_none_or(|bytes| bytes > MAX_FRAME)
        || (rows > 0 && b.columns.is_empty())
        || b.columns.iter().any(|column| column.types.len() != rows)
    {
        return Err(malformed("row dimensions exceed the decode budget or column types"));
    }
    let mut out: Vec<Row> = (0..rows).map(|_| Vec::with_capacity(b.columns.len())).collect();
    for col in &b.columns {
        let (mut ints, mut doubles) = (col.ints.iter(), col.doubles.iter());
        let (mut texts, mut blobs) = (col.text_ends.iter(), col.blob_ends.iter());
        let (mut text_at, mut blob_at) = (0usize, 0usize);
        for (row, tag) in out.iter_mut().zip(&col.types) {
            let span = |ends: &mut std::slice::Iter<'_, u32>, at: &mut usize, len: usize| {
                let end = *ends.next()? as usize;
                let start = std::mem::replace(at, end);
                (start <= end && end <= len).then_some(start..end)
            };
            let cell = match ValueType::try_from(i32::from(*tag)) {
                Ok(ValueType::Null) => SqlValue::Null,
                Ok(ValueType::Int) => SqlValue::Int(*ints.next().ok_or_else(|| malformed("ints"))?),
                Ok(ValueType::Double) => {
                    SqlValue::Double(*doubles.next().ok_or_else(|| malformed("doubles"))?)
                }
                Ok(ValueType::Text) => {
                    let range = span(&mut texts, &mut text_at, col.text_data.len())
                        .ok_or_else(|| malformed("text offsets"))?;
                    let text = std::str::from_utf8(&col.text_data[range])
                        .map_err(|_| malformed("text that is not UTF-8"))?;
                    SqlValue::Text(text.to_owned())
                }
                Ok(ValueType::Blob) => {
                    let range = span(&mut blobs, &mut blob_at, col.blob_data.len())
                        .ok_or_else(|| malformed("blob offsets"))?;
                    SqlValue::Blob(col.blob_data[range].to_vec())
                }
                Err(_) => return Err(malformed("an unknown value type")),
            };
            row.push(cell);
        }
        if ints.next().is_some()
            || doubles.next().is_some()
            || texts.next().is_some()
            || blobs.next().is_some()
            || text_at != col.text_data.len()
            || blob_at != col.blob_data.len()
        {
            return Err(malformed("unannounced column values"));
        }
    }
    Ok(out)
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

type WsRead = futures_util::stream::SplitStream<RawWebSocket>;

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
    /// The runtime the connection's supervisor, heartbeat and writer run on.
    runtime: Handle,
    /// The supervisor, heartbeat and socket writers; closing the connection cancels them.
    tasks: Arc<TaskSet>,
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
            let mut st = self.state.lock();
            if st.closed {
                return;
            }
            st.closed = true;
            st.pending.clear();
            st.tx = None;
        }
        self.phase.send_replace(Phase::Closed);
        self.tasks.close();
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
    /// Dials, says `Hello`, and starts the supervisor on `runtime`.
    pub async fn dial(
        opts: Options,
        minter: TokenMinter,
        runtime: Handle,
    ) -> Result<Conn, WsdbError> {
        lk_agents_net::transport::install_tls_provider();
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
            runtime,
            tasks: Arc::default(),
        });
        let guard = Arc::new(CloseOnDrop(Arc::downgrade(&inner)));
        let conn = Conn(inner, Some(guard));
        let socket = conn.connect().await?;
        let ping_interval = socket.ping_interval;
        let owner = conn.unguarded();
        let spawner = TokioSpawner::new(conn.0.runtime.clone());
        // Refused once the connection closed: the work is then dropped unstarted.
        conn.0.tasks.try_spawn(&spawner, "agentdb supervisor", owner.supervise(socket));
        if !ping_interval.is_zero() {
            let owner = conn.unguarded();
            conn.0.tasks.try_spawn(&spawner, "agentdb heartbeat", owner.ping_loop(ping_interval));
        }
        Ok(conn)
    }

    /// A handle for the connection's own tasks: it does not keep the connection open.
    fn unguarded(&self) -> Conn {
        Conn(self.0.clone(), None)
    }

    /// The durable commit sequence last observed.
    pub fn tip(&self) -> i64 {
        self.0.state.lock().tip
    }

    pub fn database_id(&self) -> &str {
        &self.0.opts.database_id
    }

    /// Dial, `Hello`, publish the writer.
    async fn connect(&self) -> Result<Socket, WsdbError> {
        let deadline = tokio::time::Instant::now() + self.0.opts.dial_timeout;
        let token = self.0.minter.mint().map_err(|e| WsdbError::Io(e.to_string()))?;
        // the token also rides the upgrade, so a proxy in front of agent-db can admit the socket
        let mut url = url::Url::parse(&self.0.opts.url)
            .map_err(|e| WsdbError::Io(format!("dial {}: {e}", self.0.opts.url)))?;
        url.query_pairs_mut().append_pair("access_token", &token);
        let dial = connect_with_limit(url.as_str(), Vec::new(), Some(MAX_FRAME));
        let ws = tokio::time::timeout_at(deadline, dial)
            .await
            .map_err(|_| WsdbError::Timeout)?
            .map_err(|e| match e {
                // the upgrade's own refusal is final, as the same refusal at `Hello` is
                TransportError::Status { status: 401 | 403, message, .. } => {
                    WsdbError::Server { code: "unauthenticated".into(), message }
                }
                TransportError::Status { status: 404, message, .. } => {
                    WsdbError::Server { code: "not_found".into(), message }
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
        tokio::time::timeout_at(deadline, sink.send(Message::binary(hello.encode_to_vec())))
            .await
            .map_err(|_| WsdbError::Timeout)?
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
        let resp =
            tokio::time::timeout_at(deadline, first).await.map_err(|_| WsdbError::Timeout)??;
        if resp.request_id != hello.request_id {
            return Err(WsdbError::Protocol("Hello response has a different request id".into()));
        }
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
        let spawner = TokioSpawner::new(self.0.runtime.clone());
        // Refused once the connection closed: the sink and `gone` then drop unstarted.
        self.0.tasks.try_spawn(&spawner, "agentdb writer", async move {
            // Declared first, so `gone` reports the writer stopped after the sink dropped.
            let _gone = gone;
            let mut sink = sink;
            while let Some(bytes) = rx.recv().await {
                let sent = tokio::time::timeout(ping_timeout, sink.send(Message::binary(bytes)));
                if !matches!(sent.await, Ok(Ok(()))) {
                    return;
                }
            }
            let _ = tokio::time::timeout(ping_timeout, sink.close()).await;
        });
        {
            let mut st = self.0.state.lock();
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
                        let st = self.0.state.lock();
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
        let mut st = self.0.state.lock();
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
                    self.0.state.lock().fatal = Some(e);
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
        self.0.state.lock().closed
    }

    /// Open, nothing in flight: a connection another opener can take.
    pub fn is_idle(&self) -> bool {
        let st = self.0.state.lock();
        !st.closed && st.pending.is_empty()
    }

    /// Ends the connection; every waiter and every later call fails [`WsdbError::Closed`].
    pub fn close(&self) {
        self.0.close();
    }

    /// Closes and observes settlement of the supervisor, heartbeat and socket writers.
    /// Cancellation of this waiter leaves those tasks owned by the connection tracker.
    pub async fn aclose(&self) {
        self.close();
        self.0.tasks.wait_empty(&lk_agents_net::clock::RealClock::new(), None).await;
    }

    /// Why a closed connection refuses a call: what made it give up, else `Closed`.
    fn closed_error(&self) -> WsdbError {
        self.0.state.lock().fatal.clone().unwrap_or(WsdbError::Closed)
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
        let mut st = self.0.state.lock();
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
        let frame = pb::ClientMessage { request_id, message: Some(message) };
        let size = frame.encoded_len();
        if size > MAX_FRAME {
            return Err(WsdbError::TooLarge(size));
        }
        let bytes = frame.encode_to_vec();
        let st = self.0.state.lock();
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
        let mut st = self.0.state.lock();
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
        let mut results = Results::new(1);
        loop {
            match req.next().await?.message {
                Some(pb::server_message::Message::Columns(columns)) => results.columns(columns)?,
                Some(pb::server_message::Message::ColumnBatch(batch)) => {
                    results.batch(batch)?;
                    self.write(id, pb::client_message::Message::Credit(pb::Credit { batches: 1 }))?;
                }
                Some(pb::server_message::Message::Done(done)) => {
                    let mut result = results.finish(&done)?;
                    req.answered = true;
                    self.observe_tip(done.tip);
                    return match result.results.pop() {
                        Some(StatementResult::Rows(rows)) => Ok(rows),
                        _ => Err(WsdbError::Protocol("a query did not return rows".into())),
                    };
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
        self.write(id, pb::client_message::Message::Batch(pb::Batch { statements }))?;
        self.write(
            id,
            pb::client_message::Message::Credit(pb::Credit { batches: self.0.opts.credits }),
        )?;
        let mut results = Results::new(n);
        loop {
            match req.next().await?.message {
                Some(pb::server_message::Message::Columns(columns)) => results.columns(columns)?,
                Some(pb::server_message::Message::ColumnBatch(batch)) => {
                    results.batch(batch)?;
                    self.write(id, pb::client_message::Message::Credit(pb::Credit { batches: 1 }))?;
                }
                Some(pb::server_message::Message::ExecResult(result)) => results.exec(result)?,
                Some(pb::server_message::Message::Done(done)) => {
                    let result = results.finish(&done)?;
                    req.answered = true;
                    self.observe_tip(done.tip);
                    return Ok(result);
                }
                other => {
                    return Err(WsdbError::Protocol(format!(
                        "unexpected batch response {other:?}"
                    )));
                }
            }
        }
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
        self.conn.0.state.lock().pending.remove(&self.id);
        if !self.answered {
            let _ = self.conn.write(self.id, pb::client_message::Message::Cancel(pb::Cancel {}));
        }
    }
}

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

    /// A fresh token with the `agent.databaseAdmin` grant, signed by net.
    pub fn mint(&self) -> Result<String, StorageError> {
        lk_agents_net::token::database_admin_token(&self.api_key, &self.api_secret)
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
pub(crate) mod tests;
