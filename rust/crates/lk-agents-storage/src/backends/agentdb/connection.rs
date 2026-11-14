use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use futures_util::{SinkExt as _, StreamExt as _};
pub use livekit_protocol::agent_db::wire;
use lk_agents_abi::{Clock, Headers, Spawner, Ts};
use lk_agents_net::signal::{self, Producer};
use lk_agents_net::task::TaskSet;
use lk_agents_net::transport::TransportError;
use lk_agents_net::websocket::{Message, RawWebSocket, connect_with_limit};
use parking_lot::Mutex;
use prost::Message as _;
use tokio::sync::mpsc;
use wire as pb;

use crate::StorageError;
use crate::sql::{BatchResult, Row, SqlValue, Statement, StatementResult};
use lk_agents_abi::storage::agentdb_error::{ErrorCode, Retry};

// Wire protocol: `Hello` first, with a token minted for this dial. `Exec` answers one `ExecResult`;
// `Query` streams `Columns`, `ColumnBatch`* and `Done`, paced by `Credit`; `Batch` applies its
// statements atomically, one result per statement, then `Done`. Every request carries its caller's
// deadline: it waits for a socket and for its answer until then, and is never sent again here. A
// dropped socket (the server's `idle_evicted` notice on request id 0 included) is redialed only
// while a request waits for it, so the redial lives exactly as long as a caller's deadline; those
// in flight fail `Disconnected` (outcome unknown). Socket-less, the connection is dormant, which
// no pool takes back. Answers and outgoing frames queue in bounded channels: a full one holds its
// sender back.

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
        sql: s.sql().to_owned(),
        params: s.params().iter().map(to_wire).collect(),
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
    /// A server-reported error with its stable code (`not_found`, `unauthenticated`, `unavailable`,
    /// `retry`, `outcome_unknown`, ..).
    #[error("agentdb {}: {message}", code.as_str())]
    Server { code: ErrorCode, message: String },
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

/// A write's failure, by whether its frame reached the socket writer: once it did, the server
/// may have committed it whatever the answer's fate.
#[derive(Debug, Clone, thiserror::Error)]
pub enum WriteError {
    #[error(transparent)]
    NotSent(WsdbError),
    #[error(transparent)]
    Sent(WsdbError),
}

impl WsdbError {
    /// A `Hello` refusal that redialing cannot repair.
    fn is_final(&self) -> bool {
        match self {
            Self::Server { code, .. } => code.retry() == Retry::Never,
            Self::Protocol(_) => true,
            _ => false,
        }
    }

    /// A code this build does not know keeps its name in the message.
    fn server(error: pb::Error) -> Self {
        let code = ErrorCode::from_wire(&error.code);
        let message = match code {
            ErrorCode::Unknown => format!("{}: {}", error.code, error.message),
            _ => error.message,
        };
        Self::Server { code, message }
    }
}

/// The largest frame the server reads; a larger one closes the socket.
pub const MAX_FRAME: usize = 16 << 20;
/// How long a frame may take to send, and the read deadline past a missed pong, when the server
/// names no ping timeout.
const DEFAULT_PING_TIMEOUT: Duration = Duration::from_secs(10);

/// Reconnection, on by default: a node draining or an idle database evicted severs the socket.
/// Redials back off while requests wait; none waiting, nothing redials.
#[derive(Debug, Clone)]
pub struct Reconnect {
    pub disable: bool,
    pub min_backoff: Duration,
    pub max_backoff: Duration,
}

impl Default for Reconnect {
    fn default() -> Self {
        Self {
            disable: false,
            min_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(5),
        }
    }
}

/// Frames queued for the socket writer before a request's send waits.
const WRITE_QUEUE: usize = 32;

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

struct State {
    pending: HashMap<u32, mpsc::Sender<pb::ServerMessage>>,
    /// The live socket's writer; `None` between a drop and the redial.
    tx: Option<mpsc::Sender<Vec<u8>>>,
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
    /// Whether a socket is live, published by the supervisor; ended once it is gone.
    ready: signal::Cell<bool>,
    /// How many requests wait for a socket: the supervisor redials while any does.
    waiting: signal::Shared<usize>,
    /// The executor the connection's supervisor, heartbeat and writer run on.
    spawner: Arc<dyn Spawner>,
    /// What its dial bound, heartbeat, redial backoff and read deadline wait on.
    clock: Arc<dyn Clock>,
    /// The supervisor, heartbeat and socket writers; closing the connection cancels them.
    tasks: Arc<TaskSet>,
}

/// One authenticated connection to one database; cheap to clone, closed by [`Conn::close`] or
/// when the last clone drops. The connection's own tasks hold it without the drop guard, so they
/// do not keep it open.
#[derive(Clone)]
pub struct Conn(
    Arc<Inner>,
    #[allow(dead_code, reason = "held only for its drop, which closes the connection")]
    Option<Arc<CloseOnDrop>>,
);

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

impl Socket {
    /// An error when the writer stopped, nothing (not even a pong) arrived within the read
    /// deadline on `clock`, or the socket ended or failed.
    async fn next_message(&mut self, clock: &dyn Clock) -> Result<Message, WsdbError> {
        let next = async {
            match self.read_deadline {
                Some(d) => lk_agents_abi::timeout(clock, d, self.read.next()).await.ok(),
                None => Some(self.read.next().await),
            }
        };
        let next = tokio::select! {
            next = next => next,
            _ = &mut self.writer_gone => return Err(WsdbError::Io("the socket stopped sending".into())),
        };
        match next {
            None => Err(WsdbError::Timeout), // not even a pong within the deadline
            Some(Some(Ok(message))) => Ok(message),
            Some(None) => Err(WsdbError::Io("socket closed".into())),
            Some(Some(Err(e))) => Err(WsdbError::Io(e.to_string())),
        }
    }
}

/// The task owning the socket's lifetime and the ready gate. However it ends (a give-up, a
/// close, a panic, its runtime dropped), it closes the connection, failing every request in
/// flight, then the gate ends, failing every request waiting on it.
struct Supervisor {
    conn: Conn,
    ready: Producer<bool>,
}

impl Drop for Supervisor {
    fn drop(&mut self) {
        self.conn.close();
    }
}

impl Supervisor {
    /// Reads the live socket and, once it dies, redials while a request waits for it.
    async fn run(self, mut socket: Socket) {
        let conn = &self.conn;
        loop {
            let cause = conn.read_loop(&mut socket).await;
            if conn.is_closed() {
                return;
            }
            self.ready.set(false);
            conn.dropped(&cause);
            if conn.0.opts.reconnect.disable {
                return;
            }
            match conn.redial().await {
                Some(next) => socket = next,
                None => return,
            }
            self.ready.set(true);
        }
    }
}

/// A request waiting for a socket, counted while it waits: what keeps the redial going.
struct Waiting<'a>(&'a signal::Shared<usize>);

impl<'a> Waiting<'a> {
    fn new(waiting: &'a signal::Shared<usize>) -> Self {
        waiting.modify(|n| *n += 1);
        Self(waiting)
    }
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.modify(|n| *n -= 1);
    }
}

impl Conn {
    /// Dials, says `Hello`, and starts the supervisor on `spawner`; every wait reads `clock`.
    pub async fn dial(
        opts: Options,
        minter: TokenMinter,
        spawner: Arc<dyn Spawner>,
        clock: Arc<dyn Clock>,
    ) -> Result<Conn, WsdbError> {
        let (ready, ready_cell) = signal::signal(false);
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
            ready: ready_cell,
            waiting: signal::Shared::new(0),
            spawner,
            clock,
            tasks: Arc::default(),
        });
        let guard = Arc::new(CloseOnDrop(Arc::downgrade(&inner)));
        let conn = Conn(inner, Some(guard));
        let socket = conn.connect().await?;
        ready.set(true);
        let ping_interval = socket.ping_interval;
        let supervisor = Supervisor { conn: conn.unguarded(), ready };
        let spawner = &*conn.0.spawner;
        // Refused once the connection closed: the work is then dropped unstarted.
        conn.0.tasks.try_spawn(spawner, "agentdb supervisor", supervisor.run(socket));
        if !ping_interval.is_zero() {
            let owner = conn.unguarded();
            conn.0.tasks.try_spawn(spawner, "agentdb heartbeat", owner.ping_loop(ping_interval));
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

    /// The caller's deadline, else the default one from now on the connection's clock.
    pub fn deadline(&self) -> Ts {
        crate::deadline::or_default(&*self.0.clock)
    }

    /// Dial, `Hello`, publish the writer.
    async fn connect(&self) -> Result<Socket, WsdbError> {
        let clock = &*self.0.clock;
        let deadline = Some(clock.now().saturating_add(self.0.opts.dial_timeout));
        let token = self
            .0
            .minter
            .mint(&self.0.opts.database_id)
            .map_err(|e| WsdbError::Io(e.to_string()))?;
        // the token also rides the upgrade, so a proxy in front of agent-db can admit the socket;
        // a header, never the URL, which proxies log
        let mut headers = Headers::new();
        headers.insert("Authorization", format!("Bearer {token}"));
        let dial = connect_with_limit(&self.0.opts.url, headers, Some(MAX_FRAME));
        let ws = lk_agents_abi::before(clock, deadline, dial)
            .await
            .map_err(|_| WsdbError::Timeout)?
            .map_err(|e| match e {
                // the upgrade's own refusal is final, as the same refusal at `Hello` is
                TransportError::Status { status: 401 | 403, message, .. } => {
                    WsdbError::Server { code: ErrorCode::Unauthenticated, message }
                }
                TransportError::Status { status: 404, message, .. } => {
                    WsdbError::Server { code: ErrorCode::NotFound, message }
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
        lk_agents_abi::before(clock, deadline, sink.send(Message::binary(hello.encode_to_vec())))
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
        let resp = lk_agents_abi::before(clock, deadline, first)
            .await
            .map_err(|_| WsdbError::Timeout)??;
        if resp.request_id != hello.request_id {
            return Err(WsdbError::Protocol("Hello response has a different request id".into()));
        }
        let ok = match resp.message {
            Some(pb::server_message::Message::HelloOk(ok)) => ok,
            Some(pb::server_message::Message::Error(e)) => {
                return Err(WsdbError::server(e));
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
        let (tx, mut rx) = mpsc::channel::<Vec<u8>>(WRITE_QUEUE);
        let (gone, writer_gone) = tokio::sync::oneshot::channel();
        let writer_clock = self.0.clock.clone();
        // Refused once the connection closed: the sink and `gone` then drop unstarted.
        self.0.tasks.try_spawn(&*self.0.spawner, "agentdb writer", async move {
            // Declared first, so `gone` reports the writer stopped after the sink dropped.
            let _gone = gone;
            let (mut sink, clock) = (sink, &*writer_clock);
            while let Some(bytes) = rx.recv().await {
                let sent =
                    lk_agents_abi::timeout(clock, ping_timeout, sink.send(Message::binary(bytes)));
                if !matches!(sent.await, Ok(Ok(()))) {
                    return;
                }
            }
            // The peer's close frame is a courtesy: the socket drops either way.
            if !matches!(lk_agents_abi::timeout(clock, ping_timeout, sink.close()).await, Ok(Ok(()))) {
                log::debug!(target: lk_agents_abi::LOG_TARGET, "agentdb socket close not acknowledged");
            }
        });
        {
            let mut st = self.0.state.lock();
            st.tip = st.tip.max(ok.tip); // a reconnect can only observe a later tip
            st.tx = Some(tx);
        }
        let read_deadline = (!ping_interval.is_zero())
            .then(|| ping_interval + ping_timeout + Duration::from_secs(1));
        Ok(Socket { read, writer_gone, read_deadline, ping_interval })
    }

    async fn read_loop(&self, socket: &mut Socket) -> WsdbError {
        loop {
            let routed = match socket.next_message(&*self.0.clock).await {
                Ok(Message::Binary(bytes)) => self.route_response(bytes.as_ref()).await,
                Ok(Message::Close(_)) => Err(WsdbError::Io("socket closed".into())),
                Ok(_) => Ok(()),
                Err(error) => Err(error),
            };
            if let Err(error) = routed {
                return error;
            }
        }
    }

    /// Hands a response to the request awaiting it, waiting while its answers are queued full;
    /// a pong only proves the socket is alive.
    async fn route_response(&self, bytes: &[u8]) -> Result<(), WsdbError> {
        let resp =
            pb::ServerMessage::decode(bytes).map_err(|e| WsdbError::Protocol(e.to_string()))?;
        match &resp.message {
            Some(pb::server_message::Message::Pong(_)) => return Ok(()),
            // the connection's own notice, answering no request: its socket ends for being idle
            Some(pb::server_message::Message::Error(e))
                if resp.request_id == 0
                    && ErrorCode::from_wire(&e.code) == ErrorCode::IdleEvicted =>
            {
                return Err(WsdbError::server(e.clone()));
            }
            _ => {}
        }
        let ch = self.0.state.lock().pending.get(&resp.request_id).cloned();
        if let Some(ch) = ch
            && ch.send(resp).await.is_err()
        {
            log::trace!(target: lk_agents_abi::LOG_TARGET, "agentdb answer for a request dropped meanwhile");
        }
        Ok(())
    }

    /// Fails everything in flight. Only after the read loop returned, so no response can still be
    /// delivered.
    fn dropped(&self, cause: &WsdbError) {
        let mut st = self.0.state.lock();
        st.pending.clear();
        st.tx = None;
        drop(st);
        log::warn!(target: lk_agents_abi::LOG_TARGET, databaseId = self.0.opts.database_id.as_str(); "agentdb connection lost: {cause}");
    }

    /// Dials once a request waits for the socket, again with exponential backoff while one
    /// still does; `None` once a refusal is final.
    async fn redial(&self) -> Option<Socket> {
        let (r, clock) = (&self.0.opts.reconnect, &*self.0.clock);
        let mut backoff = r.min_backoff;
        loop {
            self.0.waiting.wait_until(|n| (*n > 0).then_some(())).await;
            match self.connect().await {
                Ok(socket) => {
                    log::info!(target: lk_agents_abi::LOG_TARGET, databaseId = self.0.opts.database_id.as_str(); "agentdb reconnected");
                    return Some(socket);
                }
                Err(e) if e.is_final() => {
                    log::warn!(target: lk_agents_abi::LOG_TARGET, databaseId = self.0.opts.database_id.as_str(); "agentdb redial refused for good: {e}");
                    self.0.state.lock().fatal = Some(e);
                    return None;
                }
                Err(e) => {
                    log::warn!(target: lk_agents_abi::LOG_TARGET, databaseId = self.0.opts.database_id.as_str(); "agentdb redial failed: {e}")
                }
            }
            if self.is_closed() {
                return None;
            }
            clock.wait_until(clock.now().saturating_add(backoff)).await;
            backoff = (backoff * 2).min(r.max_backoff);
        }
    }

    async fn ping_loop(self, interval: Duration) {
        let clock = self.0.clock.clone();
        let mut next = clock.now();
        loop {
            next = next.saturating_add(interval);
            clock.wait_until(next).await;
            if !self.0.ready.get() {
                continue; // mid-reconnect: a ping before Hello is a protocol error
            }
            let now = (lk_agents_abi::unix_secs(clock.wall()) * 1000.0) as i64;
            // a writer queued full is busy sending: the ping is not needed to prove the socket
            let ping = pb::client_message::Message::Ping(pb::Ping { timestamp_ms: now });
            self.try_write(self.next_id(), ping);
        }
    }

    fn is_closed(&self) -> bool {
        self.0.state.lock().closed
    }

    /// Open, its socket live, nothing in flight: a connection another opener can take. A
    /// dormant one (its socket dropped or evicted idle) is not: whoever needs the database dials
    /// afresh.
    pub fn is_idle(&self) -> bool {
        let st = self.0.state.lock();
        !st.closed && self.0.ready.get() && st.pending.is_empty()
    }

    /// Ends the connection; every waiter and every later call fails [`WsdbError::Closed`].
    pub fn close(&self) {
        self.0.close();
    }

    /// Closes and observes settlement of the supervisor, heartbeat and socket writers.
    /// Cancellation of this waiter leaves those tasks owned by the connection tracker.
    pub async fn aclose(&self) {
        self.close();
        self.0.tasks.wait_empty(&*self.0.clock, None).await;
    }

    /// Why a closed connection refuses a call: what made it give up, else `Closed`.
    fn closed_error(&self) -> WsdbError {
        self.0.state.lock().fatal.clone().unwrap_or(WsdbError::Closed)
    }

    fn next_id(&self) -> u32 {
        self.0.next_id.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// A request issued without a socket waits for the redial until `deadline`, then fails
    /// `Timeout`, unsent; the supervisor gone, it closed the connection first. Nothing is sent
    /// with no time left for its answer.
    async fn register(&self, deadline: Ts) -> Result<Request<'_>, WsdbError> {
        if self.is_closed() {
            return Err(self.closed_error());
        }
        if !self.0.ready.get() {
            let _waiting = Waiting::new(&self.0.waiting);
            let ready = self.0.ready.wait_for(|ready| *ready);
            match lk_agents_abi::before(&*self.0.clock, Some(deadline), ready).await {
                Err(_) => return Err(WsdbError::Timeout),
                Ok(Err(_)) => return Err(self.closed_error()),
                Ok(Ok(_)) => {}
            }
        }
        if self.0.clock.now() >= deadline {
            return Err(WsdbError::Timeout);
        }
        let id = self.next_id();
        let (tx, rx) = mpsc::channel(self.0.opts.credits as usize + 2);
        let mut st = self.0.state.lock();
        if st.closed {
            drop(st);
            return Err(self.closed_error());
        }
        st.pending.insert(id, tx);
        Ok(Request { conn: self, id, rx, deadline, answered: false })
    }

    /// The frame for the socket writer, refused when over the server's limit.
    fn frame(request_id: u32, message: pb::client_message::Message) -> Result<Vec<u8>, WsdbError> {
        let frame = pb::ClientMessage { request_id, message: Some(message) };
        let size = frame.encoded_len();
        if size > MAX_FRAME {
            return Err(WsdbError::TooLarge(size));
        }
        Ok(frame.encode_to_vec())
    }

    /// The live socket's writer.
    fn writer(&self) -> Result<mpsc::Sender<Vec<u8>>, WsdbError> {
        let st = self.0.state.lock();
        if st.closed {
            drop(st);
            return Err(self.closed_error());
        }
        st.tx.clone().ok_or(WsdbError::Disconnected)
    }

    /// Queues a frame for the socket, waiting while the writer's queue is full.
    async fn write(
        &self,
        request_id: u32,
        message: pb::client_message::Message,
    ) -> Result<(), WsdbError> {
        let bytes = Self::frame(request_id, message)?;
        self.writer()?.send(bytes).await.map_err(|_| WsdbError::Disconnected)
    }

    /// Queues a frame nothing waits on (a ping, a cancel), dropped when the writer's queue is
    /// full or the socket is gone: neither changes what the server committed.
    fn try_write(&self, request_id: u32, message: pb::client_message::Message) {
        let queued = Self::frame(request_id, message)
            .and_then(|bytes| self.writer()?.try_send(bytes).map_err(|_| WsdbError::Disconnected));
        if let Err(error) = queued {
            log::trace!(target: lk_agents_abi::LOG_TARGET, "agentdb frame not queued: {error}");
        }
    }

    /// The request's next answer, until its deadline: past it, `Timeout`.
    async fn wait(
        &self,
        rx: &mut mpsc::Receiver<pb::ServerMessage>,
        deadline: Ts,
    ) -> Result<pb::ServerMessage, WsdbError> {
        let next = lk_agents_abi::before(&*self.0.clock, Some(deadline), rx.recv());
        match next.await.map_err(|_| WsdbError::Timeout)? {
            None if self.is_closed() => Err(self.closed_error()),
            None => Err(WsdbError::Disconnected),
            Some(pb::ServerMessage {
                message: Some(pb::server_message::Message::Error(e)),
                ..
            }) => Err(WsdbError::server(e)),
            Some(resp) => Ok(resp),
        }
    }

    fn observe_tip(&self, tip: i64) {
        let mut st = self.0.state.lock();
        st.tip = st.tip.max(tip);
    }

    /// One write statement; returns once durable, or fails by `deadline`.
    pub async fn exec(&self, s: &Statement, deadline: Ts) -> Result<ExecResult, WriteError> {
        let stmt = statement(s);
        let mut req = self.register(deadline).await.map_err(WriteError::NotSent)?;
        let exec = pb::client_message::Message::Exec(stmt);
        self.write(req.id, exec).await.map_err(WriteError::NotSent)?;
        self.exec_answer(&mut req).await.map_err(WriteError::Sent)
    }

    async fn exec_answer(&self, req: &mut Request<'_>) -> Result<ExecResult, WsdbError> {
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

    /// A query's rows, all of them, by `deadline`: one credit back per batch consumed, so the
    /// server sends no further than the caller reads.
    pub async fn query_all(&self, s: &Statement, deadline: Ts) -> Result<Vec<Row>, WsdbError> {
        let stmt = statement(s);
        let mut req = self.register(deadline).await?;
        let id = req.id;
        self.write(id, pb::client_message::Message::Query(stmt)).await?;
        let mut results = Results::new(1);
        loop {
            match req.next().await?.message {
                Some(pb::server_message::Message::Columns(columns)) => results.columns(columns)?,
                Some(pb::server_message::Message::ColumnBatch(batch)) => {
                    results.batch(batch)?;
                    let credit = pb::client_message::Message::Credit(pb::Credit { batches: 1 });
                    self.write(id, credit).await?;
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

    /// Statements applied atomically, in order, each answered in order, by `deadline`.
    pub async fn batch(
        &self,
        statements: &[Statement],
        deadline: Ts,
    ) -> Result<BatchResult, WriteError> {
        let statements: Vec<pb::Statement> = statements.iter().map(statement).collect();
        let n = statements.len();
        let mut req = self.register(deadline).await.map_err(WriteError::NotSent)?;
        let batch = pb::client_message::Message::Batch(pb::Batch { statements });
        self.write(req.id, batch).await.map_err(WriteError::NotSent)?;
        self.batch_answer(&mut req, n).await.map_err(WriteError::Sent)
    }

    async fn batch_answer(
        &self,
        req: &mut Request<'_>,
        n: usize,
    ) -> Result<BatchResult, WsdbError> {
        let id = req.id;
        let credit = pb::Credit { batches: self.0.opts.credits };
        self.write(id, pb::client_message::Message::Credit(credit)).await?;
        let mut results = Results::new(n);
        loop {
            match req.next().await?.message {
                Some(pb::server_message::Message::Columns(columns)) => results.columns(columns)?,
                Some(pb::server_message::Message::ColumnBatch(batch)) => {
                    results.batch(batch)?;
                    let credit = pb::client_message::Message::Credit(pb::Credit { batches: 1 });
                    self.write(id, credit).await?;
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
    rx: mpsc::Receiver<pb::ServerMessage>,
    /// The caller's: its answer is awaited until then.
    deadline: Ts,
    answered: bool,
}

impl Request<'_> {
    async fn next(&mut self) -> Result<pb::ServerMessage, WsdbError> {
        self.conn.wait(&mut self.rx, self.deadline).await
    }
}

impl Drop for Request<'_> {
    fn drop(&mut self) {
        self.conn.0.state.lock().pending.remove(&self.id);
        if !self.answered {
            self.conn.try_write(self.id, pb::client_message::Message::Cancel(pb::Cancel {}));
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

    /// A fresh data-plane token for `database_id` alone, signed by net.
    pub fn mint(&self, database_id: &str) -> Result<String, StorageError> {
        lk_agents_net::token::database_token(&self.api_key, &self.api_secret, database_id)
            .map_err(|e| StorageError::Invalid(format!("token: {e}")))
    }
}

enum ResultSet {
    Rows { width: usize, rows: Vec<Row> },
    Exec { rows_affected: i64, last_insert_id: i64 },
}

struct Results {
    sets: Vec<ResultSet>,
    expected: usize,
    rows: u64,
}

impl Results {
    fn new(expected: usize) -> Self {
        Self { sets: Vec::new(), expected, rows: 0 }
    }

    fn start(&self, statement: u32) -> Result<(), WsdbError> {
        if statement as usize != self.sets.len() || self.sets.len() >= self.expected {
            return Err(WsdbError::Protocol(format!(
                "result for statement {statement}, expected {} of {}",
                self.sets.len(),
                self.expected,
            )));
        }
        Ok(())
    }

    fn columns(&mut self, columns: pb::Columns) -> Result<(), WsdbError> {
        self.start(columns.statement)?;
        self.sets.push(ResultSet::Rows { width: columns.names.len(), rows: Vec::new() });
        Ok(())
    }

    fn batch(&mut self, batch: pb::ColumnBatch) -> Result<(), WsdbError> {
        if batch.statement as usize + 1 != self.sets.len() {
            return Err(WsdbError::Protocol("rows arrived outside their statement".into()));
        }
        let Some(ResultSet::Rows { width, rows }) = self.sets.last_mut() else {
            return Err(WsdbError::Protocol("rows arrived without Columns".into()));
        };
        if batch.columns.len() != *width {
            return Err(WsdbError::Protocol("row width differs from Columns".into()));
        }
        let decoded = decode_batch(&batch)?;
        self.rows += u64::from(batch.rows);
        rows.extend(decoded);
        Ok(())
    }

    fn exec(&mut self, result: pb::ExecResult) -> Result<(), WsdbError> {
        self.start(result.statement)?;
        self.sets.push(ResultSet::Exec {
            rows_affected: result.rows_affected,
            last_insert_id: result.last_insert_id,
        });
        Ok(())
    }

    fn finish(self, done: &pb::Done) -> Result<BatchResult, WsdbError> {
        if self.sets.len() != self.expected || self.rows != done.total_rows {
            return Err(WsdbError::Protocol(format!(
                "Done reports {} rows after {} rows and {} of {} statements",
                done.total_rows,
                self.rows,
                self.sets.len(),
                self.expected,
            )));
        }
        Ok(BatchResult {
            results: self
                .sets
                .into_iter()
                .map(|set| match set {
                    ResultSet::Rows { rows, .. } => StatementResult::Rows(rows),
                    ResultSet::Exec { rows_affected, last_insert_id } => {
                        StatementResult::Exec { rows_affected, last_insert_id }
                    }
                })
                .collect(),
        })
    }
}

#[cfg(test)]
#[path = "connection_tests.rs"]
pub(crate) mod tests;
