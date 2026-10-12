//! The `wsdb` data-plane client: one authenticated WebSocket to one database, speaking
//! `ClientMessage`/`ServerMessage` (one binary protobuf per frame, correlated by a client-chosen
//! `request_id`). The Go client in agents-private/agent-db (`pkg/agentdb/client`) is the reference;
//! the semantics are its:
//!
//! * `Hello` first, with a token minted for this dial and the database id; the answer carries the
//!   durable tip and the ping interval.
//! * `Exec` answers one `ExecResult`. `Query` streams `Columns`, `ColumnBatch`* and `Done`, paced by
//!   `Credit`: the server grants an initial window on `Query` and this client replenishes one
//!   credit per batch it consumes; `Cancel` stops a query early.
//! * `Batch` applies its statements atomically and answers one result per statement, tagged with
//!   the statement index (plan 3.3.2), then one `Done`.
//! * `Begin`/`Commit`/`Rollback` drive the database's interactive transaction.
//! * A dropped socket is redialed with backoff inside a budget, with a fresh `Hello`, so a request
//!   issued mid-reconnect waits instead of failing; requests in flight fail [`WsdbError::Disconnected`]
//!   (outcome unknown), and a transaction open at the drop is gone: every call fails
//!   [`WsdbError::TxnAborted`] until `rollback` acknowledges it. The statements layer maps both to
//!   `StorageUnavailable`, retryable (plan 13.5).

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use futures_util::{SinkExt as _, StreamExt as _};
use prost::Message as _;
use tokio::sync::{mpsc, watch};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest as _;

use crate::token::TokenMinter;
use crate::wire::{self, Row, SqlValue, decode_batch, wire as pb};

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
    /// A reconnect discarded the interactive transaction; `rollback` acknowledges it.
    #[error("agentdb client: transaction aborted by reconnect")]
    TxnAborted,
    /// A server-reported error with its stable code (`not_found`, `unauthenticated`, `unavailable`, ..).
    #[error("agentdb {code}: {message}")]
    Server { code: String, message: String },
    #[error("agentdb client: protocol: {0}")]
    Protocol(String),
    #[error("agentdb client: {0}")]
    Io(String),
    #[error("agentdb client: dial timed out")]
    Timeout,
}

/// Reconnection, on by default: databases are severed for ordinary reasons (a node draining, an
/// idle database evicted, an owner lost), so redialing is the normal path.
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

/// A durable write's report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExecResult {
    pub rows_affected: i64,
    pub last_insert_id: i64,
    pub tip: i64,
}

/// A query's rows, collected.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RowSet {
    pub columns: Vec<String>,
    pub rows: Vec<Row>,
    pub total_rows: u64,
    pub tip: i64,
}

/// One statement's result in a `Batch`.
#[derive(Debug, Clone, PartialEq)]
pub enum StatementResult {
    Rows(RowSet),
    Exec { rows_affected: i64, last_insert_id: i64 },
}

impl StatementResult {
    /// The rows of a statement that returned some; empty for one that did not.
    pub fn rows(&self) -> &[Row] {
        match self {
            StatementResult::Rows(r) => &r.rows,
            StatementResult::Exec { .. } => &[],
        }
    }

    pub fn rows_affected(&self) -> i64 {
        match self {
            StatementResult::Rows(_) => 0,
            StatementResult::Exec { rows_affected, .. } => *rows_affected,
        }
    }
}

/// A `Batch`'s answer: one result per statement, in order, and the tip the batch committed at.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchResult {
    pub results: Vec<StatementResult>,
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
    in_txn: bool,
    txn_lost: bool,
    closed: bool,
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
/// when the last clone drops.
#[derive(Clone)]
pub struct Conn(Arc<Inner>);

impl Drop for Inner {
    fn drop(&mut self) {
        for t in self.tasks.get_mut().map(std::mem::take).unwrap_or_default() {
            t.abort();
        }
    }
}

impl Conn {
    /// Dials, says `Hello`, and starts the supervisor; a service that cannot be reached, or a
    /// refused token, is reported here.
    pub async fn dial(opts: Options, minter: TokenMinter) -> Result<Conn, WsdbError> {
        crate::install_tls_provider();
        let (phase, _) = watch::channel(Phase::Connecting);
        let inner = Arc::new(Inner {
            opts,
            minter,
            next_id: AtomicU32::new(0),
            state: Mutex::new(State {
                pending: HashMap::new(),
                tx: None,
                in_txn: false,
                txn_lost: false,
                closed: false,
                tip: 0,
            }),
            phase,
            tasks: Mutex::new(Vec::new()),
        });
        let conn = Conn(inner);
        let (read, ping_interval) = conn.connect().await?;
        let supervisor = tokio::spawn(conn.clone().supervise(read));
        let mut tasks = vec![supervisor];
        if !ping_interval.is_zero() {
            tasks.push(tokio::spawn(conn.clone().ping_loop(ping_interval)));
        }
        conn.0.tasks.lock().expect("tasks").extend(tasks);
        Ok(conn)
    }

    /// The durable commit sequence last observed.
    pub fn tip(&self) -> i64 {
        self.0.state.lock().expect("state").tip
    }

    pub fn database_id(&self) -> &str {
        &self.0.opts.database_id
    }

    /// Dial, `Hello`, publish the writer. The handshake is a strict request/response exchange, so it
    /// runs before the demux loop sees the socket.
    async fn connect(&self) -> Result<(WsRead, Duration), WsdbError> {
        let token = self.0.minter.mint().map_err(|e| WsdbError::Io(e.to_string()))?;
        // the token travels on the upgrade too, as access_token, so a proxy in front of agent-db
        // (the public API server for a self-hosted worker) can admit the connection before the
        // first frame
        let mut url = url::Url::parse(&self.0.opts.url)
            .map_err(|e| WsdbError::Io(format!("dial {}: {e}", self.0.opts.url)))?;
        url.query_pairs_mut().append_pair("access_token", &token);
        let request = url
            .as_str()
            .into_client_request()
            .map_err(|e| WsdbError::Io(format!("dial {}: {e}", self.0.opts.url)))?;
        let dial = tokio_tungstenite::connect_async(request);
        let (ws, _) = tokio::time::timeout(self.0.opts.dial_timeout, dial)
            .await
            .map_err(|_| WsdbError::Timeout)?
            .map_err(|e| WsdbError::Io(format!("dial {}: {e}", self.0.opts.url)))?;
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
        let resp = tokio::time::timeout(self.0.opts.dial_timeout, read_one(&mut read))
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
        // The writer task owns the sink: frames are sent in the order callers hand them over, and a
        // frame handed over after the socket died is dropped, which its request sees as a drop.
        let (tx, mut rx) = mpsc::unbounded_channel::<Vec<u8>>();
        tokio::spawn(async move {
            while let Some(bytes) = rx.recv().await {
                if sink.send(Message::binary(bytes)).await.is_err() {
                    break;
                }
            }
            let _ = sink.close().await;
        });
        {
            let mut st = self.0.state.lock().expect("state");
            st.tip = st.tip.max(ok.tip); // a reconnect can only observe a later tip
            st.tx = Some(tx);
        }
        self.0.phase.send_replace(Phase::Ready);
        Ok((read, Duration::from_millis(u64::from(ok.ping_interval_ms))))
    }

    /// Reads the live socket and reconnects when it dies: one task owns the socket's lifetime.
    async fn supervise(self, mut read: WsRead) {
        loop {
            let cause = self.read_loop(&mut read).await;
            if self.is_closed() {
                return;
            }
            self.dropped(&cause);
            if self.0.opts.reconnect.disable {
                self.close();
                return;
            }
            match self.redial().await {
                Some(next) => read = next,
                None => {
                    self.close();
                    return;
                }
            }
        }
    }

    async fn read_loop(&self, read: &mut WsRead) -> WsdbError {
        loop {
            match read.next().await {
                Some(Ok(Message::Binary(bytes))) => match pb::ServerMessage::decode(bytes.as_ref())
                {
                    Ok(resp) => self.dispatch(resp),
                    Err(e) => return WsdbError::Protocol(e.to_string()),
                },
                Some(Ok(Message::Close(_))) | None => return WsdbError::Io("socket closed".into()),
                Some(Ok(_)) => {}
                Some(Err(e)) => return WsdbError::Io(e.to_string()),
            }
        }
    }

    fn dispatch(&self, resp: pb::ServerMessage) {
        if matches!(resp.message, Some(pb::server_message::Message::Pong(_))) {
            return;
        }
        let st = self.0.state.lock().expect("state");
        if let Some(ch) = st.pending.get(&resp.request_id) {
            let _ = ch.send(resp);
        }
    }

    /// Fails everything in flight and closes the ready gate. Only after the read loop returned, so
    /// no response can still be delivered.
    fn dropped(&self, cause: &WsdbError) {
        let mut st = self.0.state.lock().expect("state");
        st.pending.clear();
        st.tx = None;
        if st.in_txn {
            st.in_txn = false;
            st.txn_lost = true;
        }
        drop(st);
        self.0.phase.send_replace(Phase::Connecting);
        log::warn!(target: "livekit.agents", databaseId = self.0.opts.database_id.as_str(); "agentdb connection lost: {cause}");
    }

    /// Exponential backoff until the budget runs out.
    async fn redial(&self) -> Option<WsRead> {
        let r = &self.0.opts.reconnect;
        let deadline = tokio::time::Instant::now() + r.budget;
        let mut backoff = r.min_backoff;
        loop {
            match self.connect().await {
                Ok((read, _)) => {
                    log::info!(target: "livekit.agents", databaseId = self.0.opts.database_id.as_str(); "agentdb reconnected");
                    return Some(read);
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

    /// Ends the connection; every waiter and every later call fails [`WsdbError::Closed`].
    pub fn close(&self) {
        {
            let mut st = self.0.state.lock().expect("state");
            if st.closed {
                return;
            }
            st.closed = true;
            st.pending.clear();
            st.tx = None;
        }
        self.0.phase.send_replace(Phase::Closed);
        for t in self.0.tasks.lock().map(|mut t| std::mem::take(&mut *t)).unwrap_or_default() {
            t.abort();
        }
    }

    fn next_id(&self) -> u32 {
        self.0.next_id.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Blocks while a reconnect is in flight, so a request issued during one is delayed rather than
    /// failed. Never waits on a closed connection or a lost transaction.
    async fn wait_ready(&self) -> Result<(), WsdbError> {
        {
            let st = self.0.state.lock().expect("state");
            if st.closed {
                return Err(WsdbError::Closed);
            }
            if st.txn_lost {
                return Err(WsdbError::TxnAborted);
            }
        }
        let mut rx = self.0.phase.subscribe();
        match rx.wait_for(|p| *p != Phase::Connecting).await {
            Ok(p) if *p == Phase::Ready => Ok(()),
            _ => Err(WsdbError::Closed),
        }
    }

    async fn register(
        &self,
    ) -> Result<(u32, mpsc::UnboundedReceiver<pb::ServerMessage>), WsdbError> {
        self.wait_ready().await?;
        let id = self.next_id();
        let (tx, rx) = mpsc::unbounded_channel();
        let mut st = self.0.state.lock().expect("state");
        if st.closed {
            return Err(WsdbError::Closed);
        }
        st.pending.insert(id, tx);
        Ok((id, rx))
    }

    fn unregister(&self, id: u32) {
        self.0.state.lock().expect("state").pending.remove(&id);
    }

    fn write(
        &self,
        request_id: u32,
        message: pb::client_message::Message,
    ) -> Result<(), WsdbError> {
        let bytes = pb::ClientMessage { request_id, message: Some(message) }.encode_to_vec();
        let st = self.0.state.lock().expect("state");
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
            None if self.is_closed() => Err(WsdbError::Closed),
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

    /// One write statement; returns once durable (or, inside a transaction, once applied).
    pub async fn exec(&self, sql: &str, params: Vec<SqlValue>) -> Result<ExecResult, WsdbError> {
        let (id, mut rx) = self.register().await?;
        let r = async {
            self.write(id, pb::client_message::Message::Exec(wire::statement(sql, params)))?;
            match self.wait(&mut rx).await?.message {
                Some(pb::server_message::Message::ExecResult(r)) => Ok(r),
                other => Err(WsdbError::Protocol(format!("unexpected exec response {other:?}"))),
            }
        }
        .await;
        self.unregister(id);
        let r = r?;
        self.observe_tip(r.tip);
        Ok(ExecResult {
            rows_affected: r.rows_affected,
            last_insert_id: r.last_insert_id,
            tip: r.tip,
        })
    }

    /// A streaming query. The server grants the initial window on `Query`; [`Rows`] replenishes
    /// one credit per batch consumed and cancels the stream when dropped early.
    pub async fn query(&self, sql: &str, params: Vec<SqlValue>) -> Result<Rows, WsdbError> {
        let (id, rx) = self.register().await?;
        if let Err(e) =
            self.write(id, pb::client_message::Message::Query(wire::statement(sql, params)))
        {
            self.unregister(id);
            return Err(e);
        }
        Ok(Rows { conn: self.clone(), id, rx, columns: None, done: false, total_rows: 0, tip: 0 })
    }

    /// A query's rows, all of them.
    pub async fn query_all(&self, sql: &str, params: Vec<SqlValue>) -> Result<RowSet, WsdbError> {
        self.query(sql, params).await?.collect().await
    }

    /// Statements applied atomically, in order, each answered in order.
    pub async fn batch(&self, statements: Vec<pb::Statement>) -> Result<BatchResult, WsdbError> {
        let n = statements.len();
        let (id, mut rx) = self.register().await?;
        let r = async {
            self.write(id, pb::client_message::Message::Batch(pb::Batch { statements }))?;
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
                match self.wait(&mut rx).await?.message {
                    Some(pb::server_message::Message::Columns(c)) => {
                        let i = slot(&mut results, c.statement)?;
                        results[i] = Some(StatementResult::Rows(RowSet {
                            columns: c.names,
                            ..RowSet::default()
                        }));
                    }
                    Some(pb::server_message::Message::ColumnBatch(b)) => {
                        let i = slot(&mut results, b.statement)?;
                        let rows = decode_batch(&b);
                        match &mut results[i] {
                            Some(StatementResult::Rows(set)) => set.rows.extend(rows),
                            _ => {
                                results[i] = Some(StatementResult::Rows(RowSet {
                                    rows,
                                    ..RowSet::default()
                                }))
                            }
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
                        return Ok(BatchResult { results, tip: d.tip });
                    }
                    other => {
                        return Err(WsdbError::Protocol(format!(
                            "unexpected batch response {other:?}"
                        )));
                    }
                }
            }
        }
        .await;
        self.unregister(id);
        let r = r?;
        self.observe_tip(r.tip);
        Ok(r)
    }

    pub async fn begin(&self) -> Result<(), WsdbError> {
        self.simple(pb::client_message::Message::Begin(pb::Begin {})).await?;
        self.0.state.lock().expect("state").in_txn = true;
        Ok(())
    }

    /// Returns the tip the transaction committed at.
    pub async fn commit(&self) -> Result<i64, WsdbError> {
        let r = self.simple(pb::client_message::Message::Commit(pb::Commit {})).await?;
        self.0.state.lock().expect("state").in_txn = false;
        self.observe_tip(r.tip);
        Ok(r.tip)
    }

    /// Also acknowledges a transaction a reconnect discarded: the statements are gone either way,
    /// and the connection is usable again afterwards.
    pub async fn rollback(&self) -> Result<(), WsdbError> {
        let lost = {
            let mut st = self.0.state.lock().expect("state");
            let lost = st.txn_lost;
            st.txn_lost = false;
            st.in_txn = false;
            lost
        };
        if lost {
            return Ok(());
        }
        self.simple(pb::client_message::Message::Rollback(pb::Rollback {})).await.map(|_| ())
    }

    async fn simple(
        &self,
        message: pb::client_message::Message,
    ) -> Result<pb::ExecResult, WsdbError> {
        let (id, mut rx) = self.register().await?;
        let r = async {
            self.write(id, message)?;
            match self.wait(&mut rx).await?.message {
                Some(pb::server_message::Message::ExecResult(r)) => Ok(r),
                other => Err(WsdbError::Protocol(format!("unexpected response {other:?}"))),
            }
        }
        .await;
        self.unregister(id);
        r
    }
}

/// A streaming query's rows. Drop before the end cancels the stream.
pub struct Rows {
    conn: Conn,
    id: u32,
    rx: mpsc::UnboundedReceiver<pb::ServerMessage>,
    columns: Option<Vec<String>>,
    done: bool,
    total_rows: u64,
    tip: i64,
}

impl Rows {
    /// Blocks until the column list arrives.
    pub async fn columns(&mut self) -> Result<&[String], WsdbError> {
        while self.columns.is_none() && !self.done {
            self.pump().await?;
        }
        Ok(self.columns.as_deref().unwrap_or(&[]))
    }

    /// The next batch of rows, `None` at the end of the stream.
    pub async fn next_batch(&mut self) -> Result<Option<Vec<Row>>, WsdbError> {
        loop {
            if self.done {
                return Ok(None);
            }
            if let Some(rows) = self.pump().await? {
                return Ok(Some(rows));
            }
        }
    }

    /// Everything, collected.
    pub async fn collect(mut self) -> Result<RowSet, WsdbError> {
        let mut rows = Vec::new();
        while let Some(batch) = self.next_batch().await? {
            rows.extend(batch);
        }
        Ok(RowSet {
            columns: self.columns.take().unwrap_or_default(),
            rows,
            total_rows: self.total_rows,
            tip: self.tip,
        })
    }

    /// Valid once `next_batch` returned `None`.
    pub fn total_rows(&self) -> u64 {
        self.total_rows
    }

    async fn pump(&mut self) -> Result<Option<Vec<Row>>, WsdbError> {
        let resp = match self.conn.wait(&mut self.rx).await {
            Ok(r) => r,
            Err(e) => {
                self.done = true;
                return Err(e);
            }
        };
        match resp.message {
            Some(pb::server_message::Message::Columns(c)) => {
                self.columns = Some(c.names);
                Ok(None)
            }
            Some(pb::server_message::Message::ColumnBatch(b)) => {
                let rows = decode_batch(&b);
                let _ = self
                    .conn
                    .write(self.id, pb::client_message::Message::Credit(pb::Credit { batches: 1 }));
                Ok(Some(rows))
            }
            Some(pb::server_message::Message::Done(d)) => {
                self.done = true;
                self.total_rows = d.total_rows;
                self.tip = d.tip;
                self.conn.observe_tip(d.tip);
                Ok(None)
            }
            other => {
                self.done = true;
                Err(WsdbError::Protocol(format!("unexpected query response {other:?}")))
            }
        }
    }
}

impl Drop for Rows {
    fn drop(&mut self) {
        if !self.done {
            let _ = self.conn.write(self.id, pb::client_message::Message::Cancel(pb::Cancel {}));
        }
        self.conn.unregister(self.id);
    }
}

async fn read_one(read: &mut WsRead) -> Result<pb::ServerMessage, WsdbError> {
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
}

#[cfg(test)]
mod tests {
    //! A fake `wsdb` server on a loopback listener: enough of the protocol to exercise the
    //! client's framing, the per-statement `Batch` shape, the credit flow and the reconnect.

    use super::*;
    use crate::wire::encode_batch;
    use std::sync::atomic::AtomicBool;
    use tokio::net::TcpListener;

    type Served = Arc<Mutex<Vec<pb::ClientMessage>>>;

    /// Accepts connections forever; `script` answers one request. Every client frame is kept in
    /// the returned log. A `Begin` closes the socket when `drop_on_begin` is set, once.
    async fn serve(
        script: fn(&pb::ClientMessage) -> Vec<pb::ServerMessage>,
        drop_on_begin: bool,
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
                        if drop_on_begin
                            && matches!(msg.message, Some(pb::client_message::Message::Begin(_)))
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

    fn reply(m: pb::server_message::Message) -> pb::ServerMessage {
        pb::ServerMessage { request_id: 0, message: Some(m) }
    }

    fn hello_ok() -> pb::ServerMessage {
        reply(pb::server_message::Message::HelloOk(pb::HelloOk {
            tip: 7,
            ping_interval_ms: 0,
            ping_timeout_ms: 0,
        }))
    }

    fn script(m: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
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
            Some(C::Begin(_)) | Some(C::Rollback(_)) => {
                vec![reply(S::ExecResult(pb::ExecResult::default()))]
            }
            Some(C::Commit(_)) => {
                vec![reply(S::ExecResult(pb::ExecResult { tip: 10, ..Default::default() }))]
            }
            _ => vec![],
        }
    }

    fn minter() -> TokenMinter {
        TokenMinter::new("devkey", "secret").unwrap()
    }

    async fn dial(url: &str, db: &str) -> Result<Conn, WsdbError> {
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
        let r = conn.exec("INSERT", vec!["v".into()]).await.unwrap();
        assert_eq!((r.rows_affected, r.last_insert_id, r.tip), (1, 9, 8));
        let err = conn.exec("boom", vec![]).await.err().unwrap();
        assert!(matches!(err, WsdbError::Server { code, .. } if code == "exec"));

        let set = conn.query_all("SELECT", vec![]).await.unwrap();
        assert_eq!(set.columns, vec!["n", "s"]);
        assert_eq!(
            set.rows,
            vec![vec![SqlValue::Int(1), "a".into()], vec![SqlValue::Int(2), SqlValue::Null]]
        );
        assert_eq!(set.total_rows, 2);
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
    async fn a_batch_answers_per_statement_and_an_empty_returning_keeps_its_columns() {
        let (url, _) = serve(script, false).await;
        let conn = dial(&url, "DB_1").await.unwrap();
        let stmts = (0..4).map(|_| wire::statement("S", vec![])).collect();
        let out = conn.batch(stmts).await.unwrap();
        assert_eq!(out.tip, 9);
        assert_eq!(conn.tip(), 9);
        assert_eq!(out.results[0], StatementResult::Exec { rows_affected: 2, last_insert_id: 0 });
        assert_eq!(out.results[1].rows(), &[vec![SqlValue::Int(3)]]);
        match &out.results[2] {
            StatementResult::Rows(set) => {
                assert_eq!(set.columns, vec!["version"]);
                assert!(set.rows.is_empty(), "the acquire that lost: Columns, no ColumnBatch");
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(out.results[3].rows().len(), 2);
    }

    #[tokio::test]
    async fn a_reconnect_mid_transaction_is_txn_aborted_until_rollback() {
        let (url, log) = serve(script, true).await;
        let conn = dial(&url, "DB_1").await.unwrap();
        // The fake drops the socket on this Begin: the request fails and the client redials.
        let err = conn.begin().await.err().unwrap();
        assert!(matches!(err, WsdbError::Disconnected), "{err}");
        // A second Begin, answered this time, opens a transaction that the next drop would lose;
        // here the drop already happened, so model the lost transaction directly.
        conn.begin().await.unwrap();
        {
            let mut st = conn.0.state.lock().unwrap();
            st.in_txn = false;
            st.txn_lost = true;
        }
        assert!(matches!(conn.exec("INSERT", vec![]).await, Err(WsdbError::TxnAborted)));
        assert!(matches!(conn.commit().await, Err(WsdbError::TxnAborted)));
        conn.rollback().await.unwrap(); // acknowledges; no Rollback frame is sent for a lost txn
        let r = conn.exec("INSERT", vec![]).await.unwrap();
        assert_eq!(r.rows_affected, 1);
        let hellos = log
            .lock()
            .unwrap()
            .iter()
            .filter(|m| matches!(m.message, Some(pb::client_message::Message::Hello(_))))
            .count();
        assert_eq!(hellos, 2, "one Hello per dial: the first and the redial");
        conn.close();
        assert!(matches!(conn.exec("INSERT", vec![]).await, Err(WsdbError::Closed)));
    }
}
