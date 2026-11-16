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

use super::rows::{decode_batch, statement};
use crate::StorageError;
use crate::sql::{BatchResult, Row, Statement, StatementResult};
use lk_agents_abi::storage::agentdb_error::ErrorCode;

// Wire protocol: `Hello` first, with a token minted for this dial. `Exec` answers one `ExecResult`;
// `Query` streams `Columns`, `ColumnBatch`* and `Done`, paced by `Credit`; `Batch` applies its
// statements atomically, one result per statement, then `Done`. Every request carries its caller's
// deadline: it waits for a socket and for its answer until then. A dropped socket is redialed
// only while a request waits for it, so the redial lives exactly as long as a caller's deadline;
// those in flight fail `Disconnected` (outcome unknown). The server ends a socket with a notice on
// request id 0, as HTTP/2's GOAWAY does: `wrong_shard_server` (the database moved or is draining:
// redialed at once) or `idle_evicted` (redialed when next needed). The server answered every
// request it read before its notice, so each one still unanswered applied nothing: like gRPC's
// transparent retry, it is sent again on the next socket within its deadline, and its caller sees
// only the answer. A request answered `wrong_shard_server` applied nothing either: it is sent again
// the same way, while the socket takes no request more and ends, redialed at once, as soon as
// the requests still in flight on it are answered. A frame the socket fails to send is not sent. Socket-less, the connection is
// dormant, which no pool takes back. Answers and outgoing frames queue in bounded channels: a full
// one holds its sender back.

type WsRead = futures_util::stream::SplitStream<RawWebSocket>;

#[derive(Debug, Clone, thiserror::Error)]
pub enum WsdbError {
    /// `close` was called, or reconnection gave up.
    #[error("agentdb client: connection closed")]
    Closed,
    /// The connection dropped with this request in flight; a write's outcome is unknown.
    #[error("agentdb client: connection lost, request outcome unknown")]
    Disconnected,
    /// A server-reported error: its code, which says whether the request may have applied and
    /// may be sent again, and its description.
    #[error("agentdb {code}: {message}")]
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
    /// The request's socket drained before answering it: nothing applied, and the request is sent
    /// again on the next socket, so no caller sees this.
    #[error("agentdb client: the socket drained before answering the request")]
    Drained,
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
    /// A `Hello` refusal that redialing cannot repair: a known code that is not retryable.
    pub(crate) fn is_final(&self) -> bool {
        match self {
            Self::Server { code, .. } => code.known() && !code.retryable(),
            Self::Protocol(_) => true,
            _ => false,
        }
    }

    /// A code this build does not know keeps the name the server gave it in the message.
    fn server(error: pb::Error) -> Self {
        let code = ErrorCode(error.code);
        let message = if code.known() {
            error.description
        } else {
            format!("{}: {}", error.name, error.description)
        };
        Self::Server { code, message }
    }

    /// The server's notice ending the socket, sent on request id 0.
    fn notice(&self) -> Option<ErrorCode> {
        match self {
            Self::Server { code, .. } if is_notice(*code) => Some(*code),
            _ => None,
        }
    }
}

/// The largest frame either way, as the protocol fixes it.
pub const MAX_FRAME: usize = 16 << 20;

/// What this client names itself in `Hello`.
const CLIENT_VERSION: &str = concat!("lk-agents/", env!("CARGO_PKG_VERSION"));

/// The codes the server ends a socket with, on request id 0.
fn is_notice(code: ErrorCode) -> bool {
    code == ErrorCode::IDLE_EVICTED || code == ErrorCode::WRONG_SHARD_SERVER
}

/// A frame for the socket writer, and who learns that the socket took all of it.
struct Outgoing {
    bytes: Vec<u8>,
    sent: Option<tokio::sync::oneshot::Sender<()>>,
}

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

/// A request awaiting its answers, bound to the socket it goes out on.
struct Pending {
    answers: mpsc::Sender<pb::ServerMessage>,
    /// The generation of its socket; `None` once that socket drained (or none was live when it
    /// registered): nothing it sends there is read, so it goes out again on the next one.
    socket: Option<u64>,
}

struct State {
    pending: HashMap<u32, Pending>,
    /// The live socket's writer; `None` between a drop or a drain notice and the redial.
    tx: Option<mpsc::Sender<Outgoing>>,
    /// The live socket's generation, one per `Hello` admitted.
    generation: u64,
    closed: bool,
    /// The live socket refused a request `wrong_shard_server`: it ends once nothing is in flight.
    retiring: bool,
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
    /// The generation of the last retiring socket left with nothing in flight: its read ends.
    settled: signal::Shared<u64>,
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
            let cause = conn.read_loop(&mut socket, &self.ready).await;
            if conn.is_closed() {
                return;
            }
            self.ready.set(false);
            conn.dropped(&cause);
            if conn.0.opts.reconnect.disable {
                return;
            }
            // a database that moved or drains is redialed at once; any other drop when needed,
            // which a request a drain notice left unanswered is, as it registers again
            let now = cause.notice() == Some(ErrorCode::WRONG_SHARD_SERVER);
            match conn.redial(now).await {
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
                generation: 0,
                retiring: false,
                closed: false,
                fatal: None,
                tip: 0,
            }),
            ready: ready_cell,
            waiting: signal::Shared::new(0),
            settled: signal::Shared::new(0),
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
        // the dial's own bound, not a caller's deadline: the next attempt may find the database
        // warm
        let timed_out = |_| {
            WsdbError::Io(format!(
                "dial {}: no answer within the dial timeout, {:?}",
                self.0.opts.url, self.0.opts.dial_timeout
            ))
        };
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
        let ws = lk_agents_abi::before(clock, deadline, dial).await.map_err(timed_out)?.map_err(
            |e| match e {
                // the upgrade's own refusal is final, as the same refusal at `Hello` is
                TransportError::Status { status: 401, message, .. } => {
                    WsdbError::Server { code: ErrorCode::UNAUTHENTICATED, message }
                }
                TransportError::Status { status: 403, message, .. } => {
                    WsdbError::Server { code: ErrorCode::PERMISSION_DENIED, message }
                }
                TransportError::Status { status: 404, message, .. } => {
                    WsdbError::Server { code: ErrorCode::NOT_FOUND, message }
                }
                e => WsdbError::Io(format!("dial {}: {e}", self.0.opts.url)),
            },
        )?;
        let (mut sink, mut read) = ws.split();
        let hello = pb::ClientMessage {
            request_id: self.next_id(),
            message: Some(pb::client_message::Message::Hello(pb::Hello {
                token,
                database_id: self.0.opts.database_id.clone(),
                client_version: CLIENT_VERSION.into(),
            })),
        };
        lk_agents_abi::before(clock, deadline, sink.send(Message::binary(hello.encode_to_vec())))
            .await
            .map_err(timed_out)?
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
        let resp = lk_agents_abi::before(clock, deadline, first).await.map_err(timed_out)??;
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
        let (tx, mut rx) = mpsc::channel::<Outgoing>(WRITE_QUEUE);
        let (gone, writer_gone) = tokio::sync::oneshot::channel();
        let writer_clock = self.0.clock.clone();
        // Refused once the connection closed: the sink and `gone` then drop unstarted.
        self.0.tasks.try_spawn(&*self.0.spawner, "agentdb writer", async move {
            // Declared first, so `gone` reports the writer stopped after the sink dropped.
            let _gone = gone;
            let (mut sink, clock) = (sink, &*writer_clock);
            // a frame that fails is not sent: its `sent` drops unanswered, and so does every one
            // queued behind it
            while let Some(Outgoing { bytes, sent }) = rx.recv().await {
                let send =
                    lk_agents_abi::timeout(clock, ping_timeout, sink.send(Message::binary(bytes)));
                if !matches!(send.await, Ok(Ok(()))) {
                    return;
                }
                if let Some(sent) = sent
                    && sent.send(()).is_err()
                {
                    log::trace!(target: lk_agents_abi::LOG_TARGET, "agentdb frame sent for a request dropped meanwhile");
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
            st.generation += 1;
            st.retiring = false;
        }
        let read_deadline = (!ping_interval.is_zero())
            .then(|| ping_interval + ping_timeout + Duration::from_secs(1));
        Ok(Socket { read, writer_gone, read_deadline, ping_interval })
    }

    async fn read_loop(&self, socket: &mut Socket, ready: &Producer<bool>) -> WsdbError {
        let generation = self.0.state.lock().generation;
        loop {
            let message = tokio::select! {
                biased;
                message = socket.next_message(&*self.0.clock) => message,
                () = self.0.settled.wait_until(|g| (*g == generation).then_some(())) => {
                    let message = "the socket retired once its requests were answered".into();
                    return WsdbError::Server { code: ErrorCode::WRONG_SHARD_SERVER, message };
                }
            };
            let routed = match message {
                Ok(Message::Binary(bytes)) => self.route_response(bytes.as_ref(), ready).await,
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
    /// a pong only proves the socket is alive. The server's drain notice ends the socket; a
    /// request answered `wrong_shard_server` alone is drained, and the socket retires.
    async fn route_response(&self, bytes: &[u8], ready: &Producer<bool>) -> Result<(), WsdbError> {
        let resp =
            pb::ServerMessage::decode(bytes).map_err(|e| WsdbError::Protocol(e.to_string()))?;
        match &resp.message {
            Some(pb::server_message::Message::Pong(_)) => return Ok(()),
            // the gate closes first, so a request registering meanwhile waits for the next socket
            Some(pb::server_message::Message::Error(e))
                if resp.request_id == 0 && is_notice(ErrorCode(e.code)) =>
            {
                ready.set(false);
                self.drain(None);
                return Err(WsdbError::server(e.clone()));
            }
            Some(pb::server_message::Message::Error(e))
                if ErrorCode(e.code) == ErrorCode::WRONG_SHARD_SERVER =>
            {
                ready.set(false);
                self.drain(Some(resp.request_id));
                return Ok(());
            }
            _ => {}
        }
        let ch = self.0.state.lock().pending.get(&resp.request_id).map(|p| p.answers.clone());
        if let Some(ch) = ch
            && ch.send(resp).await.is_err()
        {
            log::trace!(target: lk_agents_abi::LOG_TARGET, "agentdb answer for a request dropped meanwhile");
        }
        Ok(())
    }

    /// The server's drain notice, read: it answered every request it read, so each one still
    /// unanswered on the live socket applied nothing, and the socket takes no request more. Or
    /// `only` one request, answered `wrong_shard_server`, applied nothing, and the socket retires.
    /// Each of those is handed the drain on request id 0, which sends it again on the next
    /// socket. A request without its last answer has none queued, so each has room for it; one
    /// whose queue is full broke that promise and ends `Disconnected` after its answers.
    fn drain(&self, only: Option<u32>) {
        let mut st = self.0.state.lock();
        match only {
            Some(_) => st.retiring = true,
            None => st.tx = None,
        }
        let live = Some(st.generation);
        st.pending.retain(|id, p| {
            if p.socket != live || only.is_some_and(|only| only != *id) {
                return true;
            }
            p.socket = None;
            p.answers.try_send(pb::ServerMessage::default()).is_ok()
        });
        self.settle(&mut st);
    }

    /// A retiring socket with nothing in flight on it takes no request more, and its read ends.
    fn settle(&self, st: &mut State) {
        let live = Some(st.generation);
        if st.retiring && st.tx.is_some() && !st.pending.values().any(|p| p.socket == live) {
            st.tx = None;
            let generation = st.generation;
            self.0.settled.modify(|settled| *settled = generation);
        }
    }

    /// Fails everything in flight on the socket. Only after the read loop returned, so no
    /// response can still be delivered: each request fails `Disconnected` after the answers it
    /// already queued. One a drain notice left unanswered stays, to be sent again.
    fn dropped(&self, cause: &WsdbError) {
        let mut st = self.0.state.lock();
        st.pending.retain(|_, p| p.socket.is_none());
        st.tx = None;
        drop(st);
        log::warn!(target: lk_agents_abi::LOG_TARGET, databaseId = self.0.opts.database_id.as_str(); "agentdb connection lost: {cause}");
    }

    /// Dials once a request waits for the socket (`now`: at once), again with exponential
    /// backoff while one still does; `None` once a refusal is final.
    async fn redial(&self, mut now: bool) -> Option<Socket> {
        let (r, clock) = (&self.0.opts.reconnect, &*self.0.clock);
        let mut backoff = r.min_backoff;
        loop {
            if !std::mem::take(&mut now) {
                self.0.waiting.wait_until(|n| (*n > 0).then_some(())).await;
            }
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
        let (answers, rx) = mpsc::channel(self.0.opts.credits as usize + 2);
        let mut st = self.0.state.lock();
        if st.closed {
            drop(st);
            return Err(self.closed_error());
        }
        // the gate read open, then a drain or drop closed it: bound to no socket, sent again
        let socket = st.tx.is_some().then_some(st.generation);
        st.pending.insert(id, Pending { answers, socket });
        Ok(Request { conn: self, id, rx, deadline, heard: false, answered: false })
    }

    /// Registers a request for `frame` and writes it, again on the next socket while the one it
    /// was bound to drained before taking it.
    async fn send(
        &self,
        frame: &mut pb::ClientMessage,
        deadline: Ts,
    ) -> Result<Request<'_>, WsdbError> {
        loop {
            let req = self.register(deadline).await?;
            frame.request_id = req.id;
            match self.write(frame).await {
                Ok(()) => return Ok(req),
                Err(WsdbError::Drained) => {}
                Err(error) => return Err(error),
            }
        }
    }

    /// The frame's bytes for the socket writer, refused when over the server's limit.
    fn encode(frame: &pb::ClientMessage) -> Result<Vec<u8>, WsdbError> {
        let size = frame.encoded_len();
        if size > MAX_FRAME {
            return Err(WsdbError::TooLarge(size));
        }
        Ok(frame.encode_to_vec())
    }

    /// The live socket's writer.
    fn writer(&self) -> Result<mpsc::Sender<Outgoing>, WsdbError> {
        let st = self.0.state.lock();
        if st.closed {
            drop(st);
            return Err(self.closed_error());
        }
        st.tx.clone().ok_or(WsdbError::Disconnected)
    }

    /// The writer of the socket `request_id` is bound to, while it is the live one: `Drained`
    /// once that socket drained, `Disconnected` once it dropped.
    fn bound_writer(&self, request_id: u32) -> Result<mpsc::Sender<Outgoing>, WsdbError> {
        let st = self.0.state.lock();
        if st.closed {
            drop(st);
            return Err(self.closed_error());
        }
        match st.pending.get(&request_id).map(|p| p.socket) {
            Some(None) => Err(WsdbError::Drained),
            Some(Some(socket)) if socket == st.generation => {
                st.tx.clone().ok_or(WsdbError::Disconnected)
            }
            _ => Err(WsdbError::Disconnected),
        }
    }

    /// Sends a request's frame on its socket, waiting while the writer's queue is full; an error
    /// when the socket did not take all of it, which no server reads as a request: `Drained` when
    /// that socket's drain notice was read meanwhile.
    async fn write(&self, frame: &pb::ClientMessage) -> Result<(), WsdbError> {
        let bytes = Self::encode(frame)?;
        let (sent, taken) = tokio::sync::oneshot::channel();
        let outgoing = Outgoing { bytes, sent: Some(sent) };
        let writer = self.bound_writer(frame.request_id)?;
        if writer.send(outgoing).await.is_ok() && taken.await.is_ok() {
            return Ok(());
        }
        match self.bound_writer(frame.request_id) {
            Err(WsdbError::Drained) => Err(WsdbError::Drained),
            Err(error @ (WsdbError::Closed | WsdbError::Server { .. })) => Err(error),
            _ => Err(WsdbError::Disconnected),
        }
    }

    /// A credit for a request's stream. Failing, the socket is gone: the request then reads what
    /// it already queued, then its end.
    async fn credit(&self, request_id: u32, batches: u32) {
        let credit = pb::client_message::Message::Credit(pb::Credit { batches });
        let frame = pb::ClientMessage { request_id, message: Some(credit) };
        if let Err(error) = self.write(&frame).await {
            log::trace!(target: lk_agents_abi::LOG_TARGET, "agentdb credit not sent: {error}");
        }
    }

    /// Queues a frame nothing waits on (a ping, a cancel), dropped when the writer's queue is
    /// full or the socket is gone: neither changes what the server committed.
    fn try_write(&self, request_id: u32, message: pb::client_message::Message) {
        let frame = pb::ClientMessage { request_id, message: Some(message) };
        let queued = Self::encode(&frame).and_then(|bytes| {
            let outgoing = Outgoing { bytes, sent: None };
            self.writer()?.try_send(outgoing).map_err(|_| WsdbError::Disconnected)
        });
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
            // request id 0 is the connection's own: its socket drained
            Some(pb::ServerMessage { request_id: 0, .. }) => Err(WsdbError::Drained),
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
        let exec = pb::client_message::Message::Exec(statement(s));
        let mut frame = pb::ClientMessage { request_id: 0, message: Some(exec) };
        loop {
            let mut req = self.send(&mut frame, deadline).await.map_err(WriteError::NotSent)?;
            match self.exec_answer(&mut req).await {
                Err(WsdbError::Drained) => {}
                answer => return answer.map_err(WriteError::Sent),
            }
        }
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
        let query = pb::client_message::Message::Query(statement(s));
        let mut frame = pb::ClientMessage { request_id: 0, message: Some(query) };
        loop {
            let mut req = self.send(&mut frame, deadline).await?;
            match self.query_answer(&mut req).await {
                Err(WsdbError::Drained) => {}
                answer => return answer,
            }
        }
    }

    async fn query_answer(&self, req: &mut Request<'_>) -> Result<Vec<Row>, WsdbError> {
        let id = req.id;
        let mut results = Results::new(1);
        loop {
            match req.next().await?.message {
                Some(pb::server_message::Message::Columns(columns)) => results.columns(columns)?,
                Some(pb::server_message::Message::ColumnBatch(batch)) => {
                    results.batch(batch)?;
                    self.credit(id, 1).await;
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
        let batch = pb::client_message::Message::Batch(pb::Batch { statements });
        let mut frame = pb::ClientMessage { request_id: 0, message: Some(batch) };
        loop {
            let mut req = self.send(&mut frame, deadline).await.map_err(WriteError::NotSent)?;
            match self.batch_answer(&mut req, n).await {
                Err(WsdbError::Drained) => {}
                answer => return answer.map_err(WriteError::Sent),
            }
        }
    }

    async fn batch_answer(
        &self,
        req: &mut Request<'_>,
        n: usize,
    ) -> Result<BatchResult, WsdbError> {
        let id = req.id;
        self.credit(id, self.0.opts.credits).await;
        let mut results = Results::new(n);
        loop {
            match req.next().await?.message {
                Some(pb::server_message::Message::Columns(columns)) => results.columns(columns)?,
                Some(pb::server_message::Message::ColumnBatch(batch)) => {
                    results.batch(batch)?;
                    self.credit(id, 1).await;
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
/// it leaves the pending table and tells the server to stop, on the live socket it went out on.
struct Request<'a> {
    conn: &'a Conn,
    id: u32,
    rx: mpsc::Receiver<pb::ServerMessage>,
    /// The caller's: its answer is awaited until then.
    deadline: Ts,
    /// Whether any answer arrived: a drain after one breaks the notice's promise.
    heard: bool,
    answered: bool,
}

impl Request<'_> {
    /// The next answer; `Drained` only before the first, when the request is sent again.
    async fn next(&mut self) -> Result<pb::ServerMessage, WsdbError> {
        match self.conn.wait(&mut self.rx, self.deadline).await {
            Err(WsdbError::Drained) if self.heard => Err(WsdbError::Disconnected),
            answer => {
                self.heard |= answer.is_ok();
                answer
            }
        }
    }
}

impl Drop for Request<'_> {
    fn drop(&mut self) {
        let live = {
            let mut st = self.conn.0.state.lock();
            let socket = st.pending.remove(&self.id).and_then(|p| p.socket);
            let live = st.tx.is_some() && socket == Some(st.generation);
            self.conn.settle(&mut st);
            live
        };
        if live && !self.answered {
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
