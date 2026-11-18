use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use futures_util::{FutureExt as _, SinkExt as _, StreamExt as _};
pub use livekit_protocol::agent_db::wire;
use lk_agents_abi::{Clock, Headers, Spawner, Ts};
use lk_agents_net::signal::{self, Producer};
use lk_agents_net::task::TaskSet;
use lk_agents_net::transport::TransportError;
use lk_agents_net::websocket::{Message, RawWebSocket, connect_with_limit};
use parking_lot::Mutex;
use prost::Message as _;
use tokio::sync::{mpsc, oneshot};
use wire as pb;

use super::TokenMinter;
use super::rows::{Results, statement};
use crate::deadline::REQUEST_WAIT;
use crate::sql::{BatchResult, Row, Statement, StatementResult};
use lk_agents_abi::storage::agentdb_error::ErrorCode;

// Wire protocol: `Hello` first, with a token minted for this dial. `Exec` answers one `ExecResult`;
// `Query` streams `Columns`, `ColumnBatch`* and `Done`, paced by `Credit` past the server's initial
// window; `Batch` applies its statements atomically, one result per statement, then `Done`. A
// request waits for a socket until its caller's deadline, and for its answer `REQUEST_WAIT` from
// the moment its frame was sent whole, as net/http's `ResponseHeaderTimeout`: one queued behind
// big frames does not spend its answer's wait being written. A dropped socket is redialed only
// while a request waits for it, so the redial lives exactly as long as a caller's deadline; those
// in flight fail `Disconnected` (outcome unknown). The server ends a socket with a notice on
// request id 0, as HTTP/2's GOAWAY does, having answered every request it read before it, so each
// one still unanswered applied nothing:
// - `wrong_shard_server`: the database moved or is draining; redialed at once.
// - `idle_evicted`: redialed when next needed.
// - `not_found`: the database was deleted; final. Each request unanswered there fails with it, as
//   does every later call, and the connection closes without a redial.
// After the first two, like gRPC's transparent retry, each unanswered request is sent again on the
// next socket within its deadline, and its caller sees only the answer. A request answered
// `wrong_shard_server` applied nothing either: it is sent again the same way, while its socket
// retires: it takes no request more, a new socket is dialed at once for the next ones, and the
// retiring one ends once the requests still in flight on it are answered. Any other frame on
// request id 0 is a protocol error that ends the socket. A frame the socket fails to send is not
// sent. Socket-less, the connection is dormant, which no pool takes back. The read loop never
// waits on a caller: each request's answers queue in a channel sized for every answer its window
// allows, and a request whose answers overrun it is cancelled. Outgoing frames queue for the
// socket's writer, which sends them in order; a request's caller waits until its own frame is
// sent, a credit or cancel is queued and forgotten.

type WsRead = futures_util::stream::SplitStream<RawWebSocket>;
type WsSink = futures_util::stream::SplitSink<RawWebSocket, Message>;

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
    /// The server sent the request more answers than its window allows, faster than its caller
    /// read them: the request was cancelled, and a write's outcome is unknown.
    #[error("agentdb client: the request's answers overran its window; cancelled")]
    Overrun,
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
    code == ErrorCode::IDLE_EVICTED
        || code == ErrorCode::WRONG_SHARD_SERVER
        || code == ErrorCode::NOT_FOUND
}

/// A frame for the socket writer, and who learns that the socket took all of it.
struct Outgoing {
    bytes: Vec<u8>,
    sent: Option<oneshot::Sender<()>>,
}

/// A socket's writer queue. Unbounded: a request's frame is held by its caller until sent, and a
/// credit, cancel or ping is one small frame per answer read or per ping interval.
type Writer = mpsc::UnboundedSender<Outgoing>;

/// The read deadline past a missed pong, when the server names no ping timeout.
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

#[derive(Debug, Clone)]
pub struct Options {
    /// The data-plane endpoint, `ws://host:port/db`.
    pub url: String,
    pub database_id: String,
    /// The batches a streaming request may have unread: at least the server's initial window,
    /// which each batch read renews by one credit.
    pub credits: u32,
    /// How long one attempt's WebSocket dial plus `Hello` may take, never past the deadline of
    /// the request it dials for.
    pub dial_timeout: Duration,
    pub reconnect: Reconnect,
}

impl Options {
    pub fn new(url: impl Into<String>, database_id: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            database_id: database_id.into(),
            credits: 8,
            dial_timeout: Duration::from_secs(20),
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

/// What a request's queue receives: the server's answers, then perhaps one ending of the
/// client's, for which the queue always keeps a slot.
enum Answer {
    Frame(pb::ServerMessage),
    /// Its socket drained before answering it: sent again on the next socket.
    Drained,
    /// Its answers overran their window: cancelled.
    Overrun,
}

/// A request awaiting its answers, bound to the socket it goes out on.
struct Pending {
    answers: mpsc::Sender<Answer>,
    /// The generation of its socket; `None` once that socket drained (or none was live when it
    /// registered): nothing it sends there is read, so it goes out again on the next one.
    socket: Option<u64>,
}

struct State {
    pending: HashMap<u32, Pending>,
    /// Each open socket's writer, by generation: the live one and those retiring.
    writers: HashMap<u64, Writer>,
    /// The socket taking new requests; `None` between a drop, a drain or a refusal and the
    /// redial.
    live: Option<u64>,
    /// The last socket's generation, one per `Hello` admitted.
    generation: u64,
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
    /// The deadlines of the requests waiting for a socket: the supervisor redials while any
    /// does, each attempt bounded by the latest.
    waiting: signal::Shared<Vec<Ts>>,
    /// The retiring sockets left with nothing in flight: their reads end.
    settled: signal::Shared<BTreeSet<u64>>,
    /// The executor the connection's supervisor, heartbeat, readers and writers run on.
    spawner: Arc<dyn Spawner>,
    /// What its dial bound, heartbeat, redial backoff and read deadline wait on.
    clock: Arc<dyn Clock>,
    /// The supervisor, heartbeat, retiring readers and socket writers; closing the connection
    /// cancels them.
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
            st.writers.clear();
            st.live = None;
        }
        self.tasks.close();
    }
}

/// A socket's read half, with what ends it besides the server.
struct Socket {
    generation: u64,
    read: WsRead,
    /// Resolves when the writer stopped: a send failed.
    writer_gone: oneshot::Receiver<()>,
    /// The writer stopped: what the socket already received is still read, then it ends.
    writer_stopped: bool,
    /// No frame for this long (pongs included) is a dead socket.
    read_deadline: Option<Duration>,
    ping_interval: Duration,
    /// Dropped with the read half: stops the writer, even mid-send to a dead peer.
    _reader: oneshot::Sender<()>,
}

impl Socket {
    /// An error when nothing (not even a pong) arrived within the read deadline on `clock`, the
    /// socket ended or failed, or the writer stopped and nothing received is left to read.
    async fn next_message(&mut self, clock: &dyn Clock) -> Result<Message, WsdbError> {
        let Socket { read, writer_gone, writer_stopped, read_deadline, .. } = self;
        if !*writer_stopped {
            let next = async {
                match *read_deadline {
                    Some(d) => lk_agents_abi::timeout(clock, d, read.next()).await.ok(),
                    None => Some(read.next().await),
                }
            };
            let next = tokio::select! {
                biased;
                next = next => Some(next),
                _ = writer_gone => None,
            };
            match next {
                Some(None) => return Err(WsdbError::Timeout), // not even a pong in time
                Some(Some(next)) => return Self::read(next),
                None => *writer_stopped = true,
            }
        }
        match read.next().now_or_never() {
            Some(next) => Self::read(next),
            None => Err(WsdbError::Io("the socket stopped sending".into())),
        }
    }

    fn read(
        next: Option<Result<Message, lk_agents_net::websocket::tungstenite::Error>>,
    ) -> Result<Message, WsdbError> {
        match next {
            Some(Ok(message)) => Ok(message),
            None => Err(WsdbError::Io("socket closed".into())),
            Some(Err(e)) => Err(WsdbError::Io(e.to_string())),
        }
    }
}

/// How a socket's read ended.
enum End {
    /// The live socket refused a request `wrong_shard_server`: it retires.
    Retiring,
    /// A retiring socket has nothing left in flight.
    Settled,
    /// The socket died, or the server ended it.
    Dropped(WsdbError),
}

/// What one frame read from a socket did.
enum Routed {
    Answer,
    /// A request answered `wrong_shard_server`: its socket retires.
    Refused,
}

/// The task owning the live socket and the ready gate. However it ends (a give-up, a close, a
/// panic, its runtime dropped), it closes the connection, failing every request in flight, then
/// the gate ends, failing every request waiting on it.
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
    /// Reads the live socket and, once it dies or retires, redials: at once for a database that
    /// moved, else while a request waits for it. A retiring socket finishes its reads meanwhile.
    async fn run(self, mut socket: Socket) {
        let conn = &self.conn;
        loop {
            let end = conn.read_loop(&mut socket, Some(&self.ready)).await;
            if conn.is_closed() {
                return;
            }
            self.ready.set(false);
            let (disable, generation) = (conn.0.opts.reconnect.disable, socket.generation);
            let cause = match end {
                End::Retiring if disable => return conn.unguarded().finish(socket).await,
                End::Retiring => {
                    let retiring = conn.unguarded().finish(socket);
                    let spawner = &*conn.0.spawner;
                    // Refused once the connection closed: its requests already failed.
                    conn.0.tasks.try_spawn(spawner, "agentdb retiring reader", retiring);
                    None
                }
                // Dropped here, its writer stops with it: a frame it never sent fails unsent
                // rather than wait out the redial. Only a retiring socket settles.
                End::Settled => {
                    drop(socket);
                    Some(WsdbError::Disconnected)
                }
                End::Dropped(cause) => {
                    drop(socket);
                    Some(cause)
                }
            };
            let now = match cause {
                None => true,
                Some(cause) => {
                    conn.ended(generation, Some(&cause));
                    if disable {
                        return;
                    }
                    cause.notice() == Some(ErrorCode::WRONG_SHARD_SERVER)
                }
            };
            match conn.redial(now).await {
                Some(next) => socket = next,
                None => return,
            }
            self.ready.set(true);
        }
    }
}

/// A request waiting for a socket, its deadline counted while it waits: what keeps the redial
/// going, and bounds its attempts.
struct Waiting<'a> {
    waiting: &'a signal::Shared<Vec<Ts>>,
    deadline: Ts,
}

impl<'a> Waiting<'a> {
    fn new(waiting: &'a signal::Shared<Vec<Ts>>, deadline: Ts) -> Self {
        waiting.modify(|w| w.push(deadline));
        Self { waiting, deadline }
    }
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.waiting.modify(|w| {
            if let Some(i) = w.iter().position(|d| *d == self.deadline) {
                w.swap_remove(i);
            }
        });
    }
}

impl Conn {
    /// Dials, says `Hello`, and starts the supervisor on `spawner`; every wait reads `clock`. The
    /// attempt is bounded by the caller's deadline, else the default one.
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
                writers: HashMap::new(),
                live: None,
                generation: 0,
                closed: false,
                fatal: None,
                tip: 0,
            }),
            ready: ready_cell,
            waiting: signal::Shared::new(Vec::new()),
            settled: signal::Shared::new(BTreeSet::new()),
            spawner,
            clock,
            tasks: Arc::default(),
        });
        let guard = Arc::new(CloseOnDrop(Arc::downgrade(&inner)));
        let conn = Conn(inner, Some(guard));
        let socket = conn.connect(Some(conn.deadline())).await?;
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

    /// Dial, `Hello`, publish the writer, within the dial timeout and by `deadline`, the latest
    /// deadline of the requests it dials for.
    async fn connect(&self, deadline: Option<Ts>) -> Result<Socket, WsdbError> {
        let clock = &*self.0.clock;
        let attempt = clock.now().saturating_add(self.0.opts.dial_timeout);
        let deadline = Some(deadline.map_or(attempt, |deadline| deadline.min(attempt)));
        let timed_out = |_| {
            WsdbError::Io(format!(
                "dial {}: no answer within the dial timeout, {:?}, or the request's deadline",
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
        // the writer owns the sink until the read half drops: a peer that stopped reading is
        // caught by the read deadline, never by timing out a send
        let (tx, rx) = mpsc::unbounded_channel::<Outgoing>();
        let (gone, writer_gone) = oneshot::channel();
        let (reader, reader_gone) = oneshot::channel::<()>();
        // Refused once the connection closed: the sink and `gone` then drop unstarted.
        self.0.tasks.try_spawn(&*self.0.spawner, "agentdb writer", async move {
            // Declared first, so `gone` reports the writer stopped after the sink dropped.
            let _gone = gone;
            tokio::select! {
                biased;
                _ = reader_gone => {}
                () = Self::send_frames(sink, rx) => {}
            }
        });
        let generation = {
            let mut st = self.0.state.lock();
            st.tip = st.tip.max(ok.tip); // a reconnect can only observe a later tip
            st.generation += 1;
            let generation = st.generation;
            st.live = Some(generation);
            st.writers.insert(generation, tx);
            generation
        };
        let read_deadline = (!ping_interval.is_zero())
            .then(|| ping_interval + ping_timeout + Duration::from_secs(1));
        Ok(Socket {
            generation,
            read,
            writer_gone,
            writer_stopped: false,
            read_deadline,
            ping_interval,
            _reader: reader,
        })
    }

    /// Sends the queued frames in order. A frame that fails is not sent: its `sent` drops
    /// unanswered, and so does every one queued behind it.
    async fn send_frames(mut sink: WsSink, mut rx: mpsc::UnboundedReceiver<Outgoing>) {
        while let Some(Outgoing { bytes, sent }) = rx.recv().await {
            if sink.send(Message::binary(bytes)).await.is_err() {
                return;
            }
            if let Some(sent) = sent
                && sent.send(()).is_err()
            {
                log::trace!(target: lk_agents_abi::LOG_TARGET, "agentdb frame sent for a request dropped meanwhile");
            }
        }
        // The peer's close frame is a courtesy: the socket drops either way.
        if sink.close().await.is_err() {
            log::debug!(target: lk_agents_abi::LOG_TARGET, "agentdb socket close not acknowledged");
        }
    }

    /// Reads a socket until it ends: the live one (`ready` given) until it dies or retires, a
    /// retiring one until nothing is in flight on it.
    async fn read_loop(&self, socket: &mut Socket, ready: Option<&Producer<bool>>) -> End {
        let (generation, clock) = (socket.generation, &*self.0.clock);
        loop {
            let message = match ready {
                Some(_) => socket.next_message(clock).await,
                None => tokio::select! {
                    biased;
                    message = socket.next_message(clock) => message,
                    () = self.0.settled.wait_until(|s| s.remove(&generation).then_some(())) => {
                        return End::Settled;
                    }
                },
            };
            let routed = match message {
                Ok(Message::Binary(bytes)) => {
                    self.route_response(bytes.as_ref(), generation, ready)
                }
                Ok(Message::Close(_)) => Err(WsdbError::Io("socket closed".into())),
                Ok(_) => Ok(Routed::Answer),
                Err(error) => Err(error),
            };
            match routed {
                Ok(Routed::Refused) if ready.is_some() => return End::Retiring,
                Ok(_) => {}
                Err(error) => return End::Dropped(error),
            }
        }
    }

    /// Reads a retiring socket until nothing is in flight on it, then ends it.
    async fn finish(self, mut socket: Socket) {
        let cause = match self.read_loop(&mut socket, None).await {
            End::Dropped(cause) => Some(cause),
            End::Retiring | End::Settled => None,
        };
        self.ended(socket.generation, cause.as_ref());
    }

    /// Hands a response to the request awaiting it on this socket, never waiting on its caller;
    /// a pong only proves the socket is alive. The server's notice ends the socket, as any
    /// other frame on request id 0 does; a request answered `wrong_shard_server` alone is drained,
    /// and the socket retires. The live socket's gate closes first, so a request registering
    /// meanwhile waits for the next socket.
    fn route_response(
        &self,
        bytes: &[u8],
        generation: u64,
        ready: Option<&Producer<bool>>,
    ) -> Result<Routed, WsdbError> {
        let resp =
            pb::ServerMessage::decode(bytes).map_err(|e| WsdbError::Protocol(e.to_string()))?;
        match (&resp.message, resp.request_id) {
            (Some(pb::server_message::Message::Pong(_)), _) => Ok(Routed::Answer),
            (Some(pb::server_message::Message::Error(e)), 0) if is_notice(ErrorCode(e.code)) => {
                if let Some(ready) = ready {
                    ready.set(false);
                }
                let notice = WsdbError::server(e.clone());
                match ErrorCode(e.code) {
                    ErrorCode::NOT_FOUND => self.gone(generation, &resp, &notice),
                    _ => self.drain(generation, None),
                }
                Err(notice)
            }
            (message, 0) => {
                Err(WsdbError::Protocol(format!("unexpected frame on request id 0: {message:?}")))
            }
            (Some(pb::server_message::Message::Error(e)), id)
                if ErrorCode(e.code) == ErrorCode::WRONG_SHARD_SERVER =>
            {
                if let Some(ready) = ready {
                    ready.set(false);
                }
                self.drain(generation, Some(id));
                Ok(Routed::Refused)
            }
            _ => {
                self.deliver(generation, resp);
                Ok(Routed::Answer)
            }
        }
    }

    /// Queues an answer for its request, keeping the slot for its ending. A request whose answers
    /// overran their window is cancelled: it ends `Overrun`, and the server is told to stop.
    fn deliver(&self, generation: u64, resp: pb::ServerMessage) {
        let id = resp.request_id;
        let mut st = self.0.state.lock();
        let Some(pending) = st.pending.get(&id).filter(|p| p.socket == Some(generation)) else {
            log::trace!(target: lk_agents_abi::LOG_TARGET, "agentdb answer for a request dropped meanwhile");
            return;
        };
        if pending.answers.capacity() > 1 {
            if pending.answers.try_send(Answer::Frame(resp)).is_err() {
                log::trace!(target: lk_agents_abi::LOG_TARGET, "agentdb answer for a request dropped meanwhile");
            }
            return;
        }
        log::warn!(target: lk_agents_abi::LOG_TARGET, databaseId = self.0.opts.database_id.as_str(); "agentdb request {id} answered past its window: cancelled");
        if let Some(pending) = st.pending.remove(&id)
            && pending.answers.try_send(Answer::Overrun).is_err()
        {
            log::trace!(target: lk_agents_abi::LOG_TARGET, "agentdb overrun request dropped meanwhile");
        }
        let writer = st.writers.get(&generation).cloned();
        self.settle(&mut st, generation);
        drop(st);
        Self::queue(writer.as_ref(), id, pb::client_message::Message::Cancel(pb::Cancel {}));
    }

    /// The server's drain notice on socket `generation`, read: it answered every request it read,
    /// so each one still unanswered there applied nothing. Or `only` one request, answered
    /// `wrong_shard_server`, applied nothing. Either way the socket takes no request more, and
    /// each of those is handed the drain, which sends it again on the next socket.
    fn drain(&self, generation: u64, only: Option<u32>) {
        let mut st = self.0.state.lock();
        if st.live == Some(generation) {
            st.live = None;
        }
        st.pending.retain(|id, p| {
            if p.socket != Some(generation) || only.is_some_and(|only| only != *id) {
                return true;
            }
            p.socket = None;
            p.answers.try_send(Answer::Drained).is_ok()
        });
        self.settle(&mut st, generation);
    }

    /// The server's `not_found` notice on socket `generation`: the database was deleted. Each
    /// request still unanswered there applied nothing and fails with the notice; every later call
    /// fails with it too, and the connection closes without a redial.
    fn gone(&self, generation: u64, frame: &pb::ServerMessage, notice: &WsdbError) {
        {
            let mut st = self.0.state.lock();
            st.fatal = Some(notice.clone());
            for pending in st.pending.values().filter(|p| p.socket == Some(generation)) {
                if pending.answers.try_send(Answer::Frame(frame.clone())).is_err() {
                    log::trace!(target: lk_agents_abi::LOG_TARGET, "agentdb request dropped meanwhile");
                }
            }
        }
        log::info!(target: lk_agents_abi::LOG_TARGET, databaseId = self.0.opts.database_id.as_str(); "agentdb database deleted: connection closed");
        self.close();
    }

    /// A retiring socket with nothing in flight on it is settled: its read ends.
    fn settle(&self, st: &mut State, generation: u64) {
        let retiring = st.live != Some(generation) && st.writers.contains_key(&generation);
        if retiring && !st.pending.values().any(|p| p.socket == Some(generation)) {
            self.0.settled.modify(|settled| settled.insert(generation));
        }
    }

    /// Ends socket `generation` once its read returned, so no response can still be delivered:
    /// each request in flight on it fails `Disconnected` after the answers it already queued. One
    /// a drain notice left unanswered stays, to be sent again.
    fn ended(&self, generation: u64, cause: Option<&WsdbError>) {
        let mut st = self.0.state.lock();
        st.pending.retain(|_, p| p.socket != Some(generation));
        st.writers.remove(&generation);
        if st.live == Some(generation) {
            st.live = None;
        }
        drop(st);
        self.0.settled.modify(|settled| settled.remove(&generation));
        if let Some(cause) = cause {
            log::warn!(target: lk_agents_abi::LOG_TARGET, databaseId = self.0.opts.database_id.as_str(); "agentdb connection lost: {cause}");
        }
    }

    /// Dials once a request waits for the socket (`now`: at once), again with exponential
    /// backoff while one still does, each attempt bounded by the latest waiting deadline; `None`
    /// once a refusal is final.
    async fn redial(&self, mut now: bool) -> Option<Socket> {
        let (r, clock) = (&self.0.opts.reconnect, &*self.0.clock);
        let latest = |waiting: &mut Vec<Ts>| waiting.iter().max().copied();
        let mut backoff = r.min_backoff;
        loop {
            let deadline = match std::mem::take(&mut now) {
                true => self.0.waiting.with(latest),
                false => Some(self.0.waiting.wait_until(latest).await),
            };
            match self.connect(deadline).await {
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
            let ping = pb::client_message::Message::Ping(pb::Ping { timestamp_ms: now });
            let writer = {
                let st = self.0.state.lock();
                st.live.and_then(|live| st.writers.get(&live).cloned())
            };
            Self::queue(writer.as_ref(), self.next_id(), ping);
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

    /// Closes and observes settlement of the supervisor, heartbeat, readers and socket writers.
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
    /// with no time left for its answer. Its queue holds `answers`, all its window allows, and
    /// the slot for its ending.
    async fn register(&self, deadline: Ts, answers: usize) -> Result<Request<'_>, WsdbError> {
        if self.is_closed() {
            return Err(self.closed_error());
        }
        if !self.0.ready.get() {
            let _waiting = Waiting::new(&self.0.waiting, deadline);
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
        let (tx, rx) = mpsc::channel(answers + 1);
        let mut st = self.0.state.lock();
        if st.closed {
            drop(st);
            return Err(self.closed_error());
        }
        // the gate read open, then a drain or drop closed it: bound to no socket, sent again
        let socket = st.live;
        st.pending.insert(id, Pending { answers: tx, socket });
        Ok(Request { conn: self, id, rx, deadline, heard: false, answered: false })
    }

    /// Registers a request for `frame` and writes it, again on the next socket while the one it
    /// was bound to drained or retired before taking it.
    async fn send(
        &self,
        frame: &mut pb::ClientMessage,
        deadline: Ts,
        answers: usize,
    ) -> Result<Request<'_>, WsdbError> {
        loop {
            let mut req = self.register(deadline, answers).await?;
            frame.request_id = req.id;
            match self.write(frame).await {
                Ok(()) => {
                    req.deadline = self.0.clock.now().saturating_add(REQUEST_WAIT);
                    return Ok(req);
                }
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

    /// The writer for request `request_id`'s frame: its socket's, while that is the live one.
    /// `Drained` once that socket drained or retired, `Disconnected` once it dropped.
    fn request_writer(&self, request_id: u32) -> Result<Writer, WsdbError> {
        let st = self.0.state.lock();
        if st.closed {
            drop(st);
            return Err(self.closed_error());
        }
        match st.pending.get(&request_id).map(|p| p.socket) {
            Some(None) => Err(WsdbError::Drained),
            Some(Some(socket)) if st.live == Some(socket) => {
                st.writers.get(&socket).cloned().ok_or(WsdbError::Disconnected)
            }
            Some(Some(socket)) if st.writers.contains_key(&socket) => Err(WsdbError::Drained),
            _ => Err(WsdbError::Disconnected),
        }
    }

    /// Sends a request's frame on its socket; an error when the socket did not take all of it,
    /// which no server reads as a request: `Drained` when that socket drained or retired
    /// meanwhile.
    async fn write(&self, frame: &pb::ClientMessage) -> Result<(), WsdbError> {
        let bytes = Self::encode(frame)?;
        let (sent, taken) = oneshot::channel();
        let writer = self.request_writer(frame.request_id)?;
        if writer.send(Outgoing { bytes, sent: Some(sent) }).is_ok() && taken.await.is_ok() {
            return Ok(());
        }
        match self.request_writer(frame.request_id) {
            Err(WsdbError::Drained) => Err(WsdbError::Drained),
            Err(error @ (WsdbError::Closed | WsdbError::Server { .. })) => Err(error),
            _ => Err(WsdbError::Disconnected),
        }
    }

    /// Queues a credit for a request's stream on the socket it went out on, live or retiring,
    /// without waiting for it to be sent. Its socket gone, the request reads what it already
    /// queued, then its end.
    fn credit(&self, request_id: u32, batches: u32) {
        let writer = {
            let st = self.0.state.lock();
            let socket = st.pending.get(&request_id).and_then(|p| p.socket);
            socket.and_then(|socket| st.writers.get(&socket).cloned())
        };
        let credit = pb::client_message::Message::Credit(pb::Credit { batches });
        Self::queue(writer.as_ref(), request_id, credit);
    }

    /// Queues a frame nothing waits on (a credit, a cancel, a ping) behind those already queued
    /// on `writer`; dropped when its socket is gone, which ends the request it was for.
    fn queue(writer: Option<&Writer>, request_id: u32, message: pb::client_message::Message) {
        let frame = pb::ClientMessage { request_id, message: Some(message) };
        let queued = Self::encode(&frame).and_then(|bytes| {
            let writer = writer.ok_or(WsdbError::Disconnected)?;
            writer.send(Outgoing { bytes, sent: None }).map_err(|_| WsdbError::Disconnected)
        });
        if let Err(error) = queued {
            log::trace!(target: lk_agents_abi::LOG_TARGET, "agentdb frame not queued: {error}");
        }
    }

    /// The request's next answer, until its deadline: past it, `Timeout`.
    async fn wait(
        &self,
        rx: &mut mpsc::Receiver<Answer>,
        deadline: Ts,
    ) -> Result<pb::ServerMessage, WsdbError> {
        let next = lk_agents_abi::before(&*self.0.clock, Some(deadline), rx.recv());
        match next.await.map_err(|_| WsdbError::Timeout)? {
            None if self.is_closed() => Err(self.closed_error()),
            None => Err(WsdbError::Disconnected),
            Some(Answer::Drained) => Err(WsdbError::Drained),
            Some(Answer::Overrun) => Err(WsdbError::Overrun),
            Some(Answer::Frame(pb::ServerMessage {
                message: Some(pb::server_message::Message::Error(e)),
                ..
            })) => Err(WsdbError::server(e)),
            Some(Answer::Frame(resp)) => Ok(resp),
        }
    }

    fn observe_tip(&self, tip: i64) {
        let mut st = self.0.state.lock();
        st.tip = st.tip.max(tip);
    }

    /// One write statement; returns once durable, or fails unsent past `deadline`.
    pub async fn exec(&self, s: &Statement, deadline: Ts) -> Result<ExecResult, WriteError> {
        let exec = pb::client_message::Message::Exec(statement(s));
        let mut frame = pb::ClientMessage { request_id: 0, message: Some(exec) };
        loop {
            // its result, or an error
            let mut req = self.send(&mut frame, deadline, 1).await.map_err(WriteError::NotSent)?;
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

    /// The batches a streaming request may have unread, and its answers besides: its `Columns`,
    /// and its `Done` or error.
    fn window(&self) -> usize {
        self.0.opts.credits as usize + 2
    }

    /// A query's rows, all of them, sent by `deadline`: one credit back per batch consumed, so the
    /// server sends no further than the caller reads.
    pub async fn query_all(&self, s: &Statement, deadline: Ts) -> Result<Vec<Row>, WsdbError> {
        let query = pb::client_message::Message::Query(statement(s));
        let mut frame = pb::ClientMessage { request_id: 0, message: Some(query) };
        loop {
            let mut req = self.send(&mut frame, deadline, self.window()).await?;
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
                    self.credit(id, 1);
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

    /// Statements applied atomically, in order, each answered in order, sent by `deadline`.
    pub async fn batch(
        &self,
        statements: &[Statement],
        deadline: Ts,
    ) -> Result<BatchResult, WriteError> {
        let statements: Vec<pb::Statement> = statements.iter().map(statement).collect();
        let n = statements.len();
        let batch = pb::client_message::Message::Batch(pb::Batch { statements });
        let mut frame = pb::ClientMessage { request_id: 0, message: Some(batch) };
        // its window, past one `Columns` or `ExecResult` per statement
        let answers = self.window() + n.saturating_sub(1);
        loop {
            let sent = self.send(&mut frame, deadline, answers).await;
            let mut req = sent.map_err(WriteError::NotSent)?;
            match self.batch_answer(&mut req, n).await {
                Err(WsdbError::Drained) => {}
                answer => return answer.map_err(WriteError::Sent),
            }
        }
    }

    /// The server grants the batch its initial window, as it does a query: a credit is sent back
    /// only per batch consumed.
    async fn batch_answer(
        &self,
        req: &mut Request<'_>,
        n: usize,
    ) -> Result<BatchResult, WsdbError> {
        let id = req.id;
        let mut results = Results::new(n);
        loop {
            match req.next().await?.message {
                Some(pb::server_message::Message::Columns(columns)) => results.columns(columns)?,
                Some(pb::server_message::Message::ColumnBatch(batch)) => {
                    results.batch(batch)?;
                    self.credit(id, 1);
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
/// it leaves the pending table and tells the server to stop, on the socket it went out on.
struct Request<'a> {
    conn: &'a Conn,
    id: u32,
    rx: mpsc::Receiver<Answer>,
    /// The caller's until its frame was sent, then `REQUEST_WAIT` after: its answer is awaited
    /// until then.
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
        let writer = {
            let mut st = self.conn.0.state.lock();
            let socket = st.pending.remove(&self.id).and_then(|p| p.socket);
            let writer = socket.and_then(|socket| {
                self.conn.settle(&mut st, socket);
                st.writers.get(&socket).cloned()
            });
            writer.filter(|_| !self.answered)
        };
        if writer.is_some() {
            let cancel = pb::client_message::Message::Cancel(pb::Cancel {});
            Conn::queue(writer.as_ref(), self.id, cancel);
        }
    }
}

#[cfg(test)]
#[path = "connection_tests.rs"]
pub(crate) mod tests;
