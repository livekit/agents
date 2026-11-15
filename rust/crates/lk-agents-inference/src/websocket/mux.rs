use std::collections::BTreeMap;
use std::sync::{Arc, Weak};
use std::time::Duration;

use async_trait::async_trait;
use lk_agents_abi::errors::{ApiError, ErrorKind, ProviderError, Timeout};
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::utils::websocket::{
    CANCELLED_ANSWER_WAIT, WebSocketHost, WebSocketMuxHost, WebSocketMuxOptions, WsFrame,
};
use lk_agents_abi::{ApiConnectOptions, Clock, Closed, Ts};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::owned::{Background, Owned};
use lk_agents_net::signal;
use lk_agents_net::task::TaskSet;
use lk_agents_net::transport::{TransportError, WsReceiver, WsSender};
use parking_lot::Mutex;
use tokio::sync::mpsc;

use super::executor::{options, ready_by};
use super::liveness::{Beat, Liveness};
use super::{Inbound, Transports, close_by, lost, open_direct, recv_heard};

/// What a shared socket hands each context: its frames, then how the socket ended.
type Routed = Result<Inbound, ProviderError>;

/// An idle socket the provider closed sooner than this after it opened is not opened again, so a
/// provider that refuses idle sockets is not reconnected in a loop.
const WARM_MIN_LIFETIME: Duration = Duration::from_secs(1);

/// Where a shared socket's provider session stands.
#[derive(Debug, Clone)]
enum State {
    /// Waiting for the provider to say it is ready.
    Opening,
    Ready,
    /// The socket ended, and why.
    Ended(ProviderError),
}

/// One provider socket carrying several streams.
struct Socket {
    id: u64,
    sender: Arc<dyn WsSender>,
    routes: Mutex<BTreeMap<String, mpsc::UnboundedSender<Routed>>>,
    /// When the last frame went out, for the keepalive.
    last_sent: Mutex<Ts>,
    state: signal::Cell<State>,
    /// The provider's id for the socket, from its handshake.
    request_id: Option<String>,
    /// Bumped when a stream takes or leaves it, or its cancel changes, so its reader rechecks its
    /// timers.
    used: signal::Shared<u64>,
    /// The contexts a stream that left mid-answer cancelled, and when the provider must have
    /// ended them for the socket to be reused.
    cancelling: Mutex<Option<Cancelling>>,
}

/// A stream left an unrouted socket mid-answer: it is reused once the provider ended each
/// cancelled context, by `until`, else closed.
#[derive(Debug, Clone)]
struct Cancelling {
    contexts: std::collections::BTreeSet<String>,
    until: Ts,
}

impl Socket {
    async fn send(&self, frame: WsFrame, clock: &dyn Clock) -> Result<(), TransportError> {
        match frame {
            WsFrame::Text(text) => self.sender.send_text(text).await?,
            WsFrame::Binary(data) => self.sender.send_binary(data).await?,
        }
        *self.last_sent.lock() = clock.now();
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct Usage {
    contexts: u32,
    /// After this the socket takes no new context and closes once its last one ends.
    retire_at: Option<Ts>,
    generation: u64,
    /// Since when no stream holds it.
    idle_since: Option<Ts>,
    /// A stream left it in a state the next one cannot start from: it closes once idle.
    spent: bool,
}

impl Usage {
    /// A current socket, not retired, with room for one more context.
    fn room(&self, generation: u64, now: Ts, max: u32) -> bool {
        self.generation == generation
            && !self.spent
            && !self.retire_at.is_some_and(|at| now >= at)
            && self.contexts < max
    }
}

#[derive(Default)]
struct Sockets {
    open: BTreeMap<u64, (Arc<Socket>, Usage)>,
    /// Bumped by `invalidate`: a socket of an older generation takes no new context.
    generation: u64,
    next_id: u64,
    /// `aclose()` ran: no socket opens again.
    closed: bool,
}

/// `_core.WebSocketMux`: the shared sockets of a multiplexing provider, connected on demand.
pub struct Mux(Arc<Shared>);

/// The pool slot is owned while readiness is awaited, before a context can own it.
struct Acquisition {
    mux: Arc<Shared>,
    socket: Option<Arc<Socket>>,
}

impl Drop for Acquisition {
    fn drop(&mut self) {
        if let Some(socket) = self.socket.take() {
            self.mux.released(&socket, false);
        }
    }
}

impl Mux {
    pub fn new(
        host: Arc<dyn WebSocketMuxHost>,
        options: WebSocketMuxOptions,
        env: ProcessEnv,
        transports: Transports,
    ) -> Self {
        Mux(Arc::new(Shared::new(host, options, env, transports)))
    }

    /// A context on a socket with room for it, connecting one when none has. `stream` ends the
    /// context when it leaves it early.
    pub(super) async fn acquire(
        &self,
        cx: &Cx,
        conn_options: &ApiConnectOptions,
        context_id: String,
        stream: Arc<dyn WebSocketHost>,
    ) -> Result<Context, ProviderError> {
        self.0.acquire(cx, conn_options, context_id, stream).await
    }

    /// Opens a socket in the background unless one has room.
    pub fn prewarm(&self) {
        Shared::prewarm(&self.0);
    }

    /// Options changed: idle sockets close now, busy ones once their last stream ends.
    pub fn invalidate(&self) {
        self.0.invalidate();
    }

    pub async fn aclose(&self) {
        self.0.aclose().await;
    }
}

/// The mux's state, shared with its readers and background connects.
pub(super) struct Shared {
    host: Arc<dyn WebSocketMuxHost>,
    max_contexts: u32,
    max_sockets: u32,
    idle_timeout: Option<Duration>,
    keep_warm: bool,
    routed: bool,
    reuses_sockets: bool,
    reads_cancel_end: bool,
    env: ProcessEnv,
    transports: Transports,
    /// Changed when a socket opens, frees room or closes: a stream at the cap waits on it.
    sockets: signal::Shared<Sockets>,
    /// One connect at a time, so streams opening together share the socket it opens.
    connecting: tokio::sync::Mutex<()>,
    /// The readers and the closes of sockets no stream holds.
    tasks: Arc<TaskSet>,
}

impl Shared {
    fn new(
        host: Arc<dyn WebSocketMuxHost>,
        options: WebSocketMuxOptions,
        env: ProcessEnv,
        transports: Transports,
    ) -> Self {
        // an unrouted socket's frames all go to its one stream
        let max_contexts =
            if options.routed { options.max_contexts.unwrap_or(u32::MAX).max(1) } else { 1 };
        Shared {
            host,
            max_contexts,
            max_sockets: options.max_sockets.unwrap_or(u32::MAX).max(1),
            idle_timeout: options
                .idle_timeout
                .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok()),
            keep_warm: options.keep_warm,
            routed: options.routed,
            reuses_sockets: options.reuses_sockets,
            reads_cancel_end: options.reads_cancel_end,
            env,
            transports,
            sockets: signal::Shared::new(Sockets::default()),
            connecting: tokio::sync::Mutex::new(()),
            tasks: Arc::new(TaskSet::default()),
        }
    }

    fn clock(&self) -> &dyn Clock {
        self.env.clock.as_ref()
    }

    fn admit(&self) -> Result<(), ApiError> {
        if self.sockets.with(|sockets| sockets.closed) {
            return Err(ApiError::new(
                ErrorKind::InvalidRequest,
                "the shared socket pool is closed",
            ));
        }
        Ok(())
    }

    fn background(&self) -> Result<Background, ApiError> {
        let spawner = (self.env.spawner)()?;
        Ok(Background::new(Arc::clone(&self.tasks), spawner, Arc::clone(&self.env.clock)))
    }

    /// A context on a ready socket with room for it, connecting one when none has; at the socket
    /// cap it waits for room within the pool limit. Registered: the caller owns its release from
    /// here on.
    pub(super) async fn acquire(
        self: &Arc<Self>,
        cx: &Cx,
        conn_options: &ApiConnectOptions,
        context_id: String,
        stream: Arc<dyn WebSocketHost>,
    ) -> Result<Context, ProviderError> {
        let background = self.background()?;
        let pool_deadline = conn_options.pool_deadline(self.clock().now());
        // a socket whose cancel is being answered is free in a round trip: a stream waits for it
        // rather than opening another
        let may_connect = |sockets: &Sockets| {
            sockets.open.len() < self.max_sockets as usize
                && !sockets.open.values().any(|(socket, _)| socket.cancelling.lock().is_some())
        };
        let socket = loop {
            self.admit()?;
            {
                let _connecting =
                    lk_agents_abi::before(self.clock(), pool_deadline, self.connecting.lock())
                        .await
                        .map_err(|_| ApiError::timed_out(Timeout::Pool))?;
                self.admit()?;
                if let Some(socket) = self.take_room(&background) {
                    break socket;
                }
                if self.sockets.with(|sockets| may_connect(sockets)) {
                    break self.connect(cx, conn_options, &background, 1).await?;
                }
            }
            // at the cap, or a cancel being answered: a socket frees room, or one closes
            let room = self.sockets.wait_until(|sockets| {
                let now = self.clock().now();
                let free = may_connect(sockets)
                    || sockets
                        .open
                        .values()
                        .any(|(_, usage)| usage.room(sockets.generation, now, self.max_contexts));
                (free || sockets.closed).then_some(())
            });
            if lk_agents_abi::before(self.clock(), pool_deadline, room).await.is_err() {
                return Err(ApiError::timed_out(Timeout::Pool).into());
            }
        };
        let mut acquisition = Acquisition { mux: Arc::clone(self), socket: Some(socket) };
        let socket = acquisition.socket.as_ref().expect("held until readiness settles");
        let ready_by = Some(ready_by(self.clock(), conn_options));
        let ready = socket.state.wait_map(|state| match state {
            State::Opening => None,
            State::Ready => Some(Ok(())),
            State::Ended(error) => Some(Err(error.clone())),
        });
        let ready = match lk_agents_abi::before(self.clock(), ready_by, ready).await {
            Ok(Ok(Ok(()))) => Ok(()),
            Ok(Ok(Err(error))) => Err(error),
            Ok(Err(signal::Ended)) => {
                Err(ApiError::connection(Some("the shared socket closed".into())).into())
            }
            Err(_) => Err(ApiError::timed_out(Timeout::Connect).into()),
        };
        ready?;
        if self.keep_warm {
            // the standby: a socket with room for the next stream
            Shared::prewarm(self);
        }
        let (tx, rx) = mpsc::unbounded_channel();
        {
            let mut routes = socket.routes.lock();
            // The reader publishes its end before draining routes. Register under the same
            // route lock, so a close between readiness and registration cannot lose its end.
            if let Some(error) = socket.state.with(|state| match state {
                State::Ended(error) => Some(error.clone()),
                State::Opening | State::Ready => None,
            }) {
                return Err(error);
            }
            routes.insert(context_id.clone(), tx);
        }
        let socket = acquisition.socket.take().expect("the context takes the acquired slot");
        Ok(Context { mux: Arc::clone(self), socket, context_id, rx, stream })
    }

    /// A current socket with room for one more context, now holding it. Idle sockets past
    /// their age close.
    fn take_room(&self, background: &Background) -> Option<Arc<Socket>> {
        let now = self.clock().now();
        let retired = |usage: &Usage| usage.retire_at.is_some_and(|at| now >= at);
        let (socket, aged) = self.sockets.modify(|sockets| {
            let aged: Vec<u64> = sockets
                .open
                .iter()
                .filter(|(_, (_, usage))| usage.contexts == 0 && retired(usage))
                .map(|(&id, _)| id)
                .collect();
            let aged: Vec<Arc<Socket>> = aged
                .into_iter()
                .filter_map(|id| sockets.open.remove(&id).map(|(socket, _)| socket))
                .collect();
            let generation = sockets.generation;
            let max = self.max_contexts;
            let socket =
                sockets.open.values_mut().find(|(_, usage)| usage.room(generation, now, max)).map(
                    |(socket, usage)| {
                        usage.contexts += 1;
                        usage.idle_since = None;
                        Arc::clone(socket)
                    },
                );
            (socket, aged)
        });
        for socket in aged {
            self.close_later(background, socket);
        }
        if let Some(socket) = &socket {
            socket.used.modify(|n| *n += 1);
        }
        socket
    }

    /// Connects a socket holding `contexts` streams from the start, its setup sent first.
    async fn connect(
        self: &Arc<Self>,
        cx: &Cx,
        conn_options: &ApiConnectOptions,
        background: &Background,
        contexts: u32,
    ) -> Result<Arc<Socket>, ProviderError> {
        loop {
            self.admit()?;
            let generation = self.sockets.with(|s| s.generation);
            let connection = self.host.connect(cx, *conn_options).await?;
            let options = options(&connection, Duration::ZERO, None)?;
            let ready_by = ready_by(self.clock(), conn_options);
            let (sender, receiver) =
                open_direct(&self.transports, &self.env, conn_options, &connection).await?;
            let clock = Arc::clone(&self.env.clock);
            let sender = Owned::new(
                background,
                sender,
                Box::new(move |sender, deadline| {
                    Box::pin(
                        async move { close_by(clock.as_ref(), deadline, sender.as_ref()).await },
                    )
                }),
            );
            let keepalive_only_while_unused =
                connection.keepalive.as_ref().is_some_and(|keepalive| keepalive.only_while_unused);
            let request_id = connection
                .target
                .request_id_header
                .as_deref()
                .and_then(|name| receiver.response_header(name));
            for frame in connection.setup {
                super::send(sender.as_ref(), frame).await?;
            }
            let initial = if connection.awaits_ready { State::Opening } else { State::Ready };
            let (state, ready) = signal::signal(initial);
            let now = self.clock().now();
            let opened = self.sockets.modify(|sockets| {
                (sockets.generation == generation && !sockets.closed).then(|| {
                    sockets.next_id += 1;
                    let socket = Arc::new(Socket {
                        id: sockets.next_id,
                        sender: Arc::clone(&sender),
                        routes: Mutex::new(BTreeMap::new()),
                        last_sent: Mutex::new(now),
                        state: ready,
                        request_id,
                        used: signal::Shared::new(0),
                        cancelling: Mutex::new(None),
                    });
                    let retire_at =
                        options.max_session_duration.map(|limit| now.saturating_add(limit));
                    let idle_since = (contexts == 0).then_some(now);
                    let usage = Usage { contexts, retire_at, generation, idle_since, spent: false };
                    sockets.open.insert(socket.id, (Arc::clone(&socket), usage));
                    socket
                })
            });
            let Some(socket) = opened else {
                // the options changed while it connected: it has the old ones
                sender.close().await;
                continue;
            };
            // The registered socket and its reader now own the close.
            drop(sender.into_inner());
            let reader = Reader {
                mux: Arc::downgrade(self),
                host: Arc::clone(&self.host),
                socket: Arc::clone(&socket),
                state,
                ready_by: connection.awaits_ready.then_some(ready_by),
                keepalive: options.keepalive,
                keepalive_only_while_unused,
                ping: options.ping,
                clock: Arc::clone(&self.env.clock),
                opened_at: now,
            };
            background.spawn("socket mux reader", reader.run(receiver));
            return Ok(socket);
        }
    }

    /// Opens a socket in the background unless a current one has room, so the first stream finds
    /// it open. A failure is left to that stream's own connect.
    fn prewarm(self: &Arc<Self>) {
        let Ok(background) = self.background() else { return };
        let mux = Arc::clone(self);
        let opener = background.clone();
        background.spawn("socket mux prewarm", async move {
            let _connecting = mux.connecting.lock().await;
            if mux.has_room() || mux.sockets.with(|s| s.open.len() >= mux.max_sockets as usize) {
                return;
            }
            let options = ApiConnectOptions::default();
            if let Err(error) = mux.connect(&Cx::NONE, &options, &opener, 0).await {
                log::debug!(target: lk_agents_abi::LOG_TARGET, "prewarming a shared socket failed: {error}");
            }
        });
    }

    /// Whether a current socket has room for one more context.
    fn has_room(&self) -> bool {
        let now = self.clock().now();
        self.sockets.with(|sockets| {
            let generation = sockets.generation;
            sockets.open.values().any(|(_, usage)| usage.room(generation, now, self.max_contexts))
        })
    }

    /// A context left `socket`; `reusable`: the next stream can start where it left the socket
    /// (never, for a pool that does not reuse its sockets). Once idle, a stale or spent socket
    /// closes, and a warm pool opens another.
    fn released(self: &Arc<Self>, socket: &Arc<Socket>, reusable: bool) {
        let reusable = reusable && self.reuses_sockets;
        let now = self.clock().now();
        let closing = self.sockets.modify(|sockets| {
            let generation = sockets.generation;
            let (_, usage) = sockets.open.get_mut(&socket.id)?;
            usage.contexts = usage.contexts.saturating_sub(1);
            usage.spent |= !reusable;
            let stale = usage.spent
                || usage.generation != generation
                || usage.retire_at.is_some_and(|at| now >= at);
            if usage.contexts > 0 {
                return None;
            }
            usage.idle_since = Some(now);
            if stale { sockets.open.remove(&socket.id).map(|(socket, _)| socket) } else { None }
        });
        socket.used.modify(|n| *n += 1);
        if let Some(socket) = closing
            && let Ok(background) = self.background()
        {
            self.close_later(&background, socket);
            if self.keep_warm {
                Shared::prewarm(self);
            }
        }
    }

    /// Whether a stream holds `socket`.
    fn in_use(&self, socket: u64) -> bool {
        self.sockets
            .with(|sockets| sockets.open.get(&socket).is_some_and(|(_, usage)| usage.contexts > 0))
    }

    /// The provider ended `context` on `socket`: once it ended every context a stream cancelled
    /// there, the socket is free for the next stream.
    fn context_ended(self: &Arc<Self>, socket: &Arc<Socket>, context: &str) {
        let settled = {
            let mut cancelling = socket.cancelling.lock();
            let Some(pending) = cancelling.as_mut() else { return };
            pending.contexts.remove(context);
            pending.contexts.is_empty() && cancelling.take().is_some()
        };
        if settled {
            self.released(socket, true);
        }
    }

    /// The provider did not end the cancelled contexts in time: the socket is not reused.
    fn cancel_unanswered(self: &Arc<Self>, socket: &Arc<Socket>) {
        if socket.cancelling.lock().take().is_some() {
            self.released(socket, false);
        }
    }

    /// When an idle socket closes: `idle_timeout` after its last stream left.
    fn idle_close_at(&self, socket: u64) -> Option<Ts> {
        let timeout = self.idle_timeout?;
        self.sockets.with(|sockets| {
            let (_, usage) = sockets.open.get(&socket)?;
            usage.idle_since.filter(|_| usage.contexts == 0).map(|at| at.saturating_add(timeout))
        })
    }

    /// Takes `socket` out of the pool if it stayed idle past its idle limit.
    fn close_if_idle(&self, socket: u64, now: Ts) -> bool {
        self.sockets.modify(|sockets| {
            let idle = sockets.open.get(&socket).is_some_and(|(_, usage)| {
                usage.contexts == 0
                    && usage
                        .idle_since
                        .zip(self.idle_timeout)
                        .is_some_and(|(since, limit)| now >= since.saturating_add(limit))
            });
            idle && sockets.open.remove(&socket).is_some()
        })
    }

    /// Options changed: idle sockets close now, busy ones once their last stream ends.
    pub fn invalidate(&self) {
        let Ok(background) = self.background() else { return };
        let idle = self.sockets.modify(|sockets| {
            sockets.generation += 1;
            let idle: Vec<u64> = sockets
                .open
                .iter()
                .filter(|(_, (_, usage))| usage.contexts == 0)
                .map(|(&id, _)| id)
                .collect();
            idle.into_iter()
                .filter_map(|id| sockets.open.remove(&id).map(|(socket, _)| socket))
                .collect::<Vec<_>>()
        });
        for socket in idle {
            self.close_later(&background, socket);
        }
    }

    pub async fn aclose(&self) {
        let open = self.sockets.modify(|sockets| {
            sockets.generation += 1;
            sockets.closed = true;
            std::mem::take(&mut sockets.open)
        });
        self.tasks.close();
        let deadline = self.clock().now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        for (socket, _) in open.into_values() {
            close_with_goodbye(self.host.as_ref(), self.clock(), deadline, &socket).await;
        }
        self.tasks.cancel_and_wait(self.clock()).await;
    }

    /// Closes `socket` on the mux's tasks, its goodbye first; its reader then ends.
    fn close_later(&self, background: &Background, socket: Arc<Socket>) {
        let clock = Arc::clone(&self.env.clock);
        let host = Arc::clone(&self.host);
        let deadline = background.deadline();
        background.cleanup("socket mux close", async move {
            close_with_goodbye(host.as_ref(), clock.as_ref(), deadline, &socket).await
        });
    }
}

/// Closes a socket the multiplexer is done with by `deadline`, its goodbye frames first while
/// the socket is still open: the provider session ends on its own terms. A goodbye that fails
/// or runs late leaves the close to end the session.
async fn close_with_goodbye(
    host: &dyn WebSocketMuxHost,
    clock: &dyn Clock,
    deadline: Ts,
    socket: &Socket,
) -> Closed {
    if !socket.sender.is_closed() {
        let goodbye = async {
            for frame in host.goodbye(&Cx::NONE).await? {
                socket.send(frame, clock).await.map_err(lost)?;
            }
            Ok::<(), ProviderError>(())
        };
        match lk_agents_abi::before(clock, Some(deadline), goodbye).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                log::debug!(target: lk_agents_abi::LOG_TARGET, "a shared socket's goodbye failed: {error}");
            }
            Err(_) => {
                log::debug!(target: lk_agents_abi::LOG_TARGET, "a shared socket's goodbye ran out of time");
            }
        }
    }
    close_by(clock, deadline, socket.sender.as_ref()).await
}

/// A shared socket's reader: each frame to its context (routed by the multiplexer, or straight
/// to an unrouted socket's one stream), the provider's readiness, the keepalive, the pings and
/// the idle close, and the socket's end to every context on it.
struct Reader {
    mux: Weak<Shared>,
    host: Arc<dyn WebSocketMuxHost>,
    socket: Arc<Socket>,
    state: signal::Producer<State>,
    /// When the provider must have said it is ready.
    ready_by: Option<Ts>,
    keepalive: Option<(Duration, WsFrame)>,
    /// The keepalive goes only while no stream holds the socket.
    keepalive_only_while_unused: bool,
    ping: Option<Duration>,
    clock: Arc<dyn Clock>,
    opened_at: Ts,
}

impl Reader {
    async fn run(self, mut receiver: Box<dyn WsReceiver>) {
        let Reader {
            mux,
            host,
            socket,
            state,
            ready_by,
            keepalive,
            keepalive_only_while_unused,
            ping,
            clock,
            opened_at,
        } = self;
        let clock = clock.as_ref();
        let routed = mux.upgrade().is_none_or(|mux| mux.routed);
        let liveness = ping.map(|interval| Mutex::new(Liveness::new(interval, clock.now())));
        let heard = || {
            if let Some(liveness) = &liveness {
                liveness.lock().heard();
            }
        };
        // the pool closed it (idle): its goodbye goes before the close
        let mut closed_by_pool = false;
        let end: Routed = loop {
            let ready = state.with(|state| matches!(state, State::Ready));
            let used = socket.used.with(|n| *n);
            let in_use = keepalive_only_while_unused
                && mux.upgrade().is_some_and(|mux| mux.in_use(socket.id));
            let keepalive_at = keepalive
                .as_ref()
                .filter(|_| !in_use)
                .map(|(interval, _)| socket.last_sent.lock().saturating_add(*interval));
            let ping_at = liveness.as_ref().map(|liveness| liveness.lock().wake_at());
            let idle_at = mux.upgrade().and_then(|mux| mux.idle_close_at(socket.id));
            let cancel_until = socket.cancelling.lock().as_ref().map(|pending| pending.until);
            tokio::select! {
                biased;
                inbound = recv_heard(receiver.as_mut(), &heard) => match inbound {
                    Ok(Inbound::Frame { frame, .. }) => {
                        let only = (!routed && ready)
                            .then(|| socket.routes.lock().values().next().cloned())
                            .flatten();
                        if let Some(route) = only {
                            let _ = route.send(Ok(Inbound::Frame { frame, message: None }));
                            continue;
                        }
                        match host.route(&Cx::NONE, frame.clone()).await {
                            Ok(route) => {
                                if route.ready && !ready {
                                    state.set(State::Ready);
                                }
                                if let Some(ended) = &route.ended_context
                                    && let Some(mux) = mux.upgrade()
                                {
                                    mux.context_ended(&socket, ended);
                                }
                                let target = route.context.and_then(|context| {
                                    socket.routes.lock().get(&context).cloned()
                                });
                                // a frame no stream reads drops its decoded message with it
                                if let Some(target) = target {
                                    let message = route.message;
                                    let _ = target.send(Ok(Inbound::Frame { frame, message }));
                                }
                            }
                            Err(error) => break Err(error),
                        }
                    }
                    Ok(closed) => break Ok(closed),
                    Err(error) => break Err(error.into()),
                },
                () = crate::wait_for_deadline(clock, ready_by.filter(|_| !ready)) => {
                    break Err(ApiError::timed_out(Timeout::Connect).into());
                }
                () = crate::wait_for_deadline(clock, keepalive_at) => {
                    if let Some((_, frame)) = &keepalive
                        && let Err(error) = socket.send(frame.clone(), clock).await
                    {
                        break Err(lost(error).into());
                    }
                }
                () = crate::wait_for_deadline(clock, ping_at) => {
                    let beat = liveness.as_ref().map(|liveness| liveness.lock().tick(clock.now()));
                    match beat {
                        // a write is never timed: the pong it owes is
                        Some(Beat::Ping) => {
                            if let Err(error) = socket.sender.ping().await {
                                break Err(lost(error).into());
                            }
                        }
                        Some(Beat::Lost) => {
                            let error = "the provider stopped answering pings";
                            break Err(ApiError::connection(Some(error.into())).into());
                        }
                        Some(Beat::Wait) | None => {}
                    }
                }
                () = crate::wait_for_deadline(clock, idle_at) => {
                    let mux = mux.upgrade();
                    if mux.is_some_and(|mux| mux.close_if_idle(socket.id, clock.now())) {
                        closed_by_pool = true;
                        break Ok(Inbound::Closed { code: None, reason: None });
                    }
                }
                () = crate::wait_for_deadline(clock, cancel_until) => {
                    if let Some(mux) = mux.upgrade() {
                        mux.cancel_unanswered(&socket);
                    }
                }
                () = socket.used.wait_until(|n| (*n != used).then_some(())) => {}
            }
        };
        if let Some(mux) = mux.upgrade() {
            let left = mux.sockets.modify(|sockets| sockets.open.remove(&socket.id));
            // an idle socket the provider closed: a warm one takes its place
            let lived = clock.now().saturating_duration_since(opened_at);
            let idle_close = left.is_some_and(|(_, usage)| usage.contexts == 0)
                && matches!(end, Ok(Inbound::Closed { .. }));
            if mux.keep_warm && idle_close && lived >= WARM_MIN_LIFETIME {
                Shared::prewarm(&mux);
            }
        }
        state.set(State::Ended(match &end {
            Err(error) => error.clone(),
            Ok(_) => ApiError::connection(Some("the shared socket closed".into())).into(),
        }));
        for route in std::mem::take(&mut *socket.routes.lock()).into_values() {
            let _ = route.send(end.clone());
        }
        let deadline = clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        if closed_by_pool {
            close_with_goodbye(host.as_ref(), clock, deadline, &socket).await;
        } else {
            close_by(clock, deadline, socket.sender.as_ref()).await;
        }
    }
}

/// A stream's context on a shared socket.
pub(super) struct Context {
    mux: Arc<Shared>,
    socket: Arc<Socket>,
    context_id: String,
    rx: mpsc::UnboundedReceiver<Routed>,
    /// The stream on it, which ends its contexts when it leaves early.
    stream: Arc<dyn WebSocketHost>,
}

impl Context {
    /// Sends the multiplexer's frames that open the context.
    pub(super) async fn open(&self, cx: &Cx) -> Result<(), ProviderError> {
        for frame in self.mux.host.open_context(cx, self.context_id.clone()).await? {
            self.socket.send(frame, self.mux.clock()).await.map_err(lost)?;
        }
        Ok(())
    }

    /// The provider's id for the socket, from its handshake.
    pub(super) fn request_id(&self) -> Option<String> {
        self.socket.request_id.clone()
    }

    /// The context's view of the socket: its sends keep the keepalive quiet, and its close
    /// leaves the socket open for the other streams.
    pub(super) fn sender(&self) -> Arc<dyn WsSender> {
        Arc::new(ContextSender(Arc::clone(&self.socket), Arc::clone(&self.mux.env.clock)))
    }

    pub(super) async fn recv(&mut self) -> Result<Inbound, ProviderError> {
        // the reader hands its end to every route before dropping it
        self.rx.recv().await.unwrap_or(Ok(Inbound::Closed { code: None, reason: None }))
    }

    /// Leaves the socket by `deadline`. A context the provider did not finish is ended with the
    /// stream's frames first: a routed socket is free at once; an unrouted one, whose late frames
    /// would reach the next stream, is reused only once the provider ended every context
    /// cancelled (with `reads_cancel_end`), else closed.
    pub(super) async fn release(self, finished: bool, deadline: Ts) -> Closed {
        self.socket.routes.lock().remove(&self.context_id);
        let mux = &self.mux;
        let cancels = mux.routed || mux.reads_cancel_end;
        if finished || !cancels || self.socket.sender.is_closed() {
            mux.released(&self.socket, mux.routed || finished);
            return Closed::Clean;
        }
        let clock = mux.clock();
        let socket = &self.socket;
        let failed = |error: ProviderError| {
            log::debug!(target: lk_agents_abi::LOG_TARGET,
                "closing a multiplexed context failed: {error}");
            Closed::Failed
        };
        let cancel = match lk_agents_abi::before(
            clock,
            Some(deadline),
            self.stream.close_context(&Cx::NONE),
        )
        .await
        {
            Ok(Ok(cancel)) => cancel,
            Ok(Err(error)) => {
                let closed = failed(error);
                mux.released(socket, mux.routed);
                return closed;
            }
            Err(_) => {
                mux.released(socket, mux.routed);
                return Closed::TimedOut;
            }
        };
        let awaits_end = !mux.routed && !cancel.frames.is_empty();
        if awaits_end {
            // armed before the frames go, so an end answered at once is read
            let until = clock.now().saturating_add(CANCELLED_ANSWER_WAIT);
            let contexts = cancel.contexts.into_iter().collect();
            *socket.cancelling.lock() = Some(Cancelling { contexts, until });
            socket.used.modify(|n| *n += 1);
        }
        let send = async {
            for frame in cancel.frames {
                socket.send(frame, clock).await.map_err(lost)?;
            }
            Ok::<(), ProviderError>(())
        };
        let closed = match lk_agents_abi::before(clock, Some(deadline), send).await {
            Ok(Ok(())) => Closed::Clean,
            Ok(Err(error)) => failed(error),
            Err(_) => Closed::TimedOut,
        };
        if awaits_end {
            // held until the provider ends the cancelled contexts or its wait runs out; a cancel
            // that did not go out is never answered
            if closed != Closed::Clean {
                mux.cancel_unanswered(socket);
            }
            return closed;
        }
        mux.released(socket, mux.routed);
        closed
    }
}

struct ContextSender(Arc<Socket>, Arc<dyn Clock>);

#[async_trait]
impl WsSender for ContextSender {
    async fn send_text(&self, text: String) -> Result<(), TransportError> {
        self.0.send(WsFrame::Text(text), self.1.as_ref()).await
    }

    async fn send_binary(&self, data: Vec<u8>) -> Result<(), TransportError> {
        self.0.send(WsFrame::Binary(data), self.1.as_ref()).await
    }

    fn is_closed(&self) -> bool {
        self.0.sender.is_closed()
    }

    /// The socket is shared: a context leaves it through `Context::release`.
    async fn close(&self) {}
}
