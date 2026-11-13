use std::collections::BTreeMap;
use std::sync::{Arc, Weak};
use std::time::Duration;

use async_trait::async_trait;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::utils::socket::{MuxHost, WsFrame};
use lk_agents_abi::{ApiConnectOptions, Clock, Closed, Ts};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::owned::Background;
use lk_agents_net::task::TaskSet;
use lk_agents_net::transport::{TransportError, WsReceiver, WsSender};
use parking_lot::Mutex;
use tokio::sync::mpsc;

use super::liveness::{Beat, Liveness};
use super::{Inbound, Transports, close_by, lost, open_direct, recv_heard};

/// What a shared socket hands each context: its frames, then how the socket ended.
type Routed = Result<Inbound, ProviderError>;

/// One provider socket carrying several streams.
struct Socket {
    id: u64,
    sender: Arc<dyn WsSender>,
    routes: Mutex<BTreeMap<String, mpsc::UnboundedSender<Routed>>>,
    /// When the last frame went out, for the keepalive.
    last_sent: Mutex<Ts>,
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
}

impl Usage {
    /// A current socket, not retired, with room for one more context.
    fn room(&self, generation: u64, now: Ts, max: u32) -> bool {
        self.generation == generation
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
}

/// `_core.SocketMux`: the shared sockets of a multiplexing provider, connected on demand.
pub struct Mux(Arc<Shared>);

impl Mux {
    pub fn new(
        host: Arc<dyn MuxHost>,
        max_contexts: Option<u32>,
        env: ProcessEnv,
        transports: Transports,
    ) -> Self {
        Mux(Arc::new(Shared::new(host, max_contexts, env, transports)))
    }

    /// A context on a socket with room for it, connecting one when none has.
    pub(super) async fn acquire(
        &self,
        cx: &Cx,
        conn_options: &ApiConnectOptions,
        context_id: String,
    ) -> Result<Context, ProviderError> {
        self.0.acquire(cx, conn_options, context_id).await
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
    host: Arc<dyn MuxHost>,
    max_contexts: u32,
    env: ProcessEnv,
    transports: Transports,
    sockets: Mutex<Sockets>,
    /// One connect at a time, so streams opening together share the socket it opens.
    connecting: tokio::sync::Mutex<()>,
    /// The readers and the closes of sockets no stream holds.
    tasks: Arc<TaskSet>,
}

impl Shared {
    fn new(
        host: Arc<dyn MuxHost>,
        max_contexts: Option<u32>,
        env: ProcessEnv,
        transports: Transports,
    ) -> Self {
        Shared {
            host,
            max_contexts: max_contexts.unwrap_or(u32::MAX).max(1),
            env,
            transports,
            sockets: Mutex::new(Sockets::default()),
            connecting: tokio::sync::Mutex::new(()),
            tasks: Arc::new(TaskSet::default()),
        }
    }

    fn clock(&self) -> &dyn Clock {
        self.env.clock.as_ref()
    }

    fn background(&self) -> Result<Background, ApiError> {
        let spawner = (self.env.spawner)()?;
        Ok(Background::new(Arc::clone(&self.tasks), spawner, Arc::clone(&self.env.clock)))
    }

    /// A context on a socket with room for it, connecting one when none has. Registered: the
    /// caller owns its release from here on.
    pub(super) async fn acquire(
        self: &Arc<Self>,
        cx: &Cx,
        conn_options: &ApiConnectOptions,
        context_id: String,
    ) -> Result<Context, ProviderError> {
        let background = self.background()?;
        let socket = {
            let _connecting = self.connecting.lock().await;
            match self.take_room(&background) {
                Some(socket) => socket,
                None => self.connect(cx, conn_options, &background, 1).await?,
            }
        };
        let (tx, rx) = mpsc::unbounded_channel();
        socket.routes.lock().insert(context_id.clone(), tx);
        Ok(Context { mux: Arc::clone(self), socket, context_id, rx })
    }

    /// A current socket with room for one more context, now holding it. Idle sockets past
    /// their age close.
    fn take_room(&self, background: &Background) -> Option<Arc<Socket>> {
        let now = self.clock().now();
        let retired = |usage: &Usage| usage.retire_at.is_some_and(|at| now >= at);
        let mut sockets = self.sockets.lock();
        let generation = sockets.generation;
        let aged: Vec<u64> = sockets
            .open
            .iter()
            .filter(|(_, (_, usage))| usage.contexts == 0 && retired(usage))
            .map(|(&id, _)| id)
            .collect();
        for id in aged {
            if let Some((socket, _)) = sockets.open.remove(&id) {
                self.close_later(background, socket);
            }
        }
        let max = self.max_contexts;
        let (socket, usage) =
            sockets.open.values_mut().find(|(_, usage)| usage.room(generation, now, max))?;
        usage.contexts += 1;
        Some(Arc::clone(socket))
    }

    /// Connects a socket holding `contexts` streams from the start.
    async fn connect(
        self: &Arc<Self>,
        cx: &Cx,
        conn_options: &ApiConnectOptions,
        background: &Background,
        contexts: u32,
    ) -> Result<Arc<Socket>, ProviderError> {
        loop {
            let generation = self.sockets.lock().generation;
            let connection = self.host.connect(cx, *conn_options).await?;
            let (sender, receiver) =
                open_direct(&self.transports, self.clock(), conn_options, &connection).await?;
            let options = super::executor::options(&connection, Duration::ZERO)?;
            let now = self.clock().now();
            let opened = {
                let mut sockets = self.sockets.lock();
                (sockets.generation == generation).then(|| {
                    sockets.next_id += 1;
                    let socket = Arc::new(Socket {
                        id: sockets.next_id,
                        sender: Arc::clone(&sender),
                        routes: Mutex::new(BTreeMap::new()),
                        last_sent: Mutex::new(now),
                    });
                    let retire_at =
                        options.max_session_duration.map(|limit| now.saturating_add(limit));
                    let usage = Usage { contexts, retire_at, generation };
                    sockets.open.insert(socket.id, (Arc::clone(&socket), usage));
                    socket
                })
            };
            let Some(socket) = opened else {
                // the options changed while it connected: it has the old ones
                close_by(self.clock(), background.deadline(), sender.as_ref()).await;
                continue;
            };
            background.spawn(
                "socket mux reader",
                read(
                    Arc::downgrade(self),
                    Arc::clone(&self.host),
                    Arc::clone(&socket),
                    receiver,
                    options.keepalive,
                    options.ping,
                    Arc::clone(&self.env.clock),
                ),
            );
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
            if mux.has_room() {
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
        let sockets = self.sockets.lock();
        let generation = sockets.generation;
        sockets.open.values().any(|(_, usage)| usage.room(generation, now, self.max_contexts))
    }

    /// A context left `socket`; once idle a stale socket closes.
    fn released(&self, socket: &Arc<Socket>) {
        let now = self.clock().now();
        let mut sockets = self.sockets.lock();
        let generation = sockets.generation;
        let Some((_, usage)) = sockets.open.get_mut(&socket.id) else { return };
        usage.contexts = usage.contexts.saturating_sub(1);
        let stale = usage.generation != generation || usage.retire_at.is_some_and(|at| now >= at);
        if usage.contexts == 0
            && stale
            && let Some((socket, _)) = sockets.open.remove(&socket.id)
            && let Ok(background) = self.background()
        {
            self.close_later(&background, socket);
        }
    }

    /// Options changed: idle sockets close now, busy ones once their last stream ends.
    pub fn invalidate(&self) {
        let Ok(background) = self.background() else { return };
        let mut sockets = self.sockets.lock();
        sockets.generation += 1;
        let idle: Vec<u64> = sockets
            .open
            .iter()
            .filter(|(_, (_, usage))| usage.contexts == 0)
            .map(|(&id, _)| id)
            .collect();
        for id in idle {
            if let Some((socket, _)) = sockets.open.remove(&id) {
                self.close_later(&background, socket);
            }
        }
    }

    pub async fn aclose(&self) {
        let open = {
            let mut sockets = self.sockets.lock();
            sockets.generation += 1;
            std::mem::take(&mut sockets.open)
        };
        self.tasks.close();
        let deadline = self.clock().now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        for (socket, _) in open.into_values() {
            close_by(self.clock(), deadline, socket.sender.as_ref()).await;
        }
        self.tasks.cancel_and_wait(self.clock()).await;
    }

    /// Closes `socket` on the mux's tasks; its reader then ends.
    fn close_later(&self, background: &Background, socket: Arc<Socket>) {
        let clock = Arc::clone(&self.env.clock);
        let deadline = background.deadline();
        background.cleanup("socket mux close", async move {
            close_by(clock.as_ref(), deadline, socket.sender.as_ref()).await
        });
    }
}

/// A shared socket's reader: each frame to the context the multiplexer routes it to, the
/// keepalive and the pings, and the socket's end to every context on it.
async fn read(
    mux: Weak<Shared>,
    host: Arc<dyn MuxHost>,
    socket: Arc<Socket>,
    mut receiver: Box<dyn WsReceiver>,
    keepalive: Option<(Duration, WsFrame)>,
    ping: Option<Duration>,
    clock: Arc<dyn Clock>,
) {
    let liveness = ping.map(|interval| Mutex::new(Liveness::new(interval, clock.now())));
    let heard = || {
        if let Some(liveness) = &liveness {
            liveness.lock().heard();
        }
    };
    let end: Routed = loop {
        let keepalive_at = keepalive
            .as_ref()
            .map(|(interval, _)| socket.last_sent.lock().saturating_add(*interval));
        let ping_at = liveness.as_ref().map(|liveness| liveness.lock().wake_at());
        tokio::select! {
            biased;
            inbound = recv_heard(receiver.as_mut(), &heard) => match inbound {
                Ok(Inbound::Frame(frame)) => match host.route(&Cx::NONE, frame.clone()).await {
                    Ok(Some(context_id)) => {
                        if let Some(route) = socket.routes.lock().get(&context_id) {
                            let _ = route.send(Ok(Inbound::Frame(frame)));
                        }
                    }
                    // a socket-level frame, which the multiplexer handled
                    Ok(None) => {}
                    Err(error) => break Err(error),
                },
                Ok(closed) => break Ok(closed),
                Err(error) => break Err(error.into()),
            },
            () = crate::wait_for_deadline(clock.as_ref(), keepalive_at) => {
                if let Some((_, frame)) = &keepalive
                    && let Err(error) = socket.send(frame.clone(), clock.as_ref()).await
                {
                    break Err(lost(error).into());
                }
            }
            () = crate::wait_for_deadline(clock.as_ref(), ping_at) => {
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
        }
    };
    if let Some(mux) = mux.upgrade() {
        mux.sockets.lock().open.remove(&socket.id);
    }
    for route in std::mem::take(&mut *socket.routes.lock()).into_values() {
        let _ = route.send(end.clone());
    }
    let deadline = clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
    close_by(clock.as_ref(), deadline, socket.sender.as_ref()).await;
}

/// A stream's context on a shared socket.
pub(super) struct Context {
    mux: Arc<Shared>,
    socket: Arc<Socket>,
    context_id: String,
    rx: mpsc::UnboundedReceiver<Routed>,
}

impl Context {
    /// Sends the multiplexer's frames that open the context.
    pub(super) async fn open(&self, cx: &Cx) -> Result<(), ProviderError> {
        for frame in self.mux.host.open_context(cx, self.context_id.clone()).await? {
            self.socket.send(frame, self.mux.clock()).await.map_err(lost)?;
        }
        Ok(())
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

    /// Leaves the socket by `deadline`; a context the provider did not finish is closed with
    /// the multiplexer's frames first.
    pub(super) async fn release(self, finished: bool, deadline: Ts) -> Closed {
        self.socket.routes.lock().remove(&self.context_id);
        let mut closed = Closed::Clean;
        if !finished && !self.socket.sender.is_closed() {
            let clock = self.mux.clock();
            let close = async {
                let frames =
                    self.mux.host.close_context(&Cx::NONE, self.context_id.clone()).await?;
                for frame in frames {
                    self.socket.send(frame, clock).await.map_err(lost)?;
                }
                Ok::<(), ProviderError>(())
            };
            closed = match lk_agents_abi::before(clock, Some(deadline), close).await {
                Ok(Ok(())) => Closed::Clean,
                Ok(Err(error)) => {
                    log::debug!(target: lk_agents_abi::LOG_TARGET,
                        "closing a multiplexed context failed: {error}");
                    Closed::Failed
                }
                Err(_) => Closed::TimedOut,
            };
        }
        self.mux.released(&self.socket);
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
