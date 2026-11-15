use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use lk_agents_abi::errors::{ApiError, ErrorKind, ProviderError};
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::utils::websocket::{WebSocketConnection, WebSocketInput, WsFrame};
use lk_agents_abi::{BoxFuture, Clock, Closed, Ts};
use lk_agents_core::request::{Policy, Request, Retry, Step};
use lk_agents_net::owned::{Background, Owned};
use lk_agents_net::signal;
use lk_agents_net::transport::{WsReceiver, WsSender};
use parking_lot::Mutex;

use super::connection::{Connection, ConnectionOptions, Finish, Readiness, WebSocketEffect};
use super::liveness::{Beat, Liveness};
use super::mux::Context;
use super::{DEFAULT_WAIT, Inbound, Links, Session, close_by, lost, open_direct, recv_heard, send};

/// One connection's socket: its sender, and where its frames come from.
pub(super) struct Link {
    sender: Arc<dyn WsSender>,
    inbound: Source,
    /// The provider finished the stream: a shared context owes no close of its own.
    finished: bool,
}

enum Source {
    Direct(Box<dyn WsReceiver>),
    Shared(Context),
}

impl Link {
    /// Held from the moment it exists: dropping it unclosed closes it in the background.
    fn owned(background: &Background, sender: Arc<dyn WsSender>, inbound: Source) -> Owned<Link> {
        let clock = Arc::clone(&background.clock);
        let link = Link { sender, inbound, finished: false };
        Owned::new(
            background,
            link,
            Box::new(move |link: Link, deadline: Ts| {
                Box::pin(async move { link.close(clock.as_ref(), deadline).await })
            }),
        )
    }

    /// The next frame or close; `heard` learns of everything a direct peer sends.
    async fn recv(&mut self, heard: &(dyn Fn() + Sync)) -> Result<Inbound, ProviderError> {
        match &mut self.inbound {
            Source::Direct(receiver) => Ok(recv_heard(receiver.as_mut(), heard).await?),
            Source::Shared(context) => context.recv().await,
        }
    }

    async fn close(self, clock: &dyn Clock, deadline: Ts) -> Closed {
        match self.inbound {
            Source::Direct(_) => close_by(clock, deadline, self.sender.as_ref()).await,
            Source::Shared(context) => context.release(self.finished, deadline).await,
        }
    }
}

/// The stream's input: the item being taken, the one taken but not fully written (and how many of
/// its frames its connection wrote), and whether the input ended. A connection that ends
/// mid-item leaves the item to go whole on the next one.
#[derive(Default)]
struct Input<'a> {
    taking: Option<BoxFuture<'a, Result<Option<WebSocketInput>, ProviderError>>>,
    in_flight: Option<InFlight>,
    ended: bool,
}

struct InFlight {
    input: WebSocketInput,
    written: usize,
}

impl<'a> Input<'a> {
    /// Whether the next item is still to take.
    fn wants_item(&self) -> bool {
        self.in_flight.is_none() && !self.ended
    }

    /// Takes the next item, or learns the input ended. The taking survives a dropped call.
    async fn take(&mut self, session: &'a Session, cx: &'a Cx) -> Result<(), ProviderError> {
        while self.wants_item() {
            let taking = self.taking.get_or_insert_with(|| session.host.next_input(cx));
            let taken = taking.await?;
            self.taking = None;
            match taken {
                Some(input) if input.frames.is_empty() => {}
                Some(input) => self.in_flight = Some(InFlight { input, written: 0 }),
                None => self.ended = true,
            }
        }
        Ok(())
    }
}

/// A control message being taken.
type TakingControl<'a> = BoxFuture<'a, Result<Vec<WsFrame>, ProviderError>>;

/// The control message one connection is taking. The host hands over only messages sent for the
/// connection open when it does, and runs its take on its own: one it completed after the
/// connection ended (sent while none was open) belongs to no connection, so the take ends with its
/// connection, never carried to the next. Dropping a pending take leaves its message queued.
#[derive(Default)]
struct Controls<'a> {
    taking: Option<TakingControl<'a>>,
}

impl<'a> Controls<'a> {
    async fn take(
        &mut self,
        session: &'a Session,
        cx: &'a Cx,
    ) -> Result<Vec<WsFrame>, ProviderError> {
        let taking = self.taking.get_or_insert_with(|| session.host.next_control(cx));
        let frames = taking.await;
        self.taking = None;
        frames
    }
}

/// What a run carries from one connection to the next, across replacements and retries.
#[derive(Default)]
struct Carried<'a> {
    input: Input<'a>,
}

/// `run()`: attempts under `conn_options`, each one connection after another until the stream
/// ends; a failed attempt is retried while its error and the policy allow.
pub(super) async fn run(session: &Session, cx: &Cx) -> Result<(), ProviderError> {
    attempts(session, cx, &mut Carried::default()).await
}

async fn attempts<'a>(
    session: &'a Session,
    cx: &'a Cx,
    carried: &mut Carried<'a>,
) -> Result<(), ProviderError> {
    let mut policy = Policy::from_options(&session.conn_options);
    // each connection bounds its own phases; the attempts keep the total limit
    policy.connect_timeout = None;
    policy.write_timeout = None;
    policy.read_timeout = None;
    let clock = session.env.clock.as_ref();
    let mut steps = Vec::new();
    let mut attempts = Request::start(policy, clock.now(), &mut steps);
    loop {
        if session.stop.is_cancelled() {
            attempts.cancel(clock.now(), &mut Vec::new());
            return Ok(());
        }
        let started = clock.now();
        attempts.wake(started, None, &mut steps);
        let mut gave_up = None;
        for step in steps.drain(..) {
            match step {
                Step::GiveUp(error) => gave_up = Some(error),
                Step::Retry(backoff) => {
                    let error = backoff.error().clone();
                    let _at = backoff.announce(&session.label, &error);
                }
                Step::Launch(_) | Step::Stop(_) | Step::Unresolved | Step::CleanupIncomplete(_) => {
                }
            }
        }
        if let Some(error) = gave_up {
            return Err(error.into());
        }
        let Some(id) = attempts.current() else { return Ok(()) };
        let outcome = attempt(session, cx, carried).await.map_err(|error| stamped(session, error));
        let now = clock.now();
        let error = match outcome {
            Ok(()) => {
                attempts.succeeded(id, now);
                attempts.closed(id, now, &mut Vec::new());
                return Ok(());
            }
            Err(error) => error,
        };
        let failure = error.attempt_error();
        let lasted = now.saturating_duration_since(started).as_secs_f64();
        if session.conn_options.attempt_connected(lasted, &failure) {
            attempts.reset_retries();
        }
        match attempts.attempt_failed(failure.clone(), now) {
            Retry::At(backoff) => {
                let at = backoff.announce(&session.label, &failure);
                session.host.retrying(error);
                if until_stopped(session, clock.wait_until(at)).await.is_none() {
                    attempts.cancel(clock.now(), &mut Vec::new());
                    return Ok(());
                }
            }
            // the error as the protocol raised it: an outer retry owner reads its own flag
            Retry::GiveUp(_) => return Err(error),
        }
    }
}

/// `error` carrying the provider's id for what failed, unless it names its own.
fn stamped(session: &Session, error: ProviderError) -> ProviderError {
    match error {
        ProviderError::Local(mut error) if error.request_id.is_none() => {
            error.request_id = session.request_id.lock().clone();
            ProviderError::Local(error)
        }
        other => other,
    }
}

/// `None` once the session was stopped.
async fn until_stopped<T>(session: &Session, fut: impl Future<Output = T>) -> Option<T> {
    tokio::select! {
        biased;
        () = session.stop.cancelled() => None,
        value = fut => Some(value),
    }
}

/// One attempt: connections until the stream ends or one fails.
async fn attempt<'a>(
    session: &'a Session,
    cx: &'a Cx,
    carried: &mut Carried<'a>,
) -> Result<(), ProviderError> {
    let background = session.env.background().map_err(ApiError::from)?;
    loop {
        // read before the connection opens: a reconnect asked for while it opens replaces it
        let rotations = session.rotations.with(|n| *n);
        let Some(opened) = until_stopped(session, open(session, cx, &background)).await else {
            return Ok(());
        };
        let (link, options, setup) = opened?;
        if let Some(in_flight) = &mut carried.input.in_flight {
            // an item left mid-write goes whole on the new connection
            in_flight.written = 0;
        }
        let current = Current { rotations, setup };
        match connection(session, cx, link, options, current, carried).await? {
            Finish::Ended => return Ok(()),
            Finish::Replaced => {}
            Finish::SessionFinished => {
                session.host.session_finished();
                // the next connection opens for the next input, not before, so it never idles
                let Some(taken) = until_stopped(session, carried.input.take(session, cx)).await
                else {
                    return Ok(());
                };
                taken?;
                if carried.input.in_flight.is_none() {
                    return Ok(());
                }
            }
        }
    }
}

/// What one connection is: the rotations it was opened after, and the setup it sends first.
struct Current {
    rotations: u64,
    setup: Vec<WsFrame>,
}

async fn open(
    session: &Session,
    cx: &Cx,
    background: &Background,
) -> Result<(Owned<Link>, ConnectionOptions, Vec<WsFrame>), ProviderError> {
    let reply_limit =
        session.conn_options.read_timeout.map_or(DEFAULT_WAIT, Duration::from_secs_f64);
    // the stream's own, read at each connection whether it is direct or shared
    let input_idle = session.host.input_idle_after().map_err(ProviderError::from)?;
    let input_idle = seconds(input_idle, "input_idle_after")?;
    let response_idle = session.host.response_idle_after().map_err(ProviderError::from)?;
    let response_idle = seconds(response_idle, "response_idle_after")?;
    match &session.links {
        Links::Direct(transports) => {
            let connection = session.host.connect(cx, session.conn_options).await?;
            if connection.keepalive.as_ref().is_some_and(|k| k.only_while_unused) {
                let message = "a keepalive only while unused is a multiplexer's: a stream's own \
                               socket is always in use";
                return Err(ApiError::new(ErrorKind::InvalidRequest, message).into());
            }
            let clock = session.env.clock.as_ref();
            // the connect limit covers the handshake and the provider's readiness
            let ready_by = ready_by(clock, &session.conn_options);
            let (sender, receiver) =
                open_direct(transports, &session.env, &session.conn_options, &connection).await?;
            let request_id = connection
                .target
                .request_id_header
                .as_deref()
                .and_then(|name| receiver.response_header(name));
            opened(session, request_id);
            let link = Link::owned(background, sender, Source::Direct(receiver));
            let mut options = options(&connection, reply_limit, input_idle)?;
            options.response_idle = response_idle;
            if connection.awaits_ready {
                options.readiness = Readiness::Awaited { by: ready_by };
            }
            Ok((link, options, connection.setup))
        }
        Links::Shared { mux, context_id } => {
            let host = Arc::clone(&session.host);
            let context = mux.acquire(cx, &session.conn_options, context_id.clone(), host).await?;
            let sender = context.sender();
            opened(session, context.request_id());
            let link = Link::owned(background, sender, Source::Shared(context));
            if let Source::Shared(context) = &link.inbound {
                context.open(cx).await?;
            }
            // the socket's own keepalive, readiness and rotation belong to the shared socket
            let options = ConnectionOptions {
                keepalive: None,
                max_session_duration: None,
                reply_limit,
                readiness: Readiness::Open,
                ping: None,
                input_idle,
                response_idle,
                goodbye: Vec::new(),
            };
            Ok((link, options, Vec::new()))
        }
    }
}

/// When a provider opened now must have said it is ready: the connect limit, else the default
/// wait, never unbounded.
pub(super) fn ready_by(clock: &dyn Clock, conn_options: &lk_agents_abi::ApiConnectOptions) -> Ts {
    let now = clock.now();
    conn_options.connect_deadline(now).unwrap_or_else(|| now.saturating_add(DEFAULT_WAIT))
}

/// The connection now carries `request_id`, the provider's own: the protocol learns it.
fn opened(session: &Session, request_id: Option<String>) {
    *session.request_id.lock() = request_id.clone();
    session.host.opened(request_id);
}

/// A positive duration of `value` seconds, named `what` when it is not.
pub(super) fn seconds(value: Option<f64>, what: &str) -> Result<Option<Duration>, ApiError> {
    value
        .map(|value| {
            Duration::try_from_secs_f64(value).ok().filter(|d| !d.is_zero()).ok_or_else(|| {
                ApiError::new(ErrorKind::InvalidRequest, format!("{what} must be positive"))
            })
        })
        .transpose()
}

pub(super) fn options(
    connection: &WebSocketConnection,
    reply_limit: Duration,
    input_idle: Option<Duration>,
) -> Result<ConnectionOptions, ApiError> {
    let keepalive = match &connection.keepalive {
        Some(keepalive) => seconds(Some(keepalive.interval), "keepalive.interval")?
            .map(|every| (every, keepalive.frame.clone())),
        None => None,
    };
    Ok(ConnectionOptions {
        keepalive,
        max_session_duration: seconds(connection.max_session_duration, "max_session_duration")?,
        reply_limit,
        readiness: Readiness::Open,
        ping: seconds(connection.ping_interval, "ping_interval")?,
        input_idle,
        response_idle: None,
        goodbye: connection.goodbye.clone(),
    })
}

/// What a connection's loops share: its socket's sender and its machine.
struct Loops<'a> {
    session: &'a Session,
    cx: &'a Cx,
    sender: Arc<dyn WsSender>,
    machine: Mutex<Connection>,
    /// Bumped on every change of the machine, so each loop's wait rechecks it.
    changed: signal::Shared<u64>,
}

impl Loops<'_> {
    fn clock(&self) -> &dyn Clock {
        self.session.env.clock.as_ref()
    }

    /// Writes `frame`; a failed write is the machine's input, its outcome decided by the peer's
    /// close or the machine's deadline.
    async fn write(&self, frame: WsFrame) -> bool {
        match send(self.sender.as_ref(), frame).await {
            Ok(()) => true,
            Err(error) => {
                self.write_failed(error);
                false
            }
        }
    }

    fn write_failed(&self, error: ApiError) {
        self.machine.lock().write_failed(self.clock().now(), error);
        self.changed.modify(|n| *n += 1);
    }

    /// Waits for the pace's slot of a frame lasting `duration`.
    async fn paced(&self, duration: Duration) {
        if let Some(pacing) = &self.session.pacing {
            let at = pacing.reserve(self.clock().now(), duration);
            self.clock().wait_until(at).await;
        }
    }

    /// Writes frames that are not input, at the session's pace; `false` once a write failed.
    async fn write_other(&self, frames: Vec<WsFrame>) -> bool {
        for frame in frames {
            self.paced(Duration::ZERO).await;
            if !self.write(frame).await {
                return false;
            }
            self.machine.lock().wrote(self.clock().now());
        }
        true
    }

    /// Sends the effects' frames; the connection's end once one ends it. A failed write leaves
    /// the rest unsent.
    async fn apply(&self, effects: Vec<WebSocketEffect>) -> Result<Option<Finish>, ProviderError> {
        self.changed.modify(|n| *n += 1);
        for effect in effects {
            let frames = match effect {
                WebSocketEffect::Send(frame) => vec![frame],
                WebSocketEffect::InputIdle => self.session.host.input_idle(self.cx).await?,
                WebSocketEffect::ResponseIdle => self.session.host.response_idle(self.cx).await?,
                WebSocketEffect::Finish(finish) => return Ok(Some(finish)),
                WebSocketEffect::Fail(error) => return Err(error.into()),
            };
            if !self.write_other(frames).await {
                return Ok(None);
            }
        }
        Ok(None)
    }

    /// Writes the item in flight at the session's pace, frame by frame, keeping count; it leaves
    /// the slot only once all of its frames went. A failed write leaves it there and waits for
    /// the connection to end.
    async fn write_in_flight(
        &self,
        in_flight: &mut Option<InFlight>,
    ) -> Result<Option<Finish>, ProviderError> {
        let Some(InFlight { input, written }) = in_flight.as_mut() else { return Ok(None) };
        self.machine.lock().writing_input(self.clock().now());
        // the pace counts the item's duration once, at its first frame
        let mut pace = Duration::try_from_secs_f64(input.pace).unwrap_or_default();
        while let Some(frame) = input.frames.get(*written).cloned() {
            self.paced(std::mem::take(&mut pace)).await;
            if !self.write(frame).await {
                return std::future::pending().await;
            }
            *written += 1;
        }
        *in_flight = None;
        let mut effects = Vec::new();
        self.machine.lock().input_written(self.clock().now(), &mut effects);
        match self.apply(effects).await? {
            Some(finish) => Ok(Some(finish)),
            None => self.replace_if_due().await,
        }
    }

    /// Replaces the connection once `max_session_duration` passed and the stream has nothing in
    /// progress a new connection would cut, after draining the provider session its goodbye ends.
    async fn replace_if_due(&self) -> Result<Option<Finish>, ProviderError> {
        if !self.machine.lock().replacement_due() {
            return Ok(None);
        }
        if !self.session.host.can_reconnect().map_err(ProviderError::from)? {
            return Ok(None);
        }
        let mut effects = Vec::new();
        self.machine.lock().replace_due(self.clock().now(), &mut effects);
        self.apply(effects).await
    }
}

/// One connection, from its opening to its finish: the receive, send, keepalive and replacement
/// loops race, and the first to end it decides; the socket is then closed, bounded.
async fn connection<'a>(
    session: &'a Session,
    cx: &'a Cx,
    mut link: Owned<Link>,
    options: ConnectionOptions,
    current: Current,
    carried: &mut Carried<'a>,
) -> Result<Finish, ProviderError> {
    let clock = session.env.clock.as_ref();
    let Current { rotations, setup } = current;
    let liveness = options.ping.map(|interval| Mutex::new(Liveness::new(interval, clock.now())));
    let loops = Loops {
        session,
        cx,
        sender: Arc::clone(&link.sender),
        machine: Mutex::new(Connection::open(options, clock.now())),
        changed: signal::Shared::new(0u64),
    };
    let loops = &loops;

    let outcome = {
        let receive = async {
            loop {
                let inbound = link
                    .recv(&|| {
                        if let Some(liveness) = &liveness {
                            liveness.lock().heard();
                        }
                    })
                    .await?;
                let mut effects = Vec::new();
                match inbound {
                    Inbound::Frame { frame, message } => {
                        let read = session.host.on_message(cx, frame, message).await?;
                        if read.response {
                            loops.machine.lock().received_response(clock.now());
                        }
                        if let Some(request_id) = read.request_id {
                            *session.request_id.lock() = Some(request_id);
                        }
                        if let Some(signal) = read.signal {
                            loops.machine.lock().signal(signal, clock.now(), &mut effects);
                        }
                    }
                    Inbound::Closed { code, reason } => {
                        let expected = session.host.expects_close().map_err(ProviderError::from)?;
                        // asked outside the lock: whether the provider closed with words owed
                        let finished = loops.machine.lock().is_finished();
                        let nothing_pending = !finished
                            && session.host.nothing_pending().map_err(ProviderError::from)?;
                        let mut machine = loops.machine.lock();
                        // a reconnect asked for before the close replaces the connection, whether
                        // or not the replacement loop ran first
                        if session.rotations.with(|n| *n) != rotations {
                            machine.replace(&mut effects);
                        }
                        machine.closed(code, reason, expected, nothing_pending, &mut effects);
                    }
                }
                if let Some(finish) = loops.apply(effects).await? {
                    return Ok(finish);
                }
                if let Some(finish) = loops.replace_if_due().await? {
                    return Ok(finish);
                }
            }
        };
        let send = async {
            let Carried { input } = carried;
            let mut controls = Controls::default();
            // the provider session's setup goes first, ahead of anything carried from before
            if !loops.write_other(setup).await {
                return std::future::pending().await;
            }
            loop {
                // ready first, then take: a connection that stops taking input takes nothing
                loops
                    .changed
                    .wait_until(|_| loops.machine.lock().taking_input().then_some(()))
                    .await;
                if input.wants_item() {
                    // a control message goes now, ahead of the input not yet taken
                    tokio::select! {
                        biased;
                        frames = controls.take(session, cx) => {
                            if !loops.write_other(frames?).await {
                                return std::future::pending().await;
                            }
                            continue;
                        }
                        taken = input.take(session, cx) => taken?,
                    }
                }
                let mut effects = Vec::new();
                if session.session_ended.swap(false, Ordering::SeqCst) {
                    // the stream read the session as finished: its next item opens a new one
                    loops.machine.lock().end_session(&mut effects);
                } else if !loops.machine.lock().taking_input() {
                    continue;
                } else if input.in_flight.is_some() {
                    if let Some(finish) = loops.write_in_flight(&mut input.in_flight).await? {
                        return Ok(finish);
                    }
                    continue;
                } else {
                    let owes_reply = session.host.owes_reply().map_err(ProviderError::from)?;
                    loops.machine.lock().input_ended(clock.now(), owes_reply, &mut effects);
                }
                match loops.apply(effects).await? {
                    Some(finish) => return Ok(finish),
                    // waiting for the provider's reply: the receive or timer loop finishes it
                    None => return std::future::pending().await,
                }
            }
        };
        let timers = async {
            loop {
                let seen = loops.changed.with(|n| *n);
                let at = loops.machine.lock().wake_at();
                tokio::select! {
                    biased;
                    () = loops.changed.wait_until(|n| (*n != seen).then_some(())) => continue,
                    () = crate::wait_for_deadline(clock, at) => {}
                }
                let now = clock.now();
                // asked outside the lock: the host knows whether a silent provider owes anything
                let runs_out = loops.machine.lock().reply_wait_runs_out(now);
                let nothing_pending =
                    runs_out && session.host.nothing_pending().map_err(ProviderError::from)?;
                let mut effects = Vec::new();
                loops.machine.lock().tick(now, nothing_pending, &mut effects);
                if let Some(finish) = loops.apply(effects).await? {
                    return Ok(finish);
                }
                if let Some(finish) = loops.replace_if_due().await? {
                    return Ok(finish);
                }
            }
        };
        let pings = async {
            let Some(liveness) = &liveness else { return std::future::pending().await };
            loop {
                let at = liveness.lock().wake_at();
                clock.wait_until(at).await;
                let beat = liveness.lock().tick(clock.now());
                match beat {
                    Beat::Wait => {}
                    // a write is never timed: the pong it owes is
                    Beat::Ping => {
                        if let Err(error) = loops.sender.ping().await {
                            loops.write_failed(lost(error));
                            return std::future::pending().await;
                        }
                    }
                    Beat::Lost => {
                        let error = "the provider stopped answering pings";
                        return Err(ApiError::connection(Some(error.into())).into());
                    }
                }
            }
        };
        let replacement = async {
            session.rotations.wait_until(|n| (*n != rotations).then_some(())).await;
            let mut effects = Vec::new();
            loops.machine.lock().replace(&mut effects);
            match loops.apply(effects).await? {
                Some(finish) => Ok(finish),
                // waiting for the item being written: the send loop finishes it
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            biased;
            () = session.stop.cancelled() => Ok(Finish::Ended),
            outcome = receive => outcome,
            outcome = send => outcome,
            outcome = timers => outcome,
            outcome = pings => outcome,
            outcome = replacement => outcome,
        }
    };
    link.finished = matches!(outcome, Ok(Finish::Ended | Finish::SessionFinished))
        && loops.machine.lock().is_finished();
    link.close().await;
    outcome
}
