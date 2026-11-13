use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::errors::{ApiError, ErrorKind, ProviderError};
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::utils::socket::{SocketConnection, SocketSend, WsFrame};
use lk_agents_abi::{BoxFuture, Clock, Closed, Ts};
use lk_agents_core::request::{Policy, Request, Retry, Step};
use lk_agents_net::owned::{Background, Owned};
use lk_agents_net::signal;
use lk_agents_net::transport::{WsReceiver, WsSender};
use parking_lot::Mutex;

use super::connection::{Connection, ConnectionOptions, Finish, Readiness, SocketEffect};
use super::liveness::{Beat, Liveness};
use super::mux::Context;
use super::{DRAIN_LIMIT, Inbound, Links, Session, close_by, lost, open_direct, recv_heard, send};

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

/// `run()`: attempts under `conn_options`, each one connection after another until the stream
/// ends; a failed attempt is retried while its error and the policy allow.
pub(super) async fn run(session: &Session, cx: &Cx) -> Result<(), ProviderError> {
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
        let outcome = attempt(session, cx).await;
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

/// `None` once the session was stopped.
async fn until_stopped<T>(session: &Session, fut: impl Future<Output = T>) -> Option<T> {
    tokio::select! {
        biased;
        () = session.stop.cancelled() => None,
        value = fut => Some(value),
    }
}

/// The input item being taken, and the connection it is taken for.
type Taking<'a> = Option<(BoxFuture<'a, Result<SocketSend, ProviderError>>, u64)>;

/// One attempt: connections until the stream ends or one fails. The input item being taken and
/// one taken but not sent survive a rotation; a command belongs to the connection it was taken
/// for, and goes back to the protocol when that one could not write it.
async fn attempt(session: &Session, cx: &Cx) -> Result<(), ProviderError> {
    let background = session.env.background().map_err(ApiError::from)?;
    let mut next: Taking<'_> = None;
    let mut held: Option<SocketSend> = None;
    let mut connections = 0u64;
    let result = loop {
        // read before the connection opens: a reconnect asked for while it opens rotates it
        let rotations = session.rotations.with(|n| *n);
        let opened = until_stopped(session, open(session, cx, &background)).await;
        let Some(opened) = opened else { break Ok(()) };
        let (link, options, setup) = match opened {
            Ok(opened) => opened,
            Err(error) => break Err(error),
        };
        connections += 1;
        let current = Current { number: connections, rotations, setup };
        match connection(session, cx, link, options, current, &mut next, &mut held).await {
            Ok(Finish::Ended) => break Ok(()),
            Ok(Finish::Rotate) => {}
            Err(error) => break Err(error),
        }
        if let Some(command) = held.as_ref().and_then(|item| item.command) {
            held = None;
            session.host.returned(command);
        }
    };
    if let Some(command) = held.and_then(|item| item.command) {
        session.host.returned(command);
    }
    result
}

/// What one connection is: its number in the attempt, the rotations it was opened after, and
/// the setup it sends first.
struct Current {
    number: u64,
    rotations: u64,
    setup: Vec<WsFrame>,
}

async fn open(
    session: &Session,
    cx: &Cx,
    background: &Background,
) -> Result<(Owned<Link>, ConnectionOptions, Vec<WsFrame>), ProviderError> {
    let drain_limit =
        session.conn_options.read_timeout.map_or(DRAIN_LIMIT, Duration::from_secs_f64);
    match &session.links {
        Links::Direct(transports) => {
            let connection = session.host.connect(cx, session.conn_options).await?;
            let clock = session.env.clock.as_ref();
            // the connect limit covers the handshake and the provider's readiness
            let ready_by = session.conn_options.connect_deadline(clock.now());
            let (sender, receiver) =
                open_direct(transports, clock, &session.conn_options, &connection).await?;
            if let Some(name) = &connection.target.request_id_header {
                let request_id = receiver.response_header(name).unwrap_or_default();
                log::debug!(target: lk_agents_abi::LOG_TARGET,
                    request_id = request_id.as_str(); "{}: connected", session.label
                );
            }
            let link = Link::owned(background, sender, Source::Direct(receiver));
            let mut options = options(&connection, drain_limit)?;
            if connection.awaits_ready {
                options.readiness = Readiness::Awaited { by: ready_by };
            }
            Ok((link, options, connection.setup))
        }
        Links::Shared { mux, context_id } => {
            let context = mux.acquire(cx, &session.conn_options, context_id.clone()).await?;
            let sender = context.sender();
            let link = Link::owned(background, sender, Source::Shared(context));
            if let Source::Shared(context) = &link.inbound {
                context.open(cx).await?;
            }
            // the keepalive and the rotation belong to the shared socket
            let options = ConnectionOptions {
                keepalive: None,
                goodbye: None,
                max_session_duration: None,
                drain_limit,
                readiness: Readiness::Open,
                ping: None,
            };
            Ok((link, options, Vec::new()))
        }
    }
}

pub(super) fn options(
    connection: &SocketConnection,
    drain_limit: Duration,
) -> Result<ConnectionOptions, ApiError> {
    let seconds = |value: f64, what: &str| {
        Duration::try_from_secs_f64(value).ok().filter(|d| !d.is_zero()).ok_or_else(|| {
            ApiError::new(ErrorKind::InvalidRequest, format!("{what} must be positive"))
        })
    };
    let keepalive = match &connection.keepalive {
        Some(keepalive) => {
            Some((seconds(keepalive.interval, "keepalive.interval")?, keepalive.frame.clone()))
        }
        None => None,
    };
    let max_session_duration = match connection.max_session_duration {
        Some(limit) => Some(seconds(limit, "max_session_duration")?),
        None => None,
    };
    let ping = match connection.ping_interval {
        Some(interval) => Some(seconds(interval, "ping_interval")?),
        None => None,
    };
    Ok(ConnectionOptions {
        keepalive,
        goodbye: connection.goodbye.clone(),
        max_session_duration,
        drain_limit,
        readiness: Readiness::Open,
        ping,
    })
}

/// Sends the effects' frames; the connection's end once one ends it.
async fn apply(
    sender: &dyn WsSender,
    effects: Vec<SocketEffect>,
    changed: &signal::Shared<u64>,
) -> Result<Option<Finish>, ProviderError> {
    changed.modify(|n| *n += 1);
    for effect in effects {
        match effect {
            SocketEffect::Send(frame) => send(sender, frame).await?,
            SocketEffect::Finish(finish) => return Ok(Some(finish)),
            SocketEffect::Fail(error) => return Err(error.into()),
        }
    }
    Ok(None)
}

/// One connection, from its opening to its finish: the receive, send, keepalive and rotation
/// loops race, and the first to end it decides; the socket is then closed, bounded.
async fn connection<'a>(
    session: &'a Session,
    cx: &'a Cx,
    mut link: Owned<Link>,
    options: ConnectionOptions,
    current: Current,
    next: &mut Taking<'a>,
    held: &mut Option<SocketSend>,
) -> Result<Finish, ProviderError> {
    let clock = session.env.clock.as_ref();
    let drain_limit = options.drain_limit;
    let Current { number, rotations, setup } = current;
    let readies = session.readies.with(|n| *n);
    let liveness = options.ping.map(|interval| Mutex::new(Liveness::new(interval, clock.now())));
    let machine = Mutex::new(Connection::open(options, clock.now()));
    // bumped on every change of the machine, so each loop's wait rechecks it
    let changed = signal::Shared::new(0u64);
    let sender = Arc::clone(&link.sender);
    // the provider session's setup goes first, ahead of anything held from before
    for frame in setup {
        send(sender.as_ref(), frame).await?;
    }

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
                    Inbound::Frame(frame) => {
                        let done = session.host.on_message(cx, frame).await?;
                        let mut machine = machine.lock();
                        if session.readies.with(|n| *n != readies) {
                            machine.ready(clock.now());
                        }
                        if done {
                            machine.done(&mut effects);
                        }
                    }
                    Inbound::Closed { code, reason } => {
                        machine.lock().closed(code, reason, &mut effects);
                    }
                }
                if let Some(finish) = apply(sender.as_ref(), effects, &changed).await? {
                    return Ok(finish);
                }
            }
        };
        let send_input = async {
            loop {
                changed.wait_until(|_| machine.lock().taking_input().then_some(())).await;
                let item = match held.take() {
                    Some(item) => item,
                    None => {
                        let (taking, taken_for) =
                            next.get_or_insert_with(|| (session.host.next(cx), number));
                        let item = taking.await?;
                        let taken_for = *taken_for;
                        *next = None;
                        if let Some(command) = item.command
                            && taken_for != number
                        {
                            // taken for a connection that ended before writing it
                            session.host.returned(command);
                            continue;
                        }
                        item
                    }
                };
                if !machine.lock().taking_input() {
                    match item.command {
                        Some(command) => session.host.returned(command),
                        None => *held = Some(item),
                    }
                    continue;
                }
                let (boundary, end, done, command) =
                    (item.boundary, item.end, item.done, item.command);
                for frame in item.frames {
                    if let Err(error) = send(sender.as_ref(), frame).await {
                        // a write fails once the peer closed: its close says why, so the
                        // receive loop decides, unless it never hears
                        clock.wait_until(clock.now().saturating_add(drain_limit)).await;
                        return Err(error.into());
                    }
                }
                if let Some(command) = command {
                    session.host.sent(command);
                }
                let mut effects = Vec::new();
                {
                    let mut machine = machine.lock();
                    match (done, end) {
                        (true, _) => machine.done(&mut effects),
                        (false, true) => machine.input_ended(clock.now(), &mut effects),
                        (false, false) => machine.sent(clock.now(), boundary, &mut effects),
                    }
                }
                if let Some(finish) = apply(sender.as_ref(), effects, &changed).await? {
                    return Ok(finish);
                }
            }
        };
        let timers = async {
            loop {
                let seen = changed.with(|n| *n);
                let at = machine.lock().wake_at();
                tokio::select! {
                    biased;
                    () = changed.wait_until(|n| (*n != seen).then_some(())) => continue,
                    () = crate::wait_for_deadline(clock, at) => {}
                }
                let mut effects = Vec::new();
                machine.lock().tick(clock.now(), &mut effects);
                if let Some(finish) = apply(sender.as_ref(), effects, &changed).await? {
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
                    Beat::Ping => sender.ping().await.map_err(lost)?,
                    Beat::Lost => {
                        let error = "the provider stopped answering pings";
                        return Err(ApiError::connection(Some(error.into())).into());
                    }
                }
            }
        };
        let rotation = async {
            session.rotations.wait_until(|n| (*n != rotations).then_some(())).await;
            let mut effects = Vec::new();
            machine.lock().rotate(clock.now(), &mut effects);
            match apply(sender.as_ref(), effects, &changed).await? {
                Some(finish) => Ok(finish),
                // draining: the receive or timer loop finishes it
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            biased;
            () = session.stop.cancelled() => Ok(Finish::Ended),
            outcome = receive => outcome,
            outcome = send_input => outcome,
            outcome = timers => outcome,
            outcome = pings => outcome,
            outcome = rotation => outcome,
        }
    };
    link.finished = matches!(outcome, Ok(Finish::Ended)) && machine.lock().is_finished();
    link.close().await;
    outcome
}
