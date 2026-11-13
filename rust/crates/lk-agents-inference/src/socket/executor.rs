use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::errors::{ApiError, ErrorKind, ProviderError};
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::utils::socket::{SocketConnection, SocketSend};
use lk_agents_abi::{BoxFuture, Clock, Closed, Ts};
use lk_agents_core::request::{Policy, Request, Retry, Step};
use lk_agents_net::owned::{Background, Owned};
use lk_agents_net::signal;
use lk_agents_net::transport::{WsReceiver, WsSender};
use parking_lot::Mutex;

use super::connection::{Connection, ConnectionOptions, Finish, SocketEffect};
use super::mux::Context;
use super::{DRAIN_LIMIT, Inbound, Links, Session, close_by, open_direct, recv_direct, send};

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

    async fn recv(&mut self) -> Result<Inbound, ProviderError> {
        match &mut self.inbound {
            Source::Direct(receiver) => Ok(recv_direct(receiver.as_mut()).await?),
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

/// One attempt: connections until the stream ends or one fails. The input item being taken and
/// one taken but not sent survive a rotation.
async fn attempt(session: &Session, cx: &Cx) -> Result<(), ProviderError> {
    let background = session.env.background().map_err(ApiError::from)?;
    let mut next: Option<BoxFuture<'_, Result<SocketSend, ProviderError>>> = None;
    let mut held: Option<SocketSend> = None;
    loop {
        let opened = until_stopped(session, open(session, cx, &background)).await;
        let Some(opened) = opened else { return Ok(()) };
        let (link, options) = opened?;
        match connection(session, cx, link, options, &mut next, &mut held).await? {
            Finish::Ended => return Ok(()),
            Finish::Rotate => {}
        }
    }
}

async fn open(
    session: &Session,
    cx: &Cx,
    background: &Background,
) -> Result<(Owned<Link>, ConnectionOptions), ProviderError> {
    let drain_limit =
        session.conn_options.read_timeout.map_or(DRAIN_LIMIT, Duration::from_secs_f64);
    match &session.links {
        Links::Direct(transports) => {
            let connection = session.host.connect(cx, session.conn_options).await?;
            let clock = session.env.clock.as_ref();
            let (sender, receiver) =
                open_direct(transports, clock, &session.conn_options, &connection).await?;
            let link = Link::owned(background, sender, Source::Direct(receiver));
            Ok((link, options(&connection, drain_limit)?))
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
            };
            Ok((link, options))
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
    Ok(ConnectionOptions {
        keepalive,
        goodbye: connection.goodbye.clone(),
        max_session_duration,
        drain_limit,
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
    next: &mut Option<BoxFuture<'a, Result<SocketSend, ProviderError>>>,
    held: &mut Option<SocketSend>,
) -> Result<Finish, ProviderError> {
    let clock = session.env.clock.as_ref();
    let drain_limit = options.drain_limit;
    let machine = Mutex::new(Connection::open(options, clock.now()));
    // bumped on every change of the machine, so each loop's wait rechecks it
    let changed = signal::Shared::new(0u64);
    let sender = Arc::clone(&link.sender);
    let rotations = session.rotations.with(|n| *n);

    let outcome = {
        let receive = async {
            loop {
                let inbound = link.recv().await?;
                let mut effects = Vec::new();
                match inbound {
                    Inbound::Frame(frame) => {
                        if session.host.on_message(cx, frame).await? {
                            machine.lock().done(&mut effects);
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
                        let taking = next.get_or_insert_with(|| session.host.next(cx));
                        let item = taking.await?;
                        *next = None;
                        item
                    }
                };
                if !machine.lock().taking_input() {
                    *held = Some(item);
                    continue;
                }
                let (boundary, end) = (item.boundary, item.end);
                for frame in item.frames {
                    if let Err(error) = send(sender.as_ref(), frame).await {
                        // a write fails once the peer closed: its close says why, so the
                        // receive loop decides, unless it never hears
                        clock.wait_until(clock.now().saturating_add(drain_limit)).await;
                        return Err(error.into());
                    }
                }
                let mut effects = Vec::new();
                {
                    let mut machine = machine.lock();
                    match end {
                        true => machine.input_ended(clock.now(), &mut effects),
                        false => machine.sent(clock.now(), boundary, &mut effects),
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
            outcome = rotation => outcome,
        }
    };
    link.finished = matches!(outcome, Ok(Finish::Ended)) && machine.lock().is_finished();
    link.close().await;
    outcome
}
