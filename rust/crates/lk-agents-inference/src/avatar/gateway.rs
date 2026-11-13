use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use parking_lot::Mutex;

use crate::avatar::{
    AvatarOptions, AvatarRoom, ProviderSession, create_request, decode_created, terminate_request,
};
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::inference::avatar_session::{AvatarError, InferenceAvatar};
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::{ApiConnectOptions, Clock, Closed, Extra, Ts};
use lk_agents_core::request::{self, AttemptFailure, Policy, Step};
use lk_agents_net::transport::HttpRequest;
use tokio::sync::{mpsc, oneshot};

use lk_agents_abi::{OwnedTask, SpawnerExt, TaskOutcome};
use lk_agents_net::env::ProcessEnv;

// The driver outlives its caller: it ends only once the instance is dropped and nothing is owed,
// so a session created after its caller left is still terminated. A create the gateway may have
// received is never abandoned: it is retried under its request id (one idempotency key) until the
// gateway answers, and a session it made is terminated.
use lk_agents_net::http::{GatewayHttp, Reporter, Sent, Stamped, read_body};
use lk_agents_net::task::{TaskSet, TokioSpawner};

/// Transport observations attempts may report ahead of the driver.
const OBSERVATION_QUEUE: usize = 32;

/// A gateway answer larger than this is not an avatar session.
const ANSWER_LIMIT: usize = 64 * 1024;

/// How long a create no caller takes keeps asking the gateway for the session it may have made.
pub const RETIREMENT_TIMEOUT: Duration = Duration::from_secs(30);

/// An exchange's response is expected whenever it is being read: the caller always waits.
const EXPECTING: Option<Ts> = Some(Ts::ZERO);

/// What the host's views read.
#[derive(Debug, Default)]
struct View {
    session_id: Option<String>,
    provider_session_id: Option<String>,
}

enum Input {
    Create {
        caller: u64,
        room: AvatarRoom,
        cx: Cx,
        reply: oneshot::Sender<Result<u32, AvatarError>>,
    },
    /// The create call `caller` stopped waiting.
    Cancel {
        caller: u64,
    },
    Close {
        cx: Cx,
        reply: oneshot::Sender<Closed>,
    },
}

/// `_core.InferenceAvatar`.
pub struct GatewayAvatar {
    opts: Arc<AvatarOptions>,
    view: Arc<Mutex<View>>,
    /// One message per host call, so the queue is bounded by the calls in flight.
    inputs: mpsc::UnboundedSender<Input>,
    callers: AtomicU64,
    /// The driver's owner; dropping the instance does not cancel it.
    _driver: Arc<TaskSet>,
}

impl GatewayAvatar {
    pub fn new(
        opts: AvatarOptions,
        credentials: (String, String),
        policy: Policy,
        http: Arc<GatewayHttp>,
        runtime: &tokio::runtime::Handle,
        clock: Arc<dyn Clock>,
    ) -> Self {
        let opts = Arc::new(opts);
        let view = Arc::new(Mutex::new(View::default()));
        let (inputs, rx) = mpsc::unbounded_channel();
        let (driver, observed) =
            Driver::new(opts.clone(), credentials, policy, http, runtime, clock, view.clone());
        // Self-owned: see the module docs.
        let tasks = Arc::new(TaskSet::default());
        tasks.spawn(
            &TokioSpawner::new(runtime.clone()),
            "avatar gateway driver",
            driver.run(rx, observed),
        );
        GatewayAvatar { opts, view, inputs, callers: AtomicU64::new(1), _driver: tasks }
    }

    /// Creates the provider session for the job `cx` runs for and returns the sample rate the
    /// avatar takes.
    pub async fn create(&self, room: AvatarRoom, cx: Cx) -> Result<u32, AvatarError> {
        let caller = self.callers.fetch_add(1, Ordering::Relaxed);
        let (reply, answer) = oneshot::channel();
        self.send(Input::Create { caller, room, cx, reply });
        let mut waiting = CancelOnDrop { inputs: &self.inputs, caller, armed: true };
        let outcome = answer.await;
        waiting.armed = false;
        outcome.unwrap_or_else(|_| Err(AvatarError::Closed))
    }

    /// Stops creating and terminates every held session; `TimedOut` when an attempt missed its
    /// cleanup deadline (it was reported and the runtime still owns it).
    pub async fn close(&self, cx: Cx) -> Closed {
        let (reply, settled) = oneshot::channel();
        self.send(Input::Close { cx, reply });
        settled.await.unwrap_or(Closed::Clean)
    }

    fn send(&self, input: Input) {
        // The driver outlives every handle; a send cannot fail while `self` exists.
        let _ = self.inputs.send(input);
    }
}

struct CancelOnDrop<'a> {
    inputs: &'a mpsc::UnboundedSender<Input>,
    caller: u64,
    armed: bool,
}

impl Drop for CancelOnDrop<'_> {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.inputs.send(Input::Cancel { caller: self.caller });
        }
    }
}

impl InferenceAvatar for GatewayAvatar {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        model: String,
        avatar_identity: Option<String>,
        avatar_name: Option<String>,
        extra_kwargs: Option<Extra>,
        base_url: String,
        api_key: String,
        api_secret: String,
        conn_options: ApiConnectOptions,
    ) -> Result<Self, AvatarError> {
        conn_options
            .validate()
            .map_err(|e| AvatarError::from(ApiError::new(e.to_string()).with_retryable(false)))?;
        // Signing then cannot fail when an attempt launches.
        lk_agents_net::token::inference_bearer(&api_key, &api_secret)?;
        let opts = AvatarOptions::new(
            &model,
            avatar_identity,
            avatar_name,
            extra_kwargs.unwrap_or_default(),
            base_url,
        )?;
        let runtime = (env.runtime)().map_err(ApiError::from)?;
        let http = (env.http)().map_err(ApiError::from)?;
        let policy = Policy::from_options(&conn_options);
        Ok(GatewayAvatar::new(
            opts,
            (api_key, api_secret),
            policy,
            http,
            &runtime,
            env.clock.clone(),
        ))
    }

    fn provider(&self) -> String {
        self.opts.provider.clone()
    }

    fn avatar_identity(&self) -> String {
        self.opts.identity.clone()
    }

    fn session_id(&self) -> Option<String> {
        self.view.lock().session_id.clone()
    }

    fn provider_session_id(&self) -> Option<String> {
        self.view.lock().provider_session_id.clone()
    }

    async fn create(
        &self,
        room_name: String,
        room_sid: String,
        livekit_url: Option<String>,
        agent_identity: String,
        cx: Cx,
    ) -> Result<u32, AvatarError> {
        let livekit_url =
            livekit_url.filter(|url| !url.is_empty()).ok_or(AvatarError::MissingUrl)?;
        let room = AvatarRoom { livekit_url, room_name, room_sid, agent_identity };
        GatewayAvatar::create(self, room, cx).await
    }

    async fn aclose(&self, cx: Cx) {
        // Decision 14: a missed cleanup deadline was logged and reported; close returns.
        let _ = self.close(cx).await;
    }
}

/// The create call waiting for its outcome.
struct Caller {
    id: u64,
    reply: oneshot::Sender<Result<u32, AvatarError>>,
}

/// A gateway request of one avatar instance: a create (its operation identity) or a terminate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct RequestId(u64);

/// What the attempts report, the host's calls, and what delivering a session decided. Every
/// attempt-scoped event names its request and attempt.
#[derive(Debug)]
enum Event {
    /// A deadline from [`Driver::wake_at`] is due.
    Wake,
    /// A caller asks for a session.
    Create,
    /// The create caller stopped waiting.
    Cancel,
    /// Terminate every held session and stop creating (`aclose()`); a repeat retries what a
    /// failed terminate left held.
    Close,
    /// The caller received the delivered session: the instance holds it.
    Adopted,
    /// The delivered session found no caller to receive it.
    DeliveryRefused,
    Connected {
        request: RequestId,
        attempt: u32,
    },
    RequestWritten {
        request: RequestId,
        attempt: u32,
    },
    Created {
        request: RequestId,
        attempt: u32,
        session: ProviderSession,
    },
    Terminated {
        request: RequestId,
        attempt: u32,
    },
    Failed {
        request: RequestId,
        attempt: u32,
        failure: AttemptFailure,
    },
    /// The attempt's transport resources are released.
    AttemptClosed {
        request: RequestId,
        attempt: u32,
    },
}

/// Why a session is terminated without being held: the journalled disposition of a result no
/// caller adopted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Retirement {
    /// Its caller stopped waiting.
    CallerGone,
    /// It was delivered and nobody received it.
    Refused,
    /// The instance was closed.
    Closed,
    /// A replayed answer the operation had settled without.
    Late,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Create,
    Terminate,
}

struct Request {
    kind: Kind,
    attempts: request::Request,
}

/// A create in flight; `caller` is owed its outcome until it stops waiting.
struct Creating {
    request: RequestId,
    caller: bool,
}

struct Driver {
    opts: Arc<AvatarOptions>,
    credentials: (String, String),
    policy: Policy,
    http: Arc<GatewayHttp>,
    runtime: tokio::runtime::Handle,
    reporter: Reporter<Event>,
    view: Arc<Mutex<View>>,
    /// The instant the driver last stepped at; supplied time never goes back.
    now: Ts,
    caller: Option<Caller>,
    /// A create call yet to be accepted or refused.
    arriving: Option<Caller>,
    /// The accepted create's room, read when its first attempt launches.
    pending_create: Option<AvatarRoom>,
    /// The context of the latest host call: its requests carry the identity of its job.
    cx: Cx,
    /// Each request's exchange, built once: every attempt sends the same operation.
    exchanges: BTreeMap<RequestId, Arc<HttpRequest>>,
    tasks: BTreeMap<(RequestId, u32), OwnedTask>,
    closers: Vec<oneshot::Sender<Closed>>,
    cleanup_incomplete: bool,
    /// Events delivering or launching produced, stepped next.
    queued: Vec<Event>,
    next_request: u64,
    /// Every request whose outcome is undecided or whose attempts still owe their release.
    requests: BTreeMap<RequestId, Request>,
    creating: Option<Creating>,
    delivering: Option<ProviderSession>,
    /// Sessions the instance owns with no terminate running.
    held: Vec<ProviderSession>,
    terminating: BTreeMap<RequestId, ProviderSession>,
    /// A session was adopted: one instance is one avatar.
    adopted: bool,
    closing: bool,
}

impl Driver {
    fn new(
        opts: Arc<AvatarOptions>,
        credentials: (String, String),
        policy: Policy,
        http: Arc<GatewayHttp>,
        runtime: &tokio::runtime::Handle,
        clock: Arc<dyn Clock>,
        view: Arc<Mutex<View>>,
    ) -> (Self, mpsc::Receiver<Stamped<Event>>) {
        let (tx, observed) = mpsc::channel(OBSERVATION_QUEUE);
        let driver = Driver {
            opts,
            credentials,
            policy,
            http,
            runtime: runtime.clone(),
            now: clock.now(),
            reporter: Reporter { tx, clock },
            view,
            caller: None,
            arriving: None,
            pending_create: None,
            cx: Cx::NONE,
            exchanges: BTreeMap::new(),
            tasks: BTreeMap::new(),
            closers: Vec::new(),
            cleanup_incomplete: false,
            queued: Vec::new(),
            next_request: 1,
            requests: BTreeMap::new(),
            creating: None,
            delivering: None,
            held: Vec::new(),
            terminating: BTreeMap::new(),
            adopted: false,
            closing: false,
        };
        (driver, observed)
    }

    async fn run(
        mut self,
        mut inputs: mpsc::UnboundedReceiver<Input>,
        mut observed: mpsc::Receiver<Stamped<Event>>,
    ) {
        let clock = self.reporter.clock.clone();
        let mut dropped = false;
        while !(dropped && self.is_settled()) {
            let wake = self.wake_at();
            tokio::select! {
                biased;
                Some(observation) = observed.recv() => self.step(observation.event, observation.at),
                input = inputs.recv(), if !dropped => match input {
                    Some(input) => self.input(input, clock.now()),
                    None => {
                        // The instance is gone: closing is the only cleanup left to run.
                        dropped = true;
                        self.step(Event::Close, clock.now());
                    }
                },
                () = async { clock.wait_until(wake.unwrap_or(Ts::from_nanos(u64::MAX))).await }, if wake.is_some() => {
                    self.step(Event::Wake, wake.unwrap_or(self.now));
                }
            }
        }
    }

    fn input(&mut self, input: Input, at: Ts) {
        match input {
            Input::Create { caller, room, cx, reply } => {
                self.cx = cx;
                self.pending_create = Some(room);
                self.arriving = Some(Caller { id: caller, reply });
                self.step(Event::Create, at);
                self.pending_create = None;
            }
            Input::Cancel { caller } => {
                if self.caller.as_ref().is_some_and(|c| c.id == caller) {
                    self.caller = None;
                    self.step(Event::Cancel, at);
                }
            }
            Input::Close { cx, reply } => {
                self.cx = cx;
                self.closers.push(reply);
                self.step(Event::Close, at);
            }
        }
    }

    /// The earliest instant an [`Event::Wake`] has work to do.
    fn wake_at(&self) -> Option<Ts> {
        self.requests.values().filter_map(|r| r.attempts.wake_at(EXPECTING)).min()
    }

    /// Nothing is in flight: no request, no attempt owing its release, no pending delivery.
    fn is_settled(&self) -> bool {
        self.requests.is_empty() && self.creating.is_none() && self.delivering.is_none()
    }

    /// Applies `event` at `at`, after every deadline due by then, then the events it queued.
    fn step(&mut self, event: Event, at: Ts) {
        let at = at.max(self.now);
        while let Some(due) = self.wake_at().filter(|due| *due <= at) {
            self.now = due.max(self.now);
            let acted = self.wake(self.now);
            if !acted && self.wake_at() == Some(due) {
                break;
            }
        }
        self.now = at;
        if let Event::AttemptClosed { request, attempt } = &event {
            self.tasks.remove(&(*request, *attempt));
        }
        self.handle(event, at);
        while !self.queued.is_empty() {
            for event in std::mem::take(&mut self.queued) {
                self.handle(event, self.now);
            }
        }
        self.exchanges.retain(|request, _| {
            self.tasks.keys().any(|(r, _)| r == request) || self.requests.contains_key(request)
        });
        if self.is_settled() {
            let outcome = if std::mem::take(&mut self.cleanup_incomplete) {
                Closed::TimedOut
            } else {
                Closed::Clean
            };
            for closer in self.closers.drain(..) {
                let _ = closer.send(outcome);
            }
        }
    }

    /// Runs every request's due deadlines; whether any of them acted.
    fn wake(&mut self, now: Ts) -> bool {
        let ids: Vec<RequestId> = self.requests.keys().copied().collect();
        let mut acted = false;
        for id in ids {
            self.advance(id, now, |attempts, steps| {
                attempts.wake(now, EXPECTING, steps);
                acted |= !steps.is_empty();
            });
        }
        acted
    }

    fn handle(&mut self, event: Event, now: Ts) {
        match event {
            Event::Wake => {
                self.wake(now);
            }
            Event::Create => self.create(now),
            Event::Cancel => self.stop_creating(now),
            Event::Close => {
                self.closing = true;
                if self.creating.as_ref().is_some_and(|c| c.caller) {
                    self.deliver(Err(AvatarError::Closed));
                }
                self.stop_creating(now);
                for session in std::mem::take(&mut self.held) {
                    self.terminate(session, None, now);
                }
            }
            Event::Adopted => {
                if let Some(session) = self.delivering.take() {
                    self.adopted = true;
                    if self.closing {
                        self.terminate(session, None, now);
                    } else {
                        self.held.push(session);
                    }
                }
            }
            Event::DeliveryRefused => {
                if let Some(session) = self.delivering.take() {
                    self.terminate(session, Some(Retirement::Refused), now);
                }
            }
            Event::Connected { request, attempt } => {
                if let Some(r) = self.requests.get_mut(&request) {
                    r.attempts.connected(attempt, now);
                }
            }
            Event::RequestWritten { request, attempt } => {
                if let Some(r) = self.requests.get_mut(&request) {
                    r.attempts.written(attempt, now);
                }
            }
            Event::Created { request, attempt, session } => {
                self.created(request, attempt, session, now);
            }
            Event::Terminated { request, attempt } => {
                if self.answered(request, attempt, now)
                    && let Some(session) = self.terminating.remove(&request)
                {
                    let mut view = self.view.lock();
                    if view.provider_session_id == session.provider_session_id {
                        view.provider_session_id = None;
                    }
                }
                self.forget_settled(request);
            }
            Event::Failed { request, attempt, failure } => {
                self.advance(request, now, |attempts, steps| {
                    attempts.failed(attempt, failure, now, steps)
                });
            }
            Event::AttemptClosed { request, attempt } => {
                self.advance(request, now, |attempts, steps| attempts.closed(attempt, now, steps));
            }
        }
    }

    /// Reserves the creation before its first attempt launches; refused while a creation is in
    /// flight, a session was adopted or the instance is closed.
    fn create(&mut self, now: Ts) {
        if self.closing || self.adopted || self.creating.is_some() || self.delivering.is_some() {
            if let Some(caller) = self.arriving.take() {
                let _ = caller.reply.send(Err(AvatarError::AlreadyStarted));
            }
            return;
        }
        let request = self.reserve();
        self.creating = Some(Creating { request, caller: true });
        let mut steps = Vec::new();
        let attempts = request::Request::start(self.policy, now, &mut steps);
        self.requests.insert(request, Request { kind: Kind::Create, attempts });
        self.apply(request, steps, now);
    }

    /// No caller is owed the creation any more: it is retired.
    fn stop_creating(&mut self, now: Ts) {
        if let Some(creating) = &mut self.creating {
            creating.caller = false;
            let request = creating.request;
            self.retire(request, now);
            self.forget_settled(request);
        }
    }

    /// `request` keeps asking under its identity until the gateway answers it, or stops now when
    /// the gateway cannot have received it.
    fn retire(&mut self, request: RequestId, now: Ts) {
        let Some(r) = self.requests.get_mut(&request) else { return };
        let mut steps = Vec::new();
        r.attempts.retire(now, now.saturating_add(RETIREMENT_TIMEOUT), &mut steps);
        self.apply(request, steps, now);
    }

    fn created(&mut self, request: RequestId, attempt: u32, session: ProviderSession, now: Ts) {
        let answered = self.answered(request, attempt, now);
        let creating = self.creating.as_ref().filter(|c| c.request == request);
        match creating {
            Some(c) if answered && c.caller && !self.closing => {
                self.creating = None;
                self.delivering = Some(session.clone());
                self.deliver(Ok(session));
            }
            Some(_) if answered => {
                self.creating = None;
                let reason = if self.closing { Retirement::Closed } else { Retirement::CallerGone };
                self.terminate(session, Some(reason), now);
            }
            _ if self.knows(&session) => {}
            _ => self.terminate(session, Some(Retirement::Late), now),
        }
        self.forget_settled(request);
    }

    /// Whether this answer decides `request`. Any attempt of an undecided request answers it: the
    /// operation's identity makes every attempt's answer the same, so a still-running attempt is
    /// stopped.
    fn answered(&mut self, request: RequestId, attempt: u32, now: Ts) -> bool {
        let Some(r) = self.requests.get_mut(&request) else { return false };
        if r.attempts.is_decided() {
            return false;
        }
        if !r.attempts.succeeded(attempt, now) {
            let mut steps = Vec::new();
            r.attempts.cancel(now, &mut steps);
            self.apply(request, steps, now);
        }
        true
    }

    /// Whether `session` is already held, delivered or being terminated.
    fn knows(&self, session: &ProviderSession) -> bool {
        self.delivering
            .iter()
            .chain(&self.held)
            .chain(self.terminating.values())
            .any(|s| s.same_as(session))
    }

    /// Terminates `session`; `reason` journals a session no caller holds. A session that cannot be
    /// terminated (no provider id or token) is released to its idle timeout.
    fn terminate(&mut self, session: ProviderSession, reason: Option<Retirement>, now: Ts) {
        let provider = self.opts.provider.as_str();
        let session_id = session.session_id.as_deref().unwrap_or_default();
        if let Some(reason) = reason {
            log::info!(target: lk_agents_abi::LOG_TARGET, provider, session_id, reason = format!("{reason:?}").as_str();
                "terminating an avatar session no caller holds");
        }
        if !session.terminable() {
            log::warn!(target: lk_agents_abi::LOG_TARGET, provider, session_id;
                "avatar session has no provider_session_id or terminate_token; it cannot be explicitly terminated and will bill until its provider idle timeout");
            return;
        }
        let request = self.reserve();
        self.terminating.insert(request, session);
        let mut steps = Vec::new();
        let attempts = request::Request::start(self.policy, now, &mut steps);
        self.requests.insert(request, Request { kind: Kind::Terminate, attempts });
        self.apply(request, steps, now);
    }

    fn reserve(&mut self) -> RequestId {
        let id = RequestId(self.next_request);
        self.next_request += 1;
        id
    }

    fn advance(
        &mut self,
        request: RequestId,
        now: Ts,
        f: impl FnOnce(&mut request::Request, &mut Vec<Step>),
    ) {
        let Some(r) = self.requests.get_mut(&request) else { return };
        let mut steps = Vec::new();
        f(&mut r.attempts, &mut steps);
        self.apply(request, steps, now);
        self.forget_settled(request);
    }

    fn apply(&mut self, request: RequestId, steps: Vec<Step>, now: Ts) {
        let Some(kind) = self.requests.get(&request).map(|r| r.kind) else { return };
        let opts = self.opts.clone();
        let provider = opts.provider.as_str();
        for step in steps {
            match step {
                Step::Launch(attempt) => self.launch(request, kind, attempt),
                Step::Stop(attempt) => {
                    if let Some(task) = self.tasks.get_mut(&(request, attempt)) {
                        task.cancel();
                    }
                }
                Step::Retry { attempt, error } => {
                    log::warn!(target: lk_agents_abi::LOG_TARGET, provider, attempt, error = error.message.as_str(); "avatar gateway request failed, retrying");
                }
                Step::CleanupIncomplete(attempt) => {
                    // The runtime still owns the attempt.
                    self.cleanup_incomplete = true;
                    log::error!(target: lk_agents_abi::LOG_TARGET, provider, request = request.0, attempt;
                        "avatar gateway request did not settle by its cleanup deadline; a provider session it created may bill until its idle timeout");
                }
                Step::Unresolved => {
                    // A later answer is still terminated.
                    self.cleanup_incomplete = true;
                    log::error!(target: lk_agents_abi::LOG_TARGET, provider, request = request.0;
                        "the gateway never answered an avatar create no caller took; a provider session it made may bill until its idle timeout");
                }
                Step::GiveUp(error) => self.give_up(request, kind, error, now),
            }
        }
    }

    /// `request` failed for good: a create reports to its caller and retires what the gateway may
    /// have made; a terminate holds the session again, so the next close retries it.
    fn give_up(&mut self, request: RequestId, kind: Kind, error: ApiError, now: Ts) {
        match kind {
            Kind::Create => {
                if let Some(creating) = self.creating.take_if(|c| c.request == request)
                    && creating.caller
                {
                    self.deliver(Err(AvatarError::Api(ProviderError::Local(error))));
                }
                self.retire(request, now);
            }
            Kind::Terminate => {
                let Some(session) = self.terminating.remove(&request) else { return };
                log::warn!(target: lk_agents_abi::LOG_TARGET, provider = self.opts.provider.as_str(), provider_session_id = session.provider_session_id.as_deref().unwrap_or_default(), error = error.message.as_str();
                    "failed to terminate inference avatar session; it will keep billing until its provider idle timeout unless aclose() is called again");
                self.held.push(session);
            }
        }
    }

    /// Drops `request` once nothing is owed for it; a decided create no longer reserves.
    fn forget_settled(&mut self, request: RequestId) {
        let decided = self.requests.get(&request).is_none_or(|r| r.attempts.is_decided());
        if decided {
            self.creating.take_if(|c| c.request == request);
        }
        if self.requests.get(&request).is_some_and(|r| r.attempts.is_settled()) {
            self.requests.remove(&request);
        }
    }

    /// Hands the create's outcome to its caller; whether a session was received decides adoption.
    fn deliver(&mut self, outcome: Result<ProviderSession, AvatarError>) {
        let Some(caller) = self.caller.take() else {
            // A session is delivered only to a caller known to be waiting.
            if outcome.is_ok() {
                self.queued.push(Event::DeliveryRefused);
            }
            return;
        };
        let session = match outcome {
            Ok(session) => session,
            Err(error) => {
                let _ = caller.reply.send(Err(error));
                return;
            }
        };
        if caller.reply.send(Ok(session.sample_rate)).is_err() {
            self.queued.push(Event::DeliveryRefused);
            return;
        }
        *self.view.lock() = View {
            session_id: session.session_id.clone(),
            provider_session_id: session.provider_session_id.clone(),
        };
        if !session.terminable() {
            log::warn!(target: lk_agents_abi::LOG_TARGET, provider = self.opts.provider.as_str(), session_id = session.session_id.as_deref().unwrap_or_default();
                "avatar gateway create response had no provider_session_id or terminate_token; this session cannot be explicitly terminated and will bill until its provider idle timeout");
        }
        self.queued.push(Event::Adopted);
    }

    /// Launches one attempt of `request`; a create's first attempt takes the arriving caller.
    fn launch(&mut self, request: RequestId, kind: Kind, attempt: u32) {
        if kind == Kind::Create
            && let Some(caller) = self.arriving.take()
        {
            self.caller = Some(caller);
        }
        let exchange = match self.exchanges.get(&request) {
            Some(exchange) => Some(exchange.clone()),
            None => self.exchange(request, kind, attempt),
        };
        let Some(exchange) = exchange else { return };
        let (http, reporter) = (self.http.clone(), self.reporter.clone());
        let work =
            async move { kind.run(&http, (*exchange).clone(), request, attempt, &reporter).await };
        let closed = self.reporter.clone();
        let task = TokioSpawner::new(self.runtime.clone()).spawn_settled(work, move |outcome| async move {
            if outcome == TaskOutcome::Panicked {
                log::error!(target: lk_agents_abi::LOG_TARGET, request = request.0, attempt; "avatar gateway request attempt panicked");
            }
            closed.report(Event::AttemptClosed { request, attempt }).await;
        });
        self.tasks.insert((request, attempt), task);
    }

    /// The request's one exchange: a create's fresh idempotency key, and a signed bearer. A launch
    /// failure is reported like any other failed attempt.
    fn exchange(
        &mut self,
        request: RequestId,
        kind: Kind,
        attempt: u32,
    ) -> Option<Arc<HttpRequest>> {
        if kind == Kind::Create && self.pending_create.is_none() {
            return None;
        }
        let bearer =
            lk_agents_net::token::inference_bearer(&self.credentials.0, &self.credentials.1);
        let exchange = match (kind, bearer) {
            (_, Err(error)) => {
                let failure = AttemptFailure::Connection(error.message);
                self.queued.push(Event::Failed { request, attempt, failure });
                self.queued.push(Event::AttemptClosed { request, attempt });
                return None;
            }
            (Kind::Create, Ok(bearer)) => {
                let room = self.pending_create.as_ref()?;
                let key: [u8; 16] = rand::random();
                let key: String = key.iter().map(|b| format!("{b:02x}")).collect();
                create_request(&self.opts, room, &self.cx, &bearer, &key)
            }
            (Kind::Terminate, Ok(bearer)) => {
                let session = &self.terminating[&request];
                terminate_request(&self.opts, session, &self.cx, &bearer)?
            }
        };
        let exchange = Arc::new(exchange);
        self.exchanges.insert(request, exchange.clone());
        Some(exchange)
    }
}

impl Kind {
    async fn run(
        self,
        http: &GatewayHttp,
        exchange: HttpRequest,
        request: RequestId,
        attempt: u32,
        reporter: &Reporter<Event>,
    ) {
        let sent = |sent| match sent {
            Sent::Connected => Event::Connected { request, attempt },
            Sent::RequestWritten => Event::RequestWritten { request, attempt },
        };
        let answer = match http.send(exchange, reporter, sent).await {
            Ok(response) => read_body(response, ANSWER_LIMIT).await,
            Err(failure) => Err(failure),
        };
        let event = match (self, answer) {
            (Kind::Create, Ok(body)) => match decode_created(&body) {
                Ok(session) => Event::Created { request, attempt, session },
                Err(failure) => Event::Failed { request, attempt, failure },
            },
            (Kind::Terminate, Ok(_)) => Event::Terminated { request, attempt },
            (_, Err(failure)) => Event::Failed { request, attempt, failure },
        };
        reporter.report(event).await;
    }
}

#[cfg(test)]
#[path = "gateway_tests.rs"]
mod tests;
