//! The gateway avatar: one [`AvatarSessionLifecycle`] per instance, stepped by a driver task that
//! executes its effects. Host calls and transport observations reach the driver as stamped
//! inputs, applied in order with every deadline due before them. The host's `session_id` and
//! `provider_session_id` are views the driver updates from the core's effects.
//!
//! The driver is its own owner: it ends once the instance is dropped (which closes it) and the
//! core owes nothing, so a session created after its caller left is still terminated.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::avatar::lifecycle::{
    AvatarRequestId, AvatarSessionEffect, AvatarSessionEvent, AvatarSessionLifecycle,
    CreateFailure, ProviderSession,
};
use crate::avatar::{AvatarOptions, AvatarRoom, create_request, decode_created, terminate_request};
use crate::headers::Headers;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::inference::avatar_session::{AvatarError, InferenceAvatar};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::{ApiConnectOptions, Clock, Closed, Extra, Ts};
use lk_agents_net::attempts::RequestPolicy;
use lk_agents_net::transport::HttpRequest;
use tokio::sync::{mpsc, oneshot};

use lk_agents_abi::{OwnedTask, SpawnerExt, TaskOutcome};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::http::{GatewayHttp, Reporter, Sent, Stamped, read_body};
use lk_agents_net::task::{TaskSet, TokioSpawner};

const LOG: &str = "livekit.agents";

/// Transport observations attempts may report ahead of the driver.
const OBSERVATION_QUEUE: usize = 32;

/// A gateway answer larger than this is not an avatar session.
const ANSWER_LIMIT: usize = 64 * 1024;

static NEXT_AVATAR: AtomicU64 = AtomicU64::new(1);

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
        headers: Headers,
        reply: oneshot::Sender<Result<u32, AvatarError>>,
    },
    /// The create call `caller` stopped waiting.
    Cancel {
        caller: u64,
    },
    Close {
        headers: Headers,
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
        policy: RequestPolicy,
        http: Arc<GatewayHttp>,
        runtime: &tokio::runtime::Handle,
        clock: Arc<dyn Clock>,
    ) -> Self {
        let opts = Arc::new(opts);
        let view = Arc::new(Mutex::new(View::default()));
        let (inputs, rx) = mpsc::unbounded_channel();
        let (tx, observed) = mpsc::channel(OBSERVATION_QUEUE);
        let avatar = ResourceId(NEXT_AVATAR.fetch_add(1, Ordering::Relaxed));
        let driver = Driver {
            core: AvatarSessionLifecycle::new(avatar, policy),
            opts: opts.clone(),
            credentials,
            http,
            runtime: runtime.clone(),
            reporter: Reporter { tx, clock: clock.clone() },
            view: view.clone(),
            now: clock.now(),
            caller: None,
            arriving: None,
            pending_create: None,
            headers: Headers::new(),
            exchanges: BTreeMap::new(),
            attempts: BTreeMap::new(),
            closers: Vec::new(),
            cleanup_incomplete: false,
            queued: Vec::new(),
        };
        // Self-owned: see the module docs.
        let tasks = Arc::new(TaskSet::default());
        tasks.spawn(
            &TokioSpawner::new(runtime.clone()),
            "avatar gateway driver",
            driver.run(rx, observed),
        );
        GatewayAvatar { opts, view, inputs, callers: AtomicU64::new(1), _driver: tasks }
    }

    pub fn provider(&self) -> &str {
        &self.opts.provider
    }

    /// Creates the provider session and returns the sample rate the avatar takes.
    pub async fn create(&self, room: AvatarRoom, headers: Headers) -> Result<u32, AvatarError> {
        let caller = self.callers.fetch_add(1, Ordering::Relaxed);
        let (reply, answer) = oneshot::channel();
        self.send(Input::Create { caller, room, headers, reply });
        let mut waiting = CancelOnDrop { inputs: &self.inputs, caller, armed: true };
        let outcome = answer.await;
        waiting.armed = false;
        outcome.unwrap_or_else(|_| Err(AvatarError::Closed))
    }

    /// Stops creating and terminates every held session; `TimedOut` when an attempt missed its
    /// cleanup deadline (it was reported and the runtime still owns it).
    pub async fn close(&self, headers: Headers) -> Closed {
        let (reply, settled) = oneshot::channel();
        self.send(Input::Close { headers, reply });
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
        let unavailable = |e: String| AvatarError::from(ApiError::new(e).with_retryable(false));
        conn_options.validate().map_err(|e| unavailable(e.to_string()))?;
        // Signing then cannot fail when an attempt launches.
        lk_agents_net::token::inference_bearer(&api_key, &api_secret)?;
        let opts = AvatarOptions::new(
            &model,
            avatar_identity,
            avatar_name,
            extra_kwargs.unwrap_or_default(),
            base_url,
        )?;
        let runtime = (env.runtime)().map_err(unavailable)?;
        let http = (env.http)().map_err(unavailable)?;
        let policy = RequestPolicy::from_options(&conn_options);
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
        self.view.lock().unwrap_or_else(|p| p.into_inner()).session_id.clone()
    }

    fn provider_session_id(&self) -> Option<String> {
        self.view.lock().unwrap_or_else(|p| p.into_inner()).provider_session_id.clone()
    }

    async fn create(
        &self,
        room_name: String,
        room_sid: String,
        livekit_url: Option<String>,
        agent_identity: String,
        headers: Vec<(String, String)>,
    ) -> Result<u32, AvatarError> {
        let livekit_url =
            livekit_url.filter(|url| !url.is_empty()).ok_or(AvatarError::MissingUrl)?;
        let room = AvatarRoom { livekit_url, room_name, room_sid, agent_identity };
        GatewayAvatar::create(self, room, headers.into_iter().collect()).await
    }

    async fn aclose(&self, headers: Vec<(String, String)>) {
        // Decision 14: a missed cleanup deadline was logged and reported; close returns.
        let _ = self.close(headers.into_iter().collect()).await;
    }
}

/// The create call waiting for its outcome.
struct Caller {
    id: u64,
    reply: oneshot::Sender<Result<u32, AvatarError>>,
}

struct Driver {
    core: AvatarSessionLifecycle,
    opts: Arc<AvatarOptions>,
    credentials: (String, String),
    http: Arc<GatewayHttp>,
    runtime: tokio::runtime::Handle,
    reporter: Reporter<AvatarSessionEvent>,
    view: Arc<Mutex<View>>,
    /// The instant the core was last stepped at; supplied time never goes back.
    now: Ts,
    caller: Option<Caller>,
    /// A create call the core has yet to accept or refuse.
    arriving: Option<Caller>,
    /// The accepted create's room, read when its first attempt launches.
    pending_create: Option<AvatarRoom>,
    /// The latest identification headers a host call supplied.
    headers: Headers,
    /// Each request's exchange, built once: every attempt sends the same operation.
    exchanges: BTreeMap<AvatarRequestId, Arc<HttpRequest>>,
    attempts: BTreeMap<(AvatarRequestId, u32), OwnedTask>,
    closers: Vec<oneshot::Sender<Closed>>,
    cleanup_incomplete: bool,
    /// Events the driver's own effect execution produced, stepped next.
    queued: Vec<AvatarSessionEvent>,
}

impl Driver {
    async fn run(
        mut self,
        mut inputs: mpsc::UnboundedReceiver<Input>,
        mut observed: mpsc::Receiver<Stamped<AvatarSessionEvent>>,
    ) {
        let clock = self.reporter.clock.clone();
        let mut dropped = false;
        while !(dropped && self.core.is_settled()) {
            let wake = self.core.wake_at();
            tokio::select! {
                biased;
                Some(observation) = observed.recv() => self.step(observation.event, observation.at),
                input = inputs.recv(), if !dropped => match input {
                    Some(input) => self.input(input, clock.now()),
                    None => {
                        // The instance is gone: closing is the only cleanup left to run.
                        dropped = true;
                        self.step(AvatarSessionEvent::Close, clock.now());
                    }
                },
                () = async { clock.wait_until(wake.unwrap_or(Ts::from_nanos(u64::MAX))).await }, if wake.is_some() => {
                    self.step(AvatarSessionEvent::Wake, wake.unwrap_or(self.now));
                }
            }
        }
    }

    fn input(&mut self, input: Input, at: Ts) {
        match input {
            Input::Create { caller, room, headers, reply } => {
                self.headers = headers;
                self.pending_create = Some(room);
                self.arriving = Some(Caller { id: caller, reply });
                self.step(AvatarSessionEvent::Create, at);
                self.pending_create = None;
            }
            Input::Cancel { caller } => {
                if self.caller.as_ref().is_some_and(|c| c.id == caller) {
                    self.caller = None;
                    self.step(AvatarSessionEvent::Cancel, at);
                }
            }
            Input::Close { headers, reply } => {
                self.headers = headers;
                self.closers.push(reply);
                self.step(AvatarSessionEvent::Close, at);
            }
        }
    }

    /// Applies `event` at `at`, after every deadline due by then, then the events its effects
    /// produced.
    fn step(&mut self, event: AvatarSessionEvent, at: Ts) {
        let at = at.max(self.now);
        while let Some(due) = self.core.wake_at().filter(|due| *due <= at) {
            self.now = due.max(self.now);
            let woken = self.core.handle(AvatarSessionEvent::Wake, self.now);
            if woken.is_empty() && self.core.wake_at() == Some(due) {
                break;
            }
            self.apply(woken);
        }
        self.now = at;
        if let AvatarSessionEvent::AttemptClosed { request, attempt } = &event {
            self.attempts.remove(&(*request, *attempt));
        }
        let effects = self.core.handle(event, at);
        self.apply(effects);
        while !self.queued.is_empty() {
            for event in std::mem::take(&mut self.queued) {
                let effects = self.core.handle(event, self.now);
                self.apply(effects);
            }
        }
        self.exchanges.retain(|request, _| {
            self.attempts.keys().any(|(r, _)| r == request) || self.core.tracks(*request)
        });
        if self.core.is_settled() {
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

    fn apply(&mut self, effects: Vec<AvatarSessionEffect>) {
        let opts = self.opts.clone();
        let provider = opts.provider.as_str();
        for effect in effects {
            match effect {
                AvatarSessionEffect::CreateRefused => {
                    if let Some(caller) = self.arriving.take() {
                        let _ = caller.reply.send(Err(AvatarError::AlreadyStarted));
                    }
                }
                AvatarSessionEffect::AvatarSessionCreate { request, attempt, .. } => {
                    // The accepted caller now waits for this operation.
                    if let Some(caller) = self.arriving.take() {
                        self.caller = Some(caller);
                    }
                    let exchange = match self.exchanges.get(&request) {
                        Some(exchange) => Some(exchange.clone()),
                        None => self.create_exchange(request, attempt),
                    };
                    if let Some(exchange) = exchange {
                        self.launch(request, attempt, exchange, ExchangeKind::Create);
                    }
                }
                AvatarSessionEffect::AvatarSessionTerminate {
                    request, attempt, session, ..
                } => {
                    let exchange = match self.exchanges.get(&request) {
                        Some(exchange) => Some(exchange.clone()),
                        None => self.terminate_exchange(request, attempt, &session),
                    };
                    if let Some(exchange) = exchange {
                        self.launch(request, attempt, exchange, ExchangeKind::Terminate);
                    }
                }
                AvatarSessionEffect::AvatarRequestCancel { request, attempt } => {
                    if let Some(task) = self.attempts.get_mut(&(request, attempt)) {
                        task.cancel();
                    }
                }
                AvatarSessionEffect::AttemptFailed { attempt, error, .. } => {
                    log::warn!(target: LOG, provider, attempt, error = error.message.as_str(); "avatar gateway request failed, retrying");
                }
                AvatarSessionEffect::CreateFinished(outcome) => self.deliver(outcome),
                AvatarSessionEffect::Retiring { session, reason } => {
                    log::info!(target: LOG, provider, session_id = session.session_id.as_deref().unwrap_or_default(), reason = format!("{reason:?}").as_str();
                        "terminating an avatar session no caller holds");
                }
                AvatarSessionEffect::Unterminable(session) => {
                    log::warn!(target: LOG, provider, session_id = session.session_id.as_deref().unwrap_or_default();
                        "avatar session has no provider_session_id or terminate_token; it cannot be explicitly terminated and will bill until its provider idle timeout");
                }
                AvatarSessionEffect::SessionTerminated(session) => {
                    let mut view = self.view.lock().unwrap_or_else(|p| p.into_inner());
                    if view.provider_session_id == session.provider_session_id {
                        view.provider_session_id = None;
                    }
                }
                AvatarSessionEffect::TerminateFailed { session, error } => {
                    log::warn!(target: LOG, provider, provider_session_id = session.provider_session_id.as_deref().unwrap_or_default(), error = error.message.as_str();
                        "failed to terminate inference avatar session; it will keep billing until its provider idle timeout unless aclose() is called again");
                }
                AvatarSessionEffect::CleanupIncomplete { request, attempt } => {
                    self.cleanup_incomplete = true;
                    log::error!(target: LOG, provider, request = request.0, attempt;
                        "avatar gateway request did not settle by its cleanup deadline; a provider session it created may bill until its idle timeout");
                }
                AvatarSessionEffect::RetirementIncomplete { request } => {
                    self.cleanup_incomplete = true;
                    log::error!(target: LOG, provider, request = request.0;
                        "the gateway never answered an avatar create no caller took; a provider session it made may bill until its idle timeout");
                }
            }
        }
    }

    /// Hands the created session to its caller; whether it was received decides adoption.
    fn deliver(&mut self, outcome: Result<ProviderSession, CreateFailure>) {
        let Some(caller) = self.caller.take() else {
            // The core delivers only to a caller it was told is waiting.
            if outcome.is_ok() {
                self.queued.push(AvatarSessionEvent::DeliveryRefused);
            }
            return;
        };
        match outcome {
            Ok(session) => {
                if caller.reply.send(Ok(session.sample_rate)).is_ok() {
                    *self.view.lock().unwrap_or_else(|p| p.into_inner()) = View {
                        session_id: session.session_id.clone(),
                        provider_session_id: session.provider_session_id.clone(),
                    };
                    if !session.terminable() {
                        log::warn!(target: LOG, provider = self.opts.provider.as_str(), session_id = session.session_id.as_deref().unwrap_or_default();
                            "avatar gateway create response had no provider_session_id or terminate_token; this session cannot be explicitly terminated and will bill until its provider idle timeout");
                    }
                    self.queued.push(AvatarSessionEvent::Adopted);
                } else {
                    self.queued.push(AvatarSessionEvent::DeliveryRefused);
                }
            }
            Err(CreateFailure::Api(error)) => {
                let _ = caller.reply.send(Err(AvatarError::Api(ProviderError::Local(error))));
            }
            Err(CreateFailure::Closed) => {
                let _ = caller.reply.send(Err(AvatarError::Closed));
            }
        }
    }

    /// The operation's one exchange: a fresh idempotency key and a signed bearer.
    fn create_exchange(
        &mut self,
        request: AvatarRequestId,
        attempt: u32,
    ) -> Option<Arc<HttpRequest>> {
        let room = self.pending_create.as_ref()?;
        let key: [u8; 16] = rand::random();
        let key: String = key.iter().map(|b| format!("{b:02x}")).collect();
        let exchange = match lk_agents_net::token::inference_bearer(
            &self.credentials.0,
            &self.credentials.1,
        ) {
            Ok(bearer) => create_request(&self.opts, room, self.headers.clone(), &bearer, &key),
            Err(error) => {
                // A launch failure is reported like any other failed attempt.
                self.failed_launch(request, attempt, error);
                return None;
            }
        };
        let exchange = Arc::new(exchange);
        self.exchanges.insert(request, exchange.clone());
        Some(exchange)
    }

    fn terminate_exchange(
        &mut self,
        request: AvatarRequestId,
        attempt: u32,
        session: &ProviderSession,
    ) -> Option<Arc<HttpRequest>> {
        let bearer = match lk_agents_net::token::inference_bearer(
            &self.credentials.0,
            &self.credentials.1,
        ) {
            Ok(bearer) => bearer,
            Err(error) => {
                self.failed_launch(request, attempt, error);
                return None;
            }
        };
        let exchange =
            Arc::new(terminate_request(&self.opts, session, self.headers.clone(), &bearer)?);
        self.exchanges.insert(request, exchange.clone());
        Some(exchange)
    }

    fn failed_launch(&mut self, request: AvatarRequestId, attempt: u32, error: ApiError) {
        let failure = lk_agents_net::attempts::AttemptFailure::Connection(error.message);
        self.queued.push(AvatarSessionEvent::Failed { request, attempt, failure });
        self.queued.push(AvatarSessionEvent::AttemptClosed { request, attempt });
    }

    fn launch(
        &mut self,
        request: AvatarRequestId,
        attempt: u32,
        exchange: Arc<HttpRequest>,
        kind: ExchangeKind,
    ) {
        let (http, reporter) = (self.http.clone(), self.reporter.clone());
        let work =
            async move { kind.run(&http, (*exchange).clone(), request, attempt, &reporter).await };
        let closed = self.reporter.clone();
        let task = TokioSpawner::new(self.runtime.clone()).spawn_settled(work, move |outcome| async move {
            if outcome == TaskOutcome::Panicked {
                log::error!(target: LOG, request = request.0, attempt; "avatar gateway request attempt panicked");
            }
            closed.report(AvatarSessionEvent::AttemptClosed { request, attempt }).await;
        });
        self.attempts.insert((request, attempt), task);
    }
}

/// How an attempt reads its answer.
#[derive(Clone, Copy)]
enum ExchangeKind {
    Create,
    Terminate,
}

impl ExchangeKind {
    async fn run(
        self,
        http: &GatewayHttp,
        exchange: HttpRequest,
        request: AvatarRequestId,
        attempt: u32,
        reporter: &Reporter<AvatarSessionEvent>,
    ) {
        let sent = |sent| match sent {
            Sent::Connected => AvatarSessionEvent::Connected { request, attempt },
            Sent::RequestWritten => AvatarSessionEvent::RequestWritten { request, attempt },
        };
        let answer = match http.send(exchange, reporter, sent).await {
            Ok(response) => read_body(response, ANSWER_LIMIT).await,
            Err(failure) => Err(failure),
        };
        let event = match (self, answer) {
            (ExchangeKind::Create, Ok(body)) => match decode_created(&body) {
                Ok(session) => AvatarSessionEvent::Created { request, attempt, session },
                Err(failure) => AvatarSessionEvent::Failed { request, attempt, failure },
            },
            (ExchangeKind::Terminate, Ok(_)) => AvatarSessionEvent::Terminated { request, attempt },
            (_, Err(failure)) => AvatarSessionEvent::Failed { request, attempt, failure },
        };
        reporter.report(event).await;
    }
}
