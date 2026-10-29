//! The gateway LLM: each `chat()` is one [`LlmRequest`] core whose effects this module executes,
//! pulled by its consumer. The request is stepped only while it is pulled or closed, so nothing
//! runs ahead of demand; transport observations wait in a bounded queue, stamped with the instant
//! they were made, and are applied in that order with every deadline due before them.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::headers::{Headers, identification_headers};
use crate::llm::lifecycle::{
    LlmRequest, LlmRequestEffect, LlmRequestEvent, LlmRequestId, RequestPolicy,
};
use crate::llm::request::ChatArgs;
use crate::llm::{LlmOptions, models};
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::{
    ApiConnectOptions, ChatContext, ChatItem, ChatRequest, Clock, Closed, Extra, Llm, LlmChunk,
    LlmStream, StreamSource, ToolChoice, ToolSpec, Ts,
};
use lk_agents_core::llm::provider_format::PreResolved;
mod request;
use lk_agents_net::transport::HttpRequest;
use request::build_request;
use tokio::sync::mpsc;

use crate::llm::parse::WireChunk;
use futures_util::StreamExt;
use lk_agents_net::attempts::AttemptFailure;
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::http::{GatewayHttp, Reporter, Sent, Stamped, error_chain};
use lk_agents_net::sse::SseDecoder;
use lk_agents_net::task::{OwnedTask, TaskOutcome};

/// Transport observations an attempt may report ahead of the consumer before it stops reading.
const OBSERVATION_QUEUE: usize = 32;

static NEXT_LLM: AtomicU64 = AtomicU64::new(1);

/// One `inference.LLM` and its views per job identity: the options are shared, each view stamps
/// its own job's headers on its requests.
pub struct GatewayLlm {
    opts: Arc<Mutex<LlmOptions>>,
    http: Arc<GatewayHttp>,
    runtime: tokio::runtime::Handle,
    clock: Arc<dyn Clock>,
    identity: Option<Headers>,
    resource: ResourceId,
    requests: Arc<AtomicU64>,
    /// The host provider whose `error` events report the attempts this view retries.
    reporter: Option<Arc<dyn Llm>>,
    prewarm: Mutex<Option<OwnedTask>>,
}

impl GatewayLlm {
    pub fn new(
        opts: LlmOptions,
        http: Arc<GatewayHttp>,
        runtime: tokio::runtime::Handle,
        clock: Arc<dyn Clock>,
    ) -> Self {
        GatewayLlm {
            opts: Arc::new(Mutex::new(opts)),
            http,
            runtime,
            clock,
            identity: None,
            resource: ResourceId(NEXT_LLM.fetch_add(1, Ordering::Relaxed)),
            requests: Arc::new(AtomicU64::new(1)),
            reporter: None,
            prewarm: Mutex::new(None),
        }
    }

    /// The same client as a session of the job `identity` names holds it.
    pub fn view(&self, identity: Headers, reporter: Option<Arc<dyn Llm>>) -> Self {
        GatewayLlm {
            opts: self.opts.clone(),
            http: self.http.clone(),
            runtime: self.runtime.clone(),
            clock: self.clock.clone(),
            identity: Some(identity),
            resource: self.resource,
            requests: self.requests.clone(),
            reporter,
            prewarm: Mutex::new(None),
        }
    }

    pub fn options(&self) -> LlmOptions {
        self.opts.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Read by the next `chat()`.
    pub fn update_options(&self, update: impl FnOnce(&mut LlmOptions)) {
        update(&mut self.opts.lock().unwrap_or_else(|p| p.into_inner()));
    }

    /// Opens a completion. No I/O until the first pull, which accepts the request: its total
    /// limit runs from there.
    /// `identification` is the caller's job identity, read where it called; else this view's.
    pub fn chat(
        &self,
        chat_ctx: &ChatContext,
        tools: &[ToolSpec],
        args: ChatArgs,
        policy: RequestPolicy,
        identification: Option<&Headers>,
    ) -> Result<LlmChunks, ApiError> {
        let opts = self.options();
        // The host serialized every image onto its item already.
        let images = PreResolved::from_pairs(Vec::<(String, _)>::new());
        let identification = identification.or(self.identity.as_ref());
        let bearer = lk_agents_net::token::inference_bearer(&opts.api_key, &opts.api_secret)?;
        let request =
            build_request(&opts, identification, &images, chat_ctx, tools, args, &bearer)?;
        let id = LlmRequestId(self.requests.fetch_add(1, Ordering::Relaxed));
        let (tx, rx) = mpsc::channel(OBSERVATION_QUEUE);
        Ok(LlmChunks {
            start: Some(Start {
                id,
                llm: self.resource,
                policy,
                think_tags: models::think_tags(&opts.model),
            }),
            core: None,
            rx,
            reporter: Reporter { tx, clock: self.clock.clone() },
            exchange: Arc::new(Exchange { http: self.http.clone(), request }),
            runtime: self.runtime.clone(),
            attempts: BTreeMap::new(),
            ready: VecDeque::new(),
            outcome: None,
            ended: false,
            cleanup_incomplete: false,
            now: self.clock.now(),
            host: self.reporter.clone(),
        })
    }
}

/// `_core.InferenceLlm`: the generated adapter calls this.
impl lk_agents_abi::llm::inference::InferenceLlm for GatewayLlm {
    type Chunks = LlmChunks;
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        model: String,
        provider: Option<String>,
        base_url: String,
        api_key: String,
        api_secret: String,
        inference_class: Option<String>,
        extra_kwargs: Option<Extra>,
    ) -> Result<Self, ApiError> {
        let unavailable = |e: String| ApiError::new(e).with_retryable(false);
        let runtime = (env.runtime)().map_err(unavailable)?;
        let http = (env.http)().map_err(unavailable)?;
        let opts = LlmOptions {
            model,
            base_url,
            api_key,
            api_secret,
            provider,
            inference_class,
            extra_kwargs: extra_kwargs.unwrap_or_default(),
        };
        Ok(GatewayLlm::new(opts, http, runtime, env.clock.clone()))
    }

    fn identified(&self, headers: Vec<(String, String)>, reporter: Option<Arc<dyn Llm>>) -> Self {
        self.view(headers.into_iter().collect(), reporter)
    }

    fn model(&self) -> String {
        self.options().model
    }

    fn extra_kwargs(&self) -> Extra {
        self.options().extra_kwargs
    }

    fn update_options(&self, model: Option<String>, extra_kwargs: Option<Extra>) {
        self.update_options(|opts| {
            if let Some(model) = model {
                opts.model = model;
            }
            if let Some(extra) = extra_kwargs {
                opts.extra_kwargs = extra;
            }
        });
    }

    fn chat(
        &self,
        items: Vec<ChatItem>,
        tools: Vec<ToolSpec>,
        conn_options: ApiConnectOptions,
        parallel_tool_calls: Option<bool>,
        tool_choice: Option<ToolChoice>,
        response_format: Option<serde_json::Value>,
        extra_kwargs: Option<Extra>,
        headers: Option<Vec<(String, String)>>,
    ) -> Result<LlmChunks, ApiError> {
        conn_options.validate().map_err(|e| ApiError::new(e.to_string()).with_retryable(false))?;
        let args = ChatArgs {
            extra_kwargs,
            parallel_tool_calls,
            tool_choice,
            response_format: response_format.filter(|f| !f.is_null()),
            inference_class: None,
        };
        let identification: Option<Headers> = headers.map(|pairs| pairs.into_iter().collect());
        let policy = RequestPolicy::from_options(&conn_options);
        GatewayLlm::chat(
            self,
            &ChatContext::new(items),
            &tools,
            args,
            policy,
            identification.as_ref(),
        )
    }
}

impl Llm for GatewayLlm {
    fn failed(&self, error: ProviderError, recoverable: bool) {
        if let Some(reporter) = &self.reporter {
            reporter.failed(error, recoverable);
        }
    }

    fn label(&self) -> String {
        "livekit.agents.inference.llm.LLM".into()
    }

    fn model(&self) -> String {
        self.options().model
    }

    fn id(&self) -> lk_agents_abi::ProviderId {
        lk_agents_abi::ProviderId {
            label: self.label(),
            model: self.model(),
            provider: "livekit".into(),
        }
    }

    /// Opens a connection in the background so the first request finds it pooled.
    fn prewarm(&self) {
        let opts = self.options();
        let Ok(bearer) = lk_agents_net::token::inference_bearer(&opts.api_key, &opts.api_secret)
        else {
            return;
        };
        let mut headers = identification_headers(self.identity.as_ref(), None, None);
        headers.set("Authorization", format!("Bearer {bearer}"));
        let (http, url) =
            (self.http.clone(), format!("{}/models", opts.base_url.trim_end_matches('/')));
        let task = OwnedTask::spawn(
            &self.runtime,
            async move { http.warm(url, headers.into_vec()).await },
            |_| async {},
        );
        *self.prewarm.lock().unwrap_or_else(|p| p.into_inner()) = Some(task);
    }

    /// The voice session's generation retries its own LLM calls until it delegates them to the
    /// request core, so each call here is one attempt under the session's phase limits.
    fn chat<'a>(
        &'a self,
        req: &'a ChatRequest,
    ) -> lk_agents_abi::BoxFuture<'a, Result<LlmStream, ProviderError>> {
        Box::pin(async move {
            req.conn_options
                .validate()
                .map_err(|e| ApiError::new(e.to_string()).with_retryable(false))?;
            let policy =
                RequestPolicy { max_retry: 0, ..RequestPolicy::from_options(&req.conn_options) };
            let args = ChatArgs {
                extra_kwargs: req.extra.clone(),
                parallel_tool_calls: req.parallel_tool_calls,
                tool_choice: req.tool_choice.clone(),
                response_format: None,
                inference_class: req.inference_class.clone(),
            };
            Ok(Box::new(GatewayLlm::chat(self, &req.chat_ctx, &req.tools, args, policy, None)?)
                as LlmStream)
        })
    }
}

struct Exchange {
    http: Arc<GatewayHttp>,
    request: HttpRequest,
}

impl Exchange {
    /// Each attempt is its own gateway session: a job's session id is minted afresh per attempt.
    fn attempt(&self) -> HttpRequest {
        let mut request = self.request.clone();
        for (name, value) in &mut request.headers {
            if name.eq_ignore_ascii_case(crate::headers::HEADER_SESSION_ID) {
                *value = crate::ids::shortuuid("inference_");
            }
        }
        request
    }
}

/// What accepts the request at its first pull.
struct Start {
    id: LlmRequestId,
    llm: ResourceId,
    policy: RequestPolicy,
    think_tags: (&'static str, &'static str),
}

/// One request's chunks, in order, then its outcome.
pub struct LlmChunks {
    start: Option<Start>,
    /// The accepted request; `None` until the first pull.
    core: Option<LlmRequest>,
    rx: mpsc::Receiver<Stamped<LlmRequestEvent>>,
    reporter: Reporter<LlmRequestEvent>,
    exchange: Arc<Exchange>,
    runtime: tokio::runtime::Handle,
    attempts: BTreeMap<u32, OwnedTask>,
    ready: VecDeque<LlmChunk>,
    outcome: Option<Result<(), ApiError>>,
    ended: bool,
    cleanup_incomplete: bool,
    /// The instant the core was last stepped at; supplied time never goes back.
    now: Ts,
    host: Option<Arc<dyn Llm>>,
}

impl LlmChunks {
    /// The next chunk, `None` at the end. A failure ends the stream. Dropping the returned future
    /// loses nothing: observations stay queued and chunks stay buffered.
    pub async fn next(&mut self) -> Result<Option<LlmChunk>, ApiError> {
        if let Some(chunk) = self.ready.pop_front() {
            return Ok(Some(chunk));
        }
        if self.ended {
            return Ok(None);
        }
        if let Some(start) = self.start.take() {
            let now = self.reporter.clock.now().max(self.now);
            let (core, effects) =
                LlmRequest::start(start.id, start.llm, start.policy, start.think_tags, now);
            self.core = Some(core);
            self.now = now;
            self.apply(effects);
        }
        let demand = Demand::start(self);
        demand.0.wait().await
    }

    /// Stops the request and waits, until `deadline`, for every attempt to release its
    /// transport. A missed deadline is reported, never waited on further.
    pub async fn close(mut self, deadline: Ts) -> Closed {
        let now = self.reporter.clock.now();
        self.step(LlmRequestEvent::Cancel, now);
        while self.core.as_ref().is_some_and(|core| !core.is_settled()) {
            let wake = self.wake_at().map_or(deadline, |at| at.min(deadline));
            tokio::select! {
                biased;
                Some(observed) = self.rx.recv() => self.step(observed.event, observed.at),
                () = self.reporter.clock.wait_until(wake) => {
                    if wake >= deadline {
                        log::error!(target: "livekit.agents", "LLM request cleanup missed its deadline");
                        return Closed::TimedOut;
                    }
                    self.step(LlmRequestEvent::Wake, wake);
                }
            }
        }
        if self.cleanup_incomplete { Closed::TimedOut } else { Closed::Clean }
    }

    async fn wait(&mut self) -> Result<Option<LlmChunk>, ApiError> {
        loop {
            if let Some(chunk) = self.ready.pop_front() {
                return Ok(Some(chunk));
            }
            if let Some(outcome) = self.outcome.take() {
                self.ended = true;
                return outcome.map(|()| None);
            }
            let wake = self.wake_at();
            let clock = self.reporter.clock.clone();
            tokio::select! {
                biased;
                Some(observed) = self.rx.recv() => self.step(observed.event, observed.at),
                () = async { clock.wait_until(wake.unwrap_or(Ts::from_nanos(u64::MAX))).await }, if wake.is_some() => {
                    self.step(LlmRequestEvent::Wake, wake.unwrap_or(self.now));
                }
            }
        }
    }

    fn wake_at(&self) -> Option<Ts> {
        self.core.as_ref().and_then(LlmRequest::wake_at)
    }

    /// Applies `event` at `at`, after every deadline due by then. Before the first pull there is
    /// no request to apply it to.
    fn step(&mut self, event: LlmRequestEvent, at: Ts) {
        let Some(core) = self.core.as_mut() else { return };
        let at = at.max(self.now);
        let mut effects = Vec::new();
        while let Some(due) = core.wake_at().filter(|due| *due <= at) {
            self.now = due.max(self.now);
            let woken = core.handle(LlmRequestEvent::Wake, self.now);
            if woken.is_empty() && core.wake_at() == Some(due) {
                break;
            }
            effects.extend(woken);
        }
        self.now = at;
        if let LlmRequestEvent::AttemptClosed { attempt } = &event {
            self.attempts.remove(attempt);
        }
        effects.extend(core.handle(event, at));
        self.apply(effects);
    }

    fn apply(&mut self, effects: Vec<LlmRequestEffect>) {
        for effect in effects {
            match effect {
                LlmRequestEffect::LlmRequestStart { attempt, .. } => {
                    let (exchange, reporter) = (self.exchange.clone(), self.reporter.clone());
                    let closed = self.reporter.clone();
                    let work = async move {
                        send_attempt(&exchange.http, exchange.attempt(), attempt, reporter).await
                    };
                    let task = OwnedTask::spawn(&self.runtime, work, move |outcome| async move {
                        if outcome == TaskOutcome::Panicked {
                            log::error!(target: "livekit.agents", attempt; "LLM request attempt panicked");
                        }
                        let _ = closed
                            .tx
                            .send(Stamped {
                                at: closed.clock.now(),
                                event: LlmRequestEvent::AttemptClosed { attempt },
                            })
                            .await;
                    });
                    self.attempts.insert(attempt, task);
                }
                LlmRequestEffect::LlmRequestCancel { attempt, .. } => {
                    if let Some(task) = self.attempts.get_mut(&attempt) {
                        task.cancel();
                    }
                }
                LlmRequestEffect::Deliver(chunk) => self.ready.push_back(chunk),
                LlmRequestEffect::AttemptFailed { attempt, error } => {
                    log::warn!(target: "livekit.agents", attempt, error = error.message.as_str(); "LLM request attempt failed, retrying");
                    if let Some(host) = &self.host {
                        host.failed(ProviderError::Local(error), true);
                    }
                }
                LlmRequestEffect::Finished(outcome) => self.outcome = Some(outcome),
                LlmRequestEffect::CleanupIncomplete { request_id, attempt } => {
                    self.cleanup_incomplete = true;
                    log::error!(target: "livekit.agents", request = request_id.0, attempt; "LLM request attempt did not release its transport in time");
                }
            }
        }
    }
}

/// The consumer waits for a chunk from creation to drop: the read limit runs only meanwhile.
struct Demand<'a>(&'a mut LlmChunks);

impl<'a> Demand<'a> {
    fn start(chunks: &'a mut LlmChunks) -> Self {
        let now = chunks.reporter.clock.now();
        chunks.step(LlmRequestEvent::DemandStarted, now);
        Demand(chunks)
    }
}

impl Drop for Demand<'_> {
    fn drop(&mut self) {
        let now = self.0.reporter.clock.now();
        self.0.step(LlmRequestEvent::DemandEnded, now);
    }
}

impl StreamSource<LlmChunk> for LlmChunks {
    fn cancel_safe(&self) -> bool {
        true
    }

    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<LlmChunk>, ProviderError>> {
        Box::pin(async move { LlmChunks::next(self).await.map_err(ProviderError::Local) })
    }

    fn close(self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { LlmChunks::close(*self, deadline).await })
    }
}

/// One LLM request attempt: the completion's event stream decoded into chunk events.
async fn send_attempt(
    http: &GatewayHttp,
    request: HttpRequest,
    attempt: u32,
    reporter: Reporter<LlmRequestEvent>,
) {
    let sent = |sent| match sent {
        Sent::Connected => LlmRequestEvent::Connected { attempt },
        Sent::RequestWritten => LlmRequestEvent::RequestWritten { attempt },
    };
    let failure = match http.send(request, &reporter, sent).await {
        Err(failure) => Some(failure),
        Ok(response) => read_events(response, attempt, &reporter).await,
    };
    if let Some(failure) = failure {
        reporter.report(LlmRequestEvent::Failed { attempt, failure }).await;
    }
}

/// Decodes the event stream into chunk events; `None` once the response ended cleanly.
async fn read_events(
    response: reqwest::Response,
    attempt: u32,
    reporter: &Reporter<LlmRequestEvent>,
) -> Option<AttemptFailure> {
    let mut decoder = SseDecoder::new();
    let mut bytes = response.bytes_stream();
    let mut events = Vec::new();
    loop {
        let Some(chunk) = bytes.next().await else {
            return match decoder.finish() {
                Ok(()) => {
                    reporter.report(LlmRequestEvent::Ended { attempt }).await;
                    None
                }
                Err(err) => Some(AttemptFailure::Malformed(err.to_string())),
            };
        };
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(err) => return Some(AttemptFailure::Connection(error_chain(&err))),
        };
        if let Err(err) = decoder.feed(&chunk, &mut events) {
            return Some(AttemptFailure::Malformed(err.to_string()));
        }
        if events.is_empty() {
            reporter.report(LlmRequestEvent::Progress { attempt }).await;
        }
        for event in events.drain(..) {
            if event.data == "[DONE]" {
                reporter.report(LlmRequestEvent::Ended { attempt }).await;
                return None;
            }
            match serde_json::from_str::<WireChunk>(&event.data) {
                Ok(chunk) => reporter.report(LlmRequestEvent::Chunk { attempt, chunk }).await,
                Err(err) => {
                    return Some(AttemptFailure::Malformed(format!("undecodable chunk: {err}")));
                }
            }
        }
    }
}
