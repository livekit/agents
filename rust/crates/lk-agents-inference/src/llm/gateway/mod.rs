use crate::telemetry;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

use crate::headers::{Headers, gateway_headers};
use crate::llm::request::ChatArgs;
use crate::llm::{Auth, BodyEncoding, LlmOptions, models};
use lk_agents_abi::errors::{ApiError, Invalid, ProviderError};
use lk_agents_abi::llm::fallback::{HostChatContext, HostTools};
use lk_agents_abi::llm::inference::{ProviderEndpoint, ResponseFormat};
use lk_agents_abi::types::provider::{NativeLabel, ProviderEvents};
use lk_agents_abi::{
    ApiConnectOptions, ChatContext, ChatRequest, Clock, Closed, Extra, Llm, LlmChunk, LlmStream,
    StreamOutput, ToolChoice, ToolSpec, Ts,
};
use lk_agents_core::llm::provider_format::PreResolved;
mod request;
use lk_agents_net::transport::HttpRequest;
use request::build_request;
use tokio::sync::mpsc;

use crate::llm::model_text::SaidText;
use crate::llm::parse::{ChoiceAccumulator, WireChunk, usage_chunk};
use futures_util::StreamExt;
use lk_agents_abi::{OwnedTask, Spawner, SpawnerExt, TaskOutcome};
pub use lk_agents_core::request::Policy;
use lk_agents_core::request::{AttemptFailure, Request, Step};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::http::{GatewayHttp, Reporter, Sent, Stamped, error_chain};
use lk_agents_net::sse::SseDecoder;

// The request is stepped only while it is pulled or closed, so nothing runs ahead of demand.
// The read limit runs only while the consumer waits for the next chunk: downstream backpressure
// is not remote inactivity. A failure is retried only while nothing visible was delivered.

/// Transport observations an attempt may report ahead of the consumer before it stops reading.
const OBSERVATION_QUEUE: usize = 32;

/// One `inference.LLM`. Every job it is handed to shares it; each request carries the identity
/// of the job its telemetry context runs for.
pub struct GatewayLlm {
    /// The name logs and metrics give it: its host class's when its host built it, else
    /// `opts.label`.
    label: String,
    opts: Arc<Mutex<LlmOptions>>,
    http: Arc<GatewayHttp>,
    spawner: Arc<dyn Spawner>,
    clock: Arc<dyn Clock>,
    requests: Arc<AtomicU64>,
    /// The host object it was built as, which hears the attempts it retries fail.
    events: Option<Arc<dyn ProviderEvents>>,
    prewarm: Mutex<Option<OwnedTask>>,
}

impl GatewayLlm {
    pub fn new(
        opts: LlmOptions,
        http: Arc<GatewayHttp>,
        spawner: Arc<dyn Spawner>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        GatewayLlm {
            label: opts.label.clone(),
            opts: Arc::new(Mutex::new(opts)),
            http,
            spawner,
            clock,
            requests: Arc::new(AtomicU64::new(1)),
            events: None,
            prewarm: Mutex::new(None),
        }
    }

    pub fn options(&self) -> LlmOptions {
        self.opts.lock().clone()
    }

    /// Read by the next `chat()`.
    pub fn update_options(&self, update: impl FnOnce(&mut LlmOptions)) {
        update(&mut self.opts.lock());
    }

    /// Opens a completion for the job `cx` runs for. No I/O until the first pull, which accepts
    /// the request: its total limit runs from there.
    pub fn chat(
        &self,
        cx: &telemetry::Cx,
        chat_ctx: &ChatContext,
        tools: &[ToolSpec],
        args: ChatArgs,
        policy: Policy,
    ) -> Result<LlmChunks, ApiError> {
        let opts = self.options();
        // The host serialized every image onto its item already.
        let images = PreResolved::from_pairs(Vec::<(String, _)>::new());
        let bearer = opts.auth.bearer()?;
        let request = build_request(&opts, cx, &images, chat_ctx, tools, args, &bearer)?;
        let (tx, rx) = mpsc::channel(OBSERVATION_QUEUE);
        Ok(LlmChunks {
            id: self.requests.fetch_add(1, Ordering::Relaxed),
            model_syntax: models::model_syntax(&opts.model),
            phase: Phase::Pending(policy),
            rx,
            reporter: Reporter { tx, clock: self.clock.clone() },
            exchange: Arc::new(Exchange { http: self.http.clone(), request }),
            spawner: self.spawner.clone(),
            tasks: BTreeMap::new(),
            ready: VecDeque::new(),
            outcome: None,
            ended: false,
            cleanup_incomplete: false,
            now: self.clock.now(),
            host: self.events.clone(),
        })
    }
}

/// `_core.InferenceLlm`: the generated adapter calls this.
impl lk_agents_abi::llm::inference::InferenceLlm for GatewayLlm {
    type Chunks = LlmChunks;
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        label: String,
        events: Arc<dyn ProviderEvents>,
        model: String,
        provider: Option<String>,
        base_url: Option<String>,
        api_key: Option<String>,
        api_secret: Option<String>,
        inference_class: Option<String>,
        extra_kwargs: Option<Extra>,
        endpoint: Option<ProviderEndpoint>,
    ) -> Result<Self, Invalid> {
        use lk_agents_abi::inference::InferenceCredentials as _;
        let unavailable = |e: lk_agents_net::env::ProcessError| Invalid(e.to_string());
        let spawner = (env.spawner)().map_err(unavailable)?;
        let http = (env.http)().map_err(unavailable)?;
        let (base_url, auth, strict_tools, body) = match endpoint {
            Some(e) => (
                e.base_url,
                Auth::ProviderKey(e.api_key),
                e.strict_tool_schema,
                BodyEncoding { msgpack: e.msgpack_body, gzip: e.gzip_body },
            ),
            None => {
                let c = crate::credentials::Credentials::inference_credentials(
                    base_url, api_key, api_secret,
                )?;
                let auth = Auth::Gateway { api_key: c.api_key, api_secret: c.api_secret };
                (c.base_url, auth, true, BodyEncoding::default())
            }
        };
        let opts = LlmOptions {
            model,
            base_url,
            auth,
            provider,
            inference_class,
            strict_tools,
            body,
            label: NativeLabel::InferenceLlm.as_str().into(),
            extra_kwargs: extra_kwargs.unwrap_or_default(),
        };
        let llm = GatewayLlm::new(opts, http, spawner, env.clock.clone());
        Ok(GatewayLlm { label, events: Some(events), ..llm })
    }

    fn model(&self) -> String {
        self.options().model
    }

    fn extra_kwargs(&self) -> Extra {
        self.options().extra_kwargs
    }

    fn inference_class(&self) -> Option<String> {
        self.options().inference_class
    }

    fn update_options(
        &self,
        model: Option<String>,
        extra_kwargs: Option<Extra>,
        endpoint: Option<ProviderEndpoint>,
    ) {
        self.update_options(|opts| {
            if let Some(model) = model {
                opts.model = model;
            }
            if let Some(extra) = extra_kwargs {
                opts.extra_kwargs = extra;
            }
            if let Some(e) = endpoint {
                opts.base_url = e.base_url;
                opts.auth = Auth::ProviderKey(e.api_key);
                opts.strict_tools = e.strict_tool_schema;
                opts.body = BodyEncoding { msgpack: e.msgpack_body, gzip: e.gzip_body };
            }
        });
    }

    fn chat(
        &self,
        cx: telemetry::Cx,
        chat_ctx: HostChatContext,
        tools: Option<HostTools>,
        conn_options: Option<ApiConnectOptions>,
        parallel_tool_calls: Option<bool>,
        tool_choice: Option<ToolChoice>,
        response_format: Option<ResponseFormat>,
        inference_class: Option<String>,
        extra_kwargs: Option<Extra>,
    ) -> Result<LlmChunks, ApiError> {
        let conn_options = conn_options.unwrap_or_default();
        conn_options.validate().map_err(|e| ApiError::new(e.to_string()).with_retryable(false))?;
        let args = ChatArgs {
            extra_kwargs,
            parallel_tool_calls,
            tool_choice,
            response_format: response_format.map(|f| f.param).filter(|f| !f.is_null()),
            inference_class,
        };
        let specs = tools.map(|t| t.specs).unwrap_or_default();
        let policy = Policy::from_options(&conn_options);
        GatewayLlm::chat(self, &cx, &ChatContext::new(chat_ctx.items), &specs, args, policy)
    }
}

impl Llm for GatewayLlm {
    fn failed(&self, error: ProviderError, recoverable: bool) {
        if let Some(events) = &self.events {
            events.failed(error, recoverable);
        }
    }

    fn label(&self) -> String {
        self.label.clone()
    }

    fn model(&self) -> String {
        self.options().model
    }

    fn id(&self) -> lk_agents_abi::ProviderId {
        lk_agents_abi::ProviderId {
            label: self.label(),
            model: self.model(),
            provider: match self.options() {
                LlmOptions { auth: Auth::Gateway { .. }, .. } => "livekit".into(),
                // the provider's host, as the Python plugins name it
                LlmOptions { base_url, .. } => url::Url::parse(&base_url)
                    .ok()
                    .and_then(|url| url.host_str().map(str::to_owned))
                    .unwrap_or(base_url),
            },
        }
    }

    /// Opens a connection in the background so the first request finds it pooled.
    fn prewarm(&self, cx: &telemetry::Cx) {
        let opts = self.options();
        let Ok(bearer) = opts.auth.bearer() else {
            return;
        };
        let headers = match opts.auth {
            Auth::Gateway { .. } => gateway_headers(cx, None, None, &bearer),
            Auth::ProviderKey(_) => {
                let mut headers = Headers::new();
                headers.insert("Authorization", format!("Bearer {bearer}"));
                headers
            }
        };
        let (http, url) =
            (self.http.clone(), format!("{}/models", opts.base_url.trim_end_matches('/')));
        let task = self.spawner.spawn(async move { http.warm(url, headers).await });
        *self.prewarm.lock() = Some(task);
    }

    /// The voice session's generation retries its own LLM calls until it delegates them to the
    /// request core, so each call here is one attempt under the session's phase limits.
    fn chat<'a>(
        &'a self,
        cx: &telemetry::Cx,
        req: &'a ChatRequest,
    ) -> lk_agents_abi::BoxFuture<'a, Result<LlmStream, ProviderError>> {
        let cx = cx.clone();
        Box::pin(async move {
            req.conn_options
                .validate()
                .map_err(|e| ApiError::new(e.to_string()).with_retryable(false))?;
            let policy = Policy::from_options(&req.conn_options);
            let args = ChatArgs {
                extra_kwargs: req.extra.clone(),
                parallel_tool_calls: req.parallel_tool_calls,
                tool_choice: req.tool_choice.clone(),
                response_format: None,
                inference_class: req.inference_class.clone(),
            };
            Ok(Box::new(GatewayLlm::chat(self, &cx, &req.chat_ctx, &req.tools, args, policy)?)
                as LlmStream)
        })
    }
}

struct Exchange {
    http: Arc<GatewayHttp>,
    request: HttpRequest,
}

/// What the attempts report, and the consumer's own inputs. Every attempt-scoped event names its
/// attempt; one from an attempt that is no longer current only settles that attempt's cleanup.
#[derive(Debug)]
enum Event {
    /// A deadline from [`LlmChunks::wake_at`] is due.
    Wake,
    /// The consumer started or stopped waiting for the next chunk.
    DemandStarted,
    DemandEnded,
    Connected {
        attempt: u32,
    },
    RequestWritten {
        attempt: u32,
    },
    /// Response bytes that completed no event.
    Progress {
        attempt: u32,
    },
    /// One decoded completion chunk.
    Chunk {
        attempt: u32,
        chunk: WireChunk,
    },
    /// The `[DONE]` sentinel.
    Ended {
        attempt: u32,
    },
    Failed {
        attempt: u32,
        failure: AttemptFailure,
    },
    /// The attempt's transport resources are released.
    AttemptClosed {
        attempt: u32,
    },
    /// The consumer gave up on the request.
    Cancel,
}

enum Phase {
    /// No I/O until the first pull, which accepts the request: its total limit runs from there.
    Pending(Policy),
    Accepted(Box<Accepted>),
}

struct Accepted {
    attempts: Request,
    /// The current attempt's response assembly; a retry starts over.
    choices: ChoiceAccumulator,
    /// What the attempt has said: once anything has, a failure is final.
    said: SaidText,
    /// Since when the consumer has been waiting, if it is.
    demand: Option<Ts>,
}

/// One request's chunks, in order, then its outcome.
pub struct LlmChunks {
    id: u64,
    model_syntax: lk_agents_core::lexer::ModelSyntax,
    phase: Phase,
    rx: mpsc::Receiver<Stamped<Event>>,
    reporter: Reporter<Event>,
    exchange: Arc<Exchange>,
    spawner: Arc<dyn Spawner>,
    tasks: BTreeMap<u32, OwnedTask>,
    ready: VecDeque<LlmChunk>,
    outcome: Option<Result<(), ApiError>>,
    ended: bool,
    cleanup_incomplete: bool,
    /// The instant the request was last stepped at; supplied time never goes back.
    now: Ts,
    /// The host object the client was built as, which hears each retried attempt fail.
    host: Option<Arc<dyn ProviderEvents>>,
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
        if let Phase::Pending(policy) = self.phase {
            let now = self.reporter.clock.now().max(self.now);
            self.accept(policy, now);
        }
        let demand = Demand::start(self);
        demand.0.wait().await
    }

    /// Stops the request and waits, until `deadline`, for every attempt to release its
    /// transport. A missed deadline is reported, never waited on further.
    pub async fn close(mut self, deadline: Ts) -> Closed {
        let now = self.reporter.clock.now();
        self.step(Event::Cancel, now);
        while matches!(&self.phase, Phase::Accepted(a) if !a.attempts.is_settled()) {
            let wake = self.wake_at().map_or(deadline, |at| at.min(deadline));
            tokio::select! {
                biased;
                Some(observed) = self.rx.recv() => self.step(observed.event, observed.at),
                () = self.reporter.clock.wait_until(wake) => {
                    if wake >= deadline {
                        log::error!(target: lk_agents_abi::LOG_TARGET, "LLM request cleanup missed its deadline");
                        return Closed::TimedOut;
                    }
                    self.step(Event::Wake, wake);
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
                    self.step(Event::Wake, wake.unwrap_or(self.now));
                }
            }
        }
    }

    /// Accepts the request at `now` and launches its first attempt.
    fn accept(&mut self, policy: Policy, now: Ts) {
        let mut steps = Vec::new();
        let attempts = Request::start(policy, now, &mut steps);
        self.phase = Phase::Accepted(Box::new(Accepted {
            attempts,
            choices: ChoiceAccumulator::new(),
            said: SaidText::new(self.model_syntax, None),
            demand: None,
        }));
        self.now = now;
        self.apply(steps);
    }

    /// The earliest instant an [`Event::Wake`] has work to do.
    fn wake_at(&self) -> Option<Ts> {
        match &self.phase {
            Phase::Pending(_) => None,
            Phase::Accepted(a) => a.attempts.wake_at(a.demand),
        }
    }

    /// Applies `event` at `at`, after every deadline due by then. Before the first pull there is
    /// no request to apply it to.
    fn step(&mut self, event: Event, at: Ts) {
        if matches!(self.phase, Phase::Pending(_)) {
            return;
        }
        let at = at.max(self.now);
        while let Some(due) = self.wake_at().filter(|due| *due <= at) {
            self.now = due.max(self.now);
            let mut steps = Vec::new();
            let Phase::Accepted(a) = &mut self.phase else { return };
            a.attempts.wake(self.now, a.demand, &mut steps);
            let idle = steps.iter().all(|step| matches!(step, Step::Unresolved));
            self.apply(steps);
            if idle && self.wake_at() == Some(due) {
                break;
            }
        }
        self.now = at;
        if let Event::AttemptClosed { attempt } = &event {
            self.tasks.remove(attempt);
        }
        let Phase::Accepted(a) = &mut self.phase else { return };
        let mut steps = Vec::new();
        match event {
            Event::Wake => a.attempts.wake(at, a.demand, &mut steps),
            Event::DemandStarted => a.demand = Some(at),
            Event::DemandEnded => a.demand = None,
            Event::Cancel => a.attempts.cancel(at, &mut steps),
            Event::AttemptClosed { attempt } => a.attempts.closed(attempt, at, &mut steps),
            Event::Connected { attempt } => a.attempts.connected(attempt, at),
            Event::RequestWritten { attempt } => a.attempts.written(attempt, at),
            Event::Progress { attempt } => a.attempts.progress(attempt, at),
            Event::Chunk { attempt, chunk } => {
                a.attempts.progress(attempt, at);
                if a.attempts.current() == Some(attempt) {
                    for choice in &chunk.choices {
                        if let Some(parsed) = a.choices.parse_choice(&chunk.id, choice) {
                            // reasoning is never shown, so an attempt that only reasoned retries
                            let delta = parsed.delta.as_ref();
                            let text = delta.and_then(|d| d.content.as_deref()).unwrap_or("");
                            let calls = delta.is_some_and(|d| !d.tool_calls.is_empty());
                            if !a.said.push(text).is_empty() || calls {
                                a.attempts.delivered();
                            }
                            self.ready.push_back(parsed);
                        }
                    }
                    self.ready.extend(usage_chunk(&chunk));
                }
            }
            Event::Ended { attempt } => {
                if a.attempts.succeeded(attempt, at) {
                    self.outcome = Some(Ok(()));
                }
            }
            Event::Failed { attempt, failure } => {
                a.attempts.failed(attempt, failure, at, &mut steps)
            }
        }
        self.apply(steps);
    }

    fn apply(&mut self, steps: Vec<Step>) {
        for step in steps {
            match step {
                Step::Launch(attempt) => {
                    if let Phase::Accepted(a) = &mut self.phase {
                        a.choices = ChoiceAccumulator::new();
                        a.said = SaidText::new(self.model_syntax, None);
                    }
                    self.start_attempt(attempt);
                }
                Step::Stop(attempt) => {
                    if let Some(task) = self.tasks.get_mut(&attempt) {
                        task.cancel();
                    }
                }
                Step::Retry { attempt, error } => {
                    log::warn!(target: lk_agents_abi::LOG_TARGET, attempt, error = error.message.as_str(); "LLM request attempt failed, retrying");
                    if let Some(host) = &self.host {
                        host.failed(ProviderError::Local(error), true);
                    }
                }
                Step::GiveUp(error) => self.outcome = Some(Err(error)),
                Step::CleanupIncomplete(attempt) => {
                    self.cleanup_incomplete = true;
                    log::error!(target: lk_agents_abi::LOG_TARGET, request = self.id, attempt; "LLM request attempt did not release its transport in time");
                }
                // A generation is never retired: its consumer's cancel ends it.
                Step::Unresolved => {}
            }
        }
    }

    /// Spawns one attempt; however it settles, the request hears that the attempt closed.
    fn start_attempt(&mut self, attempt: u32) {
        let (exchange, reporter) = (self.exchange.clone(), self.reporter.clone());
        let closed = self.reporter.clone();
        let work = async move {
            send_attempt(&exchange.http, exchange.request.clone(), attempt, reporter).await
        };
        let settle = move |outcome| async move {
            if outcome == TaskOutcome::Panicked {
                log::error!(target: lk_agents_abi::LOG_TARGET, attempt; "LLM request attempt panicked");
            }
            closed.report(Event::AttemptClosed { attempt }).await;
        };
        let task = self.spawner.spawn_settled(work, settle);
        self.tasks.insert(attempt, task);
    }
}

/// The consumer waits for a chunk from creation to drop: the read limit runs only meanwhile.
struct Demand<'a>(&'a mut LlmChunks);

impl<'a> Demand<'a> {
    fn start(chunks: &'a mut LlmChunks) -> Self {
        let now = chunks.reporter.clock.now();
        chunks.step(Event::DemandStarted, now);
        Demand(chunks)
    }
}

impl Drop for Demand<'_> {
    fn drop(&mut self) {
        let now = self.0.reporter.clock.now();
        self.0.step(Event::DemandEnded, now);
    }
}

impl StreamOutput<LlmChunk> for LlmChunks {
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

impl lk_agents_abi::utils::stream::ProviderStream<LlmChunk> for LlmChunks {}

/// One LLM request attempt: the completion's event stream decoded into chunk events.
async fn send_attempt(
    http: &GatewayHttp,
    request: HttpRequest,
    attempt: u32,
    reporter: Reporter<Event>,
) {
    let sent = |sent| match sent {
        Sent::Connected => Event::Connected { attempt },
        Sent::RequestWritten => Event::RequestWritten { attempt },
    };
    let failure = match http.send(request, &reporter, sent).await {
        Err(failure) => Some(failure),
        Ok(response) => read_events(response, attempt, &reporter).await,
    };
    if let Some(failure) = failure {
        reporter.report(Event::Failed { attempt, failure }).await;
    }
}

/// Decodes the event stream into chunk events; `None` once `[DONE]` ended the response. A body that
/// ends without it was cut short.
async fn read_events(
    response: reqwest::Response,
    attempt: u32,
    reporter: &Reporter<Event>,
) -> Option<AttemptFailure> {
    let request_id = response
        .headers()
        .get("X-LiveKit-Request-Id")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let mut decoder = SseDecoder::new();
    let mut bytes = response.bytes_stream();
    let mut events = Vec::new();
    loop {
        let Some(chunk) = bytes.next().await else {
            return Some(match decoder.finish() {
                Ok(()) => AttemptFailure::Connection("the stream ended before [DONE]".into()),
                Err(err) => AttemptFailure::Malformed(err.to_string()),
            });
        };
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(err) => return Some(AttemptFailure::Connection(error_chain(&err))),
        };
        if let Err(err) = decoder.feed(&chunk, &mut events) {
            return Some(AttemptFailure::Malformed(err.to_string()));
        }
        if events.is_empty() {
            reporter.report(Event::Progress { attempt }).await;
        }
        for event in events.drain(..) {
            let event = match decode_event(&event.data, attempt, request_id.as_deref()) {
                Ok(event) => event,
                Err(failure) => return Some(failure),
            };
            let ended = matches!(event, Event::Ended { .. });
            reporter.report(event).await;
            if ended {
                return None;
            }
        }
    }
}

/// One event of the stream: a chunk, the end at `[DONE]`, or the attempt's failure.
fn decode_event(
    data: &str,
    attempt: u32,
    request_id: Option<&str>,
) -> Result<Event, AttemptFailure> {
    if data == "[DONE]" {
        return Ok(Event::Ended { attempt });
    }
    match serde_json::from_str::<CompletionEvent>(data) {
        Ok(CompletionEvent::Chunk(chunk)) => Ok(Event::Chunk { attempt, chunk }),
        Ok(CompletionEvent::Error { error }) => Err(AttemptFailure::Rejected {
            status: i64::from(error.code),
            message: error.message,
            request_id: request_id.map(str::to_owned),
            retryable: None,
        }),
        Err(err) => Err(AttemptFailure::Malformed(format!("undecodable chunk: {err}"))),
    }
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum CompletionEvent {
    Error { error: InbandError },
    Chunk(WireChunk),
}

#[derive(serde::Deserialize)]
struct InbandError {
    message: String,
    #[serde(rename = "type")]
    _kind: String,
    code: u16,
}

#[cfg(test)]
#[path = "gateway_tests.rs"]
mod tests;
