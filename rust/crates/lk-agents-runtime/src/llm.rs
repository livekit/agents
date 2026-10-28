//! The gateway LLM: each `chat()` is one [`LlmRequest`] core whose effects this module executes,
//! pulled by its consumer. The request is stepped only while it is pulled or closed, so nothing
//! runs ahead of demand; transport observations wait in a bounded queue, stamped with the instant
//! they were made, and are applied in that order with every deadline due before them.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::llm::provider_format::PreResolved;
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::{ChatContext, ChatRequest, Clock, Closed, Llm, LlmChunk, LlmStream, StreamSource, ToolSpec, Ts};
use lk_agents_inference::headers::{Headers, gateway_headers};
use lk_agents_inference::llm::lifecycle::{LlmRequest, LlmRequestEffect, LlmRequestEvent, LlmRequestId, RequestPolicy};
use lk_agents_inference::llm::request::ChatArgs;
use lk_agents_inference::llm::{LlmOptions, build_request, models};
use lk_agents_inference::transport::HttpRequest;
use tokio::sync::mpsc;

use crate::http::{GatewayHttp, Reporter, Stamped};
use crate::task::{OwnedTask, TaskOutcome};

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
    pub fn new(opts: LlmOptions, http: Arc<GatewayHttp>, runtime: tokio::runtime::Handle, clock: Arc<dyn Clock>) -> Self {
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
    pub fn identified(&self, identity: Headers, reporter: Option<Arc<dyn Llm>>) -> Self {
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

    /// Opens a completion; the first attempt starts at once, its response is read on demand.
    pub fn chat(
        &self,
        chat_ctx: &ChatContext,
        tools: &[ToolSpec],
        args: ChatArgs,
        policy: RequestPolicy,
    ) -> Result<LlmChunks, ApiError> {
        let opts = self.options();
        let images = PreResolved::from_pairs(Vec::<(String, _)>::new());
        let request = build_request(&opts, self.identity.as_ref(), &images, chat_ctx, tools, args)?;
        let id = LlmRequestId(self.requests.fetch_add(1, Ordering::Relaxed));
        let now = self.clock.now();
        let (core, effects) = LlmRequest::start(id, self.resource, policy, models::think_tags(&opts.model), now);
        let (tx, rx) = mpsc::channel(OBSERVATION_QUEUE);
        let mut chunks = LlmChunks {
            core,
            rx,
            reporter: Reporter { tx, clock: self.clock.clone() },
            exchange: Arc::new(Exchange { http: self.http.clone(), request }),
            runtime: self.runtime.clone(),
            attempts: BTreeMap::new(),
            ready: VecDeque::new(),
            outcome: None,
            ended: false,
            cleanup_incomplete: false,
            now,
            host: self.reporter.clone(),
        };
        chunks.apply(effects);
        Ok(chunks)
    }
}

#[async_trait]
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
        lk_agents_abi::ProviderId { label: self.label(), model: self.model(), provider: "livekit".into() }
    }

    /// Opens a connection in the background so the first request finds it pooled.
    fn prewarm(&self) {
        let opts = self.options();
        let Ok(headers) = gateway_headers(self.identity.as_ref(), None, None, &opts.api_key, &opts.api_secret) else {
            return;
        };
        let (http, url) = (self.http.clone(), format!("{}/models", opts.base_url.trim_end_matches('/')));
        let task = OwnedTask::spawn(&self.runtime, async move { http.warm(url, headers.into_vec()).await }, |_| async {});
        *self.prewarm.lock().unwrap_or_else(|p| p.into_inner()) = Some(task);
    }

    /// The voice session's generation retries its own LLM calls until it delegates them to the
    /// request core, so each call here is one attempt; its single `timeout` bounds every wait.
    async fn chat(&self, req: &ChatRequest) -> Result<LlmStream, ProviderError> {
        let limit = Some(Duration::from_secs_f64(req.conn_options.timeout.max(0.0)));
        let policy = RequestPolicy {
            connect_timeout: limit,
            write_timeout: limit,
            read_timeout: limit,
            total_timeout: None,
            max_retry: 0,
            retry_interval: Duration::ZERO,
        };
        let args = ChatArgs {
            extra_kwargs: req.extra.clone(),
            parallel_tool_calls: req.parallel_tool_calls,
            tool_choice: req.tool_choice.clone(),
            response_format: None,
            inference_class: req.inference_class.clone(),
        };
        Ok(Box::new(self.chat(&req.chat_ctx, &req.tools, args, policy)?))
    }
}

struct Exchange {
    http: Arc<GatewayHttp>,
    request: HttpRequest,
}

/// One request's chunks, in order, then its outcome.
pub struct LlmChunks {
    core: LlmRequest,
    rx: mpsc::Receiver<Stamped>,
    reporter: Reporter,
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
        let demand = Demand::start(self);
        demand.0.wait().await
    }

    /// Stops the request and waits, until `deadline`, for every attempt to release its
    /// transport. A missed deadline is reported, never waited on further.
    pub async fn close(mut self, deadline: Ts) -> Closed {
        let now = self.reporter.clock.now();
        self.step(LlmRequestEvent::Cancel, now);
        while !self.core.is_settled() {
            let wake = self.core.wake_at().map_or(deadline, |at| at.min(deadline));
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
            let wake = self.core.wake_at();
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

    /// Applies `event` at `at`, after every deadline due by then.
    fn step(&mut self, event: LlmRequestEvent, at: Ts) {
        let at = at.max(self.now);
        while let Some(due) = self.core.wake_at().filter(|due| *due <= at) {
            self.now = due.max(self.now);
            let effects = self.core.handle(LlmRequestEvent::Wake, self.now);
            if effects.is_empty() && self.core.wake_at() == Some(due) {
                break;
            }
            self.apply(effects);
        }
        self.now = at;
        if let LlmRequestEvent::AttemptClosed { attempt } = &event {
            self.attempts.remove(attempt);
        }
        let effects = self.core.handle(event, at);
        self.apply(effects);
    }

    fn apply(&mut self, effects: Vec<LlmRequestEffect>) {
        for effect in effects {
            match effect {
                LlmRequestEffect::LlmRequestStart { attempt, .. } => {
                    let (exchange, reporter) = (self.exchange.clone(), self.reporter.clone());
                    let closed = self.reporter.clone();
                    let work = async move { exchange.http.exchange(exchange.request.clone(), attempt, reporter).await };
                    let task = OwnedTask::spawn(&self.runtime, work, move |outcome| async move {
                        if outcome == TaskOutcome::Panicked {
                            log::error!(target: "livekit.agents", attempt; "LLM request attempt panicked");
                        }
                        let _ = closed.tx.send(Stamped { at: closed.clock.now(), event: LlmRequestEvent::AttemptClosed { attempt } }).await;
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

#[async_trait]
impl StreamSource<LlmChunk> for LlmChunks {
    fn cancel_safe(&self) -> bool {
        true
    }

    async fn next(&mut self) -> Result<Option<LlmChunk>, ProviderError> {
        LlmChunks::next(self).await.map_err(ProviderError::Local)
    }

    async fn close(self: Box<Self>, deadline: Ts) -> Closed {
        LlmChunks::close(*self, deadline).await
    }
}
