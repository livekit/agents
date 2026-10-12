//! `inference/llm.py` -- the gateway LLM, HTTPS and OpenAI-compatible (§1.4): model tables
//! ([`models`]), kwargs composition ([`request`]), streaming decode ([`parse`]) and error mapping.
//! `retryable` is cleared once a chunk `has_response()`, so a failure after the model began
//! answering is not retried.

pub mod models;
pub mod parse;
#[cfg(feature = "python")]
pub(crate) mod python;
pub mod request;

use crate::abi::{ChatRequest, Llm, LlmStream, ToolSpec};
use crate::headers::{Headers, extract_quota_usage, gateway_headers};
use crate::transport::{HttpRequest, HttpStream, HttpTransport, TransportError, within};
use async_trait::async_trait;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::llm::provider_format::ImageResolver;
use lk_agents_abi::{ApiConnectOptions, ChatContext, Clock, LlmChunk, RealClock};
use parse::{ChoiceAccumulator, TagFilter, WireChunk, usage_chunk};
use request::ChatArgs;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

/// `_LLMOptions`.
#[derive(Debug, Clone)]
pub struct LlmOptions {
    pub model: String,
    pub base_url: String,
    pub api_key: String,
    pub api_secret: String,
    /// `X-LiveKit-Inference-Provider`. `None` lets the gateway route.
    pub provider: Option<String>,
    /// `X-LiveKit-Inference-Priority`.
    pub inference_class: Option<String>,
    pub extra_kwargs: serde_json::Map<String, serde_json::Value>,
}

/// `_core.InferenceLlm`'s keyword constructor: [`LlmOptions`] as the host passes them, in its order.
#[lk_agents_abi::host(client(class = crate::llm::python::InferenceLlm, positional = model))]
#[derive(Debug, Clone)]
pub struct LlmClientConfig {
    pub model: String,
    pub base_url: String,
    pub api_key: String,
    pub api_secret: String,
    #[host(kwarg(default = None))]
    pub provider: Option<String>,
    #[host(kwarg(default = None))]
    pub inference_class: Option<String>,
    #[host(kwarg(default = None))]
    pub extra_kwargs: Option<crate::extra::ExtraKwargs>,
}

impl From<LlmClientConfig> for LlmOptions {
    fn from(c: LlmClientConfig) -> Self {
        LlmOptions {
            model: c.model,
            base_url: c.base_url,
            api_key: c.api_key,
            api_secret: c.api_secret,
            provider: c.provider,
            inference_class: c.inference_class,
            extra_kwargs: c.extra_kwargs.unwrap_or_default().0,
        }
    }
}

/// `_log_rate_limited`'s destination; the host owns the logger (§1.10).
pub trait RateLimitLog: Send + Sync {
    fn rate_limited(&self, model: &str, request_id: Option<&str>, quota: &[(&'static str, String)]);
}

/// Records nothing; the 429 still reaches the caller.
pub struct NoRateLimitLog;

impl RateLimitLog for NoRateLimitLog {
    fn rate_limited(&self, _: &str, _: Option<&str>, _: &[(&'static str, String)]) {}
}

pub struct NativeLlm {
    opts: LlmOptions,
    http: Arc<dyn HttpTransport>,
    images: Arc<dyn ImageResolver + Send + Sync>,
    identification: Option<Headers>,
    log: Arc<dyn RateLimitLog>,
    conn_options: ApiConnectOptions,
    /// Bounds `conn_options.timeout` around the request and each SSE event.
    clock: Arc<dyn Clock>,
}

impl NativeLlm {
    pub fn new(
        opts: LlmOptions,
        http: Arc<dyn HttpTransport>,
        images: Arc<dyn ImageResolver + Send + Sync>,
    ) -> Self {
        NativeLlm {
            opts,
            http,
            images,
            identification: None,
            log: Arc::new(NoRateLimitLog),
            conn_options: lk_agents_abi::DEFAULT_API_CONNECT_OPTIONS,
            clock: Arc::new(RealClock::new()),
        }
    }

    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    /// `get_inference_headers(inference_class=...)`; replaces the native `User-Agent` and priority.
    pub fn with_identification(mut self, headers: Headers) -> Self {
        self.identification = Some(headers);
        self
    }

    pub fn with_rate_limit_log(mut self, log: Arc<dyn RateLimitLog>) -> Self {
        self.log = log;
        self
    }

    pub fn with_conn_options(mut self, conn_options: ApiConnectOptions) -> Self {
        self.conn_options = conn_options;
        self
    }

    /// `chat()` with the two arguments `ChatRequest` has no field for: an explicit
    /// `response_format`, and `extra_kwargs` beyond what the request carries.
    pub fn chat_with(
        &self,
        chat_ctx: &ChatContext,
        tools: &[ToolSpec],
        args: ChatArgs,
    ) -> Result<LlmStream, ProviderError> {
        let prepared = self.prepare(chat_ctx, tools, args)?;
        Ok(Box::new(NativeLlmStream::pending(prepared)))
    }

    fn prepare(
        &self,
        chat_ctx: &ChatContext,
        tools: &[ToolSpec],
        args: ChatArgs,
    ) -> Result<Prepared, ApiError> {
        let opts = &self.opts;

        let mut extra = request::compose_extra(&args, &opts.extra_kwargs);
        // Before `tool_choice` is popped (as in `LLMStream.__init__`): the model must see the
        // tool list as passed to `chat()`.
        models::drop_unsupported_params(&opts.model, &mut extra, !tools.is_empty());
        let extra_headers = extra.remove("extra_headers");

        let formatted = lk_agents_abi::llm::provider_format::openai::to_chat_ctx(chat_ctx, self.images.as_ref())
            .map_err(|e| ApiError::new(format!("failed to format chat context: {e}")))?;
        let schemas = request::tool_schemas(tools)?;
        let body = request::build_body(&opts.model, formatted.messages, &schemas, extra);

        // `extra_headers` (the composed kwarg, so the instance's replaces the call's), overlaid
        // with the job's identification, the provider pin and the bearer.
        let mut headers: Headers = match extra_headers {
            Some(serde_json::Value::Object(map)) => map
                .into_iter()
                .map(|(k, v)| (k, v.as_str().map_or_else(|| v.to_string(), str::to_owned)))
                .collect(),
            _ => Headers::new(),
        };
        for (name, value) in gateway_headers(
            self.identification.as_ref(),
            args.inference_class.as_deref().or(opts.inference_class.as_deref()),
            opts.provider.as_deref(),
            &opts.api_key,
            &opts.api_secret,
        )?
        .into_vec()
        {
            headers.set(name, value);
        }
        headers.set("Content-Type", "application/json");

        Ok(Prepared {
            request: HttpRequest {
                url: request::completions_url(&opts.base_url),
                headers: headers.into_vec(),
                body: serde_json::to_vec(&serde_json::Value::Object(body))
                    .map_err(|e| ApiError::new(e.to_string()))?,
            },
            timeout: Duration::from_secs_f64(self.conn_options.timeout.max(0.0)),
            clock: self.clock.clone(),
            http: self.http.clone(),
            log: self.log.clone(),
            model: opts.model.clone(),
            think_tags: models::think_tags(&opts.model),
        })
    }
}

#[async_trait]
impl Llm for NativeLlm {
    /// `LLM.label`, which the reference builds as `f"{provider}.{model}"` on the base class.
    fn label(&self) -> String {
        format!("livekit.{}", self.model())
    }

    fn model(&self) -> String {
        self.opts.model.clone()
    }

    /// Fails synchronously on an unserialisable tool schema or chat context, so `FallbackChain`
    /// sees a pre-chunk (retryable) failure. The request itself is issued on the first `next()`.
    /// `chat_ctx_host`/`tools_host` are ignored (no plugin here).
    async fn chat(&self, req: &ChatRequest) -> Result<LlmStream, ProviderError> {
        self.chat_with(
            &req.chat_ctx,
            &req.tools,
            ChatArgs {
                // `extra_kwargs`, which composes *under* the instance's own (see `compose_extra`).
                extra_kwargs: req.extra.clone(),
                parallel_tool_calls: req.parallel_tool_calls,
                tool_choice: req.tool_choice.clone(),
                response_format: None,
                inference_class: req.inference_class.clone(),
            },
        )
    }
}

struct Prepared {
    request: HttpRequest,
    /// `conn_options.timeout`: bounds the response and then each SSE event (a per-read timeout,
    /// like `httpx.Timeout`).
    timeout: Duration,
    clock: Arc<dyn Clock>,
    http: Arc<dyn HttpTransport>,
    log: Arc<dyn RateLimitLog>,
    model: String,
    think_tags: (&'static str, &'static str),
}

enum State {
    Pending(Box<Prepared>),
    Streaming(HttpStream),
    Done,
}

/// Tracks whether retry would duplicate visible output, shared with host SDK streams.
#[lk_agents_abi::class(name = "LlmResponseState", mutable)]
#[derive(Default)]
pub struct ResponseState { produced_output: bool }
#[lk_agents_abi::host(object(name = "LlmResponseState"))]
impl ResponseState {
    #[host(new)]
    pub fn new() -> Self {
        Self::default()
    }
    #[host(method)]
    pub fn observe(&mut self, chunk: &LlmChunk) {
        self.produced_output |= chunk.has_response();
    }
    #[host(getter)]
    pub fn retryable(&self) -> bool { !self.produced_output }
}

/// One `chat.completions.create` stream. The request is issued on the first `next()`.
pub struct NativeLlmStream {
    state: State,
    acc: ChoiceAccumulator,
    /// The reasoning-token filter, from the model's own tags; without it the agent speaks its
    /// reasoning.
    filter: TagFilter,
    pending: VecDeque<LlmChunk>,
    /// `_run`'s local. Cleared the first time a chunk carries a response.
    response: ResponseState,
    timeout: Duration,
    clock: Arc<dyn Clock>,
    log: Arc<dyn RateLimitLog>,
    model: String,
}

impl NativeLlmStream {
    fn pending(prepared: Prepared) -> Self {
        NativeLlmStream {
            filter: TagFilter::new(prepared.think_tags.0, prepared.think_tags.1),
            acc: ChoiceAccumulator::new(),
            pending: VecDeque::new(),
            response: ResponseState::default(),
            timeout: prepared.timeout,
            clock: prepared.clock.clone(),
            log: prepared.log.clone(),
            model: prepared.model.clone(),
            state: State::Pending(Box::new(prepared)),
        }
    }

    /// The five `except` arms of `_run`. `retryable` is passed to every arm, including
    /// `APIStatusError`, where it overrides the status-derived default.
    fn map_error(&self, err: TransportError) -> ApiError {
        let retryable = self.response.retryable();
        if let TransportError::Status { status: 429, .. } = err {
            // `_log_rate_limited`: the quota snapshot stamped on rejections.
            let quota = extract_quota_usage(|name| err.header(name));
            self.log.rate_limited(&self.model, None, &quota);
        }
        match err {
            TransportError::Timeout => ApiError::timeout(None).with_retryable(retryable),
            // `retryable=retryable` overrides the status-derived default in both directions.
            TransportError::Status { status, message, .. } => {
                ApiError::status(message, status, None, Some(retryable))
            }
            // `except Exception as e: raise APIConnectionError(retryable=retryable) from e` --
            // also where a mid-stream close lands.
            _ => ApiError::connection(None).with_retryable(retryable),
        }
    }

    fn ingest(&mut self, payload: &str) {
        let Ok(chunk) = serde_json::from_str::<WireChunk>(payload) else {
            // The reference's SDK raises; an undecodable event (keep-alives) is skipped here.
            return;
        };
        for choice in &chunk.choices {
            if let Some(parsed) = self.acc.parse_choice(&chunk.id, choice, &mut self.filter) {
                self.response.observe(&parsed);
                self.pending.push_back(parsed);
            }
        }
        if let Some(usage) = usage_chunk(&chunk) {
            self.pending.push_back(usage);
        }
    }
}

#[async_trait]
impl lk_agents_abi::StreamSource<LlmChunk> for NativeLlmStream {
    /// Safe to abandon mid-poll.
    fn cancel_safe(&self) -> bool {
        true
    }

    async fn next(&mut self) -> Result<Option<LlmChunk>, ProviderError> {
        loop {
            if let Some(chunk) = self.pending.pop_front() {
                return Ok(Some(chunk));
            }
            // Both waits are bounded on the session clock: a silent gateway is an
            // `APITimeoutError`, retryable until content was emitted.
            let step = match std::mem::replace(&mut self.state, State::Done) {
                State::Done => return Ok(None),
                State::Pending(prepared) => {
                    let Prepared { request, http, .. } = *prepared;
                    within(&*self.clock, self.timeout, http.post_stream(request))
                        .await
                        .map(|stream| self.state = State::Streaming(stream))
                }
                // Per event, not per response: inactivity between chunks (`: ping` comments
                // count) fails in `conn_options.timeout`.
                State::Streaming(mut stream) => {
                    match within(&*self.clock, self.timeout, stream.body.next_event()).await {
                        Ok(Some(payload)) => {
                            self.state = State::Streaming(stream);
                            self.ingest(&payload);
                            Ok(())
                        }
                        Ok(None) => return Ok(None),
                        Err(err) => Err(err),
                    }
                }
            };
            if let Err(err) = step {
                return Err(self.map_error(err).into());
            }
        }
    }

    /// Ends the stream: the socket is dropped with `state`, and nothing buffered is delivered
    /// afterwards. Nothing here awaits the network, so the deadline is not consulted.
    async fn close(mut self: Box<Self>, _deadline: lk_agents_abi::Ts) -> lk_agents_abi::Closed {
        self.state = State::Done;
        self.pending.clear();
        lk_agents_abi::Closed::Clean
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{FakeHttp, NoImages};
    use lk_agents_abi::errors::ApiErrorKind;
    use std::sync::Mutex;

    fn opts() -> LlmOptions {
        LlmOptions {
            model: "openai/gpt-4o".into(),
            provider: None,
            base_url: "https://gw/v1".into(),
            api_key: "k".into(),
            api_secret: "s".into(),
            inference_class: None,
            extra_kwargs: Default::default(),
        }
    }

    fn llm(events: Vec<Result<String, TransportError>>) -> (NativeLlm, Arc<FakeHttp>) {
        let http = Arc::new(FakeHttp::new(events));
        (NativeLlm::new(opts(), http.clone(), Arc::new(NoImages)), http)
    }

    /// One poll with a no-op waker; the virtual clock is what moves the future.
    fn poll_once<F: Future>(f: std::pin::Pin<&mut F>) -> std::task::Poll<F::Output> {
        let waker = std::task::Waker::noop();
        let mut cx = std::task::Context::from_waker(waker);
        f.poll(&mut cx)
    }

    // Catches: an unbounded wait on the response, or one bounded on tokio's clock; 10 s on the
    // session clock is `APITimeoutError(retryable=True)`.
    #[tokio::test]
    async fn a_request_the_gateway_never_answers_times_out_on_the_session_clock_and_is_retryable() {
        let clock = Arc::new(lk_agents_abi::VirtualClock::new());
        let http = Arc::new(FakeHttp::hanging());
        let llm = NativeLlm::new(opts(), http.clone(), Arc::new(NoImages)).with_clock(clock.clone());
        let mut stream = open(&llm).await;

        let mut next = Box::pin(stream.next());
        assert!(poll_once(next.as_mut()).is_pending(), "the request is in flight");
        clock.advance(Duration::from_millis(9_900));
        assert!(poll_once(next.as_mut()).is_pending(), "9.9 s is not the 10 s bound");
        clock.advance(Duration::from_millis(100));
        let err = match poll_once(next.as_mut()) {
            std::task::Poll::Ready(Err(err)) => err,
            std::task::Poll::Ready(Ok(chunk)) => panic!("a chunk from a silent gateway: {chunk:?}"),
            std::task::Poll::Pending => panic!("10 s passed and the request still waits"),
        };
        assert_eq!(err.api_kind(), Some(ApiErrorKind::Timeout));
        assert!(err.retryable(), "nothing was emitted, so the retry is safe");
        assert_eq!(http.requests(), 1, "the request was issued exactly once; retrying is not ours");
    }

    // Catches: bounding only the request and not each event, or bounding the whole response.
    #[tokio::test]
    async fn a_stream_that_stalls_after_content_times_out_per_event_and_is_not_retryable() {
        let clock = Arc::new(lk_agents_abi::VirtualClock::new());
        let http = Arc::new(
            FakeHttp::new(vec![ev(
                serde_json::json!({"id": "c", "choices": [{"delta": {"content": "Hel"}}]}),
            )])
            .then_hang(),
        );
        let llm = NativeLlm::new(opts(), http, Arc::new(NoImages)).with_clock(clock.clone());
        let mut stream = open(&llm).await;
        assert!(stream.next().await.unwrap().is_some(), "the first chunk arrives");

        let mut next = Box::pin(stream.next());
        assert!(poll_once(next.as_mut()).is_pending(), "and then the gateway goes quiet");
        clock.advance(Duration::from_secs(10));
        let err = match poll_once(next.as_mut()) {
            std::task::Poll::Ready(Err(err)) => err,
            std::task::Poll::Ready(Ok(chunk)) => panic!("a chunk from a stalled stream: {chunk:?}"),
            std::task::Poll::Pending => panic!("10 s of silence mid-stream and still waiting"),
        };
        assert_eq!(err.api_kind(), Some(ApiErrorKind::Timeout));
        assert!(!err.retryable(), "content was already emitted");
    }

    /// One user turn, which is all any of these tests needs on the request side.
    async fn open(llm: &NativeLlm) -> LlmStream {
        llm.chat(&ChatRequest::default()).await.expect("chat() itself does not fail here")
    }

    fn ev(json: serde_json::Value) -> Result<String, TransportError> {
        Ok(json.to_string())
    }

    async fn drain(
        stream: &mut dyn lk_agents_abi::StreamSource<LlmChunk>,
    ) -> Result<Vec<LlmChunk>, ProviderError> {
        let mut out = Vec::new();
        while let Some(chunk) = stream.next().await? {
            out.push(chunk);
        }
        Ok(out)
    }

    // Catches: retrying a request that already delivered text.
    #[tokio::test]
    async fn a_failure_after_the_first_content_chunk_is_not_retryable() {
        let (llm, _) = llm(vec![
            ev(serde_json::json!({"id": "c", "choices": [{"delta": {"content": "Hel"}}]})),
            Err(TransportError::Closed),
        ]);
        let mut stream = open(&llm).await;
        assert!(stream.next().await.unwrap().is_some());
        let err = stream.next().await.expect_err("the socket died");
        assert!(!err.retryable(), "content was already emitted");
        assert_eq!(err.api_kind(), Some(ApiErrorKind::Connection));
    }

    // Catches: a pre-content failure marked non-retryable.
    #[tokio::test]
    async fn a_failure_before_any_content_is_retryable() {
        let (llm, _) = llm(vec![Err(TransportError::Timeout)]);
        let mut stream = open(&llm).await;
        let err = stream.next().await.expect_err("timed out");
        assert!(err.retryable());
        assert_eq!(err.api_kind(), Some(ApiErrorKind::Timeout));
    }

    // Catches: `retryable` clobbered by `retryable_for_status` after content.
    #[tokio::test]
    async fn an_http_status_error_uses_the_streams_flag_not_the_status_rule() {
        let (llm, _) = llm(vec![
            ev(serde_json::json!({"id": "c", "choices": [{"delta": {"content": "x"}}]})),
            Err(TransportError::Status {
                status: 500,
                message: "boom".into(),
                headers: vec![],
                body: None,
            }),
        ]);
        let mut stream = open(&llm).await;
        stream.next().await.unwrap();
        let err = stream.next().await.expect_err("500");
        assert_eq!(err.status_code(), Some(500));
        assert!(!err.retryable(), "a 5xx is retryable by default; the stream's flag overrides");
    }

    // Catches: losing the quota snapshot on a 429.
    #[tokio::test]
    async fn a_429_reports_the_quota_snapshot_the_gateway_stamped() {
        type Logged = (String, Vec<(&'static str, String)>);
        struct Recorder(Mutex<Vec<Logged>>);
        impl RateLimitLog for Recorder {
            fn rate_limited(&self, model: &str, _: Option<&str>, quota: &[(&'static str, String)]) {
                self.0.lock().unwrap().push((model.to_owned(), quota.to_vec()));
            }
        }
        let rec = Arc::new(Recorder(Mutex::new(Vec::new())));
        let (llm, _) = llm(vec![Err(TransportError::Status {
            status: 429,
            message: "Too Many Requests".into(),
            // lower-cased, as a real HTTP/2 response delivers them
            headers: vec![("x-livekit-inference-rpm-used".into(), "61".into())],
            body: None,
        })]);
        let llm = llm.with_rate_limit_log(rec.clone());
        let mut stream = open(&llm).await;
        assert!(stream.next().await.is_err());
        let logged = rec.0.lock().unwrap().clone();
        assert_eq!(
            logged,
            vec![("openai/gpt-4o".to_string(), vec![("rpm_used", "61".to_string())])]
        );
    }

    // Catches: ending the stream on an undecodable event.
    #[tokio::test]
    async fn an_undecodable_event_is_skipped_not_fatal() {
        let (llm, _) = llm(vec![
            Ok("{not json".into()),
            ev(serde_json::json!({"id": "c", "choices": [{"delta": {"content": "hi"}}]})),
        ]);
        let mut stream = open(&llm).await;
        let chunks = drain(stream.as_mut()).await.unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].delta.as_ref().unwrap().content.as_deref(), Some("hi"));
    }

    // Catches: issuing the HTTP request inside `chat()`.
    #[tokio::test]
    async fn the_request_is_issued_on_the_first_next_not_in_chat() {
        let (llm, http) = llm(vec![ev(serde_json::json!({"id": "c", "choices": []}))]);
        let mut stream = open(&llm).await;
        assert_eq!(http.requests(), 0, "chat() must not touch the socket");
        let _ = stream.next().await;
        assert_eq!(http.requests(), 1);
    }

    // Catches: sending job headers when there is no job.
    #[tokio::test]
    async fn with_no_job_context_the_request_carries_no_livekit_job_headers() {
        let (llm, http) = llm(vec![ev(serde_json::json!({"id": "c", "choices": []}))]);
        let mut stream = open(&llm).await;
        let _ = stream.next().await;
        let request = http.last_request().expect("a request");
        let names: Vec<&str> = request
            .headers
            .iter()
            .map(|(k, _)| k.as_str())
            .filter(|k| k.starts_with("X-LiveKit-"))
            .collect();
        assert!(names.is_empty(), "got {names:?}");
        assert!(
            request.headers.iter().any(|(k, v)| k == "Authorization" && v.starts_with("Bearer "))
        );
        assert!(request.url.ends_with("/v1/chat/completions"), "{}", request.url);
    }

    // Catches: the host's identification headers (the job's `X-LiveKit-*` identity, read where
    // `chat()` runs) not reaching the request, or the native priority/User-Agent leaking in
    // beside them. The pinned provider and the bearer are still the core's.
    #[tokio::test]
    async fn the_hosts_identification_headers_ride_on_the_request() {
        let http = Arc::new(FakeHttp::new(vec![ev(serde_json::json!({"id": "c", "choices": []}))]));
        let opts = LlmOptions { provider: Some("openai".into()), inference_class: Some("low".into()), ..opts() };
        let ident: Headers = [
            ("User-Agent", "LiveKit Agents/1 (python 3)"),
            ("X-LiveKit-Room-ID", "RM_1"),
            ("X-LiveKit-Job-ID", "AJ_1"),
            ("X-LiveKit-Agent-ID", "PA_1"),
            ("X-LiveKit-Session-ID", "inference_abc"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
        let llm = NativeLlm::new(opts, http.clone(), Arc::new(NoImages)).with_identification(ident);
        let mut stream = open(&llm).await;
        let _ = stream.next().await;
        let request = http.last_request().expect("a request");
        let get = |name: &str| {
            request.headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
        };
        assert_eq!(get("X-LiveKit-Room-ID"), Some("RM_1"));
        assert_eq!(get("X-LiveKit-Job-ID"), Some("AJ_1"));
        assert_eq!(get("X-LiveKit-Agent-ID"), Some("PA_1"));
        assert_eq!(get("X-LiveKit-Session-ID"), Some("inference_abc"));
        assert_eq!(get("User-Agent"), Some("LiveKit Agents/1 (python 3)"));
        assert_eq!(get("X-LiveKit-Inference-Priority"), None, "the host's map already decided it");
        assert_eq!(get("X-LiveKit-Inference-Provider"), Some("openai"));
        assert!(get("Authorization").is_some_and(|v| v.starts_with("Bearer ")));
    }

    // Catches: `Llm::model` returning `"unknown"`, or a label other than `provider.model`.
    #[test]
    fn the_trait_reports_the_configured_model_and_label() {
        let (llm, _) = llm(vec![]);
        let dynamic: &dyn Llm = &llm;
        assert_eq!(dynamic.model(), "openai/gpt-4o");
        assert_eq!(llm.label(), "livekit.openai/gpt-4o", "the label is provider.model");
    }
}
