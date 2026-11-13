use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{
    ApiConnectOptions, BoxFuture, ChatRequest, ChoiceDelta, Clock, Closed, Llm, LlmChunk,
    LlmStream, StreamOutput, Ts,
};
use lk_agents_net::clock::RealClock;
use lk_agents_net::env::ProcessEnv;
use parking_lot::Mutex;

use super::FallbackLlm;
use crate::fallback::ChainPolicy;
use crate::testkit::fallback::{NO_DEADLINE, boom, parts, until, within};

struct Provider(Arc<AtomicBool>);

impl Llm for Provider {
    fn label(&self) -> String {
        "scripted".into()
    }

    fn model(&self) -> String {
        "native-model".into()
    }

    fn chat<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a ChatRequest,
    ) -> lk_agents_abi::BoxFuture<'a, Result<LlmStream, ProviderError>> {
        Box::pin(async move {
            Ok(Box::new(Response { sent: false, closed: self.0.clone() }) as LlmStream)
        })
    }
}

struct Response {
    sent: bool,
    closed: Arc<AtomicBool>,
}

impl StreamOutput<LlmChunk> for Response {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<LlmChunk>, ProviderError>> {
        Box::pin(async move {
            if self.sent {
                return std::future::pending().await;
            }
            self.sent = true;
            Ok(Some(LlmChunk {
                id: "reply".into(),
                delta: Some(ChoiceDelta {
                    content: Some("native response".into()),
                    ..Default::default()
                }),
                ..Default::default()
            }))
        })
    }

    fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move {
            self.closed.store(true, Ordering::SeqCst);
            Closed::Clean
        })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<LlmChunk> for Response {}

#[tokio::test]
async fn declared_fallback_keeps_native_dispatch_and_cancels_its_open_stream() {
    use lk_agents_abi::llm::fallback::FallbackLlm as DeclaredFallback;
    let empty = <FallbackLlm as DeclaredFallback>::new(
        &ProcessEnv::current_runtime(),
        "llm.FallbackAdapter".into(),
        Arc::new(crate::fallback::IgnoreEvents),
        vec![],
        5.0,
        0,
        0.5,
        false,
    );
    assert!(empty.is_err());

    let closed = Arc::new(AtomicBool::new(false));
    let fallback = <FallbackLlm as DeclaredFallback>::new(
        &ProcessEnv::current_runtime(),
        "llm.FallbackAdapter".into(),
        Arc::new(crate::fallback::IgnoreEvents),
        vec![Arc::new(Provider(closed.clone()))],
        5.0,
        0,
        0.5,
        false,
    )
    .expect("native provider accepted");
    assert_eq!(fallback.model(), "native-model");
    let mut stream =
        Llm::chat(&fallback, &lk_agents_abi::telemetry::Cx::NONE, &ChatRequest::default())
            .await
            .expect("opened");
    let chunk = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("first chunk arrived")
        .expect("provider succeeded")
        .expect("provider emitted a chunk");
    assert_eq!(chunk.delta.unwrap().content.as_deref(), Some("native response"));
    assert_eq!(DeclaredFallback::active_index(&fallback), 0);

    let deadline = RealClock::new().now().saturating_add(Duration::from_secs(5));
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(5), stream.close(deadline))
            .await
            .expect("close settled"),
        Closed::Clean,
    );
    assert!(closed.load(Ordering::SeqCst));
    DeclaredFallback::aclose(&fallback).await;
}

/// Drain to the end, bounded, so a stream that never ends fails instead of starving the process.
async fn drain_llm(stream: &mut LlmStream) -> Vec<LlmChunk> {
    let mut out = Vec::new();
    for _ in 0..32 {
        match within("the next chunk", stream.next()).await {
            Ok(None) => return out,
            Ok(Some(c)) => out.push(c),
            Err(e) => panic!("the stream failed: {e}"),
        }
    }
    panic!("the stream never reported its end");
}

struct FakeLlm {
    label: String,
    /// `None` leaves the trait default (`"unknown"`).
    model: Option<String>,
    seen: Mutex<Vec<ApiConnectOptions>>,
    /// `None` means `chat()` itself raises.
    chunks: Option<Vec<LlmChunk>>,
    /// The stream dies after the scripted chunks instead of ending cleanly.
    fail_after: bool,
    /// `chat()` raises this many more times before it serves.
    outages: AtomicUsize,
}

impl FakeLlm {
    fn build(
        label: &str,
        model: Option<&str>,
        chunks: Option<Vec<LlmChunk>>,
        fail_after: bool,
    ) -> Arc<Self> {
        let (label, model, seen) = (label.into(), model.map(Into::into), Mutex::new(vec![]));
        Arc::new(FakeLlm { label, model, seen, chunks, fail_after, outages: AtomicUsize::new(0) })
    }
    fn new(label: &str, chunks: Option<Vec<LlmChunk>>) -> Arc<Self> {
        Self::build(label, None, chunks, false)
    }
    fn dying(label: &str, chunks: Vec<LlmChunk>) -> Arc<Self> {
        Self::build(label, None, Some(chunks), true)
    }
}

struct VecStream {
    left: VecDeque<LlmChunk>,
    fail_after: bool,
}

impl StreamOutput<LlmChunk> for VecStream {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<LlmChunk>, ProviderError>> {
        Box::pin(async move {
            match self.left.pop_front() {
                Some(c) => Ok(Some(c)),
                None if self.fail_after => Err(boom("stream died")),
                None => Ok(None),
            }
        })
    }
    fn close(self: Box<Self>, _deadline: Ts) -> BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<LlmChunk> for VecStream {}

impl Llm for FakeLlm {
    fn label(&self) -> String {
        self.label.clone()
    }
    fn model(&self) -> String {
        self.model.clone().unwrap_or_else(|| "unknown".into())
    }
    fn chat<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        req: &'a ChatRequest,
    ) -> lk_agents_abi::BoxFuture<'a, Result<LlmStream, ProviderError>> {
        Box::pin(async move {
            self.seen.lock().push(req.conn_options);
            if self.outages.load(Ordering::SeqCst) > 0 {
                self.outages.fetch_sub(1, Ordering::SeqCst);
                return Err(boom(&format!("{} is down for now", self.label)));
            }
            match &self.chunks {
                None => Err(boom(&format!("{} is down", self.label))),
                Some(c) => Ok(Box::new(VecStream {
                    left: c.iter().cloned().collect(),
                    fail_after: self.fail_after,
                }) as LlmStream),
            }
        })
    }
}

fn llm_chain(entries: Vec<Arc<dyn Llm>>) -> FallbackLlm {
    let (clock, spawner, obs) = parts();
    FallbackLlm::new(entries, ChainPolicy::llm_defaults(), clock, spawner, obs).expect("entries")
}

#[tokio::test(start_paused = true)]
async fn the_retrying_chat_entry_does_not_retry_a_fallback_chain_twice() {
    let primary = FakeLlm::new("primary", None);
    let backup = FakeLlm::new("backup", None);
    let (_, spawner, observer) = parts();
    let clock: Arc<dyn lk_agents_abi::Clock> = Arc::new(lk_agents_net::clock::SimClock::new());
    let policy = ChainPolicy { max_retry_per_backend: 1, ..ChainPolicy::llm_defaults() };
    let chain = FallbackLlm::new(
        vec![primary.clone(), backup.clone()],
        policy,
        clock.clone(),
        spawner,
        observer,
    )
    .unwrap();
    let mut stream = crate::llm::chat(
        Arc::new(chain),
        &crate::telemetry::Cx::NONE,
        &ChatRequest::default(),
        clock,
        None,
    )
    .await
    .unwrap();
    assert!(stream.next().await.is_err());
    // Each backend gets two attempts and their two recovery probes. A replay of the chain
    // would start another set after these failures.
    until("the recovery probes to finish", || {
        primary.seen.lock().len() == 4 && backup.seen.lock().len() == 4
    })
    .await;
    assert_eq!(primary.seen.lock().len(), 4);
    assert_eq!(backup.seen.lock().len(), 4);
    stream.close(NO_DEADLINE).await;
}

// Catches: `sticky` going back to a recovered primary instead of the entry that served
// (`llm:96-105`), or the default leaving a recovered primary behind.
#[tokio::test]
async fn a_sticky_chain_keeps_the_entry_that_served_after_the_primary_recovers() {
    for sticky in [false, true] {
        let primary = FakeLlm::new("p", Some(vec![chunk("primary")]));
        primary.outages.store(1, Ordering::SeqCst);
        let backup = FakeLlm::new("b", Some(vec![chunk("backup")]));
        let (clock, spawner, obs) = parts();
        let policy = ChainPolicy { sticky, ..ChainPolicy::llm_defaults() };
        let fb: FallbackLlm = FallbackLlm::new(
            vec![primary.clone(), backup.clone()],
            policy,
            clock.clone(),
            spawner,
            obs,
        )
        .expect("entries");

        let text = |chunks: Vec<LlmChunk>| chunks[0].delta.as_ref().unwrap().content.clone();
        let mut stream =
            within("chat", fb.chat(&lk_agents_abi::telemetry::Cx::NONE, &ChatRequest::default()))
                .await
                .expect("served");
        assert_eq!(text(drain_llm(&mut stream).await).as_deref(), Some("backup"));
        until("the primary's probe to bring it back", || fb.chain().is_available(0)).await;

        let mut stream =
            within("chat", fb.chat(&lk_agents_abi::telemetry::Cx::NONE, &ChatRequest::default()))
                .await
                .expect("served");
        let want = if sticky { "backup" } else { "primary" };
        assert_eq!(text(drain_llm(&mut stream).await).as_deref(), Some(want), "sticky={sticky}");
        fb.aclose();
    }
}

fn chunk(text: &str) -> LlmChunk {
    LlmChunk {
        id: "c".into(),
        delta: Some(ChoiceDelta { content: Some(text.into()), ..Default::default() }),
        ..Default::default()
    }
}

// Catches: the chain forwarding the caller's `conn_options` unchanged (N x M retries).
#[tokio::test]
async fn a_wrapped_llm_receives_the_chains_conn_options_not_the_callers() {
    let primary = FakeLlm::new("p", None);
    let backup = FakeLlm::new("b", Some(vec![chunk("hi")]));
    let fb = llm_chain(vec![primary.clone(), backup.clone()]);

    let req = ChatRequest { conn_options: ApiConnectOptions::default(), ..Default::default() };
    let mut stream = within("chat", fb.chat(&lk_agents_abi::telemetry::Cx::NONE, &req))
        .await
        .expect("the adapter hands back a stream");
    let first = within("the first chunk", stream.next()).await.expect("no error");
    assert_eq!(first.expect("a chunk").delta.expect("delta").content.as_deref(), Some("hi"));

    // A probe also goes through `open`; assert on the first call.
    for who in [&primary, &backup] {
        let seen = who.seen.lock().clone();
        let first = *seen.first().unwrap_or_else(|| panic!("{} was never called", who.label));
        assert_eq!(first.max_retry, 0, "max_retry_per_llm defaults to 0");
        assert_eq!(first.read_timeout, Some(5.0), "attempt_timeout caps each phase");
        assert_eq!(first.total_timeout, None, "the caller's total stays the operation's");
        assert_eq!(first.retry_interval, 0.5);
    }
    assert_eq!(backup.seen.lock().len(), 1, "the backup served and was not probed");
    assert_eq!(fb.active().label(), "b");
    stream.close(NO_DEADLINE).await;
}

/// Metadata-only chunks (an empty frame, a role-only delta, usage) are not output, so a primary
/// that sends only those and dies falls over to the backup, whose tool call arrives once.
/// https://github.com/livekit/agents/issues/1207 https://github.com/livekit/agents/issues/5660
#[tokio::test]
async fn a_contentless_chunk_does_not_block_the_failover() {
    let bare = LlmChunk { id: "p".into(), ..Default::default() };
    let role_only = LlmChunk {
        id: "p".into(),
        delta: Some(ChoiceDelta {
            role: Some(lk_agents_abi::Role::Assistant),
            ..Default::default()
        }),
        ..Default::default()
    };
    let usage = LlmChunk { id: "p".into(), usage: Some(Box::default()), ..Default::default() };
    let tool_call = LlmChunk {
        id: "b".into(),
        delta: Some(ChoiceDelta {
            tool_calls: vec![lk_agents_abi::FunctionToolCall {
                name: "lookup".into(),
                arguments: r#"{"q":"x"}"#.into(),
                call_id: "call_1".into(),
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let primary = FakeLlm::dying("p", vec![bare, role_only, usage]);
    let backup = FakeLlm::new("b", Some(vec![tool_call]));
    let fb = llm_chain(vec![primary, backup]);

    let mut stream =
        within("chat", fb.chat(&lk_agents_abi::telemetry::Cx::NONE, &ChatRequest::default()))
            .await
            .expect("a stream");
    let calls: Vec<String> = drain_llm(&mut stream)
        .await
        .into_iter()
        .flat_map(|c| c.delta.map(|d| d.tool_calls).unwrap_or_default())
        .map(|call| call.call_id)
        .collect();

    assert_eq!(calls, ["call_1"]);
    stream.close(NO_DEADLINE).await;
}

// Catches: failing over after real content.
#[tokio::test]
async fn a_content_chunk_does_block_the_failover() {
    let primary = FakeLlm::dying("p", vec![chunk("Hello the")]);
    let backup = FakeLlm::new("b", Some(vec![chunk("Hello there")]));
    let fb = llm_chain(vec![primary, backup.clone()]);

    let mut stream =
        within("chat", fb.chat(&lk_agents_abi::telemetry::Cx::NONE, &ChatRequest::default()))
            .await
            .expect("a stream");
    assert!(within("the first chunk", stream.next()).await.expect("no error").is_some());
    let err = within("the failure", stream.next()).await.expect_err("the turn must abort");
    assert!(err.message().contains("stream died"), "{}", err.message());
    assert!(backup.seen.lock().is_empty(), "the backup must never be opened");
    stream.close(NO_DEADLINE).await;
}

/// An `Llm` whose stream never ends, recording when its chain task is dropped.
struct HangingLlm {
    opened: Arc<AtomicBool>,
    dropped: Arc<AtomicBool>,
}

struct HangingStream(Arc<AtomicBool>);

impl StreamOutput<LlmChunk> for HangingStream {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<LlmChunk>, ProviderError>> {
        Box::pin(async move { std::future::pending().await })
    }
    fn close(self: Box<Self>, _deadline: Ts) -> BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<LlmChunk> for HangingStream {}

impl Drop for HangingStream {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

impl Llm for HangingLlm {
    fn label(&self) -> String {
        "hanging".into()
    }
    fn chat<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _req: &'a ChatRequest,
    ) -> lk_agents_abi::BoxFuture<'a, Result<LlmStream, ProviderError>> {
        Box::pin(async move {
            self.opened.store(true, Ordering::SeqCst);
            Ok(Box::new(HangingStream(Arc::clone(&self.dropped))) as LlmStream)
        })
    }
}

// Catches: an exhaustion error not naming the providers, or the adapter borrowing a child's label.
#[tokio::test]
async fn a_total_outage_names_every_provider_and_the_adapter_names_itself() {
    let primary = FakeLlm::new("openai", None);
    let backup = FakeLlm::new("anthropic", None);
    let fb = llm_chain(vec![primary, backup]);

    assert_eq!(fb.label(), "llm.FallbackAdapter");

    let mut stream =
        within("chat", fb.chat(&lk_agents_abi::telemetry::Cx::NONE, &ChatRequest::default()))
            .await
            .expect("a stream");
    let err = within("the failure", stream.next()).await.expect_err("every entry is down");
    assert!(
        err.message().contains("all backends failed (openai, anthropic)"),
        "the message must name the list, in order: {}",
        err.message()
    );
    stream.close(NO_DEADLINE).await;
    fb.aclose();
}

// Catches: marking the serving entry on success rather than on its first chunk (`llm:200-205`).
#[tokio::test]
async fn the_llm_that_emitted_is_the_one_metrics_are_labelled_with() {
    let primary = FakeLlm::new("p", None); // `chat()` raises: nothing is ever emitted
    let backup = FakeLlm::dying("b", vec![chunk("half a sen")]);
    let fb = llm_chain(vec![primary, backup]);

    assert_eq!(fb.active().label(), "p", "before any traffic the primary is active");

    let mut stream =
        within("chat", fb.chat(&lk_agents_abi::telemetry::Cx::NONE, &ChatRequest::default()))
            .await
            .expect("a stream");
    assert!(within("the first chunk", stream.next()).await.expect("no error").is_some());
    let err = within("the failure", stream.next()).await.expect_err("the backup died mid-stream");
    assert!(err.message().contains("stream died"), "{}", err.message());

    assert_eq!(fb.active().label(), "b", "the entry that emitted owns the turn's metrics");
    stream.close(NO_DEADLINE).await;
    fb.aclose();
}

// Catches: an adapter `aclose` that does not reach the chain (`tts:155-159`), leaving later probes
// uncancelled.
#[tokio::test]
async fn a_closed_llm_adapter_starts_no_more_recovery_probes() {
    let primary = FakeLlm::new("p", None);
    let backup = FakeLlm::new("b", Some(vec![chunk("hi")]));
    let fb = llm_chain(vec![primary.clone(), backup]);

    fb.aclose();

    let mut stream =
        within("chat", fb.chat(&lk_agents_abi::telemetry::Cx::NONE, &ChatRequest::default()))
            .await
            .expect("a stream");
    assert!(within("closed refusal", stream.next()).await.is_err());
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }

    assert_eq!(primary.seen.lock().len(), 0, "close prevents every new attempt and probe");
    stream.close(NO_DEADLINE).await;
}

// Catches: the returned stream's `aclose` not aborting the task filling it.
#[tokio::test]
async fn closing_the_llm_stream_aborts_the_chain_task_and_releases_the_provider() {
    let opened = Arc::new(AtomicBool::new(false));
    let dropped = Arc::new(AtomicBool::new(false));
    let llm = Arc::new(HangingLlm { opened: Arc::clone(&opened), dropped: Arc::clone(&dropped) });
    let fb = llm_chain(vec![llm]);

    let stream =
        within("chat", fb.chat(&lk_agents_abi::telemetry::Cx::NONE, &ChatRequest::default()))
            .await
            .expect("a stream");
    until("the provider's generation to be open", || opened.load(Ordering::SeqCst)).await;
    assert!(!dropped.load(Ordering::SeqCst), "it is still running");

    stream.close(NO_DEADLINE).await;
    until("the chain task to be aborted and the generation released", || {
        dropped.load(Ordering::SeqCst)
    })
    .await;
}

// Catches: `FallbackLlm` inheriting `Llm::model()`'s `"unknown"` default instead of following the
// active entry (`_evaluate_with_llm` excludes `temperature` on gpt-5).
#[tokio::test]
async fn the_chain_reports_the_active_entrys_model_not_its_own() {
    // The primary raises on `chat()`, so the failover is real.
    let primary = FakeLlm::build("p", Some("gpt-5"), None, false);
    let backup = FakeLlm::build("b", Some("gpt-4o-mini"), Some(vec![chunk("hi")]), false);
    let fb = llm_chain(vec![primary.clone(), backup.clone()]);

    assert_eq!(fb.model(), "gpt-5", "before any traffic the primary's model is the answer");

    let mut stream =
        within("chat", fb.chat(&lk_agents_abi::telemetry::Cx::NONE, &ChatRequest::default()))
            .await
            .expect("the backup serves the turn");
    while within("next", stream.next()).await.expect("drain").is_some() {}

    assert_eq!(fb.model(), "gpt-4o-mini", "after a failover it must follow the active entry");
}
