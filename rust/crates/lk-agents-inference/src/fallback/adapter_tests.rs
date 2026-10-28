//! End-to-end over the three modality adapters: what a wrapped provider receives and what comes
//! out. The chain's own machine is pinned in `tests.rs`.

use async_trait::async_trait;
use std::collections::VecDeque;

#[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
use std::sync::{Arc, Mutex};

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::tokenize::language::LanguageCode;
use lk_agents_abi::{
    AlignedTranscript, AudioBuffer, AudioFrame, BoxFuture, ChatRequest, Closed, Llm, LlmStream,
    StreamSink, StreamSource, Stt, SttCapabilities, SttSink, SttSource, Ts, TtsCapabilities,
};
use lk_agents_abi::{
    ApiConnectOptions, ChoiceDelta, LlmChunk, SpeechData, SpeechEvent, SpeechEventType,
};
use tokio::sync::mpsc;

use crate::fallback::llm::FallbackLlm;
use crate::fallback::policy::ChainPolicy;
use crate::fallback::stt::{FallbackStt, SttChainError};
use lk_agents_net::env::Standalone;

use crate::testkit::fallback::{NO_DEADLINE, boom, parts, until, within};

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

// ---------------------------------------------------------------------------------------------
// LLM
// ---------------------------------------------------------------------------------------------

struct FakeLlm {
    label: String,
    /// `None` leaves the trait default (`"unknown"`).
    model: Option<String>,
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    seen: Mutex<Vec<ApiConnectOptions>>,
    /// `None` means `chat()` itself raises.
    chunks: Option<Vec<LlmChunk>>,
    /// The stream dies after the scripted chunks instead of ending cleanly.
    fail_after: bool,
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    /// `chat()` raises this many more times before it serves.
    outages: AtomicUsize,
}

impl FakeLlm {
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
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

impl StreamSource<LlmChunk> for VecStream {
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

#[async_trait]
impl Llm for FakeLlm {
    fn label(&self) -> String {
        self.label.clone()
    }
    fn model(&self) -> String {
        self.model.clone().unwrap_or_else(|| "unknown".into())
    }
    async fn chat(&self, req: &ChatRequest) -> Result<LlmStream, ProviderError> {
        self.seen.lock().expect("seen").push(req.conn_options);
        if self.outages.load(Ordering::SeqCst) > 0 {
            self.outages.fetch_sub(1, Ordering::SeqCst);
            return Err(boom(&format!("{} is down for now", self.label)));
        }
        match &self.chunks {
            None => Err(boom(&format!("{} is down", self.label))),
            Some(c) => Ok(Box::new(VecStream {
                left: c.iter().cloned().collect(),
                fail_after: self.fail_after,
            })),
        }
    }
}

fn llm_chain(entries: Vec<Arc<dyn Llm>>) -> FallbackLlm {
    let (clock, spawner, obs) = parts();
    FallbackLlm::new(entries, ChainPolicy::llm_defaults(), clock, spawner, obs).expect("entries")
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
        let fb: FallbackLlm =
            FallbackLlm::new(vec![primary.clone(), backup.clone()], policy, clock, spawner, obs)
                .expect("entries");

        let text = |chunks: Vec<LlmChunk>| chunks[0].delta.as_ref().unwrap().content.clone();
        let mut stream = within("chat", fb.chat(&ChatRequest::default())).await.expect("served");
        assert_eq!(text(drain_llm(&mut stream).await).as_deref(), Some("backup"));
        until("the primary's probe to bring it back", || fb.chain().is_available(0)).await;

        let mut stream = within("chat", fb.chat(&ChatRequest::default())).await.expect("served");
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
    let mut stream = within("chat", fb.chat(&req)).await.expect("the adapter hands back a stream");
    let first = within("the first chunk", stream.next()).await.expect("no error");
    assert_eq!(first.expect("a chunk").delta.expect("delta").content.as_deref(), Some("hi"));

    // A probe also goes through `open`; assert on the first call.
    for who in [&primary, &backup] {
        let seen = who.seen.lock().expect("seen").clone();
        let first = *seen.first().unwrap_or_else(|| panic!("{} was never called", who.label));
        assert_eq!(first.max_retry, 0, "max_retry_per_llm defaults to 0");
        assert_eq!(first.read_timeout, Some(5.0), "attempt_timeout caps each phase");
        assert_eq!(first.total_timeout, None, "the caller's total stays the operation's");
        assert_eq!(first.retry_interval, 0.5);
    }
    assert_eq!(backup.seen.lock().expect("seen").len(), 1, "the backup served and was not probed");
    assert_eq!(fb.active().label(), "b");
    stream.close(NO_DEADLINE).await;
}

// Catches: counting a content-less chunk (the usage frame) as output sent (`llm:281-285`).
#[tokio::test]
async fn a_contentless_chunk_does_not_block_the_failover() {
    let bare = LlmChunk { id: "u".into(), ..Default::default() };
    let primary = FakeLlm::dying("p", vec![bare]);
    let backup = FakeLlm::new("b", Some(vec![chunk("recovered")]));
    let fb = llm_chain(vec![primary, backup.clone()]);

    let mut stream = within("chat", fb.chat(&ChatRequest::default())).await.expect("a stream");
    let texts: Vec<String> = drain_llm(&mut stream)
        .await
        .into_iter()
        .map(|c| c.delta.and_then(|d| d.content).unwrap_or_default())
        .collect();
    assert_eq!(texts, vec![String::new(), "recovered".to_string()]);
    stream.close(NO_DEADLINE).await;
}

// Catches: failing over after real content.
#[tokio::test]
async fn a_content_chunk_does_block_the_failover() {
    let primary = FakeLlm::dying("p", vec![chunk("Hello the")]);
    let backup = FakeLlm::new("b", Some(vec![chunk("Hello there")]));
    let fb = llm_chain(vec![primary, backup.clone()]);

    let mut stream = within("chat", fb.chat(&ChatRequest::default())).await.expect("a stream");
    assert!(within("the first chunk", stream.next()).await.expect("no error").is_some());
    let err = within("the failure", stream.next()).await.expect_err("the turn must abort");
    assert!(err.message().contains("stream died"), "{}", err.message());
    assert!(backup.seen.lock().expect("seen").is_empty(), "the backup must never be opened");
    stream.close(NO_DEADLINE).await;
}

/// An `Llm` whose stream never ends, recording when its chain task is dropped.
struct HangingLlm {
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    opened: Arc<AtomicBool>,
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    dropped: Arc<AtomicBool>,
}

#[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
struct HangingStream(Arc<AtomicBool>);

impl StreamSource<LlmChunk> for HangingStream {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<LlmChunk>, ProviderError>> {
        Box::pin(async move { std::future::pending().await })
    }
    fn close(self: Box<Self>, _deadline: Ts) -> BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}

impl Drop for HangingStream {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[async_trait]
impl Llm for HangingLlm {
    fn label(&self) -> String {
        "hanging".into()
    }
    async fn chat(&self, _req: &ChatRequest) -> Result<LlmStream, ProviderError> {
        self.opened.store(true, Ordering::SeqCst);
        Ok(Box::new(HangingStream(Arc::clone(&self.dropped))))
    }
}

// Catches: an exhaustion error not naming the providers, or the adapter borrowing a child's label.
#[tokio::test]
async fn a_total_outage_names_every_provider_and_the_adapter_names_itself() {
    let primary = FakeLlm::new("openai", None);
    let backup = FakeLlm::new("anthropic", None);
    let fb = llm_chain(vec![primary, backup]);

    assert_eq!(fb.label(), "llm.FallbackAdapter");

    let mut stream = within("chat", fb.chat(&ChatRequest::default())).await.expect("a stream");
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

    let mut stream = within("chat", fb.chat(&ChatRequest::default())).await.expect("a stream");
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

    let mut stream = within("chat", fb.chat(&ChatRequest::default())).await.expect("a stream");
    assert!(within("closed refusal", stream.next()).await.is_err());
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }

    assert_eq!(
        primary.seen.lock().expect("seen").len(),
        0,
        "close prevents every new attempt and probe"
    );
    stream.close(NO_DEADLINE).await;
}

// Catches: the returned stream's `aclose` not aborting the task filling it.
#[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
#[tokio::test]
async fn closing_the_llm_stream_aborts_the_chain_task_and_releases_the_provider() {
    let opened = Arc::new(AtomicBool::new(false));
    let dropped = Arc::new(AtomicBool::new(false));
    let llm = Arc::new(HangingLlm { opened: Arc::clone(&opened), dropped: Arc::clone(&dropped) });
    let fb = llm_chain(vec![llm]);

    let stream = within("chat", fb.chat(&ChatRequest::default())).await.expect("a stream");
    until("the provider's generation to be open", || opened.load(Ordering::SeqCst)).await;
    assert!(!dropped.load(Ordering::SeqCst), "it is still running");

    stream.close(NO_DEADLINE).await;
    until("the chain task to be aborted and the generation released", || {
        dropped.load(Ordering::SeqCst)
    })
    .await;
}

// ---------------------------------------------------------------------------------------------
// STT
// ---------------------------------------------------------------------------------------------

struct FakeStt {
    label: String,
    conn: ApiConnectOptions,
    fails: bool,
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    calls: Arc<Mutex<Vec<(String, ApiConnectOptions)>>>,
}

impl Stt for FakeStt {
    fn label(&self) -> String {
        self.label.clone()
    }
    /// The view the chain binds: the same provider, recording the options it was built for.
    fn with_conn_options(&self, conn: &ApiConnectOptions) -> Option<Arc<dyn Stt>> {
        Some(Arc::new(FakeStt {
            label: self.label.clone(),
            conn: *conn,
            fails: self.fails,
            calls: Arc::clone(&self.calls),
        }))
    }
    fn capabilities(&self) -> SttCapabilities {
        SttCapabilities { streaming: true, ..Default::default() }
    }
    fn recognize(
        &self,
        _buffer: AudioBuffer<'_>,
        _language: Option<&LanguageCode>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
        Box::pin(async move {
            self.calls.lock().expect("calls").push((self.label.clone(), self.conn));
            if self.fails {
                return Err(boom("stt down"));
            }
            Ok(SpeechEvent::new(SpeechEventType::FinalTranscript))
        })
    }
    fn stream(
        &self,
        _language: Option<&LanguageCode>,
        _start_time_offset: Option<f64>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<(SttSink, SttSource)>, ProviderError>> {
        Box::pin(async move { Ok(None) })
    }
}

// Catches: an unbounded STT retry loop under the chain; also pins probe 0 vs attempt 1 retries.
#[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
#[tokio::test]
async fn a_wrapped_stt_never_runs_an_independent_retry_loop() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let fake = |label: &str, fails: bool| -> Arc<dyn Stt> {
        let conn = ApiConnectOptions::default();
        Arc::new(FakeStt { label: label.into(), conn, fails, calls: Arc::clone(&calls) })
    };
    let (clock, spawner, obs) = parts();
    let entries = vec![fake("p", true), fake("b", false)];

    let fb = FallbackStt::<Standalone>::new(
        entries,
        ChainPolicy {
            max_retry_per_backend: 0,
            retry_interval: Some(0.0),
            ..ChainPolicy::stt_defaults()
        },
        clock,
        spawner,
        obs,
    )
    .expect("two streaming entries");

    let frame = AudioFrame::new(vec![0u8; 4], 16_000, 1, 2);
    let ev = within("recognize", fb.recognize(&[frame], None)).await.expect("the backup answers");
    assert_eq!(ev.kind, SpeechEventType::FinalTranscript);

    until("the primary's probe to have run", || {
        calls.lock().expect("calls").iter().filter(|(l, _)| l == "p").count() >= 2
    })
    .await;

    let seen = calls.lock().expect("calls").clone();
    let p: Vec<u32> = seen.iter().filter(|(l, _)| l == "p").map(|(_, c)| c.max_retry).collect();
    assert_eq!(p[0], 0, "the chain owns retry attempts");
    assert_eq!(p[1], 0, "the probe pins it to 0");
    // A listening stream outlives one attempt's timeout: it bounds phases, never the stream.
    let primary = seen.iter().find(|(l, _)| l == "p").expect("primary").1;
    assert_eq!((primary.connect_timeout, primary.total_timeout), (Some(10.0), None));
    fb.aclose();
}

// --- the duplex STT the fallback adapter actually runs ----------------------------------------
// Exercises `FallbackStt::stream`, `InputTee` and `TeeSink` (`stt/fallback_adapter.py:341-361`).

#[derive(Default)]
struct SttSess {
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    frames: Mutex<Vec<AudioFrame>>,
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    flushes: AtomicUsize,
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    ended: AtomicBool,
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    closed: AtomicBool,
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    /// Events this session handed to whoever was draining it.
    emitted: AtomicUsize,
}

struct SttSessSink(Arc<SttSess>);

impl StreamSink<AudioFrame> for SttSessSink {
    fn push(&self, item: AudioFrame) -> Result<(), ProviderError> {
        self.0.frames.lock().expect("frames").push(item);
        Ok(())
    }
    fn flush(&self) -> Result<(), ProviderError> {
        self.0.flushes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn end_input(&self) {
        self.0.ended.store(true, Ordering::SeqCst);
    }
}

struct SttSessSource {
    sess: Arc<SttSess>,
    #[allow(clippy::disallowed_types, reason = "S0: test harness runtime resource")]
    rx: mpsc::UnboundedReceiver<Result<SpeechEvent, ProviderError>>,
}

impl StreamSource<SpeechEvent> for SttSessSource {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<SpeechEvent>, ProviderError>> {
        Box::pin(async move {
            match self.rx.recv().await {
                None => Ok(None),
                Some(Ok(ev)) => {
                    self.sess.emitted.fetch_add(1, Ordering::SeqCst);
                    Ok(Some(ev))
                }
                Some(Err(e)) => Err(e),
            }
        })
    }
    fn close(mut self: Box<Self>, _deadline: Ts) -> BoxFuture<'static, Closed> {
        Box::pin(async move {
            self.sess.closed.store(true, Ordering::SeqCst);
            self.rx.close();
            Closed::Clean
        })
    }
}

#[allow(clippy::disallowed_types, reason = "S0: test harness runtime resource")]
type SttSender = mpsc::UnboundedSender<Result<SpeechEvent, ProviderError>>;

struct DuplexStt {
    label: String,
    streaming: bool,
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    /// `stream()` refuses this many times before it works.
    refusals: AtomicUsize,
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    made: Mutex<Vec<(Arc<SttSess>, Option<SttSender>)>>,
}

impl DuplexStt {
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    fn new(label: &str, refusals: usize) -> Arc<Self> {
        Arc::new(DuplexStt {
            label: label.into(),
            streaming: true,
            refusals: AtomicUsize::new(refusals),
            made: Mutex::new(Vec::new()),
        })
    }
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    fn batch_only(label: &str) -> Arc<Self> {
        Arc::new(DuplexStt {
            label: label.into(),
            streaming: false,
            refusals: AtomicUsize::new(0),
            made: Mutex::new(Vec::new()),
        })
    }
    fn sessions(&self) -> usize {
        self.made.lock().expect("made").len()
    }
    fn sess(&self, i: usize) -> Arc<SttSess> {
        Arc::clone(&self.made.lock().expect("made")[i].0)
    }
    fn send(&self, i: usize, ev: Result<SpeechEvent, ProviderError>) {
        let made = self.made.lock().expect("made");
        made[i].1.as_ref().expect("session is still open").send(ev).expect("nobody dropped it");
    }
    /// The provider's own end of stream.
    fn end(&self, i: usize) {
        self.made.lock().expect("made")[i].1 = None;
    }
}

impl Stt for DuplexStt {
    fn label(&self) -> String {
        self.label.clone()
    }
    fn capabilities(&self) -> SttCapabilities {
        SttCapabilities { streaming: self.streaming, interim_results: true, ..Default::default() }
    }
    fn recognize(
        &self,
        _buffer: AudioBuffer<'_>,
        _language: Option<&LanguageCode>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
        Box::pin(async move { Err(boom("this fake only has a streaming path")) })
    }
    #[allow(clippy::disallowed_methods, reason = "S0: existing test harness live resources")]
    fn stream(
        &self,
        _language: Option<&LanguageCode>,
        _start_time_offset: Option<f64>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<(SttSink, SttSource)>, ProviderError>> {
        Box::pin(async move {
            let left = self.refusals.load(Ordering::SeqCst);
            if left > 0 {
                self.refusals.store(left - 1, Ordering::SeqCst);
                return Err(boom(&format!("{} has no socket", self.label)));
            }
            let sess = Arc::new(SttSess::default());
            let (tx, rx) = mpsc::unbounded_channel();
            self.made.lock().expect("made").push((Arc::clone(&sess), Some(tx)));
            Ok(Some((
                Box::new(SttSessSink(Arc::clone(&sess))) as SttSink,
                Box::new(SttSessSource { sess, rx }) as SttSource,
            )))
        })
    }
}

fn speech(kind: SpeechEventType, text: &str) -> SpeechEvent {
    let mut ev = SpeechEvent::new(kind);
    ev.alternatives = vec![SpeechData::new("en", text)];
    ev
}

fn frame(seq: u8) -> AudioFrame {
    AudioFrame::new(vec![seq; 4], 16_000, 1, 2)
}

fn stt_chain(entries: Vec<Arc<DuplexStt>>) -> FallbackStt {
    let (clock, spawner, obs) = parts();
    FallbackStt::new(
        entries.into_iter().map(|e| e as Arc<dyn Stt>).collect(),
        ChainPolicy {
            max_retry_per_backend: 0,
            retry_interval: Some(0.0),
            ..ChainPolicy::stt_defaults()
        },
        clock,
        spawner,
        obs,
    )
    .expect("every entry streams")
}

// Catches: the consumer's `close` aborting the chain task before it closes the serving member.
#[tokio::test]
async fn closing_the_fallback_stream_early_closes_the_serving_member() {
    let a = DuplexStt::new("a", 0);
    let fb = stt_chain(vec![Arc::clone(&a)]);
    let (_sink, mut source) =
        within("stream", fb.stream(None, None)).await.expect("opened").expect("duplex");
    until("the member to be serving", || a.sessions() == 1).await;
    a.send(0, Ok(speech(SpeechEventType::InterimTranscript, "hel")));
    let first =
        within("the first event", source.next()).await.expect("no error").expect("an event");
    assert_eq!(first.kind, SpeechEventType::InterimTranscript);
    assert!(
        !a.sess(0).closed.load(Ordering::SeqCst),
        "the member is live while the consumer reads"
    );

    let closed = within("the close", source.close(fb.chain().teardown_deadline())).await;
    assert_eq!(closed, Closed::Clean, "the task honoured the stop and reported done");
    assert!(
        a.sess(0).closed.load(Ordering::SeqCst),
        "the serving member's stream was closed, not dropped"
    );
}

// Catches: an audio-in leg that goes nowhere (`set_main`, `fanout`, `push`, `flush`,
// `end_input`), or a late push reaching a released provider after `clear_main`.
#[tokio::test]
async fn the_streaming_path_wires_the_callers_audio_into_the_provider_it_opened() {
    let p = DuplexStt::new("p", 0);
    let fb = stt_chain(vec![Arc::clone(&p)]);

    assert_eq!(fb.label(), "stt.FallbackAdapter");

    let (sink, mut source) = within("stream", fb.stream(None, None))
        .await
        .expect("opened")
        .expect("the adapter is duplex");
    until("the provider's session to be open", || p.sessions() == 1).await;
    let s = p.sess(0);

    sink.push(frame(1)).expect("push");
    sink.push(frame(2)).expect("push");
    sink.flush().expect("flush");
    sink.end_input();

    assert_eq!(s.frames.lock().expect("frames").len(), 2, "every frame reaches the provider");
    assert_eq!(s.flushes.load(Ordering::SeqCst), 1, "a flush is a segment boundary, not a close");
    assert!(s.ended.load(Ordering::SeqCst), "end_input reaches the stream that is serving");

    p.send(0, Ok(speech(SpeechEventType::FinalTranscript, "hello")));
    let ev =
        within("the transcript", source.next()).await.expect("no error").expect("one transcript");
    assert_eq!(ev.best_alternative().expect("an alternative").text, "hello");

    p.end(0);
    assert!(within("the end of the stream", source.next()).await.expect("no error").is_none());

    sink.push(frame(3)).expect("push");
    assert_eq!(
        s.frames.lock().expect("frames").len(),
        2,
        "the turn is over: audio must not reach a provider the chain has released"
    );
    source.close(NO_DEADLINE).await;
    fb.aclose();
}

// Catches: the streaming probe never joining the tee, recovering on an interim or empty final
// (`stt:456-466`), or staying in the tee after it finishes.
#[tokio::test]
async fn a_streaming_probe_shadows_the_live_audio_and_only_a_real_transcript_is_recovery() {
    let p = DuplexStt::new("p", 1); // its first `stream()` refuses; the probe's succeeds
    let b = DuplexStt::new("b", 0);
    let fb = stt_chain(vec![Arc::clone(&p), Arc::clone(&b)]);

    let (sink, source) =
        within("stream", fb.stream(None, None)).await.expect("opened").expect("duplex");
    until("the backup to be serving", || b.sessions() == 1).await;
    until("the primary's probe to join the tee", || p.sessions() == 1).await;
    assert!(!fb.chain().is_available(0), "the primary failed to open");

    sink.push(frame(1)).expect("push");
    assert_eq!(
        p.sess(0).frames.lock().expect("frames").len(),
        1,
        "the probe shadows the live audio -- there is nothing else for it to listen to"
    );
    assert_eq!(b.sess(0).frames.lock().expect("frames").len(), 1, "and the serving stream gets it");

    p.send(0, Ok(speech(SpeechEventType::InterimTranscript, "hel")));
    p.send(0, Ok(speech(SpeechEventType::FinalTranscript, "")));
    p.send(0, Ok(speech(SpeechEventType::FinalTranscript, "hello")));

    until("the primary to be marked recovered", || fb.chain().is_available(0)).await;
    assert_eq!(
        p.sess(0).emitted.load(Ordering::SeqCst),
        3,
        "neither the interim result nor the empty final is a recovery; only the third event is"
    );
    assert!(p.sess(0).closed.load(Ordering::SeqCst), "a finished probe closes its own stream");

    sink.push(frame(2)).expect("push");
    assert_eq!(
        p.sess(0).frames.lock().expect("frames").len(),
        1,
        "a finished probe has left the tee and stops consuming the user's audio"
    );
    assert_eq!(b.sess(0).frames.lock().expect("frames").len(), 2);
    source.close(NO_DEADLINE).await;
    fb.aclose();
}

// Catches: the streaming drain never marking the serving entry, or an exhaustion error not naming
// the STTs.
#[tokio::test]
async fn the_stt_that_produced_transcripts_owns_the_turn_and_the_failure_names_the_list() {
    let p = DuplexStt::new("p", 1);
    let b = DuplexStt::new("b", 0);
    let fb = stt_chain(vec![Arc::clone(&p), Arc::clone(&b)]);

    assert_eq!(fb.active().label(), "p", "before any traffic the primary is active");

    let (sink, mut source) =
        within("stream", fb.stream(None, None)).await.expect("opened").expect("duplex");
    until("the backup to be serving", || b.sessions() == 1).await;

    b.send(0, Ok(speech(SpeechEventType::FinalTranscript, "hello")));
    let ev = within("the transcript", source.next()).await.expect("no error").expect("an event");
    assert_eq!(ev.best_alternative().expect("an alternative").text, "hello");

    b.send(0, Err(boom("socket died")));
    let err = within("the failure", source.next()).await.expect_err("nothing is left to try");
    assert!(
        err.message().contains("socket died"),
        "the failure after a delivered transcript must remain explicit: {}",
        err.message()
    );
    assert_eq!(fb.active().label(), "b", "the entry that transcribed owns the turn's metrics");

    let _ = sink;
    source.close(NO_DEADLINE).await;
    fb.aclose();
}

// Catches: an STT adapter `aclose` that does not reach the chain.
#[tokio::test]
async fn a_closed_stt_adapter_starts_no_more_recovery_probes() {
    let p = DuplexStt::new("p", 1);
    let b = DuplexStt::new("b", 0);
    let fb = stt_chain(vec![Arc::clone(&p), Arc::clone(&b)]);

    fb.aclose();

    let (sink, mut source) =
        within("stream", fb.stream(None, None)).await.expect("opened").expect("duplex");
    assert!(within("closed refusal", source.next()).await.is_err());
    assert_eq!(b.sessions(), 0);
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }

    assert_eq!(
        p.sessions(),
        0,
        "the only attempt on the primary refused; a probe would have opened a second session"
    );
    let _ = sink;
    source.close(NO_DEADLINE).await;
}

// Catches: advertising `Default` capabilities instead of the merged list, or losing the
// `stream_adapter` guard.
#[tokio::test]
async fn the_stt_adapter_advertises_the_merged_capabilities_and_refuses_a_batch_only_entry() {
    let fb = stt_chain(vec![DuplexStt::new("p", 0), DuplexStt::new("b", 0)]);
    assert!(fb.capabilities().streaming);
    assert!(fb.capabilities().interim_results, "both entries have them, so the chain does");

    let (clock, spawner, obs) = parts();
    let err = FallbackStt::<Standalone>::new(
        vec![
            DuplexStt::new("good", 0) as Arc<dyn Stt>,
            DuplexStt::batch_only("whisper") as Arc<dyn Stt>,
        ],
        ChainPolicy {
            max_retry_per_backend: 0,
            retry_interval: Some(0.0),
            ..ChainPolicy::stt_defaults()
        },
        clock,
        spawner,
        obs,
    );
    match err {
        Err(e @ SttChainError::NotStreaming(_)) => assert_eq!(
            e.to_string(),
            "STTs do not support streaming: whisper. Wrap them with stt.StreamAdapter first."
        ),
        Err(other) => panic!("wrong error: {other:?}"),
        Ok(_) => panic!("a batch-only entry must be refused, not silently wrapped"),
    }
}

mod capabilities;
mod options;
