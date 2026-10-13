//! End-to-end over the three modality adapters: what a wrapped provider receives and what comes
//! out. The chain's own machine is pinned in `tests.rs`.

use std::collections::VecDeque;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use lk_agents_abi::{
    AlignedTranscript, AudioBuffer, AudioFrame, ChatRequest, Closed, Llm, LlmStream, StreamSink,
    StreamSource, Stt, SttCapabilities, SttSink, SttSource, SynthesizedAudio, Ts, Tts,
    TtsCapabilities, TtsSink, TtsSource,
};
use lk_agents_abi::{Spawner, TokioSpawner};
use lk_agents_abi::{Clock, VirtualClock};
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::tokenize::language::LanguageCode;
use lk_agents_abi::{
    ApiConnectOptions, ChoiceDelta, LlmChunk, SpeechData, SpeechEvent, SpeechEventType,
};
use tokio::sync::mpsc;

use crate::fallback::chain::{ChainObserver, IgnoreEvents};
use crate::fallback::llm::FallbackLlm;
use crate::fallback::policy::ChainPolicy;
use crate::fallback::stt::{FallbackStt, SttChainError};
use crate::fallback::tts::FallbackTts;

/// Every stream closed here is native and ignores the deadline.
const NO_DEADLINE: Ts = Ts::from_nanos(u64::MAX);

const LIMIT: Duration = Duration::from_secs(5);

async fn within<F: std::future::Future>(what: &str, f: F) -> F::Output {
    match tokio::time::timeout(LIMIT, f).await {
        Ok(v) => v,
        Err(_) => panic!("timed out after {LIMIT:?} waiting for {what}"),
    }
}

async fn until(what: &str, mut cond: impl FnMut() -> bool) {
    within(what, async {
        while !cond() {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await;
}

fn parts() -> (Arc<dyn Clock>, Arc<dyn Spawner>, Arc<dyn ChainObserver>) {
    (
        Arc::new(VirtualClock::new()) as Arc<dyn Clock>,
        Arc::new(TokioSpawner::current()) as Arc<dyn Spawner>,
        Arc::new(IgnoreEvents) as Arc<dyn ChainObserver>,
    )
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

async fn drain_tts(source: &mut TtsSource) -> Vec<SynthesizedAudio> {
    let mut out = Vec::new();
    for _ in 0..64 {
        match within("the next frame", source.next()).await {
            Ok(None) => return out,
            Ok(Some(a)) => out.push(a),
            Err(e) => panic!("the source failed: {e}"),
        }
    }
    panic!("the source never reported its end");
}

fn boom(what: &str) -> ProviderError {
    ProviderError::Local(ApiError::new(what.to_owned()))
}

// ---------------------------------------------------------------------------------------------
// LLM
// ---------------------------------------------------------------------------------------------

struct FakeLlm {
    label: String,
    /// `None` leaves the trait default (`"unknown"`).
    model: Option<String>,
    seen: Mutex<Vec<ApiConnectOptions>>,
    /// `None` means `chat()` itself raises.
    chunks: Option<Vec<LlmChunk>>,
    /// The stream dies after the scripted chunks instead of ending cleanly.
    fail_after: bool,
}

impl FakeLlm {
    fn build(label: &str, model: Option<&str>, chunks: Option<Vec<LlmChunk>>, fail_after: bool) -> Arc<Self> {
        let (label, model, seen) = (label.into(), model.map(Into::into), Mutex::new(vec![]));
        Arc::new(FakeLlm { label, model, seen, chunks, fail_after })
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

#[async_trait]
impl StreamSource<LlmChunk> for VecStream {
    async fn next(&mut self) -> Result<Option<LlmChunk>, ProviderError> {
        match self.left.pop_front() {
            Some(c) => Ok(Some(c)),
            None if self.fail_after => Err(boom("stream died")),
            None => Ok(None),
        }
    }
    async fn close(self: Box<Self>, _deadline: Ts) -> Closed {
        Closed::Clean
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

    let req = ChatRequest {
        conn_options: ApiConnectOptions { max_retry: 3, retry_interval: 2.0, timeout: 10.0 },
        ..Default::default()
    };
    let mut stream = within("chat", fb.chat(&req)).await.expect("the adapter hands back a stream");
    let first = within("the first chunk", stream.next()).await.expect("no error");
    assert_eq!(first.expect("a chunk").delta.expect("delta").content.as_deref(), Some("hi"));

    // A probe also goes through `open`; assert on the first call.
    for who in [&primary, &backup] {
        let seen = who.seen.lock().expect("seen").clone();
        let first = *seen.first().unwrap_or_else(|| panic!("{} was never called", who.label));
        assert_eq!(first.max_retry, 0, "max_retry_per_llm defaults to 0");
        assert_eq!(first.timeout, 5.0, "attempt_timeout, not the caller's 10.0");
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
    opened: Arc<AtomicBool>,
    dropped: Arc<AtomicBool>,
}

struct HangingStream(Arc<AtomicBool>);

#[async_trait]
impl StreamSource<LlmChunk> for HangingStream {
    async fn next(&mut self) -> Result<Option<LlmChunk>, ProviderError> {
        std::future::pending().await
    }
    async fn close(self: Box<Self>, _deadline: Ts) -> Closed {
        Closed::Clean
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
    drain_llm(&mut stream).await;
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }

    assert_eq!(
        primary.seen.lock().expect("seen").len(),
        1,
        "one real attempt and no probe: a probe would have opened the dead provider a second time"
    );
    stream.close(NO_DEADLINE).await;
}

// Catches: the returned stream's `aclose` not aborting the task filling it.
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
    calls: Arc<Mutex<Vec<(String, ApiConnectOptions)>>>,
}

#[async_trait]
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
    async fn recognize(
        &self,
        _buffer: AudioBuffer<'_>,
        _language: Option<&LanguageCode>,
    ) -> Result<SpeechEvent, ProviderError> {
        self.calls.lock().expect("calls").push((self.label.clone(), self.conn));
        if self.fails {
            return Err(boom("stt down"));
        }
        Ok(SpeechEvent::new(SpeechEventType::FinalTranscript))
    }
    async fn stream(
        &self,
        _language: Option<&LanguageCode>,
        _start_time_offset: Option<f64>,
    ) -> Option<Result<(SttSink, SttSource), ProviderError>> {
        None
    }
}

// Catches: an unbounded STT retry loop under the chain; also pins probe 0 vs attempt 1 retries.
#[tokio::test]
async fn a_wrapped_stt_sees_max_retry_one_on_an_attempt_and_zero_on_a_probe() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let fake = |label: &str, fails: bool| -> Arc<dyn Stt> {
        let conn = ApiConnectOptions::default();
        Arc::new(FakeStt { label: label.into(), conn, fails, calls: Arc::clone(&calls) })
    };
    let (clock, spawner, obs) = parts();
    let entries = vec![fake("p", true), fake("b", false)];

    let fb = FallbackStt::new(entries, ChainPolicy::stt_defaults(), clock, spawner, obs)
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
    assert_eq!(p[0], 1, "max_retry_per_stt defaults to 1, not 0");
    assert_eq!(p[1], 0, "the probe pins it to 0");
    let attempt_timeout = seen.iter().find(|(l, _)| l == "p").expect("primary").1.timeout;
    assert_eq!(attempt_timeout, 10.0);
    fb.aclose();
}

// --- the duplex STT the fallback adapter actually runs ----------------------------------------
// Exercises `FallbackStt::stream`, `InputTee` and `TeeSink` (`stt/fallback_adapter.py:341-361`).

#[derive(Default)]
struct SttSess {
    frames: Mutex<Vec<AudioFrame>>,
    flushes: AtomicUsize,
    ended: AtomicBool,
    closed: AtomicBool,
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
    rx: mpsc::UnboundedReceiver<Result<SpeechEvent, ProviderError>>,
}

#[async_trait]
impl StreamSource<SpeechEvent> for SttSessSource {
    async fn next(&mut self) -> Result<Option<SpeechEvent>, ProviderError> {
        match self.rx.recv().await {
            None => Ok(None),
            Some(Ok(ev)) => {
                self.sess.emitted.fetch_add(1, Ordering::SeqCst);
                Ok(Some(ev))
            }
            Some(Err(e)) => Err(e),
        }
    }
    async fn close(mut self: Box<Self>, _deadline: Ts) -> Closed {
        self.sess.closed.store(true, Ordering::SeqCst);
        self.rx.close();
        Closed::Clean
    }
}

type SttSender = mpsc::UnboundedSender<Result<SpeechEvent, ProviderError>>;

struct DuplexStt {
    label: String,
    streaming: bool,
    /// `stream()` refuses this many times before it works.
    refusals: AtomicUsize,
    made: Mutex<Vec<(Arc<SttSess>, Option<SttSender>)>>,
}

impl DuplexStt {
    fn new(label: &str, refusals: usize) -> Arc<Self> {
        Arc::new(DuplexStt {
            label: label.into(),
            streaming: true,
            refusals: AtomicUsize::new(refusals),
            made: Mutex::new(Vec::new()),
        })
    }
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

#[async_trait]
impl Stt for DuplexStt {
    fn label(&self) -> String {
        self.label.clone()
    }
    fn capabilities(&self) -> SttCapabilities {
        SttCapabilities { streaming: self.streaming, interim_results: true, ..Default::default() }
    }
    async fn recognize(
        &self,
        _buffer: AudioBuffer<'_>,
        _language: Option<&LanguageCode>,
    ) -> Result<SpeechEvent, ProviderError> {
        Err(boom("this fake only has a streaming path"))
    }
    async fn stream(
        &self,
        _language: Option<&LanguageCode>,
        _start_time_offset: Option<f64>,
    ) -> Option<Result<(SttSink, SttSource), ProviderError>> {
        let left = self.refusals.load(Ordering::SeqCst);
        if left > 0 {
            self.refusals.store(left - 1, Ordering::SeqCst);
            return Some(Err(boom(&format!("{} has no socket", self.label))));
        }
        let sess = Arc::new(SttSess::default());
        let (tx, rx) = mpsc::unbounded_channel();
        self.made.lock().expect("made").push((Arc::clone(&sess), Some(tx)));
        Some(Ok((
            Box::new(SttSessSink(Arc::clone(&sess))) as SttSink,
            Box::new(SttSessSource { sess, rx }) as SttSource,
        )))
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
        ChainPolicy::stt_defaults(),
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
        within("stream", fb.stream(None, None)).await.expect("duplex").expect("opened");
    until("the member to be serving", || a.sessions() == 1).await;
    a.send(0, Ok(speech(SpeechEventType::InterimTranscript, "hel")));
    let first = within("the first event", source.next()).await.expect("no error").expect("an event");
    assert_eq!(first.kind, SpeechEventType::InterimTranscript);
    assert!(!a.sess(0).closed.load(Ordering::SeqCst), "the member is live while the consumer reads");

    let closed = within("the close", source.close(fb.chain().teardown_deadline())).await;
    assert_eq!(closed, Closed::Clean, "the task honoured the stop and reported done");
    assert!(a.sess(0).closed.load(Ordering::SeqCst), "the serving member's stream was closed, not dropped");
}

// Catches: an audio-in leg that goes nowhere (`set_main`, `fanout`, `push`, `flush`,
// `end_input`), or a late push reaching a released provider after `clear_main`.
#[tokio::test]
async fn the_streaming_path_wires_the_callers_audio_into_the_provider_it_opened() {
    let p = DuplexStt::new("p", 0);
    let fb = stt_chain(vec![Arc::clone(&p)]);

    assert_eq!(fb.label(), "stt.FallbackAdapter");

    let (sink, mut source) =
        within("stream", fb.stream(None, None)).await.expect("the adapter is duplex").expect("opened");
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
        within("stream", fb.stream(None, None)).await.expect("duplex").expect("opened");
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
        within("stream", fb.stream(None, None)).await.expect("duplex").expect("opened");
    until("the backup to be serving", || b.sessions() == 1).await;

    b.send(0, Ok(speech(SpeechEventType::FinalTranscript, "hello")));
    let ev = within("the transcript", source.next()).await.expect("no error").expect("an event");
    assert_eq!(ev.best_alternative().expect("an alternative").text, "hello");

    b.send(0, Err(boom("socket died")));
    let err = within("the failure", source.next()).await.expect_err("nothing is left to try");
    assert!(
        err.message().contains("all backends failed (p, b)"),
        "the message must name the list, in order: {}",
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

    let (sink, source) =
        within("stream", fb.stream(None, None)).await.expect("duplex").expect("opened");
    until("the backup to be serving", || b.sessions() == 1).await;
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
    let err = FallbackStt::new(
        vec![
            DuplexStt::new("good", 0) as Arc<dyn Stt>,
            DuplexStt::batch_only("whisper") as Arc<dyn Stt>,
        ],
        ChainPolicy::stt_defaults(),
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

// Catches: merging STT capabilities with the wrong quantifier (`stt:79-93`), or inverting the
// `aligned_transcript` test.
#[test]
fn merging_stt_capabilities_uses_a_different_rule_per_field() {
    let rich = SttCapabilities {
        streaming: true,
        interim_results: true,
        diarization: true,
        aligned_transcript: AlignedTranscript::Word,
        offline_recognize: true,
        keyterms: true,
        chat_context: false,
    };
    let plain = SttCapabilities {
        streaming: true,
        interim_results: false,
        diarization: true,
        aligned_transcript: AlignedTranscript::Chunk,
        offline_recognize: true,
        keyterms: false,
        chat_context: true,
    };

    let merged = crate::fallback::stt::merge_capabilities(&[rich, plain]);
    assert!(merged.streaming, "the adapter always streams; every entry is required to");
    assert!(!merged.interim_results, "one entry without them loses them for the whole chain");
    assert!(merged.diarization, "both have it");
    assert_eq!(
        merged.aligned_transcript,
        AlignedTranscript::Word,
        "the primary's granularity, because every entry has some alignment"
    );
    assert!(merged.keyterms, "any");
    assert!(merged.chat_context, "any");
    assert!(merged.offline_recognize, "not merged -- the reference takes the field default");

    let unaligned = SttCapabilities { aligned_transcript: AlignedTranscript::None, ..plain };
    assert_eq!(
        crate::fallback::stt::merge_capabilities(&[rich, unaligned]).aligned_transcript,
        AlignedTranscript::None,
        "one entry with no alignment drops it for everyone"
    );
}

// ---------------------------------------------------------------------------------------------
// TTS
// ---------------------------------------------------------------------------------------------

struct FakeTts {
    label: String,
    rate: u32,
    channels: u16,
    frames: usize,
    /// `stream()` refuses this many times before it works; `usize::MAX` never comes back.
    refusals: AtomicUsize,
    /// The source raises after its scripted frames instead of ending cleanly.
    dies: bool,
    pushed: Arc<Mutex<Vec<String>>>,
}

struct Frames {
    left: usize,
    rate: u32,
    channels: u16,
    dies: bool,
}

#[async_trait]
impl StreamSource<SynthesizedAudio> for Frames {
    async fn next(&mut self) -> Result<Option<SynthesizedAudio>, ProviderError> {
        if self.left == 0 {
            return if self.dies { Err(boom("the provider's socket died")) } else { Ok(None) };
        }
        self.left -= 1;
        // 10 ms of interleaved silence at the provider's own rate.
        let n = (self.rate / 100) as usize;
        let ch = self.channels as usize;
        let mut a = SynthesizedAudio::new(
            AudioFrame::new(vec![0u8; n * ch * 2], self.rate, self.channels, n as u32),
            "r",
        );
        a.is_final = self.left == 0 && !self.dies;
        Ok(Some(a))
    }
    async fn close(self: Box<Self>, _deadline: Ts) -> Closed {
        Closed::Clean
    }
}

struct RecordingSink(Arc<Mutex<Vec<String>>>);

impl StreamSink<String> for RecordingSink {
    fn push(&self, item: String) -> Result<(), ProviderError> {
        self.0.lock().expect("pushed").push(item);
        Ok(())
    }
    fn flush(&self) -> Result<(), ProviderError> {
        self.0.lock().expect("pushed").push("<flush>".into());
        Ok(())
    }
    fn end_input(&self) {
        self.0.lock().expect("pushed").push("<end>".into());
    }
}

#[async_trait]
impl Tts for FakeTts {
    fn label(&self) -> String {
        self.label.clone()
    }
    fn capabilities(&self) -> TtsCapabilities {
        TtsCapabilities { streaming: true, aligned_transcript: false }
    }
    fn sample_rate(&self) -> u32 {
        self.rate
    }
    fn num_channels(&self) -> u16 {
        self.channels
    }
    async fn synthesize(&self, _text: &str) -> Result<TtsSource, ProviderError> {
        Ok(Box::new(self.frames()))
    }
    async fn stream(&self) -> Result<(TtsSink, TtsSource), ProviderError> {
        let left = self.refusals.load(Ordering::SeqCst);
        if left > 0 {
            self.refusals.store(left.saturating_sub(1), Ordering::SeqCst);
            return Err(boom("no socket"));
        }
        Ok((
            Box::new(RecordingSink(Arc::clone(&self.pushed))) as TtsSink,
            Box::new(self.frames()) as TtsSource,
        ))
    }
}

impl FakeTts {
    fn frames(&self) -> Frames {
        Frames { left: self.frames, rate: self.rate, channels: self.channels, dies: self.dies }
    }
}

fn built(
    label: &str,
    rate: u32,
    channels: u16,
    frames: usize,
    refusals: usize,
    dies: bool,
) -> (Arc<FakeTts>, Arc<Mutex<Vec<String>>>) {
    let pushed = Arc::new(Mutex::new(Vec::new()));
    (
        Arc::new(FakeTts {
            label: label.into(),
            rate,
            channels,
            frames,
            refusals: AtomicUsize::new(refusals),
            dies,
            pushed: Arc::clone(&pushed),
        }),
        pushed,
    )
}

// Catches: dropping `_pushed_tokens` replay on failover (`tts:428-434`).
#[tokio::test]
async fn a_failover_replays_the_text_already_pushed_into_the_new_provider() {
    let (broken, _) = built("broken", 24_000, 1, 0, usize::MAX, false);
    let (good, good_pushed) = built("good", 24_000, 1, 2, 0, false);
    let fb = tts_chain(vec![broken, good]);

    let (sink, mut source) =
        within("stream", fb.stream()).await.expect("opened");
    sink.push("Hello there, ".into()).expect("push");
    sink.flush().expect("flush");
    sink.push("how are you?".into()).expect("push");
    sink.end_input();

    let frames = drain_tts(&mut source).await.len();
    assert_eq!(frames, 2, "the healthy provider served the whole utterance");
    assert_eq!(
        good_pushed.lock().expect("pushed").as_slice(),
        &["Hello there, ".to_string(), "<flush>".into(), "how are you?".into(), "<end>".into()],
        "everything pushed before the failover is replayed, in order"
    );
    source.close(NO_DEADLINE).await;
    fb.aclose();
}

fn tts_chain(entries: Vec<Arc<dyn Tts>>) -> FallbackTts {
    let (clock, spawner, obs) = parts();
    FallbackTts::new(entries, None, ChainPolicy::tts_defaults(), clock, spawner, obs)
        .expect("a usable list")
}

// Catches: a resampled frame whose header and payload disagree (run at two channels so `/` vs
// `*` differ); also pins label, channel count and capabilities.
#[tokio::test]
async fn a_resampled_stereo_frame_carries_exactly_the_samples_its_header_claims() {
    let (slow, _) = built("slow", 16_000, 2, 3, 0, false);
    let (fast, _) = built("fast", 24_000, 2, 3, 0, false);
    let fb = tts_chain(vec![slow, fast]);

    assert_eq!(fb.label(), "tts.FallbackAdapter");
    assert_eq!(fb.sample_rate(), 24_000, "the maximum of the entries");
    assert_eq!(fb.num_channels(), 2, "both entries are stereo; the adapter must not invent mono");
    assert!(fb.capabilities().streaming, "the merged shape, not Default");
    assert!(fb.needs_resampling(0), "the 16 kHz primary");
    assert!(!fb.needs_resampling(1), "the entry already at the chain's rate");

    let mut source = within("synthesize", fb.synthesize("hello")).await.expect("a source");
    let mut samples = 0usize;
    let mut frames = 0usize;
    for a in drain_tts(&mut source).await {
        frames += 1;
        assert_eq!(a.frame.sample_rate, 24_000);
        assert_eq!(a.frame.num_channels, 2);
        assert_eq!(
            a.frame.data.len(),
            a.frame.samples_per_channel as usize * 2 * 2,
            "bytes = samples_per_channel x channels x 2, or the consumer slices the wrong frame"
        );
        samples += a.frame.samples_per_channel as usize;
    }
    assert!(frames > 0, "the primary served");
    // soxr's filter delay makes the exact count implementation-defined; assert the ratio.
    assert!(samples > 600, "expected ~720 upsampled samples per channel, got {samples}");
    source.close(NO_DEADLINE).await;
}

// Catches: the TTS drain never marking the serving entry (`AfterOutput::Stop`, `tts:293-297`).
#[tokio::test]
async fn the_tts_that_emitted_audio_owns_a_truncated_turn() {
    let (broken, _) = built("broken", 24_000, 1, 0, usize::MAX, false);
    let (dying, _) = built("dying", 24_000, 1, 1, 0, true);
    let fb = tts_chain(vec![broken, dying]);

    assert_eq!(fb.active().label(), "broken", "before any traffic the primary is active");

    let (sink, mut source) = within("stream", fb.stream()).await.expect("opened");
    sink.push("hello".into()).expect("push");
    sink.end_input();

    assert!(within("the one frame", source.next()).await.expect("no error").is_some());
    assert!(
        within("the end", source.next())
            .await
            .expect("truncation is success, not an error")
            .is_none()
    );
    assert_eq!(fb.active().label(), "dying", "the entry that emitted audio owns the turn");
    source.close(NO_DEADLINE).await;
    fb.aclose();
}

// Catches: a streaming probe that replays nothing from `snapshot` (`tts:504-531`).
#[tokio::test]
async fn a_streaming_probe_replays_the_text_recorded_so_far() {
    let (flaky, flaky_pushed) = built("flaky", 24_000, 1, 2, 1, false);
    let (good, _) = built("good", 24_000, 1, 2, 0, false);
    let fb = tts_chain(vec![flaky.clone(), good]);

    let (sink, mut source) = within("stream", fb.stream()).await.expect("opened");
    sink.push("Hello there".into()).expect("push");
    sink.flush().expect("flush");
    sink.end_input();
    drain_tts(&mut source).await;

    until("the probe to bring the primary back", || fb.chain().is_available(0)).await;
    assert_eq!(
        flaky_pushed.lock().expect("pushed").as_slice(),
        &["Hello there".to_string(), "<flush>".into(), "<end>".into()],
        "the probe replays the recorded tokens, in order, into a private channel"
    );
    source.close(NO_DEADLINE).await;
    fb.aclose();
}

/// A backup that speaks only once its input ends, like a real provider waiting for the text: the
/// request it serves cannot finish before the caller's last token.
struct SpeaksOnEnd;

struct EndSignal(Arc<tokio::sync::Notify>);

impl StreamSink<String> for EndSignal {
    fn push(&self, _item: String) -> Result<(), ProviderError> {
        Ok(())
    }
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn end_input(&self) {
        self.0.notify_one();
    }
}

struct AfterEnd {
    ended: Arc<tokio::sync::Notify>,
    frames: Option<Frames>,
}

#[async_trait]
impl StreamSource<SynthesizedAudio> for AfterEnd {
    async fn next(&mut self) -> Result<Option<SynthesizedAudio>, ProviderError> {
        if self.frames.is_none() {
            self.ended.notified().await;
            self.frames = Some(Frames { left: 1, rate: 24_000, channels: 1, dies: false });
        }
        self.frames.as_mut().expect("just set").next().await
    }
    async fn close(self: Box<Self>, _deadline: Ts) -> Closed {
        Closed::Clean
    }
}

#[async_trait]
impl Tts for SpeaksOnEnd {
    fn label(&self) -> String {
        "backup".into()
    }
    fn capabilities(&self) -> TtsCapabilities {
        TtsCapabilities { streaming: true, aligned_transcript: false }
    }
    fn sample_rate(&self) -> u32 {
        24_000
    }
    fn num_channels(&self) -> u16 {
        1
    }
    async fn synthesize(&self, _text: &str) -> Result<TtsSource, ProviderError> {
        Ok(Box::new(Frames { left: 1, rate: 24_000, channels: 1, dies: false }))
    }
    async fn stream(&self) -> Result<(TtsSink, TtsSource), ProviderError> {
        let ended = Arc::new(tokio::sync::Notify::new());
        Ok((
            Box::new(EndSignal(Arc::clone(&ended))) as TtsSink,
            Box::new(AfterEnd { ended, frames: None }) as TtsSource,
        ))
    }
}

// Catches: probing a skipped entry as the scan passes it, before any text has arrived, so the
// streamed probe declines for want of text and the provider stays down for good (#6678,
// `tts:417-420`).
#[tokio::test]
async fn an_entry_skipped_on_a_streamed_request_is_probed_with_that_requests_text() {
    let (flaky, flaky_pushed) = built("flaky", 24_000, 1, 1, 2, false);
    let fb = tts_chain(vec![flaky.clone(), Arc::new(SpeaksOnEnd)]);

    // First request: the attempt and its probe are both refused, so "flaky" stays down.
    let (sink, mut source) = within("stream", fb.stream()).await.expect("opened");
    sink.push("one".into()).expect("push");
    sink.end_input();
    drain_tts(&mut source).await;
    source.close(NO_DEADLINE).await;
    until("the first probe to be refused", || flaky.refusals.load(Ordering::SeqCst) == 0).await;
    assert!(!fb.chain().is_available(0));
    flaky_pushed.lock().expect("pushed").clear();

    // Second request: "flaky" is skipped before the caller has pushed anything.
    let (sink, mut source) = within("stream", fb.stream()).await.expect("opened");
    tokio::time::sleep(Duration::from_millis(20)).await;
    sink.push("two".into()).expect("push");
    sink.end_input();
    drain_tts(&mut source).await;

    until("the skipped entry to be probed and recover", || fb.chain().is_available(0)).await;
    assert_eq!(
        flaky_pushed.lock().expect("pushed").as_slice(),
        &["two".to_string(), "<end>".into()],
        "the probe replays the text this request received"
    );
    source.close(NO_DEADLINE).await;
    fb.aclose();
}

// Catches: a `TokenLog` still pointing at a provider the chain has finished with.
#[tokio::test]
async fn the_child_sink_is_released_when_the_stream_that_owns_it_ends() {
    let (good, pushed) = built("good", 24_000, 1, 1, 0, false);
    let fb = tts_chain(vec![good]);

    let (sink, mut source) = within("stream", fb.stream()).await.expect("opened");
    sink.push("hello".into()).expect("push");
    sink.end_input();
    drain_tts(&mut source).await;

    let before = pushed.lock().expect("pushed").len();
    sink.push("late".into()).expect("push");
    assert_eq!(
        pushed.lock().expect("pushed").len(),
        before,
        "the turn is over: a late token must not reach a provider the chain has released"
    );
    source.close(NO_DEADLINE).await;
}

// Catches: a TTS adapter `aclose` that does not reach the chain, or an exhaustion error not naming
// the list.
#[tokio::test]
async fn a_closed_tts_adapter_starts_no_more_probes_and_a_total_outage_names_the_list() {
    let (alpha, alpha_pushed) = built("alpha", 24_000, 1, 1, 1, false);
    let (beta, _) = built("beta", 24_000, 1, 1, 0, false);
    let fb = tts_chain(vec![alpha.clone(), beta]);

    fb.aclose();
    let (sink, mut source) = within("stream", fb.stream()).await.expect("opened");
    sink.push("hello".into()).expect("push");
    sink.end_input();
    drain_tts(&mut source).await;
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
    assert!(
        alpha_pushed.lock().expect("pushed").is_empty(),
        "a probe started after aclose is a task nothing will ever cancel"
    );
    source.close(NO_DEADLINE).await;

    // And with every entry refusing, the error has to say who was tried.
    let (a, _) = built("alpha", 24_000, 1, 0, usize::MAX, false);
    let (b, _) = built("beta", 24_000, 1, 0, usize::MAX, false);
    let fb = tts_chain(vec![a, b]);
    let (sink, mut source) = within("stream", fb.stream()).await.expect("opened");
    sink.push("hello".into()).expect("push");
    sink.end_input();
    let err = within("the failure", source.next()).await.expect_err("both entries refused");
    assert!(
        err.message().contains("all backends failed (alpha, beta)"),
        "the message must name the list, in order: {}",
        err.message()
    );
    source.close(NO_DEADLINE).await;
    fb.aclose();
}

// Catches: one quantifier for both TTS capabilities (`streaming` any, `aligned_transcript` all,
// `tts:92-95`).
#[test]
fn merging_tts_capabilities_is_any_for_streaming_and_all_for_alignment() {
    let duplex = TtsCapabilities { streaming: true, aligned_transcript: true };
    let batch = TtsCapabilities { streaming: false, aligned_transcript: true };
    let unaligned = TtsCapabilities { streaming: true, aligned_transcript: false };

    let merged = crate::fallback::tts::merge_capabilities(&[duplex, batch]);
    assert!(merged.streaming, "any -- see the function's own docs for why this over-promises");
    assert!(merged.aligned_transcript, "both align");

    assert!(!crate::fallback::tts::merge_capabilities(&[duplex, unaligned]).aligned_transcript, "all");
    assert!(!crate::fallback::tts::merge_capabilities(&[batch]).streaming, "no entry streams");
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
        within("chat", fb.chat(&ChatRequest::default())).await.expect("the backup serves the turn");
    while within("next", stream.next()).await.expect("drain").is_some() {}

    assert_eq!(fb.model(), "gpt-4o-mini", "after a failover it must follow the active entry");
}
