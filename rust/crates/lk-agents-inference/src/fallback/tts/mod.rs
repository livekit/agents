use crate::fallback::chain::{
    Attempt, Backend, ChainCall, ChainObserver, FallbackChain, RecoveryMode,
};
use crate::fallback::channel::{ServeAttempt, drain_probe, next_or_stop};
use crate::fallback::traced::{self, Traced};
use crate::fallback::{ChainPolicy, output_rate};
use crate::telemetry;
use crate::tts::fed::{Fed, FixedLog, TextLog};
use crate::tts::sentence_adapter::StreamAdapter;
use crate::tts::stage::{Stage, one_shot};
use async_trait::async_trait;
use lk_agents_abi::ApiConnectOptions;
use lk_agents_abi::errors::Invalid;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::llm::fallback as port;
use lk_agents_abi::types::provider::NativeLabel;
use lk_agents_abi::utils::stream::{Entry, ProviderStream};
use lk_agents_abi::{BoxFuture, Spawner};
use lk_agents_abi::{Clock, HOST_TEARDOWN_TIMEOUT};
use lk_agents_abi::{
    SpeechChunk, StreamInput, SynthesizedAudio, Tts, TtsCapabilities, TtsSink, TtsSource,
};
use lk_agents_media::{AudioFormat, AudioResampler, Quality};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::signal;
use lk_agents_net::source::{Gated, InputGate};
use parking_lot::Mutex;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

// A failure after output remains a failure: delivered audio cannot be replayed.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    Text(String),
    /// A segment boundary, not a close (`SynthesizeStream.flush`).
    Flush,
}

/// The pushed-token log as the host's `SynthesizeStream` keeps it (`self._pushed_tokens`): the
/// text each attempt and recovering stream replays. The text leg may not drop (§1.2d), so nothing
/// here is lossy or bounded.
pub struct TokenLog {
    inner: signal::Shared<TokenLogState>,
}

impl Default for TokenLog {
    fn default() -> Self {
        TokenLog { inner: signal::Shared::new(TokenLogState::default()) }
    }
}

#[derive(Default)]
struct TokenLogState {
    tokens: Vec<Token>,
    ended: bool,
    /// The request finished (its task's guard dropped): no more text can arrive.
    finished: bool,
}

impl TokenLog {
    pub fn new() -> Arc<Self> {
        Arc::new(TokenLog::default())
    }

    pub fn snapshot(&self) -> (Vec<Token>, bool) {
        self.inner.with(|s| (s.tokens.clone(), s.ended))
    }

    /// Waits until the input ends or the request finishes, so a probe started early still finds
    /// text (#6678; the reference starts probes after the request, `tts:417-420,490-493`).
    async fn settled_tokens(&self) -> Vec<Token> {
        self.inner.wait_until(|s| (s.ended || s.finished).then(|| s.tokens.clone())).await
    }

    fn finish(&self) {
        self.inner.modify(|s| s.finished = true);
    }

    /// Kept only before the input ended: a late token reaches no provider.
    pub fn push(&self, token: Token) {
        self.inner.modify(|s| {
            if !s.ended {
                s.tokens.push(token);
            }
        });
    }

    pub fn end(&self) {
        self.inner.modify(|s| s.ended = true);
    }
}

impl TextLog for TokenLog {
    fn entry(&self, index: usize) -> BoxFuture<'_, Entry<String>> {
        Box::pin(self.inner.wait_until(move |s| match s.tokens.get(index) {
            Some(token) => Some(token.clone().into()),
            None => (s.ended && index == s.tokens.len()).then_some(Entry::EndInput),
        }))
    }
}

impl From<Token> for Entry<String> {
    fn from(token: Token) -> Self {
        match token {
            Token::Text(text) => Entry::Item(text),
            Token::Flush => Entry::Flush,
        }
    }
}

struct FinishOnDrop(Option<Arc<TokenLog>>);

impl Drop for FinishOnDrop {
    fn drop(&mut self) {
        if let Some(log) = &self.0 {
            log.finish();
        }
    }
}

/// The request's text, refused once the input ended or the request failed, with its failure.
struct LoggingSink {
    log: Arc<TokenLog>,
    gate: InputGate,
}

impl LoggingSink {
    fn log(&self, token: Token) -> Result<(), ProviderError> {
        self.gate.admit_with(|| {
            self.log.push(token);
            Ok(())
        })
    }
}

/// The text as written ([`TextUnit::Text`]): each member prepares it for itself.
impl StreamInput<SpeechChunk> for LoggingSink {
    fn push(&self, item: SpeechChunk) -> Result<(), ProviderError> {
        // `tts:409-410` records only non-empty strings.
        if item.text.is_empty() {
            return self.gate.admit();
        }
        self.log(Token::Text(item.text))
    }

    fn flush(&self) -> Result<(), ProviderError> {
        self.log(Token::Flush)
    }

    fn end_input(&self) {
        self.gate.end();
        self.log.end();
    }
}

/// `cx`: where each instance's `tts_request` opens (the adapter request's attempt).
#[derive(Clone)]
pub enum TtsReq {
    OneShot { text: Arc<str>, cx: telemetry::Cx },
    Stream { log: Arc<TokenLog>, cx: telemetry::Cx },
}

impl TtsReq {
    fn cx(&self) -> &telemetry::Cx {
        match self {
            TtsReq::OneShot { cx, .. } | TtsReq::Stream { cx, .. } => cx,
        }
    }

    /// The text the request synthesized: all of it for a one-shot, what was pushed for a stream.
    pub(super) fn spoken(&self) -> String {
        match self {
            TtsReq::OneShot { text, .. } => text.to_string(),
            TtsReq::Stream { log, .. } => (log.snapshot().0.into_iter())
                .filter_map(|t| match t {
                    Token::Text(text) => Some(text),
                    Token::Flush => None,
                })
                .collect(),
        }
    }
}

pub struct TtsBackend {
    label: String,
    inner: Arc<dyn Tts>,
    clock: Arc<dyn Clock>,
}

impl TtsBackend {
    pub fn inner(&self) -> &Arc<dyn Tts> {
        &self.inner
    }

    pub fn new(tts: &Arc<dyn Tts>, clock: Arc<dyn Clock>) -> Arc<Self> {
        Arc::new(TtsBackend { label: tts.label(), inner: tts.clone(), clock })
    }

    fn provider(&self, conn: &ApiConnectOptions) -> Arc<dyn Tts> {
        let tts = self.inner.with_conn_options(conn).unwrap_or_else(|| self.inner.clone());
        // The chain owns retries; the sentence adapter only segments this attempt's text.
        StreamAdapter::adapted(tts, self.clock.clone(), *conn)
    }
}

#[async_trait]
impl Backend for TtsBackend {
    type Req = TtsReq;

    type Session = TtsSource;

    fn resource_id(&self) -> lk_agents_abi::resources::ResourceId {
        super::backend_resource(self.inner.as_ref(), self.inner.instance_id())
    }

    fn label(&self) -> String {
        self.label.clone()
    }

    fn failed(&self, error: ProviderError, recoverable: bool) {
        self.inner.failed(error, recoverable);
    }

    async fn open(
        &self,
        req: &Self::Req,
        conn: &ApiConnectOptions,
    ) -> Result<Self::Session, ProviderError> {
        let tts = self.provider(conn);
        let cx = req.cx();
        // This instance's `tts_request`, under the adapter request's attempt.
        let streaming = matches!(req, TtsReq::Stream { .. });
        let mut request = cx.span().map(|_| telemetry::TtsTrace::start(cx, &*tts, streaming));
        let run = request.as_mut().and_then(telemetry::TtsTrace::attempt);
        let inner = run.as_ref().map_or(cx, |run| run.cx());
        let opened = match req {
            TtsReq::OneShot { text, .. } => {
                one_shot_member(&tts, inner, text, conn, &self.clock).await
            }
            TtsReq::Stream { log, .. } => {
                tts.stream(inner).await.map(|s| {
                    let (sink, source) = s.split();
                    let log: Arc<dyn TextLog> = log.clone();
                    // each member's text prepared for it: its dialect, its unit, its options
                    let clock = self.clock.clone();
                    let stage = Stage::new(&*tts, log, true, inner, clock);
                    Box::new(Fed::new(stage, sink, source)) as TtsSource
                })
            }
        };
        match (opened, request) {
            (Ok(source), Some(request)) => Ok(Box::new(Traced {
                source,
                trace: Some(traced::Tts { trace: request, req: req.clone(), run }),
            })),
            (Ok(source), None) => Ok(source),
            (Err(error), request) => {
                if let Some(mut request) = request {
                    request.end_attempt(run, Some(&error));
                    request.fail(&error);
                }
                Err(error)
            }
        }
    }

    async fn recover(
        &self,
        req: &Self::Req,
        conn: &ApiConnectOptions,
        _slot: RecoveryMode,
        stop: CancellationToken,
    ) -> Result<bool, ProviderError> {
        let tts = self.provider(conn);
        // A probe runs in the background, past the request it recovers for: in its job, under no
        // span.
        let cx = req.cx().clone().continuing(None);
        let source = match req {
            TtsReq::OneShot { text, .. } => {
                one_shot_member(&tts, &cx, text, conn, &self.clock).await?
            }
            TtsReq::Stream { log, .. } => {
                let tokens = tokio::select! {
                    biased;
                    () = stop.cancelled() => return Ok(false),
                    tokens = log.settled_tokens() => tokens,
                };
                if tokens.is_empty() {
                    // `tts:504-506`: a probe needs text, and the request pushed none.
                    return Ok(false);
                }
                let (sink, source) = tts.stream(&cx).await?.split();
                // A private replay through the member's stage; unlike STT, the probe does not
                // join the live leg. Each entry is awaited: a slow input holds the replay.
                let entries = tokens.into_iter().map(Entry::from).chain([Entry::EndInput]);
                let log: Arc<dyn TextLog> = Arc::new(FixedLog(entries.collect()));
                let stage = Stage::new(&*tts, log, false, &cx, self.clock.clone());
                Box::new(Fed::new(stage, sink, source)) as TtsSource
            }
        };
        drain_probe(source, self.clock.as_ref(), stop).await
    }
}

/// A member's whole text: one request, or one per chunk when it passes the member's limit. The
/// chain owns the retries.
async fn one_shot_member(
    tts: &Arc<dyn Tts>,
    cx: &telemetry::Cx,
    text: &str,
    conn: &ApiConnectOptions,
    clock: &Arc<dyn Clock>,
) -> Result<TtsSource, ProviderError> {
    let mut chunks = one_shot(tts.as_ref(), text);
    if chunks.len() > 1 {
        let once = ApiConnectOptions { max_retry: 0, ..*conn };
        return crate::tts::request::in_chunks(tts.clone(), cx, chunks, once, clock.clone());
    }
    tts.synthesize(cx, &chunks.pop().unwrap_or_default()).await
}

/// `tts:92-95`: `streaming` is `any`, `aligned_transcript` is `all`, over the members as given.
/// The chain takes the text as written: each member's is prepared for it by its own attempt.
pub fn merge_capabilities(entries: &[TtsCapabilities]) -> TtsCapabilities {
    TtsCapabilities {
        streaming: entries.iter().any(|c| c.streaming),
        aligned_transcript: entries.iter().all(|c| c.aligned_transcript),
        text_unit: lk_agents_abi::tts::types::TextUnit::Text,
        max_text_len: None,
    }
}

#[derive(Clone)]
pub struct FallbackTts {
    chain: Arc<FallbackChain<TtsBackend>>,
    capabilities: TtsCapabilities,
    sample_rate: u32,
    num_channels: u16,
    caller_conn: ApiConnectOptions,
    /// The adapter that built the chain: a request's failures are its `error` events.
    observer: Arc<dyn ChainObserver>,
    /// The name logs give it: its host class's when its host built it.
    label: String,
}

impl FallbackTts {
    fn configured(&self, conn_options: Option<ApiConnectOptions>) -> Self {
        Self { caller_conn: conn_options.unwrap_or(self.caller_conn), ..self.clone() }
    }

    pub fn new(
        ttss: Vec<Arc<dyn Tts>>,
        sample_rate: Option<u32>,
        policy: ChainPolicy,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        observer: Arc<dyn ChainObserver>,
    ) -> Result<Self, Invalid> {
        let capabilities =
            merge_capabilities(&ttss.iter().map(|t| t.capabilities()).collect::<Vec<_>>());
        let entries: Vec<Arc<TtsBackend>> =
            ttss.iter().map(|t| TtsBackend::new(t, Arc::clone(&clock))).collect();
        let channels: Vec<u16> = entries.iter().map(|b| b.inner.num_channels()).collect();
        let Some(&num_channels) = channels.first() else {
            return Err(Invalid("at least one TTS instance must be provided".into()));
        };
        if channels.iter().any(|c| *c != num_channels) {
            return Err(Invalid(format!(
                "all TTS must have the same number of channels, got {channels:?}"
            )));
        }
        let target = match sample_rate {
            Some(rate) => rate,
            None => {
                let rates = entries.iter().map(|b| b.inner.sample_rate());
                let rates = rates
                    .collect::<Result<Vec<u32>, _>>()
                    .map_err(|e| Invalid(format!("reading a TTS's sample rate failed: {e}")))?;
                output_rate(&rates, None)
            }
        };
        let chain = FallbackChain::new(entries, policy, clock, spawner, observer.clone())?;
        Ok(FallbackTts {
            chain: Arc::new(chain),
            capabilities,
            sample_rate: target,
            num_channels,
            caller_conn: crate::fallback::caller_default(),
            observer,
            label: NativeLabel::TtsFallback.as_str().into(),
        })
    }

    pub fn chain(&self) -> &Arc<FallbackChain<TtsBackend>> {
        &self.chain
    }

    pub fn active(&self) -> Arc<dyn Tts> {
        Arc::clone(&self.chain.active().inner)
    }

    fn spawn_pump(&self, req: TtsReq, slot: RecoveryMode, log: Option<Arc<TokenLog>>) -> TtsSource {
        let call = ChainCall {
            conn_options: self.caller_conn,
            accepted_at: self.chain.clock().now(),
            slot,
        };
        let leg = Arc::new(TtsLeg {
            target: self.sample_rate,
            channels: self.num_channels,
            log: log.clone(),
            started: Mutex::default(),
        });
        // Dropped with the task, so an aborted request still releases its probes.
        let source = self.chain.serve_stream(req, call, leg, |_, _| {}, FinishOnDrop(log));
        Box::new(source) as TtsSource
    }
}

/// One request's attempts: each resamples its entry's audio to the adapter's rate.
struct TtsLeg {
    target: u32,
    channels: u16,
    /// The streamed request's text; `None` for a one-shot.
    log: Option<Arc<TokenLog>>,
    /// The streamed request is measured as one TTS node (`tts:350-355`): its started time is the
    /// first attempt's that handed text to a provider, kept across a failover.
    started: Mutex<Option<f64>>,
}

/// What became of one frame handed to the consumer.
enum Delivery {
    Delivered,
    ConsumerGone,
    Failed(ProviderError),
}

impl ServeAttempt<TtsBackend, SynthesizedAudio> for TtsLeg {
    fn serve<'a>(
        self: Arc<Self>,
        chain: Arc<FallbackChain<TtsBackend>>,
        index: usize,
        mut source: TtsSource,
        tx: &'a mut mpsc::Sender<Result<SynthesizedAudio, ProviderError>>,
        stop: CancellationToken,
    ) -> BoxFuture<'a, Attempt<()>> {
        Box::pin(async move {
            let outcome = self.forward_audio(&chain, index, &mut source, tx, &stop).await;
            // An attempt that failed before any audio still anchors the request.
            self.anchor_started(&source);
            source.close(chain.clock().now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await;
            outcome
        })
    }
}

impl TtsLeg {
    /// The attempt's resampler for a frame at `from` (`tts:262-269`): none at the adapter's rate,
    /// rebuilt when the rate a frame carries changes, never crossing a failover.
    fn resampler_at<'r>(
        &self,
        slot: &'r mut Option<(u32, AudioResampler)>,
        from: u32,
    ) -> Result<Option<&'r mut AudioResampler>, ProviderError> {
        if from == self.target {
            return Ok(None);
        }
        if slot.as_ref().is_none_or(|(rate, _)| *rate != from) {
            let format = AudioFormat::new(from, self.channels);
            let resampler = AudioResampler::fixed(format, self.target, Quality::Medium)
                .map_err(|e| ProviderError::Local(ApiError::new(format!("resampler: {e}"))))?;
            *slot = Some((from, resampler));
        }
        Ok(slot.as_mut().map(|(_, resampler)| resampler))
    }

    async fn forward_audio(
        &self,
        chain: &FallbackChain<TtsBackend>,
        index: usize,
        source: &mut TtsSource,
        tx: &mut mpsc::Sender<Result<SynthesizedAudio, ProviderError>>,
        stop: &CancellationToken,
    ) -> Attempt<()> {
        let mut produced = false;
        let mut resampler = None;
        loop {
            let mut audio = match next_or_stop(source.as_mut(), stop).await {
                Ok(Some(audio)) => audio,
                Ok(None) => return Attempt::Served(()),
                Err(error) => return Attempt::Failed { error, produced_output: produced },
            };
            if !produced {
                produced = true;
                chain.mark_serving(index);
            }
            if let Some(at) = self.anchor_started(source) {
                audio.started_time = Some(at);
            }
            let resampler = match self.resampler_at(&mut resampler, audio.frame.sample_rate) {
                Ok(resampler) => resampler,
                Err(error) => return Attempt::Failed { error, produced_output: produced },
            };
            match Self::deliver(audio, resampler, tx).await {
                Delivery::Delivered => {}
                Delivery::ConsumerGone => return Attempt::Served(()),
                Delivery::Failed(error) => {
                    return Attempt::Failed { error, produced_output: produced };
                }
            }
        }
    }

    /// Sends `audio` at the adapter's rate; at every `is_final` the filter's tail follows it
    /// (`tts:464-466`).
    async fn deliver(
        audio: SynthesizedAudio,
        resampler: Option<&mut AudioResampler>,
        tx: &mut mpsc::Sender<Result<SynthesizedAudio, ProviderError>>,
    ) -> Delivery {
        let Some(resampler) = resampler else {
            return match tx.send(Ok(audio)).await {
                Ok(()) => Delivery::Delivered,
                Err(_) => Delivery::ConsumerGone,
            };
        };
        let ready = match resampler.push_frame(&audio.frame) {
            Ok(ready) => ready,
            Err(e) => {
                return Delivery::Failed(ProviderError::Local(ApiError::new(format!(
                    "resample: {e}"
                ))));
            }
        };
        let with_frame = |frame| SynthesizedAudio { frame, ..audio.clone() };
        if let Some(frame) = ready.into_iter().next()
            && tx.send(Ok(with_frame(frame))).await.is_err()
        {
            return Delivery::ConsumerGone;
        }
        if audio.is_final
            && let Ok(Some(tail)) = resampler.flush()
            && tx.send(Ok(with_frame(tail))).await.is_err()
        {
            return Delivery::ConsumerGone;
        }
        Delivery::Delivered
    }

    /// The request's started time, anchored by the first streamed attempt that reports one.
    /// `started_time` is a plain read, so concurrent attempts settle the anchor under the lock.
    fn anchor_started(&self, source: &TtsSource) -> Option<f64> {
        self.log.as_ref()?;
        let mut anchor = self.started.lock();
        if anchor.is_none() {
            *anchor = source.started_time();
        }
        *anchor
    }
}

impl FallbackTts {
    /// Cancel every recovery probe in flight, idempotently.
    pub fn aclose(&self) {
        self.chain.aclose();
    }
}

impl Tts for FallbackTts {
    /// The chain retries and falls back on its own.
    fn retries_itself(&self) -> bool {
        true
    }

    /// The adapter's `error` event.
    fn failed(&self, error: ProviderError, recoverable: bool) {
        self.observer.failed(error, recoverable);
    }

    fn with_conn_options(&self, conn: &ApiConnectOptions) -> Option<Arc<dyn Tts>> {
        Some(Arc::new(self.configured(Some(*conn))))
    }

    fn label(&self) -> String {
        self.label.clone()
    }

    fn request_span(&self) -> telemetry::RequestSpan {
        telemetry::RequestSpan::FallbackAdapter
    }

    /// The **active** member's identity (L-25: adapters are transparent to the metrics).
    fn id(&self) -> lk_agents_abi::ProviderId {
        self.active().id()
    }

    fn prewarm(&self, cx: &telemetry::Cx) {
        for entry in self.chain.entries() {
            entry.inner().prewarm(cx);
        }
    }

    fn capabilities(&self) -> TtsCapabilities {
        self.capabilities
    }

    fn sample_rate(&self) -> Result<u32, lk_agents_abi::errors::HostError> {
        Ok(self.sample_rate)
    }

    fn num_channels(&self) -> u16 {
        self.num_channels
    }

    fn synthesize<'a>(
        &'a self,
        cx: &telemetry::Cx,
        chunk: &'a SpeechChunk,
    ) -> lk_agents_abi::BoxFuture<'a, Result<TtsSource, ProviderError>> {
        let cx = cx.clone();
        Box::pin(async move {
            Ok(self.spawn_pump(
                TtsReq::OneShot { text: Arc::from(chunk.text.as_str()), cx },
                RecoveryMode::OneShot,
                None,
            ))
        })
    }

    fn stream(
        &self,
        cx: &telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<lk_agents_abi::DuplexStream<TtsSink, TtsSource>, ProviderError>,
    > {
        let cx = cx.clone();
        Box::pin(async move {
            let log = TokenLog::new();
            let source = self.spawn_pump(
                TtsReq::Stream { log: Arc::clone(&log), cx },
                RecoveryMode::Streaming,
                Some(Arc::clone(&log)),
            );
            let gate = InputGate::default();
            let sink = LoggingSink { log, gate: gate.clone() };
            Ok(lk_agents_abi::DuplexStream::new(
                Box::new(sink) as TtsSink,
                Box::new(Gated::new(source, gate)) as TtsSource,
            ))
        })
    }
}

impl port::FallbackTts for FallbackTts {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        label: String,
        observer: Arc<dyn ChainObserver>,
        ttss: Vec<Arc<dyn Tts>>,
        max_retry_per_tts: u32,
        attempt_timeout: Option<f64>,
        sample_rate: Option<u32>,
    ) -> Result<Self, Invalid> {
        // `sample_rate = None` takes the maximum of the entries' rates and upsamples the rest.
        let policy = ChainPolicy {
            max_retry_per_backend: max_retry_per_tts,
            attempt_timeout,
            ..ChainPolicy::tts_defaults()
        };
        let spawner = (env.spawner)().map_err(|e| Invalid(e.to_string()))?;
        let chain =
            FallbackTts::new(ttss, sample_rate, policy, env.clock.clone(), spawner, observer)?;
        Ok(FallbackTts { label, ..chain })
    }
    fn active_index(&self) -> usize {
        self.chain.active_index()
    }
    fn capabilities(&self) -> TtsCapabilities {
        self.capabilities
    }
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn num_channels(&self) -> u16 {
        self.num_channels
    }
    /// Every recovery probe in flight closes its stream, awaited (bounded).
    async fn aclose(&self) {
        self.chain.close().await;
    }
}

#[cfg(test)]
#[path = "tts_tests.rs"]
mod tests;
