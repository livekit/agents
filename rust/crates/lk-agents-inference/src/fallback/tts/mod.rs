//! `tts/fallback_adapter.py`. The output rate is chosen for the whole list and per-entry
//! resampling rides on the chain (`tts:86-118,262-282`). [`TokenLog`] replays pushed tokens into a
//! fresh input on failover (`tts:428-434`) and into the streaming probe (`tts:504-531`). A failure
//! after output remains a failure and cannot replay delivered audio.

use crate::fallback::chain::{
    Attempt, Backend, ChainCall, ChainObserver, FallbackChain, NoBackends, Recovery, RecoveryMode,
};
use crate::fallback::channel::{ServeAttempt, drain_probe, next_or_stop};
use crate::fallback::{ChainPolicy, resampling_plan};
use crate::telemetry;
use crate::tts::sentence_adapter::StreamAdapter;
use crate::tts::stream::TtsStream;
use async_trait::async_trait;
use lk_agents_abi::ApiConnectOptions;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::llm::fallback as port;
use lk_agents_abi::llm::inference::InvalidPayload;
use lk_agents_abi::tts::stream_adapter::{SplitterFactory, TextProcessorFactory};
use lk_agents_abi::utils::stream::{ProviderStream, StreamOutput};
use lk_agents_abi::{BoxFuture, Spawner};
use lk_agents_abi::{Clock, HOST_TEARDOWN_TIMEOUT};
use lk_agents_abi::{Closed, Ts};
use lk_agents_abi::{StreamInput, SynthesizedAudio, Tts, TtsCapabilities, TtsSink, TtsSource};
use lk_agents_media::{AudioFormat, AudioResampler, Quality};
use lk_agents_net::env::ProcessEnv;
use parking_lot::Mutex;
use std::sync::Arc;
use tokio::sync::{mpsc, watch};

fn send(sink: &dyn StreamInput<String>, token: &Token) {
    let _ = match token {
        Token::Text(text) => sink.push(text.clone()),
        Token::Flush => sink.flush(),
    };
}

// ---------------------------------------------------------------------------------------------
// The replayable text leg
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    Text(String),
    /// A segment boundary, not a close (`SynthesizeStream.flush`).
    Flush,
}

/// The pushed-token log as the host's `SynthesizeStream` keeps it (`_core.TtsTokenLog`): the text
/// a recovering stream replays.
///
/// `self._pushed_tokens` plus the live child sink. The text leg may not drop (§1.2d), so nothing
/// here is lossy or bounded.
pub struct TokenLog {
    inner: Mutex<TokenLogState>,
    /// Raised once no more text can arrive; what a streamed probe waits for.
    settled: watch::Sender<bool>,
}

impl Default for TokenLog {
    fn default() -> Self {
        TokenLog { inner: Mutex::default(), settled: watch::channel(false).0 }
    }
}

#[derive(Default)]
struct TokenLogState {
    tokens: Vec<Token>,
    ended: bool,
    main: Option<Arc<dyn StreamInput<String>>>,
}

impl TokenLog {
    pub fn new() -> Arc<Self> {
        Arc::new(TokenLog::default())
    }

    pub fn snapshot(&self) -> (Vec<Token>, bool) {
        let s = self.inner.lock();
        (s.tokens.clone(), s.ended)
    }

    /// Waits until the input ends or the request finishes, so a probe started early still finds
    /// text (#6678; the reference starts probes after the request, `tts:417-420,490-493`).
    async fn settled_tokens(&self) -> Vec<Token> {
        let _ = self.settled.subscribe().wait_for(|settled| *settled).await;
        self.snapshot().0
    }

    fn finish(&self) {
        self.settled.send_replace(true);
    }

    /// Catch up a fresh child sink (`tts:428-434`): it is installed once it has every token, so
    /// a token pushed during the failover is neither replayed twice nor lost. The sink is called
    /// with no lock held: a host TTS's sink takes the GIL.
    fn set_main(&self, sink: TtsSink) {
        let sink: Arc<dyn StreamInput<String>> = Arc::from(sink);
        let mut sent = 0;
        let ended = loop {
            let missing = {
                let mut s = self.inner.lock();
                if s.tokens.len() == sent {
                    s.main = Some(Arc::clone(&sink));
                    break s.ended;
                }
                s.tokens[sent..].to_vec()
            };
            missing.iter().for_each(|t| send(sink.as_ref(), t));
            sent += missing.len();
        };
        if ended {
            sink.end_input();
        }
    }

    fn clear_main(&self) {
        self.inner.lock().main = None;
    }

    pub fn push(&self, token: Token) {
        let main = {
            let mut s = self.inner.lock();
            s.tokens.push(token.clone());
            s.main.clone()
        };
        if let Some(sink) = main {
            send(sink.as_ref(), &token);
        }
    }

    pub fn end(&self) {
        let main = {
            let mut s = self.inner.lock();
            s.ended = true;
            s.main.clone()
        };
        if let Some(sink) = main {
            sink.end_input();
        }
        self.settled.send_replace(true);
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

struct LoggingSink(Arc<TokenLog>);

impl StreamInput<String> for LoggingSink {
    fn push(&self, item: String) -> Result<(), ProviderError> {
        // `tts:409-410` records only non-empty strings.
        if !item.is_empty() {
            self.0.push(Token::Text(item));
        }
        Ok(())
    }

    fn flush(&self) -> Result<(), ProviderError> {
        self.0.push(Token::Flush);
        Ok(())
    }

    fn end_input(&self) {
        self.0.end();
    }
}

// ---------------------------------------------------------------------------------------------
// The backend
// ---------------------------------------------------------------------------------------------

/// `cx`: where each instance's `tts_request` opens (the adapter request's attempt).
#[derive(Clone)]
pub enum TtsReq {
    OneShot { text: Arc<str>, cx: telemetry::Cx },
    Stream { log: Arc<TokenLog>, expressive: bool, cx: telemetry::Cx },
}

impl TtsReq {
    fn cx(&self) -> &telemetry::Cx {
        match self {
            TtsReq::OneShot { cx, .. } | TtsReq::Stream { cx, .. } => cx,
        }
    }

    /// The text the request synthesized: all of it for a one-shot, what was pushed for a stream.
    fn spoken(&self) -> String {
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
    splitter: SplitterFactory,
    text_processor: TextProcessorFactory,
    clock: Arc<dyn Clock>,
}

impl TtsBackend {
    pub fn inner(&self) -> &Arc<dyn Tts> {
        &self.inner
    }

    pub fn new(tts: &Arc<dyn Tts>, clock: Arc<dyn Clock>) -> Arc<Self> {
        Arc::new(TtsBackend {
            label: tts.label(),
            inner: tts.clone(),
            splitter: Arc::new(|expressive| {
                Box::new(crate::tts::sentences::sentence_splitter(None, expressive))
            }),
            text_processor: Arc::new(lk_agents_core::tts::markup::text_processor),
            clock,
        })
    }

    fn provider(&self, conn: &ApiConnectOptions) -> Arc<dyn Tts> {
        let tts = self.inner.with_conn_options(conn).unwrap_or_else(|| self.inner.clone());
        // The chain owns retries; the sentence adapter only segments this attempt's text.
        StreamAdapter::adapted(
            tts,
            self.splitter.clone(),
            self.text_processor.clone(),
            self.clock.clone(),
            *conn,
        )
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
            TtsReq::OneShot { text, .. } => tts.synthesize(inner, text).await,
            TtsReq::Stream { log, expressive, .. } => {
                tts.stream(inner, *expressive).await.map(|s| {
                    let (sink, source) = s.split();
                    log.set_main(sink);
                    source
                })
            }
        };
        match (opened, request) {
            (Ok(source), Some(request)) => {
                Ok(Box::new(Traced { source, req: req.clone(), request: Some(request), run }))
            }
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
        stop: tokio::sync::watch::Receiver<bool>,
    ) -> Result<Recovery, ProviderError> {
        let tts = self.provider(conn);
        // A probe runs in the background, past the request it recovers for: in its job, under no
        // span.
        let cx = req.cx().clone().continuing(None);
        let source = match req {
            TtsReq::OneShot { text, .. } => tts.synthesize(&cx, text).await?,
            TtsReq::Stream { log, expressive, .. } => {
                let tokens = log.settled_tokens().await;
                if tokens.is_empty() {
                    // `tts:504-506`: a probe needs text, and the request pushed none.
                    return Ok(Recovery::NotYet);
                }
                let (sink, source) = tts.stream(&cx, *expressive).await?.split();
                // A private replay channel; unlike STT, the probe does not join the live leg.
                for t in &tokens {
                    send(sink.as_ref(), t);
                }
                sink.end_input();
                source
            }
        };
        drain_probe(source, self.clock.as_ref(), stop).await
    }
}

/// An instance's stream, ending its `tts_request` with the stream: at its end with the text the
/// adapter was given, on its error, or as cancelled when closed before either.
struct Traced {
    source: TtsSource,
    req: TtsReq,
    /// `None` once ended.
    request: Option<telemetry::TtsTrace>,
    run: Option<telemetry::ScopedSpan<telemetry::TtsRequestRun>>,
}

impl StreamOutput<SynthesizedAudio> for Traced {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<SynthesizedAudio>, ProviderError>> {
        Box::pin(async move {
            let next = self.source.next().await;
            match &next {
                Ok(Some(audio)) => {
                    if let Some(request) = &mut self.request {
                        request.observe(audio);
                    }
                }
                Ok(None) => {
                    if let Some(mut request) = self.request.take() {
                        request.end_attempt(self.run.take(), None);
                        request.end(self.req.spoken(), None);
                    }
                }
                Err(error) => {
                    if let Some(mut request) = self.request.take() {
                        request.end_attempt(self.run.take(), Some(error));
                        request.fail(error);
                    }
                }
            }
            next
        })
    }

    fn cancel_safe(&self) -> bool {
        self.source.cancel_safe()
    }

    fn close(self: Box<Self>, deadline: Ts) -> BoxFuture<'static, Closed> {
        self.source.close(deadline)
    }
}

impl ProviderStream<SynthesizedAudio> for Traced {
    fn started_time(&self) -> Option<f64> {
        self.source.started_time()
    }
}

// ---------------------------------------------------------------------------------------------
// The adapter
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TtsChainError {
    NoBackends,
    /// `ValueError("all TTS must have the same number of channels")` (`tts:83-84`).
    MixedChannels(Vec<u16>),
}

impl std::fmt::Display for TtsChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TtsChainError::NoBackends => f.write_str("at least one TTS instance must be provided"),
            TtsChainError::MixedChannels(c) => {
                write!(f, "all TTS must have the same number of channels, got {c:?}")
            }
        }
    }
}
impl std::error::Error for TtsChainError {}

/// `tts:92-95`: `streaming` is `any`, `aligned_transcript` is `all`, over the members as given.
pub fn merge_capabilities(entries: &[TtsCapabilities]) -> TtsCapabilities {
    TtsCapabilities {
        streaming: entries.iter().any(|c| c.streaming),
        aligned_transcript: entries.iter().all(|c| c.aligned_transcript),
    }
}

#[derive(Clone)]
pub struct FallbackTts {
    chain: Arc<FallbackChain<TtsBackend>>,
    capabilities: TtsCapabilities,
    sample_rate: u32,
    num_channels: u16,
    needs_resampling: Arc<Vec<bool>>,
    entry_rates: Arc<Vec<u32>>,
    caller_conn: ApiConnectOptions,
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
    ) -> Result<Self, TtsChainError> {
        let capabilities =
            merge_capabilities(&ttss.iter().map(|t| t.capabilities()).collect::<Vec<_>>());
        let entries: Vec<Arc<TtsBackend>> =
            ttss.iter().map(|t| TtsBackend::new(t, Arc::clone(&clock))).collect();
        let channels: Vec<u16> = entries.iter().map(|b| b.inner.num_channels()).collect();
        let Some(&num_channels) = channels.first() else { return Err(TtsChainError::NoBackends) };
        if channels.iter().any(|c| *c != num_channels) {
            return Err(TtsChainError::MixedChannels(channels));
        }
        let entry_rates: Vec<u32> = entries.iter().map(|b| b.inner.sample_rate()).collect();
        let (target, needs_resampling) = resampling_plan(&entry_rates, sample_rate);
        let chain = FallbackChain::new(entries, policy, clock, spawner, observer)
            .map_err(|NoBackends| TtsChainError::NoBackends)?;
        Ok(FallbackTts {
            chain: Arc::new(chain),
            capabilities,
            sample_rate: target,
            num_channels,
            needs_resampling: Arc::new(needs_resampling),
            entry_rates: Arc::new(entry_rates),
            caller_conn: crate::fallback::caller_default(),
        })
    }

    pub fn chain(&self) -> &Arc<FallbackChain<TtsBackend>> {
        &self.chain
    }

    pub fn active(&self) -> Arc<dyn Tts> {
        Arc::clone(&self.chain.active().inner)
    }

    pub fn needs_resampling(&self, index: usize) -> bool {
        self.needs_resampling.get(index).copied().unwrap_or(false)
    }

    fn spawn_pump(&self, req: TtsReq, slot: RecoveryMode, log: Option<Arc<TokenLog>>) -> TtsSource {
        let call = ChainCall {
            conn_options: self.caller_conn,
            accepted_at: self.chain.clock().now(),
            slot,
        };
        let leg = Arc::new(TtsLeg {
            needs_resampling: Arc::clone(&self.needs_resampling),
            entry_rates: Arc::clone(&self.entry_rates),
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
    needs_resampling: Arc<Vec<bool>>,
    entry_rates: Arc<Vec<u32>>,
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
        mut stop: watch::Receiver<bool>,
    ) -> BoxFuture<'a, Attempt<()>> {
        Box::pin(async move {
            let resampler = match self.resampler_for(index) {
                Ok(resampler) => resampler,
                Err(error) => {
                    source.close(chain.clock().now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await;
                    return Attempt::Failed { error, produced_output: false };
                }
            };
            let outcome =
                self.forward_audio(&chain, index, &mut source, resampler, tx, &mut stop).await;
            if let Some(log) = self.log.as_ref() {
                log.clear_main();
            }
            // An attempt that failed before any audio still anchors the request.
            self.anchor_started(&source);
            source.close(chain.clock().now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await;
            outcome
        })
    }
}

impl TtsLeg {
    /// One resampler per attempt (`tts:262-269`): no history crosses a failover. Built from the
    /// entry's declared rate, which every frame is read as.
    fn resampler_for(&self, index: usize) -> Result<Option<AudioResampler>, ProviderError> {
        if !self.needs_resampling.get(index).copied().unwrap_or(false) {
            return Ok(None);
        }
        let from = self.entry_rates.get(index).copied().unwrap_or(self.target);
        AudioResampler::fixed(AudioFormat::new(from, self.channels), self.target, Quality::Medium)
            .map(Some)
            .map_err(|e| ProviderError::Local(ApiError::new(format!("resampler: {e}"))))
    }

    async fn forward_audio(
        &self,
        chain: &FallbackChain<TtsBackend>,
        index: usize,
        source: &mut TtsSource,
        mut resampler: Option<AudioResampler>,
        tx: &mut mpsc::Sender<Result<SynthesizedAudio, ProviderError>>,
        stop: &mut watch::Receiver<bool>,
    ) -> Attempt<()> {
        let mut produced = false;
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
            match Self::deliver(audio, resampler.as_mut(), tx).await {
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

    /// The request's started time, anchored by the first streamed attempt that reports one. The
    /// source is asked with no lock held: a host stream answers under the GIL.
    fn anchor_started(&self, source: &TtsSource) -> Option<f64> {
        if self.log.is_none() {
            return None;
        }
        // `started_time` is a plain read, so concurrent attempts settle the anchor under the lock.
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
    fn with_conn_options(&self, conn: &ApiConnectOptions) -> Option<Arc<dyn Tts>> {
        Some(Arc::new(self.configured(Some(*conn))))
    }

    fn label(&self) -> String {
        "tts.FallbackAdapter".into()
    }

    fn request_span(&self) -> telemetry::RequestSpan {
        telemetry::RequestSpan::FallbackAdapter
    }

    /// The **active** member's identity (L-25: adapters are transparent to the metrics).
    fn id(&self) -> lk_agents_abi::ProviderId {
        self.active().id()
    }

    fn prewarm(&self) {
        for entry in self.chain.entries() {
            entry.inner().prewarm();
        }
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

    fn synthesize<'a>(
        &'a self,
        cx: &telemetry::Cx,
        text: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, Result<TtsSource, ProviderError>> {
        let cx = cx.clone();
        Box::pin(async move {
            Ok(self.spawn_pump(
                TtsReq::OneShot { text: Arc::from(text), cx },
                RecoveryMode::OneShot,
                None,
            ))
        })
    }

    fn stream(
        &self,
        cx: &telemetry::Cx,
        expressive: bool,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<lk_agents_abi::DuplexStream<TtsSink, TtsSource>, ProviderError>,
    > {
        let cx = cx.clone();
        Box::pin(async move {
            let log = TokenLog::new();
            let source = self.spawn_pump(
                TtsReq::Stream { log: Arc::clone(&log), expressive, cx },
                RecoveryMode::Streaming,
                Some(Arc::clone(&log)),
            );
            Ok(lk_agents_abi::DuplexStream::new(Box::new(LoggingSink(log)) as TtsSink, source))
        })
    }
}

impl port::FallbackTts for FallbackTts {
    type Stream = TtsStream;
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        ttss: Vec<Arc<dyn Tts>>,
        max_retry_per_tts: u32,
        attempt_timeout: Option<f64>,
        sample_rate: Option<u32>,
        on_availability: Option<Arc<dyn ChainObserver>>,
    ) -> Result<Self, InvalidPayload> {
        // `sample_rate = None` takes the maximum of the entries' rates and upsamples the rest.
        let policy = ChainPolicy {
            max_retry_per_backend: max_retry_per_tts,
            attempt_timeout,
            ..ChainPolicy::tts_defaults()
        };
        let observer = on_availability.unwrap_or_else(|| Arc::new(crate::fallback::IgnoreEvents));
        let spawner = (env.spawner)().map_err(InvalidPayload)?;
        FallbackTts::new(ttss, sample_rate, policy, env.clock.clone(), spawner, observer)
            .map_err(|error| InvalidPayload(error.to_string()))
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
    async fn synthesize(
        &self,
        text: String,
        conn_options: Option<ApiConnectOptions>,
    ) -> Result<Self::Stream, ProviderError> {
        Tts::synthesize(&self.configured(conn_options), &telemetry::Cx::NONE, &text)
            .await
            .map(|source| TtsStream::new(None, source, self.chain.clock()))
    }
    async fn stream(
        &self,
        conn_options: Option<ApiConnectOptions>,
        expressive: bool,
    ) -> Result<Self::Stream, ProviderError> {
        Tts::stream(&self.configured(conn_options), &telemetry::Cx::NONE, expressive).await.map(
            |stream| {
                let (sink, source) = stream.split();
                TtsStream::new(Some(sink), source, self.chain.clock())
            },
        )
    }
}

impl port::TtsTokenLog for TokenLog {
    fn new() -> Self {
        Self::default()
    }
    fn push(&self, text: String) {
        TokenLog::push(self, Token::Text(text));
    }
    fn flush(&self) {
        TokenLog::push(self, Token::Flush);
    }
    fn end(&self) {
        TokenLog::end(self);
    }
    /// Everything pushed so far (`None` for a flush) and whether the input ended.
    fn snapshot(&self) -> (Vec<Option<String>>, bool) {
        let (tokens, ended) = TokenLog::snapshot(self);
        let tokens = tokens
            .into_iter()
            .map(|token| match token {
                Token::Text(text) => Some(text),
                Token::Flush => None,
            })
            .collect();
        (tokens, ended)
    }
}
