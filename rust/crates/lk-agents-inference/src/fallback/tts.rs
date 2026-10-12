//! `tts/fallback_adapter.py`. The output rate is chosen for the whole list and per-entry
//! resampling rides on the chain (`tts:86-118,262-282`). [`TokenLog`] replays pushed tokens into a
//! fresh input on failover (`tts:428-434`) and into the streaming probe (`tts:504-531`). A failure
//! after output returns success ([`crate::fallback::policy::AfterOutput`], `tts:293-297`).

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lk_agents_abi::{StreamSink, SynthesizedAudio, Tts, TtsCapabilities, TtsSink, TtsSource};
use lk_agents_abi::{BoxFuture, Spawner, SpawnerExt};
use lk_agents_abi::Clock;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_media::{AudioFormat, FrameResampler, Quality};
use lk_agents_abi::ApiConnectOptions;
use tokio::sync::{mpsc, watch};

use crate::fallback::chain::{
    Attempt, Backend, ChainObserver, FallbackChain, NoBackends, Recovery, RecoverySlot,
};
use crate::fallback::channel::{ChainTask, ChannelSource, drain_probe, next_or_stop};
use crate::fallback::policy::{ChainPolicy, bound_views, resampling_plan};

fn send(sink: &TtsSink, token: &Token) {
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
#[lk_agents_abi::class(name = "TtsTokenLog")]
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
    main: Option<TtsSink>,
}

#[lk_agents_abi::host(object(name = "TtsTokenLog"))]
impl TokenLog {
    #[host(new)]
    pub fn for_host() -> Self {
        TokenLog::default()
    }

    #[host(method(name = "push"))]
    pub fn push_text(&self, text: String) {
        self.push(Token::Text(text));
    }

    #[host(method(name = "flush"))]
    pub fn push_flush(&self) {
        self.push(Token::Flush);
    }

    #[host(method(name = "end"))]
    pub fn end_input(&self) {
        self.end();
    }

    /// Everything pushed so far (`None` for a flush) and whether the input ended: what the host
    /// replays into a fresh stream's channel.
    #[host(method(name = "snapshot"))]
    pub fn replay_tokens(&self) -> (Vec<Option<String>>, bool) {
        let (tokens, ended) = self.snapshot();
        let tokens = tokens
            .into_iter()
            .map(|t| match t {
                Token::Text(text) => Some(text),
                Token::Flush => None,
            })
            .collect();
        (tokens, ended)
    }
}

impl TokenLog {
    pub fn new() -> Arc<Self> {
        Arc::new(TokenLog::default())
    }

    pub fn snapshot(&self) -> (Vec<Token>, bool) {
        let s = self.inner.lock().expect("token log");
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

    /// Catch up a fresh child sink (`tts:428-434`) under one lock, so a token pushed during the
    /// failover is neither replayed twice nor lost.
    fn set_main(&self, sink: TtsSink) {
        let mut s = self.inner.lock().expect("token log");
        for t in &s.tokens {
            send(&sink, t);
        }
        if s.ended {
            sink.end_input();
        }
        s.main = Some(sink);
    }

    fn clear_main(&self) {
        self.inner.lock().expect("token log").main = None;
    }

    pub fn push(&self, token: Token) {
        let mut s = self.inner.lock().expect("token log");
        if let Some(sink) = s.main.as_ref() {
            send(sink, &token);
        }
        s.tokens.push(token);
    }

    pub fn end(&self) {
        let mut s = self.inner.lock().expect("token log");
        s.ended = true;
        if let Some(sink) = s.main.as_ref() {
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

impl StreamSink<String> for LoggingSink {
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

#[derive(Clone)]
pub enum TtsReq {
    OneShot { text: Arc<str> },
    Stream { log: Arc<TokenLog> },
}


pub struct TtsBackend {
    inner: Arc<dyn Tts>,
    probe: Arc<dyn Tts>,
    clock: Arc<dyn Clock>,
}

impl TtsBackend {
    pub fn inner(&self) -> &Arc<dyn Tts> {
        &self.inner
    }

    pub fn new(tts: &Arc<dyn Tts>, policy: &ChainPolicy, clock: Arc<dyn Clock>) -> Arc<Self> {
        let (inner, probe) = bound_views(tts, policy, |conn| tts.with_conn_options(conn));
        Arc::new(TtsBackend { inner, probe, clock })
    }
}

#[async_trait]
impl Backend for TtsBackend {
    type Req = TtsReq;
    type Session = TtsSource;

    fn label(&self) -> String {
        self.inner.label()
    }

    async fn open(
        &self,
        req: &Self::Req,
        _conn: &ApiConnectOptions,
    ) -> Result<Self::Session, ProviderError> {
        let tts = &self.inner;
        match req {
            TtsReq::OneShot { text } => tts.synthesize(text).await,
            TtsReq::Stream { log } => {
                // No `StreamAdapter` substitution here (`tts:326-335`); a member without a duplex
                // path fails its open.
                let (sink, source) = tts.stream().await?;
                log.set_main(sink);
                Ok(source)
            }
        }
    }

    async fn recover(
        &self,
        req: &Self::Req,
        _conn: &ApiConnectOptions,
        _slot: RecoverySlot,
        stop: tokio::sync::watch::Receiver<bool>,
    ) -> Result<Recovery, ProviderError> {
        let tts = &self.probe;
        let source = match req {
            TtsReq::OneShot { text } => tts.synthesize(text).await?,
            TtsReq::Stream { log } => {
                let tokens = log.settled_tokens().await;
                if tokens.is_empty() {
                    // `tts:504-506`: a probe needs text, and the request pushed none.
                    return Ok(Recovery::NotYet);
                }
                let (sink, source) = tts.stream().await?;
                // A private replay channel; unlike STT, the probe does not join the live leg.
                for t in &tokens {
                    send(&sink, t);
                }
                sink.end_input();
                source
            }
        };
        drain_probe(source, self.clock.as_ref(), stop).await
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

/// `tts:92-95`: `streaming` is `any`, `aligned_transcript` is `all`. Without the reference's
/// `StreamAdapter` substitution, `any` over-promises for entries lacking a duplex path.
pub fn merge_capabilities(entries: &[TtsCapabilities]) -> TtsCapabilities {
    TtsCapabilities {
        streaming: entries.iter().any(|c| c.streaming),
        aligned_transcript: entries.iter().all(|c| c.aligned_transcript),
    }
}

pub struct FallbackTts {
    chain: Arc<FallbackChain<TtsBackend>>,
    spawner: Arc<dyn Spawner>,
    capabilities: TtsCapabilities,
    sample_rate: u32,
    num_channels: u16,
    needs_resampling: Arc<Vec<bool>>,
    entry_rates: Arc<Vec<u32>>,
    caller_conn: ApiConnectOptions,
}

impl FallbackTts {
    pub fn new(
        ttss: Vec<Arc<dyn Tts>>,
        sample_rate: Option<u32>,
        policy: ChainPolicy,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        observer: Arc<dyn ChainObserver>,
    ) -> Result<Self, TtsChainError> {
        let entries: Vec<Arc<TtsBackend>> =
            ttss.iter().map(|t| TtsBackend::new(t, &policy, Arc::clone(&clock))).collect();
        let channels: Vec<u16> = entries.iter().map(|b| b.inner.num_channels()).collect();
        let Some(&num_channels) = channels.first() else { return Err(TtsChainError::NoBackends) };
        if channels.iter().any(|c| *c != num_channels) {
            return Err(TtsChainError::MixedChannels(channels));
        }
        let entry_rates: Vec<u32> = entries.iter().map(|b| b.inner.sample_rate()).collect();
        let (target, needs_resampling) = resampling_plan(&entry_rates, sample_rate);
        let capabilities =
            merge_capabilities(&entries.iter().map(|b| b.inner.capabilities()).collect::<Vec<_>>());
        let chain = FallbackChain::new(entries, policy, clock, Arc::clone(&spawner), observer)
            .map_err(|NoBackends| TtsChainError::NoBackends)?;
        Ok(FallbackTts {
            chain: Arc::new(chain),
            spawner,
            capabilities,
            sample_rate: target,
            num_channels,
            needs_resampling: Arc::new(needs_resampling),
            entry_rates: Arc::new(entry_rates),
            caller_conn: crate::fallback::policy::caller_default(),
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

    fn spawn_pump(&self, req: TtsReq, slot: RecoverySlot, log: Option<Arc<TokenLog>>) -> TtsSource {
        let (tx, rx) = mpsc::channel::<Result<SynthesizedAudio, ProviderError>>(32);
        let chain = Arc::clone(&self.chain);
        let drain_chain = Arc::clone(&self.chain);
        let caller_conn = self.caller_conn;
        let needs = Arc::clone(&self.needs_resampling);
        let rates = Arc::clone(&self.entry_rates);
        let target = self.sample_rate;
        let channels = self.num_channels;

        // The streamed request is measured as one TTS node (`tts:350-355`): its started time is
        // the first attempt's that handed text to a provider, kept across a failover.
        let started: Arc<Mutex<Option<f64>>> = Arc::default();
        let (stop_rx, done_tx, chain_task) = ChainTask::pair(chain.clock());
        let handle = self.spawner.spawn_unboxed(async move {
            // Dropped with the task, so an aborted request still releases its probes.
            let _finish = FinishOnDrop(log.clone());
            let mut tx = tx;
            let result = chain
                .run(&req, &caller_conn, slot, &mut tx, move |index, _b, session, tx| {
                    let chain = Arc::clone(&drain_chain);
                    let needs = Arc::clone(&needs);
                    let rates = Arc::clone(&rates);
                    let log = log.clone();
                    let started = Arc::clone(&started);
                    let mut stop = stop_rx.clone();
                    Box::pin(async move {
                        let mut source = session;
                        let streamed = log.is_some();
                        let capture = |source: &TtsSource| -> Option<f64> {
                            let mut anchor = started.lock().expect("tts started time");
                            if streamed && anchor.is_none() {
                                *anchor = source.started_time();
                            }
                            *anchor
                        };
                        // One resampler per attempt (`tts:262-269`): no history crosses a failover.
                        // Built from the entry's declared rate, which every frame is read as.
                        let mut resampler = if needs.get(index).copied().unwrap_or(false) {
                            let from = rates.get(index).copied().unwrap_or(target);
                            match FrameResampler::fixed(
                                AudioFormat::new(from, channels),
                                target,
                                Quality::Medium,
                            ) {
                                Ok(r) => Some(r),
                                Err(e) => {
                                    source.close(chain.teardown_deadline()).await;
                                    return Attempt::Failed {
                                        error: ProviderError::Local(ApiError::new(format!(
                                            "resampler: {e}"
                                        ))),
                                        produced_output: false,
                                    };
                                }
                            }
                        } else {
                            None
                        };

                        let mut produced = false;
                        let outcome = loop {
                            match next_or_stop(source.as_mut(), &mut stop).await {
                                Ok(None) => break Attempt::Served(()),
                                Ok(Some(mut audio)) => {
                                    if !produced {
                                        produced = true;
                                        chain.mark_serving(index);
                                    }
                                    if let Some(at) = capture(&source) {
                                        audio.started_time = Some(at);
                                    }
                                    let out = match resampler.as_mut() {
                                        None => Some(audio.clone()),
                                        Some(r) => match r.push(&audio.frame) {
                                            Ok(ready) => ready.into_iter().next().map(|frame| {
                                                let mut a = audio.clone();
                                                a.frame = frame;
                                                a
                                            }),
                                            Err(e) => {
                                                break Attempt::Failed {
                                                    error: ProviderError::Local(ApiError::new(
                                                        format!("resample: {e}"),
                                                    )),
                                                    produced_output: produced,
                                                };
                                            }
                                        },
                                    };
                                    // `tts:464-466`: drain the filter tail at every `is_final`.
                                    let is_final = audio.is_final;
                                    if let Some(a) = out
                                        && tx.send(Ok(a)).await.is_err()
                                    {
                                        break Attempt::Served(());
                                    }
                                    if is_final
                                        && let Some(r) = resampler.as_mut()
                                        && let Ok(Some(tail)) = r.flush()
                                    {
                                        let mut a = audio.clone();
                                        a.frame = tail;
                                        if tx.send(Ok(a)).await.is_err() {
                                            break Attempt::Served(());
                                        }
                                    }
                                }
                                Err(error) => {
                                    break Attempt::Failed { error, produced_output: produced };
                                }
                            }
                        };
                        if let Some(log) = log.as_ref() {
                            log.clear_main();
                        }
                        // An attempt that failed before any audio still anchors the request.
                        capture(&source);
                        source.close(chain.teardown_deadline()).await;
                        outcome
                    }) as BoxFuture<'_, Attempt<()>>
                })
                .await;
            match result {
                // Truncated is a success: the audio already sent plays out (`tts:293-297`).
                Ok(_) => {}
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                }
            }
            let _ = done_tx.send(());
        });

        Box::new(ChannelSource { rx, task: Some(handle), chain: chain_task }) as TtsSource
    }
}

/// `tts.FallbackAdapter`'s engine as the host builds it; the public class stays Python.
#[lk_agents_abi::host(object(cell = "arc", as_dyn(Tts = as_tts)))]
impl FallbackTts {
    /// `ttss` in order. `sample_rate = None` takes the maximum of the entries' rates and upsamples
    /// the rest, as the reference does. Views are bound here, as in `FallbackStt::for_host`.
    #[host(new(kw_only = "max_retry_per_tts", defaults(max_retry_per_tts = 2)))]
    pub fn for_host(
        ttss: Vec<Arc<dyn Tts>>,
        max_retry_per_tts: u32,
        sample_rate: Option<u32>,
        on_availability: Option<Arc<dyn ChainObserver>>,
        #[host(ambient)] clock: Arc<dyn Clock>,
        #[host(ambient)] spawner: Arc<dyn Spawner>,
    ) -> Result<Self, TtsChainError> {
        let policy = ChainPolicy { max_retry_per_backend: max_retry_per_tts, ..ChainPolicy::tts_defaults() };
        let observer = on_availability.unwrap_or_else(|| Arc::new(crate::fallback::IgnoreEvents));
        FallbackTts::new(ttss, sample_rate, policy, clock, spawner, observer)
    }

    /// The chain as the provider the session holds (a native `_core.FallbackTts` is not proxied).
    pub fn as_tts(self: &Arc<Self>) -> Arc<dyn Tts> {
        Arc::clone(self) as Arc<dyn Tts>
    }

    #[host(getter)]
    pub fn active_index(&self) -> usize {
        self.chain.active_index()
    }

    /// Cancel every recovery probe in flight, idempotently.
    pub fn aclose(&self) {
        self.chain.aclose();
    }

    /// `aclose()`: every recovery probe in flight closes its stream, awaited (bounded).
    #[host(method(name = "aclose"))]
    pub async fn aclose_for_host(&self) {
        self.chain.close().await;
    }

    /// The merged capabilities, as `TTSCapabilities` keyword arguments.
    #[host(getter(name = "capabilities"))]
    pub fn merged_capabilities(&self) -> TtsCapabilities {
        self.capabilities
    }

    #[host(getter(name = "sample_rate"))]
    pub fn output_sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[host(getter(name = "num_channels"))]
    pub fn output_num_channels(&self) -> u16 {
        self.num_channels
    }

    /// The one-shot path: a stream with no push half.
    #[host(method(name = "synthesize", stream = lk_agents_abi::FallbackTtsStream))]
    pub async fn synthesize_for_host(&self, text: String) -> Result<TtsSource, ProviderError> {
        Tts::synthesize(self, &text).await
    }

    /// `stream()`: the duplex path, text in, audio out.
    #[host(method(name = "stream", stream = lk_agents_abi::FallbackTtsStream))]
    pub async fn stream_for_host(&self) -> Result<(TtsSink, TtsSource), ProviderError> {
        Tts::stream(self).await
    }
}

#[async_trait]
impl Tts for FallbackTts {
    fn label(&self) -> String {
        "tts.FallbackAdapter".into()
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

    async fn synthesize(&self, text: &str) -> Result<TtsSource, ProviderError> {
        Ok(self.spawn_pump(TtsReq::OneShot { text: Arc::from(text) }, RecoverySlot::OneShot, None))
    }

    async fn stream(&self) -> Result<(TtsSink, TtsSource), ProviderError> {
        let log = TokenLog::new();
        let source = self.spawn_pump(
            TtsReq::Stream { log: Arc::clone(&log) },
            RecoverySlot::Streaming,
            Some(Arc::clone(&log)),
        );
        Ok((Box::new(LoggingSink(log)) as TtsSink, source))
    }
}
