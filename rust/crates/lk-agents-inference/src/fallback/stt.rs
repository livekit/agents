//! `stt/fallback_adapter.py`. Streaming recovery is a live tee, not a replayed request: the probe
//! hears a copy of the live audio (`stt:341-361`) and recovers on a non-empty `FINAL_TRANSCRIPT`
//! (`stt:456-466`), so [`Backend::Req`] carries an [`InputTee`]. One availability flag serves two
//! recovery slots, hence one [`FallbackChain`] with a [`RecoveryMode`] argument. Each attempt binds the resolved policy immediately before opening its provider.

use crate::telemetry;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::tokenize::language::LanguageCode;
use lk_agents_abi::{
    AlignedTranscript, AudioFrame, StreamInput, Stt, SttCapabilities, SttSink, SttSource,
};
use lk_agents_abi::{ApiConnectOptions, SpeechEvent, SpeechEventType};
use lk_agents_abi::{BoxFuture, Spawner};
use lk_agents_abi::{Clock, HOST_TEARDOWN_TIMEOUT};
use parking_lot::Mutex;

use lk_agents_abi::llm::inference::InvalidPayload;
use lk_agents_net::env::ProcessEnv;
use tokio::sync::{mpsc, watch};

use crate::fallback::ChainPolicy;
use crate::fallback::chain::{
    Attempt, Backend, ChainCall, ChainObserver, ChainOutcome, FallbackChain, NoBackends, Recovery,
    RecoveryMode,
};
use crate::fallback::channel::{ServeAttempt, forward, probe_next};

// ---------------------------------------------------------------------------------------------
// The live input tee
// ---------------------------------------------------------------------------------------------

type InputProbe = (u64, Arc<dyn StreamInput<AudioFrame>>);

/// Fans the audio-in leg out to the serving stream and every probe. `end` is recorded so a stream
/// installed by a later failover still gets it (the reference's hole at `stt:363-365`).
/// Sinks are called with no lock held: a host STT's sink takes the GIL.
pub struct InputTee {
    main: Mutex<Option<Arc<dyn StreamInput<AudioFrame>>>>,
    probes: Mutex<Vec<InputProbe>>,
    next_id: AtomicU64,
    /// Replayed into the next serving stream; taken before `main`.
    preroll: Mutex<Preroll>,
}

/// One buffered input. `Flush` keeps its position, or two utterances would merge.
#[derive(Clone)]
enum PreFrame {
    Audio(AudioFrame),
    /// Pre-connect audio: replayed as head, never evicted and outside the bound.
    Head(AudioFrame),
    Flush,
}

#[derive(Default)]
struct Preroll {
    pending: std::collections::VecDeque<PreFrame>,
    /// The `Head` entries in `pending`.
    head: usize,
    dropped_frames: u64,
    ended: bool,
}

impl Default for InputTee {
    fn default() -> Self {
        InputTee {
            main: Mutex::new(None),
            probes: Mutex::new(Vec::new()),
            next_id: AtomicU64::new(0),
            preroll: Mutex::new(Preroll::default()),
        }
    }
}

impl InputTee {
    pub fn new() -> Arc<Self> {
        Arc::new(InputTee::default())
    }

    /// How much audio is held while no stream is serving: the bound of every reconnect.
    pub const PREROLL_FRAMES: usize = lk_agents_abi::stt::HELD_FRAMES;

    /// Replays what arrived while no stream served (as the reference's `_input_ch` buffers), then
    /// ends it if the caller already finished. Installed once nothing is buffered, so a
    /// concurrent `push` can neither overtake a buffered frame nor be left behind in the preroll.
    fn set_main(&self, sink: SttSink) {
        let sink: Arc<dyn StreamInput<AudioFrame>> = Arc::from(sink);
        let ended = loop {
            let pending: Vec<PreFrame> = {
                let mut preroll = self.preroll.lock();
                if preroll.pending.is_empty() {
                    if !preroll.ended {
                        *self.main.lock() = Some(Arc::clone(&sink));
                    }
                    break preroll.ended;
                }
                preroll.head = 0;
                preroll.pending.drain(..).collect()
            };
            for item in pending {
                let _ = match item {
                    PreFrame::Audio(f) => sink.push(f),
                    PreFrame::Head(f) => sink.head(f),
                    PreFrame::Flush => sink.flush(),
                };
            }
        };
        // Replay first, then end: a short utterance ended before the socket opened is still transcribed.
        if ended {
            sink.end_input();
        }
    }

    /// The serving stream, or `item` buffered for the next one when none serves.
    fn serving(
        &self,
        item: PreFrame,
    ) -> Result<Option<Arc<dyn StreamInput<AudioFrame>>>, ProviderError> {
        let mut preroll = self.preroll.lock();
        let main = self.main.lock().clone();
        if main.is_none() {
            Self::buffer(&mut preroll, item)?;
        }
        Ok(main)
    }

    /// Preserve utterance boundaries, accounting for each evicted audio frame. A full queue
    /// of boundaries refuses more input rather than growing or merging distinct utterances.
    fn buffer(preroll: &mut Preroll, item: PreFrame) -> Result<(), ProviderError> {
        if matches!(item, PreFrame::Head(_)) {
            preroll.head += 1;
        } else if preroll.pending.len() - preroll.head >= Self::PREROLL_FRAMES {
            let Some(index) =
                preroll.pending.iter().position(|item| matches!(item, PreFrame::Audio(_)))
            else {
                return Err(ApiError::new("STT preroll is full of pending flush boundaries")
                    .with_retryable(false)
                    .into());
            };
            preroll.pending.remove(index);
            preroll.dropped_frames += 1;
        }
        preroll.pending.push_back(item);
        Ok(())
    }

    pub fn dropped_preroll_frames(&self) -> u64 {
        self.preroll.lock().dropped_frames
    }

    fn clear_main(&self) {
        *self.main.lock() = None;
    }

    fn add_probe(&self, sink: SttSink) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.probes.lock().push((id, Arc::from(sink)));
        id
    }

    fn remove_probe(&self, id: u64) {
        self.probes.lock().retain(|(i, _)| *i != id);
    }

    /// Errors are swallowed (`stt:349-350`): a dead probe must not take down the serving path.
    fn fanout(
        &self,
        main: Option<Arc<dyn StreamInput<AudioFrame>>>,
        f: impl Fn(&dyn StreamInput<AudioFrame>),
    ) {
        let probes: Vec<_> = self.probes.lock().iter().map(|(_, sink)| Arc::clone(sink)).collect();
        probes.iter().chain(&main).for_each(|sink| f(sink.as_ref()));
    }
}

struct TeeSink(Arc<InputTee>);

impl StreamInput<AudioFrame> for TeeSink {
    fn push(&self, item: AudioFrame) -> Result<(), ProviderError> {
        // Buffered when no stream is serving; probes still get it.
        let main = self.0.serving(PreFrame::Audio(item.clone()))?;
        self.0.fanout(main, |s| {
            let _ = s.push(item.clone());
        });
        Ok(())
    }

    fn head(&self, item: AudioFrame) -> Result<(), ProviderError> {
        let main = self.0.serving(PreFrame::Head(item.clone()))?;
        self.0.fanout(main, |s| {
            let _ = s.head(item.clone());
        });
        Ok(())
    }

    fn flush(&self) -> Result<(), ProviderError> {
        let main = self.0.serving(PreFrame::Flush)?;
        self.0.fanout(main, |s| {
            let _ = s.flush();
        });
        Ok(())
    }

    fn end_input(&self) {
        // Only the serving stream is ended; a probe closes when its drain finishes (`stt:432`).
        let main = {
            let mut preroll = self.0.preroll.lock();
            preroll.ended = true;
            self.0.main.lock().clone()
        };
        if let Some(sink) = main {
            sink.end_input();
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The backend
// ---------------------------------------------------------------------------------------------

#[derive(Clone)]
pub enum SttReq {
    /// `_recognize_impl`. `Arc` because recovery replays the same buffer (`stt:217-222`).
    Batch { frames: Arc<[AudioFrame]>, language: Option<LanguageCode>, cx: telemetry::Cx },
    /// `STT.stream`. `tee` is live; every leg opens under `cx`.
    Stream {
        language: Option<LanguageCode>,
        tee: Arc<InputTee>,
        offset: Option<LegOffset>,
        cx: telemetry::Cx,
    },
}

/// The caller's `start_time_offset` when its stream opened, and that instant: a leg opened later
/// is offset by the time since (`stt:377-380`), so its timestamps stay on the input's timeline.
#[derive(Clone, Copy, Debug)]
pub struct LegOffset {
    pub start_time_offset: f64,
    pub opened: lk_agents_abi::Ts,
}

impl LegOffset {
    fn at(&self, now: lk_agents_abi::Ts) -> f64 {
        self.start_time_offset + now.saturating_duration_since(self.opened).as_secs_f64()
    }
}

pub enum SttSession {
    Batch(SpeechEvent),
    Stream(SttSource),
}

impl SttSession {
    /// A batch request's attempt: the transcript it opened with.
    fn into_batch(self) -> Attempt<SpeechEvent> {
        match self {
            SttSession::Batch(ev) => Attempt::Served(ev),
            SttSession::Stream(_) => unreachable!("a Batch request opens Batch"),
        }
    }
}

pub struct SttBackend {
    inner: Arc<dyn Stt>,
    clock: Arc<dyn Clock>,
}

impl SttBackend {
    pub fn inner(&self) -> &Arc<dyn Stt> {
        &self.inner
    }

    pub fn new(stt: &Arc<dyn Stt>, clock: Arc<dyn Clock>) -> Arc<Self> {
        Arc::new(SttBackend { inner: stt.clone(), clock })
    }
}

fn no_duplex(label: &str) -> ProviderError {
    ProviderError::Local(ApiError::new(format!("{label} has no streaming path")))
}

#[async_trait]
impl Backend for SttBackend {
    type Req = SttReq;

    type Session = SttSession;

    fn resource_id(&self) -> lk_agents_abi::resources::ResourceId {
        super::backend_resource(self.inner.as_ref(), self.inner.instance_id())
    }

    fn label(&self) -> String {
        self.inner.label()
    }

    async fn open(
        &self,
        req: &Self::Req,
        conn: &ApiConnectOptions,
    ) -> Result<Self::Session, ProviderError> {
        let stt = self.inner.with_conn_options(conn).unwrap_or_else(|| self.inner.clone());
        match req {
            SttReq::Batch { frames, language, cx } => {
                Ok(SttSession::Batch(stt.recognize(cx, frames, language.as_ref()).await?))
            }
            SttReq::Stream { language, tee, offset, cx } => {
                let offset = offset.map(|o| o.at(self.clock.now()));
                let (sink, source) = stt
                    .stream(cx, language.as_ref(), offset)
                    .await?
                    .ok_or_else(|| no_duplex(&self.inner.label()))?
                    .split();
                tee.set_main(sink);
                Ok(SttSession::Stream(source))
            }
        }
    }

    async fn recover(
        &self,
        req: &Self::Req,
        conn: &ApiConnectOptions,
        slot: RecoveryMode,
        stop: tokio::sync::watch::Receiver<bool>,
    ) -> Result<Recovery, ProviderError> {
        let stt = self.inner.with_conn_options(conn).unwrap_or_else(|| self.inner.clone());
        match (req, slot) {
            (SttReq::Batch { frames, language, cx }, _) => {
                // `_try_recognize(recovering=True)`: one call, output discarded, in the job past
                // the request it recovers for, under no span.
                stt.recognize(&cx.clone().continuing(None), frames, language.as_ref()).await?;
                Ok(Recovery::Recovered)
            }
            (SttReq::Stream { language, tee, cx, .. }, _) => {
                self.probe_stream(stt.as_ref(), language.as_ref(), tee, cx, stop).await
            }
        }
    }
}

impl SttBackend {
    /// A probe stream joined to the live tee, read until a non-empty final transcript; a clean
    /// drain with none is `NotYet` (`stt:465-466`). A probe's timestamps are never read, and the
    /// reference leaves its offset at 0.
    async fn probe_stream(
        &self,
        stt: &dyn Stt,
        language: Option<&LanguageCode>,
        tee: &InputTee,
        cx: &telemetry::Cx,
        mut stop: tokio::sync::watch::Receiver<bool>,
    ) -> Result<Recovery, ProviderError> {
        let (sink, mut source) = stt
            .stream(cx, language, None)
            .await?
            .ok_or_else(|| no_duplex(&self.inner.label()))?
            .split();
        let id = tee.add_probe(sink);
        let verdict = loop {
            match probe_next(source.as_mut(), &mut stop).await {
                None | Some(Ok(None)) => break Ok(Recovery::NotYet),
                Some(Ok(Some(ev)))
                    if ev.kind == SpeechEventType::FinalTranscript
                        && ev.alternatives.first().is_some_and(|a| !a.text.is_empty()) =>
                {
                    break Ok(Recovery::Recovered);
                }
                Some(Ok(Some(_))) => {}
                Some(Err(e)) => break Err(e),
            }
        };
        tee.remove_probe(id);
        source.close(self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await;
        verdict
    }
}

// ---------------------------------------------------------------------------------------------
// The adapter
// ---------------------------------------------------------------------------------------------

/// The caller wraps non-streaming entries in `StreamAdapter` (the reference does it, `stt:64-77`).
#[derive(Debug)]
pub enum SttChainError {
    NoBackends(NoBackends),
    NotStreaming(Vec<String>),
}

impl std::fmt::Display for SttChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SttChainError::NoBackends(e) => e.fmt(f),
            SttChainError::NotStreaming(labels) => write!(
                f,
                "STTs do not support streaming: {}. Wrap them with stt.StreamAdapter first.",
                labels.join(", ")
            ),
        }
    }
}
impl std::error::Error for SttChainError {}

/// `stt:79-93`, field for field; `offline_recognize` is not merged, as in the reference.
pub fn merge_capabilities(entries: &[SttCapabilities]) -> SttCapabilities {
    let all = |f: fn(&SttCapabilities) -> bool| entries.iter().all(f);
    let any = |f: fn(&SttCapabilities) -> bool| entries.iter().any(f);
    SttCapabilities {
        streaming: true,
        interim_results: all(|c| c.interim_results),
        diarization: all(|c| c.diarization),
        // "Use the primary's granularity if all providers support it, since the SDK only checks
        // truthiness, not the specific granularity."
        aligned_transcript: if all(|c| c.aligned_transcript != AlignedTranscript::None) {
            entries.first().map_or(AlignedTranscript::None, |c| c.aligned_transcript)
        } else {
            AlignedTranscript::None
        },
        offline_recognize: true,
        keyterms: any(|c| c.keyterms),
        chat_context: any(|c| c.chat_context),
    }
}

/// A host-built chain takes its executor and clock from its env.
pub struct FallbackStt {
    chain: Arc<FallbackChain<SttBackend>>,
    spawner: Arc<dyn Spawner>,
    capabilities: SttCapabilities,
    caller_conn: ApiConnectOptions,
}

impl Clone for FallbackStt {
    fn clone(&self) -> Self {
        FallbackStt {
            chain: Arc::clone(&self.chain),
            spawner: Arc::clone(&self.spawner),
            capabilities: self.capabilities,
            caller_conn: self.caller_conn,
        }
    }
}

impl FallbackStt {
    fn configured(&self, conn_options: Option<ApiConnectOptions>) -> Self {
        Self { caller_conn: conn_options.unwrap_or(self.caller_conn), ..self.clone() }
    }

    pub fn new(
        stts: Vec<Arc<dyn Stt>>,
        policy: ChainPolicy,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        observer: Arc<dyn ChainObserver>,
    ) -> Result<Self, SttChainError> {
        let entries: Vec<Arc<SttBackend>> =
            stts.iter().map(|s| SttBackend::new(s, Arc::clone(&clock))).collect();
        let not_streaming: Vec<String> = entries
            .iter()
            .filter(|b| !b.inner.capabilities().streaming)
            .map(|b| b.inner.label())
            .collect();
        if !not_streaming.is_empty() {
            return Err(SttChainError::NotStreaming(not_streaming));
        }
        let capabilities =
            merge_capabilities(&entries.iter().map(|b| b.inner.capabilities()).collect::<Vec<_>>());
        let chain = FallbackChain::new(entries, policy, clock, Arc::clone(&spawner), observer)
            .map_err(SttChainError::NoBackends)?;
        Ok(FallbackStt {
            chain: Arc::new(chain),
            spawner,
            capabilities,
            caller_conn: crate::fallback::caller_default(),
        })
    }

    pub fn chain(&self) -> &Arc<FallbackChain<SttBackend>> {
        &self.chain
    }

    pub fn active(&self) -> Arc<dyn Stt> {
        Arc::clone(&self.chain.active().inner)
    }

    /// Cancel every recovery probe in flight, idempotently.
    pub fn aclose(&self) {
        self.chain.aclose();
    }
}

/// `stt.FallbackAdapter`'s engine as the host builds it: the generated adapter calls this.
impl lk_agents_abi::stt::fallback::FallbackStt for FallbackStt {
    type Audio = SttSink;
    type Events = SttSource;
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        stts: Vec<Arc<dyn Stt>>,
        attempt_timeout: f64,
        max_retry_per_stt: u32,
        retry_interval: f64,
        on_availability: Option<Arc<dyn ChainObserver>>,
    ) -> Result<Self, InvalidPayload> {
        let policy = ChainPolicy {
            max_retry_per_backend: max_retry_per_stt,
            attempt_timeout: Some(attempt_timeout),
            retry_interval: Some(retry_interval),
            ..ChainPolicy::stt_defaults()
        };
        let observer = on_availability.unwrap_or_else(|| Arc::new(crate::fallback::IgnoreEvents));
        FallbackStt::new(
            stts,
            policy,
            env.clock.clone(),
            (env.spawner)().map_err(InvalidPayload)?,
            observer,
        )
        .map_err(|error| InvalidPayload(error.to_string()))
    }

    fn active_index(&self) -> usize {
        self.chain.active_index()
    }

    fn capabilities(&self) -> SttCapabilities {
        self.capabilities
    }

    async fn aclose(&self) {
        self.chain.close().await;
    }

    async fn recognize(
        &self,
        frames: Vec<AudioFrame>,
        language: Option<LanguageCode>,
        conn_options: Option<ApiConnectOptions>,
    ) -> Result<SpeechEvent, ProviderError> {
        Stt::recognize(
            &self.configured(conn_options),
            &telemetry::Cx::NONE,
            &frames,
            language.as_ref(),
        )
        .await
    }

    async fn stream(
        &self,
        language: Option<LanguageCode>,
        start_time_offset: Option<f64>,
        conn_options: Option<ApiConnectOptions>,
    ) -> Result<lk_agents_abi::DuplexStream<SttSink, SttSource>, ProviderError> {
        // Python's own call: no framework span is current to open under.
        Stt::stream(
            &self.configured(conn_options),
            &telemetry::Cx::NONE,
            language.as_ref(),
            start_time_offset,
        )
        .await
        .map(|opened| opened.expect("the chain's `stream` is always duplex"))
    }
}

impl Stt for FallbackStt {
    fn with_conn_options(&self, conn: &ApiConnectOptions) -> Option<Arc<dyn Stt>> {
        Some(Arc::new(self.configured(Some(*conn))))
    }

    fn label(&self) -> String {
        "stt.FallbackAdapter".into()
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

    /// Every member (`fallback_adapter.py:130`): the one that takes over must already know them.
    fn update_keyterms(&self, keyterms: &[String]) {
        for entry in self.chain.entries() {
            entry.inner().update_keyterms(keyterms);
        }
    }

    /// Every member (`fallback_adapter.py:134`), as keyterms.
    fn push_conversation_item(&self, role: &str, text: &str) {
        for entry in self.chain.entries() {
            entry.inner().push_conversation_item(role, text);
        }
    }

    fn capabilities(&self) -> SttCapabilities {
        self.capabilities
    }

    fn recognize(
        &self,
        cx: &telemetry::Cx,
        buffer: &[lk_agents_abi::AudioFrame],
        language: Option<&LanguageCode>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
        let (cx, buffer, language) = (cx.clone(), buffer.to_vec(), language.cloned());
        Box::pin(async move {
            let req = SttReq::Batch { frames: Arc::from(buffer), language: language.clone(), cx };
            let mut unused = ();
            let ChainOutcome::Served(ev) = self
                .chain
                .run(
                    &req,
                    &self.caller_conn,
                    RecoveryMode::OneShot,
                    &mut unused,
                    |_i, _b, session, _s| Box::pin(std::future::ready(session.into_batch())),
                )
                .await?;
            Ok(ev)
        })
    }

    /// Each leg's `start_time_offset` is `start_time_offset` plus the time since this call.
    fn stream(
        &self,
        cx: &telemetry::Cx,
        language: Option<&LanguageCode>,
        start_time_offset: Option<f64>,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Option<lk_agents_abi::DuplexStream<SttSink, SttSource>>, ProviderError>,
    > {
        let language = language.cloned();
        let cx = cx.clone();
        Box::pin(async move {
            let tee = InputTee::new();
            let opened = self.chain.clock().now();
            let offset =
                start_time_offset.map(|start_time_offset| LegOffset { start_time_offset, opened });
            let req = SttReq::Stream { language, tee: Arc::clone(&tee), offset, cx };
            let call = ChainCall {
                conn_options: self.caller_conn,
                accepted_at: opened,
                slot: RecoveryMode::Streaming,
            };
            let leg = Arc::new(SttLeg { tee: Arc::clone(&tee) });
            let source = self.chain.serve_stream(req, call, leg, |_, _| {}, ());
            Ok(Some(lk_agents_abi::DuplexStream::new(
                Box::new(TeeSink(tee)) as SttSink,
                Box::new(source) as SttSource,
            )))
        })
    }
}

/// An entry's attempt: its transcripts forwarded while it holds the live tee.
struct SttLeg {
    tee: Arc<InputTee>,
}

impl ServeAttempt<SttBackend, SpeechEvent> for SttLeg {
    fn serve<'a>(
        self: Arc<Self>,
        chain: Arc<FallbackChain<SttBackend>>,
        index: usize,
        session: SttSession,
        tx: &'a mut mpsc::Sender<Result<SpeechEvent, ProviderError>>,
        mut stop: watch::Receiver<bool>,
    ) -> BoxFuture<'a, Attempt<()>> {
        Box::pin(async move {
            let SttSession::Stream(mut source) = session else {
                unreachable!("a Stream request opens Stream")
            };
            let outcome = forward(&chain, index, source.as_mut(), &mut stop, tx, |_| true).await;
            // The dead stream leaves the tee before the next entry installs its own.
            self.tee.clear_main();
            source.close(chain.clock().now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await;
            outcome
        })
    }
}

#[cfg(test)]
mod preroll_tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    /// A sink that records what it was handed, in order.
    #[derive(Default)]
    struct Log {
        got: Mutex<Vec<String>>,
        ended: AtomicBool,
    }

    /// The sink half, over a shared [`Log`] the test can read after installing it.
    struct Recorder(Arc<Log>);

    impl StreamInput<AudioFrame> for Recorder {
        fn push(&self, item: AudioFrame) -> Result<(), ProviderError> {
            self.0.got.lock().push(format!("push:{}", item.sample_rate));
            Ok(())
        }
        fn head(&self, item: AudioFrame) -> Result<(), ProviderError> {
            self.0.got.lock().push(format!("head:{}", item.sample_rate));
            Ok(())
        }
        fn flush(&self) -> Result<(), ProviderError> {
            self.0.got.lock().push("flush".to_owned());
            Ok(())
        }
        fn end_input(&self) {
            self.0.ended.store(true, Ordering::SeqCst);
        }
    }

    /// `sample_rate` is used as an identity so assertions read as a sequence.
    fn frame(rate: u32) -> AudioFrame {
        AudioFrame {
            data: vec![0u8; 960].into(),
            sample_rate: rate,
            num_channels: 1,
            samples_per_channel: 480,
        }
    }

    /// Catches: audio pushed before the first entry's socket is open being dropped (shipped: the
    /// first syllable of the first turn went missing).
    #[test]
    fn audio_pushed_before_a_stream_exists_is_replayed_into_it() {
        let tee = InputTee::new();
        let sink = TeeSink(Arc::clone(&tee));

        sink.push(frame(1)).unwrap();
        sink.flush().unwrap();
        sink.push(frame(2)).unwrap();

        let rec = Arc::new(Log::default());
        tee.set_main(Box::new(Recorder(Arc::clone(&rec))));

        assert_eq!(*rec.got.lock(), ["push:1", "flush", "push:2"]);
    }

    /// Catches: the replay running after `end_input`, or not at all when the caller finished early.
    #[test]
    fn a_whole_utterance_pushed_before_the_stream_opens_still_reaches_it() {
        let tee = InputTee::new();
        let sink = TeeSink(Arc::clone(&tee));

        sink.push(frame(1)).unwrap();
        sink.push(frame(2)).unwrap();
        sink.end_input();

        let rec = Arc::new(Log::default());
        tee.set_main(Box::new(Recorder(Arc::clone(&rec))));

        assert_eq!(*rec.got.lock(), ["push:1", "push:2"]);
        assert!(rec.ended.load(Ordering::SeqCst), "the stream was never told the input had ended");
    }

    /// Catches: pre-connect audio buffered before the first stream opens replayed as live audio,
    /// or evicted by the bound, so the serving stream drops the start of the first turn.
    #[test]
    fn pre_connect_audio_is_replayed_as_head_in_full() {
        let tee = InputTee::new();
        let sink = TeeSink(Arc::clone(&tee));
        let head = InputTee::PREROLL_FRAMES as u32 + 10;
        for i in 0..head {
            sink.head(frame(i)).unwrap();
        }
        sink.push(frame(head)).unwrap();

        let rec = Arc::new(Log::default());
        tee.set_main(Box::new(Recorder(Arc::clone(&rec))));
        sink.head(frame(head + 1)).unwrap();

        let mut expected: Vec<String> = (0..head).map(|i| format!("head:{i}")).collect();
        expected.extend([format!("push:{head}"), format!("head:{}", head + 1)]);
        assert_eq!(*rec.got.lock(), expected);
        assert_eq!(tee.dropped_preroll_frames(), 0);
    }

    /// Catches: an unbounded pre-roll.
    #[test]
    fn the_preroll_is_bounded_and_keeps_the_most_recent() {
        let tee = InputTee::new();
        let sink = TeeSink(Arc::clone(&tee));
        for i in 0..(InputTee::PREROLL_FRAMES as u32 + 10) {
            sink.push(frame(i)).unwrap();
        }

        let rec = Arc::new(Log::default());
        tee.set_main(Box::new(Recorder(Arc::clone(&rec))));

        let got = rec.got.lock();
        assert_eq!(got.len(), InputTee::PREROLL_FRAMES);
        // The oldest ten went; the newest is still there.
        assert_eq!(got[0], "push:10");
        assert_eq!(got[got.len() - 1], format!("push:{}", InputTee::PREROLL_FRAMES + 9));
    }

    #[test]
    fn stt_preroll_flush_conservation_under_audio_overflow() {
        let tee = InputTee::new();
        let sink = TeeSink(tee.clone());
        sink.push(frame(1)).unwrap();
        sink.flush().unwrap();
        for i in 0..InputTee::PREROLL_FRAMES {
            sink.push(frame(i as u32 + 2)).unwrap();
        }
        let rec = Arc::new(Log::default());
        tee.set_main(Box::new(Recorder(rec.clone())));
        let got = rec.got.lock();
        assert!(got.iter().any(|item| item == "flush"), "overflow discarded an utterance boundary");
        assert_eq!(got.len(), InputTee::PREROLL_FRAMES);
        assert_eq!(tee.dropped_preroll_frames(), 2);
    }

    /// A serving stream that pushes `rate + 1` back into the tee for each odd `rate`, as a host
    /// sink re-entering through the GIL or a concurrent caller would.
    struct Echo(Arc<Log>, std::sync::Weak<InputTee>);

    impl StreamInput<AudioFrame> for Echo {
        fn push(&self, item: AudioFrame) -> Result<(), ProviderError> {
            self.0.got.lock().push(format!("push:{}", item.sample_rate));
            match (item.sample_rate % 2, self.1.upgrade()) {
                (1, Some(tee)) => TeeSink(tee).push(frame(item.sample_rate + 1)),
                _ => Ok(()),
            }
        }
        fn flush(&self) -> Result<(), ProviderError> {
            Ok(())
        }
        fn end_input(&self) {}
    }

    /// Catches: the serving stream called under the tee's lock, so a stream that pushes again
    /// deadlocks.
    #[test]
    fn a_stream_that_pushes_again_from_its_push_does_not_deadlock() {
        let (done, finished) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let tee = InputTee::new();
            let rec = Arc::new(Log::default());
            tee.set_main(Box::new(Echo(Arc::clone(&rec), Arc::downgrade(&tee))));
            TeeSink(Arc::clone(&tee)).push(frame(1)).unwrap();
            done.send(rec.got.lock().clone()).unwrap();
        });
        let got = finished.recv_timeout(std::time::Duration::from_secs(5)).expect("deadlocked");
        assert_eq!(got, ["push:1", "push:2"]);
    }

    /// Catches: a frame pushed while the preroll is replayed lost (buffered after the drain, the
    /// stream installed over it) or delivered ahead of the replay.
    #[test]
    fn a_push_during_the_replay_arrives_after_it_in_order() {
        let tee = InputTee::new();
        let sink = TeeSink(Arc::clone(&tee));
        sink.push(frame(1)).unwrap();
        sink.push(frame(4)).unwrap();

        let rec = Arc::new(Log::default());
        tee.set_main(Box::new(Echo(Arc::clone(&rec), Arc::downgrade(&tee))));
        sink.push(frame(6)).unwrap();

        assert_eq!(*rec.got.lock(), ["push:1", "push:4", "push:2", "push:6"]);
    }

    /// Catches: buffering continuing after a stream is installed.
    #[test]
    fn nothing_is_buffered_while_a_stream_is_serving() {
        let tee = InputTee::new();
        let sink = TeeSink(Arc::clone(&tee));
        let first = Arc::new(Log::default());
        tee.set_main(Box::new(Recorder(Arc::clone(&first))));

        sink.push(frame(1)).unwrap();
        // The failover: only what arrived in the gap is replayed.
        tee.clear_main();
        sink.push(frame(2)).unwrap();

        let second = Arc::new(Log::default());
        tee.set_main(Box::new(Recorder(Arc::clone(&second))));

        assert_eq!(*first.got.lock(), ["push:1"]);
        assert_eq!(*second.got.lock(), ["push:2"]);
    }
}
