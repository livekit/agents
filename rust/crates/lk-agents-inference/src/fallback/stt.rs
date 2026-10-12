//! `stt/fallback_adapter.py`. Streaming recovery is a live tee, not a replayed request: the probe
//! hears a copy of the live audio (`stt:341-361`) and recovers on a non-empty `FINAL_TRANSCRIPT`
//! (`stt:456-466`), so [`Backend::Req`] carries an [`InputTee`]. One availability flag serves two
//! recovery slots, hence one [`FallbackChain`] with a [`RecoverySlot`] argument. Each entry runs
//! through the two views [`bound_views`] builds.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lk_agents_abi::{
    AlignedTranscript, AudioFrame, StreamSink, Stt, SttCapabilities, SttSink, SttSource,
};
use lk_agents_abi::{BoxFuture, Spawner, SpawnerExt};
use lk_agents_abi::Clock;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::tokenize::language::LanguageCode;
use lk_agents_abi::{ApiConnectOptions, SpeechEvent, SpeechEventType};
use tokio::sync::mpsc;

use crate::fallback::chain::{
    Attempt, Backend, ChainObserver, ChainOutcome, FallbackChain, NoBackends, Recovery,
    RecoverySlot,
};
use crate::fallback::channel::{ChainTask, ChannelSource, forward, probe_next, teardown_deadline};
use crate::fallback::policy::{ChainPolicy, bound_views};

// ---------------------------------------------------------------------------------------------
// The live input tee
// ---------------------------------------------------------------------------------------------

/// Fans the audio-in leg out to the serving stream and every probe. `end` is recorded so a stream
/// installed by a later failover still gets it (the reference's hole at `stt:363-365`).
pub struct InputTee {
    main: Mutex<Option<SttSink>>,
    probes: Mutex<Vec<(u64, SttSink)>>,
    next_id: AtomicU64,
    ended: AtomicBool,
    /// Replayed into the next serving stream.
    preroll: Mutex<std::collections::VecDeque<PreFrame>>,
}

/// One buffered input. `Flush` keeps its position, or two utterances would merge.
#[derive(Clone)]
enum PreFrame {
    Audio(AudioFrame),
    Flush,
}

impl Default for InputTee {
    fn default() -> Self {
        InputTee {
            main: Mutex::new(None),
            probes: Mutex::new(Vec::new()),
            next_id: AtomicU64::new(0),
            ended: AtomicBool::new(false),
            preroll: Mutex::new(std::collections::VecDeque::new()),
        }
    }
}

impl InputTee {
    pub fn new() -> Arc<Self> {
        Arc::new(InputTee::default())
    }

    /// How much audio is held while no stream is serving: 2 s of 20 ms frames, oldest dropped
    /// first.
    pub const PREROLL_FRAMES: usize = 100;

    /// Replays what arrived while no stream served (as the reference's `_input_ch` buffers), then
    /// ends it if the caller already finished.
    fn set_main(&self, sink: SttSink) {
        // Drained before install, so a concurrent `push` cannot overtake a buffered frame.
        let pending: Vec<PreFrame> =
            self.preroll.lock().expect("tee preroll").drain(..).collect();
        for item in pending {
            let _ = match item {
                PreFrame::Audio(f) => sink.push(f),
                PreFrame::Flush => sink.flush(),
            };
        }
        // Replay first, then end: a short utterance ended before the socket opened is still transcribed.
        if self.ended.load(Ordering::SeqCst) {
            sink.end_input();
            return;
        }
        *self.main.lock().expect("tee main") = Some(sink);
    }

    /// Oldest-first eviction past the cap.
    fn buffer(&self, item: PreFrame) {
        let mut q = self.preroll.lock().expect("tee preroll");
        if q.len() >= Self::PREROLL_FRAMES {
            q.pop_front();
        }
        q.push_back(item);
    }

    fn clear_main(&self) {
        *self.main.lock().expect("tee main") = None;
    }

    fn add_probe(&self, sink: SttSink) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.probes.lock().expect("tee probes").push((id, sink));
        id
    }

    fn remove_probe(&self, id: u64) {
        self.probes.lock().expect("tee probes").retain(|(i, _)| *i != id);
    }

    /// Errors are swallowed (`stt:349-350`): a dead probe must not take down the serving path.
    fn fanout(&self, f: impl Fn(&dyn StreamSink<AudioFrame>)) {
        for (_, sink) in self.probes.lock().expect("tee probes").iter() {
            f(sink.as_ref());
        }
        if let Some(sink) = self.main.lock().expect("tee main").as_ref() {
            f(sink.as_ref());
        }
    }
}

struct TeeSink(Arc<InputTee>);

impl StreamSink<AudioFrame> for TeeSink {
    fn push(&self, item: AudioFrame) -> Result<(), ProviderError> {
        // Buffered when no stream is serving; probes still get it.
        if self.0.main.lock().expect("tee main").is_none() {
            self.0.buffer(PreFrame::Audio(item.clone()));
        }
        self.0.fanout(|s| {
            let _ = s.push(item.clone());
        });
        Ok(())
    }

    fn flush(&self) -> Result<(), ProviderError> {
        if self.0.main.lock().expect("tee main").is_none() {
            self.0.buffer(PreFrame::Flush);
        }
        self.0.fanout(|s| {
            let _ = s.flush();
        });
        Ok(())
    }

    fn end_input(&self) {
        // Only the serving stream is ended; a probe closes when its drain finishes (`stt:432`).
        self.0.ended.store(true, Ordering::SeqCst);
        if let Some(sink) = self.0.main.lock().expect("tee main").as_ref() {
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
    Batch { frames: Arc<[AudioFrame]>, language: Option<LanguageCode> },
    /// `STT.stream`. `tee` is live.
    Stream { language: Option<LanguageCode>, tee: Arc<InputTee>, offset: Option<LegOffset> },
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

pub struct SttBackend {
    inner: Arc<dyn Stt>,
    probe: Arc<dyn Stt>,
    clock: Arc<dyn Clock>,
}

impl SttBackend {
    pub fn inner(&self) -> &Arc<dyn Stt> {
        &self.inner
    }

    pub fn new(stt: &Arc<dyn Stt>, policy: &ChainPolicy, clock: Arc<dyn Clock>) -> Arc<Self> {
        let (inner, probe) = bound_views(stt, policy, |conn| stt.with_conn_options(conn));
        Arc::new(SttBackend { inner, probe, clock })
    }
}

fn no_duplex(label: &str) -> ProviderError {
    ProviderError::Local(ApiError::new(format!("{label} has no streaming path")))
}

#[async_trait]
impl Backend for SttBackend {
    type Req = SttReq;
    type Session = SttSession;

    fn label(&self) -> String {
        self.inner.label()
    }

    async fn open(
        &self,
        req: &Self::Req,
        _conn: &ApiConnectOptions,
    ) -> Result<Self::Session, ProviderError> {
        let stt = &self.inner;
        match req {
            SttReq::Batch { frames, language } => {
                Ok(SttSession::Batch(stt.recognize(frames, language.as_ref()).await?))
            }
            SttReq::Stream { language, tee, offset } => {
                let offset = offset.map(|o| o.at(self.clock.now()));
                let (sink, source) = stt
                    .stream(language.as_ref(), offset)
                    .await
                    .ok_or_else(|| no_duplex(&self.inner.label()))??;
                tee.set_main(sink);
                Ok(SttSession::Stream(source))
            }
        }
    }

    async fn recover(
        &self,
        req: &Self::Req,
        _conn: &ApiConnectOptions,
        slot: RecoverySlot,
        mut stop: tokio::sync::watch::Receiver<bool>,
    ) -> Result<Recovery, ProviderError> {
        let stt = &self.probe;
        match (req, slot) {
            (SttReq::Batch { frames, language }, _) => {
                // `_try_recognize(recovering=True)`: one call, output discarded.
                stt.recognize(frames, language.as_ref()).await?;
                Ok(Recovery::Recovered)
            }
            // A probe's timestamps are never read, and the reference leaves its offset at 0.
            (SttReq::Stream { language, tee, .. }, _) => {
                let (sink, mut source) = stt
                    .stream(language.as_ref(), None)
                    .await
                    .ok_or_else(|| no_duplex(&self.inner.label()))??;
                // Read until a non-empty final transcript; a clean drain with none is `NotYet` (`stt:465-466`).
                let id = tee.add_probe(sink);
                let verdict = loop {
                    match probe_next(source.as_mut(), &mut stop).await {
                        None | Some(Ok(None)) => break Ok(Recovery::NotYet),
                        Some(Ok(Some(ev))) => {
                            if ev.kind == SpeechEventType::FinalTranscript
                                && ev.alternatives.first().is_some_and(|a| !a.text.is_empty())
                            {
                                break Ok(Recovery::Recovered);
                            }
                        }
                        Some(Err(e)) => break Err(e),
                    }
                };
                tee.remove_probe(id);
                source.close(teardown_deadline(self.clock.as_ref())).await;
                verdict
            }
        }
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

pub struct FallbackStt {
    chain: Arc<FallbackChain<SttBackend>>,
    spawner: Arc<dyn Spawner>,
    capabilities: SttCapabilities,
    caller_conn: ApiConnectOptions,
}

impl FallbackStt {
    pub fn new(
        stts: Vec<Arc<dyn Stt>>,
        policy: ChainPolicy,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        observer: Arc<dyn ChainObserver>,
    ) -> Result<Self, SttChainError> {
        let entries: Vec<Arc<SttBackend>> =
            stts.iter().map(|s| SttBackend::new(s, &policy, Arc::clone(&clock))).collect();
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
            caller_conn: crate::fallback::policy::caller_default(),
        })
    }

    pub fn chain(&self) -> &Arc<FallbackChain<SttBackend>> {
        &self.chain
    }

    pub fn active(&self) -> Arc<dyn Stt> {
        Arc::clone(&self.chain.active().inner)
    }
}

/// `stt.FallbackAdapter`'s engine as the host builds it; the public class stays Python.
#[lk_agents_abi::host(object(cell = "arc", as_dyn(Stt = as_stt)))]
impl FallbackStt {
    /// `stts` in order, each already streaming (the host wraps a recognize-only entry in its
    /// `StreamAdapter`, which composes a host VAD); the keywords are `FallbackAdapter.__init__`'s.
    /// Each entry gets the two views the chain binds it to ([`bound_views`]), built here, while
    /// the host loop is live: the chain runs on a tokio worker with no loop.
    #[host(new(
        kw_only = "attempt_timeout",
        defaults(attempt_timeout = 10.0, max_retry_per_stt = 1, retry_interval = 5.0)
    ))]
    pub fn for_host(
        stts: Vec<Arc<dyn Stt>>,
        attempt_timeout: f64,
        max_retry_per_stt: u32,
        retry_interval: f64,
        on_availability: Option<Arc<dyn ChainObserver>>,
        #[host(ambient)] clock: Arc<dyn Clock>,
        #[host(ambient)] spawner: Arc<dyn Spawner>,
    ) -> Result<Self, SttChainError> {
        let policy = ChainPolicy {
            max_retry_per_backend: max_retry_per_stt,
            attempt_timeout: Some(attempt_timeout),
            retry_interval: Some(retry_interval),
            ..ChainPolicy::stt_defaults()
        };
        let observer = on_availability.unwrap_or_else(|| Arc::new(crate::fallback::IgnoreEvents));
        FallbackStt::new(stts, policy, clock, spawner, observer)
    }

    /// The chain as the provider the session holds (a native `_core.FallbackStt` is not proxied).
    pub fn as_stt(self: &Arc<Self>) -> Arc<dyn Stt> {
        Arc::clone(self) as Arc<dyn Stt>
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

    /// The merged capabilities, not the primary's: a caller must not wait for interim results a
    /// secondary never sends.
    #[host(getter(name = "capabilities"))]
    pub fn merged_capabilities(&self) -> SttCapabilities {
        self.capabilities
    }

    /// `_recognize_impl`: the batch path, over whichever entry answers.
    #[host(method(name = "recognize"))]
    pub async fn recognize_for_host(
        &self,
        frames: Vec<AudioFrame>,
        language: Option<LanguageCode>,
    ) -> Result<SpeechEvent, ProviderError> {
        Stt::recognize(self, &frames, language.as_ref()).await
    }

    /// The duplex path: audio in (tee'd to the serving stream and every recovery probe), events out.
    #[host(method(name = "stream", stream = lk_agents_abi::FallbackSttStream))]
    pub async fn stream_for_host(
        &self,
        language: Option<LanguageCode>,
        start_time_offset: Option<f64>,
    ) -> Result<(SttSink, SttSource), ProviderError> {
        Stt::stream(self, language.as_ref(), start_time_offset)
            .await
            .expect("the chain's `stream` is always duplex")
    }
}

#[async_trait]
impl Stt for FallbackStt {
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

    fn capabilities(&self) -> SttCapabilities {
        self.capabilities
    }

    async fn recognize(
        &self,
        buffer: lk_agents_abi::AudioBuffer<'_>,
        language: Option<&LanguageCode>,
    ) -> Result<SpeechEvent, ProviderError> {
        let req = SttReq::Batch { frames: Arc::from(buffer.to_vec()), language: language.cloned() };
        let mut unused = ();
        let out = self
            .chain
            .run(
                &req,
                &self.caller_conn,
                RecoverySlot::OneShot,
                &mut unused,
                |_i, _b, session, _s| {
                    Box::pin(async move {
                        match session {
                            SttSession::Batch(ev) => Attempt::Served(ev),
                            SttSession::Stream(_) => unreachable!("a Batch request opens Batch"),
                        }
                    }) as BoxFuture<'_, Attempt<SpeechEvent>>
                },
            )
            .await?;
        match out {
            ChainOutcome::Served(ev) => Ok(ev),
            ChainOutcome::Truncated => unreachable!("`AfterOutput::Stop` is not a policy STT uses"),
        }
    }

    /// Each leg's `start_time_offset` is `start_time_offset` plus the time since this call.
    async fn stream(
        &self,
        language: Option<&LanguageCode>,
        start_time_offset: Option<f64>,
    ) -> Option<Result<(SttSink, SttSource), ProviderError>> {
        let tee = InputTee::new();
        let opened = self.chain.clock().now();
        let offset = start_time_offset.map(|start_time_offset| LegOffset { start_time_offset, opened });
        let (tx, rx) = mpsc::channel::<Result<SpeechEvent, ProviderError>>(32);

        let chain = Arc::clone(&self.chain);
        let drain_chain = Arc::clone(&self.chain);
        let req = SttReq::Stream { language: language.cloned(), tee: Arc::clone(&tee), offset };
        let caller_conn = self.caller_conn;
        let tee_for_task = Arc::clone(&tee);

        let (stop_rx, done_tx, chain_task) = ChainTask::pair(chain.clock());
        let handle = self.spawner.spawn_unboxed(async move {
            let mut tx = tx;
            let result = chain
                .run(
                    &req,
                    &caller_conn,
                    RecoverySlot::Streaming,
                    &mut tx,
                    move |index, _b, session, tx| {
                        let chain = Arc::clone(&drain_chain);
                        let tee = Arc::clone(&tee_for_task);
                        let mut stop = stop_rx.clone();
                        Box::pin(async move {
                            let SttSession::Stream(mut source) = session else {
                                unreachable!("a Stream request opens Stream")
                            };
                            let outcome =
                                forward(&chain, index, source.as_mut(), &mut stop, tx, |_| true).await;
                            // The dead stream leaves the tee before the next entry installs its own.
                            tee.clear_main();
                            source.close(chain.teardown_deadline()).await;
                            outcome
                        }) as BoxFuture<'_, Attempt<()>>
                    },
                )
                .await;
            if let Err(e) = result {
                let _ = tx.send(Err(e)).await;
            }
            let _ = done_tx.send(());
        });

        Some(Ok((
            Box::new(TeeSink(tee)) as SttSink,
            Box::new(ChannelSource { rx, task: Some(handle), chain: chain_task }) as SttSource,
        )))
    }
}

#[cfg(test)]
mod preroll_tests {
    use super::*;

    /// A sink that records what it was handed, in order.
    #[derive(Default)]
    struct Log {
        got: Mutex<Vec<String>>,
        ended: AtomicBool,
    }

    /// The sink half, over a shared [`Log`] the test can read after installing it.
    struct Recorder(Arc<Log>);

    impl StreamSink<AudioFrame> for Recorder {
        fn push(&self, item: AudioFrame) -> Result<(), ProviderError> {
            self.0.got.lock().unwrap().push(format!("push:{}", item.sample_rate));
            Ok(())
        }
        fn flush(&self) -> Result<(), ProviderError> {
            self.0.got.lock().unwrap().push("flush".to_owned());
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

        assert_eq!(*rec.got.lock().unwrap(), ["push:1", "flush", "push:2"]);
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

        assert_eq!(*rec.got.lock().unwrap(), ["push:1", "push:2"]);
        assert!(rec.ended.load(Ordering::SeqCst), "the stream was never told the input had ended");
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

        let got = rec.got.lock().unwrap();
        assert_eq!(got.len(), InputTee::PREROLL_FRAMES);
        // The oldest ten went; the newest is still there.
        assert_eq!(got[0], "push:10");
        assert_eq!(got[got.len() - 1], format!("push:{}", InputTee::PREROLL_FRAMES + 9));
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

        assert_eq!(*first.got.lock().unwrap(), ["push:1"]);
        assert_eq!(*second.got.lock().unwrap(), ["push:2"]);
    }
}
