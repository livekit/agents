//! [`StreamAdapter`] (`tts/stream_adapter.py`): a non-streaming TTS as a duplex one. Frames pass
//! through; `is_final` needs a one-frame hold, taken only once no further sentence can arrive.
//! Its sentences ([`Sentences`]) are split natively or by a host tokenizer's stream, and paced
//! by the audio synthesized when asked.

use super::pacing::{AdapterInput, HostSentences, Sentences, Split, trim_python_whitespace};
use super::stream::TtsStream;
use crate::telemetry;
use lk_agents_abi::ApiConnectOptions;
use lk_agents_abi::TranscriptChunk;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::metrics::base::AgentMetrics;
use lk_agents_abi::tts::stream_adapter::StreamAdapterFunctions;
use lk_agents_abi::tts::stream_adapter::{
    SentenceTextProcessor, SplitterFactory, TextProcessorFactory,
};
use lk_agents_abi::tts::stream_pacer::HostSentenceStream;
use lk_agents_abi::tts::types::{SynthesizedAudio, Tts, TtsCapabilities, TtsSink, TtsSource};
use lk_agents_abi::types::identity::ProviderId;
use lk_agents_abi::utils::clock::{Clock, Ts};
use lk_agents_abi::utils::stream::{Closed, HOST_TEARDOWN_TIMEOUT, StreamInput, StreamOutput};
use lk_agents_core::metrics::provider::TtsMetricState;
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::strings::shortuuid;
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::mpsc;

// ---------------------------------------------------------------------------- StreamAdapter

pub struct StreamAdapter {
    tts: Arc<dyn Tts>,
    splitter: SplitterFactory,
    text_processor: TextProcessorFactory,
    /// Bounds the per-sentence inner closes done inside `next()`.
    clock: Arc<dyn Clock>,
    /// Each sentence's retries (`ChunkedStream`'s own loop in the reference): a sentence failing
    /// after earlier ones were heard is synthesized again, not the whole reply lost.
    budget: lk_agents_abi::ApiConnectOptions,
    /// `(min_remaining_audio, max_text_length)`: sentences released as the audio plays out.
    pacing: Option<(f64, usize)>,
}

impl StreamAdapter {
    /// No sentence is retried; see [`with_budget`](Self::with_budget).
    pub fn new(
        tts: Arc<dyn Tts>,
        splitter: SplitterFactory,
        text_processor: TextProcessorFactory,
        clock: Arc<dyn Clock>,
    ) -> Self {
        let budget = lk_agents_abi::ApiConnectOptions {
            max_retry: 0,
            ..lk_agents_abi::ApiConnectOptions::default()
        };
        StreamAdapter { tts, splitter, text_processor, clock, budget, pacing: None }
    }

    /// Each sentence retried within `budget`.
    pub fn with_budget(mut self, budget: lk_agents_abi::ApiConnectOptions) -> Self {
        self.budget = budget;
        self
    }

    /// Its sentences paced by `(min_remaining_audio, max_text_length)`.
    pub fn with_pacing(mut self, pacing: Option<(f64, usize)>) -> Self {
        self.pacing = pacing;
        self
    }

    /// Itself when it has a duplex path, else adapted, each sentence retried within `budget`;
    /// decided per instance by its capabilities.
    pub fn adapted(
        tts: Arc<dyn Tts>,
        splitter: SplitterFactory,
        text_processor: TextProcessorFactory,
        clock: Arc<dyn Clock>,
        budget: lk_agents_abi::ApiConnectOptions,
    ) -> Arc<dyn Tts> {
        if tts.capabilities().streaming {
            return tts;
        }
        Arc::new(StreamAdapter::new(tts, splitter, text_processor, clock).with_budget(budget))
    }

    /// [`adapted`](Self::adapted) with the default sentence splitting and provider markup: a
    /// voice session's TTS leg.
    pub fn adapted_for(
        tts: Arc<dyn Tts>,
        clock: Arc<dyn Clock>,
        budget: lk_agents_abi::ApiConnectOptions,
    ) -> Arc<dyn Tts> {
        let splitter: SplitterFactory =
            Arc::new(|expressive| Box::new(super::sentences::sentence_splitter(None, expressive)));
        Self::adapted(
            tts,
            splitter,
            Arc::new(lk_agents_core::tts::markup::text_processor),
            clock,
            budget,
        )
    }
}

impl Tts for StreamAdapter {
    fn label(&self) -> String {
        format!("StreamAdapter[{}]", self.tts.label())
    }

    fn request_span(&self) -> telemetry::RequestSpan {
        telemetry::RequestSpan::StreamAdapter
    }

    /// The adapter is transparent to the metrics.
    fn id(&self) -> ProviderId {
        self.tts.id()
    }

    fn prewarm(&self) {
        self.tts.prewarm();
    }

    fn failed(&self, error: ProviderError, recoverable: bool) {
        self.tts.failed(error, recoverable);
    }

    /// Verbatim from `stream_adapter.py:33`.
    fn capabilities(&self) -> TtsCapabilities {
        TtsCapabilities { streaming: true, aligned_transcript: true }
    }

    fn sample_rate(&self) -> u32 {
        self.tts.sample_rate()
    }

    fn num_channels(&self) -> u16 {
        self.tts.num_channels()
    }

    /// A pass-through speaks whatever dialect it wraps.
    fn markup_provider(&self) -> String {
        self.tts.markup_provider()
    }

    /// The framework's input path for every non-streaming TTS: it lowers the markup.
    fn lowers_markup(&self) -> bool {
        true
    }

    /// `stream_adapter.py:58`.
    fn synthesize<'a>(
        &'a self,
        cx: &telemetry::Cx,
        text: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, Result<TtsSource, ProviderError>> {
        let cx = cx.clone();
        Box::pin(async move { self.tts.synthesize(&cx, text).await })
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
        Box::pin(async move { self.open(&cx, expressive, None) })
    }
}

impl StreamAdapter {
    /// One stream, split natively, or by `sentences`, a host tokenizer's stream.
    pub(crate) fn open(
        &self,
        cx: &telemetry::Cx,
        expressive: bool,
        sentences: Option<Arc<dyn HostSentenceStream>>,
    ) -> Result<lk_agents_abi::DuplexStream<TtsSink, TtsSource>, ProviderError> {
        self.budget
            .validate()
            .map_err(|error| ApiError::new(error.to_string()).with_retryable(false))?;
        // Unbounded on purpose: text -> TTS may not drop, a lost token mangles speech.
        let (tx, rx) = mpsc::unbounded_channel();
        let processor =
            || expressive.then(|| (self.text_processor)(&self.tts.markup_provider())).flatten();
        let lowering = processor();
        let split = match sentences {
            Some(stream) => Split::Host(HostSentences::read(stream, cx)),
            None => Split::Native((self.splitter)(lowering.is_some())),
        };
        let sentences =
            Sentences::new(rx, split, processor(), self.pacing, Arc::clone(&self.clock));
        let sink = AdapterSink { tx: Mutex::new(Some(tx)), closed: AtomicBool::new(false) };
        let source = AdapterSource {
            tts: self.tts.clone(),
            cx: cx.clone(),
            budget: self.budget,
            sentence: None,
            sentence_audio: false,
            report: None,
            clock: Arc::clone(&self.clock),
            sentences,
            lowering,
            current: None,
            inner_closed: Closed::Clean,
            duration: 0.0,
            pending_transcript: Vec::new(),
            held: None,
            ledger: Arc::new(Mutex::new(Ledger {
                clock: Arc::clone(&self.clock),
                call: None,
                calls: Vec::new(),
            })),
            request_id: shortuuid(""),
            segment_id: shortuuid(""),
        };
        Ok(lk_agents_abi::DuplexStream::new(
            Box::new(sink) as TtsSink,
            Box::new(source) as TtsSource,
        ))
    }
}

struct AdapterSink {
    tx: Mutex<Option<mpsc::UnboundedSender<AdapterInput>>>,
    closed: AtomicBool,
}

impl AdapterSink {
    fn send(&self, item: AdapterInput) -> Result<(), ProviderError> {
        let guard = self.tx.lock();
        match guard.as_ref() {
            // The source was dropped without `aclose`. Nothing will ever read this.
            None => Ok(()),
            Some(tx) => tx
                .send(item)
                .map_err(|_| ApiError::new("stream adapter: the source half is gone").into()),
        }
    }
}

impl StreamInput<String> for AdapterSink {
    fn push(&self, item: String) -> Result<(), ProviderError> {
        // `tts/tts.py:744`: an empty token or a closed input is a silent no-op.
        if item.is_empty() || self.closed.load(Ordering::Acquire) {
            return Ok(());
        }
        self.send(AdapterInput::Text(item))
    }

    fn flush(&self) -> Result<(), ProviderError> {
        // `tts/tts.py:771`: silent once the input channel is closed.
        if self.closed.load(Ordering::Acquire) {
            return Ok(());
        }
        self.send(AdapterInput::Flush)
    }

    fn end_input(&self) {
        // `tts/tts.py:785`: `flush()` then `close()`, in that order.
        let _ = self.flush();
        self.closed.store(true, Ordering::Release);
        drop(self.tx.lock().take());
    }
}

struct AdapterSource {
    tts: Arc<dyn Tts>,
    /// What each sentence's synthesis runs under: the stream's caller.
    cx: telemetry::Cx,
    budget: lk_agents_abi::ApiConnectOptions,
    /// The sentence being synthesized and its failed attempts: what a retry synthesizes again.
    sentence: Option<(String, u32)>,
    /// The current sentence delivered audio, so a failure can no longer be retried (F-10).
    sentence_audio: bool,
    /// Who hears of each retried sentence (the engine's error event and journal).
    report: Option<lk_agents_abi::utils::stream::RetryReport>,
    clock: Arc<dyn Clock>,
    sentences: Sentences,
    /// The markup lowered before synthesis; `None` when expressive is off or there is no dialect.
    lowering: Option<Box<dyn SentenceTextProcessor>>,
    current: Option<TtsSource>,
    /// Folded into `close`.
    inner_closed: Closed,
    /// The `start_time` stamped on each sentence's `TranscriptChunk`.
    duration: f64,
    /// Ride on the next frame sent (`tts/tts.py:1104`).
    pending_transcript: Vec<TranscriptChunk>,
    held: Option<SynthesizedAudio>,
    /// Each sentence metered as its own call; shared, so a node's reader can end the one in
    /// flight while the node's pump waits on it.
    ledger: Arc<Mutex<Ledger>>,
    /// One request and one segment for the whole stream (`stream_adapter.py:96`).
    request_id: String,
    segment_id: String,
}

/// One sentence's synthesis: its text, when it was asked for, and its fold.
struct Call {
    text: String,
    start: f64,
    fold: TtsMetricState,
}

/// The sentence being synthesized, metered as its own call (`ChunkedStream`'s
/// `_metrics_monitor_task`): the reference's adapter forwards each one's `TTSMetrics`.
struct Ledger {
    clock: Arc<dyn Clock>,
    call: Option<Call>,
    /// The calls finished and not yet read.
    calls: Vec<AgentMetrics>,
}

impl Ledger {
    /// The sentence's record; none for a failed one (`_current_attempt_has_error`).
    fn end_call(&mut self, cancelled: bool) {
        let Some(mut call) = self.call.take() else { return };
        let now = lk_agents_abi::unix_secs(self.clock.wall());
        let record = call.fold.finish(now, call.start, &call.text, now, cancelled);
        self.calls.extend(record.map(AgentMetrics::Tts));
    }

    fn take(&mut self, in_flight: bool) -> Vec<AgentMetrics> {
        if in_flight {
            self.end_call(true);
        }
        std::mem::take(&mut self.calls)
    }
}

impl AdapterSource {
    fn ledger(&self) -> parking_lot::MutexGuard<'_, Ledger> {
        self.ledger.lock()
    }

    /// Bounded on the session clock; a timeout is remembered for `close`.
    async fn close_current(&mut self) {
        if let Some(done) = self.current.take() {
            let deadline = self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
            self.inner_closed = self.inner_closed.and(done.close(deadline).await);
        }
    }

    async fn raw_next(&mut self) -> Result<Option<SynthesizedAudio>, ProviderError> {
        loop {
            if self.current.is_some() {
                if let Some(audio) = self.read_sentence_audio().await? {
                    return Ok(Some(audio));
                }
                continue;
            }
            let Some(sentence) = self.sentences.next(self.duration).await? else { return Ok(None) };
            // The transcript uses the unstripped token and precedes the empty check
            // (`stream_adapter.py:125`).
            self.pending_transcript.push(TranscriptChunk {
                text: sentence.clone(),
                start_time: Some(self.duration),
                ..TranscriptChunk::default()
            });
            let text = self.speakable(&sentence);
            if text.is_empty() {
                continue;
            }
            self.sentence = Some((text, 0));
            self.sentence_audio = false;
            self.open_sentence().await?;
        }
    }

    /// The current sentence's next audio; `None` once the sentence ended or was retried.
    async fn read_sentence_audio(&mut self) -> Result<Option<SynthesizedAudio>, ProviderError> {
        let Some(current) = self.current.as_mut() else { return Ok(None) };
        match current.next().await {
            Ok(Some(audio)) => {
                self.sentence_audio = true;
                let clock = Arc::clone(&self.clock);
                if let Some(call) = self.ledger().call.as_mut() {
                    let start = call.start;
                    call.fold.observe(&audio, || lk_agents_abi::unix_secs(clock.wall()) - start);
                }
                Ok(Some(audio))
            }
            Ok(None) => {
                self.ledger().end_call(false);
                // Dropping a host stream does not cancel its task; close it, bounded.
                self.close_current().await;
                Ok(None)
            }
            // Close a failed inner source too, or it counts as open after the retry decision.
            Err(error) => {
                self.ledger().call = None;
                self.close_current().await;
                self.retry_sentence(error).await?;
                Ok(None)
            }
        }
    }

    /// The sentence as spoken: whitespace-only is falsy in Python, and several plugins raise on
    /// empty input. Markup is re-normalized: a marker split across two input chunks escaped the
    /// push's pass.
    fn speakable(&self, sentence: &str) -> String {
        let text = trim_python_whitespace(sentence);
        let Some(markup) = &self.lowering else { return text.to_owned() };
        let lowered = markup.convert(&markup.normalize(text));
        trim_python_whitespace(&lowered).to_owned()
    }

    /// The current sentence's synthesis, retried while its open fails within the budget.
    async fn open_sentence(&mut self) -> Result<(), ProviderError> {
        loop {
            let Some((text, _)) = self.sentence.clone() else { return Ok(()) };
            let start = lk_agents_abi::unix_secs(self.clock.wall());
            match self.tts.synthesize(&self.cx, &text).await {
                Ok(source) => {
                    self.current = Some(source);
                    let id = self.tts.id();
                    let fold = TtsMetricState::new(false, id.label.clone(), Some(id.metadata()));
                    self.ledger().call = Some(Call { text, start, fold });
                    return Ok(());
                }
                Err(e) => self.retry_sentence(e).await?,
            }
        }
    }

    /// `Ok` once the wait before the sentence's next attempt is over, and that attempt opened; the
    /// failure, final, when the sentence's budget is spent or it may not be retried.
    async fn retry_sentence(&mut self, error: ProviderError) -> Result<(), ProviderError> {
        let Some((_, retries)) = self.sentence.as_mut() else { return Err(error) };
        let eligible = error.api_kind().is_some()
            && error.retryable()
            && error.status_code() != Some(499)
            && !self.sentence_audio;
        let policy = lk_agents_core::attempts::RequestPolicy::from_options(&self.budget);
        let Some(wait) = policy.retry_delay(*retries, eligible) else {
            return Err(error.into_final());
        };
        let delay = wait.as_secs_f64();
        *retries += 1;
        log::warn!(
            target: "livekit.agents",
            "{}: failed to synthesize speech: {error}, retrying in {delay}s (attempt {retries})",
            self.tts.label()
        );
        if let Some(report) = &self.report {
            report(error.clone(), *retries, wait);
        }
        self.tts.failed(error, true);
        let at = self.clock.now().saturating_add(wait);
        self.clock.wait_until(at).await;
        Box::pin(self.open_sentence()).await
    }

    /// The inner frame as this stream's: its duration counted, then input drained (and paced
    /// against it), its ids ours, and the pending transcript on it. A sentence boundary is not a
    /// segment boundary (`tts/tts.py:1183`): the inner provider's ids and `is_final` are not ours.
    fn stamp(&mut self, mut frame: SynthesizedAudio) -> SynthesizedAudio {
        self.duration += frame.frame.duration();
        self.sentences.drain(self.duration);
        frame.is_final = false;
        frame.request_id.clone_from(&self.request_id);
        frame.segment_id.clone_from(&self.segment_id);
        if !self.pending_transcript.is_empty() {
            frame.timed_transcript = std::mem::take(&mut self.pending_transcript);
        }
        frame
    }

    /// The frame to hand out now: the inner frame, stamped, while more audio can come. Once no
    /// further audio can arrive the last frame is held to be marked final, and the one held
    /// before it goes out.
    fn pass_or_hold(&mut self, frame: SynthesizedAudio) -> Option<SynthesizedAudio> {
        let frame = self.stamp(frame);
        if self.more_audio_coming() {
            return Some(frame);
        }
        self.held.replace(frame)
    }

    /// A queued whitespace-only sentence is not audio. Once false it stays false.
    fn more_audio_coming(&self) -> bool {
        self.sentences.more_coming()
    }

    /// The held last frame, final. Alignment for trailing sentences that produced no audio rides
    /// on it (no silence marker, which is `AudioEmitter`'s job).
    fn finish_held(&mut self) -> Option<SynthesizedAudio> {
        let mut last = self.held.take()?;
        last.is_final = true;
        last.timed_transcript.append(&mut self.pending_transcript);
        Some(last)
    }
}

impl StreamOutput<SynthesizedAudio> for AdapterSource {
    fn next(
        &mut self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<SynthesizedAudio>, ProviderError>> {
        Box::pin(async move {
            loop {
                match self.raw_next().await {
                    Ok(Some(frame)) => match self.pass_or_hold(frame) {
                        Some(frame) => return Ok(Some(frame)),
                        None => continue,
                    },
                    Ok(None) => return Ok(self.finish_held()),
                    Err(e) => {
                        // Drop the hold and report now: `FallbackChain` needs the error to decide
                        // `retry_on_chunk_sent`.
                        self.held = None;
                        return Err(e);
                    }
                }
            }
        })
    }

    /// The worst inner close: `TimedOut` if any did not ack, else `Failed` if any teardown failed.
    fn close(mut self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move {
            let current = match self.current.take() {
                Some(cur) => cur.close(deadline).await,
                None => Closed::Clean,
            };
            self.sentences.close(&self.cx, deadline).await;
            self.inner_closed.and(current)
        })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<SynthesizedAudio> for AdapterSource {
    fn take_call_metrics(&mut self, in_flight: bool) -> Option<Vec<AgentMetrics>> {
        Some(self.ledger().take(in_flight))
    }
    fn report_retries(&mut self, report: lk_agents_abi::utils::stream::RetryReport) {
        self.report = Some(report);
    }
    fn call_ledger(&self) -> Option<lk_agents_abi::utils::stream::CallLedger> {
        let ledger = Arc::clone(&self.ledger);
        Some(Arc::new(move |in_flight| ledger.lock().take(in_flight)))
    }
}

pub struct AdapterStreams;

impl StreamAdapterFunctions for AdapterStreams {
    type Stream = TtsStream;
    type Env = ProcessEnv;

    async fn tts_adapter_stream(
        env: &ProcessEnv,
        cx: telemetry::Cx,
        provider: Arc<dyn Tts>,
        conn_options: ApiConnectOptions,
        expressive: bool,
        sentences: Option<Arc<dyn HostSentenceStream>>,
        pacing: Option<(f64, usize)>,
    ) -> Result<Self::Stream, ProviderError> {
        conn_options.validate().map_err(|error| {
            ProviderError::Local(ApiError { retryable: false, ..ApiError::new(error.to_string()) })
        })?;
        let provider = provider.with_conn_options(&conn_options).unwrap_or(provider);
        let splitter: SplitterFactory =
            Arc::new(|xml_aware| Box::new(super::sentences::sentence_splitter(None, xml_aware)));
        let (sink, source) = StreamAdapter::new(
            provider,
            splitter,
            Arc::new(lk_agents_core::tts::markup::text_processor),
            env.clock.clone(),
        )
        .with_budget(conn_options)
        .with_pacing(pacing)
        .open(&cx, expressive, sentences)?
        .split();
        Ok(TtsStream::new(Some(sink), source, env.clock.clone()))
    }
}
