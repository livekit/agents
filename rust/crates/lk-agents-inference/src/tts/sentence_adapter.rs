//! [`StreamAdapter`] (`tts/stream_adapter.py`): a non-streaming TTS as a duplex one. Frames pass
//! through; `is_final` needs a one-frame hold, taken only once no further sentence can arrive.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lk_agents_abi::TimedString;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::metrics::base::AgentMetrics;
use lk_agents_abi::telemetry::provider::{Context, TtsMetricState};
use lk_agents_abi::tts::stream_adapter::{
    SentenceSplitter, SentenceTextProcessor, SplitterFactory, TextProcessorFactory,
};
use lk_agents_abi::tts::text::TtsTextProcessorFactory;
use lk_agents_abi::tts::tts::{SynthesizedAudio, Tts, TtsCapabilities, TtsSink, TtsSource};
use lk_agents_abi::types::identity::ProviderId;
use lk_agents_abi::utils::clock::{Clock, Ts};
use lk_agents_abi::utils::stream::{Closed, HOST_TEARDOWN_TIMEOUT, StreamSink, StreamSource};
use lk_agents_net::strings::shortuuid;
use tokio::sync::mpsc;

// ---------------------------------------------------------------------------- StreamAdapter

pub struct StreamAdapter {
    tts: Arc<dyn Tts>,
    splitter: SplitterFactory,
    text_processor: TextProcessorFactory,
    /// Bounds the per-sentence inner closes done inside `next()`.
    clock: Arc<dyn Clock>,
    /// `_expressive`, set by the framework before each synthesis and snapshotted at `stream()`.
    expressive: AtomicBool,
    /// Each sentence's retries (`ChunkedStream`'s own loop in the reference): a sentence failing
    /// after earlier ones were heard is synthesized again, not the whole reply lost.
    budget: lk_agents_abi::ApiConnectOptions,
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
        StreamAdapter {
            tts,
            splitter,
            text_processor,
            clock,
            expressive: AtomicBool::new(false),
            budget,
        }
    }

    /// Each sentence retried within `budget`.
    pub fn with_budget(mut self, budget: lk_agents_abi::ApiConnectOptions) -> Self {
        self.budget = budget;
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

    /// [`adapted`](Self::adapted) with `E`'s sentence splitting and provider markup: a voice
    /// session's TTS leg.
    pub fn adapted_for<E: TtsTextProcessorFactory>(
        tts: Arc<dyn Tts>,
        clock: Arc<dyn Clock>,
        budget: lk_agents_abi::ApiConnectOptions,
    ) -> Arc<dyn Tts> {
        let splitter: SplitterFactory =
            Arc::new(|expressive| E::sentence_splitter(None, expressive));
        Self::adapted(tts, splitter, Arc::new(E::markup), clock, budget)
    }
}

#[async_trait]
impl Tts for StreamAdapter {
    fn label(&self) -> String {
        format!("StreamAdapter[{}]", self.tts.label())
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

    fn set_expressive(&self, on: bool) {
        self.expressive.store(on, Ordering::Release);
        self.tts.set_expressive(on);
    }

    /// `stream_adapter.py:58`.
    async fn synthesize(&self, text: &str) -> Result<TtsSource, ProviderError> {
        self.tts.synthesize(text).await
    }
    async fn stream(&self) -> Result<(TtsSink, TtsSource), ProviderError> {
        // Unbounded on purpose: text -> TTS may not drop, a lost token mangles speech.
        let (tx, rx) = mpsc::unbounded_channel();
        // Snapshotted now, while the framework holds it fixed for this synthesis.
        let lowering = self
            .expressive
            .load(Ordering::Acquire)
            .then(|| (self.text_processor)(&self.tts.markup_provider()))
            .flatten();
        let sink = AdapterSink { tx: Mutex::new(Some(tx)), closed: AtomicBool::new(false) };
        let source = AdapterSource {
            tts: self.tts.clone(),
            budget: self.budget,
            sentence: None,
            sentence_audio: false,
            report: None,
            clock: Arc::clone(&self.clock),
            rx,
            splitter: (self.splitter)(lowering.is_some()),
            lowering,
            pending: VecDeque::new(),
            current: None,
            inner_closed: Closed::Clean,
            input_done: false,
            duration: 0.0,
            pending_transcript: Vec::new(),
            held: None,
            ledger: Arc::new(Mutex::new(Ledger {
                tts: self.tts.clone(),
                clock: Arc::clone(&self.clock),
                call: None,
                calls: Vec::new(),
            })),
            request_id: shortuuid(""),
            segment_id: shortuuid(""),
        };
        Ok((Box::new(sink) as TtsSink, Box::new(source) as TtsSource))
    }
}

enum AdapterInput {
    Text(String),
    Flush,
}

struct AdapterSink {
    tx: Mutex<Option<mpsc::UnboundedSender<AdapterInput>>>,
    closed: AtomicBool,
}

impl AdapterSink {
    fn send(&self, item: AdapterInput) -> Result<(), ProviderError> {
        let guard = self.tx.lock().expect("adapter sink poisoned");
        match guard.as_ref() {
            // The source was dropped without `aclose`. Nothing will ever read this.
            None => Ok(()),
            Some(tx) => tx
                .send(item)
                .map_err(|_| ApiError::new("stream adapter: the source half is gone").into()),
        }
    }
}

impl StreamSink<String> for AdapterSink {
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
        drop(self.tx.lock().expect("adapter sink poisoned").take());
    }
}

struct AdapterSource {
    tts: Arc<dyn Tts>,
    budget: lk_agents_abi::ApiConnectOptions,
    /// The sentence being synthesized and its failed attempts: what a retry synthesizes again.
    sentence: Option<(String, u32)>,
    /// The current sentence delivered audio, so a failure can no longer be retried (F-10).
    sentence_audio: bool,
    /// Who hears of each retried sentence (the engine's error event and journal).
    report: Option<lk_agents_abi::utils::stream::RetryReport>,
    clock: Arc<dyn Clock>,
    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
    rx: mpsc::UnboundedReceiver<AdapterInput>,
    splitter: Box<dyn SentenceSplitter>,
    /// The markup lowered before synthesis; `None` when expressive is off or there is no dialect.
    lowering: Option<Box<dyn SentenceTextProcessor>>,
    pending: VecDeque<String>,
    current: Option<TtsSource>,
    /// Folded into `close`.
    inner_closed: Closed,
    input_done: bool,
    /// The `start_time` stamped on each sentence's `TimedString`.
    duration: f64,
    /// Ride on the next frame sent (`tts/tts.py:1104`).
    pending_transcript: Vec<TimedString>,
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
    tts: Arc<dyn Tts>,
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
        let id = self.tts.id();
        let label = id.label.clone();
        let context = Context { timestamp: now, label, cancelled, metadata: Some(id.metadata()) };
        let record = call.fold.finish(now, call.start, &call.text, 0, 0, 0.0, false, context);
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
    fn ledger(&self) -> std::sync::MutexGuard<'_, Ledger> {
        self.ledger.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Bounded on the session clock; a timeout is remembered for `close`.
    async fn close_current(&mut self) {
        if let Some(done) = self.current.take() {
            let deadline = self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
            self.inner_closed = self.inner_closed.and(done.close(deadline).await);
        }
    }

    fn push_text(&mut self, text: &str) {
        let sentences = match &self.lowering {
            Some(markup) => self.splitter.push(&markup.normalize(text)),
            None => self.splitter.push(text),
        };
        self.pending.extend(sentences);
    }

    /// Without awaiting. Stands in for the reference's concurrent `_forward_input`: otherwise the
    /// adapter would learn input ended only after the last frame, and never mark it final.
    fn drain_input(&mut self) {
        loop {
            match self.rx.try_recv() {
                Ok(AdapterInput::Text(t)) => self.push_text(&t),
                Ok(AdapterInput::Flush) => self.pending.extend(self.splitter.flush()),
                Err(mpsc::error::TryRecvError::Empty) => return,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    if !self.input_done {
                        self.input_done = true;
                        self.pending.extend(self.splitter.flush());
                    }
                    return;
                }
            }
        }
    }

    async fn next_sentence(&mut self) -> Option<String> {
        loop {
            self.drain_input();
            if let Some(s) = self.pending.pop_front() {
                return Some(s);
            }
            if self.input_done {
                return None;
            }
            match self.rx.recv().await {
                Some(AdapterInput::Text(t)) => self.push_text(&t),
                Some(AdapterInput::Flush) => self.pending.extend(self.splitter.flush()),
                None => {
                    // Sender dropped. A sink dropped without `end_input` never flushed, so flush
                    // again rather than lose the last half-sentence.
                    self.input_done = true;
                    self.pending.extend(self.splitter.flush());
                }
            }
        }
    }

    async fn raw_next(&mut self) -> Result<Option<SynthesizedAudio>, ProviderError> {
        loop {
            if let Some(cur) = self.current.as_mut() {
                match cur.next().await {
                    Ok(Some(a)) => {
                        self.sentence_audio = true;
                        let clock = Arc::clone(&self.clock);
                        if let Some(call) = self.ledger().call.as_mut() {
                            let start = call.start;
                            call.fold
                                .observe(&a, || lk_agents_abi::unix_secs(clock.wall()) - start);
                        }
                        return Ok(Some(a));
                    }
                    Ok(None) => {
                        self.ledger().end_call(false);
                        // Dropping a host stream does not cancel its task; close it, bounded.
                        self.close_current().await;
                        continue;
                    }
                    // Close a failed inner source too, or it counts as open after the retry
                    // decision.
                    Err(e) => {
                        self.ledger().call = None;
                        self.close_current().await;
                        self.retry_sentence(e).await?;
                        continue;
                    }
                }
            }
            let Some(sentence) = self.next_sentence().await else { return Ok(None) };

            // The transcript uses the unstripped token and precedes the empty check
            // (`stream_adapter.py:125`).
            self.pending_transcript.push(TimedString {
                text: sentence.clone(),
                start_time: Some(self.duration),
                ..TimedString::default()
            });
            // Whitespace-only is falsy in Python; several plugins raise on empty input.
            let mut text = sentence
                .trim_matches(|c| lk_agents_abi::utils::strings::PYTHON_WHITESPACE.contains(&c))
                .to_owned();
            if let Some(markup) = &self.lowering {
                // Re-normalized: a marker split across two input chunks escaped the push's pass.
                text = markup
                    .convert(&markup.normalize(&text))
                    .trim_matches(|c| lk_agents_abi::utils::strings::PYTHON_WHITESPACE.contains(&c))
                    .to_owned();
            }
            if text.is_empty() {
                continue;
            }
            self.sentence = Some((text, 0));
            self.sentence_audio = false;
            self.open_sentence().await?;
        }
    }

    /// The current sentence's synthesis, retried while its open fails within the budget.
    async fn open_sentence(&mut self) -> Result<(), ProviderError> {
        loop {
            let Some((text, _)) = self.sentence.clone() else { return Ok(()) };
            let start = lk_agents_abi::unix_secs(self.clock.wall());
            match self.tts.synthesize(&text).await {
                Ok(source) => {
                    self.current = Some(source);
                    let fold = TtsMetricState::for_host(false);
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
        use lk_agents_abi::types::retry::{RequestKind, RetryDecision, decide};
        let Some((_, retries)) = self.sentence.as_mut() else { return Err(error) };
        let decision = match error.api_kind() {
            None => RetryDecision::Raise,
            Some(_) => decide(
                RequestKind::TtsBatch,
                &self.budget,
                *retries,
                error.retryable(),
                error.status_code(),
                self.sentence_audio,
            ),
        };
        let RetryDecision::Retry(delay) = decision else { return Err(error.into_final()) };
        *retries += 1;
        log::warn!(
            target: "livekit.agents",
            "{}: failed to synthesize speech: {error}, retrying in {delay}s (attempt {retries})",
            self.tts.label()
        );
        let wait = std::time::Duration::from_secs_f64(delay.max(0.0));
        if let Some(report) = &self.report {
            report(error.clone(), *retries, wait);
        }
        self.tts.failed(error, true);
        let at = self.clock.now().saturating_add(wait);
        self.clock.wait_until(at).await;
        Box::pin(self.open_sentence()).await
    }
}

impl StreamSource<SynthesizedAudio> for AdapterSource {
    fn next(
        &mut self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<SynthesizedAudio>, ProviderError>> {
        Box::pin(async move {
            loop {
                match self.raw_next().await {
                    Ok(Some(mut frame)) => {
                        self.drain_input();
                        self.duration += frame.frame.duration();
                        // A sentence boundary is not a segment boundary (`tts/tts.py:1183`): the inner
                        // provider's ids and `is_final` are not ours.
                        frame.is_final = false;
                        frame.request_id.clone_from(&self.request_id);
                        frame.segment_id.clone_from(&self.segment_id);
                        if !self.pending_transcript.is_empty() {
                            frame.timed_transcript = std::mem::take(&mut self.pending_transcript);
                        }
                        // Hold only once no further audio can arrive; a queued whitespace-only sentence
                        // is not audio. Once true it stays true.
                        let more_audio_coming = !self.input_done
                            || self.pending.iter().any(|s| {
                                !s.trim_matches(|c| {
                                    lk_agents_abi::utils::strings::PYTHON_WHITESPACE.contains(&c)
                                })
                                .is_empty()
                            });
                        if !more_audio_coming {
                            if let Some(previous) = self.held.replace(frame) {
                                return Ok(Some(previous));
                            }
                            continue;
                        }
                        return Ok(Some(frame));
                    }
                    Ok(None) => {
                        return match self.held.take() {
                            Some(mut last) => {
                                last.is_final = true;
                                // Alignment for trailing sentences that produced no audio rides on the
                                // real last frame (no silence marker, which is `AudioEmitter`'s job).
                                last.timed_transcript.append(&mut self.pending_transcript);
                                Ok(Some(last))
                            }
                            None => Ok(None),
                        };
                    }
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

    fn take_call_metrics(&mut self, in_flight: bool) -> Option<Vec<AgentMetrics>> {
        Some(self.ledger().take(in_flight))
    }

    fn report_retries(&mut self, report: lk_agents_abi::utils::stream::RetryReport) {
        self.report = Some(report);
    }

    fn call_ledger(&self) -> Option<lk_agents_abi::utils::stream::CallLedger> {
        let ledger = Arc::clone(&self.ledger);
        Some(Arc::new(move |in_flight| {
            ledger.lock().unwrap_or_else(|p| p.into_inner()).take(in_flight)
        }))
    }

    /// The worst inner close: `TimedOut` if any did not ack, else `Failed` if any teardown failed.
    fn close(mut self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move {
            let current = match self.current.take() {
                Some(cur) => cur.close(deadline).await,
                None => Closed::Clean,
            };
            self.rx.close();
            self.inner_closed.and(current)
        })
    }
}
