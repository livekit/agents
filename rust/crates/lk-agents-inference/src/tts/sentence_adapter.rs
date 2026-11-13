use super::pacing::trim_python_whitespace;
use crate::telemetry;
use lk_agents_abi::ApiConnectOptions;
use lk_agents_abi::TranscriptChunk;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::metrics::base::AgentMetrics;
use lk_agents_abi::tts::stream_adapter::TtsStreamAdapter;
use lk_agents_abi::tts::types::{
    SharedTtsOptions, SpeechChunk, SynthesizedAudio, TextUnit, Tts, TtsCapabilities, TtsSink,
    TtsSource,
};
use lk_agents_abi::types::identity::ProviderId;
use lk_agents_abi::types::provider::ProviderEvents;
use lk_agents_abi::utils::clock::{Clock, Ts};
use lk_agents_abi::utils::stream::{Closed, HOST_TEARDOWN_TIMEOUT, StreamInput, StreamOutput};
use lk_agents_core::metrics::provider::TtsMetricState;
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::strings::shortuuid;
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::mpsc;

// `is_final` needs a one-frame hold, taken only once no further sentence can arrive.

/// A TTS with no duplex path as a streaming one: each chunk its stage prepared is synthesized on
/// its own.
#[derive(Clone)]
pub struct StreamAdapter {
    tts: Arc<dyn Tts>,
    /// The host object it was built as: its name, and its `error` events; else it names and
    /// reports through the TTS it wraps.
    host: Option<(String, Arc<dyn ProviderEvents>)>,
    /// Bounds the per-sentence inner closes done inside `next()`.
    clock: Arc<dyn Clock>,
    /// Each sentence's retries (`ChunkedStream`'s own loop in the reference): a sentence failing
    /// after earlier ones were heard is synthesized again, not the whole reply lost.
    budget: lk_agents_abi::ApiConnectOptions,
}

impl StreamAdapter {
    /// No sentence is retried; see [`with_budget`](Self::with_budget).
    pub fn new(tts: Arc<dyn Tts>, clock: Arc<dyn Clock>) -> Self {
        let budget = lk_agents_abi::ApiConnectOptions {
            max_retry: 0,
            ..lk_agents_abi::ApiConnectOptions::default()
        };
        StreamAdapter { tts, host: None, clock, budget }
    }

    /// Each sentence retried within `budget`.
    pub fn with_budget(mut self, budget: lk_agents_abi::ApiConnectOptions) -> Self {
        self.budget = budget;
        self
    }

    /// Itself when it has a duplex path, else adapted, each sentence retried within `budget`;
    /// Itself when it has a duplex path, else adapted, each chunk retried within `budget`.
    pub fn adapted(
        tts: Arc<dyn Tts>,
        clock: Arc<dyn Clock>,
        budget: lk_agents_abi::ApiConnectOptions,
    ) -> Arc<dyn Tts> {
        if tts.capabilities().streaming {
            return tts;
        }
        Arc::new(StreamAdapter::new(tts, clock).with_budget(budget))
    }
}

impl Tts for StreamAdapter {
    fn label(&self) -> String {
        match &self.host {
            Some((label, _)) => label.clone(),
            None => format!("StreamAdapter[{}]", self.tts.label()),
        }
    }

    fn request_span(&self) -> telemetry::RequestSpan {
        telemetry::RequestSpan::StreamAdapter
    }

    /// The adapter is transparent to the metrics.
    fn id(&self) -> ProviderId {
        self.tts.id()
    }

    fn prewarm(&self, cx: &telemetry::Cx) {
        self.tts.prewarm(cx);
    }

    fn failed(&self, error: ProviderError, recoverable: bool) {
        match &self.host {
            Some((_, events)) => events.failed(error, recoverable),
            None => self.tts.failed(error, recoverable),
        }
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

    fn text_unit(&self) -> TextUnit {
        self.tts.text_unit()
    }

    fn max_text_len(&self) -> Option<usize> {
        self.tts.max_text_len()
    }

    /// Each chunk is retried on its own, within the budget its view carries.
    fn retries_itself(&self) -> bool {
        true
    }

    fn with_conn_options(&self, conn: &ApiConnectOptions) -> Option<Arc<dyn Tts>> {
        let tts = self.tts.with_conn_options(conn).unwrap_or_else(|| self.tts.clone());
        let (host, clock) = (self.host.clone(), self.clock.clone());
        Some(Arc::new(StreamAdapter { tts, host, budget: *conn, clock }))
    }

    fn options(&self) -> Arc<dyn SharedTtsOptions> {
        self.tts.options()
    }

    /// `stream_adapter.py:58`.
    fn synthesize<'a>(
        &'a self,
        cx: &telemetry::Cx,
        chunk: &'a SpeechChunk,
    ) -> lk_agents_abi::BoxFuture<'a, Result<TtsSource, ProviderError>> {
        let cx = cx.clone();
        Box::pin(async move { self.tts.synthesize(&cx, chunk).await })
    }
    fn stream(
        &self,
        cx: &telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<lk_agents_abi::DuplexStream<TtsSink, TtsSource>, ProviderError>,
    > {
        let cx = cx.clone();
        Box::pin(async move { self.open(&cx) })
    }
}

impl StreamAdapter {
    /// One stream: each chunk pushed synthesized in turn.
    pub(crate) fn open(
        &self,
        cx: &telemetry::Cx,
    ) -> Result<lk_agents_abi::DuplexStream<TtsSink, TtsSource>, ProviderError> {
        self.budget
            .validate()
            .map_err(|error| ApiError::new(error.to_string()).with_retryable(false))?;
        // Unbounded on purpose: text -> TTS may not drop, a lost token mangles speech.
        let (tx, chunks) = mpsc::unbounded_channel();
        let sink = AdapterSink { tx: Mutex::new(Some(tx)), closed: AtomicBool::new(false) };
        let source = AdapterSource {
            tts: self.tts.clone(),
            cx: cx.clone(),
            budget: self.budget,
            chunk: None,
            report: None,
            clock: Arc::clone(&self.clock),
            chunks,
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
    tx: Mutex<Option<mpsc::UnboundedSender<SpeechChunk>>>,
    closed: AtomicBool,
}

impl AdapterSink {
    fn send(&self, item: SpeechChunk) -> Result<(), ProviderError> {
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

impl StreamInput<SpeechChunk> for AdapterSink {
    fn push(&self, item: SpeechChunk) -> Result<(), ProviderError> {
        // a chunk with nothing to say is not input; nor is one after the input closed
        if trim_python_whitespace(&item.text).is_empty() || self.closed.load(Ordering::Acquire) {
            return Ok(());
        }
        self.send(item)
    }

    /// Each chunk is its own synthesis already.
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }

    fn end_input(&self) {
        self.closed.store(true, Ordering::Release);
        drop(self.tx.lock().take());
    }
}

struct AdapterSource {
    tts: Arc<dyn Tts>,
    /// What each sentence's synthesis runs under: the stream's caller.
    cx: telemetry::Cx,
    budget: lk_agents_abi::ApiConnectOptions,
    /// The chunk being synthesized: what a retry synthesizes again.
    chunk: Option<SpeechChunk>,
    /// Who hears of each retried sentence (the engine's error event and journal).
    report: Option<lk_agents_abi::utils::stream::RetryReport>,
    clock: Arc<dyn Clock>,
    /// The chunks pushed, each in the wrapped TTS's syntax with its words; closed once the input
    /// ended.
    chunks: mpsc::UnboundedReceiver<SpeechChunk>,
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
            let Some(chunk) = self.chunks.recv().await else { return Ok(None) };
            if trim_python_whitespace(&chunk.text).is_empty() {
                continue;
            }
            // the chunk's words start where its audio does
            if !chunk.said.is_empty() {
                self.pending_transcript.push(TranscriptChunk {
                    text: chunk.said.clone(),
                    start_time: Some(self.duration),
                    ..TranscriptChunk::default()
                });
            }
            self.chunk = Some(chunk);
            self.open_chunk().await?;
        }
    }

    /// The current sentence's next audio; `None` once the sentence ended or was retried.
    async fn read_sentence_audio(&mut self) -> Result<Option<SynthesizedAudio>, ProviderError> {
        let Some(current) = self.current.as_mut() else { return Ok(None) };
        match current.next().await {
            Ok(Some(audio)) => {
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
                Err(error.into_final())
            }
        }
    }

    /// The current chunk's synthesis, retried while its open fails within the budget.
    async fn open_chunk(&mut self) -> Result<(), ProviderError> {
        let Some(chunk) = self.chunk.clone() else { return Ok(()) };
        let start = lk_agents_abi::unix_secs(self.clock.wall());
        let (report, clock, tts) = (self.report.clone(), self.clock.clone(), self.tts.clone());
        let observer: crate::retry::Observer = Arc::new(move |error, retry, attempt| {
            if let lk_agents_core::request::Retry::At(at) = retry {
                tts.failed(error.clone(), true);
                if let Some(report) = &report {
                    report(error.clone(), attempt, at.saturating_duration_since(clock.now()));
                }
            }
        });
        let text = chunk.text.clone();
        let open = super::request::synthesis(self.tts.clone(), &self.cx, chunk, self.budget);
        self.current = Some(
            crate::retry::stream(
                &self.budget,
                self.clock.clone(),
                open,
                |_| true,
                observer,
                tokio_util::sync::CancellationToken::new(),
            )
            .await,
        );
        let id = self.tts.id();
        let fold = TtsMetricState::new(false, id.label.clone(), Some(id.metadata()));
        self.ledger().call = Some(Call { text, start, fold });
        Ok(())
    }

    /// The inner frame as this stream's: its duration counted, then input drained (and paced
    /// against it), its ids ours, and the pending transcript on it. A sentence boundary is not a
    /// segment boundary (`tts/tts.py:1183`): the inner provider's ids and `is_final` are not ours.
    fn stamp(&mut self, mut frame: SynthesizedAudio) -> SynthesizedAudio {
        self.duration += frame.frame.duration();
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

    /// Once the input ended and every chunk was taken, no more audio can come.
    fn more_audio_coming(&self) -> bool {
        !(self.chunks.is_closed() && self.chunks.is_empty())
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
            self.chunks.close();
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

/// `_core.TtsStreamAdapter`: the generated adapter calls this.
impl TtsStreamAdapter for StreamAdapter {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        label: String,
        events: Arc<dyn ProviderEvents>,
        tts: Arc<dyn Tts>,
    ) -> Self {
        StreamAdapter { host: Some((label, events)), ..StreamAdapter::new(tts, env.clock.clone()) }
    }

    fn capabilities(&self) -> TtsCapabilities {
        Tts::capabilities(self)
    }

    fn sample_rate(&self) -> u32 {
        Tts::sample_rate(self)
    }

    fn num_channels(&self) -> u16 {
        Tts::num_channels(self)
    }
}

#[cfg(test)]
#[path = "sentence_adapter_tests.rs"]
mod tests;
