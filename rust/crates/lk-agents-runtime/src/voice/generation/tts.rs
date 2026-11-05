//! `perform_tts_inference`: one segment's duplex synthesis, retried, and its buffered playout.

use super::forward::{PlaybackGate, TextLeg, capture, drain_segment_text, take_text};
use super::llm::{Retried, report_metrics, retry};
use super::playout::{Next, Segment, SourceFailed, first_frame, perform_audio_forwarding, unheard};
use crate::telemetry;
use crate::voice::session::SessionShared;
use crate::voice::speech_handle::SpeechHandle;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{SynthesizedAudio, Tts};
use lk_agents_voice::agent_session::DriverError;
use lk_agents_voice::generation::SpeechSegmentEvent;
use lk_agents_voice::generation::effects::{AlignedTranscriptEffect, StreamingTtsEffect};
pub(super) use lk_agents_voice::generation::machine::tts::Lost;
use lk_agents_voice::generation::machine::tts::{
    AlignedTranscript, MAX_BUFFERED_TTS_FRAMES, Read, StreamingTts, SynthesisMetrics,
};
use lk_agents_voice::generation::machine::tts::{AttemptFailure, RetryOutcome, SynthesisAttempt};
use lk_agents_voice::generation::{GenEvent, TurnAccumulator};
use lk_agents_voice::trace::{GenerationStep, ProviderKind};
use lk_agents_voice::transcription::Heard;
use lk_agents_voice::transcription::Visible;
use lk_agents_voice::transcription::{Pipeline, TextTransform as _};
use std::sync::Arc;
use tokio::sync::mpsc;

/// One failed duplex stream: the error, whether audio came first, the words fed, the
/// synchroniser's words up to where its audio played, and the sink's account of that audio.
struct Failed {
    error: DriverError,
    produced: bool,
    words: Visible,
    synchronized: Option<String>,
    heard: Option<Heard>,
}

/// `perform_tts_inference`, retried: opens are retried until a frame arrives; a later failure, or a
/// spent budget, loses the voice.
#[allow(clippy::too_many_arguments)]
pub(super) async fn perform_tts_inference(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    acc: &mut TurnAccumulator,
    tts: &dyn Tts,
    expressive: bool,
    segment: usize,
    first: Visible,
    rx: &mut mpsc::UnboundedReceiver<SpeechSegmentEvent>,
    text_out: Option<&TextLeg<'_>>,
    mut aligned: Option<&mut Aligned<'_>>,
    playback: &mut SynthesisPlayout<'_>,
    node: &mut TtsNodeTrace,
) -> Result<Heard, Box<Lost>> {
    let mut state =
        SynthesisAttempt::new(&shared.conn.tts.unwrap_or_default(), first, shared.time.now());
    // The segment's `tts_request`, one `tts_request_run` per open. A registered `tts_node` runs
    // under the node span and makes its own request, if any.
    let mut trace = telemetry::TtsTrace::start(node.cx(shared), tts, true);
    loop {
        if let Some(lost) = state.before_open(handle.interrupted(), *shared.closing.borrow()) {
            return Err(Box::new(lost));
        }
        // The driver's TTS always streams (`tts_leg` adapts one that does not). A duplex open's
        // `Err` is a failed attempt, never a downgrade. What the words leg already closed is fed at
        // the open, so a preemptive generation's whole segment goes in at authorization; a live
        // segment streams the rest.
        if !state.input_done {
            state.input_done = take_text(rx, &mut state.opening, text_out, false).await;
        }
        let id = tts.id();
        shared.report(GenEvent::RequestStarted {
            speech: handle.id(),
            kind: lk_agents_core::usage::ModelKind::Tts,
            provider: id.provider,
            model: id.model,
        });
        let attempt = trace.attempt();
        // An adapter's own requests open under this attempt.
        let context = attempt.as_ref().map_or(trace.cx(), |a| a.cx()).clone();
        let (error, produced) = match tts.stream(&context, expressive).await {
            Err(error) => {
                trace.end_attempt(attempt, Some(&error));
                (DriverError::Provider(error), false)
            }
            Ok(stream) => {
                let (sink, mut source) = stream.split();
                report_retries(shared, handle, &mut source);
                let aligned = aligned.as_deref_mut();
                let tts_data = SynthesisMetrics::new(
                    tts.id(),
                    segment,
                    shared.time.wall_secs(),
                    shared.time.wall_secs(),
                );
                let streaming = Streaming {
                    shared,
                    handle,
                    acc,
                    sink: &sink,
                    source,
                    rx,
                    text_out,
                    transforms: None,
                    state: StreamingTts::new(state.opening.clone(), state.input_done),
                    text_buffer: SynthesisText::new(state.opening.text.clone()),
                    tts_data,
                    aligned,
                };
                let mut done = Synthesized::default();
                match synthesize(shared, handle, streaming, playback, &mut done).await {
                    (Ok(heard), _) => {
                        node.first_audio(done.metrics.as_ref().and_then(|m| m.ttfb));
                        if let Some(id) = done.metrics.as_ref().map(|m| m.request_id.clone()) {
                            trace.request_id(id);
                        }
                        trace.end_attempt(attempt, None);
                        trace.end(done.text, done.metrics.as_ref());
                        return Ok(heard);
                    }
                    (Err(failed), done) => {
                        let provider = match &failed.error {
                            DriverError::Provider(error) => Some(error.clone()),
                            _ => None,
                        };
                        trace.end_attempt(attempt, provider.as_ref());
                        state.stream_failed(failed.words, done, failed.synchronized, failed.heard);
                        (failed.error, failed.produced)
                    }
                }
            }
        };
        let error = match state.failed(error, produced) {
            AttemptFailure::Retry(error) => error,
            AttemptFailure::Stop(lost) => {
                fail(&mut trace, &lost);
                return Err(Box::new(lost));
            }
        };
        let failed = |e, r| tts.failed(e, r);
        let outcome =
            match retry(shared, handle, &mut state.retry, ProviderKind::Tts, &error, &failed).await
            {
                Retried::Again => RetryOutcome::Again,
                Retried::Interrupted => RetryOutcome::Interrupted,
                Retried::GiveUp => RetryOutcome::Exhausted,
            };
        if let Some(lost) = state.after_retry(outcome) {
            fail(&mut trace, &lost);
            return Err(Box::new(lost));
        }
    }
}

/// A lost voice fails its request with the provider's error.
fn fail(trace: &mut telemetry::TtsTrace, lost: &Lost) {
    if let Some(DriverError::Provider(error)) = &lost.failure {
        trace.fail(error);
    }
}

/// What a segment's synthesis fed and measured, for its `tts_request`.
#[derive(Default)]
struct Synthesized {
    text: String,
    metrics: Option<lk_agents_abi::metrics::base::TtsMetrics>,
}

/// A step's `tts_node`, under the speech's `agent_turn`: the first segment's time to first audio.
/// The model and provider are its request's: the TTS's labels only its time to first audio
/// measurement.
pub(super) struct TtsNodeTrace {
    span: Option<telemetry::ScopedSpan<telemetry::TtsNode>>,
    ttfb: Option<f64>,
    id: Option<lk_agents_abi::ProviderId>,
}

impl TtsNodeTrace {
    pub(super) fn start(
        shared: &SessionShared,
        turn: Option<telemetry::SpanContext>,
        tts: Option<&dyn Tts>,
    ) -> Self {
        let span = turn.zip(tts).map(|(turn, _)| {
            telemetry::start(
                &shared.trace.clone().under::<telemetry::AgentTurn>(turn),
                telemetry::TtsNode,
            )
        });
        TtsNodeTrace { span, ttfb: None, id: tts.map(Tts::id) }
    }

    fn cx<'a>(&'a self, shared: &'a SessionShared) -> &'a telemetry::Cx {
        self.span.as_ref().map_or(&shared.trace, |span| span.cx())
    }

    fn first_audio(&mut self, ttfb: Option<f64>) {
        self.ttfb = self.ttfb.or(ttfb);
    }

    pub(super) fn end(self) {
        let Some(span) = self.span else { return };
        if let (Some(ttfb), Some(id)) = (self.ttfb, &self.id) {
            let model = telemetry::model_labels(id);
            telemetry::measure(span.cx(), &telemetry::TurnTtsTtfb { model }, ttfb);
        }
        telemetry::end(
            span,
            telemetry::TtsNodeEnd { ttfb: telemetry::Measured(self.ttfb.map(telemetry::Seconds)) },
        );
    }
}

/// A voice that does not stream retries its sentences itself: each retry is the session's
/// recoverable error and a retry the journal waits out, whichever reply opened the stream.
pub(super) fn report_retries(
    shared: &Arc<SessionShared>,
    handle: &SpeechHandle,
    source: &mut lk_agents_abi::TtsSource,
) {
    let (on, speech) = (Arc::clone(shared), handle.id());
    source.report_retries(Arc::new(move |error, attempt, wait| {
        on.report(GenEvent::ProviderFailed {
            speech,
            leg: ProviderKind::Tts,
            recoverable: true,
            error: Box::new(error),
        });
        let at = on.time.now().saturating_add(wait);
        on.trace(GenerationStep::RetryArmed { speech, leg: ProviderKind::Tts, attempt, at });
    }));
}

/// One duplex stream: `end_input` is always sent, or a duplex stream never yields its final frame.
/// `Err` carries what [`Failed`] says; the `bool` is whether the
/// segment's text has ended.
async fn synthesize(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    mut streaming: Streaming<'_, '_>,
    playback: &mut SynthesisPlayout<'_>,
    done: &mut Synthesized,
) -> (Result<Heard, Failed>, bool) {
    let sink = streaming.sink;
    // Post-transform, like the chunked path: the journal says what the synthesiser was given.
    let text = transformed(shared, &streaming.state.words.raw);
    let segment = streaming.tts_data.segment as u32;
    shared.trace(GenerationStep::TtsRequested { speech: handle.id(), segment, text });
    streaming.transforms =
        (!shared.tts_transforms.is_empty()).then(|| Pipeline::new(&shared.tts_transforms));
    // The words in hand (first chunk, or a failed attempt's feed) go in first.
    let error = push_text(sink, &mut streaming.transforms, &streaming.state.words.raw).err();
    if streaming.state.opening_pushed(error) {
        end_input(sink, &mut streaming.transforms);
    }
    let heard = playback.run(shared, handle, &mut streaming).await;
    take_tts_calls(&mut streaming.tts_data, shared, handle, &mut streaming.source, true);
    let Streaming { state, source, rx, text_out, tts_data, aligned, .. } = streaming;
    let StreamingTts { words, input_done, .. } = state;
    // Always: `end_input` before teardown (a second call is a no-op).
    sink.end_input();
    let fed = transformed(shared, &words.raw);
    let _ =
        source.close(shared.time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT)).await;
    let produced = tts_data.produced();
    match heard {
        Ok(mut heard) => {
            if let Some(record) = tts_data.finish(
                &fed,
                shared.time.wall_secs(),
                handle.interrupted(),
                handle.speech_id(),
            ) {
                if let lk_agents_abi::metrics::base::AgentMetrics::Tts(m) = &record {
                    done.metrics = Some(m.clone());
                }
                report_metrics(shared, handle, record);
            }
            done.text = fed;
            // Drain the rest so the next segment starts clean; it joins the item only if the audio
            // played out whole (on the aligned path the item is the stamps). Its words still go
            // down the transcript leg, which an interruption keeps from the output.
            if !input_done {
                let rest = drain_segment_text(rx, Visible::default(), text_out).await;
                if heard.complete && aligned.is_none() {
                    heard.text.push_str(&rest.text);
                }
            }
            (Ok(heard), true)
        }
        Err(failed) => {
            let SourceFailed { error, synchronized, heard } = *failed;
            (Err(Failed { error, produced, words, synchronized, heard }), input_done)
        }
    }
}

/// One text chunk through the segment's filters. Empty output is normal (a filter mid-pattern).
pub(super) fn push_text(
    sink: &lk_agents_abi::TtsSink,
    transforms: &mut Option<Pipeline>,
    chunk: &str,
) -> Result<(), ProviderError> {
    if chunk.is_empty() {
        return Ok(());
    }
    match transforms {
        Some(p) => p.push(chunk).into_iter().try_for_each(|piece| sink.push(piece)),
        None => sink.push(chunk.to_owned()),
    }
}

/// The input is over: the filters' tail, then `end_input` (not `flush`: a `StreamAdapter` holds its
/// final frame until input is done).
pub(super) fn end_input(sink: &lk_agents_abi::TtsSink, transforms: &mut Option<Pipeline>) {
    for piece in transforms.as_mut().map(|p| p.flush()).unwrap_or_default() {
        let _ = sink.push(piece);
    }
    sink.end_input();
}

/// `next()` is created **once** per frame into `pending` and borrowed by `select!`, so a winning
/// text arm drops a borrow, not the future. The text arm is `biased` first.
struct Streaming<'a, 'b> {
    shared: &'a Arc<SessionShared>,
    handle: &'a SpeechHandle,
    acc: &'a mut TurnAccumulator,
    sink: &'a lk_agents_abi::TtsSink,
    source: lk_agents_abi::TtsSource,
    rx: &'a mut mpsc::UnboundedReceiver<SpeechSegmentEvent>,
    text_out: Option<&'a TextLeg<'a>>,
    transforms: Option<Pipeline>,
    state: StreamingTts,
    text_buffer: SynthesisText,
    tts_data: SynthesisMetrics,
    aligned: Option<&'a mut Aligned<'b>>,
}

impl Segment for Streaming<'_, '_> {
    fn started_speaking(&mut self, at: f64) {
        first_frame(self.shared, self.acc, at);
    }

    async fn next(&mut self) -> Result<Next, DriverError> {
        match self.state.before_read() {
            Read::End => return Ok(Next::End),
            Read::Failed(error) => return Err(DriverError::from(error)),
            Read::Provider => {}
        }
        let Streaming { sink, source, rx, text_out, transforms, state, text_buffer, .. } = self;
        let chunk = {
            let mut pending = Box::pin(source.next());
            loop {
                if state.input_done {
                    break pending.await;
                }
                tokio::select! {
                    biased;
                    event = rx.recv() => {
                        let chunk = match event {
                            Some(SpeechSegmentEvent::Chunk(chunk)) => Some(chunk),
                            Some(SpeechSegmentEvent::End) | None => None,
                        };
                        let mut effects: std::collections::VecDeque<_> = state.text(chunk).into();
                        while let Some(effect) = effects.pop_front() {
                            match effect {
                                StreamingTtsEffect::Capture(chunk) => {
                                    capture(*text_out, &chunk.text).await;
                                    text_buffer.append(&chunk.text);
                                    effects.push_front(state.captured(chunk));
                                }
                                StreamingTtsEffect::Push(raw) => {
                                    if let Err(error) = push_text(sink, transforms, &raw) {
                                        state.push_failed(error);
                                    }
                                }
                                StreamingTtsEffect::EndInput => end_input(sink, transforms),
                                StreamingTtsEffect::FlushText => {
                                    if let Some(out) = text_out { out.flush(); }
                                }
                            }
                        }
                    },
                    got = &mut pending => break got,
                }
            }
        };
        take_tts_calls(&mut self.tts_data, self.shared, self.handle, &mut self.source, false);
        let chunk = match chunk {
            Ok(Some(chunk)) => chunk,
            Ok(None) => {
                self.state.source_ended();
                if let Some(a) = self.aligned.as_deref_mut() {
                    a.ended(&self.state.words.text).await;
                }
                return Ok(Next::End);
            }
            Err(e) => return Err(DriverError::from(e)),
        };
        if let Some(stats) = self.tts_data.note(&chunk, self.shared.time.wall_secs()) {
            self.acc.record_tts(self.tts_data.segment, stats);
        }
        if let Some(a) = self.aligned.as_deref_mut() {
            a.chunk(&chunk).await;
        }
        // An `is_final` frame ends the segment's audio; a pooled connection never closes.
        if self.state.frame(chunk.is_final)
            && let Some(a) = self.aligned.as_deref_mut()
        {
            a.ended(&self.state.words.text).await;
        }
        Ok(Next::Frame(chunk.frame))
    }

    fn text(&self) -> String {
        match self.aligned.as_deref() {
            Some(a) => a.state.words.clone(),
            None => self.state.words.text.clone(),
        }
    }
}

/// Run whole on a scratch pipeline. The duplex path journals this, so `TtsRequested{text}` is what
/// the TTS was given.
pub(super) fn transformed(shared: &Arc<SessionShared>, raw: &str) -> String {
    if shared.tts_transforms.is_empty() {
        return raw.to_owned();
    }
    let mut p = Pipeline::new(&shared.tts_transforms);
    let mut out = p.push(raw);
    out.extend(p.flush());
    out.concat()
}

/// Provider-owned sentence metrics are observations, not a second synthesis record.
fn take_tts_calls(
    state: &mut SynthesisMetrics,
    shared: &SessionShared,
    handle: &SpeechHandle,
    source: &mut lk_agents_abi::TtsSource,
    in_flight: bool,
) {
    let Some(records) = source.take_call_metrics(in_flight) else { return };
    for record in state.calls(records, handle.speech_id()) {
        report_metrics(shared, handle, record);
    }
}

/// Append-only text for one synthesis attempt. Queued frames retain prefix positions in one
/// buffer instead of copying the complete accumulated transcript into every audio event.
struct SynthesisText {
    buffer: Arc<parking_lot::Mutex<String>>,
    aligned: Option<Arc<str>>,
}

impl SynthesisText {
    fn new(text: String) -> Self {
        Self { buffer: Arc::new(parking_lot::Mutex::new(text)), aligned: None }
    }

    fn append(&self, delta: &str) {
        self.buffer.lock().push_str(delta);
    }

    fn snapshot(&self) -> SynthesisTextSnapshot {
        let end = self.buffer.lock().len();
        SynthesisTextSnapshot(SnapshotText::Prefix { buffer: self.buffer.clone(), end })
    }

    /// Provider-aligned words can differ from generated text. Copy once when those words change;
    /// frames with the same stamped transcript share its immutable snapshot.
    fn aligned_snapshot(&mut self, words: &str) -> SynthesisTextSnapshot {
        let text = match &self.aligned {
            Some(text) if text.as_ref() == words => text.clone(),
            _ => {
                let text: Arc<str> = Arc::from(words);
                self.aligned = Some(text.clone());
                text
            }
        };
        SynthesisTextSnapshot(SnapshotText::Aligned(text))
    }
}

/// The private end offset is always a captured UTF-8 boundary; the buffer only appends, so older
/// snapshots remain valid. Aligned words retain their own immutable provider transcript.
pub(super) struct SynthesisTextSnapshot(SnapshotText);

enum SnapshotText {
    Prefix { buffer: Arc<parking_lot::Mutex<String>>, end: usize },
    Aligned(Arc<str>),
}

impl SynthesisTextSnapshot {
    fn text(&self) -> String {
        match &self.0 {
            SnapshotText::Prefix { buffer, end } => buffer.lock()[..*end].to_owned(),
            SnapshotText::Aligned(text) => text.to_string(),
        }
    }
}

/// Text and audio share a queue so aligned captions cannot overtake their audio. Each audio event
/// holds a permit until playback consumes it, backpressuring the synthesiser.
pub(super) enum SynthesisEvent {
    Text(String),
    TimedText(lk_agents_abi::TranscriptChunk),
    FlushText,
    Audio(Result<Next, DriverError>, SynthesisTextSnapshot, tokio::sync::OwnedSemaphorePermit),
}

pub(super) struct SynthesisPlayout<'a> {
    pub(super) binding: &'a mut super::output::OutputBinding,
    pub(super) tx: mpsc::UnboundedSender<SynthesisEvent>,
    pub(super) rx: mpsc::UnboundedReceiver<SynthesisEvent>,
    pub(super) gate: &'a mut PlaybackGate,
}

impl SynthesisPlayout<'_> {
    /// Authorization blocks only the consumer; it never cancels a pending provider read.
    async fn run(
        &mut self,
        shared: &Arc<SessionShared>,
        handle: &SpeechHandle,
        segment: &mut Streaming<'_, '_>,
    ) -> Result<Heard, Box<SourceFailed>> {
        let permits = Arc::new(tokio::sync::Semaphore::new(MAX_BUFFERED_TTS_FRAMES));
        let stopped = tokio::sync::Notify::new();
        let tx = &self.tx;
        let produce = async {
            loop {
                let next = async {
                    let permit = permits.clone().acquire_owned().await.expect("buffer is open");
                    (segment.next().await, permit)
                };
                let (result, permit) = tokio::select! {
                    biased;
                    () = handle.wait_for_interrupted() => break,
                    () = stopped.notified() => break,
                    v = next => v,
                };
                let ended = matches!(result, Ok(Next::End) | Err(_));
                let text = match segment.aligned.as_deref() {
                    Some(aligned) => segment.text_buffer.aligned_snapshot(&aligned.state.words),
                    None => segment.text_buffer.snapshot(),
                };
                let event = SynthesisEvent::Audio(result, text, permit);
                if tx.send(event).is_err() || ended {
                    break;
                }
            }
        };
        let consume = async {
            let mut started_at = None;
            let admission = if self.gate.wait(shared, handle).await {
                self.binding.activate().await
            } else {
                Ok(false)
            };
            let result = match admission {
                Err(error) => Err(Box::new(SourceFailed::from(error))),
                Ok(true) => {
                    let audio = self.binding.audio();
                    let leg = TextLeg::new(self.binding.text(), None, shared, handle.id());
                    let mut buffered = SynthesizedSegment {
                        rx: &mut self.rx,
                        text: None,
                        leg: &leg,
                        started_at: None,
                    };
                    // Activated with no audio output: the words start the turn speaking as they
                    // show, and the synthesised audio is dropped unheard.
                    if audio.is_none() {
                        shared.report(GenEvent::FirstFrame { speech: handle.id() });
                        buffered.started_at = Some(shared.time.wall_secs());
                    }
                    let result =
                        perform_audio_forwarding(shared, handle, audio.as_deref(), &mut buffered)
                            .await;
                    started_at = buffered.started_at;
                    result
                }
                Ok(false) => Ok(unheard()),
            };
            // A failed output stops synthesis too; the caller still ends input and closes the source.
            stopped.notify_one();
            (result, started_at)
        };
        let (_, (result, started_at)) = tokio::join!(produce, consume);
        if let Some(at) = started_at {
            segment.started_speaking(at);
        }
        result
    }

    /// A failed synthesis can leave text without audio. It still belongs behind authorization.
    pub(super) async fn flush_text(
        &mut self,
        shared: &Arc<SessionShared>,
        handle: &SpeechHandle,
    ) -> Result<(), DriverError> {
        if !self.gate.wait(shared, handle).await || !self.binding.activate().await? {
            return Ok(());
        }
        let leg = TextLeg::new(self.binding.text(), None, shared, handle.id());
        while let Ok(event) = self.rx.try_recv() {
            leg.forward(event).await;
        }
        Ok(())
    }
}

/// The buffered side of [`SynthesisPlayout`]: what the synthesiser produced, played in order.
struct SynthesizedSegment<'a, 'b> {
    rx: &'a mut mpsc::UnboundedReceiver<SynthesisEvent>,
    text: Option<SynthesisTextSnapshot>,
    leg: &'a TextLeg<'b>,
    started_at: Option<f64>,
}

impl Segment for SynthesizedSegment<'_, '_> {
    async fn next(&mut self) -> Result<Next, DriverError> {
        while let Some(event) = self.rx.recv().await {
            match event {
                SynthesisEvent::Audio(result, text, _permit) => {
                    self.text = Some(text);
                    return result;
                }
                text => self.leg.forward(text).await,
            }
        }
        Ok(Next::End)
    }
    fn text(&self) -> String {
        self.text.as_ref().map_or_else(String::new, SynthesisTextSnapshot::text)
    }
    fn started_speaking(&mut self, at: f64) {
        self.started_at.get_or_insert(at);
    }
}

/// `_aligned_transcript_or_text`: the item is the stamped words, with timings; if nothing was
/// stamped by the audio's end, the model's text is forwarded whole.
pub(super) struct Aligned<'a> {
    pub(super) leg: Option<&'a TextLeg<'a>>,
    pub(super) state: AlignedTranscript,
}

impl Aligned<'_> {
    async fn chunk(&mut self, chunk: &SynthesizedAudio) {
        for word in &chunk.timed_transcript {
            let effect = self.state.word(word);
            self.apply(effect).await;
        }
    }

    async fn ended(&mut self, fallback: &str) {
        for effect in self.state.end(fallback) {
            self.apply(effect).await;
        }
    }

    async fn apply(&self, effect: AlignedTranscriptEffect) {
        match effect {
            AlignedTranscriptEffect::CaptureTimed(word) => {
                if let Some(leg) = self.leg {
                    leg.capture_timed(&word).await;
                }
            }
            AlignedTranscriptEffect::Capture(text) => capture(self.leg, &text).await,
            AlignedTranscriptEffect::WarnUnaligned => log::warn!(
                target: "livekit.agents",
                "no aligned transcript was returned from tts, forwarding the generated text \
                 without word timings"
            ),
            AlignedTranscriptEffect::Flush => {
                if let Some(leg) = self.leg {
                    leg.flush();
                }
            }
        }
    }
}

#[cfg(test)]
mod text_snapshot_tests {
    use super::*;

    #[test]
    fn frame_snapshots_retain_their_utf8_prefix_after_later_text_arrives() {
        let text = SynthesisText::new("hello 🌍".into());
        let first = text.snapshot();
        text.append(" — more");
        let second = text.snapshot();
        assert_eq!(first.text(), "hello 🌍");
        assert_eq!(second.text(), "hello 🌍 — more");
        assert!(matches!((&first.0, &second.0), (
            SnapshotText::Prefix { buffer: first, .. },
            SnapshotText::Prefix { buffer: second, .. },
        ) if Arc::ptr_eq(first, second)));
    }

    #[test]
    fn a_full_audio_buffer_shares_one_transcript_allocation() {
        let words = "x".repeat(2 * 1024 * 1024);
        let text = SynthesisText::new(words);
        let snapshots: Vec<_> = (0..MAX_BUFFERED_TTS_FRAMES).map(|_| text.snapshot()).collect();
        assert!(snapshots.iter().all(|snapshot| matches!(&snapshot.0,
            SnapshotText::Prefix { buffer, end }
                if Arc::ptr_eq(&text.buffer, buffer) && *end == 2 * 1024 * 1024
        )));
        assert_eq!(text.buffer.lock().len(), 2 * 1024 * 1024);
    }

    #[test]
    fn queued_prefix_survives_its_producer() {
        let snapshot = {
            let text = SynthesisText::new("spoken".into());
            text.snapshot()
        };
        assert_eq!(snapshot.text(), "spoken");
    }

    #[test]
    fn aligned_frames_share_provider_words_and_keep_old_snapshots() {
        let mut text = SynthesisText::new("generated words differ".into());
        let first = text.aligned_snapshot("spoken 🌍");
        let same = text.aligned_snapshot("spoken 🌍");
        assert!(matches!((&first.0, &same.0), (
            SnapshotText::Aligned(first), SnapshotText::Aligned(same),
        ) if Arc::ptr_eq(first, same)));
        let later = text.aligned_snapshot("spoken 🌍 again");
        assert_eq!(first.text(), "spoken 🌍");
        assert_eq!(later.text(), "spoken 🌍 again");
        // Before stamps or fallback arrive, aligned frames carry no generated transcript.
        assert_eq!(text.aligned_snapshot("").text(), "");
    }
}
