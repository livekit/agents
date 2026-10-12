//! `perform_tts_inference`: one segment's duplex synthesis, retried, and its buffered playout.

use super::forward::{PlaybackGate, TextLeg, capture, drain_segment_text, take_text};
use super::llm::{Retried, SegEvent, extend, report_metrics, retry};
use super::pipeline_reply::{fatal, teardown_deadline};
use super::playout::{Next, Segment, first_frame, perform_audio_forwarding, unheard};
use crate::agent_session::SessionShared;
use crate::agent_session::DriverError;
use crate::generation::{Attempt, TurnAccumulator};
use crate::speech_handle::SpeechHandle;
use crate::trace::{GenerationStep, ProviderLeg};
use crate::transcription::Heard;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::metrics::base::AgentMetrics;
use lk_agents_abi::telemetry::provider::{Context, TtsMetricState};
use lk_agents_abi::{SynthesizedAudio, Tts};
use lk_agents_voice::transcription::Visible;
use lk_agents_voice::transcription::text_transforms::{Pipeline, TextTransform as _};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Backpressure speculative synthesis after about five seconds of 20 ms frames (reads pause; no
/// frame is dropped).
const MAX_BUFFERED_TTS_FRAMES: usize = 256;

/// `opening` is the text so far, still the item's. `lost`: the TTS failed for good; otherwise the
/// speech was interrupted or closing before the synthesiser was asked, and no leg is opened.
pub(super) struct Lost {
    pub(super) lost: bool,
    pub(super) opening: Visible,
}

/// `perform_tts_inference`, retried: opens are retried until a frame arrives; a later failure, or a
/// spent budget, loses the voice.
#[allow(clippy::too_many_arguments)]
pub(super) async fn perform_tts_inference(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    acc: &mut TurnAccumulator,
    tts: &dyn Tts,
    segment: usize,
    first: Visible,
    rx: &mut mpsc::UnboundedReceiver<SegEvent>,
    text_out: Option<&TextLeg<'_>>,
    mut aligned: Option<&mut Aligned<'_>>,
    playback: &mut SynthesisPlayout<'_>,
) -> Result<Heard, Lost> {
    let mut attempt = Attempt::new(lk_agents_abi::ApiConnectOptions::default());
    // The segment's words so far and whether its text has ended: what a retried open replays.
    let mut text = first;
    let mut input_done = false;
    loop {
        // Never open for a speech that is over or a closing session.
        if handle.interrupted() || *shared.closing.borrow() {
            return Err(Lost { lost: false, opening: text });
        }
        // The driver's TTS always streams (`tts_leg` adapts one that does not). A duplex open's
        // `Err` is a failed attempt, never a downgrade. What the words leg already closed is fed at
        // the open, so a preemptive generation's whole segment goes in at authorization; a live
        // segment streams the rest.
        if !input_done {
            input_done = take_text(rx, &mut text, text_out, false).await;
        }
        let (error, produced) = match tts.stream().await {
            Err(error) => (error, false),
            Ok((sink, source)) => {
                let aligned = aligned.as_deref_mut();
                let tts_data = TtsGenerationData::new(tts, shared, segment);
                let streaming = Streaming {
                    shared,
                    acc,
                    sink: &sink,
                    source,
                    rx,
                    text_out,
                    transforms: None,
                    words: text.clone(),
                    input_done,
                    ended: false,
                    send_err: None,
                    tts_data,
                    aligned,
                };
                match synthesize(shared, handle, streaming, playback).await {
                    (Ok(heard), _) => return Ok(heard),
                    (Err((error, produced, so_far)), done) => {
                        (text, input_done) = (so_far, done);
                        (error, produced)
                    }
                }
            }
        };
        if produced {
            attempt.produced_output();
        }
        match retry(shared, handle, &mut attempt, ProviderLeg::Tts, &error).await {
            Retried::Again => {}
            Retried::Interrupted => return Err(Lost { lost: false, opening: text }),
            Retried::GiveUp => return Err(Lost { lost: true, opening: text }),
        }
    }
}

/// One duplex stream: `end_input` is always sent, or a duplex stream never yields its final frame.
/// `Err` carries the error, whether audio came first, and the words fed; the `bool` is whether the
/// segment's text has ended.
async fn synthesize(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    mut streaming: Streaming<'_, '_>,
    playback: &mut SynthesisPlayout<'_>,
) -> (Result<Heard, (ProviderError, bool, Visible)>, bool) {
    let sink = streaming.sink;
    // Post-transform, like the chunked path: the journal says what the synthesiser was given.
    let text = transformed(shared, &streaming.words.raw);
    let segment = streaming.tts_data.segment as u32;
    shared.trace(GenerationStep::TtsRequested { speech: handle.id(), segment, text });
    streaming.transforms =
        (!shared.tts_transforms.is_empty()).then(|| Pipeline::new(&shared.tts_transforms));
    // The words in hand (first chunk, or a failed attempt's feed) go in first.
    streaming.send_err = push_text(sink, &mut streaming.transforms, &streaming.words.raw).err();
    if streaming.input_done && streaming.send_err.is_none() {
        end_input(sink, &mut streaming.transforms);
    }
    let heard = playback.run(shared, handle, &mut streaming).await;
    let Streaming { words, input_done, source, rx, tts_data, aligned, .. } = streaming;
    // Always: `end_input` before teardown (a second call is a no-op).
    sink.end_input();
    let fed = transformed(shared, &words.raw);
    let _ = source.close(teardown_deadline(shared)).await;
    let produced = tts_data.produced();
    match heard {
        Ok(mut heard) => {
            tts_data.finish(shared, handle, &fed);
            // Drain the rest so the next segment starts clean; it joins the item only if the audio
            // played out whole (on the aligned path the item is the stamps).
            if !input_done {
                let rest = drain_segment_text(rx, Visible::default(), None).await;
                if heard.complete && aligned.is_none() {
                    heard.text.push_str(&rest.text);
                }
            }
            (Ok(heard), true)
        }
        Err(DriverError::Provider(error)) => (Err((error, produced, words)), input_done),
        Err(other) => (Err((fatal(other.to_string()), produced, words)), input_done),
    }
}

/// One text chunk through the segment's filters. Empty output is normal (a filter mid-pattern).
fn push_text(
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
fn end_input(sink: &lk_agents_abi::TtsSink, transforms: &mut Option<Pipeline>) {
    for piece in transforms.as_mut().map(|p| p.flush()).unwrap_or_default() {
        let _ = sink.push(piece);
    }
    sink.end_input();
}

/// `next()` is created **once** per frame into `pending` and borrowed by `select!`, so a winning
/// text arm drops a borrow, not the future. The text arm is `biased` first.
struct Streaming<'a, 'b> {
    shared: &'a Arc<SessionShared>,
    acc: &'a mut TurnAccumulator,
    sink: &'a lk_agents_abi::TtsSink,
    source: lk_agents_abi::TtsSource,
    rx: &'a mut mpsc::UnboundedReceiver<SegEvent>,
    text_out: Option<&'a TextLeg<'a>>,
    transforms: Option<Pipeline>,
    /// `raw` fed to the synthesiser, `text` for the item.
    words: Visible,
    input_done: bool,
    /// The provider's last frame was seen: the next `next` is `End` without asking the socket.
    ended: bool,
    send_err: Option<ProviderError>,
    tts_data: TtsGenerationData,
    aligned: Option<&'a mut Aligned<'b>>,
}

impl Segment for Streaming<'_, '_> {
    fn started_speaking(&mut self, at: f64) {
        first_frame(self.shared, self.acc, at);
    }

    async fn next(&mut self) -> Result<Next, DriverError> {
        if self.ended {
            return Ok(Next::End);
        }
        if let Some(e) = self.send_err.take() {
            return Err(DriverError::from(e));
        }
        let Streaming {
            sink, source, rx, text_out, transforms, words, input_done, send_err, ..
        } = self;
        let chunk = {
            let mut pending = Box::pin(source.next());
            loop {
                if *input_done {
                    // Nothing more to feed: wait on the provider alone.
                    break pending.await;
                }
                tokio::select! {
                    biased;
                    ev = rx.recv() => match ev {
                        Some(SegEvent::Chunk(more)) => {
                            // The transcript gets the delta just given to the synthesiser, in order.
                            capture(*text_out, &more.text).await;
                            extend(words, &more);
                            if let Err(e) = push_text(sink, transforms, &more.raw) {
                                *send_err = Some(e);
                                *input_done = true;
                            }
                        }
                        Some(SegEvent::End) | None => {
                            end_input(sink, transforms);
                            if let Some(out) = text_out {
                                out.flush();
                            }
                            *input_done = true;
                        }
                    },
                    got = &mut pending => break got,
                }
            }
        };
        let chunk = match chunk {
            Ok(Some(chunk)) => chunk,
            Ok(None) => {
                self.ended = true;
                if let Some(a) = self.aligned.as_deref_mut() {
                    a.ended(&self.words.text).await;
                }
                return Ok(Next::End);
            }
            Err(e) => return Err(DriverError::from(e)),
        };
        self.tts_data.note(self.shared, self.acc, &chunk);
        if let Some(a) = self.aligned.as_deref_mut() {
            a.chunk(&chunk).await;
        }
        // An `is_final` frame ends the segment's audio; a pooled connection never closes.
        if chunk.is_final && self.input_done {
            self.ended = true;
            if let Some(a) = self.aligned.as_deref_mut() {
                a.ended(&self.words.text).await;
            }
        }
        Ok(Next::Frame(chunk.frame))
    }

    fn text(&self) -> String {
        match self.aligned.as_deref() {
            Some(a) => a.words.clone(),
            None => self.words.text.clone(),
        }
    }
}

/// Run whole on a scratch pipeline. The duplex path journals this, so `TtsRequested{text}` is what
/// the TTS was given.
fn transformed(shared: &Arc<SessionShared>, raw: &str) -> String {
    if shared.tts_transforms.is_empty() {
        return raw.to_owned();
    }
    let mut p = Pipeline::new(&shared.tts_transforms);
    let mut out = p.push(raw);
    out.extend(p.flush());
    out.concat()
}

/// `_TTSGenerationData` and `_metrics_monitor_task`'s fold for one request. A failed request
/// leaves no `TTSMetrics`; only the first synthesised segment feeds the turn's TTS metrics.
struct TtsGenerationData {
    tts: lk_agents_abi::ProviderId,
    segment: usize,
    synthesis_started_at: f64,
    started_at: f64,
    metrics: TtsMetricState,
}

impl TtsGenerationData {
    fn new(tts: &dyn Tts, shared: &Arc<SessionShared>, segment: usize) -> Self {
        TtsGenerationData {
            tts: tts.id(),
            segment,
            synthesis_started_at: shared.time.wall_secs(),
            started_at: shared.time.wall_secs(),
            metrics: TtsMetricState::for_host(true),
        }
    }

    /// Whether audio came.
    fn produced(&self) -> bool {
        self.metrics.ttfb().is_some()
    }

    fn note(
        &mut self,
        shared: &SessionShared,
        acc: &mut TurnAccumulator,
        chunk: &SynthesizedAudio,
    ) {
        if !self.produced() {
            // The turn's, from `synthesis_started_at`; the record's is the request's.
            let turn = shared.time.wall_secs() - self.synthesis_started_at;
            let stats = crate::generation::TtsStats {
                ttfb: Some(turn),
                synthesis_started_at: Some(self.synthesis_started_at),
            };
            acc.record_tts(self.segment, stats);
        }
        let started_at = self.started_at;
        self.metrics.observe(chunk, || shared.time.wall_secs() - started_at);
    }

    fn finish(mut self, shared: &SessionShared, handle: &SpeechHandle, text: &str) {
        let now = shared.time.wall_secs();
        let context = Context {
            timestamp: now,
            label: self.tts.label.clone(),
            cancelled: handle.interrupted(),
            metadata: Some(self.tts.metadata()),
        };
        // No token counts, acquire time or connection reuse on this path.
        let record = self.metrics.finish(now, self.started_at, text, 0, 0, 0.0, false, context);
        if let Some(mut record) = record {
            record.speech_id = Some(handle.speech_id().to_owned());
            report_metrics(shared, handle, AgentMetrics::Tts(record));
        }
    }
}

/// Text and audio share a queue so aligned captions cannot overtake their audio. Each audio event
/// holds a permit until playback consumes it, backpressuring the synthesiser.
pub(super) enum SynthesisEvent {
    Text(String),
    TimedText(lk_agents_abi::TimedString),
    FlushText,
    Audio(Result<Next, DriverError>, String, tokio::sync::OwnedSemaphorePermit),
}

pub(super) struct SynthesisPlayout<'a> {
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
    ) -> Result<Heard, DriverError> {
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
                let event = SynthesisEvent::Audio(result, segment.text(), permit);
                if tx.send(event).is_err() || ended {
                    break;
                }
            }
        };
        let consume = async {
            let mut started_at = None;
            let result = if self.gate.wait(shared, handle).await {
                // Outputs can change while a speculative reply waits for its turn.
                let audio = self.gate.audio_enabled.then(|| shared.audio_out()).flatten();
                let text = self.gate.text_enabled.then(|| shared.text_out()).flatten();
                let leg = TextLeg::new(text.as_deref(), None, shared, handle.id());
                let mut buffered = SynthesizedSegment {
                    rx: &mut self.rx,
                    text: String::new(),
                    leg: &leg,
                    started_at: None,
                };
                let result =
                    perform_audio_forwarding(shared, handle, audio.as_deref(), &mut buffered).await;
                started_at = buffered.started_at;
                result
            } else {
                Ok(unheard())
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
    pub(super) async fn flush_text(&mut self, shared: &Arc<SessionShared>, handle: &SpeechHandle) {
        if !self.gate.wait(shared, handle).await {
            return;
        }
        let out = self.gate.text_enabled.then(|| shared.text_out()).flatten();
        let leg = TextLeg::new(out.as_deref(), None, shared, handle.id());
        while let Ok(event) = self.rx.try_recv() {
            leg.forward(event).await;
        }
    }
}

/// The buffered side of [`SynthesisPlayout`]: what the synthesiser produced, played in order.
struct SynthesizedSegment<'a, 'b> {
    rx: &'a mut mpsc::UnboundedReceiver<SynthesisEvent>,
    text: String,
    leg: &'a TextLeg<'b>,
    started_at: Option<f64>,
}

impl Segment for SynthesizedSegment<'_, '_> {
    async fn next(&mut self) -> Result<Next, DriverError> {
        while let Some(event) = self.rx.recv().await {
            match event {
                SynthesisEvent::Audio(result, text, _permit) => {
                    self.text = text;
                    return result;
                }
                text => self.leg.forward(text).await,
            }
        }
        Ok(Next::End)
    }
    fn text(&self) -> String {
        self.text.clone()
    }
    fn started_speaking(&mut self, at: f64) {
        self.started_at.get_or_insert(at);
    }
}

/// `use_aligned_transcript` less its audio-output half: `use_tts_aligned_transcript` is `True`
/// (not-given is off) and the TTS aligns.
pub(super) fn aligned_transcript(shared: &SessionShared, tts: &dyn Tts) -> bool {
    shared.opts.speech.use_tts_aligned_transcript == Some(true)
        && tts.capabilities().aligned_transcript
}

/// `_aligned_transcript_or_text`: the item is the stamped words, with timings; if nothing was
/// stamped by the audio's end, the model's text is forwarded whole.
pub(super) struct Aligned<'a> {
    pub(super) leg: Option<&'a TextLeg<'a>>,
    pub(super) words: String,
    /// A stamp arrived: the fallback is off for good.
    pub(super) stamped: bool,
    pub(super) closed: bool,
}

impl Aligned<'_> {
    async fn chunk(&mut self, chunk: &SynthesizedAudio) {
        for word in &chunk.timed_transcript {
            self.stamped = true;
            self.words.push_str(&word.text);
            if let Some(leg) = self.leg {
                leg.capture_timed(word).await;
            }
        }
    }

    /// Closes the leg's segment before the playout, so a synchroniser is not left with open text
    /// input.
    async fn ended(&mut self, fallback: &str) {
        if std::mem::replace(&mut self.closed, true) {
            return;
        }
        if !self.stamped {
            self.words.push_str(fallback);
            capture(self.leg, fallback).await;
        }
        if let Some(leg) = self.leg {
            leg.flush();
        }
    }
}
