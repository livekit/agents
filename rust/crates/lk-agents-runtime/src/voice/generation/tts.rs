//! `perform_tts_inference`: one segment's duplex synthesis, retried, and its buffered playout.

use super::forward::{PlaybackGate, TextLeg, capture, drain_segment_text, take_text};
use super::llm::{Retried, SegEvent, report_metrics, retry};
use super::playout::{Next, Segment, SourceFailed, first_frame, perform_audio_forwarding, unheard};
use crate::voice::session::SessionShared;
use crate::voice::speech_handle::SpeechHandle;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{SynthesizedAudio, Tts};
use lk_agents_voice::agent_session::DriverError;
pub(super) use lk_agents_voice::generation::tts::Lost;
use lk_agents_voice::generation::tts::{
    AlignedTranscript, MAX_BUFFERED_TTS_FRAMES, Read, StreamingTts, SynthesisMetrics, TextEffect,
    TranscriptEffect,
};
use lk_agents_voice::generation::tts::{AttemptFailure, RetryOutcome, SynthesisAttempt};
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
    rx: &mut mpsc::UnboundedReceiver<SegEvent>,
    text_out: Option<&TextLeg<'_>>,
    mut aligned: Option<&mut Aligned<'_>>,
    playback: &mut SynthesisPlayout<'_>,
) -> Result<Heard, Box<Lost>> {
    let mut state =
        SynthesisAttempt::new(&shared.conn.tts.unwrap_or_default(), first, shared.time.now());
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
        let (error, produced) = match tts.stream(expressive).await {
            Err(error) => (DriverError::Provider(error), false),
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
                    tts_data,
                    aligned,
                };
                match synthesize(shared, handle, streaming, playback).await {
                    (Ok(heard), _) => return Ok(heard),
                    (Err(failed), done) => {
                        state.stream_failed(failed.words, done, failed.synchronized, failed.heard);
                        (failed.error, failed.produced)
                    }
                }
            }
        };
        let error = match state.failed(error, produced) {
            AttemptFailure::Retry(error) => error,
            AttemptFailure::Stop(lost) => return Err(Box::new(lost)),
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
            return Err(Box::new(lost));
        }
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
                report_metrics(shared, handle, record);
            }
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
    rx: &'a mut mpsc::UnboundedReceiver<SegEvent>,
    text_out: Option<&'a TextLeg<'a>>,
    transforms: Option<Pipeline>,
    state: StreamingTts,
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
        let Streaming { sink, source, rx, text_out, transforms, state, .. } = self;
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
                            Some(SegEvent::Chunk(chunk)) => Some(chunk),
                            Some(SegEvent::End) | None => None,
                        };
                        let mut effects: std::collections::VecDeque<_> = state.text(chunk).into();
                        while let Some(effect) = effects.pop_front() {
                            match effect {
                                TextEffect::Capture(chunk) => {
                                    capture(*text_out, &chunk.text).await;
                                    effects.push_front(state.captured(chunk));
                                }
                                TextEffect::Push(raw) => {
                                    if let Err(error) = push_text(sink, transforms, &raw) {
                                        state.push_failed(error);
                                    }
                                }
                                TextEffect::EndInput => end_input(sink, transforms),
                                TextEffect::FlushText => {
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

/// Text and audio share a queue so aligned captions cannot overtake their audio. Each audio event
/// holds a permit until playback consumes it, backpressuring the synthesiser.
pub(super) enum SynthesisEvent {
    Text(String),
    TimedText(lk_agents_abi::TranscriptChunk),
    FlushText,
    Audio(Result<Next, DriverError>, String, tokio::sync::OwnedSemaphorePermit),
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
                let event = SynthesisEvent::Audio(result, segment.text(), permit);
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
                        text: String::new(),
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

    async fn apply(&self, effect: TranscriptEffect) {
        match effect {
            TranscriptEffect::CaptureTimed(word) => {
                if let Some(leg) = self.leg {
                    leg.capture_timed(&word).await;
                }
            }
            TranscriptEffect::Capture(text) => capture(self.leg, &text).await,
            TranscriptEffect::WarnUnaligned => log::warn!(
                target: "livekit.agents",
                "no aligned transcript was returned from tts, forwarding the generated text \
                 without word timings"
            ),
            TranscriptEffect::Flush => {
                if let Some(leg) = self.leg {
                    leg.flush();
                }
            }
        }
    }
}
