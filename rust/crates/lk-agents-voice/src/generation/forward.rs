//! `forward_generation`: the voice half of a step, segment by segment, and the transcript leg
//! (`perform_text_forwarding`).

use super::llm::{SegEvent, extend};
use super::playout::{Next, Segment, first_frame, perform_audio_forwarding, silent, unheard};
use super::tts::{
    Aligned, Lost, SynthesisEvent, SynthesisPlayout, aligned_transcript, perform_tts_inference,
};
use crate::agent_activity::scheduling::SpeechId;
use crate::generation::{GenEvent, ReplyContext, ReplySource};
use crate::agent_session::SessionShared;
use crate::agent_session::DriverError;
use crate::generation::TurnAccumulator;
use crate::speech_handle::SpeechHandle;
use crate::trace::PlayoutStep;
use crate::transcription::Heard;
use lk_agents_abi::{AudioFrame, Tts};
use lk_agents_voice::io::TextOutput;
use lk_agents_voice::transcription::Visible;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use tokio::sync::mpsc;

/// A TTS that fails for good moves this to [`Voice::Silent`].
pub(super) enum Voice {
    Tts(Arc<dyn Tts>),
    /// Pre-synthesised (`say(audio=)`); the transcript is the words half's.
    Frames(mpsc::Receiver<AudioFrame>),
    /// No audio: no TTS, no sink, or the voice was lost.
    Silent,
}

/// Synthesis starts when permitted; playback is authorized separately. `None`: never authorized.
pub(super) async fn voice_leg(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    ctx: &ReplyContext,
    voice: Voice,
    rx: mpsc::UnboundedReceiver<SegEvent>,
) -> Result<Option<Spoken>, DriverError> {
    let mut gate = PlaybackGate::default();
    let synthesis = matches!(voice, Voice::Tts(_))
        && shared.audio_out_enabled_flag()
        && shared.audio_out().is_some();
    if synthesis {
        let early = matches!(ctx.source, ReplySource::Llm) && ctx.preemptive_tts;
        let ready = early
            || tokio::select! {
                biased;
                () = handle.wait_for_interrupted() => false,
                () = handle.wait_for_scheduled() => true,
                _ = handle.wait_for_authorization() => true,
            };
        if !ready {
            return Ok(None);
        }
    } else if !gate.wait(shared, handle).await {
        return Ok(None);
    }
    if handle.authorized() && !gate.wait(shared, handle).await {
        return Ok(None);
    }
    let mut acc = TurnAccumulator::new(shared.time.wall(), ctx.user_metrics.clone());
    let audio_enabled = shared.audio_out_enabled_flag();
    let text_enabled = shared.text_out_enabled_flag();
    // No sink: the caller's frames are left undrained, so `say(audio=)` does not wait on them.
    let has_sink = audio_enabled && shared.audio_out().is_some();
    let voice = if has_sink { voice } else { Voice::Silent };
    let mut pump = Pump {
        shared,
        handle,
        voice,
        gate: &mut gate,
        acc: &mut acc,
        text_enabled,
        audio_enabled,
        rx,
    };
    let interrupted = pump.run().await?;
    if !gate.wait(shared, handle).await && gate.at.is_none() {
        return Ok(None);
    }
    if let Some(authorized_at) = gate.at {
        acc.set_reply_started_at(authorized_at);
    }
    Ok(Some(Spoken { acc, interrupted }))
}

pub(super) struct Spoken {
    pub(super) acc: TurnAccumulator,
    pub(super) interrupted: bool,
}

/// Authorization belongs to a whole generation step, not to each synthesis segment.
#[derive(Default)]
pub(super) struct PlaybackGate {
    pub(super) at: Option<std::time::SystemTime>,
    pub(super) audio_enabled: bool,
    pub(super) text_enabled: bool,
}

impl PlaybackGate {
    pub(super) async fn wait(&mut self, shared: &SessionShared, handle: &SpeechHandle) -> bool {
        if self.at.is_some() {
            return !handle.interrupted();
        }
        let ready = tokio::select! {
            biased;
            () = handle.wait_for_interrupted() => false,
            _ = handle.wait_for_authorization() => true,
        };
        if ready {
            handle.clear_authorization();
            self.at = Some(shared.time.wall());
            self.audio_enabled = shared.audio_out_enabled_flag();
            self.text_enabled = shared.text_out_enabled_flag();
        }
        ready
    }
}

/// The `_next_segment` loop: each segment's audio finishes playing before the next one starts.
struct Pump<'a> {
    shared: &'a Arc<SessionShared>,
    handle: &'a Arc<SpeechHandle>,
    voice: Voice,
    gate: &'a mut PlaybackGate,
    acc: &'a mut TurnAccumulator,
    /// Both flags are fixed for the speech, as of the voice leg.
    text_enabled: bool,
    audio_enabled: bool,
    rx: mpsc::UnboundedReceiver<SegEvent>,
}

impl Pump<'_> {
    /// Ends when the words leg drops its sender.
    async fn run(&mut self) -> Result<bool, DriverError> {
        // TTS requests of a step are numbered in order across segments.
        let mut idx = 0usize;
        loop {
            if self.handle.interrupted() {
                return Ok(true);
            }
            // An `End` with nothing before it is an empty segment: skip, don't synthesise silence.
            let first = loop {
                match self.rx.recv().await {
                    // The words leg is done (or stopped by the interruption ending this step).
                    None => return Ok(self.handle.interrupted()),
                    Some(SegEvent::Chunk(v)) if !v.raw.is_empty() || !v.text.is_empty() => break v,
                    Some(SegEvent::Chunk(_) | SegEvent::End) => {}
                }
            };
            if self.handle.interrupted() {
                return Ok(true);
            }
            let heard = self.forward_generation(&mut idx, first).await?;
            self.acc.record(&heard.text);
            if self.handle.interrupted() {
                return Ok(true);
            }
        }
    }

    /// One segment's `forward_generation`: its synthesis (or frames), audio and transcript.
    async fn forward_generation(
        &mut self,
        idx: &mut usize,
        first: Visible,
    ) -> Result<Heard, DriverError> {
        let shared = self.shared;
        // A swap since authorization re-targets this segment, audio and words together.
        if self.gate.at.is_some() {
            self.audio_enabled = self.gate.audio_enabled;
            self.text_enabled = self.gate.text_enabled;
        }
        // Re-read at every segment's start: a swap moves the next segment (only a playing one is
        // interrupted), and a synchroniser swapped in between segments gets their words.
        let audio_out = self.audio_enabled.then(|| shared.audio_out()).flatten();
        let text = self.text_enabled.then(|| shared.text_out()).flatten();
        let synthesis = matches!(self.voice, Voice::Tts(_)) && audio_out.is_some();
        if !synthesis && !self.gate.wait(shared, self.handle).await {
            return Ok(unheard());
        }
        // One transcript leg per segment: its flush is once-only and the next segment gets its own.
        // Before authorization a synthesised segment's words queue behind its audio.
        let (tx, rx) = mpsc::unbounded_channel();
        let deferred = (synthesis && self.gate.at.is_none()).then_some(&tx);
        let leg = (synthesis || text.is_some())
            .then(|| TextLeg::new(text.as_deref(), deferred, shared, self.handle.id()));
        let text_out = leg.as_ref();
        // The aligned path: an audio output, the option on, and a TTS that aligns. The leg then
        // carries the synthesiser's stamped words *instead of* the deltas.
        let aligned_path = match (&self.voice, &audio_out) {
            (Voice::Tts(tts), Some(_)) => aligned_transcript(shared, tts.as_ref()),
            _ => false,
        };
        let mut aligned = aligned_path.then(|| Aligned {
            leg: text_out,
            words: String::new(),
            stamped: false,
            closed: false,
        });
        let deltas = if aligned_path { None } else { text_out };
        // The segment's first delta; every other one is forwarded where it is consumed.
        capture(deltas, &first.text).await;
        // With no audio output a forwarded delta starts the turn speaking. With no text leg either,
        // the agent never speaks; the words are still committed.
        if audio_out.is_none() && text_out.is_some() {
            first_text_started_speaking(shared, self.handle, self.acc);
        }
        let mut playback = SynthesisPlayout { tx: tx.clone(), rx, gate: &mut *self.gate };
        let heard = match (&mut self.voice, audio_out.as_deref()) {
            (Voice::Tts(tts), Some(_)) => {
                let tts = Arc::clone(tts);
                let rx = &mut self.rx;
                let synthesized = perform_tts_inference(
                    shared,
                    self.handle,
                    self.acc,
                    tts.as_ref(),
                    *idx,
                    first,
                    rx,
                    deltas,
                    aligned.as_mut(),
                    &mut playback,
                )
                .await;
                match synthesized {
                    Ok(heard) => {
                        *idx += 1;
                        heard
                    }
                    // The voice is gone for the rest of the step; the words are forwarded as text.
                    // A cut speech lost nothing.
                    Err(Lost { lost, opening }) => {
                        if lost {
                            self.voice = Voice::Silent;
                        }
                        match aligned.as_ref().filter(|a| a.stamped) {
                            // Stamps were already forwarded: the aligned stream never falls back,
                            // so the rest (never heard) is drained and the words are the stamps.
                            Some(a) => {
                                let words = a.words.clone();
                                drain_segment_text(rx, opening, None).await;
                                silent(words, true)
                            }
                            // `opening` holds what the failed synthesiser was fed.
                            None => {
                                silent(drain_segment_text(rx, opening, text_out).await.text, true)
                            }
                        }
                    }
                }
            }
            (Voice::Frames(frames), Some(out)) => {
                let words = drain_segment_text(&mut self.rx, first, text_out).await;
                let mut segment = Frames { shared, acc: self.acc, rx: frames };
                let mut heard =
                    perform_audio_forwarding(shared, self.handle, Some(out), &mut segment).await?;
                heard.text = words.text;
                heard
            }
            (Voice::Silent, Some(_)) | (_, None) => {
                silent(drain_segment_text(&mut self.rx, first, text_out).await.text, true)
            }
        };
        // Backstop flush for a segment whose loop broke before its `End`. Once-only.
        if let Some(out) = text_out {
            out.flush();
        }
        if synthesis {
            playback.flush_text(shared, self.handle).await;
        }
        Ok(heard)
    }
}

/// The first-frame fan-out still fires; no TTS metrics.
struct Frames<'a> {
    shared: &'a Arc<SessionShared>,
    acc: &'a mut TurnAccumulator,
    rx: &'a mut mpsc::Receiver<AudioFrame>,
}

impl Segment for Frames<'_> {
    fn started_speaking(&mut self, at: f64) {
        first_frame(self.shared, self.acc, at);
    }

    async fn next(&mut self) -> Result<Next, DriverError> {
        Ok(self.rx.recv().await.map_or(Next::End, Next::Frame))
    }

    fn text(&self) -> String {
        String::new()
    }
}

/// `perform_text_forwarding`'s sink for one segment: a once-only flush issued from all three end
/// points. `buffered` queues the words behind the audio synthesised before authorization.
pub(super) struct TextLeg<'a> {
    out: Option<&'a dyn TextOutput>,
    buffered: Option<&'a mpsc::UnboundedSender<SynthesisEvent>>,
    flushed: AtomicBool,
    shared: &'a Arc<SessionShared>,
    speech: SpeechId,
}

impl<'a> TextLeg<'a> {
    pub(super) fn new(
        out: Option<&'a dyn TextOutput>,
        buffered: Option<&'a mpsc::UnboundedSender<SynthesisEvent>>,
        shared: &'a Arc<SessionShared>,
        speech: SpeechId,
    ) -> Self {
        TextLeg { out, buffered, flushed: AtomicBool::new(false), shared, speech }
    }

    /// Errors are dropped: a text stream that will not open must not silence the agent.
    async fn capture(&self, delta: &str) {
        if !delta.is_empty() {
            self.forward(SynthesisEvent::Text(delta.to_owned())).await;
        }
    }

    /// Timings kept, so a synchroniser paces on `start_time`.
    pub(super) async fn capture_timed(&self, word: &lk_agents_abi::TimedString) {
        if !word.text.is_empty() {
            self.forward(SynthesisEvent::TimedText(word.clone())).await;
        }
    }

    pub(super) async fn forward(&self, event: SynthesisEvent) {
        if let Some(tx) = self.buffered {
            let _ = tx.send(event);
            return;
        }
        let Some(out) = self.out else { return };
        let text = match event {
            SynthesisEvent::Text(text) => {
                let _ = out.capture_text(&text).await;
                text
            }
            SynthesisEvent::TimedText(word) => {
                let _ = out.capture_timed(&word).await;
                word.text
            }
            SynthesisEvent::FlushText => return out.flush(),
            SynthesisEvent::Audio(..) => return,
        };
        self.shared.trace(PlayoutStep::TranscriptForwarded { speech: self.speech, text });
    }

    pub(super) fn flush(&self) {
        if !self.flushed.swap(true, std::sync::atomic::Ordering::Relaxed) {
            if let Some(tx) = self.buffered {
                let _ = tx.send(SynthesisEvent::FlushText);
            } else if let Some(out) = self.out {
                out.flush();
            }
        }
    }
}

pub(super) async fn capture(text_out: Option<&TextLeg<'_>>, delta: &str) {
    if let Some(out) = text_out {
        out.capture(delta).await;
    }
}

/// The text half of the first-frame fan-out, only with no audio output (with a TTS it would
/// report `speaking` too early). Idempotent.
fn first_text_started_speaking(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    acc: &mut TurnAccumulator,
) {
    if acc.started_speaking_at().is_some() || handle.interrupted() {
        return;
    }
    // `at: None`: the accumulator falls back to the wall clock.
    if acc.on_first_frame(None, shared.time.wall_secs()) {
        shared.report(GenEvent::FirstFrame { speech: handle.id() });
    }
}

/// `wait`: everything left (stopping at what is buffered made every chunk its own segment).
/// Otherwise only what is buffered: the duplex open must not wait on the model.
pub(super) async fn take_text(
    rx: &mut mpsc::UnboundedReceiver<SegEvent>,
    words: &mut Visible,
    text_out: Option<&TextLeg<'_>>,
    wait: bool,
) -> bool {
    loop {
        let ev = if wait {
            rx.recv().await
        } else {
            match rx.try_recv() {
                Ok(ev) => Some(ev),
                Err(mpsc::error::TryRecvError::Disconnected) => None,
                Err(mpsc::error::TryRecvError::Empty) => return false,
            }
        };
        let Some(SegEvent::Chunk(more)) = ev else {
            // Flushed **here**, not after the audio: the synchroniser rotates on
            // `playback_finished`, and text still open then is lost.
            if let Some(out) = text_out {
                out.flush();
            }
            return true;
        };
        capture(text_out, &more.text).await;
        extend(words, &more);
    }
}

pub(super) async fn drain_segment_text(
    rx: &mut mpsc::UnboundedReceiver<SegEvent>,
    mut words: Visible,
    text_out: Option<&TextLeg<'_>>,
) -> Visible {
    take_text(rx, &mut words, text_out, true).await;
    words
}
