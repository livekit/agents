//! `forward_generation`: the voice half of a step, segment by segment, and the transcript leg
//! (`perform_text_forwarding`).

use super::llm::{SegEvent, extend};
use super::playout::{
    Next, Segment, end_unvoiced, first_frame, perform_audio_forwarding, silent, unheard,
};
use super::tts::{Aligned, SynthesisEvent, SynthesisPlayout, perform_tts_inference};
use lk_agents_abi::voice::io::TextOutput;
use lk_agents_abi::{AudioFrame, Tts};
use lk_agents_voice::agent_activity::scheduling::SpeechId;
use lk_agents_voice::agent_session::DriverError;
use lk_agents_voice::agent_session::SessionShared;
use lk_agents_voice::agent_session::nodes::OutputNodes;
use lk_agents_voice::generation::TurnAccumulator;
use lk_agents_voice::generation::forward::{
    ForwardEffect, ForwardEvent, PreparationEffect, PreparationEvent, SynthesisPreparation,
};
use lk_agents_voice::generation::forward::{SegmentPlan, SegmentRoute, VoiceKind};
use lk_agents_voice::generation::forward::{TextLegEffect, TextLegEvent, TextLegState};
use lk_agents_voice::generation::tts::{LossEffect, LossEvent, SynthesisLoss};
use lk_agents_voice::generation::{GenEvent, ReplyContext, ReplySource};
use lk_agents_voice::speech_handle::SpeechHandle;
use lk_agents_voice::trace::PlayoutStep;
use lk_agents_voice::transcription::Heard;
use lk_agents_voice::transcription::Visible;
use std::sync::Arc;
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
    // `tts_node` sets it just before `stream()` snapshots it, for this synthesis only.
    if let Voice::Tts(tts) = &voice {
        tts.set_expressive(ctx.request.expressive.is_some());
    }
    let mut preparation = SynthesisPreparation::default();
    let mut effect = preparation.handle(PreparationEvent::Begin {
        has_tts: matches!(voice, Voice::Tts(_)),
        audio_enabled: shared.audio_enabled(),
        has_audio_output: shared.audio_output().is_some(),
        llm_reply: matches!(ctx.source, ReplySource::Llm),
        preemptive_tts: ctx.preemptive_tts,
        authorized: handle.authorized(),
    });
    loop {
        effect = match effect {
            PreparationEffect::WaitReady => {
                let ready = tokio::select! {
                    biased;
                    () = handle.wait_for_interrupted() => false,
                    () = handle.wait_for_scheduled() => true,
                    _ = handle.wait_for_authorization() => true,
                };
                preparation
                    .handle(PreparationEvent::Ready { ready, authorized: handle.authorized() })
            }
            PreparationEffect::WaitPlayback => {
                let accepted = gate.wait(shared, handle).await;
                preparation.handle(PreparationEvent::Playback {
                    accepted,
                    authorized: handle.authorized(),
                })
            }
            PreparationEffect::Start => break,
            PreparationEffect::Refuse => return Ok(None),
        };
    }
    let mut acc = TurnAccumulator::new(shared.time.wall(), ctx.user_metrics.clone());
    let audio_enabled = shared.audio_enabled();
    let text_enabled = shared.transcription_enabled();
    let mut pump = Pump {
        shared,
        handle,
        nodes: &ctx.providers.output_nodes,
        voice,
        aligned_transcript: ctx.speech.use_tts_aligned_transcript == Some(true),
        gate: &mut gate,
        acc: &mut acc,
        text_enabled,
        audio_enabled,
        rx,
    };
    let interrupted = pump.run().await?;
    if !gate.wait(shared, handle).await && gate.state.at.is_none() {
        return Ok(None);
    }
    if let Some(authorized_at) = gate.state.at {
        acc.set_reply_started_at(authorized_at);
    }
    Ok(Some(Spoken { acc, interrupted }))
}

pub(super) struct Spoken {
    pub(super) acc: TurnAccumulator,
    pub(super) interrupted: bool,
}

/// Waits for the authorization effect without owning the generation's playback policy.
#[derive(Default)]
pub(super) struct PlaybackGate {
    pub(super) state: lk_agents_voice::generation::forward::PlaybackGate,
}

impl PlaybackGate {
    pub(super) async fn wait(&mut self, shared: &SessionShared, handle: &SpeechHandle) -> bool {
        use lk_agents_voice::generation::forward::{PlaybackEffect, PlaybackEvent};
        let mut effect =
            self.state.handle(PlaybackEvent::Observe { interrupted: handle.interrupted() });
        if matches!(effect, PlaybackEffect::Wait) {
            let event = tokio::select! {
                biased;
                () = handle.wait_for_interrupted() => PlaybackEvent::Interrupted,
                _ = handle.wait_for_authorization() => PlaybackEvent::Authorized {
                    at: shared.time.wall(),
                    audio_enabled: shared.audio_enabled(),
                    text_enabled: shared.transcription_enabled(),
                },
            };
            effect = self.state.handle(event);
        }
        match effect {
            PlaybackEffect::Play { consume_authorization } => {
                if consume_authorization {
                    handle.clear_authorization();
                }
                true
            }
            PlaybackEffect::Refuse => false,
            PlaybackEffect::Wait => unreachable!("a playback observation has arrived"),
        }
    }
}

/// The `_next_segment` loop: each segment's audio finishes playing before the next one starts.
struct Pump<'a> {
    shared: &'a Arc<SessionShared>,
    handle: &'a Arc<SpeechHandle>,
    /// The reply's agent's `transcription_node`, opened over each segment's transcript leg.
    nodes: &'a OutputNodes,
    voice: Voice,
    /// `use_tts_aligned_transcript` is `True` (not-given is off).
    aligned_transcript: bool,
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
        use lk_agents_voice::generation::forward::{SegmentEffect, SegmentInput};
        let mut idx = 0usize;
        let mut effect =
            self.acc.on_segment(SegmentInput::Ready { interrupted: self.handle.interrupted() });
        loop {
            let input = match effect {
                SegmentEffect::Read => match self.rx.recv().await {
                    Some(SegEvent::Chunk(words)) => SegmentInput::Chunk {
                        words,
                        caller_frames: matches!(self.voice, Voice::Frames(_)),
                        interrupted: self.handle.interrupted(),
                    },
                    Some(SegEvent::End) => SegmentInput::Boundary,
                    None => SegmentInput::Closed { interrupted: self.handle.interrupted() },
                },
                SegmentEffect::Forward(first) => {
                    let heard = self.forward_generation(&mut idx, first).await?;
                    SegmentInput::Played { heard, interrupted: self.handle.interrupted() }
                }
                SegmentEffect::End { interrupted } => return Ok(interrupted),
            };
            effect = self.acc.on_segment(input);
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
        if self.gate.state.at.is_some() {
            self.audio_enabled = self.gate.state.audio_enabled;
            self.text_enabled = self.gate.state.text_enabled;
        }
        let Some(mut binding) = super::output::OutputBinding::acquire(shared, self.handle).await?
        else {
            return Ok(unheard());
        };
        let audio_out = binding.audio();
        let text = binding.text();
        let text = super::output::text_output(self.nodes, text);
        binding.retain_text_processing(text.clone());
        let (kind, aligned_provider) = match &self.voice {
            Voice::Tts(tts) => (VoiceKind::Synthesis, tts.capabilities().aligned_transcript),
            Voice::Frames(_) => (VoiceKind::CallerFrames, false),
            Voice::Silent => (VoiceKind::Silent, false),
        };
        let plan = SegmentPlan::new(
            kind,
            audio_out.is_some(),
            text.is_some(),
            self.aligned_transcript,
            aligned_provider,
        );
        let synthesis = matches!(plan.route, SegmentRoute::Synthesis);
        if !synthesis && (!self.gate.wait(shared, self.handle).await || !binding.activate().await?)
        {
            return Ok(unheard());
        }
        // One transcript leg per segment: its flush is once-only and the next segment gets its own.
        let (tx, rx) = mpsc::unbounded_channel();
        let deferred = plan.transcript.filter(|plan| plan.deferred).map(|_| &tx);
        let leg =
            plan.transcript.map(|_| TextLeg::new(text.clone(), deferred, shared, self.handle.id()));
        let text_out = leg.as_ref();
        // The aligned path: an audio output, the option on, and a TTS that aligns. The leg then
        // carries the synthesiser's stamped words instead of the deltas.
        let aligned_path = plan.transcript.is_some_and(|plan| plan.aligned);
        let mut aligned =
            aligned_path.then(|| Aligned { leg: text_out, state: Default::default() });
        let deltas = if aligned_path { None } else { text_out };
        // The segment's first delta; every other one is forwarded where it is consumed.
        capture(deltas, &first.text).await;
        // With no audio output the first words start the turn speaking, text output or not (the
        // reference's `first_text_fut`): a session with neither still speaks, then listens.
        first_text_started_speaking(
            shared,
            self.handle,
            self.acc,
            ForwardEvent::SegmentOpened {
                has_audio_output: audio_out.is_some(),
                interrupted: self.handle.interrupted(),
            },
        );
        let mut playback = SynthesisPlayout {
            tx: tx.clone(),
            rx,
            gate: &mut *self.gate,
            text: text.clone(),
            binding: &mut binding,
        };
        let heard = match (plan.route, &mut self.voice, audio_out.as_deref()) {
            (SegmentRoute::Synthesis, Voice::Tts(tts), Some(_)) => {
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
                    Err(lost) => {
                        let mut state = SynthesisLoss::default();
                        let mut effects: std::collections::VecDeque<_> = state
                            .handle(LossEvent::Failed {
                                lost,
                                aligned: aligned.as_ref().map(|a| a.state.clone()),
                            })
                            .into();
                        loop {
                            match effects
                                .pop_front()
                                .expect("synthesis loss settles after draining")
                            {
                                LossEffect::Fail(error) => return Err(error),
                                LossEffect::ReleaseVoice => self.voice = Voice::Silent,
                                LossEffect::CutShort => self.acc.cut_short(),
                                LossEffect::Capture(text) => capture(text_out, &text).await,
                                LossEffect::Drain { opening, forward_text } => {
                                    let text = if forward_text { text_out } else { None };
                                    let drained = drain_segment_text(rx, opening, text).await;
                                    effects.extend(state.handle(LossEvent::Drained(drained)));
                                }
                                LossEffect::Heard(heard) => break heard,
                            }
                        }
                    }
                }
            }
            (SegmentRoute::CallerFrames, Voice::Frames(frames), Some(out)) => {
                let words = drain_segment_text(&mut self.rx, first, text_out).await;
                let mut segment = Frames { shared, acc: self.acc, rx: frames };
                let mut heard =
                    perform_audio_forwarding(shared, self.handle, Some(out), &mut segment)
                        .await
                        .map_err(|failed| failed.error)?;
                heard.text = words.text;
                heard
            }
            (SegmentRoute::Text, _, _) => {
                silent(drain_segment_text(&mut self.rx, first, text_out).await.text, true)
            }
            _ => unreachable!("segment plan matches its bound ports"),
        };
        // Backstop flush for a segment whose loop broke before its `End`. Once-only.
        if let Some(out) = text_out {
            out.flush();
        }
        // Both legs end every segment, one whose words went down the transcript leg with no
        // audio (none synthesised, a voice lost before its first frame, or its audio switched
        // off) included.
        let leg_state = text_out.map(|leg| *leg.state());
        if plan.end_unvoiced(&heard, leg_state) {
            end_unvoiced(
                shared,
                self.handle.id(),
                audio_out.as_deref().expect("bound audio output"),
            );
        }
        if synthesis {
            playback.flush_text(shared, self.handle).await?;
        }
        if let Some(text) = text {
            let result = finish_text(text.as_ref(), self.handle, false).await;
            binding.text_drained();
            result?;
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
    out: Option<Arc<dyn TextOutput>>,
    buffered: Option<&'a mpsc::UnboundedSender<SynthesisEvent>>,
    state: std::sync::Mutex<TextLegState>,
    shared: &'a Arc<SessionShared>,
    speech: SpeechId,
}

impl<'a> TextLeg<'a> {
    pub(super) fn new(
        out: Option<Arc<dyn TextOutput>>,
        buffered: Option<&'a mpsc::UnboundedSender<SynthesisEvent>>,
        shared: &'a Arc<SessionShared>,
        speech: SpeechId,
    ) -> Self {
        TextLeg { out, buffered, state: Default::default(), shared, speech }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, TextLegState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn handle(&self, event: TextLegEvent) -> Option<TextLegEffect> {
        self.state().handle(event)
    }

    /// Errors are dropped: a text stream that will not open must not silence the agent.
    pub(super) async fn capture(&self, delta: &str) {
        if matches!(
            self.handle(TextLegEvent::Capture { empty: delta.is_empty() }),
            Some(TextLegEffect::Forward)
        ) {
            self.forward(SynthesisEvent::Text(delta.to_owned())).await;
        }
    }

    /// Timings kept, so a synchroniser paces on `start_time`.
    pub(super) async fn capture_timed(&self, word: &lk_agents_abi::TimedString) {
        if matches!(
            self.handle(TextLegEvent::Capture { empty: word.text.is_empty() }),
            Some(TextLegEffect::Forward)
        ) {
            self.forward(SynthesisEvent::TimedText(word.clone())).await;
        }
    }

    pub(super) async fn forward(&self, event: SynthesisEvent) {
        if matches!(event, SynthesisEvent::Text(_) | SynthesisEvent::TimedText(_)) {
            self.handle(TextLegEvent::Forwarded);
        }
        if let Some(tx) = self.buffered {
            let _ = tx.send(event);
            return;
        }
        let Some(out) = self.out.as_deref() else { return };
        let (text, start_time) = match event {
            SynthesisEvent::Text(text) => {
                let _ = out.capture_text(&text).await;
                (text, None)
            }
            SynthesisEvent::TimedText(word) => {
                let _ = out.capture_timed(&word).await;
                (word.text, word.start_time)
            }
            SynthesisEvent::FlushText => return out.flush(),
            SynthesisEvent::Audio(..) => return,
        };
        let speech = self.speech;
        self.shared.trace(PlayoutStep::TranscriptForwarded { speech, text, start_time });
    }

    pub(super) fn flush(&self) {
        if matches!(self.handle(TextLegEvent::Flush), Some(TextLegEffect::Flush)) {
            if let Some(tx) = self.buffered {
                let _ = tx.send(SynthesisEvent::FlushText);
            } else if let Some(out) = self.out.as_deref() {
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
pub(super) fn first_text_started_speaking(
    shared: &Arc<SessionShared>,
    handle: &Arc<SpeechHandle>,
    acc: &mut TurnAccumulator,
    event: ForwardEvent<'_>,
) {
    match acc.on_forwarded(event, shared.time.wall_secs()) {
        Some(ForwardEffect::StartedSpeaking) => {
            shared.report(GenEvent::FirstFrame { speech: handle.id() })
        }
        None => {}
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

/// A segment owns its text node until output delivery or cancellation cleanup settles.
pub(super) async fn finish_text(
    text: &dyn TextOutput,
    handle: &SpeechHandle,
    cancel: bool,
) -> Result<(), DriverError> {
    if cancel {
        text.cancel_flush();
        return text.wait_for_flush().await.map_err(DriverError::TextOutput);
    }
    let mut completion = text.wait_for_flush();
    let mut interrupted = Box::pin(handle.wait_for_interrupted());
    let outcome = std::future::poll_fn(|cx| {
        if interrupted.as_mut().poll(cx).is_ready() {
            return std::task::Poll::Ready(None);
        }
        completion.as_mut().poll(cx).map(Some)
    })
    .await;
    match outcome {
        Some(result) => result.map_err(DriverError::TextOutput),
        None => {
            text.cancel_flush();
            completion.await.map_err(DriverError::TextOutput)
        }
    }
}
