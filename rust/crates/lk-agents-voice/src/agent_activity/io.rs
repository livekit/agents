//! The activity's IO: the STT and VAD legs (reopened by a handoff or `update_models`), the STT's
//! keyterms, audio and video input, output enablement, and the recorder tap.

use crate::agent_session::cycle::{LegKind, TurnTimer};
use crate::agent_session::cycle::Cycle;
use crate::agent_session::{Driver, IoLeg};
use crate::audio::{LegEnd, LegId, LegOp, Ticket, on_leg_ended, spawn_drain, tagged};
use crate::keyterm_detection::{
    KeytermEffect, KeytermIn, KeytermSetup, KeytermSlot, detection_pass, parse_record_keyterms,
};
use crate::room_io::{LegIntent, LegOpened, Opened, PendingOpen, RECORDER_FLUSH};
use crate::trace::{
    CoreStep, DetectorLeg, DetectorStep, InputLeg, InputStep, OutputLeg, PlayoutStep, Step, TurnStep,
};
use lk_agents_abi::metrics::base::LlmMetrics;
use lk_agents_abi::{ChatItem, ChatMessage, Llm, Maybe, Role, SpawnerExt, Ts};
use lk_agents_voice::io::{AudioInput, AudioOutput};
use std::sync::Arc;

impl Driver {
    /// The one door every leg mutation goes through; a new source is swapped by the performer.
    pub(crate) fn apply_leg_op(&mut self, op: LegOp, cycle: &mut Cycle) -> Option<Ticket> {
        let source = matches!(op, LegOp::SetSource(_));
        let ticket = self.input.apply_leg_op(op);
        if source {
            cycle.machine().push(crate::agent_session::cycle::Effect::Audio(
                crate::agent_session::cycle::AudioOp::SetInput,
            ));
        }
        ticket
    }

    /// `Given` reopens off the loop; the new sink is installed before the old ends, so the pump
    /// never reads an empty row. `Given(None)` closes the leg; `NotGiven` leaves it.
    pub(crate) fn reopen_legs(
        &mut self,
        stt: Maybe<Option<Arc<dyn lk_agents_abi::Stt>>>,
        vad: Maybe<Option<Arc<dyn lk_agents_abi::Vad>>>,
        cycle: &mut Cycle,
    ) -> LegIntent {
        let given = stt.is_given() || vad.is_given();
        // Presence facts are the configured providers (`self._stt`/`self._vad`), not the live
        // streams: a dead stream is still a recogniser.
        let mut intent =
            LegIntent { stt: self.session.stt().is_some(), vad: self.session.vad().is_some() };
        if let Maybe::Given(stt) = stt {
            match stt {
                // A different STT: reopened; the row follows when the stream lands.
                Some(stt) => {
                    let same_stt = self.session.stt().is_some_and(|old| Arc::ptr_eq(old, &stt));
                    self.session.set_stt(Some(Arc::clone(&stt)));
                    self.input.set_stt_open_same(same_stt);
                    let open = self.request_open(PendingOpen::Stt(stt), LegKind::Stt, cycle);
                    self.input.expect_open(LegKind::Stt, open);
                    intent.stt = true;
                }
                // `Agent(stt=None)`: the leg closes and `stt_enabled` follows via `set_legs`.
                None => {
                    self.session.set_stt(None);
                    self.input.expect_open(LegKind::Stt, 0);
                    self.apply_leg_op(LegOp::Clear(LegId::Stt), cycle);
                    intent.stt = false;
                }
            }
        }
        if let Maybe::Given(vad) = vad {
            match vad {
                Some(vad) => {
                    self.session.set_vad(Some(Arc::clone(&vad)));
                    let open = self.request_open(PendingOpen::Vad(vad), LegKind::Vad, cycle);
                    self.input.expect_open(LegKind::Vad, open);
                    intent.vad = true;
                }
                None => {
                    self.session.set_vad(None);
                    self.input.expect_open(LegKind::Vad, 0);
                    self.apply_leg_op(LegOp::Clear(LegId::Vad), cycle);
                    intent.vad = false;
                }
            }
        }
        if given {
            // `decide_commit` reads `stt_enabled`; a stale flag would wait for a transcript that
            // never comes.
            self.activity.audio_recognition_mut().set_legs(intent.stt, intent.vad);
        }
        intent
    }

    fn request_open(&mut self, pending: PendingOpen, leg: LegKind, cycle: &mut Cycle) -> u64 {
        let open = self.input.request_open(pending);
        cycle.machine().open_leg(leg, open);
        open
    }

    /// A superseded open is closed unused (`LegOpenDiscarded`); a refused one leaves the old row
    /// running.
    pub(crate) fn on_leg_opened(&mut self, now: Ts, opened: LegOpened, cycle: &mut Cycle) {
        let LegOpened { open, leg, outcome } = opened;
        let wanted = self.input.expected_open(leg);
        let live = open == wanted && open != 0 && !self.session.closing_past_edge();
        match (outcome, live) {
            (Err(_), _) => {
                cycle.trace(CoreStep::PortFailure(match leg {
                    LegKind::Stt => "the new agent's STT could not be opened",
                    LegKind::Vad => "the new agent's VAD could not be opened",
                }));
            }
            (Ok(stream), false) => {
                cycle.trace(InputStep::LegOpenDiscarded { leg: leg.label() });
                self.discard_open(stream);
            }
            (Ok(Opened::Stt(sink, source)), true) => {
                if let Some(stt) = self.session.stt() {
                    stt.prewarm();
                }
                // A different recogniser is told the full list.
                let same_stt = self.input.stt_open_same();
                self.keyterms_on_stt_swap(now, same_stt, cycle);
                // A fresh detector per open; the replaced stream keeps its own for its remaining
                // items.
                let speaker = self.input.new_speaker_filter();
                let ticket = self
                    .apply_leg_op(LegOp::InstallStt { sink: Arc::from(sink), speaker }, cycle)
                    .expect("InstallStt returns its ticket");
                spawn_drain(
                    &self.shared.spawner,
                    self.shared.time.clone(),
                    self.shared.closing.subscribe(),
                    tagged(ticket, source),
                    self.stt_tx.clone(),
                    Arc::clone(self.input.legs()),
                    ticket,
                    self.input.leg_ended().clone(),
                );
                cycle.trace(InputStep::LegOpened { leg: LegId::Stt.label() });
            }
            (Ok(Opened::Vad(sink, source)), true) => {
                let ticket = self
                    .apply_leg_op(LegOp::Install { leg: LegId::Vad, sink: Arc::from(sink) }, cycle)
                    .expect("Install returns its ticket");
                spawn_drain(
                    &self.shared.spawner,
                    self.shared.time.clone(),
                    self.shared.closing.subscribe(),
                    source,
                    self.vad_tx.clone(),
                    Arc::clone(self.input.legs()),
                    ticket,
                    self.input.leg_ended().clone(),
                );
                cycle.trace(InputStep::LegOpened { leg: LegId::Vad.label() });
            }
        }
    }

    /// A stream nobody wants: closed on a bounded task (dropping a host stream doesn't cancel it).
    fn discard_open(&self, stream: Opened) {
        let time = self.shared.time.clone();
        let deadline = time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        match stream {
            Opened::Stt(_sink, source) => {
                lk_agents_abi::SpawnerExt::spawn_unboxed(&*self.shared.spawner, async move {
                    let _ = source.close(deadline).await;
                });
            }
            Opened::Vad(_sink, source) => {
                lk_agents_abi::SpawnerExt::spawn_unboxed(&*self.shared.spawner, async move {
                    let _ = source.close(deadline).await;
                });
            }
        }
    }

    /// Only a given leg is reopened.
    pub(crate) fn update_models(
        &mut self,
        stt: Maybe<Option<Arc<dyn lk_agents_abi::Stt>>>,
        vad: Maybe<Option<Arc<dyn lk_agents_abi::Vad>>>,
        llm: Maybe<Option<Arc<dyn Llm>>>,
        tts: Maybe<Option<Arc<dyn lk_agents_abi::Tts>>>,
        cycle: &mut Cycle,
    ) {
        self.session.apply_providers(llm, tts);
        let (stt_given, vad_given) =
            (matches!(stt, Maybe::Given(_)), matches!(vad, Maybe::Given(_)));
        let (stt_reopened, vad_reopened) =
            (matches!(stt, Maybe::Given(Some(_))), matches!(vad, Maybe::Given(Some(_))));
        let _ = self.reopen_legs(stt, vad, cycle);
        cycle.trace(InputStep::ModelsUpdated { stt_given, vad_given, stt_reopened, vad_reopened });
    }

    /// A leg's drain ended. A failed STT is terminal (`STTError{recoverable: False}`); a VAD
    /// failure is only journalled.
    pub(crate) fn on_leg_ended(&mut self, now: Ts, mut end: LegEnd, cycle: &mut Cycle) {
        let error = end.error.take();
        let leg = end.leg;
        // Every item the drain sent is ahead of its end in the queue.
        if leg == LegId::Stt.label() {
            self.input.legs().retire_speaker_filter(end.open);
        }
        on_leg_ended(end, cycle);
        if let Some(error) = error
            && leg == LegId::Stt.label()
        {
            self.report_error(now, crate::events::ErrorSource::Stt, false, error, None, cycle);
        }
    }

    /// Detaching under a speaking user ends the speech like the reference's mute; a re-attach
    /// forgets the acoustic anchor and reopens the STT leg on the same recogniser.
    pub(crate) fn set_audio_input(
        &mut self,
        now: Ts,
        input: Option<Arc<dyn AudioInput>>,
        cycle: &mut Cycle,
    ) {
        let attached = input.is_some();
        cycle.trace(InputStep::AudioInputSet { attached });
        self.input.replace_audio_in(input.clone());
        self.apply_leg_op(LegOp::SetSource(input), cycle);
        if attached {
            self.input.forget_first_input();
            self.activity.audio_recognition_mut().set_input_started_at(None);
            if let Some(stt) = self.session.stt().cloned() {
                let _ = self.reopen_legs(Maybe::Given(Some(stt)), Maybe::NotGiven, cycle);
            }
        } else {
            self.host_end_of_speech(now, cycle);
        }
    }

    /// The host ended the user's speech (mic detached or muted: `on_end_of_speech(None)`).
    pub(crate) fn host_end_of_speech(&mut self, now: Ts, cycle: &mut Cycle) {
        if !self.activity.audio_recognition().user_speaking() {
            return;
        }
        let eff = self.activity.audio_recognition_mut().on_host_end_of_speech(now);
        cycle.machine().append(eff);
        self.start_false_interruption_timer(now, cycle);
    }

    /// The leg stays installed; the host object is told.
    pub(crate) fn set_io_enabled(&mut self, now: Ts, leg: IoLeg, enabled: bool, cycle: &mut Cycle) {
        let changed = match leg {
            IoLeg::AudioInput => self.input.set_audio_enabled(enabled),
            IoLeg::AudioOutput => {
                let changed = self.shared.set_audio_out_enabled(enabled);
                if changed && let Some(out) = self.shared.audio_out() {
                    if enabled { out.on_attached() } else { out.on_detached() }
                }
                changed
            }
            IoLeg::Transcription => {
                let changed = self.shared.set_text_out_enabled(enabled);
                if changed && let Some(out) = self.shared.text_out() {
                    if enabled { out.on_attached() } else { out.on_detached() }
                }
                changed
            }
        };
        if !changed {
            return;
        }
        cycle.trace(match leg {
            IoLeg::AudioInput => {
                Step::Input(InputStep::InputEnabled { leg: InputLeg::Audio, enabled })
            }
            IoLeg::AudioOutput => {
                Step::Input(InputStep::OutputEnabled { leg: OutputLeg::Audio, enabled })
            }
            IoLeg::Transcription => {
                Step::Input(InputStep::OutputEnabled { leg: OutputLeg::Transcription, enabled })
            }
        });
        match leg {
            // "End user speaking state when audio is disabled" (`_on_audio_enabled_changed`).
            IoLeg::AudioInput if !enabled => self.host_end_of_speech(now, cycle),
            IoLeg::AudioInput => {}
            // The pause machine's "audio enabled" fact moved with the flag.
            IoLeg::AudioOutput => self.refresh_pause_facts(now, cycle),
            IoLeg::Transcription => {}
        }
    }

    /// `_commit_user_turn`'s "flush the stt by pushing silence", `ceil(duration / 0.2)` frames.
    /// Silent before any audio.
    pub(crate) fn flush_stt_with_silence(&self, seconds: f64) {
        if let Some(frames) = self.input.push_stt_silence(seconds) {
            // `&self` from a command handler with no cycle.
            self.journal(Step::Turn(TurnStep::SttFlushed { frames }));
        }
    }

    /// A second attach replaces the first.
    pub(crate) fn attach_recorder(
        &mut self,
        now: Ts,
        recorder: Arc<dyn lk_agents_media::recorder::AudioRecorder>,
        cycle: &mut Cycle,
    ) {
        let tap = crate::recorder_io::RecorderTap::new(recorder, self.shared.time.clock());
        *self.shared.recorder.write().unwrap_or_else(|p| p.into_inner()) = Some(tap);
        self.observe_recorder(None);
        cycle.machine().arm(TurnTimer::RecorderFlush, now.saturating_add(RECORDER_FLUSH));
    }

    /// One last flush, observer off, timer cancelled.
    pub(crate) fn detach_recorder(&mut self, now: Ts, cycle: &mut Cycle) {
        if self.shared.recorder_tap().is_none() {
            return;
        }
        self.flush_recorder(now, false, cycle);
        if let Some(observer) =
            self.shared.recorder_observer.lock().unwrap_or_else(|p| p.into_inner()).take()
            && let Some(out) = self.shared.audio_out()
        {
            out.tracker().unobserve(&observer);
        }
        *self.shared.recorder.write().unwrap_or_else(|p| p.into_inner()) = None;
        cycle.machine().cancel(TurnTimer::RecorderFlush);
        cycle.trace(PlayoutStep::RecorderDetached);
    }

    /// Off `previous` first.
    pub(crate) fn observe_recorder(&self, previous: Option<&dyn AudioOutput>) {
        let Some(tap) = self.shared.recorder_tap() else { return };
        let mut slot = self.shared.recorder_observer.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(old) = slot.take()
            && let Some(prev) = previous
        {
            prev.tracker().unobserve(&old);
        }
        if let Some(out) = self.shared.audio_out() {
            let observer: Arc<dyn crate::io::PlaybackObserver> =
                Arc::new(crate::recorder_io::RecorderPlaybackObserver(tap));
            out.tracker().observe(observer.clone());
            *slot = Some(observer);
        }
    }

    pub(crate) fn flush_recorder(&mut self, now: Ts, rearm: bool, cycle: &mut Cycle) {
        let Some(tap) = self.shared.recorder_tap() else { return };
        let ok = tap.recorder.flush(tap.now()).is_ok();
        cycle.trace(PlayoutStep::RecorderFlushed { ok });
        if rearm && !self.session.closing_past_edge() {
            cycle.machine().arm(TurnTimer::RecorderFlush, now.saturating_add(RECORDER_FLUSH));
        }
    }

    /// `VoiceActivityVideoSampler`'s rate gate (faster while the user speaks). Rejected frames are
    /// dropped without journalling.
    pub(crate) fn on_video_frame(
        &mut self,
        now: Ts,
        frame: lk_agents_rtc::VideoFrameRef,
        cycle: &mut Cycle,
    ) {
        if self.session.closing_past_edge() {
            return;
        }
        if !self.input.samples_video(self.activity.audio_recognition().user_speaking(), now) {
            return;
        }
        let image = lk_agents_abi::ImageContent {
            id: self.session.mint("img"),
            image: lk_agents_abi::ImageSource::Frame(frame.media),
            inference_width: None,
            inference_height: None,
            inference_detail: lk_agents_abi::InferenceDetail::Auto,
            mime_type: None,
        };
        let sent = self.push_video(image, cycle);
        cycle.trace(InputStep::VideoFramePushed { sent });
    }

    /// `AgentActivity.push_video`. Dropping a `MediaRef` releases the host object. A pipeline
    /// session has no realtime session to push to, so it is said once and dropped.
    pub(crate) fn push_video(
        &mut self,
        image: lk_agents_abi::ImageContent,
        cycle: &mut Cycle,
    ) -> bool {
        // `if not self._started: return`.
        if self.session.closing_past_edge() {
            return false;
        }

        if self.activity.rt_session().is_none() {
            if self.input.warn_video_sink_once() {
                cycle.trace(CoreStep::PortFailure(
                    "push_video: no realtime session, so a sampled frame reaches no model",
                ));
            }
            return false;
        };

        let item = ChatMessage::new(
            // The message id is the driver's; the image's `img_` id came from the host.
            self.session.mint("video"),
            Role::User,
            vec![lk_agents_abi::ChatContent::Image(image)],
            self.shared.time.wall(),
        );
        self.rt_send(
            lk_agents_abi::RealtimeCommand::PushItem(lk_agents_abi::PushItem {
                item: Box::new(lk_agents_abi::ChatItem::Message(item)),
                reply_required: false,
            }),
            cycle,
        );
        true
    }

    /// `KeytermDetector.start(session, stt)`: the state survives handoffs, and static terms reach
    /// the recogniser at once.
    pub(crate) fn keyterms_start(&mut self, setup: KeytermSetup, cycle: &mut Cycle) {
        let (slot, initial) = KeytermSlot::start(setup);
        self.session.set_keyterms(slot);
        if !initial.is_empty() {
            self.push_keyterms(initial, cycle);
        }
    }

    fn keyterms_detecting(&self) -> bool {
        let Some(slot) = self.session.keyterms() else { return false };
        slot.detecting() && self.session.stt().is_some_and(|stt| stt.capabilities().keyterms)
    }

    pub(crate) fn keyterms_step(&mut self, now: Ts, input: KeytermIn, cycle: &mut Cycle) {
        let Some(slot) = self.session.keyterms_mut() else { return };
        let out = slot.step(input);
        self.perform_keyterms(now, out, cycle);
    }

    /// `_on_conversation_item_added`; fed only while a pass could run, as the reference subscribes.
    pub(crate) fn keyterms_on_item(&mut self, now: Ts, item: &ChatItem, cycle: &mut Cycle) {
        if !self.keyterms_detecting() {
            return;
        }
        let ChatItem::Message(m) = item else { return };
        let text = m.raw_text_content().unwrap_or_default();
        self.keyterms_step(now, KeytermIn::Item { role: m.role, text }, cycle);
    }

    /// The STT leg changed under a handoff: a different recogniser gets the full list.
    pub(crate) fn keyterms_on_stt_swap(&mut self, now: Ts, same_stt: bool, cycle: &mut Cycle) {
        if self.session.keyterms().is_none() {
            return;
        }
        cycle.trace(DetectorStep::KeytermsRebound { same_stt });
        self.keyterms_step(now, KeytermIn::AgentUpdated { same_stt }, cycle);
    }

    /// The session's close: the request in flight is cancelled, nothing is pushed after.
    pub(crate) fn keyterms_close(&mut self, now: Ts, cycle: &mut Cycle) {
        if self.session.keyterms().is_none() {
            return;
        }
        self.keyterms_step(now, KeytermIn::Close, cycle);
    }

    pub(crate) fn on_keyterm_answer(
        &mut self,
        now: Ts,
        req: u64,
        arguments: Option<String>,
        failed: bool,
        metrics: Option<Box<LlmMetrics>>,
        cycle: &mut Cycle,
    ) {
        let Some(slot) = self.session.keyterms_mut() else { return };
        if slot.answered(req) {
            cycle.machine().cancel(TurnTimer::KeytermTimeout);
        }
        // Rides the session stream like the reply LLM's metrics, with no speech.
        if let Some(m) = metrics {
            cycle.machine().emit(crate::events::SessionEvent::MetricsCollected {
                metrics: Box::new(lk_agents_abi::metrics::base::AgentMetrics::Llm(*m)),
                speech: None,
            });
        }
        if failed {
            return self.keyterms_failed(now, req, cycle);
        }
        cycle.trace(InputStep::LegVerdict { leg: DetectorLeg::Keyterms, req });
        // Strict JSON; a malformed document is an empty update.
        let parsed =
            arguments.as_deref().and_then(|a| serde_json::from_str::<serde_json::Value>(a).ok());
        let update = parse_record_keyterms(parsed.as_ref());
        self.keyterms_step(now, KeytermIn::Answer { req, update }, cycle);
    }

    /// `asyncio.wait_for(.., timeout)` expired: no change, the slot frees.
    pub(crate) fn on_keyterm_timeout(&mut self, now: Ts, cycle: &mut Cycle) {
        let Some(slot) = self.session.keyterms_mut() else { return };
        let Some(req) = slot.time_out() else { return };
        self.keyterms_failed(now, req, cycle);
    }

    fn keyterms_failed(&mut self, now: Ts, req: u64, cycle: &mut Cycle) {
        cycle.trace(InputStep::LegFailed { leg: DetectorLeg::Keyterms, recoverable: true });
        self.keyterms_step(now, KeytermIn::Failed { req }, cycle);
    }

    fn perform_keyterms(&mut self, now: Ts, out: Vec<KeytermEffect>, cycle: &mut Cycle) {
        for effect in out {
            match effect {
                KeytermEffect::Request { req, prompt } => {
                    self.request_keyterms(now, req, prompt, cycle)
                }
                KeytermEffect::Push(terms) => self.push_keyterms(terms, cycle),
                KeytermEffect::Cancel { req } => {
                    if let Some(slot) = self.session.keyterms_mut() {
                        slot.cancel(req);
                    }
                    cycle.machine().cancel(TurnTimer::KeytermTimeout);
                }
            }
        }
    }

    fn push_keyterms(&mut self, terms: Vec<String>, cycle: &mut Cycle) {
        if let Some(stt) = self.session.stt() {
            stt.update_keyterms(&terms);
        }
        cycle.trace(DetectorStep::KeytermsPushed { terms });
    }

    /// One request, bounded by the pass's timeout, run off the loop.
    fn request_keyterms(&mut self, now: Ts, req: u64, prompt: String, cycle: &mut Cycle) {
        let Some(slot) = self.session.keyterms_mut() else { return };
        let Some(llm) = slot.llm() else {
            // The machine asked with no model to ask: fed back as a failure so the slot frees.
            self.keyterms_step(now, KeytermIn::Failed { req }, cycle);
            return;
        };
        cycle.trace(InputStep::LegRequested { leg: DetectorLeg::Keyterms, req });
        cycle.machine().arm(TurnTimer::KeytermTimeout, now.saturating_add(slot.timeout()));
        let request = slot.request(prompt, self.shared.time.wall());
        let shared = Arc::clone(&self.shared);
        let handle =
            shared.spawner.clone().spawn_unboxed(detection_pass(shared.clone(), llm, request, req));
        slot.set_inflight(req, handle);
    }
}
