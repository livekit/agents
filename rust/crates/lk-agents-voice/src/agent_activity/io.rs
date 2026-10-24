//! The activity's IO: the STT and VAD legs (reopened by a handoff or `update_models`), audio and
//! video input, and output enablement.

use crate::agent_session::effects::{LegKind, Out};
use crate::agent_session::{Driver, IoLeg};
use crate::audio::{LegEnd, LegId, LegOp, Ticket, on_leg_ended, spawn_drain, tagged};
use crate::room_io::{LegIntent, LegOpened, Opened, PendingOpen};
use crate::trace::{CoreStep, InputLeg, InputStep, OutputLeg, Step, TurnStep};
use lk_agents_abi::{ChatMessage, Llm, Maybe, Role, Ts};
use lk_agents_voice::io::AudioInput;
use std::sync::Arc;

impl Driver {
    /// The one door every leg mutation goes through; a new source is swapped by the performer.
    pub(crate) fn apply_leg_op(&mut self, op: LegOp, out: &mut Out) -> Option<Ticket> {
        let source = matches!(op, LegOp::SetSource(_));
        let ticket = self.input.apply_leg_op(op);
        if source {
            out.machine().push(crate::agent_session::effects::Effect::Audio(
                crate::agent_session::effects::AudioOp::SetInput,
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
        out: &mut Out,
    ) -> LegIntent {
        let given = stt.is_given() || vad.is_given();
        // AgentUserState facts are the configured providers (`self._stt`/`self._vad`), not the live
        // streams: a dead stream is still a recogniser.
        let mut intent =
            LegIntent { stt: self.session.stt.is_some(), vad: self.session.vad.is_some() };
        if let Maybe::Given(stt) = stt {
            match stt {
                // A different STT: reopened; the row follows when the stream lands.
                Some(stt) => {
                    let same_stt =
                        self.session.stt.as_ref().is_some_and(|old| Arc::ptr_eq(old, &stt));
                    self.session.stt = Some(Arc::clone(&stt));
                    self.input.set_stt_open_same(same_stt);
                    let open = self.request_open(PendingOpen::Stt(stt), LegKind::Stt, out);
                    self.input.expect_open(LegKind::Stt, open);
                    intent.stt = true;
                }
                // `Agent(stt=None)`: the leg closes and `stt_enabled` follows via `set_legs`.
                None => {
                    self.session.stt = None;
                    self.input.expect_open(LegKind::Stt, 0);
                    self.apply_leg_op(LegOp::Clear(LegId::Stt), out);
                    intent.stt = false;
                }
            }
        }
        if let Maybe::Given(vad) = vad {
            match vad {
                Some(vad) => {
                    self.session.vad = Some(Arc::clone(&vad));
                    let open = self.request_open(PendingOpen::Vad(vad), LegKind::Vad, out);
                    self.input.expect_open(LegKind::Vad, open);
                    intent.vad = true;
                }
                None => {
                    self.session.vad = None;
                    self.input.expect_open(LegKind::Vad, 0);
                    self.apply_leg_op(LegOp::Clear(LegId::Vad), out);
                    intent.vad = false;
                }
            }
        }
        if given {
            // `decide_commit` reads `stt_enabled`; a stale flag would wait for a transcript that
            // never comes.
            self.activity.audio_recognition.set_legs(intent.stt, intent.vad);
        }
        intent
    }

    fn request_open(&mut self, pending: PendingOpen, leg: LegKind, out: &mut Out) -> u64 {
        let open = self.input.request_open(pending);
        let call = crate::agent_session::effects::ProviderCall::OpenLeg { leg, open };
        out.machine().push(crate::agent_session::effects::Effect::Provider(call));
        open
    }

    /// A superseded open is closed unused (`LegOpenDiscarded`); a refused one leaves the old row
    /// running.
    pub(crate) fn on_leg_opened(&mut self, now: Ts, opened: LegOpened, out: &mut Out) {
        let LegOpened { open, leg, outcome } = opened;
        let wanted = self.input.expected_open(leg);
        let live = open == wanted && open != 0 && !self.session.closing_past_edge();
        match (outcome, live) {
            (Err(_), _) => {
                out.trace(CoreStep::PortFailure(match leg {
                    LegKind::Stt => "the new agent's STT could not be opened",
                    LegKind::Vad => "the new agent's VAD could not be opened",
                }));
            }
            (Ok(stream), false) => {
                out.trace(InputStep::LegOpenDiscarded { leg: leg.label() });
                self.discard_open(stream);
            }
            (Ok(Opened::Stt(sink, source)), true) => {
                if let Some(stt) = self.session.stt.as_ref() {
                    stt.prewarm();
                }
                // A different recogniser is told the full list.
                let same_stt = self.input.stt_open_same();
                self.keyterms_on_stt_swap(now, same_stt, out);
                // A fresh detector per open; the replaced stream keeps its own for its remaining
                // items.
                let speaker = self.input.new_speaker_filter();
                let ticket = self
                    .apply_leg_op(LegOp::InstallStt { sink: Arc::from(sink), speaker }, out)
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
                out.trace(InputStep::LegOpened { leg: LegId::Stt.label() });
            }
            (Ok(Opened::Vad(sink, source)), true) => {
                let ticket = self
                    .apply_leg_op(LegOp::Install { leg: LegId::Vad, sink: Arc::from(sink) }, out)
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
                out.trace(InputStep::LegOpened { leg: LegId::Vad.label() });
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
        out: &mut Out,
    ) {
        self.session.apply_providers(llm, tts);
        let (stt_given, vad_given) = (stt.is_given(), vad.is_given());
        let (stt_reopened, vad_reopened) =
            (matches!(stt, Maybe::Given(Some(_))), matches!(vad, Maybe::Given(Some(_))));
        let _ = self.reopen_legs(stt, vad, out);
        out.trace(InputStep::ModelsUpdated { stt_given, vad_given, stt_reopened, vad_reopened });
    }

    /// A leg's drain ended. A failed STT is terminal (`STTError{recoverable: False}`); a VAD
    /// failure is only journalled.
    pub(crate) fn on_leg_ended(&mut self, now: Ts, mut end: LegEnd, out: &mut Out) {
        let error = end.error.take();
        let leg = end.leg;
        // Every item the drain sent is ahead of its end in the queue.
        if leg == LegId::Stt.label() {
            self.input.legs().retire_speaker_filter(end.open);
        }
        on_leg_ended(end, out);
        if let Some(error) = error
            && leg == LegId::Stt.label()
        {
            self.report_error(now, crate::events::ErrorSource::Stt, false, error, None, out);
        }
    }

    /// Detaching under a speaking user ends the speech like the reference's mute; a re-attach
    /// forgets the acoustic anchor and reopens the STT leg on the same recogniser.
    pub(crate) fn set_audio_input(
        &mut self,
        now: Ts,
        input: Option<Arc<dyn AudioInput>>,
        out: &mut Out,
    ) {
        let attached = input.is_some();
        out.trace(InputStep::AudioInputSet { attached });
        self.input.replace_audio_in(input.clone());
        self.apply_leg_op(LegOp::SetSource(input), out);
        if attached {
            self.input.forget_first_input();
            self.activity.audio_recognition.input_started_at = None;
            if let Some(stt) = self.session.stt.as_ref().cloned() {
                let _ = self.reopen_legs(Maybe::Given(Some(stt)), Maybe::NotGiven, out);
            }
        } else {
            self.host_end_of_speech(now, out);
        }
    }

    /// The host ended the user's speech (mic detached or muted: `on_end_of_speech(None)`).
    pub(crate) fn host_end_of_speech(&mut self, now: Ts, out: &mut Out) {
        if !self.activity.audio_recognition.user_speaking() {
            return;
        }
        let eff = self.activity.audio_recognition.on_host_end_of_speech(now);
        out.machine().append(eff);
        self.start_false_interruption_timer(now, out);
    }

    /// The leg stays installed; the host object is told.
    pub(crate) fn set_io_enabled(&mut self, now: Ts, leg: IoLeg, enabled: bool, out: &mut Out) {
        let changed = match leg {
            IoLeg::AudioInput => self.input.set_audio_enabled(enabled),
            IoLeg::AudioOutput => {
                let changed = self.shared.output.set_audio_enabled(enabled);
                if changed && let Some(out) = self.shared.output.audio() {
                    if enabled { out.on_attached() } else { out.on_detached() }
                }
                changed
            }
            IoLeg::Transcription => {
                let changed = self.shared.output.set_transcription_enabled(enabled);
                if changed && let Some(out) = self.shared.output.transcription() {
                    if enabled { out.on_attached() } else { out.on_detached() }
                }
                changed
            }
        };
        if !changed {
            return;
        }
        out.trace(match leg {
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
            IoLeg::AudioInput if !enabled => self.host_end_of_speech(now, out),
            IoLeg::AudioInput => {}
            // The pause machine's "audio enabled" fact moved with the flag.
            IoLeg::AudioOutput => self.refresh_pause_facts(now, out),
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

    /// `VoiceActivityVideoSampler`'s rate gate (faster while the user speaks). Rejected frames are
    /// dropped without journalling.
    pub(crate) fn on_video_frame(
        &mut self,
        now: Ts,
        frame: lk_agents_rtc::VideoFrameRef,
        out: &mut Out,
    ) {
        if self.session.closing_past_edge() {
            return;
        }
        if !self.input.samples_video(self.activity.audio_recognition.user_speaking(), now) {
            return;
        }
        let image = lk_agents_abi::ImageContent {
            id: self.session.ids.mint("img"),
            image: lk_agents_abi::ImageSource::Frame(frame.media),
            inference_width: None,
            inference_height: None,
            inference_detail: lk_agents_abi::InferenceDetail::Auto,
            mime_type: None,
        };
        let sent = self.push_video(image, out);
        out.trace(InputStep::VideoFramePushed { sent });
    }

    /// `AgentActivity.push_video`. Dropping a `MediaRef` releases the host object. A pipeline
    /// session has no realtime session to push to, so it is said once and dropped.
    pub(crate) fn push_video(&mut self, image: lk_agents_abi::ImageContent, out: &mut Out) -> bool {
        // `if not self._started: return`.
        if self.session.closing_past_edge() {
            return false;
        }

        if self.activity.rt_session.is_none() {
            if self.input.warn_video_sink_once() {
                out.trace(CoreStep::PortFailure(
                    "push_video: no realtime session, so a sampled frame reaches no model",
                ));
            }
            return false;
        };

        let item = ChatMessage::new(
            // The message id is the driver's; the image's `img_` id came from the host.
            self.session.ids.mint("video"),
            Role::User,
            vec![lk_agents_abi::ChatContent::Image(image)],
            self.step.wall(),
        );
        self.rt_send(
            lk_agents_abi::RealtimeCommand::PushItem(lk_agents_abi::PushItem {
                item: Box::new(lk_agents_abi::ChatItem::Message(item)),
                reply_required: false,
            }),
            out,
        );
        true
    }
}
