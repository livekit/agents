//! The activity's IO: [`AgentInput`], the STT and VAD legs (reopened by a handoff or
//! `update_models`), audio and video input, and output enablement.

use crate::agent_session::effects::{AudioOp, Effect, Out, StreamKind};
use crate::agent_session::{Driver, IoKind};
use crate::audio::{AudioChannel, ChannelEnd, on_leg_ended};
use crate::data_plane::ForwarderCommand;
use crate::room_io::InputProviders;
use crate::trace::{CoreStep, InputKind, InputStep, OutputKind, Step, TurnStep};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::voice::io::AudioInput;
use lk_agents_abi::{ChatMessage, Maybe, Role, Ts};
use std::sync::Arc;

/// `voice/io.py`'s `AgentInput` as the decision core holds it: the sink installed on each audio
/// channel, the gate's last verdict and the input switch (all three told to the data plane), the
/// camera with the task pumping it, and the STT and VAD streams being reopened for a new agent.
pub(crate) struct AgentInput {
    /// What this core installed on each channel; the data plane holds the same.
    pub(crate) sinks: std::collections::BTreeMap<crate::audio::AudioChannel, ResourceId>,
    /// The STT stream whose items the loop consumes: retired by a replacement or a removal, not
    /// by the stream's own end.
    pub(crate) live_stt: Option<ResourceId>,
    pub(crate) stt_provider: Option<lk_agents_abi::ProviderId>,
    /// The verdict last sent to the data plane; `None` before the first, so it is an edge.
    pub(crate) gate_discard: Option<bool>,
    /// `input.set_audio_enabled(..)`.
    pub(crate) audio_enabled: bool,
    /// When audio started flowing from the current source, as the data plane reported it.
    pub(crate) first_input_at: Option<Ts>,
    /// So the same leg handed back is a no-op. Runtime retains the object.
    pub(crate) video_instance: Option<usize>,
    /// Shared: its rates are public attributes a caller may change mid-call.
    video_sampler: Option<crate::room_io::video_sampler::VideoSampler>,
    /// A pipeline session has nowhere to push a frame: said the first time only.
    video_sink_warned: bool,
    next_open: u64,
    /// The open a leg's `StreamOpened` must carry to land; any other is discarded (0: none).
    stt_open: u64,
    vad_open: u64,
    /// The STT being opened is the same recogniser as before.
    stt_open_same: bool,
    /// The VAD stream's inferences since its last `VADMetrics`.
    pub(crate) vad_metrics: VadMetricsFold,
}

/// `VADStream._metrics_monitor_task`'s fold: the inferences since the last record, and when the
/// stream last heard speech start or end.
#[derive(Default)]
pub(crate) struct VadMetricsFold {
    pub(crate) count: u32,
    pub(crate) inference_total: f64,
    pub(crate) activity_at: Option<f64>,
}

impl AgentInput {
    /// The legs as the session opened them: the installed sinks, the camera with its pump (if one
    /// was attached), nothing pending.
    pub(crate) fn new(
        sinks: std::collections::BTreeMap<crate::audio::AudioChannel, ResourceId>,
        video_instance: Option<usize>,
        video_sampler: Option<crate::room_io::video_sampler::VideoSampler>,
    ) -> Self {
        AgentInput {
            live_stt: sinks.get(&crate::audio::AudioChannel::Stt).copied(),
            stt_provider: None,
            sinks,
            gate_discard: None,
            audio_enabled: true,
            first_input_at: None,
            video_instance,
            video_sampler,
            video_sink_warned: false,
            next_open: 0,
            stt_open: 0,
            vad_open: 0,
            stt_open_same: false,
            vad_metrics: VadMetricsFold::default(),
        }
    }
}

impl Driver {
    /// The runtime prepared the sink; acceptance installs it on the data plane.
    fn install_sink(&mut self, channel: AudioChannel, sink: ResourceId, out: &mut Out) {
        self.input.sinks.insert(channel, sink);
        if channel == AudioChannel::Stt {
            self.input.live_stt = Some(sink);
            self.stt_stream_opened(sink);
        }
        out.machine().push(Effect::Forwarder(ForwarderCommand::Install { channel, sink }));
        out.machine().push(Effect::ActivateLeg { channel, sink });
    }

    /// Take `channel`'s sink down, its input ended. `Agent(stt=None)`: this agent does not listen.
    fn remove_sink(&mut self, channel: AudioChannel, out: &mut Out) {
        if channel == AudioChannel::Stt {
            self.input.live_stt = None;
        }
        if let Some(sink) = self.input.sinks.remove(&channel) {
            out.machine().push(Effect::Forwarder(ForwarderCommand::Remove(sink)));
        }
    }

    /// `Given` reopens off the loop; the new sink replaces the old in one command, so no frame
    /// finds the channel empty. `Given(None)` closes the leg; `NotGiven` leaves it.
    pub(crate) fn reopen_legs(
        &mut self,
        stt: Maybe<Option<lk_agents_abi::resources::SttRef>>,
        vad: Maybe<Option<lk_agents_abi::resources::VadRef>>,
        out: &mut Out,
    ) -> InputProviders {
        let given = stt.is_given() || vad.is_given();
        // AgentUserState facts are the configured providers (`self._stt`/`self._vad`), not the live
        // streams: a dead stream is still a recogniser.
        let mut intent =
            InputProviders { stt: self.session.stt.is_some(), vad: self.session.vad.is_some() };
        if let Maybe::Given(stt) = stt {
            match stt {
                // A different STT: reopened; the row follows when the stream lands.
                Some(stt) => {
                    let same_stt =
                        self.session.stt.as_ref().is_some_and(|old| old.instance == stt.instance);
                    let open = self.request_open(StreamKind::Stt, stt.id, out);
                    self.session.stt = Some(stt);
                    self.input.stt_open_same = same_stt;
                    self.input.stt_open = open;
                    intent.stt = true;
                }
                // `Agent(stt=None)`: the leg closes and `stt_enabled` follows via `set_legs`.
                None => {
                    self.session.stt = None;
                    self.input.stt_open = 0;
                    self.remove_sink(AudioChannel::Stt, out);
                    intent.stt = false;
                }
            }
        }
        if let Maybe::Given(vad) = vad {
            match vad {
                Some(vad) => {
                    let open = self.request_open(StreamKind::Vad, vad.id, out);
                    self.session.vad = Some(vad);
                    self.input.vad_metrics = VadMetricsFold::default();
                    self.input.vad_open = open;
                    intent.vad = true;
                }
                None => {
                    self.session.vad = None;
                    self.input.vad_open = 0;
                    self.remove_sink(AudioChannel::Vad, out);
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

    fn request_open(
        &mut self,
        leg: StreamKind,
        resource: lk_agents_abi::resources::ResourceId,
        out: &mut Out,
    ) -> u64 {
        self.input.next_open += 1;
        let open = self.input.next_open;
        out.machine().push(match leg {
            StreamKind::Stt => Effect::SttStreamOpen { open, stt: resource },
            StreamKind::Vad => Effect::VadStreamOpen { open, vad: resource },
        });
        open
    }

    /// A superseded open is closed unused (`LegOpenDiscarded`); a refused one leaves the old row
    /// running.
    pub(crate) fn on_leg_opened(
        &mut self,
        now: Ts,
        open: u64,
        leg: StreamKind,
        outcome: Result<ResourceId, lk_agents_abi::errors::ProviderError>,
        out: &mut Out,
    ) {
        let wanted = match leg {
            StreamKind::Stt => self.input.stt_open,
            StreamKind::Vad => self.input.vad_open,
        };
        let live = open == wanted && open != 0 && !self.session.closing_past_edge();
        match (outcome, live) {
            (Err(_), _) => {
                out.trace(CoreStep::PortFailure(match leg {
                    StreamKind::Stt => "the new agent's STT could not be opened",
                    StreamKind::Vad => "the new agent's VAD could not be opened",
                }));
            }
            (Ok(sink), false) => {
                out.trace(InputStep::LegOpenDiscarded { leg: leg.label() });
                out.machine().push(Effect::DiscardLeg(sink));
            }
            (Ok(sink), true) => {
                let channel = match leg {
                    StreamKind::Stt => {
                        if let Some(stt) = &self.session.stt {
                            out.machine().push(Effect::ProviderPrewarm(stt.id));
                        }
                        self.input.stt_provider = None;
                        self.keyterms_on_stt_swap(now, self.input.stt_open_same, out);
                        AudioChannel::Stt
                    }
                    StreamKind::Vad => AudioChannel::Vad,
                };
                self.install_sink(channel, sink, out);
                out.trace(InputStep::StreamOpened { leg: channel.label() });
            }
        }
    }

    /// Only a given leg is reopened.
    pub(crate) fn update_models(
        &mut self,
        stt: Maybe<Option<lk_agents_abi::resources::SttRef>>,
        vad: Maybe<Option<lk_agents_abi::resources::VadRef>>,
        llm: Maybe<Option<lk_agents_abi::resources::LlmRef>>,
        tts: Maybe<Option<lk_agents_abi::resources::TtsRef>>,
        out: &mut Out,
    ) {
        self.session.apply_providers(llm, tts, out);
        let (stt_given, vad_given) = (stt.is_given(), vad.is_given());
        let (stt_reopened, vad_reopened) =
            (matches!(stt, Maybe::Given(Some(_))), matches!(vad, Maybe::Given(Some(_))));
        let _ = self.reopen_legs(stt, vad, out);
        out.trace(InputStep::ModelsUpdated { stt_given, vad_given, stt_reopened, vad_reopened });
    }

    /// A leg's drain ended: its sink is taken down if still installed. A failed STT is terminal
    /// (`STTError{recoverable: False}`); a VAD failure is only journalled.
    pub(crate) fn on_leg_ended(&mut self, now: Ts, mut end: ChannelEnd, out: &mut Out) {
        let error = end.error.take();
        let leg = end.leg;
        if let Some(stream) = end.sink {
            self.stt_stream_ended(stream);
        }
        // Not `remove_sink`: the items the stream sent before its end are still consumed.
        let installed = self.input.sinks.iter().find(|(_, sink)| Some(**sink) == end.sink);
        let installed = installed.map(|(&channel, &sink)| (channel, sink));
        if let Some((channel, sink)) = installed {
            self.input.sinks.remove(&channel);
            out.machine().push(Effect::Forwarder(ForwarderCommand::Remove(sink)));
        }
        // The live recogniser died or ran out mid-utterance: it will never say the speech it
        // began has ended, so it ends here, or the user speaks for the rest of the call and the
        // session is never idle again.
        let recogniser_gone = matches!(installed, Some((AudioChannel::Stt, _)))
            && end.cause != crate::trace::ChannelEndCause::Closing;
        on_leg_ended(end, out);
        if recogniser_gone && self.activity.audio_recognition.speaking_on_recogniser() {
            self.host_end_of_speech(now, out);
        }
        if let Some(error) = error
            && leg == AudioChannel::Stt.label()
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
        out.machine().push(Effect::Audio(AudioOp::SetAudioInput(
            input.map(crate::agent_session::commands::HostAudio),
        )));
        if attached {
            self.input.first_input_at = None;
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
    pub(crate) fn set_io_enabled(&mut self, now: Ts, leg: IoKind, enabled: bool, out: &mut Out) {
        let changed = match leg {
            IoKind::AudioInput => {
                let changed = std::mem::replace(&mut self.input.audio_enabled, enabled) != enabled;
                if changed {
                    let command = ForwarderCommand::SetInputEnabled(enabled);
                    out.machine().push(Effect::Forwarder(command));
                }
                changed
            }
            IoKind::AudioOutput => {
                self.set_outputs(|o| std::mem::replace(&mut o.audio_enabled, enabled)) != enabled
            }
            IoKind::Transcription => {
                self.set_outputs(|o| std::mem::replace(&mut o.transcription_enabled, enabled))
                    != enabled
            }
        };
        if !changed {
            return;
        }
        if matches!(leg, IoKind::AudioOutput | IoKind::Transcription) {
            // The current segment owns its captured flags; selection changes retire only
            // destinations no admitted binding retains.
            self.retire_outputs(out);
        }
        out.trace(match leg {
            IoKind::AudioInput => {
                Step::Input(InputStep::InputEnabled { leg: InputKind::Audio, enabled })
            }
            IoKind::AudioOutput => {
                Step::Input(InputStep::OutputEnabled { leg: OutputKind::Audio, enabled })
            }
            IoKind::Transcription => {
                Step::Input(InputStep::OutputEnabled { leg: OutputKind::Transcription, enabled })
            }
        });
        match leg {
            // "End user speaking state when audio is disabled" (`_on_audio_enabled_changed`).
            IoKind::AudioInput if !enabled => self.host_end_of_speech(now, out),
            IoKind::AudioInput => {}
            // The pause machine's "audio enabled" fact moved with the flag.
            IoKind::AudioOutput => self.refresh_pause_facts(now, out),
            IoKind::Transcription => {}
        }
    }

    /// `_commit_user_turn`'s "flush the stt by pushing silence", `ceil(duration / 0.2)` frames.
    /// Silent before any audio or with no STT leg.
    pub(crate) fn flush_stt_with_silence(&self, seconds: f64, out: &mut Out) {
        let heard = self.input.first_input_at.is_some();
        if !heard || !(seconds > 0.0) || !self.input.sinks.contains_key(&AudioChannel::Stt) {
            return;
        }
        // `ceil`, so at least the asked duration is flushed.
        let frames = (seconds / 0.2).ceil() as u32;
        out.machine().push(Effect::Forwarder(ForwarderCommand::FlushSttWithSilence { frames }));
        out.trace(TurnStep::SttFlushed { frames });
    }

    /// `VoiceActivityVideoSampler`'s rate gate (faster while the user speaks). Rejected frames are
    /// dropped without journalling.
    pub(crate) fn on_video_frame(
        &mut self,
        now: Ts,
        frame: lk_agents_abi::voice::rtc::VideoFrameRef,
        out: &mut Out,
    ) {
        if self.session.closing_past_edge() {
            return;
        }
        let speaking = self.activity.audio_recognition.user_speaking();
        let sampler = self.input.video_sampler.as_mut();
        if !sampler.is_none_or(|s| s.should_sample_at(speaking, now)) {
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
            if !std::mem::replace(&mut self.input.video_sink_warned, true) {
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
