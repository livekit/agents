use crate::agent_session::effects::{AudioOp, StreamKind, VoiceEffect};
use crate::agent_session::{IoKind, VoiceDriver};
use crate::audio::{AudioChannel, ChannelEnd, on_leg_ended};
use crate::data_plane::AudioInputRouterCommand;
use crate::trace::{CoreStep, InputKind, InputStep, OutputKind, Step, TurnStep};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::{ChatMessage, Maybe, Role, Ts};

pub(crate) struct AgentInput {
    /// Mirrors what the data plane holds.
    pub(crate) sinks: std::collections::BTreeMap<crate::audio::AudioChannel, ResourceId>,
    /// The STT stream whose items the loop consumes: retired by a replacement or a removal, not
    /// by the stream's own end.
    pub(crate) live_stt: Option<ResourceId>,
    pub(crate) stt_provider: Option<lk_agents_abi::ProviderId>,
    /// The verdict last sent to the data plane; `None` before the first, so it is an edge.
    pub(crate) gate_discard: Option<bool>,
    pub(crate) audio_enabled: bool,
    pub(crate) first_input_at: Option<Ts>,
    /// So the same leg handed back is a no-op.
    pub(crate) video_instance: Option<lk_agents_abi::resources::VideoInputRef>,
    /// Shared: its rates are public attributes a caller may change mid-call.
    video_sampler: Option<crate::agent_session::video_sampler::VideoSampler>,
    video_sink_warned: bool,
    next_open: u64,
    /// The open a leg's `StreamOpened` must carry to land; any other is discarded (0: none).
    stt_open: u64,
    vad_open: u64,
    /// The STT being opened is the same recogniser as before.
    stt_open_same: bool,
    pub(crate) vad_metrics: VadMetricsFold,
}

/// The VAD inferences since the last metrics record, and when speech last started or ended.
#[derive(Default)]
pub(crate) struct VadMetricsFold {
    pub(crate) count: u32,
    pub(crate) inference_total: f64,
    pub(crate) activity_at: Option<f64>,
}

impl AgentInput {
    pub(crate) fn new(
        sinks: std::collections::BTreeMap<crate::audio::AudioChannel, ResourceId>,
        video_instance: Option<lk_agents_abi::resources::VideoInputRef>,
        video_sampler: Option<crate::agent_session::video_sampler::VideoSampler>,
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

impl AgentInput {
    #[must_use = "the open request identity must be sent to the runtime"]
    fn request_open(
        &mut self,
        leg: StreamKind,
        resource: lk_agents_abi::resources::ResourceId,
        span: Option<lk_agents_abi::telemetry::record::SpanContext>,
    ) -> (u64, VoiceEffect) {
        self.next_open += 1;
        let open = self.next_open;
        let effect = match leg {
            StreamKind::Stt => VoiceEffect::OpenSttStream { open, stt: resource, span },
            StreamKind::Vad => VoiceEffect::OpenVadStream { open, vad: resource },
        };
        (open, effect)
    }
}

impl VoiceDriver {
    fn install_sink(&mut self, channel: AudioChannel, sink: ResourceId) {
        self.input.sinks.insert(channel, sink);
        if channel == AudioChannel::Stt {
            self.input.live_stt = Some(sink);
            self.stt_stream_opened(sink);
        }
        self.record_effect(VoiceEffect::UpdateAudioInputRouting(
            AudioInputRouterCommand::Install { channel, sink },
        ));
        self.record_effect(VoiceEffect::AdoptOpenedInput { channel, sink });
    }

    fn remove_sink(&mut self, channel: AudioChannel) {
        if channel == AudioChannel::Stt {
            self.input.live_stt = None;
        }
        if let Some(sink) = self.input.sinks.remove(&channel) {
            self.record_effect(VoiceEffect::UpdateAudioInputRouting(
                AudioInputRouterCommand::Remove(sink),
            ));
        }
    }

    /// `Given` reopens; the new sink replaces the old in one command, so no frame finds the
    /// channel empty. `Given(None)` closes the leg; `NotGiven` leaves it.
    pub(crate) fn reopen_legs(
        &mut self,
        stt: Maybe<Option<lk_agents_abi::resources::SttRef>>,
        vad: Maybe<Option<lk_agents_abi::resources::VadRef>>,
    ) -> InputProviders {
        let given = stt.is_given() || vad.is_given();
        // User-state facts follow the configured providers: a dead stream is still a recogniser.
        let mut intent =
            InputProviders { stt: self.session.stt.is_some(), vad: self.session.vad.is_some() };
        if let Maybe::Given(stt) = stt {
            match stt {
                Some(stt) => {
                    let same_stt =
                        self.session.stt.as_ref().is_some_and(|old| old.instance == stt.instance);
                    let activity = self.activity_span();
                    let (open, effect) = self.input.request_open(StreamKind::Stt, stt.id, activity);
                    self.record_effect(effect);
                    self.session.stt = Some(stt);
                    self.input.stt_open_same = same_stt;
                    self.input.stt_open = open;
                    intent.stt = true;
                }
                None => {
                    self.session.stt = None;
                    self.input.stt_open = 0;
                    self.remove_sink(AudioChannel::Stt);
                    intent.stt = false;
                }
            }
        }
        if let Maybe::Given(vad) = vad {
            match vad {
                Some(vad) => {
                    let (open, effect) = self.input.request_open(StreamKind::Vad, vad.id, None);
                    self.record_effect(effect);
                    self.session.vad = Some(vad);
                    self.input.vad_metrics = VadMetricsFold::default();
                    self.input.vad_open = open;
                    intent.vad = true;
                }
                None => {
                    self.session.vad = None;
                    self.input.vad_open = 0;
                    self.remove_sink(AudioChannel::Vad);
                    intent.vad = false;
                }
            }
        }
        if given {
            // A stale `stt_enabled` would make the commit wait for a transcript that never comes.
            self.activity.audio_recognition.set_legs(intent.stt, intent.vad);
        }
        intent
    }

    /// A superseded open is closed unused; a refused one leaves the old leg running.
    pub(crate) fn on_leg_opened(
        &mut self,
        now: Ts,
        open: u64,
        leg: StreamKind,
        outcome: Result<ResourceId, lk_agents_abi::errors::ProviderError>,
    ) {
        let wanted = match leg {
            StreamKind::Stt => self.input.stt_open,
            StreamKind::Vad => self.input.vad_open,
        };
        let live = open == wanted && open != 0 && !self.session.closing_past_edge();
        match (outcome, live) {
            (Err(_), _) => {
                self.record_trace(CoreStep::PortFailure(match leg {
                    StreamKind::Stt => "the new agent's STT could not be opened",
                    StreamKind::Vad => "the new agent's VAD could not be opened",
                }));
            }
            (Ok(sink), false) => {
                self.record_trace(InputStep::LegOpenDiscarded { leg: leg.leg() });
                self.record_effect(VoiceEffect::DiscardOpenedInput(sink));
            }
            (Ok(sink), true) => {
                let channel = self.adopt_opened_leg(now, leg);
                self.install_sink(channel, sink);
                self.record_trace(InputStep::StreamOpened { leg: channel.leg() });
            }
        }
    }

    fn adopt_opened_leg(&mut self, now: Ts, leg: StreamKind) -> AudioChannel {
        match leg {
            StreamKind::Stt => {
                if let Some(stt) = &self.session.stt {
                    self.record_effect(VoiceEffect::ProviderPrewarm(stt.id));
                }
                self.input.stt_provider = None;
                self.keyterms_on_stt_swap(now, self.input.stt_open_same);
                AudioChannel::Stt
            }
            StreamKind::Vad => AudioChannel::Vad,
        }
    }

    pub(crate) fn update_models(
        &mut self,
        stt: Maybe<Option<lk_agents_abi::resources::SttRef>>,
        vad: Maybe<Option<lk_agents_abi::resources::VadRef>>,
        llm: Maybe<Option<lk_agents_abi::resources::LlmRef>>,
        tts: Maybe<Option<lk_agents_abi::resources::TtsRef>>,
    ) {
        let effects = self.session.apply_providers(llm, tts);
        for effect in effects {
            self.record_effect(effect);
        }
        let (stt_given, vad_given) = (stt.is_given(), vad.is_given());
        let (stt_reopened, vad_reopened) =
            (matches!(stt, Maybe::Given(Some(_))), matches!(vad, Maybe::Given(Some(_))));
        self.reopen_legs(stt, vad);
        self.record_trace(InputStep::ModelsUpdated {
            stt_given,
            vad_given,
            stt_reopened,
            vad_reopened,
        });
    }

    /// A failed STT is a non-recoverable error; a VAD failure is only journalled.
    pub(crate) fn on_leg_ended(&mut self, now: Ts, mut end: ChannelEnd) {
        let error = end.error.take();
        let leg = end.leg;
        if let Some(stream) = end.sink {
            self.stt_stream_ended(stream);
            self.user_turn_stream_ended(now, stream);
        }
        // Not `remove_sink`: the items the stream sent before its end are still consumed.
        let installed = self.input.sinks.iter().find(|(_, sink)| Some(**sink) == end.sink);
        let installed = installed.map(|(&channel, &sink)| (channel, sink));
        if let Some((channel, sink)) = installed {
            self.input.sinks.remove(&channel);
            self.record_effect(VoiceEffect::UpdateAudioInputRouting(
                AudioInputRouterCommand::Remove(sink),
            ));
        }
        // A recogniser that died mid-utterance never reports its end: end it here, or the
        // session is never idle again.
        let recogniser_gone = matches!(installed, Some((AudioChannel::Stt, _)))
            && end.cause != crate::trace::ChannelEndCause::Closing;
        for observation in on_leg_ended(end) {
            self.record_trace(observation);
        }
        if recogniser_gone && self.activity.audio_recognition.speaking_on_recogniser() {
            self.host_end_of_speech(now);
        }
        if let Some(error) = error
            && leg == lk_agents_abi::voice::events::Leg::Stt
        {
            self.report_error(now, crate::events::ErrorSource::Stt, false, error, None);
        }
    }

    /// Detaching under a speaking user ends the speech; a re-attach forgets the acoustic anchor
    /// and reopens the STT leg on the same recogniser.
    pub(crate) fn set_audio_input(&mut self, now: Ts, input: Option<ResourceId>) {
        let attached = input.is_some();
        // The runtime's source reader owns the registration from here.
        if let Some(source) = input {
            self.session.providers.remove(&source);
        }
        self.record_trace(InputStep::AudioInputSet { attached });
        self.record_effect(VoiceEffect::Audio(AudioOp::SetAudioInput(
            input.map(crate::agent_session::commands::AudioInputHandle),
        )));
        if attached {
            self.input.first_input_at = None;
            self.activity.audio_recognition.input_started_at = None;
            if let Some(stt) = self.session.stt.as_ref().cloned() {
                self.reopen_legs(Maybe::Given(Some(stt)), Maybe::NotGiven);
            }
        } else {
            self.host_end_of_speech(now);
        }
    }

    /// The host ended the user's speech (mic detached or muted).
    pub(crate) fn host_end_of_speech(&mut self, now: Ts) {
        if !self.activity.audio_recognition.user_speaking() {
            return;
        }
        let eff = self.activity.audio_recognition.on_host_end_of_speech(now);
        self.apply_recognition_effects(eff);
        self.start_false_interruption_timer(now);
    }

    /// The leg stays installed; the host object is told.
    pub(crate) fn set_io_enabled(&mut self, now: Ts, leg: IoKind, enabled: bool) {
        let changed = match leg {
            IoKind::AudioInput => {
                let changed = std::mem::replace(&mut self.input.audio_enabled, enabled) != enabled;
                if changed {
                    let command = AudioInputRouterCommand::SetInputEnabled(enabled);
                    self.record_effect(VoiceEffect::UpdateAudioInputRouting(command));
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
            self.retire_outputs();
        }
        self.record_trace(match leg {
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
            IoKind::AudioInput if !enabled => self.host_end_of_speech(now),
            IoKind::AudioInput => {}
            IoKind::AudioOutput => self.refresh_pause_facts(now),
            IoKind::Transcription => {}
        }
    }

    /// Flushes the STT with `ceil(seconds / 0.2)` silent frames; a no-op before any audio or
    /// with no STT leg.
    pub(crate) fn flush_stt_with_silence(&mut self, seconds: f64) {
        let heard = self.input.first_input_at.is_some();
        if !heard
            || seconds.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
            || !self.input.sinks.contains_key(&AudioChannel::Stt)
        {
            return;
        }
        let frames = (seconds / 0.2).ceil() as u32;
        self.record_effect(VoiceEffect::UpdateAudioInputRouting(
            AudioInputRouterCommand::FlushSttWithSilence { frames },
        ));
        self.record_trace(TurnStep::SttFlushed { frames });
    }

    /// Rate-gated (faster while the user speaks); rejected frames are dropped unjournalled.
    pub(crate) fn on_video_frame(
        &mut self,
        now: Ts,
        frame: lk_agents_abi::voice::rtc::VideoFrameRef,
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
            image: lk_agents_abi::ImageSource::Frame(frame.frame),
            inference_width: None,
            inference_height: None,
            inference_detail: lk_agents_abi::InferenceDetail::Auto,
            mime_type: None,
        };
        let sent = self.push_video(image);
        self.record_trace(InputStep::VideoFramePushed { sent });
    }

    /// A pipeline session has no realtime session to push to: warned once, then dropped.
    pub(crate) fn push_video(&mut self, image: lk_agents_abi::ImageContent) -> bool {
        if self.session.closing_past_edge() {
            return false;
        }

        if self.activity.realtime().is_none() {
            if !std::mem::replace(&mut self.input.video_sink_warned, true) {
                self.record_trace(CoreStep::PortFailure(
                    "push_video: no realtime session, so a sampled frame reaches no model",
                ));
            }
            return false;
        };

        let item = ChatMessage::new(
            self.session.ids.mint("video"),
            Role::User,
            vec![lk_agents_abi::ChatContent::Image(image)],
            self.step.wall(),
        );
        self.rt_send(lk_agents_abi::RealtimeCommand::PushItem(lk_agents_abi::PushItem {
            item: Box::new(lk_agents_abi::ChatItem::Message(item)),
            reply_required: false,
        }));
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InputProviders {
    pub(crate) stt: bool,
    pub(crate) vad: bool,
}
