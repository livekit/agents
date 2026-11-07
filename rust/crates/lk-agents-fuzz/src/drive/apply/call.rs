//! Host calls grouped by the input family and the resources each call returns.
use super::Pending as AnswerWait;
use super::*;

#[derive(Default)]
enum Pending {
    #[default]
    Sent,
    Speech {
        handle: Arc<SpeechHandle>,
        kind: SpeechKind,
        stream: Option<Feeding>,
    },
    Answer {
        waiting: AnswerWait,
        owner: AwaitOwner,
    },
}

enum AwaitOwner {
    Host,
    Tool(String),
    Dtmf,
}

impl Pending {
    fn answer(answer: AnswerWait) -> Self {
        Self::Answer { waiting: answer, owner: AwaitOwner::Host }
    }

    fn speech(speech: Arc<SpeechHandle>, kind: SpeechKind, stream: Option<Feeding>) -> Self {
        Self::Speech { handle: speech, kind, stream }
    }
}

impl Run {
    pub(super) fn call(&mut self, at: Observed, input: &Input) -> bool {
        let pending = match input {
            Input::Say { .. }
            | Input::GenerateReply { .. }
            | Input::Interrupt { .. }
            | Input::InterruptSpeech { .. }
            | Input::SetAllowInterruptions { .. } => self.call_speech(input),
            Input::CommitUserTurn { .. }
            | Input::WaitForIdle { .. }
            | Input::WaitForIdleAndHold { .. }
            | Input::ReleaseIdleHold { .. }
            | Input::ClaimUserTurn { .. }
            | Input::ReleaseUserTurn { .. }
            | Input::CommitTurn { .. }
            | Input::ClearTurn { .. } => self.call_turn(input),
            Input::ToolUpdate { .. }
            | Input::EndCall { .. }
            | Input::FillerStart { .. }
            | Input::FillerStop { .. }
            | Input::VoiceTask { .. } => self.call_tool(input),
            Input::Configure { .. }
            | Input::Override { .. }
            | Input::ReleaseOverride { .. }
            | Input::Handoff { .. }
            | Input::UpdateInstructions { .. }
            | Input::UpdateTools { .. }
            | Input::UpdateChatCtx { .. }
            | Input::UpdateModels { .. } => self.call_configuration(input),
            Input::SetAudioInput { .. }
            | Input::SetVideoInput { .. }
            | Input::SetIoEnabled { .. }
            | Input::SetAudioOutput { .. }
            | Input::SetTranscriptionOutput { .. }
            | Input::Recorder { .. }
            | Input::PushVideo { .. } => self.call_io(at, input),
            Input::Dtmf { .. }
            | Input::Ivr { .. }
            | Input::SetStaticKeyterms { .. }
            | Input::HookDone { .. }
            | Input::HookDoneLate { .. }
            | Input::HostMetrics { .. }
            | Input::PauseAuthorization { .. }
            | Input::ResumeAuthorization { .. } => self.call_workflow(input),
            Input::Drain { .. } | Input::Close { .. } | Input::Run { .. } => {
                self.call_lifecycle(input)
            }
            _ => unreachable!("{input:?} is not a call"),
        };
        let Some(pending) = pending else { return false };
        let (answer, owner) = match pending {
            Pending::Sent => return true,
            Pending::Answer { waiting, owner } => (waiting, owner),
            Pending::Speech { handle, kind, stream } => {
                let id = handle.reserved().expect("a host call's handle has its id");
                self.handles.entry(id.0).or_insert(handle.clone());
                self.open_once(
                    at,
                    Kind::Speech,
                    Id::Speech(id),
                    Detail::Speech { kind, user_initiated: true },
                );
                if let Some(feeding) = stream {
                    let audio = matches!(feeding, Feeding::Audio { .. });
                    let nth =
                        self.record.open(at, Kind::Stream, Id::Pending, Detail::Stream { audio });
                    self.streams.insert(nth, feeding);
                }
                (AnswerWait::speech(handle), AwaitOwner::Host)
            }
        };
        let nth = self.record.open(at, Kind::Answer, Id::Pending, Detail::None);
        self.pending.insert(nth, answer);
        match owner {
            AwaitOwner::Host => {}
            AwaitOwner::Tool(call_id) => {
                self.task_awaits.insert(call_id, nth);
            }
            AwaitOwner::Dtmf => self.dtmf_await = Some(nth),
        }
        true
    }

    fn call_speech(&mut self, input: &Input) -> Option<Pending> {
        let api = &self.api;
        match input {
            Input::Say { text: AgentText(text), input, allow, to_ctx } => {
                let options = SayOptions { allow_interruptions: *allow, add_to_chat_ctx: *to_ctx };
                let (speech, stream) = match input {
                    SayInput::Words => (
                        api.say(
                            OneOrStream::One(text.clone()),
                            None,
                            options.allow_interruptions,
                            options.add_to_chat_ctx,
                            None,
                        )
                        .ok()?,
                        None,
                    ),
                    SayInput::TextStream => {
                        let (speech, pusher) = api
                            .say_text_channel(
                                options.allow_interruptions,
                                options.add_to_chat_ctx,
                                None,
                            )
                            .ok()?;
                        (speech, Some(Feeding::Text(pusher)))
                    }
                    SayInput::Audio => {
                        let (speech, pusher) = api
                            .say_audio_channel(
                                text.clone(),
                                options.allow_interruptions,
                                options.add_to_chat_ctx,
                                None,
                            )
                            .ok()?;
                        (speech, Some(Feeding::Audio { pusher, chunks: 0 }))
                    }
                };
                Some(Pending::speech(speech, SpeechKind::Say, stream))
            }
            Input::GenerateReply { user_input, instructions, allow } => {
                let speech = api
                    .generate_reply(
                        user_input.clone().map(|UserText(t)| t),
                        instructions.clone().map(|AgentText(t)| t),
                        None,
                        None,
                        *allow,
                        None,
                        None,
                    )
                    .ok()?;
                Some(Pending::speech(speech, SpeechKind::GenerateReply, None))
            }
            Input::Interrupt { force, wait } => {
                let reply = api.interrupt(*force).ok()?;
                Some(if *wait {
                    Pending::answer(AnswerWait::reply(reply, Answer::Interrupt))
                } else {
                    Pending::default()
                })
            }
            Input::InterruptSpeech { speech, force } => {
                HostSpeech::interrupt(&*self.host_speech(*speech)?, *force).ok()?;
                Some(Pending::default())
            }
            Input::SetAllowInterruptions { speech, allow } => {
                HostSpeech::set_allow_interruptions(&*self.host_speech(*speech)?, *allow).ok()?;
                Some(Pending::default())
            }
            _ => unreachable!(),
        }
    }

    fn call_turn(&mut self, input: &Input) -> Option<Pending> {
        let api = &self.api;
        match input {
            Input::CommitUserTurn { transcript_timeout, stt_flush, skip_reply } => {
                let reply = api
                    .commit_user_turn(secs(*transcript_timeout), secs(*stt_flush), *skip_reply)
                    .ok()?;
                return Some(Pending::answer(AnswerWait::reply(reply, Answer::Commit)));
            }
            Input::WaitForIdle { hold } => {
                let hold = match hold {
                    Some(h) => Some(self.hold_id(*h)?.0),
                    None => None,
                };
                let reply = api.wait_for_idle(hold).ok()?;
                return Some(Pending::answer(AnswerWait::reply(reply, Answer::Idle)));
            }
            Input::WaitForIdleAndHold {} => {
                let reply = api.wait_for_idle_and_hold(None).ok()?;
                return Some(Pending::answer(AnswerWait::reply(reply, Answer::Hold)));
            }
            Input::ReleaseIdleHold { hold } => {
                api.release_idle_hold(self.hold_id(*hold)?.0);
                return self.loop_runs().then(Pending::default);
            }
            Input::ReleaseUserTurn {} => {
                api.release_user_turn();
                return self.loop_runs().then(Pending::default);
            }
            Input::ClaimUserTurn {} => api.claim_user_turn().ok()?,
            Input::CommitTurn { skip_reply } => {
                api.send(TurnCommand::Commit { skip_reply: *skip_reply }).ok()?
            }
            Input::ClearTurn {} => api.clear_user_turn().ok()?,
            _ => unreachable!(),
        }
        Some(Pending::default())
    }

    fn call_tool(&mut self, input: &Input) -> Option<Pending> {
        let api = &self.api;
        match input {
            Input::ToolUpdate { tool, progress, end } => {
                let Detail::Tool { invocation_id, operation, call_id, .. } =
                    self.record.resource(Kind::Tool, tool.nth)?.detail()
                else {
                    return None;
                };
                let progress = progress.as_ref().map(|AgentText(m)| m.as_str());
                let outcome = crate::fakes::tools::update_outcome(progress, *end);
                api.tool_update(invocation_id.clone(), *operation, call_id.clone(), outcome.into())
                    .ok()?;
            }
            Input::EndCall { tool } => {
                let Detail::Tool { speech, .. } =
                    self.record.resource(Kind::Tool, tool.nth)?.detail()
                else {
                    return None;
                };
                HostSpeech::end_call(&**self.handles.get(speech)?).ok()?;
            }
            Input::FillerStart { tool, speech, delay, interval, text } => {
                api.filler_start(
                    self.tool_call_id(*tool)?,
                    self.speech_id(*speech)?.to_string(),
                    secs(*delay),
                    interval.map(secs),
                    None,
                    text.clone().map(|AgentText(t)| t),
                    None,
                    self.tool_origin(*tool),
                )
                .ok()?;
            }
            Input::FillerStop { filler } => api.filler_stop(self.filler_id(*filler)?.0).ok()?,
            Input::VoiceTask { tool, agent, reply } => {
                let call_id = self.tool_call_id(*tool)?;
                let args = frame(
                    &self.setup,
                    &self.legs,
                    &self.registry,
                    agent,
                    &format!("You are {agent}."),
                    &self.setup.tools,
                );
                let reply = api.run_voice_task(args, self.tool_origin(*tool), *reply, None).ok()?;
                return Some(Pending::Answer {
                    owner: AwaitOwner::Tool(call_id),
                    waiting: AnswerWait::reply(reply, |r| {
                        Answer::VoiceTask(r.map(|v| match v {
                            lk_agents_abi::voice::session::VoiceTaskValue::Value(value) => {
                                value.id()
                            }
                            lk_agents_abi::voice::session::VoiceTaskValue::Digits(_) => {
                                unreachable!("this input did not configure a keypad collector")
                            }
                        }))
                    }),
                });
            }
            _ => unreachable!(),
        }
        Some(Pending::default())
    }

    fn call_configuration(&mut self, input: &Input) -> Option<Pending> {
        let api = &self.api;
        match input {
            Input::Configure { knob, mode, ms, on, words } => {
                match turn_update(*knob, *mode, *ms) {
                    Some(update) => api.update_turn_options(update).ok()?,
                    None => api
                        .send(TurnCommand::Configure(Box::new(patch(
                            *knob, *mode, *ms, *on, *words,
                        ))))
                        .ok()?,
                }
            }
            Input::Override { knob, mode, ms } => {
                self.overrides.push(api.override_options(turn_update(*knob, *mode, *ms)?).ok()?);
            }
            Input::ReleaseOverride { nth } => {
                if self.overrides.is_empty() {
                    return None;
                }
                api.release_override(self.overrides.remove(nth.0 as usize % self.overrides.len()));
                return self.loop_runs().then(Pending::default);
            }
            Input::Handoff { agent, instructions, tools, turn } => {
                let mut args =
                    frame(&self.setup, &self.legs, &self.registry, agent, &instructions.0, tools);
                args.turn_handling =
                    turn.and_then(|t| turn_update(t.knob, t.mode, t.ms)).map(|update| {
                        TurnHandlingKwargs {
                            turn_detection: update.turn_detection,
                            endpointing: update.endpointing_opts,
                            ..TurnHandlingKwargs::default()
                        }
                    });
                return Some(Pending::answer(AnswerWait::reply(
                    api.update_agent(args).ok()?,
                    |ok| Answer::Handoff(ok.then_some(()).ok_or(())),
                )));
            }
            Input::UpdateInstructions { text: AgentText(text) } => {
                api.update_instructions(text.clone()).ok()?
            }
            Input::UpdateTools { tools } => api
                .update_tools(
                    Some(tool_specs(tools)),
                    None,
                    Some(fakes::frame(&self.registry, "tools", tools)),
                )
                .ok()?,
            Input::UpdateChatCtx { items, text: AgentText(text) } => {
                api.update_chat_ctx(history(items, text)).ok()?
            }
            Input::UpdateModels { models } => return self.call_models(*models),
            _ => unreachable!(),
        }
        Some(Pending::default())
    }

    fn call_models(&mut self, models: Models) -> Option<Pending> {
        let llm = match models {
            Models::Stt | Models::Tts => Maybe::NotGiven,
            Models::Llm | Models::LlmAndTts | Models::All => {
                Maybe::Given(Some(fakes::FakeLlm::new(self.registry.clone()) as Arc<Llm>))
            }
        };
        let tts = match models {
            Models::Stt | Models::Llm => Maybe::NotGiven,
            Models::Tts | Models::LlmAndTts | Models::All => {
                Maybe::Given(Some(fakes::FakeTts::new(
                    self.registry.clone(),
                    fakes::TtsOptions {
                        latency: self.setup.tts_latency.map(|seed| !seed),
                        timing: self.setup.timing(),
                        streaming: !self.setup.chunked_tts,
                        alignment: self.setup.tts_alignment,
                    },
                ) as Arc<Tts>))
            }
        };
        let stt = match models {
            Models::Llm | Models::Tts | Models::LlmAndTts => Maybe::NotGiven,
            Models::Stt | Models::All => Maybe::Given(Some(session_stt(
                &self.setup,
                &self.registry,
                Duration::from_millis(self.setup.connect_ms.unwrap_or(0)),
            ))),
        };
        if let Maybe::Given(new) = &llm {
            self.legs.llm = new.clone();
        }
        if let Maybe::Given(new) = &tts {
            self.legs.tts = new.clone();
        }
        if let Maybe::Given(new) = &stt {
            self.legs.stt = new.clone();
        }
        self.api.update_models(llm, tts, stt, Maybe::NotGiven, None, None, None).ok()?;
        Some(Pending::default())
    }

    fn call_io(&mut self, at: Observed, input: &Input) -> Option<Pending> {
        let api = &self.api;
        let registry = &self.registry;
        match input {
            Input::SetAudioInput { attach } => {
                for (nth, _) in std::mem::take(&mut self.mics) {
                    self.record.close(at, Kind::Mic, nth);
                }
                self.host_mic = true;
                let source = attach.then(|| {
                    let (mic, frames) = fakes::FakeAudioInput::new();
                    self.mics.insert(registry.new_mic(), (Mic::Host(frames), 0));
                    mic as Arc<dyn lk_agents_abi::voice::io::AudioInput>
                });
                api.set_audio_input(source).ok()?;
            }
            Input::SetVideoInput { attach } => api
                .set_video_input(attach.then(|| fakes::FakeVideoInput::new(registry) as _))
                .ok()?,
            Input::SetIoEnabled { leg, enabled } => api.set_io_enabled(*leg, *enabled).ok()?,
            Input::SetAudioOutput { attach } => api
                .set_audio_output(attach.then(|| {
                    fakes::FakeAudioOutput::new(
                        registry.clone(),
                        self.setup.pausable_output,
                        self.setup.lagging_output,
                    ) as _
                }))
                .ok()?,
            Input::SetTranscriptionOutput { attach } => api
                .set_transcription_output(attach.then(|| {
                    fakes::FakeTextOutput::new(registry.clone(), self.setup.text_output_raises) as _
                }))
                .ok()?,
            Input::Recorder { attach } => {
                api.attach_recorder(
                    attach.then(|| fakes::FakeRecorder::new(registry.clone()) as _),
                );
                return self.loop_runs().then(Pending::default);
            }
            Input::PushVideo {} => {
                return api
                    .push_video(ImageContent {
                        id: "img_case".into(),
                        image: ImageSource::Frame(FakeFrame::shared(self.record.now().seq)),
                        inference_width: None,
                        inference_height: None,
                        inference_detail: Default::default(),
                        mime_type: None,
                    })
                    .then(Pending::default);
            }
            _ => unreachable!(),
        }
        Some(Pending::default())
    }

    fn call_workflow(&mut self, input: &Input) -> Option<Pending> {
        let api = &self.api;
        match input {
            Input::PauseAuthorization {} => {
                api.send_from(VoiceCommandOrigin::Amd, SpeechCommand::PauseAuthorization).ok()?
            }
            Input::ResumeAuthorization {} => {
                api.send_from(VoiceCommandOrigin::Amd, SpeechCommand::ResumeAuthorization).ok()?
            }
            Input::HostMetrics {} => api.report_avatar_metrics(avatar_metrics()).ok()?,
            Input::HookDone { hook, answer, text: AgentText(text) } => {
                return self
                    .answer_hook(self.hook_id(*hook)?, *answer, text.clone())
                    .then(Pending::default);
            }
            Input::HookDoneLate { hook, answer, text: AgentText(text) } => {
                return self
                    .answer_hook(self.hook_id(Ref::new(hook.nth))?, *answer, text.clone())
                    .then(Pending::default);
            }
            Input::SetStaticKeyterms { terms } => {
                api.send(Command::SetStaticKeyterms(terms.clone())).ok()?
            }
            Input::Ivr { input, silence, text: UserText(text) } => match input {
                IvrInput::Start => api.ivr_start(secs(*silence), None, None).ok()?,
                IvrInput::SeedTranscript => api.ivr_final(text.clone()).ok()?,
                IvrInput::Close => api.send(IvrCommand::Close).ok()?,
            },
            Input::Dtmf { input, key, digits, confirm, timeout } => match input {
                DtmfInput::Start => {
                    let reply = api
                        .collect_dtmf(digits.0 as usize + 1, *confirm, secs(*timeout), "#".into())
                        .ok()?;
                    return Some(Pending::Answer {
                        owner: AwaitOwner::Dtmf,
                        waiting: AnswerWait::reply(reply, |answer| {
                            Answer::Dtmf(dtmf_outcome(answer))
                        }),
                    });
                }
                DtmfInput::Digit | DtmfInput::GuestDigit => {
                    let who = if *input == DtmfInput::Digit {
                        super::super::start::CALLER
                    } else {
                        super::super::start::GUEST
                    };
                    return ((self.setup.room.is_some() || self.setup.room_legs.is_some())
                        && self
                            .room
                            .press_dtmf(&rtc::ParticipantIdentity::from(who), key.digit()))
                    .then(Pending::default);
                }
                DtmfInput::Stop => {
                    if let Some(nth) = self.dtmf_await.take()
                        && self.pending.remove(&nth).is_some()
                    {
                        self.record_abandoned(nth);
                    }
                    return self.loop_runs().then(Pending::default);
                }
            },
            _ => unreachable!(),
        }
        Some(Pending::default())
    }

    fn call_lifecycle(&mut self, input: &Input) -> Option<Pending> {
        match input {
            Input::Drain { wait } => {
                let reply = self.api.drain().ok()?;
                Some(if *wait {
                    Pending::answer(AnswerWait::reply(reply, |refused| {
                        Answer::Drain(refused.map_or(Ok(()), Err))
                    }))
                } else {
                    Pending::default()
                })
            }
            Input::Close { drain } => {
                self.record.enter_phase(Phase::Closing);
                let mut closing: BoxFuture<'static, ()> =
                    Box::pin(self.api.aclose("user_initiated".into(), *drain).ok()?);
                if ready_now(closing.as_mut()).is_none() {
                    self.closing = Some(closing);
                }
                self.loop_runs().then(Pending::default)
            }
            Input::Run { user_input } => {
                self.api.run(user_input.clone().map(|UserText(t)| t), None, None).ok()?;
                Some(Pending::default())
            }
            _ => unreachable!(),
        }
    }
}

struct SayOptions {
    allow_interruptions: Option<bool>,
    add_to_chat_ctx: bool,
}
