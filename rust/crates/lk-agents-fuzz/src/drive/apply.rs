//! One input into what the product sees: a call on the host API, a stream's next piece, a fake's
//! answer, something in the room, or a wait.
//!
//! Every input is recorded, taken or refused. A reference that no longer resolves -- a speech the
//! driver has forgotten, a hook already finished -- makes the input a refused attempt, and so does
//! a call the API refuses. Nothing is invented to stand in for it, because a call carrying an id
//! the driver never issued is a different test from the one the case asked for.
use std::task::{Context, Poll, Waker};

use lk_agents_abi::voice::session_options::UpdateKwargs;
use lk_agents_rtc::{ConnectionState, DisconnectReason, RtcRoom};
use lk_agents_voice::agent_session::{CommitSuperseded, DrainRefused, HostMetricsRecord};
use lk_agents_voice::speech_handle::SpeechWireId;
use lk_agents_voice::workflows::dtmf::{DtmfOutcome, GetDtmfResult};

use super::start::{
    Llm, Tts, frame, history, instructions, patch, session_stt, tool_specs, turn_detection,
    vad_metrics,
};
use super::*;
use crate::record::{Task, TaskEnd};

impl Run {
    /// Apply one input, then write down everything that became true because of it.
    ///
    /// Every input is recorded, taken or refused. A delivery to a receiver the product has already
    /// closed is a fact about the run, not an error, and the case is never steered around one.
    pub async fn apply(&mut self, input: &Input) {
        // `VOICE_TRACE=1` narrates every input as it is applied, whichever phase sends it: the
        // way to find the one a run stops answering on.
        static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *TRACE.get_or_init(|| std::env::var_os("VOICE_TRACE").is_some()) {
            eprintln!("> {input:?}");
        }
        let at = self.stamps.next();
        let accepted = match input {
            Input::Advance { ms } => {
                let until = self.time.now().saturating_add(Duration::from_millis(ms.0));
                self.drive(Some(until), std::future::pending::<()>()).await;
                true
            }
            Input::AwaitAnswer { answer } => self.await_answer(answer.nth).await,
            Input::AwaitSpeech { speech } => self.await_speech(speech.nth).await,
            Input::AwaitExit {} => self.await_exit().await,
            Input::Release {} => self.phase(at, Phase::Releasing),
            Input::Settle {} => self.phase(at, Phase::Settling),
            Input::FeedText { .. } | Input::FeedAudio { .. } | Input::FeedEnd { .. } => {
                self.feed(at, input)
            }
            Input::MicFrames { mic, chunks } => self.speak(mic.nth, chunks.0),
            Input::TtsOutage { down } => {
                self.registry.set_voice_down(*down);
                true
            }
            Input::Room { event } => self.room_event(*event),
            Input::Amd { input } => self.amd(*input),
            Input::Console { ask } => self.console_ask(*ask),
            _ if fakes::boundary_of(input).is_some() => self.deliver(at, input),
            _ => self.call(at, input),
        };
        self.record.record_applied(at, input.clone(), accepted);
        self.collect();
    }

    /// Enter a phase. A progress deadline starts at `Settling` and not before, so both marks are
    /// kept: releasing advances virtual time to let the driver's own timers fire, which is work
    /// the run asked for rather than time the driver spent stuck.
    fn phase(&mut self, at: Observed, phase: Phase) -> bool {
        self.record.enter_phase(phase);
        if phase >= Phase::Settling {
            self.record.mark_settling(at);
        } else {
            self.record.mark_released(at);
        }
        true
    }
}

// ==============================================================================================
// Calling the host API
// ==============================================================================================

/// An answer that arrived, plus the live handle it carried, which only this module keeps.
pub(super) struct Settled {
    pub(super) answer: Answer,
    pub(super) handle: Option<Arc<SpeechHandle>>,
}

impl Settled {
    fn of(answer: Answer) -> Self {
        Settled { answer, handle: None }
    }
}

/// What a call returned that the host awaits: a `Reply`, or the speech handle's creation. Kept
/// until it resolves; how it is spelled in the recording is decided when it is opened.
pub(super) struct Pending(BoxFuture<'static, Settled>);

impl Pending {
    fn reply<T: 'static>(
        reply: impl std::future::Future<Output = T> + Send + 'static,
        answer: fn(T) -> Answer,
    ) -> Self {
        Pending(Box::pin(async move { Settled::of(answer(reply.await)) }))
    }

    /// `await handle.wait_until_created()`: the speech, or the handle's refusal when the loop
    /// dropped it uncreated.
    fn speech(speech: HostSpeech) -> Self {
        Pending(Box::pin(async move {
            match speech.wait_until_created().await {
                Ok(()) => match speech.peek() {
                    Some(h) => Settled { answer: Answer::Speech(h.id().0), handle: Some(h) },
                    None => Settled::of(Answer::Lost),
                },
                Err(_) => Settled::of(Answer::Lost),
            }
        }))
    }

    pub(super) fn try_recv(&mut self) -> Option<Settled> {
        ready_now(self.0.as_mut())
    }

    pub(super) async fn recv(&mut self) -> Settled {
        self.0.as_mut().await
    }
}

/// One poll, with nobody to wake: what was ready without giving anything another turn. Outside
/// Tokio's budget, so a resource that is ready never answers "try later".
pub(super) fn ready_now<F: std::future::Future>(future: F) -> Option<F::Output> {
    let mut future = std::pin::pin!(tokio::task::unconstrained(future));
    match future.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(value) => Some(value),
        Poll::Pending => None,
    }
}

/// A `ResolvedSessionOptions` leg name, as `set_io_enabled` takes it.
fn io_leg(leg: IoKind) -> &'static str {
    match leg {
        IoKind::AudioInput => "audio_input",
        IoKind::AudioOutput => "audio_output",
        IoKind::Transcription => "transcription",
    }
}

fn secs(ms: Millis) -> f64 {
    ms.0 as f64 / 1000.0
}

/// A `Ref` is the n-th resource of its kind; the production identity is whatever the run
/// recorded when it opened that one. Resolution can fail, and a failure is the case pointing at
/// something that is no longer there -- never a reason to substitute anything.
macro_rules! resolvers {
    ($($name:ident: $marker:ident -> $id:ident($ty:ty);)*) => {
        impl Run {$(
            fn $name(&self, reference: Ref<crate::input::$marker>) -> Option<$ty> {
                match self.record.resource(Kind::$marker, reference.nth)?.id() {
                    Id::$id(id) => Some(id),
                    _ => None,
                }
            }
        )*}
    };
}

resolvers! {
    speech_id: Speech -> Speech(SpeechId);
    hold_id: Hold -> Hold(HoldId);
    filler_id: Filler -> Filler(FillerId);
    hook_id: Hook -> Hook(HookId);
}

impl Run {
    /// One thing that happened in the room, done to the fake room: the real room io sees it, and
    /// decides what the session hears of it. Refused in a room nothing happens in (no room
    /// switches and no room legs), and once the room is gone.
    fn room_event(&mut self, event: RoomIn) -> bool {
        if (self.setup.room.is_none() && self.setup.room_legs.is_none())
            || self.room.connection_state() == ConnectionState::Disconnected
        {
            return false;
        }
        let who = lk_agents_rtc::ParticipantIdentity::from(super::start::CALLER);
        let reason = DisconnectReason::ClientInitiated;
        match event {
            RoomIn::Linked => {
                self.join_caller();
                true
            }
            RoomIn::Unlinked => {
                // Their microphone and camera leave with them.
                for nth in self.room_mics() {
                    self.mics.remove(&nth);
                    self.record.close(self.stamps.next(), Kind::Mic, nth);
                }
                let cameras: Vec<u32> = self
                    .record
                    .live()
                    .of(Kind::Video)
                    .into_iter()
                    .filter(|nth| {
                        self.record
                            .resource(Kind::Video, *nth)
                            .is_some_and(|r| *r.detail() == Detail::RoomCamera)
                    })
                    .collect();
                for nth in cameras {
                    self.record.close(self.stamps.next(), Kind::Video, nth);
                }
                self.room.remove_participant(&who, reason)
            }
            RoomIn::StateChanged => {
                self.room.set_connection_state(ConnectionState::Reconnecting);
                true
            }
            RoomIn::Closed => {
                self.room.close(reason);
                true
            }
            RoomIn::Chat => {
                let info = lk_agents_rtc::StreamInfo {
                    id: format!("chat_{}", self.record.now().seq),
                    topic: lk_agents_abi::TOPIC_CHAT.into(),
                    mime_type: "text/plain".into(),
                    total_length: None,
                    attributes: Default::default(),
                    timestamp: 0,
                    attachments: Vec::new(),
                };
                self.room.send_text_stream(who, info, ["can you hear me"]).is_ok()
            }
        }
    }

    /// The speech handle the host holds for a speech the run saw created.
    fn host_speech(&self, reference: Ref<crate::input::Speech>) -> Option<HostSpeech> {
        let handle = self.handles.get(&self.speech_id(reference)?.0)?.clone();
        Some(HostSpeech::from_core(handle, self.api.api().ok()?))
    }

    /// Whether the session's loop is still there to take a call that returns nothing.
    fn loop_runs(&self) -> bool {
        self.runtime.is_some()
    }

    /// Make the host API call an input mirrors, with the arguments a host passes.
    ///
    /// A few inputs have no host call: they are sent as the command their real sender sends
    /// (`send_from`), and say so where they are made.
    fn call(&mut self, at: Observed, input: &Input) -> bool {
        let api = self.api.clone();
        let registry = self.registry.clone();
        let mut waiting: Option<Pending> = None;
        let mut stream: Option<(bool, Feeding)> = None;
        let mut wait = |pending: Pending| waiting = Some(pending);

        let sent = match input {
            Input::Say { text: AgentText(text), input: say, allow, to_ctx } => {
                let speech = match say {
                    SayInput::Words => api.say(text.clone(), *allow, *to_ctx, None).ok(),
                    SayInput::TextStream => {
                        api.say_stream(*allow, *to_ctx, None).ok().map(|(speech, pusher)| {
                            stream = Some((false, Feeding::Text(pusher)));
                            speech
                        })
                    }
                    SayInput::Audio => api.say_audio(text.clone(), *allow, *to_ctx, None).ok().map(
                        |(speech, pusher)| {
                            stream = Some((true, Feeding::Audio { pusher, chunks: 0 }));
                            speech
                        },
                    ),
                };
                speech.map(|speech| wait(Pending::speech(speech))).is_some()
            }
            Input::GenerateReply { user_input, instructions, allow } => api
                .generate_reply(
                    user_input.clone().map(|UserText(t)| t),
                    instructions.clone().map(|AgentText(t)| t),
                    *allow,
                    None,
                    None,
                    None,
                    None,
                    None,
                    false,
                    None,
                )
                .map(|speech| wait(Pending::speech(speech)))
                .is_ok(),
            // `interrupt()` always returns its future; `wait` is whether the case awaits it.
            Input::Interrupt { force, wait: awaited } => api
                .interrupt(*force)
                .map(|reply| {
                    if *awaited {
                        wait(Pending::reply(reply, |r| {
                            r.map_or(Answer::InterruptRefused, |()| Answer::InterruptOk)
                        }))
                    }
                })
                .is_ok(),
            Input::InterruptSpeech { speech, force } => self
                .host_speech(*speech)
                .is_some_and(|speech| speech.interrupt(*force, self.tasks.clone()).is_ok()),
            Input::SetAllowInterruptions { speech, allow } => self
                .host_speech(*speech)
                .is_some_and(|speech| speech.set_allow_interruptions(*allow).is_ok()),
            Input::CommitUserTurn { transcript_timeout, stt_flush, skip_reply } => api
                .commit_user_turn(secs(*transcript_timeout), secs(*stt_flush), *skip_reply)
                .map(|reply| {
                    wait(Pending::reply(reply, |r| {
                        r.map_or_else(
                            |CommitSuperseded| Answer::CommitSuperseded,
                            Answer::Committed,
                        )
                    }))
                })
                .is_ok(),
            Input::WaitForIdle { hold } => {
                let hold = match hold.map(|r| self.hold_id(r)) {
                    Some(None) => return false,
                    given => given.flatten(),
                };
                api.wait_for_idle(hold.map(|h| h.0))
                    .map(|reply| {
                        wait(Pending::reply(reply, |r| {
                            r.map_or_else(Answer::IdleRefused, |()| Answer::IdleOk)
                        }))
                    })
                    .is_ok()
            }
            Input::WaitForIdleAndHold {} => api
                .wait_for_idle_and_hold(None)
                .map(|reply| {
                    wait(Pending::reply(reply, |r| {
                        r.map_or_else(Answer::IdleRefused, Answer::Hold)
                    }))
                })
                .is_ok(),
            Input::ReleaseIdleHold { hold } => {
                let Some(hold) = self.hold_id(*hold) else { return false };
                api.release_idle_hold(hold.0);
                self.loop_runs()
            }
            Input::ClaimUserTurn {} => api.claim_user_turn().is_ok(),
            Input::ReleaseUserTurn {} => {
                api.release_user_turn();
                self.loop_runs()
            }
            Input::SetRunOutput { speech, ok } => {
                let Some(speech) = self.host_speech(*speech) else { return false };
                let output = Some(self.host_ref());
                let (value, error) = if *ok { (output, None) } else { (None, output) };
                api.set_run_output(&speech, value, error).is_ok()
            }
            Input::ToolUpdate { tool, progress } => {
                let Some(call_id) = self.tool_call_id(*tool) else { return false };
                match progress {
                    Some(AgentText(m)) => {
                        api.tool_update(call_id, "progress", Some(m.clone()), Some(m.clone()), None)
                    }
                    None => api.tool_update(
                        call_id,
                        "final",
                        None,
                        None,
                        Some(Answered::Returned { value: None }),
                    ),
                }
                .is_ok()
            }
            Input::FillerStart { tool, speech, delay, interval, text } => {
                let (Some(call_id), Some(speech)) =
                    (self.tool_call_id(*tool), self.speech_id(*speech))
                else {
                    return false;
                };
                api.filler_start(
                    call_id,
                    &SpeechWireId(speech).to_string(),
                    secs(*delay),
                    interval.map(secs),
                    None,
                    text.clone().map(|AgentText(t)| t),
                    None,
                )
                .is_ok()
            }
            Input::FillerStop { filler } => {
                let Some(id) = self.filler_id(*filler) else { return false };
                api.filler_stop(id.0).is_ok()
            }
            // No host call: the answering-machine detector's own hold, sent as it sends it.
            Input::PauseAuthorization {} => {
                api.send_from(Origin::Amd, SpeechCommand::PauseAuthorization).is_ok()
            }
            Input::ResumeAuthorization {} => {
                api.send_from(Origin::Amd, SpeechCommand::ResumeAuthorization).is_ok()
            }
            // No host call: the commit `commit_user_turn` makes once its flush and final wait end.
            Input::CommitTurn { skip_reply } => {
                api.send(TurnCommand::Commit { skip_reply: *skip_reply }).is_ok()
            }
            Input::ClearTurn {} => api.clear_user_turn().is_ok(),
            Input::Configure { knob, mode, ms, on, words } => {
                let endpointing = |patch: fn(&mut EndpointingKwargs, f64)| {
                    let mut kwargs = EndpointingKwargs::default();
                    patch(&mut kwargs, secs(*ms));
                    Some(kwargs)
                };
                let update = match knob {
                    Knob::Detection => Some(UpdateKwargs {
                        turn_detection: Maybe::Given(
                            turn_detection(*mode)
                                .map_or(TurnDetectionKwarg::Null, TurnDetectionKwarg::Mode),
                        ),
                        ..UpdateKwargs::default()
                    }),
                    Knob::MinEndpointing => Some(UpdateKwargs {
                        endpointing_opts: endpointing(|k, s| k.min_delay = Some(s)),
                        ..UpdateKwargs::default()
                    }),
                    Knob::MaxEndpointing => Some(UpdateKwargs {
                        endpointing_opts: endpointing(|k, s| k.max_delay = Some(s)),
                        ..UpdateKwargs::default()
                    }),
                    _ => None,
                };
                match update {
                    Some(update) => api.update_turn_options(update).is_ok(),
                    // No host call: `update_options` takes endpointing and detection only.
                    None => api
                        .send(TurnCommand::Configure(patch(*knob, *mode, *ms, *on, *words)))
                        .is_ok(),
                }
            }
            Input::Handoff { agent, instructions, tools } => api
                .update_agent(frame(
                    &self.setup,
                    &self.legs,
                    &registry,
                    agent,
                    &instructions.0,
                    tools,
                ))
                .map(|reply| {
                    wait(Pending::reply(reply, |ok| {
                        if ok { Answer::HandoffOk } else { Answer::HandoffRefused }
                    }))
                })
                .is_ok(),
            Input::UpdateInstructions { text: AgentText(text) } => {
                api.update_instructions(instructions(text)).is_ok()
            }
            Input::UpdateTools { tools } => api
                .update_tools(
                    Some(tool_specs(tools)),
                    None,
                    Some(fakes::FakeTools::new(registry.clone())),
                )
                .is_ok(),
            Input::SetAudioInput { attach } => {
                // A new microphone replaces the old one as a detach does: nothing said into the
                // old one reaches the session any more.
                for (nth, _) in std::mem::take(&mut self.mics) {
                    self.record.close(at, Kind::Mic, nth);
                }
                self.host_mic = true;
                let source = attach.then(|| {
                    let (mic, frames) = fakes::FakeAudioInput::new();
                    let nth = self.registry.new_mic();
                    self.mics.insert(nth, (Mic::Host(frames), 0));
                    mic as Arc<dyn lk_agents_abi::voice::io::AudioInput>
                });
                api.set_audio_input(source).is_ok()
            }
            Input::SetVideoInput { attach } => api
                .set_video_input(
                    attach.then(|| fakes::FakeVideoInput::new(&registry, self.refs.clone()) as _),
                )
                .is_ok(),
            Input::SetIoEnabled { leg, enabled } => {
                api.set_io_enabled(io_leg(*leg), *enabled).is_ok()
            }
            Input::UpdateChatCtx { items, text: AgentText(text) } => {
                api.update_chat_ctx(history(items, text)).is_ok()
            }
            Input::SetAudioOutput { attach } => {
                let (pausable, lagging) = (self.setup.pausable_output, self.setup.lagging_output);
                api.set_audio_output(attach.then(|| {
                    fakes::FakeAudioOutput::new(registry.clone(), pausable, lagging) as _
                }))
                .is_ok()
            }
            Input::SetTranscriptionOutput { attach } => api
                .set_transcription_output(
                    attach.then(|| fakes::FakeTextOutput::new(registry.clone()) as _),
                )
                .is_ok(),
            Input::UpdateModels { models } => {
                let (stt, llm, tts) = match models {
                    Models::Stt => (true, false, false),
                    Models::Llm => (false, true, false),
                    Models::Tts => (false, false, true),
                    Models::LlmAndTts => (false, true, true),
                    Models::All => (true, true, true),
                };
                fn given<T: ?Sized>(
                    on: bool,
                    make: impl FnOnce() -> Arc<T>,
                ) -> Maybe<Option<Arc<T>>> {
                    if on { Maybe::Given(Some(make())) } else { Maybe::NotGiven }
                }
                let connect = Duration::from_millis(self.setup.connect_ms.unwrap_or(0));
                let (llm, tts, stt) = (
                    given(llm, || fakes::FakeLlm::new(registry.clone()) as Arc<Llm>),
                    given(tts, || fakes::FakeTts::new(registry.clone()) as Arc<Tts>),
                    given(stt, || session_stt(&self.setup, &registry, connect)),
                );
                // What a later handoff gives again.
                for (leg, held) in [(&llm, &mut self.legs.llm)] {
                    if let Maybe::Given(new) = leg {
                        *held = new.clone();
                    }
                }
                if let Maybe::Given(new) = &tts {
                    self.legs.tts = new.clone();
                }
                if let Maybe::Given(new) = &stt {
                    self.legs.stt = new.clone();
                }
                api.update_models(
                    llm,
                    tts,
                    stt,
                    Maybe::NotGiven,
                    None,
                    None,
                    None,
                )
                .is_ok()
            }
            // `drain()` always returns its future; `wait` is whether the case awaits it.
            Input::Drain { wait: awaited } => api
                .drain()
                .map(|reply| {
                    if *awaited {
                        wait(Pending::reply(reply, |refused| match refused.as_deref() {
                            None => Answer::DrainOk,
                            Some(r) if r == DrainRefused::ActivityChanging.as_str() => {
                                Answer::DrainRefused(DrainRefused::ActivityChanging)
                            }
                            Some(_) => Answer::DrainRefused(DrainRefused::Closing),
                        }))
                    }
                })
                .is_ok(),
            Input::PushVideo {} => api
                .host_push_video(lk_agents_abi::ChatContent::Image(ImageContent {
                    id: "img_case".into(),
                    image: ImageSource::Frame(MediaRef::new(self.record.now().seq, &self.refs)),
                    inference_width: None,
                    inference_height: None,
                    inference_detail: Default::default(),
                    mime_type: None,
                }))
                .unwrap_or(false),
            Input::Close { drain } => {
                self.record.enter_phase(Phase::Closing);
                let Ok(closing) = api.aclose("user_initiated", *drain, false) else { return false };
                // `await session.aclose()`, then the room io's teardown, as the host runs them.
                let mut closing: BoxFuture<'static, ()> = Box::pin(async move {
                    closing.await;
                    api.close_io().await;
                });
                // The first poll is what sends the close.
                if ready_now(closing.as_mut()).is_none() {
                    self.closing = Some(closing);
                }
                return self.loop_runs();
            }
            Input::HookDone { hook, answer, text: AgentText(text) } => {
                let Some(id) = self.hook_id(*hook) else { return false };
                self.answer_hook(id, *answer, text.clone())
            }
            // The hook is gone from the live set, but the record still knows its id: the answer
            // reaches the driver as a host's late answer does.
            Input::HookDoneLate { hook, answer, text: AgentText(text) } => {
                let Some(id) = self.hook_id(Ref::new(hook.nth)) else { return false };
                self.answer_hook(id, *answer, text.clone())
            }
            Input::HostMetrics {} => api.host_metrics(HostMetricsRecord(vad_metrics())).is_ok(),
            Input::Ivr { input, silence, text: UserText(text) } => match input {
                IvrInput::Start => api.ivr_start(secs(*silence), None, None).is_ok(),
                IvrInput::SeedTranscript => api.ivr_final(text).is_ok(),
                IvrInput::Close => api.ivr_close().is_ok(),
            },
            Input::SetStaticKeyterms { terms } => api.set_static_keyterms(terms.clone()).is_ok(),
            Input::Dtmf { input, key, digits, confirm, timeout } => match input {
                DtmfInput::Start => api
                    .dtmf_start(digits.0 as usize + 1, *confirm, secs(*timeout), "#".into())
                    .map(|reply| {
                        wait(Pending::reply(reply, |wire| Answer::Dtmf(dtmf_outcome(wire))))
                    })
                    .is_ok(),
                DtmfInput::Digit => api.dtmf_digit(key.digit().to_owned()).is_ok(),
                DtmfInput::Stop => {
                    api.dtmf_stop();
                    self.loop_runs()
                }
            },
            Input::Recorder { attach } => {
                api.host_attach_recorder(
                    attach.then(|| fakes::FakeRecorder::new(registry.clone()) as _),
                );
                self.loop_runs()
            }
            Input::Run { user_input, expects_output, retries } => api
                .run(
                    user_input.clone().map(|UserText(t)| t),
                    *expects_output,
                    retries.0,
                    "Please try again.".into(),
                    None,
                    None,
                )
                .is_ok(),
            _ => unreachable!("{input:?} is not a call"),
        };
        if !sent {
            return false;
        }
        if let Some((audio, feeding)) = stream {
            let nth = self.record.open(at, Kind::Stream, Id::Pending, Detail::Stream { audio });
            self.streams.insert(nth, feeding);
        }
        if let Some(waiting) = waiting {
            let nth = self.record.open(at, Kind::Answer, Id::Pending, Detail::None);
            self.pending.insert(nth, waiting);
        }
        true
    }

    /// The host's answer to a hook call, through the method each kind of answer is.
    fn answer_hook(&self, id: HookId, answer: HookAnswer, text: String) -> bool {
        let api = &self.api;
        let id = id.0;
        match answer {
            HookAnswer::Returned => api.hook_returned(id),
            HookAnswer::Failed => api.hook_failed(id),
            HookAnswer::TurnInstructions => api.hook_turn_done(id, Some(text), None, None, false),
            HookAnswer::TurnStop => api.hook_turn_done(id, None, None, None, true),
            HookAnswer::Rendered => api.hook_rendered(id, text),
            HookAnswer::Filler => api.hook_filler(id, Some(text), false),
        }
        .is_ok()
    }

    /// The live tool invocation running under this call id.
    pub(super) fn tool_by_call_id(&self, call_id: &str) -> Option<u32> {
        self.record.live().of(Kind::Tool).into_iter().find(|nth| {
            matches!(self.record.resource(Kind::Tool, *nth).map(|r| r.detail()),
                Some(Detail::Tool { call_id: id, .. }) if id == call_id)
        })
    }

    /// Whether this tool answered `Detached` and is still running: what the host ends with a
    /// final `ToolUpdate` rather than a second answer.
    pub fn tool_detached(&self, nth: u32) -> bool {
        matches!(self.controls.get(&Slot { kind: Kind::Tool, nth }), Some(Control::Running))
    }

    fn tool_call_id(&self, reference: Ref<crate::input::Tool>) -> Option<String> {
        match self.record.resource(Kind::Tool, reference.nth)?.detail() {
            Detail::Tool { call_id, .. } => Some(call_id.clone()),
            _ => None,
        }
    }

    /// The host's side of a `RunOutput`: an object the driver only ever hands back.
    fn host_ref(&self) -> HostRef {
        HostRef::new(self.record.now().seq, &self.refs)
    }
}

/// `dtmf_start`'s answer, read back from the record the host gets.
fn dtmf_outcome(wire: lk_agents_abi::voice::workflows::DtmfOutcomeWire) -> DtmfOutcome {
    use lk_agents_abi::voice::workflows::DtmfOutcomeWire as W;
    match wire {
        W::Complete(c) => DtmfOutcome::Complete(GetDtmfResult { user_input: c.user_input }),
        W::Incomplete(i) => DtmfOutcome::Incomplete { message: i.message },
        W::Stopped(_) => DtmfOutcome::Stopped,
    }
}

// ==============================================================================================
// Feeding a stream, answering a fake
// ==============================================================================================

/// A stream a `say` opened, until the case ends it with `FeedEnd`: the pusher the host feeds.
pub(super) enum Feeding {
    Text(TextPusher),
    Audio { pusher: FramePusher, chunks: u16 },
}

impl Run {
    /// Close every utterance still open on a recognition stream.
    pub(super) fn end_utterances_on(&mut self, at: Observed, stt: u32) {
        let open: Vec<u32> = self
            .utterances
            .iter()
            .filter(|(nth, on)| **on == stt && self.record.live().has(Kind::Utterance, **nth))
            .map(|(nth, _)| *nth)
            .collect();
        for nth in open {
            self.record.close(at, Kind::Utterance, nth);
        }
    }

    /// Push tagged frames into an attached microphone. The driver's audio leg forwards them to
    /// every provider listening, which is where they are recorded.
    fn speak(&mut self, mic: u32, chunks: u32) -> bool {
        let now = self.record.now().at_ms;
        let Some((leg, count)) = self.mics.get_mut(&mic) else { return false };
        // A track's queues drain only as the session's tasks run, which is when time moves: a burst
        // they cannot hold is dropped oldest-first, as a real track would, so it is refused whole.
        if let Mic::Room { burst, .. } = leg {
            if burst.0 != now {
                *burst = (now, 0);
            }
            if burst.1 + chunks > ROOM_BURST_FRAMES {
                return false;
            }
            burst.1 += chunks;
        }
        for _ in 0..chunks {
            let sample = Tag::mic(mic, *count % (Tag::MAX_CHUNK + 1));
            let data = sample.to_le_bytes().repeat(SAMPLES_PER_CHUNK);
            let frame = AudioFrame::new(data, SAMPLE_RATE, 1, SAMPLES_PER_CHUNK as u32);
            // A track nobody reads yet (the room io has not subscribed) carries the words nowhere.
            let heard = match leg {
                Mic::Host(sender) => sender.send(frame).is_ok(),
                Mic::Room { track, .. } => track.push_frame(frame) > 0,
                Mic::Console(cli) => cli.send(Message::AudioInput(console_io::AudioFrame {
                    data: frame.data.to_vec(),
                    sample_rate: frame.sample_rate,
                    num_channels: u32::from(frame.num_channels),
                    samples_per_channel: frame.samples_per_channel,
                })),
            };
            if !heard {
                return false;
            }
            *count = count.wrapping_add(1);
        }
        true
    }

    /// `AMD`'s calls past its attach, as `detector.py` makes them. `execute()` awaits the
    /// verdict, so it is refused before one; on `machine-ivr` under `ivr_detection` it starts the
    /// navigator with the greeting first (`_start_ivr_detection(transcript)`).
    fn amd(&mut self, input: AmdInput) -> bool {
        use lk_agents_abi::voice::amd::AmdCategory;
        let api = self.api.clone();
        match input {
            AmdInput::Listen => {
                api.amd_start_detection_timer().is_ok() && api.amd_start_listening().is_ok()
            }
            AmdInput::Settle => api.amd_settle("uncertain", "participant_missing").is_ok(),
            AmdInput::Execute => {
                let verdict =
                    self.record.host_view().events().into_iter().rev().find_map(|(_, event)| {
                        match event {
                            SessionEvent::AmdResult { verdict } => Some(verdict.clone()),
                            _ => None,
                        }
                    });
                let Some(verdict) = verdict else { return false };
                if verdict.category == AmdCategory::MachineIvr
                    && self.setup.amd.is_some_and(|amd| amd.ivr_detection)
                {
                    let dtmf = tool_specs(&["send_dtmf_events".to_owned()]);
                    if api.ivr_start(5.0, Some(dtmf), None).is_err()
                        || api.ivr_final(&verdict.transcript).is_err()
                    {
                        return false;
                    }
                }
                api.amd_execute().is_ok()
            }
            AmdInput::Close => api.amd_close().is_ok(),
        }
    }

    /// The CLI asks the session something, as its TUI does; refused with no console.
    fn console_ask(&mut self, ask: ConsoleAsk) -> bool {
        use proto::session_request::{self as rq, Request};
        let Some((_, cli)) = &self.console else { return false };
        let request = match ask {
            ConsoleAsk::Ping => Request::Ping(rq::Ping {}),
            ConsoleAsk::History => Request::GetChatHistory(rq::GetChatHistory {}),
            ConsoleAsk::AgentInfo => Request::GetAgentInfo(rq::GetAgentInfo {}),
            ConsoleAsk::RunInput => Request::RunInput(rq::RunInput { text: "hello".into() }),
            ConsoleAsk::State => Request::GetSessionState(rq::GetSessionState {}),
            ConsoleAsk::RtcStats => Request::GetRtcStats(rq::GetRtcStats {}),
            ConsoleAsk::Usage => Request::GetSessionUsage(rq::GetSessionUsage {}),
            ConsoleAsk::Framework => Request::GetFrameworkInfo(rq::GetFrameworkInfo {}),
            ConsoleAsk::Mute => Request::UpdateIo(rq::UpdateIo {
                input: Some(rq::update_io::Input { audio_enabled: Some(false), video_enabled: None }),
                output: None,
            }),
            ConsoleAsk::Finalize => Request::FinalizeSimulation(rq::FinalizeSimulation {
                provisional_success: true,
                provisional_reason: "done".into(),
            }),
        };
        self.next_request += 1;
        cli.send(Message::Request(proto::SessionRequest {
            request_id: format!("req_{}", self.next_request),
            request: Some(request),
        }))
    }

    /// The avatar worker and the agent hear what the other sent since the last look.
    pub(super) fn avatar_reports(&mut self) {
        if let Some(worker) = &self.avatar {
            worker.relay(&self.room, super::start::AVATAR);
        }
    }

    /// The person's microphones in the room, which leave the room with them.
    fn room_mics(&self) -> Vec<u32> {
        self.mics.iter().filter(|(_, (leg, _))| matches!(leg, Mic::Room { .. })).map(|(n, _)| *n).collect()
    }

    /// Push one piece of a `say`'s input stream. Each audio chunk carries its stream and its index
    /// in its samples, so what reaches the sink can be decoded back to what the case fed.
    ///
    /// A push is awaited by the host, and a full channel would make it wait: polled once here, it
    /// must be taken at once, and a channel too full to take it is the fixture's mistake.
    fn feed(&mut self, at: Observed, input: &Input) -> bool {
        match input {
            Input::FeedText { stream, text: AgentText(text) } => {
                let Some(Feeding::Text(pusher)) = self.streams.get_mut(&stream.nth) else {
                    return false;
                };
                ready_now(pusher.push(text.clone())).expect("fixture text input full")
            }
            Input::FeedAudio { stream, chunks: Count(count) } => {
                let nth = stream.nth;
                let Some(Feeding::Audio { pusher, chunks }) = self.streams.get_mut(&nth) else {
                    return false;
                };
                for _ in 0..*count {
                    let sample = Tag::fed(nth, *chunks);
                    let data = sample.to_le_bytes().repeat(SAMPLES_PER_CHUNK);
                    let frame = AudioFrame::new(data, SAMPLE_RATE, 1, SAMPLES_PER_CHUNK as u32);
                    if !ready_now(pusher.push(frame)).expect("fixture audio input full") {
                        return false;
                    }
                    *chunks += 1;
                }
                true
            }
            Input::FeedEnd { stream } => {
                match self.streams.remove(&stream.nth) {
                    Some(Feeding::Text(pusher)) => pusher.end(),
                    Some(Feeding::Audio { pusher, .. }) => pusher.end(),
                    None => return false,
                }
                self.record.close(at, Kind::Stream, stream.nth);
                true
            }
            _ => false,
        }
    }

    /// Hand one answer to the fake waiting for it. An utterance is reached through the recognition
    /// stream it was opened on, which is the only indirection here: the recogniser is the thing
    /// with a channel, and an utterance is a span of what the case sent down it.
    fn deliver(&mut self, at: Observed, input: &Input) -> bool {
        let (slot, utterance) = match input {
            Input::LlmText { chat, .. }
            | Input::LlmToolCall { chat, .. }
            | Input::LlmEnd { chat }
            | Input::LlmFail { chat } => (Slot { kind: Kind::Chat, nth: chat.nth }, None),
            Input::TtsHold { tts } | Input::TtsRelease { tts } | Input::TtsFail { tts } => {
                (Slot { kind: Kind::Tts, nth: tts.nth }, None)
            }
            Input::SttStart { stt } | Input::SttFail { stt } => {
                (Slot { kind: Kind::Stt, nth: stt.nth }, None)
            }
            Input::SttInterim { utterance, .. }
            | Input::SttPreflight { utterance, .. }
            | Input::SttFinal { utterance, .. }
            | Input::SttEnd { utterance } => {
                let Some(stt) = self.utterances.get(&utterance.nth).copied() else { return false };
                (Slot { kind: Kind::Stt, nth: stt }, Some(utterance.nth))
            }
            Input::VadInference { vad, .. }
            | Input::VadStart { vad, .. }
            | Input::VadEnd { vad, .. } => (Slot { kind: Kind::Vad, nth: vad.nth }, None),
            Input::EotAnswer { prediction, .. } | Input::EotFail { prediction } => {
                (Slot { kind: Kind::Prediction, nth: prediction.nth }, None)
            }
            Input::OverlapVerdict { port, .. } | Input::OverlapFail { port } => {
                (Slot { kind: Kind::Interruption, nth: port.nth }, None)
            }
            Input::ToolAnswered { tool, .. } => (Slot { kind: Kind::Tool, nth: tool.nth }, None),
            Input::PlayoutStarted { segment }
            | Input::PlayoutFinished { segment }
            | Input::PlayoutProgressed { segment, .. } => {
                (Slot { kind: Kind::Segment, nth: segment.nth }, None)
            }
            Input::SinkReset { output } => (Slot { kind: Kind::Output, nth: output.nth }, None),
            Input::VideoFrames { video, .. } => (Slot { kind: Kind::Video, nth: video.nth }, None),
            Input::RealtimeEvent { socket, .. } => {
                (Slot { kind: Kind::Realtime, nth: socket.nth }, None)
            }
            _ => return false,
        };
        if !self.record.live().has(slot.kind, slot.nth) || !self.deliver_to(slot, input) {
            return false;
        }
        match input {
            // An utterance runs from the start of speech to whatever ends it; the recogniser keeps
            // its stream either way, which is why only the utterance closes here.
            // A new start of speech on a stream supersedes whatever utterance was still open on it.
            Input::SttStart { .. } => {
                self.end_utterances_on(at, slot.nth);
                let nth = self.record.open(at, Kind::Utterance, Id::Slot(slot.nth), Detail::None);
                self.utterances.insert(nth, slot.nth);
            }
            // A final transcript does not end the utterance: a real provider sends its
            // end-of-speech after the final, and the driver must see both in that order.
            Input::SttEnd { .. } => {
                if let Some(nth) = utterance {
                    self.record.close(at, Kind::Utterance, nth);
                }
            }
            Input::SttFail { .. } => self.end_utterances_on(at, slot.nth),
            _ => {}
        }
        true
    }

    /// Hand one input to the fake behind a slot, through the control this run owns.
    ///
    /// No lock is taken here and none is held inside: the fake writes what it did to a channel
    /// this run drains afterwards. A fake recording from inside the call it is being given is
    /// therefore ordinary, which is the whole reason the controls live here and not beside the
    /// fakes' own records.
    fn deliver_to(&mut self, slot: Slot, input: &Input) -> bool {
        if fakes::boundary_of(input) != Some(slot.kind) {
            return false;
        }
        let Some(control) = self.controls.remove(&slot) else {
            return false;
        };
        let (accepted, keep) = self.registry.deliver(slot, control, input);
        if let Some(control) = keep {
            self.controls.insert(slot, control);
        }
        accepted
    }
}

// ==============================================================================================
// Waiting
// ==============================================================================================

/// How long one awaited thing gets, in virtual milliseconds. Not a production SLA: every fake
/// answers the moment it is released, so anything still pending after this is stuck.
pub const BUDGET_MS: u64 = 30_000;

/// When a close that has not finished never will: a virtual day, where the longest close the
/// phases allow is under a minute. The paused clock jumps to the next armed timer whenever nothing
/// can run, so a session left idle with nothing armed to wake it gets here at once, for free.
const NEVER_MS: u64 = 86_400_000;

/// How many times closing will wait for a fresh batch of tasks before giving up on the driver.
const MAX_JOIN_ROUNDS: u32 = 100;

impl Run {
    async fn await_answer(&mut self, nth: u32) -> bool {
        let Some(mut waiting) = self.pending.remove(&nth) else { return false };
        match self.drive(Some(self.budget()), waiting.recv()).await {
            Some(settled) => {
                self.record_answer(nth, settled);
                true
            }
            None => {
                self.pending.insert(nth, waiting);
                false
            }
        }
    }

    async fn await_speech(&mut self, nth: u32) -> bool {
        let Some(resource) = self.record.resource(Kind::Speech, nth) else { return false };
        let Id::Speech(speech) = resource.id() else { return false };
        let Some(handle) = self.handles.get(&speech.0).cloned() else { return false };
        let Some(api) = self.api.api().ok() else { return false };
        let host = Arc::new(HostSpeech::from_core(handle, api));
        // `await handle.wait_for_generation()`, `await handle`, then what a program reads off it.
        let waited = host.clone();
        let done = self.drive(Some(self.budget()), async move {
            let _ = waited.wait_for_generation(None).await;
            waited.wait_for_playout().await
        });
        if done.await.is_none() {
            return false;
        }
        let _ = (host.id(), host.num_steps(), host.chat_items(), host.scheduled());
        let _ = (host.interrupted(), host.allow_interruptions(), host.exception());
        true
    }

    /// Before a close nothing obliges the session to exit, so that wait is an ordinary one. Once a
    /// close is begun it is waited out however long it takes: its phases are each bounded, and
    /// virtual time is free.
    ///
    /// Once the loop has exited, the host's `aclose()` and `close_io()` are waited out too, as a
    /// host awaits them; each of their waits is bounded on the session clock.
    async fn await_exit(&mut self) -> bool {
        let wait = if self.record.close_begun().is_some() { NEVER_MS } else { BUDGET_MS };
        let until = self.time.now().saturating_add(Duration::from_millis(wait));
        self.drive(Some(until), std::future::pending::<()>()).await;
        let exited = self.runtime.is_none();
        // The player goes with the session (`BackgroundAudioPlayer.aclose()`).
        if exited && let Some(player) = self.background.take() {
            player.close().await;
        }
        if exited && let Some(worker) = self.avatar.take() {
            worker.close();
        }
        if exited {
            if let Some(closing) = self.closing.take() {
                let until = self.time.now().saturating_add(Duration::from_millis(NEVER_MS));
                tokio::select! {
                    () = closing => {}
                    () = self.time.wait_until(until) => {}
                }
            }
            self.record.enter_phase(Phase::Closed);
        }
        exited
    }

    /// How long one awaited thing may take from now.
    pub(super) fn budget(&self) -> Ts {
        self.time.now().saturating_add(Duration::from_millis(BUDGET_MS))
    }

    /// Step the session until `done` resolves, virtual time reaches `until`, or the driver is gone.
    /// What reached the queue is pending; a due wake is delivered before any same-instant input;
    /// then the policy chooses between the next pending event and a scheduling round for the
    /// driver's ready tasks. Once the driver stopped it finishes and is dropped, as a session's
    /// loop is, so what waited on it is released. `None` unless `done` resolved.
    pub(super) async fn drive<T>(
        &mut self,
        until: Option<Ts>,
        done: impl std::future::Future<Output = T>,
    ) -> Option<T> {
        let mut done = std::pin::pin!(done);
        // Iterations at one virtual instant: an event, a wake, frames or a task's turn each.
        let mut instant = (self.time.now(), 0u32);
        loop {
            tokio::select! {
                biased;
                value = &mut done => return Some(value),
                () = std::future::ready(()) => {}
            }
            let now = self.time.now();
            instant = if now == instant.0 { (now, instant.1 + 1) } else { (now, 0) };
            let counted = (instant.1 >= NO_PROGRESS_STEPS).then(|| {
                format!(
                    "{NO_PROGRESS_STEPS} steps of the session at {} ms without virtual time moving",
                    self.stamps.now().at_ms
                )
            });
            let observed = self.livelock.lock().unwrap().take();
            if let Some(detail) = observed.or(counted) {
                self.record.set_livelock(detail);
                self.drop_runtime();
            }
            let Some(runtime) = &mut self.runtime else {
                // The host's loop outlives the session's: what is ready still runs, so a task the
                // driver cancelled on its way out is gone before the case sends anything else.
                // Bounded, as the session's own instants are: a task left spinning never stops.
                for _ in 0..NO_PROGRESS_STEPS {
                    if !tasks_runnable() {
                        break;
                    }
                    one_task_turn().await;
                }
                return None;
            };
            while let Ok(event) = runtime.queue.try_recv() {
                self.queue.push(event);
            }
            // Frames are not scheduled: each reaches the data plane as it arrives.
            if !runtime.frames.is_empty() {
                let now = self.time.read();
                self.step(|runtime| {
                    while let Ok(frame) = runtime.frames.try_recv() {
                        runtime.on_frame(frame, now);
                    }
                });
                continue;
            }
            if runtime.driver.stopped() {
                let undelivered = self.queue.drain();
                self.step(|runtime| runtime.finish(undelivered));
                self.drop_runtime();
                continue;
            }
            let now = self.time.now();
            let wake_at = runtime.driver.wake_at();
            if wake_at.is_some_and(|at| at <= now) {
                let now = self.time.read();
                self.step(|runtime| runtime.handle(Event::Wake, now));
                continue;
            }
            match self.queue.next(tasks_runnable()) {
                Some(Next::Event(event)) => {
                    let now = self.time.read();
                    self.step(|runtime| runtime.handle(event, now));
                    continue;
                }
                Some(Next::Tasks) => {
                    one_task_turn().await;
                    continue;
                }
                None => {}
            }
            if until.is_some_and(|until| now >= until) {
                return None;
            }
            let next = [wake_at, until].into_iter().flatten().min();
            tokio::select! {
                biased;
                value = &mut done => return Some(value),
                () = self.time.wait_until_opt(next) => {}
                Some(frame) = runtime.frames.recv() => {
                    let now = self.time.read();
                    self.step(|runtime| runtime.on_frame(frame, now));
                }
                event = runtime.queue.recv() => match event {
                    Some(event) => self.queue.push(event),
                    None => return None,
                },
            }
        }
    }

    /// One call into the session's cores. A panic is a finding about them, recorded, and the end
    /// of them: they are dropped.
    fn step(&mut self, call: impl FnOnce(&mut Runtime)) {
        let Some(runtime) = &mut self.runtime else { return };
        let called = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| call(runtime)));
        // The observer unwound out of a livelocked step: the loop reports it, not as a panic.
        if self.livelock.lock().unwrap().is_some() {
            self.drop_runtime();
        } else if let Err(panic) = called {
            self.record.set_driver_panic(panic_message(&*panic));
            self.drop_runtime();
        }
    }

    /// The cores are gone, as production drops them: the session's answer pump closes every
    /// call never answered.
    fn drop_runtime(&mut self) {
        self.runtime = None;
    }

    /// Await every retained task, and record how each ended.
    ///
    /// A task may spawn another before finishing, so this loops -- and because the work is the
    /// product's, the loop can fail to terminate. It is bounded for the same reason the release
    /// loop is: an unbounded loop here is a hang, not a slow run, and a driver that keeps
    /// spawning while closing is a result worth reporting rather than waiting on forever.
    ///
    /// Each batch is raced against a deadline a budget away, the session stepped meanwhile as its
    /// own loop would be: a session whose close never finished, frozen, would hold every task it
    /// ends. The clock is paused and moves only
    /// when nothing can run, so the deadline passes exactly when the tasks still pending can never
    /// finish: those are recorded as stuck and aborted, and the run completes.
    pub async fn join_tasks(&mut self) {
        // A livelocked session may have left a task spinning, which no deadline would outrun: the
        // livelock is the finding, and the tasks are aborted with the run.
        if self.record.livelock().is_some() {
            return;
        }
        for _ in 0..MAX_JOIN_ROUNDS {
            let batch = std::mem::take(&mut *self.tasks.spawned.lock().unwrap());
            if batch.is_empty() {
                return;
            }
            let deadline = self.budget();
            for Spawned { at, site, mut task } in batch {
                let joined = match self.drive(Some(deadline), &mut task).await {
                    Some(joined) => Some(joined),
                    // The session has exited: only its tasks are left to run.
                    None => tokio::select! {
                        joined = &mut task => Some(joined),
                        () = self.time.wait_until(deadline) => None,
                    },
                };
                let end = match joined {
                    Some(Err(error)) if error.is_panic() => {
                        TaskEnd::Panicked(panic_message(&*error.into_panic()))
                    }
                    Some(_) => TaskEnd::Finished,
                    None => {
                        task.abort();
                        TaskEnd::Stuck
                    }
                };
                self.record.record_task(Task { spawned: at, site, end });
            }
        }
        // Still spawning after every round: what is left is not going to finish either.
        for Spawned { at, site, task } in std::mem::take(&mut *self.tasks.spawned.lock().unwrap()) {
            let end = if task.is_finished() { TaskEnd::Finished } else { TaskEnd::Stuck };
            task.abort();
            self.record.record_task(Task { spawned: at, site, end });
        }
    }
}

/// What a panic said, when it said it with a string.
fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_default()
}

/// Let the runtime poll one task, then come back: the run's future wakes itself and, with an event
/// interval of one, is polled again after the next task. `yield_now` defers its wake until every
/// ready task has run, which would never let the driver step between two of them.
async fn one_task_turn() {
    let mut turned = false;
    std::future::poll_fn(|cx| {
        if std::mem::replace(&mut turned, true) {
            return std::task::Poll::Ready(());
        }
        cx.waker().wake_by_ref();
        std::task::Poll::Pending
    })
    .await
}

/// Whether a task the driver spawned is ready to run: a turn would move it.
fn tasks_runnable() -> bool {
    let metrics = tokio::runtime::Handle::current().metrics();
    metrics.global_queue_depth() + metrics.worker_local_queue_depth(0) > 0
}
