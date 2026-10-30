//! One input into what the product sees: a call on the host API, a stream's next piece, a fake's
//! answer, something in the room, or a wait.
//!
//! Every input is recorded, taken or refused. A reference that no longer resolves -- a speech the
//! driver has forgotten, a hook already finished -- makes the input a refused attempt, and so does
//! a call the API refuses. Nothing is invented to stand in for it, because a call carrying an id
//! the driver never issued is a different test from the one the case asked for.
use lk_agents_abi::voice::speech_handle::SpeechHandle as HostSpeech;
use std::task::{Context, Poll, Waker};

use lk_agents_abi::voice::rtc::{self, ConnectionState, DisconnectReason, RtcRoom};
use lk_agents_abi::voice::session::{VoiceSession as _, VoiceTaskEnded};
use lk_agents_abi::voice::session_options::UpdateKwargs;
use lk_agents_voice::agent_session::{
    Command, CommitSuperseded, DrainRefused, IvrCommand,
};

use lk_agents_voice::agent_activity::scheduling::SpeechKind;
use lk_agents_voice::workflows::dtmf::{DtmfOutcome, GetDtmfResult};

use super::start::{
    Llm, Tts, frame, history, patch, session_stt, tool_specs, turn_detection, vad_metrics,
};
use super::*;
mod wait;

pub use wait::BUDGET_MS;

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
    pub(super) fn of(answer: Answer) -> Self {
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

    /// `await HostSpeech::wait_until_created(&*handle)`: the speech, or the handle's refusal when the loop
    /// dropped it uncreated.
    fn speech(speech: Arc<SpeechHandle>) -> Self {
        Pending(Box::pin(async move {
            match HostSpeech::wait_until_created(&*speech).await {
                Ok(()) => Settled { answer: Answer::Speech(speech.id().0), handle: Some(speech) },
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
            || self.room.connection_state().expect("fake state") == ConnectionState::Disconnected
        {
            return false;
        }
        let who = rtc::ParticipantIdentity::from(super::start::CALLER);
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
            RoomIn::Reconnected => self.room.reconnected(),
            // Someone else on the call, while the person is: never the one the session listens
            // to, so the person's link and microphone stay where they are.
            RoomIn::GuestJoined => {
                let guest = rtc::ParticipantIdentity::from(super::start::GUEST);
                if self.room.fake_participant(&who).is_none()
                    || self.room.fake_participant(&guest).is_some()
                {
                    return false;
                }
                self.room
                    .add_participant(guest, rtc::ParticipantKind::Standard)
                    .publish_microphone("TR_guest_mic");
                true
            }
            RoomIn::GuestLeft => {
                let guest = rtc::ParticipantIdentity::from(super::start::GUEST);
                self.room.remove_participant(&guest, DisconnectReason::ClientInitiated)
            }
            RoomIn::Closed => {
                for (leg, _) in self.mics.values_mut() {
                    if let Mic::Room { read, .. } = leg {
                        *read = None;
                    }
                }
                self.room.close(reason);
                true
            }
            RoomIn::Resubscribed => {
                use rtc::TrackPublication as _;
                let now = self.record.now().at_ms;
                let mut any = false;
                for (leg, _) in self.mics.values_mut() {
                    let Mic::Room { track, read, .. } = leg else { continue };
                    let _ = ready_now(track.set_subscribed(false));
                    let _ = ready_now(track.set_subscribed(true));
                    if read.is_some() {
                        *read = Some(now);
                    }
                    any = true;
                }
                any
            }
            RoomIn::Chat => {
                let info = rtc::StreamInfo {
                    id: format!("chat_{}", self.record.now().seq),
                    topic: lk_agents_abi::TOPIC_CHAT.into(),
                    mime_type: "text/plain".into(),
                    ..Default::default()
                };
                self.room.send_text_stream(who, info, ["can you hear me"]).is_ok()
            }
        }
    }

    /// The speech handle the host holds for a speech the run saw created.
    fn host_speech(&self, reference: Ref<crate::input::Speech>) -> Option<Arc<SpeechHandle>> {
        let handle = self.handles.get(&self.speech_id(reference)?.0)?.clone();
        self.api.api().ok()?;
        Some(handle)
    }

    /// Whether the session's loop is still there to take a call that returns nothing.
    fn loop_runs(&self) -> bool {
        self.session.is_some()
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
        // The tool body a `VoiceTask.run()` is awaited in.
        let mut task_from: Option<String> = None;
        // A DTMF collection's await, which its task's end drops.
        let mut collecting = false;
        // The handle `say()` / `generate_reply()` returned: the host holds it, uncreated, at once.
        let mut asked: Option<(Arc<SpeechHandle>, SpeechKind)> = None;

        let sent = match input {
            Input::Say { text: AgentText(text), input: say, allow, to_ctx } => {
                let speech = match say {
                    SayInput::Words => api.say(text.clone(), *allow, *to_ctx, None).ok(),
                    SayInput::TextStream => {
                        api.say_text_channel(*allow, *to_ctx, None).ok().map(|(speech, pusher)| {
                            stream = Some((false, Feeding::Text(pusher)));
                            speech
                        })
                    }
                    SayInput::Audio => api
                        .say_audio_channel(text.clone(), *allow, *to_ctx, None)
                        .ok()
                        .map(|(speech, pusher)| {
                            stream = Some((true, Feeding::Audio { pusher, chunks: 0 }));
                            speech
                        }),
                };
                speech
                    .map(|speech| {
                        asked = Some((Arc::clone(&speech), SpeechKind::Say));
                        wait(Pending::speech(speech))
                    })
                    .is_some()
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
                .map(|speech| {
                    asked = Some((Arc::clone(&speech), SpeechKind::GenerateReply));
                    wait(Pending::speech(speech))
                })
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
                .is_some_and(|speech| HostSpeech::interrupt(&*speech, *force).is_ok()),
            Input::SetAllowInterruptions { speech, allow } => {
                self.host_speech(*speech).is_some_and(|speech| {
                    HostSpeech::set_allow_interruptions(&*speech, *allow).is_ok()
                })
            }
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
            Input::ToolUpdate { tool, progress, end } => {
                let Some(call_id) = self.tool_call_id(*tool) else { return false };
                let outcome = match progress {
                    Some(AgentText(m)) => {
                        ToolOutcome::Update { message: m.clone(), raw: m.clone() }
                    }
                    // A later update is progress, however it was drawn; the end is anything else.
                    None if *end == crate::input::ToolAnswer::Detached => ToolOutcome::Nothing,
                    None => crate::fakes::tools::outcome_of(*end, "the tool finished"),
                };
                api.tool_update(call_id, outcome.into()).is_ok()
            }
            Input::FillerStart { tool, speech, delay, interval, text } => {
                let (Some(call_id), Some(speech)) =
                    (self.tool_call_id(*tool), self.speech_id(*speech))
                else {
                    return false;
                };
                api.filler_start(
                    call_id,
                    speech.to_string(),
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
                api.update_instructions(text.clone()).is_ok()
            }
            Input::UpdateTools { tools } => api
                .update_tools(
                    Some(tool_specs(tools)),
                    None,
                    Some(fakes::frame(&registry, "tools", tools)),
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
                api.set_io_enabled(io_leg(*leg).into(), *enabled).is_ok()
            }
            Input::UpdateChatCtx { items, text: AgentText(text) } => {
                api.update_chat_ctx(history(items, text)).is_ok()
            }
            Input::SetAudioOutput { attach } => {
                let (pausable, lagging) = (self.setup.pausable_output, self.setup.lagging_output);
                api.set_audio_output(
                    attach.then(|| {
                        fakes::FakeAudioOutput::new(registry.clone(), pausable, lagging) as _
                    }),
                )
                .is_ok()
            }
            Input::SetTranscriptionOutput { attach } => api
                .set_transcription_output(attach.then(|| {
                    fakes::FakeTextOutput::new(registry.clone(), self.setup.text_output_raises) as _
                }))
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
                    given(tts, || {
                        let latency = self.setup.tts_latency.map(|seed| !seed);
                        fakes::FakeTts::new(
                            registry.clone(),
                            latency,
                            self.setup.timing(),
                            !self.setup.chunked_tts,
                            self.setup.tts_alignment,
                        ) as Arc<Tts>
                    }),
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
                api.update_models(llm, tts, stt, Maybe::NotGiven, None, None, None).is_ok()
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
            Input::PushVideo {} => api.push_video(ImageContent {
                id: "img_case".into(),
                image: ImageSource::Frame(MediaRef::new(self.record.now().seq, &self.refs)),
                inference_width: None,
                inference_height: None,
                inference_detail: Default::default(),
                mime_type: None,
            }),
            Input::Close { drain } => {
                self.record.enter_phase(Phase::Closing);
                let Ok(closing) = api.aclose("user_initiated".into(), *drain) else { return false };
                // `await session.aclose()`, the room io's teardown included.
                let mut closing: BoxFuture<'static, ()> = Box::pin(closing);
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
            Input::HostMetrics {} => api.host_metrics(vad_metrics()).is_ok(),
            Input::Ivr { input, silence, text: UserText(text) } => match input {
                IvrInput::Start => api.ivr_start(secs(*silence), None, None).is_ok(),
                IvrInput::SeedTranscript => api.ivr_final(text.clone()).is_ok(),
                IvrInput::Close => api.send(IvrCommand::Close).is_ok(),
            },
            Input::SetStaticKeyterms { terms } => {
                api.send(Command::SetStaticKeyterms(terms.clone())).is_ok()
            }
            Input::Dtmf { input, key, digits, confirm, timeout } => match input {
                DtmfInput::Start => {
                    collecting = true;
                    api.collect_dtmf(digits.0 as usize + 1, *confirm, secs(*timeout), "#".into())
                        .map(|reply| wait(Pending::reply(reply, |answer| Answer::Dtmf(dtmf_outcome(answer)))))
                        .is_ok()
                }
                // The caller presses it on their keypad; only a room carries it.
                DtmfInput::Digit => {
                    let in_room = self.setup.room.is_some() || self.setup.room_legs.is_some();
                    if in_room {
                        self.room.press_dtmf(key.digit());
                    }
                    in_room
                }
                // `GetDtmfTask` ends: its await of the collection is dropped.
                DtmfInput::Stop => {
                    if let Some(nth) = self.dtmf_await.take()
                        && self.pending.remove(&nth).is_some()
                    {
                        self.record_abandoned(nth);
                    }
                    self.loop_runs()
                }
            },
            Input::Recorder { attach } => {
                api.attach_recorder(
                    attach.then(|| fakes::FakeRecorder::new(registry.clone()) as _),
                );
                self.loop_runs()
            }
            Input::VoiceTask { tool, agent, reply } => {
                let Some(call_id) = self.tool_call_id(*tool) else { return false };
                let tools = self.setup.tools.clone();
                let instructions = format!("You are {agent}.");
                let args = frame(&self.setup, &self.legs, &registry, agent, &instructions, &tools);
                let origin = format!("tool:{call_id}");
                task_from = Some(call_id);
                api.run_voice_task(args, Some(origin), *reply)
                    .map(|reply| {
                        wait(Pending::reply(reply, |r| Answer::VoiceTask(r.map(|v| v.id()))))
                    })
                    .is_ok()
            }
            Input::Run { user_input } => {
                api.run(user_input.clone().map(|UserText(t)| t), None, None).is_ok()
            }
            _ => unreachable!("{input:?} is not a call"),
        };
        if !sent {
            return false;
        }
        if let Some((handle, kind)) = asked {
            let id = handle.reserved().expect("a host call's handle has its id");
            self.handles.entry(id.0).or_insert(handle);
            let detail = Detail::Speech { kind, user_initiated: true };
            self.open_once(at, Kind::Speech, Id::Speech(id), detail);
        }
        if let Some((audio, feeding)) = stream {
            let nth = self.record.open(at, Kind::Stream, Id::Pending, Detail::Stream { audio });
            self.streams.insert(nth, feeding);
        }
        if let Some(waiting) = waiting {
            let nth = self.record.open(at, Kind::Answer, Id::Pending, Detail::None);
            self.pending.insert(nth, waiting);
            if let Some(call_id) = task_from {
                self.task_awaits.insert(call_id, nth);
            }
            if collecting {
                self.dtmf_await = Some(nth);
            }
        }
        true
    }

    /// The body of tool `nth` awaits a `VoiceTask.run()` not answered yet.
    pub fn awaits_task(&self, nth: u32) -> bool {
        let call_id = self.tool_call_id(Ref::new(nth));
        let answer = call_id.and_then(|c| self.task_awaits.get(&c));
        answer.is_some_and(|nth| self.pending.contains_key(nth))
    }

    /// The body of tool `nth` already ran a `VoiceTask`: it returns now.
    pub fn awaited_task(&self, nth: u32) -> bool {
        self.tool_call_id(Ref::new(nth)).is_some_and(|c| self.task_awaits.contains_key(&c))
    }

    /// The body of `tool` ends while it still awaits a `VoiceTask.run()`: the await is dropped,
    /// which ends the task, and the body never sees an answer.
    fn drop_task_await(&mut self, tool: Ref<crate::input::Tool>) {
        let Some(call_id) = self.tool_call_id(tool) else { return };
        let Some(nth) = self.task_awaits.remove(&call_id) else { return };
        if self.pending.remove(&nth).is_some() {
            self.record_abandoned(nth);
        }
    }

    /// The host's answer to a hook call, as the command it is.
    fn answer_hook(&self, id: HookId, answer: HookAnswer, text: String) -> bool {
        use lk_agents_voice::agent::TurnEdit;
        use lk_agents_voice::agent_session::HookOutcome;
        let outcome = match answer {
            HookAnswer::Returned => HookOutcome::Returned,
            HookAnswer::Failed => HookOutcome::Failed,
            HookAnswer::TurnInstructions => {
                HookOutcome::Turn(vec![TurnEdit::UpdateInstructions(text)])
            }
            HookAnswer::TurnStop => HookOutcome::Turn(vec![TurnEdit::StopResponse]),
            HookAnswer::Filler => HookOutcome::Filler { text: Some(text), spoke: false },
        };
        self.api.send(Command::HookDone { id, outcome }).is_ok()
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
}

/// `collect_dtmf`'s answer, read back as the host gets it.
fn dtmf_outcome(answer: Result<String, VoiceTaskEnded>) -> DtmfOutcome {
    match answer {
        Ok(user_input) => DtmfOutcome::Complete(GetDtmfResult { user_input }),
        Err(VoiceTaskEnded::Failed(message)) => DtmfOutcome::Incomplete { message },
        Err(_) => DtmfOutcome::Stopped,
    }
}

// ==============================================================================================
// Feeding a stream, answering a fake
// ==============================================================================================

/// A stream a `say` opened, until the case ends it with `FeedEnd`: the pusher the host feeds.
pub(super) enum Feeding {
    Text(Pusher<String>),
    Audio { pusher: Pusher<AudioFrame>, chunks: u16 },
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
                // A track the room io read, and had its turn to read again since it was last
                // subscribed, carries the words even when nothing reads them: the agent went deaf,
                // which the rules are to see.
                Mic::Room { track, read, .. } => {
                    use rtc::TrackPublication as _;
                    if track.push_frame(frame) > 0 {
                        read.get_or_insert(now);
                        true
                    } else {
                        track.is_subscribed().expect("fake subscription")
                            && read.is_some_and(|since| now > since)
                    }
                }
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

    /// `AMD`'s calls past its attach, as `detector.py` makes them, and the SIP callee's answer
    /// it awaits. `execute()` awaits the
    /// verdict, so it is refused before one; on `machine-ivr` under `ivr_detection` it starts the
    /// navigator with the greeting first (`_start_ivr_detection(transcript)`).
    fn amd(&mut self, input: AmdInput) -> bool {
        use lk_agents_abi::voice::amd::{
            AmdCategory, SIP_CALL_STATUS_ACTIVE, SIP_CALL_STATUS_ATTR,
        };
        use rtc::RemoteParticipant;
        let api = self.api.clone();
        match input {
            // The SIP callee picks up; refused when the callee is not a SIP call in the room.
            AmdInput::Answer => {
                let callee = rtc::ParticipantIdentity::from(super::start::CALLER);
                let Some(callee) = self.room.fake_participant(&callee) else { return false };
                if callee.kind() != rtc::ParticipantKind::Sip {
                    return false;
                }
                callee.set_attribute(SIP_CALL_STATUS_ATTR, SIP_CALL_STATUS_ACTIVE);
                true
            }
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
                        || api.ivr_final(verdict.transcript.clone()).is_err()
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
        let Some(cli) = &self.console else { return false };
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
                input: Some(rq::update_io::Input {
                    audio_enabled: Some(false),
                    video_enabled: None,
                }),
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
            for (at, step) in worker.drain() {
                self.record.record_avatar(at, step);
            }
        }
    }

    /// The person's microphones in the room, which leave the room with them.
    fn room_mics(&self) -> Vec<u32> {
        self.mics
            .iter()
            .filter(|(_, (leg, _))| matches!(leg, Mic::Room { .. }))
            .map(|(n, _)| *n)
            .collect()
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
    /// stream it was opened on: the recogniser has the channel, an utterance spans what it carried.
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
            | Input::SttUsage { utterance }
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
            Input::ToolAnswered { tool, .. } => {
                self.drop_task_await(*tool);
                (Slot { kind: Kind::Tool, nth: tool.nth }, None)
            }
            Input::PlayoutStarted { segment }
            | Input::PlayoutFinished { segment }
            | Input::PlayoutFinishLost { segment }
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
