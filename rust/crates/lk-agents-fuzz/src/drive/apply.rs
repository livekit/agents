//! One input into what the product sees: a command, a stream's next piece, a fake's answer, or a
//! wait.
//!
//! Every input is recorded, taken or refused. A reference that no longer resolves -- a speech the
//! driver has forgotten, a hook already finished -- makes the input a refused attempt. Nothing is
//! invented to stand in for it, because a command carrying an id the driver never issued is a
//! different test from the one the case asked for.
use super::start::{
    Llm, Stt, Tts, frame, given, history, instructions, patch, toolset, vad_metrics,
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
        if std::env::var_os("VOICE_TRACE").is_some() {
            eprintln!("> {input:?}");
        }
        let at = self.stamps.next();
        // The synchroniser closes with the session, as the session layer closes its own: its
        // pacing task must be gone before the driver's close is judged for what it left behind.
        if let Input::Close { .. } = input
            && let Some(sync) = self.sync.take()
        {
            self.drive(Some(self.budget()), async move { sync.aclose().await }).await;
        }
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
// Sending a command
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
    /// The channel closed without an answer. Recorded as an outcome, because a caller left
    /// waiting forever is exactly what several rules are looking for.
    fn lost() -> Self {
        Settled::of(Answer::Lost)
    }
}

/// One real answer receiver per command that carries one, kept until it answers or closes. Which
/// variant it is decides how the value is spelled in the recording, and nothing else -- so the
/// table below is the whole of it, and both the polling and the awaiting form come from it.
macro_rules! answers {
    ($($variant:ident($carried:ty) $value:pat => $settled:expr;)*) => {
        pub(super) enum Pending { $($variant(oneshot::Receiver<$carried>)),* }

        impl Pending {
            pub(super) fn try_recv(&mut self) -> Option<Settled> {
                use oneshot::error::TryRecvError::*;
                match self {$(
                    Pending::$variant(rx) => match rx.try_recv() {
                        Ok($value) => Some($settled),
                        Err(Empty) => None,
                        Err(Closed) => Some(Settled::lost()),
                    },
                )*}
            }

            pub(super) async fn recv(&mut self) -> Settled {
                match self {$(
                    Pending::$variant(rx) => match rx.await {
                        Ok($value) => $settled,
                        Err(_) => Settled::lost(),
                    },
                )*}
            }
        }
    };
}

answers! {
    Speech(Arc<SpeechHandle>) h => Settled { answer: Answer::Speech(h.id().0), handle: Some(h) };
    Interrupt(Result<(), InterruptRefusal>) r =>
        Settled::of(r.map_or_else(Answer::InterruptRefused, |()| Answer::InterruptOk));
    Idle(Result<(), IdleRefused>) r =>
        Settled::of(r.map_or_else(Answer::IdleRefused, |()| Answer::IdleOk));
    Hold(Result<HoldId, IdleRefused>) r =>
        Settled::of(r.map_or_else(Answer::IdleRefused, |hold| Answer::Hold(hold.0)));
    Commit(Result<String, CommitSuperseded>) r =>
        Settled::of(r.map_or(Answer::CommitSuperseded, Answer::Committed));
    Handoff(Result<(), HandoffRefused>) r =>
        Settled::of(r.map_or_else(Answer::HandoffRefused, |()| Answer::HandoffOk));
    Drain(Result<(), DrainRefused>) r =>
        Settled::of(r.map_or_else(Answer::DrainRefused, |()| Answer::DrainOk));
    Dtmf(DtmfOutcome) outcome => Settled::of(Answer::Dtmf(outcome));
}

/// What a hook answers with, as the driver takes it.
fn hook_outcome(answer: HookAnswer, text: String) -> HookOutcome {
    match answer {
        HookAnswer::Returned => HookOutcome::Returned,
        HookAnswer::Failed => HookOutcome::Failed,
        HookAnswer::TurnInstructions => HookOutcome::Turn(vec![TurnEdit::UpdateInstructions(text)]),
        HookAnswer::TurnStop => HookOutcome::Turn(vec![TurnEdit::StopResponse]),
        HookAnswer::Rendered => HookOutcome::Rendered(text),
        HookAnswer::Filler => HookOutcome::Filler { text: Some(text), spoke: false },
    }
}

/// A `Ref` is the n-th resource of its kind; the production identity is whatever the run
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
    /// One thing the room did, on whichever of its two channels reports it. Refused by a session
    /// built without room legs: there is no room, so nothing happened in it.
    fn room_event(&mut self, event: RoomIn) -> bool {
        let who = || lk_agents_rtc::ParticipantIdentity::from("caller");
        let reason = lk_agents_rtc::DisconnectReason::ClientInitiated;
        match event {
            RoomIn::Linked => self
                .room_io
                .send(RoomIoEvent::ParticipantLinked(ParticipantLinked { identity: who() }))
                .is_ok(),
            RoomIn::Unlinked => self
                .room_io
                .send(RoomIoEvent::ParticipantUnlinked(ParticipantUnlinked {
                    identity: who(),
                    reason,
                }))
                .is_ok(),
            RoomIn::StateChanged => self
                .room
                .send(RoomEvent::ConnectionStateChanged {
                    state: lk_agents_rtc::ConnectionState::Reconnecting,
                })
                .is_ok(),
            RoomIn::Closed => self.room.send(RoomEvent::Close { reason }).is_ok(),
        }
    }

    /// Turn one input into the production command it mirrors and enqueue it.
    ///
    /// A reference that no longer resolves -- a speech the driver has forgotten, a hook already
    /// finished -- makes the input a refused attempt. Nothing is invented to stand in for it,
    /// because a command carrying an id the driver never issued is a different test from the one
    /// the case asked for.
    fn call(&mut self, at: Observed, input: &Input) -> bool {
        let mut waiting: Option<Pending> = None;
        let mut stream: Option<(bool, Feeding)> = None;
        let registry = self.registry.clone();
        let calls = registry.calls();

        let command: Command = match input {
            Input::Say { text: AgentText(text), input: say, allow, to_ctx } => {
                let (call, rx) = calls.open(DriverAnswer::speech);
                waiting = Some(Pending::Speech(rx));
                let call = Some(call);
                let text = match say {
                    SayInput::Words => SayText::Text(text.clone()),
                    SayInput::TextStream => {
                        let (sender, receiver) = mpsc::channel(256);
                        stream = Some((false, Feeding::Text(sender)));
                        SayText::Stream(receiver)
                    }
                    SayInput::Audio => {
                        let (sender, rx) = mpsc::channel(256);
                        stream = Some((true, Feeding::Audio { sender, chunks: 0 }));
                        SayText::Audio { text: text.clone(), rx }
                    }
                };
                let (allow_interruptions, add_to_chat_ctx) = (*allow, *to_ctx);
                SpeechCommand::Say { text, allow_interruptions, add_to_chat_ctx, call }.into()
            }
            Input::GenerateReply { user_input, instructions, allow } => {
                let (call, rx) = calls.open(DriverAnswer::speech);
                waiting = Some(Pending::Speech(rx));
                SpeechCommand::GenerateReply {
                    user_input: user_input.clone().map(|UserText(t)| t),
                    instructions: instructions.clone().map(|AgentText(t)| t),
                    allow_interruptions: *allow,
                    tool_choice: None,
                    chat_ctx: None,
                    toolset: None,
                    on_enter: false,
                    call: Some(call),
                }
                .into()
            }
            Input::Interrupt { force, wait } => {
                let call = wait.then(|| {
                    let (call, rx) = calls.open(DriverAnswer::interrupt);
                    waiting = Some(Pending::Interrupt(rx));
                    call
                });
                SpeechCommand::Interrupt { force: *force, call }.into()
            }
            Input::InterruptSpeech { speech, force } => {
                let Some(speech) = self.speech_id(*speech) else { return false };
                SpeechCommand::InterruptSpeech { speech, force: *force }.into()
            }
            Input::SetAllowInterruptions { speech, allow } => {
                let Some(speech) = self.speech_id(*speech) else { return false };
                SpeechCommand::SetAllowInterruptions { speech, allow: *allow }.into()
            }
            Input::CommitUserTurn { transcript_timeout, stt_flush, skip_reply } => {
                let (call, rx) = calls.open(DriverAnswer::commit);
                waiting = Some(Pending::Commit(rx));
                SpeechCommand::CommitUserTurn {
                    transcript_timeout: Duration::from_millis(transcript_timeout.0),
                    stt_flush_duration: Duration::from_millis(stt_flush.0),
                    skip_reply: *skip_reply,
                    call: Some(call),
                }
                .into()
            }
            Input::WaitForIdle { hold } => {
                let hold = match hold.map(|r| self.hold_id(r)) {
                    Some(None) => return false,
                    given => given.flatten(),
                };
                let (call, rx) = calls.open(DriverAnswer::idle);
                waiting = Some(Pending::Idle(rx));
                SpeechCommand::WaitForIdle { hold, call: Some(call) }.into()
            }
            Input::WaitForIdleAndHold {} => {
                let (call, rx) = calls.open(DriverAnswer::hold);
                waiting = Some(Pending::Hold(rx));
                SpeechCommand::WaitForIdleAndHold { call: Some(call) }.into()
            }
            Input::ReleaseIdleHold { hold } => {
                let Some(hold) = self.hold_id(*hold) else { return false };
                SpeechCommand::ReleaseIdleHold { hold }.into()
            }
            Input::ClaimUserTurn {} => SpeechCommand::ClaimUserTurn.into(),
            Input::ReleaseUserTurn {} => SpeechCommand::ReleaseUserTurn.into(),
            Input::SetRunOutput { speech, ok } => {
                let Some(speech) = self.speech_id(*speech) else { return false };
                let host = self.host_ref();
                let output = if *ok { RunOutput::Value(host) } else { RunOutput::Error(host) };
                SpeechCommand::SetRunOutput { speech, output }.into()
            }
            Input::ToolUpdate { tool, progress } => {
                let Some(call_id) = self.tool_call_id(*tool) else { return false };
                let update = match progress {
                    Some(AgentText(m)) => {
                        ToolUpdate::Progress { raw: m.clone(), message: m.clone() }
                    }
                    None => ToolUpdate::Final {
                        answer: Answered::Returned { value: None, agents: vec![] },
                    },
                };
                SpeechCommand::ToolUpdate { call_id, update }.into()
            }
            Input::FillerStart { tool, speech, delay, interval, text } => {
                let (Some(call_id), Some(speech)) =
                    (self.tool_call_id(*tool), self.speech_id(*speech))
                else {
                    return false;
                };
                let id = FillerId(self.next_filler);
                self.next_filler += 1;
                SpeechCommand::FillerStart {
                    id,
                    call_id,
                    speech,
                    delay: Duration::from_millis(delay.0),
                    interval: interval.map(|ms| Duration::from_millis(ms.0)),
                    max_steps: None,
                    source: text
                        .clone()
                        .map_or(FillerSource::Host, |AgentText(t)| FillerSource::Text(t)),
                }
                .into()
            }
            Input::FillerStop { filler } => {
                let Some(id) = self.filler_id(*filler) else { return false };
                SpeechCommand::FillerStop { id }.into()
            }
            Input::PauseAuthorization {} => SpeechCommand::PauseAuthorization.into(),
            Input::ResumeAuthorization {} => SpeechCommand::ResumeAuthorization.into(),

            Input::CommitTurn { skip_reply } => {
                TurnCommand::Commit { skip_reply: *skip_reply }.into()
            }
            Input::ClearTurn {} => TurnCommand::Clear.into(),
            Input::Configure { knob, mode, ms, on, words } => {
                TurnCommand::Configure(patch(*knob, *mode, *ms, *on, *words)).into()
            }

            Input::Handoff { agent, instructions, tools, pause_previous, resume } => {
                let (call, rx) = calls.open(DriverAnswer::handoff);
                waiting = Some(Pending::Handoff(rx));
                let mut frame = frame(&registry, agent, &instructions.0, tools);
                frame.previous =
                    if *pause_previous { PreviousActivity::Pause } else { PreviousActivity::Close };
                frame.next = if *resume { NewActivity::Resume } else { NewActivity::Start };
                frame.call = Some(call);
                ActivityCommand::Update(Box::new(frame)).into()
            }
            Input::UpdateInstructions { text: AgentText(text) } => {
                ActivityCommand::UpdateInstructions(instructions(text)).into()
            }
            Input::UpdateTools { tools } => {
                ActivityCommand::UpdateTools(toolset(&registry, tools)).into()
            }
            Input::SetAudioInput { attach } => {
                // A new microphone replaces the old one as a detach does: nothing said into the
                // old one reaches the session any more.
                for (nth, _) in std::mem::take(&mut self.mics) {
                    self.record.close(at, Kind::Mic, nth);
                }
                let source = attach.then(|| {
                    let (mic, frames) = fakes::FakeAudioInput::new();
                    let nth = self.registry.new_mic();
                    self.mics.insert(nth, (frames, 0));
                    HostAudio(mic as _)
                });
                ActivityCommand::SetAudioInput(source).into()
            }
            Input::SetVideoInput { attach } => {
                let source = attach.then(|| {
                    HostVideo(fakes::FakeVideoInput::new(&registry, self.refs.clone()) as _)
                });
                ActivityCommand::SetVideoInput(source).into()
            }
            Input::SetIoEnabled { leg, enabled } => {
                ActivityCommand::SetIoEnabled { leg: *leg, enabled: *enabled }.into()
            }
            Input::UpdateChatCtx { items, text: AgentText(text) } => {
                ActivityCommand::UpdateChatCtx(history(items, text)).into()
            }
            Input::SetAudioOutput { attach } => {
                let out = attach.then(|| {
                    let pausable = self.setup.pausable_output;
                    let sink = fakes::FakeAudioOutput::new(registry.clone(), pausable);
                    self.resources.register_audio_output(sink)
                });
                ActivityCommand::SetAudioOutput(out).into()
            }
            Input::SetTranscriptionOutput { attach } => {
                let out = attach.then(|| {
                    let sink = fakes::FakeTextOutput::new(registry.clone());
                    self.resources.register_text_output(sink)
                });
                ActivityCommand::SetTranscriptionOutput(out).into()
            }
            Input::UpdateModels { models } => {
                let (stt, llm, tts) = match models {
                    Models::Stt => (true, false, false),
                    Models::Llm => (false, true, false),
                    Models::Tts => (false, false, true),
                    Models::LlmAndTts => (false, true, true),
                    Models::All => (true, true, true),
                };
                ActivityCommand::UpdateModels {
                    stt: given(stt.then(|| {
                        let connect = Duration::from_millis(self.setup.connect_ms.unwrap_or(0));
                        let stt = fakes::FakeStt::new(registry.clone(), connect) as Arc<Stt>;
                        self.resources.register_stt(stt)
                    })),
                    llm: given(llm.then(|| {
                        let llm = fakes::FakeLlm::new(registry.clone()) as Arc<Llm>;
                        self.resources.register_llm(llm)
                    })),
                    tts: given(tts.then(|| {
                        let tts = fakes::FakeTts::new(registry.clone()) as Arc<Tts>;
                        self.resources.register_tts(tts)
                    })),
                    vad: Maybe::NotGiven,
                }
                .into()
            }

            Input::Drain { wait } => {
                let call = wait.then(|| {
                    let (call, rx) = calls.open(DriverAnswer::drain);
                    waiting = Some(Pending::Drain(rx));
                    call
                });
                ActivityCommand::Drain { call }.into()
            }
            Input::PushVideo {} => Command::Video(ImageContent {
                id: "img_case".into(),
                image: ImageSource::Frame(MediaRef::new(self.record.now().seq, &self.refs)),
                inference_width: None,
                inference_height: None,
                inference_detail: Default::default(),
                mime_type: None,
            }),
            Input::Close { drain } => {
                self.record.enter_phase(Phase::Closing);
                Close { drain: *drain, ..Default::default() }.into()
            }
            Input::HookDone { hook, answer, text: AgentText(text) } => {
                let Some(id) = self.hook_id(*hook) else { return false };
                Command::HookDone { id, outcome: hook_outcome(*answer, text.clone()) }
            }
            // The hook is gone from the live set, but the record still knows its id: the answer
            // reaches the driver as a host's late answer does.
            Input::HookDoneLate { hook, answer, text: AgentText(text) } => {
                let Some(id) = self.hook_id(Ref::new(hook.nth)) else { return false };
                Command::HookDone { id, outcome: hook_outcome(*answer, text.clone()) }
            }
            Input::HostMetrics {} => Command::HostMetrics(Box::new(vad_metrics())),
            Input::Ivr { input, silence, text: UserText(text) } => match input {
                IvrInput::Start => {
                    IvrCommand::Start { max_silence: Duration::from_millis(silence.0) }.into()
                }
                IvrInput::SeedTranscript => {
                    IvrCommand::SeedTranscript { text: text.clone() }.into()
                }
                IvrInput::Close => IvrCommand::Close.into(),
            },
            Input::SetStaticKeyterms { terms } => Command::SetStaticKeyterms(terms.clone()),
            Input::Dtmf { input, key, digits, confirm, timeout } => match input {
                DtmfInput::Start => {
                    let Ok(collector) = DtmfCollector::new(
                        digits.0 as usize + 1,
                        *confirm,
                        Duration::from_millis(timeout.0),
                        DtmfEvent::Pound,
                    ) else {
                        return false;
                    };
                    let (call, rx) = calls.open(DriverAnswer::dtmf);
                    waiting = Some(Pending::Dtmf(rx));
                    DtmfCommand::Start { collector, call: Some(call) }.into()
                }
                DtmfInput::Digit => DtmfCommand::Digit { digit: key.digit().to_owned() }.into(),
                DtmfInput::Stop => DtmfCommand::Stop.into(),
            },
            Input::Task { agent, speech } => {
                let speech = match speech.map(|r| self.speech_id(r)) {
                    Some(None) => return false,
                    given => given.flatten(),
                };
                let task_id = format!("task_{:04}", self.next_task);
                self.next_task += 1;
                TaskCommand::Enter(Box::new(TaskEntry {
                    handle: TaskHandle::new(task_id.into()),
                    frame: frame(&registry, agent, &format!("You are {agent}."), &[]),
                    speech,
                    preserve_function_call_history: true,
                    parent: None,
                }))
                .into()
            }
            Input::Recorder { attach } => Command::Recorder(if *attach {
                RecorderCommand::Attach(HostRecorder(fakes::FakeRecorder::new(registry.clone())))
            } else {
                RecorderCommand::Detach
            }),
            Input::Run { user_input, expects_output, retries } => Command::Run(Box::new(RunSpec {
                user_input: user_input.clone().map(|UserText(t)| t),
                expects_output: *expects_output,
                max_retries: retries.0,
                retry_instructions: "Please try again.".into(),
            })),
            _ => unreachable!("{input:?} is not a command"),
        };

        // A command the driver never received made no call: the loop is gone, the input is
        // refused, and nothing is opened for it. Only a command the driver has opens the stream it
        // feeds and the call it will answer.
        if self.session.api.send(command).is_err() {
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
    /// recorded when it opened that one. Resolution can fail, and a failure is the case pointing
    /// at something that is no longer there -- never a reason to substitute anything.
    fn host_ref(&self) -> HostRef {
        HostRef::new(self.record.now().seq, &self.refs)
    }
}

// ==============================================================================================
// Feeding a stream, answering a fake
// ==============================================================================================

/// A stream a `say` opened, until the case ends it with `FeedEnd`.
pub(super) enum Feeding {
    Text(mpsc::Sender<String>),
    Audio { sender: mpsc::Sender<AudioFrame>, chunks: u16 },
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
        let Some((sender, count)) = self.mics.get_mut(&mic) else { return false };
        for _ in 0..chunks {
            let sample = Tag::mic(mic, *count % (Tag::MAX_CHUNK + 1));
            let data = sample.to_le_bytes().repeat(SAMPLES_PER_CHUNK);
            let frame = AudioFrame::new(data, SAMPLE_RATE, 1, SAMPLES_PER_CHUNK as u32);
            if sender.send(frame).is_err() {
                return false;
            }
            *count = count.wrapping_add(1);
        }
        true
    }

    /// Push one piece of a `say`'s input stream. Each audio chunk carries its stream and its index
    /// in its samples, so what reaches the sink can be decoded back to what the case fed.
    fn feed(&mut self, at: Observed, input: &Input) -> bool {
        use mpsc::error::TrySendError::*;
        match input {
            Input::FeedText { stream, text: AgentText(text) } => {
                let Some(Feeding::Text(sender)) = self.streams.get_mut(&stream.nth) else {
                    return false;
                };
                match sender.try_send(text.clone()) {
                    Ok(()) => true,
                    Err(Closed(_)) => false,
                    Err(Full(_)) => panic!("fixture text input full"),
                }
            }
            Input::FeedAudio { stream, chunks: Count(count) } => {
                let nth = stream.nth;
                let Some(Feeding::Audio { sender, chunks }) = self.streams.get_mut(&nth) else {
                    return false;
                };
                for _ in 0..*count {
                    let sample = Tag::fed(nth, *chunks);
                    let data = sample.to_le_bytes().repeat(SAMPLES_PER_CHUNK);
                    let frame = AudioFrame::new(data, SAMPLE_RATE, 1, SAMPLES_PER_CHUNK as u32);
                    match sender.try_send(frame) {
                        Ok(()) => *chunks += 1,
                        Err(Closed(_)) => return false,
                        Err(Full(_)) => panic!("fixture audio input full"),
                    }
                }
                true
            }
            Input::FeedEnd { stream } => {
                if self.streams.remove(&stream.nth).is_none() {
                    return false;
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
        self.drive(Some(self.budget()), async move { handle.wait_for_done().await }).await.is_some()
    }

    async fn await_exit(&mut self) -> bool {
        self.drive(Some(self.budget()), std::future::pending::<()>()).await;
        let exited = self.runtime.is_none();
        if exited {
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
            let Some(runtime) = &mut self.runtime else { return None };
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
            let settled = runtime.driver.settled();
            tokio::select! {
                biased;
                value = &mut done => return Some(value),
                () = self.time.wait_until_opt(next) => {}
                () = settled => {}
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
        self.settle_answers();
        // The observer unwound out of a livelocked step: the loop reports it, not as a panic.
        if self.livelock.lock().unwrap().is_some() {
            self.drop_runtime();
        } else if let Err(panic) = called {
            self.record.set_driver_panic(panic_message(&*panic));
            self.drop_runtime();
        }
    }

    /// Hands every answer the step produced to its caller, as the runtime's answer pump does.
    fn settle_answers(&mut self) {
        while let Ok((call, answer)) = self.session.answers.try_recv() {
            self.registry.calls().settle(call, answer);
        }
    }

    /// The cores are gone, as production drops them: a call never answered is closed.
    fn drop_runtime(&mut self) {
        self.runtime = None;
        self.registry.calls().close();
    }

    /// Await every retained task, and record how each ended.
    ///
    /// A task may spawn another before finishing, so this loops -- and because the work is the
    /// product's, the loop can fail to terminate. It is bounded for the same reason the release
    /// loop is: an unbounded loop here is a hang, not a slow run, and a driver that keeps
    /// spawning while closing is a result worth reporting rather than waiting on forever.
    ///
    /// Each batch is raced against a deadline a budget away, the session stepped meanwhile as its
    /// own loop would be: a close is a chain of bounded phases that can outlast `AwaitExit`, and a
    /// session frozen mid-close would hold every task it ends. The clock is paused and moves only
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
