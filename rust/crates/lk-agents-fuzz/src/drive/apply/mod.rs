//! One input into what the product sees: a call on the host API, a stream's next piece, a fake's
//! answer, something in the room, or a wait.
//!
//! Every input is recorded, taken or refused. A reference that no longer resolves -- a speech the
//! driver has forgotten, a hook already finished -- makes the input a refused attempt, and so does
//! a call the API refuses. Nothing is invented to stand in for it, because a call carrying an id
//! the driver never issued is a different test from the one the case asked for.
use lk_agents_abi::OneOrStream;
use lk_agents_abi::voice::speech_handle::SpeechHandle as HostSpeech;
use std::task::{Context, Poll, Waker};

use lk_agents_abi::voice::rtc::{self, ConnectionState, DisconnectReason, RtcRoom};
use lk_agents_abi::voice::session::{VoiceSession as _, VoiceTaskEnded};
use lk_agents_abi::voice::session_options::{TurnHandlingKwargs, UpdateKwargs};
use lk_agents_voice::agent_session::{Command, DrainRefused, IvrCommand};

use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechKind;
use lk_agents_voice::workflows::machine::dtmf::{DtmfOutcome, GetDtmfResult};

use super::start::{
    Llm, Tts, avatar_metrics, frame, history, patch, session_stt, tool_specs, turn_detection,
};
use super::*;
mod call;
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

    /// Enter a phase. The progress deadline starts at `Settling`: releasing may advance virtual
    /// time to let the driver's own timers fire without counting that time as stalled progress.
    fn phase(&mut self, at: Observed, phase: Phase) -> bool {
        self.record.enter_phase(phase);
        if phase >= Phase::Settling {
            self.record.mark_settling(at);
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

/// What `update_options` / `override_options` say for `knob`; `None` for the options they do not
/// take (endpointing and detection only).
fn turn_update(knob: Knob, mode: TurnMode, ms: Millis) -> Option<UpdateKwargs> {
    let endpointing = |patch: fn(&mut EndpointingKwargs, f64)| {
        let mut kwargs = EndpointingKwargs::default();
        patch(&mut kwargs, secs(ms));
        Some(kwargs)
    };
    match knob {
        Knob::Detection => Some(UpdateKwargs {
            turn_detection: Maybe::Given(
                turn_detection(mode).map_or(TurnDetectionKwarg::Null, TurnDetectionKwarg::Mode),
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
                    for (enabled, operation) in [
                        (false, crate::record::BoundaryOperation::Unsubscribe),
                        (true, crate::record::BoundaryOperation::Resubscribe),
                    ] {
                        let error = ready_now(
                            track.set_subscribed(&lk_agents_abi::telemetry::Cx::NONE, enabled),
                        )
                        .expect("fake subscription ready")
                        .err()
                        .map(|e| e.to_string());
                        self.record.record_boundary(
                            self.stamps.next(),
                            crate::record::BoundaryOutcome { operation, error },
                        );
                    }
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
            HookAnswer::Filler => {
                HookOutcome::Filler(lk_agents_abi::voice::session_effects::FillerReply::Text(text))
            }
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

    fn tool_origin(
        &self,
        reference: Ref<crate::input::Tool>,
    ) -> Option<lk_agents_abi::voice::session::VoiceCommandOrigin> {
        match self.record.resource(Kind::Tool, reference.nth)?.detail() {
            Detail::Tool { invocation_id, operation, call_id, .. } => {
                Some(lk_agents_abi::voice::session::VoiceCommandOrigin::Tool {
                    invocation_id: invocation_id.clone(),
                    operation: *operation,
                    call_id: call_id.clone(),
                })
            }
            _ => None,
        }
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
