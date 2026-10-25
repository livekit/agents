//! `AgentActivity`'s realtime side: [`RealtimeSessionState`] (the session's requests and its open
//! generations), the driver's handling of the model's events, and the turn detection it agrees on.

use crate::agent::{Source, Visibility};
use crate::agent_activity::false_interruption::AudioInterruptTrigger;
use crate::agent_activity::options::TurnDetection;
use crate::agent_activity::scheduling::{SpeechId, SpeechKind};
use crate::agent_activity::speech::SpeechSpec;
use crate::agent_session::effects::{Out, TurnTimer};
use crate::agent_session::{Answer, Close, ClosePhase, Closing, Driver, RequestId, Tx};
use crate::audio_recognition::{ServerSpeech, TurnDetectionMode};
use crate::events::{DiagnosticEvent, SessionEvent};
use crate::generation::ReplySource;
use crate::generation::pipeline_reply::fatal;
use crate::speech_handle::{InputDetails, SpeechHandle};
use crate::trace::RealtimeStep;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::llm::realtime::{
    Event, PushAudio, RealtimeCapabilities, RealtimeCommand, RealtimeError, RealtimeSession,
};
use lk_agents_abi::voice::events::AgentState;
use lk_agents_abi::{
    AudioFrame, ChatItem, ChatMessage, Role, Spawner, StreamSink, ToolChoice, Ts,
    llm::realtime as rt,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use realtime_state::{Pending, Reuse, Route, bare_reply, response_id_of};
pub use realtime_state::{RealtimeSessionState, Request, RequestKey, Settled};

/// The realtime requests' state, behind its methods.
pub(crate) mod realtime_state {
    use super::*;

    /// The realtime session's requests and its one open generation. The loop owns it; the `Driver`
    /// methods below decide and act. State is updated before any action that can re-enter (a fatal
    /// error closes the session, and the close interrupts this machine).
    #[derive(Default)]
    pub struct RealtimeSessionState {
        pub(super) caps: RealtimeCapabilities,
        next_key: u64,
        pub(super) pending: BTreeMap<String, Pending>,
        /// Every generation still generating, oldest first: a server-initiated turn created while
        /// another plays is queued behind it (`_on_generation_created` schedules a new speech).
        open: Vec<Open>,
        /// Every generation this session closed; their late deltas are dropped silently.
        closed: BTreeSet<String>,
        /// Tools ran on a self-replying model: its next server-initiated generation is the
        /// follow-up, awaited under the timer keyed here.
        pub(super) auto_follow_up: Option<(Reuse, RequestKey)>,
        /// The last step's tools asked the server for a reply it generates itself: the agent stays
        /// `thinking` when that speech ends (the reference resets only when no reply is expected).
        /// Cleared when a generation opens.
        auto_reply_expected: bool,
        /// Requests cancelled (`CancelReply`) before the server answered: a generation that still
        /// answers one is interrupted (`_realtime_reply_task`: "the response already landed").
        /// Forgotten once asked.
        pub(super) cancelled_requests: BTreeSet<String>,
        /// A generation answering a request already settled: its `GenerationDone` is dropped.
        pub(super) discarded: Option<String>,
        /// The generation whose foreign deltas were warned about; reset when a generation opens.
        foreign: Option<String>,
        pub(super) closing: bool,
        ended: bool,
    }

    impl RealtimeSessionState {
        pub fn new(caps: RealtimeCapabilities) -> Self {
            RealtimeSessionState { caps, ..RealtimeSessionState::default() }
        }
    }

    /// The timer set cannot key on a `String`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct RequestKey(pub u64);

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum Request {
        Say {
            text: String,
            add_to_chat_ctx: bool,
        },
        GenerateReply {
            instructions: Option<String>,
            tool_choice: Option<ToolChoice>,
        },
        /// No caller waits. `commit_audio`: commit the input buffer and ask, both or neither.
        UserTurn {
            commit_audio: bool,
        },
        ToolFollowUp {
            speech: SpeechId,
            step: u32,
        },
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum Settled {
        Failed(RealtimeError),
        /// A late answer is discarded.
        TimedOut(RealtimeError),
        Interrupted,
        Closing,
        Ended,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) struct Reuse {
        pub(super) speech: SpeechId,
        pub(super) step: u32,
    }

    pub(super) struct Pending {
        pub(super) key: RequestKey,
        pub(super) request: Request,
        pub(super) call: Option<RequestId>,
        pub(super) allow_interruptions: Option<bool>,
    }

    struct Open {
        response_id: String,
        /// `None` until the generation's speech is started, in the same cycle.
        speech: Option<SpeechId>,
        message_id: Option<String>,
        /// `Interrupt` was sent for this response; a second stop sends nothing more.
        cancelled: bool,
    }

    pub(super) fn bare_reply(req_id: &str) -> RealtimeCommand {
        RealtimeCommand::GenerateReply(rt::GenerateReply {
            req_id: req_id.to_owned(),
            instructions: None,
            tool_choice: None,
        })
    }

    pub(crate) fn response_id_of(ev: &Event) -> Option<&str> {
        match ev {
            Event::MessageStarted(m) => m.response_id.as_deref(),
            Event::TextDelta(d) => d.response_id.as_deref(),
            Event::AudioDelta(d) => d.response_id.as_deref(),
            Event::MessageModalities(m) => m.response_id.as_deref(),
            Event::MessageDone(m) => m.response_id.as_deref(),
            _ => None,
        }
    }

    /// Where a generation's event goes.
    pub(super) enum Route {
        Dropped,
        /// A generation that is not the open one, warned about once: its id and the open one's.
        Foreign {
            id: String,
            open: String,
        },
        To(SpeechId),
    }

    impl RealtimeSessionState {
        pub fn ended(&self) -> bool {
            self.ended
        }

        /// `true` when this starts the close (it was not closing).
        pub fn begin_closing(&mut self) -> bool {
            !std::mem::replace(&mut self.closing, true)
        }

        /// The socket ended: every open generation closes, nothing is followed up any more.
        pub(super) fn end(&mut self) {
            self.ended = true;
            for o in std::mem::take(&mut self.open) {
                self.closed.insert(o.response_id);
            }
            self.auto_follow_up = None;
            self.cancelled_requests.clear();
        }

        pub(super) fn message_started(&mut self, response_id: Option<&str>, message_id: &str) {
            if let Some(open) = self.find_mut(response_id) {
                open.message_id = Some(message_id.to_owned());
            }
        }

        pub(super) fn message_done(&mut self, response_id: Option<&str>, message_id: &str) {
            if let Some(open) = self.find_mut(response_id)
                && open.message_id.as_deref() == Some(message_id)
            {
                open.message_id = None;
            }
        }

        /// The generation `response_id` names, or the newest when it names none.
        fn find_mut(&mut self, response_id: Option<&str>) -> Option<&mut Open> {
            match response_id {
                Some(id) => self.open.iter_mut().find(|o| o.response_id == id),
                None => self.open.last_mut(),
            }
        }

        /// A request sent: pending under `req_id` until answered, its timer keyed by the returned
        /// key.
        pub(super) fn register(
            &mut self,
            req_id: String,
            request: Request,
            call: Option<RequestId>,
            allow_interruptions: Option<bool>,
        ) -> RequestKey {
            let key = self.mint_key();
            self.pending.insert(req_id, Pending { key, request, call, allow_interruptions });
            key
        }

        pub(super) fn take_by_key(&mut self, key: RequestKey) -> Option<(String, Pending)> {
            let req_id =
                self.pending.iter().find(|(_, p)| p.key == key).map(|(id, _)| id.clone())?;
            let pending = self.pending.remove(&req_id)?;
            Some((req_id, pending))
        }

        /// One key space for requests and the auto follow-up's wait: both time out on
        /// `TurnTimer::RealtimeRequest`.
        fn mint_key(&mut self) -> RequestKey {
            self.next_key += 1;
            RequestKey(self.next_key - 1)
        }

        /// Tools ran on a self-replying model: its next server-initiated generation continues
        /// `speech`, awaited under the returned timer key.
        pub(super) fn follow_up_with(&mut self, speech: SpeechId, step: u32) -> RequestKey {
            let key = self.mint_key();
            self.auto_follow_up = Some((Reuse { speech, step }, key));
            self.auto_reply_expected = true;
            key
        }

        /// The follow-up's wait timed out: the speech it would have continued.
        pub(super) fn auto_follow_up_timed_out(&mut self, key: RequestKey) -> Option<SpeechId> {
            let (reuse, k) = self.auto_follow_up?;
            (k == key).then(|| {
                self.auto_follow_up = None;
                reuse.speech
            })
        }

        /// The speeches of the generations still generating.
        pub(super) fn open_speeches(&self) -> Vec<SpeechId> {
            self.open.iter().filter_map(|o| o.speech).collect()
        }

        pub(crate) fn auto_reply_expected(&self) -> bool {
            self.auto_reply_expected
        }

        pub(super) fn open_generation(&mut self, response_id: String, speech: Option<SpeechId>) {
            self.foreign = None;
            self.auto_reply_expected = false;
            self.open.push(Open { response_id, speech, message_id: None, cancelled: false });
        }

        pub(super) fn set_open_speech(&mut self, response_id: &str, speech: SpeechId) {
            if let Some(open) = self.open.iter_mut().find(|o| o.response_id == response_id) {
                open.speech = Some(speech);
            }
        }

        /// Where an event of `response_id` goes: its generation's speech, or nowhere. Without an
        /// id, the newest generation's.
        pub(super) fn route(&mut self, response_id: Option<&str>) -> Route {
            let Some(id) = response_id else {
                return match self.open.last().and_then(|o| o.speech) {
                    Some(speech) => Route::To(speech),
                    None => Route::Dropped,
                };
            };
            if let Some(open) = self.open.iter().find(|o| o.response_id == id) {
                return open.speech.map_or(Route::Dropped, Route::To);
            }
            let late = self.discarded.as_deref() == Some(id) || self.closed.contains(id);
            match self.open.last() {
                Some(newest) if !late && self.foreign.as_deref() != Some(id) => {
                    self.foreign = Some(id.to_owned());
                    Route::Foreign { id: id.to_owned(), open: newest.response_id.clone() }
                }
                Some(_) | None => Route::Dropped,
            }
        }

        /// `GenerationDone` for `response_id`: `Some(speech)` when it closed an open generation; a
        /// discarded one ends like a closed one (its tail is late, not foreign).
        pub(super) fn generation_done(&mut self, response_id: &str) -> Option<Option<SpeechId>> {
            if self.discarded.as_deref() == Some(response_id) {
                self.discarded = None;
                self.closed.insert(response_id.to_owned());
                return None;
            }
            let at = self.open.iter().position(|o| o.response_id == response_id)?;
            let open = self.open.remove(at);
            self.closed.insert(open.response_id);
            Some(open.speech)
        }

        /// One `Interrupt` per generation still generating whose speech `stops` (or that has
        /// none), once each; with its message when it had one. The cancel is session-wide on the
        /// provider, so a generation whose speech carries on is spared.
        pub(super) fn cancel_where(
            &mut self,
            stops: impl Fn(Option<SpeechId>) -> bool,
        ) -> Vec<(RealtimeCommand, Option<SpeechId>, Option<String>)> {
            if self.auto_follow_up.is_some_and(|(reuse, _)| stops(Some(reuse.speech))) {
                self.auto_follow_up = None;
            }
            if self.ended {
                return Vec::new();
            }
            let mut out = Vec::new();
            for open in self.open.iter_mut().filter(|o| !o.cancelled && stops(o.speech)) {
                open.cancelled = true;
                let response_id = Some(open.response_id.clone());
                let cmd = RealtimeCommand::Interrupt(rt::Interrupt { response_id });
                out.push((cmd, open.speech, open.message_id.clone()));
            }
            out
        }
    }
}

/// How long a request may wait for its `GenerationCreated` before it fails; one bound for every
/// request kind (the reference leaves it to each plugin).
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

/// A live `update_options(turn_detection=..)` that would flip a realtime model's server-side turn
/// detection, which is resolved at the open, one per state.
pub const REALTIME_TURN_DETECTION_STAYS_ENABLED: &str = "changing turn_detection at runtime does not update a realtime model's server-side turn detection (resolved at session start); it stays enabled for this session.";

pub const REALTIME_TURN_DETECTION_STAYS_DISABLED: &str = "changing turn_detection at runtime does not update a realtime model's server-side turn detection (resolved at session start); it stays disabled for this session.";

/// How long a self-replying model's tools wait for the server's reply.
pub const AUTO_TOOL_REPLY_TIMEOUT: Duration = Duration::from_secs(5);

/// `"timed out waiting for realtime auto tool reply from %s"`.
pub const AUTO_TOOL_REPLY_TIMED_OUT: &str = "timed out waiting for realtime auto tool reply";

/// `flush` is a no-op, not `CommitAudio`: an input boundary means the gate substituted silence, not
/// that the user ended a turn.
pub(crate) struct RealtimeAudioSink {
    session: Arc<dyn RealtimeSession>,
}

impl RealtimeAudioSink {
    pub(crate) fn new(session: Arc<dyn RealtimeSession>) -> Self {
        RealtimeAudioSink { session }
    }
}

impl StreamSink<AudioFrame> for RealtimeAudioSink {
    fn push(&self, frame: AudioFrame) -> Result<(), ProviderError> {
        self.session.send(RealtimeCommand::PushAudio(PushAudio { frame }));
        Ok(())
    }

    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }

    fn end_input(&self) {
        // Not `aclose`: a handoff replaces the leg, not the socket (the server's mirror survives).
    }
}

/// One wake on the realtime leg's queue. `Ended` exists because `select!` cannot observe a closed
/// channel. `ToolsDone` is the player's one report, on the socket's queue so the two are ordered.
pub enum RealtimeWake {
    Event(Event),
    Ended,
    /// The `aclose` the close asked for returned.
    Closed,
    ToolsDone {
        speech: SpeechId,
        step: u32,
        outputs: Vec<ChatItem>,
        ask: bool,
    },
}

pub(crate) fn spawn_event_pump(
    spawner: &Arc<dyn Spawner>,
    session: Arc<dyn RealtimeSession>,
    tx: Tx<RealtimeWake>,
) -> lk_agents_abi::AbortHandle {
    use lk_agents_abi::SpawnerExt;
    spawner.spawn_unboxed(async move {
        while let Some(ev) = session.next_event().await {
            if tx.send(RealtimeWake::Event(ev)).is_err() {
                return; // the loop is gone; nothing to tell it
            }
        }
        let _ = tx.send(RealtimeWake::Ended);
    })
}

impl Driver {
    /// A command for the realtime socket, as an effect; none without a session. Per-frame
    /// `PushAudio` never comes through here.
    pub(crate) fn rt_send(&self, cmd: lk_agents_abi::RealtimeCommand, out: &mut Out) {
        if self.activity.rt_session.is_none() {
            return;
        }
        let send = crate::agent_session::effects::RealtimeSend(cmd);
        out.machine().push(crate::agent_session::effects::Effect::Realtime(send));
    }

    pub(crate) fn on_realtime(&mut self, now: Ts, wake: RealtimeWake, out: &mut Out) {
        match wake {
            RealtimeWake::Event(ev) => self.on_realtime_event(now, ev, out),
            RealtimeWake::Ended => {
                self.activity.realtime_state.end();
                self.settle_all(Settled::Ended, false, out);
                // Every player's channel closes, finishing with the words the user heard.
                self.activity.realtime_players.clear();
            }
            RealtimeWake::Closed => {
                if let Some(Closing { phase: ClosePhase::ClosingActivity { closed, .. }, .. }) =
                    &mut self.session.closing
                {
                    *closed = true;
                    out.signal(crate::agent_session::signals::Signal::CloseProgress);
                }
            }
            RealtimeWake::ToolsDone { speech, step, outputs, ask } => {
                // Minted here whether asked or not: the id sequence does not depend on the answer.
                let req_id = self.shared.mint_item_id();
                self.realtime_tools_done(now, speech, step, outputs, ask, req_id, out);
            }
        }
    }

    fn on_realtime_event(&mut self, now: Ts, ev: Event, out: &mut Out) {
        match ev {
            Event::InputSpeechStarted(_) => {
                // Server-side VAD in the local `StartOfSpeech`'s position: the turn learns the
                // onset, the resume timer is cancelled, then the guarded interruption.
                let eff = self.activity.audio_recognition.on_server(now, ServerSpeech::Started);
                out.machine().append(eff);
                self.on_start_of_speech(out);
                self.interrupt_by_audio_activity(now, out, AudioInterruptTrigger::ServerOnset);
            }
            Event::InputSpeechStopped(_) => {
                let eff = self.activity.audio_recognition.on_server(now, ServerSpeech::Stopped);
                out.machine().append(eff);
                self.start_false_interruption_timer(now, out)
            }
            Event::InputTranscription(t) => {
                // A native session's onset is on the session clock.
                let started_at = t.turn_started_at.map(|at| self.step.wall_secs_at(at));
                self.on_input_transcription(now, t, started_at, out);
            }
            // A host session's onset is the host's `time.time()`, which the session's clock may
            // never have shown (`turn_started_at` precedes the session): it is used as given.
            Event::HostInputTranscription(rt::HostInputTranscription {
                transcription,
                turn_started_at,
            }) => {
                let started_at = turn_started_at.filter(|at| at.is_finite());
                self.on_input_transcription(now, transcription, started_at, out);
            }
            Event::GenerationCreated(rt::GenerationCreated { response_id, req_id }) => {
                self.on_generation_created(now, response_id, req_id, out)
            }
            Event::GenerateReplyFailed(rt::GenerateReplyFailed { req_id, error }) => {
                self.fail_request(&req_id, error, out)
            }
            Event::MessageStarted(m) => {
                let t = &mut self.activity.realtime_state;
                t.message_started(m.response_id.as_deref(), &m.message_id);
                self.deliver(Event::MessageStarted(m), out);
            }
            Event::MessageDone(m) => {
                let t = &mut self.activity.realtime_state;
                t.message_done(m.response_id.as_deref(), &m.message_id);
                self.deliver(Event::MessageDone(m), out);
            }
            ev @ (Event::TextDelta(_)
            | Event::AudioDelta(_)
            | Event::MessageModalities(_)
            | Event::FunctionCall(_)) => self.deliver(ev, out),
            Event::GenerationDone(done) => self.on_generation_done(done, out),
            Event::RemoteItemAdded(rt::RemoteItemAdded { previous_item_id, item }) => {
                self.mirror_remote_item(previous_item_id, *item, out)
            }
            // The native session's `UpdateChatCtx` ack, and the connection: nothing to route. A host
            // error's stamp is placed by the host leg before it gets here.
            Event::ChatCtxUpdated { .. }
            | Event::ConnectionAcquired { .. }
            | Event::HostError(_) => {}
            Event::Disconnected { .. } => {
                if self.activity.realtime_state.begin_closing() {
                    self.close_on_realtime_failure(now, out);
                }
            }
            Event::SessionReconnected(_) => {
                // The socket was recycled: the frame is replayed once onto the new connection.
                let ctx = self.session.chat_ctx.for_realtime();
                self.push_activity(Some(ctx), out);
            }
            Event::Error(err) => self.on_realtime_error(now, err, out),
        }
    }

    /// Commits the input buffer when the server did not segment, and asks for the reply: one
    /// request, both or neither.
    pub(crate) fn realtime_user_turn(&mut self, now: Ts, out: &mut Out) {
        let Some(rt) = self.activity.rt_session.as_ref() else { return };
        let server_segments =
            rt.capabilities().turn_detection && !self.activity.realtime_turn_detection_disabled;
        let request = Request::UserTurn { commit_audio: !server_segments };
        self.realtime_request(now, request, None, None, out);
    }

    pub(crate) fn realtime_request(
        &mut self,
        now: Ts,
        request: Request,
        call: Option<RequestId>,
        allow_interruptions: Option<bool>,
        out: &mut Out,
    ) {
        let req_id = self.shared.mint_item_id();
        self.send_request(now, req_id, request, call, allow_interruptions, out);
    }

    fn send_request(
        &mut self,
        now: Ts,
        req_id: String,
        request: Request,
        call: Option<RequestId>,
        allow_interruptions: Option<bool>,
        out: &mut Out,
    ) {
        let t = &self.activity.realtime_state;
        if t.ended() {
            return self.settle(request, call, Settled::Ended);
        }
        if t.closing {
            // Closing: a turn's own actuation still goes out whole and settles at once; a caller's
            // `Say`/`GenerateReply` is refused (its handle would never finish).
            if let Request::UserTurn { commit_audio } = &request {
                if *commit_audio {
                    self.rt_send(RealtimeCommand::CommitAudio(rt::CommitAudio {}), out);
                }
                self.rt_send(bare_reply(&req_id), out);
            }
            return self.settle(request, call, Settled::Closing);
        }
        let cmd = match &request {
            Request::Say { .. } if !t.caps.supports_say => {
                let error = RealtimeError::new("say() on a realtime model that has no say");
                return self.settle(request, call, Settled::Failed(error));
            }
            Request::Say { text, add_to_chat_ctx } => RealtimeCommand::Say(rt::Say {
                req_id: req_id.clone(),
                text: text.clone(),
                add_to_chat_ctx: *add_to_chat_ctx,
            }),
            Request::GenerateReply { instructions, tool_choice } => {
                RealtimeCommand::GenerateReply(rt::GenerateReply {
                    req_id: req_id.clone(),
                    instructions: instructions.clone(),
                    // Honoured where the model takes one per response, dropped otherwise.
                    tool_choice: tool_choice.clone().filter(|_| t.caps.per_response_tool_choice),
                })
            }
            Request::UserTurn { commit_audio } => {
                if *commit_audio {
                    self.rt_send(RealtimeCommand::CommitAudio(rt::CommitAudio {}), out);
                }
                bare_reply(&req_id)
            }
            Request::ToolFollowUp { .. } => bare_reply(&req_id),
        };
        let key = self.activity.realtime_state.register(req_id, request, call, allow_interruptions);
        self.rt_send(cmd, out);
        out.machine().arm(TurnTimer::RealtimeRequest(key), now.saturating_add(REQUEST_TIMEOUT));
    }

    /// Its timer goes with it.
    fn take_pending(&mut self, req_id: &str, out: &mut Out) -> Option<Pending> {
        let pending = self.activity.realtime_state.pending.remove(req_id)?;
        out.machine().cancel(TurnTimer::RealtimeRequest(pending.key));
        Some(pending)
    }

    fn settle(&mut self, request: Request, call: Option<RequestId>, how: Settled) {
        // A follow-up that will never come: the player finishes with what it has.
        if let Request::ToolFollowUp { speech, .. } = request {
            self.activity.realtime_players.remove(&speech);
        }
        self.answer_waiter(call, how);
    }

    /// Every pending request leaves through exactly one settlement; `cancel` also sends
    /// `CancelReply`.
    fn settle_all(&mut self, how: Settled, cancel: bool, out: &mut Out) {
        for (req_id, Pending { key, request, call, .. }) in
            std::mem::take(&mut self.activity.realtime_state.pending)
        {
            out.machine().cancel(TurnTimer::RealtimeRequest(key));
            if cancel && !self.activity.realtime_state.ended() {
                self.activity.realtime_state.cancelled_requests.insert(req_id.clone());
                self.rt_send(RealtimeCommand::CancelReply(rt::CancelReply { req_id }), out);
            }
            self.settle(request, call, how.clone());
        }
    }

    /// The request timed out: the server is told not to answer, the caller learns the timeout.
    pub(crate) fn realtime_request_timed_out(&mut self, key: RequestKey, out: &mut Out) {
        // The server never generated the reply its tools asked for: the speech ends on what it
        // spoke (5 s wait).
        if let Some(speech) = self.activity.realtime_state.auto_follow_up_timed_out(key) {
            diagnose(out, DiagnosticEvent::warning(AUTO_TOOL_REPLY_TIMED_OUT));
            self.activity.realtime_players.remove(&speech);
            return;
        }
        let t = &mut self.activity.realtime_state;
        // Answered in the same instant the timer fired: nothing owed.
        let Some((req_id, Pending { request, call, .. })) = t.take_by_key(key) else { return };
        if !t.ended() && !t.closing {
            t.cancelled_requests.insert(req_id.clone());
            self.rt_send(RealtimeCommand::CancelReply(rt::CancelReply { req_id }), out);
        }
        let error = RealtimeError::new(format!(
            "generate_reply timed out: no generation was created within {REQUEST_TIMEOUT:?}"
        ));
        self.settle(request, call, Settled::TimedOut(error));
    }

    fn fail_request(&mut self, req_id: &str, error: RealtimeError, out: &mut Out) {
        // A cancelled request that will not generate is forgotten.
        let _ = self.activity.realtime_state.cancelled_requests.remove(req_id);
        // Unknown: already claimed by a `GenerationCreated`, or cancelled by the driver.
        if let Some(Pending { request, call, .. }) = self.take_pending(req_id, out) {
            self.settle(request, call, Settled::Failed(error));
        }
    }

    /// `_on_generation_created`: a request's answer, or a server-initiated turn.
    fn on_generation_created(
        &mut self,
        now: Ts,
        response_id: String,
        req_id: Option<String>,
        out: &mut Out,
    ) {
        let pending = req_id.as_deref().map(|id| self.take_pending(id, out));
        // Settled before the server answered (interrupted, closed, timed out), or closing.
        if matches!(pending, Some(None)) || self.activity.realtime_state.closing {
            let t = &mut self.activity.realtime_state;
            // Cancelled while the request was in flight: the response landed anyway, and only an
            // interrupt reaches it now.
            let cancelled = req_id.as_deref().is_some_and(|id| t.cancelled_requests.remove(id));
            t.discarded = Some(response_id.clone());
            if cancelled && !t.ended() {
                let response_id = Some(response_id.clone());
                self.rt_send(RealtimeCommand::Interrupt(rt::Interrupt { response_id }), out);
            }
            let message = "realtime server answered a request the driver had already settled; the generation is discarded";
            return diagnose(
                out,
                DiagnosticEvent::warning(message).with("response_id", response_id),
            );
        }
        let t = &mut self.activity.realtime_state;
        let (asked, allow_interruptions, add_to_chat_ctx, reuse) = match pending.flatten() {
            Some(Pending { request, call, allow_interruptions, .. }) => match request {
                Request::Say { add_to_chat_ctx, .. } => {
                    (Some(call), allow_interruptions, add_to_chat_ctx, None)
                }
                Request::GenerateReply { .. } => (Some(call), allow_interruptions, true, None),
                Request::UserTurn { .. } => (None, allow_interruptions, true, None),
                Request::ToolFollowUp { speech, step } => {
                    (None, allow_interruptions, true, Some(Reuse { speech, step }))
                }
            },
            // Server-initiated: the follow-up of a model that answers its own tools, or a new turn.
            None => {
                let follow_up = t.auto_follow_up.take();
                if let Some((_, key)) = follow_up {
                    out.machine().cancel(TurnTimer::RealtimeRequest(key));
                }
                (None, None, true, follow_up.map(|(reuse, _)| reuse))
            }
        };
        t.open_generation(response_id.clone(), reuse.map(|r| r.speech));
        // The server is answering, so a parked speech was interrupted for real: the pause ends by
        // interrupting, never by resuming under the answer.
        if self.activity.paused_speech.paused_handle().is_some() {
            self.cancel_speech_pause(now, out);
        }
        // A tool follow-up continues its speech; if that player is gone, a fresh speech.
        let speech = match reuse.filter(|r| self.activity.realtime_players.contains_key(&r.speech))
        {
            Some(r) => r.speech,
            None => {
                self.start_realtime_speech(now, asked, allow_interruptions, add_to_chat_ctx, out)
            }
        };
        self.activity.realtime_state.set_open_speech(&response_id, speech);
    }

    /// `asked`: the speech is a host command's (`say`, `generate_reply`), with its call.
    fn start_realtime_speech(
        &mut self,
        now: Ts,
        asked: Option<Option<RequestId>>,
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        out: &mut Out,
    ) -> SpeechId {
        // `allow_interruptions=false` holds only when the client owns replies; a self-replying
        // server interrupts on onset, so the caller is told once.
        let server_replies = self.activity.rt_session.as_ref().is_some_and(|s| {
            let caps = s.capabilities();
            caps.turn_detection && caps.auto_turn_reply_generation
        }) && self.activity.audio_recognition.turn_detection_mode()
            == TurnDetectionMode::RealtimeLlm;
        let mut allow =
            allow_interruptions.unwrap_or(self.session.config.effective.turn.interruption.enabled);
        if !allow && server_replies {
            allow = true;
            diagnose(
                out,
                DiagnosticEvent::warning(
                    "allow_interruptions=False is ignored on a realtime model whose server replies on its own",
                ),
            );
        }
        // The same `start_speech` as every speech, fed by the player channel.
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let events = tokio::sync::Mutex::new(rx);
        let source = ReplySource::Realtime { events, wake: self.realtime_tx.clone() };
        let spec = SpeechSpec {
            add_to_chat_ctx,
            host_asked: asked.is_some(),
            ..SpeechSpec::new(SpeechKind::Realtime, source, allow)
        };
        let handle = self.start_speech(now, spec, out);
        self.activity.realtime_players.insert(handle.id(), tx);
        if let Some(call) = asked {
            // Asked for by the host, so an open run owns it.
            self.run_own(crate::run_result::Owned::Speech(handle.id()), out);
            self.answer(call, Answer::Speech(Arc::clone(&handle)));
        }
        handle.id()
    }

    /// Route a generation's event to its speech. With a `response_id` the routing is exact and a
    /// foreign delta is dropped (warned once); without one the open slot is the only routing.
    fn deliver(&mut self, ev: Event, out: &mut Out) {
        let speech = match self.activity.realtime_state.route(response_id_of(&ev)) {
            Route::Dropped => return,
            Route::Foreign { id, open } => {
                let message = "realtime server streamed a generation that is not the open one; its deltas are dropped";
                let d = DiagnosticEvent::warning(message).with("response_id", id);
                return diagnose(out, d.with("open", open));
            }
            Route::To(speech) => speech,
        };
        // A gone player was interrupted; nothing is journalled per frame.
        self.activity.play_realtime(speech, ev);
    }

    fn on_generation_done(&mut self, done: rt::GenerationDone, out: &mut Out) {
        // Refused, discarded, or another generation's end while this one is open: nothing to end.
        let Some(speech) = self.activity.realtime_state.generation_done(&done.response_id) else {
            return;
        };
        if let Some(speech) = speech {
            let status = format!("{:?}", done.status);
            self.activity.play_realtime(speech, Event::GenerationDone(done));
            // The generation ended on the server; the speech finishes by its own playout.
            out.trace(RealtimeStep::GenerationStatus { speech, status });
        }
    }

    /// `RealtimeModelError` reaches `_on_error`, which has no counter for it and closes on the
    /// first unrecoverable one.
    fn on_realtime_error(&mut self, now: Ts, err: rt::RealtimeModelError, out: &mut Out) {
        let error = RealtimeError::with_code(err.message.clone(), err.code.clone());
        let recoverable = err.recoverable;
        let reported = ProviderError::Local(ApiError::new(err.message).with_retryable(recoverable));
        let source = crate::events::ErrorSource::Realtime;
        if recoverable {
            let pending = err
                .req_id
                .as_deref()
                .and_then(|id| self.activity.realtime_state.pending.remove(id));
            self.report_error(now, source, true, reported, None, out);
            if let Some(Pending { key, request, call, .. }) = pending {
                out.machine().cancel(TurnTimer::RealtimeRequest(key));
                self.settle(request, call, Settled::Failed(error));
            }
            return;
        }
        // Fatal: pending callers learn the code, the generation is cancelled, the session closes.
        // The state moves first, so the close this error may start finds everything settled.
        let t = &mut self.activity.realtime_state;
        let interrupts = t.cancel_where(|_| true);
        let pending = std::mem::take(&mut t.pending);
        let close = t.begin_closing();
        self.report_error(now, source, false, reported, None, out);
        for (cmd, ..) in interrupts {
            self.rt_send(cmd, out);
        }
        for (_, Pending { key, request, call, .. }) in pending {
            out.machine().cancel(TurnTimer::RealtimeRequest(key));
            self.settle(request, call, Settled::Failed(error.clone()));
        }
        if close {
            self.close_on_realtime_failure(now, out);
        }
    }

    /// The leg is dead (a realtime session is the LLM): the session closes with
    /// `CloseReason::Error`.
    fn close_on_realtime_failure(&mut self, now: Ts, out: &mut Out) {
        let error = ApiError::new("the realtime session ended with an unrecoverable error");
        let close = Close {
            reason: lk_agents_abi::voice::events::CloseReason::Error,
            drain: false,
            error: Some(ProviderError::Local(error)),
        };
        self.begin_closing(now, close, out);
    }

    #[allow(clippy::too_many_arguments)]
    fn realtime_tools_done(
        &mut self,
        now: Ts,
        speech: SpeechId,
        step: u32,
        outputs: Vec<ChatItem>,
        ask: bool,
        req_id: String,
        out: &mut Out,
    ) {
        let auto = self.activity.realtime_state.caps.auto_tool_reply_generation;
        for item in outputs {
            // Pushed while closing too (the server holds the call open until answered); only an
            // ended socket takes nothing.
            if !self.activity.realtime_state.ended() {
                let reply_required = ask && !auto;
                let push = rt::PushItem { item: Box::new(item), reply_required };
                self.rt_send(RealtimeCommand::PushItem(push), out);
            }
        }
        let t = &mut self.activity.realtime_state;
        if !ask || t.ended() || t.closing {
            self.activity.realtime_players.remove(&speech);
        } else if auto {
            // The server replies on its own; it is given 5 s to start.
            let key = t.follow_up_with(speech, step);
            out.machine()
                .arm(TurnTimer::RealtimeRequest(key), now.saturating_add(AUTO_TOOL_REPLY_TIMEOUT));
        } else {
            let request = Request::ToolFollowUp { speech, step };
            self.send_request(now, req_id, request, None, None, out);
        }
    }

    pub(crate) fn fail_waiter(&mut self, call: Option<RequestId>, message: &str) {
        self.answer_waiter(call, Settled::Failed(RealtimeError::new(message)));
    }

    /// A handle never in the speech queue, minted only when a caller awaits it.
    pub(crate) fn answer_waiter(&mut self, call: Option<RequestId>, how: Settled) {
        if call.is_none() {
            return;
        }
        let id = self.session.ids.next_speech_id();
        let handle = Arc::new(SpeechHandle::new(
            id,
            crate::speech_handle::SpeechWireId(id).to_string(),
            self.session.config.effective.turn.interruption.enabled,
            InputDetails::default(),
        ));
        match how {
            Settled::Failed(err) | Settled::TimedOut(err) => {
                handle.mark_done(Some(fatal(err.message)))
            }
            Settled::Interrupted | Settled::Closing | Settled::Ended => {
                let _ = handle.interrupt(true);
                handle.mark_done(None);
            }
        }
        self.answer(call, Answer::Speech(handle));
    }
    /// The server's turn detection was fixed at the open. After a re-resolution the mode is
    /// refined through the open's table and written back; a change that takes (local to local)
    /// or is refused (the server keeps the turn) is said once, never flipped silently.
    pub(crate) fn reconcile_realtime_turn_detection(
        &mut self,
        caps: &lk_agents_abi::llm::realtime::RealtimeCapabilities,
        out: &mut Out,
    ) -> Option<crate::audio_recognition::TurnDetectionMode> {
        let resolved = self.session.config.effective.turn.owner;
        let requested = self
            .activity
            .note_resolved_mode(resolved, self.session.config.effective.turn.detection);
        let side = SessionSide {
            stt: self.activity.audio_recognition.stt_enabled(),
            vad: self.input.has_vad(),
            detection: self.session.config.effective.turn.detection,
            streaming_detector: self.session.config.legs.detector
                == Some(crate::agent_activity::options::DetectorKind::Streaming),
            interruption_given: self.session.config.effective.turn.interruption.mode.is_some(),
            turn_detection_mode: resolved,
        };
        let disabled = self.activity.realtime_turn_detection_disabled;
        let mode = realtime_turn_detection_mode(caps, side, disabled);
        self.session.config.effective.turn.owner = mode;
        let previous = self.activity.audio_recognition.turn_detection_mode();
        let honoured = mode != previous;
        // `update_options(turn_detection=..)`: the server's detection was resolved at the open and
        // is not re-synced, so an explicit change that would flip it is said, once
        //; reverting to automatic is not.
        let enabled_at_open = !disabled;
        let explicit = side.detection != TurnDetection::Auto;
        if requested
            && explicit
            && caps.can_disable_turn_detection
            && server_turn_detection_enabled(caps, side) != enabled_at_open
        {
            let message = if enabled_at_open {
                REALTIME_TURN_DETECTION_STAYS_ENABLED
            } else {
                REALTIME_TURN_DETECTION_STAYS_DISABLED
            };
            let d = DiagnosticEvent::warning(message).with("requested", format!("{resolved:?}"));
            diagnose(out, d.with("owner", format!("{mode:?}")));
        }
        honoured.then_some(mode)
    }

    /// `generate_reply()` on a realtime session. `user_input` is pushed as an item first.
    pub(crate) fn realtime_reply(
        &mut self,
        now: Ts,
        user_input: Option<String>,
        instructions: Option<String>,
        allow_interruptions: Option<bool>,
        tool_choice: Option<lk_agents_abi::ToolChoice>,
        call: Option<RequestId>,
        out: &mut Out,
    ) {
        if self.activity.rt_session.is_none() {
            return;
        }
        if let Some(text) = user_input {
            let wall = self.step.wall();
            let item = ChatMessage::text(self.session.ids.mint("item"), Role::User, &text, wall);
            let push = rt::PushItem { item: Box::new(item.clone().into()), reply_required: true };
            self.rt_send(RealtimeCommand::PushItem(push), out);
            self.commit_item(item.into(), Source::UserTyped, Visibility::Both, out);
        }
        let request = Request::GenerateReply { instructions, tool_choice };
        self.realtime_request(now, request, call, allow_interruptions, out);
    }

    fn on_input_transcription(
        &mut self,
        now: Ts,
        t: lk_agents_abi::llm::realtime::InputTranscription,
        started_at: Option<f64>,
        out: &mut Out,
    ) {
        let speech =
            ServerSpeech::Transcription { text: t.transcript.clone(), is_final: t.is_final };
        let eff = self.activity.audio_recognition.on_server(now, speech);
        out.machine().append(eff);
        if t.is_final {
            self.upsert_user_transcript(t, started_at, out);
        }
    }

    /// A final server transcript, upserted by `item_id` (a repeated final replaces in place).
    /// Placed at the turn's onset (wall seconds) when reported, with it as `started_speaking_at`
    /// (`_on_input_audio_transcription_completed`), else now.
    pub(crate) fn upsert_user_transcript(
        &mut self,
        t: lk_agents_abi::llm::realtime::InputTranscription,
        started_at: Option<f64>,
        out: &mut Out,
    ) {
        let started = started_at.and_then(|at| {
            let since_epoch = Duration::try_from_secs_f64(at).ok()?;
            Some((at, std::time::UNIX_EPOCH + since_epoch))
        });
        let created_at = started.map_or_else(|| self.step.wall(), |(_, wall)| wall);
        let mut item = ChatMessage::text(t.item_id.clone(), Role::User, &t.transcript, created_at);
        item.metrics.started_speaking_at = started.map(|(at, _)| at);
        let source = Source::Remote { item_id: t.item_id };
        self.commit_item(item.into(), source, Visibility::Both, out);
    }

    /// An item the server added to its mirror, recorded once (duplicate ids ignored).
    pub(crate) fn mirror_remote_item(
        &mut self,
        previous_item_id: Option<String>,
        item: lk_agents_abi::ChatItem,
        out: &mut Out,
    ) {
        let id = item.id().to_owned();
        if self.session.chat_ctx.contains(&id) {
            return;
        }
        if let Some(anchor) = &previous_item_id
            && !self.session.chat_ctx.contains(anchor)
        {
            let message = "realtime server anchored an item to one this session does not hold; placed by its stamp";
            diagnose(out, DiagnosticEvent::warning(message).with("item_id", id.clone()));
        }
        self.commit_item(item, Source::Remote { item_id: id }, Visibility::Both, out);
    }

    /// The open generation cancelled and truncated to what was heard. The interruption decision is
    /// `decide_interrupt`'s; this only actuates it.
    pub(crate) fn realtime_interrupt(&mut self, out: &mut Out) {
        if self.activity.rt_session.is_none() {
            return;
        }
        if self.session.closing_past_edge() {
            return self.realtime_on_close(out);
        }
        if self.activity.realtime_state.ended() {
            return self.settle_all(Settled::Ended, false, out);
        }
        // What the user heard, only while speaking: otherwise the tracker holds the previous
        // speech's mark.
        let played = (self.session.states.agent_state() == AgentState::Speaking)
            .then(|| self.shared.output.audio().map(|out| out.tracker().position()))
            .flatten();
        // Only a generation whose speech stopped: the cancel is session-wide, and a speech that
        // plays on keeps its response.
        let current = self.activity.speech_scheduler.current_speech();
        let playing_on: BTreeSet<SpeechId> = {
            let tasks = &self.activity.speech_scheduler;
            let open = self.activity.realtime_state.open_speeches();
            open.into_iter()
                .filter(|s| tasks.handle(*s).is_some_and(|h| !h.interrupted()))
                .collect()
        };
        let interrupts = self
            .activity
            .realtime_state
            .cancel_where(|speech| speech.is_none_or(|s| !playing_on.contains(&s)));
        let truncation = self.activity.realtime_state.caps.message_truncation;
        for (cmd, speech, message_id) in interrupts {
            self.rt_send(cmd, out);
            // Only what was heard, and only when something was: the playing speech's message.
            if let Some(message_id) = message_id.filter(|_| speech.is_some() && speech == current)
                && let Some(played) = played.filter(|p| !p.is_zero() && truncation)
            {
                let truncate = rt::Truncate {
                    message_id,
                    has_audio: true,
                    audio_end_ms: u64::try_from(played.as_millis()).unwrap_or(u64::MAX),
                    audio_transcript: None,
                };
                self.rt_send(RealtimeCommand::Truncate(truncate), out);
            }
        }
        self.settle_all(Settled::Interrupted, true, out);
    }

    /// The session closes: every pending request settles `Closing`; every open generation is
    /// cancelled.
    pub(crate) fn realtime_on_close(&mut self, out: &mut Out) {
        self.activity.realtime_state.closing = true;
        for (cmd, ..) in self.activity.realtime_state.cancel_where(|_| true) {
            self.rt_send(cmd, out);
        }
        self.settle_all(Settled::Closing, true, out);
    }
}

fn diagnose(out: &mut Out, d: DiagnosticEvent) {
    out.machine().emit(SessionEvent::Diagnostic(d));
}

/// The client-side turn inputs `_resolve_rt_turn_detection_enabled` reads beside the capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionSide {
    pub stt: bool,
    pub vad: bool,
    /// What the user said, five-way. `Auto` is nothing said.
    pub detection: TurnDetection,
    /// Named by the caller (the host only keeps one on a realtime session when named).
    pub streaming_detector: bool,
    /// `interruption_detection` given on the agent or the session (`interruption["mode"]` said).
    pub interruption_given: bool,
    /// The mode the options resolved to, before the model's refinement.
    pub turn_detection_mode: TurnDetectionMode,
}

/// `_resolve_rt_turn_detection_enabled`: a model that cannot hand detection over keeps its own
/// setting; explicit `realtime_llm` keeps it on, `manual` turns it off; with a VAD, `vad` mode, a
/// named streaming detector or a given `interruption_detection` turn it off; else it stays on.
pub fn server_turn_detection_enabled(caps: &RealtimeCapabilities, side: SessionSide) -> bool {
    if !caps.turn_detection || !caps.can_disable_turn_detection {
        return caps.turn_detection;
    }
    match side.detection {
        TurnDetection::RealtimeLlm => return true,
        TurnDetection::Manual => return false,
        TurnDetection::Vad | TurnDetection::Stt | TurnDetection::Auto => {}
    }
    let client_side =
        side.detection == TurnDetection::Vad || side.streaming_detector || side.interruption_given;
    !(side.vad && client_side)
}

/// `turn_detection_disabled = not enabled and caps.can_disable_turn_detection` (`_start_session`).
pub fn turn_detection_disabled(caps: &RealtimeCapabilities, side: SessionSide) -> bool {
    !server_turn_detection_enabled(caps, side) && caps.can_disable_turn_detection
}

/// Who ends the turn on a realtime session (`realtime_llm`): the server iff it segments and replies
/// on its own and was left detection; otherwise the local legs, or the host.
pub fn realtime_turn_detection_mode(
    caps: &RealtimeCapabilities,
    side: SessionSide,
    disabled: bool,
) -> TurnDetectionMode {
    if caps.turn_detection && caps.auto_turn_reply_generation && !disabled {
        return TurnDetectionMode::RealtimeLlm;
    }
    match side.turn_detection_mode {
        TurnDetectionMode::Manual => TurnDetectionMode::Manual,
        TurnDetectionMode::Vad | TurnDetectionMode::Stt | TurnDetectionMode::RealtimeLlm => {
            if side.vad {
                TurnDetectionMode::Vad
            } else if side.stt {
                TurnDetectionMode::Stt
            } else {
                TurnDetectionMode::Manual
            }
        }
    }
}
