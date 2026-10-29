//! `AgentActivity`'s realtime side: [`RealtimeSessionState`] (the session's requests and its open
//! generations), the driver's handling of the model's events, and the turn detection it agrees on.

use crate::agent::{Source, Visibility};
use crate::agent_activity::false_interruption::InterruptRefusal;
use crate::agent_activity::options::TurnDetection;
use crate::agent_activity::scheduling::{SpeechId, SpeechKind};
use crate::agent_activity::speech::SpeechSpec;
use crate::agent_session::effects::{Out, Timer};
use crate::agent_session::{Close, ClosePhase, Closing, Driver, SpeechCall, Tx};
use crate::audio_recognition::{ServerSpeech, TurnDetectionMode};
use crate::events::{DiagnosticEvent, SessionEvent};
use crate::generation::ReplySource;
use crate::generation::fatal;
use crate::trace::{InterruptCause, InterruptStep, RealtimeStep};
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::llm::realtime::{
    Event, PushAudio, RealtimeCapabilities, RealtimeCommand, RealtimeError, RealtimeSession,
};
use lk_agents_abi::voice::events::AgentState;
use lk_agents_abi::{
    AudioFrame, ChatMessage, Role, Spawner, StreamSink, ToolChoice, Ts, llm::realtime as rt,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

pub use realtime_state::response_id_of;
use realtime_state::{Pending, Phase, Reuse, Route, Swap, bare_reply};
pub use realtime_state::{RealtimeSessionState, Request, RequestKey, Settled};

/// The realtime requests' state, behind its methods.
pub(crate) mod realtime_state;

/// How many closed generations' late deltas are recognised.
const CLOSED_REMEMBERED: usize = 32;

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

const REALTIME_SESSION_ENDED: &str = "the realtime session ended";

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
    PlayerEnded(SpeechId),
    Ended,
    /// The `aclose` the close asked for returned.
    Closed,
    ToolsDone {
        speech: SpeechId,
        step: u32,
        ask: bool,
        /// The follow-up's `tool_choice`, as the step decided it.
        tool_choice: Option<lk_agents_abi::ToolChoice>,
    },
    /// A cut reply's messages, truncated to what the caller heard of each.
    Heard {
        speech: SpeechId,
        cuts: Vec<rt::Truncate>,
    },
}

pub(crate) fn spawn_event_pump(
    spawner: &Arc<dyn Spawner>,
    session: Arc<dyn RealtimeSession>,
    tx: Tx,
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
    /// A text `say()` is the realtime model's to speak: it says, and there is no TTS
    /// (`AgentActivity.say`); otherwise the TTS speaks it, as on a pipeline.
    pub(crate) fn realtime_says(&self) -> bool {
        self.activity.rt_session.is_some()
            && self.session.tts.is_none()
            && self.activity.realtime_state.says()
    }

    /// A command for the realtime socket, as an effect; none without a session. Per-frame
    /// `PushAudio` never comes through here.
    fn play_realtime(&self, speech: SpeechId, event: Event, out: &mut Out) {
        if self.activity.realtime_players.contains(&speech) {
            out.machine().push(crate::agent_session::effects::Effect::RealtimePlayer {
                speech,
                event: Box::new(event),
            });
        }
    }

    fn close_realtime_player(&mut self, speech: SpeechId, out: &mut Out) {
        if self.activity.realtime_players.remove(&speech) {
            out.machine().push(crate::agent_session::effects::Effect::CloseRealtimePlayer(speech));
        }
    }

    pub(crate) fn rt_send(&self, cmd: lk_agents_abi::RealtimeCommand, out: &mut Out) {
        let Some(session) = self.activity.rt_session.as_ref().map(|s| s.id) else { return };
        out.machine()
            .push(crate::agent_session::effects::Effect::Realtime { session, command: cmd });
    }

    pub(crate) fn on_realtime(&mut self, now: Ts, wake: RealtimeWake, out: &mut Out) {
        self.on_realtime_wake(now, wake, out);
        // Whatever the wake ended (a reply done, a request answered or settled) may free the
        // server for the next queued request.
        self.realtime_send_queued(now, out);
    }

    fn on_realtime_wake(&mut self, now: Ts, wake: RealtimeWake, out: &mut Out) {
        match wake {
            RealtimeWake::Event(ev) => self.on_realtime_event(now, ev, out),
            RealtimeWake::PlayerEnded(speech) => {
                self.close_realtime_player(speech, out);
            }
            RealtimeWake::Ended => {
                self.activity.realtime_state.end();
                self.settle_all(Settled::Ended, false, out);
                // Every player's channel closes, finishing with the words the user heard.
                self.activity.realtime_players.clear();
                out.machine().push(crate::agent_session::effects::Effect::CloseRealtimePlayers);
            }
            RealtimeWake::Closed => {
                if let Some(Closing { phase: ClosePhase::ClosingActivity { closed, .. }, .. }) =
                    &mut self.session.closing
                {
                    *closed = true;
                    out.signal(crate::agent_session::effects::Signal::CloseProgress);
                }
            }
            // Not past the close's edge: the socket is going away with what it holds.
            RealtimeWake::Heard { speech, cuts } => {
                if self.session.closing_past_edge() {
                    return;
                }
                // Cut while its response still generates: stopped, or the server keeps adding to
                // what was truncated ("send it only while this response is still generating").
                for cmd in self.activity.realtime_state.cancel_where(|s| s == Some(speech)) {
                    self.rt_send(cmd, out);
                }
                let t = &self.activity.realtime_state;
                if t.caps.message_truncation && !t.ended() {
                    for cut in cuts {
                        self.rt_send(RealtimeCommand::Truncate(cut), out);
                    }
                }
            }
            RealtimeWake::ToolsDone { speech, step, ask, tool_choice } => {
                // Minted here whether asked or not: the id sequence does not depend on the answer.
                let req_id = self.session.ids.mint("rt");
                self.realtime_tools_done(now, speech, step, ask, tool_choice, req_id, out);
            }
        }
    }

    fn on_realtime_event(&mut self, now: Ts, ev: Event, out: &mut Out) {
        if !matches!(ev, Event::TranscriptAttribution(_))
            && response_id_of(&ev).is_some_and(|id| self.activity.realtime_state.attributed(id))
        {
            let result = self.activity.realtime_state.transcript_event(ev);
            self.commit_attributed_transcript(now, result, out);
            return;
        }
        match ev {
            Event::TranscriptAttribution(attribution) => {
                let created_at = self
                    .session
                    .chat_ctx
                    .entry(&attribution.source_response_id)
                    .map_or_else(|| self.step.wall(), |entry| entry.item.created_at());
                let result =
                    self.activity.realtime_state.attribute_transcript(attribution, created_at);
                self.commit_attributed_transcript(now, result.map(|()| None), out);
            }
            Event::InputSpeechStarted(_) => {
                // Server-side VAD in the local `StartOfSpeech`'s position: the turn learns the
                // onset, then `interrupt(source="audio_activity")`, a barge-in that is never a
                // pause; a speech that forbids it keeps playing.
                let eff = self.activity.audio_recognition.on_server(now, ServerSpeech::Started);
                out.machine().append(eff);
                if self.interrupt_all(now, false, InterruptCause::ByAudioActivity, out).is_err() {
                    out.trace(InterruptStep::ByAudioActivityRefused(InterruptRefusal::NotAllowed));
                }
            }
            Event::InputSpeechStopped(_) => {
                let eff = self.activity.audio_recognition.on_server(now, ServerSpeech::Stopped);
                out.machine().append(eff);
                self.start_false_interruption_timer(now, out)
            }
            // The onset is wall seconds, which the session's clock may never have shown (it may
            // precede the session): used as given.
            Event::InputTranscription(t) => {
                let started_at = t.turn_started_at.filter(|at| at.is_finite());
                self.on_input_transcription(now, t, started_at, out);
            }
            Event::GenerationCreated(rt::GenerationCreated { response_id, req_id }) => {
                self.on_generation_created(now, response_id, req_id, out)
            }
            Event::GenerateReplyFailed(rt::GenerateReplyFailed { req_id, error }) => {
                self.fail_request(&req_id, error, out)
            }
            ev @ (Event::MessageStarted(_)
            | Event::MessageDone(_)
            | Event::TextDelta(_)
            | Event::AudioDelta(_)
            | Event::MessageModalities(_)
            | Event::FunctionCall(_)) => self.deliver(ev, out),
            Event::GenerationDone(done) => self.on_generation_done(done, out),
            Event::RemoteItemAdded(rt::RemoteItemAdded { previous_item_id, item }) => {
                self.mirror_remote_item(previous_item_id, *item, out)
            }
            // A host error's stamp is placed by the host leg before it gets here; a server
            // generation is the host leg's to drain.
            Event::HostError(_) | Event::ServerGeneration(_) => {}
            // `_on_metrics_collected`: into the session's metrics and usage, attributed to no speech.
            Event::Metrics(rt::RealtimeMetrics { metrics }) => {
                out.machine()
                    .emit(crate::events::SessionEvent::MetricsCollected { metrics, speech: None });
            }
            // The session replays its conversation, instructions and tools onto the new connection
            // itself (`_reconnect`); replaying them again from here re-created every item the
            // replay had just restored, which the server refuses (`item_create_duplicate_item_id`).
            Event::SessionReconnected(_) => {}
            Event::SessionSwapped { regenerate } => self.on_session_swapped(now, regenerate, out),
            Event::Error(err) => self.on_realtime_error(now, err, out),
        }
    }

    fn commit_attributed_transcript(
        &mut self,
        now: Ts,
        result: Result<Option<ChatMessage>, ApiError>,
        out: &mut Out,
    ) {
        match result {
            Ok(Some(message)) => {
                let source = Source::Remote { item_id: message.id.clone() };
                self.commit_item(message.into(), source, Visibility::Both, out);
            }
            Ok(None) => {}
            Err(error) => self.on_realtime_error(
                now,
                rt::RealtimeModelError {
                    timestamp: now,
                    label: "RealtimeSession".into(),
                    message: error.message.clone(),
                    recoverable: false,
                    req_id: None,
                    code: Some("realtime_transcript_attribution_refused".into()),
                    provider_error: Some(error),
                },
                out,
            ),
        }
    }

    /// Commits the input buffer when the server did not segment, and asks for the reply: one
    /// request, both or neither. The reply's speech, if one was announced.
    pub(crate) fn realtime_user_turn(&mut self, now: Ts, out: &mut Out) -> Option<SpeechId> {
        let rt = self.activity.rt_session.as_ref()?;
        let server_segments =
            rt.capabilities.turn_detection && !self.activity.realtime_turn_detection_disabled;
        let request = Request::UserTurn { commit_audio: !server_segments };
        self.realtime_request(now, request, None, None, out)
    }

    /// The speech announced for it, if the request has one and it was not refused.
    pub(crate) fn realtime_request(
        &mut self,
        now: Ts,
        request: Request,
        call: Option<SpeechCall>,
        allow_interruptions: Option<bool>,
        out: &mut Out,
    ) -> Option<SpeechId> {
        let req_id = self.session.ids.mint("rt");
        self.send_request(now, req_id, request, call, allow_interruptions, out)
    }

    fn send_request(
        &mut self,
        now: Ts,
        req_id: String,
        request: Request,
        call: Option<SpeechCall>,
        allow_interruptions: Option<bool>,
        out: &mut Out,
    ) -> Option<SpeechId> {
        let t = &self.activity.realtime_state;
        if t.ended() {
            self.settle_request(request, call, Settled::Ended, out);
            return None;
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
            let closing = Settled::Interrupted(InterruptCause::Closing);
            self.settle_request(request, call, closing, out);
            return None;
        }
        let cmd = match &request {
            Request::Say { text, add_to_chat_ctx } => RealtimeCommand::Say(rt::Say {
                req_id: req_id.clone(),
                text: text.clone(),
                add_to_chat_ctx: *add_to_chat_ctx,
            }),
            Request::GenerateReply { instructions, tool_choice, .. } => {
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
            // `tool_choice="none"` when the step budget is spent or the agent drains
            // (`_realtime_generation_task`), where the model takes one per response.
            Request::ToolFollowUp { tool_choice, .. } => {
                RealtimeCommand::GenerateReply(rt::GenerateReply {
                    req_id: req_id.clone(),
                    instructions: None,
                    tool_choice: tool_choice.clone().filter(|_| t.caps.per_response_tool_choice),
                })
            }
        };
        // A caller's reply, or a user turn's, is announced now and asks the server once its speech
        // is authorized (`_realtime_reply_task`); a tool follow-up continues a speech that plays.
        let (allow, mut call) = (allow_interruptions, call);
        let speech = match &request {
            Request::ToolFollowUp { .. } => None,
            Request::Say { add_to_chat_ctx, .. } => {
                let add = *add_to_chat_ctx;
                Some(self.start_realtime_speech(now, Some(call.take()), false, allow, add, out))
            }
            Request::GenerateReply { read_back, .. } => {
                let asked = Some(call.take());
                let speech = self.start_realtime_speech(now, asked, false, allow, true, out);
                if *read_back {
                    let ev = crate::workflows::dtmf::DtmfTaskEvent::ReadBack(Some(speech));
                    self.dtmf_step(ev, out);
                }
                Some(speech)
            }
            Request::UserTurn { .. } => {
                Some(self.start_realtime_speech(now, None, true, allow, true, out))
            }
        };
        let t = &mut self.activity.realtime_state;
        match speech {
            // Refused by the scheduler: it never plays, so nothing is asked.
            Some(speech) if !self.activity.speech_scheduler.contains(speech) => {
                self.close_realtime_player(speech, out);
                None
            }
            // The caller was answered with the speech.
            Some(speech) => {
                t.register(req_id, request, None, allow, Some(speech), Phase::Unsent(cmd));
                Some(speech)
            }
            None => {
                let key =
                    t.register(req_id.clone(), request, call, allow, None, Phase::Unsent(cmd));
                self.send_request_now(now, &req_id, key, out);
                None
            }
        }
    }

    /// The speech announced for a request is authorized: the request goes to the server now.
    pub(crate) fn realtime_authorized(&mut self, now: Ts, speech: SpeechId, out: &mut Out) {
        let Some((req_id, _, key)) = self.activity.realtime_state.unsent(speech) else { return };
        self.send_request_now(now, &req_id, key, out);
    }

    /// A reply's request goes to the server, with its timer, unless the server is busy: then it
    /// waits in the queue and goes when the reply the server generates (or the request it
    /// answers) is done -- one at a time, as speeches are scheduled; nothing is cut or lost.
    fn send_request_now(&mut self, now: Ts, req_id: &str, key: RequestKey, out: &mut Out) {
        if let Some(cmd) = self.activity.realtime_state.send_or_queue(req_id) {
            self.rt_send(cmd, out);
            out.machine().arm(Timer::RealtimeRequest(key), now.saturating_add(REQUEST_TIMEOUT));
        }
    }

    /// The queue's next request, if the server is free for it.
    pub(crate) fn realtime_send_queued(&mut self, now: Ts, out: &mut Out) {
        if self.activity.realtime_state.ended() || self.activity.realtime_state.closing {
            return;
        }
        if let Some((cmd, key)) = self.activity.realtime_state.next_queued() {
            self.rt_send(cmd, out);
            out.machine().arm(Timer::RealtimeRequest(key), now.saturating_add(REQUEST_TIMEOUT));
        }
    }

    /// Its timer goes with it.
    fn take_pending(&mut self, req_id: &str, out: &mut Out) -> Option<Pending> {
        let pending = self.activity.realtime_state.pending.remove(req_id)?;
        out.machine().cancel(Timer::RealtimeRequest(pending.key));
        Some(pending)
    }

    /// A pending request leaves. One announced with its speech ends that speech: a failure plays
    /// into it as its error (`speech_handle._mark_done(error=e)`); an interruption already cut it.
    fn settle(&mut self, pending: Pending, how: Settled, out: &mut Out) {
        let Some(speech) = pending.speech else {
            return self.settle_request(pending.request, pending.call, how, out);
        };
        let error = match how {
            Settled::Interrupted(_) => None,
            Settled::Failed(error) | Settled::TimedOut(error) => Some(error),
            Settled::Ended => Some(RealtimeError::new(REALTIME_SESSION_ENDED)),
        };
        if let Some(error) = error {
            let failed = rt::GenerateReplyFailed { req_id: String::new(), error };
            self.play_realtime(speech, Event::GenerateReplyFailed(failed), out);
        }
        self.close_realtime_player(speech, out);
    }

    fn settle_request(
        &mut self,
        request: Request,
        call: Option<SpeechCall>,
        how: Settled,
        out: &mut Out,
    ) {
        let kind = match request {
            Request::Say { .. } => SpeechKind::Say,
            // A follow-up that will never come: the player finishes with what it has.
            Request::ToolFollowUp { speech, .. } => {
                self.close_realtime_player(speech, out);
                return;
            }
            Request::GenerateReply { read_back, .. } => {
                // A read-back that will never be spoken releases the collection's guard now.
                if read_back {
                    self.dtmf_step(crate::workflows::dtmf::DtmfTaskEvent::ReadBack(None), out);
                }
                SpeechKind::GenerateReply
            }
            Request::UserTurn { .. } => SpeechKind::GenerateReply,
        };
        self.answer_waiter(call, kind, how, out);
    }

    /// Every pending request leaves through exactly one settlement; `cancel` also sends
    /// `CancelReply` for those on the wire. An interruption leaves the request of a speech it did
    /// not cut: queued, it is asked when that speech is authorized.
    fn settle_all(&mut self, how: Settled, cancel: bool, out: &mut Out) {
        for (req_id, pending) in std::mem::take(&mut self.activity.realtime_state.pending) {
            let playing_on = pending.speech.is_some_and(|s| {
                self.activity.speech_scheduler.handle(s).is_some_and(|h| !h.interrupted())
            });
            if playing_on && matches!(how, Settled::Interrupted(_)) {
                self.activity.realtime_state.pending.insert(req_id, pending);
                continue;
            }
            out.machine().cancel(Timer::RealtimeRequest(pending.key));
            let asked = matches!(pending.phase, Phase::Sent(_));
            if cancel && asked && !self.activity.realtime_state.ended() {
                // The server is busy with it until it answers, within a bound of its own.
                let at = self.step.now.saturating_add(REQUEST_TIMEOUT);
                out.machine().arm(Timer::RealtimeRequest(pending.key), at);
                self.activity.realtime_state.cancelled_requests.insert(req_id.clone(), pending.key);
                self.rt_send(RealtimeCommand::CancelReply(rt::CancelReply { req_id }), out);
            }
            self.settle(pending, how.clone(), out);
        }
    }

    /// The request timed out: the server is told not to answer, the caller learns the timeout.
    pub(crate) fn realtime_request_timed_out(&mut self, key: RequestKey, out: &mut Out) {
        // The server never generated the reply its tools asked for: the speech ends on what it
        // spoke (5 s wait).
        if let Some(speech) = self.activity.realtime_state.auto_follow_up_timed_out(key) {
            diagnose(out, DiagnosticEvent::warning(AUTO_TOOL_REPLY_TIMED_OUT));
            self.close_realtime_player(speech, out);
            return;
        }
        let t = &mut self.activity.realtime_state;
        // A cancelled request the server never answered: it no longer holds the next one back.
        let cancelled =
            t.cancelled_requests.iter().find(|(_, k)| **k == key).map(|(id, _)| id.clone());
        if let Some(id) = cancelled {
            t.cancelled_requests.remove(&id);
            let now = self.step.now;
            return self.realtime_send_queued(now, out);
        }
        // Answered in the same instant the timer fired: nothing owed.
        let Some((req_id, pending)) = t.take_by_key(key) else { return };
        if !t.ended() && !t.closing {
            // Not counted busy: ten seconds unanswered, the server may never create it (a socket
            // that cannot reconnect); the session discards it by id if it does.
            self.rt_send(RealtimeCommand::CancelReply(rt::CancelReply { req_id }), out);
        }
        let error = RealtimeError::new(format!(
            "generate_reply timed out: no generation was created within {REQUEST_TIMEOUT:?}"
        ));
        self.settle(pending, Settled::TimedOut(error), out);
    }

    fn fail_request(&mut self, req_id: &str, error: RealtimeError, out: &mut Out) {
        // Refused for a reply the server had started as the request crossed it on the wire: it
        // waits for that reply, as a request asked over it would have.
        if error.code.as_deref() == Some("conversation_already_has_active_response")
            && let Some(key) = self.activity.realtime_state.requeue(req_id)
        {
            out.machine().cancel(Timer::RealtimeRequest(key));
            return;
        }
        // A cancelled request that will not generate is forgotten.
        if let Some(key) = self.activity.realtime_state.cancelled_requests.remove(req_id) {
            out.machine().cancel(Timer::RealtimeRequest(key));
        }
        // Unknown: already claimed by a `GenerationCreated`, or cancelled by the driver.
        if let Some(pending) = self.take_pending(req_id, out) {
            self.settle(pending, Settled::Failed(error), out);
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
        // The speech announced for it is over: nothing plays the answer.
        let players = &self.activity.realtime_players;
        let orphaned =
            matches!(&pending, Some(Some(Pending { speech: Some(s), .. })) if !players.contains(s));
        // Settled before the server answered (interrupted, closed, timed out), or closing.
        if matches!(pending, Some(None)) || orphaned || self.activity.realtime_state.closing {
            let t = &mut self.activity.realtime_state;
            // Cancelled while the request was in flight: the response landed anyway, and only an
            // interrupt reaches it now. A reply answered while closing, asked by the client or
            // opened by the server itself: nothing plays it either, and left generating it holds
            // the server, which refuses the next reply asked meanwhile.
            let cancelled = orphaned
                || req_id.as_deref().and_then(|id| t.cancelled_requests.remove(id)).is_some()
                || t.closing;
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
        let mut read_back = false;
        // `turn`: the reply to a user turn the client committed, which the reference creates
        // through `_generate_reply` and so announces as user-initiated.
        let (asked, allow_interruptions, add_to_chat_ctx, reuse, turn) = match pending.flatten() {
            // Announced at the request: the answer plays into it.
            Some(Pending { speech: Some(speech), allow_interruptions, .. }) => {
                (None, allow_interruptions, true, Some(Reuse { speech, step: 0 }), false)
            }
            Some(Pending { request, call, allow_interruptions, .. }) => match request {
                Request::Say { add_to_chat_ctx, .. } => {
                    (Some(call), allow_interruptions, add_to_chat_ctx, None, false)
                }
                Request::GenerateReply { read_back: asked_back, .. } => {
                    read_back = asked_back;
                    (Some(call), allow_interruptions, true, None, false)
                }
                Request::UserTurn { .. } => (None, allow_interruptions, true, None, true),
                Request::ToolFollowUp { speech, step, .. } => {
                    (None, allow_interruptions, true, Some(Reuse { speech, step }), false)
                }
            },
            // Server-initiated: the follow-up of a model that answers its own tools, or a new turn.
            None => {
                let follow_up = t.auto_follow_up.take();
                if let Some((_, key)) = follow_up {
                    out.machine().cancel(Timer::RealtimeRequest(key));
                }
                (None, None, true, follow_up.map(|(reuse, _)| reuse), false)
            }
        };
        t.open_generation(response_id.clone(), reuse.map(|r| r.speech));
        // The server is answering, so a parked speech was interrupted for real: the pause ends by
        // interrupting, never by resuming under the answer.
        if self.activity.paused_speech.paused_handle().is_some() {
            self.cancel_speech_pause(now, out);
        }
        // A tool follow-up continues its speech; if that player is gone, a fresh speech.
        let speech = match reuse.filter(|r| self.activity.realtime_players.contains(&r.speech)) {
            Some(r) => r.speech,
            None => self.start_realtime_speech(
                now,
                asked,
                turn,
                allow_interruptions,
                add_to_chat_ctx,
                out,
            ),
        };
        self.activity.realtime_state.set_open_speech(&response_id, speech);
        if read_back {
            let ev = crate::workflows::dtmf::DtmfTaskEvent::ReadBack(Some(speech));
            self.dtmf_step(ev, out);
        }
    }

    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    /// `asked`: the speech is a host command's (`say`, `generate_reply`), with its call. `turn`: it
    /// answers a user turn the client committed.
    fn start_realtime_speech(
        &mut self,
        now: Ts,
        asked: Option<Option<SpeechCall>>,
        turn: bool,
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        out: &mut Out,
    ) -> SpeechId {
        // `allow_interruptions=false` holds only when the client owns replies; a self-replying
        // server interrupts on onset, so the caller is told once.
        let server_replies = self.activity.rt_session.as_ref().is_some_and(|s| {
            let caps = s.capabilities;
            caps.turn_detection && caps.auto_turn_reply_generation
        }) && self.activity.audio_recognition.turn_detection_mode()
            == TurnDetectionMode::RealtimeLlm;
        let enabled = self.session.config.effective.turn.interruption.enabled();
        let mut allow = allow_interruptions.unwrap_or(enabled);
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
        let source = ReplySource::Realtime;
        let spec = SpeechSpec {
            add_to_chat_ctx,
            host_asked: asked.is_some() || turn,
            host_handle: asked.as_ref().and_then(Option::as_ref).map(|c| Arc::clone(&c.speech)),
            ..SpeechSpec::new(SpeechKind::Realtime, source, allow)
        };
        let handle = self.start_speech(now, spec, out);
        self.activity.realtime_players.insert(handle.id());
        if let Some(call) = asked {
            // Asked for by the host, so an open run owns it.
            self.run_own(crate::run_result::Owned::Speech(handle.id()), out);
            self.answer_speech(call);
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
        self.play_realtime(speech, ev, out);
    }

    fn on_generation_done(&mut self, done: rt::GenerationDone, out: &mut Out) {
        // Refused, discarded, or another generation's end while this one is open: nothing to end.
        let Some(speech) = self.activity.realtime_state.generation_done(&done.response_id) else {
            return;
        };
        if let Some(speech) = speech {
            let status = format!("{:?}", done.status);
            self.play_realtime(speech, Event::GenerationDone(done), out);
            // The generation ended on the server; the speech finishes by its own playout.
            out.trace(RealtimeStep::GenerationStatus { speech, status });
        }
    }

    /// `RealtimeModelError` reaches `_on_error`, which has no counter for it and closes on the
    /// first unrecoverable one.
    fn on_realtime_error(&mut self, now: Ts, err: rt::RealtimeModelError, out: &mut Out) {
        let error = RealtimeError::with_code(err.message.clone(), err.code.clone());
        let recoverable = err.recoverable;
        let reported = ProviderError::Local(
            err.provider_error
                .unwrap_or_else(|| ApiError::new(err.message))
                .with_retryable(recoverable),
        );
        let source = crate::events::ErrorSource::Realtime;
        if recoverable {
            let pending = err
                .req_id
                .as_deref()
                .and_then(|id| self.activity.realtime_state.pending.remove(id));
            self.report_error(now, source, true, reported, None, out);
            if let Some(pending) = pending {
                out.machine().cancel(Timer::RealtimeRequest(pending.key));
                self.settle(pending, Settled::Failed(error), out);
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
        for cmd in interrupts {
            self.rt_send(cmd, out);
        }
        for (_, pending) in pending {
            out.machine().cancel(Timer::RealtimeRequest(pending.key));
            self.settle(pending, Settled::Failed(error.clone()), out);
        }
        if close {
            self.close_on_realtime_failure(now, out);
        }
    }

    /// `_swap`: a new session holds nothing. What plays is cut (`interrupt(force=True)`), so what
    /// the caller heard is committed; the new session is given the frame once those speeches are
    /// released, and a reply in progress is asked for again.
    fn on_session_swapped(&mut self, now: Ts, regenerate: bool, out: &mut Out) {
        if self.activity.realtime_state.closing || self.activity.realtime_state.ended() {
            return;
        }
        let state = self.session.states.agent_state();
        let busy = matches!(state, AgentState::Speaking | AgentState::Thinking);
        // The new server's VAD heard no onset: a turn the old one opened ends here, or nothing
        // would ever end it.
        if self.activity.audio_recognition.turn_detection_mode() == TurnDetectionMode::RealtimeLlm
            && self.activity.audio_recognition.user_speaking()
        {
            let eff = self.activity.audio_recognition.on_server(now, ServerSpeech::Stopped);
            out.machine().append(eff);
        }
        let cut =
            self.interrupt_all(now, true, InterruptCause::ModelSwapped, out).unwrap_or_default();
        let waiting =
            cut.into_iter().filter(|s| self.activity.speech_scheduler.contains(*s)).collect();
        let t = &mut self.activity.realtime_state;
        // What the old session was asked and cancelled will never be answered.
        t.cancelled_requests.clear();
        t.swap = Some(Swap { waiting, regenerate: regenerate && busy });
        self.realtime_replay(now, None, out);
    }

    /// The swap's frame goes to the new session once the last speech it cut (`released`) is
    /// released.
    pub(crate) fn realtime_replay(&mut self, now: Ts, released: Option<SpeechId>, out: &mut Out) {
        let t = &mut self.activity.realtime_state;
        let Some(swap) = t.swap.as_mut() else { return };
        if let Some(speech) = released {
            swap.waiting.remove(&speech);
        }
        if !swap.waiting.is_empty() {
            return;
        }
        let regenerate = t.swap.take().is_some_and(|s| s.regenerate);
        if t.closing || t.ended() {
            return;
        }
        self.push_activity(Some(self.session.chat_ctx.for_realtime()), out);
        if regenerate {
            let request =
                Request::GenerateReply { instructions: None, tool_choice: None, read_back: false };
            self.realtime_request(now, request, None, None, out);
        }
    }

    /// The leg is dead (a realtime session is the LLM): the session closes with
    /// `CloseReason::Error`.
    fn close_on_realtime_failure(&mut self, now: Ts, out: &mut Out) {
        let error = ApiError::new("the realtime session ended with an unrecoverable error");
        let close = Close {
            reason: lk_agents_abi::voice::events::CloseReason::Error,
            drain: false,
            error: Some(crate::events::ErrorEvent {
                source: crate::events::ErrorSource::Realtime,
                recoverable: false,
                error: ProviderError::Local(error),
                speech: None,
            }),
        };
        self.begin_closing(now, close, out);
    }

    #[allow(clippy::too_many_arguments)]
    fn realtime_tools_done(
        &mut self,
        now: Ts,
        speech: SpeechId,
        step: u32,
        ask: bool,
        tool_choice: Option<lk_agents_abi::ToolChoice>,
        req_id: String,
        out: &mut Out,
    ) {
        let auto = self.activity.realtime_state.caps.auto_tool_reply_generation;
        let t = &mut self.activity.realtime_state;
        if !ask || t.ended() || t.closing {
            self.close_realtime_player(speech, out);
        } else if auto {
            // The server replies on its own; it is given 5 s to start.
            let key = t.follow_up_with(speech, step);
            out.machine()
                .arm(Timer::RealtimeRequest(key), now.saturating_add(AUTO_TOOL_REPLY_TIMEOUT));
        } else {
            // A draining activity's follow-up starts no new round (`draining`).
            let draining = self.activity.speech_scheduler.is_paused()
                || self.activity.speech_agent(speech).is_some_and(|f| *f != self.activity.agent);
            let tool_choice =
                if draining { Some(lk_agents_abi::ToolChoice::Forbidden) } else { tool_choice };
            let request = Request::ToolFollowUp { speech, step, tool_choice };
            self.send_request(now, req_id, request, None, None, out);
        }
    }

    pub(crate) fn fail_waiter(
        &mut self,
        call: Option<SpeechCall>,
        kind: SpeechKind,
        message: &str,
        out: &mut Out,
    ) {
        self.answer_waiter(call, kind, Settled::Failed(RealtimeError::new(message)), out);
    }

    /// A handle never in the speech queue, minted only when a caller awaits it. The host holds
    /// it, so it is announced and ended on the host stream like every other speech.
    pub(crate) fn answer_waiter(
        &mut self,
        call: Option<SpeechCall>,
        kind: SpeechKind,
        how: Settled,
        out: &mut Out,
    ) {
        if call.is_none() {
            return;
        }
        let allow_interruptions = self.session.config.effective.turn.interruption.enabled();
        let (id, handle) = self.new_speech(call.as_ref().map(|c| &c.speech), allow_interruptions);
        match how {
            Settled::Failed(err) | Settled::TimedOut(err) => {
                handle.mark_done(Some(fatal(err.message)))
            }
            Settled::Ended => handle.mark_done(Some(fatal(REALTIME_SESSION_ENDED.to_owned()))),
            Settled::Interrupted(cause) => {
                out.trace(InterruptStep::Interrupted { speech: id, cause, force: true });
                let _ = handle.interrupt(true);
                handle.mark_done(None);
            }
        }
        let speech = Arc::clone(&handle);
        out.machine().emit(SessionEvent::SpeechCreated { speech, kind, user_initiated: true });
        out.machine().emit(SessionEvent::SpeechDone { speech: id, outcome: handle.outcome() });
        self.answer_speech(call);
    }

    /// The server's turn detection was fixed at the open. After a re-resolution the mode is
    /// refined through the open's table and written back; a change that takes (local to local)
    /// or is refused (the server keeps the turn) is said once, never flipped silently.
    pub(crate) fn reconcile_realtime_turn_detection(
        &mut self,
        caps: &lk_agents_abi::llm::realtime::RealtimeCapabilities,
        out: &mut Out,
    ) -> Option<crate::audio_recognition::TurnDetectionMode> {
        let resolved = self.session.config.effective.turn.owner();
        let requested = self
            .activity
            .note_resolved_mode(resolved, self.session.config.effective.turn.detection());
        let side = SessionSide {
            stt: self.activity.audio_recognition.stt_enabled(),
            vad: self.input.sinks.contains_key(&crate::audio::AudioChannel::Vad),
            detection: self.session.config.effective.turn.detection(),
            streaming_detector: self.session.config.legs.detector
                == Some(crate::agent_activity::options::DetectorKind::Streaming),
            interruption_given: self.session.config.effective.turn.interruption.mode.is_some(),
            turn_detection_mode: resolved,
        };
        let disabled = self.activity.realtime_turn_detection_disabled;
        let mode = realtime_turn_detection_mode(caps, side, disabled);
        self.session.config.effective.turn.owner = Some(mode);
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
        call: Option<SpeechCall>,
        read_back: bool,
        out: &mut Out,
    ) -> Option<SpeechId> {
        self.activity.rt_session.as_ref()?;
        if let Some(text) = user_input {
            let wall = self.step.wall();
            let item = ChatMessage::text(self.session.ids.mint("item"), Role::User, &text, wall);
            let push = rt::PushItem { item: Box::new(item.clone().into()), reply_required: true };
            self.rt_send(RealtimeCommand::PushItem(push), out);
            self.commit_item(item.into(), Source::UserTyped, Visibility::Both, out);
        }
        let request = Request::GenerateReply { instructions, tool_choice, read_back };
        self.realtime_request(now, request, call, allow_interruptions, out)
    }

    fn on_input_transcription(
        &mut self,
        now: Ts,
        t: lk_agents_abi::llm::realtime::InputTranscription,
        started_at: Option<f64>,
        out: &mut Out,
    ) {
        let speech = ServerSpeech::Transcription {
            text: t.transcript.clone(),
            is_final: t.is_final,
            item_id: t.item_id.clone(),
        };
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
        // A placeholder in the model's context (`_on_remote_item_added`): the message or the
        // transcript that fills it is what reaches history.
        self.commit_item(item, Source::Remote { item_id: id }, Visibility::Model, out);
    }

    /// The open generation cancelled and truncated to what was heard. The interruption decision is
    /// `decide_interrupt`'s; this only actuates it.
    pub(crate) fn realtime_interrupt(&mut self, cause: InterruptCause, out: &mut Out) {
        if self.activity.rt_session.is_none() {
            return;
        }
        if self.session.closing_past_edge() {
            return self.realtime_on_close(out);
        }
        if self.activity.realtime_state.ended() {
            return self.settle_all(Settled::Ended, false, out);
        }
        // Only a generation whose speech stopped: the cancel is session-wide, and a speech that
        // plays on keeps its response. What was heard of it is truncated once its playout says
        // (`RealtimeWake::Heard`).
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
        for cmd in interrupts {
            self.rt_send(cmd, out);
        }
        self.settle_all(Settled::Interrupted(cause), true, out);
    }

    /// The session-level `interrupt()`'s `self._rt_session.interrupt()`, unconditional: with no
    /// generation of its own to cancel (`generating`, read before the speeches were cut) the
    /// socket is still told, unaddressed.
    pub(crate) fn realtime_interrupt_session(&mut self, generating: bool, out: &mut Out) {
        let ended = self.activity.realtime_state.ended();
        if self.activity.rt_session.is_some() && !ended && !generating {
            self.rt_send(RealtimeCommand::Interrupt(rt::Interrupt { response_id: None }), out);
        }
    }

    /// The session closes: every pending request settles `Closing`; every open generation is
    /// cancelled.
    pub(crate) fn realtime_on_close(&mut self, out: &mut Out) {
        self.activity.realtime_state.closing = true;
        for cmd in self.activity.realtime_state.cancel_where(|_| true) {
            self.rt_send(cmd, out);
        }
        self.settle_all(Settled::Interrupted(InterruptCause::Closing), true, out);
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
        // The reference ignores `stt` under a realtime model: its unset, VAD-based mode.
        TurnDetectionMode::Vad | TurnDetectionMode::Stt | TurnDetectionMode::RealtimeLlm => {
            if side.vad || side.stt { TurnDetectionMode::Vad } else { TurnDetectionMode::Manual }
        }
    }
}
