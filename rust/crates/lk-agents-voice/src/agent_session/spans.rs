//! The session's own spans, opened and ended through `self.telemetry`: `agent_session` (the
//! session's), `session_start` (the opening's), `session_close` (the close's) and `amd` (the
//! detection's), the session's events, the turn span of each chat item, and the close that ends
//! every span the session's entities still own.

use crate::agent::Source;
use crate::agent_session::{AmdSpan, VoiceDriver};
use crate::events::ErrorEvent;
use crate::telemetry::{
    self, AmdDetection, Holding, SessionTelemetry, Trace, UserTurnSpan, participant_attrs,
};
use lk_agents_abi::Ts;
use lk_agents_abi::telemetry::decl::{Content, Seconds, SpanDecl, SpanRef};
use lk_agents_abi::telemetry::spans::common::{GenAiOperation, ParticipantAttrs};
use lk_agents_abi::telemetry::spans::session::{
    AgentSession, AgentStateChanged, Amd, AmdVerdict, CloseReason, ConnectionStateChanged,
    EventLoopBlockedEvent, ParticipantDisconnected, ParticipantLinked, SessionClose, SessionStart,
    UserStateChanged,
};
use lk_agents_abi::voice::events::{AgentState, UserState};
use lk_agents_abi::voice::room_io::ParticipantInfo;
use lk_agents_abi::voice::rtc::{ConnectionState, DisconnectReason};
use lk_agents_core::telemetry::span::SpanFailure;
use lk_agents_core::tts::markup::ChatMessageExt;

/// The session's trace: span ids from `telemetry.ids`, else seeded from the session id (tests,
/// the DST); `gen_ai.conversation.id` the call's, else the session's own id.
pub(crate) fn trace(session: &str, telemetry: &mut SessionTelemetry, now: super::Now) -> Trace {
    let ids =
        telemetry.ids.take().unwrap_or_else(|| telemetry::seeded_ids(telemetry::fnv(session)));
    let conversation = telemetry.conversation.take().or_else(|| Some(session.to_owned()));
    Trace::new(ids, now.at, now.wall, telemetry.job.take(), conversation)
}

impl VoiceDriver {
    /// `start()`, as the session is built: `agent_session` and `session_start` open, as `start()`
    /// reserved them.
    pub(super) fn open_session_spans(&mut self, telemetry: SessionTelemetry, agent: &str) {
        let SessionTelemetry { parent, job_ids, opening, .. } = telemetry;
        let decl = AgentSession {
            agent_label: agent.to_owned(),
            operation: GenAiOperation::InvokeWorkflow,
            workflow: AgentSession::NAME.to_owned(),
            dispatch_id: job_ids.dispatch_id,
            worker_id: job_ids.worker_id,
            agent_id: job_ids.agent_id,
        };
        let trace = &mut self.telemetry;
        let (root, start) = match opening {
            Some(o) => {
                self.activity.arriving = o.activity.map(|activity| telemetry::Arriving {
                    agent: agent.to_owned(),
                    update: None,
                    activity,
                });
                let root = trace.open_root(parent, Some(o.session), decl);
                let start = trace.open_in_reserved(&root, o.start, SessionStart);
                (root, start)
            }
            None => {
                let root = trace.open_root(parent, None, decl);
                let start = trace.open_in(&root, SessionStart);
                (root, start)
            }
        };
        self.session.starting = Some(Holding { span: start, done: false });
        self.session.span = Some(root);
    }

    /// `session_start`'s context while it is open: the room connection and the waits the runtime
    /// runs for it nest under it.
    pub fn session_start_span(&self) -> Option<SpanRef> {
        self.telemetry.context(&self.session.starting.as_ref()?.span)
    }

    /// The opening cycle is done: `session_start` ends, unless the first `on_enter` still runs.
    pub(crate) fn session_opened(&mut self) {
        if let Some(start) = &mut self.session.starting {
            start.done = true;
        }
        self.settle_holdings();
    }

    /// `session_close`, from the request to the close.
    pub(crate) fn session_close_span(
        &mut self,
        reason: lk_agents_abi::voice::events::CloseReason,
        drain: bool,
        error: Option<&ErrorEvent>,
    ) -> Option<telemetry::Span<SessionClose>> {
        use lk_agents_abi::voice::events::CloseReason as R;
        let root = self.session.span.as_ref()?;
        let reason = match reason {
            R::Error => CloseReason::Error,
            R::JobShutdown => CloseReason::JobShutdown,
            R::ParticipantDisconnected => CloseReason::ParticipantDisconnected,
            R::UserInitiated => CloseReason::UserInitiated,
            R::TaskCompleted => CloseReason::TaskCompleted,
        };
        let decl = SessionClose { reason, drain, error: error.map(|e| e.error.error_type()) };
        Some(self.telemetry.open_in(root, decl))
    }

    /// An event-loop stall: counted on the session and marked as its event.
    pub(crate) fn event_loop_blocked(&mut self, duration: std::time::Duration) {
        let seconds = duration.as_secs_f64();
        self.session.blocking.stalled(seconds);
        let Some(root) = &self.session.span else { return };
        self.telemetry.event(root, EventLoopBlockedEvent { duration: Seconds(seconds) });
    }

    /// The AMD detector attached (`amd` opens, as the detector reserved it) or released its
    /// verdict (it ends).
    pub(crate) fn amd_span(&mut self, ev: &AmdSpan) {
        let Some(root) = &self.session.span else { return };
        let p = match ev {
            AmdSpan::Attached(reserved) => {
                if self.session.amd.is_none() {
                    let span = self.telemetry.open_in_reserved(root, *reserved, Amd);
                    self.session.amd = Some(AmdDetection { span });
                }
                return;
            }
            AmdSpan::Verdict(p) => p,
        };
        let Some(amd) = self.session.amd.take() else { return };
        let verdict = AmdVerdict {
            category: p.category.as_str().to_owned(),
            reason: p.reason.as_str().to_owned(),
            speech_duration: Seconds(p.speech_duration.as_secs_f64()),
            delay: Seconds(p.delay.as_secs_f64()),
            transcript: Content(p.transcript.clone()),
        };
        self.telemetry.end(amd.span, Some(verdict));
    }

    /// The room's connection state changed: the session's event.
    pub(crate) fn connection_span_event(&mut self, state: ConnectionState) {
        let Some(root) = &self.session.span else { return };
        let state = format!("CONN_{}", state.as_str().to_ascii_uppercase());
        self.telemetry.event(root, ConnectionStateChanged { state });
    }

    /// The linked participant joined or left (`reason`): the session's event, with the room's
    /// description of it when it gave one before the link (else its identity only).
    pub(crate) fn participant_span_event(
        &mut self,
        identity: &str,
        known: Option<ParticipantInfo>,
        left: Option<DisconnectReason>,
    ) {
        let Some(root) = &self.session.span else { return };
        let participant = known.as_ref().map_or_else(
            || ParticipantAttrs { id: None, identity: Content(identity.to_owned()), kind: None },
            participant_attrs,
        );
        match left {
            None => self.telemetry.event(root, ParticipantLinked { participant }),
            Some(reason) => {
                let reason = telemetry::disconnect_reason(reason);
                self.telemetry.event(root, ParticipantDisconnected { participant, reason });
            }
        }
    }

    /// The agent state changed: the session's event; leaving `speaking` ends `agent_speaking`.
    pub(crate) fn agent_state_span_event(&mut self, old: AgentState, new: AgentState) {
        if let Some(root) = &self.session.span {
            let event = AgentStateChanged { old: old.as_str().into(), new: new.as_str().into() };
            self.telemetry.event(root, event);
        }
        if old == AgentState::Speaking {
            self.agent_stopped_speaking();
        }
    }

    /// The user state changed at `at`: the session's event.
    pub(crate) fn user_state_span_event(&mut self, old: UserState, new: UserState, at: Ts) {
        let Some(root) = &self.session.span else { return };
        let at = at.min(self.telemetry.now());
        let event = UserStateChanged { at, old: old.as_str().into(), new: new.as_str().into() };
        self.telemetry.event(root, event);
    }

    /// A chat item goes to the host: the turn it was produced in, for the session report. A
    /// reply's words join its turn's response; typed input is a user turn of its own.
    pub(crate) fn item_turn(&mut self, item: &lk_agents_abi::ChatItem, source: &Source) {
        let text = match item {
            lk_agents_abi::ChatItem::Message(m) => Some(m.text_content().unwrap_or_default()),
            _ => None,
        };
        let turns = &self.session.item_turns;
        let turn = match (source, text) {
            (Source::Reply { speech, .. } | Source::Say { speech }, Some(text)) => {
                if let Some(turn) = self.agent_turn(*speech) {
                    turn.response(&text);
                }
                self.session.item_turns.of_speech(*speech)
            }
            (Source::UserInput { .. } | Source::UserTyped, Some(text)) => {
                let participant = self.session.room.linked_participant();
                UserTurnSpan::typed(&mut self.telemetry, participant, &text)
            }
            (Source::UserTurn { turn }, _) => turns.of_user_turn(*turn),
            (source, _) => source.speech().and_then(|s| turns.of_speech(s)),
        };
        if let Some(turn) = turn {
            self.session.item_turns.item(item.id(), turn);
        }
    }

    /// The close: every entity still owning a span ends it, children first (the activity's user
    /// turns, agent turns and callbacks, the AMD detection, the keyterm pass, `session_start`,
    /// `session_close`), then the session. The sweep after it finds nothing.
    pub(crate) fn close_spans(&mut self) {
        let Some(root) = self.session.span.take() else { return };
        self.close_activity();
        if let Some(amd) = self.session.amd.take() {
            self.telemetry.end(amd.span, None);
        }
        self.keyterm_span_ended(None);
        if let Some(start) = self.session.starting.take() {
            self.telemetry.end(start.span, ());
        }
        if let Some(close) = self.session.closing_span.take() {
            self.telemetry.end(close, ());
        }
        self.telemetry.end(root, self.session.blocking.session_end());
        let swept = self.telemetry.close_root();
        if swept > 0 {
            log::warn!(
                target: "livekit.agents",
                swept = swept;
                "spans of the session were still open after their owners ended them"
            );
        }
    }

    /// The runtime drops the session without its close (a cut-off loop): every span still open
    /// ends as at a close and joins the outbox. `true` when the session was open.
    pub fn abandon_spans(&mut self) -> bool {
        if self.session.span.is_none() {
            return false;
        }
        log::warn!(target: "livekit.agents", "the voice session ended without its close");
        self.close_spans();
        self.flush_spans();
        true
    }

    /// Spans the session's close sweep ended because their owner never did: a bug.
    pub fn spans_swept(&self) -> u32 {
        self.telemetry.swept()
    }
}

/// Only the close (or the runtime's abandon, which exports what it ends) ends a session's spans:
/// ending them here would lose them with the trace.
impl Drop for VoiceDriver {
    fn drop(&mut self) {
        if self.session.span.is_none() || std::thread::panicking() {
            return;
        }
        debug_assert!(!self.exported, "the session's spans were dropped without its close");
        // A unit test's core, or a release build's bug: the spans end, the records go nowhere.
        self.close_spans();
    }
}
