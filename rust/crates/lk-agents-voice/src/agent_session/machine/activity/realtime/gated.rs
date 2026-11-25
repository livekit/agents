//! Realtime replies under the AMD reply gate.
//!
//! A realtime session reaches the gate only with client-side turn detection: a server that
//! detects turns starts its replies itself, past the gate. A gated turn commits the server's
//! audio buffer at commit, before its decision, so a denied turn leaves no audio for the next
//! allowed reply. An allowed reply asks the server once, at authorization, so a reply held for
//! silence starts generating after it. Its responses carry the decision's stage instructions as
//! per-response instructions, passed through unchanged, and the session's tools carry AMD's DTMF
//! tool from its first response until its speech ends, since a response request names no tools.

use super::super::RealtimeActivity;
use crate::agent::{Source, Visibility};
use crate::agent_session::VoiceDriver;
use crate::agent_session::machine::activity::scheduling::SpeechId;
use crate::amd::AmdAttachRefused;
use crate::audio_recognition::{EndOfTurnInfo, TurnId};
use crate::tool_executor::Toolset;
use lk_agents_abi::llm::realtime as rt;
use lk_agents_abi::{ChatItem, ChatMessage, RealtimeCommand, Role, ToolSpec};
use std::sync::Arc;

/// What an AMD decision adds to its realtime reply's responses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatedResponse {
    /// Stage instructions for the reply and its tool follow-ups.
    pub instructions: Option<String>,
    /// AMD's DTMF tool, offered next to the agent's.
    pub dtmf: Option<ToolSpec>,
}

impl RealtimeActivity {
    /// The server segments the input itself: a client commit of its buffer is refused.
    pub(in crate::agent_session::machine::activity) fn server_segments(&self) -> bool {
        self.session.capabilities.turn_detection && !self.turn_detection_disabled
    }
}

impl VoiceDriver {
    /// Why AMD cannot run on this realtime session (spec §5.1), in `detector.py`'s order: the
    /// server detects turns, a response cannot carry the decision's tools or the server answers
    /// tool results itself, or nothing transcribes the callee: neither session STT nor AMD's own.
    pub(crate) fn realtime_amd_refusal(&self, amd_stt: bool) -> Option<AmdAttachRefused> {
        let rt = self.activity.realtime_activity()?;
        let caps = rt.session.capabilities;
        if rt.server_segments() {
            Some(AmdAttachRefused::RealtimeServerTurnDetection)
        } else if caps.auto_tool_reply_generation || !caps.per_response_tool_choice {
            Some(AmdAttachRefused::RealtimeTools)
        } else if !self.activity.audio_recognition.stt_enabled() && !amd_stt {
            Some(AmdAttachRefused::RealtimeNoStt)
        } else {
            None
        }
    }

    /// A turn committed under the gate asks for a reply: its audio becomes the server's user item
    /// now, before the decision.
    pub(crate) fn realtime_gate_commit(&mut self) {
        if self.activity.realtime_activity().is_some_and(|rt| !rt.server_segments()) {
            self.rt_send(RealtimeCommand::CommitAudio(rt::CommitAudio {}));
        }
    }

    /// The user item of a turn the gate decided, whether it replies, is denied or is superseded:
    /// the one owner of its words. A pipeline keeps the STT's words. On a realtime model the
    /// server's transcript of the committed audio is the turn's item, so only a model without
    /// user transcription keeps the session STT's words, when there are some; with no session
    /// STT, AMD's `InsertUserItem` brings them. A realtime item stays off the wire: the server
    /// holds the turn's audio.
    pub(crate) fn commit_gated_turn_item(&mut self, info: &EndOfTurnInfo, text: Option<&str>) {
        let transcribes = self.activity.realtime().map(|rt| rt.capabilities.user_transcription);
        let words =
            self.activity.audio_recognition.stt_enabled() && !info.new_transcript.is_empty();
        if transcribes.is_some_and(|transcribes| transcribes || !words) {
            return;
        }
        let user = self.user_turn_item(info, text);
        if transcribes.is_some() {
            self.session.chat_ctx.server_holds_audio(&user.id);
        }
        let source = Source::UserTurn { turn: info.turn_id };
        self.commit_item(ChatItem::Message(user), source, Visibility::Both);
    }

    /// `InsertUserItem`: AMD's transcript of a turn, for a realtime model with neither user
    /// transcription nor session STT, so the item is never added twice. It goes to the agent's
    /// view and history, not to the server, which holds the turn's audio.
    pub(crate) fn amd_insert_user_item(&mut self, turn: TurnId, text: String) -> bool {
        let stt = self.activity.audio_recognition.stt_enabled();
        let needed = self.activity.realtime().is_some_and(|rt| !rt.capabilities.user_transcription);
        if !needed || stt || text.trim().is_empty() {
            return false;
        }
        let id = self.user_input_item_id(turn);
        self.session.chat_ctx.server_holds_audio(&id);
        let mut item = ChatMessage::text(id, Role::User, &text, self.step.wall());
        item.transcript_confidence = Some(0.0);
        self.commit_item(ChatItem::Message(item), Source::UserTurn { turn }, Visibility::Both);
        true
    }

    /// The tools a gated reply's speech runs: the agent's, with AMD's DTMF tool.
    pub(super) fn gated_tools(&self, gated: &Option<GatedResponse>) -> Option<Arc<Toolset>> {
        let dtmf = gated.as_ref()?.dtmf.as_ref()?;
        let mut toolset = (*self.activity.advertised_toolset(&self.session.tools)).clone();
        toolset.specs.push(dtmf.clone());
        Some(Arc::new(toolset))
    }

    /// A request goes on the wire. The server first gets the tools of the speech it answers: a
    /// gated reply's, or the agent's again.
    pub(super) fn rt_send_request(&mut self, cmd: RealtimeCommand) {
        let Some(rt) = self.activity.realtime_activity() else { return };
        let t = &rt.state;
        let pending = cmd.req_id().and_then(|id| t.pending.get(id));
        let speech = pending.and_then(|p| match p.request {
            super::Request::ToolFollowUp { speech, .. } => Some(speech),
            _ => p.speech,
        });
        let wanted = speech.filter(|s| t.gated.get(s).is_some_and(|g| g.dtmf.is_some()));
        if wanted != t.tools_for {
            self.set_server_tools(wanted);
        }
        self.rt_send(cmd);
    }

    /// The speech's responses are over: its tools leave the server.
    pub(super) fn realtime_response_closed(&mut self, speech: SpeechId) {
        let Some(rt) = self.activity.realtime_activity_mut() else { return };
        rt.state.gated.remove(&speech);
        if rt.state.tools_for == Some(speech) {
            self.set_server_tools(None);
        }
    }

    /// The agent's tools, or `speech`'s gated tools: the agent's with AMD's DTMF tool.
    fn set_server_tools(&mut self, speech: Option<SpeechId>) {
        let Some(rt) = self.activity.realtime_activity_mut() else { return };
        let dtmf = speech.and_then(|s| rt.state.gated.get(&s)?.dtmf.clone());
        rt.state.tools_for = speech.filter(|_| dtmf.is_some());
        let mut tools = self.activity.toolset.specs.clone();
        tools.extend(dtmf.filter(|tool| !tools.iter().any(|t| t.name == tool.name)));
        self.rt_send(RealtimeCommand::UpdateTools(rt::UpdateTools { tools }));
    }

    /// Something sent the agent's tools to the server: no gated tools are left on it.
    pub(crate) fn realtime_tools_reset(&mut self) {
        if let Some(rt) = self.activity.realtime_activity_mut() {
            rt.state.tools_for = None;
        }
    }
}
