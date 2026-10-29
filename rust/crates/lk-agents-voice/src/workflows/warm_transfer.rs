//! Warm transfer configuration and lifecycle decisions. Runtime owns rooms, requests and tasks.

use super::instructions::render;
use super::prompts::warm_transfer as p;
pub use lk_agents_abi::voice::workflows::warm_transfer::{
    HUMAN_AGENT_IDENTITY, TWILIO_RINGING_TIMEOUT, WarmTransferRefused,
};
use lk_agents_abi::voice::workflows::{InstructionsArg, IoEnableState, TwilioDial};
use lk_agents_abi::{ChatContext, Maybe};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::time::Duration;

/// The human agent's room: the caller's, suffixed.
pub fn human_agent_room_name(caller_room: &str) -> String {
    format!("{caller_room}-human-agent")
}

/// The caller conversation as the briefing reads it: the caller's and the assistant's turns that
/// have text, one per line.
pub fn format_conversation_history(chat_ctx: &ChatContext) -> String {
    let mut out = String::new();
    for msg in chat_ctx.messages() {
        let role = match msg.role {
            lk_agents_abi::Role::User => "Caller",
            lk_agents_abi::Role::Assistant => "Assistant",
            lk_agents_abi::Role::System | lk_agents_abi::Role::Developer => continue,
        };
        let Some(text) = msg.raw_text_content() else { continue };
        if text.is_empty() {
            continue;
        }
        out.push_str(role);
        out.push_str(": ");
        out.push_str(&text);
        out.push('\n');
    }
    out
}

/// The caller session's io while the transfer runs: off, then back to what it was. The originals
/// are read once, so a leg off before the transfer stays off after it.
#[derive(Debug, Default)]
pub struct IoGate {
    original: Option<IoEnableState>,
}

impl IoGate {
    /// What the setters should be given. `current` is read only the first time.
    pub fn desired(&mut self, enabled: bool, current: IoEnableState) -> IoEnableState {
        let original = *self.original.get_or_insert(current);
        IoEnableState {
            audio_input: enabled && original.audio_input,
            audio_output: enabled && original.audio_output,
            transcription_output: enabled && original.transcription_output,
        }
    }
}

/// What dials the human agent.
#[derive(Debug, Clone, PartialEq)]
pub enum Origination {
    /// `CreateSIPParticipant` on a trunk or a custom SIP connection, waiting until answered.
    Sip {
        sip_call_to: String,
        sip_trunk_id: Option<String>,
        sip_number: String,

        sip_headers: BTreeMap<String, String>,
        dtmf: Option<String>,
    },
    /// The LiveKit Twilio connector's stream, called through Twilio's Calls API.
    Twilio(TwilioDial),
}

/// `[(role, text)]` as a context with just what [`format_conversation_history`] reads. Anything
/// that is not a role literal reads as `developer`, which the briefing skips.
fn conversation_of(items: Vec<(String, String)>) -> ChatContext {
    use lk_agents_abi::{ChatContent, ChatItem, ChatMessage, Role};
    ChatContext {
        items: items
            .into_iter()
            .enumerate()
            .map(|(i, (role, text))| {
                let role = Role::from_literal(&role).unwrap_or(Role::Developer);
                ChatItem::Message(ChatMessage::new(
                    format!("history-{i}"),
                    role,
                    vec![ChatContent::Text(text)],
                    std::time::UNIX_EPOCH,
                ))
            })
            .collect(),
    }
}

/// `connect_to_caller`, `decline_transfer` and `voicemail_detected`, as raw function schemas.
pub fn transfer_tools() -> Vec<BTreeMap<String, Value>> {
    let tool = |name: &str, description: &str, reason: Option<&str>| {
        let (properties, required) = match reason {
            Some(d) => {
                (json!({ "reason": { "type": "string", "description": d } }), json!(["reason"]))
            }
            None => (json!({}), json!([])),
        };
        BTreeMap::from([
            ("name".to_owned(), Value::String(name.to_owned())),
            ("description".to_owned(), Value::String(description.to_owned())),
            (
                "parameters".to_owned(),
                json!({
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": false,
                }),
            ),
        ])
    };
    vec![
        tool(p::CONNECT_TOOL, p::CONNECT_DESC, None),
        tool(p::DECLINE_TOOL, p::DECLINE_DESC, Some(p::DECLINE_ARG_REASON)),
        tool(p::VOICEMAIL_TOOL, p::VOICEMAIL_DESC, None),
    ]
}

/// The briefing instructions: the template over the persona, the extra section and the caller
/// conversation, unless fixed instructions were given.
pub fn briefing(
    instructions: Option<InstructionsArg>,
    conversation: Option<Vec<(String, String)>>,
) -> String {
    let (persona, extra) = match instructions {
        Some(InstructionsArg::Fixed(text)) => return text,
        Some(InstructionsArg::Workflow(sections)) => (sections.persona.given(), sections.extra),
        None => (None, String::new()),
    };
    let history = conversation.map(conversation_of).as_ref().map(format_conversation_history);
    render(
        p::INSTRUCTIONS_TEMPLATE,
        &[
            ("persona", persona.as_deref().unwrap_or(p::PERSONA)),
            ("extra", &extra),
            ("_conversation_history", &history.unwrap_or_default()),
        ],
    )
}

/// Unresolved constructor inputs; omission is kept until the plan is resolved.
pub struct TransferConfig {
    pub sip_call_to: Option<String>,
    pub sip_trunk_id: Maybe<Option<String>>,
    pub has_sip_connection: bool,
    pub sip_number: Maybe<String>,
    pub sip_headers: Option<BTreeMap<String, String>>,
    pub dtmf: Option<String>,
    pub ringing_timeout: Option<f64>,
    pub instructions: Maybe<InstructionsArg>,
    pub conversation: Option<Vec<(String, String)>>,
    pub env_sip_trunk: Option<String>,
    pub env_sip_number: String,
    pub twilio: Option<TwilioDial>,
}

pub struct TransferPlan {
    pub instructions: String,
    pub origination: Origination,
    pub ringing_timeout: Option<Duration>,
}

impl TransferPlan {
    pub fn resolve(config: TransferConfig) -> Result<Self, WarmTransferRefused> {
        let TransferConfig {
            sip_call_to,
            sip_trunk_id,
            has_sip_connection,
            sip_number,
            sip_headers,
            dtmf,
            ringing_timeout,
            instructions,
            conversation,
            env_sip_trunk,
            env_sip_number,
            twilio,
        } = config;
        let instructions = briefing(instructions.given(), conversation);
        let origination = match twilio {
            Some(t) => {
                if t.account_sid.is_empty() || t.auth_token.is_empty() {
                    return Err(WarmTransferRefused::NoTwilioCredentials);
                }
                if t.call_token.as_deref().is_some_and(|c| !c.is_empty())
                    && t.original_caller_number.as_deref().is_none_or(str::is_empty)
                {
                    return Err(WarmTransferRefused::CallTokenWithoutCaller);
                }
                Origination::Twilio(t)
            }
            None => {
                let sip_call_to = sip_call_to
                    .filter(|s| !s.is_empty())
                    .ok_or(WarmTransferRefused::NoSipCallTo)?;
                let sip_trunk_id = match sip_trunk_id {
                    Maybe::Given(id) => id,
                    Maybe::NotGiven if has_sip_connection => None,
                    Maybe::NotGiven => env_sip_trunk,
                };
                if sip_trunk_id.is_none() && !has_sip_connection {
                    return Err(WarmTransferRefused::NoTrunk);
                }
                Origination::Sip {
                    sip_call_to,
                    sip_trunk_id,
                    sip_number: sip_number.given().unwrap_or(env_sip_number),
                    sip_headers: sip_headers.unwrap_or_default(),
                    dtmf,
                }
            }
        };
        Ok(Self {
            instructions,
            origination,
            ringing_timeout: ringing_timeout
                .map(|seconds| Duration::from_secs_f64(seconds.max(0.0))),
        })
    }
}

/// Only settled user-visible outcomes are retained, never live host exceptions or resources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferFailure {
    Dial,
    Declined(String),
    Voicemail,
    RoomClosed(String),
    Host(String),
}

impl From<TransferFailure> for lk_agents_abi::voice::workflows::warm_transfer::TransferFailed {
    fn from(failure: TransferFailure) -> Self {
        match failure {
            TransferFailure::Dial => Self::Dial,
            TransferFailure::Declined(reason) => Self::Declined(reason),
            TransferFailure::Voicemail => Self::Voicemail,
            TransferFailure::RoomClosed(reason) => Self::RoomClosed(reason),
            TransferFailure::Host(message) => {
                Self::Host(lk_agents_abi::errors::HostError::new(message))
            }
        }
    }
}

pub type TransferOutcome = Result<String, TransferFailure>;

/// Resource-specific effects of a lifecycle transition. The runtime supplies their handles.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct TransferEffects {
    pub close_human: Option<bool>,
    pub stop_hold: bool,
    pub io: Option<IoEnableState>,
}

#[derive(Debug, Default, PartialEq, Eq)]
enum Phase {
    #[default]
    Idle,
    Dialling,
    Briefing,
    Merging,
    Settled,
}

pub enum TransferTool {
    Connect,
    Decline(String),
    Voicemail,
}

pub enum TransferToolEffect {
    MoveParticipant,
    Settled(TransferEffects),
}

/// The active transfer's phase, original I/O flags, and one correlated outcome.
#[derive(Default)]
pub struct Transfer {
    phase: Phase,
    human_open: bool,
    io: IoGate,
    outcome: Option<TransferOutcome>,
}

impl Transfer {
    pub fn begin(&mut self, current: IoEnableState) -> TransferEffects {
        *self = Self::default();
        self.phase = Phase::Dialling;
        TransferEffects { io: Some(self.io.desired(false, current)), ..Default::default() }
    }

    pub fn human_connected(&mut self) {
        if self.phase == Phase::Dialling {
            self.human_open = true;
            self.phase = Phase::Briefing;
        }
    }

    pub fn tool(&mut self, tool: TransferTool) -> Option<TransferToolEffect> {
        if self.phase != Phase::Briefing {
            return None;
        }
        match tool {
            TransferTool::Connect => {
                self.begin_merge();
                Some(TransferToolEffect::MoveParticipant)
            }
            TransferTool::Decline(reason) => {
                self.settle(Err(TransferFailure::Declined(reason))).map(TransferToolEffect::Settled)
            }
            TransferTool::Voicemail => {
                self.settle(Err(TransferFailure::Voicemail)).map(TransferToolEffect::Settled)
            }
        }
    }

    pub fn begin_merge(&mut self) -> bool {
        if self.phase != Phase::Briefing {
            return false;
        }
        self.phase = Phase::Merging;
        true
    }

    pub fn merge_failed(&mut self) {
        if self.phase == Phase::Merging {
            self.phase = Phase::Briefing;
        }
    }

    pub fn abort_dial(&mut self) -> TransferEffects {
        TransferEffects {
            close_human: std::mem::take(&mut self.human_open).then_some(false),
            ..Default::default()
        }
    }

    pub fn room_closed(&mut self, reason: String) -> Option<TransferEffects> {
        if self.phase == Phase::Merging {
            return None;
        }
        self.settle(Err(TransferFailure::RoomClosed(reason)))
    }

    pub fn settle(&mut self, outcome: TransferOutcome) -> Option<TransferEffects> {
        if matches!(self.phase, Phase::Idle | Phase::Settled) {
            return None;
        }
        self.outcome = Some(outcome);
        Some(self.finish(true))
    }

    pub fn cancel(&mut self) -> TransferEffects {
        self.finish(false)
    }

    pub fn take_outcome(&mut self) -> Option<TransferOutcome> {
        self.outcome.take()
    }

    fn finish(&mut self, drain: bool) -> TransferEffects {
        if self.phase == Phase::Settled {
            return TransferEffects::default();
        }
        self.phase = Phase::Settled;
        TransferEffects {
            close_human: std::mem::take(&mut self.human_open).then_some(drain),
            stop_hold: true,
            io: self.io.original.map(|original| self.io.desired(true, original)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sip(
        sip_call_to: Option<&str>,
        has_sip_connection: bool,
        env_trunk: Option<&str>,
    ) -> Result<TransferPlan, WarmTransferRefused> {
        TransferPlan::resolve(TransferConfig {
            sip_call_to: sip_call_to.map(str::to_owned),
            sip_trunk_id: Maybe::NotGiven,
            has_sip_connection,
            sip_number: Maybe::NotGiven,
            sip_headers: None,
            dtmf: None,
            ringing_timeout: None,
            instructions: Maybe::NotGiven,
            conversation: None,
            env_sip_trunk: env_trunk.map(str::to_owned),
            env_sip_number: String::new(),
            twilio: None,
        })
    }
    // Catches: the trunk fallback losing its middle arm (an env trunk overriding an explicit
    // `sip_connection`), or a transfer built with nothing to dial.
    #[test]
    fn an_explicit_sip_connection_beats_the_env_trunk() {
        let t = sip(Some("+15550100"), true, Some("env")).unwrap();
        assert!(matches!(t.origination, Origination::Sip { sip_trunk_id: None, .. }));
        let t = sip(Some("+15550100"), false, Some("env")).unwrap();
        assert!(
            matches!(t.origination, Origination::Sip { sip_trunk_id: Some(ref id), .. } if id == "env")
        );
        assert_eq!(sip(Some("+15550100"), false, None).err(), Some(WarmTransferRefused::NoTrunk));
        assert_eq!(sip(None, false, Some("env")).err(), Some(WarmTransferRefused::NoSipCallTo));
    }

    // Catches: the briefing losing the caller conversation or the persona.
    #[test]
    fn the_briefing_carries_the_conversation() {
        let i = briefing(
            None,
            Some(vec![("user".into(), "my order is late".into()), ("system".into(), "x".into())]),
        );
        assert!(i.contains("Caller: my order is late\n"), "{i}");
        assert!(i.contains("# Identity"));
        assert!(!i.contains("x\n## End"));
    }

    // Catches: a restore that re-enables a leg the caller had off before the transfer.
    #[test]
    fn the_io_gate_restores_what_was_there() {
        let mut gate = IoGate::default();
        let before =
            IoEnableState { audio_input: true, audio_output: false, transcription_output: true };
        let off = gate.desired(false, before);
        assert_eq!(
            off,
            IoEnableState { audio_input: false, audio_output: false, transcription_output: false }
        );
        assert_eq!(gate.desired(true, off), before);
    }

    #[test]
    fn merging_ignores_its_room_close_and_settles_cleanup_once() {
        let mut transfer = Transfer::default();
        let io = IoEnableState { audio_output: false, ..Default::default() };
        assert!(transfer.tool(TransferTool::Decline("too early".to_owned())).is_none());
        transfer.begin(io);
        assert!(!transfer.begin_merge());
        assert!(transfer.tool(TransferTool::Voicemail).is_none());
        transfer.human_connected();
        assert!(transfer.begin_merge());
        assert!(transfer.room_closed("moved".to_owned()).is_none());
        let cleanup = transfer.settle(Ok(HUMAN_AGENT_IDENTITY.to_owned())).unwrap();
        assert_eq!(cleanup.close_human, Some(true));
        assert!(cleanup.stop_hold);
        assert_eq!(cleanup.io, Some(io));
        assert!(transfer.room_closed("late".to_owned()).is_none());
        assert!(!transfer.begin_merge());
        assert_eq!(transfer.cancel(), TransferEffects::default());
        assert_eq!(transfer.take_outcome(), Some(Ok(HUMAN_AGENT_IDENTITY.to_owned())));
        assert!(transfer.take_outcome().is_none());
    }

    #[test]
    fn a_failed_move_restores_room_failure_observation() {
        let mut transfer = Transfer::default();
        transfer.begin(IoEnableState::default());
        transfer.human_connected();
        assert!(transfer.begin_merge());
        transfer.merge_failed();
        let cleanup = transfer.room_closed("disconnected".to_owned()).unwrap();
        assert_eq!(cleanup.close_human, Some(true));
        assert_eq!(
            transfer.take_outcome(),
            Some(Err(TransferFailure::RoomClosed("disconnected".to_owned())))
        );
    }
}
