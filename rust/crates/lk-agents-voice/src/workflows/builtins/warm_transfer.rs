use crate::workflows::builtins::instructions::{
    InstructionsArg, TemplateValue, WorkflowInstructions, render,
};
use crate::workflows::builtins::prompts::warm_transfer as p;
use crate::workflows::builtins::schema::{Param, nullary, tool};
use crate::workflows::{
    Task, TaskCtx, TaskOptions, ToolError, ToolReply, Warning, args, update_tool_flags,
};
use async_trait::async_trait;
use lk_agents_abi::voice::workflows::SipParticipantRequest;
use lk_agents_abi::{ChatContext, ToolSpec};
use lk_agents_rtc::{ParticipantIdentity, SipParticipantReq};
use lk_agents_voice::generation::Instructions;
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;

/// `self._human_agent_identity`: the SIP participant's identity, the second session's
/// `participant_identity`, and what `move_participant` names.
pub const HUMAN_AGENT_IDENTITY: &str = "human-agent-sip";

pub use lk_agents_abi::voice::workflows::WarmTransferResult;

/// `room.name + "-human-agent"` (`warm_transfer.WarmTransferTask._dial_human_agent`).
pub fn human_agent_room_name(caller_room: &str) -> String {
    format!("{caller_room}-human-agent")
}

/// `_format_conversation_history`: skips non-user/assistant roles and messages with no text. Uses
/// `raw_text_content`; callers with expressive TTS should strip markup first.
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

// --------------------------------------------------------------------------------- the io gate

pub use lk_agents_abi::voice::workflows::IoEnableState;

/// `self._original_io_state`. The originals are saved once (a second disable would capture the
/// muted values), and the restore is `enabled and original[..]`, so a channel off before the
/// transfer stays off.
#[derive(Debug, Default)]
pub struct IoGate {
    original: Option<IoEnableState>,
}

impl IoGate {
    /// What the five setters should be given. `current` is read only the first time.
    pub fn desired(&mut self, enabled: bool, current: IoEnableState) -> IoEnableState {
        let original = *self.original.get_or_insert(current);
        IoEnableState {
            audio_input: enabled && original.audio_input,
            video_input: enabled && original.video_input,
            audio_output: enabled && original.audio_output,
            transcription_output: enabled && original.transcription_output,
            video_output: enabled && original.video_output,
        }
    }
}

// ------------------------------------------------------------------------------------ the SIP

/// Filled into `CreateSIPParticipantRequest` (`WarmTransferTask._dial_human_agent`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SipDialConfig {
    /// `sip_call_to`. Required -- an empty one is [`WarmTransferError::NoSipCallTo`].
    pub sip_call_to: String,
    /// `sip_trunk_id`, after the env fallback.
    pub sip_trunk_id: Option<String>,
    /// `sip_connection is not None`; only presence is modelled (it decides the trunk fallback).
    pub has_sip_connection: bool,
    pub sip_number: String,
    pub sip_headers: HashMap<String, String>,
    pub dtmf: Option<String>,
    pub ringing_timeout: Option<Duration>,
}

/// Config the reference rejects in `__init__` with a `ValueError`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WarmTransferError {
    #[error("`sip_call_to` must be set")]
    NoSipCallTo,
    #[error(
        "`LIVEKIT_SIP_OUTBOUND_TRUNK` environment variable, `sip_trunk_id`, or `sip_connection` \
         must be set"
    )]
    NoTrunk,
}

/// `WarmTransferTask.__init__`'s parameters.
#[derive(Default)]
pub struct WarmTransferConfig {
    pub sip_call_to: Option<String>,
    /// `NOT_GIVEN` is the outer `None`; an explicit `None` is `Some(None)`. See [`resolve_trunk`].
    pub sip_trunk_id: Option<Option<String>>,
    pub has_sip_connection: bool,
    pub sip_number: Option<String>,
    pub sip_headers: HashMap<String, String>,
    pub dtmf: Option<String>,
    pub ringing_timeout: Option<Duration>,
    pub instructions: Option<InstructionsArg>,
    /// `extra_instructions`, deprecated in favour of `instructions=`.
    pub extra_instructions: String,
    /// The caller conversation, summarised into the instructions; not the task's own context.
    pub chat_ctx: Option<ChatContext>,
    /// `target_phone_number`, deprecated in favour of `sip_call_to`.
    pub target_phone_number: Option<String>,
    /// `os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK", None)`, read by the driver.
    pub env_sip_trunk: Option<String>,
    /// `os.getenv("LIVEKIT_SIP_NUMBER", "")`, read by the driver.
    pub env_sip_number: String,
    pub options: TaskOptions,
}

/// The trunk fallback, with three arms: "explicit sip_connection: don't override with the env var
/// trunk".
pub fn resolve_trunk(
    sip_trunk_id: &Option<Option<String>>,
    has_sip_connection: bool,
    env_sip_trunk: &Option<String>,
) -> Option<String> {
    match sip_trunk_id {
        // `is_given(sip_trunk_id)`: an explicit `None` disables the trunk and is not overridden.
        Some(explicit) => explicit.clone(),
        None if has_sip_connection => None,
        None => env_sip_trunk.clone(),
    }
}

/// The transfer's plan: its prompt and tools, the SIP request, and the io gate. The host dials,
/// moves the participant and settles.
pub struct WarmTransferTask {
    instructions: Instructions,
    user_tools: Vec<ToolSpec>,
    sip: SipDialConfig,
    io: IoGate,
}

impl WarmTransferTask {
    /// `__init__`. Returns the warnings the reference logs: the core has no logger.
    pub fn new(cfg: WarmTransferConfig) -> Result<(Self, Vec<Warning>), WarmTransferError> {
        let mut warnings = Vec::new();

        // Resolved before the deprecation handling, as in the reference.
        let history = cfg.chat_ctx.as_ref().map(format_conversation_history).unwrap_or_default();
        let instructions = match &cfg.instructions {
            None => {
                WorkflowInstructions {
                    persona: None,
                    extra: Some(TemplateValue::plain(cfg.extra_instructions.clone())),
                }
                .resolve(
                    p::INSTRUCTIONS_TEMPLATE,
                    p::PERSONA,
                    &[("_conversation_history", TemplateValue::plain(history))],
                )
                .instructions
            }
            Some(InstructionsArg::Workflow(w)) => {
                if !cfg.extra_instructions.is_empty() {
                    warnings.push(Warning::ExtraInstructionsIgnored);
                }
                w.resolve(
                    p::INSTRUCTIONS_TEMPLATE,
                    p::PERSONA,
                    &[("_conversation_history", TemplateValue::plain(history))],
                )
                .instructions
            }
            Some(InstructionsArg::Fixed(i)) => {
                if !cfg.extra_instructions.is_empty() {
                    warnings.push(Warning::ExtraInstructionsIgnored);
                }
                i.clone()
            }
        };

        // `if target_phone_number:` fills only a falsy `sip_call_to`.
        let mut sip_call_to = cfg.sip_call_to.clone().unwrap_or_default();
        if let Some(t) = cfg.target_phone_number.as_deref().filter(|t| !t.is_empty()) {
            warnings.push(Warning::TargetPhoneNumberDeprecated);
            if sip_call_to.is_empty() {
                sip_call_to = t.to_owned();
            }
        }
        if sip_call_to.is_empty() {
            return Err(WarmTransferError::NoSipCallTo);
        }

        let sip_trunk_id =
            resolve_trunk(&cfg.sip_trunk_id, cfg.has_sip_connection, &cfg.env_sip_trunk);
        if sip_trunk_id.is_none() && !cfg.has_sip_connection {
            return Err(WarmTransferError::NoTrunk);
        }

        Ok((
            WarmTransferTask {
                instructions,
                user_tools: cfg.options.tools,
                sip: SipDialConfig {
                    sip_call_to,
                    sip_trunk_id,
                    has_sip_connection: cfg.has_sip_connection,
                    sip_number: cfg.sip_number.unwrap_or(cfg.env_sip_number),
                    sip_headers: cfg.sip_headers,
                    dtmf: cfg.dtmf,
                    ringing_timeout: cfg.ringing_timeout,
                },
                io: IoGate::default(),
            },
            warnings,
        ))
    }

    pub fn io(&mut self) -> &mut IoGate {
        &mut self.io
    }

    /// The `CreateSIPParticipantRequest`: `sip_number or None` and `dtmf or ""`, as the reference.
    pub fn sip_request(&self, human_agent_room: &str) -> SipParticipantReq {
        SipParticipantReq {
            sip_trunk_id: self.sip.sip_trunk_id.clone(),
            sip_call_to: self.sip.sip_call_to.clone(),
            room_name: human_agent_room.to_owned(),
            participant_identity: ParticipantIdentity::from(HUMAN_AGENT_IDENTITY),
            wait_until_answered: true,
            sip_number: Some(self.sip.sip_number.clone()).filter(|s| !s.is_empty()),
            headers: self.sip.sip_headers.clone(),
            dtmf: self.sip.dtmf.clone().unwrap_or_default(),
            ringing_timeout: self.sip.ringing_timeout,
        }
    }
}

// ------------------------------------------------------------------------------------ the tools

/// All three tools are `IGNORE_ON_ENTER`: the entering turn summarises the call to the human, and
/// connecting then would cut the summary off.
fn tools() -> Vec<ToolSpec> {
    vec![
        {
            let mut t = nullary(p::CONNECT_TOOL, p::CONNECT_DESC);
            t.flags = update_tool_flags(true);
            t
        },
        tool(
            p::DECLINE_TOOL,
            p::DECLINE_DESC,
            &[Param::string("reason", p::DECLINE_ARG_REASON)],
            update_tool_flags(true),
        ),
        {
            let mut t = nullary(p::VOICEMAIL_TOOL, p::VOICEMAIL_DESC);
            t.flags = update_tool_flags(true);
            t
        },
    ]
}

#[async_trait]
impl Task for WarmTransferTask {
    type Output = WarmTransferResult;

    fn instructions(&self) -> Instructions {
        self.instructions.clone()
    }

    fn tools(&self) -> Vec<ToolSpec> {
        let mut v = self.user_tools.clone();
        v.extend(tools());
        v
    }

    /// `connect_to_caller` is not dispatched here: the host performs the `move_participant`.
    async fn call(
        &mut self,
        name: &str,
        args_json: Value,
        ctx: &mut TaskCtx<WarmTransferResult>,
    ) -> ToolReply {
        match name {
            p::DECLINE_TOOL => match args::string(&args_json, name, "reason") {
                Ok(reason) => {
                    ctx.fail(ToolError::plain(render(p::DECLINED, &[("reason", &reason)])));
                    ToolReply::Silent
                }
                Err(e) => ToolReply::Error(e),
            },
            p::VOICEMAIL_TOOL => {
                ctx.fail(ToolError::plain(p::VOICEMAIL));
                ToolReply::Silent
            }
            p::CONNECT_TOOL => ToolReply::error("connect_to_caller is performed by the host"),
            other => ToolReply::unknown_tool(other),
        }
    }
}

/// A plan construction refusal (`ValueError`): both instruction forms, or no number or trunk.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WarmTransferRefused {
    #[error(transparent)]
    Instructions(#[from] crate::workflows::builtins::instructions::BothInstructionForms),
    #[error(transparent)]
    Config(#[from] WarmTransferError),
}

/// `beta/workflows/warm_transfer.py`'s `__init__`, plus the two strings the transfer is built on.
///
/// The tool bodies stay Python (they drive host rooms, APIs and sessions); the decisions are here:
/// the three-arm trunk fallback, `target_phone_number` filling in only for a falsy `sip_call_to`,
/// and `sip_number or None` / `dtmf or ""`.
#[lk_agents_abi::class(name = "WarmTransferPlan", mutable)]
pub struct HostWarmTransfer {
    task: WarmTransferTask,
    warnings: Vec<Warning>,
}

/// `[(role, text)]` as a context with just what `_format_conversation_history` reads. Anything that
/// is not a `ChatRole` literal reads as `developer`, which the formatter skips like `system`.
fn conversation_of(items: Vec<(String, String)>) -> ChatContext {
    use lk_agents_abi::{ChatContent, ChatItem, ChatMessage, Role};
    ChatContext {
        items: items
            .into_iter()
            .enumerate()
            .map(|(i, (role, text))| {
                let role = Role::from_literal(&role).unwrap_or(Role::Developer);
                ChatItem::Message(ChatMessage::new(format!("history-{i}"), role, vec![ChatContent::Text(text)], std::time::UNIX_EPOCH))
            })
            .collect(),
    }
}

#[lk_agents_abi::host(object(name = "WarmTransferPlan", pid = false))]
impl HostWarmTransfer {
    /// `conversation` is `[(role, text)]` off the caller's `ChatContext`, unfiltered; the core
    /// keeps the `user`/`assistant` turns that have text. `sip_trunk_id_given` says whether an
    /// explicit `sip_trunk_id=None` was passed: it disables the trunk and beats the env var.
    #[host(new(
        kw_only,
        defaults(
            sip_trunk_id_given = false,
            has_sip_connection = false,
            extra_instructions = String::new(),
            env_sip_number = String::new()
        )
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sip_call_to: Option<String>,
        sip_trunk_id: Option<String>,
        sip_trunk_id_given: bool,
        has_sip_connection: bool,
        sip_number: Option<String>,
        sip_headers: Option<HashMap<String, String>>,
        dtmf: Option<String>,
        ringing_timeout: Option<f64>,
        instructions: Option<Instructions>,
        sections: Option<(Option<Instructions>, Option<Instructions>)>,
        extra_instructions: String,
        conversation: Option<Vec<(String, String)>>,
        target_phone_number: Option<String>,
        env_sip_trunk: Option<String>,
        env_sip_number: String,
        tools: Option<Vec<ToolSpec>>,
    ) -> Result<Self, WarmTransferRefused> {
        let (task, warnings) = WarmTransferTask::new(WarmTransferConfig {
            sip_call_to,
            sip_trunk_id: sip_trunk_id_given.then_some(sip_trunk_id),
            has_sip_connection,
            sip_number,
            sip_headers: sip_headers.unwrap_or_default(),
            dtmf,
            ringing_timeout: ringing_timeout.map(Duration::from_secs_f64),
            instructions: InstructionsArg::from_host(instructions, sections)?,
            extra_instructions,
            chat_ctx: conversation.map(conversation_of),
            target_phone_number,
            env_sip_trunk,
            env_sip_number,
            options: TaskOptions { tools: tools.unwrap_or_default(), ..TaskOptions::default() },
        })?;
        Ok(HostWarmTransfer { task, warnings })
    }

    #[host(getter)]
    pub fn instructions(&self) -> (String, Option<String>, Option<String>) {
        let i = Task::instructions(&self.task);
        (i.common, i.audio, i.text)
    }

    #[host(getter)]
    pub fn warnings(&self) -> Vec<String> {
        self.warnings.iter().map(|w| w.as_str().to_owned()).collect()
    }

    /// `connect_to_caller`, `decline_transfer` and `voicemail_detected`, all three
    /// `IGNORE_ON_ENTER`.
    #[host(method)]
    pub fn tools(&self) -> Vec<lk_agents_abi::voice::workflows::RawFunctionDescription> {
        Task::tools(&self.task).iter().map(Into::into).collect()
    }

    /// `f"{caller_room}-human-agent"`. One place, so the dialer and the mover cannot disagree
    /// about which room the human is in.
    #[host(method)]
    pub fn human_agent_room(caller_room: &str) -> String {
        human_agent_room_name(caller_room)
    }

    /// `"human-agent-sip"` -- the identity the SIP participant joins under and the one
    /// `move_participant` looks for.
    #[host(method)]
    pub fn human_agent_identity() -> String {
        HUMAN_AGENT_IDENTITY.to_owned()
    }

    /// The `CreateSIPParticipantRequest`'s fields, after the reference's two normalisations.
    #[host(method)]
    pub fn sip_request(&self, human_agent_room: &str) -> SipParticipantRequest {
        sip_request_record(&self.task.sip_request(human_agent_room))
    }

    /// `_set_io_enabled`'s five flags. The first `desired(False, ..)` remembers the session's io,
    /// and `desired(True, ..)` restores that rather than enabling every leg.
    #[host(method)]
    pub fn io_enabled(&mut self, enabled: bool, current: lk_agents_abi::voice::workflows::IoEnableState) -> lk_agents_abi::voice::workflows::IoEnableState {
        self.task.io().desired(enabled, current)
    }
}

/// A [`SipParticipantReq`] as the host fills the protobuf from it.
pub fn sip_request_record(req: &SipParticipantReq) -> SipParticipantRequest {
    SipParticipantRequest {
        sip_trunk_id: req.sip_trunk_id.clone(),
        sip_call_to: req.sip_call_to.clone(),
        room_name: req.room_name.clone(),
        participant_identity: req.participant_identity.to_string(),
        wait_until_answered: req.wait_until_answered,
        sip_number: req.sip_number.clone(),
        headers: req.headers.clone(),
        dtmf: req.dtmf.clone(),
        ringing_timeout: req.ringing_timeout.map(|d| d.as_secs_f64()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(sip_call_to: &str, has_sip_connection: bool) -> WarmTransferConfig {
        WarmTransferConfig {
            sip_call_to: Some(sip_call_to.to_owned()),
            has_sip_connection,
            env_sip_trunk: Some("env-trunk".to_owned()),
            ..Default::default()
        }
    }

    // Catches: the trunk fallback losing its middle arm (an env trunk overriding an explicit
    // `sip_connection`).
    #[test]
    fn an_explicit_sip_connection_beats_the_env_trunk() {
        let (task, _) = WarmTransferTask::new(config("+15550100", true)).unwrap();
        let req = task.sip_request("room-human-agent");
        assert!(
            req.sip_trunk_id.is_none(),
            "the env trunk must not override an explicit sip_connection"
        );
        // `sip_number=... or None`.
        assert!(req.sip_number.is_none());
        // `dtmf=... or ""`: absent becomes empty, because that field is not optional.
        assert_eq!(req.dtmf, "");
    }

    // Catches: `target_phone_number` winning over an explicit `sip_call_to`, or its deprecation
    // warning not reaching the host.
    #[test]
    fn the_deprecated_number_only_fills_a_hole_and_still_warns() {
        let (task, warnings) = WarmTransferTask::new(WarmTransferConfig {
            target_phone_number: Some("+15550999".to_owned()),
            ..config("+15550100", false)
        })
        .unwrap();
        assert!(warnings.contains(&Warning::TargetPhoneNumberDeprecated));
        assert_eq!(task.sip_request("r").sip_call_to, "+15550100", "the explicit number wins");
    }
}
