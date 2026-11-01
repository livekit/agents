//! What the prebuilt workflows hand the host and read back, and the prompt constants the
//! reference's `beta/workflows/` modules publish. The workflows are `lk-agents-voice`'s.

use crate::utils::dates::Date;
use std::collections::BTreeMap;

/// `CreateSIPParticipantRequest`'s fields as the host fills the protobuf. `sip_number` is `None`
/// when empty and `dtmf` `""` when absent, as in the reference.
#[crate::host(record(module = "workflows", typed_dict, emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct SipParticipantRequest {
    pub sip_trunk_id: Option<String>,
    pub sip_call_to: String,
    pub room_name: String,
    pub participant_identity: String,
    pub wait_until_answered: bool,
    pub sip_number: Option<String>,
    pub headers: BTreeMap<String, String>,
    pub dtmf: String,
    pub ringing_timeout: Option<f64>,
}

/// The caller session's io legs the warm transfer turns off while the caller is on hold, read
/// and restored. Absent or `None` reads as `true`.
#[crate::host(record(module = "workflows", typed_dict, extract, emit))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IoEnableState {
    #[host(default = true)]
    pub audio_input: bool,
    #[host(default = true)]
    pub audio_output: bool,
    #[host(default = true)]
    pub transcription_output: bool,
}

impl Default for IoEnableState {
    fn default() -> Self {
        IoEnableState { audio_input: true, audio_output: true, transcription_output: true }
    }
}

/// How `TwilioConnectorWarmTransfer` reaches the human agent: Twilio's Calls API, streaming into
/// the LiveKit Twilio connector. `call_token` (with `original_caller_number`) places the call from
/// the caller's own number.
#[crate::host(record(module = "workflows", typed_dict, extract))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TwilioDial {
    pub phone_number: String,
    pub from_number: String,
    pub account_sid: String,
    pub auth_token: String,
    #[host(default = None, annotation = "str | None")]
    pub call_token: Option<String>,
    #[host(default = None, annotation = "str | None")]
    pub original_caller_number: Option<String>,
}

/// `WorkflowInstructions` as the host passes it: the two fields a workflow's template reads.
#[derive(Clone, PartialEq, Eq)]
#[crate::host(record(module = "workflows", name = "WorkflowInstructions", extract))]
pub struct WorkflowInstructionsArg {
    /// `NOT_GIVEN` keeps the workflow's built-in persona; `""` removes the section.
    pub persona: crate::Maybe<String>,
    #[host(default = "")]
    pub extra: String,
}

/// A workflow's `instructions: WorkflowInstructions | str`: a
/// `WorkflowInstructions` fills the built-in template, anything else is fixed instructions.
#[derive(Clone, PartialEq, Eq)]
#[crate::host(record(module = "workflows", extract, transparent))]
pub enum InstructionsArg {
    Workflow(WorkflowInstructionsArg),
    Fixed(String),
}

pub mod dtmf;

pub mod prompts;

pub mod warm_transfer;

/// Why a date of birth is refused (`ValueError`).
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DateOfBirthRefused {
    #[error(transparent)]
    Invalid(#[from] crate::utils::dates::DateError),
    #[error("Date of birth cannot be in the future")]
    Future,
}

/// What a capture tool heard is not what it collects: the reference's `ToolError`, its message
/// the model's.
#[crate::host(error(kind = "livekit.agents.llm.tool_context.ToolError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct CaptureRefused(pub String);

/// A key that is no DTMF symbol (`ValueError`).
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown DTMF symbol {0:?}")]
pub struct UnknownDtmfSymbol(pub String);

/// The checks the capture tools run on what the model heard, and the DTMF symbols they read.
#[crate::host(functions(pid = false))]
pub trait WorkflowChecks {
    #[host(function)]
    fn dtmf_capture_plan(confirmation: bool) -> CapturePlan;
    #[host(function)]
    fn email_capture_plan() -> CapturePlan;
    #[host(function)]
    fn email_read_back(email_address: String) -> String;
    /// The Luhn checksum over an all-digit string.
    #[host(function)]
    fn validate_card_number(card_number: String) -> bool;
    /// Whether a two-digit expiry year and its month precede `today`'s month.
    #[host(function)]
    fn card_expired(month: i64, year: i64, today: Date) -> bool;
    /// A two-digit year is this century's when it is not past this year's two digits, else the
    /// last century's; an invalid or future date is refused.
    #[host(function)]
    fn normalize_date_of_birth(
        year: i64,
        month: i64,
        day: i64,
        today: Date,
    ) -> Result<Date, DateOfBirthRefused>;
    /// Stripped, and refused unless it reads `local@label.….tld`.
    #[host(function)]
    fn normalize_email_address(email: String) -> Result<String, CaptureRefused>;
    /// Whitespace, hyphens, parentheses and dots dropped, and refused unless it reads `+?[1-9]`
    /// followed by six to fourteen digits.
    #[host(function)]
    fn normalize_phone_number(phone_number: String) -> Result<String, CaptureRefused>;
    /// Digits are themselves, `*` is 10, `#` is 11 and `A`-`D` are 12-15.
    #[host(function)]
    fn dtmf_code(symbol: String) -> Result<u8, UnknownDtmfSymbol>;
    /// The space-joined digits that fill the DTMF read-back's `{digits}` hole.
    #[host(function)]
    fn format_dtmf(symbols: Vec<String>) -> Result<String, UnknownDtmfSymbol>;
}

/// Built-in capture policy; language hosts attach typed result constructors and callbacks.
#[crate::host(record(module = "workflows", emit))]
#[derive(Clone)]
pub struct CapturePlan {
    pub agent_name: String,
    pub instructions: String,
    pub tool_name: String,
    pub tool_description: String,
}

// Public data contracts. Python behavior remains in its adapters.

#[crate::host(record(module = "workflows", path = "livekit.agents.beta.workflows", dataclass))]
pub struct GetEmailResult {
    pub email_address: String,
}

#[crate::host(record(module = "workflows", path = "livekit.agents.beta.workflows", dataclass))]
pub struct GetDtmfResult {
    /// The pressed digits, space-separated (``format_dtmf``).
    pub user_input: String,
}

#[crate::host(record(module = "workflows", path = "livekit.agents.beta.workflows", dataclass))]
pub struct WarmTransferResult {
    pub human_agent_identity: String,
}
