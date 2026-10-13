//! What the prebuilt workflows hand the host and read back, and the prompt constants the
//! reference's `beta/workflows/` modules publish. The workflows are `lk-agents-voice`'s.

use crate::ToolSpec;
use crate::utils::dates::{Date, Time};
use lk_agents_host_derive::Host;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

// Each task's `complete(..)` value (a frozen dataclass in the reference). `GetDtmfResult` and
// `GetDOBResult` carry methods, so each is a `#[host(object)]` instead.

/// GetAddressResult(address: 'str')
#[crate::class(eq, from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(dataclass, module = "workflows", path = "livekit.agents.beta.workflows.address", name = "GetAddressResult")]
#[node(interface)]
#[binding(class, new)]
pub struct GetAddressResult {
    pub address: String,
}

/// GetEmailResult(email_address: 'str')
#[crate::class(eq, from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(dataclass, module = "workflows", path = "livekit.agents.beta.workflows.email_address", name = "GetEmailResult")]
#[node(interface)]
#[binding(class, new)]
pub struct GetEmailResult {
    pub email_address: String,
}

/// GetPhoneNumberResult(phone_number: 'str')
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(dataclass, module = "workflows", path = "livekit.agents.beta.workflows.phone_number", name = "GetPhoneNumberResult")]
#[node(interface)]
#[binding(class, new)]
#[crate::class(eq, from_py_object)]
pub struct GetPhoneNumberResult {
    pub phone_number: String,
}

/// `GetDtmfResult` (`beta/workflows/dtmf_inputs.py`): the digits, space-separated (`format_dtmf`).
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(fields_only)]
pub struct GetDtmfResult {
    pub user_input: String,
}

/// What a driver-run DTMF collection came to: `complete`, `incomplete`, or `stopped`.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit)]
pub enum DtmfOutcomeWire {
    Complete(DtmfComplete),
    Incomplete(DtmfIncomplete),
    Stopped(DtmfStopped),
}

/// `self.complete(GetDtmfResult(user_input))`.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit, tag = "kind", tag_value = "complete")]
pub struct DtmfComplete {
    pub user_input: String,
}

/// `self.complete(ToolError(message))`: the debounce fired before `num_digits` arrived.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit, tag = "kind", tag_value = "incomplete")]
pub struct DtmfIncomplete {
    pub message: String,
}

/// The collection was stopped, replaced or closed first.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Host)]
#[binding(emit, tag = "kind", tag_value = "stopped")]
pub struct DtmfStopped {}

// A field is `None` unless the task was configured to collect that part, whatever the model
// volunteered.
/// GetNameResult(first_name: 'str | None' = None, middle_name: 'str | None' = None, last_name: 'str | None' = None)
#[crate::class(eq, from_py_object)]
#[derive(Debug, Clone, Default, PartialEq, Eq, Host)]
#[python(dataclass, zero_defaults, module = "workflows", path = "livekit.agents.beta.workflows.name", name = "GetNameResult")]
#[node(interface)]
#[binding(class, new)]
pub struct GetNameResult {
    #[binding(default)]
    pub first_name: Option<String>,
    #[binding(default)]
    pub middle_name: Option<String>,
    #[binding(default)]
    pub last_name: Option<String>,
}

// Through converters (Node reads ISO strings).
/// GetDOBResult(date_of_birth: 'date', time_of_birth: 'time | None' = None)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Host)]
#[binding(fields_only)]
#[python(dataclass, module = "workflows", path = "livekit.agents.beta.workflows.dob", name = "GetDOBResult")]
#[node(interface)]
pub struct GetDOBResult {
    #[python(ty = "date", import = "datetime:date")]
    #[node(ty = "string")]
    pub date_of_birth: Date,
    #[python(ty = "time | None", import = "datetime:time", default = "None")]
    #[node(ty = "string | null")]
    pub time_of_birth: Option<Time>,
}

/// GetDOBResult(date_of_birth: 'date', time_of_birth: 'time | None' = None)
///
/// `GetDOBResult` as a `_core` value: `date_of_birth` a `datetime.date`, `time_of_birth` a
/// `datetime.time | None`, and the two strings the prompt interpolates. Frozen data: no fork guard.
#[crate::host(object(cell = "value", pid = false, pyclass(from_py_object), derive(Clone)))]
impl GetDOBResult {
    /// Out-of-range components raise `ValueError`. `Date::new` validates rather than saturating,
    /// and saturating a date of birth is how a typo becomes a plausible-looking record.
    #[host(new(kw_only))]
    pub fn for_host(date_of_birth: Date, time_of_birth: Option<Time>) -> Self {
        GetDOBResult { date_of_birth, time_of_birth }
    }

    #[host(getter(name = "date_of_birth"))]
    pub fn host_date_of_birth(&self) -> Date {
        self.date_of_birth
    }

    #[host(getter(name = "time_of_birth"))]
    pub fn host_time_of_birth(&self) -> Option<Time> {
        self.time_of_birth
    }

    /// `strftime("%B %d, %Y")` -- `"January 15, 1990"`. The prompt interpolates this exact form.
    #[host(method)]
    pub fn format_long(&self) -> String {
        self.date_of_birth.format_long()
    }

    /// `strftime("%I:%M %p")` -- `"03:05 PM"`. `None` when no time was collected.
    #[host(method)]
    pub fn format_time(&self) -> Option<String> {
        self.time_of_birth.map(|t| t.format_12h())
    }

    #[host(method(name = "__eq__"))]
    pub fn host_eq(&self, other: GetDOBResult) -> bool {
        *self == other
    }

    #[host(method(name = "__repr__"))]
    pub fn host_repr(&self) -> String {
        format!("GetDOBResult({self:?})")
    }
}

/// GetCardNumberResult(issuer: 'str', card_number: 'str')
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(dataclass, module = "workflows", path = "livekit.agents.beta.workflows.credit_card", name = "GetCardNumberResult")]
#[node(interface)]
#[binding(class, new)]
#[crate::class(eq, from_py_object)]
pub struct GetCardNumberResult {
    pub issuer: String,
    pub card_number: String,
}

/// GetSecurityCodeResult(security_code: 'str')
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(dataclass, module = "workflows", path = "livekit.agents.beta.workflows.credit_card", name = "GetSecurityCodeResult")]
#[node(interface)]
#[binding(class, new)]
#[crate::class(eq, from_py_object)]
pub struct GetSecurityCodeResult {
    pub security_code: String,
}

// The field is `date`, not `expiration_date`: `assemble_credit_card_result` maps it.
/// GetExpirationDateResult(date: 'str')
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(dataclass, module = "workflows", path = "livekit.agents.beta.workflows.credit_card", name = "GetExpirationDateResult")]
#[node(interface)]
#[binding(class, new)]
#[crate::class(eq, from_py_object)]
pub struct GetExpirationDateResult {
    pub date: String,
}

/// GetCreditCardResult(cardholder_name: 'str', issuer: 'str', card_number: 'str', security_code: 'str', expiration_date: 'str')
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(dataclass, module = "workflows", path = "livekit.agents.beta.workflows.credit_card", name = "GetCreditCardResult")]
#[node(interface)]
#[binding(class, new)]
#[crate::class(eq, from_py_object)]
pub struct GetCreditCardResult {
    pub cardholder_name: String,
    pub issuer: String,
    pub card_number: String,
    pub security_code: String,
    pub expiration_date: String,
}

// Python-only: the host builds the dataclass itself.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(
    dataclass,
    module = "workflows",
    path = "livekit.agents.beta.workflows.warm_transfer",
    name = "WarmTransferResult"
)]
#[node(interface)]
pub struct WarmTransferResult {
    pub human_agent_identity: String,
}

// Keyed exactly as published (hence capitals); the strings live in `workflows::builtins::prompts`.

/// The five a capture module publishes, in the reference's own order: `address`.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit)]
pub struct CapturePrompts {
    #[host(rename = "PERSONA")]
    pub persona: String,
    #[host(rename = "AUDIO_SPECIFIC")]
    pub audio_specific: String,
    #[host(rename = "TEXT_SPECIFIC")]
    pub text_specific: String,
    #[host(rename = "CONFIRMATION_INSTRUCTION")]
    pub confirmation_instruction: String,
    #[host(rename = "INSTRUCTIONS_TEMPLATE")]
    pub instructions_template: String,
}

/// `email_address`: the capture five plus `EMAIL_REGEX`, the reference's validator string verbatim.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit)]
pub struct EmailPrompts {
    #[host(rename = "PERSONA")]
    pub persona: String,
    #[host(rename = "AUDIO_SPECIFIC")]
    pub audio_specific: String,
    #[host(rename = "TEXT_SPECIFIC")]
    pub text_specific: String,
    #[host(rename = "CONFIRMATION_INSTRUCTION")]
    pub confirmation_instruction: String,
    #[host(rename = "INSTRUCTIONS_TEMPLATE")]
    pub instructions_template: String,
    #[host(rename = "EMAIL_REGEX")]
    pub email_regex: String,
}

/// `phone_number`: only `PHONE_REGEX` is public there.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit)]
pub struct PhonePrompts {
    #[host(rename = "PHONE_REGEX")]
    pub phone_regex: String,
}

/// `warm_transfer`: the persona and the template alone.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit)]
pub struct TransferPrompts {
    #[host(rename = "PERSONA")]
    pub persona: String,
    #[host(rename = "INSTRUCTIONS_TEMPLATE")]
    pub instructions_template: String,
}

/// `credit_card`: `CARD_ISSUERS_LOOKUP`, first digit -> issuer name.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit)]
pub struct CardPrompts {
    #[host(rename = "CARD_ISSUERS_LOOKUP")]
    pub card_issuers_lookup: BTreeMap<String, String>,
}

/// `_core.workflow_prompts()`: the five modules' constants, keyed by the reference's module name.
/// Exactly the names the reference publishes, and no others.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit)]
pub struct WorkflowPrompts {
    pub email_address: EmailPrompts,
    pub address: CapturePrompts,
    pub phone_number: PhonePrompts,
    pub warm_transfer: TransferPrompts,
    pub credit_card: CardPrompts,
}

/// A [`ToolSpec`] in the shape `@function_tool(raw_schema=..)` takes: a `RawFunctionDescription`.
// The schema is keyed `parameters` here and `json_schema` inbound; the wrong key gives the model a
// tool with no parameters.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(emit)]
pub struct RawFunctionDescription {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub ignore_on_enter: bool,
}

impl From<&ToolSpec> for RawFunctionDescription {
    fn from(spec: &ToolSpec) -> Self {
        RawFunctionDescription {
            name: spec.name.clone(),
            description: spec.description.clone(),
            parameters: spec.json_schema.clone(),
            ignore_on_enter: spec.flags.ignore_on_enter,
        }
    }
}

/// `TaskGroupStep::Run`: call the factory for `task_id`, append `out_of_scope` to its tools when
/// there is something to regress to, then await it.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(emit, tag = "kind", tag_value = "run")]
pub struct TaskGroupRun {
    pub task_id: String,
    pub out_of_scope: Option<RawFunctionDescription>,
}

/// `TaskGroupStep::Summarize`: the stack drained and `summarize_chat_ctx` is set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Host)]
#[binding(emit, tag = "kind", tag_value = "summarize")]
pub struct TaskGroupSummarize {}

/// `TaskGroupStep::Complete`: `self.complete(TaskGroupResult(task_results=...))`.
/// `recorded` is `(task_id, "value" | "error")` in order; an `"error"` result is an exception.
#[derive(Debug, Clone, Default, PartialEq, Eq, Host)]
#[binding(emit, tag = "kind", tag_value = "complete")]
pub struct TaskGroupComplete {
    pub recorded: Vec<(String, String)>,
}

/// `TaskGroupStep::Fail`: `self.complete(e)`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Host)]
#[binding(emit, tag = "kind", tag_value = "fail")]
pub struct TaskGroupFail {
    pub error: String,
}

/// A task group's next step, as one of the four records above (tagged `kind`).
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(emit)]
pub enum TaskGroupStepWire {
    Run(TaskGroupRun),
    Summarize(TaskGroupSummarize),
    Complete(TaskGroupComplete),
    Fail(TaskGroupFail),
}

/// The five `copy()` flags `TaskGroup`'s summarisation passes: all `False`, so the summary sees the
/// tool calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Host)]
#[binding(emit)]
pub struct TaskGroupSummaryFlags {
    pub exclude_instructions: bool,
    pub exclude_handoff: bool,
    pub exclude_config_update: bool,
    pub exclude_empty_message: bool,
    pub exclude_function_call: bool,
}

/// `CreateSIPParticipantRequest`'s fields as the host fills the protobuf. `sip_number` is `None`
/// when empty and `dtmf` `""` when absent, as in the reference.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(emit)]
pub struct SipParticipantRequest {
    pub sip_trunk_id: Option<String>,
    pub sip_call_to: String,
    pub room_name: String,
    pub participant_identity: String,
    pub wait_until_answered: bool,
    pub sip_number: Option<String>,
    pub headers: HashMap<String, String>,
    pub dtmf: String,
    pub ringing_timeout: Option<f64>,
}

/// The five enable flags `WarmTransferTask._set_io_enabled` toggles, read and restored. Absent or
/// `None` reads as `true`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Host)]
#[binding(extract, emit, from_items)]
pub struct IoEnableState {
    #[binding(default = true)]
    pub audio_input: bool,
    #[binding(default = true)]
    pub audio_output: bool,
    #[binding(default = true)]
    pub transcription_output: bool,
    #[binding(default = true)]
    pub video_input: bool,
    #[binding(default = true)]
    pub video_output: bool,
}

impl Default for IoEnableState {
    fn default() -> Self {
        IoEnableState {
            audio_input: true,
            audio_output: true,
            transcription_output: true,
            video_input: true,
            video_output: true,
        }
    }
}

// A workflow's collected value reaches the host (`CoreObject`) as its class, over a clone.
#[cfg(feature = "python")]
const _: () = {
    use crate::python::{CoreObjectClass, core_object_to_host};
    inventory::submit! { CoreObjectClass { to_host: core_object_to_host::<GetEmailResult> } }
    inventory::submit! { CoreObjectClass { to_host: core_object_to_host::<GetAddressResult> } }
    inventory::submit! { CoreObjectClass { to_host: core_object_to_host::<GetPhoneNumberResult> } }
    inventory::submit! { CoreObjectClass { to_host: core_object_to_host::<GetNameResult> } }
    inventory::submit! { CoreObjectClass { to_host: core_object_to_host::<GetCardNumberResult> } }
    inventory::submit! { CoreObjectClass { to_host: core_object_to_host::<GetSecurityCodeResult> } }
    inventory::submit! { CoreObjectClass { to_host: core_object_to_host::<GetExpirationDateResult> } }
    inventory::submit! { CoreObjectClass { to_host: core_object_to_host::<GetDOBResult> } }
};
