//! What the prebuilt workflows hand the host and read back, and the prompt constants the
//! reference's `beta/workflows/` modules publish. The workflows are `lk-agents-voice`'s.

use crate::utils::dates::{Date, Time};
use lk_agents_host_derive::Host;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::collections::HashMap;

// Each task's `complete(..)` value (a frozen dataclass in the reference). `GetDtmfResult` and
// `GetDOBResult` carry methods, so each is a `#[host(object)]` instead.

/// GetAddressResult(address: 'str')
#[crate::class(eq, from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(
    dataclass,
    module = "workflows",
    path = "livekit.agents.beta.workflows",
    name = "GetAddressResult"
)]
#[binding(class, new)]
pub struct GetAddressResult {
    pub address: String,
}

/// GetEmailResult(email_address: 'str')
#[crate::class(eq, from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(
    dataclass,
    module = "workflows",
    path = "livekit.agents.beta.workflows",
    name = "GetEmailResult"
)]
#[binding(class, new)]
pub struct GetEmailResult {
    pub email_address: String,
}

/// GetPhoneNumberResult(phone_number: 'str')
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(
    dataclass,
    module = "workflows",
    path = "livekit.agents.beta.workflows",
    name = "GetPhoneNumberResult"
)]
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
#[python(
    dataclass,
    zero_defaults,
    module = "workflows",
    path = "livekit.agents.beta.workflows",
    name = "GetNameResult"
)]
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
#[python(
    dataclass,
    module = "workflows",
    path = "livekit.agents.beta.workflows",
    name = "GetDOBResult"
)]
pub struct GetDOBResult {
    #[python(ty = "date", import = "datetime:date")]
    pub date_of_birth: Date,
    #[python(ty = "time | None", import = "datetime:time", default = "None")]
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
#[python(
    dataclass,
    module = "workflows",
    path = "livekit.agents.beta.workflows",
    name = "GetCardNumberResult"
)]
#[binding(class, new)]
#[crate::class(eq, from_py_object)]
pub struct GetCardNumberResult {
    pub issuer: String,
    pub card_number: String,
}

/// GetSecurityCodeResult(security_code: 'str')
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(
    dataclass,
    module = "workflows",
    path = "livekit.agents.beta.workflows",
    name = "GetSecurityCodeResult"
)]
#[binding(class, new)]
#[crate::class(eq, from_py_object)]
pub struct GetSecurityCodeResult {
    pub security_code: String,
}

// The field is `date`, not `expiration_date`: `assemble_credit_card_result` maps it.
/// GetExpirationDateResult(date: 'str')
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(
    dataclass,
    module = "workflows",
    path = "livekit.agents.beta.workflows",
    name = "GetExpirationDateResult"
)]
#[binding(class, new)]
#[crate::class(eq, from_py_object)]
pub struct GetExpirationDateResult {
    pub date: String,
}

/// GetCreditCardResult(cardholder_name: 'str', issuer: 'str', card_number: 'str', security_code: 'str', expiration_date: 'str')
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(
    dataclass,
    module = "workflows",
    path = "livekit.agents.beta.workflows",
    name = "GetCreditCardResult"
)]
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
    path = "livekit.agents.beta.workflows",
    name = "WarmTransferResult"
)]
pub struct WarmTransferResult {
    pub human_agent_identity: String,
}

/// `CreateSIPParticipantRequest`'s fields as the host fills the protobuf. `sip_number` is `None`
/// when empty and `dtmf` `""` when absent, as in the reference.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(emit, from_items)]
pub struct SipParticipantRequest {
    pub sip_trunk_id: Option<String>,
    pub sip_call_to: String,
    pub room_name: String,
    pub participant_identity: String,
    pub wait_until_answered: bool,
    pub sip_number: Option<String>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    pub headers: HashMap<String, String>,
    pub dtmf: String,
    pub ringing_timeout: Option<f64>,
}

/// The caller session's io legs the warm transfer turns off while the caller is on hold, read
/// and restored. Absent or `None` reads as `true`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Host)]
#[binding(extract, emit, from_items)]
pub struct IoEnableState {
    #[binding(default = true)]
    pub audio_input: bool,
    #[binding(default = true)]
    pub audio_output: bool,
    #[binding(default = true)]
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
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(extract, from_items)]
pub struct TwilioDial {
    pub phone_number: String,
    pub from_number: String,
    pub account_sid: String,
    pub auth_token: String,
    #[binding(default)]
    pub call_token: Option<String>,
    #[binding(default)]
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
};

pub mod dtmf;

pub mod warm_transfer;
