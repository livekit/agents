# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/workflows.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NotRequired, TypeAlias

from typing_extensions import TypedDict

from ..types import NOT_GIVEN

if TYPE_CHECKING:
    from ..types import NotGivenOr


class SipParticipantRequest(TypedDict):
    """`CreateSIPParticipantRequest`'s fields as the host fills the protobuf. `sip_number` is `None`
    when empty and `dtmf` `""` when absent, as in the reference.
    """

    sip_trunk_id: str | None
    sip_call_to: str
    room_name: str
    participant_identity: str
    wait_until_answered: bool
    sip_number: str | None
    headers: dict[str, str]
    dtmf: str
    ringing_timeout: float | None


class IoEnableState(TypedDict, total=False):
    """The caller session's io legs the warm transfer turns off while the caller is on hold, read
    and restored. Absent or `None` reads as `true`.
    """

    audio_input: bool
    audio_output: bool
    transcription_output: bool


class TwilioDial(TypedDict):
    """How `TwilioConnectorWarmTransfer` reaches the human agent: Twilio's Calls API, streaming into
    the LiveKit Twilio connector. `call_token` (with `original_caller_number`) places the call from
    the caller's own number.
    """

    phone_number: str
    from_number: str
    account_sid: str
    auth_token: str
    call_token: NotRequired[str | None]
    original_caller_number: NotRequired[str | None]


@dataclass(frozen=True, kw_only=True, slots=True)
class WorkflowInstructions:
    """`WorkflowInstructions` as the host passes it: the two fields a workflow's template reads."""

    persona: NotGivenOr[str] = NOT_GIVEN
    """`NOT_GIVEN` keeps the workflow's built-in persona; `""` removes the section."""
    extra: str = ""


InstructionsArg: TypeAlias = WorkflowInstructions | str
"""A workflow's `instructions: WorkflowInstructions | str`: a
`WorkflowInstructions` fills the built-in template, anything else is fixed instructions.
"""
