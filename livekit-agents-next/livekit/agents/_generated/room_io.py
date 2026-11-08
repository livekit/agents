# GENERATED FILE -- DO NOT EDIT.
# From ABI declarations and authored plugin bases:
#     rust/crates/lk-agents-abi/src/voice/room_io.rs

from __future__ import annotations

from typing import Final

from livekit import rtc

DEFAULT_PARTICIPANT_KINDS: Final[list[rtc.ParticipantKind.ValueType]] = [
    rtc.ParticipantKind.PARTICIPANT_KIND_CONNECTOR,
    rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
    rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
]
"""The participant kinds a room links to when none are given: no agents or egress, so agents
never link to each other.
"""
