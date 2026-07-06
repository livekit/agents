from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hotel_db import HotelDB, RoomBooking

from livekit.agents import llm


@dataclass
class Userdata:
    db: HotelDB
    # Departments already transferred to this call - guards against a duplicate transfer
    # row when the agent re-calls transfer_call after the caller's reaction.
    transferred_to: set[str] = field(default_factory=set)
    # The refund outcome from the last room cancellation, and the caller-turn count when it
    # happened - so a re-invoked cancel (no caller input since) re-surfaces that answer
    # instead of re-verifying into a confusing "already cancelled" dead end.
    last_cancel_message: str = ""
    caller_turns_at_last_cancel: int = -1
    verified_booking: RoomBooking | None = None
    # The most recent completed room booking, and the caller-turn count at the moment
    # it completed - together they catch a model that re-runs the booking flow with no
    # caller input since, which would silently double-book the guest.
    last_room_booking: RoomBooking | None = None
    caller_turns_at_last_booking: int = 0


def _speak_code(code: str) -> str:
    # Hand the raw code to the TTS - its own parser reads alphanumeric codes
    # correctly; we don't pre-spell it character by character.
    return code.upper()


def _count_caller_turns(chat_ctx: llm.ChatContext) -> int:
    """How many times the caller has spoken so far - the signal for whether a
    booking flow was actually driven by the caller or silently re-run by the model."""
    return sum(1 for it in chat_ctx.items if it.type == "message" and it.role == "user")
