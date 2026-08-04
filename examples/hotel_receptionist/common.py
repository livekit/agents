from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from zoneinfo import ZoneInfo

from livekit.agents import llm

from .hotel import RoomBooking
from .hotel_db import HotelDB

# scenarios.yaml pins every date literal to SIM_TODAY, so simulation runs must
# not use the real clock. HOTEL_TODAY=YYYY-MM-DD overrides in any mode.
SIM_TODAY = date(2026, 6, 8)


def resolve_today() -> date:
    """The hotel's local date (it's in San Francisco, the server may not be).
    Set HOTEL_TODAY=YYYY-MM-DD to pin it for deterministic sim runs."""
    if pinned := os.environ.get("HOTEL_TODAY"):
        return date.fromisoformat(pinned)
    # `lk agent simulate` always passes --simulation; job subprocesses inherit argv
    if "--simulation" in sys.argv:
        return SIM_TODAY
    return datetime.now(ZoneInfo("America/Los_Angeles")).date()


@dataclass
class Userdata:
    db: HotelDB
    today: date
    # Departments already transferred to this call - guards against a duplicate transfer
    # row when the agent re-calls transfer_call after the caller's reaction.
    transferred_to: set[str] = field(default_factory=set)
    # The refund outcome from the last room cancellation, and the caller-turn count when it
    # happened - so a re-invoked cancel (no caller input since) re-surfaces that answer
    # instead of re-verifying into a confusing "already cancelled" dead end.
    last_cancel_message: str = ""
    caller_turns_at_last_cancel: int = -1
    # The last document re-send (what was sent + the caller-turn count when it went out) -
    # so a re-invoked resend with no caller input since relays "it's sent" instead of
    # emailing the same document twice.
    last_resend_message: str = ""
    caller_turns_at_last_resend: int = -1
    # The last completed booking modification's outcome, and the caller-turn count when
    # it finished - so a re-invoked modification (no caller input since) relays that
    # outcome instead of re-opening the flow on a booking that was just updated.
    last_modification_message: str = ""
    caller_turns_at_last_modification: int = -1
    verified_booking: RoomBooking | None = None
    # The most recent completed room booking, and the caller-turn count at the moment
    # it completed - together they catch a model that re-runs the booking flow with no
    # caller input since, which would silently double-book the guest.
    last_room_booking: RoomBooking | None = None
    caller_turns_at_last_booking: int = 0
    # Whether a room-booking flow is currently awaiting its inline BookRoomTask.
    # Parallel inline AgentTasks are unsupported (the second activation orphans the
    # first call's future and wedges the session), so a second start_room_booking
    # while one is in flight must bounce with a ToolError instead of running.
    room_booking_in_flight: bool = False
    # Whether the pre-hangup policy audit already handed the agent its one nudge -
    # after that, say_goodbye_and_close_call always closes, so the agent can never
    # get stuck unable to hang up.
    end_call_nudged: bool = False
    # Whether the farewell was already delivered on this call - repeat close attempts
    # (the caller answered the goodbye, the model called the tool again) must not
    # produce a "Goodbye!" / "Take care!" / "Goodbye!" loop.
    goodbye_said: bool = False


def _speak_code(code: str) -> str:
    # Hand the raw code to the TTS - its own parser reads alphanumeric codes
    # correctly; we don't pre-spell it character by character.
    return code.upper()


def _count_caller_turns(chat_ctx: llm.ChatContext) -> int:
    """How many times the caller has spoken so far - the signal for whether a
    booking flow was actually driven by the caller or silently re-run by the model."""
    return sum(1 for it in chat_ctx.items if it.type == "message" and it.role == "user")
