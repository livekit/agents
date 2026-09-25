from __future__ import annotations

from dataclasses import dataclass, field

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
    # Spell character by character, with "-" spoken as the single word "dash" -
    # NOT spelled D, A, S, H (that reads as four more code characters).
    return ", ".join("dash" if c == "-" else c for c in code.upper())


def _count_caller_turns(chat_ctx: llm.ChatContext) -> int:
    """How many times the caller has spoken so far - the signal for whether a
    booking flow was actually driven by the caller or silently re-run by the model."""
    return sum(1 for it in chat_ctx.items if it.type == "message" and it.role == "user")


def speech_only(chat_ctx: llm.ChatContext) -> llm.ChatContext:
    """The conversation without tool mechanics, for handing to a sub-task.

    Tool calls in the history are scoped to the agent that made them. A
    sub-task whose schema doesn't include those tools will still see them
    being called and imitate them - smaller models invent similar-sounding
    tool names instead of using the ones they actually have. Hand every
    sub-task the words only; anything that matters from a tool result was
    spoken to the caller and survives in the messages.
    """
    return chat_ctx.copy(exclude_function_call=True, exclude_handoff=True)
