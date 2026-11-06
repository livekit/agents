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
    verified_booking: RoomBooking | None = None


def _speak_code(code: str) -> str:
    # Spell character by character, with "-" spoken as the single word "dash" -
    # NOT spelled D, A, S, H (that reads as four more code characters).
    return ", ".join("dash" if c == "-" else c for c in code.upper())


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
