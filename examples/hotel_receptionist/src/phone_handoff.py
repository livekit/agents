from __future__ import annotations

import asyncio
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common import Userdata
from hotel_db import RoomBooking, speak_usd
from instructions import build_instructions

from livekit import rtc
from livekit.agents import Agent, AgentSession

logger = logging.getLogger("hotel-receptionist.phone-handoff")

HANDOFF_GUEST_LAST_NAME = "Smith"
HANDOFF_GUEST_CODE = "HTL-AB12"


def handoff_instructions(booking: RoomBooking) -> str:
    stay = (
        f"{booking.check_in.strftime('%A, %B %-d')} to {booking.check_out.strftime('%A, %B %-d')}, "
        f"{booking.nights} nights, {booking.guests} guests, a {booking.room_type.replace('_', ' ')} room"
    )
    extras = ", ".join(e.replace("_", " ") for e in booking.extras) or "no extras"
    return f"""\
# You're now on a phone call you placed
The conversation just moved from the website to a phone call: the guest asked the hotel to call them, and you're the one who called. The person on the line is {booking.first_name} {booking.last_name}, who has a confirmed stay with us: {stay}, with {extras}, total {speak_usd(booking.total)}. You dialed the number on her booking, so she is already verified for that booking - never ask her for a confirmation code, card digits, or her name, and use the booking tools directly if she wants to change anything.

This is a warm pre-arrival call, not a sales call. Open it yourself: greet her by first name, say it's the front desk at The LiveKit Hotel calling ahead of her stay, and ask if she has any questions about check-in. Once check-in is covered, ask - as its own question - whether the trip is for anything special. Then offer to have something waiting in the room when she arrives: a bottle of champagne or sparkling water, on the house. Only offer those two; if she picks one, record it with record_followup (kind="other", caller_name "{booking.first_name} {booking.last_name}", caller_phone "{booking.phone}", and a summary naming the drink, the occasion if she gave one, and the {booking.check_in.strftime("%B %-d")} arrival), then confirm it'll be waiting. If she declines, that's fine - don't push.

If she mentions an occasion like an anniversary or birthday, the special-occasions guidance still applies, but mention the suite at most once. Keep every reply as short as the rest of your instructions require, and wrap up warmly once she has nothing else.
"""


def greet_on_phone_handoff(session: AgentSession[Userdata], agent: Agent, room: rtc.Room) -> None:
    started = False
    tasks: set[asyncio.Task[None]] = set()

    def maybe_start(participant: rtc.RemoteParticipant) -> None:
        nonlocal started
        if (
            started
            or participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_SIP
            or participant.attributes.get("sip.callStatus") != "active"
        ):
            return
        started = True
        task = asyncio.create_task(_start_phone_script(session, agent))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    room.on("participant_connected", maybe_start)
    room.on(
        "participant_attributes_changed",
        lambda _changed, participant: maybe_start(participant),
    )


async def _start_phone_script(session: AgentSession[Userdata], agent: Agent) -> None:
    booking = await session.userdata.db.find_booking(
        last_name=HANDOFF_GUEST_LAST_NAME, confirmation_code=HANDOFF_GUEST_CODE
    )
    if booking is None:
        logger.warning("phone handoff guest %s not found", HANDOFF_GUEST_CODE)
        return
    session.userdata.verified_booking = booking
    await agent.update_instructions(build_instructions() + "\n\n" + handoff_instructions(booking))
    session.generate_reply(
        instructions=(
            f"The call to {booking.first_name} just connected. Greet her by first name, say it's "
            "the front desk at The LiveKit Hotel calling ahead of her stay, and ask if she has any "
            "questions about check-in."
        )
    )
