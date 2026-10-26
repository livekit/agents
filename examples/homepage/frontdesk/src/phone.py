from __future__ import annotations

import logging

from livekit import api, rtc
from livekit.agents import JobContext

logger = logging.getLogger("phone")

PHONE = "phone"
CALL_ERROR_ATTRIBUTE = "call.error"


def outbound_line(meta: dict[str, str]) -> tuple[str, str]:
    """The line a call to a real phone goes out on, as (trunk, from): the job metadata's, which the
    demo page fills from its own config."""
    return meta.get("trunk") or "", meta.get("from") or ""


async def ring(
    ctx: JobContext, number: str, trunk: str, from_number: str
) -> rtc.RemoteParticipant | None:
    """Rings a real phone and returns it once someone picks up; None if nobody did, after telling
    the watching page why."""
    try:
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=trunk,
                sip_number=from_number,
                sip_call_to=number,
                participant_identity=PHONE,
                wait_until_answered=True,
            )
        )
    except api.TwirpError as e:
        logger.warning("outbound call failed: %s", e)
        await ctx.room.local_participant.set_attributes(
            {CALL_ERROR_ATTRIBUTE: dial_error(e, number)}
        )
        return None
    return await ctx.wait_for_participant(identity=PHONE)


def dial_error(e: api.TwirpError, number: str) -> str:
    """Why a call to a phone couldn't be placed, the way the page says it."""
    sip = e.metadata or {}
    status = sip.get("sip_status_code", "")
    if status in ("408", "480", "487"):
        return "The phone didn't answer."
    if status in ("486", "600"):
        return "The line was busy."
    if status == "603":
        return "The call was declined."
    detail = f"SIP {status}: {sip.get('sip_status', '')}" if status else e.message
    return f"The call couldn't be placed to {number} ({detail})."
