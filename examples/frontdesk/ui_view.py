"""UI rendering for the LiveKit Playground card attached to this example.

The Front Desk agent itself is UI-agnostic; this module is the only
place anything playground-specific lives. The agent receives an
optional ``UIView`` on its ``Userdata`` and calls two semantic
methods on it (``slots_listed`` / ``appointment_booked``). When
running outside the playground, ``Userdata.ui`` is ``None`` and
these helpers are simply never called.

Pushes are fire-and-forget so a slow or absent peer never blocks the
function-tool reply path.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from zoneinfo import ZoneInfo

from calendar_api import AvailableSlot

from livekit.agents import JobContext

logger = logging.getLogger("frontdesk.ui")

# The playground reads `views[].rpc` from playground.yaml and registers
# an RPC handler with this exact method name. Keep both in sync.
_VIEW_METHOD = "set_appointment_status"


def _relative(local: datetime.datetime, now: datetime.datetime) -> str:
    """Human "tomorrow" / "in 3 weeks" phrasing for a slot."""
    delta = local - now
    days = delta.days
    if local.date() == now.date():
        return "in less than an hour" if delta.seconds < 3600 else "later today"
    if local.date() == (now.date() + datetime.timedelta(days=1)):
        return "tomorrow"
    if days < 7:
        return f"in {days} days"
    if days < 14:
        return "in 1 week"
    return f"in {days // 7} weeks"


class UIView:
    """Pushes markdown to the playground card associated with this example."""

    def __init__(self, ctx: JobContext) -> None:
        self._ctx = ctx

    def slots_listed(
        self,
        slots: list[AvailableSlot],
        now: datetime.datetime,
        tz: ZoneInfo,
        range_days: int,
    ) -> None:
        """Render the search range + slot count. Empty input hides the card."""
        if not slots:
            self._push("")
            return
        # The agent's tool reply already walks the LLM through specific
        # options; the card just summarises the window it searched so
        # the caller has a visual anchor without an overwhelming list.
        if range_days <= 14:
            window = "Next 2 weeks"
        elif range_days <= 30:
            window = "Next month"
        else:
            window = f"Next {range_days // 30} months"
        last = max(s.start_time for s in slots).astimezone(tz)
        span = f"{now.strftime('%b %d')} – {last.strftime('%b %d')}"
        plural = "slot" if len(slots) == 1 else "slots"
        self._push(f"**{window}**\n\n[[{len(slots)}]] available {plural} · *{span}*")

    def appointment_booked(self, slot: AvailableSlot, tz: ZoneInfo) -> None:
        local = slot.start_time.astimezone(tz)
        self._push(
            f"**Booked: {local.strftime('%A, %B %d, %Y')}**\n\nat [[{local.strftime('%H:%M %Z')}]]"
        )

    # ---- internals ----

    def _push(self, payload: str) -> None:
        for p in list(self._ctx.room.remote_participants.values()):
            asyncio.create_task(self._push_to(p.identity, payload))

    async def _push_to(self, identity: str, payload: str) -> None:
        try:
            await self._ctx.room.local_participant.perform_rpc(
                destination_identity=identity,
                method=_VIEW_METHOD,
                payload=payload,
            )
        except Exception:
            logger.exception("UI push to %s failed", identity)
