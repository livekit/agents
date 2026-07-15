from __future__ import annotations

from livekit.agents import Agent, RunContext

from .common import Userdata
from .context import speech_only
from .hotel import RoomBooking
from .verify_booking import VerifyBookingTask


class HotelAgent(Agent):
    async def _verified_booking(self, ctx: RunContext[Userdata]) -> RoomBooking:
        if ctx.userdata.verified_booking is None:
            result = await VerifyBookingTask(
                ctx.userdata.db,
                ctx.userdata.today,
                chat_ctx=speech_only(self.chat_ctx),
            )
            ctx.userdata.verified_booking = result.booking
        return ctx.userdata.verified_booking
