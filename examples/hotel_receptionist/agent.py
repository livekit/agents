from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from livekit.agents import AgentServer, AgentSession, JobContext, cli, inference
from livekit.agents.voice import presets

from .common import Userdata, resolve_today
from .evaluation import on_session_end, on_simulation_end
from .fake_data.seed import build_seed_bytes
from .hotel_db import HotelDB
from .instructions import build_instructions
from .policies import build_lookup_policy_tool
from .suggested_replies import SuggestedReplies
from .tools_restaurant import RestaurantToolsMixin
from .tools_rooms import RoomToolsMixin
from .tools_services import ServicesToolsMixin
from .ui_view import UiView

logger = logging.getLogger("hotel-receptionist")

# Set by the frontend on the caller's participant ("true"/"false") to opt in
# to expressive TTS delivery for the session.
EXPRESSIVE_ATTRIBUTE = "expressive"


class HotelReceptionistAgent(RoomToolsMixin, RestaurantToolsMixin, ServicesToolsMixin):
    def __init__(self, *, today: date) -> None:
        super().__init__(instructions=build_instructions(today), tools=[build_lookup_policy_tool()])

    async def on_enter(self) -> None:
        # The caller may have already said what they want before we speak -
        # pick up from there instead of re-asking "how can I help?".
        await self.session.generate_reply(
            instructions=(
                "Greet the caller in one short sentence. If they've already named a need "
                "(a room, a table, a cancellation...), move straight into helping; "
                "otherwise ask how you can help."
            )
        )


server = AgentServer()


async def _close_session_resources(db: HotelDB, ui: UiView) -> None:
    db.on_change = None
    try:
        await ui.aclose()
    finally:
        await db.aclose()


@server.rtc_session(
    on_session_end=on_session_end,
    on_simulation_end=on_simulation_end,
    agent_name="hotel_receptionist",
)
async def hotel_receptionist_agent(ctx: JobContext) -> None:
    await ctx.connect()

    # HOTEL_EXPRESSIVE=1 forces it on for sessions whose caller can't set
    # attributes (sims, console runs).
    caller = await ctx.wait_for_participant()
    expressive = (
        os.getenv("HOTEL_EXPRESSIVE") == "1"
        or caller.attributes.get(EXPRESSIVE_ATTRIBUTE) == "true"
    )
    logger.info("expressive mode: %s", "on" if expressive else "off")

    today = resolve_today()
    db = HotelDB.from_bytes(build_seed_bytes(today), today)

    ui = UiView(ctx.room, db.connection)

    async def close_session_resources() -> None:
        await _close_session_resources(db, ui)

    ctx.add_shutdown_callback(close_session_resources)

    db.on_change = ui.on_change
    await ui.start()

    userdata = Userdata(db=db, today=today)
    session = AgentSession[Userdata](
        userdata=userdata,
        vad=inference.VAD(model="silero"),
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("google/gemma-4-31b-it"),
        tts=inference.TTS(
            model="xai/tts-1",
            voice="carina",
        ),
        max_tool_steps=5,
    )
    session.update_options(expressive={"speech_steering": presets.FORMAL} if expressive else False)

    SuggestedReplies(
        session,
        ctx.room,
        llm=inference.LLM("google/gemma-4-31b-it"),
        expressive=expressive,
    ).attach()

    await session.start(agent=HotelReceptionistAgent(today=today), room=ctx.room)


def main() -> None:
    load_dotenv(Path(__file__).with_name(".env.local"))
    cli.run_app(server)


if __name__ == "__main__":
    main()
