from __future__ import annotations

import logging
import os
import time
from collections import deque
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from livekit.agents import (
    AgentServer,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    ToolError,
    cli,
    function_tool,
    inference,
    llm,
)

from .capabilities import CAPABILITIES, render
from .common import Userdata, resolve_today
from .evaluation import on_session_end, on_simulation_end
from .fake_data.seed import build_seed_bytes
from .hotel_db import HotelDB
from .persona_core import core_instructions
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


def _expressive_options(expressive: bool) -> bool | dict[str, object]:
    """The session's expressive-pipeline config: composed, formal delivery
    (the old presets.FORMAL) - breathing and light fillers stay on, every
    other non-verbal sound is disabled."""
    if not expressive:
        return False
    return {
        "speech_steering": {
            "nonverbal_sounds": {
                "laughing": False,
                "sighing": False,
                "crying": False,
                "vocalizing": False,
                "mouth_sounds": False,
                "reflex_sounds": False,
            },
        },
    }


class HotelReceptionistAgent(RoomToolsMixin, RestaurantToolsMixin, ServicesToolsMixin):
    def __init__(self, *, today: date) -> None:
        super().__init__(instructions=core_instructions(today), tools=[build_lookup_policy_tool()])

        # Agent.__init__ collects every mixin tool. Hold them in a registry keyed by
        # name and expose only the resident pair, so the re-sent prefix carries the
        # router rather than 35 schemas; load_capability switches the rest on.
        self._registry = {tool.info.name: tool for tool in self._tools}
        self._loaded: set[str] = set()
        self._tools = self._visible_tools()
        self._chat_ctx = self._chat_ctx.copy(tools=self._tools)

    def _visible_tools(self) -> list[llm.Tool | llm.Toolset]:
        # Returns rather than assigns self._tools: update_tools() diffs against the
        # current list to emit the AgentConfigUpdate that records the switch, so
        # assigning first would make every capability load invisible.
        resident = ["load_capability", "say_goodbye_and_close_call"]
        names = dict.fromkeys(
            resident + [name for area in self._loaded for name in CAPABILITIES[area].tools]
        )
        return [self._registry[name] for name in names if name in self._registry]

    @function_tool
    async def load_capability(self, area: str) -> str:
        """Switch on the tools and procedure for one area of the job. Call this as soon as
        the caller names what they need, before promising anything in that area.

        Args:
            area: One of rooms, billing, restaurant, concierge, guest_services, groups,
                emergency, transfer, policy.
        """
        if area not in CAPABILITIES:
            raise ToolError(f"unknown area {area!r} - valid areas: {', '.join(CAPABILITIES)}")
        self._loaded.add(area)
        await self.update_tools(self._visible_tools())
        return render(area)

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
        # A dense caller turn legitimately chains several recording tools before the
        # reply (set_stay + choose_room + a dialog + confirm_booking); at 5 the cap
        # was hit mid-booking-flow and the closing confirm_booking got suppressed,
        # leaving the task wedged with nothing written.
        max_tool_steps=8,
    )
    # The expressive pipeline is framework-internal (AgentSession exposes no
    # public switch yet - update_options() grew no expressive kwarg when the
    # pipeline landed upstream), so flip the same private attribute the
    # framework's own expressive tests use.
    session._expressive = _expressive_options(expressive)

    # Token-usage instrumentation: the inference gateway enforces a per-minute LLM
    # token quota project-wide, so log every LLM request's token counts plus a
    # rolling 60s total to see exactly what consumes the budget.
    llm_events: deque[tuple[float, int]] = deque()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent) -> None:
        m = ev.metrics
        if m.type != "llm_metrics":
            return
        now = time.monotonic()
        llm_events.append((now, m.total_tokens))
        while llm_events and now - llm_events[0][0] > 60:
            llm_events.popleft()
        window_tokens = sum(t for _, t in llm_events)
        logger.info(
            "LLM usage: prompt=%d (cached=%d) completion=%d total=%d ttft=%.2fs "
            "| last-60s (this session, agent LLM only): %d tokens across %d requests",
            m.prompt_tokens,
            m.prompt_cached_tokens,
            m.completion_tokens,
            m.total_tokens,
            m.ttft,
            window_tokens,
            len(llm_events),
        )

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
