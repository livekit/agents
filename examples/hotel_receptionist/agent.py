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
from livekit.plugins import openai

from .capabilities import CAPABILITIES, render
from .common import Userdata, resolve_today
from .evaluation import on_session_end, on_simulation_end
from .fake_data.seed import build_seed_bytes
from .hotel_db import HotelDB
from .persona_core import core_instructions
from .policies import build_lookup_policy_tool
from .raw_generation import publish_raw_generations
from .suggested_replies import SuggestedReplies
from .tools_restaurant import RestaurantToolsMixin
from .tools_rooms import RoomToolsMixin
from .tools_services import ServicesToolsMixin
from .ui_view import UiView

# Loaded at import time, not in main(): `lk agent simulate`/`start` boot the
# agent through `python -m livekit.agents start`, which imports this module and
# never calls main(). Existing environment variables win over the file. Skipped
# under pytest - tests import this module too, and the demo endpoint's env
# (OPENAI_BASE_URL, LIVEKIT_URL...) would leak into unrelated tests.
if not os.environ.get("PYTEST_VERSION"):
    load_dotenv(Path(__file__).with_name(".env.local"))

logger = logging.getLogger("hotel-receptionist")

# The receptionist's own LLM. `MODEL` overrides it so a benchmark sweep can
# point a *deployed* agent at a different model with `lk agent update-secrets`
# (a restart, not a rebuild) instead of redeploying once per model. Unset keeps
# the previous behaviour. The suggested-replies LLM below is deliberately not
# swept: it is a UI side-feature, and holding it fixed keeps the receptionist's
# own LLM the only variable under test.
DEFAULT_LLM_MODEL = "google/gemma-4-31b-it"


def _session_llm_model() -> str:
    return os.environ.get("MODEL", "").strip() or DEFAULT_LLM_MODEL


def _session_llm():
    """The receptionist's LLM: LiveKit Inference, or an OpenAI-compatible endpoint.

    `OPENAI_BASE_URL` switches to a self-hosted server (Ollama, vLLM, LM Studio) so a
    model running on the operator's own machine can be benchmarked with the same
    scenarios. Only works when this agent runs locally — a cloud deployment cannot
    reach a private address — which Benchbin enforces before launching a run.
    """
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    model = _session_llm_model()
    if not base_url:
        return inference.LLM(model)

    logger.info("session llm via self-hosted endpoint: %s @ %s", model, base_url)
    return openai.LLM(
        model=model,
        base_url=base_url,
        # Local servers usually ignore the key, but the client requires one.
        api_key=os.environ.get("OPENAI_API_KEY", "").strip() or "local",
    )


# Set by the frontend on the caller's participant ("true"/"false") to opt in
# to expressive TTS delivery for the session.
EXPRESSIVE_ATTRIBUTE = "expressive"

# Appended to the expressive pipeline's TTS-markup instructions, so it reaches
# every turn - including task/sub-flow turns - and is removed with the rest of
# the expressive instructions if the mode is toggled off.
EXPRESSIVE_EXTRA_INSTRUCTIONS = (
    "Match your delivery to the caller's mood: brighter and warmer for a happy "
    "booking, steadier and more measured for a complaint or a stressful moment."
    "Your goal is to really be expressive, dynamic, and show the customer you care."
    "Customer service is extremely important at the Gilded Rose hotel, make sure "
    "that the customers can feel it! Speak softly when they're disappointed, and "
    "add more energy when the customer is feeling energetic or happy."
    "Only emphasize a single word per sentence. Emphasize only the most important word."
    "Not every sentence needs emphasis!"
    "If it's a date, emphasize the day of the week."
    "Sprinkle in disfluencies like 'um' or 'so' at thought boundaries."
)


def _expressive_options(expressive: bool) -> bool | dict[str, object]:
    """The session's expressive-pipeline config: composed, formal delivery -
    light fillers stay on (the default, and the extra instructions ask for
    them), every non-verbal sound is disabled."""
    if not expressive:
        return False
    return {
        "speech_steering": {"nonverbal_sounds": False},
        "tts_instructions_append": EXPRESSIVE_EXTRA_INSTRUCTIONS,
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
    llm_model = _session_llm_model()
    # Logged per session so a sweep's evidence shows which model actually served
    # it — the secret is read at session start, so a restart mid-sweep is visible.
    # The endpoint is logged too: the same model id means something different when
    # it is served from a self-hosted box than from Inference.
    logger.info(
        "session llm: %s | endpoint: %s | today: %s",
        llm_model,
        os.environ.get("OPENAI_BASE_URL", "").strip() or "livekit-inference",
        today.isoformat(),
    )
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
        llm=_session_llm(),
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

    if expressive:
        publish_raw_generations(session, ctx.room)

    await session.start(agent=HotelReceptionistAgent(today=today), room=ctx.room)


def main() -> None:
    cli.run_app(server)


if __name__ == "__main__":
    main()
