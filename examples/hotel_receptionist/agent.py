from __future__ import annotations

import logging
import os
import sys
import time
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchmark import build_expected, diff_databases
from common import Userdata
from dotenv import load_dotenv
from fake_data.seed import build_seed_bytes
from hotel_db import (
    TODAY,
    HotelDB,
)
from instructions import build_instructions
from policies import build_lookup_policy_tool
from run_artifacts import dump_run_artifacts
from tools_restaurant import RestaurantToolsMixin
from tools_rooms import RoomToolsMixin
from tools_services import ServicesToolsMixin
from ui_view import UiView

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    SimulationContext,
    cli,
    inference,
)

load_dotenv(".env.local")

logger = logging.getLogger("hotel-receptionist")


def _expected_state_statements(userdata: dict[str, object]) -> list[str] | None:
    """Return configured state statements; an explicit empty list means unchanged state."""
    if "expected_state" not in userdata:
        return None
    expected_state = userdata["expected_state"]
    if expected_state is None:
        return []
    if not isinstance(expected_state, list):
        raise TypeError("expected_state must be a list of SQL statements")
    statements: list[str] = []
    for statement in expected_state:
        if not isinstance(statement, str):
            raise TypeError("expected_state must be a list of SQL statements")
        statements.append(statement)
    return statements


def _tag_work_activity(ctx: JobContext, *, state_changes: list[str], served_reads: bool) -> None:
    """Record whether deterministic app activity was observed, without judging outcome."""
    if state_changes or served_reads:
        ctx.tagger.add("work:observed")
    else:
        ctx.tagger.add("work:none")


class HotelReceptionistAgent(RoomToolsMixin, RestaurantToolsMixin, ServicesToolsMixin, Agent):
    def __init__(self) -> None:
        super().__init__(instructions=build_instructions(), tools=[build_lookup_policy_tool()])

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

_SEED_DB_BYTES = build_seed_bytes(TODAY)


async def on_simulation_end(ctx: SimulationContext) -> None:
    db_diffs: list[str] = []
    expected_state = _expected_state_statements(ctx.userdata())
    if expected_state is not None:
        # Grade the run on final DB state: build the scenario's `expected_state` on a
        # fresh seed, then diff it against the agent's DB. The diff compares
        # agent-decided facts only (room type, dates, extras, status), so minted
        # codes / order / which-king don't matter and the agent need not reproduce the
        # statements — while collateral damage still surfaces.
        session = ctx.job_context.primary_session
        expected = await build_expected(_SEED_DB_BYTES, expected_state)
        try:
            db_diffs = diff_databases(expected.connection, session.userdata.db.connection)
        finally:
            await expected.aclose()

    # The session outcome is the simulator's conversation judgment AND the optional
    # DB-state check. In particular, conversation-only scenarios must not be failed
    # merely because they correctly completed without a tool call or state change.
    if db_diffs:
        reason = "final DB diverges from expected: " + " | ".join(db_diffs[:8])
        ctx.fail(reason=reason)
        ctx.job_context.tagger.fail(reason=reason)
    elif ctx.simulator_verdict.success:
        ctx.job_context.tagger.success(reason=ctx.simulator_verdict.reason)
    else:
        ctx.job_context.tagger.fail(reason=ctx.simulator_verdict.reason)


async def on_session_end(ctx: JobContext) -> None:
    try:
        report = ctx.make_session_report()
    except RuntimeError:
        return

    chat = report.chat_history.copy(exclude_function_call=True, exclude_instructions=True)
    if len(chat.items) < 3:
        return

    userdata = ctx.primary_session.userdata

    # "Did the call do real work?" is a DB question, not per-tool bookkeeping:
    # compare the final DB against the untouched seed. Any change in the
    # transactional tables (booking, cancellation, modification, dispute,
    # followup, late-arrival note...) counts.
    try:
        seed_db = HotelDB.from_bytes(_SEED_DB_BYTES)
        try:
            state_changes = diff_databases(seed_db.connection, userdata.db.connection)
        finally:
            await seed_db.aclose()
    except Exception:
        logger.exception("error diffing final DB against seed")
        state_changes = []

    # Read-only calls (policy questions, availability checks, booking lookups)
    # are real work too - a Q&A call that answered from a successful read tool
    # shouldn't be tagged as having accomplished nothing.
    read_tools = {
        "lookup_policy",
        "lookup_booking",
        "lookup_invoice",
        "lookup_restaurant_reservation",
        "check_room_availability",
        "check_restaurant_availability",
        "lookup_guest_history",
    }
    call_names = {
        item.call_id: item.name
        for item in report.chat_history.items
        if item.type == "function_call"
    }
    served_reads = any(
        item.type == "function_call_output"
        and not item.is_error
        and call_names.get(item.call_id) in read_tools
        for item in report.chat_history.items
    )

    # This is useful activity metadata, not an outcome judgment. Simulations set
    # lk.success/lk.fail from their actual verdict in on_simulation_end; production
    # sessions may set an outcome through app-specific instrumentation.
    _tag_work_activity(ctx, state_changes=state_changes, served_reads=served_reads)

    logger.info("session tags: %s", ctx.tagger.tags)

    dump_run_artifacts(ctx, report, userdata.db)

    try:
        await userdata.db.aclose()
    except Exception:
        logger.exception("error closing hotel DB")


@server.rtc_session(on_session_end=on_session_end, on_simulation_end=on_simulation_end)
async def hotel_receptionist_agent(ctx: JobContext) -> None:
    await ctx.connect()

    db = HotelDB.from_bytes(_SEED_DB_BYTES)

    ui = UiView(ctx.room, db.connection)
    db.on_change = ui.on_change
    await ui.start()

    userdata = Userdata(db=db)
    session = AgentSession[Userdata](
        userdata=userdata,
        # An explicit VAD is required (not the bundled default): without it the
        # speaking anchor falls back to the STT stream clock, which drifts into the
        # future across a long call / nested-task switch and makes the turn-commit
        # logic sleep for that offset (~the elapsed call time) before replying.
        vad=inference.VAD(model="silero"),
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("google/gemma-4-31b-it"),
        tts=inference.TTS("inworld/inworld-tts-2"),
        max_tool_steps=5,
    )

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

    await session.start(agent=HotelReceptionistAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
