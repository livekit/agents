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
    # Grade the run on final DB state: build the scenario's `expected_state` on a
    # fresh seed, then diff it against the agent's DB. The diff compares
    # agent-decided facts only (room type, dates, extras, status), so minted
    # codes / order / which-king don't matter and the agent need not reproduce the
    # statements — while collateral damage still surfaces.
    expected_state = _expected_state_statements(ctx.userdata())
    if expected_state is None:
        return

    session = ctx.job_context.primary_session
    expected = await build_expected(_SEED_DB_BYTES, expected_state)
    try:
        diffs = diff_databases(expected.connection, session.userdata.db.connection)
    finally:
        await expected.aclose()

    # Veto the run if the final DB state diverged. The effective result is the AND of
    # this check and the simulator's conversation judgment, so a mismatch fails a run
    # the simulator passed; a match simply leaves the simulator's verdict to stand.
    if diffs:
        ctx.fail(reason="final DB diverges from expected: " + " | ".join(diffs[:8]))


async def on_session_end(ctx: JobContext) -> None:
    try:
        report = ctx.make_session_report()
    except RuntimeError:
        return

    chat = report.chat_history.copy(exclude_function_call=True, exclude_instructions=True)
    if len(chat.items) < 3:
        return

    userdata = ctx.primary_session.userdata

    db_diffs: list[str] = []
    expected_state_configured = False
    try:
        sim_ctx = ctx.simulation_context()
        if sim_ctx is None:
            logger.info(
                "local expected-state diff skipped: no simulation context "
                "(job/room metadata carried no SimulationDispatch)"
            )
        expected_state = _expected_state_statements(sim_ctx.userdata()) if sim_ctx else None
        expected_state_configured = expected_state is not None
        if sim_ctx is not None and expected_state is None:
            logger.info("local expected-state diff skipped: scenario has no expected_state")
        if expected_state is not None:
            logger.info("running local expected-state diff (%d statement(s))", len(expected_state))
            expected = await build_expected(_SEED_DB_BYTES, expected_state)
            try:
                db_diffs = diff_databases(expected.connection, userdata.db.connection)
            finally:
                await expected.aclose()
    except Exception:
        logger.exception("error running local expected-state diff")

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

    if db_diffs:
        ctx.tagger.fail(reason="final DB diverges from expected: " + " | ".join(db_diffs[:8]))
    elif expected_state_configured:
        ctx.tagger.success()
    elif state_changes or served_reads:
        ctx.tagger.success()
    else:
        ctx.tagger.fail(
            reason="The call accomplished nothing: no state was changed (booking, "
            "cancellation, modification, dispute, followup, message, wake-up call...) "
            "and no information was looked up for the caller."
        )

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
