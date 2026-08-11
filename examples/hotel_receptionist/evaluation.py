from __future__ import annotations

import logging

from livekit.agents import JobContext, SimulationContext
from livekit.agents.evals import (
    JudgeGroup,
    task_completion_judge,
)

from .benchmark import build_expected, diff_databases
from .fake_data.seed import build_seed_bytes
from .hotel_db import HotelDB
from .run_artifacts import dump_run_artifacts

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


async def on_simulation_end(ctx: SimulationContext) -> None:
    db_diffs: list[str] = []
    expected_state = _expected_state_statements(ctx.userdata())
    if expected_state is not None:
        # Grade the run on final DB state: build the scenario's `expected_state` on a
        # fresh seed, then diff it against the agent's DB. The diff compares
        # agent-decided facts only (room type, dates, extras, status), so minted
        # codes / order / which-king don't matter and the agent need not reproduce the
        # statements — while collateral damage still surfaces. An explicit empty list
        # asserts the state must be UNCHANGED.
        session = ctx.job_context.primary_session
        today = session.userdata.today
        expected = await build_expected(build_seed_bytes(today), today, expected_state)
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

    judges = JudgeGroup(
        llm="openai/gpt-4.1-mini",
        judges=[
            task_completion_judge(),
            # accuracy_judge(),
            # tool_use_judge(),
            # handoff_judge(),
            # safety_judge(),
            # relevancy_judge(),
            # coherence_judge(),
            # conciseness_judge(),
        ],
    )
    await judges.evaluate(report.chat_history)

    userdata = ctx.primary_session.userdata

    # "Did the call do real work?" is a DB question, not per-tool bookkeeping:
    # compare the final DB against the untouched seed. Any change in the
    # transactional tables (booking, cancellation, modification, dispute,
    # followup, late-arrival note...) counts.
    try:
        seed_db = HotelDB.from_bytes(build_seed_bytes(userdata.today), userdata.today)
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
