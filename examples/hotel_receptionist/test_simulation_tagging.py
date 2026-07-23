from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace
from typing import cast

import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import _tag_work_activity, on_simulation_end
from common import Userdata
from fake_data.seed import build_seed_bytes
from hotel_db import TODAY, HotelDB

from livekit.agents import (
    JobContext,
    Scenario,
    SimulationContext,
    SimulationDispatch,
    SimulationVerdict,
)
from livekit.agents.observability import Tagger

pytestmark = pytest.mark.unit

_MISSING = object()


def _simulation_context(
    db: HotelDB,
    *,
    simulator_success: bool,
    expected_state: object = _MISSING,
) -> SimulationContext:
    scenario_userdata = (
        "" if expected_state is _MISSING else json.dumps({"expected_state": expected_state})
    )
    dispatch = SimulationDispatch(scenario=Scenario(userdata=scenario_userdata))
    job_ctx = cast(
        JobContext,
        SimpleNamespace(
            tagger=Tagger(),
            primary_session=SimpleNamespace(userdata=Userdata(db=db)),
        ),
    )
    ctx = SimulationContext(dispatch, job_ctx)
    ctx._begin_finalize(
        simulator_verdict=SimulationVerdict(
            success=simulator_success,
            reason="conversation passed" if simulator_success else "conversation failed",
        ),
        run=None,
        job=None,
    )
    return ctx


@pytest.mark.asyncio
async def test_conversation_only_pass_sets_success_outcome() -> None:
    db = HotelDB.from_bytes(build_seed_bytes(TODAY))
    try:
        ctx = _simulation_context(db, simulator_success=True)
        await on_simulation_end(ctx)

        assert ctx.job_context.tagger.outcome == "success"
        assert ctx.job_context.tagger.outcome_reason == "conversation passed"
        assert ctx.user_verdict is None
    finally:
        await db.aclose()


@pytest.mark.asyncio
async def test_conversation_only_failure_sets_failure_outcome() -> None:
    db = HotelDB.from_bytes(build_seed_bytes(TODAY))
    try:
        ctx = _simulation_context(db, simulator_success=False)
        await on_simulation_end(ctx)

        assert ctx.job_context.tagger.outcome == "fail"
        assert ctx.job_context.tagger.outcome_reason == "conversation failed"
        assert ctx.user_verdict is None
    finally:
        await db.aclose()


@pytest.mark.asyncio
async def test_matching_expected_state_preserves_conversation_success() -> None:
    db = HotelDB.from_bytes(build_seed_bytes(TODAY))
    try:
        ctx = _simulation_context(db, simulator_success=True, expected_state=[])
        await on_simulation_end(ctx)

        assert ctx.job_context.tagger.outcome == "success"
        assert ctx.user_verdict is None
    finally:
        await db.aclose()


@pytest.mark.asyncio
async def test_db_divergence_vetoes_conversation_success_and_sets_failure_outcome() -> None:
    db = HotelDB.from_bytes(build_seed_bytes(TODAY))
    try:
        await db.record_followup(
            kind="other",
            caller_name="Test Guest",
            caller_phone="401",
            summary="Unexpected state change",
        )
        ctx = _simulation_context(db, simulator_success=True, expected_state=[])
        await on_simulation_end(ctx)

        assert ctx.job_context.tagger.outcome == "fail"
        assert ctx.user_verdict is not None
        assert ctx.user_verdict.success is False
        assert "final DB diverges from expected" in ctx.user_verdict.reason
    finally:
        await db.aclose()


@pytest.mark.parametrize(
    ("state_changes", "served_reads", "expected_tag"),
    [
        ([], False, "work:none"),
        (["followups changed"], False, "work:observed"),
        ([], True, "work:observed"),
    ],
)
def test_work_activity_tags_do_not_set_an_outcome(
    state_changes: list[str], served_reads: bool, expected_tag: str
) -> None:
    ctx = cast(JobContext, SimpleNamespace(tagger=Tagger()))

    _tag_work_activity(ctx, state_changes=state_changes, served_reads=served_reads)

    assert ctx.tagger.tags == {expected_tag}
    assert ctx.tagger.outcome is None
