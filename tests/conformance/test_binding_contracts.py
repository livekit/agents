"""B-19 S0 vectors through today's generated native entrypoints, without test-only exports."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from livekit.agents import _core
from livekit.agents.types import NOT_GIVEN

pytestmark = pytest.mark.unit

_TABLES = json.loads(Path(__file__).with_name("cases.json").read_text())


def _cases(family: str) -> list[Any]:
    return [
        pytest.param(
            case,
            id=case["id"],
            marks=(
                pytest.mark.xfail(
                    reason=case["defect"],
                    strict=True,
                    raises=(AssertionError, pytest.fail.Exception),
                )
                if "defect" in case
                else ()
            ),
        )
        for case in _TABLES[family]
    ]


@pytest.mark.parametrize("case", _cases("presence"))
def test_presence_defaults_only_on_omission(case: dict[str, Any]) -> None:
    filtered = _core.ExpFilter(0.5, initial=8.0)
    arguments = {k: NOT_GIVEN if v == "NOT_GIVEN" else v for k, v in case["arguments"].items()}
    if "error" in case:
        with pytest.raises(TypeError, match=case["error"]):
            filtered.reset(**arguments)
        assert filtered.value == 8.0
    else:
        filtered.reset(**arguments)
        assert filtered.value == case["expected"]


@pytest.mark.parametrize("case", _cases("scalars"))
def test_scalar_conversion_preserves_values_and_types(case: dict[str, Any]) -> None:
    value = case["value"]
    if case["kind"] == "string":
        assert _core.GetDtmfResult(user_input=value).user_input == case["expected"]
    elif case["kind"] == "float":
        assert _core.ExpFilter(0.5, initial=value).value == case["expected"]
    elif case["kind"] == "bool":
        if "error" in case:
            with pytest.raises(TypeError, match=case["error"]):
                _core.RecordingOptions(audio=value)
        else:
            assert _core.RecordingOptions(audio=value).audio is case["expected"]
    else:
        error_type = TypeError if isinstance(value, bool) else OverflowError
        with pytest.raises(error_type, match=case["error"]):
            _core.MovingAverage(value)


@pytest.mark.parametrize("case", _cases("time"))
async def test_duration_reports_validate_before_settling_playout(case: dict[str, Any]) -> None:
    tracker = _core.PlaybackTracker()
    tracker.note_capture()
    tracker.note_flush()
    value = float(case["value"])
    if "error" in case:
        with pytest.raises(ValueError, match=case["error"]):
            tracker.playback_finished(value, False, None)
        # Refusing a report must leave the outstanding segment available for a valid report.
        assert tracker.playback_finished(0.125, False, None)
    else:
        assert tracker.playback_finished(value, False, None)
        position, interrupted, transcript = await tracker.wait_for_playout()
        assert position == case["expected"]
        assert not interrupted and transcript is None


@pytest.mark.parametrize("case", _cases("collections"))
def test_typed_collections_validate_every_element(case: dict[str, Any]) -> None:
    if "error" in case:
        with pytest.raises(TypeError, match=case["error"]):
            _core.GetDtmfResult.from_dtmf_inputs(case["value"])
    else:
        values = list(case["value"])
        result = _core.GetDtmfResult.from_dtmf_inputs(values)
        values.append("9")
        assert result.user_input == case["expected"]


@pytest.mark.parametrize("case", _cases("errors"))
def test_declared_errors_keep_argument_and_domain_details(case: dict[str, Any]) -> None:
    if case["operation"] == "construct":
        with pytest.raises(ValueError, match=case["error"]):
            _core.ExpFilter(case["value"])
    elif case["operation"] == "reset":
        filtered = _core.ExpFilter(0.5, initial=8.0)
        with pytest.raises(AssertionError, match=case["error"]):
            filtered.reset(alpha=case["value"])
        assert filtered.apply(1, 0.0) == 4.0
    else:
        with pytest.raises(TypeError, match=case["error"]):
            _core.ExpFilter(case["value"])


@pytest.mark.parametrize("case", _cases("cancellation"))
async def test_cancelling_native_wait_does_not_cancel_playout(case: dict[str, Any]) -> None:
    tracker = _core.PlaybackTracker()
    tracker.note_capture()
    tracker.note_flush()

    async def wait() -> tuple[float, bool, str | None]:
        return await tracker.wait_for_playout()

    waiter = asyncio.create_task(wait())
    try:
        if case["start"]:
            await asyncio.sleep(0)
            assert not waiter.done()
        waiter.cancel()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert tracker.playback_finished(case["position"], False, "settled")
        assert await tracker.wait_for_playout() == (case["position"], False, "settled")
        assert waiter.cancelled(), "late completion revived the cancelled waiter"
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
