from __future__ import annotations

import os
import sys
from datetime import date, time
from typing import cast

import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from book_restaurant import BookRestaurantTask
from hotel_db import HotelDB

from livekit.agents.llm.tool_context import ToolError
from livekit.agents.llm.utils import build_strict_openai_schema, prepare_function_arguments

pytestmark = pytest.mark.unit


def _task_with_open_time(at_time: time) -> BookRestaurantTask:
    task = BookRestaurantTask(cast(HotelDB, None))
    task._date = date(2026, 6, 9)
    task._open_times = {at_time}
    return task


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"hr": 5, "minute": 30, "ampm": "am", "notes": None}, time(5, 30)),
        ({"hr": 5, "minute": 30, "ampm": "pm", "notes": "anniversary"}, time(17, 30)),
        ({"hr": 12, "minute": 0, "ampm": "am", "notes": None}, time(0, 0)),
        ({"hr": 12, "minute": 0, "ampm": "pm", "notes": None}, time(12, 0)),
    ],
)
@pytest.mark.asyncio
async def test_choose_time_converts_twelve_hour_time(
    arguments: dict[str, object], expected: time
) -> None:
    task = _task_with_open_time(expected)
    args, kwargs = prepare_function_arguments(
        fnc=task.choose_time,
        json_arguments=arguments,
    )

    result = await task.choose_time(*args, **kwargs)

    assert task._time == expected
    assert task._notes == arguments["notes"]
    assert result.startswith("time recorded: ")


@pytest.mark.parametrize(
    "arguments",
    [
        {"hr": 0, "minute": 30, "ampm": "am", "notes": None},
        {"hr": 13, "minute": 30, "ampm": "pm", "notes": None},
        {"hr": 5, "minute": -1, "ampm": "pm", "notes": None},
        {"hr": 5, "minute": 60, "ampm": "pm", "notes": None},
        {"hr": 5, "minute": 30, "ampm": "evening", "notes": None},
    ],
)
@pytest.mark.asyncio
async def test_choose_time_rejects_invalid_components(arguments: dict[str, object]) -> None:
    task = _task_with_open_time(time(17, 30))

    with pytest.raises(ToolError, match="Error parsing arguments"):
        prepare_function_arguments(fnc=task.choose_time, json_arguments=arguments)


@pytest.mark.asyncio
async def test_choose_time_strict_schema_uses_twelve_hour_components() -> None:
    task = _task_with_open_time(time(17, 30))
    parameters = build_strict_openai_schema(task.choose_time)["function"]["parameters"]

    assert set(parameters["properties"]) == {"hr", "minute", "ampm", "notes"}
    assert parameters["properties"]["hr"]["minimum"] == 1
    assert parameters["properties"]["hr"]["maximum"] == 12
    assert parameters["properties"]["minute"]["minimum"] == 0
    assert parameters["properties"]["minute"]["maximum"] == 59
    assert parameters["properties"]["ampm"]["enum"] == ["am", "pm"]
