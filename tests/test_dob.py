from datetime import date

import pytest

from livekit.agents import beta

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_dob_two_digit_year_normalized() -> None:
    # The prompt asks the model to normalize two-digit years, but smaller/faster
    # models often pass the spoken value through literally. The tool layer must not
    # accept "90" as year 90 AD. https://github.com/livekit/agents/issues/6067
    task = beta.workflows.GetDOBTask(require_confirmation=True)

    await task._update_dob_impl(90, 5, 15, ctx=None)  # type: ignore[arg-type]
    assert task._current_dob == date(1990, 5, 15)

    await task._update_dob_impl(5, 3, 1, ctx=None)  # type: ignore[arg-type]
    assert task._current_dob == date(2005, 3, 1)


@pytest.mark.asyncio
async def test_dob_four_digit_year_unchanged() -> None:
    task = beta.workflows.GetDOBTask(require_confirmation=True)

    await task._update_dob_impl(1962, 7, 4, ctx=None)  # type: ignore[arg-type]
    assert task._current_dob == date(1962, 7, 4)

    await task._update_dob_impl(2001, 12, 31, ctx=None)  # type: ignore[arg-type]
    assert task._current_dob == date(2001, 12, 31)
