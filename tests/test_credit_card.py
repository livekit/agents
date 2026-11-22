from datetime import date

import pytest

from livekit.agents.beta.workflows.credit_card import GetExpirationDateTask

pytestmark = pytest.mark.unit

# a few years ahead so the card is never expired, whenever the suite runs
FUTURE_YEAR = date.today().year + 3
FUTURE_YY = FUTURE_YEAR % 100


@pytest.mark.asyncio
async def test_expiration_four_digit_year_is_shortened() -> None:
    # The prompt itself lists "04/2025" and "April 2025" as expected input, and smaller
    # models pass the year through as spoken. The tool should accept it rather than ask
    # the caller to repeat a date that was fine.
    task = GetExpirationDateTask(require_confirmation=False)

    reply = await task._update_expiration_date_impl(None, 4, FUTURE_YEAR)  # type: ignore[arg-type]

    assert reply is None
    assert task._expiration_date == f"04/{FUTURE_YY:02d}"


@pytest.mark.asyncio
async def test_expiration_two_digit_year_unchanged() -> None:
    task = GetExpirationDateTask(require_confirmation=False)

    reply = await task._update_expiration_date_impl(None, 11, FUTURE_YY)  # type: ignore[arg-type]

    assert reply is None
    assert task._expiration_date == f"11/{FUTURE_YY:02d}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("said_first", "said_again"),
    [(FUTURE_YEAR, FUTURE_YY), (FUTURE_YY, FUTURE_YEAR), (FUTURE_YEAR, FUTURE_YEAR)],
)
async def test_expiration_confirmation_matches_either_year_form(
    said_first: int, said_again: int
) -> None:
    task = GetExpirationDateTask(require_confirmation=True)

    await task._update_expiration_date_impl(None, 4, said_first)  # type: ignore[arg-type]
    confirm = {t.id: t for t in task.tools}["confirm_expiration_date"]
    reply = await confirm(4, said_again)  # type: ignore[operator]

    assert reply is None
    assert task.done()


@pytest.mark.asyncio
@pytest.mark.parametrize("year", [1999, 2100, 250, -1])
async def test_expiration_implausible_year_is_rejected(year: int) -> None:
    task = GetExpirationDateTask(require_confirmation=False)

    reply = await task._update_expiration_date_impl(None, 4, year)  # type: ignore[arg-type]

    assert reply is not None and "year is invalid" in reply
    assert not task.done()
