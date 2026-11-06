import pytest

from livekit.agents import beta
from livekit.agents.llm.tool_context import ToolError

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("07700 900123", "07700900123"),
        ("020 7946 0958", "02079460958"),
        ("030 12345678", "03012345678"),
        ("06 12 34 56 78", "0612345678"),
        ("0412 345 678", "0412345678"),
        ("090-1234-5678", "09012345678"),
    ],
    ids=["uk-mobile", "uk-london", "de-berlin", "fr-mobile", "au-mobile", "jp-mobile"],
)
async def test_phone_number_accepts_national_numbers_with_a_leading_zero(
    spoken: str, expected: str
) -> None:
    # Outside North America a number dialled within the country starts with a 0. The
    # model passes on what the caller said and is told not to invent digits, so it has
    # no country code to add. These must be accepted like a US number without +1.
    task = beta.workflows.GetPhoneNumberTask(require_confirmation=True)

    await task._update_phone_number_impl(spoken, ctx=None)  # type: ignore[arg-type]
    assert task._current_phone_number == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("spoken", "expected"),
    [("415-555-0626", "4155550626"), ("+44 7700 900123", "+447700900123")],
    ids=["us-national", "e164"],
)
async def test_phone_number_accepts_us_and_international_numbers(
    spoken: str, expected: str
) -> None:
    task = beta.workflows.GetPhoneNumberTask(require_confirmation=True)

    await task._update_phone_number_impl(spoken, ctx=None)  # type: ignore[arg-type]
    assert task._current_phone_number == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spoken",
    ["+0 7700 900123", "123 456", "1234 5678 9012 3456", "+1 234 567 890 123 456"],
    ids=["plus-zero", "too-short", "too-long", "e164-too-long"],
)
async def test_phone_number_rejects_invalid_numbers(spoken: str) -> None:
    task = beta.workflows.GetPhoneNumberTask(require_confirmation=True)

    with pytest.raises(ToolError):
        await task._update_phone_number_impl(spoken, ctx=None)  # type: ignore[arg-type]
