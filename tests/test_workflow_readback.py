from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from livekit.agents import beta

pytestmark = pytest.mark.unit


def _audio_ctx() -> Any:
    return SimpleNamespace(
        speech_handle=SimpleNamespace(input_details=SimpleNamespace(modality="audio"))
    )


@pytest.mark.asyncio
async def test_email_is_spelled_once_a_confirmation_is_refused() -> None:
    task = beta.workflows.GetEmailTask()
    ctx = _audio_ctx()

    first = await task._update_email_impl("shayne.cole@gmail.com", ctx)
    second = await task._update_email_impl("shayne.cole@gmail.com", ctx)

    assert first is not None and second is not None
    assert first != second
    assert " ".join("shayne.cole@gmail.com") in second


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_name", "last_name"),
    [("Shayne", "Cole"), ("Anne-Marie", "O'Neill"), ("Ana", "García")],
)
async def test_name_is_spelled_on_subsequent_updates(first_name: str, last_name: str) -> None:
    task = beta.workflows.GetNameTask(first_name=True, last_name=True)
    ctx = _audio_ctx()

    first = await task._update_name_impl(ctx, first_name=first_name, last_name=last_name)
    second = await task._update_name_impl(ctx, first_name=first_name, last_name=last_name)
    third = await task._update_name_impl(ctx, first_name=first_name, last_name=last_name)

    assert first is not None and second is not None
    assert first != second
    spelled = " ".join(f"{first_name}{last_name}")
    assert spelled not in first
    assert spelled in second
    assert "  " not in second
    assert third == second
    assert task._first_name == first_name
    assert task._last_name == last_name


@pytest.mark.asyncio
async def test_name_with_verify_spelling_is_spelled_from_the_start() -> None:
    task = beta.workflows.GetNameTask(first_name=True, verify_spelling=True)
    ctx = _audio_ctx()

    first = await task._update_name_impl(ctx, first_name="Shayne")
    second = await task._update_name_impl(ctx, first_name="Shayne")

    assert first == second
    assert first is not None
    assert "S h a y n e" in first


@pytest.mark.asyncio
async def test_phone_is_read_digit_by_digit_once_a_confirmation_is_refused() -> None:
    task = beta.workflows.GetPhoneNumberTask()
    ctx = _audio_ctx()

    first = await task._update_phone_number_impl("415-555-0626", ctx)
    second = await task._update_phone_number_impl("415-555-0626", ctx)

    assert first is not None and second is not None
    assert first != second
    assert " ".join("4155550626") in second


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("street", "unit", "locality", "country"),
    [
        ("1 Main St", "", "Springfield", "US"),
        ("88 Harbor Lane", "Apartment 3", "Portland Oregon 97209", "United States"),
        ("5 Rue de l’Ancienne Comédie", "App C4", "75006 Paris", "France"),
    ],
)
async def test_address_is_spelled_on_subsequent_updates(
    street: str, unit: str, locality: str, country: str
) -> None:
    task = beta.workflows.GetAddressTask()
    ctx = _audio_ctx()

    first = await task._update_address_impl(street, unit, locality, country, ctx)
    second = await task._update_address_impl(street, unit, locality, country, ctx)
    third = await task._update_address_impl(street, unit, locality, country, ctx)

    assert first is not None and second is not None
    assert first != second
    spelled = " ".join(street.replace(" ", ""))
    assert spelled not in first
    assert spelled in second
    assert "  " not in second
    assert third == second
    fields = [street, unit, locality, country] if unit else [street, locality, country]
    assert task._current_address == " ".join(fields)
    assert str([spelled, *fields[1:]]) in second


@pytest.mark.asyncio
async def test_dob_is_read_part_by_part_once_a_confirmation_is_refused() -> None:
    task = beta.workflows.GetDOBTask()
    ctx = _audio_ctx()

    first = await task._update_dob_impl(1990, 5, 17, ctx)
    second = await task._update_dob_impl(1990, 5, 17, ctx)

    assert first is not None and second is not None
    assert first != second
