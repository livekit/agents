from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from livekit.agents.beta.workflows.credit_card import (
    GetCardNumberResult,
    GetCardNumberTask,
    _card_issuer,
)

pytestmark = pytest.mark.unit


# test numbers published by the card networks and payment processors, all Luhn-valid
@pytest.mark.parametrize(
    "card_number, issuer",
    [
        ("4111111111111111", "Visa"),
        ("4012888888881881", "Visa"),
        ("5555555555554444", "Mastercard"),
        ("5105105105105100", "Mastercard"),
        ("2223003122003222", "Mastercard"),
        ("2720992720992720", "Mastercard"),
        ("378282246310005", "American Express"),
        ("371449635398431", "American Express"),
        ("3530111333300000", "JCB"),
        ("3566002020360505", "JCB"),
        ("30569309025904", "Diners Club"),
        ("36227206271667", "Diners Club"),
        ("38520000023237", "Diners Club"),
        ("6011111111111117", "Discover"),
        ("6445644564456445", "Discover"),
        ("6200000000000005", "UnionPay"),
        ("2000000000000000", "Other"),
        ("5018000000000009", "Other"),
    ],
)
def test_card_issuer(card_number: str, issuer: str) -> None:
    assert _card_issuer(card_number) == issuer


def _text_ctx() -> Any:
    return SimpleNamespace(
        speech_handle=SimpleNamespace(input_details=SimpleNamespace(modality="text"))
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "card_number, issuer",
    [("2223 0031 2200 3222", "Mastercard"), ("3530 1113 3330 0000", "JCB")],
)
async def test_card_number_task_reports_issuer(card_number: str, issuer: str) -> None:
    task = GetCardNumberTask(require_confirmation=False)

    with patch.object(task, "complete") as complete:
        assert await task._update_card_number_impl(_text_ctx(), card_number) is None

    complete.assert_called_once_with(
        GetCardNumberResult(issuer=issuer, card_number=card_number.replace(" ", ""))
    )
