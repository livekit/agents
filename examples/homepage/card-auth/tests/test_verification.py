import json
from dataclasses import dataclass

import pytest

from bank import ON_FILE, Bank
from card_services import BACKEND_INSTRUCTIONS, CallState, CardServices

JORDAN = {
    "full_name": "Jordan Rivera",
    "date_of_birth": "1990-04-22",
    "card_last_four": "4417",
    "last_purchase_merchant": "Trader Joe's",
    "last_purchase_amount": "42.18",
}


@dataclass
class FakeContext:
    userdata: CallState


@pytest.fixture
def published() -> dict:
    return {}


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch, published: dict) -> CardServices:
    monkeypatch.setenv("LIVEKIT_API_KEY", "unused")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "unused")

    async def publish(key, value):
        published.setdefault(f"{key}_history", []).append(value)
        published[key] = value

    return CardServices(Bank(publish=publish, lookup_seconds=0))


@pytest.fixture
def context() -> FakeContext:
    return FakeContext(CallState())


async def verify(agent, context, **overrides) -> dict:
    return json.loads(await agent.verify_identity(context, **{**JORDAN, **overrides}))


async def test_all_five_details_authenticate(agent, context, published):
    assert await verify(agent, context) == {"authenticated": True}
    assert context.userdata.cardholder.name == "Jordan Rivera"
    assert published["status_history"] == ["authenticating", "authenticated"]


@pytest.mark.parametrize(
    "wrong",
    [
        {"full_name": "Jordan Rivers"},
        {"date_of_birth": "1990-04-23"},
        {"card_last_four": "4471"},
        {"last_purchase_merchant": "Whole Foods"},
        {"last_purchase_amount": "42.81"},
    ],
)
async def test_any_wrong_detail_fails_without_saying_which(
    agent, context, published, wrong
):
    assert await verify(agent, context, **wrong) == {"authenticated": False}
    assert context.userdata.cardholder is None
    assert published["status"] == "unauthenticated"


async def test_details_from_two_cardholders_do_not_combine(agent, context):
    priya_card = {"card_last_four": "0932"}
    assert await verify(agent, context, **priya_card) == {"authenticated": False}


async def test_a_failed_attempt_can_be_retried(agent, context, published):
    await verify(agent, context, last_purchase_amount="40.00")
    assert await verify(agent, context) == {"authenticated": True}
    assert [a["authenticated"] for a in published["attempts"]] == [False, True]


async def test_spoken_formats_are_accepted(agent, context):
    result = await verify(
        agent,
        context,
        full_name="jordan rivera",
        card_last_four="44 17",
        last_purchase_merchant="Trader Joes",
        last_purchase_amount="$42.18",
    )
    assert result == {"authenticated": True}


async def test_a_vague_amount_is_just_a_failed_attempt(agent, context, published):
    result = await verify(agent, context, last_purchase_amount="about forty dollars")
    assert result == {"authenticated": False}
    assert published["attempts"][0]["submitted"]["amount"] == "about forty dollars"


def test_neither_model_is_told_anything_on_file(agent):
    prompts = agent.instructions + BACKEND_INSTRUCTIONS
    for cardholder in ON_FILE:
        for value in cardholder.to_json().values():
            assert value not in prompts
