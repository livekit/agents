from __future__ import annotations

import pytest
from agent import RosaAgent, Userdata
from backend import demo_pos

from livekit.agents import AgentSession, inference, llm
from livekit.agents.voice.run_result import RunResult

pytestmark = pytest.mark.evals


def _main_llm() -> llm.LLM:
    # the model the agent runs on
    return inference.LLM("openai/gpt-4.1-mini")


def _judge_llm() -> llm.LLM:
    # judge must be a text-based LLM
    return inference.LLM("openai/gpt-4.1-mini")


@pytest.mark.asyncio
async def test_ro02_corrections_during_the_read_back() -> None:
    pos = demo_pos()
    async with (
        _main_llm() as model,
        AgentSession[Userdata](llm=model, userdata=Userdata(pos=pos)) as sess,
    ):
        await sess.start(RosaAgent(pos))
        await sess.run(
            user_input="Hi, can I get a large pepperoni, a dozen hot wings, and a large Coke?"
        )
        await sess.run(user_input="No thanks, that's it. It's for pickup.")
        await sess.run(user_input="Sam Carter, four one five, five five five, zero one four two.")
        # by now she's priced it and is reading it back
        assert pos.quote is not None, pos.cart_summary()
        await sess.run(user_input="Oh wait, make that pizza a medium. And drop the Coke.")
        await sess.run(user_input="Yep, that's right.")

    assert pos.order is not None, pos.cart_summary()
    assert [ln.describe() for ln in pos.lines] == [
        "L1: 1 × medium Pepperoni",
        "L2: 1 × dozen Bone-in Wings, hot sauce",
    ]
    assert pos.quote_current


@pytest.mark.asyncio
async def test_a_large_pizza_and_a_coke_asks_the_coke_size() -> None:
    # the pizza's "large" mustn't carry over: the Coke waits for its own size
    pos = demo_pos()
    async with (
        _main_llm() as model,
        _judge_llm() as judge,
        AgentSession[Userdata](llm=model, userdata=Userdata(pos=pos)) as sess,
    ):
        await sess.start(RosaAgent(pos))
        result: RunResult[None] = await sess.run(user_input="Large pepperoni and a Coke.")
        await result.expect.contains_message(role="assistant").judge(
            judge,
            intent="Asks what size the Coke should be, offering small, medium, or large.",
        )
        assert not any(ln.item.id == "coke" for ln in pos.lines), pos.cart_summary()

        await sess.run(user_input="Medium.")

    assert [ln.describe() for ln in pos.lines] == [
        "L1: 1 × large Pepperoni",
        "L2: 1 × medium Coke",
    ]


@pytest.mark.asyncio
async def test_ro06_ro03_nut_allergy_and_the_86d_side() -> None:
    pos = demo_pos()
    async with (
        _main_llm() as model,
        _judge_llm() as judge,
        AgentSession[Userdata](llm=model, userdata=Userdata(pos=pos)) as sess,
    ):
        await sess.start(RosaAgent(pos))
        result: RunResult[None] = await sess.run(
            user_input="My kid has a nut allergy. What's safe to order?"
        )
        result.expect.contains_function_call(name="items_with_allergen")
        await result.expect.contains_message(role="assistant").judge(
            judge,
            intent=(
                "Names the pesto pizza and the Thai wings (or Thai sauce) as containing nuts, "
                "and does not claim any item is nut-free, allergen-free, or guaranteed safe."
            ),
        )

        await sess.run(user_input="Okay. Can I get the Buffalo cauliflower?")

    assert not any(ln.item.id == "buffalo_cauliflower" for ln in pos.lines)
    assert "nuts" in pos.allergies


@pytest.mark.asyncio
async def test_ro07_bursts_and_repeats_make_no_duplicates() -> None:
    pos = demo_pos()
    async with (
        _main_llm() as model,
        AgentSession[Userdata](llm=model, userdata=Userdata(pos=pos)) as sess,
    ):
        await sess.start(RosaAgent(pos))
        await sess.run(user_input="Two large pepperonis, a medium margherita,")
        await sess.run(user_input="yeah two pepperonis, and a dozen wings, BBQ, and oh, two cokes")
        await sess.run(user_input="Small. Small is fine. And two pepperonis, right?")

    by_item = {ln.item.id: ln for ln in pos.lines}
    assert sorted(by_item) == ["coke", "margherita", "pepperoni", "wings"], pos.cart_summary()
    assert len(pos.lines) == 4, pos.cart_summary()
    assert by_item["pepperoni"].quantity == 2
    assert by_item["pepperoni"].size.id == "L"
    assert by_item["margherita"].size.id == "M"
    assert by_item["wings"].sauce is not None and by_item["wings"].sauce.id == "bbq"
    assert by_item["coke"].quantity == 2 and by_item["coke"].size.id == "S"
