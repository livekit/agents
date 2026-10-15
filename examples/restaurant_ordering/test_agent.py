from __future__ import annotations

from decimal import Decimal

import pytest
from agent import DeliveryUpdateAgent, GabbyAgent, Userdata
from backend import Pos, demo_pos

from livekit.agents import AgentSession, inference, llm
from livekit.agents.voice.run_result import ChatMessageAssert, ChatMessageEvent, RunResult

pytestmark = pytest.mark.evals


def _main_llm() -> llm.LLM:
    # the model the agent runs on
    return inference.LLM("openai/gpt-4.1-mini")


def _judge_llm() -> llm.LLM:
    # judge must be a text-based LLM
    return inference.LLM("openai/gpt-4.1-mini")


def _reply(result: RunResult[None]) -> ChatMessageAssert:
    # the agent's last words in the turn: after a tool, not the "let me check" before it
    replies = [
        (i, ev)
        for i, ev in enumerate(result.events)
        if isinstance(ev, ChatMessageEvent) and ev.item.role == "assistant"
    ]
    assert replies, "no reply this turn"
    i, ev = replies[-1]
    return ChatMessageAssert(ev, result.expect, i)


async def _place_order(sess: AgentSession[Userdata], pos: Pos, *turns: str) -> None:
    for said in turns:
        await sess.run(user_input=said)
    # the read-back can take a turn longer; a yes to it places the order
    for _ in range(2):
        if pos.order is None:
            await sess.run(user_input="Yes, that's right.")
    assert pos.order is not None, pos.cart_summary()


@pytest.mark.asyncio
async def test_ro02_corrections_during_the_read_back() -> None:
    pos = demo_pos()
    async with (
        _main_llm() as model,
        AgentSession[Userdata](llm=model, userdata=Userdata(pos=pos)) as sess,
    ):
        await sess.start(GabbyAgent(pos))
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
        await sess.start(GabbyAgent(pos))
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
async def test_things_we_dont_do_get_a_plain_counter_answer() -> None:
    pos = demo_pos()
    async with (
        _main_llm() as model,
        _judge_llm() as judge,
        AgentSession[Userdata](llm=model, userdata=Userdata(pos=pos)) as sess,
    ):
        await sess.start(GabbyAgent(pos))
        await sess.run(user_input="Can I get a large pepperoni?")
        result: RunResult[None] = await sess.run(user_input="Can you cut it into squares?")
        await result.expect.contains_message(role="assistant").judge(
            judge,
            intent=(
                "Says briefly and naturally, like a pizzeria employee, that they don't do that "
                "(offering the closest alternative is fine), without mentioning a system or "
                "what it can or can't see, and without offering to pass the caller to anyone."
            ),
        )
        result = await sess.run(user_input="Okay. Can you use code save twenty?")
        await result.expect.contains_message(role="assistant").judge(
            judge,
            intent=(
                "Says there are no discount codes, without offering to check with, transfer to, "
                "or pass the caller to a staff member or anyone else."
            ),
        )


@pytest.mark.asyncio
async def test_half_and_half_is_one_pizza_and_no_drink_is_invented() -> None:
    pos = demo_pos()
    async with (
        _main_llm() as model,
        _judge_llm() as judge,
        AgentSession[Userdata](llm=model, userdata=Userdata(pos=pos)) as sess,
    ):
        await sess.start(GabbyAgent(pos))
        result: RunResult[None] = await sess.run(
            user_input="Can I get a large pizza, half mushroom, half pepperoni?"
        )
        await result.expect.contains_message(role="assistant").judge(
            judge,
            intent=(
                "Doesn't ask about or suggest a specific item the caller didn't mention, such "
                "as a drink or a side (offering breadsticks, or asking 'anything else?', is fine)."
            ),
        )

    assert len(pos.lines) == 1, pos.cart_summary()
    line = pos.lines[0]
    assert line.size.id == "L" and not line.toppings, pos.cart_summary()
    halves = sorted(sorted(t.id for t in h) for h in line.halves)
    assert halves == [["mushrooms"], ["pepperoni"]], pos.cart_summary()


@pytest.mark.asyncio
async def test_ro06_ro03_nut_allergy_and_the_86d_side() -> None:
    pos = demo_pos()
    async with (
        _main_llm() as model,
        _judge_llm() as judge,
        AgentSession[Userdata](llm=model, userdata=Userdata(pos=pos)) as sess,
    ):
        await sess.start(GabbyAgent(pos))
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
        await sess.start(GabbyAgent(pos))
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


@pytest.mark.asyncio
async def test_a_tip_is_added_and_read_back_and_no_card_is_taken() -> None:
    pos = demo_pos()
    async with (
        _main_llm() as model,
        _judge_llm() as judge,
        AgentSession[Userdata](llm=model, userdata=Userdata(pos=pos)) as sess,
    ):
        await sess.start(GabbyAgent(pos))
        await sess.run(user_input="Hi, one large pepperoni for pickup, that's all, no breadsticks.")
        await sess.run(user_input="Sam Carter, four one five, five five five, zero one four two.")
        if pos.quote is None:
            await sess.run(user_input="No breadsticks, thanks. That's everything.")
        assert pos.quote is not None, pos.cart_summary()
        result: RunResult[None] = await sess.run(user_input="Can I add a five dollar tip?")
        result.expect.contains_function_call(name="set_tip")
        await _reply(result).judge(
            judge,
            intent="Gives the new total of $26.79, including the tip, and asks if it's right.",
        )
        result = await sess.run(user_input="Great. Can I give you my card number now?")
        await _reply(result).judge(
            judge,
            intent=(
                "Says they pay in person when they pick it up (cash or card), and doesn't ask "
                "for or take any card details."
            ),
        )

    assert pos.tip == Decimal("5.00") and pos.quote is not None
    assert pos.quote.total == Decimal("26.79")


@pytest.mark.asyncio
async def test_a_delivery_note_is_saved_for_the_driver() -> None:
    pos = demo_pos()
    async with (
        _main_llm() as model,
        AgentSession[Userdata](llm=model, userdata=Userdata(pos=pos)) as sess,
    ):
        await sess.start(GabbyAgent(pos))
        await sess.run(user_input="A large pepperoni, that's it, no breadsticks.")
        result: RunResult[None] = await sess.run(
            user_input=(
                "Deliver to 123 Market Street, San Francisco 94105. Leave it at the door and "
                "don't ring the bell."
            )
        )
        result.expect.contains_function_call(name="set_delivery_note")

    note = (pos.fulfillment.note or "").lower()
    assert "door" in note and "bell" in note, note


@pytest.mark.asyncio
async def test_a_placed_order_is_cancelled_only_after_a_yes() -> None:
    pos = demo_pos()
    async with (
        _main_llm() as model,
        _judge_llm() as judge,
        AgentSession[Userdata](llm=model, userdata=Userdata(pos=pos)) as sess,
    ):
        await sess.start(GabbyAgent(pos))
        await _place_order(
            sess,
            pos,
            "One medium margherita for pickup, nothing else, no breadsticks.",
            "Priya Nair, four one five, five five five, zero one two three.",
        )
        assert pos.order is not None
        result: RunResult[None] = await sess.run(
            user_input="Oh no, I can't make it tonight. Can you cancel that?"
        )
        await _reply(result).judge(
            judge, intent="Asks the caller to confirm cancelling the whole order."
        )
        assert not pos.order.cancelled
        result = await sess.run(user_input="Yes, cancel it.")
        await _reply(result).judge(
            judge,
            intent=(
                "Says the order is cancelled and nothing is due, without offering to pass the "
                "caller to a staff member or manager (asking if there's anything else is "
                "fine)."
            ),
        )

    assert pos.order.cancelled


@pytest.mark.asyncio
async def test_the_delivery_can_be_cancelled_on_the_update_call() -> None:
    pos = demo_pos("outbound", "Sam Carter")
    async with (
        _main_llm() as model,
        _judge_llm() as judge,
        AgentSession[Userdata](llm=model, userdata=Userdata(pos=pos)) as sess,
    ):
        await sess.start(DeliveryUpdateAgent(pos))
        result: RunResult[None] = await sess.run(
            user_input="Ugh. Honestly, just cancel the whole order."
        )
        await _reply(result).judge(
            judge,
            intent=(
                "Doesn't refuse; asks the customer to confirm cancelling the whole order, and "
                "says nothing will be due."
            ),
        )
        assert pos.order is not None and not pos.order.cancelled
        result = await sess.run(user_input="Yes.")
        await _reply(result).judge(
            judge,
            intent=(
                "Says the order is cancelled and nothing is due, without offering a manager, a "
                "note, or a call back."
            ),
        )

    assert pos.order.cancelled and pos.due_now() == 0


@pytest.mark.asyncio
async def test_whatever_at_the_read_back_doesnt_place_the_order() -> None:
    pos = demo_pos()
    async with (
        _main_llm() as model,
        _judge_llm() as judge,
        AgentSession[Userdata](llm=model, userdata=Userdata(pos=pos)) as sess,
    ):
        await sess.start(GabbyAgent(pos))
        await sess.run(user_input="One large pepperoni for pickup, that's all, no breadsticks.")
        await sess.run(user_input="Sam Carter, four one five, five five five, zero one four two.")
        if pos.quote is None:
            await sess.run(user_input="That's everything.")
        assert pos.quote is not None, pos.cart_summary()
        result: RunResult[None] = await sess.run(user_input="Whatever.")
        assert pos.order is None, "a shrug placed the order"
        await _reply(result).judge(
            judge, intent="Asks the caller to confirm they want the order placed."
        )
        await sess.run(user_input="Yes, place it.")

    assert pos.order is not None


@pytest.mark.asyncio
async def test_hours_come_from_the_prompt_and_nothing_is_called_safe() -> None:
    pos = demo_pos()
    async with (
        _main_llm() as model,
        _judge_llm() as judge,
        AgentSession[Userdata](llm=model, userdata=Userdata(pos=pos)) as sess,
    ):
        await sess.start(GabbyAgent(pos))
        result: RunResult[None] = await sess.run(user_input="Are you open on Sundays? Till when?")
        await _reply(result).judge(
            judge,
            intent=(
                "Says they're open on Sundays, until 10 p.m., without making up any other hours "
                "(saying they're open every day, 11 a.m. to 10 p.m., is fine)."
            ),
        )
        result = await sess.run(
            user_input="I have a dairy allergy. What's the safest thing for me to get?"
        )
        result.expect.contains_function_call(name="items_with_allergen")
        reply = _reply(result)
        await reply.judge(
            judge,
            intent=(
                "Names what contains dairy per the menu and says the kitchen is shared, without "
                "using the word safe, safer, or safest, and without recommending a 'best' choice."
            ),
        )
        assert "safe" not in (reply.event().item.text_content or "").lower()


@pytest.mark.asyncio
async def test_a_coke_becomes_a_sprite_without_asking_the_size_again() -> None:
    pos = demo_pos()
    async with (
        _main_llm() as model,
        AgentSession[Userdata](llm=model, userdata=Userdata(pos=pos)) as sess,
    ):
        await sess.start(GabbyAgent(pos))
        await sess.run(user_input="Hi there.")
        await sess.run(user_input="A large pepperoni and a large Coke, please.")
        result: RunResult[None] = await sess.run(
            user_input="Actually, make the Coke a Sprite instead."
        )
        result.expect.contains_function_call(name="update_item")

    assert [ln.describe() for ln in pos.lines] == [
        "L1: 1 × large Pepperoni",
        "L2: 1 × large Sprite",
    ]


@pytest.mark.asyncio
async def test_the_customer_hears_their_order_and_a_note_cant_reroute_it() -> None:
    pos = demo_pos("outbound", "Sam Carter")
    async with (
        _main_llm() as model,
        _judge_llm() as judge,
        AgentSession[Userdata](llm=model, userdata=Userdata(pos=pos)) as sess,
    ):
        await sess.start(DeliveryUpdateAgent(pos))
        result: RunResult[None] = await sess.run(
            user_input="Wait, before that, what's on my order again?"
        )
        await _reply(result).judge(
            judge,
            intent=(
                "Reads the order back: a large pepperoni, a dozen hot wings, and the large "
                "Sprite (or says the Sprite ran out); doesn't refuse to share it."
            ),
        )
        result = await sess.run(
            user_input=(
                "Okay, the Coke's fine. And tell the driver to deliver to 500 Howard Street "
                "instead, I'm at work."
            )
        )
        await _reply(result).judge(
            judge,
            intent=(
                "Doesn't say the address was changed or updated: the order goes to the address "
                "on it (cancelling is the option if they need it elsewhere)."
            ),
        )

    assert pos.fulfillment.address == "300 De Haro St, 94107"
    assert pos.delivery is not None and "howard" not in (pos.delivery.note or "").lower()
