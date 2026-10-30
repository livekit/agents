"""Turns against the real agent on Gemma, in text. The point of sale is in memory, so nothing is
mocked: every assertion reads what the agent did to it."""

import json
from decimal import Decimal

import pytest
from livekit.agents import AgentSession, inference

from gabby import DeliveryUpdate, TakeOrder
from pizzeria import Pizzeria

MODEL = "google/gemma-4-31b-it"


async def let_the_opener_finish(session: AgentSession) -> None:
    """The caller answers once they've heard Gabby's opener, not over it."""
    if speech := session.current_speech:
        await speech.wait_for_playout()


@pytest.fixture
async def judge():
    async with inference.LLM(model=MODEL) as llm:
        yield llm


async def test_an_order_is_placed_only_in_a_turn_after_its_read_back(judge):
    pos = Pizzeria()
    read_back_turn = placed_turn = None
    async with AgentSession(llm=judge) as session:
        await session.start(TakeOrder(pos))
        for turn, line in enumerate(
            (
                "Hi, can I get a large pepperoni pizza and a dozen wings with hot sauce?",
                "No thanks, that's everything. It's for pickup.",
                "It's Sam Carter.",
                "My number is 415-555-0142.",
                "Yes, that's right.",
                "Yes, go ahead and place it.",
            )
        ):
            result = await session.run(user_input=line)
            for event in result.events:
                item = event.item
                if (
                    read_back_turn is None
                    and item.type == "function_call_output"
                    and item.name == "place_order"
                    and "readBack" in json.loads(item.output)
                ):
                    read_back_turn = turn
            if pos.order is not None:
                placed_turn = turn
                break

    assert read_back_turn is not None and placed_turn is not None
    assert placed_turn > read_back_turn
    assert sorted(ln.describe() for ln in pos.lines) == [
        "1 dozen Bone-in Wings, hot sauce",
        "1 large Pepperoni",
    ]
    assert pos.quote().total == Decimal("40.31")
    assert (pos.name, pos.phone) == ("Sam Carter", "(415) 555-0142")


async def test_a_size_said_for_one_item_isnt_borrowed_for_another(judge):
    pos = Pizzeria()
    async with AgentSession(llm=judge) as session:
        await session.start(TakeOrder(pos))
        result = await session.run(user_input="Can I get a large pepperoni and a Coke?")
        result.expect.contains_function_call(name="add_items")
        assert [ln.describe() for ln in pos.lines] == ["1 large Pepperoni"]
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(judge, intent="Asks what size Coke they'd like.")
        )


async def test_saying_an_item_again_doesnt_order_it_twice(judge):
    pos = Pizzeria()
    async with AgentSession(llm=judge) as session:
        await session.start(TakeOrder(pos))
        await session.run(user_input="Okay so two large pepperonis to start.")
        await session.run(
            user_input="Yeah, the two pepperonis, right? And a large Coke."
        )
    assert sorted(ln.describe() for ln in pos.lines) == [
        "1 large Coke",
        "2 large Pepperoni",
    ]


async def test_an_allergy_question_is_answered_from_the_menu(judge):
    pos = Pizzeria()
    async with AgentSession(llm=judge) as session:
        await session.start(TakeOrder(pos))
        result = await session.run(
            user_input="My kid has a nut allergy. What's safe for them to eat?"
        )
        result.expect.contains_function_call(
            name="check_allergen", arguments={"allergen": "nuts"}
        )
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge,
                intent="Names the pesto chicken pizza and the Thai chili peanut wings as "
                "containing nuts, and doesn't call any item safe or nut-free.",
            )
        )


async def test_a_coke_is_fine_swaps_the_sprite_at_the_same_price(judge):
    pos = Pizzeria.out_for_delivery("Sam Carter")
    async with AgentSession(llm=judge) as session:
        await session.start(DeliveryUpdate(pos))
        await let_the_opener_finish(session)
        result = await session.run(user_input="Oh, sure, a Coke is fine.")
        result.expect.contains_function_call(
            name="swap_sold_out_item", arguments={"item_id": "coke"}
        )
    assert not pos.sold_out_unresolved()
    assert pos.balance() == 0


async def test_a_wrong_number_hears_nothing_about_the_order(judge):
    pos = Pizzeria.out_for_delivery("Sam Carter")
    async with AgentSession(llm=judge) as session:
        await session.start(DeliveryUpdate(pos))
        await let_the_opener_finish(session)
        result = await session.run(
            user_input="Sorry, I think you've got the wrong number. There's no Sam here."
        )
        result.expect.contains_function_call(
            name="end_without_update", arguments={"reason": "wrong_person"}
        )
    assert pos.changes() == []


async def test_a_tip_after_the_read_back_gets_read_back_before_its_placed(judge):
    pos = Pizzeria()
    async with AgentSession(llm=judge) as session:
        await session.start(TakeOrder(pos))
        for line in (
            "Just one large pepperoni for pickup, nothing else.",
            "It's Sam Carter, 415-555-0142.",
        ):
            await session.run(user_input=line)
        result = await session.run(user_input="Can I add a five dollar tip?")
        assert pos.order is None
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge,
                intent="Reads back the order with the new total of twenty-six seventy-nine, "
                "including the five dollar tip, and asks if it's right.",
            )
        )
        await session.run(user_input="Yes, that's right.")
    assert pos.order is not None and pos.tip == 5


async def test_pickup_said_up_front_is_set_without_placeholders(judge):
    pos = Pizzeria()
    async with AgentSession(llm=judge) as session:
        await session.start(TakeOrder(pos))
        await session.run(user_input="Could I get a medium margherita for pickup?")
    assert pos.fulfillment.mode == "pickup"
    assert [ln.describe() for ln in pos.lines] == ["1 medium Margherita"]


async def test_cancelling_the_delivery_waits_for_their_yes(judge):
    pos = Pizzeria.out_for_delivery("Sam Carter")
    async with AgentSession(llm=judge) as session:
        await session.start(DeliveryUpdate(pos))
        await let_the_opener_finish(session)
        result = await session.run(user_input="Honestly, just cancel the whole order.")
        result.expect.contains_function_call(name="cancel_order")
        assert not pos.order.cancelled
        await session.run(user_input="Yes, cancel it.")
    assert pos.order.cancelled and pos.due_now() == 0


async def test_dropping_an_order_thats_not_placed_waits_for_their_yes(judge):
    pos = Pizzeria()
    async with AgentSession(llm=judge) as session:
        await session.start(TakeOrder(pos))
        await session.run(
            user_input="Can I get a medium margherita and a large Coke for pickup?"
        )
        result = await session.run(
            user_input="Actually, never mind, cancel the whole thing."
        )
        result.expect.contains_function_call(name="cancel_order")
        assert len(pos.lines) == 2
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(judge, intent="Asks them to confirm dropping the whole order.")
        )
        await session.run(user_input="Yes, drop it.")
    assert pos.order is None and pos.lines == []


async def test_how_big_the_large_is_comes_from_the_menu(judge):
    async with AgentSession(llm=judge) as session:
        await session.start(TakeOrder(Pizzeria()))
        result = await session.run(user_input="How big is the large pizza?")
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(judge, intent="Says the large pizza is fourteen inches.")
        )


async def test_wings_with_a_sauce_we_dont_have_wait_for_the_callers_pick(judge):
    pos = Pizzeria()
    async with AgentSession(llm=judge) as session:
        await session.start(TakeOrder(pos))
        result = await session.run(
            user_input="Can I get a large pepperoni and a dozen wings with honey garlic "
            "sauce?"
        )
        assert sorted(ln.describe() for ln in pos.lines) == [
            "1 dozen Bone-in Wings",
            "1 large Pepperoni",
        ]
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge,
                intent="Says there's no honey garlic and asks which of its sauces they'd "
                "like, without choosing one for them.",
            )
        )
        await session.run(user_input="Lemon pepper, then.")
    assert sorted(ln.describe() for ln in pos.lines) == [
        "1 dozen Bone-in Wings, lemon pepper sauce",
        "1 large Pepperoni",
    ]


async def test_forget_it_after_a_missing_sauce_doesnt_end_the_call(judge):
    pos = Pizzeria()
    async with AgentSession(llm=judge) as session:
        await session.start(TakeOrder(pos))
        await session.run(
            user_input="Large pepperoni and a dozen bone-in wings with honey mustard. "
            "That's it, for pickup."
        )
        result = await session.run(user_input="Ugh, okay, forget it then.")
        hung_up = [
            e.item
            for e in result.events
            if e.item.type == "function_call_output"
            and e.item.name == "end_call"
            and not e.item.is_error
        ]
        assert not hung_up
    assert "1 large Pepperoni" in [ln.describe() for ln in pos.lines]


async def test_a_number_given_in_one_breath_is_saved_as_said(judge):
    pos = Pizzeria()
    async with AgentSession(llm=judge) as session:
        await session.start(TakeOrder(pos))
        await session.run(user_input="Can I get a large cheese pizza for pickup?")
        await session.run(user_input="That's it.")
        await session.run(
            user_input="Jamie Fox, 415-555-0167. And make it pepperoni on one half."
        )
    assert (pos.name, pos.phone) == ("Jamie Fox", "(415) 555-0167")


async def test_a_gate_code_is_said_back_digit_by_digit_but_written_as_numerals(judge):
    pos = Pizzeria.out_for_delivery("Sam Carter")
    async with AgentSession(llm=judge) as session:
        await session.start(DeliveryUpdate(pos))
        await let_the_opener_finish(session)
        await session.run(user_input="A large Coke is fine, thanks.")
        result = await session.run(
            user_input="Yes, the gate code is 4512, and I'm in unit 3B."
        )
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge,
                intent="Repeats the gate code back to them in words, as four five one two.",
            )
        )
    assert "4512" in pos.fulfillment.driver_note


async def test_the_changes_are_read_back_before_asking_for_anything_else(judge):
    pos = Pizzeria.out_for_delivery("Sam Carter")
    async with AgentSession(llm=judge) as session:
        await session.start(DeliveryUpdate(pos))
        await let_the_opener_finish(session)
        await session.run(user_input="A large Coke is fine, thanks.")
        result = await session.run(user_input="No, nothing for the driver.")
        result.expect.contains_function_call(name="read_back_changes")
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge,
                intent="Says the Sprite is now a Coke, that forty-nine sixty-five is still due "
                "at the door, and that Marco will be there in about twenty minutes.",
            )
        )
