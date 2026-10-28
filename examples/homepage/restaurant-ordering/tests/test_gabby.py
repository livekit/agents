"""Gabby's two calls without a model: which tools each has, and what the tools that don't depend
on the conversation hand back."""

import json
from types import SimpleNamespace

import pytest
from livekit.agents import llm
from livekit.agents.llm import ToolError

from gabby import DeliveryUpdate, TakeOrder, delivery_opener
from pizzeria import Pizzeria


def tools(agent):
    return sorted(t.id for t in agent.tools)


def test_each_call_has_only_its_own_tools():
    assert tools(TakeOrder(Pizzeria())) == [
        "add_items",
        "cancel_order",
        "check_allergen",
        "end_call",
        "place_order",
        "remove_item",
        "set_contact",
        "set_delivery",
        "set_pickup",
        "set_tip",
        "update_item",
    ]
    assert tools(DeliveryUpdate(Pizzeria.out_for_delivery(None))) == [
        "cancel_order",
        "end_call",
        "end_without_update",
        "read_back_changes",
        "set_breadsticks",
        "set_driver_note",
        "set_tip",
        "swap_sold_out_item",
        "take_off_sold_out_item",
    ]


async def test_a_tip_comes_back_with_the_total_or_whats_due_at_the_door():
    pos = Pizzeria()
    await pos.add_items([{"item_id": "pepperoni", "size": "large"}])
    taking = json.loads(await TakeOrder(pos).set_tip(None, dollars=5))
    assert (taking["tip"], taking["total"]) == ("$5.00", "$26.79")

    out = Pizzeria.out_for_delivery("Sam Carter")
    update = json.loads(await DeliveryUpdate(out).set_tip(None, dollars=5))
    assert update == {
        "dueAtTheDoor": "$54.65",
        "comparedToBefore": "$5.00 more than before",
        "tip": "$5.00",
    }


def test_the_opener_uses_the_customers_name_only_when_its_known():
    assert delivery_opener(Pizzeria.out_for_delivery("Sam Carter")).startswith(
        "Hi Sam, this is"
    )
    assert delivery_opener(Pizzeria.out_for_delivery(None)).startswith("Hi, this is")


class FakeContext:
    def __init__(self, call_id: str, items: list[llm.ChatItem]) -> None:
        self.function_call = llm.FunctionCall(
            call_id=call_id, name="end_call", arguments="{}"
        )
        self.session = SimpleNamespace(history=SimpleNamespace(items=items))
        self.speech_handle = SimpleNamespace(add_done_callback=lambda cb: None)

    async def wait_for_playout(self) -> None:
        pass


def said(role: str, text: str, *, interrupted: bool = False) -> llm.ChatMessage:
    return llm.ChatMessage(role=role, content=[text], interrupted=interrupted)


def tool_output(
    call_id: str, name: str, *, is_error: bool = False
) -> llm.FunctionCallOutput:
    return llm.FunctionCallOutput(
        call_id=call_id, name=name, output="{}", is_error=is_error
    )


async def test_a_cut_off_confirmation_is_replayed_before_hanging_up():
    agent = TakeOrder(Pizzeria())
    agent._must_tell(FakeContext("call_done", []), "their order number")
    items: list[llm.ChatItem] = [
        tool_output("call_done", "place_order"),
        said("assistant", "You're order two fifteen, ready in", interrupted=True),
        said("user", "Sorry, what was that?"),
        said("assistant", "Order two fifteen, ready in twenty minutes."),
        said("user", "Great, bye."),
        said("assistant", "Bye!"),
    ]
    with pytest.raises(ToolError, match="first tell them their order number"):
        await agent.end_call(FakeContext("call_end", items))
    items += [
        tool_output("call_end", "end_call", is_error=True),
        said("assistant", "It's order two fifteen, ready in twenty minutes. Bye!"),
    ]
    assert await agent.end_call(FakeContext("call_end_again", items)) is None
