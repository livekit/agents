"""Turns against the real agent on Gemma, in text. The desk is in memory, so nothing is mocked:
every assertion reads what the agent did to it."""

import json
from datetime import date

import pytest
from livekit.agents import AgentSession, inference

from advisor import RepairCallback, ServiceAdvisor
from service_desk import ServiceDesk

MONDAY = date(2026, 10, 19)
MODEL = "google/gemma-4-31b-it"


async def let_the_opener_finish(session: AgentSession) -> None:
    """The caller answers once they've heard Nick's opener, not over it."""
    if speech := session.current_speech:
        await speech.wait_for_playout()


@pytest.fixture
async def judge():
    async with inference.LLM(model=MODEL) as llm:
        yield llm


async def test_a_visit_is_booked_only_in_a_turn_after_its_read_back(judge):
    desk = ServiceDesk(MONDAY)
    read_back_turn = booked_turn = None
    async with AgentSession(llm=judge) as session:
        await session.start(ServiceAdvisor(desk))
        for turn, line in enumerate(
            (
                "Hi, this is Alex Morgan, and my number is 415-555-0142.",
                "It's my X3. It's been grinding from the front when I brake, for about a "
                "week now. No warning lights, and nothing's been done to it lately.",
                "Yeah, a diagnostic is fine, and please add the recall too.",
                "Wednesday the twenty-first, first thing in the morning.",
                "I'll take the shuttle.",
                "That's everything, go ahead and book it.",
                "Yes, that's all right.",
            )
        ):
            result = await session.run(user_input=line)
            for event in result.events:
                item = event.item
                if (
                    item.type == "function_call_output"
                    and item.name == "book_visit"
                    and "readBack" in json.loads(item.output)
                    and read_back_turn is None
                ):
                    read_back_turn = turn
            if desk.visit.booked is not None:
                booked_turn = turn
                break

    assert read_back_turn is not None and booked_turn is not None
    assert booked_turn > read_back_turn
    booked = desk.visit.booked
    assert booked is not None
    assert booked.slot.id == "2026-10-21T07:30"
    assert booked.services == ["diagnostic", "recall"]
    assert booked.ride == "shuttle"


async def test_failing_brakes_are_flagged_unsafe_right_away(judge):
    desk = ServiceDesk(MONDAY)
    async with AgentSession(llm=judge) as session:
        await session.start(ServiceAdvisor(desk))
        result = await session.run(
            user_input="My brake pedal goes all the way to the floor and the car barely stops."
        )
        result.expect.contains_function_call(name="flag_unsafe_to_drive")
        assert desk.visit.problem.unsafe == "brakes_failing"
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge,
                intent="Tells the caller the car isn't safe to drive: not to drive it, or to have it towed in.",
            )
        )


async def test_a_repair_off_the_menu_gets_no_made_up_price(judge):
    desk = ServiceDesk(MONDAY)
    async with AgentSession(llm=judge) as session:
        await session.start(ServiceAdvisor(desk))
        result = await session.run(user_input="How much would a brake job run me?")
        result.expect.contains_function_call(
            name="quote_price", arguments={"item": "something_else"}
        )
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge,
                intent="Doesn't give a price for the brake job itself. It may say the repair is "
                "quoted after a diagnostic, and may give the diagnostic's fee.",
            )
        )


async def test_a_yes_to_the_findings_approves_the_front_brakes(judge):
    desk = ServiceDesk.for_callback(MONDAY)
    async with AgentSession(llm=judge) as session:
        await session.start(RepairCallback(desk))
        await let_the_opener_finish(session)
        result = await session.run(user_input="Yes, this is Alex.")
        result.expect.contains_function_call(name="go_over_findings")
        assert desk.shop is not None and not desk.shop.decisions
        await session.run(user_input="Yes, go ahead with it.")
        assert desk.shop.decisions.get("front_brakes") == "approved"


async def test_a_wrong_number_hears_nothing_about_the_car(judge):
    desk = ServiceDesk.for_callback(MONDAY)
    async with AgentSession(llm=judge) as session:
        await session.start(RepairCallback(desk))
        await let_the_opener_finish(session)
        result = await session.run(
            user_input="Sorry, I think you have the wrong number."
        )
        result.expect.contains_function_call(
            name="end_without_update", arguments={"reason": "wrong_person"}
        )
        assert desk.shop is not None and not desk.shop.decisions
        assert not desk.shop.findings_shared
