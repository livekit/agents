from __future__ import annotations

import datetime as dt
import re

import pytest
from agent import LeoAgent, RepairCallAgent, Userdata
from backend import ServiceDesk, demo_desk

from livekit.agents import AgentSession, inference, llm
from livekit.agents.voice.run_result import RunResult

pytestmark = pytest.mark.evals

TODAY = dt.date(2026, 9, 29)  # a Tuesday
X3_VIN = "WBX57DP0XPN123456"
SEDAN_VIN = "WBA5R1C05KA765432"
CONFIRMS = ["Yes, that's all correct, go ahead and book it.", "Yes, book it please."]


def _main_llm() -> llm.LLM:
    # the model the agent runs on
    return inference.LLM("openai/gpt-4.1-mini")


async def _talk(turns: list[str], desk: ServiceDesk) -> list[str]:
    """Play the caller's turns, then confirm until it's booked. Returns what Leo said."""
    said: list[str] = []
    async with (
        _main_llm() as model,
        AgentSession(llm=model, userdata=Userdata(desk=desk)) as sess,
    ):
        await sess.start(LeoAgent(desk))
        for text in [*turns, *CONFIRMS]:
            if desk.appointment and text in CONFIRMS:
                break
            result: RunResult[None] = await sess.run(user_input=text)
            said += [
                e.item.text_content or ""
                for e in result.events
                if e.type == "message" and e.item.role == "assistant"
            ]
    # printed on failure: what Leo said
    print("\n".join(said))
    return said


@pytest.mark.asyncio
async def test_oil_change_with_recall_no_loaner_and_a_day_correction() -> None:
    desk = demo_desk(today=TODAY)
    said = await _talk(
        [
            "Hi, I need to book an oil change.",
            "Sure, it's 415 555 0142.",
            "It's for the X3.",
            "Yeah, go ahead and add the recall.",
            "Can I get a loaner while it's in?",
            "Okay, fine, the shuttle then.",
            "Tuesday at eight works.",
            "No wait, sorry, make it Wednesday, first thing in the morning.",
        ],
        desk,
    )
    appt = desk.appointment
    assert appt is not None, said
    assert appt.vin == X3_VIN
    assert appt.services == ["Oil change"]
    assert (appt.recall_id, appt.recall) == ("23V-418", "added")
    assert appt.slot_id == "wed-0730"
    assert appt.transport == "shuttle"


@pytest.mark.asyncio
async def test_brake_job_price_then_switches_cars() -> None:
    desk = demo_desk(today=TODAY)
    said = await _talk(
        [
            "Hi, how much is a brake job? My brakes are grinding when I stop.",
            "My number is 415 555 0142, and it's the X3.",
            "Okay, let's book the diagnostic then.",
            "Oh, actually, sorry, it's the 330i, not the X3.",
            "Thursday, whatever's earliest, and I'll just wait there.",
        ],
        desk,
    )
    assert desk.quotes and desk.quotes[0].price is None, said
    # the only prices Leo may ever say are the menu's
    for line in said:
        for amount in re.findall(r"\$\s?([\d,]+)", line):
            assert amount.replace(",", "") in {"159", "79", "39", "0"}, line
    appt = desk.appointment
    assert appt is not None, said
    assert appt.vin == SEDAN_VIN
    assert appt.services == ["Diagnostic"]
    assert SEDAN_VIN in desk.recalls_checked
    assert appt.slot_id.startswith("thu")
    assert appt.transport == "wait"


@pytest.mark.asyncio
async def test_follow_up_approves_the_front_brakes() -> None:
    desk = demo_desk(today=TODAY, direction="outbound")
    async with (
        _main_llm() as model,
        AgentSession(llm=model, userdata=Userdata(desk=desk)) as sess,
    ):
        await sess.start(RepairCallAgent(desk))
        result: RunResult[None] = await sess.run(user_input="Speaking.")
        result.expect.contains_function_call(name="confirm_customer")

        result = await sess.run(user_input="Yes, go ahead with the front brakes.")
        result.expect.contains_function_call(name="approve_repair")
    assert desk.visit is not None
    assert desk.visit.decisions == {"front_brakes": "approved"}
    assert desk.ticket_total() == 648
