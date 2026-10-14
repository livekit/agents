from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable

import pytest
from agent import NickAgent, RepairCallAgent, Userdata
from backend import ServiceDesk, demo_desk

from livekit.agents import AgentSession, inference, llm
from livekit.agents.voice.run_result import RunResult

pytestmark = pytest.mark.evals

TODAY = dt.date(2026, 9, 29)  # a Tuesday
X3_VIN = "WBX57DP0XPN123456"
SEDAN_VIN = "WBA5R1C05KA765432"
CONFIRMS = [
    "Yes, that's all correct, go ahead and book it.",
    "Yes, book it please.",
    "Yes, that's everything, please book it.",
]


def _main_llm() -> llm.LLM:
    # the model the agent runs on
    return inference.LLM("openai/gpt-4.1-mini")


async def _talk(turns: list[str], desk: ServiceDesk) -> list[str]:
    """Play the caller's turns, then confirm until it's booked. Returns what Nick said."""
    said: list[str] = []
    async with (
        _main_llm() as model,
        AgentSession(llm=model, userdata=Userdata(desk=desk)) as sess,
    ):
        await sess.start(NickAgent(desk))
        for text in [*turns, *CONFIRMS]:
            if desk.appointment and text in CONFIRMS:
                break
            result: RunResult[None] = await sess.run(user_input=text)
            said += [
                e.item.text_content or ""
                for e in result.events
                if e.type == "message" and e.item.role == "assistant"
            ]
    # printed on failure: what Nick said
    print("\n".join(said))
    return said


@pytest.mark.asyncio
async def test_oil_change_with_recall_no_loaner_and_a_day_correction() -> None:
    desk = demo_desk(today=TODAY)
    said = await _talk(
        [
            "Hi, I need to book an oil change.",
            "Alex Morgan.",
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
            "I'm Alex Morgan, my number is 415 555 0142, and it's the X3.",
            "Okay, let's book the diagnostic then.",
            "Oh, actually, sorry, it's the 330i, not the X3.",
            "Thursday, whatever's earliest, and I'll just wait there.",
        ],
        desk,
    )
    assert desk.quotes and desk.quotes[0].price is None, said
    # the only prices Nick may ever say are the menu's
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


async def _say(
    turns: list[str], desk: ServiceDesk, *, outbound: bool = False, calls: list[str] | None = None
) -> list[str]:
    """Play the caller's turns as they are. Returns what Nick said; `calls` collects the
    tools he called."""
    said: list[str] = []
    async with (
        _main_llm() as model,
        AgentSession(llm=model, userdata=Userdata(desk=desk)) as sess,
    ):
        await sess.start(RepairCallAgent(desk) if outbound else NickAgent(desk))
        for text in turns:
            result: RunResult[None] = await sess.run(user_input=text)
            said += [
                e.item.text_content or ""
                for e in result.events
                if e.type == "message" and e.item.role == "assistant"
            ]
            if calls is not None:
                calls += [e.item.name for e in result.events if e.type == "function_call"]
    print("\n".join(said))
    return said


@pytest.mark.asyncio
async def test_already_booked_is_looked_up_and_moved_not_booked_twice() -> None:
    desk = demo_desk(today=TODAY)
    said = await _say(
        [
            "Hi, I think I'm already booked in for a tire rotation, can you check?",
            "Alex Morgan, 415 555 0142.",
            "Could we move it to Thursday morning instead?",
            "Nine works.",
            "Yes, move it.",
        ],
        desk,
    )
    e = desk.on_file
    assert e is not None and e.status == "booked", said
    assert e.slot_id == "thu-0900" and e.moved_from == "Tuesday October 6, 2:30 PM", said
    assert desk.appointment is None, said
    assert not any("connect you" in line.lower() for line in said), said


@pytest.mark.asyncio
async def test_cancelling_the_appointment_on_file_waits_for_a_clear_yes() -> None:
    desk = demo_desk(today=TODAY)
    said = await _say(
        [
            "Hi, it's Alex Morgan, 415 555 0142. I need to cancel my appointment.",
            "Hmm, hang on, I'm not sure yet.",
        ],
        desk,
    )
    # Alex's appointment, whether or not his file was opened
    on_file = desk._customers[0].on_file
    assert on_file is not None and on_file.status == "booked", said
    desk = demo_desk(today=TODAY)
    said = await _say(
        [
            "Hi, it's Alex Morgan, 415 555 0142. I need to cancel my appointment.",
            "Yes, cancel it please.",
        ],
        desk,
    )
    assert desk.on_file is not None and desk.on_file.status == "cancelled", said


@pytest.mark.asyncio
async def test_payment_is_in_person_and_a_service_is_added_after_booking() -> None:
    desk = demo_desk(today=TODAY)
    async with _Call(desk) as call:
        await call.say("Hi, it's Alex Morgan, 415 555 0142. I need an oil change on my X3.")
        await call.until(
            lambda: desk.appointment,
            "No recall this time. Wednesday at 7:30 works, I'll wait in the lounge, and if "
            "you've read it all back, yes, that's correct, book it.",
            tries=6,
        )
        booked = desk.appointment
        assert booked is not None, call.said
        await call.say("Can I just pay by card now, over the phone?")
        # paid in person when they pick it up: never over the phone
        answer = call.said[-1].lower()
        assert "pick" in answer or "service desk" in answer or "in person" in answer, call.said
        await call.say("Oh, and can you add a tire rotation to that?")
        await call.until(
            lambda: desk.appointment and len(desk.appointment.services) == 2,
            "Yes, that's right.",
        )
    appt = desk.appointment
    assert appt is not None and appt.status == "booked", call.said
    assert appt.services == ["Oil change", "Tire rotation"], call.said
    assert appt.confirmation == booked.confirmation, call.said


class _Call:
    """One call with Nick (or the follow-up), played turn by turn."""

    def __init__(self, desk: ServiceDesk, *, outbound: bool = False) -> None:
        self.desk, self.outbound = desk, outbound
        self.said: list[str] = []
        self.calls: list[str] = []

    async def __aenter__(self) -> _Call:
        self._model = _main_llm()
        self._sess = AgentSession(llm=self._model, userdata=Userdata(desk=self.desk))
        await self._sess.__aenter__()
        agent = RepairCallAgent(self.desk) if self.outbound else NickAgent(self.desk)
        await self._sess.start(agent)
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._sess.__aexit__(None, None, None)
        await self._model.aclose()
        print("\n".join(self.said))

    async def say(self, *turns: str) -> None:
        for text in turns:
            result: RunResult[None] = await self._sess.run(user_input=text)
            for e in result.events:
                if e.type == "message" and e.item.role == "assistant":
                    self.said.append(e.item.text_content or "")
                elif e.type == "function_call":
                    self.calls.append(e.item.name)

    async def until(self, done: Callable[[], object], reply: str, tries: int = 4) -> None:
        """Answers with `reply` until `done()` holds, a few times at most."""
        for _ in range(tries):
            if done():
                return
            await self.say(reply)


# ---- what the edge-case simulations found ----


@pytest.mark.asyncio
async def test_smoke_means_dont_drive_it_before_any_recall_pitch() -> None:
    desk = demo_desk(today=TODAY)
    said = await _say(
        [
            "Hi, it's Alex Morgan, 415 555 0142. There's smoke coming from under the hood of "
            "my X3. Can you book me in next week?",
        ],
        desk,
    )
    text = " ".join(said).lower()
    assert "tow" in text and ("not drive" in text or "n't drive" in text), said
    # the warning comes before any recall pitch
    assert "recall" not in text or text.index("tow") < text.index("recall"), said
    assert desk.stop_driving() is not None, said


@pytest.mark.asyncio
async def test_grinding_brakes_are_never_a_warranty_repair() -> None:
    desk = demo_desk(today=TODAY)
    said = await _say(
        [
            "Hi, it's Alex Morgan, 415 555 0142. My X3 grinds when I brake.",
            "It's from the front, it started a couple of weeks ago, and no warning lights.",
            "It's under warranty, so that's covered, right?",
        ],
        desk,
    )
    assert "warranty_repair" not in desk.draft.services, said


_REFUSALS = ("match", "account holder", "can't find", "unable to")


@pytest.mark.asyncio
async def test_someone_elses_name_on_a_known_number_gets_a_new_file() -> None:
    desk = demo_desk(today=TODAY)
    said = await _say(
        [
            "Hi, I need to check on a car.",
            "Jordan Morgan.",
            "415 555 0142.",
        ],
        desk,
    )
    text = " ".join(said)
    for secret in ("Alex", "X3", "330i", "recall", "21,400", "tire rotation"):
        assert secret not in text, said
    assert not any(r in text.lower() for r in _REFUSALS), said
    c = desk.customer
    assert c is not None and c.new and c.name == "Jordan Morgan" and c.vehicles == [], said


@pytest.mark.asyncio
async def test_an_unknown_caller_books_a_diagnostic_for_a_car_they_add() -> None:
    desk = demo_desk(today=TODAY)
    async with _Call(desk) as call:
        await call.say(
            "Hi, my car's been clunking over bumps, can I get it looked at?",
            "It's Dana Zhou.",
            "437 465 7465.",
        )
        await call.until(
            lambda: desk.draft_vehicle(),
            "It's a 2021 BMW X5, with about 40,000 miles on it.",
        )
        await call.until(
            lambda: "diagnostic" in desk.draft.services,
            "It's a clunk from the front right, only over bumps, for about a week. No warning "
            "lights, and nothing recent. Yes, book the diagnostic.",
        )
        await call.until(
            lambda: desk.draft.read_back_revision is not None,
            "Whatever's earliest on Thursday, and I'll take the shuttle.",
        )
        await call.until(lambda: desk.appointment, "Yes, that's right, book it.", tries=3)
    text = " ".join(call.said)
    assert not any(r in text.lower() for r in _REFUSALS), call.said
    for secret in ("Alex", "X3", "330i", "tire rotation"):
        assert secret not in text, call.said
    c = desk.customer
    assert c is not None and c.new and c.phone == "4374657465", call.said
    assert [v.label for v in c.vehicles] == ["2021 BMW X5"], call.said
    assert c.vehicles[0].miles == 40_000 and not c.vehicles[0].in_warranty(TODAY)
    appt = desk.appointment
    assert appt is not None and appt.vehicle == "2021 BMW X5", call.said
    assert appt.services == ["Diagnostic"] and appt.slot_id.startswith("thu"), call.said
    assert appt.transport == "shuttle" and appt.vin in desk.recalls_checked, call.said
    assert "add_vehicle" in call.calls, call.calls


@pytest.mark.asyncio
async def test_a_ride_is_asked_not_picked() -> None:
    desk = demo_desk(today=TODAY)
    calls: list[str] = []
    said = await _say(
        [
            "Hi, it's Alex Morgan, 415 555 0142. I need an oil change on the 330i.",
            "Wednesday at 7:30 is fine.",
            "I'll need transportation.",
        ],
        desk,
        calls=calls,
    )
    assert desk.draft.transport is None, (said, calls)


@pytest.mark.asyncio
async def test_a_yes_with_a_question_still_books_and_the_limit_is_noted() -> None:
    desk = demo_desk(today=TODAY)
    async with _Call(desk) as call:
        await call.say(
            "Hi, it's Alex Morgan, 415 555 0142. My 330i has a check engine light, can you "
            "look at it?",
            "It's steady, came on yesterday, no noises. Don't do anything over $500 without "
            "calling me.",
            "Thursday at 9, and I'll wait in the lounge.",
        )
        await call.until(
            lambda: desk.draft.read_back_revision is not None,
            "Thursday at 9 a.m., and I'll wait in the lounge. What have you got down?",
        )
        assert desk.draft.read_back_revision is not None, call.said
        await call.say("Yes, that's right. Do you have wifi in the lounge?")
    appt = desk.appointment
    assert appt is not None and appt.approval_limit == 500, call.said
    assert appt.slot_id == "thu-0900", call.said


@pytest.mark.asyncio
async def test_the_3_is_asked_about() -> None:
    desk = demo_desk(today=TODAY)
    said = await _say(["Hi, it's Alex Morgan, 415 555 0142. I need an oil change on my 3."], desk)
    assert desk.draft.vin is None, said
    assert "X3" in said[-1] and "330i" in said[-1], said


@pytest.mark.asyncio
async def test_follow_up_says_the_new_total_and_a_friend_collects() -> None:
    desk = demo_desk(today=TODAY, direction="outbound")
    calls: list[str] = []
    said = await _say(
        [
            "Speaking.",
            "Yes, go ahead with the front brakes. So what's the total now?",
            "Since the last time you worked on it there's been a new rattle, by the way.",
            "My friend Sam will pick up the car for me.",
        ],
        desk,
        outbound=True,
        calls=calls,
    )
    assert desk.visit is not None and desk.visit.decisions == {"front_brakes": "approved"}
    assert any("$648" in line for line in said), said
    assert desk.visit.concerns, (said, calls)
    assert desk.visit.collected_by and "authorized" not in " ".join(said).lower(), said
    assert "pickup_summary" in calls and desk.visit.ended_early is None, calls
