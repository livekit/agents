"""The examples' text endpoints served as ``lk agent console`` serves them, on scripted models:
persisted state across requests and the business rules their tools enforce."""

from __future__ import annotations

import json
import sys
from collections import deque
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import BaseModel

from livekit.agents import Agent, llm
from tests.fakes import FakeLLM, Reply, last_input, tool_call

from .conftest import EXAMPLES, ChatWith

LoadExample = Callable[[str, FakeLLM], ModuleType]


def script(model: FakeLLM, replies: list[Reply]) -> None:
    """Queues ``replies`` once the example (and the ids it seeds) is loaded."""
    assert isinstance(model.replies, deque)
    model.replies.extend(replies)


def conversation(model: FakeLLM, request: int) -> list[tuple[str, str | None]]:
    """The user and assistant messages the model was given in its ``request``-th request."""
    items = model.requests[request].items
    return [(i.role, i.text_content) for i in items if i.type == "message" and i.role != "system"]


@pytest.mark.parametrize("name", ["frontdesk", "hotel-receptionist", "airline-support"])
async def test_the_frontend_continues_its_text_conversation_on_the_next_request(
    name: str,
    load_example: LoadExample,
    source_module: Callable[[str, Path], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    chat_with: ChatWith,
) -> None:
    """A second message on the frontend's thread is answered with the first exchange in view."""
    model = FakeLLM(["Hello.", "Welcome back."])
    backend = load_example(name, model)
    monkeypatch.setitem(sys.modules, "backend_agent", backend)
    frontend = source_module(f"{name}_frontend", EXAMPLES / name / "frontend_agent.py")
    ask = await chat_with("frontend", frontend.front_desk, backend.back_office)

    await ask("Hi.")
    reply = await ask("I'm back.")

    assert reply.text == "Welcome back."
    assert conversation(model, 1) == [
        ("user", "Hi."),
        ("assistant", "Hello."),
        ("user", "I'm back."),
    ]


async def test_frontdesk_books_only_after_the_email_and_the_read_back_are_confirmed(
    load_example: LoadExample, chat_with: ChatWith
) -> None:
    """A booking asks for the email, reads the terms back, and books once on yes."""
    model = FakeLLM()
    example = load_example("frontdesk", model)
    slot = example.cal.slots[0]
    script(
        model,
        [
            tool_call("schedule_appointment", slot_id=slot.id),
            "What email address should the confirmation go to?",
            tool_call("schedule_appointment", slot_id=slot.id, email="patient@example.com"),
            tool_call("schedule_appointment", slot_id=slot.id, email="patient@example.com"),
            "Booked.",
        ],
    )
    ask = await chat_with("backend", example.back_office)

    asked_email = await ask("Book the first opening.")
    read_back = await ask("patient@example.com")
    booked_before_yes = list(example.cal.appointments)
    done = await ask("yes")

    assert (asked_email.state, read_back.state, done.state) == (
        "input-required",
        "input-required",
        "completed",
    )
    assert "Please confirm" in read_back.text
    assert booked_before_yes == []
    assert [(a.slot.id, a.email) for a in example.cal.appointments] == [
        (slot.id, "patient@example.com")
    ]


async def test_frontdesk_reads_back_the_saved_email_for_the_next_booking(
    load_example: LoadExample, chat_with: ChatWith
) -> None:
    """An email given for one booking is saved and offered in the next booking's read-back."""
    model = FakeLLM()
    example = load_example("frontdesk", model)
    slot, next_slot = example.cal.slots[:2]
    script(
        model,
        [
            tool_call("schedule_appointment", slot_id=slot.id, email="patient@example.com"),
            tool_call("schedule_appointment", slot_id=slot.id, email="patient@example.com"),
            "Booked.",
            tool_call("schedule_appointment", slot_id=next_slot.id),
        ],
    )
    ask = await chat_with("backend", example.back_office)
    await ask("Book the first opening for patient@example.com.")
    await ask("yes")

    read_back = await ask("Book another.")

    assert read_back.state == "input-required"
    assert "patient@example.com" in read_back.text


async def test_hotel_shows_and_cancels_a_booking_only_once_verified_and_confirmed(
    load_example: LoadExample, chat_with: ChatWith
) -> None:
    """A guest is verified before the booking is shown, and cancelling it needs a yes."""
    model = FakeLLM(
        [
            tool_call("lookup_booking"),
            "What are the last name and confirmation code on the booking?",
            tool_call("verify_booking", last_name="Smith", code="HTL-AB12"),
            "Verified.",
            tool_call("lookup_booking"),
            "Your booking is confirmed.",
            tool_call("cancel_room_booking"),
            tool_call("cancel_room_booking"),
            "Cancelled.",
        ]
    )
    example = load_example("hotel-receptionist", model)
    booking = example.hotel.find(last_name="Smith", code="HTL-AB12")
    ask = await chat_with("backend", example.back_office)

    unverified = await ask("My booking?")
    await ask("Smith HTL-AB12")
    await ask("Check it again.")
    shown = last_input(model.requests[-1]) or ""
    read_back = await ask("Cancel it.")
    status_before_yes = booking.status
    await ask("yes")

    assert unverified.state == "input-required"
    assert "HTL-AB12" in shown and "confirmed" in shown
    assert read_back.state == "input-required" and "cancel HTL-AB12" in read_back.text
    assert status_before_yes == "confirmed"
    assert booking.status == "cancelled"


# -- airline support: the back office triages an email through its specialist agents


def outputs(*replies: str | BaseModel) -> FakeLLM:
    """A specialist's model: a pydantic model is its final_result, a str its reply."""
    return FakeLLM(
        tool_call("final_result", **reply.model_dump(mode="json"))
        if isinstance(reply, BaseModel)
        else reply
        for reply in replies
    )


def specialists(
    airline: ModuleType,
    triage: BaseModel,
    drafts: tuple[str, ...] = (),
    verdicts: tuple[BaseModel, ...] = (),
) -> None:
    """Scripts the classifier, the reply writer and the reviewer the triage tool calls."""
    classifier = Agent(
        instructions="classify", llm=outputs(triage), output_type=airline.Classification
    )
    writer = Agent(instructions="draft", llm=outputs(*drafts), output_type=str)
    reviewer = Agent(instructions="review", llm=outputs(*verdicts), output_type=airline.Verdict)
    vars(airline).update(classifier=classifier, writer=writer, reviewer=reviewer)


def triage(sender: str, subject: str, body: str) -> llm.FunctionToolCall:
    return tool_call("triage", email={"sender": sender, "subject": subject, "body": body})


def reported(model: FakeLLM) -> dict[str, object]:
    """The TriageResult the back office's last tool call returned to its model."""
    result: dict[str, object] = json.loads(last_input(model.requests[-1]) or "{}")
    return result


DELAY = ("ana.garcia@example.com", "SK204 late", "Booking SKY4F2.")
LONG_HAUL = ("marcus.lee@example.com", "SK880 late", "SKY7Q9.")
NO_REF = ("tom.okafor@example.com", "Lisbon", "You cancelled SK517.")
SPAM = ("deals@cheap-meds.example", "You WON", "Click here.")


async def test_an_eu261_delay_is_paid_after_an_approved_reply(
    load_example: LoadExample, chat_with: ChatWith
) -> None:
    """An approved reply to a delay claim pays the EU261 amount once."""
    model = FakeLLM([triage(*DELAY), "Done."])
    airline = load_example("airline-support", model)
    specialists(
        airline,
        airline.Classification(kind="delay_or_cancellation", booking_ref="SKY4F2"),
        ("EUR 250.",),
        (airline.Verdict(approved=True),),
    )
    ask = await chat_with("backend", airline.back_office)

    await ask("A customer email.")

    result = reported(model)
    assert (result["action"], result["paid_eur"], result["reply_draft"]) == (
        "resolved",
        250,
        "EUR 250.",
    )
    assert [(p.booking_ref, p.eur) for p in airline.skyline.payments] == [("SKY4F2", 250)]


async def test_an_earlier_payment_is_taken_off(
    load_example: LoadExample, chat_with: ChatWith
) -> None:
    """A booking already compensated is not paid again."""
    model = FakeLLM([triage(*DELAY), "Done."])
    airline = load_example("airline-support", model)
    airline.skyline.pay("SKY4F2", 250, "paid before")
    specialists(
        airline,
        airline.Classification(kind="delay_or_cancellation", booking_ref="SKY4F2"),
        ("Paid.",),
        (airline.Verdict(approved=True),),
    )
    ask = await chat_with("backend", airline.back_office)

    await ask("A customer email.")

    assert (reported(model)["action"], reported(model)["paid_eur"]) == ("resolved", 0)
    assert len(airline.skyline.payments) == 1


async def test_a_claim_over_the_limit_goes_to_a_person_unpaid(
    load_example: LoadExample, chat_with: ChatWith
) -> None:
    """A long-haul claim above the agent's payment limit is handed to a person, unpaid."""
    model = FakeLLM([triage(*LONG_HAUL), "Done."])
    airline = load_example("airline-support", model)
    specialists(
        airline,
        airline.Classification(kind="delay_or_cancellation", booking_ref="SKY7Q9"),
        ("EUR 600.",),
        (airline.Verdict(approved=True),),
    )
    ask = await chat_with("backend", airline.back_office)

    await ask("A customer email.")

    result = reported(model)
    assert (result["action"], result["priority"], result["paid_eur"]) == ("to_human", "high", 0)
    assert airline.skyline.payments == []


async def test_a_reply_failing_review_twice_goes_to_a_person_unpaid(
    load_example: LoadExample, chat_with: ChatWith
) -> None:
    """Two drafts the reviewer rejects hand the claim to a person, unpaid."""
    model = FakeLLM([triage(*DELAY), "Done."])
    airline = load_example("airline-support", model)
    specialists(
        airline,
        airline.Classification(kind="delay_or_cancellation", booking_ref="SKY4F2"),
        ("EUR 600.", "EUR 400."),
        (
            airline.Verdict(approved=False, objection="not 600"),
            airline.Verdict(approved=False, objection="not 400"),
        ),
    )
    ask = await chat_with("backend", airline.back_office)

    await ask("A customer email.")

    assert (reported(model)["action"], reported(model)["paid_eur"]) == ("to_human", 0)
    assert airline.skyline.payments == []


async def test_spam_closes_without_a_reply(load_example: LoadExample, chat_with: ChatWith) -> None:
    """Spam is closed with no reply drafted and nothing paid."""
    model = FakeLLM([triage(*SPAM), "Done."])
    airline = load_example("airline-support", model)
    specialists(airline, airline.Classification(kind="spam"))
    ask = await chat_with("backend", airline.back_office)

    await ask("A customer email.")

    result = reported(model)
    assert (result["action"], result["reply_draft"], result["paid_eur"]) == ("close", None, 0)


async def test_a_missing_booking_reference_is_asked_and_the_claim_continues(
    load_example: LoadExample, chat_with: ChatWith
) -> None:
    """A claim without a booking reference asks for one, refuses someone else's, then pays."""
    model = FakeLLM(
        [
            triage(*NO_REF),
            "Your booking reference?",
            tool_call("continue_request", booking_ref="SKY4F2"),
            "Your booking reference?",
            tool_call("continue_request", booking_ref="SKY5C1"),
            "Paid.",
        ]
    )
    airline = load_example("airline-support", model)
    specialists(
        airline,
        airline.Classification(kind="delay_or_cancellation"),
        ("EUR 400.",),
        (airline.Verdict(approved=True),),
    )
    ask = await chat_with("backend", airline.back_office)

    asked = await ask("A customer email.")
    question = last_input(model.requests[-1]) or ""
    someone_elses = await ask("SKY4F2")
    asked_again = last_input(model.requests[-1]) or ""
    done = await ask("SKY5C1")

    assert (asked.state, someone_elses.state, done.state) == (
        "input-required",
        "input-required",
        "completed",
    )
    assert "booking reference" in question and "booking reference" in asked_again
    assert [(p.booking_ref, p.eur) for p in airline.skyline.payments] == [("SKY5C1", 400)]
