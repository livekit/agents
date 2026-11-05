"""Scripted example conversations exercise the real handlers and persisted business state."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from livekit.agents import voice as voice_agent
from livekit.agents.storage.session import ContextStore
from livekit.agents.voice.agent_run import RunHooks, reset_run_hooks, set_run_hooks

from .test_text_session import Reply, Script, call

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.fixture
def load_example(monkeypatch: pytest.MonkeyPatch) -> Callable[[str, Script], ModuleType]:
    def source_module(name: str, path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return module

    def load(name: str, model: Script) -> ModuleType:
        directory = Path(__file__).parents[1] / "examples" / name
        monkeypatch.setenv("FRONTDESK_NOW", "2026-06-12T09:00")
        monkeypatch.setenv("HOTEL_TODAY", "2026-06-12")
        monkeypatch.delenv("CAL_API_KEY", raising=False)
        monkeypatch.setattr(voice_agent, "llm_from_model_string", lambda _: model)
        dependency = {"frontdesk": "calendar_api", "hotel-receptionist": "hotel"}[name]
        source_module(dependency, directory / f"{dependency}.py")
        return source_module(
            f"example_{name.replace('-', '_')}_backend", directory / "backend_agent.py"
        )

    return load


async def test_frontdesk_booking_approval_and_saved_email_survive_handler_restart(
    load_example: Callable[[str, Script], ModuleType],
) -> None:
    model = Script()
    example = load_example("frontdesk", model)
    slot, next_slot = example.cal.slots[:2]
    model.replies.extend(
        [
            call("schedule_appointment", {"slot_id": slot.id}, "missing-email"),
            Reply("What email address should the confirmation go to?"),
            call(
                "schedule_appointment",
                {"slot_id": slot.id, "email": "patient@example.com"},
                "approval",
            ),
            call(
                "schedule_appointment",
                {"slot_id": slot.id, "email": "patient@example.com"},
                "book",
            ),
            Reply("Booked."),
            call("schedule_appointment", {"slot_id": next_slot.id}, "saved-email"),
        ]
    )
    scopes = []
    history = []
    for message in ("Book the first opening.", "patient@example.com", "yes", "Book another."):
        store = ContextStore(scopes=scopes)
        token = set_run_hooks(RunHooks(store=store))
        try:
            result = await example.text.handler(
                SimpleNamespace(message=message, message_history=history, store=store)
            )
            scopes = [write.scope for write in store._scope_writes]
            history = result.all_messages()
        finally:
            reset_run_hooks(token)
            store._close()
        if message == "Book the first opening.":
            assert result.status == "input_required" and "email" in result.question
            assert example.cal.appointments == []
        elif message == "patient@example.com":
            assert result.status == "input_required" and "Please confirm" in result.question
            assert example.cal.appointments == []
        elif message == "yes":
            assert result.status == "completed" and result.output == "Booked."
            assert [(a.slot.id, a.email) for a in example.cal.appointments] == [
                (slot.id, "patient@example.com")
            ], result.new_messages()
            assert slot not in example.cal.slots
        else:
            assert result.status == "input_required"
            assert "Please confirm" in result.question and "patient@example.com" in result.question
            assert len(example.cal.appointments) == 1
    assert not model.replies


async def test_hotel_verification_and_cancellation_resume_after_handler_restart(
    load_example: Callable[[str, Script], ModuleType],
) -> None:
    model = Script(
        call("lookup_booking", {}, "unverified"),
        Reply("What are the last name and confirmation code on the booking?"),
        call("verify_booking", {"last_name": "Smith", "code": "HTL-AB12"}, "verify"),
        Reply("Verified."),
        call("lookup_booking", {}, "lookup"),
        Reply("Your booking is confirmed."),
        call("cancel_room_booking", {}, "approval"),
        call("cancel_room_booking", {}, "cancel"),
        Reply("Cancelled."),
    )
    example = load_example("hotel-receptionist", model)
    booking = example.hotel.find(last_name="Smith", code="HTL-AB12")
    assert booking is not None
    scopes = []
    history = []
    for message in ("My booking?", "Smith HTL-AB12", "Check it again.", "Cancel it.", "yes"):
        store = ContextStore(scopes=scopes)
        token = set_run_hooks(RunHooks(store=store))
        try:
            result = await example.text.handler(
                SimpleNamespace(message=message, message_history=history, store=store)
            )
            scopes = [write.scope for write in store._scope_writes]
            history = result.all_messages()
        finally:
            reset_run_hooks(token)
            store._close()
        if message == "My booking?":
            assert result.status == "input_required" and "confirmation code" in result.question
        elif message == "Check it again.":
            assert result.status == "completed", result.new_messages()
            output = next(i for i in result.new_messages() if i.type == "function_call_output")
            assert "HTL-AB12" in output.output and "confirmed" in output.output
        elif message == "Cancel it.":
            assert result.status == "input_required" and "cancel HTL-AB12" in result.question
        elif message == "yes":
            assert result.status == "completed" and result.output == "Cancelled."
            assert booking.status == "cancelled"
        if message != "yes":
            assert booking.status == "confirmed"
    assert not model.replies
