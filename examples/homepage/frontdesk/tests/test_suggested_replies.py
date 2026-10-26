import asyncio
import json
from datetime import date
from types import SimpleNamespace

from livekit.agents import ConversationItemAddedEvent, llm

from clinic import Clinic
from receptionist import Caller, CallState, caller_script
from suggested_replies import SUGGESTIONS_ATTRIBUTE, SuggestedReplies

MONDAY = date(2026, 9, 28)
BIRTHDAY_LINE = {
    "label": "My birthday",
    "value": "March fourteenth, nineteen eighty-eight.",
}


def test_script_moves_the_seeded_appointment_to_the_fourth_clinic_day():
    script = caller_script(Clinic(today=MONDAY), CallState())

    assert "Taylor Smith" in script and "March 14, 1988" in script
    assert "Knee pain follow-up" in script
    assert "move their appointment to Friday morning" in script
    assert "Alex Morgan" in script and "A L E X, M O R G A N" in script


async def test_script_follows_the_call_from_verified_to_moved():
    clinic = Clinic(today=MONDAY)
    state = CallState(caller=Caller("Taylor Smith", date(1988, 3, 14)))

    assert "take the first one" in caller_script(clinic, state)

    friday = clinic.clinic_days()[3]
    opening = (await clinic.check_availability(friday, friday, "morning"))[0]
    await clinic.reschedule("Taylor Smith", clinic.appointments[0].id, opening)

    assert "goodbye" in caller_script(clinic, state)


class FakeSession:
    def __init__(self) -> None:
        self.user_state = "listening"
        self.history = llm.ChatContext()
        self.handlers: dict = {}

    def on(self, event, handler) -> None:
        self.handlers[event] = handler


class FakeRoom:
    def __init__(self) -> None:
        self.local_participant = self
        self.lines: list[list[dict]] = []

    async def set_attributes(self, attributes: dict[str, str]) -> None:
        self.lines.append(json.loads(attributes[SUGGESTIONS_ATTRIBUTE]))


class FakeLLM:
    def __init__(self, reply: dict) -> None:
        self.reply = json.dumps(reply)
        self.calls = 0

    def chat(self, *, chat_ctx):
        self.calls += 1
        return FakeStream(self.reply)


class FakeStream:
    def __init__(self, text: str) -> None:
        self._text = text

    async def __aiter__(self):
        yield SimpleNamespace(delta=SimpleNamespace(content=self._text))

    async def aclose(self) -> None:
        pass


def attached() -> tuple[FakeSession, FakeRoom, FakeLLM]:
    session, room = FakeSession(), FakeRoom()
    model = FakeLLM({"suggestions": [BIRTHDAY_LINE]})
    SuggestedReplies(
        session, room, llm=model, persona="Riley", script=lambda: "script"
    ).attach()
    return session, room, model


async def agent_says(session: FakeSession, line: str, *, interrupted=False) -> None:
    item = llm.ChatMessage(role="assistant", content=[line], interrupted=interrupted)
    session.handlers["conversation_item_added"](ConversationItemAddedEvent(item=item))
    await asyncio.gather(*(asyncio.all_tasks() - {asyncio.current_task()}))


async def test_a_new_question_replaces_the_line():
    session, room, _ = attached()

    await agent_says(session, "Sure, what's your date of birth?")

    assert room.lines == [[], [BIRTHDAY_LINE]]


async def test_the_line_stays_up_while_the_caller_talks_over_the_agent():
    session, room, model = attached()
    session.user_state = "speaking"

    await agent_says(session, "Mm-hm.")

    assert room.lines == []
    assert model.calls == 0


async def test_an_interrupted_reply_leaves_the_line_up():
    session, room, model = attached()

    await agent_says(session, "And what's your", interrupted=True)

    assert room.lines == []
    assert model.calls == 0
