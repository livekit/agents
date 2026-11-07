import asyncio
import json
from types import SimpleNamespace

from livekit.agents import ConversationItemAddedEvent, llm

from suggested_replies import SUGGESTIONS_ATTRIBUTE, SuggestedReplies

CAR_LINE = {"label": "The X3", "value": "It's the X3."}


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
    model = FakeLLM({"suggestions": [CAR_LINE]})
    SuggestedReplies(
        session, room, llm=model, persona="Nick", script=lambda: "script"
    ).attach()
    return session, room, model


async def agent_says(session: FakeSession, line: str, *, interrupted=False) -> None:
    item = llm.ChatMessage(role="assistant", content=[line], interrupted=interrupted)
    session.handlers["conversation_item_added"](ConversationItemAddedEvent(item=item))
    await asyncio.gather(*(asyncio.all_tasks() - {asyncio.current_task()}))


async def test_a_new_question_replaces_the_line():
    session, room, _ = attached()

    await agent_says(session, "Which car is it for?")

    assert room.lines == [[], [CAR_LINE]]


async def test_the_line_stays_up_while_the_caller_talks_over_the_agent():
    session, room, model = attached()
    session.user_state = "speaking"

    await agent_says(session, "Mm-hmm.")

    assert room.lines == []
    assert model.calls == 0


async def test_an_interrupted_reply_leaves_the_line_up():
    session, room, model = attached()

    await agent_says(session, "Which car is", interrupted=True)

    assert room.lines == []
    assert model.calls == 0
