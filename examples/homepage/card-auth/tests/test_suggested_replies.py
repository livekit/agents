import asyncio
import json
from types import SimpleNamespace

from livekit.agents import ConversationItemAddedEvent, llm

from bank import Bank, Identity
from card_services import SAMPLE_BIRTH_DATE, SAMPLE_NAME, caller_script
from suggested_replies import SUGGESTIONS_ATTRIBUTE, SuggestedReplies, parse_suggestions

CARD_LINE = {"label": "My card", "value": "It ends in one one one one."}


def test_parses_the_models_json():
    raw = '{"suggestions": [{"label": "My name", "value": "It\'s Jordan Lee."}]}'

    assert parse_suggestions(raw) == [{"label": "My name", "value": "It's Jordan Lee."}]


def test_tolerates_fences_and_prose_around_the_json():
    raw = 'Sure:\n```json\n{"suggestions": [{"label": "Yes", "value": "Yes, that\'s right."}]}\n```'

    assert parse_suggestions(raw) == [{"label": "Yes", "value": "Yes, that's right."}]


def test_drops_anything_that_is_not_a_labelled_line():
    raw = (
        '{"suggestions": [{"label": "", "value": "x"}, {"value": "no label"}, "text",'
        ' {"label": " Ok ", "value": " Okay. "}]}'
    )

    assert parse_suggestions(raw) == [{"label": "Ok", "value": "Okay."}]


def test_caps_the_list_and_survives_garbage():
    four = [{"label": str(i), "value": f"line {i}"} for i in range(4)]

    assert len(parse_suggestions(str(four).replace("'", '"'))) == 3
    assert parse_suggestions("I don't know") == []


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
    model = FakeLLM({"suggestions": [CARD_LINE]})
    SuggestedReplies(
        session, room, llm=model, persona="Avery", script=lambda: "script"
    ).attach()
    return session, room, model


async def agent_says(session: FakeSession, line: str, *, interrupted=False) -> None:
    item = llm.ChatMessage(role="assistant", content=[line], interrupted=interrupted)
    session.handlers["conversation_item_added"](ConversationItemAddedEvent(item=item))
    await asyncio.gather(*(asyncio.all_tasks() - {asyncio.current_task()}))


async def test_a_new_question_replaces_the_line():
    session, room, _ = attached()

    await agent_says(session, "And the last four digits of your card?")

    assert room.lines == [[], [CARD_LINE]]


async def test_the_line_stays_up_while_the_caller_talks_over_the_agent():
    session, room, model = attached()
    session.user_state = "speaking"

    await agent_says(session, "Mm-hmm.")

    assert room.lines == []
    assert model.calls == 0


async def test_an_interrupted_reply_leaves_the_line_up():
    session, room, model = attached()

    await agent_says(session, "And the last four", interrupted=True)

    assert room.lines == []
    assert model.calls == 0


async def test_script_walks_from_the_reason_for_calling_to_goodbye():
    bank = Bank()

    first = caller_script(bank)
    assert SAMPLE_NAME in first and "April 22, 1990" in first
    assert "one one one one" in first
    assert "don't recognize" in first

    def caller(last_four: str) -> Identity:
        return Identity(SAMPLE_NAME, SAMPLE_BIRTH_DATE, last_four)

    await bank.authenticate(caller("0000"))
    assert "didn't go through" in caller_script(bank)

    await bank.authenticate(caller("1111"))
    assert "verified" in caller_script(bank)

    await bank.recent_charges()
    assert "didn't make it" in caller_script(bank)

    await bank.record_answer(made_by_cardholder=False)
    assert "goodbye" in caller_script(bank)


async def test_script_says_goodbye_once_the_caller_is_locked_out():
    bank = Bank()
    for _ in range(3):
        await bank.authenticate(Identity(SAMPLE_NAME, SAMPLE_BIRTH_DATE, "0000"))

    assert "can't verify them" in caller_script(bank)
