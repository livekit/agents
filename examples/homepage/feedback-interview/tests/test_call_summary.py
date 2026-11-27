import json
from dataclasses import dataclass, field

from livekit.agents import llm

from call_summary import SUMMARY_ATTRIBUTE, CallSummary, markdown_list


@dataclass
class FakeParticipant:
    attributes: dict[str, str] = field(default_factory=dict)

    async def set_attributes(self, attributes: dict[str, str]) -> None:
        self.attributes.update(attributes)


@dataclass
class FakeRoom:
    local_participant: FakeParticipant = field(default_factory=FakeParticipant)


@dataclass
class FakeSession:
    history: llm.ChatContext = field(default_factory=llm.ChatContext.empty)

    def on(self, event, callback):
        pass


class FakeReply:
    def __init__(self, text: str) -> None:
        self.text = text

    async def collect(self):
        return self


class FakeLLM:
    def __init__(self, reply: dict) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def chat(self, *, chat_ctx, response_format):
        self.prompts.append(chat_ctx.items[-1].text_content)
        return FakeReply(json.dumps(self.reply))


def summary_for(
    *turns: tuple[str, str], reply: dict
) -> tuple[CallSummary, FakeRoom, FakeLLM]:
    session = FakeSession()
    for role, text in turns:
        session.history.add_message(role=role, content=text)
    room, model = FakeRoom(), FakeLLM(reply)
    return CallSummary(session, room, llm=model), room, model


async def test_finish_publishes_name_role_and_summary():
    summary, room, model = summary_for(
        ("assistant", "Hi, I'm Morgan. Who are you?"),
        ("user", "I'm Shane, I run a foot clinic. We went with Vapi."),
        reply={
            "name": "Shane",
            "role": None,
            "points": ["Runs a foot clinic.", "Chose Vapi."],
        },
    )
    await summary.finish()
    published = json.loads(room.local_participant.attributes[SUMMARY_ATTRIBUTE])
    assert published == {
        "name": "Shane",
        "role": None,
        "summary": "- Runs a foot clinic.\n- Chose Vapi.",
    }
    assert "Interviewee: I'm Shane" in model.prompts[0]


async def test_nothing_is_published_before_the_interviewee_speaks():
    summary, room, model = summary_for(
        ("assistant", "Hi, I'm Morgan. Who are you?"),
        reply={"name": None, "role": None, "points": []},
    )
    await summary.finish()
    assert room.local_participant.attributes == {} and model.prompts == []


def test_points_become_one_markdown_bullet_each():
    points = ["- Already a bullet.", "Spans\ntwo lines.", "  ", "* Starred."]
    assert (
        markdown_list(points) == "- Already a bullet.\n- Spans two lines.\n- Starred."
    )
