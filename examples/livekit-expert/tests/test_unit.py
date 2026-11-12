from collections.abc import AsyncIterator
from typing import Any, Literal

import behaviors
import pytest
from knowledge import LOOKUP_DESCRIPTION, PRODUCTS, lookup_product

from livekit.agents import (
    ChatMessage,
    ConversationItemAddedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    VoiceSession,
)

pytestmark = pytest.mark.unit


def test_lookup_tool_lists_every_product() -> None:
    assert set(PRODUCTS) == {
        "agent-builder",
        "agent-observability",
        "agents-on-livekit-cloud",
        "livekit-inference",
        "livekit-phone-numbers",
        "platform",
    }
    assert lookup_product.info.description == LOOKUP_DESCRIPTION
    for name, product in PRODUCTS.items():
        assert f"- {name}: {product.summary}" in LOOKUP_DESCRIPTION
        assert product.body


@pytest.mark.parametrize(
    "chunks,spoken",
    [
        (["I love LiveKit."], "I love Lyve Kit."),
        (["livekit is great"], "Lyve Kit is great"),
        (["Live", "Kit rocks"], "Lyve Kit rocks"),
        (["LiveKit's SDK"], "Lyve Kit's SDK"),
        (["LiveKit’s SDK"], "Lyve Kit's SDK"),
        (["LiveKitten is not a product"], "LiveKitten is not a product"),
    ],
)
async def test_livekit_is_respelled_for_the_tts(chunks: list[str], spoken: str) -> None:
    async def text() -> AsyncIterator[str]:
        for chunk in chunks:
            yield chunk

    assert "".join([word async for word in behaviors.respell_livekit(text())]) == spoken


async def test_checks_in_once_when_the_user_goes_away() -> None:
    session = VoiceSession()
    replies: list[dict[str, Any]] = []
    session.generate_reply = lambda **kwargs: replies.append(kwargs)  # type: ignore[method-assign,assignment]
    behaviors.check_in_when_user_away(session)

    session.emit(
        "user_state_changed", UserStateChangedEvent(old_state="listening", new_state="away")
    )
    session.emit(
        "user_state_changed", UserStateChangedEvent(old_state="away", new_state="speaking")
    )

    assert replies == [{"instructions": behaviors.CHECK_IN, "allow_interruptions": True}]


class _Job:
    def __init__(self) -> None:
        self.rooms_deleted = 0

    def delete_room(self) -> None:
        self.rooms_deleted += 1


def _caller(monkeypatch: pytest.MonkeyPatch) -> tuple[VoiceSession[Any], _Job]:
    session = VoiceSession()
    session.interrupt = lambda **kwargs: None  # type: ignore[method-assign,assignment]
    job = _Job()
    monkeypatch.setattr(behaviors, "get_job_context", lambda: job)
    behaviors.disconnect_scripted_callers(session)
    return session, job


def _say(session: VoiceSession[Any], role: Literal["user", "assistant"], text: str) -> None:
    item = ChatMessage(role=role, content=[text])
    session.emit("conversation_item_added", ConversationItemAddedEvent(item=item))


@pytest.mark.parametrize(
    "opening",
    [
        "What is one plus one?",
        "  what IS the speed of light.  ",
        "What do you think a healthy lifestyle includes? "
        "Please limit your answers to about forty words.",
    ],
)
async def test_scripted_opening_hangs_up(monkeypatch: pytest.MonkeyPatch, opening: str) -> None:
    session, job = _caller(monkeypatch)
    for _ in range(5):
        _say(session, "assistant", "How familiar are you with LiveKit?")
    _say(session, "user", opening)
    _say(session, "user", opening)

    assert job.rooms_deleted == 1


@pytest.mark.parametrize(
    "turns",
    [
        ["Hello?", "Does LiveKit support SIP and what does it cost?", "what is two plus two"],
        ["Hello?", "I have a question.", "About pricing.", "What is one plus one?"],
    ],
    ids=["real-questions", "scripted-line-after-the-opening"],
)
async def test_real_callers_stay_on(monkeypatch: pytest.MonkeyPatch, turns: list[str]) -> None:
    session, job = _caller(monkeypatch)
    for turn in turns:
        _say(session, "user", turn)

    assert job.rooms_deleted == 0


async def test_only_whole_turns_match(monkeypatch: pytest.MonkeyPatch) -> None:
    session, job = _caller(monkeypatch)
    segments = [
        "Please list a few environmental actions",
        "that you think help protect the earth.",
        "Please limit your answers to about forty words.",
    ]
    for segment in segments:
        event = UserInputTranscribedEvent(transcript=segment, is_final=True)
        session.emit("user_input_transcribed", event)
    assert job.rooms_deleted == 0

    _say(session, "user", " ".join(segments))
    assert job.rooms_deleted == 1
