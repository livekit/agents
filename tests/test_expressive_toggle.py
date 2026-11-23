from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, ExpressiveOptions, inference, tokenize, tts
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import is_given
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.generation import (
    EXPRESSIVE_INSTRUCTIONS_MESSAGE_ID,
    _strip_assistant_markup,
    remove_expressive_instructions,
    update_expressive_instructions,
)

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SESSION_TIMEOUT = 60


class _GeminiMarkupTTS(tts.TTS):
    """A TTS that declares a markup dialect but cannot lower it itself."""

    def __init__(self, *, streaming: bool = False) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=streaming),
            sample_rate=24000,
            num_channels=1,
        )

    class Markup(tts.TTS.Markup):
        def _provider_key(self) -> str:
            return "gemini"

    def synthesize(self, text, *, conn_options=DEFAULT_API_CONNECT_OPTIONS):  # type: ignore[override]
        raise NotImplementedError


# what an expressive turn leaves in history; square brackets are prose, not markup
MARKED_UP = (
    '<expr type="expression" label="happy"/> Welcome back! <sound value="chuckle"/> '
    "Glad you called again. Docs: [the guide](https://docs.livekit.io)."
)


def test_strip_assistant_markup() -> None:
    ctx = ChatContext.empty()
    ctx.add_message(role="assistant", content=MARKED_UP)
    ctx.add_message(role="user", content='I typed <expression value="happy"/> literally')
    plain = ctx.add_message(role="assistant", content="No tags here.")
    plain_content = plain.content

    _strip_assistant_markup(ctx)

    assistant_texts = [
        item.text_content
        for item in ctx.items
        if item.type == "message" and item.role == "assistant"
    ]
    assert "<expr" not in (assistant_texts[0] or "")
    assert "<sound" not in (assistant_texts[0] or "")
    assert "Welcome back!" in (assistant_texts[0] or "")
    assert "Glad you called again." in (assistant_texts[0] or "")
    # markdown links survive: brackets are prose, not markup
    assert "[the guide](https://docs.livekit.io)" in (assistant_texts[0] or "")

    # user content is never touched, tag-shaped or not
    user_text = next(
        item.text_content for item in ctx.items if item.type == "message" and item.role == "user"
    )
    assert '<expression value="happy"/>' in (user_text or "")

    # tag-free assistant content is left as-is (fast path)
    assert plain.content is plain_content


def test_update_and_remove_expressive_instructions() -> None:
    ctx = ChatContext.empty()
    update_expressive_instructions(ctx, text="markup guide v1")
    update_expressive_instructions(ctx, text="markup guide v2")

    guides = [item for item in ctx.items if item.id == EXPRESSIVE_INSTRUCTIONS_MESSAGE_ID]
    assert len(guides) == 1, "re-injection must replace, not stack"
    assert guides[0].text_content == "markup guide v2"

    remove_expressive_instructions(ctx)
    assert all(item.id != EXPRESSIVE_INSTRUCTIONS_MESSAGE_ID for item in ctx.items)


def test_update_options_expressive() -> None:
    session = AgentSession(expressive=True)
    assert session._expressive is True

    # turn off
    session.update_options(expressive=False)
    assert session._expressive is False

    # turn back on with options
    opts: ExpressiveOptions = {"tts_instructions_append": "Stay upbeat."}
    session.update_options(expressive=opts)
    assert session._expressive == opts

    # untouched when not given
    session.update_options()
    assert session._expressive == opts


def test_agent_expressive() -> None:
    # unset by default: falls back to the session value at runtime
    assert not is_given(Agent(instructions="test").expressive)

    agent = Agent(instructions="test", expressive=True)
    assert agent.expressive is True

    agent.update_options(expressive=False)
    assert agent.expressive is False

    opts: ExpressiveOptions = {"tts_instructions_append": "Stay upbeat."}
    agent.update_options(expressive=opts)
    assert agent.expressive == opts

    # untouched when not given
    agent.update_options()
    assert agent.expressive == opts


def test_agent_expressive_overrides_session() -> None:
    # constructed hermetically; fishaudio declares a markup dialect, which expressive requires
    tts = inference.TTS("fishaudio/s2.1-pro", api_key="fake", api_secret="fake")

    session_on = AgentSession(expressive=True, tts=tts)
    session_off = AgentSession(tts=tts)

    def resolves(agent: Agent, session: AgentSession) -> bool:
        return AgentActivity(agent, session)._resolve_expressive_options() is not None

    # agent unset: the session value applies
    assert resolves(Agent(instructions="test"), session_on)
    assert not resolves(Agent(instructions="test"), session_off)

    # agent value wins over the session in both directions
    assert not resolves(Agent(instructions="test", expressive=False), session_on)
    assert resolves(Agent(instructions="test", expressive=True), session_off)


async def test_expressive_off_turn_scrubs_history() -> None:
    """A turn with expressive off drops the markup guide and scrubs past assistant markup."""
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hello, how are you?", stt_delay=0.2)
    actions.add_llm("I'm doing well, thank you!", ttft=0.1, duration=0.3)
    actions.add_tts(2.0, ttfb=0.2, duration=0.3)

    # FakeTTS has no markup dialect, so expressive resolves to off despite the session option
    session = create_session(
        actions,
        extra_kwargs={"expressive": {"speech_steering": {"nonverbal_sounds": {"laughing": False}}}},
    )

    # history as if previous turns ran expressive
    seeded = ChatContext.empty()
    seeded.add_message(role="assistant", content=MARKED_UP)
    update_expressive_instructions(seeded, text="Use the formatting tags below…")
    agent = Agent(instructions="You are a helpful assistant.", chat_ctx=seeded)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    # the injected guide is gone
    assert all(item.id != EXPRESSIVE_INSTRUCTIONS_MESSAGE_ID for item in agent.chat_ctx.items)

    assistant_texts = [
        item.text_content
        for item in agent.chat_ctx.items
        if item.type == "message" and item.role == "assistant"
    ]
    assert assistant_texts, "expected assistant messages in history"
    for text in assistant_texts:
        assert "<expr" not in (text or "")
        assert "<sound" not in (text or "")
    # the seeded message's visible text survives the scrub, links included
    assert any("Welcome back!" in (t or "") for t in assistant_texts)
    assert any("[the guide](https://docs.livekit.io)" in (t or "") for t in assistant_texts)
    # and the new reply went through normally
    assert any("I'm doing well" in (t or "") for t in assistant_texts)


def test_expressive_needs_a_tts_the_framework_can_lower_for() -> None:
    """Declaring a dialect is not enough — something has to lower the markers."""

    def resolves(tts_obj: tts.TTS) -> bool:
        session = AgentSession(expressive=True, tts=tts_obj)
        return (
            AgentActivity(Agent(instructions="test"), session)._resolve_expressive_options()
            is not None
        )

    # non-streaming: the StreamAdapter the framework wraps it in does the lowering
    assert resolves(_GeminiMarkupTTS(streaming=False))
    # natively streaming: nothing in the framework can lower for it
    assert not resolves(_GeminiMarkupTTS(streaming=True))
    # the gateway streams too, but lowers inside its own stream
    assert resolves(inference.TTS("fishaudio/s2.1-pro", api_key="fake", api_secret="fake"))


def test_a_caller_supplied_stream_adapter_stays_expressive() -> None:
    """A StreamAdapter is streaming only at its surface; inside is the lowering path."""

    wrapped = _GeminiMarkupTTS()
    adapter = tts.StreamAdapter(tts=wrapped)
    assert adapter.capabilities.streaming

    session = AgentSession(expressive=True, tts=adapter)
    activity = AgentActivity(Agent(instructions="test"), session)
    assert activity._resolve_expressive_options() is not None

    # StreamAdapterWrapper reads the wrapped instance's flag, so it has to pass through
    adapter._set_expressive(True)
    assert wrapped._expressive


async def test_stream_adapter_snapshots_expressive_per_stream() -> None:
    """A stream keeps the expressive flag it started with, even if the shared TTS flips."""

    wrapped = _GeminiMarkupTTS()
    adapter = tts.StreamAdapter(tts=wrapped)

    adapter._set_expressive(True)
    stream = adapter.stream()
    adapter._set_expressive(False)  # a second turn, before this one's _run runs

    try:
        assert stream._expressive is True
        assert wrapped._expressive is False
    finally:
        await stream.aclose()
        await adapter.aclose()


async def test_stream_adapter_tokenizes_xml_aware_while_lowering() -> None:
    """A marker label containing a period stays one token while lowering."""
    marked = '<expr type="expression" label="Calm. Steady"/> All set.'

    async def tokens(tokenizer: tokenize.SentenceTokenizer) -> list[str]:
        stream = tokenizer.stream()
        stream.push_text(marked)
        stream.end_input()
        out = [ev.token async for ev in stream]
        await stream.aclose()
        return out

    adapter = tts.StreamAdapter(tts=_GeminiMarkupTTS())
    try:
        assert await tokens(adapter._tokenizer_for(lowering=True)) == [marked]
        # the default splits it mid-tag, which is why lowering needs its own
        assert await tokens(adapter._tokenizer_for(lowering=False)) != [marked]
        # one tokenizer, reused across syntheses
        assert adapter._tokenizer_for(lowering=True) is adapter._tokenizer_for(lowering=True)

        mine = tokenize.blingfire.SentenceTokenizer(retain_format=True)
        explicit = tts.StreamAdapter(tts=_GeminiMarkupTTS(), sentence_tokenizer=mine)
        assert explicit._tokenizer_for(lowering=True) is mine  # never second-guessed
        await explicit.aclose()
    finally:
        await adapter.aclose()
