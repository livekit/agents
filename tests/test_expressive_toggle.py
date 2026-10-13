from __future__ import annotations

import pytest

from livekit.agents import Agent, ExpressiveOptions, LiveSession, inference, tokenize, tts
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

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SESSION_TIMEOUT = 60


# what an expressive turn actually leaves in history: the expr markers the LLM emitted,
# plus (defensively) a hallucinated native tag. Square brackets are *not* markup here —
# they reach history as prose or markdown links, so the scrub must leave them alone.
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


def test_expressive_param_defaults_off() -> None:
    assert LiveSession()._expressive is False
    assert LiveSession(expressive=True)._expressive is True

    opts: ExpressiveOptions = {"tts_instructions_append": "Stay upbeat."}
    assert LiveSession(expressive=opts)._expressive == opts


def test_update_options_expressive() -> None:
    session = LiveSession(expressive=True)
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

    session_on = LiveSession(expressive=True, tts=tts)
    session_off = LiveSession(tts=tts)

    def resolves(agent: Agent, session: LiveSession) -> bool:
        return AgentActivity(agent, session)._resolve_expressive_options() is not None

    # agent unset: the session value applies
    assert resolves(Agent(instructions="test"), session_on)
    assert not resolves(Agent(instructions="test"), session_off)

    # agent value wins over the session in both directions
    assert not resolves(Agent(instructions="test", expressive=False), session_on)
    assert resolves(Agent(instructions="test", expressive=True), session_off)


def test_expressive_needs_a_tts_the_framework_can_lower_for() -> None:
    """Declaring a dialect is not enough — something has to lower the markers."""

    class _Declaring(tts.TTS):
        def __init__(self, *, streaming: bool) -> None:
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

    def resolves(tts_obj: tts.TTS) -> bool:
        session = LiveSession(expressive=True, tts=tts_obj)
        return (
            AgentActivity(Agent(instructions="test"), session)._resolve_expressive_options()
            is not None
        )

    # non-streaming: the StreamAdapter the framework wraps it in does the lowering
    assert resolves(_Declaring(streaming=False))
    # natively streaming: nothing in the framework can lower for it
    assert not resolves(_Declaring(streaming=True))
    # the gateway streams too, but lowers inside its own stream
    assert resolves(inference.TTS("fishaudio/s2.1-pro", api_key="fake", api_secret="fake"))


def test_a_caller_supplied_stream_adapter_stays_expressive() -> None:
    """A StreamAdapter is streaming only at its surface; inside is the lowering path."""

    class _NonStreaming(tts.TTS):
        def __init__(self) -> None:
            super().__init__(
                capabilities=tts.TTSCapabilities(streaming=False),
                sample_rate=24000,
                num_channels=1,
            )

        class Markup(tts.TTS.Markup):
            def _provider_key(self) -> str:
                return "gemini"

        def synthesize(self, text, *, conn_options=DEFAULT_API_CONNECT_OPTIONS):  # type: ignore[override]
            raise NotImplementedError

    wrapped = _NonStreaming()
    adapter = tts.StreamAdapter(tts=wrapped)
    assert adapter.capabilities.streaming  # what the old guard rejected it for

    session = LiveSession(expressive=True, tts=adapter)
    activity = AgentActivity(Agent(instructions="test"), session)
    assert activity._resolve_expressive_options() is not None

    # StreamAdapterWrapper reads the wrapped instance's flag, so it has to pass through
    adapter._set_expressive(True)
    assert wrapped._expressive


@pytest.mark.asyncio
async def test_stream_adapter_snapshots_expressive_per_stream() -> None:
    """_expressive lives on the shared TTS, but a stream is one synthesis.

    The pipeline sets the flag synchronously before stream(); _run happens later in its
    own task, so another turn or session sharing the TTS could flip it in the gap and
    send that turn's markers through unlowered.
    """

    class _NonStreaming(tts.TTS):
        def __init__(self) -> None:
            super().__init__(
                capabilities=tts.TTSCapabilities(streaming=False),
                sample_rate=24000,
                num_channels=1,
            )

        class Markup(tts.TTS.Markup):
            def _provider_key(self) -> str:
                return "gemini"

        def synthesize(self, text, *, conn_options=DEFAULT_API_CONNECT_OPTIONS):  # type: ignore[override]
            raise NotImplementedError

    wrapped = _NonStreaming()
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


@pytest.mark.asyncio
async def test_stream_adapter_tokenizes_xml_aware_while_lowering() -> None:
    """A marker split across two tokens would be lowered as two halves.

    Labels are free-form English and may contain a period, which an unguarded sentence
    tokenizer treats as a boundary. The framework passes an xml-aware tokenizer when it
    builds the adapter; a caller using the default has to get one too.
    """
    marked = '<expr type="expression" label="Calm. Steady"/> All set.'

    async def tokens(tokenizer: tokenize.SentenceTokenizer) -> list[str]:
        stream = tokenizer.stream()
        stream.push_text(marked)
        stream.end_input()
        out = [ev.token async for ev in stream]
        await stream.aclose()
        return out

    class _NonStreaming(tts.TTS):
        def __init__(self) -> None:
            super().__init__(
                capabilities=tts.TTSCapabilities(streaming=False),
                sample_rate=24000,
                num_channels=1,
            )

        class Markup(tts.TTS.Markup):
            def _provider_key(self) -> str:
                return "gemini"

        def synthesize(self, text, *, conn_options=DEFAULT_API_CONNECT_OPTIONS):  # type: ignore[override]
            raise NotImplementedError

    adapter = tts.StreamAdapter(tts=_NonStreaming())
    try:
        assert await tokens(adapter._tokenizer_for(lowering=True)) == [marked]
        # the default splits it mid-tag, which is why lowering needs its own
        assert await tokens(adapter._tokenizer_for(lowering=False)) != [marked]
        # one tokenizer, reused across syntheses
        assert adapter._tokenizer_for(lowering=True) is adapter._tokenizer_for(lowering=True)

        mine = tokenize.blingfire.SentenceTokenizer(retain_format=True)
        explicit = tts.StreamAdapter(tts=_NonStreaming(), sentence_tokenizer=mine)
        assert explicit._tokenizer_for(lowering=True) is mine  # never second-guessed
        await explicit.aclose()
    finally:
        await adapter.aclose()
