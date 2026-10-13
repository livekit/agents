"""say() text uses the same markup rules as an expressive LLM reply.

The TTS gets the marked-up text: ``<expr/>`` markers lowered to its dialect (or dropped
when nothing can speak them), native markup passed through as-is. The transcript is
always plain. Chat history keeps ``<expr/>`` only while expressive is active, and never
keeps the TTS's native markup.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable

import pytest

from livekit.agents import Agent, tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from .fake_io import FakeTextOutput
from .fake_session import FakeActions, create_session, run_session
from .fake_tts import FakeChunkedStream, FakeSynthesizeStream, FakeTTS

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SESSION_TIMEOUT = 60

BREAK = 'Hi <expr type="break" label="1s"/> there.'


class _MarkupTTS(FakeTTS):
    """A fake TTS speaking the *key* dialect that records what it was sent."""

    def __init__(
        self, key: str = "", *, streaming: bool = False, audio_duration: float = 0.3
    ) -> None:
        super().__init__(fake_audio_duration=audio_duration)
        self._capabilities = tts.TTSCapabilities(streaming=streaming)
        self.key = key
        self.inputs: list[str] = []

    class Markup(tts.TTS.Markup):
        def _provider_key(self) -> str:
            assert isinstance(self._tts, _MarkupTTS)
            return self._tts.key

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> FakeChunkedStream:
        self.inputs.append(text)
        return super().synthesize(text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> FakeSynthesizeStream:
        return _RecordingStream(tts=self, conn_options=conn_options)


class _RecordingStream(FakeSynthesizeStream):
    def push_text(self, token: str) -> None:
        assert isinstance(self._tts, _MarkupTTS)
        self._tts.inputs.append(token)
        super().push_text(token)


async def _say(
    text: str | AsyncIterable[str], tts_obj: _MarkupTTS, *, expressive: bool = False
) -> tuple[str, str]:
    """Run one say(); return what the room transcript showed and what history stored."""

    class _SayAgent(Agent):
        async def on_enter(self) -> None:
            self.session.say(text)

    session = create_session(FakeActions(), with_stt=False, extra_kwargs={"expressive": expressive})
    text_output = session.output.transcription
    assert text_output is not None and isinstance(text_output.next_in_chain, FakeTextOutput)
    fake_output = text_output.next_in_chain

    agent = _SayAgent(instructions="test", tts=tts_obj)
    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)
    transcript = "".join(fake_output._messages)

    messages = [
        item for item in agent.chat_ctx.items if item.type == "message" and item.role == "assistant"
    ]
    assert len(messages) == 1
    return transcript, messages[0].raw_text_content or ""


async def test_expr_lowered_for_the_tts_and_stripped_from_history() -> None:
    tts_obj = _MarkupTTS("ssml")
    transcript, history = await _say(BREAK, tts_obj)

    assert "".join(tts_obj.inputs) == 'Hi <break time="1000ms"/> there.'
    assert transcript == "Hi there."
    assert history == "Hi there."


async def test_expr_dropped_for_a_tts_without_a_dialect() -> None:
    tts_obj = _MarkupTTS("")
    transcript, history = await _say(BREAK, tts_obj)

    assert "".join(tts_obj.inputs) == "Hi there."
    assert transcript == "Hi there."
    assert history == "Hi there."


async def test_expr_dropped_for_a_natively_streaming_tts() -> None:
    # nothing in the framework lowers for it, so a marker would be read aloud
    tts_obj = _MarkupTTS("cartesia", streaming=True)
    transcript, history = await _say(BREAK, tts_obj)

    assert "<expr" not in "".join(tts_obj.inputs)
    assert "".join(tts_obj.inputs).split() == ["Hi", "there."]
    assert history == "Hi there."


async def test_expr_kept_in_history_while_expressive() -> None:
    tts_obj = _MarkupTTS("cartesia")
    marked = '<expr type="expression" label="happy"/> Welcome back!'
    transcript, history = await _say(marked, tts_obj, expressive=True)

    assert "".join(tts_obj.inputs) == '<emotion value="happy"/> Welcome back!'
    # left for the room output, which strips it and surfaces lk.expression, as it does
    # for an expressive LLM reply
    assert transcript == marked
    # the LLM is taught expr, so its own past markers are useful examples
    assert history == marked


async def test_native_markup_reaches_the_tts_but_not_history() -> None:
    tts_obj = _MarkupTTS("ssml")
    marked = 'Say <phoneme alphabet="ipa" ph="t">tomato</phoneme> <break time="1s"/>now.'
    transcript, history = await _say(marked, tts_obj, expressive=True)

    assert "".join(tts_obj.inputs) == marked
    assert transcript == "Say tomato now."
    assert history == "Say tomato now."


async def test_bracket_cues_stripped_for_a_bracket_dialect() -> None:
    tts_obj = _MarkupTTS("eleven_v3")
    transcript, history = await _say("[whispers] Hello there.", tts_obj)

    assert "".join(tts_obj.inputs) == "[whispers] Hello there."
    assert transcript == "Hello there."
    assert history == "Hello there."


async def test_brackets_are_prose_for_other_dialects() -> None:
    tts_obj = _MarkupTTS("ssml")
    _, history = await _say("Option [A] is fine.", tts_obj)
    assert history == "Option [A] is fine."


async def test_streamed_say_with_a_marker_split_across_chunks() -> None:
    async def chunks() -> AsyncIterable[str]:
        for chunk in ["Hi <expr type=", '"break" label="1s"/>', " there."]:
            yield chunk

    tts_obj = _MarkupTTS("ssml")
    transcript, history = await _say(chunks(), tts_obj)

    assert "".join(tts_obj.inputs) == 'Hi <break time="1000ms"/> there.'
    assert transcript == "Hi there."
    assert history == "Hi there."


async def test_marker_spanning_sentences_is_one_request() -> None:
    # a non-streaming SSML TTS gets one request per sentence; a scope split across two
    # would be malformed for the provider
    tts_obj = _MarkupTTS("ssml")
    marked = '<prosody rate="slow">First sentence here. Second sentence here.</prosody>'
    _, history = await _say(marked, tts_obj)

    assert tts_obj.inputs == [marked]
    assert history == "First sentence here. Second sentence here."


async def test_plain_text_unchanged() -> None:
    tts_obj = _MarkupTTS("ssml")
    plain = "Hello there. How are you today?"
    transcript, history = await _say(plain, tts_obj)

    assert "".join(tts_obj.inputs).split() == plain.split()
    assert transcript == plain
    assert history == plain


async def test_interrupted_say_stores_plain_text() -> None:
    marked = (
        '<prosody rate="slow">I saw your application</prosody> <break time="1s"/>for a Honda '
        "and wanted to follow up on it with you today."
    )

    class _SayAgent(Agent):
        async def on_enter(self) -> None:
            self.session.say(marked)

    actions = FakeActions()
    actions.add_user_speech(1.0, 1.2, "Who is this?")
    actions.add_llm("This is John.")
    actions.add_tts(1.0)
    session = create_session(actions)
    agent = _SayAgent(instructions="test", tts=_MarkupTTS("ssml", audio_duration=10.0))
    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    said = next(
        item for item in agent.chat_ctx.items if item.type == "message" and item.role == "assistant"
    )
    assert said.interrupted is True
    assert said.raw_text_content
    assert "<" not in said.raw_text_content
    assert said.raw_text_content.startswith("I saw")


async def test_held_aligned_words_keep_their_start_time() -> None:
    # a tag split inside itself across aligned chunks holds the earlier words until it
    # closes; they must not take the timing of the chunk that closed it
    from types import SimpleNamespace

    from livekit.agents.types import TimedString
    from livekit.agents.voice.agent_activity import AgentActivity

    activity = SimpleNamespace(tts=_MarkupTTS("ssml"), _resolve_expressive_options=lambda: None)

    async def aligned() -> AsyncIterable[str]:
        yield TimedString("Hello <prosody rate=", start_time=0.0, end_time=0.5)
        yield TimedString('"slow">world</prosody> ', start_time=2.0, end_time=2.5)
        yield TimedString("again.", start_time=3.0, end_time=3.5)

    out = [c async for c in AgentActivity._strip_say_markup(activity, aligned())]  # type: ignore[arg-type]

    assert "".join(out) == "Hello world again."
    assert isinstance(out[0], TimedString)
    assert (out[0].start_time, out[0].end_time) == (0.0, 2.5)
    assert isinstance(out[-1], TimedString) and out[-1].start_time == 3.0
