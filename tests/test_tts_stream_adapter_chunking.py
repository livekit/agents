from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from livekit.agents import APIConnectOptions, tokenize, tts
from livekit.agents.voice.agent import Agent, ModelSettings

from .fake_tts import FakeTTS

pytestmark = pytest.mark.unit

TEST_CONN_OPTIONS = APIConnectOptions(max_retry=0, retry_interval=0.0, timeout=1.0)


class RecordingTTS(FakeTTS):
    def __init__(self, *, streaming: bool = True) -> None:
        super().__init__(fake_audio_duration=0.01)
        self.inputs: list[str] = []
        self._capabilities = tts.TTSCapabilities(streaming=streaming)

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = TEST_CONN_OPTIONS) -> Any:
        self.inputs.append(text)
        return super().synthesize(text, conn_options=conn_options)


async def _collect_stream(adapter: tts.StreamAdapter, text: str, *, flush: bool = False) -> None:
    async with adapter.stream(conn_options=TEST_CONN_OPTIONS) as stream:
        stream.push_text(text)
        if flush:
            stream.flush()
        stream.end_input()

        async for _ in stream:
            pass


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _normalized_inputs(recording_tts: RecordingTTS) -> list[str]:
    return [_normalize(text) for text in recording_tts.inputs]


async def test_default_behavior_remains_per_sentence() -> None:
    wrapped = RecordingTTS()
    adapter = tts.StreamAdapter(tts=wrapped)

    await _collect_stream(
        adapter,
        "This is the first complete sentence. This is the second complete sentence.",
    )

    assert _normalized_inputs(wrapped) == [
        "This is the first complete sentence.",
        "This is the second complete sentence.",
    ]


async def test_min_chars_batches_multiple_sentences() -> None:
    wrapped = RecordingTTS()
    adapter = tts.StreamAdapter(tts=wrapped, chunking=tts.ChunkingOptions(min_chars=70))

    await _collect_stream(
        adapter,
        "This is the first complete sentence. This is the second complete sentence.",
    )

    assert _normalized_inputs(wrapped) == [
        "This is the first complete sentence. This is the second complete sentence."
    ]


async def test_max_chars_flushes_batch_before_next_sentence_exceeds_cap() -> None:
    wrapped = RecordingTTS()
    adapter = tts.StreamAdapter(
        tts=wrapped,
        chunking=tts.ChunkingOptions(min_chars=200, max_chars=80),
    )

    await _collect_stream(
        adapter,
        (
            "This is the first complete sentence. "
            "This is the second complete sentence. "
            "This is the third complete sentence."
        ),
    )

    assert _normalized_inputs(wrapped) == [
        "This is the first complete sentence. This is the second complete sentence.",
        "This is the third complete sentence.",
    ]


async def test_flush_emits_buffered_text() -> None:
    wrapped = RecordingTTS()
    adapter = tts.StreamAdapter(tts=wrapped, chunking=tts.ChunkingOptions(min_chars=200))

    await _collect_stream(adapter, "This buffered text has no terminator", flush=True)

    assert _normalized_inputs(wrapped) == ["This buffered text has no terminator"]


async def test_end_input_emits_buffered_text() -> None:
    wrapped = RecordingTTS()
    adapter = tts.StreamAdapter(tts=wrapped, chunking=tts.ChunkingOptions(min_chars=200))

    await _collect_stream(adapter, "This buffered text has no terminator")

    assert _normalized_inputs(wrapped) == ["This buffered text has no terminator"]


async def test_oversized_sentence_passes_through() -> None:
    wrapped = RecordingTTS()
    adapter = tts.StreamAdapter(
        tts=wrapped,
        chunking=tts.ChunkingOptions(max_chars=20),
    )

    await _collect_stream(adapter, "This single sentence is longer than the configured cap.")

    assert _normalized_inputs(wrapped) == [
        "This single sentence is longer than the configured cap."
    ]


async def test_agent_tts_chunking_is_passed_to_automatic_stream_adapter(monkeypatch: Any) -> None:
    chunking = tts.ChunkingOptions(min_chars=70, max_chars=120)
    wrapped = RecordingTTS(streaming=False)
    captured: dict[str, Any] = {}

    class EmptyStream:
        def push_text(self, text: str) -> None:
            pass

        def end_input(self) -> None:
            pass

        def __aiter__(self) -> EmptyStream:
            return self

        async def __anext__(self) -> Any:
            raise StopAsyncIteration

        async def __aenter__(self) -> EmptyStream:
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

    class CapturingStreamAdapter:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def stream(self, *, conn_options: APIConnectOptions) -> EmptyStream:
            return EmptyStream()

    monkeypatch.setattr(tts, "StreamAdapter", CapturingStreamAdapter)

    agent = Agent(instructions="test", tts=wrapped, tts_chunking=chunking)
    agent._activity = SimpleNamespace(
        tts=wrapped,
        session=SimpleNamespace(conn_options=SimpleNamespace(tts_conn_options=TEST_CONN_OPTIONS)),
        _resolve_expressive_options=lambda: None,
    )

    async def text() -> Any:
        yield "Hello."

    async for _ in Agent.default.tts_node(agent, text(), ModelSettings()):
        pass

    assert captured["tts"] is wrapped
    assert captured["chunking"] is chunking
    assert captured["_xml_aware"] is False
    assert agent.tts_chunking is chunking


def test_chunking_options_is_exported() -> None:
    from livekit.agents.tts import ChunkingOptions

    assert ChunkingOptions is tts.ChunkingOptions


def test_sentence_tokenizer_and_chunking_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="cannot both be provided"):
        tts.StreamAdapter(
            tts=RecordingTTS(),
            sentence_tokenizer=tokenize.blingfire.SentenceTokenizer(retain_format=True),
            chunking=tts.ChunkingOptions(min_chars=70),
        )


@pytest.mark.parametrize(
    ("chunking", "expected_min", "expected_max"),
    [
        (tts.ChunkingOptions(min_chars=1), 1, None),
        (tts.ChunkingOptions(max_chars=1), None, 1),
        (tts.ChunkingOptions(min_chars=10, max_chars=5), 10, 5),
    ],
)
def test_valid_chunking_options(
    chunking: tts.ChunkingOptions, expected_min: int | None, expected_max: int | None
) -> None:
    assert chunking.min_chars == expected_min
    assert chunking.max_chars == expected_max


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_chars": 0}, "min_chars"),
        ({"max_chars": 0}, "max_chars"),
    ],
)
def test_invalid_chunking_options_raise(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        tts.ChunkingOptions(**kwargs)
