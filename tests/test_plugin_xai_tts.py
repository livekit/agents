"""Unit tests for xAI TTS websocket behavior."""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

from livekit.agents import APIConnectOptions
from livekit.agents.types import TimedString
from livekit.plugins.xai import tts as xai_tts

pytestmark = pytest.mark.plugin("xai")


class _FakeWebSocket:
    def __init__(self, audio: bytes = b"audio", timestamps: list[dict] | None = None) -> None:
        self._audio = audio
        self._timestamps = timestamps or [{}]
        self._messages: asyncio.Queue[Any] = asyncio.Queue()
        self.close_code: int | None = None
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    async def send_str(self, data: str) -> None:
        packet = json.loads(data)
        self.sent.append(packet)
        if packet["type"] == "text.done":
            for ts in self._timestamps:
                frame = {
                    "type": "audio.delta",
                    "delta": base64.b64encode(self._audio).decode("ascii"),
                }
                if ts:
                    frame["audio_timestamps"] = ts
                await self._messages.put(
                    SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(frame))
                )
            await self._messages.put(
                SimpleNamespace(
                    type=aiohttp.WSMsgType.TEXT,
                    data=json.dumps({"type": "audio.done"}),
                )
            )

    async def receive(self) -> Any:
        return await self._messages.get()

    async def close(self) -> None:
        self.closed = True


class _FakeEmitter:
    def __init__(self) -> None:
        self.started_segments: list[str] = []
        self.ended_segments = 0
        self.audio_chunks: list[bytes] = []
        self.timed: list[TimedString] = []
        self.calls: list[str] = []

    def start_segment(self, *, segment_id: str) -> None:
        self.started_segments.append(segment_id)

    def push(self, audio: bytes) -> None:
        self.audio_chunks.append(audio)
        self.calls.append("audio")

    def end_segment(self) -> None:
        self.ended_segments += 1

    def push_timed_transcript(self, delta_text: list[TimedString]) -> None:
        self.timed += delta_text
        self.calls.append("timed")


def _new_word_stream(tts: xai_tts.TTS, text: str):
    word_stream = tts._opts.tokenizer.stream()  # pyright: ignore[reportPrivateUsage]
    word_stream.push_text(text)
    word_stream.end_input()
    return word_stream


def _new_synthesize_stream(tts: xai_tts.TTS) -> xai_tts.SynthesizeStream:
    stream = object.__new__(xai_tts.SynthesizeStream)
    stream._tts = tts  # pyright: ignore[reportPrivateUsage]
    stream._opts = replace(tts._opts)  # pyright: ignore[reportPrivateUsage]
    stream._conn_options = APIConnectOptions(max_retry=0, timeout=1.0)  # pyright: ignore[reportPrivateUsage]
    stream._started_time = 0.0  # pyright: ignore[reportPrivateUsage]
    stream._acquire_time = 0.0  # pyright: ignore[reportPrivateUsage]
    stream._connection_reused = False  # pyright: ignore[reportPrivateUsage]
    stream._expressive = tts._expressive  # pyright: ignore[reportPrivateUsage]
    return stream


def _graph(text: str, start: float, step: float = 0.1) -> dict[str, list]:
    chars = list(text)
    return {
        "graph_chars": chars,
        "graph_times": [[start + i * step, start + (i + 1) * step] for i in range(len(chars))],
    }


async def _run_with_timestamps(
    monkeypatch: pytest.MonkeyPatch, tts: xai_tts.TTS, text: str, frames: list[dict]
) -> tuple[_FakeWebSocket, _FakeEmitter]:
    ws = _FakeWebSocket(timestamps=frames)

    async def connect_ws(*_args: object, **_kwargs: object) -> _FakeWebSocket:
        return ws

    monkeypatch.setattr(tts, "_connect_ws", connect_ws)
    emitter = _FakeEmitter()
    stream = _new_synthesize_stream(tts)
    await stream._run_ws(_new_word_stream(tts, text), emitter)  # pyright: ignore[reportPrivateUsage]
    return ws, emitter


@pytest.mark.asyncio
async def test_streaming_segments_reuse_websocket_connection(monkeypatch: pytest.MonkeyPatch):
    tts = xai_tts.TTS(api_key="test-key")
    stream = _new_synthesize_stream(tts)
    emitter = _FakeEmitter()
    websockets: list[_FakeWebSocket] = []

    async def connect_ws(*_args: object, **_kwargs: object) -> _FakeWebSocket:
        # xAI restarts timings at 0 for each utterance on a reused socket (verified live);
        # this fake replays the same timings for every utterance
        ws = _FakeWebSocket(timestamps=[_graph("word ", 0.0)])
        websockets.append(ws)
        return ws

    async def close_ws(ws: _FakeWebSocket) -> None:
        await ws.close()

    monkeypatch.setattr(tts, "_connect_ws", connect_ws)
    monkeypatch.setattr(tts, "_close_ws", close_ws)

    await stream._run_ws(_new_word_stream(tts, "first"), emitter)  # pyright: ignore[reportPrivateUsage]
    await stream._run_ws(_new_word_stream(tts, "second"), emitter)  # pyright: ignore[reportPrivateUsage]

    assert len(websockets) == 1
    assert websockets[0].closed is False
    assert emitter.audio_chunks == [b"audio", b"audio"]
    assert emitter.ended_segments == 2
    # nothing carries over from the first utterance: same single word, same times
    assert [(str(t), t.start_time, t.end_time) for t in emitter.timed] == [
        ("word ", 0.0, pytest.approx(0.5)),
        ("word ", 0.0, pytest.approx(0.5)),
    ]

    await tts.aclose()

    assert websockets[0].closed is True


@pytest.mark.asyncio
async def test_char_timings_become_word_timed_transcripts(monkeypatch: pytest.MonkeyPatch):
    tts = xai_tts.TTS(api_key="test-key")
    # "Hi there " arrives in two frames, splitting "there" across them
    frames = [_graph("Hi th", 0.0), _graph("ere ", 0.5)]
    ws, emitter = await _run_with_timestamps(monkeypatch, tts, "Hi there", frames)

    assert tts.capabilities.aligned_transcript
    assert [p["delta"] for p in ws.sent if p["type"] == "text.delta"] == ["Hi ", "there "]
    assert [(str(t), t.start_time, t.end_time) for t in emitter.timed] == [
        ("Hi ", 0.0, pytest.approx(0.3)),
        ("there ", pytest.approx(0.3), pytest.approx(0.9)),
    ]
    # each frame's words are pushed ahead of its audio; the split word is flushed on audio.done
    assert emitter.calls == ["timed", "audio", "audio", "timed"]
    await tts.aclose()


@pytest.mark.asyncio
async def test_word_timestamps_off(monkeypatch: pytest.MonkeyPatch):
    tts = xai_tts.TTS(api_key="test-key", word_timestamps=False)
    # even if timings arrive anyway, they must not become timed transcripts
    _, emitter = await _run_with_timestamps(monkeypatch, tts, "Hi", [_graph("Hi ", 0.0)])

    assert emitter.timed == []
    assert emitter.audio_chunks == [b"audio"]
    await tts.aclose()


@pytest.mark.asyncio
async def test_unspaced_script_gets_one_word_per_character(monkeypatch: pytest.MonkeyPatch):
    tts = xai_tts.TTS(api_key="test-key", language="ja")
    frames = [_graph("こんに", 0.0), _graph("ちは。 ", 0.3)]
    _, emitter = await _run_with_timestamps(monkeypatch, tts, "こんにちは。", frames)

    assert [str(t) for t in emitter.timed] == ["こ", "ん", "に", "ち", "は", "。 "]
    assert (emitter.timed[2].start_time, emitter.timed[2].end_time) == (
        pytest.approx(0.2),
        pytest.approx(0.3),
    )
    await tts.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("expressive", [True, False])
async def test_bracket_cues_dropped_in_expressive_mode(
    expressive: bool, monkeypatch: pytest.MonkeyPatch
):
    tts = xai_tts.TTS(api_key="test-key")
    tts._set_expressive(expressive)  # pyright: ignore[reportPrivateUsage]
    # xAI echoes the [pause] cue in its timings, split here across two frames
    frames = [_graph("Sure [pau", 0.0), _graph("se] one ", 0.9)]
    _, emitter = await _run_with_timestamps(monkeypatch, tts, "Sure [pause] one", frames)

    # outside expressive mode a bracket is the reply's own text, so it stays
    expected = "Sure one " if expressive else "Sure [pause] one "
    assert "".join(str(t) for t in emitter.timed) == expected
    one = next(t for t in emitter.timed if str(t) == "one ")
    assert (one.start_time, one.end_time) == (pytest.approx(1.3), pytest.approx(1.7))
    await tts.aclose()


@pytest.mark.asyncio
async def test_cue_attached_to_word_keeps_word_timing(monkeypatch: pytest.MonkeyPatch):
    tts = xai_tts.TTS(api_key="test-key")
    tts._set_expressive(True)  # pyright: ignore[reportPrivateUsage]
    # "[pause]" spans 0.6-1.3 s and "go" is spoken from 1.3 s
    frames = [_graph("Ready [pause]go ", 0.0)]
    _, emitter = await _run_with_timestamps(monkeypatch, tts, "Ready [pause]go", frames)

    assert [(str(t), t.start_time, t.end_time) for t in emitter.timed] == [
        ("Ready ", 0.0, pytest.approx(0.6)),
        ("go ", pytest.approx(1.3), pytest.approx(1.6)),
    ]
    await tts.aclose()


@pytest.mark.asyncio
async def test_malformed_timestamps_keep_audio(monkeypatch: pytest.MonkeyPatch):
    tts = xai_tts.TTS(api_key="test-key")
    short = {"graph_chars": ["H", "i"], "graph_times": [[0.0, 0.1]]}
    missing = {"graph_chars": ["H", "i"]}
    not_text = {"graph_chars": [None, "i"], "graph_times": [[0.0, 0.1], [0.1, 0.2]]}
    bad_time = {"graph_chars": ["H"], "graph_times": [["x", 0.1]]}
    frames = [short, missing, not_text, bad_time]
    _, emitter = await _run_with_timestamps(monkeypatch, tts, "Hi", frames)

    assert emitter.timed == []
    assert emitter.audio_chunks == [b"audio"] * 4
    await tts.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("word_timestamps", [True, False])
async def test_with_timestamps_query_param(word_timestamps: bool, monkeypatch: pytest.MonkeyPatch):
    tts = xai_tts.TTS(api_key="test-key", word_timestamps=word_timestamps)
    urls: list[str] = []

    class _FakeSession:
        async def ws_connect(self, url: str, **_kwargs: object) -> _FakeWebSocket:
            urls.append(url)
            return _FakeWebSocket()

    monkeypatch.setattr(tts, "_ensure_session", lambda: _FakeSession())
    await tts._connect_ws(1.0, tts._opts)  # pyright: ignore[reportPrivateUsage]
    assert ("with_timestamps=true" in urls[0]) is word_timestamps
    assert tts.capabilities.aligned_transcript is word_timestamps
