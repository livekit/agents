from __future__ import annotations

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from livekit.agents import APIStatusError, utils
from livekit.agents.metrics import TTSMetrics
from livekit.agents.tts import TTSError
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT, APIConnectOptions
from livekit.plugins import breezeblue
from livekit.plugins.breezeblue._pcm import PcmPause
from livekit.plugins.breezeblue._realtime import RealtimeConnection
from livekit.plugins.breezeblue._text import utf8_chunks

pytestmark = pytest.mark.unit


class _PcmResponse:
    content_type = "audio/pcm"
    headers: dict[str, str] = {}

    def __init__(self, pcm: bytes, status: int) -> None:
        self.status = status
        self._pcm = pcm
        self.content = self

    async def iter_any(self):
        yield self._pcm


class _FakeHttp:
    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.texts: list[str] = []

    @asynccontextmanager
    async def post(self, url, *, json, **options):
        self.texts.append(json["text"])
        value = 1000 * len(self.texts)
        yield _PcmResponse(value.to_bytes(2, "little") * 480, self.status)


class _TimedResponse(_PcmResponse):
    content_type = "application/x-ndjson"

    def __init__(self, pcm: bytes, text: str, start: float) -> None:
        super().__init__(pcm, 200)
        self._text = text
        self._start = start

    async def __aiter__(self):
        yield json.dumps(
            {
                "audio_base64": base64.b64encode(self._pcm).decode(),
                "word_timestamps": [
                    {"index": 0, "word": self._text.strip(), "start": self._start, "end": 0.02}
                ],
            }
        ).encode()


class _FakeTimedHttp(_FakeHttp):
    @asynccontextmanager
    async def post(self, url, *, json, **options):
        self.texts.append(json["text"])
        value = 1000 * len(self.texts)
        yield _TimedResponse(
            value.to_bytes(2, "little") * 480, json["text"], 0.01 if value > 1000 else 0.0
        )


async def test_http_session_uses_livekit_context_without_closing_it():
    async with utils.http_context.open() as shared, aiohttp.ClientSession() as caller:
        async with breezeblue.TTS(voice="test", api_key="test") as provider:
            assert provider._http_session() is shared
        assert not shared.closed
        async with breezeblue.TTS(voice="test", api_key="test", http_session=caller) as provider:
            assert provider._http_session() is caller
        assert not caller.closed

    async with breezeblue.TTS(voice="test", api_key="test") as provider:
        owned = provider._http_session()
        assert provider._http_session() is owned
    assert owned.closed


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "reply, expected_caption", [("train leaves", "train leaves"), ("training", "training")]
)
async def test_long_http_reply_preserves_pauses_and_caption_boundaries(
    streaming, reply, expected_caption
):
    http = _FakeTimedHttp()
    async with breezeblue.TTS(
        voice="test", api_key="test", http_session=http, max_characters=6, word_timestamps=True
    ) as provider:
        if streaming:
            stream = provider.stream()
            stream.push_text(reply)
            stream.end_input()
        else:
            stream = provider.synthesize(reply)
        async with stream:
            frames = [event.frame async for event in stream]

    assert "".join(http.texts) == reply
    samples = [sample for frame in frames for sample in frame.data]
    assert samples.index(2000) - 480 == 160 * 24
    words = [word for frame in frames for word in frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, ())]
    assert "".join(words) == expected_caption
    assert words[1].start_time == pytest.approx(0.19)
    assert words[1].end_time == pytest.approx(0.20)


@pytest.mark.virtual_time
async def test_sentence_mode_synthesizes_short_sentence_before_input_ends():
    http = _FakeHttp()
    async with (
        breezeblue.TTS(
            voice="test", api_key="test", http_session=http, input_mode="sentence"
        ) as provider,
        provider.stream() as stream,
    ):
        stream.push_text("Got it. One moment")
        assert (await asyncio.wait_for(anext(stream), timeout=1)).frame.samples_per_channel
        assert http.texts == ["Got it."]
        stream.push_text(".")
        stream.end_input()
        assert [event async for event in stream]
    assert [text.strip() for text in http.texts] == ["Got it.", "One moment."]


async def test_default_waits_for_complete_response():
    http = _FakeHttp()
    async with (
        breezeblue.TTS(voice="test", api_key="test", http_session=http) as provider,
        provider.stream() as stream,
    ):
        stream.push_text("The first sentence. ")
        stream.push_text("The final sentence.")
        await asyncio.sleep(0)
        assert http.texts == []
        stream.end_input()
        assert [event async for event in stream]
    assert http.texts == ["The first sentence. The final sentence."]


@pytest.mark.virtual_time
async def test_http_chunk_schedule_starts_before_end_input_and_repeats_last_count():
    http = _FakeHttp()
    async with (
        breezeblue.TTS(
            voice="test", api_key="test", http_session=http, input_chunk_length_schedule=[4, 6]
        ) as provider,
        provider.stream() as stream,
    ):
        stream.push_text("甲乙")
        await asyncio.sleep(0)
        assert http.texts == []
        stream.push_text("丙丁戊")
        assert (await asyncio.wait_for(anext(stream), timeout=1)).frame.samples_per_channel
        assert http.texts == ["甲乙丙丁"]

        stream.push_text("己庚辛壬癸子丑寅卯辰巳午未")
        stream.end_input()
        assert [event async for event in stream]

    assert http.texts == ["甲乙丙丁", "戊己庚辛壬癸", "子丑寅卯辰巳", "午未"]


@pytest.mark.parametrize("schedule", [[], [0], [True], [4.5], [1001]])
async def test_chunk_schedule_rejects_invalid_character_counts(schedule):
    with pytest.raises(ValueError, match="input_chunk_length_schedule"):
        breezeblue.TTS(voice="test", api_key="test", input_chunk_length_schedule=schedule)


async def test_sentence_mode_adds_a_gap_between_requests():
    http = _FakeHttp()
    async with (
        breezeblue.TTS(
            voice="test", api_key="test", http_session=http, input_mode="sentence"
        ) as provider,
        provider.stream() as stream,
    ):
        stream.push_text("The first complete sentence. The second complete sentence.")
        stream.end_input()
        audio = [sample async for event in stream for sample in event.frame.data]

    assert len(http.texts) == 2
    assert audio[:480] == [1000] * 480
    assert audio.index(2000) - 480 == 160 * 24
    assert len(audio) - audio.index(2000) - 480 < 480


async def test_retry_after_and_input_limit():
    http = _FakeHttp()
    async with (
        breezeblue.TTS(voice="test", api_key="test", http_session=http) as provider,
        provider.synthesize("x" * 1020) as stream,
    ):
        samples = [sample async for event in stream for sample in event.frame.data]
    assert [len(text) for text in http.texts] == [1000, 20]
    assert samples.count(1000) == samples.count(2000) == 480

    class _RateLimited:
        status = 429
        headers = {"Retry-After": "0.15"}

        def release(self):
            pass

    pause = AsyncMock()
    with (
        patch("livekit.plugins.breezeblue.tts.asyncio.sleep", pause),
        pytest.raises(APIStatusError) as failure,
    ):
        await breezeblue.TTS._raise_for_status(_RateLimited(), timeout=1.0)
    pause.assert_awaited_once_with(0.15)
    assert failure.value.retryable


@pytest.mark.virtual_time
@pytest.mark.parametrize("streaming", [False, True])
async def test_http_synthesis_uses_one_retry_budget_and_reports_one_terminal_error(streaming):
    http = _FakeHttp(status=503)
    conn_options = APIConnectOptions(max_retry=1)
    reply = "The complete reply."
    errors: list[TTSError] = []
    async with breezeblue.TTS(voice="test", api_key="test", http_session=http) as provider:
        provider.on("error", errors.append)
        if streaming:
            stream = provider.stream(conn_options=conn_options)
            stream.push_text(reply)
            stream.end_input()
        else:
            stream = provider.synthesize(reply, conn_options=conn_options)
        async with stream:
            with pytest.raises(APIStatusError) as failure:
                await anext(stream)

    assert failure.value.status_code == 503
    assert http.texts == [reply, reply]
    assert [event.recoverable for event in errors] == [True, False]
    assert errors[-1].error is failure.value


async def test_http_stream_reports_metrics_once_per_segment():
    http = _FakeHttp()
    metrics: list[TTSMetrics] = []
    sentences = ["The first complete sentence.", "The longer second complete sentence."]
    async with (
        breezeblue.TTS(
            voice="test", api_key="test", http_session=http, input_mode="sentence"
        ) as provider,
        provider.stream() as stream,
    ):
        provider.on("metrics_collected", metrics.append)
        stream.push_text(" ".join(sentences))
        stream.end_input()
        assert [event async for event in stream]

    assert [text.strip() for text in http.texts] == sentences
    assert [event.characters_count for event in metrics] == [len(text) for text in http.texts]


class _RealtimeWire:
    max_characters = 1000
    max_append_bytes = 2048
    options: dict[str, str] = {}
    last_used = 0.0

    def __init__(self, *, finish_turns: bool = False) -> None:
        self.finish_turns = finish_turns
        self.socket = None
        self.turn_id: str | None = None
        self.commands: list[str] = []
        self.text = ""
        self.flushed: list[str] = []
        self.events: asyncio.Queue[bytes | dict[str, str]] = asyncio.Queue()

    def reusable(self) -> bool:
        return True

    async def send(self, kind: str, **fields: str) -> None:
        self.commands.append(kind)
        if kind == "turn.start":
            assert self.turn_id is None
            self.turn_id = fields["turn_id"]
        if kind == "text.append":
            self.text += fields["text"]
        if kind == "text.flush":
            self.flushed.append(self.text)
            self.text = ""
            self.events.put_nowait(b"\x00\x08" * 1920)
        if kind == "turn.cancel" or (
            kind == "turn.end" and (self.finish_turns or "turn.cancel" in self.commands)
        ):
            assert self.turn_id is not None
            event = "turn.cancelled" if kind == "turn.cancel" else "turn.done"
            self.events.put_nowait({"type": event, "turn_id": self.turn_id})
            self.turn_id = None

    async def receive(self, socket):
        return await self.events.get()

    async def close(self) -> None:
        pass


@pytest.mark.parametrize(
    ("code", "meta", "status", "retryable", "delay"),
    [
        ("AUTH_SESSION_EXPIRED", {}, 502, True, None),
        ("AUTH_REQUIRED", {}, 401, False, None),
        ("RATE_LIMITED", {"retry_after_ms": 250}, 429, True, 0.25),
        ("RATE_LIMITED", {"retry_after_seconds": 0.5, "retry_after_ms": "bad"}, 429, True, 0.5),
        ("RATE_LIMITED", {"retry_after_ms": "bad"}, 429, True, None),
    ],
)
async def test_realtime_error_reports_only_safe_retry_metadata(
    code, meta, status, retryable, delay
):
    payload = {"type": "error", "code": code, "message": "raw provider details", "meta": meta}
    socket = Mock()
    socket.receive = AsyncMock(return_value=Mock(type=aiohttp.WSMsgType.TEXT))
    socket.receive.return_value.json.return_value = payload

    with pytest.raises(APIStatusError) as failure:
        await RealtimeConnection.receive(socket)

    assert failure.value.status_code == status
    assert failure.value.retryable is retryable
    assert failure.value.body == ({"retry_after": delay} if delay is not None else None)
    assert "raw provider details" not in str(failure.value)


async def test_realtime_cancel_then_reuse_connection():
    wire = _RealtimeWire()
    with patch.object(RealtimeConnection, "connect", AsyncMock(return_value=wire)) as connect:
        async with breezeblue.TTS(voice="test", api_key="test", transport="websocket") as provider:
            async with provider.stream() as stream:
                stream.push_text("The first reply.")
                await asyncio.sleep(0)
                assert wire.commands == []
                stream.end_input()
                assert (await anext(stream)).frame.samples_per_channel
            assert wire.turn_id is None
            assert "turn.cancel" in wire.commands
            async with provider.stream() as stream:
                stream.push_text("The next reply.")
                stream.end_input()
                assert [event async for event in stream]
    assert wire.commands.count("turn.start") == 2
    connect.assert_awaited_once()


@pytest.mark.virtual_time
async def test_realtime_chunk_schedule_flushes_each_chunk_on_one_connection():
    wire = _RealtimeWire(finish_turns=True)
    with patch.object(RealtimeConnection, "connect", AsyncMock(return_value=wire)) as connect:
        async with (
            breezeblue.TTS(
                voice="test",
                api_key="test",
                transport="websocket",
                input_chunk_length_schedule=[5, 7],
            ) as provider,
            provider.stream() as stream,
        ):
            stream.push_text("abcd")
            await asyncio.sleep(0)
            assert wire.commands == []
            stream.push_text("efghi")
            assert (await asyncio.wait_for(anext(stream), timeout=1)).frame.samples_per_channel
            assert wire.flushed == ["abcde"]
            stream.push_text("jklm")
            stream.end_input()
            assert [event async for event in stream]

    assert wire.flushed == ["abcde", "fghijkl", "m"]
    connect.assert_awaited_once()


def test_existing_quiet_audio_counts_toward_the_pause():
    pause = PcmPause(160)
    voice = b"\x00\x08" * 480
    pause.begin()
    assert pause.push(voice + bytes(60 * 48)) == (b"", voice + bytes(60 * 48))
    pause.begin()
    assert pause.push(bytes(40 * 48)) == (b"", b"")
    assert pause.push(voice) == (bytes(60 * 48), bytes(40 * 48) + voice)
    assert pause.finish() == b""


def test_append_chunks_respect_the_utf8_byte_limit():
    text = "列车🚆a" * 500
    chunks = list(utf8_chunks(text, 2048))
    assert "".join(chunks) == text
    assert all(0 < len(chunk.encode("utf-8")) <= 2048 for chunk in chunks)
