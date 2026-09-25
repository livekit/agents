"""Regression tests for Gradium TTS text messages and stream lifecycle."""

from __future__ import annotations

import asyncio
import base64
import json
from types import TracebackType
from typing import Any, cast
from unittest.mock import Mock

import aiohttp
import pytest

from livekit.agents import APIConnectOptions, tokenize, tts
from livekit.plugins.gradium import TTS

pytestmark = pytest.mark.unit

_BREAK = '<break time="1.5s" />'
_TEXT = f"Hello. {_BREAK} Next."
_PCM = b"\x00\x01" * 9600
_CONNECT_OPTIONS = APIConnectOptions(max_retry=0, timeout=2.0)


class _FakeWebSocket:
    """Capture requests and return PCM after the client finishes sending text."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.text_sent = asyncio.Event()
        self.closed = False
        self._messages: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()

    async def send_str(self, data: str) -> None:
        packet = json.loads(data)
        self.sent.append(packet)
        if packet["type"] == "text":
            self.text_sent.set()
        elif packet["type"] == "end_of_stream":
            for response in (
                {"type": "audio", "audio": base64.b64encode(_PCM).decode("ascii")},
                {"type": "end_of_stream"},
            ):
                self._messages.put_nowait(
                    aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(response), "")
                )

    async def receive(self) -> aiohttp.WSMessage:
        return await self._messages.get()

    async def __aenter__(self) -> _FakeWebSocket:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.closed = True

    @property
    def texts(self) -> list[str]:
        return [packet["text"] for packet in self.sent if packet["type"] == "text"]


class _FakeSession:
    """Create a separate fake WebSocket for each synthesis request."""

    def __init__(self) -> None:
        self.sockets: list[_FakeWebSocket] = []
        self.connected = asyncio.Event()

    def ws_connect(self, url: str, **kwargs: object) -> _FakeWebSocket:
        ws = _FakeWebSocket()
        self.sockets.append(ws)
        self.connected.set()
        return ws


def _provider(
    session: _FakeSession, *, word_tokenizer: tokenize.WordTokenizer | None = None
) -> TTS:
    return TTS(
        api_key="test-key",
        http_session=cast(aiohttp.ClientSession, session),
        word_tokenizer=word_tokenizer,
    )


async def _collect_audio(stream: tts.SynthesizeStream | tts.ChunkedStream) -> bytes:
    return b"".join([event.frame.data.tobytes() async for event in stream])


async def _stream_text(
    chunks: list[str], *, word_tokenizer: tokenize.WordTokenizer | None = None
) -> list[str]:
    session = _FakeSession()
    provider = _provider(session, word_tokenizer=word_tokenizer)
    async with provider.stream(conn_options=_CONNECT_OPTIONS) as stream:
        for chunk in chunks:
            stream.push_text(chunk)
        stream.end_input()
        assert await asyncio.wait_for(_collect_audio(stream), timeout=2.0) == _PCM

    (ws,) = session.sockets
    assert ws.closed
    assert ws.sent[0] == {
        "type": "setup",
        "model_name": "default",
        "output_format": "pcm",
        "voice_id": "4SZHfMpw-p46Ywgs",
    }
    assert ws.sent[-1] == {"type": "end_of_stream"}
    assert all(packet["type"] == "text" for packet in ws.sent[1:-1])
    return ws.texts


@pytest.mark.parametrize("retain_format", [False, True])
@pytest.mark.parametrize(
    "chunks",
    [
        pytest.param([_TEXT], id="whole-input"),
        pytest.param(list(_TEXT), id="character-chunks"),
        pytest.param(["Hello. <br", "eak ti", 'me="1.', '5s" /', "> Next."], id="split-tag"),
        pytest.param(["Hello. ", _BREAK, " Next."], id="whole-tag"),
    ],
)
async def test_break_is_sent_in_one_message(chunks: list[str], retain_format: bool) -> None:
    word_tokenizer = tokenize.basic.WordTokenizer(
        ignore_punctuation=False, retain_format=retain_format
    )
    expected = ["Hello. ", f" {_BREAK} ", "Next. "]
    if retain_format:
        expected = ["Hello. ", '  <break  time="1.5s"  /> ', " Next. "]

    assert await _stream_text(chunks, word_tokenizer=word_tokenizer) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"{_BREAK} Next.", [f" {_BREAK} ", "Next. "]),
        (f"Hello. {_BREAK}", ["Hello. ", f" {_BREAK} "]),
        (_BREAK, [f" {_BREAK} "]),
        (
            'First. <break time="0.1s" /> <break time="2.0s" /> Last.',
            ["First. ", ' <break time="0.1s" /> ', ' <break time="2.0s" /> ', "Last. "],
        ),
        (
            f"First. {_BREAK} Second. {_BREAK} Last.",
            ["First. ", f" {_BREAK} ", "Second. ", f" {_BREAK} ", "Last. "],
        ),
    ],
)
async def test_break_placement_and_duration(text: str, expected: list[str]) -> None:
    assert await _stream_text([text]) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello, world! <flush> Next.", ["Hello, ", "world! ", "<flush> ", "Next. "]),
        ("<breakfast is ready.", ["<breakfast ", "is ", "ready. "]),
        ('Hello<break time="1.5s" />world', ["Hello<break ", 'time="1.5s" ', "/>world "]),
    ],
)
async def test_other_text_is_unchanged(text: str, expected: list[str]) -> None:
    assert await _stream_text(list(text)) == expected


async def test_speech_before_an_incomplete_break_is_sent_immediately() -> None:
    session = _FakeSession()
    async with _provider(session).stream(conn_options=_CONNECT_OPTIONS) as stream:
        stream.push_text("Hello. <break time=")
        await asyncio.wait_for(session.connected.wait(), timeout=2.0)
        (ws,) = session.sockets
        await asyncio.wait_for(ws.text_sent.wait(), timeout=2.0)
        assert ws.texts == ["Hello. "]

        stream.push_text('"1.5s" /> Next.')
        stream.end_input()
        assert await asyncio.wait_for(_collect_audio(stream), timeout=2.0) == _PCM

    assert ws.texts == ["Hello. ", f" {_BREAK} ", "Next. "]


@pytest.mark.parametrize("explicit_flush", [False, True])
async def test_incomplete_break_is_preserved_without_leaking_into_next_stream(
    explicit_flush: bool,
) -> None:
    session = _FakeSession()
    provider = _provider(session)
    async with provider.stream(conn_options=_CONNECT_OPTIONS) as stream:
        stream.push_text('Hello. <break time="1.5s"')
        if explicit_flush:
            stream.flush()
        stream.end_input()
        assert await asyncio.wait_for(_collect_audio(stream), timeout=2.0) == _PCM

    async with provider.stream(conn_options=_CONNECT_OPTIONS) as stream:
        stream.push_text("Next.")
        stream.end_input()
        assert await asyncio.wait_for(_collect_audio(stream), timeout=2.0) == _PCM

    assert session.sockets[0].texts == ["Hello. ", "<break ", 'time="1.5s" ']
    assert session.sockets[1].texts == ["Next. "]
    assert all(ws.closed for ws in session.sockets)


async def test_complete_break_from_custom_tokenizer() -> None:
    word_tokenizer = Mock(spec=tokenize.WordTokenizer)
    word_tokenizer.stream.return_value = tokenize.BufferedWordStream(
        tokenizer=lambda text: [text] if text else [], min_token_len=1, min_ctx_len=1
    )
    word_tokenizer.format_words.side_effect = " ".join

    assert await _stream_text([_BREAK], word_tokenizer=word_tokenizer) == [f" {_BREAK} "]
    word_tokenizer.format_words.assert_called_once_with([_BREAK])


async def test_cancellation_does_not_send_buffered_break() -> None:
    session = _FakeSession()
    async with _provider(session).stream(conn_options=_CONNECT_OPTIONS) as stream:
        stream.push_text("Hello. <break time=")
        await asyncio.wait_for(session.connected.wait(), timeout=2.0)
        (ws,) = session.sockets
        await asyncio.wait_for(ws.text_sent.wait(), timeout=2.0)
        assert ws.texts == ["Hello. "]
        before_close = ws.sent.copy()

    assert ws.closed
    assert ws.sent == before_close
    assert stream._task.done()


async def test_one_shot_synthesis_preserves_complete_input() -> None:
    session = _FakeSession()
    async with _provider(session).synthesize(_TEXT, conn_options=_CONNECT_OPTIONS) as stream:
        audio = await asyncio.wait_for(_collect_audio(stream), timeout=2.0)
        assert audio[: len(_PCM)] == _PCM

    (ws,) = session.sockets
    assert ws.texts == [_TEXT]
    assert ws.closed
