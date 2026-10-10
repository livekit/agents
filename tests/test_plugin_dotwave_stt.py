# Copyright 2026 Dollyglot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the .wave STT plugin against a local fake of the /v1/listen socket."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, urlsplit

import aiohttp
import pytest
from websockets.asyncio.server import Request, Response, ServerConnection, serve

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, stt
from livekit.agents.types import TimedString
from livekit.plugins import dotwave

pytestmark = pytest.mark.unit

API_KEY = "test"
REQUEST_ID = "sess_test"
SAMPLE_RATE = 16000

EventType = stt.SpeechEventType


def _results(
    transcript: str,
    words: list[tuple[str, float, float]],
    *,
    is_final: bool,
    speech_final: bool = False,
    from_finalize: bool = False,
    language: str | None = "en-US",
) -> str:
    alternative: dict[str, Any] = {
        "transcript": transcript,
        "confidence": 1.0,
        "words": [
            {
                "word": text.lower(),
                "punctuated_word": text,
                "start": start,
                "end": end,
                "confidence": 1.0,
                **({"language": language} if language else {}),
            }
            for text, start, end in words
        ],
    }
    if language:
        alternative["languages"] = [language]
    start = words[0][1] if words else 0.0
    end = words[-1][2] if words else 0.0
    return json.dumps(
        {
            "type": "Results",
            "channel_index": [0, 1],
            "start": start,
            "duration": end - start,
            "is_final": is_final,
            "speech_final": speech_final,
            "from_finalize": from_finalize,
            "metadata": {
                "request_id": REQUEST_ID,
                "model_info": {
                    "name": "nemotron-asr-streaming",
                    "arch": "nemotron-asr-streaming",
                    "version": "v1",
                },
                "model_uuid": "nemotron-asr-streaming",
            },
            "channel": {"alternatives": [alternative]},
        }
    )


class FakeListen:
    """A minimal /v1/listen server.

    ``modes`` lists how each successive connection behaves:

    - "normal": transcribes the first audio, answers Finalize, closes 1000 on CloseStream.
    - "restart": closes with 1012 right after Metadata.
    - "bad_request": sends an Error, then closes with 4400.
    - "idle": sends Metadata and waits for CloseStream.
    """

    def __init__(
        self,
        modes: list[str] | None = None,
        *,
        finalize_text: str = "three",
        detected_language: str | None = "en-US",
    ) -> None:
        self.modes = modes or ["normal"]
        self.finalize_text = finalize_text
        self.detected_language = detected_language
        self.queries: list[dict[str, list[str]]] = []
        self.paths: list[str] = []
        self.messages: list[str] = []
        self.audio_bytes = 0
        self.connected = asyncio.Condition()
        self.port = 0

    async def wait_for_connections(self, count: int) -> None:
        async with self.connected:
            await asyncio.wait_for(
                self.connected.wait_for(lambda: len(self.queries) >= count), timeout=5
            )

    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        if request.headers.get("Authorization") != f"Token {API_KEY}":
            return connection.respond(HTTPStatus.UNAUTHORIZED, "invalid API key\n")
        return None

    async def _handler(self, ws: ServerConnection) -> None:
        assert ws.request is not None
        split = urlsplit(ws.request.path)
        index = len(self.queries)
        mode = self.modes[min(index, len(self.modes) - 1)]
        async with self.connected:
            self.paths.append(split.path)
            self.queries.append(parse_qs(split.query))
            self.connected.notify_all()

        await ws.send(json.dumps({"type": "Metadata", "request_id": REQUEST_ID, "channels": 1}))

        if mode == "restart":
            await ws.close(code=1012, reason="restarting")
            return
        if mode == "bad_request":
            await ws.send(json.dumps({"type": "Error", "message": "language is not supported"}))
            await ws.close(code=4400, reason="protocol error")
            return

        transcribed = mode == "idle"
        audio_since_finalize = False
        async for message in ws:
            if isinstance(message, bytes):
                self.audio_bytes += len(message)
                audio_since_finalize = True
                if not transcribed:
                    transcribed = True
                    lang = self.detected_language
                    await ws.send(
                        json.dumps({"type": "SpeechStarted", "channel": [0], "timestamp": 0.0})
                    )
                    if self.queries[index].get("interim_results") != ["false"]:
                        await ws.send(
                            _results("One", [("One", 0.08, 0.16)], is_final=False, language=lang)
                        )
                    await ws.send(
                        _results("One", [("One", 0.08, 0.16)], is_final=True, language=lang)
                    )
                    await ws.send(
                        _results("two", [("two", 0.16, 0.24)], is_final=True, language=lang)
                    )
                continue

            msg_type = json.loads(message)["type"]
            self.messages.append(msg_type)
            if msg_type == "Finalize":
                # nothing left to finalize without new audio: the answer is empty
                text = self.finalize_text if audio_since_finalize else ""
                audio_since_finalize = False
                words = [(text, 0.24, 0.32)] if text else []
                await ws.send(
                    _results(
                        text,
                        words,
                        is_final=True,
                        speech_final=True,
                        from_finalize=True,
                        language=None,
                    )
                )
            elif msg_type == "CloseStream":
                await ws.close(code=1000)
                return


@asynccontextmanager
async def fake_listen(fake: FakeListen) -> AsyncIterator[FakeListen]:
    async with serve(fake._handler, "127.0.0.1", 0, process_request=fake._process_request) as srv:
        fake.port = srv.sockets[0].getsockname()[1]
        yield fake


def _make_stt(fake: FakeListen, session: aiohttp.ClientSession, **kwargs: Any) -> dotwave.STT:
    kwargs.setdefault("api_key", API_KEY)
    return dotwave.STT(
        base_url=f"http://127.0.0.1:{fake.port}/v1/listen", http_session=session, **kwargs
    )


def _push_silence(stream: stt.RecognizeStream, *, frames: int = 10) -> None:
    # 20 ms frames of silence at 16 kHz
    for _ in range(frames):
        stream.push_frame(rtc.AudioFrame.create(SAMPLE_RATE, 1, SAMPLE_RATE // 50))


async def _collect(stream: stt.RecognizeStream) -> list[stt.SpeechEvent]:
    async def run() -> list[stt.SpeechEvent]:
        return [ev async for ev in stream]

    try:
        return await asyncio.wait_for(run(), timeout=10)
    finally:
        await stream.aclose()


def _speech(events: list[stt.SpeechEvent]) -> list[stt.SpeechEvent]:
    return [ev for ev in events if ev.type != EventType.RECOGNITION_USAGE]


@pytest.mark.asyncio
async def test_stream_events_and_query() -> None:
    async with fake_listen(FakeListen()) as fake, aiohttp.ClientSession() as session:
        stream = _make_stt(fake, session, utterance_end_ms=1000).stream()
        _push_silence(stream)
        stream.flush()
        stream.end_input()
        events = await _collect(stream)

    speech = _speech(events)
    assert [ev.type for ev in speech] == [
        EventType.START_OF_SPEECH,
        EventType.INTERIM_TRANSCRIPT,
        EventType.FINAL_TRANSCRIPT,
        EventType.FINAL_TRANSCRIPT,
        EventType.FINAL_TRANSCRIPT,
        EventType.END_OF_SPEECH,
    ]
    finals = [ev for ev in speech if ev.type == EventType.FINAL_TRANSCRIPT]
    assert [ev.alternatives[0].text for ev in finals] == ["One", "two", "three"]
    assert speech[1].alternatives[0].text == "One"
    assert all(ev.request_id == REQUEST_ID for ev in finals)

    # the detected language comes from languages[0]; with automatic detection and no
    # languages on the message, the language is empty
    assert [ev.alternatives[0].language for ev in finals] == ["en-US", "en-US", ""]

    first = finals[0].alternatives[0]
    assert first.words is not None and len(first.words) == 1
    word = first.words[0]
    assert isinstance(word, TimedString)
    assert str(word) == "One"
    # times carry the stream's start_time_offset, a few microseconds in this test
    assert word.start_time == pytest.approx(0.08, abs=1e-3)
    assert word.end_time == pytest.approx(0.16, abs=1e-3)
    assert word.confidence == 1.0
    assert first.start_time == pytest.approx(0.08, abs=1e-3)
    assert first.end_time == pytest.approx(0.16, abs=1e-3)
    assert first.confidence == 1.0

    usage = [ev for ev in events if ev.type == EventType.RECOGNITION_USAGE]
    assert usage, "expected RECOGNITION_USAGE events"
    assert all(ev.recognition_usage is not None for ev in usage)
    total = sum(ev.recognition_usage.audio_duration for ev in usage if ev.recognition_usage)
    assert total == pytest.approx(0.2)
    assert fake.audio_bytes == int(0.2 * SAMPLE_RATE) * 2

    assert fake.paths == ["/v1/listen"]
    query = fake.queries[0]
    assert "language" not in query
    assert query["model"] == ["nemotron-asr-streaming"]
    assert query["encoding"] == ["linear16"]
    assert query["sample_rate"] == ["16000"]
    assert query["channels"] == ["1"]
    assert query["interim_results"] == ["true"]
    assert query["endpointing"] == ["false"]
    assert query["utterance_end_ms"] == ["1000"]

    # flush() and end_input() both flush; only the first one follows new audio
    assert fake.messages.count("Finalize") == 1
    assert fake.messages.count("CloseStream") == 1
    assert fake.messages.index("Finalize") < fake.messages.index("CloseStream")


@pytest.mark.asyncio
async def test_language_sent_verbatim_and_empty_finalize() -> None:
    fake = FakeListen(finalize_text="", detected_language=None)
    async with fake_listen(fake), aiohttp.ClientSession() as session:
        stream = _make_stt(fake, session, language="pt-BR", interim_results=False).stream()
        _push_silence(stream)
        stream.flush()
        stream.end_input()
        events = await _collect(stream)

    query = fake.queries[0]
    assert query["language"] == ["pt-BR"]
    assert query["interim_results"] == ["false"]
    assert "utterance_end_ms" not in query

    speech = _speech(events)
    # an empty Finalize answer emits no transcript, but still ends the speech
    assert [ev.type for ev in speech] == [
        EventType.START_OF_SPEECH,
        EventType.FINAL_TRANSCRIPT,
        EventType.FINAL_TRANSCRIPT,
        EventType.END_OF_SPEECH,
    ]
    # without languages[] on the message, the configured language is reported
    finals = [ev for ev in speech if ev.type == EventType.FINAL_TRANSCRIPT]
    assert [ev.alternatives[0].language for ev in finals] == ["pt-BR", "pt-BR"]


@pytest.mark.asyncio
async def test_vad_events_disabled_starts_with_first_transcript() -> None:
    async with fake_listen(FakeListen()) as fake, aiohttp.ClientSession() as session:
        stream = _make_stt(fake, session, vad_events=False).stream()
        _push_silence(stream)
        stream.flush()
        stream.end_input()
        events = await _collect(stream)

    speech = _speech(events)
    assert speech[0].type == EventType.START_OF_SPEECH
    assert speech[1].type == EventType.INTERIM_TRANSCRIPT
    assert sum(ev.type == EventType.START_OF_SPEECH for ev in speech) == 1


@pytest.mark.asyncio
async def test_reconnects_after_restart_close() -> None:
    fake = FakeListen(["restart", "normal"])
    async with fake_listen(fake), aiohttp.ClientSession() as session:
        stream = _make_stt(fake, session).stream(
            conn_options=APIConnectOptions(max_retry=2, retry_interval=0.0, timeout=5)
        )
        await fake.wait_for_connections(2)
        _push_silence(stream)
        stream.flush()
        stream.end_input()
        events = await _collect(stream)

    finals = [ev for ev in events if ev.type == EventType.FINAL_TRANSCRIPT]
    assert [ev.alternatives[0].text for ev in finals] == ["One", "two", "three"]
    assert len(fake.queries) == 2


@pytest.mark.asyncio
async def test_update_options_reconnects_with_new_query() -> None:
    fake = FakeListen(["idle", "normal"])
    async with fake_listen(fake), aiohttp.ClientSession() as session:
        dotwave_stt = _make_stt(fake, session)
        stream = dotwave_stt.stream()
        await fake.wait_for_connections(1)
        dotwave_stt.update_options(language="de", utterance_end_ms=2000)
        await fake.wait_for_connections(2)
        _push_silence(stream)
        stream.flush()
        stream.end_input()
        events = await _collect(stream)

    assert "language" not in fake.queries[0]
    assert fake.queries[1]["language"] == ["de"]
    assert fake.queries[1]["utterance_end_ms"] == ["2000"]
    assert any(ev.type == EventType.FINAL_TRANSCRIPT for ev in events)


@pytest.mark.asyncio
async def test_error_close_is_not_retried() -> None:
    fake = FakeListen(["bad_request"])
    async with fake_listen(fake), aiohttp.ClientSession() as session:
        stream = _make_stt(fake, session).stream(
            conn_options=APIConnectOptions(max_retry=3, retry_interval=0.0, timeout=5)
        )
        with pytest.raises(APIStatusError) as exc_info:
            await _collect(stream)

    err = exc_info.value
    assert err.status_code == 4400
    assert err.retryable is False
    assert "language is not supported" in err.message
    assert len(fake.queries) == 1


@pytest.mark.asyncio
async def test_bad_key_is_not_leaked() -> None:
    secret = "dw_secret_key_value"
    async with fake_listen(FakeListen()) as fake, aiohttp.ClientSession() as session:
        stream = _make_stt(fake, session, api_key=secret).stream(
            conn_options=APIConnectOptions(max_retry=0, timeout=5)
        )
        with pytest.raises(APIStatusError) as exc_info:
            await _collect(stream)

    err = exc_info.value
    assert err.status_code == 401
    assert err.retryable is False
    chain: list[BaseException] = []
    current: BaseException | None = err
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    for exc in chain:
        assert secret not in str(exc)
        assert secret not in repr(exc)


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOTWAVE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DOTWAVE_API_KEY"):
        dotwave.STT()


def test_api_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOTWAVE_API_KEY", "from-env")
    dotwave_stt = dotwave.STT()
    assert dotwave_stt.provider == ".wave"
    assert dotwave_stt.model == "nemotron-asr-streaming"
    assert dotwave_stt.capabilities.streaming is True
    assert dotwave_stt.capabilities.aligned_transcript == "word"
    assert dotwave_stt.capabilities.offline_recognize is False


@pytest.mark.parametrize(
    "kwargs",
    [{"sample_rate": 8000}, {"utterance_end_ms": 500}, {"utterance_end_ms": 5000}, {"model": ""}],
)
def test_invalid_options_raise(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        dotwave.STT(api_key=API_KEY, **kwargs)


@pytest.mark.asyncio
async def test_recognize_is_not_supported() -> None:
    dotwave_stt = dotwave.STT(api_key=API_KEY)
    with pytest.raises(NotImplementedError):
        await dotwave_stt.recognize(rtc.AudioFrame.create(SAMPLE_RATE, 1, 160))
