from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable
from typing import Any

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions
from livekit.agents.beta.translation import (
    InputTranscript,
    OutputTranscript,
    TranslatedAudioFrame,
    TranslationModelError,
)
from livekit.plugins.openai.beta import RealtimeTranslationModel, RealtimeTranslationSession
from livekit.plugins.openai.beta.translation_model import _translation_url

pytestmark = pytest.mark.unit


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.incoming: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()

    def emit(self, event: dict[str, Any]) -> None:
        self.incoming.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(event), ""))

    def drop(self) -> None:
        self.incoming.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, ""))

    async def send_str(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def receive(self) -> aiohttp.WSMessage:
        return await self.incoming.get()

    async def close(self) -> None:
        self.drop()


def _connect_hook(monkeypatch: pytest.MonkeyPatch) -> list[_FakeWS]:
    conns: list[_FakeWS] = []

    async def _create_ws_conn(self: RealtimeTranslationSession) -> _FakeWS:
        ws = _FakeWS()
        conns.append(ws)
        return ws

    monkeypatch.setattr(RealtimeTranslationSession, "_create_ws_conn", _create_ws_conn)
    return conns


def _model(**kwargs: Any) -> RealtimeTranslationModel:
    return RealtimeTranslationModel(target_language="es", api_key="sk-test", **kwargs)


async def _wait_for(pred: Callable[[], bool], timeout: float = 2.0) -> None:
    async def _poll() -> None:
        while not pred():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout)


def test_the_url_targets_the_translation_endpoint() -> None:
    assert (
        _translation_url("https://api.openai.com/v1", "gpt-realtime-translate")
        == "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"
    )
    # a full path is kept as given
    assert (
        _translation_url("wss://proxy.example/translate", "m")
        == "wss://proxy.example/translate?model=m"
    )


async def test_the_session_opens_with_its_language_and_transcription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conns = _connect_hook(monkeypatch)
    session = _model(noise_reduction="near_field").session()
    try:
        await _wait_for(lambda: bool(conns) and bool(conns[0].sent))
        assert conns[0].sent[0]["type"] == "session.update"
        assert conns[0].sent[0]["session"] == {
            "audio": {
                "output": {"language": "es"},
                "input": {
                    "transcription": {"model": "gpt-realtime-whisper"},
                    "noise_reduction": {"type": "near_field"},
                },
            }
        }
    finally:
        await session.aclose()

    # closing asks the server to end the session
    assert conns[0].sent[-1] == {"type": "session.close"}


async def test_input_audio_is_resampled_and_chunked(monkeypatch: pytest.MonkeyPatch) -> None:
    conns = _connect_hook(monkeypatch)
    session = _model().session()
    try:
        # 200ms at 24kHz, already the wire rate: two 100ms appends
        session.push_audio(
            rtc.AudioFrame(
                data=b"\x01\x00" * 4800,
                sample_rate=24000,
                num_channels=1,
                samples_per_channel=4800,
            )
        )
        await _wait_for(
            lambda: (
                bool(conns)
                and sum(e["type"] == "session.input_audio_buffer.append" for e in conns[0].sent)
                == 2
            )
        )
        appends = [e for e in conns[0].sent if e["type"] == "session.input_audio_buffer.append"]
        assert all(len(base64.b64decode(e["audio"])) == 2400 * 2 for e in appends)
    finally:
        await session.aclose()


async def test_translated_audio_reaches_the_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    conns = _connect_hook(monkeypatch)
    session = _model().session()
    try:
        await _wait_for(lambda: bool(conns))
        pcm = b"\x02\x00" * 480
        conns[0].emit(
            {
                "type": "session.output_audio.delta",
                "delta": base64.b64encode(pcm).decode(),
                "sample_rate": 24000,
                "channels": 1,
            }
        )

        stream = aiter(session.audio_stream)
        ev = await asyncio.wait_for(anext(stream), 2)
        assert isinstance(ev, TranslatedAudioFrame)
        assert ev.frame.samples_per_channel == 480
        assert ev.frame.sample_rate == 24000
        assert ev.language == "es"
    finally:
        await session.aclose()


async def test_transcripts_end_once_their_deltas_go_quiet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conns = _connect_hook(monkeypatch)
    session = _model(transcript_segment_timeout=0.05).session()
    heard: list[InputTranscript] = []
    spoken: list[OutputTranscript] = []
    session.on("input_transcript", heard.append)
    session.on("output_transcript", spoken.append)
    try:
        await _wait_for(lambda: bool(conns))
        conns[0].emit({"type": "session.input_transcript.delta", "delta": "hello "})
        conns[0].emit({"type": "session.input_transcript.delta", "delta": "there"})
        conns[0].emit({"type": "session.output_transcript.delta", "delta": "hola"})

        await _wait_for(lambda: len(heard) == 3 and len(spoken) == 2)
        assert [(t.text, t.is_final) for t in heard] == [
            ("hello ", False),
            ("there", False),
            ("", True),
        ]
        assert [(t.text, t.is_final, t.language) for t in spoken] == [
            ("hola", False, "es"),
            ("", True, "es"),
        ]
    finally:
        await session.aclose()


async def test_retargeting_updates_the_running_session(monkeypatch: pytest.MonkeyPatch) -> None:
    conns = _connect_hook(monkeypatch)
    model = _model()
    assert model.capabilities.retargetable
    session = model.session()
    try:
        await _wait_for(lambda: bool(conns))
        session.update_options(target_language="fr")
        await _wait_for(lambda: len(conns[0].sent) == 2)
        assert conns[0].sent[1]["session"] == {"audio": {"output": {"language": "fr"}}}

        # a reconnection opens with the language the session has moved to
        conns[0].drop()
        await _wait_for(lambda: len(conns) == 2 and bool(conns[1].sent))
        assert conns[1].sent[0]["session"]["audio"]["output"] == {"language": "fr"}
    finally:
        await session.aclose()


async def test_a_fatal_error_ends_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    conns = _connect_hook(monkeypatch)
    session = _model(conn_options=APIConnectOptions(max_retry=3, retry_interval=0)).session()
    errors: list[TranslationModelError] = []
    session.on("error", errors.append)
    try:
        await _wait_for(lambda: bool(conns))
        conns[0].emit(
            {
                "type": "error",
                "error": {"type": "invalid_request_error", "code": "insufficient_quota"},
            }
        )

        await _wait_for(lambda: bool(errors))
        assert not errors[-1].recoverable
        # the stream ends rather than waiting on a session that will not reconnect
        async for _ in session.audio_stream:
            pass
        assert len(conns) == 1
    finally:
        await session.aclose()


async def test_a_recoverable_error_keeps_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    conns = _connect_hook(monkeypatch)
    session = _model().session()
    errors: list[TranslationModelError] = []
    session.on("error", errors.append)
    try:
        await _wait_for(lambda: bool(conns))
        conns[0].emit(
            {"type": "error", "error": {"type": "invalid_request_error", "code": "bad_language"}}
        )
        await _wait_for(lambda: bool(errors))
        assert errors[0].recoverable
        assert len(conns) == 1
    finally:
        await session.aclose()
