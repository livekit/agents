from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from google.genai import errors, types

from livekit import rtc
from livekit.agents import APIConnectOptions
from livekit.agents.beta.translation import (
    InputTranscript,
    OutputSpeechStopped,
    OutputTranscript,
    TranslatedAudioFrame,
    TranslationModelError,
)
from livekit.plugins.google.beta import GeminiTranslationModel, GeminiTranslationSession

pytestmark = pytest.mark.unit


class _FakeLiveSession:
    def __init__(self, config: types.LiveConnectConfig) -> None:
        self.config = config
        self.sent: list[types.Blob] = []
        self.fail_sends = False
        self.stall_sends = False
        self._incoming: asyncio.Queue[types.LiveServerMessage | Exception] = asyncio.Queue()

    def emit(self, **server_content: Any) -> None:
        self._incoming.put_nowait(
            types.LiveServerMessage(server_content=types.LiveServerContent(**server_content))
        )

    def go_away(self) -> None:
        self._incoming.put_nowait(types.LiveServerMessage(go_away=types.LiveServerGoAway()))

    def drop(self) -> None:
        self._incoming.put_nowait(ConnectionError("connection lost"))

    async def send_realtime_input(self, *, audio: types.Blob) -> None:
        if self.fail_sends:
            raise ConnectionResetError("connection reset")
        if self.stall_sends:
            await asyncio.Future()
        self.sent.append(audio)

    async def receive(self) -> AsyncIterator[types.LiveServerMessage]:
        msg = await self._incoming.get()
        if isinstance(msg, Exception):
            raise msg
        yield msg


def _connect_hook(
    monkeypatch: pytest.MonkeyPatch, *, handshake_errors: list[Exception] | None = None
) -> list[_FakeLiveSession]:
    conns: list[_FakeLiveSession] = []
    pending_errors = list(handshake_errors or [])

    @contextlib.asynccontextmanager
    async def _connect(
        self: GeminiTranslationSession, config: types.LiveConnectConfig
    ) -> AsyncIterator[_FakeLiveSession]:
        if pending_errors:
            raise pending_errors.pop(0)
        conn = _FakeLiveSession(config)
        conns.append(conn)
        yield conn

    monkeypatch.setattr(GeminiTranslationSession, "_connect", _connect)
    return conns


def _pcm(duration: float, *, fill: int) -> bytes:
    return bytes([fill, 0]) * int(16000 * duration)


def _frame(duration: float, *, fill: int) -> rtc.AudioFrame:
    data = _pcm(duration, fill=fill)
    return rtc.AudioFrame(
        data=data, sample_rate=16000, num_channels=1, samples_per_channel=len(data) // 2
    )


def _audio_turn() -> types.Content:
    return types.Content(
        parts=[types.Part(inline_data=types.Blob(data=bytes([2, 0]) * 480, mime_type="audio/pcm"))]
    )


def _model(**kwargs: Any) -> GeminiTranslationModel:
    return GeminiTranslationModel(target_language="es", api_key="test", vertexai=False, **kwargs)


async def _wait_for(pred: Callable[[], bool], timeout: float = 2.0) -> None:
    async def _poll() -> None:
        while not pred():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout)


async def test_the_session_connects_with_its_translation_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conns = _connect_hook(monkeypatch)
    session = _model(echo_target_language=True).session()
    try:
        await _wait_for(lambda: bool(conns))
        config = conns[0].config
        assert config.translation_config == types.TranslationConfig(
            target_language_code="es", echo_target_language=True
        )
        assert config.input_audio_transcription is not None
        assert config.output_audio_transcription is not None
    finally:
        await session.aclose()


def test_transcription_can_be_turned_off() -> None:
    model = _model(input_audio_transcription=None)
    assert not model.capabilities.input_transcription
    assert model.capabilities.output_transcription


async def test_input_audio_is_resampled_and_chunked(monkeypatch: pytest.MonkeyPatch) -> None:
    conns = _connect_hook(monkeypatch)
    session = _model().session()
    try:
        # 200ms at 48kHz: two 100ms blobs at 16kHz once resampled
        session.push_audio(
            rtc.AudioFrame(
                data=b"\x01\x00" * 9600,
                sample_rate=48000,
                num_channels=1,
                samples_per_channel=9600,
            )
        )
        session.push_audio(
            rtc.AudioFrame(
                data=b"\x01\x00" * 9600,
                sample_rate=48000,
                num_channels=1,
                samples_per_channel=9600,
            )
        )
        await _wait_for(lambda: bool(conns) and len(conns[0].sent) >= 2)
        assert all(b.mime_type == "audio/pcm;rate=16000" for b in conns[0].sent)
        assert all(len(b.data or b"") == 1600 * 2 for b in conns[0].sent)
    finally:
        await session.aclose()


async def test_translated_audio_ends_with_a_mark_on_generation_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conns = _connect_hook(monkeypatch)
    session = _model().session()
    try:
        await _wait_for(lambda: bool(conns))
        conns[0].emit(
            model_turn=types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(data=b"\x02\x00" * 480, mime_type="audio/pcm")
                    )
                ]
            )
        )
        conns[0].emit(generation_complete=True)

        stream = aiter(session.audio_stream)
        ev = await asyncio.wait_for(anext(stream), 2)
        assert isinstance(ev, TranslatedAudioFrame)
        assert ev.frame.samples_per_channel == 480
        assert ev.frame.sample_rate == 24000
        assert ev.language == "es"
        assert isinstance(await asyncio.wait_for(anext(stream), 2), OutputSpeechStopped)
    finally:
        await session.aclose()


async def test_transcripts_end_when_finished_or_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    conns = _connect_hook(monkeypatch)
    session = _model(transcript_segment_timeout=0.05).session()
    heard: list[InputTranscript] = []
    spoken: list[OutputTranscript] = []
    session.on("input_transcript", heard.append)
    session.on("output_transcript", spoken.append)
    try:
        await _wait_for(lambda: bool(conns))
        conns[0].emit(input_transcription=types.Transcription(text=" hello", language_code="en"))
        conns[0].emit(
            input_transcription=types.Transcription(text=" there", finished=True),
            output_transcription=types.Transcription(text="hola"),
        )

        await _wait_for(lambda: len(heard) == 3 and len(spoken) == 2)
        # the leading space gemini opens with is dropped, the language carries to the end
        # the source transcript replaces itself in place, as an STT interim does
        assert [(t.text, t.is_final, t.language) for t in heard] == [
            ("hello", False, "en"),
            ("hello there", False, "en"),
            ("hello there", True, "en"),
        ]
        # the server never marked this one finished, so the quiet ended it
        assert [(t.text, t.is_final, t.language) for t in spoken] == [
            ("hola", False, "es"),
            ("", True, "es"),
        ]
    finally:
        await session.aclose()


async def test_retargeting_reconnects_with_the_new_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conns = _connect_hook(monkeypatch)
    model = _model()
    assert model.capabilities.retargetable
    session = model.session()
    try:
        await _wait_for(lambda: bool(conns))
        session.update_options(target_language="fr")
        await _wait_for(lambda: len(conns) == 2)
        assert conns[1].config.translation_config is not None
        assert conns[1].config.translation_config.target_language_code == "fr"
    finally:
        await session.aclose()


async def test_go_away_reconnects(monkeypatch: pytest.MonkeyPatch) -> None:
    conns = _connect_hook(monkeypatch)
    session = _model().session()
    errors: list[TranslationModelError] = []
    session.on("error", errors.append)
    try:
        await _wait_for(lambda: bool(conns))
        conns[0].go_away()
        await _wait_for(lambda: len(conns) == 2)
        assert not errors
    finally:
        await session.aclose()


async def test_a_dropped_connection_retries_then_gives_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conns = _connect_hook(monkeypatch)
    session = _model(conn_options=APIConnectOptions(max_retry=1, retry_interval=0)).session()
    errors: list[TranslationModelError] = []
    session.on("error", errors.append)
    try:
        await _wait_for(lambda: bool(conns))
        conns[0].drop()
        await _wait_for(lambda: len(conns) == 2)
        assert errors[0].recoverable

        conns[1].drop()
        await _wait_for(lambda: len(errors) == 2)
        assert not errors[1].recoverable
        # the stream ends rather than waiting on a session that will not reconnect
        async for _ in session.audio_stream:
            pass
    finally:
        await session.aclose()


async def test_audio_whose_send_failed_goes_out_on_the_next_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conns = _connect_hook(monkeypatch)
    session = _model(conn_options=APIConnectOptions(max_retry=3, retry_interval=0)).session()
    try:
        await _wait_for(lambda: bool(conns))
        conns[0].fail_sends = True
        session.push_audio(_frame(0.1, fill=3))

        await _wait_for(lambda: len(conns) == 2 and bool(conns[1].sent))
        assert [b.data for b in conns[1].sent] == [_pcm(0.1, fill=3)]
    finally:
        await session.aclose()


async def test_closing_sends_the_tail_short_of_a_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    conns = _connect_hook(monkeypatch)
    session = _model().session()
    await _wait_for(lambda: bool(conns))
    # 80ms: less than one 100ms chunk
    session.push_audio(_frame(0.08, fill=4))
    await session.aclose()

    assert [b.data for b in conns[0].sent] == [_pcm(0.08, fill=4)]


async def test_audio_from_before_a_retarget_keeps_its_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conns = _connect_hook(monkeypatch)
    session = _model().session()
    try:
        await _wait_for(lambda: bool(conns))
        stream = aiter(session.audio_stream)
        conns[0].emit(model_turn=_audio_turn())
        first = await asyncio.wait_for(anext(stream), 2)

        session.update_options(target_language="fr")
        await _wait_for(lambda: len(conns) == 2)
        conns[1].emit(model_turn=_audio_turn())
        # the old connection's speech is closed off before the new one starts
        assert isinstance(await asyncio.wait_for(anext(stream), 2), OutputSpeechStopped)
        second = await asyncio.wait_for(anext(stream), 2)

        assert isinstance(first, TranslatedAudioFrame) and first.language == "es"
        assert isinstance(second, TranslatedAudioFrame) and second.language == "fr"
    finally:
        await session.aclose()


async def test_a_handshake_that_fails_in_transit_is_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conns = _connect_hook(monkeypatch, handshake_errors=[ConnectionError("network down")])
    session = _model(conn_options=APIConnectOptions(max_retry=3, retry_interval=0)).session()
    errors_seen: list[TranslationModelError] = []
    session.on("error", errors_seen.append)
    try:
        await _wait_for(lambda: bool(conns))
        assert [e.recoverable for e in errors_seen] == [True]
    finally:
        await session.aclose()


async def test_a_handshake_the_server_rejects_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conns = _connect_hook(
        monkeypatch, handshake_errors=[errors.APIError(1007, {"message": "invalid argument"})]
    )
    session = _model(conn_options=APIConnectOptions(max_retry=3, retry_interval=0)).session()
    errors_seen: list[TranslationModelError] = []
    session.on("error", errors_seen.append)
    try:
        async for _ in session.audio_stream:
            pass
        assert [e.recoverable for e in errors_seen] == [False]
        assert not conns
    finally:
        await session.aclose()


async def test_audio_in_flight_at_a_retarget_goes_out_on_the_next_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conns = _connect_hook(monkeypatch)
    session = _model().session()
    try:
        await _wait_for(lambda: bool(conns))
        conns[0].stall_sends = True
        session.push_audio(_frame(0.1, fill=5))
        await asyncio.sleep(0.05)

        session.update_options(target_language="fr")
        await _wait_for(lambda: len(conns) == 2 and bool(conns[1].sent))
        assert [b.data for b in conns[1].sent] == [_pcm(0.1, fill=5)]
    finally:
        await session.aclose()


async def test_a_sample_rate_change_keeps_the_tail_of_the_old_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conns = _connect_hook(monkeypatch)
    session = _model().session()
    await _wait_for(lambda: bool(conns))
    # 10ms at 48kHz, all of it still held in the resampler
    session.push_audio(
        rtc.AudioFrame(
            data=b"\x01\x00" * 480, sample_rate=48000, num_channels=1, samples_per_channel=480
        )
    )
    session.push_audio(_frame(0.09, fill=6))
    await session.aclose()

    assert sum(len(b.data or b"") for b in conns[0].sent) == 1600 * 2


def test_transient_setup_failures_are_retried() -> None:
    import google.auth.exceptions
    from google.genai.errors import ClientError

    from livekit.plugins.google.beta.gemini_translation import _is_setup_error

    def _client_error(code: int) -> ClientError:
        return ClientError(code, {"error": {"code": code, "message": "x", "status": "x"}})

    # a cancelled request and rate limiting pass; a bad request does not
    assert not _is_setup_error(_client_error(499))
    assert not _is_setup_error(_client_error(429))
    assert _is_setup_error(_client_error(400))

    # a credential refresh lost on the network is worth another try
    assert not _is_setup_error(google.auth.exceptions.TransportError("connection reset"))
    refresh = google.auth.exceptions.RefreshError("refresh failed")
    refresh.__cause__ = google.auth.exceptions.TransportError("connection reset")
    assert not _is_setup_error(refresh)
    assert not _is_setup_error(google.auth.exceptions.RefreshError("x", retryable=True))

    # credentials that are simply wrong stay wrong
    assert _is_setup_error(google.auth.exceptions.RefreshError("invalid_grant"))
    assert _is_setup_error(google.auth.exceptions.DefaultCredentialsError("no credentials"))


async def test_a_transient_handshake_failure_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    import google.auth.exceptions

    conns: list[_FakeLiveSession] = []
    attempts = 0

    @contextlib.asynccontextmanager
    async def _connect(
        self: GeminiTranslationSession, config: types.LiveConnectConfig
    ) -> AsyncIterator[_FakeLiveSession]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise google.auth.exceptions.TransportError("connection reset")
        conn = _FakeLiveSession(config)
        conns.append(conn)
        yield conn

    monkeypatch.setattr(GeminiTranslationSession, "_connect", _connect)
    session = _model(conn_options=APIConnectOptions(max_retry=3, retry_interval=0)).session()
    errors: list[TranslationModelError] = []
    session.on("error", errors.append)
    try:
        await _wait_for(lambda: bool(conns))
        assert [e.recoverable for e in errors] == [True]
    finally:
        await session.aclose()


async def test_punctuation_trailing_past_the_quiet_does_not_open_the_next_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conns = _connect_hook(monkeypatch)
    session = _model(transcript_segment_timeout=0.05).session()
    heard: list[InputTranscript] = []
    session.on("input_transcript", heard.append)
    try:
        await _wait_for(lambda: bool(conns))
        conns[0].emit(input_transcription=types.Transcription(text="where is the restroom"))
        await _wait_for(lambda: any(t.is_final for t in heard))
        # the question mark arrives after the quiet ended the sentence, with the next one
        conns[0].emit(input_transcription=types.Transcription(text="? What are you saying?"))
        await _wait_for(lambda: sum(t.is_final for t in heard) == 2)

        assert [t.text for t in heard if t.is_final] == [
            "where is the restroom",
            "What are you saying?",
        ]
    finally:
        await session.aclose()


async def test_utterances_apart_by_more_than_the_quiet_stay_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conns = _connect_hook(monkeypatch)
    session = _model(transcript_segment_timeout=0.05).session()
    heard: list[InputTranscript] = []
    session.on("input_transcript", heard.append)
    try:
        await _wait_for(lambda: bool(conns))
        # neither carries punctuation or a finished mark
        conns[0].emit(input_transcription=types.Transcription(text="hello"))
        await asyncio.sleep(0.1)
        conns[0].emit(input_transcription=types.Transcription(text="goodbye"))
        await _wait_for(lambda: sum(t.is_final for t in heard) == 2)

        assert [t.text for t in heard if t.is_final] == ["hello", "goodbye"]
    finally:
        await session.aclose()
