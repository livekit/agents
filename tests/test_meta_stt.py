from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Callable
from typing import cast

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, APIStatusError, stt
from livekit.plugins import meta
from livekit.plugins.meta import stt as meta_stt

pytestmark = pytest.mark.unit

_TEST_OPTIONS = APIConnectOptions(max_retry=0, retry_interval=0.0, timeout=1.0)
_CLOSE_AFTER_END = object()
_CLOSE_1011_AFTER_END = object()


class _FakeWebSocket:
    def __init__(
        self,
        incoming: list[object] | None = None,
        *,
        block_audio: bool = False,
        send_error: BaseException | None = None,
    ) -> None:
        self._incoming: asyncio.Queue[object] = asyncio.Queue()
        for item in incoming or []:
            self._incoming.put_nowait(item)
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []
        self.handshake_sent = asyncio.Event()
        self.handshake_accepted = asyncio.Event()
        self.audio_started = asyncio.Event()
        self.release_audio = asyncio.Event()
        self.end_stream_sent = asyncio.Event()
        self.closed = False
        self.close_code: int | None = None
        self._send_error = send_error
        if not block_audio:
            self.release_audio.set()

    def queue_json(self, payload: dict[str, object]) -> None:
        self._incoming.put_nowait(_text_message(payload))

    async def send_str(self, data: str) -> None:
        self.sent_text.append(data)
        payload = json.loads(data)
        if "authorization" in payload:
            self.handshake_sent.set()
        if payload.get("type") == "endStream":
            self.end_stream_sent.set()

    async def send_bytes(self, data: bytes) -> None:
        assert self.handshake_accepted.is_set(), "audio was sent before handshake acknowledgement"
        self.audio_started.set()
        await self.release_audio.wait()
        if self._send_error is not None:
            raise self._send_error
        self.sent_bytes.append(bytes(data))

    async def receive(self) -> aiohttp.WSMessage:
        item = await self._incoming.get()
        if item is _CLOSE_AFTER_END:
            await self.end_stream_sent.wait()
            self.close_code = 1000
            return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, 1000, "")
        if item is _CLOSE_1011_AFTER_END:
            await self.end_stream_sent.wait()
            self.close_code = 1011
            return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, 1011, "unsafe close reason")
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, aiohttp.WSMessage)
        if item.type == aiohttp.WSMsgType.TEXT:
            try:
                payload = json.loads(item.data)
            except (json.JSONDecodeError, TypeError):
                pass
            else:
                if isinstance(payload, dict) and payload.get("sessionId"):
                    self.handshake_accepted.set()
        if (
            item.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED)
            and self.close_code is None
        ):
            self.close_code = item.data if isinstance(item.data, int) else 1000
        return item

    async def close(self) -> bool:
        self.closed = True
        self.release_audio.set()
        return True


class _FakeSession:
    def __init__(self, outcomes: list[_FakeWebSocket | BaseException]) -> None:
        self._outcomes = deque(outcomes)
        self.connect_calls: list[tuple[str, dict[str, object]]] = []

    async def ws_connect(self, url: str, **kwargs: object) -> _FakeWebSocket:
        self.connect_calls.append((url, kwargs))
        outcome = self._outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _text_message(payload: dict[str, object]) -> aiohttp.WSMessage:
    return aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(payload), "")


def _raw_text_message(payload: str) -> aiohttp.WSMessage:
    return aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, payload, "")


def _close_message(code: int = 1000, reason: str = "") -> aiohttp.WSMessage:
    return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, code, reason)


def _websocket(events: list[dict[str, object]] | None = None) -> _FakeWebSocket:
    return _FakeWebSocket(
        [
            _text_message({"sessionId": "session-1"}),
            *(_text_message(event) for event in events or []),
            _CLOSE_AFTER_END,
        ]
    )


def _frame(data: bytes, *, sample_rate: int = 24_000, num_channels: int = 1) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=len(data) // (2 * num_channels),
    )


async def _wait_until(predicate: Callable[[], bool], *, timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0)


async def _collect(
    provider: meta.STT,
    audio: bytes = b"",
    *,
    language: str | None = None,
    conn_options: APIConnectOptions = _TEST_OPTIONS,
) -> tuple[list[stt.SpeechEvent], meta.SpeechStream]:
    stream = (
        provider.stream(language=language, conn_options=conn_options)
        if language is not None
        else provider.stream(conn_options=conn_options)
    )
    if audio:
        stream.push_frame(_frame(audio))
    stream.end_input()
    events = [event async for event in stream]
    await stream.aclose()
    return events, stream


def _speech_events(events: list[stt.SpeechEvent]) -> list[stt.SpeechEvent]:
    return [event for event in events if event.type != stt.SpeechEventType.RECOGNITION_USAGE]


def _event_summary(events: list[stt.SpeechEvent]) -> list[tuple[stt.SpeechEventType, str, str]]:
    return [
        (
            event.type,
            event.request_id,
            event.alternatives[0].text if event.alternatives else "",
        )
        for event in _speech_events(events)
    ]


def test_construction_capabilities_and_package_exports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MODEL_API_KEY"):
        meta.STT()

    monkeypatch.setenv("MODEL_API_KEY", "environment-key")
    provider = meta.STT(http_session=cast(aiohttp.ClientSession, _FakeSession([])))
    explicit = meta.STT(
        api_key="explicit-key", http_session=cast(aiohttp.ClientSession, _FakeSession([]))
    )

    assert provider._api_key == "environment-key"
    assert explicit._api_key == "explicit-key"
    assert meta.__version__
    assert meta.STT is meta_stt.STT
    assert meta.SpeechStream is meta_stt.SpeechStream
    assert provider.model == "muse-voice-transcribe-1.0"
    assert provider.provider == "Meta"
    assert provider.capabilities == stt.STTCapabilities(
        streaming=True,
        interim_results=True,
        diarization=False,
        aligned_transcript=False,
        offline_recognize=False,
        keyterms=False,
    )


@pytest.mark.parametrize(
    "url",
    [
        "ws://api.meta.ai/v1/asr/realtime",
        "https://api.meta.ai",
        "/relative",
        "wss://user:secret@api.meta.ai/v1/asr/realtime",
        "wss://api.meta.ai/v1/asr/realtime#secret",
    ],
)
def test_rejects_non_secure_or_relative_websocket_urls(url: str) -> None:
    with pytest.raises(ValueError, match="wss"):
        meta.STT(api_key="test-key", url=url)


def test_normalizes_and_validates_static_hints() -> None:
    provider = meta.STT(
        api_key="test-key",
        keywords=[" Muse ", "Muse"],
        language_bias=[" en ", "en"],
        http_session=cast(aiohttp.ClientSession, _FakeSession([])),
    )
    assert provider._keywords == ["Muse"]
    assert provider._language_bias == ["en"]

    with pytest.raises(ValueError, match="keywords entries"):
        meta.STT(api_key="test-key", keywords=[" "])
    with pytest.raises(ValueError, match="language_bias entries"):
        meta.STT(api_key="test-key", language_bias=[""])


def test_normalizes_stream_language_to_primary_subtag() -> None:
    assert meta.STT._normalize_language_hint("pt_BR") == "pt"
    assert meta.STT._normalize_language_hint("EN-us") == "en"

    with pytest.raises(ValueError, match="language code"):
        meta.STT._normalize_language_hint("not a language")


async def test_batch_recognition_is_rejected_as_non_retryable() -> None:
    provider = meta.STT(
        api_key="test-key", http_session=cast(aiohttp.ClientSession, _FakeSession([]))
    )

    with pytest.raises(APIError, match="streaming recognition only") as exc_info:
        await provider.recognize([], conn_options=_TEST_OPTIONS)

    assert exc_info.value.retryable is False
    await provider.aclose()


async def test_handshake_precedes_audio_and_matches_contract() -> None:
    websocket = _FakeWebSocket()
    session = _FakeSession([websocket])
    provider = meta.STT(
        api_key="explicit-secret",
        keywords=["Muse", "Muse"],
        language_bias=["en", "fr"],
        http_session=cast(aiohttp.ClientSession, session),
    )
    stream = provider.stream(language="fr-FR", conn_options=_TEST_OPTIONS)
    stream.push_frame(_frame(b"x" * 3840))

    await _wait_until(websocket.handshake_sent.is_set)
    assert websocket.sent_bytes == []
    assert stream._audio_consumed is False

    websocket.queue_json({"sessionId": "session-1"})
    await _wait_until(websocket.audio_started.is_set)
    stream.end_input()
    websocket._incoming.put_nowait(_CLOSE_AFTER_END)
    events = [event async for event in stream]

    assert json.loads(websocket.sent_text[0]) == {
        "mode": "ENDPOINTING",
        "authorization": {"accessToken": "explicit-secret"},
        "audioEncoding": "PCM_24KHZ",
        "model": "muse-voice-transcribe-1.0",
        "partialMode": "CUMULATIVE",
        "emitAudioProgress": True,
        "keywords": ["Muse"],
        "languageBias": ["en", "fr"],
    }
    url, kwargs = session.connect_calls[0]
    assert url == "wss://api.meta.ai/v1/asr/realtime"
    assert "explicit-secret" not in url
    assert "explicit-secret" not in repr(kwargs)
    assert [event.recognition_usage.audio_duration for event in events] == [0.08]

    await stream.aclose()
    await provider.aclose()


async def test_packetizes_pcm16_into_80ms_chunks_and_preserves_tail() -> None:
    websocket = _websocket()
    provider = meta.STT(
        api_key="test-key", http_session=cast(aiohttp.ClientSession, _FakeSession([websocket]))
    )

    events, _ = await _collect(provider, b"x" * (3840 * 2 + 100))

    assert [len(packet) for packet in websocket.sent_bytes] == [3840, 3840, 100]
    assert websocket.sent_bytes == [b"x" * 3840, b"x" * 3840, b"x" * 100]
    usage = [
        event.recognition_usage.audio_duration
        for event in events
        if event.type == stt.SpeechEventType.RECOGNITION_USAGE
        and event.recognition_usage is not None
    ]
    assert usage == pytest.approx([0.08, 0.08, 100 / 48_000])
    assert sum(usage) == pytest.approx((3840 * 2 + 100) / 48_000)
    await provider.aclose()


async def test_resamples_48khz_mono_input_to_24khz_wire_audio() -> None:
    websocket = _websocket()
    provider = meta.STT(
        api_key="test-key", http_session=cast(aiohttp.ClientSession, _FakeSession([websocket]))
    )
    stream = provider.stream(conn_options=_TEST_OPTIONS)
    stream.push_frame(_frame(b"\0" * (48_000 * 2 // 5), sample_rate=48_000))
    stream.end_input()
    events = [event async for event in stream]

    assert sum(map(len, websocket.sent_bytes)) == 24_000 * 2 // 5
    assert [len(packet) for packet in websocket.sent_bytes] == [3840, 3840, 1920]
    usage = [
        event.recognition_usage.audio_duration
        for event in events
        if event.recognition_usage is not None
    ]
    assert sum(usage) == pytest.approx(0.2)
    await stream.aclose()
    await provider.aclose()


async def test_flush_sends_tail_without_ending_and_audio_can_continue() -> None:
    websocket = _websocket()
    provider = meta.STT(
        api_key="test-key", http_session=cast(aiohttp.ClientSession, _FakeSession([websocket]))
    )
    stream = provider.stream(conn_options=_TEST_OPTIONS)

    stream.push_frame(_frame(b"a" * 1000))
    stream.flush()
    await _wait_until(lambda: websocket.sent_bytes == [b"a" * 1000])
    assert all(json.loads(message).get("type") != "endStream" for message in websocket.sent_text)

    stream.push_frame(_frame(b"b" * 3840))
    stream.end_input()
    events = [event async for event in stream]

    assert websocket.sent_bytes == [b"a" * 1000, b"b" * 3840]
    assert [json.loads(message).get("type") for message in websocket.sent_text].count(
        "endStream"
    ) == 1
    usage = [
        event.recognition_usage.audio_duration
        for event in events
        if event.type == stt.SpeechEventType.RECOGNITION_USAGE
        and event.recognition_usage is not None
    ]
    assert sum(usage) == pytest.approx(4840 / 48_000)

    await stream.aclose()
    await provider.aclose()


async def test_rejects_stereo_after_input_is_consumed_without_retry() -> None:
    websocket = _websocket()
    session = _FakeSession([websocket, _websocket()])
    provider = meta.STT(api_key="test-key", http_session=cast(aiohttp.ClientSession, session))
    stream = provider.stream(
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0.0, timeout=1.0)
    )
    stream.push_frame(_frame(b"\0" * 7680, num_channels=2))
    stream.end_input()

    with pytest.raises(APIError, match="mono audio") as exc_info:
        async for _ in stream:
            pass

    assert exc_info.value.retryable is False
    assert len(session.connect_calls) == 1
    await stream.aclose()
    await provider.aclose()


async def test_absolute_pacing_accounts_for_send_time(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Clock:
        now = 0.0
        sleeps: list[float] = []

        def time(self) -> float:
            return self.now

        async def sleep(self, delay: float) -> None:
            self.sleeps.append(delay)
            self.now += delay

    class _PacingWebSocket:
        def __init__(self, clock: _Clock) -> None:
            self.clock = clock
            self.packets: list[bytes] = []
            self.text: list[str] = []

        async def send_bytes(self, packet: bytes) -> None:
            self.packets.append(packet)
            self.clock.now += 0.03

        async def send_str(self, payload: str) -> None:
            self.text.append(payload)

    async def input_items() -> AsyncIterator[rtc.AudioFrame]:
        yield _frame(b"x" * (3840 * 3))

    class _EventSink:
        def __init__(self) -> None:
            self.events: list[stt.SpeechEvent] = []

        def send_nowait(self, event: stt.SpeechEvent) -> None:
            self.events.append(event)

    clock = _Clock()
    websocket = _PacingWebSocket(clock)
    stream = object.__new__(meta.SpeechStream)
    stream._input_ch = input_items()
    stream._audio_consumed = False
    stream._end_stream_sent = False
    stream._session_id = "session-1"
    stream._event_ch = _EventSink()
    monkeypatch.setattr(meta_stt.asyncio, "get_running_loop", lambda: clock)
    monkeypatch.setattr(meta_stt.asyncio, "sleep", clock.sleep)

    await stream._send_audio(cast(aiohttp.ClientWebSocketResponse, websocket))

    assert [len(packet) for packet in websocket.packets] == [3840, 3840, 3840]
    assert clock.sleeps == pytest.approx([0.05, 0.05])
    assert websocket.text == ['{"type":"endStream"}']
    assert [
        event.recognition_usage.audio_duration for event in stream._event_ch.events
    ] == pytest.approx([0.08, 0.08, 0.08])


async def test_audio_progress_does_not_duplicate_sent_byte_usage() -> None:
    websocket = _websocket(
        [
            {"type": "audioProgress", "audioProcessedMs": 80},
            {"type": "audioProgress", "audioProcessedMs": 80},
            {"type": "audioProgress", "audioProcessedMs": 160},
        ]
    )
    provider = meta.STT(
        api_key="test-key", http_session=cast(aiohttp.ClientSession, _FakeSession([websocket]))
    )

    events, _ = await _collect(provider, b"x" * 7680)

    usage = [
        event.recognition_usage.audio_duration
        for event in events
        if event.recognition_usage is not None
    ]
    assert usage == pytest.approx([0.08, 0.08])
    await provider.aclose()


async def test_normal_turn_deduplicates_partials_final_and_end_then_evicts_state() -> None:
    events = [
        {"type": "speechStart", "turnId": 1},
        {"type": "transcript", "transcript": "hel"},
        {"type": "transcript", "turnId": 1, "transcript": "hel"},
        {"type": "transcript", "turnId": 1, "transcript": "hello"},
        {"type": "speechEnd", "turnId": 1},
        {"type": "speechEnd", "turnId": 1},
        {"type": "speechComplete", "turnId": 1, "transcript": "hello there"},
        {"type": "speechComplete", "turnId": 1, "transcript": "duplicate"},
        {"type": "speechEnd", "turnId": 1},
    ]
    websocket = _websocket(events)
    provider = meta.STT(
        api_key="test-key", http_session=cast(aiohttp.ClientSession, _FakeSession([websocket]))
    )

    collected, stream = await _collect(provider)

    assert _event_summary(collected) == [
        (stt.SpeechEventType.START_OF_SPEECH, "1", ""),
        (stt.SpeechEventType.INTERIM_TRANSCRIPT, "1", "hel"),
        (stt.SpeechEventType.INTERIM_TRANSCRIPT, "1", "hello"),
        (stt.SpeechEventType.FINAL_TRANSCRIPT, "1", "hello there"),
        (stt.SpeechEventType.END_OF_SPEECH, "1", ""),
    ]
    assert stream._turns == {}
    assert stream._completed_turn_ids == {"1"}
    assert list(stream._completed_turn_order) == ["1"]
    final = _speech_events(collected)[-2]
    assert final.alternatives[0].language == ""
    await provider.aclose()


def test_completed_turn_tombstones_are_bounded() -> None:
    stream = object.__new__(meta.SpeechStream)
    stream._completed_turn_ids = set()
    stream._completed_turn_order = deque()

    for index in range(meta_stt._MAX_COMPLETED_TURNS + 2):
        stream._remember_completed_turn(str(index))

    assert len(stream._completed_turn_ids) == meta_stt._MAX_COMPLETED_TURNS
    assert len(stream._completed_turn_order) == meta_stt._MAX_COMPLETED_TURNS
    assert "0" not in stream._completed_turn_ids
    assert "1" not in stream._completed_turn_ids
    assert str(meta_stt._MAX_COMPLETED_TURNS + 1) in stream._completed_turn_ids


async def test_interleaved_turns_are_globally_serialized() -> None:
    events = [
        {"type": "speechStart", "turnId": "first"},
        {"type": "transcript", "turnId": "first", "transcript": "one"},
        {"type": "speechEnd", "turnId": "first"},
        {"type": "speechStart", "turnId": "second"},
        {"type": "transcript", "transcript": "two"},
        {"type": "speechEnd", "turnId": "second"},
        {"type": "speechComplete", "turnId": "second", "transcript": "turn two"},
        {"type": "speechComplete", "turnId": "first", "transcript": "turn one"},
    ]
    websocket = _websocket(events)
    provider = meta.STT(
        api_key="test-key", http_session=cast(aiohttp.ClientSession, _FakeSession([websocket]))
    )

    collected, _ = await _collect(provider)

    assert _event_summary(collected) == [
        (stt.SpeechEventType.START_OF_SPEECH, "first", ""),
        (stt.SpeechEventType.INTERIM_TRANSCRIPT, "first", "one"),
        (stt.SpeechEventType.FINAL_TRANSCRIPT, "first", "turn one"),
        (stt.SpeechEventType.END_OF_SPEECH, "first", ""),
        (stt.SpeechEventType.START_OF_SPEECH, "second", ""),
        (stt.SpeechEventType.INTERIM_TRANSCRIPT, "second", "two"),
        (stt.SpeechEventType.FINAL_TRANSCRIPT, "second", "turn two"),
        (stt.SpeechEventType.END_OF_SPEECH, "second", ""),
    ]
    await provider.aclose()


async def test_empty_turnless_transcript_outside_speech_is_ignored() -> None:
    websocket = _websocket(
        [{"type": "transcript", "transcript": "", "final": False, "audioProcessedMs": 1000}]
    )
    provider = meta.STT(
        api_key="test-key", http_session=cast(aiohttp.ClientSession, _FakeSession([websocket]))
    )

    collected, _ = await _collect(provider)

    assert _speech_events(collected) == []
    await provider.aclose()


async def test_missing_turn_id_is_rejected_outside_an_active_turn() -> None:
    websocket = _websocket(
        [
            {"type": "speechStart", "turnId": 1},
            {"type": "speechEnd", "turnId": 1},
            {"type": "transcript", "transcript": "unsafe transcript body"},
        ]
    )
    provider = meta.STT(
        api_key="test-key", http_session=cast(aiohttp.ClientSession, _FakeSession([websocket]))
    )
    stream = provider.stream(conn_options=_TEST_OPTIONS)
    stream.end_input()

    with pytest.raises(APIError, match="missing turnId outside an active turn") as exc_info:
        async for _ in stream:
            pass

    assert exc_info.value.retryable is False
    assert "unsafe transcript body" not in str(exc_info.value)
    await stream.aclose()
    await provider.aclose()


async def test_raw_normal_close_code_wins_over_stale_socket_property() -> None:
    websocket = _FakeWebSocket(
        [
            _text_message({"sessionId": "session-1"}),
            _text_message(
                {
                    "type": "transcript",
                    "transcript": "",
                    "final": False,
                    "audioProcessedMs": 1000,
                }
            ),
            _CLOSE_AFTER_END,
        ]
    )
    websocket.close_code = 1006
    provider = meta.STT(
        api_key="test-key", http_session=cast(aiohttp.ClientSession, _FakeSession([websocket]))
    )

    events, _ = await _collect(provider)

    assert events == []
    await provider.aclose()


async def test_clean_close_with_incomplete_turn_is_non_retryable() -> None:
    websocket = _websocket([{"type": "speechStart", "turnId": "unfinished"}])
    provider = meta.STT(
        api_key="test-key", http_session=cast(aiohttp.ClientSession, _FakeSession([websocket]))
    )
    stream = provider.stream(conn_options=_TEST_OPTIONS)
    stream.end_input()

    with pytest.raises(APIError, match="incomplete speech turns") as exc_info:
        async for _ in stream:
            pass

    assert exc_info.value.retryable is False
    await stream.aclose()
    await provider.aclose()


async def test_pre_audio_connection_failure_retries_without_losing_input() -> None:
    websocket = _websocket()
    session = _FakeSession([RuntimeError("transient connection detail"), websocket])
    provider = meta.STT(api_key="test-key", http_session=cast(aiohttp.ClientSession, session))

    events, _ = await _collect(
        provider,
        b"x" * 3840,
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0.0, timeout=1.0),
    )

    assert len(session.connect_calls) == 2
    assert websocket.sent_bytes == [b"x" * 3840]
    assert [
        event.recognition_usage.audio_duration
        for event in events
        if event.recognition_usage is not None
    ] == [0.08]
    await provider.aclose()


async def test_empty_input_retry_sends_end_stream_on_every_attempt() -> None:
    first = _FakeWebSocket([_text_message({"sessionId": "session-1"}), _CLOSE_1011_AFTER_END])
    second = _websocket()
    session = _FakeSession([first, second])
    provider = meta.STT(api_key="test-key", http_session=cast(aiohttp.ClientSession, session))

    events, _ = await _collect(
        provider,
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0.0, timeout=1.0),
    )

    assert events == []
    assert len(session.connect_calls) == 2
    for websocket in (first, second):
        assert [json.loads(message).get("type") for message in websocket.sent_text].count(
            "endStream"
        ) == 1
    await provider.aclose()


async def test_post_consumption_failure_is_not_retried() -> None:
    websocket = _FakeWebSocket(
        [_text_message({"sessionId": "session-1"})],
        send_error=RuntimeError("secret send failure"),
    )
    session = _FakeSession([websocket, _websocket()])
    provider = meta.STT(api_key="test-key", http_session=cast(aiohttp.ClientSession, session))
    stream = provider.stream(
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0.0, timeout=1.0)
    )
    stream.push_frame(_frame(b"x" * 3840))
    stream.end_input()

    with pytest.raises(APIError) as exc_info:
        async for _ in stream:
            pass

    assert exc_info.value.retryable is False
    assert "secret send failure" not in str(exc_info.value)
    assert len(session.connect_calls) == 1
    await stream.aclose()
    await provider.aclose()


async def test_aclose_during_handshake_closes_socket() -> None:
    websocket = _FakeWebSocket()
    provider = meta.STT(
        api_key="test-key", http_session=cast(aiohttp.ClientSession, _FakeSession([websocket]))
    )
    stream = provider.stream(conn_options=_TEST_OPTIONS)
    await _wait_until(websocket.handshake_sent.is_set)

    await asyncio.wait_for(stream.aclose(), timeout=1.0)

    assert websocket.closed is True
    await provider.aclose()


async def test_aclose_cancels_blocked_transport_tasks_and_closes_socket() -> None:
    websocket = _FakeWebSocket(
        [_text_message({"sessionId": "session-1"})],
        block_audio=True,
    )
    provider = meta.STT(
        api_key="test-key", http_session=cast(aiohttp.ClientSession, _FakeSession([websocket]))
    )
    stream = provider.stream(conn_options=_TEST_OPTIONS)
    stream.push_frame(_frame(b"x" * 3840))
    await _wait_until(websocket.audio_started.is_set)

    await asyncio.wait_for(stream.aclose(), timeout=1.0)

    assert stream._task.done()
    assert stream._metrics_task.done()
    assert websocket.closed is True
    await provider.aclose()


@pytest.mark.parametrize(
    ("secret", "session"),
    [
        ("connect-secret", _FakeSession([RuntimeError("connect-secret")])),
        (
            "provider-secret",
            _FakeSession(
                [
                    _FakeWebSocket(
                        [
                            _text_message({"sessionId": "session-1"}),
                            _text_message(
                                {
                                    "type": "error",
                                    "errorCode": "invalid_request",
                                    "message": "provider-secret",
                                }
                            ),
                        ]
                    )
                ]
            ),
        ),
        (
            "close-secret",
            _FakeSession(
                [
                    _FakeWebSocket(
                        [
                            _text_message({"sessionId": "session-1"}),
                            _close_message(1011, "close-secret"),
                        ]
                    )
                ]
            ),
        ),
        (
            "payload-secret",
            _FakeSession(
                [
                    _FakeWebSocket(
                        [
                            _text_message({"sessionId": "session-1"}),
                            _raw_text_message("{payload-secret"),
                        ]
                    )
                ]
            ),
        ),
    ],
)
async def test_failures_redact_transport_and_provider_details(
    secret: str,
    session: _FakeSession,
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = meta.STT(
        api_key="api-key-secret",
        http_session=cast(aiohttp.ClientSession, session),
    )
    stream = provider.stream(conn_options=_TEST_OPTIONS)

    with pytest.raises(APIError) as exc_info:
        async for _ in stream:
            pass

    rendered = f"{exc_info.value!s}\n{exc_info.value!r}\n{caplog.text}"
    assert secret not in rendered
    assert "api-key-secret" not in rendered
    assert exc_info.value.body is None
    assert exc_info.value.__cause__ is None
    await stream.aclose()
    await provider.aclose()


async def test_handshake_rejection_redacts_token_and_body() -> None:
    websocket = _FakeWebSocket(
        [
            _text_message(
                {
                    "type": "error",
                    "errorCode": "unauthorized",
                    "message": "credential api-key-secret rejected",
                }
            )
        ]
    )
    provider = meta.STT(
        api_key="api-key-secret",
        http_session=cast(aiohttp.ClientSession, _FakeSession([websocket])),
    )
    stream = provider.stream(conn_options=_TEST_OPTIONS)

    with pytest.raises(APIStatusError) as exc_info:
        async for _ in stream:
            pass

    assert exc_info.value.status_code == 400
    assert exc_info.value.body is None
    assert "api-key-secret" not in str(exc_info.value)
    assert "credential" not in str(exc_info.value)
    assert websocket.closed is True
    await stream.aclose()
    await provider.aclose()
