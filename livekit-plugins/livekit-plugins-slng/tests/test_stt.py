from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import aiohttp
import pytest

from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError
from livekit.agents.stt import SpeechEventType
from livekit.plugins import slng

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("model", ["", "nova", "deepgram/", "/nova:3", "deepgram/nova:"])
def test_stt_rejects_invalid_explicit_model(model: str) -> None:
    with pytest.raises(ValueError, match="provider/model"):
        slng.STT(api_key="test-key", model=model)


def test_stt_builds_unmute_bridge_connection() -> None:
    stt = slng.STT(api_key="test-key", model="deepgram/nova:3")
    assert stt._connections[0].endpoint == (
        "wss://api.slng.ai/v1/bridges/unmute/stt/deepgram/nova:3"
    )


def test_stt_accepts_endpoint_without_model() -> None:
    endpoint = "wss://api.slng.ai/v1/bridges/unmute/stt/deepgram/nova:3"
    stt = slng.STT(api_key="test-key", connections=[endpoint])
    assert stt._model == "deepgram/nova:3"


def test_stt_rejects_direct_provider_endpoint() -> None:
    with pytest.raises(ValueError, match="Unmute Bridge"):
        slng.STT(
            api_key="test-key",
            connections=["wss://api.slng.ai/v1/stt/deepgram/nova:3"],
        )


def test_stt_connection_config_keeps_endpoint_settings() -> None:
    endpoint = "wss://api.slng.ai/v1/bridges/unmute/stt/deepgram/nova:3"
    stt = slng.STT(
        api_key="test-key",
        connections=[
            slng.STTConnectionConfig(
                endpoint=endpoint,
                headers={"X-Test": "yes"},
                init={"type": "init", "config": {"language": "en"}},
            )
        ],
    )
    assert stt._connections[0].headers == {"X-Test": "yes"}
    assert stt._connections[0].init == {
        "type": "init",
        "config": {"language": "en"},
    }


@pytest.mark.asyncio
async def test_stt_exhausted_failure_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_connect(self, *, model_endpoint: str, model: str | None):
        del self, model_endpoint, model
        raise APIStatusError("temporary failure")

    monkeypatch.setattr(slng.stt.SpeechStream, "_connect_ws", fail_connect)
    async with aiohttp.ClientSession() as session:
        stt = slng.STT(
            api_key="test-key",
            model="deepgram/nova:3",
            http_session=session,
        )
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))
        stream._input_ch.send_nowait(SimpleNamespace(data=memoryview(b"\x01\x02")))
        stream.end_input()

        async def drain() -> None:
            async for _ in stream:
                pass

        with pytest.raises(APIStatusError, match="temporary failure"):
            await asyncio.wait_for(drain(), timeout=1)


def test_stt_api_token_alias_warns_and_authenticates() -> None:
    with pytest.warns(DeprecationWarning, match="api_token"):
        stt = slng.STT(api_token="legacy-key", model="deepgram/nova:3")
    assert stt._api_token == "legacy-key"
    assert "api_token" not in stt._model_options


def test_stt_rejects_removed_endpoint_kwargs() -> None:
    with pytest.raises(ValueError, match="Unmute Bridge"):
        slng.STT(
            api_key="test-key",
            model="deepgram/nova:3",
            model_endpoint="wss://api.slng.ai/v1/stt/deepgram/nova:3",
        )
    with pytest.raises(ValueError, match="Unmute Bridge"):
        slng.STT(
            api_key="test-key",
            model="deepgram/nova:3",
            model_endpoints=["wss://api.slng.ai/v1/stt/deepgram/nova:3"],
        )


def test_stt_world_part_override_becomes_header() -> None:
    stt = slng.STT(
        api_key="test-key",
        model="deepgram/nova:3",
        world_part_override="EU",
    )
    assert stt._extra_headers["X-World-Part-Override"] == "eu"


def test_stt_provider_api_key_becomes_byok_header() -> None:
    stt = slng.STT(
        api_key="test-key",
        model="deepgram/nova:3",
        provider_api_key="provider-secret",
    )
    assert stt._extra_headers["X-Slng-Provider-Key"] == "provider-secret"


@pytest.mark.asyncio
async def test_stt_413_is_terminal_without_chain_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[str] = []

    async def fail_connect(self, *, model_endpoint: str, model: str | None):
        del self, model
        attempts.append(model_endpoint)
        raise APIStatusError("payload too large", status_code=413)

    monkeypatch.setattr(slng.stt.SpeechStream, "_connect_ws", fail_connect)
    async with aiohttp.ClientSession() as session:
        stt = slng.STT(
            api_key="test-key",
            connections=["deepgram/nova:3", "deepgram/nova:2"],
            http_session=session,
        )
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))
        stream._input_ch.send_nowait(SimpleNamespace(data=memoryview(b"\x01\x02")))
        stream.end_input()

        async def drain() -> None:
            async for _ in stream:
                pass

        with pytest.raises(APIStatusError, match="payload too large"):
            await asyncio.wait_for(drain(), timeout=1)
        assert len(attempts) == 1


class _ScriptedSttWs:
    """Fake bridge websocket: records frames, closes after protocol close."""

    def __init__(self) -> None:
        self.sent_texts: list[dict] = []
        self.audio_frames = 0
        self._closed = asyncio.Event()

    async def send_str(self, msg: str) -> None:
        frame = json.loads(msg)
        self.sent_texts.append(frame)
        if frame.get("type") == "close":
            self._closed.set()

    async def send_bytes(self, payload: bytes) -> None:
        del payload
        self.audio_frames += 1

    async def receive(self, timeout=None):
        del timeout
        await self._closed.wait()
        return SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None)

    async def close(self) -> None:
        self._closed.set()


@pytest.mark.asyncio
async def test_stt_sends_finalize_on_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _ScriptedSttWs()

    async def fake_connect(self, *, model_endpoint: str, model: str | None):
        del self, model_endpoint, model
        return ws

    monkeypatch.setattr(slng.stt.SpeechStream, "_connect_ws", fake_connect)
    async with aiohttp.ClientSession() as session:
        stt = slng.STT(api_key="test-key", model="deepgram/nova:3", http_session=session)
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))
        stream._input_ch.send_nowait(SimpleNamespace(data=memoryview(b"\x00\x01" * 1600)))
        stream.flush()
        stream.end_input()

        async def drain() -> None:
            async for _ in stream:
                pass

        await asyncio.wait_for(drain(), timeout=2)

    finalizes = [f for f in ws.sent_texts if f.get("type") == "finalize"]
    assert len(finalizes) == 1
    assert ws.audio_frames > 0


@pytest.mark.asyncio
async def test_stt_recognize_is_unsupported() -> None:
    stt = slng.STT(api_key="test-key", model="deepgram/nova:3")
    with pytest.raises(NotImplementedError, match="stream"):
        await stt._recognize_impl(None, conn_options=APIConnectOptions(max_retry=0))


def test_stt_rejects_mulaw_encoding() -> None:
    with pytest.raises(ValueError, match="pcm_s16le"):
        slng.STT(api_key="test-key", model="deepgram/nova:3", encoding="pcm_mulaw")


@pytest.mark.asyncio
async def test_stt_stream_language_override_is_applied() -> None:
    async with aiohttp.ClientSession() as session:
        stt = slng.STT(
            api_key="test-key",
            model="deepgram/nova:3",
            language="en",
            http_session=session,
        )
        stream = stt.stream(language="id", conn_options=APIConnectOptions(max_retry=0))
        assert stream._opts.language == "id"
        await stream.aclose()


class _DeepgramResultsSttWs:
    """Answers a finalize with a Deepgram-native Results frame."""

    def __init__(self) -> None:
        self.sent_texts: list[dict] = []
        self.audio_frames = 0
        self._q: asyncio.Queue = asyncio.Queue()

    async def send_str(self, msg: str) -> None:
        frame = json.loads(msg)
        self.sent_texts.append(frame)
        if frame.get("type") == "finalize":
            payload = {
                "type": "Results",
                "is_final": True,
                "channel": {
                    "alternatives": [
                        {
                            "transcript": "halo dunia",
                            "confidence": 0.97,
                            "language": "id",
                            "words": [
                                {"word": "halo", "start": 1.2, "end": 1.5},
                                {"word": "dunia", "start": 1.6, "end": 2.1},
                            ],
                        }
                    ]
                },
            }
            self._q.put_nowait(
                SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(payload))
            )
        if frame.get("type") == "close":
            self._q.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None))

    async def send_bytes(self, payload: bytes) -> None:
        del payload
        self.audio_frames += 1

    async def receive(self, timeout=None):
        del timeout
        return await self._q.get()

    async def close(self) -> None:
        self._q.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None))


@pytest.mark.asyncio
async def test_stt_deepgram_results_keep_words_and_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deepgram-native Results normalization must preserve word timings and
    the provider-detected language for the emitted final event."""
    ws = _DeepgramResultsSttWs()

    async def fake_connect(self, *, model_endpoint: str, model: str | None):
        del self, model_endpoint, model
        return ws

    monkeypatch.setattr(slng.stt.SpeechStream, "_connect_ws", fake_connect)
    async with aiohttp.ClientSession() as session:
        stt = slng.STT(
            api_key="test-key",
            model="deepgram/nova:3",
            language="en",
            http_session=session,
        )
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))
        stream._input_ch.send_nowait(SimpleNamespace(data=memoryview(b"\x00\x01" * 1600)))
        stream.flush()
        stream.end_input()

        finals: list = []

        async def drain() -> None:
            async for ev in stream:
                if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                    finals.append(ev.alternatives[0])

        await asyncio.wait_for(drain(), timeout=3)

    assert len(finals) == 1
    assert finals[0].text == "halo dunia"
    assert finals[0].language == "id"
    assert finals[0].start_time == 1.2
    assert finals[0].end_time == 2.1


class _DyingSttWs:
    """Closes after receiving a finalize, or audio when requested."""

    def __init__(self, *, die_after_audio: bool = False) -> None:
        self.sent_texts: list[dict] = []
        self.audio_frames = 0
        self._die = asyncio.Event()
        self._die_after_audio = die_after_audio

    async def send_str(self, msg: str) -> None:
        frame = json.loads(msg)
        self.sent_texts.append(frame)
        if frame.get("type") == "finalize":
            self._die.set()

    async def send_bytes(self, payload: bytes) -> None:
        del payload
        self.audio_frames += 1
        if self._die_after_audio:
            self._die.set()

    async def receive(self, timeout=None):
        del timeout
        await self._die.wait()
        return SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None)

    async def close(self) -> None:
        self._die.set()


class _AnsweringSttWs:
    """Answers a finalize with a final transcript, closes after protocol close."""

    def __init__(self) -> None:
        self.sent_texts: list[dict] = []
        self.audio_frames = 0
        self.audio_received = asyncio.Event()
        self._q: asyncio.Queue = asyncio.Queue()

    def _put(self, payload: dict) -> None:
        self._q.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(payload)))

    async def send_str(self, msg: str) -> None:
        frame = json.loads(msg)
        self.sent_texts.append(frame)
        if frame.get("type") == "finalize":
            self._put(
                {
                    "type": "final_transcript",
                    "transcript": "hello world",
                    "confidence": 1.0,
                }
            )
        if frame.get("type") == "close":
            self._q.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None))

    async def send_bytes(self, payload: bytes) -> None:
        del payload
        self.audio_frames += 1
        self.audio_received.set()

    async def receive(self, timeout=None):
        del timeout
        return await self._q.get()

    async def close(self) -> None:
        self._q.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None))


@pytest.mark.asyncio
async def test_stt_failover_replays_audio_and_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A close while awaiting the final fails over, and the replacement
    candidate receives both the replayed audio AND the finalize boundary."""
    ws1, ws2 = _DyingSttWs(), _AnsweringSttWs()
    sockets: list = [ws1, ws2]

    async def fake_connect(self, *, model_endpoint: str, model: str | None):
        del self, model_endpoint, model
        return sockets.pop(0)

    monkeypatch.setattr(slng.stt.SpeechStream, "_connect_ws", fake_connect)
    async with aiohttp.ClientSession() as session:
        stt = slng.STT(
            api_key="test-key",
            connections=["deepgram/nova:3", "deepgram/nova:2"],
            http_session=session,
        )
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))
        stream._input_ch.send_nowait(SimpleNamespace(data=memoryview(b"\x00\x01" * 1600)))
        stt.notify_user_state("speaking")
        stt.notify_user_state("listening")

        got_final = asyncio.Event()
        finals: list[str] = []

        async def read() -> None:
            async for ev in stream:
                if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                    finals.append(ev.alternatives[0].text)
                    got_final.set()

        reader = asyncio.create_task(read())
        await asyncio.wait_for(got_final.wait(), timeout=3)
        stream.end_input()
        await asyncio.wait_for(reader, timeout=3)

    assert finals == ["hello world"]
    assert any(f.get("type") == "finalize" for f in ws1.sent_texts)
    assert ws2.audio_frames > 0, "audio was not replayed to the fallback"
    assert any(f.get("type") == "finalize" for f in ws2.sent_texts), (
        "finalize boundary was not replayed to the fallback"
    )


@pytest.mark.asyncio
async def test_stt_failover_does_not_invent_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failover with no outstanding finalize must not replay one: a second
    connect failure must not turn buffered audio into a premature turn end."""
    first = _DyingSttWs(die_after_audio=True)
    third = _AnsweringSttWs()
    attempts = 0

    async def fake_connect(self, *, model_endpoint: str, model: str | None):
        nonlocal attempts
        del self, model_endpoint, model
        attempts += 1
        if attempts == 1:
            return first
        if attempts == 2:
            raise APIConnectionError("connect failed")
        return third

    monkeypatch.setattr(slng.stt.SpeechStream, "_connect_ws", fake_connect)
    async with aiohttp.ClientSession() as session:
        stt = slng.STT(
            api_key="test-key",
            connections=["deepgram/nova:3", "deepgram/nova:2", "deepgram/nova:1"],
            http_session=session,
        )
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))
        stream._input_ch.send_nowait(SimpleNamespace(data=memoryview(b"\x00\x01" * 1600)))

        async def drain() -> None:
            async for _ in stream:
                pass

        reader = asyncio.create_task(drain())
        await asyncio.wait_for(third.audio_received.wait(), timeout=3)
        assert not any(f.get("type") == "finalize" for f in third.sent_texts)

        stream.end_input()
        await asyncio.wait_for(reader, timeout=3)


class _InterimOnlySttWs:
    """Answers a finalize with an interim transcript and then goes silent."""

    def __init__(self) -> None:
        self.sent_texts: list[dict] = []
        self._q: asyncio.Queue = asyncio.Queue()

    async def send_str(self, msg: str) -> None:
        frame = json.loads(msg)
        self.sent_texts.append(frame)
        if frame.get("type") == "finalize":
            self._q.put_nowait(
                SimpleNamespace(
                    type=aiohttp.WSMsgType.TEXT,
                    data=json.dumps(
                        {
                            "type": "partial_transcript",
                            "transcript": "hello",
                            "confidence": 0.5,
                        }
                    ),
                )
            )

    async def send_bytes(self, payload: bytes) -> None:
        del payload

    async def receive(self, timeout=None):
        del timeout
        return await self._q.get()

    async def close(self) -> None:
        self._q.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None))


@pytest.mark.asyncio
async def test_stt_interim_does_not_disarm_final_watchdog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interim transcript after finalize must NOT cancel the final-result
    watchdog; the timeout still fires and (with one candidate) surfaces."""
    ws = _InterimOnlySttWs()

    async def fake_connect(self, *, model_endpoint: str, model: str | None):
        del self, model_endpoint, model
        return ws

    monkeypatch.setattr(slng.stt.SpeechStream, "_connect_ws", fake_connect)
    async with aiohttp.ClientSession() as session:
        stt = slng.STT(
            api_key="test-key",
            model="deepgram/nova:3",
            final_timeout_s=0.3,
            http_session=session,
        )
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))
        stream._input_ch.send_nowait(SimpleNamespace(data=memoryview(b"\x00\x01" * 1600)))
        stt.notify_user_state("speaking")
        stt.notify_user_state("listening")

        async def drain() -> None:
            async for _ in stream:
                pass

        with pytest.raises((TimeoutError, APIConnectionError)):
            await asyncio.wait_for(drain(), timeout=3)


class _SilentSttWs:
    """Records frames, never answers a finalize."""

    def __init__(self) -> None:
        self.sent_texts: list[dict] = []
        self.audio_frames = 0
        self._q: asyncio.Queue = asyncio.Queue()

    async def send_str(self, msg: str) -> None:
        self.sent_texts.append(json.loads(msg))

    async def send_bytes(self, payload: bytes) -> None:
        del payload
        self.audio_frames += 1

    async def receive(self, timeout=None):
        del timeout
        return await self._q.get()

    async def close(self) -> None:
        self._q.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None))


@pytest.mark.asyncio
async def test_stt_options_reconnect_disarms_final_watchdog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An options-change reconnect while awaiting a final transcript must
    disarm the watchdog; the stale timer must not fail over the new
    connection."""
    ws1, ws2 = _SilentSttWs(), _AnsweringSttWs()
    sockets: list = [ws1, ws2]

    async def fake_connect(self, *, model_endpoint: str, model: str | None):
        del self, model_endpoint, model
        return sockets.pop(0)

    monkeypatch.setattr(slng.stt.SpeechStream, "_connect_ws", fake_connect)
    async with aiohttp.ClientSession() as session:
        stt = slng.STT(
            api_key="test-key",
            model="deepgram/nova:3",
            final_timeout_s=0.4,
            http_session=session,
        )
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))

        got_final = asyncio.Event()
        finals: list[str] = []

        async def read() -> None:
            async for ev in stream:
                if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                    finals.append(ev.alternatives[0].text)
                    got_final.set()

        reader = asyncio.create_task(read())

        # First turn: finalize goes to ws1, which never answers.
        stream._input_ch.send_nowait(SimpleNamespace(data=memoryview(b"\x00\x01" * 1600)))
        stt.notify_user_state("speaking")
        stt.notify_user_state("listening")
        async with asyncio.timeout(2):
            while not any(f.get("type") == "finalize" for f in ws1.sent_texts):
                await asyncio.sleep(0.01)

        # Options change triggers a reconnect while the watchdog is armed.
        stream.update_options(language="id")

        # Wait past the final timeout: the stale timer must not fire a failover.
        await asyncio.sleep(0.6)

        # Second turn on the new connection completes normally.
        stream._input_ch.send_nowait(SimpleNamespace(data=memoryview(b"\x00\x01" * 1600)))
        stt.notify_user_state("speaking")
        stt.notify_user_state("listening")
        await asyncio.wait_for(got_final.wait(), timeout=3)
        stream.end_input()
        await asyncio.wait_for(reader, timeout=3)

    assert finals == ["hello world"]


@pytest.mark.asyncio
async def test_stt_options_reconnect_closes_speech_bracket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An options-change reconnect must close a dangling START_OF_SPEECH so
    the next utterance on the new connection opens a fresh bracket."""
    ws1, ws2 = _InterimOnlySttWs(), _AnsweringSttWs()
    sockets: list = [ws1, ws2]

    async def fake_connect(self, *, model_endpoint: str, model: str | None):
        del self, model_endpoint, model
        return sockets.pop(0)

    monkeypatch.setattr(slng.stt.SpeechStream, "_connect_ws", fake_connect)
    async with aiohttp.ClientSession() as session:
        stt = slng.STT(api_key="test-key", model="deepgram/nova:3", http_session=session)
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))

        events: list = []
        got_final = asyncio.Event()

        async def read() -> None:
            async for ev in stream:
                events.append(ev.type)
                if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                    got_final.set()

        reader = asyncio.create_task(read())

        # First turn: ws1 answers the finalize with an interim only, which
        # opens a START_OF_SPEECH bracket that never closes on ws1.
        stream._input_ch.send_nowait(SimpleNamespace(data=memoryview(b"\x00\x01" * 1600)))
        stt.notify_user_state("speaking")
        stt.notify_user_state("listening")
        async with asyncio.timeout(2):
            while SpeechEventType.INTERIM_TRANSCRIPT not in events:
                await asyncio.sleep(0.01)

        # Options change triggers the reconnect mid-utterance.
        stream.update_options(language="id")
        await asyncio.sleep(0.2)  # let the reconnect settle

        # Second turn completes normally on ws2.
        stream._input_ch.send_nowait(SimpleNamespace(data=memoryview(b"\x00\x01" * 1600)))
        stt.notify_user_state("speaking")
        stt.notify_user_state("listening")
        await asyncio.wait_for(got_final.wait(), timeout=3)
        stream.end_input()
        await asyncio.wait_for(reader, timeout=3)

    speech_events = [
        ev
        for ev in events
        if ev
        in (
            SpeechEventType.START_OF_SPEECH,
            SpeechEventType.INTERIM_TRANSCRIPT,
            SpeechEventType.FINAL_TRANSCRIPT,
            SpeechEventType.END_OF_SPEECH,
        )
    ]
    assert speech_events == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.INTERIM_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
    ]
