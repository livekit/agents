from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import wave
from collections.abc import AsyncIterator
from types import TracebackType
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import vad
from livekit.plugins.microsoft_ai._http import HTTPClient

DUMMY_CONFIG = """\
MICROSOFT_AI_STT_URL=wss://stt.example.invalid/v1/realtime?intent=transcription
MICROSOFT_AI_STT_MODEL=file-transcriber
MICROSOFT_AI_STT_API_KEY="dummy-stt-key"
MICROSOFT_AI_STT_LANGUAGE=en
MICROSOFT_AI_TTS_URL=https://tts.example.invalid/cognitiveservices/v1
MICROSOFT_AI_TTS_MODEL=file-synthesizer
MICROSOFT_AI_TTS_API_KEY='dummy-tts-key'
MICROSOFT_AI_TTS_VOICE=en-US-Dummy:file-synthesizer
MICROSOFT_AI_TTS_SAMPLE_RATE=24000
"""


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    async def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Hermetic Microsoft AI tests must not make network requests")

    monkeypatch.setattr(aiohttp.ClientSession, "_request", forbidden)
    for name in list(os.environ):
        if name.startswith("MICROSOFT_AI_"):
            monkeypatch.delenv(name)


@pytest.fixture
def no_http_session(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Configuration tests must not create HTTP sessions")

    monkeypatch.setattr(HTTPClient, "session", forbidden)


def audio_frame(
    samples: int = 800, *, sample_rate: int = 16000, value: bytes = b"\x81\xff"
) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=value * samples, sample_rate=sample_rate, num_channels=1, samples_per_channel=samples
    )


def wav_bytes(pcm: bytes, *, sample_rate: int = 24000, channels: int = 1, width: int = 2) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(width)
        audio.setframerate(sample_rate)
        audio.writeframes(pcm)
    return output.getvalue()


def fake_session() -> MagicMock:
    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False

    async def close() -> None:
        session.closed = True

    session.close = AsyncMock(side_effect=close)
    return session


class FakeSocket:
    def __init__(self, *, auto_update: bool = True, auto_commit: bool = True) -> None:
        self.auto_update = auto_update
        self.auto_commit = auto_commit
        self.incoming: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()
        self.sent: list[dict[str, object]] = []
        self.changed = asyncio.Condition()
        self.append_started = asyncio.Event()
        self.append_gate: asyncio.Event | None = None
        self.pending_audio = bytearray()
        self.commits: list[bytes] = []
        self.closed = False
        self.emit({"type": "session.created", "session": {"id": "test-session"}})

    def emit(self, event: dict[str, object]) -> None:
        self.incoming.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(event), ""))

    def disconnect(self) -> None:
        self.incoming.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, ""))

    def transcript(self, type: str, *, item: str = "item-1", **fields: object) -> None:
        self.emit(
            {
                "type": f"conversation.item.input_audio_transcription.{type}",
                "item_id": item,
                **fields,
            }
        )

    def complete(self, *, item: str = "item-1", text: str = "hello") -> None:
        self.emit({"type": "input_audio_buffer.committed", "item_id": item})
        self.transcript("delta", item=item, delta=text)
        self.transcript("completed", item=item, transcript=text)

    async def send_json(self, event: dict[str, object]) -> None:
        assert not self.closed
        if event["type"] == "input_audio_buffer.append":
            self.append_started.set()
            if self.append_gate is not None:
                await self.append_gate.wait()
            audio = event["audio"]
            assert isinstance(audio, str)
            self.pending_audio.extend(base64.b64decode(audio, validate=True))
        async with self.changed:
            self.sent.append(event)
            self.changed.notify_all()
        if event["type"] == "session.update" and self.auto_update:
            self.emit({"type": "session.updated"})
        if event["type"] == "input_audio_buffer.commit":
            self.commits.append(bytes(self.pending_audio))
            self.pending_audio.clear()
            if self.auto_commit:
                self.complete(item=f"item-{len(self.commits)}", text=f"turn {len(self.commits)}")

    async def receive(self) -> aiohttp.WSMessage:
        return await self.incoming.get()

    async def close(self) -> bool:
        self.closed = True
        return True

    async def wait_sent(self, type: str, count: int = 1) -> None:
        async def wait() -> None:
            async with self.changed:
                await self.changed.wait_for(
                    lambda: sum(event["type"] == type for event in self.sent) >= count
                )

        await asyncio.wait_for(wait(), 2.0)


class ScriptedVAD(vad.VAD):
    def __init__(
        self,
        boundaries: dict[int, vad.VADEventType] | None = None,
        *,
        prefix_samples: int = 16000,
    ) -> None:
        super().__init__(capabilities=vad.VADCapabilities(update_interval=0.032))
        self.boundaries = boundaries or {}
        self.prefix_samples = prefix_samples
        self.streams: list[ScriptedVADStream] = []
        self.started = asyncio.Event()

    def stream(self) -> ScriptedVADStream:
        stream = ScriptedVADStream(self)
        self.streams.append(stream)
        self.started.set()
        return stream

    async def wait_consumed(self, samples: int) -> None:
        await asyncio.wait_for(self.started.wait(), 1.0)
        await self.streams[-1].wait_consumed(samples)


class ScriptedVADStream(vad.VADStream):
    def __init__(self, detector: ScriptedVAD) -> None:
        super().__init__(detector)
        self.detector = detector
        self.closed = False
        self.retained_samples = 0
        self._previous_event: vad.VADEvent | None = None
        self._consumed_samples = 0
        self._progress = asyncio.Condition()

    async def __anext__(self) -> vad.VADEvent:
        # Asking for the next event proves the consumer finished processing the previous one.
        async with self._progress:
            if self._previous_event is not None:
                self._consumed_samples = self._previous_event.samples_index
            self._progress.notify_all()
        self._previous_event = await super().__anext__()
        return self._previous_event

    async def wait_consumed(self, samples: int) -> None:
        async def wait() -> None:
            async with self._progress:
                await self._progress.wait_for(lambda: self._consumed_samples >= samples)

        await asyncio.wait_for(wait(), 2.0)

    async def _main_task(self) -> None:
        samples = 0
        pending = bytearray()
        prefix = bytearray()
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                # Like the bundled VAD, flush does not decode a partial inference window.
                pending.clear()
                prefix.clear()
                samples = 0
                continue
            pending.extend(data.data.cast("B"))
            while len(pending) >= 1024:
                prefix.extend(pending[:1024])
                excess = len(prefix) - self.detector.prefix_samples * 2
                if excess > 0:
                    del prefix[:excess]
                self.retained_samples = len(prefix) // 2
                del pending[:1024]
                samples += 512
                self.emit(vad.VADEventType.INFERENCE_DONE, samples)
                if boundary := self.detector.boundaries.get(samples):
                    frames = []
                    if boundary == vad.VADEventType.START_OF_SPEECH and prefix:
                        frames = [
                            rtc.AudioFrame(
                                data=bytes(prefix),
                                sample_rate=16000,
                                num_channels=1,
                                samples_per_channel=len(prefix) // 2,
                            )
                        ]
                    self.emit(boundary, samples, frames=frames)

    def emit(
        self, type: vad.VADEventType, samples: int, *, frames: list[rtc.AudioFrame] | None = None
    ) -> None:
        self._event_ch.send_nowait(
            vad.VADEvent(
                type=type,
                samples_index=samples,
                timestamp=samples / 16000,
                speech_duration=samples / 16000,
                silence_duration=0.0,
                frames=frames or [],
            )
        )

    async def aclose(self) -> None:
        self.closed = True
        await super().aclose()


class FakeContent:
    def __init__(
        self,
        chunks: list[bytes],
        *,
        error: Exception | None = None,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.chunks = chunks
        self.error = error
        self.gate = gate
        self.started = asyncio.Event()

    async def iter_chunked(self, size: int) -> AsyncIterator[bytes]:
        self.started.set()
        for chunk in self.chunks:
            if self.gate is not None:
                await self.gate.wait()
            yield chunk
        if self.error is not None:
            raise self.error


class FakeResponse:
    def __init__(
        self,
        data: bytes = b"",
        *,
        status: int = 200,
        content_type: str = "audio/wav",
        content_length: int | None = None,
        error: Exception | None = None,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.status = status
        self.content_type = content_type
        self.content_length = content_length
        self.content = FakeContent([data[:17], data[17:]], error=error, gate=gate)
        self.closed = False

    async def __aenter__(self) -> FakeResponse:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.closed = True
