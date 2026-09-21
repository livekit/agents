from __future__ import annotations

import asyncio
import base64
import io
import json
import wave
from collections.abc import AsyncIterator
from types import TracebackType
from unittest.mock import AsyncMock, MagicMock

import aiohttp

from livekit import rtc
from livekit.agents import vad


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
    def __init__(self, boundaries: dict[int, vad.VADEventType] | None = None) -> None:
        super().__init__(capabilities=vad.VADCapabilities(update_interval=0.032))
        self.boundaries = boundaries or {}
        self.streams: list[ScriptedVADStream] = []

    def stream(self) -> ScriptedVADStream:
        stream = ScriptedVADStream(self)
        self.streams.append(stream)
        return stream


class ScriptedVADStream(vad.VADStream):
    def __init__(self, detector: ScriptedVAD) -> None:
        super().__init__(detector)
        self.detector = detector
        self.closed = False

    async def _main_task(self) -> None:
        samples = 0
        pending = bytearray()
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                # Like the bundled VAD, flush does not decode a partial inference window.
                pending.clear()
                samples = 0
                continue
            pending.extend(data.data.cast("B"))
            while len(pending) >= 1024:
                del pending[:1024]
                samples += 512
                self.emit(vad.VADEventType.INFERENCE_DONE, samples)
                if boundary := self.detector.boundaries.get(samples):
                    self.emit(boundary, samples)

    def emit(self, type: vad.VADEventType, samples: int) -> None:
        self._event_ch.send_nowait(
            vad.VADEvent(
                type=type,
                samples_index=samples,
                timestamp=samples / 16000,
                speech_duration=samples / 16000,
                silence_duration=0.0,
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
