"""Actual pinned Silero segmentation with a synthetic probability callable.

No VAD.load, ONNX session or model weights; the real buffering/filter/boundary
implementation executes unchanged against deterministic generated PCM.
"""

from __future__ import annotations

import asyncio
from collections import Counter

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import stt, vad
from livekit.plugins import silero
from livekit.plugins.oruk import RealtimeSTT, vad_stream_node
from livekit.plugins.silero.vad import _VADOptions

from .test_plugin_oruk_realtime import Session, audio

pytestmark = pytest.mark.unit


class ProbabilityModel:
    window_size_samples = 512
    sample_rate = 16000

    def __init__(self, probabilities):
        self.probabilities = iter(probabilities)
        self.calls = 0
        self.resets = 0

    def __call__(self, samples):
        assert samples.shape == (512,)
        self.calls += 1
        return next(self.probabilities)

    def reset(self):
        self.resets += 1


class RecordingStream:
    def __init__(self, actual):
        self.actual = actual
        self.events = []

    def push_frame(self, frame):
        self.actual.push_frame(frame)

    def end_input(self):
        self.actual.end_input()

    async def aclose(self):
        await self.actual.aclose()

    def __aiter__(self):
        return self

    async def __anext__(self):
        event = await anext(self.actual)
        self.events.append(event)
        return event


def detector_with_stub(monkeypatch, probabilities):
    model = ProbabilityModel(probabilities)

    def forbidden_load(*args, **kwargs):
        raise AssertionError("No ONNX inference session or model load is allowed")

    monkeypatch.setattr(silero.vad.onnx_model, "new_inference_session", forbidden_load)
    monkeypatch.setattr(silero.vad.onnx_model, "OnnxModel", lambda **kwargs: model)
    detector = silero.VAD(
        session=object(),
        opts=_VADOptions(
            min_speech_duration=0.064,
            min_silence_duration=0.064,
            prefix_padding_duration=0.256,
            max_buffered_speech=60.0,
            activation_threshold=0.5,
            deactivation_threshold=0.35,
            sample_rate=16000,
        ),
    )
    actual_factory = detector.stream
    streams = []

    def record_stream():
        recorder = RecordingStream(actual_factory())
        streams.append(recorder)
        return recorder

    monkeypatch.setattr(detector, "stream", record_stream)
    return detector, model, streams


@pytest.fixture
def sockets(monkeypatch):
    session = Session()
    monkeypatch.setattr("livekit.plugins.oruk.realtime.new_http_session", lambda: session)
    return session


async def test_actual_silero_overlap_and_active_eof_preserve_every_sample_once(
    monkeypatch, sockets
):
    probabilities = [0.0] * 4 + [1.0] * 4 + [0.0] * 6 + [1.0] * 4 + [0.0] * 6 + [1.0] * 4
    detector, model, streams = detector_with_stub(monkeypatch, probabilities)
    frames = [audio(index + 1) for index in range(len(probabilities))] + [audio(99, samples=128)]
    owner = RealtimeSTT(api_key="test")

    async def source():
        for frame in frames:
            yield frame
            await asyncio.sleep(0)

    try:
        events = [event async for event in vad_stream_node(owner, source(), detector=detector)]
        assert model.calls == len(probabilities) and model.resets == 1
        observed = streams[0].events
        boundaries = [
            item
            for item in observed
            if item.type in (vad.VADEventType.START_OF_SPEECH, vad.VADEventType.END_OF_SPEECH)
        ]
        assert [item.type for item in boundaries] == [
            vad.VADEventType.START_OF_SPEECH,
            vad.VADEventType.END_OF_SPEECH,
            vad.VADEventType.START_OF_SPEECH,
            vad.VADEventType.END_OF_SPEECH,
            vad.VADEventType.START_OF_SPEECH,
        ]
        for item in boundaries:
            before = observed[observed.index(item) - 1]
            assert before.type == vad.VADEventType.INFERENCE_DONE
            assert before.samples_index == item.samples_index
        assert len(sockets.sockets) == 3
        sent = b"".join(chunk for socket in sockets.sockets for chunk in socket.audio)
        # The configured 256ms prefix covers each shorter silence gap. Therefore
        # every unique input sample is expected exactly once, including the EOF tail.
        assert sent == b"".join(bytes(frame.data) for frame in frames)
        counts = Counter(np.frombuffer(sent, dtype=np.int16))
        assert counts[99] == 128
        assert all(counts[index + 1] == 512 for index in range(len(probabilities)))
        assert (
            len([item for item in events if item.type == stt.SpeechEventType.FINAL_TRANSCRIPT]) == 3
        )
        assert streams[0].actual._task.done() and streams[0].actual._metrics_task.done()
    finally:
        await owner.aclose()


async def test_cancel_actual_silero_bridge_joins_workers_without_commit(monkeypatch, sockets):
    detector, _, streams = detector_with_stub(monkeypatch, [1.0] * 4)
    owner = RealtimeSTT(api_key="test")
    waiting = asyncio.Event()

    async def source():
        for index in range(4):
            yield audio(index + 1)
            await asyncio.sleep(0)
        await waiting.wait()

    bridge = vad_stream_node(owner, source(), detector=detector)
    try:
        assert (
            await asyncio.wait_for(anext(bridge), 1)
        ).type == stt.SpeechEventType.START_OF_SPEECH
        await asyncio.wait_for(anext(bridge), 1)  # actual audio reached the fake socket
        await bridge.aclose()
        assert streams[0].actual._task.done() and streams[0].actual._metrics_task.done()
        assert all(socket.closed for socket in sockets.sockets)
        assert not any(
            command["type"] == "input_audio_buffer.commit"
            for socket in sockets.sockets
            for command in socket.commands
        )
        assert not owner._receipts
    finally:
        await bridge.aclose()
        await owner.aclose()


async def test_large_frame_backpressures_actual_silero_without_lost_or_replayed_pcm(
    monkeypatch, sockets
):
    samples = np.arange(48000, dtype=np.int16)
    detector, model, streams = detector_with_stub(monkeypatch, [1.0] * 93)
    owner = RealtimeSTT(api_key="test")

    async def source():
        yield rtc.AudioFrame(samples.tobytes(), 16000, 1, len(samples))

    try:
        events = [event async for event in vad_stream_node(owner, source(), detector=detector)]
        assert model.calls == 93  # active EOF retains the remaining384 samples
        assert len(sockets.sockets) == 1
        assert b"".join(sockets.sockets[0].audio) == samples.tobytes()
        assert all(len(chunk) <= 10240 for chunk in sockets.sockets[0].audio)
        assert (
            len([item for item in events if item.type == stt.SpeechEventType.FINAL_TRANSCRIPT]) == 1
        )
        assert streams[0].actual._task.done() and streams[0].actual._metrics_task.done()
    finally:
        await owner.aclose()


class StalledVAD(vad.VAD):
    def __init__(self):
        super().__init__(capabilities=vad.VADCapabilities(update_interval=0.032))
        self.streams = []

    def stream(self):
        stream = StalledVADStream(self)
        self.streams.append(stream)
        return stream


class StalledVADStream(vad.VADStream):
    def __init__(self, owner):
        super().__init__(owner)
        self.received_bytes = 0

    def push_frame(self, frame):
        self.received_bytes += len(bytes(frame.data))
        super().push_frame(frame)

    async def _main_task(self):
        await asyncio.Event().wait()


@pytest.mark.parametrize(
    "cancel,samples", [(False, 48000), (True, 48000), (False, 512), (True, 512)]
)
async def test_stalled_vad_capacity_wait_is_bounded_and_cancellable(
    monkeypatch, sockets, cancel, samples
):
    monkeypatch.setattr(
        "livekit.plugins.oruk.realtime_bridge._VAD_BACKLOG_TIMEOUT", 1.0 if cancel else 0.02
    )
    detector = StalledVAD()
    owner = RealtimeSTT(api_key="test")

    async def source():
        yield audio(samples=samples)

    async def consume():
        return [event async for event in vad_stream_node(owner, source(), detector=detector)]

    task = asyncio.create_task(consume())
    try:
        if cancel:
            for _ in range(100):
                if detector.streams and detector.streams[0].received_bytes == min(
                    30720, samples * 2
                ):
                    break
                await asyncio.sleep(0.001)
            assert detector.streams[0].received_bytes == min(30720, samples * 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
        else:
            message = (
                "VAD input capacity wait timed out"
                if samples > 16000
                else "VAD input drain timed out"
            )
            with pytest.raises(RuntimeError, match=message):
                await asyncio.wait_for(task, 1)
        assert 0 < detector.streams[0].received_bytes <= 32000
        assert detector.streams[0]._task.done() and detector.streams[0]._metrics_task.done()
        assert not sockets.sockets and not owner._receipts
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await owner.aclose()


@pytest.mark.parametrize("closed", [False, True])
async def test_second_stream_allocation_failure_closes_first_vad_resource(sockets, closed):
    owner = RealtimeSTT(api_key="test")
    detector = StalledVAD()
    existing = []
    if closed:
        await owner.aclose()
    else:
        existing = [owner.stream() for _ in range(4)]

    async def source():
        raise AssertionError("allocation failure must not consume caller audio")
        yield audio()

    bridge = vad_stream_node(owner, source(), detector=detector)
    try:
        with pytest.raises(RuntimeError, match="closed|more than four"):
            await anext(bridge)
        assert len(detector.streams) == 1
        assert detector.streams[0]._task.done() and detector.streams[0]._metrics_task.done()
        assert not sockets.sockets
    finally:
        await bridge.aclose()
        await owner.aclose()
    assert all(stream._task.done() for stream in existing)
