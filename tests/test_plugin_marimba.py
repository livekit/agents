import asyncio
import importlib
import threading

import numpy as np
import pytest
import soxr

pytest.importorskip("marimba")

from marimba.decision import Decision  # noqa: E402

from livekit.agents import Plugin  # noqa: E402
from livekit.plugins import marimba  # noqa: E402

pytestmark = pytest.mark.unit


def test_plugin_registration():
    plugin = next(p for p in Plugin.registered_plugins if p.title == "livekit.plugins.marimba")
    assert plugin.version == marimba.__version__
    assert plugin.package == "livekit.plugins.marimba"
    assert plugin.logger.name == "livekit.plugins.marimba"
    assert marimba.TurnDetector is not None
    assert marimba.TurnDetectorStream is not None


@pytest.fixture
def integration():
    pytest.importorskip("marimba")
    return importlib.import_module("livekit.plugins.marimba.turn_detector")


class FakeModel:
    def __init__(self, probability=0.75):
        self.probability = probability
        self.calls = []

    def __call__(self, audio, questions, sampling_rate=None):
        self.calls.append((audio.copy(), questions, sampling_rate))
        q = questions[0]
        return [
            Decision(
                question=q,
                task=q.task,
                kind=q.kind,
                value=self.probability,
                probability=self.probability,
            )
        ]


def frame(samples, rate=48000, channels=1):
    from livekit import rtc

    return rtc.AudioFrame(
        data=np.asarray(samples, dtype=np.int16).tobytes(),
        sample_rate=rate,
        num_channels=channels,
        samples_per_channel=len(samples) // channels,
    )


def test_protocols_and_mapping(integration):
    from livekit.agents.voice.turn import _StreamingTurnDetector, _StreamingTurnDetectorStream

    async def run():
        model = FakeModel()
        detector = integration.TurnDetector(model=model, unlikely_threshold=0.8)
        stream = detector.stream()
        assert isinstance(detector, _StreamingTurnDetector)
        assert isinstance(stream, _StreamingTurnDetectorStream)
        assert stream.provider == "marimba"
        assert stream.model == "charcot-ai/marimba"
        assert not stream.is_fallback
        assert stream.prediction_timeout > 0
        assert await stream.unlikely_threshold("en") == 0.8
        assert await stream.supports_language(None)
        assert await stream.backchannel_threshold(None) is None
        stream.on_start_of_turn(0.0)
        stream.push_audio(frame([8192, -16384], 16000))
        from livekit.agents.llm import ChatContext

        event = await stream.predict(ChatContext())
        assert event.type == "eot_prediction"
        assert event.end_of_turn_probability == 0.75
        assert event.inference_duration >= 0
        audio, questions, rate = model.calls[0]
        np.testing.assert_array_equal(audio, [0.25, -0.5])
        assert rate == 16000
        assert questions[0].text == "Has the speaker finished their turn?"
        assert questions[0].task == "turn_end"
        await stream.aclose()

    asyncio.run(run())


def test_resampling_and_window(integration):
    async def run():
        model = FakeModel()
        stream = integration.TurnDetector(model=model).stream()
        pcm = np.arange(9 * 48000, dtype=np.int16)
        for chunk in np.array_split(pcm, 90):
            stream.push_audio(frame(chunk))
        await stream.predict()
        expected = soxr.resample(
            pcm[-8 * 48000 :].astype(np.float32) / 32768, 48000, 16000, quality="HQ"
        )
        np.testing.assert_allclose(model.calls[0][0], expected)
        assert len(expected) == 128000
        await stream.aclose()

    asyncio.run(run())


def test_lazy_load(integration, monkeypatch):
    model = FakeModel()
    loads = []

    def load():
        loads.append(True)
        return model

    monkeypatch.setattr(integration.Marimba, "from_pretrained", load)

    async def run():
        detector = integration.TurnDetector()
        assert not loads
        stream = detector.stream()
        stream.push_audio(frame([0], 16000))
        await stream.predict()
        await stream.predict()
        assert loads == [True]
        await stream.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("timeout", [0, -1, float("nan")])
def test_prediction_timeout_validation(integration, timeout):
    with pytest.raises(ValueError, match="prediction_timeout"):
        integration.TurnDetector(prediction_timeout=timeout)


def test_prediction_timeout_option(integration):
    assert integration.TurnDetector(model=FakeModel()).stream().prediction_timeout == 2.0
    detector = integration.TurnDetector(model=FakeModel(), prediction_timeout=0.75)
    assert detector.stream().prediction_timeout == 0.75


def test_stream_warms_model_once_without_event_loop(integration, monkeypatch):
    model = FakeModel()
    entered, release, loaded = threading.Event(), threading.Event(), threading.Event()
    loads = []

    def load():
        loads.append(threading.current_thread())
        entered.set()
        assert release.wait(timeout=5)
        loaded.set()
        return model

    monkeypatch.setattr(integration.Marimba, "from_pretrained", load)
    detector = integration.TurnDetector()
    assert not loads
    stream = detector.stream()
    try:
        assert entered.wait(timeout=5)
        second = detector.stream()
        assert len(loads) == 1
        assert loads[0].daemon
        assert loads[0] is not threading.current_thread()
    finally:
        release.set()
    assert loaded.wait(timeout=5)

    async def run():
        stream.push_audio(frame([0], 16000))
        await stream.predict()
        await stream.aclose()
        await second.aclose()
        assert len(loads) == 1

    asyncio.run(run())


def test_warmup_failure_can_retry_at_prediction(integration, monkeypatch, caplog):
    failed = threading.Event()
    loads = []

    def load():
        loads.append(True)
        if len(loads) == 1:
            failed.set()
            raise RuntimeError("synthetic warmup failure")
        return FakeModel()

    monkeypatch.setattr(integration.Marimba, "from_pretrained", load)
    stream = integration.TurnDetector().stream()
    assert failed.wait(timeout=5)

    async def run():
        stream.push_audio(frame([0], 16000))
        assert (await stream.predict()).end_of_turn_probability == 0.75
        await stream.aclose()

    asyncio.run(run())
    assert loads == [True, True]
    assert "Marimba model warmup failed" in caplog.text


@pytest.mark.parametrize("failure", ["raise", "empty", "missing"])
def test_failure_defaults_to_complete(integration, failure):
    def broken(*args, **kwargs):
        if failure == "raise":
            raise RuntimeError("synthetic failure")
        if failure == "empty":
            return []
        return FakeModel(None)(*args, **kwargs)

    async def run():
        stream = integration.TurnDetector(model=broken).stream()
        stream.push_audio(frame([0], 16000))
        assert (await stream.predict()).end_of_turn_probability == 1.0
        await stream.aclose()

    asyncio.run(run())


def test_flush_empty_and_end_input(integration):
    async def run():
        model = FakeModel()
        stream = integration.TurnDetector(model=model).stream()
        assert (await stream.predict()).end_of_turn_probability == 1.0
        stream.push_audio(frame([1], 16000))
        stream.flush()
        assert (await stream.predict()).end_of_turn_probability == 1.0
        assert not model.calls
        stream.push_audio(frame([2], 48000))
        stream.end_input()
        stream.push_audio(frame([3], 16000))
        assert (await stream.predict()).end_of_turn_probability == 1.0
        await stream.aclose()
        await stream.aclose()

    asyncio.run(run())


def test_cancellation_ignores_stale_result(integration):
    entered, release = threading.Event(), threading.Event()

    def blocking(*args, **kwargs):
        entered.set()
        release.wait()
        return FakeModel()(*args, **kwargs)

    async def run():
        stream = integration.TurnDetector(model=blocking).stream()
        stream.push_audio(frame([0], 16000))
        future = stream.predict()
        try:
            await asyncio.to_thread(entered.wait)
            stream.cancel_inference(timed_out=True)
            assert (await future).end_of_turn_probability == 0.0
        finally:
            release.set()
            await stream.aclose()
        assert future.result().end_of_turn_probability == 0.0

    asyncio.run(run())


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_threshold_validation(integration, threshold):
    with pytest.raises(ValueError, match="unlikely_threshold"):
        integration.TurnDetector(unlikely_threshold=threshold)


def test_stereo_and_rate_change(integration):
    async def run():
        model = FakeModel()
        stream = integration.TurnDetector(model=model).stream()
        stream.push_audio(frame([8192, 24576], 16000, 2))
        with pytest.raises(ValueError, match="sample rate"):
            stream.push_audio(frame([0], 48000))
        await stream.predict()
        np.testing.assert_array_equal(model.calls[0][0], [0.5])
        await stream.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("rate", [8000, 16000, 24000, 44100, 48000])
@pytest.mark.parametrize("samples", [1, 9 * 48000])
def test_sample_rates_and_short_windows(integration, rate, samples):
    async def run():
        model = FakeModel()
        stream = integration.TurnDetector(model=model).stream()
        pcm = (np.sin(np.arange(samples) * 0.01) * 32000).astype(np.int16)
        for start in range(0, len(pcm), 960):
            stream.push_audio(frame(pcm[start : start + 960], rate))
        await stream.predict()
        expected = pcm[-8 * rate :].astype(np.float32) / 32768.0
        if rate != 16000:
            expected = soxr.resample(expected, rate, 16000, quality="HQ")
        np.testing.assert_array_equal(model.calls[0][0], expected[-128000:])
        await stream.aclose()

    asyncio.run(run())


def test_predictions_with_and_without_chat_context(integration):
    from livekit.agents.llm import ChatContext

    async def run():
        model = FakeModel()
        stream = integration.TurnDetector(model=model).stream()
        for context in [None, ChatContext()]:
            stream.on_start_of_turn(0.0)
            assert (await stream.predict(context)).end_of_turn_probability == 1.0
            stream.push_audio(frame([8192], 16000))
            assert (await stream.predict(context)).end_of_turn_probability == 0.75
            stream.flush()
        await stream.aclose()
        np.testing.assert_array_equal(model.calls[0][0], model.calls[1][0])

    asyncio.run(run())


def test_superseded_predictions_do_not_queue_workers(integration):
    entered, release = threading.Event(), threading.Event()
    model = FakeModel()

    def blocking(*args, **kwargs):
        entered.set()
        release.wait()
        return model(*args, **kwargs)

    async def run():
        stream = integration.TurnDetector(model=blocking).stream()
        stream.push_audio(frame([1], 16000))
        first = stream.predict()
        try:
            await asyncio.to_thread(entered.wait)
            for value in range(2, 102):
                previous = stream._future
                stream.flush()
                assert (await previous).end_of_turn_probability == 0.0
                stream.push_audio(frame([value], 16000))
                latest = stream.predict()
                await asyncio.sleep(0)
                assert len(stream._tasks) == 1
                assert len(stream._audio) == 1
            release.set()
            assert (await latest).end_of_turn_probability == 0.75
        finally:
            release.set()
            await stream.aclose()
        assert first.result().end_of_turn_probability == 0.0
        assert len(model.calls) == 2
        np.testing.assert_array_equal(model.calls[-1][0], [101 / 32768])
        assert not stream._tasks
        assert stream._pending is None

    asyncio.run(run())


def test_concurrent_streams_and_many_turns(integration):
    async def run():
        model = FakeModel()
        detector = integration.TurnDetector(model=model)
        streams = [detector.stream(), detector.stream()]
        for turn in range(100):
            for channel, stream in enumerate(streams):
                stream.push_audio(frame([turn + channel + 1], 16000))
            events = await asyncio.gather(*(stream.predict() for stream in streams))
            assert all(event.end_of_turn_probability == 0.75 for event in events)
            for stream in streams:
                stream.flush()
                assert not stream._audio.size
                assert len(stream._tasks) <= 1
        await asyncio.gather(*(stream.aclose() for stream in streams))
        assert len(model.calls) == 200
        assert all(not stream._tasks for stream in streams)

    asyncio.run(run())
