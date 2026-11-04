"""Rime STT compatibility, audio conversion, recognition, and streaming regressions."""

from __future__ import annotations

import asyncio
import io
import subprocess
import sys
import traceback
import wave
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import (
    AgentSession,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    llm,
    stt,
    utils,
    vad,
)
from livekit.agents.stt import SpeechEventType
from livekit.agents.vad import VADEventType
from livekit.agents.voice.audio_recognition import AudioRecognition
from livekit.agents.voice.endpointing import BaseEndpointing
from livekit.plugins import rime

pytestmark = pytest.mark.unit


@pytest.fixture
def sdk():
    # Compatibility checks must still run without the optional SDK or Python 3.11.
    if sys.version_info < (3, 11):
        pytest.skip("Rime STT requires Python 3.11+")
    return pytest.importorskip("rimelabs_sdk")


@pytest.fixture
async def cleanup():
    async with AsyncExitStack() as stack:
        yield stack


async def _cancel_task(task):
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.fixture
def collect_events(cleanup):
    def start(stream, consume=None):
        observed = []

        async def collect():
            async for event in stream:
                observed.append(event)
                if consume is not None:
                    await consume(event)

        reader = asyncio.create_task(collect())
        cleanup.push_async_callback(_cancel_task, reader)
        return observed, reader

    return start


NO_RETRY = APIConnectOptions(max_retry=0, timeout=2)


@pytest.fixture
def client(sdk, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    client = MagicMock()
    client.stt.transcribe = AsyncMock(
        return_value=sdk.Transcription(text="Hello.", language="en", request_id="request-123")
    )
    client.close = AsyncMock()
    monkeypatch.setattr(sdk, "Rime", MagicMock(return_value=client))
    return client


@pytest.fixture
async def recognizer(client: MagicMock) -> AsyncIterator[rime.STT]:
    async with rime.STT() as recognizer:
        yield recognizer


def _frame(
    sample_rate: int = 16000,
    channels: int = 1,
    samples: int | None = None,
    *,
    data: bytes | None = None,
) -> rtc.AudioFrame:
    if data is None:
        samples = samples if samples is not None else sample_rate // 10
        # Asymmetric, loud channels detect channel selection and int16 overflow.
        data = np.tile(np.arange(channels, dtype=np.int16) * 1000 + 20000, (samples, 1)).tobytes()
    else:
        samples = len(data) // (channels * 2)
    return rtc.AudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=channels,
        samples_per_channel=samples,
    )


class _SegmentVAD(vad.VAD):
    """Emit one utterance per flush, keeping actual frames for the adapter."""

    def __init__(self) -> None:
        super().__init__(capabilities=vad.VADCapabilities(update_interval=0.1))

    def stream(self) -> vad.VADStream:
        return _SegmentStream(self)


class _SegmentStream(vad.VADStream):
    async def _main_task(self) -> None:
        frames: list[rtc.AudioFrame] = []
        async for item in self._input_ch:
            if isinstance(item, rtc.AudioFrame):
                if not frames:
                    self._emit(VADEventType.START_OF_SPEECH, [])
                frames.append(item)
            elif frames:
                self._emit(VADEventType.END_OF_SPEECH, frames)
                frames = []

    def _emit(self, event_type: vad.VADEventType, frames: list[rtc.AudioFrame]) -> None:
        self._event_ch.send_nowait(
            vad.VADEvent(
                type=event_type,
                frames=frames,
                samples_index=0,
                timestamp=0,
                speech_duration=sum(frame.duration for frame in frames),
                silence_duration=0,
            )
        )


class DetectorStream:
    def __init__(self):
        self.events = utils.aio.Chan()
        self.closed = False

    def push_frame(self, frame):
        pass

    def flush(self):
        pass

    def end_input(self):
        self.events.close()

    async def aclose(self):
        self.closed = True

    def __aiter__(self):
        return self.events

    def send(self, kind, data, *, speech_duration=0.0, inference_duration=0.0):
        self.events.send_nowait(
            SimpleNamespace(
                type=kind,
                frames=[_frame(data=data)],
                silence_duration=0.5,
                speech_duration=speech_duration,
                inference_duration=inference_duration,
            )
        )

    def utterance(self, data=b"aa"):
        self.send(VADEventType.START_OF_SPEECH, data)
        self.send(VADEventType.END_OF_SPEECH, data)


class SDKStream:
    request_id = "test-request"

    def __init__(self, source, options, fail, sdk):
        self.sdk = sdk
        self.source = source
        self.options = options
        self.audio = []
        self.closed = False
        self.fail = fail

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def read_audio(self):
        async for chunk in self.source:
            self.audio.append(chunk)

    async def updates(self):
        text = ""
        async for chunk in self.source:
            self.audio.append(chunk)
            delta = "Ice" if not text else "\ncream"
            text += delta
            if (
                self.options["output_contract"]
                == self.sdk.TranscriptionOutputContract.APPEND_FRAGMENTS
            ):
                yield self.sdk.TranscriptionDelta(text=delta)
            else:
                yield self.sdk.TranscriptionPartial(text=text)
            if self.fail:
                raise self.sdk.RimeUnavailableError(
                    "private-provider-content", request_id="failed-request"
                )
        yield self.sdk.TranscriptionFinal(text=text, language="en")

    def __aiter__(self):
        return self.updates()


@pytest.fixture
async def setup(sdk, client, cleanup):
    streams, events, fail, closed = [], [], [False], []
    detector = DetectorStream()

    def stream(source, **options):
        result = SDKStream(source, options, fail[0], sdk)
        streams.append(result)
        return result

    client.stt.stream.side_effect = stream
    client.close.side_effect = lambda: closed.append(True)
    # Close the owner before the cleanup fixture joins readers and their consumers.
    async with rime.STT(
        vad=SimpleNamespace(stream=lambda: detector), output_contract="append_fragments"
    ) as owner:
        yield owner, detector, streams, events, fail, closed


async def receive(stream, kind):
    while True:
        event = await asyncio.wait_for(anext(stream), 2)
        if event.type == kind:
            return event


async def wait_until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), 2)


def test_tts_import_does_not_require_the_sdk() -> None:
    # A fresh interpreter also catches accidental eager SDK imports from rime.__init__.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from importlib.abc import MetaPathFinder
class NoSDK(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'rimelabs_sdk':
            raise ModuleNotFoundError(name=fullname)
sys.meta_path.insert(0, NoSDK())
from livekit.plugins import rime
assert rime.TTS
try:
    rime.STT()
except ImportError as error:
    assert 'livekit-plugins-rime[stt]' in str(error)
    if sys.version_info < (3, 11):
        assert 'Python 3.11+' in str(error)
else:
    raise AssertionError('Missing SDK was not detected')
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_stt_requires_python_311(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 10, 0))
    with pytest.raises(ImportError, match=r"Rime STT requires Python 3\.11\+"):
        rime.STT()


@pytest.mark.parametrize("sample_rate,channels", [(16000, 1), (48000, 1), (48000, 2), (44100, 2)])
async def test_audio_and_result_mapping(
    recognizer: rime.STT, client: MagicMock, sample_rate: int, channels: int
) -> None:
    metrics = []
    recognizer.on("metrics_collected", metrics.append)
    frames = [_frame(sample_rate, channels), _frame(sample_rate, channels)]
    result = await recognizer.recognize(frames, conn_options=NO_RETRY)

    with wave.open(io.BytesIO(client.stt.transcribe.call_args.args[0]), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        assert wav.getnframes() == 3200
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        # Ignore the resampler's edges; steady samples must have the mean channel amplitude.
        assert np.mean(audio[200:-200]) == pytest.approx(20000 + (channels - 1) * 500, abs=2)

    assert result.type == SpeechEventType.FINAL_TRANSCRIPT
    assert result.request_id == "request-123"
    assert result.alternatives[0].text == "Hello."
    assert result.alternatives[0].language == "en"
    assert result.alternatives[0].words is None
    assert len(metrics) == 1
    assert metrics[0].request_id == "request-123"
    assert metrics[0].audio_duration == pytest.approx(0.2)
    assert metrics[0].metadata.model_provider == "Rime"
    assert metrics[0].metadata.model_name == "rime-stt"
    assert client.stt.transcribe.call_args.kwargs["context_terms"] == ()
    assert not recognizer.capabilities.streaming
    assert not recognizer.capabilities.interim_results


async def test_options_are_snapshotted_and_keyterms_merged(client: MagicMock, sdk) -> None:
    terms = ["Rime"]
    async with rime.STT(language="en-US", mode="verbatim", context_terms=terms) as recognizer:
        terms.append("not part of the configuration")
        session_terms = ["Rime", "LiveKit"]
        recognizer._update_session_keyterms(session_terms)
        session_terms.clear()
        await recognizer.recognize(_frame(), language="es", conn_options=NO_RETRY)
        assert client.stt.transcribe.call_args.kwargs == {
            "language": "es",
            "mode": sdk.TranscriptionMode.VERBATIM,
            "context_terms": ("Rime", "LiveKit"),
            "timeout": 2,
        }
        await recognizer.recognize(_frame(), conn_options=NO_RETRY)
        assert client.stt.transcribe.call_args.kwargs["language"] == "en-US"

        recognizer.update_options(language="fr", mode="written", context_terms=[])
        recognizer._update_session_keyterms([])
        await recognizer.recognize(_frame(), conn_options=NO_RETRY)
        assert client.stt.transcribe.call_args.kwargs == {
            "language": "fr",
            "mode": sdk.TranscriptionMode.WRITTEN,
            "context_terms": (),
            "timeout": 2,
        }
    client.close.assert_awaited_once()


async def test_empty_transcript_and_missing_request_id(
    recognizer: rime.STT, client: MagicMock, sdk
) -> None:
    client.stt.transcribe.return_value = sdk.Transcription(text="", language="es")
    event = await recognizer.recognize(_frame(), conn_options=NO_RETRY)
    assert event.type == SpeechEventType.FINAL_TRANSCRIPT
    assert event.request_id == ""
    assert event.alternatives[0].text == ""
    assert event.alternatives[0].language == "es"


@pytest.mark.parametrize("bad_terms", ["Rime", [123]])
def test_invalid_context_terms_are_not_coerced(client: MagicMock, bad_terms: object) -> None:
    with pytest.raises(ValueError, match="sequence of strings"):
        rime.STT(context_terms=bad_terms)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "buffer",
    [[], _frame(samples=0), [_frame(), _frame(48000)], [_frame(), _frame(channels=2)]],
)
async def test_invalid_audio_is_not_retried(
    recognizer: rime.STT, client: MagicMock, buffer: utils.AudioBuffer
) -> None:
    with pytest.raises(APIError) as caught:
        await recognizer.recognize(buffer)
    assert not caught.value.retryable
    client.stt.transcribe.assert_not_called()


@pytest.mark.parametrize(
    "sdk_error,livekit_error,retryable",
    [
        ("RimeTimeoutError", APITimeoutError, True),
        ("RimeUnavailableError", APIConnectionError, True),
        ("RimeAuthenticationError", APIStatusError, False),
        ("RimePermissionError", APIStatusError, False),
        ("RimeInputError", APIStatusError, False),
        ("RimeAudioFormatError", APIStatusError, False),
        ("RimeResourceLimitError", APIStatusError, False),
        ("RimeCancelledError", APIStatusError, False),
        ("RimeStreamError", APIStatusError, False),
    ],
)
async def test_safe_errors_and_retry_policy(
    recognizer: rime.STT,
    client: MagicMock,
    sdk_error: str,
    livekit_error: type[APIError],
    retryable: bool,
    caplog: pytest.LogCaptureFixture,
    sdk,
) -> None:
    error_type = getattr(sdk, sdk_error)
    marker = "private-request-content"
    client.stt.transcribe.side_effect = error_type(marker, request_id="request-failed")
    with pytest.raises(livekit_error) as caught:
        await recognizer.recognize(_frame(), conn_options=NO_RETRY)
    error = caught.value
    assert error.retryable is retryable
    assert error.body == {"request_id": "request-failed"}
    assert error.__cause__ is None
    assert marker not in "".join(traceback.format_exception(error))
    assert marker not in repr(error)
    assert marker not in caplog.text

    client.stt.transcribe.reset_mock()
    client.stt.transcribe.side_effect = [
        error_type(marker),
        sdk.Transcription(text="Recovered.", language="en"),
    ]
    options = APIConnectOptions(max_retry=1, retry_interval=0, timeout=2)
    if retryable:
        event = await recognizer.recognize(_frame(), conn_options=options)
        assert event.alternatives[0].text == "Recovered."
        assert client.stt.transcribe.await_count == 2
    else:
        with pytest.raises(livekit_error):
            await recognizer.recognize(_frame(), conn_options=options)
        assert client.stt.transcribe.await_count == 1
    assert marker not in caplog.text


@pytest.mark.parametrize("finish", ["timeout", "cancel", "close"])
async def test_real_sdk_pending_request_lifecycle(
    monkeypatch: pytest.MonkeyPatch, finish: str, sdk
) -> None:
    # Exercise the SDK's deadline, cancellation, and close behavior without a server.
    from rimelabs_sdk.stt import _transport

    connected = asyncio.Event()
    cancelled = asyncio.Event()

    async def channel_ready() -> None:
        connected.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    channel = MagicMock()
    channel.channel_ready = channel_ready
    channel.close = AsyncMock()
    monkeypatch.setattr(_transport, "make_channel", lambda _: channel)
    monkeypatch.setenv("RIME_API_KEY", "test-only")
    async with rime.STT() as recognizer:
        task = asyncio.create_task(
            recognizer.recognize(
                _frame(),
                conn_options=APIConnectOptions(
                    max_retry=0, timeout=0.05 if finish == "timeout" else 2
                ),
            )
        )
        await asyncio.wait_for(connected.wait(), timeout=1)
        if finish == "cancel":
            task.cancel()
            expected = asyncio.CancelledError
        elif finish == "close":
            await recognizer.aclose()
            expected = APIStatusError
        else:
            expected = APITimeoutError
        with pytest.raises(expected):
            await asyncio.wait_for(task, timeout=1)
        assert cancelled.is_set()
    channel.close.assert_awaited_once()


def test_sdk_without_transcribe_has_actionable_error(client: MagicMock) -> None:
    client.stt.transcribe = None
    with pytest.raises(ImportError, match="stt.transcribe"):
        rime.STT()


async def test_livekit_stream_adapter_transcribes_each_utterance(
    recognizer: rime.STT, client: MagicMock, sdk
) -> None:
    client.stt.transcribe.side_effect = [
        sdk.Transcription(text="First.", language="en"),
        sdk.Transcription(text="Second.", language="en"),
    ]
    async with stt.StreamAdapter(stt=recognizer, vad=_SegmentVAD()) as adapter:
        async with adapter.stream(conn_options=NO_RETRY) as stream:
            stream.push_frame(_frame())
            stream.push_frame(_frame())
            stream.flush()
            stream.push_frame(_frame())
            stream.end_input()
            events = [event async for event in stream]

    assert [event.type for event in events] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.END_OF_SPEECH,
        SpeechEventType.FINAL_TRANSCRIPT,
    ] * 2
    assert [event.alternatives[0].text for event in events if event.alternatives] == [
        "First.",
        "Second.",
    ]
    assert client.stt.transcribe.await_count == 2
    lengths = []
    for call in client.stt.transcribe.call_args_list:
        with wave.open(io.BytesIO(call.args[0]), "rb") as wav:
            lengths.append(wav.getnframes())
    assert lengths == [3200, 1600]


def test_append_requires_vad_and_compatible_sdk(client, monkeypatch, sdk):
    with pytest.raises(ValueError, match="requires a vad"):
        rime.STT(output_contract="append_fragments")
    monkeypatch.delattr(sdk, "TranscriptionOutputContract", raising=False)
    with pytest.raises(ImportError, match="TranscriptionOutputContract"):
        rime.STT(vad=_SegmentVAD(), output_contract="append_fragments")


@pytest.mark.parametrize("backend", ["sdk_append", "sdk_revised"])
async def test_live_audio_prefix_deltas_and_final(setup, backend):
    owner, detector, streams, events, _, _ = setup
    owner._output_contract = "append_fragments" if backend == "sdk_append" else "revised_hypotheses"
    stream = owner.stream(conn_options=APIConnectOptions(timeout=120))
    # INFERENCE before START is already included in START's prefix.
    detector.send(VADEventType.INFERENCE_DONE, b"xx")
    detector.send(VADEventType.START_OF_SPEECH, b"aabb")
    first = await receive(stream, SpeechEventType.INTERIM_TRANSCRIPT)
    assert first.alternatives[0].text == "Ice"
    assert not streams[0].closed  # text arrived while input remains open
    detector.send(VADEventType.INFERENCE_DONE, b"cc")
    second = await receive(stream, SpeechEventType.INTERIM_TRANSCRIPT)
    assert second.alternatives[0].text == "Ice\ncream"
    detector.send(VADEventType.END_OF_SPEECH, b"aabbcc")
    final = await receive(stream, SpeechEventType.FINAL_TRANSCRIPT)
    assert final.alternatives[0].text == "Ice\ncream"
    assert streams[0].audio == [b"aabb", b"cc"]  # no duplicate END buffer
    assert streams[0].closed
    assert final.request_id == "test-request"

    stream.end_input()
    assert [e.type async for e in stream] == [
        SpeechEventType.RECOGNITION_USAGE,
        SpeechEventType.END_OF_SPEECH,
    ]
    assert detector.closed


async def test_options_snapshot_and_second_utterance(setup, sdk):
    owner, detector, streams, _, _, _ = setup
    owner.update_options(context_terms=["Rime"])
    owner._update_session_keyterms(["Rime", "LiveKit"])
    stream = owner.stream()
    detector.send(VADEventType.START_OF_SPEECH, b"aa")
    await receive(stream, SpeechEventType.INTERIM_TRANSCRIPT)
    owner.update_options(mode="verbatim", language="es", context_terms=["New"])
    detector.send(VADEventType.END_OF_SPEECH, b"aa")
    await receive(stream, SpeechEventType.FINAL_TRANSCRIPT)
    detector.send(VADEventType.START_OF_SPEECH, b"bb")
    await receive(stream, SpeechEventType.INTERIM_TRANSCRIPT)
    detector.send(VADEventType.END_OF_SPEECH, b"bb")
    await receive(stream, SpeechEventType.FINAL_TRANSCRIPT)
    assert streams[0].options["mode"] == sdk.TranscriptionMode.WRITTEN
    assert streams[0].options["context_terms"] == ("Rime", "LiveKit")
    assert streams[1].options["mode"] == sdk.TranscriptionMode.VERBATIM
    assert streams[1].options["language"] == "es"
    assert streams[1].options["context_terms"] == ("New", "Rime", "LiveKit")


async def test_failure_before_vad_end_is_not_retried(setup):
    owner, detector, streams, events, fail, _ = setup
    fail[0] = True
    stream = owner.stream(conn_options=APIConnectOptions(max_retry=3))
    detector.send(VADEventType.START_OF_SPEECH, b"aa")
    observed = []
    with pytest.raises(APIStatusError) as caught:
        async for event in stream:
            observed.append(event)
    assert caught.value.request_id == "failed-request"
    assert not caught.value.retryable
    assert "private-provider-content" not in str(caught.value)
    assert not any(e.type == SpeechEventType.FINAL_TRANSCRIPT for e in observed)
    assert len(streams) == 1
    assert not any(e.get("kind") == "final" for e in events)
    assert streams[0].closed


async def test_stop_cancels_active_rpc(setup):
    owner, detector, streams, events, _, closed = setup
    stream = owner.stream()
    detector.send(VADEventType.START_OF_SPEECH, b"aa")
    await receive(stream, SpeechEventType.INTERIM_TRANSCRIPT)
    await owner.aclose()
    assert streams[0].closed and detector.closed and closed
    assert not owner._streams
    assert not any(e.get("kind") == "final" for e in events)


async def test_silence_does_not_start_rpc(setup):
    owner, detector, streams, _, _, _ = setup
    stream = owner.stream()
    detector.send(VADEventType.INFERENCE_DONE, b"aa")
    stream.end_input()
    assert [e async for e in stream] == []
    assert not streams
    await owner.aclose()


async def test_end_input_finishes_active_utterance_without_vad_end(setup):
    owner, detector, streams, _, _, _ = setup
    stream = owner.stream()
    detector.send(VADEventType.START_OF_SPEECH, b"aa")
    await receive(stream, SpeechEventType.INTERIM_TRANSCRIPT)
    stream.end_input()
    events = [event async for event in stream]
    assert [e.type for e in events] == [
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.RECOGNITION_USAGE,
        SpeechEventType.END_OF_SPEECH,
    ]
    assert events[0].alternatives[0].text == "Ice"
    assert streams[0].audio == [b"aa"]


async def test_backlog_is_bounded_in_bytes_and_does_not_finalize(setup):
    owner, detector, streams, _, _, _ = setup
    stream = owner.stream()
    detector.send(VADEventType.START_OF_SPEECH, bytes(320002))
    with pytest.raises(APIError, match="backlog"):
        await receive(stream, SpeechEventType.FINAL_TRANSCRIPT)
    assert not streams
    assert detector.closed


async def test_language_override_and_silence_final(setup, monkeypatch):
    owner, detector, streams, _, _, _ = setup

    async def silence(self):
        await self.read_audio()
        yield self.sdk.TranscriptionFinal(text="", language="es")

    monkeypatch.setattr(SDKStream, "updates", silence)
    stream = owner.stream(language="es")
    detector.utterance(b"aa")
    final = await receive(stream, SpeechEventType.FINAL_TRANSCRIPT)
    assert final.alternatives[0].text == ""
    assert final.alternatives[0].language == "es"
    assert streams[0].options["language"] == "es"


async def test_revised_hypotheses_replace_previous_text(setup, monkeypatch):
    owner, detector, streams, _, _, _ = setup
    owner._output_contract = "revised_hypotheses"

    async def revised(self):
        await self.read_audio()
        yield self.sdk.TranscriptionPartial(text="I scream")
        yield self.sdk.TranscriptionPartial(text="Ice cream")
        yield self.sdk.TranscriptionFinal(text="Ice cream.", language="en")

    monkeypatch.setattr(SDKStream, "updates", revised)
    stream = owner.stream()
    detector.utterance(b"aa")
    texts = []
    while True:
        event = await asyncio.wait_for(anext(stream), 2)
        if event.alternatives:
            texts.append(event.alternatives[0].text)
        if event.type == SpeechEventType.FINAL_TRANSCRIPT:
            break
    assert texts == ["I scream", "Ice cream", "Ice cream."]


async def test_flush_restarts_vad_and_resamples_live_frames(setup, monkeypatch):
    owner, _, streams, _, _, _ = setup
    detectors = []

    class LiveDetector(DetectorStream):
        def __init__(self):
            super().__init__()
            self.started = False
            detectors.append(self)

        def push_frame(self, frame):
            assert frame.sample_rate == 16000
            kind = VADEventType.INFERENCE_DONE if self.started else VADEventType.START_OF_SPEECH
            self.started = True
            self.send(kind, bytes(frame.data))

    owner._vad = SimpleNamespace(stream=LiveDetector)
    stream = owner.stream()
    frame = _frame(48000, data=bytes(9600))
    stream.push_frame(frame)
    stream.flush()
    stream.push_frame(frame)
    stream.end_input()
    events = [event async for event in stream]
    assert len([e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT]) == 2
    assert len(streams) == 2
    assert all(sum(map(len, s.audio)) == 3200 for s in streams)
    assert all(d.closed for d in detectors)


@pytest.mark.parametrize("contract", ["append_fragments", "revised_hypotheses"])
async def test_later_utterance_uploads_while_first_final_waits(
    setup, monkeypatch, contract, collect_events, sdk
):
    owner, detector, streams, _, _, _ = setup
    owner._output_contract = contract
    release = asyncio.Event()
    first_eof = asyncio.Event()
    second_done = asyncio.Event()

    async def delayed(self):
        first = self is streams[0]
        self.request_id = "first" if first else "second"
        text = ""
        async for chunk in self.source:
            self.audio.append(chunk)
            text += "a" if first else "b"
            if contract == "append_fragments":
                yield self.sdk.TranscriptionDelta(text="a" if first else "b")
            else:
                yield self.sdk.TranscriptionPartial(text=text)
        if first:
            first_eof.set()
            await release.wait()
        yield self.sdk.TranscriptionFinal(text=text, language=self.options["language"])
        if not first:
            second_done.set()

    monkeypatch.setattr(SDKStream, "updates", delayed)
    stream = owner.stream(conn_options=APIConnectOptions(timeout=120))
    observed, reader = collect_events(stream)
    detector.send(VADEventType.START_OF_SPEECH, b"aa")
    await wait_until(lambda: streams and streams[0].audio)
    detector.send(VADEventType.END_OF_SPEECH, b"aa")
    await asyncio.wait_for(first_eof.wait(), 2)
    owner.update_options(mode="verbatim", language="es", context_terms=["Second"])
    detector.send(VADEventType.START_OF_SPEECH, bytes(32000))
    await wait_until(lambda: len(streams) == 2 and len(streams[1].audio) == 1)
    # Eleven seconds consumed while the previous final is still unavailable.
    for count in range(2, 12):
        detector.send(VADEventType.INFERENCE_DONE, bytes(32000))
        await wait_until(lambda count=count: len(streams[1].audio) == count)
    detector.send(VADEventType.END_OF_SPEECH, bytes(352000))
    await asyncio.wait_for(second_done.wait(), 2)
    assert sum(e.type == SpeechEventType.START_OF_SPEECH for e in observed) == 1
    assert not any(e.type == SpeechEventType.END_OF_SPEECH for e in observed)
    assert not any(e.type == SpeechEventType.FINAL_TRANSCRIPT for e in observed)
    assert streams[1].options["language"] == "es"
    assert streams[1].options["mode"] == sdk.TranscriptionMode.VERBATIM
    assert streams[1].options["context_terms"] == ("Second",)
    stream.end_input()
    release.set()
    await asyncio.wait_for(reader, 2)
    finals = [e for e in observed if e.type == SpeechEventType.FINAL_TRANSCRIPT]
    assert [e.request_id for e in finals] == ["first", "second"]
    assert [e.alternatives[0].text for e in finals] == ["a", "b" * 11]
    ends = [e.speech_end_time for e in observed if e.type == SpeechEventType.END_OF_SPEECH]
    assert [e.speech_end_time for e in finals] == ends
    assert [
        e.type
        for e in observed
        if e.type
        in (
            SpeechEventType.START_OF_SPEECH,
            SpeechEventType.FINAL_TRANSCRIPT,
            SpeechEventType.END_OF_SPEECH,
        )
    ] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
    ] * 2
    usage = [e.recognition_usage.audio_duration for e in observed if e.recognition_usage]
    assert usage == pytest.approx([1 / 16000, 11])
    assert all(s.closed for s in streams)


@pytest.mark.parametrize("failing", [0, 1, None])
async def test_parallel_requests_cancel_on_error_or_close(
    setup, monkeypatch, failing, collect_events
):
    owner, detector, streams, _, _, _ = setup
    fail_now = asyncio.Event()

    async def delayed(self):
        index = streams.index(self)
        await self.read_audio()
        await fail_now.wait()
        if index == failing:
            raise self.sdk.RimeUnavailableError(
                "private-provider-content", request_id="failed-request"
            )
        await asyncio.Event().wait()
        yield  # keep this an async generator

    monkeypatch.setattr(SDKStream, "updates", delayed)
    stream = owner.stream()
    observed, reader = collect_events(stream)
    for count in (1, 2):
        detector.utterance(b"aa")
        await wait_until(lambda count=count: len(streams) == count and streams[-1].audio)
    if failing is None:
        await owner.aclose()
    else:
        fail_now.set()
        with pytest.raises(APIStatusError):
            await asyncio.wait_for(reader, 2)
    assert all(s.closed for s in streams)
    assert detector.closed
    assert not any(
        e.type in (SpeechEventType.FINAL_TRANSCRIPT, SpeechEventType.END_OF_SPEECH)
        for e in observed
    )


async def test_flush_accepts_new_audio_before_previous_final(setup, monkeypatch, collect_events):
    owner, _, streams, _, _, _ = setup
    release = asyncio.Event()
    first_eof = asyncio.Event()
    second_eof = asyncio.Event()

    class LiveDetector(DetectorStream):
        def push_frame(self, frame):
            self.send(VADEventType.START_OF_SPEECH, bytes(frame.data))

    owner._vad = SimpleNamespace(stream=LiveDetector)

    async def delayed(self):
        await self.read_audio()
        if self is streams[0]:
            first_eof.set()
            await release.wait()
        else:
            second_eof.set()
        yield self.sdk.TranscriptionFinal(text=str(streams.index(self)), language="en")

    monkeypatch.setattr(SDKStream, "updates", delayed)
    stream = owner.stream()
    frame = _frame(data=bytes(3200))
    observed, reader = collect_events(stream)
    stream.push_frame(frame)
    stream.flush()
    await asyncio.wait_for(first_eof.wait(), 2)
    stream.push_frame(frame)
    stream.end_input()
    await asyncio.wait_for(second_eof.wait(), 2)
    assert not any(e.type == SpeechEventType.FINAL_TRANSCRIPT for e in observed)
    release.set()
    await asyncio.wait_for(reader, 2)
    assert [
        e.alternatives[0].text for e in observed if e.type == SpeechEventType.FINAL_TRANSCRIPT
    ] == ["0", "1"]


async def test_pending_utterances_are_bounded(setup, monkeypatch, collect_events):
    owner, detector, streams, _, _, _ = setup

    async def delayed(self):
        await self.read_audio()
        await asyncio.Event().wait()
        yield

    monkeypatch.setattr(SDKStream, "updates", delayed)
    stream = owner.stream()

    _, reader = collect_events(stream)
    for count in range(1, 9):
        detector.utterance(b"aa")
        await wait_until(lambda count=count: len(streams) == count and streams[-1].audio)
    detector.send(VADEventType.START_OF_SPEECH, b"aa")
    with pytest.raises(APIError, match="pending utterance limit"):
        await asyncio.wait_for(reader, 2)
    assert len(streams) == 8
    assert all(s.closed for s in streams)
    assert detector.closed


@pytest.mark.parametrize("contract", ["append_fragments", "revised_hypotheses"])
@pytest.mark.parametrize("second_finishes_first", [False, True])
@pytest.mark.parametrize("commit_between_events", [False, True])
async def test_concurrent_results_reach_audio_recognition(
    setup,
    monkeypatch,
    contract,
    second_finishes_first,
    commit_between_events,
    cleanup,
    collect_events,
):
    """Exercise the real turn consumer, including both possible commit schedules."""
    owner, detector, streams, _, _, _ = setup
    owner._output_contract = contract
    release = [asyncio.Event(), asyncio.Event()]
    uploaded = [asyncio.Event(), asyncio.Event()]

    async def delayed(self):
        index = streams.index(self)
        self.request_id = str(index)
        text = "first" if index == 0 else "second"
        async for chunk in self.source:
            self.audio.append(chunk)
            if contract == "append_fragments":
                yield self.sdk.TranscriptionDelta(text=text)
            else:
                yield self.sdk.TranscriptionPartial(text=text)
        uploaded[index].set()
        await release[index].wait()
        yield self.sdk.TranscriptionFinal(text=text, language="en")

    monkeypatch.setattr(SDKStream, "updates", delayed)
    session = AgentSession(vad=None, turn_handling={"turn_detection": "stt"})
    hooks = MagicMock()
    hooks.retrieve_chat_ctx.return_value = llm.ChatContext()
    hooks.on_end_of_turn.return_value = True
    recognition = AudioRecognition(
        session,
        hooks=hooks,
        endpointing=BaseEndpointing(min_delay=0, max_delay=0),
        stt=owner,
        vad=None,
        interruption_detection=None,
        turn_detection="stt",
    )
    stream = owner.stream()
    cleanup.push_async_callback(recognition._aclose)

    async def consume(event):
        await recognition._on_stt_event(event)
        # Both committing between events and draining queued events must be safe.
        if commit_between_events and event.type == SpeechEventType.END_OF_SPEECH:
            task = recognition._end_of_turn_task
            if task is not None:
                await task

    observed, reader = collect_events(stream, consume)
    for index in range(2):
        detector.utterance(b"aa")
        await asyncio.wait_for(uploaded[index].wait(), 2)
    stream.end_input()
    if second_finishes_first:
        release[1].set()
        await wait_until(lambda: streams[1].closed)
    release[0].set()
    if not second_finishes_first:
        await wait_until(
            lambda: (
                any(e.type == SpeechEventType.FINAL_TRANSCRIPT for e in observed)
                and sum(e.type == SpeechEventType.START_OF_SPEECH for e in observed) == 2
            )
        )
        # B is still awaiting its final. A's completion must not leave B
        # in the ended/committed state established by an earlier boundary.
        assert recognition._speaking
        release[1].set()
    await asyncio.wait_for(reader, 2)
    if recognition._end_of_turn_task is not None:
        await asyncio.wait_for(recognition._end_of_turn_task, 2)
    turns = [call.args[0].new_transcript for call in hooks.on_end_of_turn.call_args_list]
    assert " ".join(turns) == "first second"
    assert recognition._audio_transcript == ""
    if commit_between_events:
        assert turns == ["first", "second"]
    assert [
        call.args[0].alternatives[0].text for call in hooks.on_final_transcript.call_args_list
    ] == ["first", "second"]


async def test_speech_onset_accounts_for_buffering_before_deferred_publication(
    setup, monkeypatch, collect_events
):
    from livekit.plugins.rime import _stt_stream

    owner, detector, streams, _, _, _ = setup
    clock = [100.0]
    # Patch only the adapter clock, leaving asyncio and the test runner untouched.
    monkeypatch.setattr(_stt_stream, "time", SimpleNamespace(time=lambda: clock[0]))
    release = asyncio.Event()
    first_eof = asyncio.Event()

    async def delayed(self):
        await self.read_audio()
        if self is streams[0]:
            first_eof.set()
            await release.wait()
        yield self.sdk.TranscriptionFinal(text="hello", language="en")

    monkeypatch.setattr(SDKStream, "updates", delayed)
    stream = owner.stream()
    observed, reader = collect_events(stream)
    detector.send(
        VADEventType.START_OF_SPEECH,
        b"aa",
        speech_duration=0.3,
        inference_duration=0.02,
    )
    await wait_until(lambda: streams and streams[0].audio and observed)
    detector.send(VADEventType.END_OF_SPEECH, b"aa")
    await asyncio.wait_for(first_eof.wait(), 2)
    clock[0] = 200.0
    detector.send(
        VADEventType.START_OF_SPEECH,
        b"bb",
        speech_duration=0.4,
        inference_duration=0.03,
    )
    detector.send(VADEventType.END_OF_SPEECH, b"bb")
    await wait_until(lambda: len(streams) == 2 and streams[1].closed)
    starts = [e for e in observed if e.type == SpeechEventType.START_OF_SPEECH]
    assert len(starts) == 1  # B's start is held behind A's final.
    assert starts[0].speech_start_time == pytest.approx(99.68)
    clock[0] = 300.0
    stream.end_input()
    release.set()
    await asyncio.wait_for(reader, 2)
    assert [
        e.speech_start_time for e in observed if e.type == SpeechEventType.START_OF_SPEECH
    ] == pytest.approx([99.68, 199.57])
