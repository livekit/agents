"""Rime STT audio conversion, SDK integration, lifecycle, and LiveKit VAD adaptation."""

from __future__ import annotations

import asyncio
import io
import sys
import traceback
import wave
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
    vad,
)
from livekit.plugins import rime

pytestmark = pytest.mark.unit

if sys.version_info >= (3, 11):
    import rimelabs_sdk as sdk
else:
    pytest.skip("Rime STT requires Python 3.11+", allow_module_level=True)


NO_RETRY = APIConnectOptions(max_retry=0, timeout=2)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
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
    sample_rate: int = 16000, channels: int = 1, samples: int | None = None
) -> rtc.AudioFrame:
    samples = samples if samples is not None else sample_rate // 10
    # Asymmetric, loud channels detect both accidental channel selection and int16 overflow.
    data = np.tile(np.arange(channels, dtype=np.int16) * 1000 + 20000, (samples, 1))
    return rtc.AudioFrame(
        data=data.tobytes(),
        sample_rate=sample_rate,
        num_channels=channels,
        samples_per_channel=samples,
    )


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

    assert result.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert result.request_id == "request-123"
    assert result.alternatives[0].text == "Hello."
    assert result.alternatives[0].language == "en"
    assert result.alternatives[0].words is None
    assert len(metrics) == 1
    assert metrics[0].request_id == "request-123"
    assert metrics[0].audio_duration == pytest.approx(0.2)
    assert metrics[0].metadata.model_provider == "Rime"
    assert not recognizer.capabilities.streaming
    assert not recognizer.capabilities.interim_results


async def test_options_are_snapshotted_and_keyterms_merged(client: MagicMock) -> None:
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
    recognizer: rime.STT, client: MagicMock
) -> None:
    client.stt.transcribe.return_value = sdk.Transcription(text="", language="es")
    event = await recognizer.recognize(_frame(), conn_options=NO_RETRY)
    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
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
        (sdk.RimeTimeoutError, APITimeoutError, True),
        (sdk.RimeUnavailableError, APIConnectionError, True),
        (sdk.RimeAuthenticationError, APIStatusError, False),
        (sdk.RimePermissionError, APIStatusError, False),
        (sdk.RimeInputError, APIStatusError, False),
        (sdk.RimeAudioFormatError, APIStatusError, False),
        (sdk.RimeResourceLimitError, APIStatusError, False),
        (sdk.RimeCancelledError, APIStatusError, False),
        (sdk.RimeStreamError, APIStatusError, False),
    ],
)
async def test_safe_errors_and_retry_policy(
    recognizer: rime.STT,
    client: MagicMock,
    sdk_error: type[sdk.RimeError],
    livekit_error: type[APIError],
    retryable: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = "private-request-content"
    client.stt.transcribe.side_effect = sdk_error(marker, request_id="request-failed")
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
        sdk_error(marker),
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
    monkeypatch: pytest.MonkeyPatch, finish: str
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
                    self._emit(vad.VADEventType.START_OF_SPEECH, [])
                frames.append(item)
            elif frames:
                self._emit(vad.VADEventType.END_OF_SPEECH, frames)
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


async def test_livekit_stream_adapter_transcribes_each_utterance(
    recognizer: rime.STT, client: MagicMock
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
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.END_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
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


def test_append_requires_vad_and_compatible_sdk(client, monkeypatch):
    with pytest.raises(ValueError, match="requires a vad"):
        rime.STT(output_contract="append_fragments")
    monkeypatch.delattr(sdk, "TranscriptionOutputContract", raising=False)
    with pytest.raises(ImportError, match="TranscriptionOutputContract"):
        rime.STT(vad=_SegmentVAD(), output_contract="append_fragments")
