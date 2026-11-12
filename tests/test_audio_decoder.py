import asyncio
import contextlib
import logging
import os
import time

import pytest

from livekit.agents.utils import media
from livekit.agents.utils.media import AudioStreamDecoder, audio_frames_from_file

# Decodes audio on background threads / executors with blocking waits; it deadlocks when forced
# to share one event loop with other tests.
pytestmark = pytest.mark.unit

TEST_AUDIO_FILEPATH = os.path.join(os.path.dirname(__file__), "change-sophie.opus")


@pytest.mark.asyncio
async def test_aclose_while_probing_is_not_an_error(caplog: pytest.LogCaptureFixture) -> None:
    """Closing a decoder mid-probe is intentional and must not be reported as a decode failure."""
    with open(TEST_AUDIO_FILEPATH, "rb") as f:
        head = f.read(16)  # far less than av.open needs to probe the container

    decoder = AudioStreamDecoder()
    decoder.push(head)
    await asyncio.sleep(0.05)  # the decode thread is now blocked reading inside av.open

    with caplog.at_level(logging.DEBUG, logger="livekit.agents"):
        await decoder.aclose()
        await asyncio.sleep(0.05)

    assert [r.getMessage() for r in caplog.records if "error decoding" in r.getMessage()] == []


@pytest.mark.asyncio
async def test_aborted_read_does_not_signal_end_of_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """The BackgroundAudioPlayer stop() path: an aborted read is a truncated file, not an EOF.

    Signalling the end of input hands the partial file to PyAV as a complete container, which
    it then reports as invalid data.
    """
    pushes: list[None] = []
    started = asyncio.Event()
    ends: list[None] = []

    class SpyDecoder:
        def __init__(self, *, sample_rate: int, num_channels: int) -> None:
            self._decoder = AudioStreamDecoder(sample_rate=sample_rate, num_channels=num_channels)

        def push(self, chunk: bytes) -> None:
            pushes.append(None)
            started.set()
            time.sleep(0.005)  # hold the reader mid-file for the whole test
            self._decoder.push(chunk)

        def end_input(self) -> None:
            ends.append(None)
            self._decoder.end_input()

        def __aiter__(self) -> AudioStreamDecoder:
            return self._decoder

        async def aclose(self) -> None:
            await self._decoder.aclose()

    monkeypatch.setattr(media, "AudioStreamDecoder", SpyDecoder)

    gen = audio_frames_from_file(TEST_AUDIO_FILEPATH)

    async def drain() -> None:
        async for _ in gen:
            pass

    task = asyncio.create_task(drain())
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    await gen.aclose()  # what PlayHandle.stop() ultimately triggers

    assert pushes, "the reader never ran, so the abort was never exercised"
    assert ends == []


@pytest.mark.asyncio
async def test_closing_partial_decoders_releases_admission_while_wrappers_remain_alive(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with open(TEST_AUDIO_FILEPATH, "rb") as source:
        head = source.read(16)
    closed = []
    for _ in range(40):
        decoder = AudioStreamDecoder(format="audio/ogg")
        decoder.push(head)
        await decoder.aclose()
        closed.append(decoder)

    # Every closed decoder released its worker admission: a new one still opens.
    with caplog.at_level(logging.ERROR, logger="livekit.agents"):
        replacement = AudioStreamDecoder(format="audio/ogg")
        replacement.push(head)
        await replacement.aclose()
    assert [r.getMessage() for r in caplog.records if "error decoding" in r.getMessage()] == []
