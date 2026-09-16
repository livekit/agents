import asyncio

import pytest

from livekit.agents import io
from livekit.agents.voice.transcription.synchronizer import TranscriptSynchronizer

pytestmark = pytest.mark.unit


class _TextOutput(io.TextOutput):
    def __init__(self) -> None:
        super().__init__(label="test", next_in_chain=None)

    async def capture_text(self, text: str) -> None:
        pass

    def flush(self) -> None:
        pass


class _AudioOutput(io.AudioOutput):
    def __init__(self) -> None:
        super().__init__(label="test", capabilities=io.AudioOutputCapabilities(pause=False))

    async def capture_frame(self, frame) -> None:
        pass

    def flush(self) -> None:
        pass

    def clear_buffer(self) -> None:
        pass

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass


def test_barrier_cancellation_does_not_cancel_segment_rotation() -> None:
    async def run() -> None:
        sync = TranscriptSynchronizer(
            next_in_chain_text=_TextOutput(), next_in_chain_audio=_AudioOutput()
        )
        release = asyncio.Event()

        async def rotate() -> None:
            await release.wait()

        rotation = asyncio.create_task(rotate())
        sync._rotate_segment_atask = rotation
        waiter = asyncio.create_task(sync.barrier())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not rotation.cancelled()
        release.set()
        await sync.barrier()

    asyncio.run(run())
