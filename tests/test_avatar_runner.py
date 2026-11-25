from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from livekit.agents.voice.avatar import AvatarOptions, AvatarRunner, QueueAudioOutput

pytestmark = pytest.mark.unit


class _VideoGen:
    async def push_audio(self, frame: object) -> None:
        pass

    def clear_buffer(self) -> None:
        pass

    def __aiter__(self) -> _VideoGen:
        return self

    async def __anext__(self) -> object:
        raise StopAsyncIteration


def _make_runner() -> AvatarRunner:
    room = MagicMock()
    room.isconnected.return_value = False
    return AvatarRunner(
        room,
        audio_recv=QueueAudioOutput(sample_rate=16000),
        video_gen=_VideoGen(),
        options=AvatarOptions(
            video_width=64,
            video_height=64,
            video_fps=25,
            audio_sample_rate=16000,
            audio_channels=1,
        ),
    )


async def test_wait_for_complete_before_start_raises_not_started() -> None:
    runner = _make_runner()
    with pytest.raises(RuntimeError, match="not started"):
        await runner.wait_for_complete()
    await runner.aclose()


async def test_aclose_before_start() -> None:
    # e.g. a shutdown callback after start() failed or was never reached
    runner = _make_runner()
    await runner.aclose()
