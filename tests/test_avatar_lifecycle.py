"""`AvatarSession`'s lifecycle: the join wait and its timeout, the close on the job's shutdown,
and what a close ends."""

import asyncio

import pytest

from livekit import rtc
from livekit.agents import VoiceSession
from livekit.agents.metrics import AvatarMetrics
from livekit.agents.voice.avatar import AvatarSession

pytestmark = pytest.mark.unit


class Avatar(AvatarSession):
    closes = 0

    @property
    def avatar_identity(self) -> str:
        return "avatar"

    @property
    def provider(self) -> str:
        return "test-provider"

    async def aclose(self) -> None:
        self.closes += 1
        await super().aclose()


class Job:
    """What the avatar's lifecycle reads of its `JobContext`."""

    def __init__(self) -> None:
        self.shutdown: list[object] = []
        self._server_env: dict[str, str] = {}

    def _core_on_avatar_shutdown(self, close: object) -> None:
        self.shutdown.append(close)


async def test_wait_for_join_times_out_without_ending_the_wait_and_a_close_ends_it():
    avatar = Avatar()
    await avatar.wait_for_join(timeout=0)  # not started: returns at once
    await avatar.start(VoiceSession(vad=None), rtc.Room())
    with pytest.raises(TimeoutError):
        await avatar.wait_for_join(timeout=0.05)
    waiting = asyncio.ensure_future(avatar.wait_for_join(timeout=None))
    await asyncio.sleep(0.05)
    assert not waiting.done()
    await avatar.aclose()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(waiting, 1)
    await avatar.wait_for_join(timeout=0)  # closed: returns at once
    await avatar.aclose()


async def test_the_jobs_shutdown_runs_the_subclass_close(monkeypatch):
    job = Job()
    monkeypatch.setattr("livekit.agents.voice.avatar.get_job_context", lambda **_: job)
    avatar = Avatar()
    await avatar.start(VoiceSession(vad=None), rtc.Room())
    assert len(job.shutdown) == 1
    close = job.shutdown[0]
    assert close == avatar.aclose  # the subclass's own aclose
    await close()  # an unconnected room removes nobody
    assert avatar.closes == 1


async def test_concurrent_closes_wait_for_one_close_and_a_new_start_begins_again():
    avatar = Avatar()
    session = VoiceSession(vad=None)
    await avatar.start(session, rtc.Room())
    await asyncio.gather(avatar.aclose(), avatar.aclose())
    await avatar.start(session, rtc.Room())
    with pytest.raises(TimeoutError):
        await avatar.wait_for_join(timeout=0.01)
    await avatar.aclose()


def test_metrics_collected_emits_the_lifecycles_metrics():
    avatar = Avatar()
    seen: list[AvatarMetrics] = []
    avatar.on("metrics_collected", seen.append)
    metrics = AvatarMetrics(timestamp=1.0, playback_latency=0.2)
    avatar._core_metrics_collected(metrics)
    assert seen == [metrics]
