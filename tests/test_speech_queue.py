"""``AgentActivity._schedule_speech`` must enqueue exactly one entry per speech.

The queue is a heap of ``(-priority, timestamp, speech)`` tuples. Two entries whose
priority and timestamp tie fall through to comparing the ``SpeechHandle`` objects,
which are not orderable — ``heapq.heappush`` raises ``TypeError`` after it has
already appended the item. The retry loop then pushes a *second* copy of the same
speech, so the queue holds one duplicate and can lose its heap invariant.
"""

from __future__ import annotations

import time

import pytest

from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_session import FakeActions, create_session
from .test_agent_session import MyAgent, _close_test_session

pytestmark = pytest.mark.unit


def _make_activity() -> AgentActivity:
    return AgentActivity(MyAgent(), create_session(FakeActions()))


async def test_tied_timestamps_enqueue_each_speech_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clock whose reads tie must not duplicate entries in the speech queue."""
    activity = _make_activity()
    activity._scheduling_paused = False
    # a coarse perf clock (e.g. Windows QPC) can return the same value for
    # back-to-back scheduling calls
    monkeypatch.setattr(time, "perf_counter_ns", lambda: 1_000)
    try:
        first = SpeechHandle.create()
        second = SpeechHandle.create()
        activity._schedule_speech(first, priority=SpeechHandle.SPEECH_PRIORITY_NORMAL)
        activity._schedule_speech(second, priority=SpeechHandle.SPEECH_PRIORITY_NORMAL)

        queued = [speech for _, _, speech in activity._speech_q]
        assert len(queued) == 2, f"expected one queue entry per speech, got {queued}"
        assert sorted(queued, key=id) == sorted([first, second], key=id)
    finally:
        await _close_test_session(activity._session)
