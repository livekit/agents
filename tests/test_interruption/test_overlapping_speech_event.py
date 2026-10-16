from unittest.mock import MagicMock

import numpy as np
import pytest

from livekit.agents.inference import OverlappingSpeechEvent
from livekit.agents.inference.interruption import InterruptionWebSocketStream

pytestmark = pytest.mark.unit


def test_interruption_event_serialization() -> None:
    ev = OverlappingSpeechEvent(type="overlapping_speech")
    ev.speech_input = np.array([1, 2, 3, 4, 5])
    assert ev.model_dump()["speech_input"] is None
    assert ev.model_dump(mode="json")["speech_input"] is None
    assert ev.speech_input is not None


async def test_agent_ended_overlap_is_not_counted_as_backchannel() -> None:
    stream = InterruptionWebSocketStream.__new__(InterruptionWebSocketStream)
    stream._model = MagicMock(model="test-model", provider="test-provider")

    async def _events():
        yield OverlappingSpeechEvent(is_interruption=False, agent_ended=True)

    await stream._metrics_monitor_task(_events())

    metrics = stream._model.emit.call_args.args[1]
    assert metrics.num_interruptions == 0
    assert metrics.num_backchannels == 0
