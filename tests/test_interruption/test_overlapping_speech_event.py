import numpy as np

from livekit.agents.inference import OverlappingSpeechEvent


def test_interruption_event_serialization() -> None:
    ev = OverlappingSpeechEvent(type="user_interruption_detected")
    ev.speech_input = np.array([1, 2, 3, 4, 5])
    assert ev.model_dump()["speech_input"] is None
    assert ev.model_dump(mode="json")["speech_input"] is None
    assert ev.speech_input is not None
