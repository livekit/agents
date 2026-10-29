import pytest

from livekit.agents.voice.amd import AMDCategory, AMDPredictionEvent

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("category", "human", "machine"),
    [
        (AMDCategory.HUMAN, True, False),
        (AMDCategory.MACHINE_VM, False, True),
        (AMDCategory.UNCERTAIN, False, False),
    ],
)
def test_a_verdict_says_whether_a_human_or_a_machine_answered(
    category: AMDCategory, human: bool, machine: bool
) -> None:
    # The properties call into _core; twice they named a _core function that did not exist.
    event = AMDPredictionEvent(
        speech_duration=1.0, category=category, reason="llm", transcript="hi", delay=0.5
    )
    assert (event.is_human, event.is_machine) == (human, machine)
