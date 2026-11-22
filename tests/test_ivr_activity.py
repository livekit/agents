import asyncio
from typing import Any

import pytest

from livekit.agents.utils import EventEmitter
from livekit.agents.voice.events import AgentStateChangedEvent
from livekit.agents.voice.ivr.ivr_activity import IVRActivity, TfidfLoopDetector

pytestmark = pytest.mark.unit

_MENU = [
    "Welcome to automated phone system",
    "Type 1 for sales",
    "Type 2 for support",
    "Type 3 for billing",
    "Type 4 for technical support",
]


@pytest.mark.parametrize(
    ("transcripts", "expected_loops"),
    [
        pytest.param(
            [*_MENU, "Type 5 for account management", "Type 6 for billing", "Type 7 for users"],
            0,
            id="unique-prompts",
        ),
        # the third repeat in a row is a loop, and every repeat after it
        pytest.param([*_MENU, *_MENU], 3, id="repeated-menu"),
        pytest.param(
            [
                "Welcome to automated phone system",
                "Press 1 for sales",
                "Press 2 for support",
                "Press 1 for sales",
                "Press 2 for support",
                "Press 1 for sales",
                "Here's a new announcement never heard before",
                "Press 2 for support",
                "Press 1 for sales",
                "Press 2 for support",
            ],
            2,
            id="novel-speech-resets",
        ),
        pytest.param(
            [
                "Welcome to automated phone system",
                "Type 1 for sales, type 2 for support, type 3 for billing",
                "Again, type 1 for sales, type 2 for support, type 3 for billing",
                "And again, type 1 for sales, type 2 for support, type 3 for billing",
                "Repeat, type 1 for sales, type 2 for support, and type 3 for billing",
            ],
            1,
            id="minor-variations",
        ),
        pytest.param(
            [
                "Hey there, I'm trying to get some help with my checking account today.",
                "Sure, before we dive in can you confirm the last four digits of the account?",
                "Yeah, it's 0 2 4 6, and I think there's a $35 fee that shouldn't be there.",
                "Thanks, give me just a moment while I pull up those recent transactions.",
                "Alright, I see the fee. Let me reverse it and send you a confirmation email.",
            ],
            0,
            id="human-small-talk",
        ),
    ],
)
def test_tfidf_loop_detection(transcripts: list[str], expected_loops: int) -> None:
    detector = TfidfLoopDetector()
    loops = 0
    for transcript in transcripts:
        detector.add_chunk(transcript)
        loops += detector.check_loop_detection()
    assert loops == expected_loops


class _FakeSession(EventEmitter[Any]):
    def __init__(self, *, agent_state: str, user_state: str) -> None:
        super().__init__()
        self.agent_state = agent_state
        self.user_state = user_state
        self.replies = 0

    def generate_reply(self, **kwargs: Any) -> None:
        self.replies += 1

    def set_agent_state(self, state: str) -> None:
        old, self.agent_state = self.agent_state, state
        self.emit(
            "agent_state_changed",
            AgentStateChangedEvent(old_state=old, new_state=state),  # type: ignore[arg-type]
        )


@pytest.mark.virtual_time
async def test_silence_check_armed_when_started_while_idle() -> None:
    # AMD starts IVR detection after the IVR greeting, with both sides already listening:
    # no state event follows, so the activity has to start from the current states
    session = _FakeSession(agent_state="listening", user_state="listening")
    ivr = IVRActivity(session, max_silence_duration=0.05)  # type: ignore[arg-type]
    await ivr.start()
    try:
        await asyncio.sleep(0.3)
        assert session.replies == 1
    finally:
        await ivr.aclose()


@pytest.mark.virtual_time
async def test_silence_check_waits_for_agent_to_go_idle() -> None:
    session = _FakeSession(agent_state="speaking", user_state="listening")
    ivr = IVRActivity(session, max_silence_duration=0.05)  # type: ignore[arg-type]
    await ivr.start()
    try:
        await asyncio.sleep(0.2)
        assert session.replies == 0

        session.set_agent_state("listening")
        await asyncio.sleep(0.3)
        assert session.replies == 1
    finally:
        await ivr.aclose()
