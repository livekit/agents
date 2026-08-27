import pytest
from behaviors.frontend_attributes import frontend_attributes
from behaviors.user_away import CHECK_IN_INSTRUCTIONS, check_in_when_user_away

from livekit.agents import AgentSession, UserStateChangedEvent

pytestmark = pytest.mark.unit


def test_frontend_attributes_carry_the_configured_tts_voice() -> None:
    assert frontend_attributes(tts_voice="Nate") == {"tts_voice": "Nate"}
    assert frontend_attributes(tts_voice=None) == {}


@pytest.mark.asyncio
async def test_user_away_checkin() -> None:
    session = AgentSession()
    check_in_when_user_away(session)

    replies = []
    session.generate_reply = lambda **kwargs: replies.append(kwargs)  # type: ignore[method-assign]

    session.emit(
        "user_state_changed",
        UserStateChangedEvent(old_state="listening", new_state="away"),
    )
    assert replies == [
        {
            "instructions": CHECK_IN_INSTRUCTIONS,
            "allow_interruptions": True,
        }
    ]

    session.emit(
        "user_state_changed",
        UserStateChangedEvent(old_state="away", new_state="speaking"),
    )
    assert len(replies) == 1
