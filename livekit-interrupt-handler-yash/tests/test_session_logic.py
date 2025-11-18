import pytest
from unittest.mock import AsyncMock

from agent.session_manager import SessionManager
from agent.state import AgentState


@pytest.mark.asyncio
async def test_session_interrupt_logic():
    # Mock LiveKit session + agent
    mock_session = AsyncMock()
    mock_agent = AsyncMock()
    mock_config = AsyncMock()

    sm = SessionManager(mock_session, mock_agent, mock_config)

    # agent currently speaking
    sm.state.agent_is_speaking = True

    # mock interrupt filter decision
    sm.interrupt_filter.should_interrupt = AsyncMock(return_value=True)

    await sm.handle_user_transcript("stop", 0.98)

    # Expect interrupt
    mock_session.interrupt.assert_called_once()
