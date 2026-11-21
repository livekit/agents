import asyncio

import pytest

from livekit.agents import Agent, AgentSession

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_next", [False, True])
async def test_handoff_to_active_agent_does_not_strand_session(queue_next: bool) -> None:
    current = Agent(instructions="Current agent")
    occupied = Agent(instructions="Running in another session")
    replacement = Agent(instructions="Valid handoff target")

    async with AgentSession() as source, AgentSession() as other:
        await source.start(agent=current)
        await other.start(agent=occupied)
        source.update_agent(occupied)
        rejected = source._update_activity_atask
        assert rejected is not None

        if queue_next:
            source.update_agent(replacement)
        else:
            # Let the attempt settle even if the SDK surfaces the rejection as a task error.
            await asyncio.gather(rejected, return_exceptions=True)
            assert source.current_agent is current
            assert source._activity is not None
            assert not source._activity._new_turns_blocked
            source.update_agent(replacement)

        accepted = source._update_activity_atask
        assert accepted is not None
        await asyncio.wait_for(accepted, timeout=5)
        assert source.current_agent is replacement
        assert source._activity is not None
        assert source._activity.agent is replacement
        assert other.current_agent is occupied
