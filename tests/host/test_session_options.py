"""A voice session's options: temporary overrides layered over the base options."""

from __future__ import annotations

import pytest

from livekit.agents import Agent, VoiceSession
from tests.fakes import FakeLLM, until


# Catches: an override leaking past its block when the block raises.
async def test_an_override_applies_inside_its_block_and_the_base_returns_even_after_a_raise() -> (
    None
):
    async with VoiceSession(
        llm=FakeLLM(), vad=None, turn_handling={"endpointing": {"min_delay": 0.3}}
    ) as session:
        await session.start(Agent(instructions="Help."))

        def min_delay() -> float:
            return session.options["endpointing"]["min_delay"]

        with pytest.raises(RuntimeError, match="interaction failed"):
            async with session.override_options(turn_handling={"endpointing": {"min_delay": 1.2}}):
                await until(lambda: min_delay() == 1.2, "the override never applied")
                raise RuntimeError("interaction failed")

        await until(lambda: min_delay() == 0.3, "the base never came back")
