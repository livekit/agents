"""Example voice agent that uses the skills system.

This agent demonstrates two ways to use skills:
1. Pre-activated skills (weather) — available immediately
2. Registry-based skills (calendar) — the LLM can activate on demand

Run with:
    python agent.py console
"""

from pathlib import Path

from livekit.agents import Agent, AgentSession, JobContext, JobProcess, cli
from livekit.agents.skills import SkillRegistry, load_skill_from_directory

SKILLS_DIR = Path(__file__).parent / "skills"


def prewarm(proc: JobProcess) -> None:
    pass


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    # Load the weather skill directly (pre-activated)
    weather_skill = load_skill_from_directory(SKILLS_DIR / "weather")

    # Create a registry with the calendar skill (LLM can activate on demand)
    registry = SkillRegistry()
    calendar_skill = load_skill_from_directory(SKILLS_DIR / "calendar")
    registry.register(calendar_skill)

    agent = Agent(
        instructions=(
            "You are a helpful voice assistant. You start with weather capabilities. "
            "If the user asks about their calendar or scheduling, activate the calendar skill."
        ),
        skills=[weather_skill],
        skill_registry=registry,
    )

    session = AgentSession()
    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(prewarm=prewarm, entrypoint=entrypoint)
