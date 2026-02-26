"""
Minimal LiveKit agent with Keyframe avatar plugin.
"""

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import keyframe, openai
from livekit.plugins.keyframe import Emotion

load_dotenv()


class AvatarAgent(Agent):
    def __init__(self, avatar: keyframe.AvatarSession) -> None:
        super().__init__(
            instructions=(
                "You are a friendly voice assistant with an avatar. "
                "Use the set_emotion tool to change your facial expression "
                "whenever the conversation mood shifts."
            ),
        )
        self._avatar = avatar

    @function_tool()
    async def set_emotion(self, context: RunContext, emotion: Emotion) -> str:
        """Set the avatar's facial expression to match the conversation mood.

        Args:
            emotion: The emotion to express. One of 'neutral', 'happy', 'sad', or 'angry'.
        """
        await self._avatar.set_emotion(emotion)
        return f"Emotion set to {emotion}"


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="marin"),
    )

    avatar = keyframe.AvatarSession(persona_slug="public:lyra_persona-1.5-live")
    await avatar.start(session, room=ctx.room)

    await session.start(
        agent=AvatarAgent(avatar=avatar),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
