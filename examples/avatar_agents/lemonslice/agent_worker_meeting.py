import json
import logging
import os
from typing import Any

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    cli,
    inference,
)
from livekit.plugins import lemonslice

logger = logging.getLogger("lemonslice-avatar-meeting-example")
logger.setLevel(logging.INFO)

load_dotenv()


server = AgentServer()


def _optional_bool(value: Any, *, default: bool) -> bool:
    """Parse a bool from job metadata (JSON) or an environment variable (string)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true"):
            return True
        if normalized in ("0", "false"):
            return False
        raise ValueError(f"invalid boolean value: {value!r}")
    raise ValueError(f"invalid boolean value: {value!r}")


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("google/gemini-2.5-flash"),
        tts=inference.TTS("cartesia/sonic-3"),
        turn_handling=TurnHandlingOptions(
            interruption={
                "resume_false_interruption": False,
            },
        ),
    )

    lemonslice_image_url = os.getenv("LEMONSLICE_IMAGE_URL")
    if lemonslice_image_url is None:
        raise ValueError("LEMONSLICE_IMAGE_URL must be set")
    avatar = lemonslice.AvatarSession(
        agent_image_url=lemonslice_image_url,
        agent_prompt="Be expressive in your movements and use your hands while talking.",
    )
    await avatar.start(session, room=ctx.room)

    meta = json.loads(ctx.job.metadata) if ctx.job.metadata else {}
    meeting_url = meta.get("meeting_url") or os.getenv("MEETING_URL")
    if not meeting_url:
        raise ValueError("Set meeting_url in job metadata or MEETING_URL env var")

    join_kwargs: dict[str, str | bool] = {
        "listen_to_meeting_chat": _optional_bool(
            meta.get("listen_to_meeting_chat", os.getenv("LISTEN_TO_MEETING_CHAT")),
            default=True,
        ),
    }
    bot_name = meta.get("bot_name") or os.getenv("MEETING_BOT_NAME")
    if bot_name:
        join_kwargs["bot_name"] = bot_name

    await avatar.join_meeting(meeting_url, **join_kwargs)

    agent = Agent(instructions="Talk to me!")

    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=avatar.room_options(),
    )

    session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(server)
