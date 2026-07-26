"""Avatar agent provisioned through LiveKit Inference (no provider key).

Unlike agent.py (which uses the BYOK lemonslice plugin with a LEMONSLICE_API_KEY),
this starts the avatar with inference.AvatarSession: the agent authenticates only
with LIVEKIT_API_KEY / LIVEKIT_API_SECRET, and the Inference gateway creates the
LemonSlice session using LiveKit's wholesale key. Media and lip-sync still flow
in-room over DataStream, exactly as the BYOK path does.

Requires the `avatar_lemonslice` feature flag to be enabled for your project on
the Inference gateway.

Run:
    python inference_agent.py dev
"""

import logging
import os

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    inference,
)

logger = logging.getLogger("inference-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("google/gemini-2.5-flash"),
        tts=inference.TTS("cartesia/sonic-3"),
    )

    # Avatar provisioning goes through LiveKit Inference: only LIVEKIT_API_KEY /
    # LIVEKIT_API_SECRET are needed (no provider key). The gateway creates the
    # LemonSlice session with LiveKit's wholesale key; media stays in-room.
    #
    # Pass a catalog agent id instead of an image with
    # inference.AvatarSession("lemonslice/<agent_id>", ...).
    avatar_image_url = os.getenv("LEMONSLICE_IMAGE_URL")
    if not avatar_image_url:
        raise ValueError("LEMONSLICE_IMAGE_URL must be set")
    avatar = inference.AvatarSession(
        "lemonslice",
        image_url=avatar_image_url,
        prompt="Be expressive in your movements and use your hands while talking.",
    )
    await avatar.start(session, room=ctx.room)
    await avatar.wait_for_join()

    await session.start(agent=Agent(instructions="Talk to me!"), room=ctx.room)
    session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(server)
