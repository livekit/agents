"""Avatar agent provisioned through LiveKit Inference (no provider key).

Unlike agent.py (which uses the BYOK lemonslice plugin with a LEMONSLICE_API_KEY),
this starts the avatar with inference.AvatarSession: no provider key is needed —
the agent authenticates with its LiveKit credentials (LIVEKIT_API_KEY /
LIVEKIT_API_SECRET, or LIVEKIT_INFERENCE_API_KEY / LIVEKIT_INFERENCE_API_SECRET
when set), and the Inference gateway creates the LemonSlice session using
LiveKit's wholesale key. Agent audio and the playback RPCs still flow in-room
over DataStream and the avatar publishes its video as a normal track, exactly as
the BYOK path does.

Requires the `avatar_lemonslice` feature flag to be enabled for your project on
the Inference gateway; without it the gateway returns HTTP 403 "Inference Avatar
is not enabled for this project".

Env: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LEMONSLICE_IMAGE_URL.

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

    avatar_image_url = os.getenv("LEMONSLICE_IMAGE_URL")
    if not avatar_image_url:
        raise ValueError("LEMONSLICE_IMAGE_URL must be set")

    # Provider-specific options go through extra_kwargs; LemonSliceOptions types
    # the keys the gateway accepts (image_url / prompt / idle_prompt /
    # idle_timeout).
    #
    # To use a pre-built LemonSlice agent instead of an image, pass its catalog id
    # in the model string and drop image_url — the two are mutually exclusive:
    #
    #     inference.AvatarSession(
    #         "lemonslice/<agent_id>",
    #         extra_kwargs=inference.LemonSliceOptions(prompt="..."),
    #     )
    avatar = inference.AvatarSession(
        "lemonslice",
        extra_kwargs=inference.LemonSliceOptions(
            image_url=avatar_image_url,
            prompt="Be expressive in your movements and use your hands while talking.",
        ),
    )
    await avatar.start(session, room=ctx.room)
    await avatar.wait_for_join()

    await session.start(agent=Agent(instructions="Talk to me!"), room=ctx.room)
    session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(server)
