from __future__ import annotations

import logging

from dotenv import load_dotenv
from livekit.agents import (
    JobContext,
    JobRequest,
    WorkerOptions,
    cli,
    multimodal,
)
from livekit.plugins import openai
from livekit.rtc import RpcInvocationData

load_dotenv()

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    logger.info("starting entrypoint")

    await ctx.connect()
    participant = await ctx.wait_for_participant()
    agent = multimodal.MultimodalAgent(
        model=openai.realtime.RealtimeModel(
            voice="alloy",
            temperature=0.8,
            instructions="You are a helpful assistant",
            turn_detection=None,
        ),
    )
    agent.start(ctx.room, participant)

    @ctx.room.local_participant.register_rpc_method("ptt.start")
    async def handle_ptt_start(data: RpcInvocationData):
        logger.info("Received PTT start")
        agent.interrupt()
        return "ok"

    # NOTE: There's a potential risk of state getting out of sync if something
    # goes wrong in the remote client. In a production environment, consider
    # implementing PTT heartbeats where the frontend sends periodic signals
    # (e.g., every few seconds) while PTT is held. This allows the agent to
    # timeout and reset its state if heartbeats stop, preventing the agent from
    # hanging indefinitely.
    @ctx.room.local_participant.register_rpc_method("ptt.end")
    async def handle_ptt_end(data: RpcInvocationData):
        logger.info("Received PTT end")
        agent.generate_reply(on_duplicate="cancel_existing")
        return "ok"


async def handle_request(request: JobRequest) -> None:
    await request.accept(
        identity="ptt-agent",
        # This attribute communicates to frontend that we support PTT
        attributes={"supports-ptt": "1"},
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_fnc=handle_request,
        )
    )
