from __future__ import annotations

import asyncio
import logging
from typing import List

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    utils,
)
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai
from livekit.plugins.openai.realtime.remote_items import _RemoteConversationItems

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


class SessionManager:
    def __init__(self, instructions: str):
        self.instructions = instructions
        self.chat_history: List[llm.ChatMessage] = []
        self.current_agent: MultimodalAgent | None = None
        self.current_model: openai.realtime.RealtimeModel | None = None
        # self.is_speaking = False

    @property
    def is_speaking(self) -> bool:
        if not self.current_agent:
            return False
        return (
            self.current_agent._playing_handle is not None
            and not self.current_agent._playing_handle.done()
        )

    def create_model(self) -> openai.realtime.RealtimeModel:
        model = openai.realtime.RealtimeModel(
            instructions=self.instructions,
            modalities=["audio", "text"],
        )
        self.current_model = model
        return model

    def create_agent(self, model: openai.realtime.RealtimeModel) -> MultimodalAgent:
        agent = MultimodalAgent(model=model)
        self.current_agent = agent
        return agent

    def setup_session(self, room: rtc.Room, participant: rtc.RemoteParticipant):
        model = self.create_model()
        agent = self.create_agent(model)
        agent.start(room, participant)

        # Initialize conversation with chat history
        session = model.sessions[0]
        session.conversation.item.create(
            llm.ChatMessage(
                role="assistant",
                content="Please begin the interaction with the user in a manner consistent with your instructions.",
            )
        )
        session.response.create()

    @utils.log_exceptions(logger=logger)
    async def restart_session(self):
        session = self.current_model.sessions[0]

        # clean up the old session
        await utils.aio.gracefully_cancel(session._main_atask)
        chat_history = session.chat_ctx_copy()
        # Patch: remove the empty conversation items
        # https://github.com/livekit/agents/pull/1245
        chat_history.messages = [
            msg
            for msg in chat_history.messages
            if msg.tool_call_id or msg.content is not None
        ]
        session._remote_conversation_items = _RemoteConversationItems()

        # create a new connection
        session._main_atask = asyncio.create_task(session._main_task())
        session.session_update()

        chat_history.append(
            text="We've just been reconnected, please continue the conversation.",
            role="assistant",
        )
        await session.set_chat_ctx(chat_history)

        await session.response.create()

    async def recycle_session(self, room: rtc.Room, participant: rtc.RemoteParticipant):
        if not self.current_agent or not self.current_model:
            return

        # Wait for any ongoing speech to complete
        while self.is_speaking:
            await asyncio.sleep(0.1)

        # Restart the session
        await self.restart_session()


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()

    session_manager = run_multimodal_agent(ctx, participant)

    logger.info("agent started")

    # Recycle session every minute
    while True:
        await asyncio.sleep(60)
        logger.info("recycling session")
        try:
            await asyncio.wait_for(
                session_manager.recycle_session(ctx.room, participant), timeout=10
            )
        except asyncio.TimeoutError:
            logger.warning("session recycling timed out, continuing")
        except Exception as e:
            logger.error(f"Error during session recycling: {e}")


def run_multimodal_agent(
    ctx: JobContext, participant: rtc.RemoteParticipant
) -> SessionManager:
    logger.info("starting multimodal agent")

    instructions = (
        "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
        "You should use short and concise responses, and avoiding usage of unpronouncable punctuation. "
        "You were created as a demo to showcase the capabilities of LiveKit's agents framework, "
        "as well as the ease of development of realtime AI prototypes. You are currently running in a "
        "LiveKit Sandbox, which is an environment that allows developers to instantly deploy prototypes "
        "of their realtime AI applications to share with others."
    )

    session_manager = SessionManager(instructions)
    session_manager.setup_session(ctx.room, participant)

    return session_manager


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
