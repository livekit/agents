"""A simultaneous interpreter on DeepL Voice: speak one language, hear another.

Run locally with headphones, so the microphone does not pick up the translation:

    DEEPL_API_KEY=... python deepl_interpreter.py console

Languages default to English -> German; override with SOURCE_LANGUAGE / TARGET_LANGUAGE.
"""

import logging
import os

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    JobContext,
    UserInputTranscribedEvent,
    cli,
    llm,
)
from livekit.plugins import deepl

logger = logging.getLogger("deepl-interpreter")

load_dotenv()

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session: AgentSession[None] = AgentSession(
        llm=deepl.realtime.VoiceModel(
            source_language=os.getenv("SOURCE_LANGUAGE", "en"),
            target_language=os.getenv("TARGET_LANGUAGE", "de"),
            voice="match",
        ),
        # an interpreter never prompts a speaker who goes quiet
        user_away_timeout=None,
    )

    @session.on("user_input_transcribed")
    def _on_heard(ev: UserInputTranscribedEvent) -> None:
        if ev.is_final:
            logger.info(f"heard: {ev.transcript}")

    @session.on("conversation_item_added")
    def _on_said(ev: ConversationItemAddedEvent) -> None:
        if isinstance(ev.item, llm.ChatMessage) and ev.item.role == "assistant":
            logger.info(f"said:  {ev.item.text_content}")

    # DeepL takes no instructions: it translates what it hears
    await session.start(agent=Agent(instructions=""), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
