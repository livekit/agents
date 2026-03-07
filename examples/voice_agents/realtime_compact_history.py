import logging

from dotenv import load_dotenv

from livekit.agents import AgentServer, JobContext, cli, llm
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai

logger = logging.getLogger("realtime-compact-history")
logger.setLevel(logging.INFO)

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant.",
            llm=openai.realtime.RealtimeModel(),
        )
        self.summarizer_llm = openai.LLM(model="gpt-4.1")

    @llm.function_tool
    async def compact_history(self) -> str:
        """Called when the user asks to compact the chat history.
        This will compact the chat history into a short, faithful summary.
        """

        logger.info("compacting chat history")

        # summarize the chat history using the LLM
        source_text = "\n".join(
            f"{item.role}: {(item.text_content or '').strip()}"
            for item in self.chat_ctx.items
            if item.type == "message" and item.role in {"user", "assistant"}
        ).strip()

        if not self.chat_ctx.items or not source_text:
            return "history is empty"

        last_updated_at = self.chat_ctx.items[-1].created_at
        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(
            role="system",
            content=(
                "Compress older chat history into a short, faithful summary.\n"
                "Focus on user goals, constraints, decisions, key facts/preferences/entities, and pending tasks.\n"
                "Exclude chit-chat and greetings. Be concise."
            ),
        )
        chat_ctx.add_message(
            role="user",
            content=f"Conversation to summarize:\n\n{source_text}",
        )
        async with self.summarizer_llm.chat(chat_ctx=chat_ctx) as stream:
            response = await stream.collect()
            summary = response.text
            logger.info(f"compacted history summary: {summary}")

        # sync the compacted history back to the agent's chat context
        compacted_history = llm.ChatContext()
        compacted_history.add_message(
            role="assistant",
            content=f"[history summary]\n{summary}",
            created_at=last_updated_at,
        )

        # keep new items and incomplete function calls for realtime session
        tail_items: list[llm.ChatItem] = []
        for item in self.realtime_llm_session.chat_ctx.items:
            if item.created_at > last_updated_at or (
                item.type == "function_call" and item.status != "completed"
            ):
                tail_items.append(item)
        if tail_items:
            compacted_history.items.extend(tail_items)
            logger.info(f"keeping {len(tail_items)} items from realtime session")

        await self.update_chat_ctx(compacted_history)

        return "history compacted"


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession()
    await session.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
