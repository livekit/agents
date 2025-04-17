import logging
from collections.abc import AsyncIterable

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    ModelSettings,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.plugins import cartesia, deepgram, groq, silero

logger = logging.getLogger("two-llm-example")
logger.setLevel(logging.INFO)

load_dotenv()

## This example shows how to use a fast LLM and a main LLM to generate a response.
## The fast LLM is used to generate a short instant response to the user's message.
## The main LLM is used to generate a more detailed response to the user's message.


class TwoLLMAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant.",
            llm=groq.LLM(model="llama-3.3-70b-versatile"),
        )
        self.fast_llm: llm.LLM = groq.LLM(model="llama-3.1-8b-instant")
        self.fast_llm_prompt = llm.ChatMessage(
            role="system",
            content=[
                "Generate a short instant response to the user's message with 5 to 10 words.",
                "Do not answer the questions directly. For example, let me think about that, "
                "wait a moment, that's a good question, etc.",
            ],
        )

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk | llm.FlushSentinel]:
        # truncate the chat ctx with a fast response prompt
        fast_chat_ctx = chat_ctx.copy(
            exclude_function_call=True, exclude_instructions=True
        ).truncate(max_items=3)
        fast_chat_ctx.items.insert(0, self.fast_llm_prompt)

        quick_response = ""
        async with self.fast_llm.chat(chat_ctx=fast_chat_ctx) as stream:
            async for chunk in stream:
                yield chunk
                if chunk.delta and chunk.delta.content:
                    quick_response += chunk.delta.content

        # yield flush the quick response to tts
        yield llm.FlushSentinel()
        logger.info(f"quick response: {quick_response}")

        # (Optional) add the quick response to the chat ctx for the main llm
        assert isinstance(self.llm, llm.LLM)
        chat_ctx.add_message(role="assistant", content=quick_response)

        # generate the response with the main llm
        async for chunk in Agent.default.llm_node(
            agent=self,
            chat_ctx=chat_ctx,
            tools=tools,
            model_settings=model_settings,
        ):
            yield chunk


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        tts=cartesia.TTS(),
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)

    await session.start(agent=TwoLLMAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
