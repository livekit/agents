import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.llm.chat_context import ChatContext, ChatMessage
from livekit.plugins import deepgram, groq, openai, silero

logger = logging.getLogger("pre-reseponse-agent")

load_dotenv()


class PreResponseAgent(Agent):
    def __init__(self):
        super().__init__(instructions="You are a helpful assistant")
        self._fast_llm = groq.LLM(model="llama-3.1-8b-instant")
        self._fast_llm_prompt = llm.ChatMessage(
            role="system",
            content=[
                "Generate a short instant response to the user's message with 5 to 10 words.",
                "Do not answer the questions directly. Examples:, let me think about that, "
                "wait a moment, that's a good question, etc.",
            ],
        )

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage):
        # Create a short "silence filler" response to quickly acknowledge the user's input
        fast_llm_ctx = turn_ctx.copy(
            exclude_instructions=True, exclude_function_call=True
        ).truncate(max_items=3)
        fast_llm_ctx.items.insert(0, self._fast_llm_prompt)
        fast_llm_ctx.items.append(new_message)

        # Intentionally not awaiting SpeechHandle to allow the main response generation to run concurrently
        self.session.say(
            self._fast_llm.chat(chat_ctx=fast_llm_ctx).to_str_iterable(),
            add_to_chat_ctx=False,
        )

        # Alternatively, if you want the reply to be aware of this "silence filler" response,
        # you can await the say method—but the tradeoff is that it may be slower since it won’t
        # execute concurrently with the main response generation:
        #
        # speech_handle = await self.session.say(
        #     self._fast_llm.chat(chat_ctx=fast_llm_ctx).to_str_iterable()
        # )
        # assert speech_handle.chat_message is not None, "say method always returns a chat message"
        # turn_ctx.items.append(speech_handle.chat_message)


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
        vad=silero.VAD.load(),
    )
    await session.start(PreResponseAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
