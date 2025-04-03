import logging
from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentSession,
    ChatContext,
    FunctionTool,
    JobContext,
    ModelSettings,
    WorkerOptions,
    cli,
)
from livekit.plugins import openai, silero, turn_detector
from typing import AsyncIterable, Callable, Optional, cast
from pydantic_core import from_json
from typing_extensions import TypedDict, Annotated

logger = logging.getLogger("structured-output")
load_dotenv()


class ResponseEmotion(TypedDict):
    voice_instructions: Annotated[
        str,
        Field(..., description="Concise TTS directive for tone, emotion, intonation, and speed"),
    ]
    response: str


async def process_structured_output(
    text: AsyncIterable[str],
    callback: Optional[Callable[[ResponseEmotion], None]] = None,
) -> AsyncIterable[str]:
    last_response = ""
    acc_text = ""
    async for chunk in text:
        acc_text += chunk
        try:
            resp: ResponseEmotion = from_json(acc_text, allow_partial="trailing-strings")
        except ValueError:
            continue

        if callback:
            callback(resp)

        if not resp.get("response"):
            continue

        new_delta = resp["response"][len(last_response) :]
        if new_delta:
            yield new_delta
        last_response = resp["response"]


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Echo. You are an extraordinarily expressive voice assistant "
                "with mastery over vocal dynamics and emotions. Adapt your voice—modulate tone, pitch, speed, intonation, "
                "and convey emotions such as happiness, sadness, excitement, or calmness—to match the conversation context. "
                "Keep responses concise, clear, and engaging, turning every interaction into a captivating auditory performance."
            ),
            stt=openai.STT(model="gpt-4o-transcribe"),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=openai.TTS(model="gpt-4o-mini-tts"),
        )

    async def llm_node(
        self, chat_ctx: ChatContext, tools: list[FunctionTool], model_settings: ModelSettings
    ):
        llm = cast(openai.LLM, self.llm)
        tool_choice = model_settings.tool_choice if model_settings else NOT_GIVEN
        async with llm.chat(
            chat_ctx=chat_ctx,
            tools=tools,
            tool_choice=tool_choice,
            response_format=ResponseEmotion,
        ) as stream:
            async for chunk in stream:
                yield chunk

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        instruction_updated = False

        def output_processed(resp: ResponseEmotion):
            nonlocal instruction_updated
            if resp.get("voice_instructions") and resp.get("response") and not instruction_updated:
                # when the response isn't empty, we can assume voice_instructions is complete.
                # (if the LLM sent the fields in the right order)
                instruction_updated = True
                logger.info(f"Updating TTS instructions: {resp['voice_instructions']}")

                tts = cast(openai.TTS, self.tts)
                tts.update_options(instructions=resp["voice_instructions"])

        return super().tts_node(
            process_structured_output(text, callback=output_processed), model_settings
        )

    async def transcription_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        async for delta in process_structured_output(text):
            yield delta


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session = AgentSession(
        vad=silero.VAD.load(),
        turn_detection=turn_detector.EOUModel(),
    )
    await session.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
