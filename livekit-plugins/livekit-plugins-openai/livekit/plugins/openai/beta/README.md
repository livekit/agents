# OpenAI Beta Features

## Assistants API

Example usage:

```python
import asyncio

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.openai.beta import (
    AssistantCreateOptions,
    AssistantLLM,
    AssistantOptions,
    OnFileUploadedInfo
)

load_dotenv()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext()

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # When using vision capabilities, files are uploaded.
    # It's up to you to remove them if desired or otherwise manage
    # them going forward.
    def on_file_uploaded(self, info: OnFileUploadedInfo):
        pass

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=AssistantLLM(
            assistant_opts=AssistantOptions(
                create_options=AssistantCreateOptions(
                    model="gpt-4o",
                    instructions="You are a voice assistant created by LiveKit. Your interface with users will be voice.",
                    name="KITT",
                )
            )
        ),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
        on_file_uploaded: on_file_uploaded,
    )
    assistant.start(ctx.room)

    # listen to incoming chat messages, only required if you'd like the agent to
    # answer incoming messages from Chat
    chat = rtc.ChatManager(ctx.room)

    async def answer_from_text(txt: str):
        chat_ctx = assistant.chat_ctx.copy()
        chat_ctx.append(role="user", text=txt)
        stream = assistant.llm.chat(chat_ctx=chat_ctx)
        await assistant.say(stream)

    @chat.on("message_received")
    def on_chat_received(msg: rtc.ChatMessage):
        if msg.message:
            asyncio.create_task(answer_from_text(msg.message))

    await asyncio.sleep(1)
    await assistant.say("Hey, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

## TODO
- tool calling