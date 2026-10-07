"""Microsoft AI STT/TTS with an existing OpenAI LLM and local Silero VAD.

Set MICROSOFT_AI_ENV_FILE to an explicit local dotenv file, or set the
MICROSOFT_AI_* variables documented in the plugin README, plus
OPENAI_API_KEY for the LLM. Confirm the provisional endpoint contracts first.
The CLI's console mode works without LiveKit Cloud.
"""

import os

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference
from livekit.plugins import microsoft_ai, openai

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    detector = inference.VAD(model="silero")
    speech_to_text = microsoft_ai.STT(vad=detector)
    text_to_speech = microsoft_ai.TTS()
    ctx.add_shutdown_callback(speech_to_text.aclose)
    ctx.add_shutdown_callback(text_to_speech.aclose)

    session: AgentSession = AgentSession(
        vad=detector,
        stt=speech_to_text,
        llm=openai.LLM(model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")),
        # The default agent TTS node supplies the sentence StreamAdapter.
        tts=text_to_speech,
    )
    await session.start(
        room=ctx.room,
        agent=Agent(instructions="You are a helpful voice assistant. Keep your replies concise."),
    )
    session.generate_reply(instructions="Greet the user briefly.")


if __name__ == "__main__":
    cli.run_app(server)
