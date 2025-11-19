import logging
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    Agent,
    AgentSession,
)
from livekit.plugins import openai, deepgram, silero

load_dotenv()
logger = logging.getLogger("voice-agent")

# CONFIGURATION: FILLER WORDS
IGNORED_WORDS = {"uh", "umm", "hmm", "haan", "ah", "like", "ok", "mhmm", "hnn", "mm", "okay", "sure", "alright"}

class InterruptHandlingAgent(Agent):
    def __init__(self):
        super().__init__(instructions="You are a helpful assistant. Answer concisely.")
        
    async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage):
        # FIX: Extract text from the list of content
        user_text = ""
        if isinstance(new_message.content, list):
            # Join all text parts in the list
            user_text = " ".join([str(item) for item in new_message.content if isinstance(item, str)])
        elif isinstance(new_message.content, str):
            user_text = new_message.content

        # Clean the text
        cleaned_text = user_text.strip().lower().replace(".", "").replace(",", "")
        logger.info(f"User said: '{cleaned_text}'")

        # 1. CHECK FOR FILLER WORDS
        if cleaned_text in IGNORED_WORDS:
            logger.info(f"Detected filler word '{cleaned_text}'. Ignoring.")
            return

        # 2. GENERATE REPLY
        # Note: This line will still fail with 'Quota Exceeded' if you have no OpenAI credits
        try:
            await self.session.generate_reply(chat_ctx=turn_ctx)
        except Exception as e:
            logger.error(f"Failed to generate reply (likely billing issue): {e}")

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
    )

    agent = InterruptHandlingAgent()
    await session.start(room=ctx.room, agent=agent)
    
    try:
        await session.generate_reply(instructions="Say hello and mention that you can ignore filler words.")
    except Exception as e:
        logger.error(f"Failed to send greeting: {e}")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))