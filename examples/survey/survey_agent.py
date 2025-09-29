import logging
from dotenv import load_dotenv
from dataclasses import dataclass
from livekit.agents import (
    Agent,
    AgentTask,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli
)
from livekit.agents.beta.workflows import Question, GetEmailTask, TaskOrchestrator
from livekit.plugins import cartesia, deepgram, openai, silero

@dataclass
class CollectedInformation:
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    birthday: str | None = None
    question_answer: dict | None = None 

class SurveyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
                You are a Survey agent screening candidates for a Software Engineer position.
            """
        )

    async def on_enter(self) -> AgentTask:
        await self.session.generate_reply(
            instructions="Greet the candidate for the Software Engineer interview."
        )
        # add questiontasks into task stack TaskOrchestrator
        email = await GetEmailTask()
        yoe_q = Question(instructions="Ask the user how many years of experience they have.", return_type="str")
        yoe = await yoe_q

logger = logging.getLogger("SurveyAgent")

load_dotenv(".env.local")

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):


    session = AgentSession[CollectedInformation](
        userdata=CollectedInformation(),
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(model="nova-3", language="multi"),
        tts=cartesia.TTS(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

   
    # add shutdown callback of conversation summary
    # ctx.add_shutdown_callback(...)

    await session.start(
        agent=SurveyAgent(),
        room=ctx.room,
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))