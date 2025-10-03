import logging
from dataclasses import dataclass

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    AgentTask,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
)
from livekit.agents.beta.workflows import GetEmailTask, Question, Task, TaskOrchestrator
from livekit.plugins import cartesia, deepgram, openai, silero


@dataclass
class CollectedInformation:
    # TODO: perhaps can be organized better
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
            instructions="Welcome the candidate for the Software Engineer interview."
        )
        task_bank = [
            Task(
                lambda: Question(
                    instructions="Ask the candidate how many years of experience they have."
                ),
                description="Collects years of experience",
            ),
            Task(
                lambda: Question(
                    instructions="Ask the candidate if they are able to commute to the office."
                ),
                description="Asks about commute flexibility",
            ),
            Task(lambda: GetEmailTask(), description="Collects email"),
        ]
        # TODO will add more complex open ended questions

        task_orchestrator = TaskOrchestrator(
            llm=openai.LLM(model="gpt-4o-mini"), task_stack=task_bank
        )
        results = await task_orchestrator

        # TODO: transfer the results to the userdata


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
