import os
import csv
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
    RunContext,
    cli,
)
from livekit.agents.beta.workflows import GetEmailTask, Question, Task, TaskOrchestrator
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, openai, silero


@dataclass
class CollectedInformation:
    name: str | None = None
    email: str | None = None
    question_answer: dict | None = None


def write_to_csv(filename: str, data):
    if type(data) != dict:
        data = data.asdict()
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = data.keys()
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not os.path.exists(filename):
            csv_writer.writeheader()
        csv_writer.writerow(data)

class ExperienceTask(AgentTask[str]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            Record how many years of experience the candidate has and the descriptions of their previous jobs if any.
            """,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Ask the candidate how many years of experience they have and for an overview of their career."
        )

    @function_tool()
    async def record_experience(self, context: RunContext, experience_description: str) -> None:
        """ Call to record the years of experience the candidate has and its descriptions.
        
        Args:
            experience_description (str): The years of experience the candidate has and a description
        """
        self.complete(experience_description)

class CommuteTask(AgentTask[bool]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            Record if the candidate is able to commute to the office.
            """,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Ask the candidate if they are able to commute to the office"
        )

    @function_tool()
    async def record_commute_flexibility(self, context: RunContext, can_commute: bool) -> None:
        """ Call to record whether or not the candidate can commute to the office.
        
        Args:
            can_commute (bool): If the candidate can commute or not
        """
        self.complete(can_commute)



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
        task_stack = [Task(lambda: ExperienceTask(), description="Collects years of experience"),
                     Task(lambda:  CommuteTask(), description="Asks about commute"),
                     Task(lambda: GetEmailTask(), description="Collects email")
                     ]

        task_orchestrator = TaskOrchestrator(llm=openai.LLM(model="gpt-4o"), task_stack=task_stack)
        results = await task_orchestrator
        write_to_csv(filename="results.csv", data=results)


logger = logging.getLogger("SurveyAgent")

load_dotenv(".env.local")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    session = AgentSession[CollectedInformation](
        userdata=CollectedInformation(),
        llm=openai.LLM(model="gpt-4o"),  
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
