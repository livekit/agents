import csv
import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    AgentTask,
    JobContext,
    JobProcess,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.beta.workflows import GetEmailTask, Task, TaskOrchestrator
from livekit.agents.llm import function_tool
from livekit.agents.types import NOT_GIVEN
from livekit.plugins import cartesia, deepgram, openai, silero


@dataclass
class Userdata:
    task_results: dict


def write_to_csv(filename: str, data: dict):
    with open(filename, "w", newline="") as csvfile:
        fieldnames = data.keys()
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not os.path.exists(filename):
            csv_writer.writeheader(data)
        csv_writer.writerow(data)


class ExperienceTask(AgentTask[str]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            Record how many years of experience the candidate has and the descriptions of their previous jobs if any. There is no set required amount for this position.
            Focus on the frameworks they have experience in and any gaps between jobs. Be sure to confirm details.
            """,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Gather the candidate's work experience including how many years of experience they have and a general overview of their career.",
            tool_choice="none",
        )

    @function_tool()
    async def record_experience(self, context: RunContext, experience_description: str) -> None:
        """Call to record the years of experience the candidate has and its descriptions.

        Args:
            experience_description (str): The years of experience the candidate has and a description
        """
        self.complete(experience_description)


class CommuteTask(AgentTask[bool]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            Record if the candidate is able to commute to the office and their flexibility. Ideally, the candidate should commute to the office three days a week.
            Maintain a friendly disposition.
            """,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Gather the candidate's commute flexibility, specfically whether or not they are able to commute to the office.",
            tool_choice="none",
        )

    @function_tool()
    async def record_commute_flexibility(self, context: RunContext, can_commute: bool) -> None:
        """Call to record whether or not the candidate can commute to the office.

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
        # maybe make the greeting a task as well so we can collect the response

        tasks = [
            Task(
                lambda: ExperienceTask(),
                id="experience_task",
                description="Collects years of experience",
            ),
            Task(lambda: CommuteTask(), id="commute_task", description="Asks about commute"),
            Task(lambda: GetEmailTask(), id="get_email_task", description="Collects email"),
        ]

        results = await TaskOrchestrator(tasks)
        # TaskOrchestrator returns a dictionary with Task IDs as the keys and the results as the values
        self.session.userdata = results

        write_to_csv(filename="results.csv", data=results)


logger = logging.getLogger("SurveyAgent")

load_dotenv(".env.local")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    session = AgentSession[Userdata](
        userdata=Userdata(task_results={}),
        llm=openai.LLM(model="gpt-4o"),
        stt=deepgram.STT(model="nova-3", language="multi"),
        tts=cartesia.TTS(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    usage_collector = metrics.UsageCollector()

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    # add shutdown callback of conversation summary
    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=SurveyAgent(),
        room=ctx.room,
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
