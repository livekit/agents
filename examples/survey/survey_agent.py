import csv
import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
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
from livekit.plugins import cartesia, deepgram, openai, silero


@dataclass
class Userdata:
    task_results: dict


def write_to_csv(filename: str, data: dict):
    with open(filename, "a", newline="") as csvfile:
        fieldnames = data.keys()
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not os.path.exists(filename):
            csv_writer.writeheader()
        csv_writer.writerow(data)


class ExperienceTask(AgentTask[str]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are an interviewer screening a candidate for a software engineering position. You have already been asking a series of questions, and this is another stage of the process.
            Record how many years of experience the candidate has and the descriptions of their previous jobs if any. There is no set required amount for this position.
            Focus on the frameworks they have experience in and any gaps between jobs. Be sure to confirm details. If the candidate wishes to change a previous answer, call out_of_scope.
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


class CommuteTask(AgentTask[dict]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are an interviewer screening a candidate for a software engineering position. You have already been asking a series of questions, and this is another stage of the process.
            Record if the candidate is able to commute to the office and their flexibility. Ideally, the candidate should commute to the office three days a week.
            """,
        )
        self._result = {}

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Gather the candidate's commute flexibility, specfically whether or not they are able to commute to the office. Be brief and to the point.",
            tool_choice="none",
        )

    @function_tool()
    async def record_commute_flexibility(self, context: RunContext, can_commute: bool) -> None:
        """Call to record whether or not the candidate can commute to the office.

        Args:
            can_commute (bool): If the candidate can commute or not
        """
        if can_commute:
            self._result["can_commute"] = can_commute
            commute_method = await CommuteMethodTask(commute_task=self)
            self._result["commute_method"] = commute_method
        self.complete(self._result)


class CommuteMethodTask(AgentTask[str]):
    def __init__(self, commute_task: CommuteTask) -> None:
        super().__init__(
            instructions="You will now be collecting the candidate's method of transportation.",
            chat_ctx=commute_task.chat_ctx,
            tools=[commute_task.tools[1]],
        )  # add out of scope method
        self._commute_task = commute_task

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Gather their transportation method.",
            tool_choice="none",
        )

    @function_tool()
    async def record_commute_method(self, context: RunContext, commute_method: str) -> None:
        """Call to record the candidate's method of transportation for their commute.

        Args:
            commute_method (str): The candidate's method of transportation for their commute.
        """
        self.complete(commute_method)


class IntroTask(AgentTask[str]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are an interviewer screening a candidate for a software engineering position. You both have just started the call.
            Welcome the candidate to the interview and remain positive and concise. Take note of their attitude and how they respond.
            You will also be collecting their name.
            """,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Welcome the candidate by introducing yourself and gather their name after their introduction.",
            tool_choice="none",
        )

    @function_tool()
    async def record_name(self, context: RunContext, name: str, notes: str) -> None:
        """Call to record the candidate's name and any notes about their attitude

        Args:
            name (str): The candidate's name
            notes (str): Any additional notes about the candidate's responses (if none, return "none")
        """
        self.complete(name)


class SurveyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
                You are a Survey agent screening candidates for a Software Engineer position.
            """
        )

    async def on_enter(self) -> AgentTask:
        tasks = [
            Task(
                lambda: ExperienceTask(),
                id="experience_task",
                description="Collects years of experience",
            ),
            Task(lambda: CommuteTask(), id="commute_task", description="Asks about commute"),
            Task(lambda: GetEmailTask(), id="get_email_task", description="Collects email"),
            Task(lambda: IntroTask(), id="get_name_intro_task", description="Collects name"),
        ]
        results = await TaskOrchestrator(tasks)
        # TaskOrchestrator returns a dictionary with Task IDs as the keys and the results as the values
        self.session.userdata = results

        write_to_csv(filename="results.csv", data=results)

    async def on_exit(self) -> None:
        await self.session.generate_reply(
            instructions="The interview is now complete, alert the user and thank them for their time. They will hear back within 3 days."
        )


logger = logging.getLogger("SurveyAgent")

load_dotenv(".env.local")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    session = AgentSession[Userdata](
        userdata=Userdata(task_results={}),
        llm=openai.LLM(model="gpt-4.1"),
        stt=deepgram.STT(model="nova-3", language="multi"),
        tts=cartesia.TTS(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=SurveyAgent(),
        room=ctx.room,
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
