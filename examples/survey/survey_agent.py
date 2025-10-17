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
    RoomInputOptions,
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


@function_tool()
async def disqualify(context: RunContext, disqualification_reason: str) -> None:
    """Call if the candidate refuses to explicitly answer or if they do not meet the prerequisites for the position.
    This function will terminate the interview and hang up.

    Args:
        disqualification_reason (str): The justification for ending the interview (ex. Refuses to answer question)
    """
    context.session.generate_reply(
        instructions=f"The interview is ending now, inform the candidate that the reason was {disqualification_reason}"
    )
    # in this scenario, disqualified candidates do not get recorded into the csv file, is that ideal?
    context.session.shutdown()


class WorkDistributionTask(AgentTask[str]):
    def __init__(self, team_size: int) -> None:
        super().__init__(
            instructions="""You will be asking the candidate about their project in relation to their team size."""
        )
        self._team_size = team_size

    async def on_enter(self) -> None:
        if self._team_size == 1:
            self.session.generate_reply(
                instructions="Have the candidate walk you through their thought process on splitting the project work between a team of four. If unspecified, probe further on why they made their decisions."
            )

        else:
            self.session.generate_reply(
                instructions="Inquire how the work was divided up between the team. If the candidate believes that there was a better way to distribute work, probe further on what they would change."
            )

    @function_tool()
    async def record_work_division_response(self, overall_response: str):
        """Call once the candidate has provided a complete overview to their perspective on work division.

        Args:
            overall_response (str): The candidate's response to approaching work division, especially regarding their aforementioned project
        """
        self.complete(overall_response)


class ExpandProjectTask(AgentTask[dict]):
    def __init__(self) -> None:
        super().__init__(
            instructions="You will be asking the candidate to expand upon their project intentions, whether it be to scale their current one or to create a new one."
        )
        self._result = {}

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Allow the candidate to choose a scenario between expanding upon the project they are currently speaking of or creating a new project entirely. Dissect their thought process and decisions."
        )

    @function_tool()
    async def record_old_project_expansion_response(
        self, new_features: str, scale_plan: str, overall_response: str
    ):
        """Call if the candidate decides to scale their previous project.

        Args:
            new_features (str): Record any new features the candidate expressed adding, such as improving GUI or adding an AI component.
            scale_plan (str): Record how the candidate expressed scaling their project, such as deploying it if not already
            overall_response (str): An overview of the candidate's response
        """
        self.session.generate_reply(
            instructions="Express interest in seeing the candidate scale their project as they described in the future."
        )
        self._result["new_features"] = new_features
        self._result["scale_plan"] = scale_plan
        self._result["overall_response"] = overall_response
        self.complete(self._result)

    @function_tool()
    async def record_new_project_creation_response(
        self, new_project_type: str, development_plan: str, overall_response: str
    ):
        """Call if the candidate decides to create a new project entirely.

        Args:
            new_project_type (str): The type of project, such as mobile app or AI program
            development_plan (str): Record how the candidate plans to develop this project from start to finish with detail
            overall_response (str): An overview of the candidate's response
        """
        self.session.generate_reply(
            instructions="Respond to the candidate's project idea and express support for pursuing it in the future."
        )
        self._result["new_project_type"] = new_project_type
        self._result["development_plan"] = development_plan
        self._result["overall_response"] = overall_response
        self.complete(self._result)


class ProjectTask(AgentTask[dict]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are an interviewer screening a candidate for a software engineering position. You have already been asking a series of questions, and this is another stage of the process.
                         Gather information about the most technical project the candidate has attempted and probe for their thinking process. Note specificities such as if they worked solo or in a team, and the technology stack they used
                         if applicable. If they have no projects to dissect, call disqualify(). Do not mention any prerequisites for this position.""",
            tools=[disqualify],
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Learn about the candidate's most technically difficult project, be inquisitive on their design choices and thinking process."
        )

    @function_tool()
    async def record_project_details(
        self, context: RunContext, team_size: int, project_description: str
    ) -> None:
        """Call to record team size, a categorization of the project, and any additional notes before another round of questions.

        Args:
            team_size (int): The size of the project team, minimum of 1
            project_description (str): A description of the project, including the type of project being described, such as "full stack application." Include the technology stack they used and their reasoning behind it.
        """
        tasks = [
            Task(
                lambda: WorkDistributionTask(team_size=team_size),
                id="work_distribution_task",
                description="Collects the candidate's perspective on work distribution regarding their project",
            ),
            Task(
                lambda: ExpandProjectTask(),
                id="expand_project_task",
                description="Collects the candidate's response on either scaling a previous project or creating a new one",
            ),
        ]
        task_group = TaskOrchestrator(tasks=tasks)
        results = await task_group
        self.complete(results)


class ExperienceTask(AgentTask[dict]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are an interviewer screening a candidate for a software engineering position. You have already been asking a series of questions, and this is another stage of the process.
            Record how many years of experience the candidate has and the descriptions of their previous jobs if any. There is no set required amount for this position.
            Focus on the frameworks they have experience in and any gaps between jobs. Be sure to confirm details. If the candidate wishes to change a previous answer, call out_of_scope.
            """,
            tools=[disqualify],
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
            experience_description (str): The years of experience the candidate has and a description of each role they previously held. Take note of the corresponding companies as well.
        """
        self.complete(experience_description)


class CommuteTask(AgentTask[dict]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are an interviewer screening a candidate for a software engineering position. You have already been asking a series of questions, and this is another stage of the process.
            Record if the candidate is able to commute to the office and their flexibility. Ideally, the candidate should commute to the office three days a week.
            """,
            tools=[disqualify],
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
        self._result["can_commute"] = can_commute
        if can_commute:
            commute_method = await CommuteMethodTask(commute_task=self)
            self._result["commute_method"] = commute_method

        self.complete(self._result)


class CommuteMethodTask(AgentTask[str]):
    def __init__(self, commute_task: CommuteTask) -> None:
        out_of_scope_tool = None
        for tool in commute_task.tools:
            if tool.__name__ == "out_of_scope":
                out_of_scope_tool = tool
        super().__init__(
            instructions="You will now be collecting the candidate's method of transportation.",
            chat_ctx=commute_task.chat_ctx,
            tools=[disqualify, out_of_scope_tool],
        )
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


class IntroTask(AgentTask[dict]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are an interviewer screening a candidate for a software engineering position. You both have just started the call.
            Welcome the candidate to the interview, remain positive and concise.
            You will also be collecting their name and introduction.
            """,
        )
        self._results = {}

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Welcome the candidate by introducing yourself and gather their name after their introduction.",
            tool_choice="none",
        )

    @function_tool()
    async def record_name(self, context: RunContext, name: str, intro_notes: str) -> None:
        """Call to record the candidate's name and any notes about their response

        Args:
            name (str): The candidate's name
            intro_notes (str): The candidate's introduction and any additional notes
        """
        self._results["name"] = name
        self._results["intro_notes"] = intro_notes
        self.complete(self._results)


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
                lambda: ProjectTask(),
                id="project_task",
                description="Probes the user about their thought process on projects",
            ),
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
        room_input_options=RoomInputOptions(
            delete_room_on_close=True,
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
