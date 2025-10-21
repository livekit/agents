import csv
import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AgentTask,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RunContext,
    cli,
    llm,
    metrics,
)
from livekit.agents.beta.workflows import GetEmailTask, TaskGroup
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, openai, silero


@dataclass
class Userdata:
    filename: str
    task_results: dict


def write_to_csv(filename: str, data: dict):
    with open(filename, "a", newline="") as csvfile:
        fieldnames = data.keys()
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not os.path.exists(filename):
            csv_writer.writeheader()
        csv_writer.writerow(data)


async def evaluate_candidate(llm_model, summary) -> str:
    """Analyzes the full conversation to determine if the candidate is a good fit for the role and position"""
    conversation_text = summary.content
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(
        role="system",
        content=(
            "Evaluate whether or not this candidate is a good fit for the company and role based on the conversation summary provided.\n"
            "Take into account their holistic and professional profile and the quality of their responses.\n"
            "Be concise and firm in the evaluation."
        ),
    )
    chat_ctx.add_message(
        role="user",
        content=f"Conversation to evaluate:\n\n{conversation_text}",
    )

    chunks: list[str] = []
    async for chunk in llm_model.chat(chat_ctx=chat_ctx):
        if chunk.delta and chunk.delta.content:
            chunks.append(chunk.delta.content)
    evaluation = "".join(chunks).strip()
    return evaluation


@function_tool()
async def disqualify(context: RunContext, disqualification_reason: str) -> None:
    """Call if the candidate refuses to cooperate, provides an unsatisfactory or inappropriate answer, or do not meet the prerequisites for the position.
    This function will terminate the interview, record their disqualification, and hang up.

    Args:
        disqualification_reason (str): The justification for ending the interview (ex. Refuses to answer question)
    """
    context.session.generate_reply(
        instructions=f"The interview is ending now, inform the candidate that the reason was {disqualification_reason}"
    )
    disqualification_reason = "[DISQUALIFIED] " + disqualification_reason
    data = {
        "name": context.session.userdata.task_results["name"],
        "disqualification reason": disqualification_reason,
    }
    write_to_csv(context.session.userdata.filename, data)
    context.session.shutdown()


class BehaviorialTask(AgentTask[dict]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are an interviewer screening a candidate for a software engineering position. You have already been asking a series of questions, and this is another stage of the process.
            You will now be learning more about the candidate holistically. This includes their strengths, weaknesses, and work and communication style. You are testing the candidate for a good fit in the company.
            The ideal candidate would be well spoken, energetic, and thorough in their answers. Do not mention the prerequisites. If the candidate does not fulfill the description or refuses to answer, call disqualify().
            """,
            tools=[disqualify],
        )
        self._results = {}

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Approach this task in a natural conversational manner and incrementally gather the candidate's strengths, weaknesses, and work style in no particular order."
        )

    @function_tool()
    async def record_strengths(self, strengths_summary: str):
        """Call to record a summary of the candidate's strengths.

        Args:
            strengths_summary (str): A summary of the candidate's strengths
        """
        self._results["strengths"] = strengths_summary
        self._check_completion()

    @function_tool()
    async def record_weaknesses(self, weaknesses_summary):
        """Call to record a summary of the candidate's weaknesses.

        Args:
            weaknesses_summary (str): A summary of the candidate's weaknesses
        """
        self._results["weaknesses"] = weaknesses_summary
        self._check_completion()

    @function_tool()
    async def record_work_style(self, work_style: str):
        """Call to record a summary of the candidate's work and communication style.

        Args:
            work_style (str): The candidate's work and communication style
        """
        self._results["work_style"] = work_style
        self._check_completion()

    def _check_completion(self):
        if self._results.keys() == {"strengths", "weaknesses", "work_style"}:
            self.complete(self._results)


class ProjectTask(AgentTask[dict]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are an interviewer screening a candidate for a software engineering position. You have already been asking a series of questions, and this is another stage of the process.
                         Gather information about the most technical project the candidate has attempted and probe for their thinking process. Note specificities such as if they worked solo or in a team, and the technology stack they used
                         if applicable. If they have no projects to dissect, call disqualify(). Do not mention any prerequisites for this position.""",
            tools=[disqualify],
        )
        self._results = {}

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Learn about the candidate's most technically difficult project, be inquisitive on their design choices and thinking process."
        )

    @function_tool()
    async def record_project_details(self, context: RunContext, project_description: str) -> None:
        """Call to record and gradually update the description of their project as they answer more questions.

        Args:
            project_description (str): A description of the project, including the type of project being described, such as "full stack application." Include the technology stack they used and their reasoning behind it.
        """

        self._results["project_description"] = project_description
        if not self._results["work_division_response"]:
            self.session.generate_reply(
                instructions="Have the candidate walk you through their thought process on splitting the project work. If they already worked in a team for that project, gather their thoughts on what they would do differently."
            )

        elif (
            self._results["work_division_response"]
            and not self._results["scaling_project_response"]
        ):
            self.session.generate_reply(
                instructions="Allow the candidate to choose a scenario between expanding upon the project they are currently speaking of or creating a new project entirely. Dissect their thought process and decisions."
            )

        else:
            self.complete(self._results)

    @function_tool()
    async def record_work_division_response(self, work_division_response: str):
        """Call once the candidate has provided a complete overview to their perspective on work division.

        Args:
            work_division_response (str): The candidate's response to approaching work division, especially regarding their aforementioned project
        """
        self._results["work_division_response"] = work_division_response

    @function_tool()
    async def record_project_scale_response(self, chosen_scenario: str, scale_response: str):
        """Call to record the candidate's response to scaling their project, either the aforementioned or a new one

        Args:
            chosen_scenario (str): The scenario the candidate chose, either 'old_project' or 'new_project'
            old_project_scale_response (str): An overview of the candidate's response
        """
        self.session.generate_reply(
            instructions="Express interest in seeing the candidate scale their project as they described in the future."
        )
        results = {"scenario": chosen_scenario, "response": scale_response}
        self._result["scaling_project_response"] = results


class ExperienceTask(AgentTask[dict]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are an interviewer screening a candidate for a software engineering position. You have already been asking a series of questions, and this is another stage of the process.
            Record how many years of experience the candidate has and the descriptions of their previous jobs if any. There is no set required amount for this position.
            Focus on the frameworks they have experience in and any gaps between jobs. Be sure to confirm details.
            """,
            tools=[disqualify],
        )
        self._results = {}

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Gather the candidate's work experience including how many years of experience they have and a general overview of their career.",
        )

    @function_tool()
    async def record_experience(
        self, context: RunContext, years_of_experience: int, experience_description: str
    ) -> None:
        """Call to record the work experience the candidate has and a description of each role with a clear timeline.

        Args:
            years_of_experience (int): The years of experience the candidate has
            experience_description (str): A description of each role they previously held. Take note of the corresponding companies as well.
        """
        self._results["years_of_experience"] = years_of_experience
        self._results["experience_description"] = experience_description
        self.complete(self._results)


class CommuteTask(AgentTask[dict]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are an interviewer screening a candidate for a software engineering position. You have already been asking a series of questions, and this is another stage of the process.
            Record if the candidate is able to commute to the office and their flexibility. Ideally, the candidate should commute to the office three days a week. Disqualify the candidate if they cannot commute at all.
            """,
            tools=[disqualify],
        )
        self._result = {}

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Gather the candidate's commute flexibility, specfically whether or not they are able to commute to the office. If they are able to commute, collect their commute method. Be brief and to the point.",
        )

    @function_tool()
    async def record_commute_flexibility(
        self, context: RunContext, office_flexibility: str, commute_method: str
    ) -> None:
        """Call to record the candidate's flexibility of going into office and notes about their commute. If they are able to commute, record their method of transportation.

        Args:
            office_flexibility (str): How often the candidate can commute to the office
            commute_method (str): The method of transportation the candidate will take to commute (e.g. personal car, bus, subway)
        """
        self._result["office_flexibility"] = office_flexibility
        self._result["commute_method"] = commute_method
        if commute_method.lower() == "personal car":
            self.session.generate_reply(
                instructions="The candidate noted that they will drive to work. Inform them that there is no designated parking lot for the office, but there is metered street parking."
            )

        self.complete(self._result)


class IntroTask(AgentTask[dict]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are Alex, an interviewer screening a candidate for a software engineering position. You both have just started the call.
            Welcome the candidate to the interview, remain positive and concise.
            You will also be collecting their name and introduction.
            """,
        )
        self._results = {}

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Welcome the candidate by introducing yourself and gather their name after their introduction.",
        )

    @function_tool()
    async def record_name(self, context: RunContext, name: str, intro_notes: str) -> None:
        """Call to record the candidate's name and any notes about their response

        Args:
            name (str): The candidate's name
            intro_notes (str): The candidate's introduction and any additional notes
        """
        self.session.userdata.task_results["name"] = name
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
        task_group = TaskGroup()
        task_group.add(
            lambda: IntroTask(),
            id="get_name_intro_task",
            description="Collects name and introduction",
        )
        task_group.add(lambda: GetEmailTask(), id="get_email_task", description="Collects email")
        task_group.add(lambda: CommuteTask(), id="commute_task", description="Asks about commute")
        task_group.add(
            lambda: ExperienceTask(),
            id="experience_task",
            description="Collects years of experience",
        )
        task_group.add(
            lambda: BehaviorialTask(),
            id="behavorial_task",
            description="Gathers a holistic view of the candidate, including their strengths, weaknesses, and work style",
        )
        task_group.add(
            lambda: ProjectTask(),
            id="project_task",
            description="Probes the user about their thought process on projects",
        )

        results = await task_group
        results = results.task_results
        # TaskGroup returns a TaskGroupResult object. The task_results field holds a dictionary with Task IDs as the keys and the results as the values
        summary = self.chat_ctx.items[-1]
        evaluation = await evaluate_candidate(llm_model=self.session.llm, summary=summary)
        results["summary"] = summary.content
        results["evaluation"] = evaluation

        self.session.userdata.task_results = results
        write_to_csv(filename=self.session.userdata.filename, data=results)

    async def on_exit(self) -> None:
        await self.session.generate_reply(
            instructions="The interview is now complete, alert the user and thank them for their time. They will hear back within 3 days."
        )


logger = logging.getLogger("SurveyAgent")

load_dotenv(".env.local")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server = AgentServer()


@server.realtime_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession[Userdata](
        userdata=Userdata(filename="results.csv", task_results={}),
        llm=openai.LLM(model="gpt-4.1"),
        stt=deepgram.STT(model="nova-3", language="multi"),
        tts=openai.TTS(),
        vad=silero.VAD.load(),
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
    cli.run_app(server)
