import logging
import os
from dataclasses import dataclass
from typing import Annotated

import aiofiles
from aiocsv import AsyncWriter
from dotenv import load_dotenv
from pydantic import Field

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AgentTask,
    JobContext,
    RunContext,
    cli,
    inference,
    llm,
    metrics,
    room_io,
)
from livekit.agents.beta.workflows import GetEmailTask, TaskGroup
from livekit.agents.llm import function_tool
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("SurveyAgent")

load_dotenv()

CommuteMethods = ["driving", "bus", "subway", "none"]
WorkStyles = ["independent", "team_player"]


@dataclass
class Userdata:
    filename: str
    candidate_name: str
    task_results: dict


@dataclass
class IntroResults:
    name: str
    intro: str


@dataclass
class CommuteResults:
    can_commute: bool
    commute_method: str


@dataclass
class ExperienceResults:
    years_of_experience: int
    experience_description: str


@dataclass
class BehavioralResults:
    strengths: str
    weaknesses: str
    work_style: str


async def write_to_csv(filename: str, data: dict):
    async with aiofiles.open(filename, "a", newline="") as csvfile:
        writer = AsyncWriter(csvfile, data.keys())
        if not os.path.exists(filename):
            await writer.writeheader()
        await writer.writerow(data.values())


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
        instructions=f"The interview is ending now, inform the candidate that the reason was {disqualification_reason}. Be respectful and natural."
    )
    disqualification_reason = "[DISQUALIFIED] " + disqualification_reason
    data = {
        "name": context.session.userdata.candidate_name,
        "disqualification reason": disqualification_reason,
    }
    await write_to_csv(context.session.userdata.filename, data)
    context.session.shutdown()


class BehavioralTask(AgentTask[BehavioralResults]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are an interviewer screening a candidate for a software engineering position. You have already been asking a series of questions, and this is another stage of the process.
            You will now be learning more about the candidate holistically. This includes their strengths, weaknesses, and work and communication style. You are testing the candidate for a good fit in the company.
            If the candidate refuses to answer, call disqualify(). Be concise and to the point.
            Avoid listing out questions with bullet points or numbers, use a natural conversational tone.
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
    async def record_weaknesses(self, weaknesses_summary: str):
        """Call to record a summary of the candidate's weaknesses.

        Args:
            weaknesses_summary (str): A summary of the candidate's weaknesses
        """
        self._results["weaknesses"] = weaknesses_summary
        self._check_completion()

    @function_tool()
    async def record_work_style(
        self, work_style: Annotated[str, Field(json_schema_extra={"enum": WorkStyles})]
    ):
        """Call to record a summary of the candidate's work style.

        Args:
            work_style (str): The candidate's work style
        """
        self._results["work_style"] = work_style
        self._check_completion()

    def _check_completion(self):
        if self._results.keys() == {"strengths", "weaknesses", "work_style"}:
            results = BehavioralResults(
                strengths=self._results["strengths"],
                weaknesses=self._results["weaknesses"],
                work_style=self._results["work_style"],
            )
            self.complete(results)
        else:
            self.session.generate_reply(
                instructions="Continue incrementally collecting the remaining answers for the behavioral stage. Maintain a conversational tone."
            )


class ExperienceTask(AgentTask[ExperienceResults]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are an interviewer screening a candidate for a software engineering position. You have already been asking a series of questions, and this is another stage of the process.
            Record how many years of experience the candidate has and the descriptions of their previous jobs if any. There is no set required amount for this position.
            Be sure to confirm details. Avoid listing out questions with bullet points or numbers, use a natural conversational tone.
            """,
            tools=[disqualify],
        )

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
        results = ExperienceResults(
            years_of_experience=years_of_experience, experience_description=experience_description
        )
        self.complete(results)


class CommuteTask(AgentTask[CommuteResults]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are an interviewer screening a candidate for a software engineering position. You have already been asking a series of questions, and this is another stage of the process.
            Record if the candidate is able to commute to the office and their flexibility. Ideally, the candidate should commute to the office three days a week. Avoiding using parentheses in your response.
            """,
            tools=[disqualify],
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Gather the candidate's commute flexibility, specfically whether or not they are able to commute to the office. If they are able to commute, collect their commute method. Be brief and to the point.",
        )

    @function_tool()
    async def record_commute_flexibility(
        self,
        context: RunContext,
        can_commute: bool,
        commute_method: Annotated[str, Field(json_schema_extra={"enum": CommuteMethods})],
    ) -> None:
        """Call to record the candidate's flexibility of going into office and notes about their commute. If they are able to commute, record their method of transportation.

        Args:
            can_commute (bool): If the candidate can commute to the office
            commute_method (str): The method of transportation the candidate will take to commute
        """
        results = CommuteResults(can_commute=can_commute, commute_method=commute_method)
        self.complete(results)


class IntroTask(AgentTask[IntroResults]):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are Alex, an interviewer screening a candidate for a software engineering position. You both have just started the call.
            Welcome the candidate to the interview, remain positive and concise.
            You will also be collecting their name and introduction.
            """,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Welcome the candidate by introducing yourself and gather their name after their introduction.",
        )

    @function_tool()
    async def record_intro(self, context: RunContext, name: str, intro_notes: str) -> None:
        """Call to record the candidate's name and any notes about their response

        Args:
            name (str): The candidate's name
            intro_notes (str): The candidate's introduction and any additional notes
        """
        self.session.userdata.candidate_name = name
        results = IntroResults(name=name, intro=intro_notes)
        self.complete(results)


class SurveyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
                You are a Survey agent screening candidates for a Software Engineer position. When the interview is concluded, call end_screening to hang up.
            """
        )

    async def on_enter(self) -> AgentTask:
        task_group = TaskGroup()
        task_group.add(
            lambda: IntroTask(),
            id="get_name_intro_task",
            description="Collects name and introduction",
        )
        task_group.add(
            lambda: GetEmailTask(
                extra_instructions="If the user refuses to provide their email, call disqualify() insted of decline_email_capture().",
                tools=[disqualify],
            ),
            id="get_email_task",
            description="Collects email",
        )
        task_group.add(
            lambda: CommuteTask(),
            id="commute_task",
            description="Asks about commute and corresponding method of transportation.",
        )
        task_group.add(
            lambda: ExperienceTask(),
            id="experience_task",
            description="Collects years of experience and a description of their professionl work history.",
        )
        task_group.add(
            lambda: BehavioralTask(),
            id="behavorial_task",
            description="Gathers a holistic view of the candidate, including their strengths, weaknesses, and work style",
        )

        results = await task_group
        results = results.task_results
        # TaskGroup returns a TaskGroupResult object. The task_results field holds a dictionary with Task IDs as the keys and the results as the values
        summary = self.chat_ctx.items[-1]
        evaluation = await evaluate_candidate(llm_model=self.session.llm, summary=summary)
        results["summary"] = summary.content
        results["evaluation"] = evaluation
        self.session.userdata.task_results = results
        await write_to_csv(filename=self.session.userdata.filename, data=results)
        await self.session.generate_reply(
            instructions="The interview is now complete, alert the user and thank them for their time. They will hear back within 3 days."
        )

    @function_tool()
    async def end_screening(self):
        """Call when the interview/screening is concluded to hang up."""
        self.session.shutdown()


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession[Userdata](
        userdata=Userdata(filename="results.csv", candidate_name="", task_results={}),
        llm=inference.LLM("google/gemini-2.5-flash"),
        stt=inference.STT("deepgram/nova-3", language="multi"),
        tts=inference.TTS("inworld/inworld-tts-1"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
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
        room_options=room_io.RoomOptions(delete_room_on_close=True),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
