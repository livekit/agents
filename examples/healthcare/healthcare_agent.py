import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated

from dotenv import load_dotenv
from pydantic import Field

# from .fake_database import FakeDatabase
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AgentTask,
    JobContext,
    cli,
)
from livekit.agents.beta.workflows import GetDOBTask, GetEmailTask, GetNameTask, TaskGroup
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("HealthcareAgent")

load_dotenv(".env.local")

ValidInsurances = ["Anthem", "Aetna", "EmblemHealth", "HealthFirst"]


@dataclass
class GetInsuranceResult:
    insurance: str


@dataclass
class ScheduleAppointmentResult:
    doctor_name: str
    appointment_time: datetime
    visit_reason: str


class GetInsuranceTask(AgentTask[GetInsuranceResult]):
    def __init__(self):
        super().__init__(
            instructions="""
            You will be gathering the user's health insurance. Be sure to confirm their answer. Avoid using dashes and special characters in your response.
        """
        )

    async def on_enter(self):
        await self.session.generate_reply(instructions="Collect the user's health insurance.")

    @function_tool()
    async def record_health_insurance(
        self, insurance: Annotated[str, Field(json_schema_extra={"enum": ValidInsurances})]
    ):
        """Record the user's health insurance.

        Args:
            insurance (str): The user's health insurance
        """
        self.complete(GetInsuranceResult(insurance=insurance))


class ScheduleAppointmentTask(AgentTask[ScheduleAppointmentResult]):
    def __init__(self):
        super().__init__(
            instructions="You will now assist the user with selecting a doctor and appointment time."
        )
        self._selected_doctor: str | None = None
        self._visit_reason: str | None = None

    async def on_enter(self): ...

    @function_tool()
    async def confirm_doctor_selection(self, doctor: str): ...

    @function_tool()
    async def schedule_appointment(self, appointment_time: str):
        if not (self._selected_doctor and self._visit_reason):
            await self.session.generate_reply(
                instructions="An appointment cannot be scheduled without selecting a doctor and specifying a visit reason"
            )
        else:
            self.complete(
                ScheduleAppointmentResult(
                    doctor_name=self._selected_doctor,
                    appointment_time=appointment_time,
                    visit_reason=self._visit_reason,
                )
            )

    @function_tool()
    async def confirm_visit_reason(self, visit_reason: str): ...


class HealthcareAgent(Agent):
    def __init__(self, database=None) -> None:
        super().__init__(
            instructions="You are a healthcare agent offering assistance to users. Maintain a friendly disposition. If the user refuses to provide any requested information or does not cooperate, call EndCallTool. if the user requests to schedule an appointment, call schedule_appointment",
            # tools=[EndCallTool(end_instructions="Disclose that the call is ending because the user refuses to cooperate or provide information and say goodbye.")]
        )
        self._database = database

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Greet the user and gather the reason for their call."
        )

    def information_tg_factory(self) -> TaskGroup:
        """Creates a TaskGroup that collects user information"""
        task_group = TaskGroup(chat_ctx=self.chat_ctx, return_exceptions=False)

        task_group.add(
            lambda: GetNameTask(), id="get_name_task", description="Gathers the user's name"
        )
        task_group.add(
            lambda: GetDOBTask(), id="get_dob_task", description="Gathers the user's date of birth"
        )
        task_group.add(
            lambda: GetEmailTask(), id="get_email_task", description="Gathers the user's email"
        )
        task_group.add(
            lambda: GetInsuranceTask(),
            id="get_insurance_task",
            description="Gathers the user's insurance",
        )

        return task_group

    @function_tool()
    async def schedule_appointment(self):
        """Call to schedule an appointment for the user."""
        await self.session.generate_reply(
            instructions="Inform the user that you will now be collecting their information."
        )

        task_group = self.information_tg_factory()

        task_group.add(
            lambda: ScheduleAppointmentTask(),
            id="schedule_appointment_task",
            description="Selects a doctor and schedules an appointment",
        )

        results = await task_group
        # TODO load results into database

        return "The appointment has been made, ask the user if they need assistance with anything else."

    # @function_tool()
    # async def medication_refill(self):
    #     """Facilitates medicine refill"""
    #     task_group = self.information_tg_factory()
    #     task_group.add(lambda: RefillPrescriptionTask(), id="refill_prescription_task", description="Refills user's prescription if available")

    # @function_tool()
    async def update_records(self, field: str):
        """Updates the user's information in the database"""
        results = await self.information_tg_factory()


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # db = FakeDatabase()
    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.responses.LLM(),
        tts=deepgram.TTS(),
        vad=silero.VAD.load(),
    )

    await session.start(
        agent=HealthcareAgent(),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
