import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Optional

from dotenv import load_dotenv
from fake_database import FakeDatabase
from pydantic import Field

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AgentTask,
    FunctionTool,
    JobContext,
    RunContext,
    cli,
)
from livekit.agents.beta.workflows import GetDOBTask, GetEmailTask, GetNameTask, TaskGroup
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("HealthcareAgent")

load_dotenv()

ValidInsurances = ["Anthem", "Aetna", "EmblemHealth", "HealthFirst"]


@dataclass
class UserData:
    database: FakeDatabase
    insurance: str | None = None


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
        await self.session.generate_reply(
            instructions="Collect the user's health insurance and inform them of the accepted insurance options."
        )

    @function_tool()
    async def record_health_insurance(
        self,
        context: RunContext,
        insurance: Annotated[str, Field(json_schema_extra={"enum": ValidInsurances})],
    ):
        """Record the user's health insurance.

        Args:
            insurance (str): The user's health insurance
        """
        context.session.userdata.insurance = insurance
        self.complete(GetInsuranceResult(insurance=insurance))


class ScheduleAppointmentTask(AgentTask[ScheduleAppointmentResult]):
    def __init__(self):
        super().__init__(
            instructions="You will now assist the user with selecting a doctor and appointment time."
        )
        self._selected_doctor: str | None = None
        self._appointment_time: datetime | None = None

    async def on_enter(self):
        database = self.session.userdata.database
        insurance = self.session.userdata.insurance

        self._compatible_doctor_records = database.get_compatible_doctors(insurance=insurance)

        available_doctors = [doctor["name"] for doctor in self._compatible_doctor_records]
        doctor_confirmation_tool = self._build_doctor_selection_tool(
            available_doctors=available_doctors
        )
        current_tools = self.tools
        current_tools.append(doctor_confirmation_tool)
        await self.update_tools(current_tools)

        if len(self._compatible_doctor_records) > 1:
            await self.session.generate_reply(
                instructions=f"These are the doctors compatible with the user's insurance: {available_doctors}, prompt the user to choose one."
            )
        else:
            await self.session.generate_reply(
                instructions=f"Inform the user that {available_doctors} accepts their insurance and confirm if they would like to select that doctor."
            )

    def _build_doctor_selection_tool(
        self, *, available_doctors: list[str]
    ) -> Optional[FunctionTool]:
        @function_tool()
        async def confirm_doctor_selection(
            selected_doctor: Annotated[
                str,
                Field(
                    description="The names of the available doctors",
                    json_schema_extra={"items": {"enum": available_doctors}},
                ),
            ],
        ) -> None:
            """Call to confirm the user's doctor selection.

            Args:
                selected_doctor (str): The doctor the user selects
            """
            self._selected_doctor = selected_doctor
            doctor_record = next(
                (d for d in self._compatible_doctor_records if d["name"] == selected_doctor), None
            )

            available_times = doctor_record["availability"]
            schedule_appointment_tool = self._build_schedule_appointment_tool(
                available_times=available_times
            )
            current_tools = self.tools
            current_tools.append(schedule_appointment_tool)
            await self.update_tools(current_tools)

            await self.session.generate_reply(
                instructions=f"The selected doctor has availabilities at {available_times}. Ask the user which time slot they prefer."
            )

        return confirm_doctor_selection

    def _build_schedule_appointment_tool(
        self, *, available_times: list[str]
    ) -> Optional[FunctionTool]:
        @function_tool()
        async def schedule_appointment(
            appointment_time: Annotated[
                str,
                Field(
                    description="The available appointment times",
                    json_schema_extra={"items": {"enum": available_times}},
                ),
            ],
        ):
            """Call to confirm the user's selected appointment time.

            Args:
                appointment_time (str): The user's appointment time selection
            """
            self._appointment_time = appointment_time

            visit_reason_tool = self._build_visit_reason_tool()
            current_tools = self.tools
            current_tools.append(visit_reason_tool)
            await self.update_tools(current_tools)

            await self.session.generate_reply(
                instructions="Prompt the user for the reason for their visit."
            )

        return schedule_appointment

    def _build_visit_reason_tool(self) -> Optional[FunctionTool]:
        @function_tool()
        async def confirm_visit_reason(visit_reason: str):
            """Call to record the user's reason for their appointment.

            Args:
                visit_reason (str): The user's reason for visiting a doctor
            """
            self.complete(
                ScheduleAppointmentResult(
                    doctor_name=self._selected_doctor,
                    appointment_time=self._appointment_time,
                    visit_reason=visit_reason,
                )
            )

        return confirm_visit_reason


class HealthcareAgent(Agent):
    def __init__(self, database=None) -> None:
        super().__init__(
            instructions="You are a healthcare agent offering assistance to users. Maintain a friendly disposition. If the user refuses to provide any requested information or does not cooperate, call EndCallTool. if the user requests to schedule an appointment, call schedule_appointment()",
            # tools=[EndCallTool(end_instructions="Disclose that the call is ending because the user refuses to cooperate or provide information and say goodbye.", delete_room=True)]
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
        task_group = self.information_tg_factory()

        # Observe how if any information is given early, TaskGroup will fast-forward the respective task. and at any time, user information can be updated
        task_group.add(
            lambda: ScheduleAppointmentTask(),
            id="schedule_appointment_task",
            description="Selects a doctor and schedules an appointment",
        )
        results = (await task_group).task_results

        appointment_results = results["schedule_appointment_task"]
        appointment = {
            "doctor_name": appointment_results.doctor_name,
            "appointment_time": appointment_results.appointment_time,
            "visit_reason": appointment_results.visit_reason,
        }

        db = self.session.userdata.database
        db.add_patient_record(
            info={
                "name": results["get_name_task"],
                "date_of_birth": results["get_dob_task"],
                "email": results["get_email_task"],
                "insurance": results["get_insurance_task"],
                "appointments": {appointment},
            }
        )
        self._database.remove_doctor_availability(
            appointment_results.doctor_name,
            {
                "date": appointment_results.appointment_time.date(),
                "time": appointment_results.appointment_time.time(),
            },
        )

        return "The appointment has been made, ask the user if they need assistance with anything else."

    # @function_tool()
    # async def modify_appointment(self):
    #     """Call if the user requests to reschedule or cancel an existing appointment"""

    # @function_tool()
    # async def medication_refill(self):
    #     """Facilitates medicine refill"""

    @function_tool()
    async def update_record(
        self,
        field: Annotated[
            str, Field(json_schema_extra={"enum": ["name", "dob", "email", "insurance"]})
        ],
    ):
        """Update a specific field in the user's records.

        Args:
            field (str): The field to update
        """
        # TODO: if user profile not found
        if field == "name":
            await self.session.generate_reply(
                instructions="The user may not change their name, they must create a new record."
            )
        field_to_task = {
            "name": GetNameTask,
            "dob": GetDOBTask,
            "email": GetEmailTask,
            "insurance": GetInsuranceTask,
        }

        task_class = field_to_task.get(field)

        result = await task_class()
        # TODO change the field in the database
        return f"The user's {field} has been updated."


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    db = FakeDatabase()
    session = AgentSession(
        userdata=UserData(database=db),
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
