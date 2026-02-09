import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Optional

from dotenv import load_dotenv
from fake_database import FakeDatabase
from openai import OpenAI
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
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.beta.workflows import (
    GetDOBTask,
    GetEmailTask,
    GetNameTask,
    TaskGroup,
    WarmTransferTask,
)
from livekit.agents.llm import ToolError, function_tool
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("HealthcareAgent")

load_dotenv()

# to test out warm transfer, ensure the following variables/env vars are set
SIP_TRUNK_ID = os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK")  # "ST_abcxyz"
SUPERVISOR_PHONE_NUMBER = os.getenv("LIVEKIT_SUPERVISOR_PHONE_NUMBER")  # "+12003004000"
SIP_NUMBER = os.getenv("LIVEKIT_SIP_NUMBER")  # "+15005006000" - caller ID shown to supervisor

ValidInsurances = ["Anthem", "Aetna", "EmblemHealth", "HealthFirst"]


@dataclass
class UserData:
    database: FakeDatabase
    profile: dict | None


@dataclass
class GetInsuranceResult:
    insurance: str


@dataclass
class ScheduleAppointmentResult:
    doctor_name: str
    appointment_time: datetime
    visit_reason: str


@dataclass
class ModifyAppointmentResult:
    new_appointment: ScheduleAppointmentResult | None
    old_appointment: dict


@function_tool()
async def transfer_to_human(context: RunContext) -> None:
    """Called when the user asks to speak to a human agent. This will put the user on
    hold while the supervisor is connected.

    Ensure that the user has confirmed that they wanted to be transferred. Do not start transfer
    until the user has confirmed.
    Examples on when the tool should be called:
    ----
    - User: Can I speak to your supervisor?
    - Assistant: Yes of course.
    ----
    - Assistant: I'm unable to help with that, would you like to speak to a human agent?
    - User: Yes please.
    ----
    """

    logger.info("tool called to transfer to human")
    await context.session.say(
        "Please hold while I connect you to a human agent.", allow_interruptions=False
    )
    try:
        assert SIP_TRUNK_ID is not None
        assert SUPERVISOR_PHONE_NUMBER is not None

        result = await WarmTransferTask(
            target_phone_number=SUPERVISOR_PHONE_NUMBER,
            sip_trunk_id=SIP_TRUNK_ID,
            sip_number=SIP_NUMBER,
            chat_ctx=context.session.history,
        )
    except ToolError as e:
        logger.error(f"failed to transfer to supervisor with tool error: {e}")
        raise e
    except Exception as e:
        logger.exception("failed to transfer to supervisor")
        raise ToolError(f"failed to transfer to supervisor with error: {e}") from e

    logger.info(
        "transfer to supervisor successful",
        extra={"supervisor_identity": result.human_agent_identity},
    )
    await context.session.say(
        "you are on the line with my supervisor. I'll be hanging up now.",
        allow_interruptions=False,
    )
    context.session.shutdown()


class GetInsuranceTask(AgentTask[GetInsuranceResult]):
    def __init__(self):
        super().__init__(
            instructions="""
            You will be gathering the user's health insurance. Be sure to confirm their answer. Avoid using dashes and special characters in your response.
        """,
            tools=[transfer_to_human],
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


@function_tool()
async def update_record(
    context: RunContext,
    field: Annotated[str, Field(json_schema_extra={"enum": ["name", "dob", "email", "insurance"]})],
):
    """Call when the user requests to modify information in their existing patient record.

    Args:
        field (str): The field to update
    """
    if field == "name":
        await context.session.generate_reply(instructions="The user may not change their name.")
        return
    field_map = {
        "dob": (GetDOBTask, "date_of_birth"),
        "email": (GetEmailTask, "email_address"),
        "insurance": (GetInsuranceTask, "insurance"),
    }

    task_class, attr = field_map.get(field)
    result = await task_class()
    value = getattr(result, attr)

    name = context.session.userdata.profile["name"]
    updated = context.session.userdata.database.update_patient_record(name, **{attr: value})
    if not updated:  # this will only execute in the main HealthcareAgent() flow
        return "No profile was found to update"
    return f"The user's {field} has been updated."


class ScheduleAppointmentTask(AgentTask[ScheduleAppointmentResult]):
    def __init__(self):
        super().__init__(
            instructions="You will now assist the user with selecting a doctor and appointment time.",
            tools=[update_record, transfer_to_human],
        )
        self._selected_doctor: str | None = None
        self._appointment_time: datetime | None = None

    async def on_enter(self):
        database = self.session.userdata.database
        insurance = self.session.userdata.profile["insurance"]

        self._compatible_doctor_records = database.get_compatible_doctors(insurance=insurance)

        available_doctors = [doctor["name"] for doctor in self._compatible_doctor_records]
        doctor_confirmation_tool = self._build_doctor_selection_tool(
            available_doctors=available_doctors
        )
        current_tools = [t for t in self.tools if t.name != "confirm_doctor_selection"]
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
            doctor_record = self.session.userdata.database.get_doctor_by_name(selected_doctor)

            available_times = doctor_record["availability"]
            schedule_appointment_tool = self._build_schedule_appointment_tool(
                available_times=available_times
            )
            current_tools = [t for t in self.tools if t.name != "schedule_appointment"]
            current_tools.append(schedule_appointment_tool)
            await self.update_tools(current_tools)

            await self.session.generate_reply(
                instructions=f"The selected doctor has availabilities at {available_times}. Ask the user which time slot they prefer."
            )

        return confirm_doctor_selection

    def _build_schedule_appointment_tool(
        self, *, available_times: list[dict]
    ) -> Optional[FunctionTool]:
        iso_times = [
            datetime.combine(slot["date"], slot["time"]).isoformat() for slot in available_times
        ]

        @function_tool()
        async def schedule_appointment(
            appointment_time: Annotated[
                str,
                Field(
                    description="The available appointment times in ISO format",
                    json_schema_extra={"items": {"enum": iso_times}},
                ),
            ],
        ):
            """Call to confirm the user's selected appointment time.

            Args:
                appointment_time (str): The user's appointment time selection in ISO format
            """
            self._appointment_time = appointment_time

            visit_reason_tool = self._build_visit_reason_tool()
            current_tools = [t for t in self.tools if t.name != "confirm_visit_reason"]
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


class ModifyAppointmentTask(AgentTask[ModifyAppointmentResult]):
    def __init__(self, function: str):
        super().__init__(
            instructions="You will now assist the user with modifying their appointment.",
            tools=[update_record, transfer_to_human],
        )
        self._function = function
        self._selected_appointment: dict | None = None

    async def on_enter(self):
        self._database = self.session.userdata.database
        self._patient_profile = self.session.userdata.profile

        name = self._patient_profile["name"]
        appointments = self._database.get_patient_by_name(name).get("appointments", [])
        if not appointments:
            await self.session.generate_reply(
                instructions="Inform the user that they have no appointments on file."
            )
            self.complete(ModifyAppointmentResult(new_appointment=None, old_appointment={}))
            return
        else:
            cancel_appt_tool = self._build_modify_appt_tool(available_appts=appointments)
            current_tools = [t for t in self.tools if t.name != "confirm_appointment_selection"]
            current_tools.append(cancel_appt_tool)
            await self.update_tools(current_tools)
            await self.session.generate_reply(
                instructions=f"The user has these outstanding appointments: {json.dumps(appointments)}, prompt them to choose one to modify."
            )

    def _build_modify_appt_tool(self, *, available_appts: list[str]) -> Optional[FunctionTool]:
        @function_tool()
        async def confirm_appointment_selection(
            selected_appointment: Annotated[
                str,
                Field(
                    description="The names of the available appointments to cancel or reschedule",
                    json_schema_extra={"items": {"enum": available_appts}},
                ),
            ],
        ) -> None:
            """Call to confirm the user's appointment selection to either cancel or reschedule

            Args:
                selected_appointment (str): The appointment the user selects for modification
            """
            self._selected_appointment = selected_appointment
            self._database.cancel_appointment(self._patient_profile["name"], selected_appointment)

            if self._function == "cancel":
                self.complete(
                    ModifyAppointmentResult(
                        new_appointment=None, old_appointment=selected_appointment
                    )
                )
            else:
                result = await ScheduleAppointmentTask()
                self.complete(
                    ModifyAppointmentResult(
                        new_appointment=result, old_appointment=selected_appointment
                    )
                )

        return confirm_appointment_selection


class HealthcareAgent(Agent):
    def __init__(self, database=None) -> None:
        super().__init__(
            instructions=(
                "You are a healthcare agent offering assistance to users. Maintain a friendly disposition. If the user refuses to provide any requested information or does not cooperate, call EndCallTool.\n"
                "Before scheduling/modifying appointments and retrieving lab results, you will be authenticating the user's information and checking for an existing profile. Do not preemptively ask for information (ex. birthday) unless instructed to.\n"
                "Call 'schedule_appointment' to schedule a new appointment."
            ),
            tools=[
                EndCallTool(
                    end_instructions="Disclose that the call is ending because the user refuses to cooperate or provide information and say goodbye.",
                    delete_room=True,
                ),
                update_record,
                transfer_to_human,
            ],
        )
        self._information_verified: bool = False
        self._database = database

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Greet the user and gather the reason for their call."
        )

    async def task_completed_callback(self, event, task_group):
        if event.task_id == "get_name_task":
            patient_name = event.result.first_name + " " + event.result.last_name
            existing_record = self._database.get_patient_by_name(patient_name)
            if existing_record:
                logger.info(f"Found existing patient profile for {patient_name}")
                self.session.userdata.profile = existing_record
                chat_ctx = task_group.chat_ctx.copy()
                chat_ctx.add_message(
                    role="system",
                    content=f"Alert the user that an existing patient record has been found. This birthday has been found linked to the existing profile, confirm it with the user: {self.session.userdata.profile['date_of_birth']}",
                )
                await task_group.update_chat_ctx(chat_ctx)
            elif not existing_record and self.session.userdata.profile:
                # in the case that the user creates a new profile or restarts, the recorded session profile is cleared
                self.session.userdata.profile = {}

        # each profile field is injected into the taskgroup context before the respective task is executed
        if event.task_id == "get_dob_task" and self.session.userdata.profile["date_of_birth"]:
            chat_ctx = task_group.chat_ctx.copy()
            chat_ctx.add_message(
                role="system",
                content=f"This email has been found linked to the existing file, confirm it with the user: {self.session.userdata.profile['email']}",
            )
            await task_group.update_chat_ctx(chat_ctx)
        if event.task_id == "get_email_task" and self.session.userdata.profile["email"]:
            chat_ctx = task_group.chat_ctx.copy()
            chat_ctx.add_message(
                role="system",
                content=f"This insurance has been found linked to the existing file, confirm it with the user: {self.session.userdata.profile['insurance']}",
            )
            await task_group.update_chat_ctx(chat_ctx)

    async def profile_authenticator(self) -> None:
        """Creates a TaskGroup that collects user information"""
        logger.info("Authenticating user information")
        if not self.session.userdata.profile:
            task_group = TaskGroup(
                chat_ctx=self.chat_ctx,
                return_exceptions=False,
                on_task_completed=lambda event: self.task_completed_callback(event, task_group),
            )

            task_group.add(
                lambda: GetNameTask(last_name=True),
                id="get_name_task",
                description="Gathers the user's name",
            )
            task_group.add(
                lambda: GetDOBTask(),
                id="get_dob_task",
                description="Gathers the user's date of birth",
            )
            task_group.add(
                lambda: GetEmailTask(),
                id="get_email_task",
                description="Gathers the user's email",
            )
            task_group.add(
                lambda: GetInsuranceTask(),
                id="get_insurance_task",
                description="Gathers the user's insurance",
            )

            results = await task_group

            patient_name = f"{results.task_results['get_name_task'].first_name} {results.task_results['get_name_task'].last_name}"
            profile = {
                "name": patient_name,
                "date_of_birth": results.task_results["get_dob_task"].date_of_birth,
                "email": results.task_results["get_email_task"].email_address,
                "insurance": results.task_results["get_insurance_task"].insurance,
            }
            self.session.userdata.profile = profile
            self._database.add_patient_record(info=profile)

    @function_tool()
    async def schedule_appointment(self):
        """Call to schedule an appointment for the user. Do not ask for any information in advance."""
        await self.profile_authenticator()
        result = await ScheduleAppointmentTask()

        appointment = {
            "doctor_name": result.doctor_name,
            "appointment_time": result.appointment_time,
            "visit_reason": result.visit_reason,
        }

        self._database.add_appointment(
            name=self.session.userdata.profile["name"], appointment=appointment
        )

        return "The appointment has been made, ask the user if they need assistance with anything else."

    @function_tool()
    async def modify_appointment(
        self,
        function: Annotated[
            str,
            Field(
                description="Available functions to modify an existing appointment",
                json_schema_extra={"items": {"enum": ["reschedule", "cancel"]}},
            ),
        ],
    ):
        """Call if the user requests to reschedule or cancel an existing appointment"""
        await self.profile_authenticator()

        result = await ModifyAppointmentTask(function=function)
        confirmation_message = (
            f"Inform the user that the old appointment ({result.old_appointment}) has been canceled"
        )
        if result.new_appointment:
            appointment = {
                "doctor_name": result.new_appointment.doctor_name,
                "appointment_time": result.new_appointment.appointment_time,
                "visit_reason": result.new_appointment.visit_reason,
            }

            self._database.add_appointment(
                name=self.session.userdata.profile["name"], appointment=appointment
            )
            confirmation_message += (
                f" and a new appointment ({json.dumps(appointment)}) has been scheduled."
            )

        return confirmation_message

    @function_tool()
    async def retrieve_lab_results(self):
        """Call if the user wishes to see their latest lab results"""
        await self.profile_authenticator()

        if not os.path.isfile("mock_checkup_report.pdf"):
            logger.warning(
                "To try out this task, 'mock_checkup_report.pdf' must be in the current directory."
            )
            return

        client = OpenAI()
        vector_store = client.vector_stores.create(name="lab_reports")
        with open("mock_checkup_report.pdf", "rb") as f:
            file = client.files.create(file=f, purpose="assistants")
        client.vector_stores.files.create(vector_store_id=vector_store.id, file_id=file.id)

        filesearch_tool = openai.tools.FileSearch(vector_store_ids=[vector_store.id])
        current_tools = self.tools
        current_tools.append(filesearch_tool)
        await self.update_tools(current_tools)

        # TODO delete upon exit
        # client.vector_stores.delete(self._vector_store.id)
        # client.files.delete(self._file.id)

    @function_tool()
    async def retrieve_available_doctors(self) -> None:
        """Call if the user inquires about the available doctors in the network"""
        await self.session.generate_reply(
            instructions=f"Inform the user about each doctor record: {self._database.doctor_records}"
        )

    @function_tool()
    async def handle_billing(self):
        """Call for any billing inquiries and if the user wants to pay their outstanding balance"""
        ...
        # results = await GetCreditCardTask()


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    db = FakeDatabase()
    session = AgentSession(
        userdata=UserData(database=db, profile=None),
        stt=deepgram.STT(),
        llm=openai.responses.LLM(model="gpt-4.1"),
        tts=deepgram.TTS(),
        vad=silero.VAD.load(),
    )

    await session.start(
        agent=HealthcareAgent(database=db),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
