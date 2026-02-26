import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated

from dotenv import load_dotenv
from fake_database import FakeDatabase
from openai import AsyncOpenAI
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
    inference,
    llm,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.beta.workflows import (
    GetCreditCardTask,
    GetDOBTask,
    GetNameTask,
    GetPhoneNumberTask,
    TaskGroup,
    WarmTransferTask,
)
from livekit.agents.llm import ToolError, function_tool
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.plugins import openai, silero

logger = logging.getLogger("HealthcareAgent")

load_dotenv(".env.local")

# to test out warm transfer, ensure the following variables/env vars are set
SIP_TRUNK_ID = os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK")  # "ST_abcxyz"
SUPERVISOR_PHONE_NUMBER = os.getenv("LIVEKIT_SUPERVISOR_PHONE_NUMBER")  # "+12003004000"
SIP_NUMBER = os.getenv("LIVEKIT_SIP_NUMBER")  # "+15005006000" - caller ID shown to supervisor

ValidInsurances = ["Anthem", "Aetna", "EmblemHealth", "HealthFirst"]

GLOBAL_INSTRUCTIONS = "Be succinct and to the point when assisting the user. Never give medical advice or diagnose users, escalate to a human whenever the user's request is out of your scope of assistance."


@dataclass
class UserData:
    database: FakeDatabase
    profile: dict | None
    oai_client: AsyncOpenAI | None = None
    vector_store_id: str | None = None
    file_id: str | None = None


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
        self.complete(GetInsuranceResult(insurance=insurance))


@function_tool()
async def update_record(
    context: RunContext,
    field: Annotated[str, Field(json_schema_extra={"enum": ["name", "dob", "phone", "insurance"]})],
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
        "phone": (GetPhoneNumberTask, "phone_number"),
        "insurance": (GetInsuranceTask, "insurance"),
    }

    task_class, attr = field_map.get(field)
    result = await task_class()
    value = getattr(result, attr)

    name = context.session.userdata.profile["name"]
    updated = context.session.userdata.database.update_patient_record(name, **{attr: value})
    if not updated:
        return "No profile was found to update"
    return f"The user's {field} has been updated."


class ScheduleAppointmentTask(AgentTask[ScheduleAppointmentResult]):
    def __init__(self, chat_ctx: llm.ChatContext | None = None):
        super().__init__(
            instructions="""You will now assist the user with selecting a doctor and appointment time.
            Do not be verbose and ask for any unnecessary information unless instructed to.
            Avoid using special characters like bullet points when listing out doctors and available timeslots, maintain a natural tone.
            """
            + GLOBAL_INSTRUCTIONS,
            tools=[
                update_record,
                transfer_to_human,
                EndCallTool(
                    end_instructions="Disclose that the call is ending because the user refuses to cooperate or provide information and say goodbye.",
                    delete_room=True,
                ),
            ],
            chat_ctx=chat_ctx,
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
        current_tools = [t for t in self.tools if t.id != "confirm_doctor_selection"]
        current_tools.append(doctor_confirmation_tool)
        await self.update_tools(current_tools)
        chat_ctx = self.chat_ctx.copy()
        chat_ctx.add_message(
            role="system",
            content=f"These doctors are compatible with the user's insurance: {available_doctors}",
        )
        await self.update_chat_ctx(chat_ctx)
        extra = (
            " Avoid special notation when listing out the doctors, this will be read out. "
            if self.session.current_speech.input_details.modality == "audio"
            else ""
        )
        if len(self._compatible_doctor_records) > 1:
            await self.session.generate_reply(
                instructions="Inform the user of the doctors compatible to them, and prompt the user to choose one."
                + extra
            )
        else:
            await self.session.generate_reply(
                instructions="Inform the user of their compatible doctor and confirm if they would like to select that doctor."
                + extra
            )

    def _build_doctor_selection_tool(self, *, available_doctors: list[str]) -> FunctionTool | None:
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
            current_tools = [t for t in self.tools if t.id != "schedule_appointment"]
            current_tools.append(schedule_appointment_tool)
            await self.update_tools(current_tools)

            chat_ctx = self.chat_ctx.copy()
            chat_ctx.add_message(
                role="system",
                content=f"The selected doctor has availabilities at {available_times}.",
            )
            await self.update_chat_ctx(chat_ctx)
            extra = (
                " Avoid special notation when listing out the available time slots, this will be read out. "
                if self.session.current_speech.input_details.modality == "audio"
                else ""
            )

            await self.session.generate_reply(
                instructions="Inform and ask the user which time slot they prefer, and do not list out the times using bullet points."
                + extra
            )

        return confirm_doctor_selection

    def _build_schedule_appointment_tool(
        self, *, available_times: list[dict]
    ) -> FunctionTool | None:
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
            self._appointment_time = datetime.fromisoformat(appointment_time)

            visit_reason_tool = self._build_visit_reason_tool()
            current_tools = [t for t in self.tools if t.id != "confirm_visit_reason"]
            current_tools.append(visit_reason_tool)
            await self.update_tools(current_tools)

            await self.session.generate_reply(
                instructions="Prompt the user for the reason for their visit."
            )

        return schedule_appointment

    def _build_visit_reason_tool(self) -> FunctionTool:
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
    def __init__(self, function: str, chat_ctx: llm.ChatContext | None = None):
        super().__init__(
            instructions="""You will now assist the user with modifying their appointment.
            Do not be verbose and ask for any unnecessary information unless instructed to.
            Avoid using special characters like bullet points, maintain a natural tone."""
            + GLOBAL_INSTRUCTIONS,
            chat_ctx=chat_ctx,
            tools=[
                update_record,
                transfer_to_human,
                EndCallTool(
                    end_instructions="Disclose that the call is ending because the user refuses to cooperate or provide information and say goodbye.",
                    delete_room=True,
                ),
            ],
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
            modify_appt_tool = self._build_modify_appt_tool(available_appts=appointments)
            current_tools = [t for t in self.tools if t.id != "confirm_appointment_selection"]
            current_tools.append(modify_appt_tool)
            await self.update_tools(current_tools)

            chat_ctx = self.chat_ctx.copy()
            chat_ctx.add_message(
                role="system",
                content=f"The user has these outstanding appointments: {json.dumps(appointments, default=str)} and requested to {self._function} one.",
            )
            await self.update_chat_ctx(chat_ctx)

            extra = (
                " Avoid special notation when listing out the appointments, this will be read out. "
                if self.session.current_speech.input_details.modality == "audio"
                else ""
            )

            await self.session.generate_reply(
                instructions="Prompt the user to choose one of the appointments to modify, and confirm if they would either like to reschedule or cancel it."
                + extra
            )

    def _build_modify_appt_tool(self, *, available_appts: list[dict]) -> FunctionTool:
        appt_by_time = {str(appt["appointment_time"]): appt for appt in available_appts}
        appt_times = list(appt_by_time.keys())

        @function_tool()
        async def confirm_appointment_selection(
            function: Annotated[
                str,
                Field(
                    description="Available functions to modify an existing appointment",
                    json_schema_extra={"enum": ["reschedule", "cancel"]},
                ),
            ],
            selected_appointment_time: Annotated[
                str,
                Field(
                    description="The appointment time to cancel or reschedule",
                    json_schema_extra={"enum": appt_times},
                ),
            ],
        ) -> None:
            """Call to confirm the user's appointment selection to either cancel or reschedule"""
            appointment = appt_by_time[selected_appointment_time]
            self._selected_appointment = appointment
            self._database.cancel_appointment(self._patient_profile["name"], appointment)

            if function == "cancel":
                self.complete(
                    ModifyAppointmentResult(new_appointment=None, old_appointment=appointment)
                )
            else:
                result = await ScheduleAppointmentTask(chat_ctx=self.chat_ctx)
                self.complete(
                    ModifyAppointmentResult(new_appointment=result, old_appointment=appointment)
                )

        return confirm_appointment_selection


class HealthcareAgent(Agent):
    def __init__(self, database=None) -> None:
        super().__init__(
            instructions=(
                "You are a healthcare agent offering assistance to users. Maintain a friendly disposition. If the user refuses to provide any requested information or does not cooperate, call EndCallTool.\n"
                "Before scheduling/modifying appointments and retrieving lab results, you will be authenticating the user's information and checking for an existing profile. Do not preemptively ask for information (ex. birthday) unless instructed to.\n"
                "Call 'schedule_appointment' to schedule a new appointment." + GLOBAL_INSTRUCTIONS
            ),
            tools=[
                EndCallTool(
                    end_instructions="Disclose that the call is ending because the user refuses to cooperate or provide information and say goodbye.",
                    delete_room=True,
                ),
                transfer_to_human,
            ],
        )
        self._found_profile: NotGivenOr[bool] = NOT_GIVEN

        self._database = database

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Greet the user and gather the reason for their call."
        )

    async def task_completed_callback(self, event, task_group):
        if event.task_id == "get_name_task":
            patient_name = event.result.first_name + " " + event.result.last_name
            existing_record = self._database.get_patient_by_name(patient_name)
            if existing_record:
                logger.info(f"Found existing patient profile for {patient_name}")
                self._found_profile = True
                self.session.userdata.profile = existing_record
                chat_ctx = task_group.chat_ctx.copy()
                chat_ctx.add_message(
                    role="system",
                    content=f"Alert the user that an existing patient record has been found. This birthday has been found linked to the existing profile, confirm it with the user: {self.session.userdata.profile['date_of_birth']}. After confirming, call 'update_dob'.",
                )
                await task_group.update_chat_ctx(chat_ctx)
            elif not existing_record and self.session.userdata.profile:
                # in the case that the user creates a new profile or restarts, the recorded session profile is cleared
                self.session.userdata.profile = {}
                self._found_profile = NOT_GIVEN

        # each profile field is injected into the taskgroup context before the respective task is executed
        if event.task_id == "get_dob_task" and self._found_profile:
            chat_ctx = task_group.chat_ctx.copy()
            chat_ctx.add_message(
                role="system",
                content=f"This phone number has been found linked to the existing profile, confirm it with the user: {self.session.userdata.profile['phone_number']}. After confirming, call 'update_phone_number'.",
            )
            await task_group.update_chat_ctx(chat_ctx)
        if event.task_id == "get_phone_number_task" and self._found_profile:
            chat_ctx = task_group.chat_ctx.copy()
            chat_ctx.add_message(
                role="system",
                content=f"This insurance has been found linked to the existing file, confirm it with the user: {self.session.userdata.profile['insurance']}. After confirming, call 'record_health_insurance'.",
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
                lambda: GetNameTask(
                    last_name=True,
                    extra_instructions="Do not be verbose and ask for any unnecessary information unless instructed to.",
                ),
                id="get_name_task",
                description="Gathers the user's name",
            )
            task_group.add(
                lambda: GetDOBTask(
                    require_confirmation=False if is_given(self._found_profile) else NOT_GIVEN,
                    extra_instructions="Do not be verbose and ask for any unnecessary information unless instructed to.",
                ),
                id="get_dob_task",
                description="Gathers the user's date of birth",
            )
            task_group.add(
                lambda: GetPhoneNumberTask(
                    require_confirmation=False if is_given(self._found_profile) else NOT_GIVEN,
                    extra_instructions="Do not be verbose and ask for any unnecessary information unless instructed to.",
                ),
                id="get_phone_number_task",
                description="Gathers the user's phone number",
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
                "phone_number": results.task_results["get_phone_number_task"].phone_number,
                "insurance": results.task_results["get_insurance_task"].insurance,
            }

            if self._found_profile:
                self._database.update_patient_record(patient_name, **profile)
                self.session.userdata.profile = self._database.get_patient_by_name(patient_name)
            else:
                self.session.userdata.profile = profile
                self._database.add_patient_record(info=profile)

            current_tools = [t for t in self.tools if t.id != "update_record"]
            current_tools.append(update_record)
            await self.update_tools(current_tools)

    @function_tool()
    async def schedule_appointment(self):
        """Call to schedule an appointment for the user. Do not ask for any information in advance."""
        await self.profile_authenticator()
        result = await ScheduleAppointmentTask(chat_ctx=self.chat_ctx)

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
                json_schema_extra={"enum": ["reschedule", "cancel"]},
            ),
        ],
    ):
        """Call if the user requests to reschedule or cancel an existing appointment"""
        await self.profile_authenticator()

        result = await ModifyAppointmentTask(function=function, chat_ctx=self.chat_ctx)
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
            confirmation_message += f" and a new appointment ({json.dumps(appointment, default=str)}) has been scheduled."

        return confirmation_message

    @function_tool()
    async def retrieve_lab_results(self):
        """Call if the user wishes to see their latest lab results"""
        await self.profile_authenticator()

        userdata = self.session.userdata
        if userdata.oai_client is None:
            if not os.path.isfile("mock_checkup_report.pdf"):
                logger.warning(
                    "To try out this task, 'mock_checkup_report.pdf' must be in the current directory."
                )
                return "No report was found"
            await self.session.generate_reply(
                instructions="Inform the user you are fetching their report."
            )
            userdata.oai_client = AsyncOpenAI()
            vector_store = await userdata.oai_client.vector_stores.create(name="lab_reports")
            userdata.vector_store_id = vector_store.id
            with open("mock_checkup_report.pdf", "rb") as f:
                file = await userdata.oai_client.files.create(file=f, purpose="assistants")
            userdata.file_id = file.id
            await userdata.oai_client.vector_stores.files.create_and_poll(
                vector_store_id=userdata.vector_store_id, file_id=userdata.file_id
            )

            filesearch_tool = openai.tools.FileSearch(vector_store_ids=[userdata.vector_store_id])
            current_tools = [t for t in self.tools if not isinstance(t, openai.tools.FileSearch)]
            current_tools.append(filesearch_tool)
            await self.update_tools(current_tools)
        await self.session.generate_reply(
            instructions="You now are able to access the user's report, invite any queries regarding it. Keep descriptions short and succinct unless requested otherwise."
        )

    @function_tool()
    async def retrieve_available_doctors(self) -> None:
        """Call if the user inquires about the available doctors in the network"""
        await self.session.generate_reply(
            instructions=f"Inform the user about each doctor record: {self._database.doctor_records}"
        )

    @function_tool()
    async def handle_billing(self):
        """Call for any billing inquiries, like if the user wants to check their outstanding balance or if they want to pay a bill."""
        await self.profile_authenticator()

        name = self.session.userdata.profile["name"]
        balance = self._database.get_outstanding_balance(name)

        payment_proceeds_tool = self._build_payment_proceeds_tool()
        current_tools = [t for t in self.tools if t.id != "confirm_payment_proceeds"]
        current_tools.append(payment_proceeds_tool)
        await self.update_tools(current_tools)
        await self.session.generate_reply(
            instructions=f"Inform the patient that their outstanding balance is ${balance} and ask if they would like to pay it now."
        )

    def _build_payment_proceeds_tool(self) -> FunctionTool:
        @function_tool()
        async def confirm_payment_proceeds(amount: float) -> None:
            """Call to proceed with payment steps regarding the user's bill.

            Args:
                amount (float): The dollar amount the user wishes to pay toward their balance.
            """
            name = self.session.userdata.profile["name"]
            balance = self._database.get_outstanding_balance(name)

            if amount <= 0:
                return "The payment amount must be greater than zero."
            if amount > balance:
                return f"The payment amount exceeds the outstanding balance of ${balance}."

            result = await GetCreditCardTask()

            last_four_digits = str(result.card_number)[-4:]
            remaining = self._database.apply_payment(name, amount)

            current_tools = [t for t in self.tools if t.id != "confirm_payment_proceeds"]
            await self.update_tools(current_tools)
            return f"The payment method ending in {last_four_digits} has been successfully charged ${amount}. Remaining balance: ${remaining}."

        return confirm_payment_proceeds


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    db = FakeDatabase()
    userdata = UserData(database=db, profile=None)
    session = AgentSession(
        userdata=userdata,
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=openai.responses.LLM(model="gpt-4.1"),  # switch to Websocket Mode once merged
        tts=inference.TTS("inworld/inworld-tts-1"),
        vad=silero.VAD.load(),
    )

    async def on_session_close() -> None:
        if userdata.oai_client is None:
            return
        try:
            if userdata.vector_store_id is not None:
                await userdata.oai_client.vector_stores.delete(userdata.vector_store_id)
        except Exception:
            logger.exception("failed to delete vector store")
        try:
            if userdata.file_id is not None:
                await userdata.oai_client.files.delete(userdata.file_id)
        except Exception:
            logger.exception("failed to delete file")
        await userdata.oai_client.close()

    ctx.add_shutdown_callback(on_session_close)

    await session.start(
        agent=HealthcareAgent(database=db),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
