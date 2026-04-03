import json
import time
from dataclasses import dataclass
from typing import Literal

from livekit.agents import Agent, AgentTask
from livekit.agents.beta.workflows import TaskGroup
from livekit.agents.llm import function_tool

DEBUG_LOG_PATH = "/Users/toubatbrian/Documents/agents-js/.cursor/debug-e6d38d.log"


def _debug_log(*, run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": "e6d38d",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


@dataclass
class UserData:
    verified_intent: str | None = None
    full_name: str | None = None
    date_of_birth: str | None = None
    preferred_date_time: str | None = None
    status: str | None = None


class VerifyIntentTask(AgentTask[str]):
    def __init__(self) -> None:
        super().__init__(
            instructions=("Verify the user's intent to schedule a patient appointment.")
        )

    async def on_enter(self) -> None:
        # region agent log
        _debug_log(
            run_id="pre-fix",
            hypothesis_id="H1",
            location="mock_patient_taskgroup_agent.py:30",
            message="VerifyIntentTask.on_enter entered",
            data={"task": "verify_intent_task"},
        )
        # endregion
        await self.session.generate_reply(
            instructions=(
                "Ask user if they want to schedule an appointment. That said, do not say anything more. Just one brief sentence is enough."
            ),
            tool_choice="none",
        )

    @function_tool()
    async def verify_intent(self, intent: Literal["schedule"]) -> None:
        # region agent log
        _debug_log(
            run_id="pre-fix",
            hypothesis_id="H2",
            location="mock_patient_taskgroup_agent.py:50",
            message="verify_intent tool executed",
            data={"intent": intent},
        )
        # endregion
        self.session.userdata.verified_intent = intent
        self.complete(intent)


class IdentifyPatientTask(AgentTask[dict]):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Ask for full name and date of birth, then call identify_patient immediately.",
            ),
        )

    async def on_enter(self) -> None:
        # region agent log
        _debug_log(
            run_id="pre-fix",
            hypothesis_id="H3",
            location="mock_patient_taskgroup_agent.py:69",
            message="IdentifyPatientTask.on_enter entered",
            data={"verified_intent": self.session.userdata.verified_intent},
        )
        # endregion
        await self.session.generate_reply(
            instructions=(
                "Ask for full name and date of birth, then call identify_patient immediately."
            ),
            tool_choice="none",
        )

    @function_tool()
    async def identify_patient(self, full_name: str, date_of_birth: str) -> None:
        self.session.userdata.full_name = full_name
        self.session.userdata.date_of_birth = date_of_birth
        self.complete({"full_name": full_name, "date_of_birth": date_of_birth})


class SchedulePatientVisitTask(AgentTask[dict]):
    def __init__(self) -> None:
        super().__init__(
            instructions=("Ask for a preferred date and time, then call schedule_patient_visit.")
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=("Ask for a preferred date and time, then call schedule_patient_visit."),
            tool_choice="none",
        )

    @function_tool()
    async def schedule_patient_visit(self, preferred_date_time: str) -> None:
        self.session.userdata.preferred_date_time = preferred_date_time
        self.session.userdata.status = "scheduled"
        self.complete(
            {
                "preferred_date_time": preferred_date_time,
                "status": "scheduled",
            }
        )


class MockPatientTaskGroupAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a concise healthcare assistant. Help the user schedule a patient appointment."
            )
        )

    async def on_enter(self) -> None:
        task_group = TaskGroup(summarize_chat_ctx=True)
        task_group.add(
            lambda: VerifyIntentTask(),
            id="verify_intent_task",
            description="confirm user scheduling intent",
        )
        task_group.add(
            lambda: IdentifyPatientTask(),
            id="identify_patient_task",
            description="collect patient identity details",
        )
        task_group.add(
            lambda: SchedulePatientVisitTask(),
            id="schedule_patient_visit_task",
            description="schedule patient visit",
        )
        await task_group
