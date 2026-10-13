"""
- Northside Family Clinic prescription refills: the task waits on the prescriber's system for
  hours, then maybe on the patient.
- `Refill.on_enter` files the request with the EHR, gives it the task id, and returns
  `Working(tool=self.handle_decision, timeout=3d)`.
- The EHR posts its decision as a `DataPart` by task id, which calls `handle_decision` with no
  model.
- If only the generic is in stock, the tool returns `InputRequired(question,
  tool=self.send_to_pharmacy)`; the patient's free-text answer reaches the task through the clinic
  agent's handoff and becomes the tool call.
- `Completed(value)` ends the task, and the value is the last message to the patient; see
  `human-in-the-loop.md` and the "Answering a task" section of `a2a.md`.
"""
from dataclasses import dataclass
from datetime import timedelta

import httpx

from livekit.agents import Agent, AgentServer, AgentTask, Completed, InputRequired, RunContext
from livekit.agents import Working, function_tool
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.storage import AgentDBStorage

EHR = "https://api.northsideclinic.example"
PHARMACY = "https://api.pharmacy-partner.example"


@dataclass
class Patient:
    """State: saved with the clinic lane after every run."""

    pharmacy_id: str | None = None


@dataclass
class Deps:
    """What the tools and the task need for one run. Never saved."""

    ehr: httpx.AsyncClient
    pharmacy: httpx.AsyncClient


clinic = Agent(
    instructions="Northside Family Clinic. Appointments, refills, directions. Warm and brief.",
    llm="google/gemma-4-31b-it",
    deps_type=Deps,
    state_type=Patient,
)


@dataclass
class Stock:
    brand: bool
    generic: bool


async def pharmacy_stock(pharmacy: httpx.AsyncClient, pharmacy_id: str, medication: str) -> Stock:
    r = await pharmacy.get(f"/stock/{pharmacy_id}/{medication}")
    return Stock(**r.json())


# Nothing here fits inside a turn: the prescriber decides in the EHR hours later, the pharmacy may
# only have the generic, and that needs the patient's consent. So it is a task. Every return value
# is the A2A state the task moves to: Working while a system decides, InputRequired while the
# patient does, Completed when the refill is at the pharmacy.


@dataclass
class Refill(AgentTask[str]):
    medication: str

    def __post_init__(self) -> None:   # a task builds its deps like a door does
        super().__init__(
            instructions=(
                "Get the patient's refill done. When the patient answers a question, call the "
                "tool that takes the answer."
            ),
            llm="google/gemma-4-31b-it",
            deps=Deps(
                ehr=httpx.AsyncClient(base_url=EHR),
                pharmacy=httpx.AsyncClient(base_url=PHARMACY),
            ),
        )

    async def on_enter(self, ctx: RunContext[Deps, Patient]) -> Working:
        # the EHR gets this task's id and posts the prescriber's decision to it, as a DataPart
        await ctx.deps.ehr.post(
            "/refill-requests",
            json={"medication": self.medication, "callback_task": ctx.task_id},
            headers={"Idempotency-Key": ctx.task_id},
        )
        return Working(
            tool=self.handle_decision, timeout=timedelta(days=3),
            message="Sent to the prescriber for approval.",
        )

    @function_tool
    async def handle_decision(
        self, ctx: RunContext[Deps, Patient], approved: bool, substitution_allowed: bool
    ) -> Completed[str] | InputRequired:
        """The prescriber's decision, posted by the EHR. A DataPart calls this with no model."""
        if not approved:
            return Completed(
                f"Dr. Patel would like to see you before renewing your {self.medication}. "
                "Text us and we'll book a visit."
            )
        stock = await pharmacy_stock(ctx.deps.pharmacy, ctx.state.pharmacy_id, self.medication)
        if stock.brand or not substitution_allowed:
            return await self.send_to_pharmacy(ctx, generic_ok=False)
        # the question is the code's, word for word; the answer is a call to send_to_pharmacy
        return InputRequired(
            f"Dr. Patel approved your {self.medication} refill. Your pharmacy only has the generic "
            "in stock right now; is the generic okay?",
            tool=self.send_to_pharmacy, timeout=timedelta(days=2),
        )

    @function_tool
    async def send_to_pharmacy(
        self, ctx: RunContext[Deps, Patient], generic_ok: bool
    ) -> Completed[str]:
        """Send the approved refill to the patient's pharmacy, with or without substitution."""
        r = await ctx.deps.pharmacy.post(
            "/fills",
            json={
                "medication": self.medication,
                "generic_ok": generic_ok,
                "pharmacy": ctx.state.pharmacy_id,
            },
            headers={"Idempotency-Key": ctx.task_id},
        )
        fill = r.json()
        return Completed(
            f"Your {self.medication} refill is ready for pickup after {fill['ready_at']} "
            f"at {fill['pharmacy_name']}."
        )

    async def on_timeout(self, ctx: RunContext[Deps, Patient]) -> Completed[str]:
        return Completed(
            f"I haven't heard back about your {self.medication} refill yet. I'll keep an eye "
            "on it; call us if you run out before then."
        )


@clinic.tool
async def use_pharmacy(ctx: RunContext[Deps, Patient], pharmacy_id: str) -> str:
    """Remember the patient's pharmacy for refills."""
    r = await ctx.deps.pharmacy.get(f"/pharmacies/{pharmacy_id}")
    if r.status_code != 200:
        return "unknown pharmacy; ask for the name and the street"
    ctx.state.pharmacy_id = pharmacy_id
    return f"{r.json()['name']} on file"


@clinic.tool
async def request_refill(ctx: RunContext[Deps, Patient], medication: str) -> str:
    """Start a prescription refill. Takes hours; the patient is told when it is ready."""
    if ctx.state.pharmacy_id is None:
        return "ask which pharmacy first"
    handle = await ctx.submit(Refill(medication))   # the task sets its own deadlines
    return f"refill started, task {handle.id}"


server = AgentServer(storage=AgentDBStorage())


@server.a2a_session(endpoint="clinic", description="Northside Family Clinic.", state=Patient)
async def text(ctx: A2ASessionContext[Patient]) -> str:
    # A message without a task id is a turn of clinic. One with a task id goes to that task,
    # whether it comes from the patient, a form, or the EHR posting the prescriber's decision.
    deps = Deps(
        ehr=httpx.AsyncClient(base_url=EHR), pharmacy=httpx.AsyncClient(base_url=PHARMACY)
    )
    result = await clinic.run(
        ctx.message, message_history=ctx.message_history, state=ctx.state, deps=deps
    )
    return result.output


# The EHR's decision arrives as message/send {taskId} with a DataPart matching handle_decision, so
# it is called directly: no model. "Yes that's fine" from the patient carries no task id: clinic's
# turn sees Refill waiting and hands the conversation to it; Refill's model calls
# send_to_pharmacy(generic_ok=True). A system answers a task exactly as a form does: by task id.
