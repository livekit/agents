import asyncio
import random
from datetime import timedelta

from livekit.agents import (
    Agent,
    AgentHandler,
    DelegationOptions,
    InputRequired,
    RunContext,
    RunResult,
    TextSession,
    inference,
)
from livekit.agents.a2a import A2ASessionContext

MODEL = "google/gemma-4-31b-it"

# Stand-ins for slow systems the backend waits on.
PREPARE_SECONDS = 8
SUBMIT_SECONDS = 3
LOOKUP_SECONDS = 4

backend = Agent(
    instructions=(
        "You are the backend. You run jobs and look things up for the user; what you say "
        "reaches them through the frontend, a voice agent. Prepare a job with prepare_job, then "
        "submit it with submit_job. Pass a priority only when the user chose one; prepare_job "
        "asks otherwise. An answer to a question is the next request: call the tool again with "
        "it. Once prepare_job reports a job ready, call submit_job right away: it asks the user "
        "to confirm by itself, so never ask for that confirmation yourself. Questions about "
        "documents go to look_up; pass the quarter only when the user said it. Handle everything in a request, and reply in one or two plain "
        "sentences."
    ),
    llm=MODEL,
)


@backend.tool
async def prepare_job(
    ctx: RunContext, description: str, priority: str | None = None
) -> str | InputRequired:
    """Check that a job can run.

    Args:
        description: what the job does, as the user said it
        priority: "normal" or "high", only when the user chose
    """
    if priority is None:
        # Progress reaches the frontend while the work goes on.
        ctx.update(f"Checking what '{description}' needs.")
        await asyncio.sleep(PREPARE_SECONDS / 2)
        ctx.update("Reserving capacity for it.")
        await asyncio.sleep(PREPARE_SECONDS / 2)
        # A question in the middle of the work: the frontend asks it, and its routing model
        # recognizes the user's answer and sends it back here.
        return InputRequired("Should it run at normal or high priority?")
    ctx.update(f"Checking capacity at {priority} priority.")
    await asyncio.sleep(2)
    return f"'{description}' is ready at {priority} priority; submit it with submit_job now."


def read_back(ctx: RunContext, description: str, priority: str) -> str:
    return f"Submit '{description}' at {priority} priority?"


@backend.tool(needs_approval=read_back)
async def submit_job(ctx: RunContext, description: str, priority: str) -> str:
    """Submit a prepared job. Runs only after the user confirmed the read-back.

    Args:
        description: what the job does
        priority: "normal" or "high"
    """
    ctx.update("Submitting the job.")
    await asyncio.sleep(SUBMIT_SECONDS)
    job_id = f"JOB-{random.randint(1000, 9999)}"
    # Exact words: the frontend speaks the ID as written, never rephrased.
    await ctx.say(f"Your job ID is {job_id}.")
    return f"submitted '{description}' at {priority} priority as {job_id}"


@backend.tool
async def look_up(ctx: RunContext, topic: str, quarter: str | None = None) -> str | InputRequired:
    """Look a topic up in the documents of one quarter.

    Args:
        topic: what to look up
        quarter: "Q1" to "Q4", only when the user said which
    """
    if quarter is None:
        return InputRequired(f"Which quarter should I search for '{topic}'?")
    ctx.update(f"Searching the {quarter} documents for '{topic}'.")
    await asyncio.sleep(LOOKUP_SECONDS)
    return f"3 {quarter} documents about '{topic}'; the most recent is from last week"


back_office = AgentHandler()


# What happens to a message that arrives while the backend is busy, by channel:
# - "delegation" (the voice frontend): "live". Each request keeps its own task and runs beside
#   the others, so two jobs are two tasks, each with its own question.
# - "a2a" (a text client): "batch". Messages sent while a turn runs are merged into one next
#   turn, and the earlier tasks complete pointing at it (livekit/mergedInto).
@back_office.a2a_session(
    partition_key="backend",
    endpoint="backend",
    description="Runs jobs and looks things up.",
    inbox={"delegation": "live", "a2a": "batch"},
    debounce=timedelta(seconds=1),
)
async def text(ctx: A2ASessionContext) -> RunResult[str]:
    async with inference.LLM(MODEL) as routing:
        return await TextSession(
            agent=backend, delegation=DelegationOptions(routing_model=routing)
        ).run(ctx.message, context=ctx.store.bind())
