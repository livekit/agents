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

agent = Agent(
    instructions=(
        "You run jobs for the user. What you say reaches them through a voice assistant. "
        "Prepare a job with prepare_job, then submit it with submit_job. Pass a priority only "
        "when the user chose one; prepare_job asks otherwise. An answer to a question is the "
        "next request: call the tool again with it. Once prepare_job reports a job ready, call "
        "submit_job right away: it asks the user to confirm by itself, so never ask for that "
        "confirmation yourself. Reply in one plain sentence."
    ),
    llm=MODEL,
)


@agent.tool
async def prepare_job(
    ctx: RunContext, description: str, priority: str | None = None
) -> str | InputRequired:
    """Check that a job can run.

    Args:
        description: what the job does, as the user said it
        priority: "normal" or "high", only when the user chose
    """
    if priority is None:
        # Progress streams to the voice assistant while the work goes on.
        ctx.update(f"Checking what '{description}' needs.")
        await asyncio.sleep(PREPARE_SECONDS / 2)
        ctx.update("Reserving capacity for it.")
        await asyncio.sleep(PREPARE_SECONDS / 2)
        # A question in the middle of the work: the task waits for the user's answer, which the
        # assistant's routing model recognizes and sends back to this task.
        return InputRequired("Should it run at normal or high priority?")
    ctx.update(f"Checking capacity at {priority} priority.")
    await asyncio.sleep(2)
    return f"'{description}' is ready at {priority} priority; submit it with submit_job now."


def read_back(ctx: RunContext, description: str, priority: str) -> str:
    return f"Submit '{description}' at {priority} priority?"


@agent.tool(needs_approval=read_back)
async def submit_job(ctx: RunContext, description: str, priority: str) -> str:
    """Submit a prepared job. Runs only after the user confirmed the read-back.

    Args:
        description: what the job does
        priority: "normal" or "high"
    """
    ctx.update("Submitting the job.")
    await asyncio.sleep(SUBMIT_SECONDS)
    job_id = f"JOB-{random.randint(1000, 9999)}"
    # Exact words: the voice assistant speaks the ID as written, never rephrased.
    await ctx.say(f"Your job ID is {job_id}.")
    return f"submitted '{description}' at {priority} priority as {job_id}"


jobs = AgentHandler()


# "batch": messages that arrive while a turn runs (the user changing their mind) are merged
# into one next turn; the earlier tasks complete pointing at it with livekit/mergedInto.
@jobs.a2a_session(
    partition_key="jobs",
    endpoint="jobs",
    description="Prepares and submits jobs.",
    inbox="batch",
    debounce=timedelta(seconds=1),
)
async def endpoint(ctx: A2ASessionContext) -> RunResult[str]:
    async with inference.LLM(MODEL) as routing:
        return await TextSession(
            agent=agent, delegation=DelegationOptions(routing_model=routing)
        ).run(ctx.message, context=ctx.store.bind())
