"""
- An auto claim's first notice of loss over text, which needs a photo of the damage.
- `request_photo` returns `InputRequired(question)`: the turn's task waits `input-required` with
  the question as its reply, and the policyholder's next message, the uploaded photo's reference,
  becomes that call's output, on any worker.
- `open_claim` takes the `Photo` as a typed argument; the thread keeps the reference, the bytes
  stay outside it.
- `output_type=Claim` gives the handler a typed result.
- The claim number is written to the `Loss` state once, so a retried call reuses it.
"""

import secrets
from dataclasses import dataclass

from pydantic import BaseModel

from livekit.agents import Agent, AgentServer, InputRequired, RunContext
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.storage import AgentDBStorage


@dataclass
class Loss:
    """State: saved with the lane after every run."""

    policy_number: str | None = None
    description: str | None = None
    claim_number: str | None = None


class Photo(BaseModel):
    """An image the policyholder uploads. The application's UI stores it and sends its reference.

    The thread keeps the reference the application resolves; the bytes are not in the thread.
    """

    ref: str


class Claim(BaseModel):
    claim_number: str
    next_step: str


fnol = Agent(
    instructions=(
        "Take a first notice of loss on an auto policy: what happened, when, where. "
        "Request a photo of the damage, then open the claim with it."
    ),
    llm="google/gemma-4-31b-it",
    state_type=Loss,
    output_type=Claim,
)


@fnol.tool
async def request_photo(ctx: RunContext[None, Loss]) -> InputRequired:
    """Ask the policyholder for a photo of the damage. Its result is the photo they send."""
    # The turn's A2A task enters input-required with this question; the pending call sits in the
    # history with a placeholder output; the reply, whenever it comes and on whatever worker,
    # replaces it.
    return InputRequired("Could you send a photo of the damage? A wide shot and a close-up.")


@fnol.tool
async def open_claim(
    ctx: RunContext[None, Loss], policy_number: str, description: str, photo: Photo
) -> str:
    ctx.state.policy_number = policy_number
    ctx.state.description = description
    if ctx.state.claim_number is None:  # idempotent: a retried call gets the same number
        ctx.state.claim_number = f"CLM-{secrets.randbelow(10**7):07d}"
    return f"{ctx.state.claim_number} opened with photo {photo.ref}"


server = AgentServer(storage=AgentDBStorage())


@server.a2a_session(endpoint="claims", description="Report an auto claim.", state=Loss)
async def claims(ctx: A2ASessionContext[Loss]) -> Claim | None:
    result = await fnol.run(ctx.message, message_history=ctx.message_history, state=ctx.state)
    return result.output  # None while the task is input-required; the question is the reply


# The policyholder reports from the roadside and uploads the photo that evening, on another worker.
# Nothing re-executes: the answer replaces the pending call's placeholder output and the model
# continues from the history. A reply that is a question instead is still the call's output; the model
# answers it and asks for the photo again, which opens a new pending call.
