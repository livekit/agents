"""An auto claim's first notice of loss over text, which needs a photo of the damage."""

import secrets
from dataclasses import dataclass

from pydantic import BaseModel

from livekit.agents import Agent, AgentServer, InputRequired, RunContext, RunResult, persisted
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.storage import AgentDBStorage


@dataclass
class Loss:
    """Application data retained through explicit context saves."""

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


@dataclass
class AppContext:
    loss: Loss = persisted(default_factory=Loss)


fnol = Agent[AppContext](
    instructions="Take a first notice of loss on an auto policy: what happened, when, where. Request a photo of the damage, then open the claim with it.",
    llm="google/gemma-4-31b-it",
    output_type=Claim,
)


@fnol.tool
async def request_photo(ctx: RunContext[AppContext]) -> InputRequired:
    """Ask the policyholder for a photo of the damage. The next invocation interprets the photo they send."""
    return InputRequired("Could you send a photo of the damage? A wide shot and a close-up.")


@fnol.tool
async def open_claim(
    ctx: RunContext[AppContext], policy_number: str, description: str, photo: Photo
) -> str:
    ctx.context.loss.policy_number = policy_number
    ctx.context.loss.description = description
    if ctx.context.loss.claim_number is None:
        ctx.context.loss.claim_number = f"CLM-{secrets.randbelow(10**7):07d}"
    await ctx.store.save_context(ctx.context)
    return f"{ctx.context.loss.claim_number} opened with photo {photo.ref}"


server = AgentServer(storage=AgentDBStorage())


@server.a2a_session(endpoint="claims", description="Report an auto claim.")
async def claims(ctx: A2ASessionContext) -> RunResult[Claim]:
    app = AppContext()
    await ctx.store.load_context(app)
    result = await fnol.run(ctx.message, message_history=ctx.message_history, context=app)
    await ctx.store.save_context(app)
    return result
