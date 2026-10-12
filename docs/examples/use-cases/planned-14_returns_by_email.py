"""
- Larkfield Outfitters' returns desk over email: attachments in, a reply in the same thread, a
  warehouse that inspects days later.
- The email ingress keys the thread by `email:<address>`, strips quoted history, and posts the
  subject and attachments as chat items; the handler is the same as for any text channel.
- `start_return` opens the RMA from the attached photos and submits `Inspection`, which waits on
  the warehouse with `Working(tool=self.inspected, timeout=21d)`.
- A question the task sends by email comes back with `In-Reply-To`, so the ingress posts the reply
  with the task id and the main agent never sees it.
- Attachments out are a link; blobs are a planned addition to storage.
"""

from dataclasses import dataclass
from datetime import timedelta

import httpx
from pydantic import BaseModel

from livekit.agents import (
    Agent,
    AgentServer,
    AgentTask,
    Completed,
    InputRequired,
    RunContext,
    Working,
    function_tool,
)
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.storage import AgentDBStorage

SHOP = "https://api.larkfield.example"
WAREHOUSE = "https://api.larkfield-returns.example"


@dataclass
class Return:
    """State: saved with the lane after every run."""

    order_id: str | None = None
    rma_id: str | None = None


class Photo(BaseModel):
    """An attachment on the email. The item keeps a reference; the bytes are not in the thread."""

    ref: str
    name: str


@dataclass
class Deps:
    shop: httpx.AsyncClient
    warehouse: httpx.AsyncClient


support = Agent(
    instructions=(
        "Larkfield Outfitters returns desk. Replies are email: a greeting, short paragraphs, "
        "every question in the message answered, a sign-off. Markdown is rendered."
    ),
    llm="google/gemma-4-31b-it",
    deps_type=Deps,
    state_type=Return,
)


@support.tool
async def start_return(
    ctx: RunContext[Deps, Return], order_id: str, reason: str, photos: list[Photo]
) -> str:
    """Open a return for an order. Damaged items need the photos the customer attached."""
    body = {"order": order_id, "reason": reason, "photos": [p.ref for p in photos]}
    r = await ctx.deps.shop.post(
        "/returns", json=body, headers={"Idempotency-Key": ctx.function_call.call_id}
    )
    rma = r.json()
    ctx.state.order_id, ctx.state.rma_id = order_id, rma["rma_id"]
    if reason == "damaged":
        await ctx.submit(Inspection(rma["rma_id"]), timeout=timedelta(days=21))
    # the label is a link: an attachment out needs blobs, a planned addition to storage
    return f"RMA {rma['rma_id']} opened. Prepaid label: {rma['label_url']}. Refund on receipt."


@dataclass
class Inspection(AgentTask[str]):  # the warehouse inspects on receipt, days from now
    rma_id: str

    def __post_init__(self) -> None:
        super().__init__(
            instructions=(
                "When the customer answers a question, call the tool that takes the answer."
            ),
            llm="google/gemma-4-31b-it",
            deps=Deps(
                shop=httpx.AsyncClient(base_url=SHOP),
                warehouse=httpx.AsyncClient(base_url=WAREHOUSE),
            ),
        )

    async def on_enter(self, ctx: RunContext[Deps, Return]) -> Working:
        await ctx.deps.warehouse.post(
            "/inspections",
            json={"rma": self.rma_id, "callback_task": ctx.task_id},
            headers={"Idempotency-Key": ctx.task_id},
        )
        return Working(
            tool=self.inspected,
            timeout=timedelta(days=21),
            message="Waiting for the warehouse to receive the item.",
        )

    @function_tool
    async def inspected(
        self, ctx: RunContext[Deps, Return], as_described: bool, refund_cents: int
    ) -> Completed[str] | InputRequired:
        """The warehouse's finding, posted by its system."""
        if as_described:
            return Completed(
                f"Your return was received and a refund of {refund_cents / 100:.2f} "
                "is on its way to your original payment method."
            )
        # delivered as an email in the same thread; the reply comes back with the task id
        return InputRequired(
            "The warehouse received your item, but the damage is not what the photos showed, so "
            f"the refund would be {refund_cents / 100:.2f}. Accept that, or have the item sent "
            "back?",
            tool=self.decide,
            timeout=timedelta(days=7),
        )

    @function_tool
    async def decide(self, ctx: RunContext[Deps, Return], accept_partial: bool) -> Completed[str]:
        """Whether the customer accepts the reduced refund."""
        action = "refund" if accept_partial else "ship-back"
        await ctx.deps.warehouse.post(
            f"/inspections/{self.rma_id}/{action}", headers={"Idempotency-Key": ctx.task_id}
        )
        return Completed(
            "Done. The partial refund is on its way."
            if accept_partial
            else "We'll ship the item back to you this week."
        )

    async def on_timeout(self, ctx: RunContext[Deps, Return]) -> Completed[str]:
        return Completed(
            f"We have not received RMA {self.rma_id}. Reply here if you still "
            "want to return it and we'll send a new label."
        )


server = AgentServer(storage=AgentDBStorage())


@server.a2a_session(endpoint="returns", description="Larkfield Outfitters returns.", state=Return)
async def inbox(ctx: A2ASessionContext[Return]) -> str:
    # The email ingress stripped the quoted thread, kept the subject and the attachments, and
    # posted the message with livekit/channel: email. The return value is sent as a reply in the
    # same thread, markdown rendered to HTML.
    deps = Deps(
        shop=httpx.AsyncClient(base_url=SHOP), warehouse=httpx.AsyncClient(base_url=WAREHOUSE)
    )
    result = await support.run(
        ctx.message, message_history=ctx.message_history, state=ctx.state, deps=deps
    )
    return result.output


# What email adds, and what it costs. The ingress keys the thread by "email:<address>", so one
# correspondent is one conversation across threads, like a phone number. Threading is free
# routing: the egress records the Message-ID of every task message it sends, so a reply to the
# inspection question arrives with In-Reply-To and the ingress posts it with the task id. The
# customer's reply reaches Inspection directly, one model call, no handoff by support. What is
# missing: attachments out (a label PDF) need blobs, and a proactive message needs a subject when
# it starts a new thread. Both are in the plan as later.
