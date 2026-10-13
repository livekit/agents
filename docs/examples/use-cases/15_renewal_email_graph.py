"""
- Harbor Mutual policy renewal emails: a writer drafts, a compliance reviewer approves or sends it
  back, at most three rounds.
- A graph: `WriteEmail` and `ReviewEmail` are `BaseNode` dataclasses, the
  return annotation of `run()` is the edge, `End(email)` is the typed result.
- `Renewal.run()` runs the graph inline to `End` inside one turn; nothing of it is stored, and a
  retry of the turn starts it over.
- An internal endpoint the CRM calls runs the graph inside one turn and returns the approved email
  for a person to send.
- `mermaid_code()` draws the graph before it runs; its output is the comment at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from livekit.agents import Agent, AgentServer
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.graph import BaseNode, End, Graph, GraphRunContext
from livekit.agents.storage import AgentDBStorage


class Policy(BaseModel):
    id: str
    holder: str
    product: str
    renews_on: str
    premium_change_pct: float


class Email(BaseModel):
    subject: str
    body: str


class Review(BaseModel):
    approved: bool
    feedback: str


@dataclass
class Draft:
    """The graph's state, for this run: the policy and every draft so far."""

    policy: Policy
    drafts: list[Email] = field(default_factory=list)


writer = Agent(
    instructions="Write policy renewal emails for Harbor Mutual. Plain, warm, under 150 words.",
    llm="google/gemma-4-31b-it",
    output_type=Email,
)
reviewer = Agent(
    instructions=(
        "Compliance review of a renewal email. Approve only if it states the premium change in "
        "numbers, promises nothing about future rates, and says how to opt out."
    ),
    llm="google/gemma-4-31b-it",
    output_type=Review,
)


# livekit.agents.graph: a node is a dataclass whose fields are its
# inputs, run() returns the next node or End, and the return annotation is the edge the diagram
# is drawn from.


@dataclass
class WriteEmail(BaseNode[Draft]):
    feedback: str | None = None

    async def run(self, ctx: GraphRunContext[Draft]) -> ReviewEmail:
        prompt = f"Policy: {ctx.state.policy.model_dump_json()}"
        if self.feedback:
            prompt += f"\nPrevious draft: {ctx.state.drafts[-1].model_dump_json()}"
            prompt += f"\nFix this: {self.feedback}"
        result = await writer.run(prompt)
        ctx.state.drafts.append(result.output)
        return ReviewEmail(result.output)


@dataclass
class ReviewEmail(BaseNode[Draft, None, Email]):
    email: Email

    async def run(self, ctx: GraphRunContext[Draft]) -> WriteEmail | End[Email]:
        prompt = f"Policy: {ctx.state.policy.model_dump_json()}"
        prompt += f"\nEmail: {self.email.model_dump_json()}"
        review = (await reviewer.run(prompt)).output
        if review.approved or len(ctx.state.drafts) >= 3:   # three rounds, then a human reads it
            return End(self.email)
        return WriteEmail(feedback=review.feedback)


class Renewal(Graph[Email, Draft]):
    nodes = [WriteEmail, ReviewEmail]


server = AgentServer(storage=AgentDBStorage())


@server.a2a_session(endpoint="renewals", description="Drafts policy renewal emails.", public=False)
async def draft(ctx: A2ASessionContext) -> Email:
    """Called by the CRM with a policy. The reply is the approved email, for a person to send."""
    policy = Policy(**ctx.message.data)
    result = await Renewal.run(WriteEmail(), state=Draft(policy))   # seconds; inline
    return result.output


# Renewal.mermaid_code(start_node=WriteEmail):
#
#   stateDiagram-v2
#     [*] --> WriteEmail
#     WriteEmail --> ReviewEmail
#     ReviewEmail --> WriteEmail
#     ReviewEmail --> [*]
#
# A graph is for code steps whose edges are known before anything runs: agents called as
# functions, branches and loops in code, a typed End. Run inline, as here, it finishes in the
# turn; awaited from a tool, a node can wait on the person (docs/graphs.md). It is not the
# conversation, whose next step the model picks.
