"""
- Northside's front desk, with an insurance eligibility expert deployed on its own.
- The voice door is thin: the realtime model's one tool is the front desk's own text door,
  `A2ADelegate(FRONT_DESK_URL)`, and the front desk owns the real ones.
- The `coverage` tool delegates over A2A with `A2ADelegate(url).run(..., progress=ctx.update)`, so
  the caller hears each portal step, and `record_delegation` notes the expert's task on the call.
- The expert's card offers the Live extension, so it writes its own lane in the caller's thread,
  on its own worker. `public=False` closes its endpoint to callers without a project token; the
  delegate mints one because the expert is on the project's own host.
- The text endpoint runs the same `front_desk` agent, so voice and text share one history.
"""

import asyncio

from livekit.agents import Agent, AgentServer, JobContext, LiveSession, RunContext
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.delegation import A2ADelegate, record_delegation
from livekit.agents.storage import AgentDBStorage
from livekit.plugins import openai

PAYERS = {"bluecross": "Blue Cross", "aetna": "Aetna", "medicare": "Medicare"}
EXPERT_URL = "https://northside.livekit.cloud/agents/eligibility/production/eligibility"
FRONT_DESK_URL = "https://northside.livekit.cloud/agents/clinic/production/front-desk"


async def portal_sign_in(payer: str) -> str:
    """Stands in for a browser session on a slow portal."""
    await asyncio.sleep(40)
    return f"{payer}-session"


async def portal_benefits(portal: str, member_id: str, visit_type: str) -> dict:
    await asyncio.sleep(20)
    return {"active": True, "copay": 30, "deductible_left": 450}


eligibility = Agent(
    instructions="Verify insurance eligibility and benefits with the payer. Report each step.",
    llm="google/gemma-4-31b-it",
)
front_desk = Agent(
    instructions="Northside Family Clinic front desk. Appointments, insurance, directions.",
    llm="google/gemma-4-31b-it",
)
voice = Agent(
    instructions="You are the clinic's voice on the phone. Hand every request to the front desk.",
    tools=[A2ADelegate(FRONT_DESK_URL)],  # one tool, ask_front_desk, described by its card
)


@eligibility.tool
async def check_eligibility(
    ctx: RunContext[None, None], payer: str, member_id: str, visit_type: str
) -> str:
    if payer not in PAYERS:
        return f"unknown payer; we take {', '.join(PAYERS.values())}"
    # the caller hears each update while waiting
    await ctx.update(f"Signing in to the {PAYERS[payer]} portal.")
    portal = await portal_sign_in(payer)
    await ctx.update(f"Reading benefits for a {visit_type} visit.")
    b = await portal_benefits(portal, member_id, visit_type)
    if not b["active"]:
        return "coverage is not active"
    return f"active; copay {b['copay']}, deductible remaining {b['deductible_left']}"


@front_desk.tool
async def coverage(ctx: RunContext[None, None], payer: str, member_id: str, visit_type: str) -> str:
    """Ask the eligibility expert whether a visit is covered. Its progress reaches the caller."""
    request = f"{payer} member {member_id}, {visit_type} visit"
    expert = A2ADelegate(EXPERT_URL)
    try:
        result = await expert.run(request, progress=ctx.update)
    finally:
        await expert.aclose()
    record_delegation(ctx.function_call, result)
    return result.output


server = AgentServer(storage=AgentDBStorage())


@server.rtc_session()
async def phone(ctx: JobContext) -> None:
    # the voice door is thin: the realtime model owns the floor, the front desk owns the tools
    session = LiveSession(llm=openai.realtime.RealtimeModel(), thread=ctx.thread)
    await session.start(agent=voice, room=ctx.room)


@server.a2a_session(endpoint="front-desk", description="Northside Family Clinic.")
async def front_desk_text(ctx: A2ASessionContext) -> str:
    result = await front_desk.run(ctx.message, message_history=ctx.message_history)
    return result.output


# The expert's own deployment. Shown here for the reader; in practice this is another server,
# with the portal tooling it needs, scaled on its own, maybe owned by another team. Were it in
# this server for real, coverage would call eligibility.run(...) directly, in-process.
@server.a2a_session(endpoint="eligibility", description="Payer eligibility checks.", public=False)
async def eligibility_text(ctx: A2ASessionContext) -> str:
    result = await eligibility.run(ctx.message, message_history=ctx.message_history)
    return result.output


# One brain, two doors: the call and the text line both run front_desk, one history. The
# realtime model gets one tool, the front desk's door; the front desk gets its own tools. Its coverage tool
# delegates again, over A2A, to the expert in another deployment: another worker, its own lane in
# the patient's thread, progress frames forwarded so the caller hears "signing in to Blue Cross".
# "Actually it's a specialist visit" reaches the same expert. Same shape for prior authorization.
