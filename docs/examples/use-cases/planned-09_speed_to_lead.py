"""An opted-in CRM lead is called with an explicit conversation thread; sales representatives can resume it."""

from dataclasses import dataclass

import httpx

from livekit import api
from livekit.agents import Agent, AgentServer, RunResult
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.storage import AgentDBStorage

CRM = "https://crm.brightpath-solar.example"
AGENT = "sales"
TRUNK = "ST_outbound"
server = AgentServer(storage=AgentDBStorage())


@dataclass
class Lead:
    id: str
    phone: str


@dataclass
class Deps:
    crm: httpx.AsyncClient


sales = Agent(
    instructions="Brightpath Solar sales. Qualify the lead: budget, timeline, roof type.",
    llm="google/gemma-4-31b-it",
)


@server.a2a_session(endpoint="sales", description="Brightpath Solar sales.")
async def text(ctx: A2ASessionContext) -> RunResult[str]:
    deps = Deps(crm=httpx.AsyncClient(base_url=CRM))
    result = await sales.run(ctx.message, message_history=ctx.message_history, context=deps)
    return result


async def on_new_lead(lk: api.LiveKitAPI, lead: Lead) -> None:
    sess = await lk.threads.create()
    async with httpx.AsyncClient(base_url=CRM) as crm:
        await crm.patch(f"/leads/{lead.id}", json={"thread_id": sess.id})
    room = f"lead-{lead.id}"
    dispatch = api.CreateAgentDispatchRequest(agent_name=AGENT, room=room, thread_id=sess.id)
    await lk.agent_dispatch.create_dispatch(dispatch)
    call = api.CreateSIPParticipantRequest(
        room_name=room, sip_trunk_id=TRUNK, sip_call_to=lead.phone
    )
    await lk.sip.create_sip_participant(call)


async def rep_console(token: str, thread_id: str) -> None:
    from livekit.agents.a2a import AgentClient

    url = "https://brightpath.livekit.cloud/agents/sales/production/sales"
    client = AgentClient(url, token=token)
    question = "Where are we with this lead? Budget, timeline, roof type."
    try:
        reply = await client.send(question, thread_id=thread_id)
        async for item in reply:
            print(item)
        reply = await client.send("Any reply from the lead since?", thread_id=thread_id)
        async for item in reply:
            print(item)
    finally:
        await client.aclose()
