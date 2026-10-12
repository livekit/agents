"""
- Around a sales agent: the CRM calling a new web lead within minutes, and a rep resuming the
  thread.
- The backend creates the thread, keeps its id against the lead, and dispatches the call with
  `thread_id=`, so the call and the conversation after it are one.
- `thread.submit(..., delay=)` starts a follow-up task with no conversation and no model;
  `thread.background_tasks()` lists and `thread.cancel()` cancels.
- `AgentClient` is thin over the stock a2a-sdk: connect, send, cancel, aclose. Resuming the thread
  later is a send with the thread id as its `contextId`.
- `sales` is the agent behind the `sales` endpoint; `FollowUp` is the pure code task the backend
  submits.
"""

from dataclasses import dataclass
from datetime import timedelta

import httpx

from livekit import api
from livekit.agents import Agent, AgentServer, AgentTask, Completed, RunContext
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
    deps_type=Deps,
)


@dataclass
class FollowUp(AgentTask[str | None]):  # pure code: no model; submitted by the backend two days out
    lead_id: str

    def __post_init__(self) -> None:
        super().__init__(deps=Deps(crm=httpx.AsyncClient(base_url=CRM)))

    async def on_enter(self, ctx: RunContext[Deps, None]) -> Completed[str | None]:
        r = await ctx.deps.crm.get(f"/leads/{self.lead_id}")
        if r.json()["status"] == "new":  # nobody reached them: a nudge, the task's last message
            return Completed("Still interested in a solar quote? Reply here and we'll call.")
        return Completed(None)


@server.a2a_session(endpoint="sales", description="Brightpath Solar sales.")
async def text(ctx: A2ASessionContext) -> str:
    deps = Deps(crm=httpx.AsyncClient(base_url=CRM))
    result = await sales.run(ctx.message, message_history=ctx.message_history, deps=deps)
    return result.output


# The CRM's backend, with the project credential. A web lead just came in: call within minutes,
# and make the call and the texts one conversation.
async def on_new_lead(lk: api.LiveKitAPI, lead: Lead) -> None:
    sess = await lk.threads.create()  # a handle
    async with httpx.AsyncClient(base_url=CRM) as crm:  # the backend keeps the id against the lead
        await crm.patch(f"/leads/{lead.id}", json={"thread_id": sess.id})

    room = f"lead-{lead.id}"
    dispatch = api.CreateAgentDispatchRequest(agent_name=AGENT, room=room, thread_id=sess.id)
    await lk.agent_dispatch.create_dispatch(dispatch)  # the job carries the thread id
    call = api.CreateSIPParticipantRequest(
        room_name=room, sip_trunk_id=TRUNK, sip_call_to=lead.phone
    )
    await lk.sip.create_sip_participant(call)

    # a task started by the backend, no conversation and no model involved: message/send carrying
    # the task under the agent session extension, as livekit/task {class, args}
    tasks = await sess.background_tasks()  # A2A Task objects; the class rides in the metadata
    for t in tasks:
        if t.status.state == "submitted" and t.metadata["livekit/task"]["class"] == "FollowUp":
            await sess.cancel(t.id)  # a leftover from a previous attempt
    await sess.submit(agent=AGENT, task=FollowUp(lead_id=lead.id), delay=timedelta(days=2))


# A sales rep's console. Thin over the stock a2a-sdk; any A2A client works without it.
async def rep_console(token: str, thread_id: str) -> None:
    from livekit.agents.a2a import AgentClient

    url = "https://brightpath.livekit.cloud/agents/sales/production/sales"
    client = AgentClient(url, token=token)
    question = "Where are we with this lead? Budget, timeline, roof type."
    try:
        reply = await client.send(question, thread_id=thread_id)
        async for item in reply:  # live
            print(item)
        # later, worker or no worker: a resume is a send with the same contextId
        reply = await client.send("Any reply from the lead since?", thread_id=thread_id)
        async for item in reply:
            print(item)
    finally:
        await client.aclose()


# The card, public by default and advertising the agent session extension:
# https://brightpath.livekit.cloud/agents/sales/production/sales/.well-known/agent-card.json
# This is a warm, opted-in web lead; cold outbound is not what this shape is for.
