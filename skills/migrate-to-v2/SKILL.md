---
name: migrate-to-v2
description: Migrate a LiveKit Agents 1.x agent (AgentSession, Agent subclasses with node overrides, @function_tool, session.userdata, AgentTask, TaskGroup, prebuilt workflows, handoffs, session.say/generate_reply) to Agents 2.0, the fast brain / slow brain pattern - a frontend that talks to the person by voice and text, delegating over A2A with RemoteAgent(...).as_tool() to a backend that holds the tools, data and business rules. Use when the user asks to migrate, port, upgrade or convert a livekit-agents 1.x agent to 2.0.
---

# Migrate a LiveKit Agents 1.x agent to 2.0

The docs in `docs/` are the API; this skill is only the 1.x → 2.0 mapping. Read
[docs/README.md](../../docs/README.md) first, then the page linked from each section.

## What changes

- **`Agent` is a constructed instance**, built once at import and shared by every
  conversation: `Agent(instructions=, llm=, context_type=, output_type=, tools=)`, with tools
  and dynamic instructions registered as `@agent.tool` / `@agent.instructions`. No subclasses,
  nothing per-conversation on it. See [agents.md](../../docs/agents.md).
- **`userdata` becomes application context**: a typed object passed per run or per call
  (`run(context=)`, `VoiceSession(context=)`) and read as `ctx.context`. Only fields marked
  `persisted(...)` are stored, and only when you call `save_context`. See
  [context.md](../../docs/context.md).
- **Fast brain / slow brain.** The frontend (`frontend_agent.py`) owns the conversation:
  persona, pacing, voice pipeline, greeting. Its one business tool is the backend,
  `RemoteAgent(url).as_tool()`. The backend (`backend_agent.py`) owns every business tool, the
  data and the rules, and never speaks to the person. See
  [voice-session.md, the thin voice door](../../docs/voice-session.md#the-thin-voice-door).
- **Doors on a handler.** Each module declares `AgentHandler` doors (`rtc_session()` for a
  call, `a2a_session(endpoint=)` for text); `main.py` serves them with
  `AgentServer.add(...)`. See [server.md](../../docs/server.md).
- **`AgentSession` is `VoiceSession`**, and the reply pipeline (nodes, turn hooks) is
  registered on the session, not overridden on the agent.

## Procedure

1. **Inventory** the 1.x agent in `MIGRATION.md` before writing code: every tool (what it
   reads, writes, asks, confirms), agent and handoff, `AgentTask`/`TaskGroup`/prebuilt
   workflow, `userdata` field, node override and session hook, `say`/`generate_reply`, the
   greeting, every voice feature (background audio, turn detection, interruptions, DTMF,
   hang-up, transfer, idle nudges) and every test.
2. **Assign each row to a side** (table below). Keep the domain modules (database, API
   clients, menus, policies) unchanged; split a mixed instruction constant into persona
   (frontend) and business rules (backend).
3. Write `backend_agent.py`, then `frontend_agent.py`, then `main.py` (templates below).
4. **Verify**, and fill the inventory's last column: where each behavior lives now, or why
   it was dropped. A behavior silently lost is a failed migration.

## Deciding the split

| 1.x thing | Goes to |
| --- | --- |
| business tools: lookups, bookings, orders, payments, anything touching data | backend `@backend.tool` |
| `userdata` fields the tools read or write | backend context: `persisted(...)` fields for data, plain fields for services |
| guided multi-turn flows (`AgentTask`, `TaskGroup`) over business data | backend: a persisted draft, one `record_*` tool per step, a status line from `@backend.instructions` |
| specialist agents with handoffs | backend: one agent with tools; a specialist with a typed result becomes `specialist.as_tool()` under a `TextSession` |
| business rules, policies, prices, constraints | backend instructions and tool docstrings |
| persona, greeting, small talk, pacing, how to say numbers and times | frontend instructions |
| STT/LLM/TTS/VAD/realtime model, turn handling, background audio, room options, node overrides, idle nudges | frontend `phone()` door, on the `VoiceSession` |
| tools that act on the call: end call, warm transfer, DTMF, collecting an email on the call | frontend voice agent, next to the backend tool |

If it needs to be right, it goes to the backend; if it needs to sound right, the frontend.
The frontend holds no business state and never confirms anything the backend did not return.

## API mapping

| 1.x | 2.0 |
| --- | --- |
| `AgentSession(stt=, llm=, tts=, vad=, ...)` | `VoiceSession(stt=, llm=, tts=, vad=, context=, thread=ctx.thread)`; model strings work: `stt="deepgram/nova-3:en"` ([models.md](../../docs/models.md#model-strings)) |
| `AgentSession(turn_detection=, min_endpointing_delay=, preemptive_generation=, ...)` | `VoiceSession(turn_detection=, endpointing_opts={"min_delay": ..., "max_delay": ...})` or `turn_handling={...}`; `session.update_options(...)` mid-call |
| `AgentSession[Userdata](userdata=Userdata(...))` | `VoiceSession(context=app)`, a `VoiceSession[AppContext]` |
| `ctx.userdata`, `RunContext[Userdata]`, `session.userdata` | `ctx.context`, `RunContext[AppContext]`, `session.context` |
| `class MyAgent(Agent)` + `@function_tool` methods | `my_agent = Agent(context_type=AppContext, instructions=..., llm=...)` + `@my_agent.tool` functions; `@function_tool` stays for reusable tools and `Toolset` methods |
| `Agent` type parameters by hand | inferred from `context_type=` and `output_type=`: `Agent[AppContext, Booking]`; no `output_type` means `str` |
| `self.update_instructions(...)`, `update_tools(...)`, `update_chat_ctx(...)` | gone. Render state with `@agent.instructions`; refuse with `ToolError` instead of swapping tools; history goes through `session.start(chat_ctx=)` / `update_agent(chat_ctx=)` |
| `async def on_enter(self)` / `on_exit(self)` | `@agent.on_enter` / `@agent.on_exit`, an `async def f(session: VoiceSession)` |
| `llm_node`, `stt_node`, `tts_node`, `transcription_node`, `on_user_turn_completed`, `on_user_turn_exceeded` overridden on an `Agent` | `@session.<name>` on the `VoiceSession` before `start()`, for every agent of the call; branch on `ctx.agent`, call `session.default.<name>(...)` for the built-in step ([customizing the pipeline](../../docs/voice-session.md#customizing-the-pipeline)) |
| `self.session`, `context.session` in a tool | `ctx.live_session` (`None` outside a call) |
| a tool that returns `NewAgent()` (handoff) | frontend: `ctx.live_session.update_agent(other_agent)` with a module-level agent; backend: one agent, or `specialist.as_tool()` |
| `class T(AgentTask[R])` + `self.complete(r)` on the call | `VoiceTask(agent=collector).run()` from a frontend tool; the collector has `output_type=R` and a tool returns `Completed(r)`, or `Failed(reason)` which raises `VoiceTaskFailed` ([voice-tasks.md](../../docs/voice-tasks.md)) |
| `AgentTask`/`TaskGroup` over business data | backend tools on a persisted draft (see the hotel example), or an agent with `output_type` as a tool under `TextSession` ([text-session.md](../../docs/text-session.md)) |
| `beta.workflows.GetEmailTask` | `GetEmailTask().run()` in a frontend tool, still in `livekit.agents.beta.workflows` |
| `GetNameTask`, `GetPhoneNumberTask`, `GetAddressTask`, `GetCreditCardTask`, ... | a custom `VoiceTask` whose agent's `output_type` is your own result type (a dataclass or pydantic model of the fields you collect), or a backend `record_*` tool when the value is business data |
| `WarmTransferTask(target_phone_number=)` | `await WarmTransfer(sip_call_to=...).run()` in a frontend tool (`livekit.agents.beta.workflows`) |
| `EndCallTool()`, `send_dtmf_events` | unchanged, from `livekit.agents.beta`, on the frontend voice agent |
| confirm before a write (prompt rule, `approved: bool` argument) | `@backend.tool(needs_approval=True)` or `needs_approval=read_back` ([tools.md, approval](../../docs/tools.md#approval)) |
| a second LLM call inside a tool | a module-level `Agent(output_type=...)` and `await judge.run(prompt)`; its `result.usage` is its own (usage.md) |
| `ChatContext` passed around to carry history | the thread carries it; a text door gets `ctx.message_history` |
| `@server.rtc_session()` + `ctx.connect()` + `session.start(agent=MyAgent(), room=ctx.room)` | `@front_desk.rtc_session()` on an `AgentHandler`, `await session.start(agent=voice, room=ctx.room)`; no `ctx.connect()` |
| `@server.rtc_session(on_session_end=, on_simulation_end=)` | the same keywords on the frontend's `rtc_session(...)` |
| `python agent.py dev\|start\|console` | `lk agent dev\|start\|console`, which drives `cli.run_app(server)`; a container runs `python main.py start` |

### userdata → context

```python
# 1.x
@dataclass
class Userdata:
    db: Database
    customer_name: str | None = None

session = AgentSession[Userdata](userdata=Userdata(db=db), llm=..., stt=..., tts=...)

@function_tool
async def set_name(context: RunContext[Userdata], name: str) -> str:
    context.userdata.customer_name = name
    return "recorded"
```

```python
# 2.0 (backend)
class Customer(BaseModel):
    name: str | None = None

@dataclass
class AppContext:
    db: Database                                                   # a service, never stored
    customer: Customer = persisted(default_factory=Customer)       # stored when you save

backend = Agent(context_type=AppContext, instructions=..., llm="openai/gpt-4.1-mini")

@backend.tool
async def set_name(ctx: RunContext[AppContext], name: str) -> str:
    ctx.context.customer.name = name
    return "recorded"
```

The door constructs the context, `await ctx.store.load_context(app)` before the run and
`await ctx.store.save_context(app)` after it (also after an `input_required` result). Nothing
saves implicitly. Data about one conversation goes in persisted fields; data that outlives it
goes in a real store.

### Node and turn hooks → session callbacks

```python
# 1.x
class Billing(Agent):
    async def tts_node(self, text, model_settings):
        return Agent.default.tts_node(self, spell_out_amounts(text), model_settings)

    async def on_user_turn_completed(self, turn_ctx, new_message):
        if new_message.text_content == "hold on":
            raise StopResponse()
```

```python
# 2.0, in the phone() door
session = VoiceSession(stt=..., llm=..., tts=..., context=app, thread=ctx.thread)

@session.tts_node
def spell_amounts(ctx: RunContext[AppContext], text: AsyncIterable[str], model_settings: ModelSettings):
    if ctx.agent is billing:
        text = spell_out_amounts(text)
    return session.default.tts_node(ctx, text, model_settings)

@session.on_user_turn_completed
async def hold_on(ctx: RunContext[AppContext], turn_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
    if (new_message.text_content or "").strip().lower() == "hold on":
        raise StopResponse()

await session.start(agent=voice_agent, room=ctx.room)
```

One callback per step per session; registering twice or after `start()` raises. `agent.run()`
and `TextSession` run none of them.

## Backend template

```python
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from livekit.agents import Agent, AgentHandler, InputRequired, RunContext, RunResult, ToolError, persisted
from livekit.agents.a2a import A2ASessionContext

from shop import OutOfStock, Shop  # the 1.x domain module, unchanged


class Customer(BaseModel):
    name: str | None = None
    order_ids: list[str] = []


@dataclass
class AppContext:
    shop: Shop
    customer: Customer = persisted(default_factory=Customer)


backend = Agent(
    context_type=AppContext,
    instructions=(
        "You are the back office of ... What you say and ask reaches the customer through the "
        "front desk. Answer from your tools in one or two plain sentences, and report only what "
        "a tool did. A write reads the details back and asks; a yes means calling the same tool "
        "again exactly the same way. <business rules from 1.x; tool rules go in docstrings>"
    ),
    llm="openai/gpt-4.1-mini",  # keep the 1.x model
)


@backend.instructions
def known(ctx: RunContext[AppContext]) -> str:
    name = ctx.context.customer.name
    return f"The customer is {name}." if name else "Ask for the customer's name before ordering."


async def read_back(ctx: RunContext[AppContext], item_id: str) -> str | InputRequired:
    item = ctx.context.shop.find(item_id)
    if item is None:
        raise ToolError("no such item; list the items again")
    if ctx.context.customer.name is None:
        return InputRequired("What name should the order be under?")
    return f"Please confirm: one {item.name} for ${item.price}, under {ctx.context.customer.name}. Yes or no?"


@backend.tool(needs_approval=read_back)
async def place_order(ctx: RunContext[AppContext], item_id: str) -> str:
    """Order an item.

    Args:
        item_id: the item's id, as list_items returned it
    """
    item, name = ctx.context.shop.find(item_id), ctx.context.customer.name
    if item is None or name is None:  # the read-back checked, but time passed
        raise ToolError("the order changed; read it back again")
    try:
        order_id = await ctx.context.shop.order(item, name)
    except OutOfStock:
        raise ToolError("that item just sold out; offer another") from None
    ctx.context.customer.order_ids.append(order_id)
    return f"ordered {item.name}, order {order_id}"


shop = Shop()  # one per worker process, shared by every thread
back_office = AgentHandler(lane="backend")  # its own lane in the customer's thread


@back_office.a2a_session(endpoint="backend", description="The shop's back office: items and orders.", inbox="queue")
async def text(ctx: A2ASessionContext) -> RunResult[str]:
    app = AppContext(shop=shop)
    await ctx.store.load_context(app)
    result = await backend.run(ctx.message, message_history=ctx.message_history, context=app)
    await ctx.store.save_context(app)
    return result  # the door maps completed / input_required / no_output to A2A
```

## Frontend template

```python
import os

from livekit.agents import Agent, AgentHandler, JobContext, RunResult, VoiceSession
from livekit.agents.a2a import A2ASessionContext, RemoteAgent
from livekit.agents.beta import EndCallTool

# /agents/<AgentServer agent_name>/<deployment>/<the backend door's endpoint>
BACKEND = os.environ["LIVEKIT_URL"].replace("ws", "http", 1) + "/agents/shop/default/backend"

PERSONA = (
    "You are the front desk of ... <persona from 1.x>. You know nothing about the business "
    "yourself: every request about it goes to the back office at once, in full, with what the "
    "customer said. Tell the customer what it answered, in your own words; never confirm "
    "anything it did not return. Only pleasantries and the conversation itself are yours."
)

back_office = RemoteAgent(BACKEND)  # one tool, named and described from the backend's card

chat = Agent(
    instructions=PERSONA + " By text: one or two plain sentences, no lists.",
    llm="openai/gpt-4.1-mini",
    tools=[back_office.as_tool()],
)

voice = Agent(
    instructions=PERSONA + " On the phone: one short sentence at a time, one question per turn.",
    tools=[back_office.as_tool(), EndCallTool()],  # EndCallTool only if 1.x could hang up
)

front_desk = AgentHandler()  # the worker's own lane: the phone and the text door share it


@front_desk.rtc_session()
async def phone(ctx: JobContext) -> None:
    session = VoiceSession(stt="deepgram/nova-3:en", llm="openai/gpt-4.1-mini", tts="cartesia/sonic-3", thread=ctx.thread)
    await session.start(agent=voice, room=ctx.room)
    if not session.history.items:  # only if 1.x greeted
        await session.generate_reply(instructions="Greet the customer in one sentence.")


@front_desk.a2a_session(endpoint="frontend", description="The shop's front desk.")
async def text(ctx: A2ASessionContext) -> RunResult[str]:
    return await chat.run(ctx.message, message_history=ctx.message_history)
```

## Entrypoint (`main.py`)

```python
import backend_agent
import frontend_agent

from livekit.agents import AgentServer, cli
from livekit.agents.storage import AgentDBStorage

server = AgentServer(storage=AgentDBStorage(), agent_name="shop")
server.add(frontend_agent.front_desk, backend_agent.back_office)

if __name__ == "__main__":
    cli.run_app(server)
```

Run it with `lk agent dev` (or `lk agent console`). The examples in `examples/` (frontdesk,
hotel-receptionist, airline-support) are complete migrations of this shape.

## Results

`agent.run()` returns a `RunResult` with three statuses: `completed` (`result.output`, typed
by `output_type`), `input_required` (`result.question`) and `no_output` (a tool raised
`StopResponse`, returned `ToolResult(reply_required=False)`, or the turn was interrupted).
Narrow on `result.status` before reading `output`; a door returns the whole `RunResult`. See
[agents.md, running a turn](../../docs/agents.md#running-a-turn).

## Speaking

The backend never speaks: it returns text and the frontend words it
([tools.md, wording and verbatim](../../docs/tools.md#wording-and-verbatim)).

| 1.x, in a tool or task | 2.0 backend |
| --- | --- |
| `session.say("Let me check...")` | `await ctx.update("Checking the calendar.")` |
| `session.say(result)` | return it from the tool |
| `session.say(question)`, `generate_reply(instructions="ask for X")` | `return InputRequired("What is your email?")` |
| text that must be said exactly (a disclaimer, a code) | `InputRequired(q, verbatim=True)` or `ToolResult(text, verbatim=True)` |
| greeting, idle nudge, hold message | frontend: `session.say()` / `generate_reply()` in `phone()`, or `ctx.live_session` in a frontend tool |

Write questions and results to the person, in the second person, with no internal ids.

## Approval and tools

- Give `needs_approval=` to every write that moves money or cannot be undone, even where 1.x
  trusted the model. The body runs only on a repeat of the same call after the person answered
  the read-back; there is no `approved` argument for the model to set.
- A read-back function takes the tool's own parameters; it may `raise ToolError` to refuse or
  return `InputRequired` for a missing detail. Amounts come from the same domain call the
  write makes. The body re-checks what the read-back checked.
- Put "a yes means calling the tool again exactly the same way" in the agent's instructions,
  not only the docstring.
- Keep 1.x docstring hints on argument formats; raise `ToolError` (or `ModelRetry` to make the
  model correct the call) and name the next step in the message.
- Guard prerequisites (verification) in the tool body or read-back, never by instructions alone.

## Tests

- `async with AgentSession(...) as s: await s.start(MyAgent()); r = await s.run(user_input=...)`
  becomes `r = await agent.run("...", context=fake_app)`; `r.expect...`, `.judge(llm, intent=)`
  and `mock_tools(agent, {...})` take the agent instance
  ([testing.md](../../docs/testing.md#programmatic-tests)).
- Multi-turn flows: pass `message_history=r.all_messages()` to the next run, or drive the door
  with `AgentClient` against a running worker ([clients.md](../../docs/clients.md#a-client-for-an-endpoint)).
- Keep pure domain tests unchanged. A test module in this repo declares its category
  (`pytestmark = pytest.mark.unit`).

## Verify

1. `python -c "import main"` with `LIVEKIT_URL`, `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`
   set (dummy values are enough).
2. `mypy --strict` and `ruff check --line-length 100` on every new module.
3. Walk the inventory: every 1.x tool, flow, rule, voice feature and test has a row.
4. Reread the frontend instructions: no business facts or tools, nothing it could confirm on
   its own.
5. If a LiveKit project is available, `lk agent dev` and hold one text conversation through
   every flow, including a backend question and an approval; check the write happened.

## Pitfalls

- Business tools on the frontend "for speed".
- `Agent(...)` built inside a door: agents are module-level and shared.
- A persisted field mutated but never saved, or `save_context` in `finally`.
- Reading `result.output` without checking `result.status`.
- Node overrides left on an `Agent` subclass: they no longer run; register them on the session.
- A `VoiceTask` on the backend: it needs a live call; backend flows are tools or `TextSession`.
