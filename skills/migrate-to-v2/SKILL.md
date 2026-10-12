---
name: migrate-to-v2
description: Migrate a LiveKit Agents v1 agent (AgentSession, Agent subclasses, @function_tool, session.userdata, AgentTask, TaskGroup, prebuilt workflows, handoffs, session.say/generate_reply) to Agents 2.0, the fast brain / slow brain pattern - a frontend that talks to the person by voice and text, delegating over A2A to a backend that holds the tools, state and business logic. Use when the user asks to migrate, port, upgrade or convert a livekit-agents v1 agent to v2 / 2.0.
---

# Migrate a LiveKit Agents v1 agent to 2.0

## What changes

v1 runs one `AgentSession` per room: the call, the model, the tools and the history live in one
process and die with the room. 2.0 splits every agent in two and keeps the conversation in a
durable **thread**:

- **Fast brain, the frontend** (`frontend_agent.py`): owns the conversation. Persona, tone,
  pacing, the voice pipeline, the greeting. It knows nothing about the business. Its one
  business tool is the backend, reached with `A2ADelegate`. It has a voice door (`rtc_session`)
  and a text door (`a2a_session`).
- **Slow brain, the backend** (`backend_agent.py`): owns the work. Every business tool, the data
  access, the rules, the typed state, the multi-step flows. It never speaks to the person: what
  it returns and the questions it asks go through the frontend.
- **Thread**: the durable conversation. Each agent keeps its own lane in it (history, typed
  state, agent stack). The framework loads the lane before every turn and saves it after, so
  state survives calls, texts, days and worker restarts.
- **`Agent` is configuration**, built once at module level and shared by every conversation.
  `await agent.run(message, message_history=, state=, deps=)` is one turn. Nothing
  per-conversation lives on the agent object.
- **One worker serves both brains** (`main.py`): each module defines its doors on an
  `AgentHandler`, and `AgentServer.add(...)` serves them.

## Procedure

1. **Inventory the v1 agent** in a table, in `MIGRATION.md`, before writing code: every tool
   (what it reads, writes, asks, confirms), every agent and handoff, every
   `AgentTask`/`TaskGroup`/prebuilt workflow, every `userdata` field, every runtime instruction
   or tool change, every `say`/`generate_reply`, the greeting, every voice feature (background
   audio, turn detection, interruptions, DTMF, hang-up, transfer, idle nudges, session hooks),
   and every test. This table is the checklist you verify against at the end.
2. **Assign each row to a side** (see "Deciding the split").
3. **Keep the domain modules** (database, fake data, API clients, menus, policies). Only the
   agent files change. If a v1 instruction constant mixes persona and business rules, split it
   section by section into the two agent files.
4. **Write `backend_agent.py`**, then **`frontend_agent.py`**, then **`main.py`**, from the
   templates.
5. **Verify** (see "Verify") and fill in the last column of the inventory: where each behavior
   lives now, or why it was dropped. A behavior silently lost is a failed migration.

## Deciding the split

| v1 thing | Goes to |
| --- | --- |
| business tools: lookups, bookings, orders, payments, anything touching data | backend `@backend.tool` |
| `userdata` fields the tools read or write | backend `state_type=` (saved) or `deps_type=` (services, per run) |
| guided multi-turn flows (`AgentTask`, `TaskGroup`, prebuilt `GetEmailTask`, ...) | backend, as `AgentTask` dataclasses awaited from a tool |
| several specialist agents with handoffs | backend: one routing agent with tools, plus `AgentTask`s for flows; a handoff (a tool returns an `Agent`) only when a specialist must own many turns |
| business rules, policies, prices, menus, constraints | backend instructions |
| persona, brand voice, greeting, small talk, pacing, how to say numbers and times | frontend instructions |
| STT/LLM/TTS/VAD/realtime model, turn handling, background audio, `RoomOptions`, noise cancellation, idle nudges, `on_session_end` hooks | frontend `phone()` door |
| tools that act on the call itself: end call, warm transfer, DTMF, play audio | frontend voice agent, next to the delegate |
| a tool named "transfer" that only writes a record | backend: decide by whether it touches the call or the data |

When the product is the questions (a screening, a survey, an intake), the backend's task asks
them; the frontend's greeting only says hello and hands over at once.

If it needs to be right, it goes to the backend; if it needs to sound right, it goes to the
frontend. The frontend holds no business state and never confirms anything the backend did not
return.

## Backend template

```python
from __future__ import annotations

from pydantic import BaseModel

from livekit.agents import Agent, AgentHandler, InputRequired, RunContext, ToolError
from livekit.agents.a2a import A2ASessionContext

from shop import Item, OutOfStock, Shop  # the v1 domain module, unchanged


class Customer(BaseModel):
    """Saved on the thread after every turn: what the backend learned. JSON only, < 64 KiB,
    every field defaulted (a new thread starts from Customer())."""

    name: str | None = None
    order_ids: list[str] = []


backend = Agent(
    instructions=(
        "You are the back office of ... You never talk to the customer yourself: the front "
        "desk relays what you say and asks your questions. Answer from your tools, in one or two "
        "plain sentences. Nothing is done until a tool says so: report only what a tool did, "
        "and when a detail is missing, ask for that one thing without saying anything was "
        "added. <business rules from v1 here; keep routing short and put the rest in the tools' "
        "docstrings>"
    ),
    llm="openai/gpt-4.1-mini",  # keep the v1 model string
    deps_type=Shop,             # services for one run, never saved
    state_type=Customer,
)


@backend.instructions
def known(ctx: RunContext[Shop, Customer]) -> str:
    """Rendered every turn: what the thread knows, and today's date when dates matter."""
    today = f"Today is {ctx.deps.today():%A, %B %d, %Y}."
    return f"{today} The customer is {ctx.state.name}." if ctx.state.name else today


async def read_back(ctx: RunContext[Shop, Customer], item_id: str) -> str | InputRequired:
    """What the customer approves before the order is written. Same parameters as the tool."""
    item = ctx.deps.find(item_id)
    if item is None:
        raise ToolError("no such item; list the items again")
    if ctx.state.name is None:
        return InputRequired("What name should the order be under?")  # missing detail first
    return f"Please confirm: one {item.name} for ${item.price}, under {ctx.state.name}. Yes or no?"


@backend.tool(requires_approval=read_back)
async def place_order(ctx: RunContext[Shop, Customer], item_id: str) -> str:
    """Order an item. It reads the order back and asks the customer; their answer comes back as
    this call's output, and a yes means calling it again exactly the same way.

    Args:
        item_id: the item's id, as list_items returned it
    """
    item = ctx.deps.find(item_id)
    assert item is not None and ctx.state.name is not None  # the read-back checked both
    try:
        order_id = await ctx.deps.order(item, ctx.state.name)
    except OutOfStock:
        raise ToolError("that item just sold out; offer another") from None
    ctx.state.order_ids.append(order_id)
    return f"ordered {item.name}, order {order_id}"


shop = Shop()  # one per worker, shared by every thread; see "State and deps"

back_office = AgentHandler(lane="backend")  # its own lane in the customer's thread


@back_office.a2a_session(
    endpoint="backend",
    description="The shop's back office: items, orders and returns.",  # the frontend reads this
    public=False,    # only callers holding a project token, i.e. the frontend
    inbox="queue",   # one delegation at a time, in order
    state=Customer,
)
async def text(ctx: A2ASessionContext[Customer]) -> str:
    result = await backend.run(
        ctx.message, message_history=ctx.message_history, state=ctx.state, deps=shop
    )
    return result.input_required or str(result.output)
```

## Frontend template

```python
import os

from livekit.agents import Agent, AgentHandler, JobContext, LiveSession, inference
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.delegation import A2ADelegate

# /agents/<the worker's agent_name>/<deployment>/<the backend's endpoint>
BACKEND = os.environ.get("BACKEND_URL", "http://127.0.0.1:7880/agents/shop/default/backend")

PERSONA = (
    "You are the front desk of ... <persona, tone, brand from v1>. You know nothing about the "
    "business yourself. Hand every request about <domain> to the back office at once, before "
    "asking anything, as the customer's own words in quotes, with only context they gave "
    "earlier: 'Customer: \"two cheeseburgers and a medium fries\"', never 'the customer wants "
    "to order'. "
    "Tell the customer what it answered, in your own words, without adding facts: confirm only "
    "what it says was done, and if it says something is not on offer, say exactly that. Only "
    "pleasantries and the conversation itself are yours."
)

back_office = A2ADelegate(BACKEND)  # one tool, "ask", described from the backend's card

chat = Agent(
    instructions=PERSONA + " By text: one or two plain sentences, no lists.",
    llm="openai/gpt-4.1-mini",
    tools=[back_office],
)

voice = Agent(
    instructions=PERSONA + " On the phone: one short sentence at a time, one question per turn, "
    "plain speech. While the back office works, keep the customer company in one line.",
    tools=[back_office, EndCallTool()],  # EndCallTool only if v1 could hang up; never business tools
)

front_desk = AgentHandler()  # the worker's own lane: the phone and the text door share it


@front_desk.rtc_session()
async def phone(ctx: JobContext) -> None:
    session: LiveSession[None] = LiveSession(
        stt=inference.STT("deepgram/nova-3"),  # the v1 AgentSession's models and voice options
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
        thread=ctx.thread,
    )
    await session.start(agent=voice, room=ctx.room)
    if not session.history.items:  # only if v1 greeted; a text conversation picks up as it was
        await session.generate_reply(instructions="Greet the customer in one sentence.")


@front_desk.a2a_session(endpoint="frontend", description="The shop's front desk.")
async def text(ctx: A2ASessionContext) -> str:
    result = await chat.run(ctx.message, message_history=ctx.message_history)
    return result.input_required or str(result.output)
```

## Entrypoint template (`main.py`)

```python
from dotenv import load_dotenv

load_dotenv()  # before the agent modules: model strings and storage read the env at import

import backend_agent  # noqa: E402
import frontend_agent  # noqa: E402

from livekit.agents import AgentServer, cli  # noqa: E402
from livekit.agents.storage import AgentDBStorage  # noqa: E402

# one worker for both: the front desk on the worker's lane, the back office on its own
server = AgentServer(storage=AgentDBStorage(), agent_name="shop")
server.add(frontend_agent.front_desk, backend_agent.back_office)

if __name__ == "__main__":
    cli.run_app(server)
```

The frontend reaches the backend at `/agents/shop/default/backend`: the worker's `agent_name`,
then the backend door's `endpoint`. Update the Dockerfile to run `python main.py start`, and the
`pyproject.toml` dependency to `livekit-agents>=2.0.0.dev0`.

## API mapping

| v1 | 2.0 |
| --- | --- |
| `AgentSession(stt=, llm=, tts=, vad=, ...)` | `LiveSession(..., thread=ctx.thread)` in the frontend's `phone()` door |
| `AgentSession(preemptive_generation=..., turn_detection=..., min_endpointing_delay=...)` | `turn_handling={"preemptive_generation": {"enabled": True}, ...}` on `LiveSession` (the old keywords are deprecated) |
| `AgentSession(max_tool_steps=N)` | `backend.run(..., max_steps=N + 1)`: it counts model requests, the last one being the answer (default 12) |
| `AgentSession(user_away_timeout=)`, `"user_state_changed"` idle nudges, `ctx.add_shutdown_callback`, `RoomOptions` | unchanged, in the frontend's `phone()` door |
| `Instructions(audio=..., text=...)` per modality | the frontend's two agents: `voice` and `chat`, the persona plus a suffix each |
| `session.userdata`, `RunContext[Userdata]`, `context.userdata` | `state_type=` (saved, JSON) and `deps_type=` (clients, DBs); tools take `RunContext[Deps, State]` |
| `@server.rtc_session()` + `ctx.connect()` + `session.start(agent=MyAgent(), room=ctx.room)` | `@front_desk.rtc_session()` on an `AgentHandler`, `session.start(agent=voice, room=ctx.room)`; no `ctx.connect()` |
| `class MyAgent(Agent)` with `@function_tool` methods | `my_agent = Agent(instructions=..., llm=..., deps_type=, state_type=)` at module level and `@my_agent.tool` functions |
| `self.update_instructions(...)`, `update_tools(...)`, `update_chat_ctx(...)` | gone. Render state with `@backend.instructions`; validate arguments against data and raise `ToolError` instead of swapping tools; put tools that only make sense later inside an `AgentTask` |
| tool schemas built per session with `enum` of valid ids | ids that never change: keep the enum, `ItemId = Annotated[str, Field(json_schema_extra={"enum": ids})]` built at import (give the domain module a synchronous loader if v1's was async), or `Literal[...]`. What changes per conversation (availability, a patient's appointments): list the valid ids in dynamic instructions and check them in the tool against `deps` |
| raw-schema tools (`function_tool(raw_schema=...)`) | unchanged in `tools=[...]`; they cannot take `requires_approval` |
| a tool that returns `NewAgent()` or `NewAgent(), "msg"` | a tool returns a module-level `Agent` alone; the handoff is saved on the lane's stack |
| confirm before a write (prompt rule, or `approved` flag) | `@backend.tool(requires_approval=read_back)`, see "Approval" |
| a second LLM call inside a tool (`llm.chat(chat_ctx=...)`) | a module-level `evaluator = Agent(...)` and `await evaluator.run(prompt, usage=ctx.usage)` |
| `class T(AgentTask[R])` with `self.complete(result)` | `@dataclass class T(AgentTask[R, TState])`, `return Completed(result)` from its tools, see "AgentTask in text" |
| `TaskGroup` / tasks awaited in sequence | one `AgentTask` whose state tracks the steps, one tool per step, `Completed` when all are recorded |
| `beta.workflows.GetEmailTask`, `GetNameTask`, `GetPhoneNumberTask`, `GetCreditCardTask` | a field in the task's state plus a `record_*` tool that validates it |
| `WarmTransferTask(target_phone_number=)` | frontend voice tool; `WarmTransferTask(sip_call_to=..., sip_number=SIP_NUMBER or NOT_GIVEN)` |
| `session.say(...)`, `generate_reply(...)` in a tool or task | see "Speaking" |
| `ChatContext` passed around to carry history | not needed: the thread carries it |

## Speaking: `say` and `generate_reply`

On the backend there is no call: the backend never speaks (`ctx.session` raises outside a
call). It returns text, and the agent facing the person words it.

| v1, in a tool or task | 2.0 on the backend | What the person gets |
| --- | --- | --- |
| `session.say("Let me check...")` (progress) | `await ctx.update("Checking the calendar.")` | the frontend says it in its own words while the backend works |
| `session.say(result)` | return it from the tool | the frontend rephrases it |
| `session.say(question)`, `generate_reply(instructions="ask for X")` | `return InputRequired("What is your email?")` | the frontend asks it in its own words; the answer comes straight back to the backend |
| text that must be said exactly (a disclaimer, a code) | `InputRequired(q, verbatim=True)`, or `ToolResult(text, verbatim=True)` for a statement | exactly as written; `reply_required=False` ends the turn after it, otherwise the model continues |
| `on_enter` → `generate_reply` in a task | the task's model speaks first (its text ends the turn as a question), or `on_enter(self, ctx)` returns an `InputRequired` | asked in the frontend's words |
| greeting, idle nudge, `say()` with audio | frontend `phone()` door: `session.say()` / `session.generate_reply()`, or `ctx.session.say()` in a frontend voice tool | as v1 |
| a backend step that ends the conversation (disqualify, done) | return a `Completed` or `Failed` whose text says so ("the screening is over; end the call") | the frontend's voice agent says goodbye and calls `EndCallTool`; tell it so in its instructions |

Write every question and result **to the person, in the second person** ("What is your email?",
never "the guest's email"): a verbatim one reaches them unchanged, on a call as on a text, so
write read-backs plainly enough to be spoken. Never put internal ids or ISO timestamps in text
meant for the person; a code the person needs (a confirmation number) is fine. Ids listed in
instructions are for tool arguments: tell the model never to say them, and describe records in
words ("the October 15 appointment with Dr. Jekyll").

## Approval

`requires_approval=` is the only way to confirm before a write. Give it to **every write that
moves money or cannot be undone** (a charge, a cancel, a change with a price difference), even
where v1 trusted the model; a model-set `bool` that chooses between outcomes of a write is the
same hole as an `approved` flag. A read-back repeats every detail the person gave (name, email,
dates, amounts), since it is the only time they hear them. The framework holds the
approval: a call returns the read-back as a verbatim question and does not run, and the body
runs only when the model repeats the **same call with the same arguments** after the person
answered. There is no `approved` parameter the model could set, so it cannot skip the ask.

- `requires_approval=True` builds the read-back from the docstring and the arguments.
- `requires_approval=read_back` takes a function with **the tool's own parameters** (the
  context, and `self` for a task method) that returns the read-back text, or an `InputRequired`
  for a detail that is still missing.
- A read-back may `raise ToolError(...)` to refuse ("already cancelled"), or `return await
  VerifyTask()` for a precondition (see "Verify, then act").
- Inside a task whose `record_*` tools collect the details, a read-back with a step missing
  raises `ToolError("choose the doctor first")`: an `InputRequired` there turns the write tool
  into a shortcut for asking, and the person gets a question with no options. Return
  `InputRequired` for a missing detail only from a standalone tool.
- It is for any yes the person must give, a write or a detail to confirm (an email read back).
- Put "a yes means calling the tool again exactly the same way" in the instructions of the
  agent or task that owns the tool, not only in its docstring: a small model ignores a rule
  that lives only in a docstring, and the approved write never happens.
- Changed details mean changed arguments, which read back again.

For a task method, define the read-back as a method above it in the class:

```python
    def _read_back(self, ctx: RunContext[Shop, Customer]) -> str:
        return f"Please confirm: {self.state.size} pizza, ${self.state.total}. Yes or no?"

    @function_tool(requires_approval=_read_back)
    async def place(self, ctx: RunContext[Shop, Customer]) -> Completed[str]: ...
```

## Porting tools

- Arguments: anything pydantic validates. Keep v1 enums as `Literal[...]` or ids the tool
  listed; constrain numbers with `Annotated[int, Field(ge=1)]`. A free-form `datetime` the model
  fills is risky: give it today's date in `@backend.instructions`, or take ids from a listing.
- Ids listed in instructions or tool output: short (`a1`, `a2`) or set apart (`[id=a1]`), never
  run into the text, since the model copies the punctuation around them. A `ToolError` for an
  unknown id lists the valid ones; otherwise the model retries the same bad id until the run
  runs out of steps.
- Return a short string: what happened, and what to do next when it failed. Raise `ToolError`
  for a recoverable failure. The backend reports only what a tool did: "I've added X" before a
  tool added it is the most common failure of a migrated backend.
- Keep the backend's routing instructions short and put each tool's rules in its docstring: a
  long v1 prompt on a small model stops it starting flows at once.
- Name the single next thing to ask in a tool's return and in dynamic instructions ("next: the
  email"), never a list: the model asks what it is told, all at once.
- A conversation tool that returns `InputRequired` never runs again with the answer; the model
  reads the answer and calls the tool again with it. Design parameters for that.
- `await ctx.update("...")` before slow work.

## State and deps

- `state_type` is JSON, defaulted, < 64 KiB, saved after every turn. `deps` are clients and
  databases, never saved.
- v1 built domain objects per room (`FakeDatabase()` in the session). In 2.0 a module-level
  deps object is shared by every thread of the worker: data that must survive a conversation
  lives there or in a real store; data about one conversation lives in `state`.
- The thread's history is durable: whatever the person types (a full card number) stays in it.
  Keep only what you need in state (the last four digits), and say so in `MIGRATION.md`.

## AgentTask in text

The backend is reached by text, so flows run as tasks across turns and workers.

```python
from dataclasses import dataclass

from pydantic import BaseModel

from livekit.agents import AgentTask, Completed, Failed, RunContext, ToolError, function_tool


class Draft(BaseModel):  # the task's own state, saved between turns
    size: str | None = None
    toppings: list[str] = []


@dataclass
class OrderPizza(AgentTask[str, Draft]):
    size: str | None = None  # what the customer already said: JSON-shaped, saved with the task

    def __post_init__(self) -> None:
        super().__init__(
            instructions="Take the pizza order. Record what the customer already said, then ask "
            "for what is missing, one thing at a time. For anything outside this order, call "
            "stop first.",
            llm="openai/gpt-4.1-mini",  # required: a task does not inherit the routing agent's
        )
        self.state = Draft(size=self.size)
        self.instructions(self._next)  # dynamic instructions: a bound method taking ctx

    def _next(self, ctx: RunContext[Shop, Customer]) -> str:
        return "Next: the size." if self.state.size is None else "Next: the toppings."

    @function_tool
    async def record_size(self, ctx: RunContext[Shop, Customer], size: str) -> str:
        """Record the size.

        Args:
            size: small, medium or large
        """
        self.state.size = size
        return "recorded; next, the toppings"

    @function_tool
    async def finish(self, ctx: RunContext[Shop, Customer]) -> Completed[str]:
        """The order is complete."""
        if self.state.size is None:
            raise ToolError("no size yet; ask for it")  # not done: the task goes on
        return Completed(f"a {self.state.size} pizza; tell the customer it is on its way")

    @function_tool
    async def stop(self, ctx: RunContext[Shop, Customer], reason: str) -> Failed:
        """Leave the order: the customer wants something else, or gave up.

        Args:
            reason: why, in a few words
        """
        return Failed(f"order stopped: {reason}; help with what they asked instead")


@backend.tool
async def start_pizza_order(ctx: RunContext[Shop, Customer], size: str | None = None) -> str:
    """Order a pizza; the flow asks for what is missing. Pass what the customer already said.

    Args:
        size: the size, if the customer said it
    """
    return await OrderPizza(size=size)  # nothing after this line runs in text
```

- A task is a module-level `@dataclass` subclass of `AgentTask[Output, State]`, rebuilt by
  class name and fields on the next message, maybe on another worker. Fields are JSON-shaped.
- Give it its own `llm=`: a task never inherits the routing agent's model.
- Inside a task tool, `ctx.state` is the lane's state (it outlives the task) and `self.state`
  the task's own.
- Don't name a field or a tool after an `Agent` attribute: `done`, `instructions`, `tools`, `id`,
  `label`, `state`, `session`, `chat_ctx`.
- A task tool returns `Completed`/`Failed` to end the task, or text and `ToolError` to go on;
  `Failed` only for the exit tool and real failures, since it ends the task.
- Fold v1 sub-tasks (`GetCardTask` awaited inside `BookRoomTask`) into the task: state fields
  and `record_*` tools; share tools between tasks with a mixin class.
- It ends only through a tool returning `Completed(value)` (what `await` returns) or `Failed`.
  **Every task needs an exit tool** returning `Failed`, and instructions to call it for anything
  outside the flow: while a task runs, the routing agent's tools are out of reach.
- **Code after `await Task(...)` does not run in text.** `return await Task(...)`, make the
  `Completed` value the text the routing agent needs, and do the final write inside the task's
  last tool.
- Pass what the person already said into the task's fields, and pre-fill `self.state` from them
  in `__post_init__`, so it does not ask again.
- Do not use `self.session` in a backend task.

### Verify, then act

v1's common "verify the caller, then do the thing" (two tasks in one tool) becomes **guard and
re-call**. Guard every tool that needs it the same way, in its body or in its read-back; a
`ToolError("verify first")` instead leaves the model to ask the person itself, or to push the
questions onto the frontend: the guarded tool returns `await Verify()` when the state says the person is not
verified, `Verify` records the verification in the lane's state and completes with "verified;
now call the tool again", and the model calls the guarded tool again, which now passes the
guard:

```python
@backend.tool  # a tool with no approval: the guard is in its body
async def lookup_booking(ctx: RunContext[Hotel, Caller]) -> str:
    booking = verified(ctx)  # reads ctx.state.verified_code
    if booking is None:
        return await VerifyBooking()  # completes with "verified; call the tool again"
    return describe(booking)


async def cancellation(ctx: RunContext[Hotel, Caller]) -> str:
    """The read-back runs before the body: the guard goes here for a tool that needs approval."""
    booking = verified(ctx)
    if booking is None:
        return await VerifyBooking()
    return f"Please confirm: cancel {booking.code}, {booking.refund_text}. Yes or no?"


@backend.tool(requires_approval=cancellation)
async def cancel_booking(ctx: RunContext[Hotel, Caller]) -> str: ...
```

Give `Verify` optional fields for the credentials the person already gave. Verification now
lasts as long as the thread, across calls and days, not one call: decide whether that is right
(add a `sign_out` tool, or keep the time it was verified in state and re-verify after a while)
and write the decision in `MIGRATION.md`.

## Running it

The stack (livekit-server with agent endpoints, and agent-db) must be up, or the environment must
point at a LiveKit project that serves agent endpoints:

```bash
uv run python main.py dev
```

Text to the frontend from Python:

```python
from livekit.agents.a2a import AgentClient, Credentials

token = Credentials(KEY, SECRET).project_token()
client = AgentClient("http://127.0.0.1:7880/agents/shop/default/frontend", token=token)
reply = await client.send("I'd like a large pizza")
async for update in reply:
    print(update.state, update.text)
reply = await client.send("pepperoni")  # continues the same thread
```

## Tests

- Test one backend turn directly: `await backend.run("...", state=Customer(), deps=fake_shop)`
  on a real model; assert on `result.output`, `result.input_required`, the state, and the
  `FunctionCall` items of `result.new_messages()`. v1's `result.expect`, `judge()` and
  `mock_tools` do not apply to `agent.run()`: judge with a small `Agent(output_type=Verdict)`,
  and inject failures and per-test data through `deps` (a menu with an item unavailable).
- Importing the backend needs the env (its model strings): set dummy `LIVEKIT_API_KEY` and
  `LIVEKIT_API_SECRET` in the test, or keep pure models and validators in the domain module.
- A multi-turn flow (a task, an approval) needs the door: it continues across messages only
  through the A2A endpoint. Test it against a running worker with `AgentClient` (see "Running
  it"; the backend door is `public=False`, so use the project token), or keep it as a
  simulation scenario.
- Keep pure domain tests unchanged. A test module in the repo declares its category:
  `pytestmark = pytest.mark.unit`.

## Verify

1. `LIVEKIT_API_KEY=x LIVEKIT_API_SECRET=x LIVEKIT_AGENTDB_URL=ws://x python -c "import main"`:
   model strings and storage need these at import; dummy values are enough.
2. `mypy --strict --follow-imports=silent` on every new module (all of them, if you split the
   backend), and `ruff check --line-length 100`: every error in the new code fixed. Untouched
   domain modules are out of scope; a call into an untyped one may take
   `# type: ignore[no-untyped-call]`.
3. Walk the inventory: every v1 tool, flow, rule, voice feature and test has a row saying where
   it lives now, or why it was dropped.
4. Read the frontend instructions once more: no business facts, no business tools, nothing it
   could confirm on its own, and it delegates before asking anything.
5. If the stack is available, run `main.py` and hold one text conversation through every flow,
   including a question the backend asks and an approval before a write. Check the result,
   not the closing words: the write happened (state, database), and the backend lane's history
   (`AgentDBStorage().threads.get_thread_items(thread_id, agent="backend")`) shows the call's
   `lk.approval` as `asked`, then `granted`.

## Pitfalls

- Business tools on the frontend "for speed". The frontend stays thin; its first progress line
  covers the wait.
- A frontend that asks business questions itself, rewrites the request in the third person
  ("the customer wants..."), or confirms what the backend did not say.
- A `state_type` that cannot construct with no arguments, or that holds a client.
- `Agent(...)` built inside a handler: agents are module-level and shared.
- A handler returning `result.output` alone: while a tool waits on the person `output` is
  `None`; return `result.input_required or str(result.output)`.
- Chaining two awaited tasks in one tool, or work after the await: only the first await happens
  in text. Use guard and re-call.
- A task with no exit tool: the person cannot leave it.
- Confirmation in the frontend's instructions, or an `approved: bool` parameter: use
  `requires_approval=`.
