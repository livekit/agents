# Tools and toolsets

A tool is a function the model can call. Its first parameter is the
`RunContext`, the rest are its arguments, and its Google-style docstring
feeds the schema.

## Tools on an agent

```python
@scheduler.tool
async def find_appointment(
    ctx: RunContext[Deps, Patient], last_name: str, date_of_birth: str
) -> str:
    """Find the patient's next appointment.

    Args:
        last_name: as spelled on the patient's file
        date_of_birth: YYYY-MM-DD
    """
    r = await ctx.deps.clinic.get("/appointments/next", params={"last_name": last_name, "dob": date_of_birth})
    if r.status_code != 200:
        return "no upcoming appointment"
    ctx.state.appointment = Appointment(**r.json())
    return f"{ctx.state.appointment.when:%A %H:%M} with {ctx.state.appointment.provider}"
```

`@agent.tool` and `@agent.tool(name=, description=, ...)` register a tool on
a module-level agent. A subclass declares its tools as methods with
`@function_tool`; `Agent(tools=[...])` takes tools built elsewhere.

- Sync or async. Arguments are anything pydantic validates: scalars,
  dataclasses, pydantic models, lists, dicts.
- The return value is what the model reads: a string, a number, a bool,
  `None`, a dataclass or a pydantic model, or lists, tuples and dicts of
  those. Dataclasses and models are serialized the way `Completed` values
  are. Anything else is recorded as an error output.
- Reads per-run data from `ctx.deps` and `ctx.state`, never from module
  globals that change.

## What a tool returns

| Return or raise | What happens |
|---|---|
| a value | the model reads it and continues |
| `ToolResult(output, reply_required=False)` | the output is recorded; the call asks for no reply |
| `ToolResult(text, verbatim=True)` | the person gets the text exactly as written; the model reads that it was delivered and continues, or the turn ends there with `reply_required=False`; see [human-in-the-loop.md](human-in-the-loop.md) |
| raise `ToolError(message)` | the model reads the message as an error output and continues |
| raise `ModelRetry(reason)` | the model reads the reason and calls again; the same in `agent.run()` and on a call |
| raise `StopResponse()` | the call is answered with nothing and asks for no reply; `run()` ends the turn with no output |
| any other exception | logged; the model reads "An internal error occurred" |
| an `Agent` | a handoff: the next model request runs that agent on the same history; see [multi-agent.md](multi-agent.md) |
| `InputRequired(question)` | the turn waits for the person; see [human-in-the-loop.md](human-in-the-loop.md) |
| `Failed(error)`, `Rejected(reason)` | the model reads the text as an error output and replies |
| `Completed(value)` | the value is the call's output and the model continues; only a task's tool ends a task with it; see [agent-tasks.md](agent-tasks.md) |

## Progress and exact text

`ctx.update(text)` is progress the model words; `ctx.say(text)` is exact
text, said as written with no model step and recorded as the agent's
message: spoken on a call, a flagged `working` status in text
([a2a-live-ext.md](a2a-live-ext.md#exact-text)). On a call the first of
them releases the tool to the background, and a first `say` is the turn's
reply; later ones are delivered at idle in call order, consecutive updates
as one reply, and a `say` cut off before it played is said again. A model
that cannot say text as written is asked to, with a warning.

The calls of one step run together. A verbatim result the model goes on
from is said when its tool returns. Those that end the turn are the
answer, joined in call order with a blank line, unless another call asks
for a reply: then they are said and the model goes on. On an agent whose
`output_type` is not `str`, a verbatim result never ends the turn.

## Approval

```python
@clinic.tool(requires_approval=True)
async def cancel_appointment(ctx: RunContext[Deps, Patient], date: str) -> str:
    """Cancel the patient's appointment on a date.

    Args:
        date: the appointment's date
    """
    ...
```

A tool declared with `requires_approval=True`, on `@agent.tool` or
`@function_tool`, does not run when the model calls it. The call asks the
person instead, as a verbatim `InputRequired`:
`Please confirm: Cancel the patient's appointment on a date (date='Monday'). Yes or no?`,
the first sentence of the docstring and the arguments. The person's answer
is that call's output, and the model reads it as on any turn.

Approval is held by the framework. The tool's schema has no parameter for
it, and the body runs only for a repeat of the same call, with the same
arguments, made after the person answered its read-back. A first call, a
call with other arguments, or a second repeat asks again. The calls carry
`lk.approval` (`asked`, `granted`) in `FunctionCall.extra`; the next call
reads them from the history. The model reads an answered read-back's output
as `not run yet: the person answered '...'`, with how to run it; the history
stores the person's words. A raw-schema tool cannot require approval.

`requires_approval=` also takes a read-back function. It takes the tool's
own parameters, the context and a method's `self` included, and returns the
question; it may return an `InputRequired` instead, for a detail that is
missing before there is anything to approve:

```python
async def read_back(ctx: RunContext[Calendar, Patient], slot_id: str, email: str | None = None) -> str | InputRequired:
    slot = await open_slot(ctx, slot_id)
    if (email := email or ctx.state.email) is None:
        return InputRequired("What email address should the confirmation go to?")
    return f"Please confirm: {spoken(slot.start)}, confirmation to {email}. Yes or no?"

@backend.tool(requires_approval=read_back)
async def schedule_appointment(ctx: RunContext[Calendar, Patient], slot_id: str, email: str | None = None) -> str: ...
```

## Toolsets

A `Toolset` is a group of tools with an id, defined once and attached to any
agent through `tools=`:

```python
from livekit.agents import RunContext, function_tool
from livekit.agents.llm import Toolset
from livekit.agents.llm.mcp import MCPServerHTTP, MCPToolset

class Cards(Toolset):
    def __init__(self) -> None:
        super().__init__(id="cards")

    @function_tool
    async def card_status(self, ctx: RunContext[Deps, Member], last_four: str) -> str: ...

    @function_tool(requires_approval=True)
    async def block_card(self, ctx: RunContext[Deps, Member], last_four: str) -> str: ...

rewards = MCPToolset(id="rewards", mcp_server=MCPServerHTTP("https://mcp.rewards-partner.example/"))

member_services = Agent(..., tools=[Cards(), rewards])
```

| Class | What |
|---|---|
| `Toolset(id=, tools=)` | a group of tools; a subclass adds its `@function_tool` methods |
| `AsyncToolset(id=, tools=, tool_handling=)` | on a call, background work from its tools survives a handoff when the toolset is passed to `LiveSession(tools=)` |
| `MCPToolset(id=, mcp_server=, tool_options=)` | the tools of an MCP server, `MCPServerHTTP(url)` or `MCPServerStdio(command, args)` |

A toolset connects per run of the agent: `setup()` when the agent's
activity starts, a call or a run, and `aclose()` when it ends. An
`MCPToolset` connects to its server and lists its tools in `setup()`.
Toolsets passed to `LiveSession(tools=)` live for the whole call.

An agent and its toolsets are shared by every conversation in the
process, so a toolset holds process resources only, a client or a pool,
and reads a conversation's state from the `RunContext` and the history.

The clients in `deps` are per run by design; see
[state-and-deps.md](state-and-deps.md).

## On a call

Tools that run while the agent speaks have options that only matter on a
call: `flags=ToolFlag.CANCELLABLE`, and `on_duplicate=` with
`duplicate_scope=` for a call made while the same tool is still running
(`"allow"`, `"reject"`, `"replace"`, or `"confirm"`, which asks the model to
call again to confirm). See [voice.md](voice.md).

A tool that needs an input to go on, an email, a yes or no on a read-back,
a card number, returns `InputRequired(question)`: the agent facing the
person asks it in its own words, or exactly as written with
`verbatim=True`, and the person's next message is the answer, which the
task's model reads and acts on. The questions of one step's calls are asked
as one. A conversational question, such as offering slots, is a plain
reply, and the next message is a new request. Only the delegate's tool takes
the answer itself, as the person said it, with no model turn in between, to
continue the expert's task ([multi-agent.md](multi-agent.md)).

## Idempotency

A tool may run again after a retry. Key every outside call on
`ctx.function_call.call_id` inside a conversation tool and on `ctx.task_id`
inside a task, so a repeated call is a no-op on the other side.

Filtered and prefixed toolsets and `Agent(toolsets=)` are designed in
[planned-agents.md](planned-agents.md).
