# Tools and toolsets

A tool is a function the model can call. Its first parameter is the
`RunContext`, the rest are its arguments, and its Google-style docstring
feeds the schema.

## Tools on an agent

```python
@scheduler.tool
async def find_appointment(
    ctx: RunContext[PatientContext], last_name: str, date_of_birth: str
) -> str:
    """Find the patient's next appointment.

    Args:
        last_name: as spelled on the patient's file
        date_of_birth: YYYY-MM-DD
    """
    r = await ctx.context.clinic.get("/appointments/next", params={"last_name": last_name, "dob": date_of_birth})
    if r.status_code != 200:
        return "no upcoming appointment"
    ctx.context.appointment = Appointment(**r.json())
    return f"{ctx.context.appointment.when:%A %H:%M} with {ctx.context.appointment.provider}"
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
- Reads per-run data from `ctx.context`, never from module
  globals that change.

## Agents as tools

`agent.as_tool(...)` exposes a specialist with a typed input schema and a
mandatory `output_type`. Attach it through the parent's `tools=` and run the
conversation with `AgentRouter`. Configure its name, description, parameters,
approval gate, and history policy in the
[agent-tool specification](agent-router.md#configuring-an-agent-tool).

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
| `InputRequired(question)` | the turn waits for the person; see [human-in-the-loop.md](human-in-the-loop.md) |
| `Failed(error)`, `Rejected(reason)` | the model reads the text as an error output and replies |
| `Completed(value)` | ordinary call output; [agent tasks](agent-router.md) complete through accepted typed final output |

## Approval

`needs_approval=` configures the same framework approval gate on
`@agent.tool`, `@function_tool`, and `agent.as_tool(...)`. The gate runs
before the tool body or child agent starts. Approval is bound to the exact
tool invocation and validated arguments; changed arguments require approval
again. The model cannot grant approval through a tool argument.

For an agent tool, the router retains the pending invocation in its state.
The application explicitly saves that state and loads it for the next reply.
See [approval persistence](agent-router.md#approval-before-starting-a-child).


```python
@clinic.tool(needs_approval=True)
async def cancel_appointment(ctx: RunContext[PatientContext], date: str) -> str:
    """Cancel the patient's appointment on a date.

    Args:
        date: the appointment's date
    """
    ...
```

A tool declared with `needs_approval=True`, on `@agent.tool` or
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

`needs_approval=` also takes a read-back function. It takes the tool's
own parameters, the context and a method's `self` included, and returns the
question; it may return an `InputRequired` instead, for a detail that is
missing before there is anything to approve:

```python
async def read_back(ctx: RunContext[PatientContext], slot_id: str, email: str | None = None) -> str | InputRequired:
    slot = await open_slot(ctx, slot_id)
    if (email := email or ctx.context.email) is None:
        return InputRequired("What email address should the confirmation go to?")
    return f"Please confirm: {spoken(slot.start)}, confirmation to {email}. Yes or no?"

@backend.tool(needs_approval=read_back)
async def schedule_appointment(ctx: RunContext[PatientContext], slot_id: str, email: str | None = None) -> str: ...
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
    async def card_status(self, ctx: RunContext[MemberContext], last_four: str) -> str: ...

    @function_tool(needs_approval=True)
    async def block_card(self, ctx: RunContext[MemberContext], last_four: str) -> str: ...

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

Runtime clients in application context are supplied per run; see
[state-and-deps.md](state-and-deps.md).

## On a call

Tools that run while the agent speaks have options that only matter on a
call: `flags=ToolFlag.CANCELLABLE`, and `on_duplicate=` with
`duplicate_scope=` for a call made while the same tool is still running
(`"allow"`, `"reject"`, `"replace"`, or `"confirm"`, which asks the model to
call again to confirm). See [voice.md](voice.md).

A tool whose result is a question for the person returns
`InputRequired(question)`: the agent facing the person asks it in its own
words, or exactly as written with `verbatim=True`, and the person's next
message is written as the call's output for the model to go on with. With `tool=`, the answer is not the model's to interpret: it is handed to
that tool as its first string parameter, on a call and in text, with no model
turn in between, and the model goes on from that tool's output. The delegate's
tool does this for an expert's `input-required`
([multi-agent composition](agents.md#working-with-other-agents)).

## Idempotency

A tool may run again after a retry. Use a stable business-operation key for external effects and enforce it in
the authoritative service. A tool-call ID identifies a repeated invocation,
but a model retry can create a new call ID for the same business action.

Filtered and prefixed toolsets are specified in
[planned-agents.md](planned-agents.md).
