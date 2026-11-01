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
a module-level agent. `@function_tool` declares a reusable tool function;
`Agent(tools=[...])` takes tools and toolsets built elsewhere.

- Sync or async. Arguments are anything pydantic validates: scalars,
  dataclasses, pydantic models, lists, dicts.
- The return value is what the model reads: a string, a number, a bool,
  `None`, a dataclass or a pydantic model, or lists, tuples and dicts of
  those. Dataclasses and models are serialized the way `Completed` values
  are. Anything else is recorded as an error output.
- Reads per-run data from `ctx.context`, never from module
  globals that change.

`@function_tool(raw_schema={"name": ..., "description": ..., "parameters": {...}})`
declares a tool from a JSON schema instead of a signature: the function
takes `raw_arguments: dict`, optionally after `ctx`, and validates them itself.

### Retries

Arguments that fail validation and `ModelRetry` from the body send the model
back to correct the call, as in pydantic-ai. Each tool counts its own
consecutive retries against the agent's `retries["tools"]` budget (default
`1`); `retries=` on the tool replaces that budget for it. A run whose tool
goes over its budget fails with `UnexpectedModelBehavior`.

```python
@scheduler.tool(retries=3)
async def find_slot(ctx: RunContext[PatientContext], day: str) -> str:
    """Find an open slot on a day."""
    slots = await ctx.context.clinic.slots(day)
    if not slots:
        raise ModelRetry(f"No slots on {day}; ask for another day or try the next weekday.")
    return slots[0].label
```

`@agent.tool(retries=)`, `@function_tool(retries=)` (also on a toolset's
methods) and `MCPToolOptions(retries=)` take the same budget. See
[agent retries](agents.md#retries) for the run-level budgets.

## Agents as tools

Tools can delegate to another agent directly, through a backend `TextSession`,
or through a frontend remote tool on `VoiceSession`. The execution host owns
the pending interaction; the tool definition controls which agents may call it.

## Direct delegation

Call another agent directly when the caller can consume its result within
one invocation:

```python
@main_line.tool
async def quick_balance(ctx: RunContext[CustomerContext], account_number: str) -> str:
    result = await billing.run(
        f"Look up the balance for account {account_number}. Do not ask follow-up questions.",
        context=ctx.context,
    )
    if result.status == "input_required":
        return f"Balance lookup needs more information: {result.question}"
    if result.status == "no_output":
        return "The balance lookup ended without an answer."
    return str(result.output)
```

The outer tool waits for this invocation to finish. Its parent model then
reads the returned tool value. This starts no session frame or persistent
parent-child relationship. To continue the expert directly, the application
must retain its history and supply it to another run.

Tool bodies enforce application prerequisites such as customer verification
before reading protected records or performing an effect. Instructions and the
presence of a tool in the model's tool list do not establish authorization.

Application context can also carry a budget shared by several tools, such as a
remaining transcript-read count. Each tool checks and decrements that budget
before accessing the service. The agent's run loop enforces no application
limit on reads, cost, or external operations. Use
explicit persistence when a budget must span human turns.

## Backend agents as tools

A local specialist declares `output_type` and is attached through `tools=`
using `.as_tool(...)`. The parent supplies validated input; accepting the
child's output resolves its originating call. A `TextSession` owns backend
frames, private histories, pending input, and parent-child continuation across
requests. See [text-session execution](text-session.md#calling-an-agent-as-a-tool)
and its [complete example](text-session.md#running-a-text-conversation).

## Remote agents as tools

A `RemoteAgent` connects to an agent endpoint using A2A. Expose it through `.as_tool()`
and attach the returned tool to the parent:

```python
from livekit.agents.a2a import RemoteAgent

back_office = RemoteAgent(
    "https://northside.livekit.cloud/agents/eligibility/production/eligibility"
)
front_desk = Agent(
    instructions="Help patients with appointments and insurance eligibility.",
    llm=MODEL,
    tools=[back_office.as_tool()],
)

session = VoiceSession(llm=REALTIME_MODEL, context=app, thread=ctx.thread)
await session.start(agent=front_desk, room=ctx.room)
```

The remote card supplies the tool's name, description, and supported input.
`RemoteAgent(url, *, token=None, conn_options=None)` accepts the same
[connection policy](models.md#connection-options) as `AgentClient`.
The remote service owns its history, application data, approval behavior,
and completion contract. The local tool takes its configuration from that
endpoint. Local agent-tool options such as `chat_history`,
`merge_chat_history`, and `needs_approval` do not configure the remote tool.

In the frontend, `VoiceSession` coordinates the interaction while the adapter
handles the A2A protocol. Expert events enter the calling frontend conversation's in-process
queue and wake its session without database polling. Persistence of frontend
event delivery is optional. The frontend
presents useful progress when the conversational floor permits and can rephrase
questions without taking ownership of the expert's task. An input-required
response records that expert as the destination for the next relevant human
reply. See [inbox delivery](voice-session.md#conversation-owned-inbox). Completion delivers the remote result to the parent tool
call, then the parent model continues. Failures and cancellation propagate
through the tool lifecycle.

The frontend session retains the remote endpoint and conversation/task
references in process. The remote service retains its private history; it can
use plain `Agent.run()` or a text session. Frontend event recovery is optional
and separate from backend persistence.

Backend agents can attach the same remote tool through `tools=`. In that case,
`TextSession` owns the pending call across requests and the application
explicitly saves its returned backend routing state. No frontend event queue
or live session is created by a backend run.

The adapter manages connection lifecycle, authentication, task correlation,
and delegation metadata. Applications configure the endpoint and attach its
tool; protocol calls and metadata recording are internal.

When the caller holds
[explicit authority](protocol/a2a.md#delegation-and-conversation-data) to share
its thread with the endpoint, the adapter sends the thread so the remote agent
keeps its own lane in the caller's thread. The card's `capabilities.extensions`
entry is only a capability hint and grants no access. Other A2A endpoints
retain their own remote context.

See [A2A](protocol/a2a.md) for the wire protocol and authentication rules.

## Configuring an agent tool

`agent.as_tool(...)` exposes the agent to a parent model through `tools=`.

| Parameter | Meaning |
| --- | --- |
| `tool_name` | Name the parent model uses to call the tool |
| `tool_description` | Explains when the parent should select this specialist |
| `parameters` | Dataclass or Pydantic model defining the parent's tool arguments |
| `needs_approval` | Requires human approval before starting the requested child invocation |
| `chat_history` | `"fresh"` starts from the tool input; `"fork"` also copies the parent history |
| `merge_chat_history` | `True` returns the child's visible dialogue alongside its typed result; `False` returns only the result |

The framework validates the parent-provided arguments against `parameters`
before starting the child. These arguments become the child's task input.
The child's `output_type` defines its final result. Runtime services and the
persisted draft are supplied through application context.

History selection and merging are independent. For example,
`chat_history="fork", merge_chat_history=False` lets a specialist use the
parent conversation while returning only its accepted result.

## What a tool returns

| Return or raise | What happens |
|---|---|
| a value | the model reads it and continues; `None` follows the mode-specific rule below |
| `ToolResult(output, reply_required=False)` | the output is recorded; the call asks for no reply |
| `ToolResult(text, verbatim=True)` | the person gets the text exactly as written; the model reads that it was delivered and continues, or the turn ends there with `reply_required=False`; see [human input](#human-input) |
| raise `ToolError(message)` | the model reads the message as an error output and continues |
| raise `ModelRetry(reason)` | the model reads the reason and calls again, within the tool's [retry budget](#retries); the same in `agent.run()` and on a call |
| raise `StopResponse()` | the call is answered with nothing and asks for no reply; `run()` ends the turn with `status="no_output"` |
| any other exception | logged; the model reads "An internal error occurred" |
| `InputRequired(question)` | the turn waits for the person; see [human input](#human-input) |
| `Failed(error)`, `Rejected(reason)` | the model reads the text as an error output and replies |
| `Completed(value)` | An ordinary agent tool returns its call output; a VoiceTask tool completes its live interaction with the validated typed value. See [custom voice tasks](voice-tasks.md#authoring-a-custom-voicetask). |

In direct runs and `TextSession`, `None` is a tool value and the model
continues. On a call, an ordinary tool returning `None` records its output
without requesting a follow-up reply. Use `ToolResult` to choose reply
behavior explicitly.

## Human input

```python
@fnol.tool
async def request_photo(ctx: RunContext[LossContext]) -> InputRequired:
    """Ask the policyholder for a photo of the damage."""
    return InputRequired("Could you send a photo of the damage? A wide shot and a close-up.")
```

The tool finishes with its question recorded as its outcome. The invocation
returns a result with `status="input_required"` and `question`. Save the
updated history and context, then deliver `result.question`.

### Wording and verbatim

The agent facing the person words what reaches them.

| | The agent facing the person | A run another agent asked |
|---|---|---|
| `InputRequired(question)` | asks in its own words: one model request with no tools in text, a `generate_reply` on a call | passes the question up as written; the agent facing the person words it |
| `InputRequired(question, verbatim=True)` | asks it exactly as written: the turn's reply in text, `say()` on a call | passes it up flagged `livekit/verbatim`; asked as written at the end |
| `ToolResult(text, verbatim=True)`, `Completed(value, verbatim=True)` | delivers the text as written; the model reads `delivered to the person as written: ...` and continues, unless `reply_required=False` ends the turn there | the same, and the exact text leads the answer, flagged `livekit/verbatim` |

A run is another agent's when its turn's messages carry
`livekit/kind: delegation`. A handler without a model asks as written. `result.verbatim` is the part of the answer or the question
that was exact; the door puts it on the task as `livekit/verbatim`, and the
delegate delivers that part as written and hands the rest to its model.
A read-back from `needs_approval` is verbatim.

The person's next message is appended to the saved history. A new invocation
reads that history and context. The model interprets the answer and chooses
its next tool or question. The tool that requested input has already finished.

A conversation tool that returns `Failed(error)` or `Rejected(reason)`
records the text as an error output and the model replies to it.
`Completed(value)` from an ordinary agent tool is that call's output. Inside
a `VoiceTask` tool, it completes that live task and returns its validated result
to the waiting frontend caller. See [custom voice tasks](voice-tasks.md#authoring-a-custom-voicetask).
An agent completes when its final output is accepted. The handler delivers the
returned framework outcome to the transport.

Text task correlation is defined by the [A2A answer contract](protocol/a2a.md#answering-a-task).
`TextSession` records backend frame continuation; `VoiceSession` owns frontend
expert questions and reply routing. Neither path resumes the Python body of
a tool that already returned `InputRequired`.

## Approval

`needs_approval=` configures the same framework approval gate on
`@agent.tool`, `@function_tool`, and `agent.as_tool(...)`. The gate runs
before the tool body or child agent starts. Approval is bound to the exact
tool invocation and its validated arguments (after defaults and validation);
changed arguments require approval again. Each approval is consumed once: a
later call, even with identical arguments, asks again. The model cannot grant
approval through a tool argument.

For an agent tool, the session retains the pending invocation in its state.
The application explicitly saves that state and loads it for the next reply.
See [approval persistence](text-session.md#approval-before-starting-a-child).


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
| `AsyncToolset(id=, tools=, tool_handling=)` | on a call, background work from its tools survives a handoff when the toolset is passed to `VoiceSession(tools=)` |
| `MCPToolset(id=, mcp_server=, tool_options=)` | an `AsyncToolset` of an MCP server's tools, `MCPServerHTTP(url)` or `MCPServerStdio(command, args)` |
| `MCPServerHTTP(url, headers=, allowed_tools=, transport_type=, timeout=, sse_read_timeout=, client_session_timeout_seconds=)` | an MCP server over streamable HTTP, or SSE (`transport_type="sse"`); `timeout` bounds the connect and `client_session_timeout_seconds` one request |
| `MCPServerStdio(command, args, env=, cwd=, client_session_timeout_seconds=)` | an MCP server run as a child process |
| `MCPToolOptions(retries=, flags=, on_duplicate=, duplicate_scope=, report_progress=)` | one MCP tool's options, keyed by tool name in `tool_options`; `report_progress` turns the server's progress into `ctx.update()` |

A toolset connects per run of the agent: `setup()` when the agent's
activity starts, a call or a run, and `aclose()` when its last using activity
ends. Concurrent runs sharing a toolset share its connection; one run finishing
does not close it beneath another. An `MCPToolset` connects to its server and
lists its tools in `setup()`.
Toolsets passed to `VoiceSession(tools=)` live for the whole call.
Setup failure aborts initialization in both text and voice. Successfully acquired
resources, including nested toolsets, close before the error reaches the caller;
the failed toolset's own `aclose()` is not called. A failed agent activation does
not replace the active voice agent with a partially initialized one.

Runtime clients in application context are supplied per run; see
[context.md](context.md).

## On a call

Tools that run while the agent speaks have options that only matter on a
call: `flags=ToolFlag.IGNORE_ON_ENTER`, which withholds the tool from replies
generated inside the agent's `on_enter`, and `on_duplicate=` with
`duplicate_scope=` for a call made while the same tool is still running
(`"allow"`, `"reject"`, `"replace"`, or `"confirm"`, which asks the model to
call again to confirm). The `DuplicateScope` is `"name"` (any running call of the
tool) or `"name_and_args"` (one with the same arguments). See
[voice-session.md](voice-session.md).

What the model reads around these calls comes from templates set with
`Agent(tool_handling={"async_options": {...}})`, a `ToolHandlingOptions` whose
`async_options` is an `AsyncToolOptions` (from `livekit.agents.voice.tool_executor`).
An `AsyncToolset(tool_handling=)` overrides them for its own tools; the most
specific wins:

| Template | Rendered for | Arguments |
|---|---|---|
| `update_template` | a `ctx.update(message)`, as the model reads it | `function_name`, `call_id`, `message` |
| `duplicate_reject_template` | a call `on_duplicate="reject"` refused | `function_name`, `fnc_calls_json`, `fnc_calls_text` |
| `duplicate_confirm_template` | a call `on_duplicate="confirm"` asks the model to repeat | `function_name`, `fnc_calls_json`, `fnc_calls_text` |
| `reply_at_tail_template` | the deferred reply to background results, when the agent has not spoken since | `call_ids` |
| `reply_maybe_covered_template` | the deferred reply, when newer items followed the results | `call_ids` |

Each is a `str.format()` template or a callable taking the arguments (for
`update_template`, an `UpdatePromptArgs`) and returning the text; an unset
template keeps the framework's text. The deferred
reply reads the templates of its latest result's tool.

A tool whose result is a question for the person returns
`InputRequired(question)`: the agent facing the person asks it in its own
words, or exactly as written with `verbatim=True`, and the person's next
message is written as the call's output for the model to go on with. With `tool=`, the answer is not the model's to interpret: it is handed to
that tool as its first string parameter, on a call and in text, with no model
turn in between, and the model goes on from that tool's output. The delegate's
tool does this for an expert's `input-required`
([multi-agent composition](agents.md#working-with-other-agents)).

## Prebuilt tools

`livekit.agents.beta` holds ready-made tools:

| Name | What |
|---|---|
| `EndCallTool(extra_description=, delete_room=, end_instructions=, ignore_on_enter=, on_tool_called=, on_tool_completed=)` | a toolset whose `end_call` tool ends the call after the agent's goodbye; the callbacks receive a `ToolCalledEvent` and a `ToolCompletedEvent` |
| `send_dtmf_events` | a tool that sends keypad (DTMF) tones on the call |
| `ToolSearchToolset(id=, tools=, max_results=, search_strategy=)` | keeps a large tool collection out of the model's context: one `tool_search` tool loads the tools a search finds |
| `ToolProxyToolset(id=, tools=, ...)` | the same collection behind two fixed tools, `tool_search` (returns schemas) and `call_tool`, so the tool list never changes |
| `SearchStrategy`, `SearchItem` | the search the two toolsets run (`build_index`, `search`, `cleanup` over `SearchItem`s); `BM25SearchStrategy(k1=, b=)` is the default, `KeywordSearchStrategy` a simpler one |

## Idempotency

A tool may run again after a retry. Use a stable business-operation key for external effects and enforce it in
the authoritative service. A tool-call ID identifies a repeated invocation,
but a model retry can create a new call ID for the same business action.
