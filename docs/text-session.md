# TextSession

`TextSession` coordinates a text conversation across a root agent and agents called as tools.
Each agent has its own model history. The current agent receives new human input; a completed
child returns its accepted output to the parent's originating tool call.

Agent definitions and TextSession configuration are reusable. Conversation state belongs to
an explicitly bound store. Without a binding, a run starts a fresh conversation.

## Running a conversation

```python
from dataclasses import dataclass
from livekit.agents import Agent, DelegationOptions, SessionState, TextSession, persisted

@dataclass
class BookingContext:
    services: HotelServices
    session: SessionState = persisted(default_factory=SessionState)
    draft: RoomDraft = persisted(default_factory=RoomDraft)

booking = Agent(
    name="booking",
    instructions="Collect the details and reserve only after approval.",
    llm=backend_model,
    output_type=Booking,
)
front_desk = Agent(
    name="front-desk",
    instructions="Help customers plan their stay.",
    llm=frontend_model,
    tools=[booking.as_tool(needs_approval=True, merge_chat_ctx=True)],
)
session = TextSession(
    agent=front_desk,
    delegation=DelegationOptions(routing_model=routing_model),
)

app = await ctx.store.load_context(BookingContext(services=services))
result = await session.run(ctx.message, context=ctx.store.bind(app))
if result.status == "input_required":
    return result
return result.output
```

The next request loads fresh application data and services and binds the same thread and
partition. It resumes the saved current agent. No Python coroutine stays suspended between
requests. TextSession itself caches no conversation between calls.

## Loading and saving session state

An optional persisted SessionState field exposes the same state that the binding advances.
The field's name is application-defined. A binding also works without such a field:
`context=ctx.store.bind(app)` still selects and checkpoints the conversation.

`store.bind()` is the only continuation input. There is no `state=` argument, implicit restore
from a plain application object, or second snapshot execution mode. A plain `context=app`
supplies dependencies to a fresh run. Bound `Agent.run()` and `Agent.run_stream()` use the
same native conversation owner.

## Results and streaming

`run()` returns `RunResult[Output]`, using the same [result contract](agents.md#running-a-turn)
as Agent.run. A completed result exposes its typed output; an input-required result exposes
its question. The conversation remains available through the bound SessionState, independently
of the invocation result. `result.usage` is an immutable snapshot of this turn's consumption
across its local agents, counted once. A reconstructed turn starts fresh accounting.

```python
async with session.run_stream(ctx.message, context=ctx.store.bind(app)) as stream:
    async for text in stream.stream_text(delta=True):
        await output.write(text)
    result = await stream.result()
```

`run_stream()` accepts the same message, context and retries as `run()`. It has one text/event
consumer and an awaitable `result()`. Child events retain their agent and invocation identity;
streaming does not merge their private histories into the parent. Cancellation and cleanup
settle before result retrieval. Failure raises rather than manufacturing a completed result.
See [streaming](agents.md#text-session-streaming).

The last text fragment is not a storage boundary. The native owner checkpoints framework
transitions and delivery receipts. Changed application fields still require
`await ctx.store.save_context(app)`. The [persistence guide](persistence.md#session-state-and-explicit-binding)
owns binding validation, object identity, snapshots and save boundaries.

`retries=` overrides every agent's retry budgets for that turn. An integer sets both budgets;
AgentRetries selects them individually. See [retries](agents.md#retries).

## Shared state and private histories

Read `app.session.current_agent.chat_ctx` for the current agent's private model history and
`app.session.chat_history` for the person's delivered conversation. Root/current views,
local call relationships and snapshot behavior belong to [conversation state](session-state.md).
The [persistence guide](persistence.md#session-state-and-explicit-binding) owns binding and saves.

## Responses to requests for input

Configure `DelegationOptions(routing_model=...)` on TextSession, or pass the same options as
`delegation=` to a bound Agent.run or Agent.run_stream. This separate model applies to the
whole conversation. It checks presented unresolved requests before the current agent plans,
then separately checks proposed remote tool calls against known work. Set `routing_model=None`
to disable automatic routing and use explicit remote `task_id` selection.

The current agent receives the whole human message. Recognition can retain a response while
another child is current, but delivery waits for an eligible agent with the relevant remote
tool. The original question and reserved human words keep their identity across that wait.
[Delegation](delegation.md#responding-to-questions) owns recognition, clarification, grouping,
corrections and dispatch rules.

Text consumed from `run_stream` establishes delivery at that execution's output boundary.
Verbatim questions use the same stream and receipts as model text. Reading only a result
without consuming its text does not establish presentation. Routing-model consumption is
included in the turn's usage.

## Local agents as tools

An agent used as a tool must declare `output_type`. That contract defines its accepted final
value and completion condition. A standalone conversational agent can omit output_type.
Construct agents once and register tools and instructions at import. Each invocation receives
the current request's application context and fresh service objects.

1. The parent model chooses the agent tool and supplies its arguments.
2. The session records the parent call and starts the child with its private history.
3. If the child needs input, the invocation returns an input-required result and the child
   remains current. The next bound human turn continues that child.
4. The child's accepted final output fills the recorded parent tool call.
5. The parent starts a fresh invocation with that result and any selected dialogue.

The parent is not invoked with its child call unresolved. The relationship is saved data,
not a Python parent await retained across requests. Agent-tool names, typed parameters,
approvals and history selection use [the tool API](tools.md#configuring-an-agent-tool).

## Approval

`needs_approval=` uses the ordinary tool approval gate, including read-back functions. The
saved decision refers to the parent's invocation and validated arguments. With approval
required, the child has not started when the session asks permission.

A recognized response is delivered to the parent that owns the read-back. If another local
child is current, the response stays held until the parent resumes. The body still requires
a validated repeat of the same call; changing its arguments requires another read-back.
Starting a specialist and authorizing a business operation are separate: a booking specialist
can collect preferences before its reservation tool asks permission for the actual booking.
See [tool approval](tools.md#approval).

Framework checkpoints retain decisions and call relationships. Ordinary application changes
remain explicit saves. External services still need their own idempotency and authorization.

## Remote agents and delivery modes

`RemoteAgent(url).as_tool()` communicates with an A2A endpoint. Remote communication is
separate from local child execution: a remote can have multiple tasks and contexts, or return
a direct Message without creating a task. See [delegation](delegation.md).

On A2A, `ctx.execution_id` is the task ID. A message addressed to a waiting task continues it.
A message without taskId follows the endpoint's [task-answering contract](protocol/a2a.md#answering-a-task).
Ending a frontend does not itself cancel remote business work.

VoiceSession owns the call's speech, interruption and presentation. It does not require a
TextSession. Durable local Agent-as-tool continuations belong to text; finish unresolved local
work before starting voice on the same conversation. Voice handoffs and VoiceTasks are live
interactions that preserve the durable root and human history. A VoiceTask's parent await stays
in memory; it does not provide durable backend continuation. See
[voice interactions](voice-tasks.md#voicetask) and [frontend events](voice-session.md#conversation-owned-inbox).
