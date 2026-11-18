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

`app.session.current_agent` and `app.session.root_agent` resolve entries in
`app.session.agents`. Their `chat_ctx` values are private model histories. They include tool
activity and any context inherited when a child started.

`app.session.chat_history` records the conversation the person experienced across agents and
supported text/voice transitions. It excludes tool calls, tool results and synthetic child task
instructions. `snapshot()` returns a detached inspection view; execution continues through
store binding. SessionSnapshot is private native data, not another public state class.

`merge_chat_ctx=True` copies only the child's conversation with the person into the parent's
model context. Human dialogue remains in chat_history regardless of that option. After the
parent receives the child's accepted output and selected dialogue, the completed child is
removed from the active agent map.

## Responses to requests for input

Configure a separate routing model with `DelegationOptions(routing_model=...)` on TextSession,
or pass the same options as `delegation=` to a bound Agent.run or Agent.run_stream. This is
conversation-wide configuration shared by every remote tool and local agent. The framework
does not reuse the frontend model or select a provider implicitly.

Set `routing_model=None` to disable automatic routing. The current agent handles incoming
messages normally; remote tools can continue known work with `task_id`, and calls without it
start new work. Task selection supports clarification and correction as well as responses;
it does not itself consume a pending input occurrence. See [remote task routing](protocol/a2a.md#selecting-work-through-a-remote-tool).

Before the current agent handles a new message, the routing model checks unresolved input
requests that were actually presented. This includes a single question and earlier presented
questions from inactive agents. A side question such as "Why do you need the date?" leaves the
date request open. An unclear or failed classification lets the current agent clarify; it does
not guess a destination. Without presented candidates, no classification call is made.
The verdict applies to the exact message classified. A revised transcript must be classified
again before its current words can answer a request.

Related tool questions can become one prompt, such as "What are your first and last names?"
The frontend selects the sources and wording together; each input request keeps its identity.
A clear full-name response can address both requests. "My first name is Ada" answers only the
first-name request, leaving the last name open. Questions omitted from the delivered prompt
are not offered to the routing model. Grouping questions never grants business approval.

The current agent still receives the entire human message and a private routing observation.
The native owner validates selected occurrences and checkpoints their canonical human response
before dispatch. Recognition can retain an answer while another agent is current. It does not
switch agents, approve a business operation or prove that a remote received the response.
When a selected response continues a remote task, it keeps that task's context and sends the
person's reserved words. Model-generated tool arguments cannot rewrite them.

The response belongs to the conversation's remote task. An eligible agent can send it even
when that agent has a fresh private history. Until an eligible agent invokes the remote tool,
the response stays held. A tool call cannot choose arbitrarily between several held responses
or create a second outgoing operation for a response whose dispatch is already being tracked.
While a response is held, the routing model can recognize a clear correction such as "Actually
Saturday". The owner checks that the original response is still current before replacing it.
Once dispatch may have started, it preserves the sent message and reconciles that operation.

The same routing model separately checks proposed remote tool calls against known tasks.
A side question need not become a new task, and an existing held response cannot redirect an
unrelated hotel request. Each model turn retains the input occurrences it was offered;
admission rechecks those occurrences before sending. A response whose outgoing operation is
already tracked is acknowledged locally without another send. This deduplication applies to
that response, so the same human turn can also start independent work.

Text consumed from run_stream establishes delivery at that output boundary. Verbatim tool
questions use the same stream and receipts as model text. Reading only a result without
consuming its text does not establish presentation. Routing model consumption is included in
the turn's usage. The [delegation design](proposals/delegation.md#input-response-detection)
describes grouping, dispatch eligibility and recovery across execution modes.

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
a direct Message without creating a task. See [delegation](proposals/delegation.md).

On A2A, `ctx.execution_id` is the task ID. A message addressed to a waiting task continues it.
A message without taskId follows the endpoint's [task-answering contract](protocol/a2a.md#answering-a-task).
Ending a frontend does not itself cancel remote business work.

VoiceSession owns the call's speech, interruption and presentation. It does not require a
TextSession. Durable local Agent-as-tool continuations belong to text; finish unresolved local
work before starting voice on the same conversation. Voice handoffs and VoiceTasks are live
interactions that preserve the durable root and human history. A VoiceTask's parent await stays
in memory; it does not provide durable backend continuation. See
[voice interactions](voice-tasks.md#voicetask) and [frontend events](voice-session.md#conversation-owned-inbox).
