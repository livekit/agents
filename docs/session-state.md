# Conversation state

`SessionState` describes one conversation across text and voice. It keeps each local agent's
private model context, the dialogue the person experienced, and observed communication with
remote agents. Agents and models remain application-supplied configuration.

## Use the existing context binding

Declare a persisted state field beside application data and services:

```python
from dataclasses import dataclass

from livekit.agents import SessionState, persisted


@dataclass
class App:
    session: SessionState = persisted(default_factory=SessionState)
    account_id: str = persisted(default="")
```

After loading the application context, pass `context=store.bind(app)` to `Agent.run`,
`TextSession.run`, or `VoiceSession`. Tools receive the original application object through
`ctx.context`. The framework keeps its state field current through that binding. Application
field saves remain explicit. [Persistence](persistence.md#session-state-and-explicit-binding)
owns the loading, saving, snapshot and transaction rules.

There is no separate state argument to `Agent.run` or `TextSession`, and `RunResult` does not
carry another state copy. Plain execution without a binding does not resume or save an
application's stored conversation.

## Root, current agent and history

```python
state = app.session
root = state.root_agent
current = state.current_agent
if current is not None:
    private_history = current.chat_ctx
human_history = state.chat_history
```

`root_agent` and `current_agent` return `AgentState` records from `state.agents`. The root is
the top-level local context; current handles unaddressed input. An empty conversation has neither. Read the properties
again to inspect later state: the returned local records are copies of that revision. The
same configured Python agent can participate in several conversations with different histories.

Each context keeps its private `chat_ctx`, including tool activity. The configured agent
supplies its instructions and model separately; those are not restored from this history.
`chat_history` contains accepted human input and assistant content actually delivered across
agents and supported mode changes. It includes interrupted prefixes and excludes private
tools, instructions and inherited duplicate history. Receiving remote content alone does
not make that content part of the person's conversation.

When a text child requests input, it becomes current while the root remains its parent.
Unaddressed human input goes to that current child. A response to an earlier presented remote
question retains its original destination; the framework can hold it until an eligible agent
has the relevant remote tool. See [delegation](delegation.md#responding-to-questions).

`SessionState.current_agent` describes the recorded local conversation. During voice,
`VoiceSession.current_agent` returns the live Python `Agent` selected for the call, including
a temporary VoiceTask's agent. That agent remains selected while listening or idle; the
property does not mean that speech is currently playing. Temporary voice roles do not create
persisted text-child continuations.

Voice and text can continue the same root history and remote work through a saved binding.
Durable local agent-as-tool continuations belong to text. Voice handoffs and `VoiceTask`
interactions remain transient; an unresolved text child must be continued through text before
switching that conversation to voice.

## Local agent and call views

The following shows the generated inspection shape. Applications read these records rather
than constructing or maintaining execution state themselves:

```python
@dataclass(frozen=True)
class ToolCallRef:
    agent_id: str                  # Context containing the parent call.
    call_id: str                   # Canonical FunctionCall.id, not the provider's wire ID.


@dataclass(frozen=True)
class AgentCallState:
    origin: ToolCallRef            # Parent call that receives the accepted child output.
    parent_invocation_id: str      # Parent invocation that admitted this local call.
    parent_execution_id: str       # Parent workflow/task correlation.
    parent_tool_name: str          # Agent-as-tool function exposed by the parent.
    agent_name: str                # Configured child behavior.
    validated_arguments: dict[str, JsonValue]  # Accepted arguments to that function.
    identity: str                  # Canonical argument identity used by approval.
    status: Literal["approval", "ready", "running"]  # Local admission, not A2A status.
    agent_id: str | None           # Child context ID; absent before the child starts.


@dataclass(frozen=True)
class AgentInputRequest:
    state: InputRequestState       # Response lifecycle for this occurrence.
    source_message: ChatMessage    # Requested question; delivery is tracked separately.
    response_message: ChatMessage | None  # Reserved human response.
    origin: ToolCallRef | None     # Asking call's provenance; never a direct-answer command.


@dataclass(frozen=True)
class AgentState:
    agent_name: str                # Configured behavior; no Agent instance is serialized.
    chat_ctx: ChatContext          # This context's private model history.
    origin: ToolCallRef | None     # Parent call receiving the output; None for the root.
    calls: dict[str, AgentCallState]  # Unresolved calls, indexed by parent call-item ID.
    input_requests: dict[str, AgentInputRequest]  # Retained local input occurrences.
    completed: bool                # An output was accepted; JSON null also counts.
    output: JsonValue | None       # Accepted output retained until parent propagation.
    error: str | None              # Diagnostic failure, separate from successful output.
```

`origin` defines the child-to-parent relationship. A call can await approval before a child
exists, so `AgentCallState.agent_id` can be absent. Once the child starts, that ID selects its
entry in `state.agents`; the child's `origin` identifies the exact parent call it resolves.
Provider call IDs can repeat, so correlation uses canonical `FunctionCall.id` values.

Accepted child output stays with the child until its owned work settles and the parent result
is committed. Propagation fills the parent tool result once, removes the unresolved call and
prunes the completed child. Historical tool results remain in the parent's model context.
`calls` is an unresolved-call index; ordering comes from that context's tool-call order.

`merge_chat_ctx=True` returns only the child's conversation with the person to the parent
model context. Private tools, instructions and synthetic task input stay private. The person's
dialogue remains in `chat_history` whether or not it is merged into the parent.
[TextSession](text-session.md) owns local agent execution and child policy options.

## Remote communication views

```python
remote = app.session.remote_agents["flights"]
message = remote.messages[message_id]
task = remote.tasks[task_id]
context = remote.contexts[task.task.context_id]
events = remote.events
```

The key is `RemoteAgent(id=...)`, defaulting to its canonical URL. The model-facing tool name
is independent of that key. `remote.endpoint` pins the destination; saved addresses and IDs
grant no access.

| View | Contains |
| --- | --- |
| `remote.messages` | Observed SDK messages in either direction, indexed by message ID; includes direct replies without tasks or contexts |
| `remote.tasks` | Observed tasks, indexed by task ID, with task-owned response/cancellation facts |
| `remote.contexts` | Those same messages and tasks grouped by context ID; no second message history |
| `remote.events` | Requests, responses and observed errors in local communication order |
| `task.task` | The upstream SDK Task: status, optional history, artifacts and metadata |
| `task.input_requests` | Input occurrences with state, source message ID and outgoing response message ID |
| `task.cancel_requested` | Explicit local cancellation intent; SDK task status confirms the outcome |

Task and message IDs are scoped to the configured remote. A context groups observations; it
does not create another ID namespace or require a task to exist. These collections describe
retained observations, not the remote server's complete inventory. Inspection performs no
network calls. Returned SDK values and collections are copies.

The journal's sequence is local observation order. Its request ID correlates one immutable
operation with responses/errors; it is not a remote task ID or an SSE replay cursor. Dispatch
markers and request lifetime change beside the log without appending copies of request content.
The native owner checkpoints related state and communication evidence together.

## Input lifecycle

Local and remote input occurrences use the same response lifecycle, independently of A2A's
task status:

| State | Meaning |
| --- | --- |
| `open` | Waiting for a response to this occurrence |
| `responding` | A human response is reserved; its local delivery or remote admission is not yet confirmed |
| `responded` | Local delivery or remote acceptance was confirmed |
| `superseded` | A replacement occurrence retired the unsent selection |
| `closed` | Input is no longer requested; this does not prove a response was accepted |

Presentation, recognition and delivery are separate facts. A question is eligible for automatic
response recognition only after its human-facing presentation is established. A side question
does not answer it. A correction can replace a held response before dispatch; once sending may
have started, recovery reconciles that immutable request. Timestamp or recognition-metadata
normalization does not change the person's words or grant another response.

## Bound and detached inspection

`state.snapshot()` returns a detached `SessionState` fixed at the observed revision. It neither
writes storage nor preserves models, callbacks, credentials, live coroutines or command rights.
Remote views retained from a bound state follow that conversation; views from a detached
snapshot stay fixed.

`await task.cancel()` is the explicit command on a bound remote task. It uses the current
native owner or a short fenced control scope and requires persisted confirmation of remote
cancellation. Inspection alone never dispatches. See
[retained task control](persistence.md#canceling-retained-remote-work).

Rust owns the snapshot encoding and generated Python views. `SessionSnapshot` and
`SessionCheckpoint` are private implementation concepts; applications use `SessionState`,
the existing store binding and the execution APIs.
