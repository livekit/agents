# Shared SessionState

This plan defines the agreed conversation state shared by VoiceSession, TextSession and
Agent.run. The logical schema below is framework-owned; applications use generated views
and store binding. Rust owns transitions and persistence contracts. The implementation
checklist tracks the native encoding and validation work still required.

## Expected experience

A person can start a request by voice and return by text through a supported transition.
Binding the same saved conversation retains its delivered dialogue and remote task/input
references. Durable local Agent-as-tool execution belongs to TextSession and bound Agent.run;
voice handoffs and VoiceTasks remain live interactions. Changing delivery mode does not itself
start remote work again or create another conversation.

Applications should be able to rely on these outcomes:

- The current local agent handles new unaddressed human input, including when it is a child.
  A targeted response keeps its original input-request destination across agent changes.
- The person sees one conversation. Each local agent still has its private model history;
  internal tool activity does not appear as dialogue the person experienced.
- Remote work can continue after the frontend closes. Returning to a stored conversation
  restores references and reconciles current work; uncertain sends or speech remain explicit.
- Inspection shows retained agent/task/message state without performing network calls or
  changing which agent has the floor.
- Framework checkpoints preserve conversation progress. Application fields are saved only
  through explicit application saves; application services are reconstructed by the application.

Recovery covers recorded state and supported continuations. It does not restore Python stacks
or a live VoiceTask await. The [delegation product direction](delegation.md#product-direction)
defines the responsive frontend and remote reasoning experience. The
[conversation expectations](delegation.md#conversation-expectations) own question pacing,
grouped progress and interruption behavior; the
[Rust end state](delegation.md#rust-end-state) maps those expectations to native components.

## Agreed direction

- Share conversation history across execution modes, with a private ChatContext per frame.
- The top-level agent runs in the root frame. Agents called as tools create child frames.
- Persist local child continuations only for text. Voice role changes preserve the durable
  root and human history without saving a resumable local child tree.
- Expose current_agent and root_agent as views scoped to the selected session.
- Store agent states in one internal map; root_agent and current_agent are properties that
  resolve saved IDs into that map, not separate state copies.
- Read each agent's private model context through that view's chat_ctx.
- Expose chat_history as the whole conversation experienced by the person, across agents/modes.
- Omit the ambiguous top-level SessionState.chat_ctx accessor.
- Omit AgentState.answer_tool; continue input through a selected question's recorded target.
- Keep completed, output and error as ordinary fields on AgentState; add no local status enum or
  Python state-variant classes. Remote task status reuses the upstream SDK TaskState.
- Keep application data in the application's context, alongside its SessionState field.
- Reuse SessionState for delegation recovery instead of adding public DelegationState.
- Use store.bind as the single continuation path across VoiceSession, TextSession and Agent.run.
- Send unaddressed human input to the current local agent; retain explicit input-request
  destinations independently of the current agent's available tools.
- With the default detector enabled, recognize responses to previously presented requests
  across agent changes and retain them until dispatch is eligible.
- Keep SessionState as a bound view or detached inspection snapshot; add no Agent.run state
  input or RunResult.state output.
- Preserve explicit application saves.
- Name the per-agent session record AgentState, the call reference ToolCallRef and the
  agent-as-tool execution record AgentCallState; use origin for the initiating call reference.
- Prefix parent-specific invocation, execution and tool metadata with parent_; origin remains
  the agreed parent-call reference. Call-owned arguments, identity and status stay unprefixed.
- Keep local AgentCallState separate from remote communication state. Track a remote agent's
  contexts, messages and tasks explicitly; a communication relationship is not a single task.
- Keep local questions on AgentState and remote input requests on the task requesting input.
  Routing gathers eligible candidates; SessionState has no second global question registry.
- Name the local agent-tool call map calls and index it by the parent tool-call ID. For now,
  chat_ctx is sufficient for ordering; add no OrderedDict or call-order list.
- Keep the frontend presentation queue separate from durable execution/question state.
- Persist mutable request progress, task state and input requests as current state. Set
  dispatch_started durably before network I/O; it does not prove remote acceptance.
- Keep one append-only communication journal for requests, responses and observed errors.
  Dispatch progress and stream lifetime are state updates, not extra journal events. Expose
  public read-only messages, tasks, contexts and events; commit related state/log changes together.
- Reuse a saved default context for new work with a configured remote. Continuing a task keeps
  its own context; another observed context does not silently replace the default.
- `merge_chat_ctx` adds only the child's conversation with the person to the parent's model
  context. It does not control whether that conversation appears in chat_history.

Storage is unreleased. Use one shared encoding and remove the old text/voice tags, aliases
and migration readers. SessionState snapshots are for inspection; store.bind is the only
continuation input. TextSession returns RunResult directly and has no state= argument.

## Root, current agent and history

The root frame represents the top-level conversation. Its identity is not a Python Agent
object identity, an invocation ID, a remote A2A task ID or a room binding key. Agent behavior
is supplied by configuration; saved frames identify which configured agent they use.

For example:

```text
root: front_desk
  ChatContext: person's conversation with the front desk
  child: booking_expert
    ChatContext: expert's private conversation and tool activity
```

Root and current are different relationships. If a local child is collecting input,
current_agent may refer to that child while root_agent still refers to front_desk. After the
child completes, its accepted output fills the parent's tool call and the parent continues.

Remote delegation is different: the frontend can remain current_agent while several remote
experts work or ask questions. Those experts own their remote histories. A remote question
does not implicitly make that expert current_agent. Only the input/output exchanged with the
person through this frontend belongs in this conversation's chat_history.

Voice and text using the same frame read and advance the same ChatContext. Changing execution
mode neither forks history nor creates another root. A new frame exists for a separate agent
conversation, not simply because another input/output mode is used.

Child histories retain fresh/fork and merge policies. A fresh child does not receive the
parent's private tool history. A fork starts from the policy-selected parent history and then
diverges. The agreed `merge_chat_ctx` policy returns only the child's conversation with the
person: user input and assistant content actually delivered, including interrupted prefixes.
Exclude private tool calls/results, instructions, synthetic task input and inherited context.
Preserve item identity and apply each transfer once. The accepted child output still resolves
the parent's tool call whether or not dialogue is merged. The child's dialogue remains in
chat_history even with merge_chat_ctx=False: the person experienced it regardless of what the
parent model receives. The option is named `merge_chat_ctx`; there is no `merge_chat_history` alias.

## Public state views

```python
state.current_agent.chat_ctx  # Current agent's private model context.
state.root_agent.chat_ctx     # Top-level agent's private model context.
state.chat_history            # Whole conversation experienced by the person.
```

current_agent and root_agent are read-only views scoped to this session. They identify agent
contexts, not merely reusable Agent configurations: the same configured agent may participate
in multiple contexts with different histories. Inspecting a saved root or inactive child must
not require that agent to be executing in the current Python callback. These views neither
serialize Agent objects nor add mutable conversation data to shared Agent configuration.
Their concrete return types remain to be specified; applications do not construct new view
objects to manage execution.

chat_history is a ChatContext containing accepted user messages and assistant content actually
delivered, in conversation order across agent switches and supported mode changes. Include
interrupted prefixes; exclude private tool calls/results, instructions, synthetic task input
and duplicate inherited history. It covers the selected conversation, not every partition
or private expert conversation in a thread.

For example, chat_history includes the entire exchange below, even if the child does not merge
its dialogue into the root agent's model context:

```text
Person:      Find me a flight.
Front desk:  Let's check.
Child:       What date?
Person:      Friday.
Child:       I found a flight.
Front desk:  Shall I book it?
```

The parent model does not automatically receive chat_history as its input. Model execution
uses that agent context's chat_ctx, with the child's dialogue added only according to the
merge policy. Reading the conversation history does not change that policy.

## Internal representation

The public views do not require independent copies of every message. Preserve canonical item
identities and record context membership and delivery facts, so each agent's model context
and the conversation history can be reconstructed without concatenating private histories.
The human-facing content may be an interrupted prefix of generated content; one item identity
must not erase that distinction. Preserve the conversation-level projection when a child is
removed, independently of whether its dialogue was merged into the parent context.

The logical records below define the target relationships. Frame remains an internal term;
applications inspect AgentState. Concrete native serialization follows the initialization
and retention rules below. ABI declarations must give every stored field a short
purpose/ownership comment, including the invariant that recovery depends on.

### SessionSnapshot, live views and checkpoints

SessionSnapshot is the private Rust data representation behind
SessionState. One encoding serves text, voice and bound Agent.run.
Applications do not construct another snapshot type or keep it synchronized with SessionState.
There is one logical conversation state with different execution and inspection interfaces.

| Name | Role |
| --- | --- |
| Session | Native decision machine advancing the conversation and emitting effects. Its live execution also has transient state. |
| SessionSnapshot | Recoverable conversation data at a coherent revision; contains no model objects, runtime tasks, connections or command capabilities. |
| SessionState | Generated public view. A bound view follows the owner's state; a detached view stays fixed. |
| SessionCheckpoint | Private acquired scope retaining storage ownership from reload through an atomic commit of framework state and related content/history/task/journal changes. Dropping it releases ownership. |
| state.snapshot() | Existing public inspection operation returning a detached SessionState; it neither writes storage nor returns a new public SessionSnapshot class. |

The recoverable data includes root/current agent identities, local agent/call/input state,
accepted outputs, admitted child policies, remote task/request/input state, and the references
needed to reconstruct private model contexts and the person's delivered history. Immutable
messages/content and the communication log retain their canonical storage identities. The
snapshot can refer to those records and indexes instead of embedding another complete copy
of every history or journal event. A detached view captures its retained values; it can share
immutable payloads internally while remaining unchanged as the live conversation advances.

For example, after a flight question is presented and the person answers Friday:

1. The native machine accepts the human message once and reserves its response against the
   original input occurrence. The recoverable state records that reservation.
2. A checkpoint groups the snapshot change with its related new history/content/task/event
   writes. The fenced transaction makes those changes durable together.
3. A fresh owner after a crash loads the committed data and canonical references, reconstructs
   native coordination and receives fresh application configuration/services.
4. It can retain the response while the flight tool is unavailable, or validate and dispatch
   it once eligible. If a send may already have reached the peer, it reconciles the recorded
   operation rather than blindly sending the answer again.

Constructing a snapshot or reading state.snapshot() does not itself make progress durable.
Durability comes from the checkpoint's successful storage receipt. Recovery starts from the
last committed state; it does not restore the previous process's Python coroutine or voice
connection. The separate configured routing model and bound RemoteTask command capability
are supplied by the new runtime, not serialized into SessionSnapshot.

The schema/codegen slice proves the concrete shared encoding. SessionSnapshot is private
recoverable data. SessionCheckpoint is a private runtime/storage scope with a commit operation;
it is not serialized or exposed as a public state class. The native decision machine emits
changes, and the I/O layer executes their checkpoint. Neither adds another public conversation
object or an event-sourced database.

### Initialization and retention

- Allocate one root agent-state ID from supplied identity inputs when a selected conversation
  has no state. Root/current references must resolve before execution. Preserve IDs across
  supported mode changes and require restored configured agents to resolve; fail explicitly
  rather than silently replacing a missing child or resetting the root.
- Store admitted child options with its continuation. After accepted output reaches the
  parent's tool result and the permitted dialogue merge is durable, remove temporary child/
  call state. Preserve the person's conversation independently of child removal or merge policy.
- Public bound histories are read-only views. Existing supported agent/session history-editing
  commands go through the owning machine and preserve canonical identity/ordering. Detached
  snapshot inspection performs no writes and creates no second continuation source.
- Maintain one shared snapshot encoding; storage has never been released and needs no old
  text/voice readers. Unknown encoding versions fail explicitly.
  Canonical protocol storage and relational schema remain declared once through storage codegen.
- Retain canonical items and operation metadata while unresolved calls, input reservations,
  uncertain dispatch, undelivered outcomes or retained journal entries reference them. Prune
  settled data only through the storage retention policy and report expired inspection scope
  truthfully. Do not discard active references to fit a checkpoint or duplicate full histories.

## Concept boundaries

```text
SessionState
  local agent states
    calls -> local child agent states
    input_requests -> local requests for human input
  remote agent states
    events -> append-only communication history: requests, responses and errors
    internal request state -> dispatch progress, lifetime and local caller/input correlation
    messages -> observed A2A Messages in both directions
    tasks -> A2A task views, input requests and task-specific cancellation
    contexts -> grouping views over the same messages and tasks
  chat_history -> conversation experienced by the person

Live frontend
  presentation queue -> events waiting for an eligible current agent
```

AgentCallState represents an invocation of a parent's tool implemented by a local agent.
RemoteAgentState represents this conversation's communication state with a configured remote
agent. It can track multiple remote contexts, multiple tasks per context and message-only
exchanges. Each outgoing message, rather than the entire remote relationship, retains its
originating local tool call. Both kinds of invocation still appear in the caller's chat_ctx.

Do not add local/remote switches to AgentCallState or create fake local AgentState records for
remote agents. Share small references and matching approval/routing/delivery policies, without
forcing the same persisted execution record onto both paths. Reuse/evolve existing Rust
RemoteInteraction/InteractionState decisions at their appropriate per-request/task scope;
do not retain a second independent remote lifecycle implementation. Long's final behavior
remains the porting reference; his Python helpers are not copied one for one.

## Combined working structure

These dataclasses show the logical relationships together, not the serialization layout or
application constructors. root_agent and current_agent are views over the internal records;
they can refer to the same record. chat_history is the conversation projection described
above, not an independent copy of every chat item. The internal map owns each agent state;
root and current IDs identify entries in it. Native snapshot records define the persisted
layout and initialization; this sketch describes the relationships applications can inspect.

```python
from collections.abc import Mapping
from typing import Literal, TypeAlias
from pydantic import JsonValue


InputRequestState: TypeAlias = Literal[
    "open",        # Eligible to receive a response.
    "responding",  # A response is reserved; delivery/continuation is still pending.
    "responded",   # Response admitted; its content may still require validation.
    "superseded",  # Explicitly replaced by another input request.
    "closed",      # No longer accepts a response.
]


@dataclass
class SessionState:
    _agent_states: dict[str, AgentState]         # Canonical agent-context ID -> state.
    _root_agent_id: str                          # Top-level context's map key.
    _current_agent_id: str                       # Foreground context's map key.

    chat_history: ChatContext                    # Conversation experienced by the person.
    remote_agents: Mapping[str, RemoteAgentState]  # Read-only configured remote ID -> state.

    @property
    def root_agent(self) -> AgentState:
        return self._agent_states[self._root_agent_id]

    @property
    def current_agent(self) -> AgentState:
        return self._agent_states[self._current_agent_id]


@dataclass(frozen=True)
class FrameOptions:
    merge_chat_ctx: bool = False                 # Merge only dialogue with the person.


@dataclass
class AgentState:
    agent_name: str                              # Configured agent to run.
    chat_ctx: ChatContext                        # This context's private model history.
    options: FrameOptions                        # Policies fixed when the child starts.

    # Return this child's final result to the parent agent's tool call; None for root.
    # Human answers use the question's continuation, not this reference.
    origin: ToolCallRef | None
    # This agent's calls to local agent tools, keyed by canonical FunctionCall.id.
    # Read call order from chat_ctx; retain entries until result propagation settles.
    calls: dict[str, AgentCallState]
    input_requests: dict[str, AgentInputRequest]  # Occurrence ID -> local request for input.

    completed: bool = False                     # True once this invocation has accepted output.
    output: JsonValue | None = None              # Accepted output; None is valid when completed.
    error: str | None = None                     # Failure detail, if this invocation failed.


@dataclass(frozen=True)
class ToolCallRef:
    agent_id: str                               # Caller context in SessionState's agent-state map.
    call_id: str                                 # Canonical FunctionCall.id; provider IDs may repeat.


@dataclass
class AgentCallState:
    """One call to a parent's tool exposed through child_agent.as_tool()."""

    # Return the child agent's accepted result to this tool call in the parent.
    # Human answers use the question's continuation, not this reference.
    origin: ToolCallRef
    parent_invocation_id: str                    # Parent invocation that made this tool call.
    parent_execution_id: str                     # Parent workflow/task correlation.
    parent_tool_name: str                        # Tool exposed by the parent, e.g. book_hotel.
    agent_name: str                              # Child agent implementing that tool.
    validated_arguments: dict[str, JsonValue]     # Validated tool inputs, passed as the child's task.
    identity: str                                # Canonical validated-argument identity.
    status: Literal["approval", "ready", "running"]  # Call admission/launch progress.
    agent_id: str | None                         # Local AgentState key; None until started.


@dataclass
class AgentInputRequest:
    state: InputRequestState                     # Response selection/continuation progress.
    source_message: ChatMessage                  # Input requested by this local agent.
    response_message: ChatMessage | None         # Selected response; None until reserved.
    origin: ToolCallRef | None                   # Asking tool call, if any; None for model input.
    # The owning AgentState identifies the local context to resume after routing.
```

AgentState is the chosen name for per-agent session state. It backs the root_agent and
current_agent views; applications do not construct an additional execution object. FrameOptions
remains a working internal name. Save the child's options snapshot with its continuation
state: restoring a pending child must not reread changed tool settings and change merge policy.

Map keys identify particular agent contexts, not configured agent names. Two calls to the same
configured agent can have separate entries and private histories. The framework owns the map
and changes the current-agent reference as execution moves between contexts. Both properties
resolve the same entry when the root is current; applications do not assign those properties
or mutate the internal map. Validate that both references resolve when restoring state.

The current SDK already names its transient activity enum AgentState. Resolve that naming
collision with an explicit migration before changing ABI exports; do not silently shadow it.
The replacement activity-enum name is still open. No implementation rename is made here.

ToolCallRef and AgentCallState are the agreed names for the originating-call reference and
agent-as-tool execution state. origin names the initiating call reference; it identifies
provenance, not the pending question to answer. A question occurrence and its owning interaction
must be selected and validated separately; origin alone cannot route a human answer.
validated_arguments contains the arguments supplied to that
tool call after parsing and validation; it is not the application context. The current local
child receives those arguments as its task input. The call map is named calls; the exact
cleanup boundaries remain open. Question-specific continuation replaces the
standalone answer_tool field. AgentState keeps ordinary completed/output/error fields for the
invocation result and failure detail. AgentCallState tracks admission and the associated agent state: agent_id must be absent
before launch and resolve into the agent-state map while running. agent_name selects the
configured agent; agent_id identifies the particular state executing this call. The framework
validates these combinations. Remote communication records are drafted below.

The local calls map provides direct lookup by the parent tool-call ID. The parent's chat_ctx owns
the order of its recorded FunctionCall items; scheduling reads those items in order and looks
up the calls still pending in the map. Do not reconstruct model order from map iteration,
including after serialization/reload. Keep originating call items available while unresolved;
history compaction must preserve the references and ordering needed to continue them.

Do not restore the old visible_dialogue list as another independent history in this target
sketch. Delivery evidence belongs with canonical chat items/projections. Fork provenance and
idempotent history merging still need concrete records before removing the existing starting
and transferred ID bookkeeping from the implementation. The field discussion stays outside
this plan; this block preserves the working structure and marks unresolved decisions.

Incoming-message routing follows [input-request continuation](#input-request-continuation).
The current local agent handles unaddressed input. Default response recognition also covers
previously presented remote questions while their tool is absent. current_agent is an
inspection view; eligibility and response validation belong to the native routing decisions.

## Accepted output and settlement

AgentState exposes three normal Python fields: completed, output and error. completed means
accepted output is available, not merely that work stopped. An input-required, failed or
canceled run is not a successfully completed invocation. error retains failure detail for
inspection; its text is not used to infer error kinds or decide retries. Existing typed
call/task outcomes still own failure classification and cancellation.

The framework updates the fields together: success sets completed and output and clears error;
failure leaves completed false, clears output and records error. A pending invocation has no
error or accepted output. Cancellation remains identified by its existing call/task outcome.
Check completed to distinguish an accepted None
output from no accepted result yet. For a child, they describe the invocation whose output
returns through origin. For the reusable root, they describe its current/latest invocation;
a completed root turn does not end the conversation. A new invocation resets them before work
starts. These are framework-managed views, not fields applications mutate to signal completion.
They describe durable local invocations. Transient voice roles, VoiceTasks and speech completion
use their live results and delivery receipts rather than creating another durable invocation.

A2A task status is separate: TaskUpdate.state and Task.status.state reuse the upstream SDK's
TaskState, including interrupted and unspecified observations. There is no framework copy of
its enum values. The AgentCallState admission states describe different decisions.

Current TextSession already saves the accepted child output as a FunctionCallOutput in the
parent's history, using origin's call ID, before marking the retained child completed. It
serializes structured output as JSON text. The proposal makes call lifecycle/result ownership
explicit; it does not introduce output persistence for the first time.

Settlement must durably retain the accepted result, resolve the originating caller's tool call,
apply the merge policy and release the settled child's state. Keep the accepted child result
until caller propagation, delivery and cleanup obligations settle. Commit caller-result/history
changes and child removal together at the final settlement boundary. If an earlier checkpoint
retains completed=True and output before propagation, recovery must propagate that accepted
output rather than re-execute the child to obtain another result.

The caller's recorded tool result retains the output after temporary child/call state is
released. Agent-state and history projections must refer to the same accepted result or be
updated atomically; do not create two independently mutable output sources. Human-facing
chat_history still omits private tool results and retains the delivered conversation.

## Input-request continuation

Do not store a session-wide instruction that invokes a named tool with whichever message
arrives next. A side question or unrelated request must not become tool input merely because
an earlier invocation asked for information.

Keep the specific input request open, with its occurrence identity, owning interaction/call
reference and continuation target. Only after routing selects that request may the framework
reserve it and deliver the answer. Ambiguous input leaves the question open and permits
clarification; stale or already-reserved requests cannot consume another answer. origin records
where the work began, not which question a message answers. A lost send acknowledgement retains
the same answer identity for reconciliation rather than reopening it for a second send.

AgentState.input_requests owns local AgentInputRequest records. RemoteTask.input_requests owns
remote InputRequest records. They share selection rules and InputRequestState, without a
global SessionState.questions table. Routing gathers transient candidates with a qualified
InputRequestRef and reserves the actual owner's record. Local ownership identifies the context
to resume; a remote reference identifies the configured remote agent, context, task and request.

InputRequired accepts the question and verbatim policy, without a tool parameter. It records
the need for input; it does not choose a tool to receive whichever message arrives next.
For local work, the matched response resumes the owning agent context, whose next invocation
interprets it and chooses any subsequent tools. For remote work, the matched response is sent
to the recorded A2A task/context. The asking tool has already returned; its Python coroutine
is not resumed and its origin is not an instruction to invoke it again.
Exact local continuation and presentation correlation fields still need native encoding.
Remove the old answer_tool fields and direct-dispatch paths as part of this migration.
The current local agent handles unaddressed human input. Once a response's target is known,
its durable reservation can wait while that agent lacks the matching remote tool. Reference
the canonical human message internally, then create the outgoing A2A message when dispatch is
eligible; response_message_id continues to identify that outgoing message. Revalidate the input
occurrence and access before sending, without appending the human message to history twice.
With the default detector enabled, input admission recognizes responses to previously
presented, unresolved requests in the authorized conversation even while their remote tool is
absent. This is conditional on pending presented requests, not a requirement to route every
utterance through a remote agent. New work and unmatched messages remain with the current
agent. Reuse the same reservation if a later tool call refers to the retained response.

## Remote agents, messages, contexts and tasks

Remote communication is separate from local child execution. A2A contexts group related tasks
and independent messages. Each task has its own status, artifacts and lifecycle. A remote agent
can reply with a Message without creating a task, and several tasks can run in the same context.
Continuing input-required work addresses its task; terminal follow-up starts new work, possibly
in the same context. A remote relationship is not a single task or originating tool call.

The existing delegation tool selects known work with an optional `task_id`; its argument
contract lives in [targeting a remote task](delegation.md#targeting-a-remote-task).
Input-occurrence IDs stay in framework routing and recovery state. The remote agent interprets
task messages and owns which information is still missing, including several answers requested
within one task.

### What we reuse from A2A

The official SDKs separate protocol models, client operations and server task storage. This
proposal follows those boundaries and adds conversation-scoped recovery and routing:

| A2A model or operation | Use here |
| --- | --- |
| Message with optional contextId/taskId | Preserve messages in both directions, including exchanges without a task. |
| Task with status, artifacts and optional history | Reuse the protocol Task as the current observed task projection. |
| StreamResponse | Preserve Task, Message, status-update and artifact-update variants before normalization. |
| Context identifier | Group observed messages and tasks; our context view is not a new wire object. |
| Send, get/list tasks, cancel and subscribe | Journal requests and outcomes, including operations that send no Message. |

Sources reviewed: [Rust models](https://github.com/a2aproject/a2a-rs/blob/700a6a28eb001f6fc32bc9446078ff87bdd18c6c/a2a/src/types.rs),
[stream events](https://github.com/a2aproject/a2a-rs/blob/700a6a28eb001f6fc32bc9446078ff87bdd18c6c/a2a/src/event.rs),
[client operations](https://github.com/a2aproject/a2a-rs/blob/700a6a28eb001f6fc32bc9446078ff87bdd18c6c/a2a-client/src/client.rs)
and [Python TaskManager](https://github.com/a2aproject/a2a-python/blob/caab6a252958b2218ea61c0cae772f0a5c159e2a/src/a2a/server/tasks/task_manager.py).
The task manager demonstrates applying events to a task snapshot. Our bidirectional journal,
local tool-call origins, answer reservations and presentation tracking are framework additions,
not concepts supplied by that SDK. These source references guide the design; they do not
require upgrading our pinned SDK dependency to its latest main branch.

### Public inspection views

```python
remote = state.remote_agents[remote_id]
remote.messages                         # messageId -> observed A2A Message, both directions.
remote.tasks                            # taskId -> observed task and local routing state.
remote.contexts[context_id].messages     # The same messages, filtered by contextId.
remote.contexts[context_id].tasks        # The same tasks, filtered by contextId.
remote.events                           # Append-only requests, responses and observed errors.
```

These are framework-managed views with read-only data. Applications can issue the explicit
cancel command through a bound RemoteTask; they do not assign task state to control execution.
Bound views advance with the owner; detached snapshots remain
inspection snapshots. Tasks and input requests expose persisted current state; message indexes
refer to communication content, and contexts group the same messages/tasks. Select outgoing
messages through their send-request entries. Local origin, answer correlation and dispatch
progress belong to durable internal request state. The journal is history, not the state machine.

All views describe what this conversation has observed and retained. They are not a complete
inventory of the remote server. Task history is optional and can be limited; it is not a full
communication log or the person's chat_history. Contextless messages stay in remote.messages.
A context can contain messages and no tasks. A task observed by polling can exist without its
initial message being known. Listing tasks adds observations; it does not prove unseen work
is absent, and inspecting these views performs no network I/O.

### Reusing protocol types

Reuse existing protocol Message, Task, request, response and StreamResponse types. Their
lossless generated Python exposure and persistence encoding use the SDK codecs;
A2AMessage below means the A2A Message shape, not a second model of its fields. The current
host Message adapter must preserve the required protocol fields when used for these views.
The aliases below only shorten the schema sketch; they introduce no new protocol objects.
TaskUpdate.state uses the SDK TaskState in both Rust and generated Python, including
auth-required/unspecified observations, rather than converting through the storage task enum.

Source method identities from the pinned SDK's `a2a::jsonrpc::methods` constants, such as
`CANCEL_TASK`. Store the operation/method on the durable internal request record; journal
entries refer to that record through request_id. The SDK exports constants rather than an
A2AMethod enum. Any required closed operation discriminator stays internal to the ABI, maps
to those constants and generates its host representation; there is no separately maintained
public method type or handwritten Python Literal list. Keep streaming versus unary send
explicit because both use SendMessageRequest.

A2ARequest and A2AResponse are local groupings of SDK types, not imported SDK-wide unions or
types scoped to one task. The sketch lists common conversation operations; implementation must
generate a complete mapping for every supported client method, including card/push operations
when used. The illustrative list does not decide what gets logged. SendMessageRequest can start
a message-only exchange, create work or continue an existing task; task methods inspect/control
that work. Reuse SDK types when adding operations rather than maintaining a replacement schema.

### Expected response shapes

We know the protocol operation, not which valid outcome or content the remote will return.
Validate each response against its request rather than accepting every member of the journal's
response union for every method:

| Operation | Existing SDK success type |
| --- | --- |
| Send message | SendMessageResponse: either Message or Task. Preserve the SDK union. |
| Send streaming message | StreamResponse: Message, Task, status update or artifact update. |
| Get task / cancel task | Task; cancellation still requires observing its status. |
| List tasks | ListTasksResponse, including pagination. |
| Subscribe to task | StreamResponse, associated with the subscribed task. |

Protocol failures reuse SDK A2AError codes/details; transport failures use the existing typed
client error boundary. Neither is a successful Task or Message. Preserve parts, structured data,
files, metadata and supported extensions without assuming text or a fixed business schema.
A business object or file is content inside Message.parts or Task.artifacts, not a new response
union member. Application output validation is separate from decoding the A2A envelope.
Preserve unknown optional metadata. Unsupported protocol versions, required extensions or
malformed response shapes fail explicitly with bounded diagnostic evidence; do not invent
tasks, silently discard content or treat arbitrary JSON as a valid A2A response envelope.

### Logical records

```python
from collections.abc import Mapping, Sequence

# Protocol names below denote the existing A2A SDK types.
# Common SDK operations shown; the generated aliases must cover every supported client method.
A2ARequest: TypeAlias = (
    SendMessageRequest | GetTaskRequest | ListTasksRequest
    | CancelTaskRequest | SubscribeToTaskRequest
)
A2AResponse: TypeAlias = SendMessageResponse | StreamResponse | Task | ListTasksResponse


# Communication event kinds, generated from ABI; separate from request/task lifecycle state.
A2AJournalKind: TypeAlias = Literal[
    "request_recorded",   # Outgoing request content recorded; sending is tracked in request state.
    "response_received",  # SDK result/stream item observed; task status is in the payload.
    "request_error",      # Observed protocol/client error; does not prove remote work failed.
]


@dataclass
class RemoteAgentState:
    # The containing remote_agents map key is remote_id; scope it to this conversation/access.
    endpoint: str                                # Pinned A2A destination; no credentials.
    events: Sequence[A2AJournalEntry]             # Append-only communication history.

    @property
    def messages(self) -> Mapping[str, A2AMessage]:
        ...  # messageId -> observed Message in either direction; no local metadata wrapper.

    @property
    def tasks(self) -> Mapping[str, RemoteTask]:
        ...  # taskId -> persisted current task and task-owned local routing state.

    @property
    def contexts(self) -> Mapping[str, RemoteContextState]:
        ...  # contextId -> grouping view over these same messages/tasks, not a wire object.


@dataclass
class RemoteContextState:
    messages: Mapping[str, A2AMessage]             # Same Message views, filtered by contextId.
    tasks: Mapping[str, RemoteTask]                # Same task views, filtered by contextId.


@dataclass
class RemoteTask:
    task: Task                                    # Existing A2A Task; persisted current observation.
    input_requests: Mapping[str, InputRequest]    # Occurrence ID -> task-owned request for input.
    # Explicit local intent for THIS task, not its context or remote agent.
    # Only observed task state confirms cancellation; frontend close does not set this.
    cancel_requested: bool = False

    async def cancel(self) -> None:
        ...  # Bound native command; success requires observed cancellation to be persisted.


@dataclass(frozen=True)
class A2AJournalEntry:
    sequence: int                                 # Stable local position; not remote causal order.
    request_id: str                               # Stable client-operation ID, shared by all its events.
    kind: A2AJournalKind                           # Request, response or observed communication error.
    payload: A2ARequest | A2AResponse | A2AError | AgentClientError  # Native content.


@dataclass(frozen=True)
class RemoteTaskRef:
    context_id: str                                # Remote A2A context, not a local thread ID.
    task_id: str                                   # Particular task inside that context.


@dataclass
class InputRequest:
    state: InputRequestState                       # Local response handling, not the A2A task status.
    source_message_id: str | None                  # Remote message requesting input; None if absent.
    response_message_id: str | None                # Outgoing A2A message reserved as its response.
    # The owning task supplies context/task identity; recovery reads this state directly.
```

AgentClientError denotes the existing typed client failure model, not a new string status.
Its durable representation must retain the failure kinds needed by recovery, including
transport failures, without persisting credentials or exception objects. Preserve peer protocol
refusals as the SDK's A2AError, including code/details, before any existing lossy client mapping;
never recover their meaning by parsing an error message.
The dataclasses describe logical fields; public collections and nested protocol values cannot
be mutated to bypass the owning machine.

RemoteTask is the generated task view with an explicit cancel() command; its task field is the
existing SDK Task value. The bound view carries a transient command capability, not serialized
credentials or a connection in the task record. cancel() resolves the authorized conversation
owner through that binding and uses its normal fenced write/dispatch path. Detached snapshots
support inspection; calling cancel() on one fails explicitly rather than opening a connection
from saved data. A fresh bound view is obtained when returning to the conversation.

The native control path serializes with an active owner, or takes a short fenced control turn
when no conversational execution is active. It does not start a model turn. Awaiting cancel()
succeeds only after the peer is observed canceled and that outcome is committed. Refusal,
terminal races and unresolved delivery are explicit existing error outcomes. No separate
public task controller or cancellation-result class is added.

SessionState.remote_agents is keyed by configured remote identity, not by task or connection.
Resolve access through runtime configuration and validate the pinned endpoint on restore;
changed configuration must not silently redirect saved work. A remote contextId is opaque,
not automatically our thread database ID. Task IDs are indexed within this remote scope;
validate that every task's context agrees with its context view.

For new work, reuse this conversation's saved default context for that configured remote.
Keep the selection explicit in native state rather than inferring it from the latest event or
task. Contextless exchanges remain valid when no context is known. Responses to an existing
task always use that task's context, even when it is not the default. An explicit separate
context is supported by the design; its API spelling and initialization remain to finalize.
Observing another context does not silently change the saved default.

### Message and request correlation

Record a SendMessageRequest before any remote task ID need be known. Its Message can continue
an existing task, start work in a context, or receive a message-only reply. Preserve its stable
messageId and protocol addressing, including referenceTaskIds. Durable request state retains
origin, the local tool call; later sends in the same task/context can have different origins.
No relationship-wide origin stands for every invocation.

request_id identifies one logical client operation and its durable request record. Every event
for that operation carries it, including the outgoing request. sequence identifies an individual
journal event; it does not identify the operation. Scope request IDs to this conversation's
remote state and keep them stable across persistence, restart and safe retries. Separate polls
and reconnect subscriptions are separate operations with their own IDs.

This is local operation identity, distinct from A2A messageId/taskId/contextId and from an
InputRequestRef's input-occurrence ID. Map transport RPC IDs to it where needed; a wire ID that
can be reused across connections is not sufficient by itself. The local ID is not a remote
idempotency guarantee. It associates responses/errors with the originating operation, not proof
that this request caused every observed task update. Preserve original-call correlation needed
for later delivery without fabricating Messages for control operations.

Index outgoing Message payloads and incoming direct messages, Task.history and TaskStatus.message
by messageId, preserving their original appearances in events. Repeated appearances are not
new messages. Define compatible enrichment when the server adds addressing or metadata, while
preserving the exact original wire payloads. Conflicting content for one identity must be
surfaced/reconciled rather than silently overwritten. Journal kind identifies requests versus
responses; durable request state supplies dispatch facts. Message role alone supplies neither.

InputRequest is local response/recovery tracking for a task in input-required state, not an A2A
wire object. source_message_id identifies the optional TaskStatus.message from the Task snapshot
or status update that opened this occurrence. Resolve it through remote.messages; None is valid
when the peer requests input without a message. Keep the reference even when later task status
changes. Recovery reads the current input state and canonical message, independently of a
particular journal entry. A later occurrence gets a distinct local input-request ID; identical
wording is not identity, and a repeated snapshot must not create a new occurrence.

Use response for a message addressing an explicit request, including InputRequest. Use reply for
general conversational output. Local AgentInputRequest follows the same source/response naming;
SDK and provider-defined names retain their protocol meaning.

Selecting a response reserves the task-owned input request before dispatch. response_message_id
then identifies the exact outgoing A2A Message in remote.messages, not the human ChatMessage or
the transport request ID. Both message IDs refer to canonical messages rather than embedded
copies. Persist the response message/reference with its reservation and retain the same identity
while delivery is uncertain. Internal request state retains the selected InputRequestRef.
Validate the remote/context/task/request tuple. dispatch_started or response_message_id alone
does not prove remote admission or semantic satisfaction. Confirmation/reconciliation advances
the input state to responded; ambiguous input leaves it open. Do not create a new task to
escape an uncertain response send.

Before dispatch starts, a replacement question or a task that stops requesting input retires
the unsent response. Revalidate that reservation at the dispatch checkpoint. After dispatch
may have started, keep its original correlation until peer evidence resolves admission.
An admitted response can leave the same task input-required again: create a new occurrence,
even if its question message ID is reused or absent. Repeated observations of an unanswered
occurrence do not create extra questions. Responded means admitted, not that the input sufficed.

### Current state and communication history

Store changing facts as current state, using the same ownership model as tasks and input_requests:

| Owner | Durable current state |
| --- | --- |
| Task | Latest observed SDK Task and local cancel_requested intent. |
| Input request | Response state and canonical source/response message references. |
| Internal request record | Stable request_id, SDK operation/method, dispatch progress, local lifetime and caller/input correlation. |
| Journal | Immutable request, response and observed-error payloads, in local observation order. |

Evolve the existing Rust request machinery for the internal records; add no public request-state
primitive. Key them by stable request_id and associate the journaled SDK request body with that
ID so the body is stored once. The operation/method belongs to this record, not each journal
event. dispatch_started is a durable mutable field on that record, set before network I/O; it
means an attempt may have happened, not that the peer received it. Response/error references and
stream lifetime belong to that state too. Exact internal fields and retry-attempt bookkeeping
remain implementation details to finalize. Core owns transitions; runtime executes storage and
network effects. No dispatch-start or stream-end entry is appended merely to change local state.

If an observer fails before dispatch starts, a fresh observer can prepare the unchanged
request again with the same message and operation identities. Reopen its local stream lifetime
without duplicating request content or deleting earlier errors. Historical errors do not
determine the current stream's ended field. A retired input reservation instead requires
reconciliation; it does not regain permission to send just because no attempt started.

Local failure and remote outcome are independent facts. Once dispatch_started is persisted,
an error or local stream/call termination must not clear it or establish that nothing happened
remotely. A locally finished request with no conclusive peer response remains unresolved for
recovery; keep its state and referenced payload/error until reconciled or explicitly retained
as unknown. No additional public outcome enum is introduced by this rule.

| Durable evidence | What recovery can conclude |
| --- | --- |
| dispatch_started=False | No network attempt began under the enforced write-before-I/O rule. |
| dispatch_started=True; connection reset/timeout; no conclusive response | The peer may have acted. Local failure does not authorize blind resend. |
| dispatch_started=True; peer response persisted | Interpret that protocol response; it does not prove earlier business effects were undone. |

The journal contains communication content and correlation. A Task reporting failed status is
response_received; an operation/protocol/transport failure is request_error. Neither journal
kind determines remote task status by itself. Stream closure updates local request state and
does not complete or cancel remote work. All client operations performed for this conversation
pass through the same recording boundary; the aliases above illustrate the common operations,
and the generated operation/type mapping must cover every supported client method.

### Task cancellation and crash recovery

cancel_requested is persisted local intent on this task. It does not mean the peer canceled,
all tasks in a context should stop, or earlier business effects were undone. Cancellation uses
the same durable request machinery as sends, polls and subscriptions:

1. In one fenced write, save cancel_requested, the internal request record and its journaled
   CancelTaskRequest. Confirm the write before continuing. This records intent, not dispatch.
2. Persist dispatch_started on the request record before attempting network I/O. A crash after
   this point can leave peer receipt unknown; recovery must reconcile rather than blindly resend.
3. Append the actual response or typed error and update the request/task state together. A peer
   refusal or lost response remains distinguishable from observed canceled status.
4. Resolve awaited cancellation as confirmed only after observing Task.status=canceled and
   persisting that observation. Surface refusal or uncertainty instead of claiming cancellation.

Framework writes use the existing idempotent storage retry/fencing mechanism and do not save
arbitrary application mutations. State and associated journal events commit atomically; a
failed or uncertain database write is resolved with the same write identity before dependent
network work or caller success. A response obtained before a crash but not durably committed
may be lost locally; recover the task state from the peer rather than assume its outcome.

For one cancellation, history and current state look like this:

```text
Communication journal (append-only):
  41 request_recorded  request_id=R1, payload=CancelTaskRequest(T1)
  42 response_received request_id=R1, payload=Task(T1, canceled)

Current state (updated in place):
  request R1: method=CancelTask, dispatch_started=True, local operation finished
  task T1: cancel_requested=True, task.status=canceled
```

A crash before entry 42 leaves the durable request record describing dispatch uncertainty.
The journal does not need an additional marker for that state change. Internal records retain
caller/result and answer correlation until settlement; the journal need not carry those fields
on every event. Before a task ID is known, local waiting can stop, but cancellation cannot claim
to stop unknown remote work. Task follow/replacement keeps separate identities.

### Communication log and recovery boundaries

Keep SDK SendMessageResponse and StreamResponse unions, protocol error codes/details and artifact
append/replacement semantics. Record responses before lossy normalization; update the current
Task with each accepted observation rather than appending synthetic cumulative TaskUpdate copies.
Outgoing addressing can group messages but cannot fabricate a task or remote acceptance.
Validate event kind and required request_id at the recording boundary. Resolve the request's
method from its durable record to validate payload/result shapes and operation correlation.
Credentials, headers, Python stacks and transport framing are not conversation history.

Restart loads durable request, task and input-request state, then reconciles unfinished or
uncertain work with the peer. It does not derive dispatch progress by replaying the communication
log. Replay can rebuild content indexes or debug history; it must not resend requests, repeat
business effects, reapply artifact appends or speak saved messages. A task snapshot can reconcile
current state without recovering missing historical events. A working task snapshot alone does
not prove that a particular message was accepted. If the peer cannot establish the outcome,
retain uncertainty rather than inventing success, failure or safe retry.

For the later durable stream-replay extension, internal subscription state retains a
peer-supported resume cursor with its remote/task scope.
Advance it atomically with the accepted incoming event and resulting state, not merely when bytes
arrive. The peer's cursor is distinct from request_id and this journal's local sequence. Outgoing
events on our served streams commit before publication so another connection/owner can replay
them. The [streaming contract](delegation.md#streaming-updates-and-reconnection) owns both-hop
resumption, replay-to-live handoff and fallback for peers without replay. No public journal
fields are added for socket or presentation lifetime.

Journal sequence is local observation order, not a remote causal order or universal replay ID.
Use peer-supported cursors/reconciliation or a fresh task snapshot for duplicate/stale input;
task/artifact IDs alone cannot deduplicate append chunks. Reconcile overlapping poll/stream
observations before updating current state. Use indexed views and storage checkpoints for reads.
Compaction retains payloads/references required by live request state, input and presentation;
terminal retention is bounded. Current state and its pinned payload references remain sufficient
for recovery as older communication history is compacted. Retain canonical source/response messages
while an input occurrence references them, independently of its journal events. Retain request ID/method
and payload association while any journal event references them, even after active request progress
is cleaned up. Prune them with their final references so older communication remains inspectable
without retaining completed execution state indefinitely. No exactly-once remote execution or
speech is promised.

### Debugging the communication

The journal explains what requests we recorded and what responses, artifacts or errors we
observed. Current request state explains dispatch progress and unfinished work. Together they
cover cancellation and recovery without inserting local state changes into communication history.
SessionState.chat_history separately records accepted human input and delivered content;
each AgentState.chat_ctx shows its private model context. Logged A2A content is not human dialogue.

```text
human input -> local tool call -> request state + recorded request/message
-> remote responses + updated task/input state -> queued presentation -> delivered frontend reply
                         -> get/cancel/subscribe requests and their outcomes
```

Correlate through message IDs, request/journal positions, context/task references, origin and
delivery records. These are linked views, not duplicated conversations. A queued response is
not proof it was heard; uncertain delivery stays explicit. Recovery reconciles durable facts
rather than restoring and blindly replaying the transient presentation queue.

The [delegation integration proposal](delegation.md) owns behavior and acceptance scenarios;
this page owns state layout. Media connections, credentials, Python stacks and the transient
presentation queue are not serialized into SessionState.

## Frontend presentation

The live frontend owns an in-process queue; it is not another field on SessionState. Backend
Agent.run/TextSession execution does not gain a presentation queue merely by running an agent.
The queue carries notifications for presentation, not the authoritative call/question records.
The [conversation expectations](delegation.md#conversation-expectations) define question pacing,
progress grouping and interruption behavior; this section defines the state they use.
The frontend can combine several eligible input-request events into one naturally worded
prompt. Each event retains its own input reference and actual presentation evidence; grouping
does not create a merged input request. A clear response can address several requests, with
each target validated and reserved independently and the human message recorded once.

```python
@dataclass(frozen=True)
class InputRequestRef:
    owner: Literal["agent", "remote_task"]         # Local agent input or remote task input.
    owner_id: str                                 # Local agent-state ID or configured remote ID.
    request_id: str                               # Occurrence ID inside that owner's request map.
    task: RemoteTaskRef | None = None              # Required only for a remote_task reference.


@dataclass
class PresentationEvent:
    event_id: str                               # Identity for this presentation notification.
    origin: ToolCallRef                          # Original request/result correlation.
    agent_tool_id: str                           # Stable configured tool identity, not display name.
    kind: Literal["progress", "input_request", "result", "error"]
    message: ChatMessage                         # Normalized content offered for presentation.
    input_request: InputRequestRef | None = None  # Owner-qualified local/remote input request.
    verbatim: str | None = None                  # Prefix requiring exact delivery.
```

When a result arrives, record it against its original interaction regardless of which agent
is current. Present it only when the current agent exposes the matching configured agent/tool
identity and access scope and the conversational floor is available. Otherwise leave the event
queued; do not force a switch, fabricate a tool call, discard the result or cancel remote work.
The same Python tool object or tool name alone does not determine eligibility.

Changing the current agent preserves the frontend queue. Closing the frontend does not cancel
remote tasks. Durable interaction/question/result records support recovery and reconciliation;
the transient queue itself is not restored or blindly replayed. Pin the correlation/history
needed by unsettled work and presentation until its retention obligations settle. Only actual
delivery updates chat_history and delivery evidence; enqueueing is not proof of presentation.

InputRequestRef is a small shared reference, not a shared execution/continuation model. A local
reference resolves into AgentState.input_requests. A remote reference resolves through
SessionState.remote_agents[owner_id].contexts[task.context_id].tasks[task.task_id].input_requests.
Validate the owner/task combination before routing. The frontend gathers candidates and
notifies the appropriate owner without a second copy of request state. Exact delivery receipt
storage and local continuation fields remain implementation design items.

## Application state

```python
@dataclass
class App:
    bookings: BookingService
    trip: Trip = persisted(default_factory=Trip)
    session: SessionState = persisted(default_factory=SessionState)
```

The application constructs runtime services and loads its marked fields through the store.
A bound execution keeps app.session current and checkpoints framework progress. Changes to
app.trip still require explicit save_context. Tools receive App through ctx.context.

```python
app = App(bookings=booking_service)
await ctx.store.load_context(app)

result = await front_desk.run(
    ctx.message,
    context=ctx.store.bind(app),
)

await ctx.store.save_context(app)  # Application fields only; not an implicit session save.
# app.session is the framework-maintained state view.
```

Agent.run, VoiceSession and TextSession accept the same binding. The store's selected
conversation supplies the execution state. An optional app.session field exposes that state;
it is not a second state source to merge with the binding. Without such a field, use
store.bind() to manage the conversation internally. Taking a snapshot does not create another
way to pass execution state. Plain unbound execution does not implicitly restore stored state.

For example, the framework records that an expert asked for a departure date and which answer
was submitted. Application code decides whether that answer changes trip.departure_date and
when that business change is saved. Framework checkpoints never implicitly save arbitrary
application mutations. The [persistence guide](../persistence.md) owns those save boundaries.

## Execution ownership and switching modes

Shared state does not allow two executors to mutate a conversation without coordination.
The selected partition's lease/fence and local execution owner serialize state transitions.
Separate partitions do not share a SessionState merely because they belong to the same thread.

A supported voice-to-text transition must:

1. Settle or explicitly account for local work and checkpoint accepted progress.
2. Restore the durable root, histories and remote context/task references.
3. Reconcile pending input, answer submissions and remote task state.
4. Continue without importing private child histories or replaying uncertain speech.

Closing the voice frontend does not cancel remote business tasks. Voice handoffs and VoiceTasks
are transient: they do not replace the durable root agent or become saved local children.
VoiceSession.current_agent identifies the live speaking role. The durable local-agent map is
continued by TextSession or bound Agent.run; it cannot resurrect a VoiceTask await or connection.

A conversation with unresolved local Agent-as-tool work must continue in text before voice can
start. Voice startup refuses that saved graph without changing it, whether supplied with the
root or child agent. It does not require a root graph or try to restore a text child on the call.
Closing a voice session releases its execution binding and temporary checkpoint sink; an
application-supplied ContextStore remains usable for the following text turn or call.

## Design and implementation checklist

The [delegation completion roadmap](delegation.md#implementation-checklist) owns the overall
implementation order and release exits. The checks here detail this plan's state contracts;
finishing them does not by itself complete live execution, voice delivery or interoperability.

### Agreed decisions

- [x] Agree on history scope: shared across modes, private per conversation frame.
- [x] Distinguish root_agent, current_agent and their private chat_ctx views.
- [x] Back root_agent/current_agent properties with one agent-state map and saved context IDs.
- [x] Define chat_history independently of the child-to-parent merge policy.
- [x] Choose store binding as the single conversation continuation API.
- [x] Let the current local agent handle unaddressed human input.
- [x] Recognize and durably retain responses to previously presented requests while the
      current agent lacks the matching remote tool; validate eligibility separately.
- [x] Reuse the saved default remote context for new work; retain the original context for
      task continuation and permit explicit separate contexts.
- [x] Separate local calls, remote delegations, owner-scoped questions and frontend presentation.
- [x] Use one ordered communication journal; expose public read-only messages, tasks, contexts
      and events, with dispatch and origin metadata on mutable internal request state.
- [x] Name local input records AgentInputRequest and use input_requests on both local/task owners.
- [x] Replace single-request DelegationRecord with remote agent/context/task communication state.
- [x] Put origin/dispatch information on requests and input/cancellation on tasks;
      omit a relationship-wide origin or cancellation flag.
- [x] Choose question-specific continuation instead of a standalone answer_tool field.
- [x] Choose InputRequired without a tool argument; local continuation returns to the owning agent,
      and remote continuation uses the selected input request's A2A task/context.
- [x] Choose RemoteTask.cancel() on bound task views for explicit cancellation.
- [x] Keep ordinary AgentState completed/output/error fields; omit a new local status enum and
      Python state-variant classes. Reuse shared TaskState for A2A task status.
### Implementation work

- [ ] Apply purpose/ownership/recovery comments to every finalized ABI/schema field and
      regenerate host documentation from those declarations.
- [ ] Finalize session/execution identity, initialization and frame relationship fields.
- [ ] Finalize internal options grouping and save the admitted policy with each child.
- [ ] Finalize local and remote question/continuation records, references and presentation
      events without adding a second global question registry.
- [ ] Persist each remote input occurrence's source/response message IDs with its response state;
      verify canonical lookup, uncertain response identity and recovery after task-status changes
      or journal compaction, including absent source messages.
- [ ] Expose lossless A2A protocol records through generated host APIs; reuse Task/TaskState,
      Message, StreamResponse and existing request/response types without parallel wire models.
- [x] Reuse SDK method constants for journal operation identities; map any internal ABI
      discriminator to them and generate host values, preserving streaming/unary send identity.
- [x] Store operation/method on internal request records keyed by stable request_id; require
      that ID on every journal event, independently of sequence and transport RPC identity.
- [ ] Cover every supported client operation in generated request/response mappings; preserve
      SDK response unions/error details and validate each method's allowed outcomes.
- [x] Persist mutable request progress using existing Rust ownership; keep dispatch/lifetime and
      caller/input correlation there, with no separate public primitive or local-state log markers.
- [x] Journal SDK requests, responses and observed errors across supported client operations;
      generate the three commented A2AJournalKind values and validate operation correlation.
- [ ] Commit related state/log changes atomically; test crashes between intent, dispatch, peer
      response and outcome persistence, including committed-then-lost database writes.
- [x] Build read-only message/task/context indexes and consistent checkpoints; preserve native
      Message responses and contextless messages without inventing tasks.
- [ ] Define bounded compaction and observed-state inspection semantics without losing
      unresolved request, origin, dispatch or delivery references; retain request ID/method
      metadata while historical events reference it and prune it with their final references.
- [ ] Deduplicate/reconcile reconnect input without assuming local sequence is a peer event ID.
- [ ] Verify restart loads request/task/input state and reconciles uncertainty without replaying
      control transitions from history; content replay repeats no artifact appends, requests,
      business effects or speech. Cover stale polls/streams, refused cancel and lost responses.
- [ ] Map existing Rust InteractionState onto per-request/task decisions under remote
      communication state; specify default-context initialization, task-follow and retention rules.
- [ ] Finalize explicit separate-context selection and task/checkpoint encoding; verify grouping
      across concurrent tasks, direct messages, optional task history and partial task-list observations.
- [ ] Finalize AgentState view typing and migrate the existing activity-enum name explicitly.
- [ ] Define agent replacement and supported child-mode transitions; verify that queued
      results survive agent changes and require matching expert/access eligibility.
- [x] Specify atomic result propagation/merge/child cleanup and recovery of a retained
      completed AgentState output without re-executing the child; preserve existing failures.
- [ ] Define shared item identity, context membership and delivery/history projection storage,
      including child removal with merge_chat_ctx disabled.
- [ ] Implement automatic response recognition across agent changes and durable response
      reservations, independently of permission to dispatch through the current agent's tools.
- [x] Define local continuation records and remove InputRequired.tool plus answer_tool from
      ABI/host/runtime/storage paths, without retaining a blind-dispatch compatibility path.
      Verify that side questions do not consume pending input or invoke an arbitrary named tool.
- [ ] Define ChatContext editing and snapshot semantics while bound.
- [x] Remove detached TextSession continuation inputs and its result/state envelope;
      use store binding as the single conversation input.
- [x] Remove old text/voice encodings and migration readers; storage has never been released.
- [ ] Move shared records and decisions into the sans-I/O core; keep runtime storage/I/O thin.
- [ ] Generate the host API from ABI declarations and integrate all three execution APIs.
- [ ] Verify text-to-voice and voice-to-text history continuity, root/current context isolation,
      child dialogue in chat_history with merge_chat_ctx on and off, interrupted delivery,
      approval identity, pending-question recovery and uncertain answer reconciliation.
- [ ] Exercise interrupted input, late tool results, competing owners and crash checkpoints
      through the real executors in DST, then host/compatibility and E2E coverage.

Implementation follows the agreed contracts above. Remaining public API questions are retained
in the delegation plan while implementation proceeds on the resolved behavior.
