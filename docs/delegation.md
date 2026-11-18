# Delegation

A frontend agent keeps the conversation responsive while remote agents reason and do business
work. Models, instructions and tools can differ between those roles. The person still speaks
to one conversational partner: the frontend controls wording, pacing and interruptions.

Delegation is communication between agents, rather than a single task. One configured remote
can own several A2A tasks and contexts, ask for input, send progress and return direct messages.
Those observations belong to the conversation's [SessionState](session-state.md).

## Configure a frontend

```python
from livekit.agents import Agent, DelegationOptions, RunContext, TextSession, VoiceSession
from livekit.agents.a2a import RemoteAgent

travel = RemoteAgent(
    travel_url,
    id="travel",
    name="travel",
    description="Find and book flights and hotels.",
)
front_desk = Agent(
    name="front_desk",
    instructions="Help travelers. Use travel for searches and bookings.",
    llm=frontend_model,
    tools=[travel.as_tool()],
)
delegation = DelegationOptions(routing_model=routing_model)

async with VoiceSession(
    llm=frontend_model,
    stt=stt_model,
    tts=tts_model,
    delegation=delegation,
    context=store.bind(app),
) as session:
    await session.start(agent=front_desk, room=room)
```

The frontend and router are separate configured models. The framework never substitutes the
frontend model or chooses a provider implicitly. Configure the same owner-level policy for
text and bound direct runs:

```python
result = await front_desk.run(
    "Find a flight to Rome.",
    context=store.bind(app),
    delegation=delegation,
)

text = TextSession(agent=front_desk, delegation=delegation)
result = await text.run("Make that Saturday.", context=store.bind(app))
```

The [persistence guide](persistence.md#session-state-and-explicit-binding) explains loading the
application and declaring its state field. Voice can use the same coordination in memory
without persistence. A durable binding adds recovery across executions.

`RemoteAgent.id` is the stable saved-state and event-eligibility key. It defaults to the
canonical URL; model-facing `name` and `description` are independent. Distinct credential
configurations at one URL use distinct IDs. Reusing an ID with a conflicting live endpoint or
authentication is refused. Saved IDs and addresses do not restore credentials or grant access.

## A natural conversation

Remote latency does not own the conversational floor. While a tool takes sixty seconds, the
frontend can listen, answer a side question and acknowledge a correction. Stopping playback
must not wait for that remote operation. A backchannel or false interruption follows the
session's [voice policy](voice-session.md); interrupting speech alone does not cancel work.

The frontend should acknowledge admitted work when useful, without claiming that an action
already succeeded. Acknowledgement is model behavior; there is no automatic `announce` option.
Results remain grounded in the remote's actual observations and the application service.

For questions and updates:

- Ask independent questions at a manageable pace. Five pending questions do not require a
  five-question utterance. Keep the remaining sources pending.
- Group related information when natural: first and last name can be requested together.
  Track the response to each selected source independently.
- A partial response answers only what it supplies. Two independent cancellation questions
  followed by a bare “yes” require clarification unless the person identifies one or both.
- Batch useful progress already available when the frontend can speak. Ten internal updates
  need not produce ten announcements, and batching does not require a fixed waiting period.
- Retain skipped questions and their destinations. A result or progress item cannot silently
  replace a question the person was answering.

Ordinary content is worded by the frontend. Explicit exact output keeps its ordered delivery
contract. Conversation history records what was actually delivered, including interrupted
prefixes; generated or received text alone is not evidence that the person heard it.

## Responding to questions

Response recognition runs before the frontend's first plan for the human turn. It considers
canonical human input, presented unresolved requests and the person's delivered conversation.
It does not inspect another agent's private model history or assume that every new message
answers the latest question.

The router may select one or several requests, identify a correction to a held response, find
no match, or request clarification. The native owner validates that decision against the
exact offered occurrences and current human words, then checkpoints the reservation before
allowing dispatch. An unavailable or ambiguous detector keeps requests pending and lets the
frontend clarify; it does not guess a destination.

Recognition and dispatch eligibility are separate. If a local child is current without the
relevant remote tool, a recognizable response to an earlier question is retained. It can be
forwarded when an eligible agent returns. Changing the speaking agent does not redirect it.
`VoiceTask` input stays with its own temporary interaction and does not automatically answer
an earlier remote question.

A correction can replace a reserved response before dispatch. Once the earlier request may
have been sent, reconcile that request before assuming it failed. A correction to ongoing
business work can be sent as another task message; it cannot imply that an already completed
external action was undone. Recognition itself never grants business approval.

## Select work through the same tool

The remote tool has an optional `task_id` argument:

```json
{"request": "Why do you need the date?", "task_id": "booking-42"}
```

The task must be known on that configured remote and nonterminal. Its saved state supplies
`contextId`. Selecting a task can clarify or correct work; it does not by itself answer the
task's question. Input-occurrence IDs remain framework correlation and do not appear in the
model's arguments.

With `task_id` omitted or null, the owner routes the proposed tool call separately from human
response recognition. It can forward a captured response, continue known work or start new
work. One human turn can answer a flight question and start an independent hotel search.
An earlier forwarding decision does not disable the remote tool for the rest of the turn.

`DelegationOptions(routing_model=None)` explicitly disables automatic routing. An explicit
`task_id` then continues known work; omitting it starts new work. Unknown, foreign and terminal
IDs return tool feedback before dispatch. A card schema cannot declare a business field named
`task_id`; the framework strips that reserved argument before sending business data.

New work reuses that remote's saved default A2A context for this conversation, without
inheriting a task ID. A response or clarification keeps the selected task's own context.
Configure distinct stable remote IDs when the application needs independent remote
conversations; each ID retains its own default context.

One live task has one native observer and original result owner. Further calls to it wait for
message admission; the original call receives the final result. A peer merging requests cannot
create another result owner. Native state, rather than copied model-history metadata, owns
continuation authority.

## Progress and exact output

```python
@booking.tool
async def reserve(ctx: RunContext[App], itinerary: str) -> str:
    await ctx.update("Checking availability.")
    reservation = await ctx.context.bookings.reserve_once(itinerary)
    await ctx.say(f"Your booking reference is {reservation.reference}.")
    return "The reservation was confirmed."
```

Both methods enqueue immediately and return a private `Awaitable[None]`. Omitting an await
allows the body to continue while admitted work remains owned and ordered. Canceling or
discarding that waiter does not cancel admitted delivery.

| Operation | Await guarantee |
| --- | --- |
| `ctx.update(text)` | Progress was admitted; the frontend may word or batch it later |
| `ctx.update(text, silent=True)` | Status was admitted without requesting conversational wording |
| `ctx.say(text)` in voice | The exact text completed the configured local output boundary: audio playout when audio is attached, or text delivery in a text-only session |
| `ctx.say(text)` in text/A2A | The exact text reached that execution's live output boundary |

An A2A backend's `await ctx.say()` does not acknowledge downstream speech. Each hop confirms
its own boundary; stock A2A provides no end-to-end “heard by the person” receipt. Incomplete
delivery, failed output and owner loss settle the delivery waiter with an existing error;
that error does not itself cancel business work or undo an external effect. A plain run
without a live output sink refuses exact delivery instead of recording invented human history.

A realtime voice session needs an explicit TTS provider or TTS node for exact tool speech.
Without that path, exact delivery is refused before audio generation.

Exact output remains ordered through final body return and nested delegation. It cannot be
coalesced away like ordinary progress. Receipts remain private; applications need no receipt
class or additional controller. External business APIs still need their own idempotency and
reconciliation, as represented by `reserve_once` above.

## Long work on a backend

An endpoint can opt into `inbox="live"`. Run its managed conversation through the existing
store binding. Identified tools can release the foreground while their bodies continue; one
native owner serializes subsequent model turns and result delivery.

An unaddressed message is handled by the current local agent and retains its own transport
task. An explicit `taskId` continues that task. Results, checkpoints and exact output retain
their original source even when another handler is current. Queue and batch policies keep
their existing behavior; live execution is opt-in. The
[server guide](server.md#a-message-while-a-turn-is-running) owns admission and policy options.

Closing a response observer detaches it from business execution. It does not cancel the task.
The endpoint owns held bodies, their cleanup and final commits until those settle or explicit
cancellation is admitted.

Cancellation follows task ownership rather than the local agent tree. If task B is already
continuing a child that task A started, canceling A leaves B and that child's current work
running. Sharing a child context does not give A cancellation authority over B.

## Result content and interoperability

Remote agents can return SDK Messages, Tasks and streaming status/artifact events. A direct
Message stays a message; it does not create a synthetic task. Task history and optional
metadata/artifacts may be absent. Inspection preserves SDK records, parts, artifact ordering,
metadata and extensions independently of frontend presentation.

For the model, text-only results remain text and a sole data part keeps its JSON value.
Mixed data and files retain their SDK structure. A named `answer` artifact does not discard
other final data/file artifacts. URL files are represented without fetching them. Inline
bytes stay in the SDK observation; the model receives filename, media type, byte count and
their retained location, without claiming to have read their contents.

Ordinary model material is bounded. An oversized completed result produces a neutral fact
that the result is retained, rather than truncated JSON or an error inviting the model to
repeat completed business work. Required exact text keeps its independent delivery promise.

Unflagged working text remains status. A namespaced progress flag offers it for conversational
presentation; explicit exact text follows the ordered output path. Authentication-required
status is not a human input question. Unsupported private protocols fail explicitly rather
than being guessed from wording. [A2A](protocol/a2a.md) owns wire shapes and metadata keys.

## Cancellation, uncertainty and recovery

```python
remote = app.session.remote_agents["travel"]
await remote.tasks[task_id].cancel()
```

Cancel through a bound task view. Its native owner journals intent and dispatch progress;
success requires persisted observation of remote cancellation. A lost reply is not proof that
the task stopped. Concurrent callers share the pending attempt; after definite refusal, a
later explicit call may issue another request. Ending a call never requests cancellation.
[Persistence](persistence.md#canceling-retained-remote-work) owns binding and current-credential
requirements for controls between runs.

Recovery restores committed task/context/input references and reconciles their current state.
An accepted terminal observation cannot be reopened by a late working poll. Only proved final
content and the accepted terminal event's own presentation policy become a recovered result.
Cached partial artifacts from a broken stream do not become a fabricated complete answer.

Preparing a request persists its immutable content before dispatch starts. After an uncertain
send, retain that identity and reconcile with a known task or supported peer evidence. Missing
task identity or unprovable admission remains unknown; do not resend business work to restore
observation. Python stacks and transient speech queues are not recovered, and uncertain speech
is not blindly replayed.

Stock A2A task lookup/resubscription can recover supported current observations without
guaranteeing every missed update. Durable cursor replay is a separate capability. Request IDs,
local journal sequence and inherited SSE event IDs do not by themselves establish replay or
deduplication guarantees.
