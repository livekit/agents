# Delegation integration

A person should be able to talk naturally to a responsive agent while a remote agent does
deeper reasoning and business work. The frontend keeps listening, handles interruptions and
maintains the conversation while remote work progresses. The remote agent can use a stronger
reasoning model, specialist instructions and tools independently of the frontend's model.

The document first explains the behavior and public API, then the native implementation and
work required. It describes the target system. Existing SDK guides remain the current contract
until implementation lands. The implementation checklist records work still to do.

## Product direction

The intended experience is one capable conversational partner. Its frontend is optimized for
listening, turn taking, personality and quick conversational responses. A remote agent is
optimized for the work: reasoning through a complicated request, checking systems, running
tools, validating information and producing a reliable outcome. Models are chosen independently
for those roles. Remote agents can themselves communicate with other agents.

Delegation is an ongoing conversation between agents. The remote agent can send useful
progress, ask for missing information, continue after a response and return results across
several exchanges. Several tasks can run at once. The frontend decides how to integrate that
work into the person's conversation, preserving its voice and pacing.

Responsiveness and task completion are separate product goals. A remote tool can take sixty
seconds while the frontend remains available to listen, acknowledge a correction, handle a
side question or explain known status. Starting remote work must not hold the speaking turn
open until it finishes. A prompt acknowledgement is grounded in admitted work; the frontend
does not claim a result, successful change or completed action before it has that evidence.

In particular, the person should experience:

- **A responsive conversation while work continues.** They can keep talking and give useful
  information without waiting for the remote model to finish reasoning. The frontend can
  answer from available knowledge and say when it is still waiting for task-specific facts.
- **Natural interruptions.** The frontend yields on a detected interruption and handles
  backchannels or false interruptions according to the session's voice policy. Remote latency
  must not delay stopping playback. Interrupting speech alone does not cancel remote work.
- **Corrections that matter.** A change such as "Saturday, not Friday" is retained and
  communicated through the supported remote path. A late Friday result cannot be presented
  as satisfying the Saturday request. If an action already happened, explain its actual state
  rather than implying the correction undid it.
- **Useful questions and progress.** Ask questions at a manageable pace, grouping related
  questions when that feels natural, and summarize useful updates. Quiet periods are acceptable;
  avoid repeated filler or reading every internal event.
  Ordinary results are worded for the current conversation; exact text retains its contract.
- **Coherent continuation.** Answers reach the correct pending request. Side questions and new
  work remain possible. Returning to a stored conversation restores the work's references and
  reconciles its status, including uncertainty, without requiring the person to start over.

For example:

```text
Person:    Find a flight from Paris to Rome next Friday morning.
Frontend:  I'll check the options.
           [The remote agent reasons and searches; the frontend remains available.]
Person:    Actually, make that Saturday.
Frontend:  Saturday—got it.
           [The correction is retained and communicated; remote work continues.]
Frontend:  What budget should I use?
Person:    Up to five hundred euros.
           [The remote agent receives the response and returns Saturday options.]
Frontend:  I found two options. The first departs at—
Person:    Just tell me the cheapest one.
           [The frontend yields, then answers using the received options.]
```

The frontend's acknowledgement of Saturday records the person's preference; it does not
announce that a booking changed. The budget question represents an actual pending remote
request. The interruption changes what is presented next without restarting the search.

We assess this experience through separate observations: how quickly speech stops on an
interruption, how quickly the frontend begins an appropriate response, whether remote updates
are presented at useful moments, and how long the remote task takes. Longer remote reasoning
must not silently become longer interruption or conversational response latency. Exact latency
budgets need measurement and agreement; this plan does not invent numerical guarantees.

## Problem and scope

A voice agent may act as a travel concierge while separate agents search flights, reserve
hotels and authorize payments. The concierge needs to keep the conversation coherent while
those experts work independently. A remote call returning a string is not enough:

- A flight expert can ask a question while the hotel expert is still working.
- The person's next utterance might be a side question rather than the requested answer.
- A result can arrive while another reply is speaking, or after the person interrupted it.
- A lost connection does not tell us whether an expert accepted an answer or made a booking.
- A person can end the call and later continue the same work by text.

The rewrite already has RemoteAgent tools, A2A transport, expert presentation and task storage.
This integration extends them into a coordinated lifecycle, using the final behavior of Long's
stack as its starting point. The source PRs are listed at the end. Some earlier PR behaviors
were replaced later; their final scenarios, not each intermediate implementation, are the porting
target.

Scope includes remote and applicable local agent-tool behavior, text and voice delivery,
durable correlation, cancellation, third-party A2A and the developer-facing controls. It preserves
the rewrite's typed results, explicit application persistence and approval checks. It does not
recover Python call stacks or guarantee exactly-once external business operations.

## Concepts

| Term | Meaning in this design |
| --- | --- |
| Frontend / fast brain | The agent handling the person's conversation. It chooses which expert to call and ordinarily words results for the person. |
| Remote agent / slow brain | An agent doing delegated work. It owns validation, business tools, approvals and results, and can communicate with other agents. |
| Delegation | Communication with a configured remote agent. It can span multiple local tool calls, messages, contexts and tasks; it is not itself one A2A task. |
| A2A message | Input or output with a messageId and optional contextId/taskId. A peer can reply directly with a Message without creating a task. |
| A2A context | A grouping identifier for related messages and tasks. Our context view groups observed state; it is not a new protocol object. |
| A2A task | The expert server's tracked work, addressed by `taskId` within its `contextId`. One task can pause for input and continue several times. |
| Turn / invocation | One execution responding to input or delivering background results. Remote tasks can outlive that invocation and continue across several local tool calls. |
| Pending question | An expert request for input, tied to a task and a particular occurrence of the question. Several can remain open; compatible questions can be presented together in one conversational prompt. |
| Question incarnation | The local identity of one question occurrence. The same remote task can ask again, so its task ID alone cannot distinguish an old answer from an answer to the new question. |
| Partition owner | The runtime that holds the lease for one stored conversation partition and serializes its changes. A fence prevents an old owner from committing after ownership changes. |
| Delivery receipt | An observation at the output boundary: progress admitted, text handed off, or audio played. It does not prove the person understood the message. |

Local conversation identity and remote A2A identity are distinct. A third-party `contextId` is
not automatically a LiveKit thread database ID. Similarly, an A2A task is not a suspended Python
coroutine: the expert server decides how to reconstruct or continue its execution.

## Agreed behavior

| Area | Contract |
| --- | --- |
| Entry point | `RemoteAgent(...).as_tool()` |
| Configuration | Connection/auth on `RemoteAgent`; caller behavior on the conversation execution owner |
| Routing | Detector enabled by default, including one eligible pending question |
| Routing model | Configure a separate routing model explicitly; frontend-model reuse is a later optimization |
| Agent switches | Recognize responses to previously presented questions even when the current agent lacks the remote tool; retain them durably until dispatch is eligible |
| Detector disabled | Explicit task reference continues work; no reference starts new work |
| Detector failure | Bounded retries; keep pending questions; return feedback to clarify or retry |
| Updates | Enqueue immediately; awaiting is optional. `update` waits for admission; `say` confirms its local output boundary |
| Control | Inspect public remote message/task/context/event views; explicitly target a remote task for cancellation |
| Scheduling | Opt-in `inbox="live"`; retain existing queue/batch defaults |
| Call end | Remote work continues; cancellation is explicit |
| Restart | Recover durable references and questions; reconcile remote state |
| New human input | The current local agent handles unaddressed messages; targeted responses retain their input-request identity |
| Remote context | Reuse the conversation's saved default context for new work with that remote; task continuation retains that task's context |
| Stream reconnect | Resume existing tasks from durable cursors where supported; reconcile snapshots otherwise; never restart business work merely to restore observation |
| Existing guarantees | Preserve typed outputs, explicit application saves, authorization and validated-argument approvals |

## Conversation expectations

The frontend coordinates one conversation with the person while several agents work. The
person should receive useful questions and outcomes at a sensible pace, without having to
manage the arrival order of remote events. These are the default presentation expectations;
they apply to local and remote agent work wherever the same interaction is supported.

| Situation | Expected experience |
| --- | --- |
| A remote model is reasoning or a tool is slow | The frontend remains available for conversation and interruptions. It can acknowledge admitted work or report known status without inventing a result or filling every pause. |
| Five questions arrive together during a voice conversation | Choose a manageable prompt and give the person a chance to respond. Group related questions when natural; keep unrelated or demanding questions pending. Presenting a group does not merge its underlying input requests. |
| Related questions can be answered together | First name and last name can become "What's your first and last name?" Choose wording and grouping for the conversation rather than imposing a fixed number of questions per utterance. |
| Several questions remain pending | Choose the next relevant prompt fairly. An answered agent can receive a bounded follow-up opportunity, without repeatedly starving other waiting agents or displacing the foreground prompt. |
| The person skips a question or changes topic | Handle the new topic and release that question's presentation priority. Keep it open for a later response; do not immediately repeat it or interpret the topic change as its answer. |
| The person answers an earlier question, or clearly answers several | Route each identifiable response to its actual input request. Presentation order does not make the latest question the only valid target. Clarify ambiguous responses. |
| Ten ordinary progress updates arrive before the frontend can speak | Combine eligible updates into one useful summary at the next conversational opportunity. Repeated or superseded status does not require ten separate announcements. |
| A result arrives before its queued progress is spoken | Present the outcome with any still-relevant context. Omit obsolete narration such as "I'm still checking" after that check has finished. |
| Progress arrives while the person is speaking or answering a question | Wait for the conversational floor. Routine progress does not interrupt the person or repeatedly distract from the question being answered. |
| Several results become ready together | The frontend can give a combined summary while keeping each task's outcome clear. Grouping does not settle unrelated calls or discard an outcome that still needs delivery. |
| A tool uses exact `ctx.say()` output | Preserve its text and delivery order. Ordinary progress batching must not replace it with a summary, silently omit it or mark it delivered without the appropriate receipt. |
| The person interrupts a question or result | Yield according to the session's interruption policy without waiting for the remote agent. Retain results and delivery facts; continue or clarify without racing a duplicate reply. Speech interruption alone does not cancel remote work. |
| The person changes a constraint while work runs | Retain and communicate the correction through a supported path. Reconcile late results against the current request; do not present an outdated answer as satisfying the new constraint or imply an already completed action was undone. |
| An agent changes while remote work is running | Retain pending work and outcomes. Present them when the current agent exposes the matching remote tool and has access; do not force an agent switch merely because an event arrived. |
| The person returns after a disconnect | Reconcile the existing work and explain the current situation. Summarize relevant missed progress rather than narrating the entire backlog. |
| The conversation is text rather than voice | Related questions or results can be grouped into a readable message when their targets remain clear. Preserve the same input identities, validation and delivery rules. |

For example, if flights asks for a date, hotels asks for a budget and payments asks for
confirmation at the same time, voice might start with "What date would you like to fly?"
The hotel and payment requests remain pending. The person's next utterance can answer the
date, ask why it is needed, or start another topic; it is not automatically a flight answer.

Conversely, asking "What's your first name?" and then "What's your last name?" as separate
turns can sound unnecessarily mechanical. The frontend can ask "What's your first and last
name?" If the person supplies only their first name, continue with the missing information.
If separate input requests contributed to that prompt, track their responses individually;
grouping alone neither resolves them nor makes an ambiguous response sufficient. A clear
response can address several requests, each validated and reserved against its own target.
Preserve per-request presentation evidence: an interruption before part of the grouped prompt
is delivered must not mark every included question as presented. Exact wording and approval
contracts still apply when choosing what can be grouped.

If ten eligible updates report flight options and an ongoing hotel search, voice might say
"I've found flight options; I'm still checking hotels." If both searches have already finished,
it can go straight to their results. Keep each fact associated with the correct task and do
not invent completion. Silent updates and unflagged A2A working status do not independently
request spoken announcements.

Batching here concerns presentation. Accepted communication events remain recorded under the
normal journal/retention rules, and each input request and task retains its identity. The
low-level client can still observe individual protocol events. This is separate from the
endpoint's `inbox="batch"` policy, which groups incoming requests.

The native machines enforce eligibility, ordered foreground delivery, per-request correlation,
exact wording and receipt accounting. The frontend model chooses wording and useful grouping
of questions and updates within those constraints. There is no fixed limit of one question per
utterance. Coalesce updates already waiting at a presentation opportunity;
do not require a fixed delay or wait for a particular number of updates before speaking.

## Public API

New signatures below are proposed spellings; the behavior above is agreed.
The frontend example is a fragment inside an RTC job handler: `ctx.room` is that handler's
room, and the models and expert URL are application configuration.

```python
flights = RemoteAgent(flights_url, token=token, conn_options=connection_options)

front_desk = Agent(
    instructions="Help travelers plan their trip.",
    llm=frontend_model,
    tools=[flights.as_tool()],
)
delegation = DelegationOptions(routing_model=routing_model)
session = VoiceSession(
    llm=frontend_model,
    stt=stt_model,
    tts=tts_model,
    delegation=delegation,
)
await session.start(front_desk, room=ctx.room)
```

Answer routing and dispatch acknowledgement are conversation-wide behavior. Configure them
on the execution owner (VoiceSession, TextSession or the run/endpoint execution), so one policy
coordinates all exposed remote tools. Automatic detection requires an explicitly configured
separate routing model for every frontend type, including pipeline and realtime voice. There
is no fallback to the frontend model or implicit provider credentials. A decision model or
Gemma can be an application choice; no model family is hardcoded by the framework. Reusing a
frontend model is a later optimization. DelegationOptions/delegation are proposed spellings;
these controls do not belong on individual as_tool calls.

| Surface | Meaning |
| --- | --- |
| `.as_tool()` | Expose the expert as a tool; conversational policy belongs to the execution owner |
| Conversation policy | Default detector and dispatch acknowledgement; acknowledgement suppression does not suppress progress/questions/results |
| `ctx.update(text, silent=False)` | Progress the frontend may word; silent updates inform only the model |
| `ctx.say(text)` | Ordered exact text, including through nested delegation |
| `InputRequired(question, verbatim=False)` | Finish this invocation waiting for input; a later invocation resumes the task |
| `ToolResult(..., verbatim=, reply_required=)` | Existing result wording and follow-up controls |
| `SessionState.remote_agents[remote_id]` | Public read-only views of observed messages, tasks, contexts and the communication journal |
| `await remote.tasks[task_id].cancel()` | Cancel through the bound RemoteTask; it retains its remote/context/task addressing |

Both update methods enqueue owned work and return awaitable receipts. `await update` confirms
admission, not spoken progress. `await say` confirms its local output boundary: audio playout
on the frontend voice session, or handoff to the text/A2A output on a backend. Forwarding exact
text through nested delegation does not supply end-to-end speech acknowledgement; stock A2A
has no such receipt. Interruption, close and uncertainty settle receipts with defined outcomes.

For example, an expert can report that it is checking availability, then deliver the reference
as written after the booking service confirms it:

```python
@booking_agent.tool
async def book(ctx: RunContext[BookingContext], itinerary: str) -> str:
    await ctx.update("Checking availability.")
    booking = await ctx.context.bookings.reserve_once(itinerary)
    await ctx.say(f"Your booking reference is {booking.reference}.")
    return "The reservation was confirmed."
```

Here `reserve_once` represents an application service with its own stable business idempotency.
The framework owns update delivery, but cannot make an arbitrary booking API idempotent.
Omitting an await allows the tool to proceed while owned delivery continues; it does not discard
the message. Exact output still follows the interaction's delivery order.

With detection disabled, the tool schema has a fixed optional reference field, provisionally
`task_id`. The framework supplies eligible references in model context:

```json
{"request": "Find flights from Paris to Rome"}
```

```json
{"request": "Next Friday", "task_id": "<pending reference>"}
```

Validate references against the conversation, expert, task, question incarnation and access.
Schema enums do not change during a realtime call. Before a remote task ID is known, the
request and dispatch facts remain inspectable. Stopping local waiting does not prove remote
work stopped; an uncertain dispatch must be reconciled before a task can be targeted.

Retain `AgentClient.send(..., thread_id=, task_id=)` and `AgentClient.cancel(...)` for low-level
A2A access. A third-party context ID is not assumed to identify a LiveKit database. Expose typed
updates through existing session events/run streams; application observers do not drive execution.

Inspect remote communication through the bound SessionState, or a detached inspection snapshot.
Use the generated RemoteTask view for task-scoped cancellation:

```python
remote = app.session.remote_agents[remote_id]
task = remote.tasks[task_id]
await task.cancel()
```

The bound view retains the addressing and sends a validated command through its native
conversation binding. The control path serializes with an active owner or takes a short
fenced control turn when idle. It does not start a model turn or add a model-visible tool.
Data properties stay read-only. Detached snapshots are inspectable but cannot dispatch
commands; rebinding supplies fresh authority. Low-level clients retain AgentClient.cancel().
The [state proposal](session-state.md#logical-records) owns the bound/detached and durable
cancellation contract. No owner-level cancel_remote_task method is introduced.

### Conversation state across execution APIs

**Proposed state carrier:** extend the existing SessionState with durable delegation and
pending-question records. Do not introduce a separate public DelegationState. The
[shared SessionState proposal](session-state.md) owns its schema: one ChatContext per frame,
shared across execution modes, with a top-level root and private child contexts. Public
current_agent/root_agent views expose each agent context's chat_ctx; chat_history describes
what the person experienced across agents and modes, independently of child-to-parent merging.
The linked proposal owns those history semantics and separates local AgentCallState from
remote agent/message/context/task communication state. Durable internal request records retain
individual tool-call origins and dispatch progress; input requests and cancellation belong to
specific tasks. Several tasks and message-only exchanges can share a remote context. Small
references and common rules are shared, not a local/remote execution record or global input
registry. Remote communication state survives supported executor changes; presentation stays
with its live frontend owner.

The public remote views follow A2A's domain model:

```python
remote = app.session.remote_agents[remote_id]
remote.messages                         # Observed A2A Messages in both directions.
remote.tasks                            # Observed A2A Tasks plus local routing state.
remote.contexts[context_id].messages     # Same messages, grouped by contextId.
remote.contexts[context_id].tasks        # Same tasks, grouped by contextId.
remote.events                           # Append-only requests, responses and observed errors.
```

Data properties are read-only and describe retained observations, not the remote server's
complete state. Inspection performs no network calls; RemoteTask.cancel() is an explicit
command on a bound view. Contextless and message-only exchanges remain valid;
Task.history is optional and does not replace the communication journal. All client operations
for this conversation record requests, responses and observed errors. Request dispatch progress,
stream lifetime and local routing live in durable current state, alongside tasks/input_requests;
they do not create extra communication events. Related state/log changes commit atomically.
The linked [remote-state design](session-state.md#remote-agents-messages-contexts-and-tasks)
owns record definitions, current-state recovery, request correlation and commit rules.

Store binding is the single proposed entry point for restoring and continuing a conversation
across all three execution APIs. The binding selects the stored state, checkpoints framework
progress and keeps an optional app.session view current. Plain application-field mutations
still require explicit saves. Snapshots remain useful detached reads, not a second execution
input path. Neither form contains Python coroutines, media connections or the transient
presentation queue.

The following examples define the proposed public shape. VoiceSession/TextSession binding
exists today; durable delegation records and Agent.run state integration are new work.
The handlers assume configured agents/models and a runtime-provided store for the selected
thread/partition. Sharing configuration or a room alone does not select the same conversation.

```python
from dataclasses import dataclass
from livekit.agents import SessionState, persisted

@dataclass
class App:
    session: SessionState = persisted(default_factory=SessionState)
```

VoiceSession uses the existing binding API. This runs inside an RTC job handler:

```python
app = App()
await ctx.store.load_context(app)

session = VoiceSession(
    llm=frontend_model,
    stt=stt_model,
    tts=tts_model,
    context=ctx.store.bind(app),
    delegation=delegation,
)
await session.start(front_desk, room=ctx.room)
```

TextSession can use the same binding inside a managed text handler:

```python
app = App()
await ctx.store.load_context(app)

result = await TextSession(agent=front_desk, delegation=delegation).run(
    ctx.message,
    context=ctx.store.bind(app),
)
return result
```

Agent.run should accept that binding too, without requiring a TextSession wrapper. This is a
proposed addition: today's Agent.run does not unwrap BoundContext or restore SessionState.

```python
app = App()
await ctx.store.load_context(app)

result = await front_desk.run(
    ctx.message,
    context=ctx.store.bind(app),
    delegation=delegation,
)
await ctx.store.save_context(app)  # Explicitly retain application-field changes.
return result
```

The state is read through app.session; no new Agent.run(state=...) input or RunResult.state
output is added. The result retains its typed output/status contract. Applications without a
declared SessionState field can use store.bind() and let the store manage the state internally.
Binding construction itself performs no I/O; execution activates the selected conversation.

Remove detached TextSession execution inputs and the result/state envelope. Storage has not
been released: keep one encoding and no legacy migration path. The continuation API has one
state source: the binding. A bound execution must reject competing explicit history/state inputs rather
than choosing one silently. A plain unbound run does not implicitly discover stored session
state. A standalone invocation must settle its local owned work before returning; it cannot
become a hidden live endpoint.

### Decisions to finish

- [ ] Finalize the common owner-level configuration signature and retry settings. A separate
  explicitly configured routing model is agreed for all automatic-detection frontends.
- [ ] Integrate Agent.run binding and shared SessionState, including transitions between
  executors, current-agent input handling and removal of detached continuation entry points.
- [ ] Finalize reference spelling, custom card-schema collisions and explicit overrides in detector mode.
- [ ] Decide inbound messages without `taskId`: continue a local waiting task or retain new-task/`mergedInto` behavior.
- [ ] Finalize RemoteTask.cancel() failure/uncertainty outcomes and bound control-activation
  lifetime; retain the agreed task-instance command instead of an owner-level method.
- [ ] Finalize read-only snapshot/retention behavior and generated protocol types for public
  messages/tasks/contexts/events, including partial observations and optional task history.
- [ ] Decide the bounded protected-step API and shutdown contract before its implementation;
  the compatibility comparison is deferred in the TODO below.

## Inbox policies

| Inbox | Message arriving while work runs |
| --- | --- |
| `queue` | Wait; each message gets its own turn and reply |
| `batch` | Answer a burst together; earlier tasks reference the answering task with `mergedInto` |
| `concurrent` | Overlap without a lease; no application-field or routing-state saves |
| `reject` | Refuse the busy request, identify the running task, create no new task |
| `live` | Continue background tools while one owner serializes conversational and result-delivery turns |

The first four exist today. A2A/delegation default to queue; SMS, WhatsApp and email to batch.
Live is selected per endpoint/partition, independently of fast-brain/slow-brain roles:

```python
@backend.a2a_session(endpoint="booking", inbox="live")
async def booking_endpoint(ctx: A2ASessionContext):
    return await booking_agent.run(
        ctx.message,
        context=ctx.store.bind(),
        delegation=delegation,
    )
```

Application loading/saving is omitted above and remains explicit. Live lets this backend handle
another turn while a tool works. VoiceSession can already converse while waiting for an expert;
a frontend text endpoint can separately enable live execution and shared call/text ownership.

Under `live`, a tool's progress can release the conversational turn while the tool continues.
Its A2A task remains working. A new request gets a new turn under the same owner. When the tool
finishes, a delivery turn incorporates its result and replies to the tasks that result belongs
to. This turn may have no new human message. There is still one writer of conversation state;
`live` does not let several invocations mutate that state independently.

## Execution model

### A conversation with two experts

Consider a person asking the travel concierge for flights and a hotel in Rome:

1. **Start the work.** The fast brain calls the flight and hotel tools. Each request gets durable
   internal state retaining its originating tool call; its SDK payload is recorded in the
   communication log before dispatch. Responses and corresponding state updates commit together.
   Another send can continue the same task or start separate work; a direct Message reply needs no task. Reusing RemoteAgent
   configuration does not share another person's communication state.
2. **Ask a question.** Flights reports `input-required`: "When do you want to fly?" The owner
   records the pending question. The voice session presents it when the conversational floor
   is available. A hotel question arriving meanwhile waits; its existence is not proof it was heard.
3. **Handle a side question.** The person asks why the date is needed. The fast brain can explain
   without invoking flights. Nothing is sent as a date answer. The flight question remains open;
   another queued question can subsequently get presentation priority.
4. **Route a relevant reply.** The person says "The flight should be next Friday." With the
   default detector enabled, the session checks previously presented, unresolved input requests
   against the conversation and this message. It can select the earlier flight question even
   if another question has since been presented. If a local child currently lacks the flight
   tool, retain the response against that question until dispatch is eligible. Ambiguous input
   permits clarification while the requests remain open; new work reaches the current agent.
5. **Continue the expert task.** The native machine validates and reserves that question. The
   runtime sends a new A2A message using the flight task's existing `taskId` and `contextId`.
   The expert validates the input and either continues, asks again, or completes. A subsequent
   question on that same task has a different local question occurrence.
6. **Handle nested work.** If flights delegates payment, each hop retains its own task and
   pending-question references. A payment answer follows those references back through the
   chain. Intermediate progress does not create several competing frontend announcements;
   explicitly exact receipts can propagate to the person.
7. **Deliver the outcome.** The flight result resolves the local call correlated with that
   request. The frontend words the outcome; exact text uses the ordered delivery path. Hotel
   work can remain pending. Task history and the remote relationship survive individual local
   call settlement. Never attach a result to an unrelated newer tool call.

```mermaid
flowchart LR
    Person --> Owner[Native conversation owner]
    Owner --> Frontend[Frontend model]
    Frontend --> Owner[Native conversation owner]
    Owner --> Routing[Detector or explicit reference]
    Routing --> Owner
    Owner --> Expert[Expert over A2A]
    Expert --> Owner
    Owner --> Output[Text or voice presentation]
    Output --> Person
    Owner <--> Store[Durable references and task records]
```

### Input-response detection

The detector matches human input to response-eligible requests already presented in the
authorized conversation. It is enabled by default, including when there is only one pending
request. Already-reserved requests are not offered as fresh targets. Without eligible
presented candidates, normal input handling proceeds without this classification call.

Its inputs are the accepted human message, relevant human-visible conversation, each
candidate's question/content and its qualified occurrence reference. Include previously
presented questions from inactive agents; do not use tool availability as a recognition filter.
Do not disclose private remote reasoning/history or classify against unseen pending questions
merely because they exist in state. Delivery evidence distinguishes presented, partial and
queued questions, including a grouped prompt that was interrupted halfway through.

The model returns a structured match to zero, one or several candidates, or an unclear result.
It does not pick an arbitrary tool, mutate input state, approve an operation or start new work.

| Presented request and new input | Decision |
| --- | --- |
| Date question; "Friday" | Select that input occurrence. |
| Date question; "Why do you need that?" | No automatic response dispatch; let the current agent handle the side question. |
| Date question; "Also find a hotel" | New work for the current agent; the date request remains open. |
| First/last-name prompt backed by separate requests; "My first name is Ada" | Match only the identifiable first-name request; keep missing input pending. |
| Several questions; a response clearly addressing more than one | Select each identified occurrence; validate/reserve each independently. |
| "Yes" with several plausible targets | Clarify rather than choosing the latest request by default. |

Core validates every returned reference against the current conversation, authority, input
occurrence and response state. A verdict arriving after a request was replaced or answered
cannot reserve the old target. Persist accepted reservations with the canonical human-message
reference before dispatch. If the current local agent lacks the matching remote tool, retain
the response until an eligible agent returns, then revalidate. No outgoing A2A request is
needed while it waits.

The current agent still receives the human message and the framework's routing observation.
If an utterance answers a question and introduces another topic, recognizing the response
must not hide the remaining intent. Record the human message once. The frontend chooses new
remote work through its tools; a relevant tool call can reuse the same routing result and
reservation rather than consuming the response again. Both input admission and tool-call
routing use the same native validation/reservation/dispatch path, with no duplicate send.

An invalid verdict, provider failure or ambiguous match uses bounded retry/clarification and
keeps unresolved requests pending. Explicit references skip model-based matching but retain
native validation. With detection disabled, continuation requires an explicit reference.
Recognition never delays local voice interruption handling. Use the separately configured
routing model; automatic frontend-model reuse is outside the initial scope. The common
configuration spelling and exact retry settings remain design decisions; no new detector
base class is required. Explicit-reference mode with detection disabled needs no routing model.

Routing a continuation does not certify that its content satisfies the question. For example,
the expert can receive a clarification about an existing request and ask for the date again.
If one input request asks for both first and last name, a partial response can continue that
request and the owner asks for the missing information; grouping does not require splitting
one request into new records. Routing never grants approval for a business action.

### Streaming updates and reconnection

A long-running tool reports progress without waiting for its final result. Each ctx.update()
produces a working task status event; the open HTTP/SSE response or task subscription carries
it to the frontend's RemoteAgent reader. The conversation owner records the event and task
state, then offers eligible progress for presentation. Voice words it when the floor permits;
a text frontend sends its output on the client's stream. Receiving an update does not require
another frontend model turn, but choosing how to word it may. Flagged progress can be worded;
unflagged working text remains status. Exact text takes the ordered say path.

Streaming updates and accepting another conversational turn are separate capabilities. The
agreed opt-in live policy lets a turn release conversational ownership while its work runs on;
one owner still serializes conversation and result delivery. Progress itself never makes the
task completed. Slow or disconnected observers detach without canceling the business work.

Streaming progress and recovery of known task references belong to the initial integration.
Durable cursor replay is a follow-up beyond the first delegation delivery. The remaining
reconnection design describes that later target. Initial recovery reconciles known task state
and resumes observation where supported, with missing intermediate progress left explicit.

Our endpoints retain communication events durably before publishing them with a stable cursor.
Reconnect subscribes to the existing task after the consumer's last applied cursor, replays
retained events and joins live delivery without a gap. The consumer stores that cursor with
the corresponding accepted event/state update. A duplicate delivery must not append artifact
content twice or create another input request. State, event and cursor writes share the normal
fence/idempotent storage-write contract; publication cannot precede the required durable write.

Apply this independently to backend-to-frontend and frontend-to-client links. Each cursor is
scoped to its serving peer, authorized conversation/task and stream; it is not request_id or
the receiving journal's sequence. Cursor ownership and persistence belong to the native client/
subscription machinery. Applications do not implement a second recovery loop around RemoteAgent.

For example, a sixty-second tool sends progress at ten and forty seconds. The connection drops
at twenty seconds, and the tool completes at sixty. A reconnect at sixty-five resumes the same
task after the last applied cursor and receives the retained forty-second update and final
outcome. It does not send the original business request again. The frontend can summarize
missed progress; replay does not blindly speak every old update or prove earlier speech played.

Stock third-party A2A resubscription does not guarantee historical replay or stable event IDs.
Use a peer's replay/cursor capability only when supported. Otherwise reconcile with GetTask
and resume observation where possible, without claiming every intermediate update was recovered.
A task snapshot does not prove acceptance of a particular message. Missing task identity or
an unprovable outcome stays unknown. Expired cursors or gaps beyond retention are explicit;
they must not silently resume midway through artifact appends or masquerade as a complete replay.

### Failure, interruption and return

The owner reserves a pending question before sending its answer so two concurrent sends cannot
both consume it. If sending definitely fails before anything leaves, the question remains
answerable. If the expert might have accepted it but the acknowledgement was lost, the owner
keeps the answer's message identity and reconciles with the peer. Starting a fresh task could
repeat the work, so uncertainty is a recorded state rather than an assumed failure.

Spoken interruption affects presentation. If the person interrupts a result, the next frontend
turn reads the retained result and what was established as delivered. It does not race a second
independently regenerated result reply. A queued question or exact utterance also cannot be
recorded as heard merely because it was generated.

Ending the call closes its local media and subscriptions without canceling the remote tasks.
On reconnect, or when continuing by text, an authorized conversation loads durable delegation
and question references and reconciles the expert's current task state. Completed work can be
presented, and still-pending questions can be continued. This requires a stored thread/partition;
an in-memory conversation gains no automatic durability.

Explicit cancellation is separate: the application selects a recorded remote task and
records cancellation intent, journals the request/outcome and updates durable dispatch/task
state. A refused or lost cancellation response is not proof a booking was stopped. Similarly, recovery does not restore Python stacks, replay a
payment, or assume an uncertain spoken receipt was heard.

### Invariants

- One coherent foreground prompt can include several compatible input requests, with separate
  identities and presentation/response tracking. Passing over a question releases presentation
  priority without closing it. Follow-up priority covers only the answered expert's next output
  and respects the foreground prompt.
- Each question has an incarnation. Replays do not create new questions; stale answers cannot
  resolve replacements. Input predating presentation is not automatically an answer.
- Keep factual history, model-request notes and presentation receipts separate. Later updates
  become correlated notes, not fabricated tool calls. Continued tasks read relevant shared history.
- Progress-only wording cannot start business tools. Honor silent/no-reply outcomes; preserve
  exact-message ordering. After interruption, the next turn reads retained results and delivery
  facts instead of independently regenerating a competing result reply.
- Reserve before dispatch. Definite pre-send failure restores the input request; lost
  acknowledgement retains its in-flight identity for reconciliation. Retries preserve request
  and message/task identities. Supported client operations use the same write-before-I/O rule.
- Mutable request/task/input state owns recovery. The append-only journal records communication
  content/errors, with related state and log changes committed together. Local dispatch or
  stream-lifetime updates do not create extra communication events.
- Local failure/close never clears persisted dispatch uncertainty. A locally finished operation
  with no conclusive remote outcome still requires reconciliation; unsupported reconciliation
  remains explicitly unknown and cannot justify blind resend.
- Task projections reuse A2A models; direct messages do not become synthetic tasks. A closed
  stream is not a terminal task outcome. Public views show observations, not a complete remote
  inventory. Reconcile stale poll/stream races and ambiguous artifact replay explicitly.
- Shared Agent/RemoteAgent objects hold configuration, not conversation state. Validate peer,
  conversation and access independently of model-visible tool names.
- Every waiter settles on completion, failure, cancellation or owner loss. Call end stops local
  frontend resources without implicitly cancelling remote work. Local cleanup remains bounded.
- Restart reconciles durable references and remote state. It restores neither Python stacks nor
  the transient presentation queue, and does not blindly replay uncertain speech. Durable recovery
  requires an authorized stored thread/partition.
- Checkpoint accepted input before delegation reads the same partition. Refuse self/ancestor
  delegation that would wait on an owned partition. Preserve explicit application saves.
- Same-partition text reaches a call through its fenced owner and retains its transport task.
  RoomIO bindings and legacy primary selection do not select that owner. VoiceTasks own their
  input independently; agent replacement cannot grant access to an unrelated expert.
- Task IDs do not guarantee exactly-once business effects or speech across crashes. Use stable
  business operation IDs. Approval still matches validated arguments and consumes each answer once.

## Rust end state

VoiceSession, TextSession and bound Agent.run use the same native conversation coordination.
The local and remote delegation subjects stay separate, composed with the existing AgentRun
and voice machines. Rust owns conversation state, routing admission, recoverable dispatch,
presentation eligibility and result propagation. Python supplies application configuration,
tools, instruction callbacks and provider adapters through generated host contracts.

The native responsibilities follow the product expectations:

| Native component | Owns | Product outcome |
| --- | --- | --- |
| ABI declarations and codegen | Shared state/views, references, protocol exposure, verdicts, receipts and host contracts | The same inspectable state and typed API across execution modes; public Python forwarding and views are generated. |
| Core session coordination | Root/current agent, accepted input, history membership and shared local/remote coordination | A supported switch between voice and text continues the same conversation, with private agent histories and one human-facing history. |
| Core local delegation | Child creation/continuation, accepted output, parent-call resolution, dialogue merge and cleanup | A child can collect information and return its result without losing the person's dialogue or repeating an already accepted output. |
| Core remote delegation | Remote contexts/tasks/messages, input reservations, request dispatch/reconciliation and result correlation | Several remote tasks can progress independently; an answer or result retains its original destination across agent changes and recovery. |
| Core and voice presentation decisions | Shared eligibility and presentation selection; voice floor, foreground prompt, playback/interruption and VoiceTask interaction | Related questions and ordinary updates can be grouped naturally; pacing and exact output retain their delivery contracts. The model words the selected material. |
| Runtime executors | Active partition owner, effect execution, notifications, checkpoint ordering and owned-task cleanup | One writer advances a bound conversation while background work and frontend delivery have explicit lifetimes. |
| Net A2A client | Cards, SDK operations, streams and transport observations | The same RemoteAgent path talks to our endpoints and supported third-party peers. Restoring observation does not restart business work. |
| Inference | Detector model/provider behind an injected routing port | Routing can interpret the person's intent; native validation still decides whether the selected input request can accept that response. |
| Storage | Atomic framework checkpoints, canonical content, encoding, fencing, write receipts and leases | A fresh owner recovers retained state after a crash, while application-field saves remain explicit. |

The [SessionState proposal](session-state.md#combined-working-structure) owns the record
definitions. Its public SessionState/AgentState and remote views are native inspection APIs.
ToolCallRef, AgentCallState and input references retain their agreed roles; remote Task,
Message, StreamResponse and error payloads reuse A2A models. Local completion/output/error
fields remain ordinary public fields, independently of remote protocol task status.

Private native records retain request method/identity, dispatch progress, response reservations
and delivery facts. A held response references the canonical human message and selected input
occurrence; creating its outgoing A2A message is a later step. These records support the public
views without introducing another public controller or a second global question registry.
Data crossing the storage boundary is declared in ABI; storage never imports a core machine.
Storage encoding uses canonical content and references rather than rewriting copied histories
and the entire communication journal into every session checkpoint.

Shared coordination is extracted from the relevant parts of today's TextSession implementation.
Retain AgentRun's model/tool loop and voice's floor/media logic. The decision machines accept
typed inputs and injected time, then emit effects; models, databases and sockets stay outside
them. The runtime executes those effects and reports their outcomes. Internal enum states and
owned handles guard execution without adding Python state-variant classes.

For a bound remote call, the native boundary is:

1. Admit the human input and validate its call/input references. Reserve a response against
   its actual input occurrence when applicable; retain it if dispatch is temporarily ineligible.
2. Checkpoint the prepared outgoing request and communication entry. Persist dispatch_started
   before handing the request to the A2A client.
3. Execute the exchange. Feed received protocol content or a typed failure back into core;
   a closed local stream is not proof that the remote task ended.
4. Atomically checkpoint related request/task/input state and journal observations before
   dependent frontend delivery. Later cursor support joins this same atomic boundary.
5. Offer eligible material for presentation. Delivery receipts update the person's history;
   a queued event or received result does not itself prove delivery.

Unbound execution uses the same decisions in memory, without a durable recovery promise.
Recovery loads current state and reconciles uncertain operations. It does not restore Python
stacks or replay communication content as commands. An accepted child result is propagated
from stored output; an unprovable remote send remains uncertain. Business operation
idempotency remains the application's responsibility.

### Delegation code layout

Keep local agent-as-tool execution and remote delegation as distinct subjects under delegation/.
The core and runtime own their respective decisions and executors; a2a_client/ owns transport.
Use the existing crate graph:

| Crate | Target responsibility/folder |
| --- | --- |
| ABI | session/ for shared conversation records/references/views; delegation/local/ and delegation/remote/ for their orchestration contracts. a2a/ retains protocol/client/endpoint contracts; storage/ retains binding/checkpoint contracts. |
| Core | session/ for shared coordination; delegation/local/ and delegation/remote/ for their distinct decisions, retention and sibling tests. agent_run/ retains the invocation loop; a2a/ retains protocol and endpoint/task behavior. |
| Runtime | session/ for bound ownership and shared effect driving; delegation/local/ for child execution and delegation/remote/ for remote orchestration across persistence, detector and transport ports. |
| Net | a2a_client/ remains the reusable transport client. Move the current remote orchestration executor to runtime/delegation/remote/; net does not gain storage or inference dependencies. |

This local/remote separation and code layout are agreed.

Move existing code into these subjects and remove superseded paths. Regenerate affected
bindings from ABI instead of editing generated files. Keep transport, storage and voice floor
implementations in their owning crates. No new crate is needed. File moves and native behavior
changes are covered by the implementation checklist below.

Completion is demonstrated through readable product stories over these real executors, with
controlled model/host/transport boundaries and production storage in the existing DST. Add the
missing remote exchange and fresh-owner reconstruction seams rather than reproducing delegation
decisions in a fake. Check delivered conversation, remote requests and committed state together;
real HTTP/SSE peers, generated bindings and E2E provide the separate conformance checks.

## Primitive and API inventory

This is the review inventory for the planned domain records, execution interfaces, public
methods and test handles. It distinguishes new concepts from existing types being renamed or
extended. The record definitions remain owned by the [SessionState proposal](session-state.md).
Entries marked proposed are not agreed names/signatures. Implementation must not silently
introduce another domain object outside this inventory; ordinary private holders and generated
wrappers are implementation details of the listed objects, not independent lifecycle owners.

### Public state inspection

Applications receive framework-owned read-only views; these are not new controllers they must
construct. The default RemoteAgent.as_tool() application path keeps its existing shape.

| Type | Change | Purpose |
| --- | --- | --- |
| SessionState | Existing public facade, extended. | Shared root/current agent, human history and remote communication views across supported modes. |
| AgentState | Replaces/evolves SessionFrame; becomes the per-agent inspection view. | Private ChatContext, local calls/input requests and accepted output/error. Current voice activity enum has a naming collision to migrate explicitly. |
| ToolCallRef | Renames/evolves ParentCall. | Specific parent tool call receiving a child's output; not human-answer routing. |
| AgentCallState | Renames/evolves PendingChildCall. | Local agent-as-tool admission/execution, validated arguments and associated child agent-state ID. |
| AgentInputRequest | New durable local input record. | Local request for human input and its continuation/selected response. |
| RemoteAgentState | New communication aggregate/view. | One configured remote's retained messages, tasks, contexts and events for this conversation. |
| RemoteContextState | New derived read-only view. | Group existing messages/tasks by context; it is not another stored message history. |
| RemoteTask | New generated bound task view around the existing SDK Task observation. | Read task-owned input/cancellation state and issue cancel() through its native binding. |
| InputRequest | New remote-task input occurrence record. | Preserve source/response message IDs and response handling state through later task changes. |
| A2AJournalEntry | New typed communication entry using the existing journal storage path. | Record an SDK request, response or observed error with local operation correlation. |

The new inspection records are AgentInputRequest, RemoteAgentState, RemoteContextState,
RemoteTask, InputRequest and A2AJournalEntry. SessionState already exists; AgentState,
ToolCallRef and AgentCallState evolve existing records rather than adding parallel copies.

| Alias/closed set | Change |
| --- | --- |
| InputRequestState | New generated Literal values: open, responding, responded, superseded, closed. |
| A2AJournalKind | New generated Literal values: request_recorded, response_received, request_error. |
| A2ARequest, A2AResponse | New generated typing unions over supported SDK operation/result types; no new wire-model classes. |
| AgentCallState.status | Reuse the current approval/ready/running lifecycle; no new public AgentCallStatus variant hierarchy. |
| AgentActivity, name proposed | Rename the existing transient voice AgentState enum if needed to free the agreed per-agent state name; this is not another state machine. |

Message/A2AMessage, Task, TaskState, TaskStatus, Part, Artifact, AgentCard, SendMessageRequest,
GetTaskRequest, ListTasksRequest, CancelTaskRequest, SubscribeToTaskRequest,
SendMessageResponse, StreamResponse, ListTasksResponse, TaskStatusUpdateEvent,
TaskArtifactUpdateEvent and A2AError are reused SDK schema types. Their constituent protocol
types/codecs are generated from that schema rather than handwritten framework copies.
AgentClientError, RunResult, InputRequired, ToolResult, ChatContext and SpeechHandle already exist.

### Public configuration and methods

| Surface | Change or remaining contract |
| --- | --- |
| RemoteAgent(...).as_tool() | Existing entry point; no per-tool answer-detector/announcement options. |
| RemoteAgent(..., id=...), proposed | Optional stable configured remote identity, separate from a model-visible name. Exact spelling/default identity is unresolved; current RemoteAgent has name/description but no separate identity option. |
| DelegationOptions, proposed spelling | Common owner-level configuration for routing mode/model and dispatch acknowledgement. Automatic detection requires a separate explicitly configured model. Reuse existing connection/retry settings; no detector/provider class tree or automatic model reuse in the initial integration. |
| VoiceSession/TextSession and Agent.run/run_stream owner policy | Accept the same delegation configuration. An execution that performs automatic input detection must be given its separate routing model. Exact keyword placement remains open. |
| Agent.run(..., context=store.bind(...)) and run_stream | Extend existing methods to activate shared state. No new state input/result path; reject competing explicit history while bound. |
| SessionState.root_agent/current_agent/chat_history/remote_agents | Generated inspection properties. The remote properties messages/tasks/contexts/events expose the records above. snapshot() keeps its existing detached-read role. |
| TextSession detached state/history entry points | Migrate existing state arguments/results to the agreed binding path, with no compatibility path; add no new parallel continuation API. |
| ctx.update(text, silent=False) | Extend the existing update method with immediate enqueue and an optional-await admission receipt; add silent progress behavior. |
| ctx.say(text) | New exact-output method with immediate enqueue and an optional-await local delivery receipt. Return/interrupt/close outcomes still require a final signature. |
| RemoteTask.cancel() | Async command on the bound task view, with no repeated remote/context/task arguments. Success requires observed cancellation to be persisted; retain explicit refusal/uncertainty outcomes. |
| Agent.as_tool(..., merge_chat_ctx=...) | Rename the existing merge_chat_history option without a compatibility alias. needs_approval and fresh/fork remain existing capabilities. |
| inbox="live" | Add a value to the existing endpoint policy; no public LiveSession/LiveLane class. |
| InputRequired(question, verbatim=False) | Remove tool= and the associated blind direct-answer path. Match input first, then continue the owning local agent or the recorded remote task. |
| ctx.uninterruptible(), proposed | Long's protected business-step context manager needs a compatibility decision. It is separate from speech interruption and cannot promise process-crash survival. |

<!-- TODO: Revisit Long's protected-step API and its relationship to IGNORE_ON_ENTER and
RunContext.disallow_interruptions() in a later compatibility pass. Keep business-step lifetime
and voice interruption semantics explicit; this comparison is outside the current alignment. -->

### Internal native records and interfaces

Names below identify the intended responsibilities. A proposed internal name does not imply
a public Python export or an additional SDK constructor.

| Type/interface | New or reused | Responsibility |
| --- | --- | --- |
| SessionSnapshot, working name | Use one private conversation snapshot for every execution mode. | Persistent shared conversation data; separate from the public native view's resource handle. |
| Session, SessionConfig, SessionInput, SessionEvent, SessionEffect | Extracted existing TextSession coordination into core/session/. | Conversation records live at the domain entry, transitions in machine/ and typed effects in effects.rs. AgentRun remains the invocation loop. |
| SessionDriver, working name | Extract/evolve runtime text-session Driver. | One bound owner executing effects; voice retains its own floor/media driver. |
| AgentStateId, working name | Rename/reuse FrameId. | Native agent-context identity; Python can continue seeing a string. |
| ClientRequestId and InputOccurrenceId, proposed native-only wrappers | New typed identities if needed at the serialization/transition boundary. | Prevent mixing a client operation ID with a task's input occurrence; no new Python ID classes. |
| FrameOptions, provisional | New extracted immutable policy record. | Child policy snapshot currently including merge_chat_ctx. Intended to remain internal; whether the options view is publicly exposed and its final name must be decided. |
| RemoteRequestState, working name | Evolve/replace current InteractionState at operation scope. | Method, request/payload identity, caller/input correlation, dispatch_started and local operation lifetime. It does not replace the SDK task's status. |
| RemoteTaskState, internal data | Persisted data underlying the public RemoteTask view. | SDK Task observation, task-owned input state and cancellation intent; no serialized command capability. |
| RemoteAgentView, RemoteContextView and RemoteTaskView, internal ABI traits | Generated bindings for the three public inspection types. | Native storage handles retain the enclosing SessionState's bound or detached owner. Getter results copy SDK values; grouping adds no duplicate histories or network calls. |
| RemoteCommands, RemoteCommand and RemoteInput, internal ABI port/data | Serializes communication observations through the active owner. | Preparation, dispatch markers, responses and errors receive their checkpoint before the transport continues; no host policy or stored command capability. |
| RemoteTaskCommands, conditional internal ABI port | Extend the existing binding/owner command path where possible. | Bound RemoteTask.cancel() reaches the authorized owner without putting a runtime dependency in storage or adding a public controller. |
| RemoteTaskRef | Reuse/promote the current core A2A TaskRef shape. | Context/task addressing; keep internal unless a public control signature requires it. |
| InputRequestRef | New shared reference, not a lifecycle owner. | Qualify a local or remote input occurrence; keep internal by default. |
| InputRoute, proposed | New internal parsed detector verdict. | Match zero/one/several input references or request clarification; use existing LLM interfaces rather than a new public detector base class. |
| PresentationEvent | Evolve existing voice Presentation/ExpertInboxEvent records. | Transient normalized notification with original correlation, grouping and exact-output information. |
| SessionCheckpoint, proposed | New typed batch extending the existing checkpoint contract. | Atomically commit related framework state/history/task/journal changes without arbitrary application-field saves. |
| VoiceHistory, private I/O record | Captures the conversation and the database baseline observed by that snapshot. | Reconciled receipts retain their fence and observation order; overlapping writes cannot relabel a stale snapshot as a new local edit. This is not a persisted format or a public constructor. |
| TranscriptForward, private I/O request | Serializes the existing transcript writer's paced output, direct captures and delivery drains. | One physical output order and owned receipts across synchronization changes; no new Python API or presentation policy. |
| DeliveryReceipt, working name only | Internal awaitable adapter over existing signal/playback machinery if required by binding generation. | Immediate enqueue with an awaitable admission/delivery result. Public constructor/result type is not agreed; do not silently invent one. |
| HttpExchangePort, working name | New injected network boundary if the current concrete GatewayHttp exchange cannot be reused directly. | Let DST run real client/executor logic over fake HTTP observations; do not fake the entire remote delegation controller. |
| FakeA2AHttp, working name, native test-only | New implementation of that boundary in the existing DST. | Model accepted requests/lost responses, SDK task snapshots, stream fragments and faults without replacing native client or delegation decisions. |
| ProtectedStepGuard, conditional working name | Needed only if the protected-step context-manager contract is adopted. | Scoped cancellation protection with explicit ownership and shutdown bounds. |

Held human responses are private fields on their owning input record, referencing canonical
human content. No separate ResponseQueue/DelegationSubmission/OutgoingA2AMessage class is
planned. Durable presentation facts evolve existing TextSessionDelivery, ConversationReceipt
and voice playback/delivery records; do not add a second independent conversation history.

LocalInteraction, RemoteInteraction, existing DelegationHost/RemoteEvent contracts,
VerifiedAuthority, AgentRun, LocalAgentTool, SessionStateSink/SessionBinding, the A2A server
task/inbox machinery, RunWrites/Commit, Resend, signal cells, Owned and TaskSet are reused,
moved or narrowed. They are not additional new owners. Local delegation can remain functions
and transitions over the shared state; the folder split does not require another local
controller class. Existing errors are extended only for outcomes callers actually branch on.

The general E2E framework separately proposes four test-only types: E2E, RunningApp,
VoiceClient and ReceivedSpeech. Their APIs, reuse and fake-provider policy are owned by the
[E2E plan](e2e.md#planned-test-only-types). Text clients and model/service fakes reuse existing types.

There is no new public Delegation/DelegationState, unified local/remote call record,
QuestionGroup, Person/people fixture, detector-provider hierarchy or parallel protocol model.
Future durable stream replay adds scoped cursor state to subscriptions; it does not add a
public replay/journal controller to the initial integration.

This is the complete planned domain/API inventory, including conditional candidates. Final
codec row shapes, generated awaitable adapters and private transition-holder layouts are not
all specified yet. Validate those in the first schema/codegen proof and update this inventory
if they require another type. Do not claim that every eventual private Rust struct is already
designed, or conceal an additional domain concept as a mechanical helper.

## Implementation readiness

The product direction and local/remote layout are aligned. The general E2E foundation,
bound-state continuation, generated remote inspection and native communication journal are
implemented. Automatic input routing, task controls, live scheduling and complete voice
delegation remain implementation work. Several public signatures and native lifetime
contracts above remain open.
Do not treat the architecture agreement as approval of every provisional type name.

| Before this slice | Resolve |
| --- | --- |
| Shared native state/codegen/storage | Review this inventory; finalize agent/remote identity and initialization, options/view visibility, one snapshot encoding, bound-history editing and retention of referenced content. Prove the generated protocol/view representation. |
| Input routing | Finalize configuration spelling for the agreed separate routing model, explicit reference spelling/occurrence validation and custom-card schema collisions. |
| Live/exact delivery | Finalize update/say receipt return and interruption/close outcomes, protected-step behavior and RemoteTask.cancel() error/control-activation lifetime. |
| Cross-channel release | Finalize no-taskId inbound continuation versus merged task behavior and UI commands without an active owner. |

The first milestone includes native state and recovery in text **and pipeline voice end to
end**. Build state/storage first, then connect the same owner to voice; the milestone is not
complete at the text boundary. The recovery proof must include the real
remote request/observation path needed for one known-task, peer-accepted-response/lost-reply
restart story, alongside atomic framework checkpoints. A companion case retains an outcome
that the peer cannot prove as unknown. Scaffold work must not be reported as that recovery
proof. The voice proof includes a slow remote operation, progress, an input-required question,
a spoken response, interruption of frontend speech and the eventual result. Confirm actual
receiver audio as well as transcripts and service effects. Realtime parity follows that proof.

The voice DST remains a release gate. The 1,024-seed survey and eight-call soak pass. The
32,768-seed campaign completed with 20 contract/subject findings; the gate is still red.
An unchanged checkout also fails its initial survey, so each finding needs a focused
reproduction before attribution. Seed numbers alone do not identify equivalent cases across
revisions. Saved storage replays and authored regressions cover snapshot baselines, canonical
tool-result projection, recognition recovery, tool-group order and paced/unpaced transcripts.
Approval checks also account for a restored call consuming a grant and reject spending the
same answer twice.

Each storage snapshot retains its own read baseline and receives the actual reconciled write
result. Native write provenance distinguishes an earlier checkpoint from an intervening
writer's edit, even after lease reacquisition; submission order spans the whole activation.
Acquiring a lease does not publish unsequenced local revisions. A mutable baseline shared by
overlapping turns cannot establish which values a late snapshot intentionally edits.

The latest 4,096-seed window beginning at seed 1,024 passes. Focused regressions cover a final
realtime transcript retaining its known timestamp and position, a newer storage observation
updating a previously loaded row while preserving local edits, and restored tool outputs
keeping their canonical timestamps. A realtime tool can also retake the idle floor after its
speech is interrupted; speech interruption does not end application work.

Retained replays include seed 1,646 under Arrival, seeds 3,024 and 3,642 under BasicPos, and
seed 4,110 under TaskBridge. Rerun a finding with `VOICE_SEED=<seed> VOICE_POLICY=<policy>` on
its recorded revision. The full 32,768-seed campaign still needs another complete pass; the
smaller passing window does not close that gate.

Local real-process E2E and all v1 client compatibility stories pass. Cloud AgentDB conformance
has not run: this environment has no LiveKit project configuration. Local SQLite tests and
lost-acknowledgement fault tests do not substitute for that external gate.

## Questions retained during implementation

Record unresolved contracts here while implementation proceeds on the agreed boundaries.
Routine private type layout and generator work do not require another public primitive.

- When interruption prevents complete `ctx.say()` delivery, should awaiting its receipt raise
  an existing delivery error or return an outcome the caller must inspect? Either way, the
  receipt settles and remote business work continues. The recommended contract is an error.

- The pinned Python A2A SDK exposes protobuf operation models but no JSON-RPC error class.
  Native storage preserves the SDK's exact code/message/data. Decide whether public inspection
  should expose that error as its JSON mapping or through the existing typed client exception;
  neither choice should introduce a second wire schema or discard protocol error details.

- What exact explicit input-reference spelling should a custom RemoteAgent tool expose? It
  must identify an input occurrence as well as its task, and reject schema-name collisions.
- Which existing typed error should a task cancellation expose when a peer refuses cancellation
  or its outcome remains unknown after reconciliation? Successful cancellation still requires
  a persisted observed canceled state.
- The bounded protected-business-step API is deferred with the existing interruption-policy
  TODO; no implicit protected section is introduced by this work.
- For a text-to-voice transition while a local child is current, confirm that VoiceSession.start
  receives the configured root graph and restores the saved child from it. Supplying only the
  child cannot reconstruct missing parent behavior; it must not discard the saved parent call.

## Implementation checklist

Work in order. Each phase builds the mechanism needed by the next and includes its public
and internal documentation updates. The checkboxes are implementation actions; each exit
criterion states what must be observable before proceeding.

Start internally with native state and recovery: generated inspection views,
atomic framework-only checkpoints and a readable text crash/restart story. Prove that boundary
before connecting live scheduling and voice in the same milestone. The session checkpoint
commits related framework state/events together without implicitly saving application mutations.
Reuse the existing text DST over
production SQLite with injected connection faults; add remote exchange/restart coverage rather
than another simulation framework. Stock-peer HTTP and AgentDB conformance remain separate gates.

### 1. Contracts and baseline

- [ ] Resolve the API decisions above; map Long's final scenarios to owning tests.
- [x] Build the general E2E foundation over existing public APIs: one ordinary app process,
  an owned text client and a passing roundtrip/cleanup story, using scripted inference.
  This groundwork does not claim the new delegation recovery behavior is implemented.
- [ ] Preserve separate local-call and remote agent/context/task owners; evolve existing Rust
  InteractionState at per-request/task scope rather than adding a parallel lifecycle machine.
- [ ] Align public inspection with messages/tasks/contexts/events; finalize task-scoped
  cancellation and pre-task-ID uncertainty without a single-task delegation abstraction.
- [x] Record baseline failures/replay seeds and current SDK compatibility requirements.
  The unchanged initial survey fails five contract/subject categories. The current extended
  campaign remains a separate failing gate; retained journals and focused regressions drive fixes.
- [x] Use an isolated branch/worktree; preserve unrelated edits.
- [x] Consolidate local/remote delegation into their owning folders without behavior changes;
  retain public imports and regenerate affected bindings.
- [ ] Prove the first state/recovery slice with generated views and atomic storage before live/voice.

**Exit:** agreed API contracts and a traceable acceptance inventory.

### 2. Shared state, A2A and first recovery proof

- [ ] Replace split text/voice state with the agreed shared snapshot, one encoding,
  canonical history membership and generated inspection views. Reuse the existing local-call
  lifecycle and preserve accepted child output through propagation and cleanup.
- [x] Extend framework-only checkpoints to commit related state, canonical content, tasks and
  communication entries atomically under the existing fence/idempotent-write contract.
- [x] Prove SDK protocol-value conversion and generated Python views before multiplying
  hand-written adapters. Keep declaration and storage dependencies within the existing graph.
- [ ] Expose existing A2A Message/Task/StreamResponse and request/response models through
  generated bindings; preserve direct Message replies before presentation normalization.
  Reuse SDK method constants and generate any internal discriminator's host representation.
- [ ] Preserve SDK SendMessageResponse/StreamResponse unions and A2AError codes/details;
  validate response shapes per operation while retaining structured parts and extensions.
- [ ] Record SDK requests, responses and observed errors before lossy normalization for every
  supported conversation client operation. Require stable request_id on every event, including
  the outgoing request; journal sequence and message IDs do not identify the operation.
- [ ] Persist mutable dispatch/lifetime and caller/input correlation in existing request machinery;
  store the SDK method on that request record. Commit related request/task/input state and
  communication events together under the fence; retain method/identity while events reference it.
- [ ] Build public read-only messages/tasks/contexts/events views with consistent checkpoints,
  optional-history handling, observed-state semantics and bounded retention. Preserve the
  supported protocol's complete task-state set when aligning existing TaskState bindings.
- [ ] Normalize status, flagged progress, exact text, input requests, artifacts and outcomes
  for presentation without overwriting the original protocol observations.
- [ ] Assemble append/replacement/interleaved artifacts; define answer selection and typed-output validation.
- [ ] Support existing/new verbatim metadata without required LiveKit negotiation. Unflagged working text remains status.
- [ ] Add bounded card/503/stream recovery and reconciliation of potentially accepted sends.
- [ ] Define the stock-peer snapshot/resubscribe fallback without assuming replay, repeated
  message-id execution is safe, or unknown task identity can always be recovered.
- [ ] Preserve cancellation intent versus observed outcome; handle refused/lost cancel responses,
  stale snapshots and overlapping subscriptions without duplicate artifact appends.
- [x] Add the native exchange seam and a readable text DST story: a known task accepts a
  response, the reply is lost, the owner crashes and a fresh owner reconciles it. Check stored
  state and remote/service observations. A companion unprovable outcome remains unknown.
- [ ] Cover pre-dispatch failure and a database commit whose acknowledgement was lost;
  framework writes remain atomic and application-field mutations remain unsaved until explicit save.

**Exit:** generated views and atomic checkpoints pass; fresh-owner text recovery uses the real
client/executors/storage without blind resend or duplicated accepted outcomes. The client
passes deterministic stock-peer transport and failure tests.

### 3. Question routing

- [ ] Add detector port/provider, usage accounting, cancellation and bounded retries.
- [ ] Recognize responses to previously presented requests at input admission, including when
  the current agent lacks the matching remote tool. Persist held responses, revalidate before
  dispatch and deduplicate against later tool calls using the same shared reservation path.
- [ ] Allow corrections to held responses before dispatch begins; invalidate stale detector
  decisions by reservation revision. After dispatch may have begun, preserve the original
  message/operation and reconcile it; a correction is a separate continuation, not an overwrite.
- [ ] Implement explicit references and the shared validation/reservation/dispatch path.
- [ ] Implement eligibility, incarnations, passing, fairness and nested references.
- [ ] Use source_message_id/response_message_id for remote input occurrences and canonical
  message lookup. Retain references through task-status changes/history compaction, including
  message-less input and uncertain response delivery; align local source/response field names.

**Exit:** side questions/new work cannot silently become answers, including with one pending question.

### 4. Live execution and recovery

- [ ] Add opt-in live policy using existing AgentRun/TextSession/tools under one partition owner.
- [ ] Implement progress yielding, ordered delivery turns, task streams, sender-aware batching,
  local stream fast paths, explicit cancellation and protected steps.
- [ ] Use in-process events and cross-worker notification/storage boundaries; recover fenced
  records, executed outcomes and explicitly saved state after owner loss.
- [ ] Integrate bound-state inspection, task-scoped cancellation and durable-reference recovery.
- [ ] Restore durable request/task/input state and reconcile uncertain work; replay communication
  content only for indexes/debugging. Do not redispatch requests, repeat artifact appends or
  replay speech; retain references needed by unresolved input and delivery.
- [ ] Test crashes around intent, dispatch, peer response and atomic state/log persistence,
  including committed-then-lost database writes and history compaction with unfinished work.
- [ ] Correlate human history, private model contexts, remote communication and delivery for
  debugging, without duplicating histories or recording received content as heard.

**Exit:** real executors safely interleave turns, results, cancellation and owner loss.

### 5. Voice and shared text/voice

- [ ] Connect question/exact-text presentation to pipeline/realtime delivery receipts.
- [ ] Implement immediate update/say receipts, request-view notes and interruption handling.
- [ ] Prove that slow/stalled remote reasoning, tools and reads do not hold the frontend's
  conversational floor or delay interruption handling. Record foreground responsiveness
  separately from remote task duration; preserve existing false-interruption behavior.
- [ ] Implement the [conversation expectations](#conversation-expectations): paced questions,
  natural grouping of related questions, fair follow-up, grouped ordinary progress/results and
  removal of obsolete progress from pending presentation, while preserving exact output and
  durable communication records.
- [ ] Add the shared E2E voice client and receiver observations; exercise the same ordinary
  application modules with deterministic speech and inference before real-model evaluation.
- [ ] Route same-partition text into a call, preserving task/outcome and answered-by-voice attribution.
- [ ] Cover agent/tool replacement, VoiceTasks, close, reconnect and independent room bindings.

**Exit:** equivalent lifecycle semantics on text, pipeline voice and realtime voice.

### 6. E2E applications and third-party acceptance

- [x] Run a real HTTP/SSE stock A2A SDK server in another process without LiveKit dependencies,
  custom metadata, shared storage or required extension negotiation.
- [x] Run ADK's real A2A server in an isolated environment with a deterministic model stub.
  The ordinary frontend reaches its agent and executes a tool through the same RemoteAgent path.
- [ ] Define failure for unsupported vendor-private input protocols; decide adapters separately.
- [ ] Author ordinary SDK applications under tests/e2e/apps/ and exercise them through the
  E2E framework below. Cover frontend -> flights/hotels -> payments, overlapping requests,
  interrupted receipts and idempotent application booking/payment operations.
- [ ] Update maintained clients/console for references, offsets, restoration and voice attribution;
  confirm where their source lives before applying upstream UI changes.

**Exit:** the E2E applications, stock A2A and ADK pass through the same RemoteAgent execution path.

### Follow-up: durable stream replay

This extends the initial integration; it is not required for the first delegation delivery.
Task/reference recovery and truthful snapshot/resubscribe fallback remain in that delivery.

- [ ] Persist outgoing stream events before publication; resume retained task streams from
  durable consumer cursors, including replay-to-live handoff, slow observers and expired cursors.
- [ ] Persist received event/state/cursor together and deduplicate supported replay without
  applying artifact appends or creating input occurrences twice. Cover both client links.
- [ ] Exercise the stream-replay acceptance stories below, including receiver/sender restart
  and a reconnect after the task completed while the observer was offline.

**Exit:** peers supporting durable replay recover retained updates without restarting work;
unsupported replay and expired cursors have explicit outcomes.

## Acceptance tests

Expand these into independent regressions. Run applicable cases across text, pipeline voice
and realtime voice, with local/remote parity where transport is irrelevant.
Use the [conversation expectations](#conversation-expectations) as readable delivery stories:
five simultaneous questions, a combined first/last-name prompt, ten queued progress updates,
results overtaking progress and interruption. Assert both the person's delivered conversation
and the separately retained input/task/event state; exact output must not be coalesced with
ordinary progress.

| Area | Required cases |
| --- | --- |
| Responsiveness | Long-running or stalled remote work while the person speaks, interrupts, asks a side question or changes a constraint; prompt local playback stop, available frontend turns, grounded acknowledgement/status, stale results not presented as satisfying corrected constraints |
| Intent/routing | One/multiple questions; answers, side questions, new work, correction, refusal, silence; new/continue/unclear, invalid verdict, retry/exhaustion/cancel; missing/foreign/stale references |
| Questions | Queued/unheard/partial questions, naturally grouped first/last-name questions, partial response to a group, clear responses to several requests, interruption partway through a grouped prompt, input before presentation, passed questions answered later, fairness, double answers, one of several requests resolved |
| Nested/isolation | Multiple experts/tasks/hops, same tool names across peers, shared configuration across users, self/ancestor partition refusal |
| Updates/history | Immediate result after progress, bursts/silent/exact/no-reply updates, results overtaking progress, no fictitious calls or stored routing instructions, shared history and partial delivery |
| Tasks/streams | Same-task continuation, concurrent tasks in one context, contextless/direct messages, repeated questions, duplicate messages, terminal follow-up, context mismatch, merged tasks; drops before/after ack, mid-artifact or post-commit; duplicate/stale events, poll/stream races, resubscribe/get fallback, unsupported replay; stream closure without task completion |
| Stream replay follow-up | Sixty-second tool with several updates; detach/reconnect and completion while offline; sender/receiver crash around event/state/cursor commit; duplicate chunks, replay-to-live races, expired cursor/retention gaps, slow observer detach, no task cancellation or request resend to restore a stream |
| Journal/views | Supported-operation request correlation, native responses/errors, no dispatch/stream-state marker entries; atomic state/log writes, restart from current state, content replay without I/O/speech or repeated artifact appends; message IDs echoed in task history/status, conflicting payloads, optional/truncated history, partial list observations, context grouping, read-only snapshots and compaction of pinned references |
| Cancellation/storage | Barge-in versus cancel, protected steps, queued-question cancel, late success, reader close, dropped executor/acquire, refused save, committed-then-lost writes, lease loss, worker death during an answer, stale status, saved state/outcomes |
| Public APIs | Both schemas, custom card schemas/overrides, bindings/types; awaited/unawaited update/say, dropped receipts, observer failure; inspect before task ID exists, task-scoped cancel after reconciliation/terminal state/reconnect, unauthorized scopes |
| Channels/recovery | Text during calls, results after newer input, question handoff, overlapping senders, reconnect, agent/tool replacement, VoiceTasks, multiple bindings, restored/expired references, uncertain speech |
| Third party | Stock SDK/ADK, direct Message versus Task on the same send path, unflagged status, input-required, interleaved/appended/replaced artifacts, empty/structured/file output, metadata/extensions, failed/rejected/canceled/auth-required, SDK error codes/details, missing IDs, wrong-operation response shapes, unsupported versions/required extensions and malformed frames |
| Authority | Forged metadata/IDs, expired grants, peer/context/task mismatch, cross-user references, URL authority and removed tool access |
| Approvals/accounting | Yes/no/side question, changed arguments with identical read-back, parallel repeats, one-time grant; lost reply after business effect; usage counted once, correlated spans, bounded resources and PII-aware diagnostics |

Use machine tests for transitions and real executors over injected ports for runtime behavior.
Extend DST execution coverage for live backend/cross-channel races; an invariant alone does not
exercise an executor. Hermetic tests use observed events and virtual time. Keep paid-model
experiments separate from deterministic stock-peer/ADK acceptance.
Use real conversation/audio evaluation to assess wording, pacing and interruption quality.
DST proves scheduling, correlation and delivery behavior under controlled inputs; passing it
alone does not establish that a model's conversation sounds natural.

### E2E authoring framework

These journeys use the general [shared E2E framework](e2e.md), which owns application launch,
voice/text clients, resource cleanup, artifacts and the deterministic-provider policy. Voice
callers are one capability of its e2e fixture; delegation does not add a separate people API.

Delegation test apps use ordinary public SDK APIs. The frontend, remote agents and input-routing
model use scripted inference for correctness runs, while real native delegation decisions,
task state, storage and RTC/A2A transport are exercised. Scripts provide model output through
the normal parser/validation path; they do not implement a second delegation state machine.
Live-provider conversation evaluation runs separately through the same applications.

### E2E journeys

Use a small set of concierge app modules with configurable remote endpoints and controlled
business services. Reuse them across journeys; keep individual regressions focused on one
outcome. The first implementation proof remains text state/storage recovery, followed by live
voice stories. These are acceptance tests, rather than new product demos under examples/.

| Journey | What the person does | Evidence |
| --- | --- | --- |
| Slow work, responsive voice | Start a long search, interrupt speech, ask a side question and correct a constraint while the remote works. | Recorded audio and foreground timings; the accepted correction and remote requests; eventual results match the current request. |
| Parallel questions and progress | Start flights and hotels, receive a burst of updates, answer a grouped first/last-name prompt partially and answer another question later. | Natural grouping and pacing; individual input reservations; responses reach their actual targets and routine updates are summarized. |
| Response across a local handoff | Hear a remote question, enter a local child without that tool, answer the earlier question and return to an eligible agent. | Response retained through the handoff without dispatch, sent to the original task once eligible, no duplicate human-history entry or forced agent switch. |
| Return and recovery | Leave the call while work runs, return by text on the same thread/partition, then exercise a separate fresh-owner restart after a peer accepted a response but its acknowledgement was lost. | Retained task/input identities, reconciled outcome, business-service ledger and atomic storage records; no blind resend, implicit cancellation or uncertain speech replay. |
| Third-party remote | Point the same frontend at a stock A2A SDK peer, then an ADK peer. | Real HTTP/SSE in separate processes, supported questions/results on the same client path, truthful behavior when optional progress/replay capabilities are absent. |

Each E2E journey uses real process/room/transport boundaries where applicable. Controlled
services make slow work and failures reproducible; deterministic tests release explicit gates
rather than sleep for a minute. A real-model voice run assesses the conversation itself.
Keep the audio/transcript, correlated communication journal, business-service ledger and
foreground timings as reviewable evidence. Exact latency budgets remain to be agreed.

## Documentation and release gate

- [ ] Lead each maintained feature guide with the expected person/application experience,
  concrete examples and limits, then explain the native mechanism and API. Link each product
  expectation to its implementation owner and a readable acceptance story.
- [ ] Update `tools.md` and `protocol/a2a.md` for public API and wire/task contracts.
- [ ] Update `server.md`, session guides and `persistence.md` for policy, delivery and recovery.
- [ ] Document machine/effect ownership and task/close behavior in internal guides.
- [ ] Publish third-party setup, E2E app authoring, journeys and replay instructions in testing guides.
- [ ] Run codegen/architecture checks, bindings, lint/types, Rust and hermetic Python suites,
  full DST survey/campaign/soak and real-process E2E. Compare baseline failures and retain seeds.

Give each exhaustive contract one owning guide. SessionState owns state/history/binding/recovery;
delegation owns conversational coordination, routing and remote communication. Update the
maintained guides with each implementation slice and retire completed proposal content.
Durable stream replay is a follow-up scope item, not a gate for the first delegation delivery.
For each public method, document immediate behavior, await guarantees, persistence and failure
outcomes. Migrate implemented contracts out of this proposal. Release requires no unaccounted
waiter/task, no new ownership/cancellation regression, no unvalidated answer dispatch,
stock-peer compatibility, and explicit disposition
of known failures. Do not weaken assertions to obtain a passing run.

## Source stack

Long's stack supplies behavior and edge cases. The official A2A SDKs supply the protocol model
and operation boundaries; pinned source references and the reuse decisions live in
[What we reuse from A2A](session-state.md#what-we-reuse-from-a2a). The framework adds durable
communication history, local routing and delivery tracking rather than copying SDK helper classes.

Base: [#7757](https://github.com/livekit/agents/pull/7757). Reviewed head:
`e0fff7d69f476b8a9491a28093d20a4b1f1a9e39`.

| PR | Contribution |
| --- | --- |
| [#8200](https://github.com/livekit/agents/pull/8200) | Hermetic storage/A2A tests |
| [#8201](https://github.com/livekit/agents/pull/8201) | Exact text, artifact answers, conversation-scoped state |
| [#8202](https://github.com/livekit/agents/pull/8202) | Failure identity, retained outcomes, expert recovery |
| [#8203](https://github.com/livekit/agents/pull/8203) | Continuation and nested question references |
| [#8204](https://github.com/livekit/agents/pull/8204) | Live inbox, background work, streams and cancellation |
| [#8255](https://github.com/livekit/agents/pull/8255) | Question queue and interrupted-result records |
| [#8256](https://github.com/livekit/agents/pull/8256) | Call/text continuity, ordering and barge-in |
| [#8421](https://github.com/livekit/agents/pull/8421) | Model-mediated routing and passed questions |
| [#8493](https://github.com/livekit/agents/pull/8493) | Stock A2A/ADK and flagged progress |
| [#8532](https://github.com/livekit/agents/pull/8532) | Final delivery/history rules, multi-hop and Getaway |

Use final stack behavior: later PRs replace forced forwarding, task-filtered conversation
history and automatic interrupted-result regeneration. Long's remaining TODOs—duplicate result
replies, re-sending in-flight work and transient 503 recovery—remain acceptance work here.
