# Voice

A call runs on a `VoiceSession`: the runtime a room attaches to, with the
models, turn detection and audio. It runs an `Agent`, and it runs it in the
same thread a text endpoint uses, so a call and a text in one thread
share one history and one state.

## The voice door

```python
@server.rtc_session()
async def phone(ctx: JobContext) -> None:
    app = SchedulingContext(clinic=clinic_service)
    session = VoiceSession(
        llm=openai.realtime.RealtimeModel(), context=app, thread=ctx.thread
    )
    await session.start(agent=scheduler, room=ctx.room)
    await session.store.load_context(app)
    await session.generate_reply(instructions="Greet the caller and ask about their appointment.")
```

The door is `{agent_name}/{endpoint}`: the agent's name from
`AgentServer(agent_name=)`, and `rtc_session(endpoint=)`, the agent's name
again when left empty.

| Parameter | What |
|---|---|
| `llm` | Session default when the agent does not select a model; supports a realtime model |
| `stt`, `tts`, `vad` | Session-owned recognition, synthesis, and voice activity models |
| `turn_detection`, `endpointing_opts` | Session-owned turn completion and endpointing policy |
| Interruption and delivery options | Session-owned interruption handling, pacing, and expressive speech |
| `context` | the application object; marked fields require explicit loading and saving |
| `thread` | the call's durable thread, `ctx.thread`; omitted, the history is kept in memory and not saved |

`session.start(agent, chat_ctx=)` and `session.update_agent(agent, chat_ctx=)`
are the two places to supply a different history. An explicit agent switch
retains the current history when no replacement is supplied.

In a tool on a call, `ctx.live_session` is the call (`say()`,
`generate_reply()`), and `ctx.thread` is the durable thread, the same
object a text handler gets. Outside a call `ctx.live_session` is `None`.

## Updating a live session

Configure audio on the session and update it explicitly during the call:

```python
session = VoiceSession(
    stt=recognizer,
    tts=front_desk_voice,
    vad=voice_activity_detector,
    turn_detection="vad",
    context=app,
    thread=ctx.thread,
)
await session.start(agent=front_desk, room=ctx.room)

# Change presentation and listening policy when the application needs it.
session.update_options(
    tts=specialist_voice,
    turn_detection="stt",
    endpointing_opts={"min_delay": 0.3, "max_delay": 2.0},
)
session.update_agent(specialist)
```

`update_options` updates only the supplied settings on that session. It can
replace STT, TTS, and VAD models and change turn detection, endpointing,
interruption, and delivery settings without restarting the call. The selected
model must support the requested behavior; incompatible updates fail explicitly.
For realtime sessions, provider capabilities determine which speech and
turn-detection settings can change on the active connection.

Updates are applied promptly at the relevant processing boundary. New speech
uses the updated synthesis settings; audio already generated or playing is
not regenerated. Listening changes apply to subsequent processing without
silently discarding buffered input or duplicating a user turn. Changing options
does not itself interrupt speech, reset history, or restart the agent.
Provider connection setup can take time; an immediate configuration update does
not promise zero-latency model replacement.

`update_agent` changes the agent's instructions, tools, and reasoning behavior.
It preserves the session's audio settings. VoiceTask activation and restoration also
preserve those settings. The application calls `update_options` when a routing
change should alter presentation, and explicitly restores the desired settings
when returning to the parent. Multiple sessions using the same agent can have
different voices and listening policies.

## The call's thread

On a server with storage, the door resolves the call's thread before the
entrypoint runs: the `thread_id` key of the dispatch's JSON metadata, as
the web console sends it, else a new thread. It is `ctx.thread` on the
`JobContext`, `None` on a server without storage, and
`VoiceSession(thread=ctx.thread)` is how the call gets it. A `VoiceSession`
without `thread=` keeps its history in memory, not saved.
For phone-number routing and conversation continuity, see
[channel routing](channels.md#channel-routing).

At `start()` the call loads the agent's lane: the stored history when the
call was handed none. Application context fields are explicitly loaded with
`session.store.load_context(app)` before generating a reply. Each
voice turn then takes the lane from the end of the person's turn to the
commit of the reply, and releases it without draining, so a text that
arrived meanwhile is answered by the worker waiting on it. When a text
turn moved the lane during the call, transport history is reloaded before the model reads it. Application code
reloads context fields at its turn boundary before using changes from other
activations. A
turn that produced no reply still releases the lane within 30 seconds.
Storage failures are logged and never end the call.

## The thin voice door

The realtime model is good at the floor: small talk, interruptions, pacing.
It is not where the tools and the state should live. An `RemoteAgent(...).as_tool()` in
the voice agent's `tools` gives it one tool, the brain behind the text
endpoint, named and described from that endpoint's card, and speaks the
brain's progress while it waits.

```python
back_office = RemoteAgent("http://127.0.0.1:7880/agents/dentist/default/backend")
voice = Agent(
    instructions="You are the practice's voice on the phone. Hand every request to the back office.",
    tools=[back_office.as_tool()],            # one tool, ask
)

@server.rtc_session()
async def phone(ctx: JobContext) -> None:
    session = VoiceSession(llm=openai.realtime.RealtimeModel(), thread=ctx.thread)
    await session.start(agent=voice, room=ctx.room)
```

The remote service owns its conversation and approval behavior. The adapter
places expert progress, questions, and outcomes in the conversation-owned inbox.
The live session schedules frontend presentation around speech and interruptions.
The fast brain phrases questions while the live session correlates replies with the
remote task. Completion supplies the
parent's tool result; a failed remote task becomes a tool error.
Connection lifecycle and delegation metadata are managed internally. See
[remote agents as tools](agents.md#remote-agents-as-tools) for routing and
persistence ownership.

One brain, two doors: voice and text share one set of tools, explicitly persisted application data
and one history. The front desk answers a call today and a text tomorrow from
the same memory.

## Expert events and frontend presentation

`VoiceSession` owns frontend remote-tool interactions: it retains pending
expert references, routes human replies, and schedules presentation. The
frontend attaches `RemoteAgent(...).as_tool()` through its agent's `tools=`
and starts the live session. It does not construct a `TextSession` or pass
backend `TextSessionState` to the session.

## Conversation-owned inbox

Expert events enter an internal inbox scoped to one frontend agent's
conversation. Each frontend conversation has its own inbox and pending reply
destination, even when several frontends share a thread database. Backend
agents do not acquire a presentation inbox merely by using `Agent.run()` or
`TextSession`.

The inbox belongs to the frontend conversation, not the shared `Agent`
configuration object or the thread as a whole. Replacing the speaking agent
within that frontend conversation preserves the inbox; switching to another
frontend conversation does not transfer its events. Applications use the existing
session APIs; they do not construct or dispatch an inbox themselves.

### Frontend queue versus backend inbox

| Mechanism | Scope | Contents | Delivery |
| --- | --- | --- | --- |
| Frontend event queue | One frontend conversation | Expert progress, questions, results, and failures | In-process push and wake-up of the live session |
| Backend request inbox | One transport lane | Incoming requests buffered by endpoint policies such as batching | Managed request scheduling and storage |

The frontend event queue needs neither AgentDB nor a durable queue service.
It is an internal in-process queue for the active frontend session. A backend
can persist its own conversation independently without making this queue durable.

### How the frontend receives events

1. The expert adapter receives an update and pushes a correlated event into
   the frontend event queue.
2. Enqueueing wakes the live session's event loop; there is no database polling.
3. The session processes queued events when the active agent exposes the
   corresponding remote tool and speech, interruption, and frontend-task
   scheduling permits presentation.
4. Useful progress may be combined; a question can trigger a fast-brain model
   invocation to phrase it. Enqueueing alone does not invoke the LLM.
5. Before delivering the question, the live session records the expert destination
   for the user's reply in the active conversation's runtime state.

Applications do not call a dequeue method or run a polling loop. While the
floor is occupied, the queue retains events for the session to process later.

Each event identifies its delegation, source, destination, and occurrence.
The live session retains its association with the originating frontend tool
call. Remote events also retain the endpoint, context, and task references
needed to continue the expert conversation. The adapter normalizes local and
remote lifecycle events before delivery.

| Event | Session responsibility | Frontend presentation |
| --- | --- | --- |
| Progress | Retain correlation and ordering; coalesce superseded progress | Present useful updates when the conversational floor permits; an update need not trigger an LLM call |
| Input required | Record the pending question and its expert as the reply destination | Ask through the fast brain, preserving meaning; deliver verbatim text unchanged |
| Completed | Resolve the recorded parent tool call with the accepted result | Let the caller's next model invocation present the result |
| Failed or cancelled | Settle the delegation with its recorded outcome | Explain the outcome through the frontend when appropriate |

### When the fast brain runs

Receiving an event does not immediately start a competing speech or model
invocation. The live session serializes presentation with its current speech,
interruptions, and frontend tasks. Progress can be combined or omitted when
stale. Questions and final outcomes remain correlated and are not discarded
as replaceable progress.

While an expert is waiting, the fast brain can make a presentation-only model
invocation to phrase its question or progress. That invocation receives a
model-valid presentation view containing the relevant visible conversation
and expert event. It does not resume the parent's ordinary tool loop, invoke
new business tools, or mark the expert call complete. The parent's pending
tool call remains recorded until the expert supplies its outcome. The full
parent history is used again only after that call is resolved.

### Routing the person's reply

An expert's input request is presented by the frontend and the person's answer
is forwarded to that expert. The presentation step does not select a local
collection tool or start a `VoiceTask` on the expert's behalf. For example,
"What email should receive the receipt?" is phrased for the passenger; their
answer goes back to the waiting refund task. Expert-triggered structured
collection, such as automatically invoking `GetEmailTask` for that question,
is outside this contract.

The live session records the expert question before presenting it. Rephrasing
changes its wording, not its identity or reply destination. An ordinary answer
to the foreground question is forwarded to that expert without requiring the
model to select the delegation tool again. Cancellation, a new request, or an
ambiguous response must be handled explicitly rather than blindly forwarded.
Only one expert question owns the foreground reply destination at a time.

A frontend `VoiceTask` can temporarily own the live interaction. Its user
turns answer that task; they are not also forwarded to the waiting expert.
Expert events remain queued until the frontend can present them. Completion
of the frontend task restores its caller without implicitly completing or
cancelling the expert delegation.

### Extending input collection

Keep the expert's pending input request separate from its presentation. Its
internal record retains request identity, delegation/task references, reply
destination, and lifecycle state. Rephrasing text, replacing the eligible
frontend agent, or temporarily running a live task must not lose that identity.
A received human answer is correlated with the request before being adapted
to the expert's supported transport format.

The current collection path presents the question and forwards the human's
answer. Keep that collection step separate from reply correlation and delivery,
so another collection strategy can later supply an answer to the same pending
request. Existing experts can continue to ask plain questions; they must not
require collection-specific metadata.

A future structured collector could use optional input hints and a local
`VoiceTask`, then explicitly submit its validated result as the answer to that
request. This is an extension point, not an additional public API or automatic
behavior. A live task's result must never be forwarded merely because an expert
is waiting. Completion, cancellation, and stale-request checks remain tied to
the exact pending request; a late collector result cannot answer a different
question. The expert remains responsible for validating the received answer
and authorizing its own business actions.

### Agent replacement and persistence

Replacing the speaking agent does not delete the inbox, cancel a delegation,
or change its originating call. Queued expert events become eligible for
presentation when the active frontend agent exposes the corresponding remote
tool and the conversational floor is available. It can be the original agent
or a different configured agent with access to that same remote tool.

Tool access is matched against the configured remote-tool identity, not merely
a similar model-visible name. A replacement without that tool cannot interact
with the expert, including its pending task, and the events remain queued.
The live session retains correlation independently of the active agent's tools;
a queued event never grants access by itself.

If an expert question was already delivered before switching to an agent
without its tool, retain the question but suspend its automatic reply routing.
The replacement's next user turn must not silently answer an expert it cannot
interact with. When an eligible agent becomes active, restore the pending
interaction and clarify or re-present the question when needed before forwarding
a relevant answer. Settled or superseded questions cannot receive late answers.

An eligible agent can present the expert's updates and continue its pending
interaction. The existing delegation, remote task, and originating call IDs
remain unchanged; regaining access does not create a new remote request or
reassign a result to an unrelated tool call. The
[airline refund example](voice.md#example-airline-refund-and-receipt-email)
shows updates waiting during a frontend inline task.

If no eligible agent returns, the application must eventually explicitly
settle or cancel the pending interaction. The session does not automatically
switch agents, start another remote request, or deliver the result through an
unrelated agent. Cancellation is not proof that a refund or other remote effect
did not happen; the authoritative service must reconcile uncertain outcomes.

The default queue and its active reply correlation live in process. They can
end with the frontend session. They are not backend routing frames or
automatically saved application fields. Saving backend history or application
fields does not make frontend event delivery recoverable.

Persistence is optional when the application needs pending frontend events and
reply routing to survive disconnects, worker changes, or restarts. That path
stores serializable event records, processing positions, and delegation/reply
references under the originating frontend conversation's identity. It never
serializes a queue object or Python await. Applications explicitly save retained
frontend records; durable delivery and recovery require transport reconciliation.

Another frontend in the same thread cannot consume these events simply because
it shares a database. Duplicate event delivery must not resolve a tool call
twice or present the same retained occurrence twice during normal processing.
A crash between delivery and acknowledgement can still cause uncertainty;
optional persistence does not promise exactly-once speech or external effects.

The frontend queue coordinates expert events and presentation. It does not add
background execution, a scheduler, or Python stack recovery. The backend request
inbox continues to serve endpoint request policies independently.


### Example: airline refund and receipt email

A passenger asks for a refund. The frontend calls the remote airline agent,
which checks the ticket and handles refund eligibility, approval, and execution.
While it works, the passenger asks to use a different email for the receipt.
The frontend collects that address with a local live task.

```python
from dataclasses import dataclass

from livekit.agents import Agent, RunContext
from livekit.agents.a2a import RemoteAgent
from livekit.agents.beta.workflows import GetEmailTask


@dataclass
class AirlineContext:
    booking_reference: str
    airline: AirlineServices


refunds = RemoteAgent(REFUND_AGENT_URL)
front_desk = Agent[AirlineContext](
    instructions=(
        "Help passengers with refunds through the airline refund expert. "
        "If the passenger asks to change their receipt email, collect and save it. "
        "Never promise a refund until the expert confirms it."
    ),
    tools=[refunds.as_tool()],
)


@front_desk.tool
async def change_receipt_email(ctx: RunContext[AirlineContext]) -> str:
    """Collect and confirm the email address for this booking's refund receipt."""
    # Passenger: "Please send the refund receipt to a different email."
    # The refund expert may still be checking the ticket.
    # INLINE TASK: temporarily owns the floor to collect and confirm the address.
    # It has no remote tool and cannot interact with the refund expert.
    # INBOX: expert progress/questions/results arrive in the frontend queue
    # and wake the session, but wait for email capture to finish.
    # The passenger's spelling and confirmation answers go only to this task.
    result = await GetEmailTask().run()

    # The task restores front_desk; no explicit update_agent() is needed.
    await ctx.context.airline.set_refund_receipt_email(
        booking_reference=ctx.context.booking_reference,
        email=result.email_address,
    )
    # INBOX: after this foreground tool finishes and the floor is available,
    # front_desk can present queued events because it exposes refunds.as_tool().
    # Another active agent with that same remote tool would also be eligible.
    # No polling or manual drain.
    # Progress can be combined. An expert question is phrased by front_desk;
    # the live session sends the next relevant answer to the recorded refund task.
    # A refund completion resolves the original remote call before the next
    # ordinary frontend model invocation announces the confirmed outcome.
    return f"The refund receipt email is now {result.email_address}."
```

For example, the refund expert may ask whether the passenger accepts a partial
refund while the email task is confirming a spelling. That question stays queued
until the task returns and the frontend tool finishes. The email confirmation
is not treated as acceptance of the refund. The frontend then asks the refund
question, and the live session sends its answer to the expert.

The frontend selects `change_receipt_email` for the passenger's explicit new
request, using a model-valid foreground view while the remote call remains
pending. Expert-event presentation alone does not invoke this tool. This
example handles the passenger's separate request to change their email; it
does not turn an expert's email question into a local collection task.
`AirlineServices` is application-provided; it saves the receipt destination in
the authoritative booking/refund record used by the expert. The service enforces
whether the receipt address can still change. Saving the address neither approves
nor executes a refund. The host supplies the authenticated booking context to
the live session. The frontend event queue is in-process and requires no DB.

## VoiceTask: frontend live interactions

`VoiceTask` is a temporary interaction run by the fast brain: the frontend
agent attached to an active voice session. It can collect an email address,
confirm a spoken name, gather DTMF input, or guide another bounded exchange
before returning a typed result to the frontend's calling tool.
`VoiceTask[Output]` describes the typed live-interaction primitive; helpers such
as `GetEmailTask` return their domain result through `.run()`.

Calling `.run()` resolves the active `VoiceSession` from the current frontend
invocation. No session argument or application-side session check is required.
The binding is invocation-local, so simultaneous sessions cannot share it.
Calling a live task without an active live-session interaction raises a clear
error before the task starts. Text-mode backend runs do not provide this binding.

The task temporarily takes the conversational floor. The parent frontend
waits in memory while the task receives live user turns and invokes its model
and tools. On accepted final output, the task returns its typed result and
restores the parent, whose model can continue from the tool result. This is
suspension and restoration of the same frontend activation: the parent's
`on_exit` and `on_enter` hooks are not called for the task transition.

The lifetime is the active `VoiceSession`. Completion, cancellation, and session
closure end the task. Cleanup releases the floor and restores any temporary
session presentation settings, including on failure. Audio configuration
remains owned by `VoiceSession`; application changes during the interaction
must have explicit restoration behavior.

This is an in-memory frontend interaction, not a stateless workload or backend
execution primitive. Its parent await remains on the running Python stack.
Saving history or application fields does not restore that await after a
worker restart, deployment, or session loss. It does not provide background
execution, scheduling, or cross-request task recovery.

Use `Agent.run()` for stateless backend invocations and `TextSession` for
backend agent-tool conversations that continue across requests using saved
routing state. A frontend task and a backend delegation have separate owners:
temporarily collecting live input does not silently cancel, transfer, or
complete an outstanding remote task.

### Authoring a custom VoiceTask

Construct a `VoiceTask` and register its tools as functions. `output_type`
declares the typed result; returning `Completed(value)` from one of the task's
tools completes the live interaction. Ordinary tool values and conversational
speech keep it active. The framework validates the completed value before
returning it to the waiting caller and restoring that caller's live role.

```python
from pydantic import BaseModel, EmailStr

from livekit.agents import Completed, VoiceTask, RunContext


class EmailResult(BaseModel):
    email_address: EmailStr


def collect_email() -> VoiceTask[EmailResult]:
    task = VoiceTask[EmailResult](
        instructions=(
            "Collect the passenger's receipt email. Clarify spelling when needed. "
            "Call confirm_email with the address. Its approval read-back asks the "
            "passenger to confirm; finish only after they approve that exact address."
        ),
        output_type=EmailResult,
    )

    async def read_back(ctx: RunContext[object], email_address: EmailStr) -> str:
        return f"Should I use {email_address} for your refund receipt?"

    @task.tool(needs_approval=read_back)
    async def confirm_email(
        ctx: RunContext[object], email_address: EmailStr
    ) -> Completed[EmailResult]:
        # EmailStr validates the address; the shared approval gate runs before
        # this body. Changing the address requires confirmation again.
        return Completed(EmailResult(email_address=email_address))

    return task


@front_desk.tool
async def change_receipt_email(ctx: RunContext[AirlineContext]) -> str:
    result = await collect_email().run()  # EmailResult; current session is implicit.
    # Completed restored front_desk. Its queued expert events become eligible
    # once this tool finishes, provided it still exposes the remote refund tool.
    await ctx.context.airline.set_refund_receipt_email(
        booking_reference=ctx.context.booking_reference,
        email=result.email_address,
    )
    return f"Refund receipt email updated to {result.email_address}."
```

The factory constructs a fresh task for each interaction. It registers tools
before `.run()` starts; no Agent or VoiceTask subclass is needed. The local task
uses the active session's frontend model and audio configuration. Its tools can
use the runtime wrapper without depending on the parent's application data;
this example types the unused application context as `object`.

`Completed[EmailResult]` ends this live task, not the remote refund task or the
whole call. The email's confirmation authorizes using that address; it does
not authorize a refund. The factory replaces the prebuilt `GetEmailTask` in
the preceding example and requires no expert-triggered collection behavior.

## Structured results in voice

A voice turn delivers conversational speech. `VoiceSession` owns frontend
remote-tool calls and their pending results; it presents completion after
resolving the originating tool call. `VoiceTask` returns its typed result to
the waiting frontend tool. Neither frontend path requires `TextSession`.

The remote backend may use plain `Agent.run()` or a backend
[TextSession](text-session.md) to coordinate its own specialists. The endpoint
returns its outcome through A2A; it does not select the frontend speaking agent.

See `examples/use-cases/01_appointments_by_phone_and_text.py`,
`03_insurance_eligibility.py` and `07_hospital_main_line.py`.
