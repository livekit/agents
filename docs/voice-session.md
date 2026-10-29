# VoiceSession

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
| `turn_handling` | a `TurnHandlingOptions`: `turn_detection`, `endpointing` (`EndpointingOptions`), `interruption` (`InterruptionOptions`) and `preemptive_generation` in one dict |
| `tools` | tools and toolsets every agent of the call is offered, besides its own |
| `context` | the application object; marked fields require explicit loading and saving |
| `thread` | the call's durable thread, `ctx.thread`; omitted, the history is kept in memory and not saved |

`session.start(agent, chat_ctx=)` and `session.update_agent(agent, chat_ctx=)`
are the two places to supply a different history. An explicit agent switch
retains the current history when no replacement is supplied.

In a tool on a call, `ctx.live_session` is the call (`say()`,
`generate_reply()`), and `ctx.thread` is the durable thread, the same
object a text handler gets. Outside a call `ctx.live_session` is `None`.

| Member | What |
|---|---|
| `start(agent, room=, chat_ctx=)` | attach the session to the room (the job's when left out) with `agent` speaking |
| `say(text)`, `generate_reply(instructions=, user_input=)` | speak fixed words, or reply through the model; each returns the speech's `SpeechHandle` |
| `interrupt()` | stop the current speech |
| `update_agent(agent)`, `update_options(...)`, `override_options(...)` | see [updating a live session](#updating-a-live-session) |
| `history` | the call's conversation, a `ChatContext`; empty on a new thread |
| `current_agent` | the agent speaking now |
| `on(event, callback)` | subscribe to a session event, below |
| `aclose()`, `async with` | end the call's session |

| Event | Payload | When |
|---|---|---|
| `"user_input_transcribed"` | `UserInputTranscribedEvent`: `transcript`, `is_final` | speech recognition produced a transcript |
| `"conversation_item_added"` | `ConversationItemAddedEvent`: `item` | a message or agent handoff joined `history` |
| `"function_tools_executed"` | `FunctionToolsExecutedEvent`: `function_calls`, `function_call_outputs` | a reply's tool calls settled |
| `"speech_created"` | `SpeechCreatedEvent`: `speech_handle`, `source`, `user_initiated` | the agent scheduled a speech |
| `"agent_state_changed"` | `AgentStateChangedEvent`: `old_state`, `new_state` | the agent moved between `initializing`, `idle`, `listening`, `thinking` and `speaking` |
| `"user_state_changed"` | `UserStateChangedEvent`: `old_state`, `new_state` | the person moved between `speaking`, `listening` and `away` |
| `"session_usage_updated"` | `SessionUsageUpdatedEvent`: `usage` | model usage grew |
| `"error"` | `ErrorEvent`: `error`, `source` | a provider failed; `error.recoverable` says whether the call goes on |
| `"close"` | `CloseEvent`: `reason`, `error` | the session closed |

The event types are exported from `livekit.agents`.

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
change should alter presentation. Multiple sessions using the same agent can have
different voices and listening policies.

### Temporary overrides

`update_options` writes the session's base settings. For a change that lasts
only as long as an operation, use an override layer:

```python
async with session.override_options(tts=specialist_voice):
    await session.say("Let me read the terms back to you.")
```

Each setting takes its value from the topmost override that sets it, else from
the base. Base changes made during an override persist and show through once the
override is removed. The layer belongs to the block that opened it and is
released when that block ends, on success, failure or cancellation. Transfer,
DTMF and `VoiceTask` interactions use the same mechanism; a task opens its
overrides inside its own code and they are released when the task ends.

## Custom I/O and recording

Audio, video and text endpoints may be implemented in Rust, Python or Node.
They use the same typed capability and stream contracts as provider plugins;
language adapters handle invocation, backpressure, errors and cleanup. An
`AudioOutput` reports playback progress and completion separately from frame
acceptance. The session calls `begin_segment(PlaybackSegment)` before capture.
A custom sink latches `playback_segment` when it accepts audio and supplies that
same key to `on_playback_started`, `on_playback_progressed` and
`on_playback_finished`. Delayed reports retain the captured key rather than
looking up the current segment when they complete. Wrapper outputs forward the
key to their downstream output.

VAD streams bound pending audio and output. Pushing audio beyond capacity or
an oversized frame raises an API error and retains no refused audio. Close
remains available at capacity and waits for bounded cleanup. See
[runtime limits](internals/rust-core.md#runtime-limits-and-backpressure).

The session owns input/output routing, transcript synchronization and recording.
Replacing its playback destination changes that destination only. Recording
continues on the same session timeline without requiring the application to
find and rebuild a recorder wrapper. Unrelated transforms and input attachments
are preserved.

Frames and playback reports carry attachment/segment identity. Reports from a
replaced output remain attributable to its old media and cannot advance the
new output. The session retains accounting for already delivered audio while
releasing obsolete endpoints. Recording failure and cleanup failure are
observable outcomes, not silent loss of recording. Recording overflow fails the
recording with a typed event, releases buffered audio and completes cleanup
within its deadline. The conversation continues.

Pre-connect audio is normalized to the session's format before entering the
same processing chain as live input. A processor that needs the source format
must declare that capability.

If an input processor such as noise cancellation fails, the session reports a
typed processing error, logs it and continues with unprocessed frames for that
span. The input stream does not end and the agent keeps hearing the user.

Playback completion waits are bounded by the delivered audio duration and a
transport allowance. A missing completion report cannot hold the speech or
session open indefinitely, and a timeout does not count unreported audio as heard.

See the [I/O composition contract](internals/rust-core.md#session-io-composition)
for ownership and the [shared primitives](internals/rust-core.md#common-runtime-primitives)
for cross-language stream semantics.

### Output handover policy

Output replacement takes effect at a segment boundary. The current playback
segment finishes on its original output; subsequent segments use the replacement
after the current segment settles. A segment is a unit of output playback, not
an inferred sentence boundary.

Replacement has one policy and does not interrupt speech. An application that
needs an immediate switch explicitly interrupts the current speech, waits for
the interruption to settle, then replaces the output. There is no separate
cut-and-switch replacement mode.

Existing work retains its original output binding until it finishes or is
interrupted. Pause, resume, stop and cleanup target that binding even after
another output is selected. Buffered audio is not silently replayed on the
replacement, and generated or accepted audio is not counted as heard audio.
Recording and transcript accounting retain the same session timeline across
the change.

## Answering machine detection

`AMD(session, ...)` classifies who answered an outbound call. Start it before
dialing so no audio is missed:

```python
from livekit.agents import AMD, AMDCategory

async with AMD(session) as detector:
    await ctx.api.sip.create_sip_participant(...)  # wait_until_answered=True
    result = await detector.execute()

if result.category == AMDCategory.MACHINE_VM:
    await session.say("Hi, this is the clinic calling about your appointment.")
```

`execute()` returns an `AMDPredictionEvent`: `category` (`human`,
`machine-ivr`, `machine-vm`, `machine-unavailable` or `uncertain`), `reason`,
`transcript`, `speech_duration` and `delay`, with `is_human` and `is_machine`.
Until the verdict the agent's speech is held. On a machine the current speech
is interrupted (`interrupt_on_machine=True`), and a `machine-ivr` verdict
starts IVR navigation (`ivr_detection=True`).

| Parameter | What |
| --- | --- |
| `llm`, `stt` | the classifier's models; by default LiveKit inference with Cloud credentials, else the session's. `None` reuses the session's |
| `interrupt_on_machine`, `ivr_detection` | what a machine verdict does, both on by default |
| `participant_identity` | listen to this participant only |
| `detection_options` | a `DetectionOptions`: the silence and speech thresholds, `timeout` and the classification `prompt` |
| `wait_until_finished` | once speech is heard, wait for the greeting to end rather than answer at `timeout` |

## Background audio

`BackgroundAudioPlayer` publishes a second audio track for ambient sound and a
"thinking" sound played while the agent is in the `thinking` state:

```python
from livekit.agents import AudioConfig, BackgroundAudioPlayer, BuiltinAudioClip

background = BackgroundAudioPlayer(
    ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.6),
    thinking_sound=[
        AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.8, probability=0.7),
        AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.7, probability=0.3),
    ],
)
await background.start(room=ctx.room, agent_session=session)

handle = background.play("assets/hold.ogg", loop=True)  # a PlayHandle
handle.stop()
```

A sound is a `BuiltinAudioClip`, a file path, or an async iterator of
`rtc.AudioFrame`, optionally wrapped in an `AudioConfig` with `volume`,
`probability`, `fade_in` and `fade_out`. From a list, one sound is picked by
probability; a total below 1 can pick silence. `play()` returns a `PlayHandle`
to `stop()`, check `done()` or await until it finishes. `aclose()` stops every
sound and unpublishes the track.

## The call's thread

On a server with storage, the door resolves the call's thread before the
entrypoint runs: the `thread_id` key of the dispatch's JSON metadata, as
the web console sends it, else a new thread. It is `ctx.thread` on the
`JobContext`, `None` on a server without storage, and
`VoiceSession(thread=ctx.thread)` is how the call gets it. A `VoiceSession`
without `thread=` keeps its history in memory, not saved. If a supplied dispatch
thread id is unknown or deleted, the call continues with an in-memory thread
and logs the refusal; it does not create a replacement durable thread.
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
    tools=[back_office.as_tool()],            # one tool, named from the card
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
[remote agents as tools](tools.md#remote-agents-as-tools) for routing and
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
[airline refund example](voice-tasks.md#example-airline-refund-and-receipt-email)
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


## Customizing the pipeline

An `Agent` subclass overrides the steps a reply runs through. Each node
receives a stream and returns a stream (an async iterable, or a coroutine
returning one); `super()` runs the default, so an override can wrap it. A
node's override can narrow its return type to what it actually returns:

```python
from collections.abc import AsyncIterable

from livekit.agents import Agent, ModelSettings, StopResponse, llm


class FrontDesk(Agent):
    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        if (new_message.text_content or "").strip().lower() == "hold on":
            raise StopResponse()  # no reply to this turn

    async def transcription_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[str]:
        async for chunk in text:
            yield chunk.replace("Live Kit", "LiveKit")
```

| Override | Runs | Receives, returns |
| --- | --- | --- |
| `on_user_turn_completed(turn_ctx, new_message)` | when the person's turn ends, before the reply | the turn's `ChatContext` and the new `ChatMessage`, both editable in place; `StopResponse` skips the reply |
| `llm_node(chat_ctx, tools, model_settings)` | each model request, on a call and in `agent.run()` | `ChatChunk`s or text; a `FlushSentinel` ends the spoken segment there |
| `stt_node(audio, model_settings)` | the person's audio | `rtc.AudioFrame`s in, `stt.SpeechEvent`s out |
| `tts_node(text, model_settings)` | each spoken segment | text in, `rtc.AudioFrame`s out |
| `transcription_node(text, model_settings)` | each segment's transcript before the transcription output | text (or `TimedString`s) in and out |
| `realtime_audio_output_node(audio, model_settings)` | a realtime model's audio before playout | `rtc.AudioFrame`s in and out |

`ModelSettings.tool_choice` is the reply's tool choice. Edits to `turn_ctx`
in `on_user_turn_completed` apply to this reply's request; the edited
`new_message` text is what the history records. A node that returns `None`
(a coroutine resolving to nothing) produces no output for that step.

## Tool calls on a call

On a call, a tool's `RunContext` also carries the speech it belongs to:

| Member | What |
| --- | --- |
| `ctx.speech_handle` | the `SpeechHandle` of the reply that made this tool call; raises in `agent.run()` |
| `await ctx.wait_for_playout()` | waits until the speech spoken before this tool ran has played out, not the whole turn |
| `async with ctx.with_filler(source, delay=0, interval=None, max_steps=None)` | while the block runs, speaks `source` (a string, or a callable taking the step count and returning text, a handle, or `None` to skip) once the session has been idle for `delay` seconds; with `interval` it repeats, up to `max_steps` |
| `async with ctx.foreground()` | waits for the session to be idle, then holds the floor for the block, so no other speech or deferred tool reply lands in between; yields the `VoiceSession` |
| `ctx.disallow_interruptions()` | makes the rest of this reply uninterruptible; raises `RuntimeError` if it was already interrupted |

## Temporary voice interactions

A [VoiceTask](voice-tasks.md) temporarily owns the floor, returns a typed result,
and restores its caller without running the parent's exit/enter hooks. Expert
events wait while the task collects input. See the
[airline refund example](voice-tasks.md#example-airline-refund-and-receipt-email)
for the exact inbox-processing boundaries and a task called from a frontend tool.

## Structured results in voice

A voice turn delivers conversational speech. `VoiceSession` owns frontend
remote-tool calls and their pending results; it presents completion after
resolving the originating tool call. `VoiceTask` returns its typed result to
the waiting frontend tool. Neither frontend path requires `TextSession`.

The remote backend may use plain `Agent.run()` or a backend
[TextSession](text-session.md) to coordinate its own specialists. The endpoint
returns its outcome through A2A; it does not select the frontend speaking agent.
