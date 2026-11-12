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
| `turn_handling` | a `TurnHandlingOptions`: `turn_detection`, `endpointing` (`EndpointingOptions`), `interruption` (`InterruptionOptions`), `preemptive_generation` (`PreemptiveGenerationOptions`) and `user_turn_limit` (`UserTurnLimitOptions`) in one dict |
| `conn_options` | a `SessionConnectOptions`: each leg's [connection policy](models.md#connection-options) (`stt_conn_options`, `llm_conn_options`, `tts_conn_options`) and `max_unrecoverable_errors` |
| `video_sampler` | a `VoiceActivityVideoSampler(speaking_fps=, silent_fps=)`: how often camera frames reach the model; `None` sends every frame |
| `stt_context_options` | an `STTContextOptions`: static `keyterms` the STT is biased toward, `keyterm_detection` (`KeytermDetectionOptions`) to add terms heard in the conversation, `forward_chat_context`. `session.keyterms` reads the current terms; `update_options(keyterms=)` replaces the static ones |
| `aec_warmup_duration` | seconds after the agent first speaks during which the person cannot interrupt, while echo cancellation adapts; 3 by default, none on an outbound SIP call the agent dialed |
| `expressive` | `True`, or an `ExpressiveOptions` (`speech_steering`: `SpeechSteeringOptions`, whose `nonverbal_sounds` takes `NonverbalOptions`; or a custom instructions template); see [expressive speech](lexer.md#expressive-speech) |
| `max_tool_steps` | model requests one reply may chain through tool calls; 3 by default |
| `min_consecutive_speech_delay` | seconds of silence between two of the agent's speeches; 0 by default |
| `user_away_timeout` | seconds of silence after which the person's state becomes `away`; 15 by default, `None` never |
| `transcription_timeout` | seconds to wait for a transcript after the person spoke before `user_transcription_timeout`; `None` (the default) never |
| `session_close_transcript_timeout` | seconds the close waits for the last transcript; 2 by default |
| `tools` | tools and toolsets every agent of the call is offered, besides its own; a name an agent's tool already has is refused (`ValueError`) when that agent runs, see [tools](tools.md#tools-on-an-agent) |
| `context` | the application object; marked fields require explicit loading and saving |
| `thread` | the call's durable thread, `ctx.thread`; omitted, the call runs in memory and no thread is created; one whose partition does not load fails `start()` ([the call's thread](#the-calls-thread)) |

`turn_detection` is a mode (`"stt"`, `"vad"`, `"realtime_llm"`, `"manual"`) or an
`inference.TurnDetector`, which predicts the end of the person's turn on their audio; any other
object is refused (`TypeError`). Left out, a session without a realtime model runs
`inference.TurnDetector()`. A detector object is set at construction;
`update_options(turn_handling={"turn_detection": ...})` on a running session takes a mode.

`session.start(agent, chat_ctx=)` and `session.update_agent(agent, chat_ctx=)`
are the two places to supply a different history. An explicit agent switch
retains the current history when no replacement is supplied. `start(agent)`
picks the agent the session starts with. Called before `start()`,
`update_agent(agent, chat_ctx=)` only sets the history the session opens on;
`start(chat_ctx=)` takes precedence over it.

In a tool on a call, `ctx.live_session` is the call (`say()`,
`generate_reply()`), and `ctx.thread` is the durable thread, the same
object a text handler gets. Outside a call `ctx.live_session` is `None`.

| Member | What |
|---|---|
| `start(agent, room=, room_options=, chat_ctx=)` | attach the session to the room (the job's when left out) with `agent` speaking; `room_options` is a `RoomOptions` of `audio_input` (`AudioInputOptions`), `audio_output` (`AudioOutputOptions`), `video_input` (`VideoInputOptions`), `text_input` (`TextInputOptions`, whose callback receives a `TextInputEvent`) and `transcription_output` (`TranscriptionOutputOptions`, the agent's words), each `True`/`False` or its options |
| `say(text, audio=, allow_interruptions=, add_to_chat_ctx=True)`, `generate_reply(instructions=, user_input=, tools=, tool_choice=, allow_interruptions=, chat_ctx=)` | speak fixed words, or reply through the model; each returns the speech's `SpeechHandle`. `allow_interruptions` overrides the session's for this speech; `add_to_chat_ctx=False` keeps a `say` out of the history; `tool_choice` and `chat_ctx` (the context this reply works on, refused on a realtime model) apply to this reply only. `text` is a `str` or an async iterable of `str` read to its end; with `audio` (pre-synthesized frames, then `text` must be a `str`, else `ValueError`) the frames play instead of the TTS and `text` is still added to the history. `tools=` (names) offers only those of the driving agent's tools for that reply, plus its provider-owned tools; a realtime model refuses it. A reply asked from the agent's `on_enter` is that hook's reply: tools flagged `ToolFlag.IGNORE_ON_ENTER` are withheld from it |
| `interrupt(force=False)` | stop the current speech and the queued ones up to the first uninterruptible one; returns a future that resolves when they are done, or with the refusal. A reply interrupted while its tools run is done at once and the person's next turn goes ahead: the tools finish in the background, their results join the history with no follow-up reply, and any still running 5 seconds after the interruption are cancelled |
| `await set_participant(identity)` | listen to another room participant (their audio, transcript and keypad); `None` listens to no one. By default the session links the first participant who joins, or `RoomOptions(participant_identity=)` |
| `update_agent(agent)`, `update_options(...)`, `override_options(...)` | see [updating a live session](#updating-a-live-session) |
| `history` | the call's conversation, a new `ChatContext` over the session's items on each read; empty on a new thread and before `start()`; see [reading the session from a callback](#reading-the-session-from-a-callback) |
| `current_agent` | the agent speaking now |
| `stt`, `tts`, `vad` | the providers the session runs on now: a held `override_options` layer's, else the base's (the constructor's, then `update_options`'); `None` when removed |
| `agent_state`, `user_state` | the agent's and the person's state, `"initializing"` and `"listening"` until their first change; read as `history` is |
| `usage` | latest immutable `Usage` snapshot; retained after shutdown; see [usage and metrics](usage.md) |
| `on(event, callback)` | subscribe to a session event, below |
| `drain()` | run the driving agent's `on_exit`, then wait until no speech is left, scheduling nothing new |
| `wait_for_idle()` | wait until the session is idle: nothing speaking, queued or deciding a turn |
| `shutdown(drain=True)` | close without waiting; `drain` lets queued speech finish first |
| `aclose()`, `async with` | end the call's session and wait for it |
| `options` | the resolved configuration the session runs on (a `ResolvedSessionOptions`), current as soon as a change is sent |
| `llm`, `tools`, `conn_options`, `context` | the constructor's model, session tools, connection policy and application context |
| `thread` | the call's thread, `None` for a call in memory |
| `store` | the call's `ContextStore` for explicit `load_context` and `save_context`; raises before `start()` |
| `amd` | the session's `AMD`, `None` when answering-machine detection is off |
| `input`, `output` | the session's media endpoints; assigning `session.output.audio` replaces the audio sink the agent speaks into |

| Event | Payload | When |
|---|---|---|
| `"user_input_transcribed"` | `UserInputTranscribedEvent`: `transcript`, `is_final` | speech recognition produced a transcript |
| `"conversation_item_added"` | `ConversationItemAddedEvent`: `item` | a message or agent handoff joined `history` |
| `"function_tools_executed"` | `FunctionToolsExecutedEvent`: `function_calls`, `function_call_outputs` | a reply's tool calls settled |
| `"speech_created"` | `SpeechCreatedEvent`: `speech_handle`, `source`, `user_initiated` | the agent scheduled a speech |
| `"agent_state_changed"` | `AgentStateChangedEvent`: `old_state`, `new_state` | the agent moved between `initializing`, `idle`, `listening`, `thinking` and `speaking` |
| `"user_state_changed"` | `UserStateChangedEvent`: `old_state`, `new_state` | the person moved between `speaking`, `listening` and `away` |
| `"error"` | `ErrorEvent`: `error`, `source` | a provider, an input audio processor, the avatar or the audio output failed, or the audio output did not report a cleared segment; `error.recoverable` says whether the call goes on |
| `"user_turn_exceeded"` | `UserTurnExceededEvent`: `transcript`, `accumulated_transcript`, `accumulated_word_count`, `duration` (a `timedelta`) | the person spoke past the turn limit without a reply |
| `"user_transcription_timeout"` | `UserTranscriptionTimeoutEvent`: `speech_duration` | the person spoke but recognition produced no transcript |
| `"agent_false_interruption"` | `AgentFalseInterruptionEvent`: `resumed` | an interruption turned out not to be one |
| `"overlapping_speech"` | `OverlappingSpeechEvent`: `is_interruption`, `agent_ended`, `overlap_started_at`, `detected_at` | adaptive interruption detection classified overlapping speech |
| `"tool_execution_updated"` | `ToolExecutionUpdatedEvent`: `update`, a `ToolCallStarted`, `ToolCallUpdated`, `ToolCallEnded` or `ToolReplyUpdated` | one step of a tool call's lifecycle on the call |
| `"close"` | `CloseEvent`: `reason` (a `CloseReason`), `error` | the session closed |

Every event type is exported from `livekit.agents.voice.events`, the common ones also from
`livekit.agents`, with `AgentEvent` (their union), `EventTypes` (the event names) and the
`AgentState` and `UserState` literals.

### Reading the session from a callback

The session's history and states are held by the native session and read,
never copied, by `session.history`, `session.agent_state` and
`session.user_state`. Each event reaches its callbacks with the session state it
belongs to. Inside an event callback, and in any task the callback starts, they
read the session as of that event: the history ends with the items that event
delivered and omits items added after it, and an `"agent_state_changed"`
callback reads its own `new_state`, even when the callback reads them later.
Anywhere else they are the latest the session's events have delivered. The items are the session's own objects: the item of a
`"conversation_item_added"` event and every later `session.history` are the same Python
objects. Each read returns a new
`ChatContext`, so editing it does not change the session; use
`update_agent(agent, chat_ctx=)` or `Agent.update_chat_ctx()` to supply a different history.

### Speech handles

`say()` and `generate_reply()` return the speech's `SpeechHandle`; the same
object is handed back wherever the speech is named (`"speech_created"`,
`ctx.speech_handle`). A speech ends twice, in this order. First the native
session finishes it: `done()` turns true, and `await handle` and
`wait_for_playout()` return; awaiting never raises, and `exception()` reads a
failure. Then the session hands the host the speech's end, after every session
event before it: only then do the callbacks given to `add_done_callback(cb)` run,
each as `cb(handle)` on the event loop, in the order added. A callback added
after that runs soon; one that raises is logged and the others still run;
`remove_done_callback(cb)` forgets registrations that have not run. A speech the
session ended before creating is reported at once, and closing the session
reports every speech it still holds, so no callback waits past the session.

Usage is read through `session.usage`, without an update event. Intermediate
snapshots contain measurements received so far; after awaiting `aclose()`, the
snapshot describes settled local accounting and any missing measurements. See
[voice usage](usage.md#voice-sessions).

## Updating a live session

Configure audio on the session and update it explicitly during the call:

```python
session = VoiceSession(
    stt=recognizer,
    tts=front_desk_voice,
    vad=voice_activity_detector,
    turn_handling={"turn_detection": "vad"},
    context=app,
    thread=ctx.thread,
)
await session.start(agent=front_desk, room=ctx.room)

# Change presentation and listening policy when the application needs it.
session.update_options(
    tts=specialist_voice,
    turn_handling={
        "turn_detection": "stt",
        "endpointing": {"min_delay": 0.3, "max_delay": 2.0},
    },
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

A replaced provider is installed in place under the current agent's nodes; only
the providers whose effective value changed are rebound. Before `start()` the
session opens on the updated settings and providers. `turn_handling` takes the
same keys as the constructor's and merges by key, nested groups included:
`{"endpointing": {"max_delay": 4.0}}` changes that one setting and keeps the
others. `turn_detection: None` returns to automatic selection. Its `interruption`
takes no `mode`, not even `None`: the interruption mode is set with
`VoiceSession(turn_handling=)` and refused with a `ValueError` anywhere else. An
update refused for its arguments changes nothing.

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
the base; each key inside a `turn_handling` group is a setting on its own. Overrides sit above the active agent's settings, including an agent or
`VoiceTask` that takes over inside the block. Base changes made during an
override persist: they show through at once on the settings no override sets,
and on the others once the override is removed. The layer belongs to the block that opened it and is
released when that block ends, on success, failure or cancellation. When the
session closed inside the block the release has nothing to restore: it ends
quietly, or, when the block raised, adds a note to that error. Transfer,
DTMF and `VoiceTask` interactions use the same mechanism; a task opens its
overrides inside its own code and they are released when the task ends.

## User turns and push-to-talk

A user turn can be opened and closed from code, whatever the turn detection: a
game key, a DTMF key, or the application's own endpointing.

```python
turn = session.start_user_turn()   # interrupts the agent; only end() or cancel() closes it
...
result = await turn.end()          # every final transcript arrives, then the message is added
if not result.empty:
    print(result.message.text_content)   # already in session.history; the reply follows

turn = session.start_user_turn()
turn.cancel()                      # nothing the turn heard is transcribed or kept
```

`end(reply=False)` adds the message and asks for no reply. A release keeps
listening for about 200 ms of audio, since the release can arrive before the
last words, then the recogniser finalizes what it heard. `end()` resolves once
the message is in the history, before the reply, and at most about 5 s after the
release: a recogniser that has not finished by then commits what arrived. A turn
with nothing said adds no message and no reply (`result.empty`).

One turn is open at a time: another `start_user_turn()` raises `RuntimeError`
until the open one's `end()` resolved or it was cancelled. A handoff keeps the
turn, and the agent active at `end()` answers it. Reaching `user_turn_limit` ends
an open turn as a release would. The session's close cancels an open turn, and a
pending `end()` raises. `user_turn_started` and `user_turn_ended` (`cancelled`,
`empty`) are emitted for each turn; once the session ended a turn itself (the turn limit,
`push_to_talk` turned off), that turn's `end()` raises `RuntimeError`.

`turn_handling={"push_to_talk": True}` makes these the only turns: outside a turn
the microphone reaches no recogniser, no voice activity detection and no
interruption; the last ~200 ms before a press is kept and heard by the turn.
Turn detection, endpointing, interruption and preemptive generation keep their
settings and apply again when `push_to_talk` is turned off; turning it off ends
an open turn as a release would.

```python
session = VoiceSession(stt=recognizer, turn_handling={"push_to_talk": True})
session.update_options(turn_handling={"push_to_talk": False})   # hands-free, live
async with session.override_options(turn_handling={"push_to_talk": True}):
    await dictation_step()
```

Applications connect their buttons, RPCs or other input controls to `start_user_turn()`,
`end()` and `cancel()`. The framework currently provides the session API; it does not register
`lk.user_turn.start/end/cancel` room RPCs or publish an `lk.agent.input_mode` attribute.

## A job's sessions

A session started on the job's room (the default room inside a job), or on
`lk agent console`'s call, is the job's. Its start runs in this order: the
thread's storage opens, the room and models are prepared, the session claims
the job, the job's recording is set from the claim, the start time is taken,
the audio recording opens, and the job's shutdown is made to close the session.
A failure at any step fails `start()` and settles what was acquired.

The first session claimed is the job's primary one: it records what the job
records (`enable_recording`), serves the remote session protocol, is the one
the job reports, and ends the job when it closes. A later session of the job
records nothing, serves no remote session and does not end the job; the
console's call serves the remote session either way, and `--record` records
its audio. The audio recording is `audio.ogg` in the job's session directory,
the report's `audio_recording_path`. Every session of the job closes as the
job shuts down, with `CloseReason.JOB_SHUTDOWN`, in the shutdown's span.

A session's report belongs to the job the session first started in; another
job's `make_session_report(session)` raises `PermissionError`. A session on
another room, or with `room=None`, still belongs to that job's report but
neither claims, records nor closes with it.

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

A synchronized transcript shows each word when the voice starts saying it. A
voice that reports word timings (Cartesia, Inworld and ElevenLabs among them)
places its words; otherwise each word is placed in the audio received so far:
the silences in the audio are matched to the text's breaks (sentence ends,
commas, list items) and the speech between two of them is shared by the words
by their expected length. A sentence end is likelier to get a pause than a
comma, and a pause where the text has no break is taken for a hesitation. The
estimate is usually within about 0.1 s, but a number or an ID a voice reads
unexpectedly can put the words after it seconds off until the next pause, so a
voice's own timings are always preferred. Behind `StreamAdapter`, each chunk's
audio holds its own words. The transcript follows the playback reports. From
`on_playback_started`, words advance with elapsed time, never past the audio
sent to the output, so a sink that reports only its start and finish still
shows words as they are spoken. `on_playback_progressed`, when a sink reports
it, corrects that estimate. Words stop while the output is paused and continue
on resume. An interrupted segment keeps the words heard up to the position its
`on_playback_finished` reports. The transcription output is flushed once per
segment that had text, never on attach or close alone.

VAD streams bound pending audio and output. Pushing audio beyond capacity or
an oversized frame raises an API error and retains no refused audio. Close
remains available at capacity and waits for bounded cleanup. See
[runtime limits](internals/runtime.md#runtime-limits-and-backpressure).

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
recording with a `recording capacity exceeded` error, logged when the session
ends, releases buffered audio and completes cleanup within its deadline. The conversation continues.

Pre-connect audio is normalized to the session's format before entering the
same processing chain as live input. A processor that needs the source format
must declare that capability.

If an input processor such as noise cancellation fails, the session emits an
`"error"` event whose `error` is a recoverable `room_io.AudioProcessingError` and whose
`source` is the processor, logs it and continues with frames that processor left
unprocessed for that span. Each processor's failing span is one error, ending at the
first frame it processes cleanly; a processor that fails again reports again. The
input stream does not end and the agent keeps hearing the user.

If the audio output raises from `capture_frame`, the speech it was playing
ends there, interrupted, and its handle's `exception()` is the output's
exception. The session emits an `"error"` event whose `error` is a recoverable
`io.AudioOutputError` and whose `source` is the output, and logs it. The
history keeps what was heard, as for any interruption. A failed or cancelled
capture holds nothing the output must report, so the output, and every link of
its chain, stays usable for the next speech.

Playback completion waits are bounded by the delivered audio duration and a
transport allowance. A missing completion report cannot hold the speech or
session open indefinitely, and a timeout does not count unreported audio as heard.

After `clear_buffer`, the output keeps the cut segment until its sink reports
`on_playback_finished` for it, since the old audio may still be playing. A sink
that has not reported within 5 seconds of the clear (a custom output that never
calls `on_playback_finished` after `clear_buffer`, or a remote avatar worker
that died) is treated as cut where it last reported playback (the start when it
reported none): the segment settles there, the recording keeps only what was
reported played, the synchronized transcript stops, and the output goes to the
next speech. The session emits one `"error"` event
whose `error` is a recoverable `io.AudioOutputError` and whose `source` is the
output, saying the sink did not report playback finished within 5 s of being
cleared and that the session may have been unable to speak meanwhile, and logs
it. A report that arrives later is ignored. A sink that reports within the 5
seconds is unaffected.

See the [I/O composition contract](internals/runtime.md#session-io-composition)
for ownership and the [shared primitives](internals/runtime.md#common-runtime-primitives)
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
`transcript`, `speech_duration` and `delay` (`timedelta`s), with `is_human` and `is_machine`.
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

## Avatars

An `AvatarSession` (a provider plugin, or `inference.AvatarSession`) replaces
the call's audio with a stream to a remote avatar worker that publishes the
video on the agent's behalf:

```python
avatar = tavus.AvatarSession(...)
await avatar.start(session, room=ctx.room)
await avatar.wait_for_join()  # raises TimeoutError after 30 s by default
await session.start(agent, room=ctx.room)
```

`start()` begins the avatar's lifecycle in the session's runtime:

- It waits in the background for the avatar participant (`avatar_identity`)
  to join the room and publish its video, once the room is connected.
  `wait_for_join(timeout=)` awaits it and raises `TimeoutError` after
  `timeout` seconds (`None` waits indefinitely); a timeout does not end the
  wait, so a later call can still see the join. It raises
  `asyncio.CancelledError` once the avatar closed before it joined, and the
  room's error if the room failed first.
- If the avatar has not joined 30 s after `start()`, the session emits one
  `"error"` event whose `error` is a recoverable `avatar.AvatarError` (its
  `source` is `None`). From then on the avatar's audio output stops waiting:
  each speech fails at once with that error while the avatar is absent. The
  join wait goes on, and once the avatar joins, speech plays normally. Before
  the 30 s, speech waits for the avatar to join.
- Inside a job, the job's shutdown runs the avatar's `aclose()` (a plugin's
  own), in the shutdown's span.
- `metrics_collected` emits `AvatarMetrics`: once when the avatar joined
  (`session_started_time` when the room was connected, `avatar_joined_time`
  when its video was published), then for each assistant message the session
  adds with a `playback_latency` (from the first audio frame sent to the
  avatar to the worker's `lk.playback_started`), until the close. Each is also
  recorded with the session's metrics once the session runs.

`aclose()` closes the avatar's audio output, stops the join wait and the
metrics, then, inside a job while the room is connected, removes the avatar
participant through RoomService `RemoveParticipant` with the job's
`LIVEKIT_URL` and key pair. A participant already gone (`not_found`) is
skipped; any other failure is logged and does not fail the close. Concurrent
closes wait for the same close, a later one does nothing, and `start()` again
begins a new lifecycle.

`DataStreamAudioOutput` is native, and it has the `io.AudioOutput` surface a
stage forwards to: `capture_frame`, `flush`, `clear_buffer`, `begin_segment`,
`playback_segment`, `wait_for_playout`, `can_pause` (`False`), `pause`,
`resume`, `on_attached` and `on_detached`. So a Python stage can wrap it
(`MyStage(next_in_chain=output)`) as it wraps a Python sink: the stage
forwards frames, flushes and clears to the output, and the output reports its
playback to the stage's `on_playback_started`, `on_playback_progressed` and
`on_playback_finished`, in order on the stage's loop, until the output
closes. The output emits no events of its own; listen on the stage.

An audio output chain reports playback upward. Building a stage with
`next_in_chain=` makes it that output's parent. When an output reports
playback (`on_playback_*`), it emits its own `"playback_started"`,
`"playback_progressed"` or `"playback_finished"` event to its listeners, then
reports the same to its parent, up to the head of the chain, so a listener on
any Python stage hears that stage's events. An output belongs to one chain:
building a second stage over it raises `ValueError`.

On the avatar worker, an `AvatarRunner` feeds an `AudioReceiver`'s audio to a
`VideoGenerator` and publishes what it generates. `DataStreamAudioReceiver`
receives the agent's stream; a custom receiver iterates frames and an
`AudioSegmentEnd` at each segment's end. The runner keys each segment from
its first frame to its end, or to the receiver's `"clear_buffer"`, which cuts
it, and reports its playback to the receiver with that key.

## The call's thread

The door resolves the job's `thread_id` in the server's storage before the entrypoint runs
([how a call gets its thread](threads.md#calls)). `ctx.thread` names that thread; a dispatch
without a thread runs in memory. An unknown or deleted dispatch thread is logged as refused.

Bind the job's store to enable automatic session restoration and incremental checkpoints:

```python
app = App()
await ctx.store.load_context(app)
binding = ctx.store.bind(app)
session = VoiceSession(context=binding, llm=model)
await session.start(agent, room=ctx.room)
```

With no application context, use `context=ctx.store.bind()`. Passing a plain application
context or a `thread=` association alone does not enable automatic session persistence.
See [session binding](context.md#session-state-and-explicit-binding) for the live view and detached snapshot APIs.

A bound session restores committed history at startup unless an explicit `chat_ctx` is supplied.
Each voice turn takes the partition lease, checkpoints accepted conversation progress, commits
explicitly saved application fields on success, and releases ownership after settlement.
A text request using the same partition waits for that ownership. Before a later voice turn,
transport history is reconciled with committed changes from other requests. Application fields
still require explicit load/save.

When the person interrupts a reply while its tools run, the conversation is freed at once, as on
a call without a thread: the reply's speech is done and the next turn is answered. Only
persistence waits. The interrupted turn keeps its hold on the partition until its tools end, or
the 5 second interruption bound cancels them, and its turn is committed or abandoned. Turns
commit in order: a later turn's commit waits for the earlier one, so the thread's history keeps
the order the turns were taken in. The interrupted turn's late tool output is placed right after
its call, in `session.history` and in the thread, before what was said since; the thread holds
back the later turn's items until it lands. The thread saves the call's items in the order the
session holds them, its opening handoff first.

The history matches what the person heard: what was said is kept in the order it was heard,
with only the words heard. An item the person never heard, a tool call, its output or a
handoff, sits with the turn it belongs to. A tool that awaited a `VoiceTask` records its output
after the task's conversation, since the task's answers produced it.

A failure to acquire, restore or persist bound state closes the session with a storage error;
it never silently switches to in-memory execution. `start()` failures propagate to the entrypoint:

```python
from livekit.agents.storage import StorageError

try:
    await session.start(agent)
except StorageError:
    ...  # end this call or explicitly create a new session
```

An uncaught error ends the job as any entrypoint exception does. For phone-number routing and
conversation continuity, see [channel routing](channels.md#channel-routing).

## The thin voice door

The realtime model is good at the floor: small talk, interruptions, pacing.
It is not where the tools and the state should live. An `RemoteAgent(...).as_tool()` in
the voice agent's `tools` gives it one tool, the brain behind the text
endpoint, named and described from that endpoint's card, and speaks the
brain's progress while it waits.

```python
back_office = RemoteAgent("http://127.0.0.1:7880/agents/dentist/_/backend")
voice = Agent(
    instructions="You are the practice's voice on the phone. Hand every request to the back office.",
    tools=[back_office.as_tool()],            # one tool, named from the card
)

@server.rtc_session()
async def phone(ctx: JobContext) -> None:
    binding = ctx.store.bind()
    session = VoiceSession(llm=openai.realtime.RealtimeModel(), context=binding)
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

Doors selecting the same partition share transport history and explicitly saved application
data. Fast and slow brains normally select different partition keys so each keeps its own
conversation; see [partitions](threads.md#partitions-and-session-frames).

## Expert events and frontend presentation

`VoiceSession` owns frontend remote-tool interactions: it retains pending
expert references, routes human replies, and schedules presentation. The
frontend attaches `RemoteAgent(...).as_tool()` through its agent's `tools=`
and starts the live session. It does not construct a `TextSession` or pass
backend `TextSessionState` to the session. A local `agent.as_tool()` on a call
is an expert the same way: its child runs in process, and its progress,
questions and result take this path ([local agents as tools](tools.md#local-agents-as-tools)).

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
| Backend request inbox | One transport partition | Incoming requests buffered by endpoint policies such as batching | Managed request scheduling and storage |

The frontend event queue is not persisted or restored across restarts.
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
4. Useful progress may be combined; presenting progress or a question runs a
   presentation-only fast-brain model reply to phrase it. Enqueueing alone does
   not invoke the LLM.
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
| Progress | Retain correlation and ordering; coalesce superseded progress | Present useful updates when the conversational floor permits, each through a presentation-only model reply with tools withheld |
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
model to select the delegation tool again. The current forwarding path does not classify
cancellations, new requests or ambiguous answers; it forwards the supplied reply. Applications
that need that distinction must handle it before reply delivery. Only one expert question owns
the foreground reply destination at a time.

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

There is currently no public application API to settle or cancel a stranded frontend
interaction. Applications must account for that limit when permanently removing expert access.
The session does not automatically
switch agents, start another remote request, or deliver the result through an
unrelated agent. Cancellation is not proof that a refund or other remote effect
did not happen; the authoritative service must reconcile uncertain outcomes.

The default queue and its active reply correlation live in process. They can
end with the frontend session. They are not backend routing frames or
automatically saved application fields. Saving backend history or application
fields does not make frontend event delivery recoverable.

Durable frontend event delivery requires an application-owned transport and recovery design.
The framework does not currently provide automatic queue persistence or replay. Such a design
must retain event positions and delegation/reply identities and reconcile uncertain remote
outcomes; serializing a queue or Python await is not sufficient.

Another frontend in the same thread cannot consume these events simply because
it shares a database. Duplicate event delivery must not resolve a tool call
twice or present the same retained occurrence twice during normal processing.
A crash between delivery and acknowledgement can still cause uncertainty;
application-owned persistence does not promise exactly-once speech or external effects.

The frontend queue coordinates expert events and presentation. It does not add
background execution, a scheduler, or Python stack recovery. The backend request
inbox continues to serve endpoint request policies independently.


## Customizing the pipeline

The steps a reply runs through belong to the session, not to an agent. Register
at most one callback per step on the session before `start()`; it runs for
whichever agent holds the call, `ctx.agent`, and for every `VoiceTask`.
`session.default.<step>` is the built-in step, so a callback can wrap it.
`agent.run()` and `TextSession` run none of these.

```python
from collections.abc import AsyncIterable

from livekit import rtc
from livekit.agents import ModelSettings, RunContext, StopResponse, VoiceSession, llm

session = VoiceSession(stt=STT, llm=MODEL, tts=TTS, context=app, thread=ctx.thread)


@session.on_user_turn_completed
async def hold_on(
    ctx: RunContext[AppContext], turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
) -> None:
    if (new_message.text_content or "").strip().lower() == "hold on":
        raise StopResponse()  # no reply to this turn


@session.tts_node
def spell_amounts(
    ctx: RunContext[AppContext], text: AsyncIterable[str], model_settings: ModelSettings
) -> AsyncIterable[rtc.AudioFrame]:
    if ctx.agent is billing:
        text = spell_out_amounts(text)
    return session.default.tts_node(ctx, text, model_settings)


await session.start(agent=front_desk, room=ctx.room)
```

| Callback | Runs | Receives after `ctx`, returns |
| --- | --- | --- |
| `on_user_turn_completed` | when the person's turn ends, before the reply | the turn's `ChatContext` and the new `ChatMessage`, both editable in place; `StopResponse` skips the reply |
| `on_user_turn_exceeded` | when the person has spoken past the turn limit without a reply | the `UserTurnExceededEvent`; by default the agent cuts in with a short reply |
| `llm_node` | each model request of a reply | `chat_ctx`, `tools`, `model_settings`; `ChatChunk`s or text, a `FlushSentinel` ending the spoken segment there |
| `stt_node` | the person's audio | `rtc.AudioFrame`s and `model_settings`; `stt.SpeechEvent`s |
| `tts_node` | each spoken segment | text and `model_settings`; `rtc.AudioFrame`s |
| `transcription_node` | each segment's transcript before the transcription output | `TranscriptChunk` records and `model_settings`; records or plain text |

A node returns an async iterable or a coroutine resolving to one; `None`
produces no output for that step. `ModelSettings.tool_choice` is the reply's
tool choice. Edits to `turn_ctx` in `on_user_turn_completed` apply to this
reply's request; the edited `new_message` text is what the history records.
Registering a step twice, or after `start()`, raises `RuntimeError`.

Transcript chunks carry `text`, optional `start_time`/`end_time`, `confidence`,
`start_time_offset`, `speaker_id`, `spoken` (whether the voice says it) and `expression` (the
delivery asked for from it on); see [the transcript](lexer.md#the-transcript). They are generated records, rather than string subclasses;
read `.text` for string operations. A `transcription_node` receives records even when timing is
absent and may return records or plain strings. Plain strings are normalized into records with
no metadata at that boundary. Return an existing record unchanged to preserve all its metadata.
Missing metadata is `None`; malformed metadata raises at the binding boundary.

Times remain seconds on the owning audio stream's timeline. Native stream adapters apply source
offsets when normalizing provider events; the recorded `start_time_offset` is not an additional
shift to apply downstream. This record representation does not change existing timeline values.
A Python `TranscriptionOutput` (`livekit.agents.voice.io`) can override
`capture_transcript(chunk)` to consume metadata; its default implementation forwards `chunk.text`
to `capture_text`.

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


## Session persistence

Use the entrypoint's store to opt into conversation persistence:

```python
await ctx.store.load_context(app)
binding = ctx.store.bind(app)
voice = VoiceSession(context=binding, llm=model)
await voice.start(agent=front_desk, room=ctx.room)
```

JobContext.store is available before startup. VoiceSession and its tools reuse that activation;
the declaration does not create another store or lease. Input/model/tool work starts only
after binding validation. The entrypoint's partition_key selects the stored conversation.

A marked SessionState field on app is optional. When present, it retains identity and follows
accepted runtime changes; otherwise the session owns its state internally. Plain context
does not opt into managed session persistence. Runtime services remain application-supplied.

Conversation checkpoints retain turn ownership and do not commit staged application saves.
An explicit store.flush is a conversation durability barrier. Detached load_session snapshots
remain unchanged; they can lag accepted in-memory state. Managed storage failure ends the
binding and surfaces an error rather than silently claiming durability in memory.

Voice snapshots recover configured agent identity and conversation references. They do not
restore the in-process expert-event queue or resume audio streams, Python coroutines or live
VoiceTask parent stacks. A stored agent unavailable in the new configuration is an error.
See [context binding](context.md#session-state-and-explicit-binding).
