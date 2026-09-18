# Multi-turn AMD

AMD classifies the call participant at client-side end of turn (EOT). It can
handle several screening, voicemail, and IVR turns before it completes.
The SDK owns classification, stage changes, turn hooks, and deadlines.
AgentSession owns the customer hook, interruption, and playback.

AMD consumes `user_state_changed` and `user_input_transcribed` session events.
At each accepted EOT, AgentSession notifies AMD internally before
`on_user_turn_completed` runs. AMD owns its turn IDs and returns turn hooks
bound to that turn. Preemptive generation prepares reply tools but only commits
an agent turn if its reply is accepted for output.
While AMD is active, AgentSession commits every turn. It skips the
`min_words` interruption filter, so a short transcript during agent speech
still reaches AMD. False-interruption pause and resume stay unchanged.
State events carry the accepted speech-boundary time,
including the STT timestamp when STT controls turn detection.

This replaces the one-shot AMD API. `execute()` now returns `AMDCompletedEvent`,
not the first `AMDPredictionEvent`. Use `amd_prediction` to observe each prediction.
The application still decides whether to continue or end the call.

## Start AMD

Start AgentSession first. Enter AMD before creating the SIP participant.
Do not generate an `on_enter` greeting before AMD starts.
Let AgentSession generate replies during AMD. Direct calls to `session.say()`
or `session.generate_reply()` during detection are not supported.

```python
import logging

from livekit.agents import AMD

logger = logging.getLogger("amd-example")

detector = AMD(
    session,
    participant_identity=callee_identity,
    # llm=None,  # Use the active Agent's LLM instead of auto-selection.
    # stt=None,  # Use only the session transcript instead of auto-selection.
)

@detector.on("amd_prediction")
def on_prediction(event):
    logger.info(
        "AMD prediction: turn_id=%s category=%s reason=%s",
        event.turn_id, event.category, event.reason,
    )

@detector.on("amd_menu_observed")
def on_menu(event):
    record_menu(event)  # Informational. This event does not authorize an action.

async with detector:
    # Create the SIP participant here.
    result = await detector.execute()
    # Choose the next call action from result.category and result.reason.
```

By default, AMD and the Agent discard SIP pre-answer audio. Set
`wait_until_answered=False` to use subscribed early media. This does not cause
the SIP provider to supply early media.

`detector.lifecycle` uses `AMDLifecycle`: `INITIALIZED` before context entry, `PENDING`
while awaiting the participant, `ACTIVE` during detection, and `FINISHED`
after detection ends. Import `AMDLifecycle` from `livekit.agents`.

AMD allows five seconds for the participant's audio track to be published and
subscribed. If that fails, it completes with `participant_missing` and releases
the session. A participant or room disconnect during setup completes with
`participant_disconnected`. Set the SIP answer timeout when placing the call.
The detection `timeout` starts when listening begins.

## Model selection and transcript race

- `llm` accepts `str | LLM | None | NotGiven`. If omitted, AMD auto-selects
  `google/gemini-3.1-flash-lite`. Pass `None` to use the active Agent's pipeline
  LLM. A model instance or string selects a different classification and menu model.
- `stt` accepts `str | STT | None | NotGiven`. If omitted, AMD auto-selects
  `cartesia/ink-whisper`. Pass `None` to use only the session transcript.
  A model instance or string selects an STT for AMD only.
- Model strings use LiveKit Inference. Provider plugins can use the caller's
  own provider credentials.
- AMD closes models it creates from strings. It does not close supplied model
  instances. It closes its requests and STT streams in either case.

Auto-selection requires a LiveKit Cloud `LIVEKIT_URL` and an API key and secret.
AMD checks `LIVEKIT_INFERENCE_API_KEY` and `LIVEKIT_INFERENCE_API_SECRET`, with
`LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` as fallbacks. Without these settings,
omitted values inherit the Agent's LLM and session transcript. Explicit `None`
always inherits. AMD resolves `llm` and `stt` independently.

The LLM must support required function calls. AMD uses a `record_result` tool
for each prediction or menu result. It validates the tool arguments against the
result schema. This tool is not added to the active Agent.

When session STT is configured, the first non-empty final transcript selects the
source for the whole AMD run. Without session STT, AMD uses its own STT directly.
Empty and interim results cannot win. AMD keeps using the selected source across
turns, so a change in relative STT latency cannot discard trailing transcript segments.
If session STT wins, AMD closes its optional STT stream and stops sending audio to it.
The Agent's transcript, history, hooks, and EOT stay unchanged. A faster AMD
transcript does not make the Agent's EOT arrive earlier.

At EOT, AMD commits the selected source's available transcript immediately, even if
it is empty. Finals received after EOT accumulate for the next turn. The AMD STT
must support streaming. It uses a single stream without flushing at EOT,
matching session STT. AMD receives the unsuppressed participant audio during AEC
warmup, while session STT receives silence. AgentSession disables AEC warmup by
default for outbound SIP calls; an explicit `aec_warmup_duration` takes precedence.
The classifier prompt treats the transcript as untrusted evidence, so echoed
agent speech cannot instruct it.

If the AMD stream fails, the open turn uses its buffered session transcript, even if
it is empty. Subsequent turns also use the session transcript. The sources are never
combined, and committed turns stay unchanged.

Each classification uses up to 20 recent chat items, including tool calls and results.
Late transcripts become part of the next committed turn. They do not change an
in-flight request or a reply that already started.
A new turn can cancel classification work without discarding its transcript.
AMD keeps its selected transcripts in an independent chat context. The next request
includes those transcripts and successful DTMF tool calls and results.
An empty EOT keeps useful pending classification for the latest turn. Older AMD
reply waits exit immediately. Reusing a prediction emits an `amd_prediction`
event with `reason="reused"`, so every committed turn produces one event.

## Realtime models

OpenAI Realtime can generate AMD replies when AgentSession controls turn detection.
Configure this before starting the session. AMD rejects server-side turn detection,
automatic tool replies, and models without per-response tool selection.

Native user transcription is optional. While AMD is active and native
transcription is disabled, committed turns use session STT for text history,
or AMD STT if session STT is not configured.

Supply a separate text LLM for classification and either session STT or AMD STT.
Realtime input transcription alone arrives after the audio commit, too late for
AMD's current turn. STT supplies the transcript; EOT commits the turn. The realtime
model still receives audio for its replies.

AMD races its STT against session transcripts only when session STT is configured.
Pass `stt=None` to AMD to reuse session STT instead.

```python
from livekit.agents import AMD, Agent, AgentSession
from livekit.plugins import openai, silero

session = AgentSession(
    llm=openai.realtime.RealtimeModel(turn_detection=None),
    vad=silero.VAD.load(),
    turn_handling={"turn_detection": "vad"},
)
await session.start(Agent(instructions="Call about an appointment."), room=ctx.room)

async with AMD(
    session,
    llm="google/gemini-3.1-flash-lite",
    stt="cartesia/ink-whisper",
    participant_identity=callee_identity,
) as detector:
    # Create the SIP participant here, as in the example above.
    result = await detector.execute()
```

AMD sends stage instructions with each authorized response and its tool follow-ups.
These instructions do not change the realtime session's base instructions or saved
history. The DTMF tool is available only to IVR replies and their tool follow-ups.

## Stage behavior

| Prediction | Reply behavior | AMD lifecycle |
| --- | --- | --- |
| `uncertain` | Before a machine stage, permit normal reply handling. In a machine stage, keep that stage's reply rule. | Keep the current stage. |
| `wait` | Skip the reply to an advertisement, promotion, or request to keep waiting. | Keep the current stage and pause the idle timer. |
| `machine-screening` | Answer the screener's latest question briefly. | Continue listening for the next turn. |
| `machine-vm` | Deliver one complete, uninterrupted message. | Keep listening during and after playback. |
| `machine-ivr` | Use the actual prompt to choose DTMF or a spoken response. | Continue listening for the next turn. |
| `human` | After a machine stage, supply temporary human instructions. Otherwise use normal Agent instructions. | Complete AMD. |
| `machine-unavailable` | Cancel held replies; do not generate a machine reply. | Complete AMD. |

`uncertain` and `wait` are per-turn predictions, not stages. Each
`amd_prediction` event carries both the `category` predicted for the turn and the
`stage` AMD keeps after it. `state_changed` is true only when the stage changes.
The stage constrains the classifier's allowed categories for the next turn, so
an uncertain turn in voicemail cannot jump to screening.

`wait` skips the current reply without a silence wait or menu extraction. While
the latest prediction is `wait`, the idle timer is paused; the overall `timeout`
still applies. Empty turns and inference failures reuse `wait`; the next
transcribed turn gets a fresh classification. A valid `wait` prediction resets
the inference-timeout and consecutive-uncertain counters. AMD completes on a
human prediction, so it does not classify later advertisements after that handoff.

Stage instructions are temporary. They do not enter the Agent's saved history.
Use `screening_instructions`, `voicemail_instructions`, `ivr_instructions`, and
`human_instructions` to customize them. `human_instructions` applies to the first
human turn after a machine stage. It is skipped if no machine stage preceded the
human, including when earlier predictions were only `uncertain`.
The customer hook runs before AMD adds its instructions.
`StopResponse` and interruption remain AgentSession responsibilities.

The normal interruption path handles a person who speaks during a message.
Interruption does not itself authorize a reply. AMD still checks the next turn.
An uncertain prediction keeps the current stage and its reply rule. AMD
classifies again on the next transcribed turn.

The FSM accepts classification results and returns the next category and effects.
Its allowed transitions also constrain the classifier's output schema. AMD owns
turn IDs, inference, deadlines, counters, fallback, reply authorization, and playback.
Timeouts and playback do not change the FSM. Repeated IVR predictions still request
menu extraction.

Accepted predictions update the category and emit an event immediately. Replies
to screening, voicemail, and IVR wait for 1.5 seconds of continuous participant
silence. Silence before EOT and during classification counts toward the threshold.
Human and initial `uncertain` replies use normal EOT timing. An unavailable result ends AMD.
Before reply authorization, new speech restarts the silence wait for the committed
turn, even if the new speech produces no accepted turn. After authorization,
AgentSession handles speech, silence, and interruptions. The next committed turn
pauses authorization until AMD decides whether to reply.
AMD enables interruptions on the session and current Agent when it starts,
then restores both settings when it finishes, including cancellation or setup failure.
AgentSession's false-interruption handling can resume paused speech when the
audio output supports pause and resume.
A new EOT uses the accepted category even if the previous reply is still waiting.
Empty turns reuse pending inference or the current category without adding empty
messages to classifier history. AMD retains earlier transcripts and successful
DTMF calls. Superseded requests cannot emit predictions or authorize old replies.

AMD records voicemail delivery only after successful, uninterrupted audio playback.
This record survives transitions through `uncertain` and IVR, so returning to
voicemail does not send a second message. An interrupted or failed attempt can be
retried on a later voicemail turn. An attempt still playing blocks another message.

## DTMF and menus

AMD observes the session's `function_tools_executed` event and retains successful
`send_dtmf_events` calls and results in its own chat context. It does not copy the
session's transcripts or agent speech. Failed or canceled calls are omitted,
including sequences that sent some digits before failing. The tool raises
`ToolError` when a publish fails. Calls are matched by the name `send_dtmf_events`,
including application overrides with that name. Tools with other names are ignored.
DTMF completion alone neither triggers classification nor proves that a person answered.

Menu extraction runs as separate best-effort work. It never holds a reply.
Each `amd_menu_observed` event identifies the turn and contains a menu description
and options with `label`, `dtmf`, and `spoken_response`. AMD does not build a tree
or execute the observed menu. Extraction has a 5-second deadline and at most
20 options. A new classification cancels the previous menu request.

## Deadlines and cleanup

| Control | Default | Behavior |
| --- | --- | --- |
| `inference_timeout` | 1.5 seconds | Use the current stage on timeout; its silence rule still applies. |
| `machine_silence_threshold` | 1.5 seconds | Wait for continuous silence before authorizing a machine reply. Set to `0` to disable. |
| `idle_timeout` | 10 seconds | Complete after inactivity outside voicemail. |
| `voicemail_idle_timeout` | 60 seconds | Allow a delayed post-message menu after playback. |
| `timeout` | 120 seconds | Fixed overall limit from the start of listening. |
| `max_uncertain_turns` | 3 | Complete with the current stage after this many consecutive uncertain predictions. Any other valid prediction, including `wait`, resets the count. |
| `max_inference_timeouts` | 3 | Complete after this many prediction timeouts. A valid prediction resets the count. |

At the inference deadline, AMD cancels the request and keeps the current stage.
Results that arrive after the deadline are ignored. The next transcribed turn
starts a new classification. Reaching `max_inference_timeouts` completes AMD
after the current stage's silence requirement.
Model errors and reused results use the same stage and silence rule.
New speech cancels the idle timer. Stage changes do not extend the overall limit.
The overall limit can end AMD during a silence wait. Without speech-end timing,
the silence wait starts at EOT.

Finishing AMD immediately clears `session.amd`, removes its listeners and turn
hooks, and releases session audio and reply authorization. Background cleanup
then cancels outstanding work and closes owned resources. `execute()` waits for
that cleanup and returns the final category and reason. Completion does not
decide the next call action.

## Run the example

The example requires a LiveKit room. Configure `.env` for the selected providers,
then run it in `dev` mode:

```sh
uv run python examples/telephony/amd.py dev
```

To place an outbound call, also set
`SIP_PHONE_NUMBER`, `SIP_PARTICIPANT_IDENTITY`, and `SIP_OUTBOUND_TRUNK_ID`.

## Current limits

- Realtime requires session or AMD STT, a separate text classifier, client-side
  turn detection, per-response tool selection, and client-controlled tool replies.
- Agent handoff during AMD is not supported.
- No audio-based hold detection.
- Session-level `ivr_detection` cannot run alongside AMD. Entry raises if it is enabled.
- The remote-session protocol maps screening to `AMD_UNKNOWN`. Full v2
  prediction, menu, and completion fields still need protocol support.
- Classification quality needs repeated model evals. Unit tests prove routing,
  transcript ownership, deadlines, and cleanup, not model accuracy.
